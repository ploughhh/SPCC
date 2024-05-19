import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Callable, Optional, Union

from scvi import REGISTRY_KEYS
from scvi.module import Classifier
from scvi._compat import Literal
from scvi.nn import one_hot

from scvi.train import TrainingPlan
from scvi.module.base import BaseModuleClass, LossRecorder, PyroBaseModuleClass
from ._components import Discriminator
from ._components import InnerpAffinity
from ._components import Sinkhorn
from pygmtools import hungarian


class AdversarialTrainingPlan(TrainingPlan):
    """
    Train vaes with adversarial loss option to encourage latent space mixing.

    Parameters
    ----------
    module
        A module instance from class ``BaseModuleClass``.
    lr
        Learning rate used for optimization :class:`~torch.optim.Adam`.
    weight_decay
        Weight decay used in :class:`~torch.optim.Adam`.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        Only activated when `n_epochs_kl_warmup` is set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from 0 to 1.
        Overrides `n_steps_kl_warmup` when both are not `None`.
    reduce_lr_on_plateau
        Whether to monitor validation loss and reduce learning rate when validation set
        `lr_scheduler_metric` plateaus.
    lr_factor
        Factor to reduce learning rate.
    lr_patience
        Number of epochs with no improvement after which learning rate will be reduced.
    lr_threshold
        Threshold for measuring the new optimum.
    lr_scheduler_metric
        Which metric to track for learning rate reduction.
    lr_min
        Minimum learning rate allowed
    adversarial_classifier
        Whether to use adversarial classifier in the latent space
    scale_adversarial_loss
        Scaling factor on the adversarial components of the loss.
        By default, adversarial loss is scaled from 1 to 0 following opposite of
        kl warmup.
    **loss_kwargs
        Keyword args to pass to the loss method of the `module`.
        `kl_weight` should not be passed here and is handled automatically.
    """

    def __init__(
        self,
        module: BaseModuleClass,
        lr=1e-3,
        weight_decay=1e-6,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = 400,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
        ] = "elbo_validation",
        lr_min: float = 0,
        adversarial_classifier: Union[bool, Classifier] = True,
        celltype_classifier: Union[bool, Classifier] = True,
        scale_adversarial_loss: Union[float, Literal["auto"]] = "auto",
        **loss_kwargs,
    ):
        super().__init__(
            module=module,
            lr=lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_min=lr_min,
        )
        self.n_celltype = self.module.n_labels
        if adversarial_classifier is True:
            self.n_output_classifier = self.module.n_batch
            self.adversarial_classifier = Classifier(
                n_input=self.module.n_latent,
                n_hidden=32,
                n_labels=self.n_output_classifier,
                n_layers=2,
                logits=True,
            )
        else:
            self.adversarial_classifier = adversarial_classifier
        if celltype_classifier is True:
            self.celltype_classifier = Classifier(
                n_input=self.module.n_latent,
                n_hidden=32,
                n_labels=6,
                n_layers=2,
                logits=True,
            )
        else:
            self.celltype_classifier = celltype_classifier
        self.scale_adversarial_loss = scale_adversarial_loss
        self.discriminator = Discriminator(10, {'dim': 20, 'norm': 'none', 'activ': 'lrelu', 'gan_type': 'lsgan'})

    def loss_adversarial_classifier(self, z, batch_index, predict_true_class=True):

        n_classes = self.n_output_classifier
        cls_logits = torch.nn.LogSoftmax(dim=1)(self.adversarial_classifier(z))

        if predict_true_class:
            cls_target = one_hot(batch_index, n_classes)
        else:
            one_hot_batch = one_hot(batch_index, n_classes)
            cls_target = torch.zeros_like(one_hot_batch)
            # place zeroes where true label is
            cls_target.masked_scatter_(
                ~one_hot_batch.bool(), torch.ones_like(one_hot_batch) / (n_classes - 1)
            )

        l_soft = cls_logits * cls_target
        loss = -l_soft.sum(dim=1).mean()

        return loss

    def loss_celltype_classifier(self, z, celltype_index, predict_true_class=True):

        n_classes = self.n_celltype
        cls_logits = torch.nn.LogSoftmax(dim=1)(self.celltype_classifier(z))


        cls_target = one_hot(celltype_index, n_classes)


        l_soft = cls_logits * cls_target
        loss = -l_soft.sum(dim=1).mean()

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        kappa = (
            1 - self.kl_weight
            if self.scale_adversarial_loss == "auto"
            else self.scale_adversarial_loss
        )
        batch_tensor = batch[REGISTRY_KEYS.BATCH_KEY]
        celltype_tensor = batch[REGISTRY_KEYS.LABELS_KEY]
        if optimizer_idx == 0:
            loss_kwargs = dict(kl_weight=self.kl_weight)
            inference_outputs, _, scvi_loss = self.forward(
                batch, loss_kwargs=loss_kwargs
            )
            loss = scvi_loss.loss
            # fool classifier if doing adversarial training
            if kappa > 0 and self.adversarial_classifier is not False:
                z = inference_outputs["z"]
                fool_loss = self.loss_adversarial_classifier(z, batch_tensor, False)
                loss += fool_loss * kappa

            self.log("train_loss", loss, on_epoch=True)
            self.compute_and_log_metrics(scvi_loss, self.elbo_train)
            return loss

        # train adversarial classifier
        # this condition will not be met unless self.adversarial_classifier is not False
        if optimizer_idx == 1:
            inference_inputs = self.module._get_inference_input(batch)
            outputs = self.module.inference(**inference_inputs)
            z = outputs["z"]
            loss = self.loss_adversarial_classifier(z.detach(), batch_tensor, True)
            loss *= kappa

            return loss

    def configure_optimizers(self):
        params1 = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer1 = torch.optim.Adam(
            params1, lr=self.lr, eps=0.01, weight_decay=self.weight_decay
        )
        config1 = {"optimizer": optimizer1}
        dis_params = list(self.discriminator.parameters())
        if self.reduce_lr_on_plateau:
            scheduler1 = ReduceLROnPlateau(
                optimizer1,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
                verbose=True,
            )
            config1.update(
                {
                    "lr_scheduler": scheduler1,
                    "monitor": self.lr_scheduler_metric,
                },
            )

        if self.adversarial_classifier is not False:
            params2 = filter(
                lambda p: p.requires_grad, self.adversarial_classifier.parameters()
            )
            optimizer2 = torch.optim.Adam(
                params2, lr=1e-3, eps=0.01, weight_decay=self.weight_decay
            )
            config2 = {"optimizer": optimizer2}

            # bug in pytorch lightning requires this way to return
            opts = [config1.pop("optimizer"), config2["optimizer"]]
            if "lr_scheduler" in config1:
                config1["scheduler"] = config1.pop("lr_scheduler")
                scheds = [config1]
                return opts, scheds
            else:
                return opts
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-4)


        return config1

class GIMVITrainingPlan(AdversarialTrainingPlan):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.discriminator = Discriminator(10, {'dim':20,'norm':'none', 'activ':'lrelu', 'gan_type':'lsgan'})
        if kwargs["adversarial_classifier"] is True:
            self.n_output_classifier = 2
            self.adversarial_classifier = Classifier(
                n_input=self.module.n_latent,
                n_hidden=32,
                n_labels=self.n_output_classifier,
                n_layers=3,
                logits=True,
            )
        else:
            self.adversarial_classifier = kwargs["adversarial_classifier"]
        # self.affinity_layer = InnerpAffinity(self.module.n_latent)
        # self.classifier = torch.nn.Linear(128, 128)
        # self.sinkhorn = Sinkhorn(max_iter=10, tau=1, epsilon=1e-4)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        kappa = (
            1 - self.kl_weight
            if self.scale_adversarial_loss == "auto"
            else self.scale_adversarial_loss
        )
        if optimizer_idx == 0:
            # batch contains both data loader outputs
            scvi_loss_objs = []
            n_obs = 0
            zs = []
            for (i, tensors) in enumerate(batch):
                n_obs += tensors[REGISTRY_KEYS.X_KEY].shape[0]
                self.loss_kwargs.update(dict(kl_weight=self.kl_weight, mode=i))
                inference_kwargs = dict(mode=i)
                generative_kwargs = dict(mode=i)
                inference_outputs, _, scvi_loss = self.forward(
                    tensors,
                    loss_kwargs=self.loss_kwargs,
                    inference_kwargs=inference_kwargs,
                    generative_kwargs=generative_kwargs,
                )
                zs.append(inference_outputs["z"])
                scvi_loss_objs.append(scvi_loss)

            loss = sum([scl.loss for scl in scvi_loss_objs])
            loss /= n_obs
            rec_loss = sum([scl.reconstruction_loss.sum() for scl in scvi_loss_objs])
            kl = sum([scl.kl_local.sum() for scl in scvi_loss_objs])

            # fool classifier if doing adversarial training
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # THIS PART IS VERY IMPORTANT
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # ADVERSARIAL TRAINING IS VERY IMPORTANT, ONE OF ADVERSARIAL CLASSIFIER MUST SET TO BE FALSE, THE OTHER MUST SET TO BE TRUE
            batch_tensor = [
                torch.zeros((z.shape[0], 1), device=z.device) + i
                for i, z in enumerate(zs)
            ]
            # celltype_tensor = batch[REGISTRY_KEYS.LABELS_KEY]
            celltype_tensor = [labels[REGISTRY_KEYS.LABELS_KEY] for labels in batch]
            if kappa > 0 and self.adversarial_classifier is not False:
                fool_loss = self.loss_adversarial_classifier(
                    torch.cat(zs), torch.cat(batch_tensor), True
                )
                loss += fool_loss * kappa
                discriminative_loss = self.discriminator.calc_gen_loss(zs[0]) + self.discriminator.calc_gen_loss_reverse(zs[1])
                loss += discriminative_loss * 5
            if kappa > 0 and self.celltype_classifier is not False:
                loss += self.loss_celltype_classifier(torch.cat(zs), torch.cat(celltype_tensor), True) * kappa
            # Ke = self.affinity_layer(zs[0], zs[1])
            # A = (Ke > 0).to(Ke.dtype)
            # emb = torch.diagonal(A)
            # v = self.classifier(emb)
            # s = v.view(v.shape[0], -1)
            # ss = self.sinkhorn(s, zs[0].shape[0], zs[1].shape[0])
            # x = hungarian(ss, torch.Tensor(zs[0].shape[0]), torch.Tensor(zs[1].shape[0]))
            # match_loss = torch.mean(torch.matmul(Ke, ss))
            #
            # loss += match_loss


            return {
                "loss": loss,
                "reconstruction_loss_sum": rec_loss,
                "kl_local_sum": kl,
                "kl_global": 0.0,
                "n_obs": n_obs,
            }

        # train adversarial classifier
        # this condition will not be met unless self.adversarial_classifier is not False
        if optimizer_idx == 1:
            zs = []
            for (i, tensors) in enumerate(batch):
                inference_inputs = self.module._get_inference_input(tensors)
                inference_inputs.update({"mode": i})
                outputs = self.module.inference(**inference_inputs)
                zs.append(outputs["z"])

            batch_tensor = [
                torch.zeros((z.shape[0], 1), device=z.device) + i
                for i, z in enumerate(zs)
            ]
            loss = self.loss_adversarial_classifier(
                torch.cat(zs).detach(), torch.cat(batch_tensor), False
            )
            loss *= kappa
            # loss += (torch.mean((zs[0] - 0) ** 2) + torch.mean((zs[1] - 1) ** 2)) * 0.05



            return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        self.loss_kwargs.update(dict(kl_weight=self.kl_weight, mode=dataloader_idx))
        inference_kwargs = dict(mode=dataloader_idx)
        generative_kwargs = dict(mode=dataloader_idx)
        _, _, scvi_loss = self.forward(
            batch,
            loss_kwargs=self.loss_kwargs,
            inference_kwargs=inference_kwargs,
            generative_kwargs=generative_kwargs,
        )
        reconstruction_loss = scvi_loss.reconstruction_loss
        return {
            "reconstruction_loss_sum": reconstruction_loss.sum(),
            "kl_local_sum": scvi_loss.kl_local.sum(),
            "kl_global": scvi_loss.kl_global,
            "n_obs": reconstruction_loss.shape[0],
        }

    def validation_epoch_end(self, outputs):
        """Aggregate validation step information."""
        n_obs, elbo, rec_loss, kl_local = 0, 0, 0, 0
        for dl_out in outputs:
            for tensors in dl_out:
                elbo += tensors["reconstruction_loss_sum"] + tensors["kl_local_sum"]
                rec_loss += tensors["reconstruction_loss_sum"]
                kl_local += tensors["kl_local_sum"]
                n_obs += tensors["n_obs"]
        # kl global same for each minibatch
        self.log("elbo_validation", elbo / n_obs)
        self.log("reconstruction_loss_validation", rec_loss / n_obs)
        self.log("kl_local_validation", kl_local / n_obs)
        self.log("kl_global_validation", 0.0)



