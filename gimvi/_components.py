import math
from typing import List, Iterable, Optional, Callable

import torch
from torch import nn
from torch.nn import functional as F, ModuleList
from torch.distributions import Normal
from torch.autograd import Function
import collections
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from src.utilities import one_hot
import pygmtools as pygm
from multiprocessing import Pool





def build_layer(layers, use_batch_norm=True, dropout_rate=0.1):
    fc_layer = nn.Sequential(
        collections.OrderedDict(
            [
                (
                    "Layer {}".format(),
                    nn.Sequential(
                        nn.Linear(in_dim, out_dim),
                        nn.LayerNorm(out_dim),
                        nn.LeakyReLU(),
                        nn.Dropout(dropout_rate)
                    ),
                )
                for i, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:]))
            ]
        )
    )

    return fc_layer


class FCLayers(nn.Module):
    """
    A helper class to build fully-connected layers for a neural network.

    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    """

    def __init__(
            self,
            n_in: int,
            n_out: int,
            n_cat_list: Iterable[int] = None,
            n_layers: int = 1,
            n_hidden: int = 128,
            dropout_rate: float = 0.1,
            use_batch_norm: bool = True,
            use_layer_norm: bool = False,
            use_activation: bool = True,
            bias: bool = True,
            inject_covariates: bool = True,
            activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        cat_dim = sum(self.n_cat_list)
        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(
                                n_in + cat_dim * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                    zip(layers_dim[:-1], layers_dim[1:])
                )
                ]
            )
        )

    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond

    def set_online_update_hooks(self, hook_first_layer=True):
        self.hooks = []

        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[:, -categorical_dims:] = grad[:, -categorical_dims:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, nn.Linear):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)

    def forward(self, x: torch.Tensor, *cat_list: int):
        """
        Forward computation on ``x``.

        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample
        x: torch.Tensor

        Returns
        -------
        py:class:`torch.Tensor`
            tensor of shape ``(n_out,)``

        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError(
                "nb. categorical args provided doesn't match init. params."
            )
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x


def reparameter(mu, var):
    return Normal(mu, var.sqrt(), validate_args=False).rsample()

def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt(), validate_args=False).rsample()

def identity(x):
    return x

def cross_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    key = value
    scores = torch.einsum('ik,jk->ij', query, key)
    prob = torch.softmax(scores, dim=-1)
    return prob

def scaled_dot_product(q, k, v, A, mask=None):
    d_k = q.size()[-1]
    k = torch.matmul(A, k)
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention



class FastMMD(nn.Module):
    """ Fast Maximum Mean Discrepancy approximated using the random kitchen sinks method.
    """

    def __init__(self, out_features, gamma):
        super().__init__()
        self.gamma = gamma
        self.out_features = out_features

    def forward(self, a, b):
        in_features = a.shape[-1]

        # W sampled from normal
        w_rand = torch.randn((in_features, self.out_features), device=a.device)
        # b sampled from uniform
        b_rand = torch.zeros((self.out_features,), device=a.device).uniform_(0, 2 * math.pi)

        phi_a = self._phi(a, w_rand, b_rand).mean(dim=0)
        phi_b = self._phi(b, w_rand, b_rand).mean(dim=0)
        mmd = torch.norm(phi_a - phi_b, 2)

        return mmd

    def _phi(self, x, w, b):
        scale_a = math.sqrt(2 / self.out_features)
        scale_b = math.sqrt(2 / self.gamma)
        out = scale_a * (scale_b * (x @ w + b)).cos()
        return out



class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha: Optional[float] = 1.0):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class Multi_head_Attention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, A, mask=None, return_attention=True):
        # batch_size, seq_length, embed_dim = x.size()
        batch_size, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        # qkv = qkv.reshape(batch_size, self.num_heads, 3 * self.head_dim)
        # qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, A, mask=mask)
        # values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        # values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class Discriminator(nn.Module):

    def __init__(self, input_dim, params):
        super(Discriminator, self).__init__()
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.input_dim = input_dim
        self.net = self._make_net()

    def _make_net(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.input_dim, self.dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.dim, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = [self.forward(input_fake)]
        outs1 = [self.forward(input_real)]
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = [self.forward(input_fake)]
        loss = 0
        for it, (out0) in enumerate(outs0):
            # 1 = real data
            loss += torch.mean((out0 - 1) ** 2)
        return loss

    def calc_gen_loss_reverse(self, input_real):
        # calculate the loss to train G
        outs0 = [self.forward(input_real)]
        loss = 0
        for it, (out0) in enumerate(outs0):
            # 0 = fake data
            loss += torch.mean((out0 - 0) ** 2)
        return loss

    def calc_gen_loss_half(self, input_fake):
        # calculate the loss to train G
        outs0 = [self.forward(input_fake)]
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0.5) ** 2)
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

class InnerpAffinity(nn.Module):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.lambda_ = Parameter(torch.Tensor(self.latent_dim, self.latent_dim))
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.lambda_.size(1))
        self.lambda_.data.uniform_(-stdv, stdv)

    def forward(self, X, Y):
        Me = torch.matmul(X, self.lambda_)
        Me = torch.matmul(Me, Y.transpost(0,1))

        return Me

class Sinkhorn(nn.Module):
    r"""
    Sinkhorn algorithm turns the input matrix into a bi-stochastic matrix.

    Sinkhorn algorithm firstly applies an ``exp`` function with temperature :math:`\tau`:

    .. math::
        \mathbf{S}_{i,j} = \exp \left(\frac{\mathbf{s}_{i,j}}{\tau}\right)

    And then turns the matrix into doubly-stochastic matrix by iterative row- and column-wise normalization:

    .. math::
        \mathbf{S} &= \mathbf{S} \oslash (\mathbf{1}_{n_2} \mathbf{1}_{n_2}^\top \cdot \mathbf{S}) \\
        \mathbf{S} &= \mathbf{S} \oslash (\mathbf{S} \cdot \mathbf{1}_{n_2} \mathbf{1}_{n_2}^\top)

    where :math:`\oslash` means element-wise division, :math:`\mathbf{1}_n` means a column-vector with length :math:`n`
    whose elements are all :math:`1`\ s.

    :param max_iter: maximum iterations (default: ``10``)
    :param tau: the hyper parameter :math:`\tau` controlling the temperature (default: ``1``)
    :param epsilon: a small number for numerical stability (default: ``1e-4``)
    :param log_forward: apply log-scale computation for better numerical stability (default: ``True``)
    :param batched_operation: apply batched_operation for better efficiency (but may cause issues for back-propagation,
     default: ``False``)

    .. note::
        ``tau`` is an important hyper parameter to be set for Sinkhorn algorithm. ``tau`` controls the distance between
        the predicted doubly-stochastic matrix, and the discrete permutation matrix computed by Hungarian algorithm (see
        :func:`~src.lap_solvers.hungarian.hungarian`). Given a small ``tau``, Sinkhorn performs more closely to
        Hungarian, at the cost of slower convergence speed and reduced numerical stability.

    .. note::
        We recommend setting ``log_forward=True`` because it is more numerically stable. It provides more precise
        gradient in back propagation and helps the model to converge better and faster.

    .. note::
        Setting ``batched_operation=True`` may be preferred when you are doing inference with this module and do not
        need the gradient.
    """
    def __init__(self, max_iter: int=10, tau: float=1., epsilon: float=1e-4,
                 log_forward: bool=True, batched_operation: bool=False):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.tau = tau
        self.epsilon = epsilon
        self.log_forward = log_forward
        if not log_forward:
            print('Warning: Sinkhorn algorithm without log forward is deprecated because log_forward is more stable.')
        self.batched_operation = batched_operation # batched operation may cause instability in backward computation,
                                                   # but will boost computation.

    def forward(self, s: torch.Tensor, nrows: torch.Tensor=None, ncols: torch.Tensor=None, dummy_row: bool=False) -> torch.Tensor:
        r"""
        :param s: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size
        :param nrows: :math:`(b)` number of objects in dim1
        :param ncols: :math:`(b)` number of objects in dim2
        :param dummy_row: whether to add dummy rows (rows whose elements are all 0) to pad the matrix to square matrix.
         default: ``False``
        :return: :math:`(b\times n_1 \times n_2)` the computed doubly-stochastic matrix

        .. note::
            We support batched instances with different number of nodes, therefore ``nrows`` and ``ncols`` are
            required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
            the batched matrices are not padded.

        .. note::
            The original Sinkhorn algorithm only works for square matrices. To handle cases where the graphs to be
            matched have different number of nodes, it is a common practice to add dummy rows to construct a square
            matrix. After the row and column normalizations, the padded rows are discarded.

        .. note::
            We assume row number <= column number. If not, the input matrix will be transposed.
        """
        if self.log_forward:
            return self.forward_log(s, nrows, ncols, dummy_row)
        else:
            return self.forward_ori(s, nrows, ncols, dummy_row) # deprecated

    def forward_log(self, s, nrows=None, ncols=None, dummy_row=False):
        """Compute sinkhorn with row/column normalization in the log space."""
        return pygm.sinkhorn(s, n1=nrows, n2=ncols, dummy_row=dummy_row, max_iter=self.max_iter, tau=self.tau, batched_operation=self.batched_operation, backend='pytorch')

    def forward_ori(self, s, nrows=None, ncols=None, dummy_row=False):
        r"""
        Computing sinkhorn with row/column normalization.

        .. warning::
            This function is deprecated because :meth:`~src.lap_solvers.sinkhorn.Sinkhorn.forward_log` is more
            numerically stable.
        """
        if len(s.shape) == 2:
            s = s.unsqueeze(0)
            matrix_input = True
        elif len(s.shape) == 3:
            matrix_input = False
        else:
            raise ValueError('input data shape not understood.')

        batch_size = s.shape[0]

        #s = s.to(dtype=dtype)

        if nrows is None:
            nrows = [s.shape[1] for _ in range(batch_size)]
        if ncols is None:
            ncols = [s.shape[2] for _ in range(batch_size)]

        # tau scaling
        ret_s = torch.zeros_like(s)
        for b, n in enumerate(nrows):
            ret_s[b, 0:n, 0:ncols[b]] = \
                nn.functional.softmax(s[b, 0:n, 0:ncols[b]] / self.tau, dim=-1)
        s = ret_s

        # add dummy elements
        if dummy_row:
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            #s = torch.cat((s, torch.full(dummy_shape, self.epsilon * 10).to(s.device)), dim=1)
            #nrows = nrows + dummy_shape[1] # non in-place
            s = torch.cat((s, torch.full(dummy_shape, 0.).to(s.device)), dim=1)
            ori_nrows = nrows
            nrows = ncols
            for b in range(batch_size):
                s[b, ori_nrows[b]:nrows[b], :ncols[b]] = self.epsilon

        row_norm_ones = torch.zeros(batch_size, s.shape[1], s.shape[1], device=s.device, dtype=s.dtype)  # size: row x row
        col_norm_ones = torch.zeros(batch_size, s.shape[2], s.shape[2], device=s.device, dtype=s.dtype)  # size: col x col
        for b in range(batch_size):
            row_slice = slice(0, nrows[b])
            col_slice = slice(0, ncols[b])
            row_norm_ones[b, row_slice, row_slice] = 1
            col_norm_ones[b, col_slice, col_slice] = 1

        s += self.epsilon

        for i in range(self.max_iter):
            if i % 2 == 0:
                # column norm
                #ones = torch.ones(batch_size, s.shape[1], s.shape[1], device=s.device)
                sum = torch.sum(torch.mul(s.unsqueeze(3), col_norm_ones.unsqueeze(1)), dim=2)
            else:
                # row norm
                # ones = torch.ones(batch_size, s.shape[2], s.shape[2], device=s.device)
                sum = torch.sum(torch.mul(row_norm_ones.unsqueeze(3), s.unsqueeze(1)), dim=2)

            tmp = torch.zeros_like(s)
            for b in range(batch_size):
                row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
                col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
                tmp[b, row_slice, col_slice] = 1 / sum[b, row_slice, col_slice]
            s = s * tmp

        if dummy_row:
            if dummy_shape[1] > 0:
                s = s[:, :-dummy_shape[1]]
            for b in range(batch_size):
                s[b, ori_nrows[b]:nrows[b], :ncols[b]] = 0

        if matrix_input:
            s.squeeze_(0)

        return s


class GumbelSinkhorn(nn.Module):
    """
    Gumbel Sinkhorn Layer turns the input matrix into a bi-stochastic matrix.
    See details in `"Mena et al. Learning Latent Permutations with Gumbel-Sinkhorn Networks. ICLR 2018"
    <https://arxiv.org/abs/1802.08665>`_

    :param max_iter: maximum iterations (default: ``10``)
    :param tau: the hyper parameter :math:`\tau` controlling the temperature (default: ``1``)
    :param epsilon: a small number for numerical stability (default: ``1e-4``)
    :param batched_operation: apply batched_operation for better efficiency (but may cause issues for back-propagation,
     default: ``False``)

    .. note::
        This module only supports log-scale Sinkhorn operation.
    """
    def __init__(self, max_iter=10, tau=1., epsilon=1e-4, batched_operation=False):
        super(GumbelSinkhorn, self).__init__()
        self.sinkhorn = Sinkhorn(max_iter, tau, epsilon, batched_operation=batched_operation)

    def forward(self, s: torch.Tensor, nrows: torch.Tensor=None, ncols: torch.Tensor=None,
                sample_num=5, dummy_row=False) -> torch.Tensor:
        r"""
        :param s: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size
        :param nrows: :math:`(b)` number of objects in dim1
        :param ncols: :math:`(b)` number of objects in dim2
        :param sample_num: number of samples
        :param dummy_row: whether to add dummy rows (rows whose elements are all 0) to pad the matrix to square matrix.
         default: ``False``
        :return: :math:`(b m\times n_1 \times n_2)` the computed doubly-stochastic matrix. :math:`m`: number of samples
         (``sample_num``)

        The samples are stacked at the fist dimension of the output tensor. You may reshape the output tensor ``s`` as:

        ::

            s = torch.reshape(s, (-1, sample_num, s.shape[1], s.shape[2]))

        .. note::
            We support batched instances with different number of nodes, therefore ``nrows`` and ``ncols`` are
            required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
            the batched matrices are not padded.

        .. note::
            The original Sinkhorn algorithm only works for square matrices. To handle cases where the graphs to be
            matched have different number of nodes, it is a common practice to add dummy rows to construct a square
            matrix. After the row and column normalizations, the padded rows are discarded.

        .. note::
            We assume row number <= column number. If not, the input matrix will be transposed.
        """
        def sample_gumbel(t_like, eps=1e-20):
            """
            randomly sample standard gumbel variables
            """
            u = torch.empty_like(t_like).uniform_()
            return -torch.log(-torch.log(u + eps) + eps)

        s_rep = torch.repeat_interleave(s, sample_num, dim=0)
        s_rep = s_rep + sample_gumbel(s_rep)
        nrows_rep = torch.repeat_interleave(nrows, sample_num, dim=0)
        ncols_rep = torch.repeat_interleave(ncols, sample_num, dim=0)
        s_rep = self.sinkhorn(s_rep, nrows_rep, ncols_rep, dummy_row)
        #s_rep = torch.reshape(s_rep, (-1, sample_num, s_rep.shape[1], s_rep.shape[2]))
        return s_rep

class CrossEntropyLoss(nn.Module):
    r"""
    Multi-class cross entropy loss between two permutations.

    .. math::
        L_{ce} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2} \left(\mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred_dsmat: torch.Tensor, gt_perm: torch.Tensor, src_ns: torch.Tensor, tgt_ns: torch.Tensor) -> torch.Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged cross-entropy loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_dsmat)
            raise err

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            gt_index = torch.max(gt_perm[batch_slice], dim=-1).indices
            loss += F.nll_loss(
                torch.log(pred_dsmat[batch_slice]),
                gt_index,
                reduction='sum')
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum

class common_encoder(nn.Module):
    def __init__(self,
                 sc_input_dim,
                 spa_input_dim,
                 output_dim: int = 3000,
                 drop_rate: int = 0.1,
                 hidden_dim: int = 6000,
                 ):
        super(common_encoder, self).__init__()

        self.drop_rate = drop_rate
        self.sc_encoder = nn.Sequential(
            nn.Linear(sc_input_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate)
        )
        self.spa_encoder = nn.Sequential(
            nn.Linear(spa_input_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate)
        )
        self.concat1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.concat2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.mu_encoder = nn.Linear(hidden_dim * 2, output_dim)
        self.var_encoder = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        sc_parameter = self.sc_encoder(x)
        spa_parameter = self.spa_encoder(y)
        combine_parameter = self.concat1(torch.cat((sc_parameter, spa_parameter), 1))
        combine_mu = self.mu_encoder(combine_parameter)
        combine_var = torch.exp(self.var_encoder(combine_parameter))
        z_latent = reparameter(combine_mu, combine_var)
        return combine_mu, combine_var, z_latent


class private_encoder(nn.Module):
    def __init__(self,
                 sc_input_dim,
                 hidden_dim,
                 output_dim,
                 drop_rate):
        super(private_encoder, self).__init__()

        self.dropout_rate = drop_rate
        self.private_sc_encoder = nn.Sequential(
            nn.Linear(sc_input_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate)
        )
        self.mu_encoder = nn.Linear(hidden_dim, output_dim)
        self.var_encoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        sc_latent = self.private_sc_encoder(x)
        parameter_mu = self.mu_encoder(sc_latent)
        parameter_var = torch.exp(self.var_encoder(sc_latent))
        z_latent = reparameter(parameter_mu, parameter_var)
        return parameter_mu, parameter_var, z_latent


# class Encoder(nn.Module):
#     def __init__(self,
#                  input_dim,
#                  output_dim,
#                  drop_rate,
#                  hidden_dim: int = 128):
#         super(Encoder, self).__init__()
#
#         self.drop_rate = drop_rate
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim, bias=True),
#             nn.LayerNorm(hidden_dim),
#             nn.LeakyReLU(),
#             nn.Dropout(self.drop_rate)
#         )
#         self.mu_encoder = nn.Linear(hidden_dim, output_dim)  # 10140, 3000
#         self.var_encoder = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         parameter = self.encoder(x)
#         mu = self.mu_encoder(parameter)
#         var = torch.exp(self.var_encoder(parameter)) + 1e-4
#         z_latent = reparameter(mu, var)
#         return mu, var, z_latent

# Encoder
class Encoder(nn.Module):
    """
    Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivity of the variance.
        When `None`, defaults to `torch.exp`.
    **kwargs
        Keyword args for :class:`~scvi.module._base.FCLayers`
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()

        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = identity
        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""
        The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        latent = self.z_transformation(reparameterize_gaussian(q_m, q_v))
        return q_m, q_v, latent



class library_encoder(nn.Module):
    def __init__(self,
                 input,
                 hidden_dim: int = 128,
                 dropout_rate: float = 0.1):
        super(library_encoder, self).__init__()
        self.dropout_rate = dropout_rate

        self.library_encoder = nn.Sequential(
            nn.Linear(input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate)
        )

        self.library_mu = nn.Linear(hidden_dim, 1, bias=True)
        self.library_var = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, x: torch.Tensor):
        z = self.library_encoder(x)
        mu = self.library_mu(z)
        var = torch.exp(self.library_var(z)) + 1e-4
        latent_z = reparameter(mu, var)

        return mu, var, latent_z


class GCN(nn.Module):
    def __init__(self,
                 feature,
                 out_feature,
                 bias=True,
                 normalized=False):
        super(GCN, self).__init__()
        self.feature = feature
        self.out_feature = out_feature
        self.normalized = normalized
        self.normalize = nn.LayerNorm(out_feature)

        self.weight = nn.parameter.Parameter(
            torch.FloatTensor(feature, out_feature)
        )

        if bias:
            self.bias = nn.parameter.Parameter(
                torch.FloatTensor(out_feature)
            )
        else:
            self.register_parameter('gcn_encoder_bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_batch(self):
        stdv_batch = 1. / math.sqrt(self.batch_weight.size(1))
        self.batch_weight.data.uniform_(-stdv_batch, stdv_batch)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv_batch, stdv_batch)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj.T, support)
        # self.batch_weight = torch.nn.parameter.Parameter(torch.FloatTensor(adj.shape[0], adj.shape[1])).to("cuda")
        self.batch_weight = torch.nn.parameter.Parameter(torch.FloatTensor(adj.shape[0], adj.shape[1]))
        self.reset_parameters_batch()
        output = torch.mm(self.batch_weight, output)
        # batch_scale_mat = torch.ones(input.shape[0], adj.shape[1]).to('cuda')
        # batch_scale_mat = torch.ones(input.shape[0], adj.shape[1])
        # output = torch.mm(batch_scale_mat, output)
        if self.normalized:
            output = self.normalize(output)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class MeanAggregator(nn.Module):
    def __init__(self,
                 input_features,
                 out_features,
                 concat: bool = True,
                 dropout: float = 0.1,
                 bias: bool = True):
        super(MeanAggregator, self).__init__()
        self.input_features = input_features
        self.out_features = out_features
        self.concat = concat
        self.bias = bias
        self.dropout = dropout

        self.self_weight = nn.Parameter(
            torch.FloatTensor(self.input_features, self.out_features)
        )
        self.neighbor_weight = nn.Parameter(
            torch.FloatTensor(self.input_features, self.out_features)
        )
        if bias:
            self.self_bias = nn.parameter.Parameter(
                torch.FloatTensor(out_features)
            )
            self.neighbor_bias = nn.parameter.Parameter(
                torch.FloatTensor(out_features)
            )
        else:
            self.register_parameter('gcn_encoder_bias', None)

    def reset_parameters(self):
        self_stdv = 1. / math.sqrt(self.self_weight.size(1))
        neighbor_stdv = 1. / math.sqrt(self.neighbor_weight.size(1))
        self.self_weight.data.uniform_(-self_stdv, self_stdv)
        self.neighbor_weight.data.uniform_(-neighbor_stdv, neighbor_stdv)
        self.self_bias.data.uniform_(-self_stdv, self_stdv)
        self.neighbor_bias.data.uniform_(-neighbor_stdv, neighbor_stdv)

    def forward(self, x, adj):
        self_x, neighbor_x = x








class Multi_GCNEncoder(nn.Module):
    def __init__(
            self,
            n_heads: int,
            n_input_list: List[int],
            n_output: int,
            num_heads: int,
            n_hidden: int = 128,
            n_layers_individual: int = 1,
            n_layers_shared: int = 2,
            n_cat_list: Iterable[int] = None,
            dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.n_hidden = n_hidden

        self.encoders = ModuleList(
            [
                FCLayers(
                    n_in=n_input_list[i],
                    n_out=n_hidden,
                    n_cat_list=n_cat_list,
                    n_layers=n_layers_individual,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    use_batch_norm=True,
                )
                for i in range(n_heads)
            ]
        )

        self.encoder_shared = FCLayers(
            n_in=n_hidden,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers_shared,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

        # self.self_attention_raw = MultiheadAttention(n_input_list, n_input_list, num_heads)
        # self.self_attention_mid = MultiheadAttention(n_input_list, n_hidden, num_heads)
        # self.self_attention_latent = MultiheadAttention(n_hidden, n_output, num_heads)
        self.self_attention = Multi_head_Attention(n_hidden, n_hidden, num_heads)
        # self.grl = ReverseLayerF()



    def attention(self, H, A, v):
        f1 = torch.matmul(H, v[0])
        f1 = A * f1
        f2 = torch.matmul(H, v[1])
        f2 = A * torch.transpose(f2, 0, 1)
        edge_weight = f1 + f2
        edge_weight = torch.sigmoid(edge_weight)
        attentions = torch.softmax(edge_weight, dim=0)
        return attentions


    def forward(self, x: torch.Tensor, spatial_metric: torch.Tensor, head_id: int, *cat_list: int):

        q = self.encoders[head_id](x, *cat_list)
        # self.vs[head_id] = {}
        # self.vs[head_id][0] = nn.init.kaiming_uniform(nn.Parameter(torch.FloatTensor(q.size(1), 1)))
        # self.vs[head_id][1] = nn.init.kaiming_uniform(nn.Parameter(torch.FloatTensor(q.size(1), 1)))

        # att1 = self.attention(q, spatial_metric, vs[head_id])
        # q = torch.matmul(att1, q)

        # self.vs['shared'] = {}
        # self.vs['shared'][0] = nn.init.kaiming_uniform(nn.Parameter(torch.FloatTensor(q.size(1), 1)))
        # self.vs['shared'][1] = nn.init.kaiming_uniform(nn.Parameter(torch.FloatTensor(q.size(1), 1)))
        q = self.encoder_shared(q, *cat_list)
        # att2 = self.attention(q, spatial_metric, vs['shared'])
        # q = torch.matmul(att2, q)
        q, attention = self.self_attention(q, spatial_metric)

        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q))
        latent = reparameter(q_m, q_v)
        latent = ReverseLayerF.apply(latent)

        return q_m, q_v, latent, attention





class GCNEncoder_shared(nn.Module):
    def __init__(
            self,
            n_heads: int,
            n_input_list: List[int],
            n_output: int,
            n_hidden: int = 128,
            n_layers_individual: int = 1,
            n_layers_shared: int = 2,
            n_cat_list: Iterable[int] = None,
            dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.encoders = ModuleList(
            [
                FCLayers(
                    n_in=n_input_list[i],
                    n_out=n_hidden,
                    n_cat_list=n_cat_list,
                    n_layers=n_layers_individual,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    use_batch_norm=True,
                )
                for i in range(n_heads)
            ]
        )

        self.encoder_shared = FCLayers(
            n_in=n_hidden,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers_shared,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.gcn_encoders = ModuleList(
            [
                GCN(n_input_list[j], 1024, bias=True)
                for j in range(n_heads)
            ]
        )
        # self.gcn_encoder_sc = GCN(n_input_list[0], 1024, bias=True)
        # self.gcn_encoder_spa = GCN(n_input_list[1], 1024, bias=True)
        # self.gcn_encoder2 = GCN_encoder(1024, n_hidden, bias=True)
        self.encoder = nn.Sequential(
            nn.Linear(1024, n_hidden, bias=True),
            nn.LayerNorm(n_hidden),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, x: torch.Tensor, spatial_metric: torch.Tensor, head_id: int, *cat_list: int):
        # if head_id == 0:
        #     q = self.encoders[0](x, *cat_list)
        #
        # if head_id == 1:
        #     q = F.relu(self.gcn_encoder1(x, spatial_metric))
        #     q = F.dropout(q, self.dropout_rate)
        #     # q = self.gcn_encoder2(q, spatial_metric)
        #     q = self.spa_encoder(q)

        q = F.relu(self.gcn_encoders[head_id](x, spatial_metric))
        q = F.dropout(q, self.dropout_rate)
        q = self.encoder(q)


        q = self.encoder_shared(q, *cat_list)

        q_m = self.mean_encoder(q)
        q_var = torch.exp(self.var_encoder(q))
        latent = reparameter(q_m, q_var)

        # q_m = self.mean_encoder(q)
        # q_v = torch.exp(self.var_encoder(q))
        # latent = reparameter(q_m, q_v)

        return q_m, q_var, latent


# class MultiEncoder(nn.Module):
#     def __init__(
#         self,
#         n_heads: int,
#         n_input_list: List[int],
#         n_output: int,
#         n_hidden: int = 128,
#         n_layers_individual: int = 1,
#         n_layers_shared: int = 2,
#         n_cat_list: Iterable[int] = None,
#         dropout_rate: float = 0.1,
#     ):
#         super().__init__()
#
#         self.encoders = ModuleList(
#             [
#                 FCLayers(
#                     n_in=n_input_list[i],
#                     n_out=n_hidden,
#                     n_cat_list=n_cat_list,
#                     n_layers=n_layers_individual,
#                     n_hidden=n_hidden,
#                     dropout_rate=dropout_rate,
#                     use_batch_norm=True,
#                 )
#                 for i in range(n_heads)
#             ]
#         )
#
#         self.encoder_shared = FCLayers(
#             n_in=n_hidden,
#             n_out=n_hidden,
#             n_cat_list=n_cat_list,
#             n_layers=n_layers_shared,
#             n_hidden=n_hidden,
#             dropout_rate=dropout_rate,
#         )
#         self.dropout_rate = dropout_rate
#         self.gcn_encoder = GCN(n_input_list[1], 128, bias=True)
#         self.spa_encoder = nn.Sequential(
#             nn.Linear(128, n_hidden, bias=True),
#             nn.LayerNorm(n_hidden),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout_rate)
#         )
#         self.mean_encoder = nn.Linear(n_hidden, n_output)
#         self.var_encoder = nn.Linear(n_hidden, n_output)
#
#     def forward(self, x: torch.Tensor, spatial_metric: torch.Tensor, head_id: int, *cat_list: int):
#         global q
#         if head_id == 0:
#             q = self.encoders[0](x, *cat_list)
#
#         if head_id == 1:
#             q = F.relu(self.gcn_encoder(x, spatial_metric))
#             q = F.dropout(q, self.dropout_rate)
#             # q = self.gcn_encoder2(q, spatial_metric)
#             q = self.spa_encoder(q)
#
#         # q = self.encoders[head_id](x, *cat_list)
#         lat = self.encoder_shared(q, *cat_list)
#
#         q_m = self.mean_encoder(lat)
#         q_v = torch.exp(self.var_encoder(lat))
#         latent = reparameterize_gaussian(q_m, q_v)
#
#         return q_m, q_v, latent

class MultiEncoder(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_input_list: List[int],
        n_output: int,
        n_hidden: int = 128,
        n_layers_individual: int = 1,
        n_layers_shared: int = 2,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoders = ModuleList(
            [
                FCLayers(
                    n_in=n_input_list[i],
                    n_out=n_hidden,
                    n_cat_list=n_cat_list,
                    n_layers=n_layers_individual,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    use_batch_norm=True,
                )
                for i in range(n_heads)
            ]
        )

        self.encoder_shared = FCLayers(
            n_in=n_hidden,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers_shared,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, head_id: int, *cat_list: int):
        q = self.encoders[head_id](x, *cat_list)
        q = self.encoder_shared(q, *cat_list)

        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q))
        latent = reparameterize_gaussian(q_m, q_v)

        return q_m, q_v, latent


class Decoder(nn.Module):
    """
    Decodes data from latent space to data space.

    ``n_input`` dimensions to ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.
    Output is the mean and variance of a multivariate Gaussian

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    kwargs
        Keyword args for :class:`~scvi.module._base.FCLayers`
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            **kwargs,
        )

        self.mean_decoder = nn.Linear(n_hidden, n_output)
        self.mean_decoder_scale = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.Softmax(dim=-1)
        )
        self.var_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        """
        The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        Parameters
        ----------
        x
            tensor with shape ``(n_input,)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        2-tuple of :py:class:`torch.Tensor`
            Mean and variance tensors of shape ``(n_output,)``

        """
        # Parameters for latent distribution
        p = self.decoder(x, *cat_list)
        p_m_scale = self.mean_decoder_scale(p)
        p_m = self.mean_decoder(p)
        p_v = torch.exp(self.var_decoder(p))
        return p_m_scale, p_m, p_v

# class GCNDecoder(nn.Module):
#     def __init__(self,
#                  n_input: int,
#                  n_output: int,
#                  n_hidden: int = 128,
#                  n_layers_conditions: int = 1,
#                  n_cat_list: Iterable[int] = None,
#                  dropout_rate: float = 0.2):
#         super().__init__()
#
#         n_out = n_hidden if n_layers_conditions else n_output
#
#         if n_layers_conditions:
#             self.px_decoder_conditioned = FCLayers(
#                 n_in=n_input,
#                 n_out=n_out,
#                 n_cat_list=n_cat_list,
#                 n_layers=n_layers_conditions,
#                 n_hidden=n_hidden,
#                 dropout_rate=dropout_rate,
#                 use_batch_norm=True,
#             )
#             n_in = n_out
#         else:
#             self.px_decoder_conditioned = None
#             n_in = n_input
#
#         self.px_scale_decoder = nn.Sequential(
#             nn.Linear(n_in, n_output),
#             nn.Softmax(dim=-1)
#         )
#         self.px_r_decoder = nn.Linear(n_in, n_output)
#         self.px_dropout_decoder = nn.Linear(n_in, n_output)
#
#     def forward(self, z: torch.Tensor, dataset_id: int, *cat_list: int):
#         px = z
#         if self.px_decoder_conditioned:
#             px = self.px_decoder_conditioned(px, *cat_list)
#         else:
#             px = z
#
#         px_scale = self.px_scale_decoder(px)
#         px_dropout = self.px_dropout_decoder(px)
#         px_r = self.px_r_decoder(px)
#
#         return px_scale, px_r, px_dropout


class GCNDecoder(nn.Module):
    def __init__(self,
                 n_input: int,
                 n_output: int,
                 n_hidden: int = 128,
                 n_layers_conditions: int = 1,
                 n_cat_list: Iterable[int] = None,
                 dropout_rate: float = 0.2):
        super().__init__()

        n_out = n_hidden if n_layers_conditions else n_output

        if n_layers_conditions:
            self.decoder = FCLayers(
                n_in=n_input,
                n_out=n_out,
                n_cat_list=n_cat_list,
                n_layers=n_layers_conditions,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                use_batch_norm=True,
            )
            n_in = n_out
        else:
            self.decoder = None
            n_in = n_input

        self.raw_decoder_soft = nn.Sequential(
            nn.Linear(n_in, n_output),
            nn.Softmax(dim=-1)
        )
        self.raw_decoder = nn.Linear(n_in, n_output)
        self.px_dropout_decoder = nn.Linear(n_in, n_output)

    def forward(self, z: torch.Tensor, dataset_id: int, library: torch.Tensor, *cat_list: int):
        px = z
        px = self.decoder(px, *cat_list)


        px_scale = self.raw_decoder_soft(px)
        px_dropout = self.px_dropout_decoder(px)
        px_raw = self.raw_decoder(px)
        px_rate = torch.exp(library) * px_scale

        return px_scale, px_raw, px_rate, px_dropout


class MultiDecoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden_conditioned: int = 32,
        n_hidden_shared: int = 128,
        n_layers_conditioned: int = 1,
        n_layers_shared: int = 1,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        n_out = n_hidden_conditioned if n_layers_shared else n_hidden_shared
        if n_layers_conditioned:
            self.px_decoder_conditioned = FCLayers(
                n_in=n_input,
                n_out=n_out,
                n_cat_list=n_cat_list,
                n_layers=n_layers_conditioned,
                n_hidden=n_hidden_conditioned,
                dropout_rate=dropout_rate,
                use_batch_norm=True,
            )
            n_in = n_out
        else:
            self.px_decoder_conditioned = None
            n_in = n_input

        if n_layers_shared:
            self.px_decoder_final = FCLayers(
                n_in=n_in,
                n_out=n_hidden_shared,
                n_cat_list=[],
                n_layers=n_layers_shared,
                n_hidden=n_hidden_shared,
                dropout_rate=dropout_rate,
                use_batch_norm=True,
            )
            n_in = n_hidden_shared
        else:
            self.px_decoder_final = None

        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_in, n_output), nn.Softmax(dim=-1)
        )
        self.px_r_decoder = nn.Linear(n_in, n_output)
        self.px_dropout_decoder = nn.Linear(n_in, n_output)

    def forward(
        self,
        z: torch.Tensor,
        dataset_id: int,
        library: torch.Tensor,
        dispersion: str,
        *cat_list: int,
    ):

        px = z
        if self.px_decoder_conditioned:
            px = self.px_decoder_conditioned(px, *cat_list)
        if self.px_decoder_final:
            px = self.px_decoder_final(px, *cat_list)

        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = torch.exp(library) * px_scale
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None

        return px_scale, px_r, px_rate, px_dropout

class common_decoder(nn.Module):
    def __init__(self,
                 input_dim,
                 sc_output_dim,
                 spa_output_dim,
                 hidden_dim: int = 256,
                 dropout_rate: int = 0):
        super(common_decoder, self).__init__()

        self.dropout = dropout_rate

        self.sc_decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )

        self.sc_raw_dim_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Linear(hidden_dim * 2, sc_output_dim)
        )

        self.spa_decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )

        self.spa_raw_dim_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Linear(hidden_dim * 2, spa_output_dim)
        )

    def forward(self, z: torch.Tensor, z_c: torch.Tensor):
        prior_sc = self.sc_decoder(z)
        sc_cluster_aux = self.sc_decoder(z_c)
        prior_spa = self.spa_decoder(z)
        spa_cluster_aux = self.spa_decoder(z_c)
        prior_sc_raw = self.sc_raw_dim_decoder(prior_sc)
        prior_spa_raw = self.spa_raw_dim_decoder(prior_spa)
        return prior_sc, prior_spa, prior_sc_raw, prior_spa_raw, sc_cluster_aux, spa_cluster_aux


# class Encoder(nn.Module):
#     def __init__(self, layer, hidden_dim, latent_dim, dropout_rate=0.1):
#         super(Encoder, self).__init__()
#
#         if len(layer) > 1:
#             self.fc = build_layer(layer, dropout_rate=dropout_rate)
#         self.layer = layer
#         self.fc_means = nn.Linear(hidden_dim, latent_dim)
#         self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
#
#     def reparameter(self, means, logvar):
#         if self.training:
#             std = logvar.mul(0.5).exp_()
#             eps = Variable(std.data.new(std.size()).normal_())
#             return eps.mul(std).add_(means)
#         else:
#             return means
#
#     def forward(self, x):
#         if len(self.layer) > 1:
#             h = self.fc(x)
#         else:
#             h = x
#         mean_x = self.fc_means(h)
#         logvar_x = self.fc_logvar(h)
#         latent = self.reparameter(mean_x, logvar_x)
#
#         return mean_x, logvar_x, latent


class Decoder_NB(nn.Module):
    def __init__(self, latent_dim, hidden_dim, raw_input_size, drop_rate):
        super(Decoder_NB, self).__init__()

        orderedDict = collections.OrderedDict()

        orderedDict['layer 0'] = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(drop_rate)
        )

        orderedDict['layer 1'] = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(drop_rate)
        )

        self.decoder = nn.Sequential(orderedDict)
        self.decoder_scale = nn.Linear(hidden_dim, raw_input_size)
        self.decoder_raw = nn.Linear(hidden_dim, raw_input_size)
        self.dropout = nn.Linear(hidden_dim, raw_input_size)

    def forward(self, z, library):
        latent = self.decoder(z)
        NB_scale = F.softmax(self.decoder_scale(latent), dim=0)

        NB_shape_r = torch.exp(library) * NB_scale
        NB_sample_rate = self.decoder_raw(latent)
        NB_sample_rate = torch.exp(NB_sample_rate)
        dropout_rate = self.dropout(latent)

        return NB_scale, NB_shape_r, NB_sample_rate, dropout_rate


class Decoder_Gaussian(nn.Module):
    def __init__(self, layer, hidden_dim, raw_input_size, drop_rate):
        super(Decoder_Gaussian, self).__init__()
        if len(layer) > 1:
            self.decoder = build_layer(layer, dropout_rate=drop_rate)

        self.decoder_raw = nn.Linear(hidden_dim, raw_input_size)
        self.layer = layer

    def forward(self, z):
        if len(self.layer) > 1:
            latent = self.decoder(z)
        else:
            latent = z

        reconstruct = self.decoder_raw(latent)

        final_x = F.softmax(reconstruct, dim=1)
