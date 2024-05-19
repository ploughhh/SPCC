import louvain
import matplotlib.pyplot as plt
import scanpy
import numpy as np
import pandas as pd
import anndata
import torch
import sklearn
import scipy

from gimvi_data import GIMVI
from scvi.data import smfish, cortex
from src.utilities import preprocess_graph
from src.validation_test import get_train_set
from src.utilities import get_spatial_metric, fast_correlation, corrcoef
from src.sinkhorn_pointcloud import sinkhorn_loss
from src.visiualize import sigmoid
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.neighbors import NearestNeighbors
import seaborn as sns


net_sc = pd.read_csv("/home/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/simulation/tissue1/sc/LR_network_sim.csv").T
net_spa = pd.read_csv("/home/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/simulation/tissue1/spa/LR_network_sim.csv").T

adata = anndata.AnnData(np.expm1(net_sc))
adata_vis = anndata.AnnData(np.expm1(net_spa))

adata.spatial_metric=np.zeros((len(adata.obs), 1))
adata_vis.spatial_metric=np.zeros((len(adata_vis.obs), 1))

GIMVI.setup_anndata(adata)
GIMVI.setup_anndata(adata_vis)

model = GIMVI(adata, adata_vis, generative_distributions=["zinb", "zinb"], model_library_size=[True, True], log_variational=False, dispersion='gene-cell')
model.train(200, use_gpu="cuda:0")
# model.load_model("/home/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/simulation/tissue1/model/")

latent_seq, latent_spatial = model.get_latent_representation()
latents = [latent_seq, latent_spatial]

#concatenate to one latent representation
latent_representation = np.concatenate([latent_seq, latent_spatial])
batch_a = np.zeros(len(latent_seq))
batch_b = np.ones(len(latent_spatial))
batch = np.concatenate([batch_a, batch_b])

latent_adata = anndata.AnnData(latent_representation)

def batch_entropy_mixing_score(data, batches, n_neighbors=100, n_pools=100, n_samples_per_pool=100):
    """
    Calculate batch entropy mixing score

    Algorithm
    -----
        * 1. Calculate the regional mixing entropies at the location of 100 randomly chosen cells from all batches
        * 2. Define 100 nearest neighbors for each randomly chosen cell
        * 3. Calculate the mean mixing entropy as the mean of the regional entropies
        * 4. Repeat above procedure for 100 iterations with different randomly chosen cells.

    Parameters
    ----------
    data
        np.array of shape nsamples x nfeatures.
    batches
        batch labels of nsamples.
    n_neighbors
        The number of nearest neighbors for each randomly chosen cell. By default, n_neighbors=100.
    n_samples_per_pool
        The number of randomly chosen cells from all batches per iteration. By default, n_samples_per_pool=100.
    n_pools
        The number of iterations with different randomly chosen cells. By default, n_pools=100.

    Returns
    -------
    Batch entropy mixing score
    """

    #     print("Start calculating Entropy mixing score")
    def entropy(batches):
        p = np.zeros(N_batches)
        adapt_p = np.zeros(N_batches)
        a = 0
        for i in range(N_batches):
            p[i] = np.mean(batches == batches_[i])
            a = a + p[i] / P[i]
        entropy = 0
        for i in range(N_batches):
            adapt_p[i] = (p[i] / P[i]) / a
            entropy = entropy - adapt_p[i] * np.log(adapt_p[i] + 10 ** -8)
        return entropy

    n_neighbors = min(n_neighbors, len(data) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(data)
    kmatrix = nne.kneighbors_graph(data) - scipy.sparse.identity(data.shape[0])

    score = 0
    batches_ = np.unique(batches)
    N_batches = len(batches_)
    if N_batches < 2:
        raise ValueError("Should be more than one cluster for batch mixing")
    P = np.zeros(N_batches)
    for i in range(N_batches):
        P[i] = np.mean(batches == batches_[i])
    for t in range(n_pools):
        indices = np.random.choice(np.arange(data.shape[0]), size=n_samples_per_pool)
        score += np.mean([entropy(batches[kmatrix[indices].nonzero()[1]
        [kmatrix[indices].nonzero()[0] == i]])
                          for i in range(n_samples_per_pool)])
    Score = score / float(n_pools)
    return Score / float(np.log2(N_batches))



# score = 0.6866839419910781
score = batch_entropy_mixing_score(latent_representation, batch)

import umap
import matplotlib.pyplot as plt

lat1 = umap.UMAP(metric='correlation').fit_transform(latents[0])
lat2 = umap.UMAP(metric='correlation').fit_transform(latents[1])
fig, ax = plt.subplots(figsize=(50, 40))
contour_c='#444444'
# plt.xlim([np.min(lat1[:,0])-50, np.max(lat1[:,0])+50])
# plt.ylim([np.min(lat1[:,1])-50, np.max(lat1[:,1])+50])
labelsize = 20
plt.xlabel('UMAP 1', fontsize=labelsize)
plt.ylabel('UMAP 2', fontsize=labelsize)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.scatter(lat1[:, 0], lat1[:, 1], lw=0, c='#D55E00', label='sc', alpha=1.0, s=180, marker="o", edgecolors='k', linewidth=2)
plt.scatter(lat2[:, 0], lat2[:, 1], label='spatial')
leg = plt.legend(prop={'size': labelsize}, loc='upper right', markerscale=2.00)
leg.get_frame().set_alpha(0.9)
plt.setp(ax, xticks=[], yticks=[])
plt.show()

# scRNA latent space cluster, with no distance weight
from scipy.spatial import distance_matrix
from src.build_cell_communication import knn_graph
dmat_sc = distance_matrix(net_sc, net_sc)
G_sc = knn_graph(dmat_sc, 10)

partition_sc = louvain.find_partition(G_sc,
                                      louvain.RBConfigurationVertexPartition)
n_cluster_sc = len(partition_sc)

for i in range(n_cluster_sc):
    ids = partition_sc[i]
    plt.scatter(net_sc.iloc[ids, 0], net_sc.iloc[ids, 1], label=str(i + 1), linewidth=0, s=15)
plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
plt.axis('off')
plt.tight_layout()
plt.show()

# lr_list = pd.DataFrame(data={'ligand': ['L1', 'L2', 'L3', 'L4', 'L5', 'L6'],
#                              'receptor': []})

# ground truth gene correlation compared to generated gene correlation
imputed_values = model.get_imputed_values()
sc_gen = imputed_values[0]
gen_cor = sc_gen.T @ sc_gen
orig_cor = (net_sc.T @ net_sc).to_numpy()
from sklearn import preprocessing
min_max_normalizer = preprocessing.MinMaxScaler(feature_range=(0,1))
gen_cor = min_max_normalizer.fit_transform(gen_cor)
orig_cor = min_max_normalizer.fit_transform(orig_cor)
# make heatmap comparison
fig, ax = plt.subplots(1, 2)
# sns.set_context({'figure.figsize':[20, 20]})
sns.heatmap(orig_cor, ax=ax[0], xticklabels=net_sc.columns, yticklabels=net_sc.columns)
sns.heatmap(gen_cor, ax=ax[1], xticklabels=net_sc.columns, yticklabels=net_sc.columns)
# plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], labels=net_sc.columns)
# plt.yticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], labels=net_sc.columns)
plt.show()