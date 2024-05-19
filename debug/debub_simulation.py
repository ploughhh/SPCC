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

adata = scanpy.read_h5ad("/home/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/simulation/tissue1/sc/sc_adata_final_withLR.h5ad")
adata_vis = scanpy.read_h5ad("/home/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/simulation/tissue1/spa/spa_adata_final_withLR.h5ad")

# scanpy.pp.filter_cells(adata_vis, min_counts=1)
# scanpy.pp.filter_cells(adata, min_counts = 1)

adata.spatial_metric=np.zeros((len(adata.obs), 1))
adata_vis.spatial_metric=np.zeros((len(adata_vis.obs), 1))

GIMVI.setup_anndata(adata, labels_key='leiden')
GIMVI.setup_anndata(adata_vis, labels_key='leiden')

model = GIMVI(adata, adata_vis)
model.train(200, use_gpu="cuda:0")

latent_seq, latent_spatial = model.get_latent_representation()
latents = [latent_seq, latent_spatial]

#concatenate to one latent representation
latent_representation = np.concatenate([latent_seq, latent_spatial])
batch_a = np.zeros(len(latent_seq))
batch_b = np.ones(len(latent_spatial))
batch = np.concatenate([batch_a, batch_b])

latent_adata = anndata.AnnData(latent_representation)
