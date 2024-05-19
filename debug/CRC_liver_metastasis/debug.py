import matplotlib.pyplot as plt
import seaborn as sns
import scanpy
import numpy as np
import pandas as pd
import anndata
import torch
import sklearn
import scipy
import os

from gimvi import GIMVI
from src.utilities import preprocess_graph, read_visium
from src.validation_test import get_train_set
from src.utilities import get_spatial_metric, fast_correlation, corrcoef
from src.sinkhorn_pointcloud import sinkhorn_loss
from src.visiualize import sigmoid
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.neighbors import NearestNeighbors
from src.utilities import knn_weight
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from spa_decom.spatial_decomposition import spatial_range_limtation, get_view_information
from src.build_cell_communication import phi_exp

# adata = scanpy.read_csv("/data03/WTG/spascer/38_output/raw/Zhangzemin_CRC_liver_metastasis/GSE164522_CRLM_MT_expression.csv").T
# adata = scanpy.read_csv("/data03/WTG/spascer/38_output/raw/Zhangzemin_CRC_liver_metastasis/GSE164522_CRLM_MT_expression.csv").T
adata = scanpy.read_h5ad('/data03/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/benchmarking/our_method/CRC_liver_metastasis/PT_data/sc.h5ad')
adata = scanpy.read_h5ad('/data03/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/benchmarking/our_method/CRC_liver_metastasis/MT/MT_data/sc.h5ad')
adata_vis = scanpy.read_h5ad("/data03/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/benchmarking/our_method/CRC_liver_metastasis/PT_data/spa.h5ad")
adata_vis = scanpy.read_h5ad('/data03/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/benchmarking/our_method/CRC_liver_metastasis/MT/MT_data/spa.h5ad')
scanpy.pp.filter_cells(adata, min_genes=1500)
def preprocess(adata):
    scanpy.pp.filter_genes(adata, min_cells=8)


    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    scanpy.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    scanpy.pp.normalize_total(adata, target_sum=1e4)
    scanpy.pp.log1p(adata)
    scanpy.pp.highly_variable_genes(adata, min_mean=0, max_mean=3, min_disp=-0.5)
    # scanpy.pl.highly_variable_genes(adata)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    # scanpy.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    scanpy.pp.scale(adata, max_value=10)

    # scanpy.tl.pca(adata, svd_solver='arpack')
    # scanpy.pl.pca(adata, color='celltype_major')
    scanpy.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    scanpy.tl.leiden(adata)
    # scanpy.tl.paga(adata)
    # scanpy.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
    # scanpy.tl.umap(adata, init_pos='paga')
    # scanpy.pl.umap(adata, color='celltype_major')

    scanpy.tl.rank_genes_groups(adata, 'celltype_major', method='t-test')
    # scanpy.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
    celltype = adata.obs['celltype_major'].drop_duplicates().to_list()
    rank_genes = []
    for i in celltype:
        rank_genes.append(adata.uns['rank_genes_groups']['names'][i][0:20])
    return rank_genes


rank_genes = preprocess(adata_sub)
train_genes = [gene for cluster in rank_genes for gene in cluster]
train_gene = pd.read_csv("/data03/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/benchmarking/our_method/CRC_liver_metastasis/PT_data/train_gene.csv", sep='\t', index_col='Unnamed: 0')
train_gene = train_gene.iloc[:,0].to_list()
train_genes = list(train_genes.iloc[:, 0].values)
spa_gene_name = list(adata_vis.var_names)
train_gene = list(set(train_genes).intersection(spa_gene_name))
train_gene = list(set(train_gene).intersection(adata.var_names))
# adata.obs_names.str.extract(r'\.(\d+)').astype(int)[0].drop_duplicates()
# adata_sub = adata[adata.obs_names.str.endswith(('.5', '.9', '.10', '.16'))]
adata = adata[:, train_gene].copy()
adata_vis.var_names_make_unique()
adata_vis = adata_vis[:, train_gene].copy()

adata_vis.obs.rename({'second_type': 'celltype_major'}, axis=1, inplace=True)
# meta = pd.read_csv("/data03/WTG/spascer/38_output/raw/Zhangzemin_CRC_liver_metastasis/GSE164522_CRLM_metadata.csv", sep=',', index_col='Unnamed: 0')
# adata_sub.obs = meta.loc[meta.index.isin(adata_sub.obs_names)]
# GIMVI.setup_anndata(adata_sub, labels_key='celltype_major')
# GIMVI.setup_anndata(adata_vis, labels_key='Classification')
GIMVI.setup_anndata(adata, labels_key='celltype_major')
GIMVI.setup_anndata(adata_vis, labels_key='celltype_major')
model = GIMVI(adata, adata_vis, generative_distributions=['zinb', 'zinb'], model_library_size=[True, True])
# model.load_model("/data2/WTG/spascer_data/49/raw/trained_model/p_CID3946_1142243F/")

model.train(250,
            use_gpu="cuda:0",
            celltype_classifier=True,
            kappa=2.5
            )
latent_seq, latent_spatial = model.get_latent_representation()
latents = [latent_seq, latent_spatial]
latent_representation = np.concatenate([latent_seq, latent_spatial])
batch_a = np.zeros(len(latent_seq))
batch_b = np.ones(len(latent_spatial))
batch = np.concatenate([batch_a, batch_b])
from src.utilities import batch_entropy_mixing_score
score = batch_entropy_mixing_score(latent_representation, batch) # score = 0.5123512570756076

lr_file = pd.read_csv("/data03/WTG/spascer/38_output/raw/Zhangzemin_CRC_liver_metastasis/MT/out/means.txt", sep='\t')
lr_file = lr_file.dropna(subset=["gene_a"])
lr_file = lr_file.dropna(subset=["gene_b"])
lr_list = lr_file[['gene_a', 'gene_b']]
ligand_gene = lr_list.loc[:, 'gene_a'].tolist()
receptor_gene = lr_list.loc[:, 'gene_b'].tolist()

def get_spatial_metric(sc_adata, spa_adata):
    position = pd.DataFrame([spa_adata.obs['array_row'], spa_adata.obs['array_col']])
    dis = pdist(position.T, metric='euclidean')
    spa_dis = squareform(dis)
    spa_adata.spatial_metric = spa_dis
    sc_adata.spatial_metric = np.zeros([sc_adata.n_obs, sc_adata.n_vars])

    return spa_adata, sc_adata

adata_vis, adata = get_spatial_metric(adata_sub, adata_vis)

from rpy2.robjects.packages import importr
# from rpy2.robjects.vectors import StrVector
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

# from rpy2.robjects import r

r_rfsrc = importr('randomForestSRC')

pandas2ri.activate()
spa_coord = adata_vis.obs[['array_row', 'array_col']]
spa_coord.reset_index(inplace=True)
spa_coord.drop('index', axis=1, inplace=True)
latent_spatial = pd.DataFrame(latent_spatial)
latent_spatial.reset_index(inplace=True)
latent_spatial.drop('index', axis=1, inplace=True)
st_train = pd.concat([latent_spatial, spa_coord], axis=1)
r_st_train = pandas2ri.py2rpy(st_train)

latent_representation = pd.DataFrame(latent_representation)
r_test_data = pandas2ri.py2rpy(latent_representation)

# 定义 formula 和 distance 参数
formula = 'Multivar(array_row, array_col) ~ .'
distance = 'all'
formula = robjects.Formula(formula)

rf_train = r_rfsrc.rfsrc(formula=formula, data=r_st_train, block_size=5, ntree=1000)
rf_pred = r_rfsrc.predict_rfsrc(rf_train, newdata=r_test_data, distance='all')
distance = rf_pred.rx2('distance')
rf_pred_dist = distance[0:len(latent_seq), len(latent_seq):len(latent_representation)]


