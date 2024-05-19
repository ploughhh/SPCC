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

#   TNBC sample
p_CID3946 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/data03/WTG/spascer/49/raw/sc_cellphonedb/CID3946_TNBC/sc.h5ad")
p_CID3963 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/data03/WTG/spascer/49/raw/sc_cellphonedb/CID3963_TNBC/sc.h5ad")
p_CID4513 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID4513/sc.h5ad")
p_CID4515 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID4515/sc.h5ad")

p_CID4465 = scanpy.read_h5ad("/data03/WTG/spascer/49/raw/sc_cellphonedb/CID4465/sc.h5ad")

p_CID4495 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID4495/sc.h5ad")
p_CID44041 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID44041/sc.h5ad")

p_CID44971 = scanpy.read_h5ad("/data03/WTG/spascer/49/raw/sc_cellphonedb/CID44971/sc.h5ad")

p_CID44991 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID44991/sc.h5ad")

# ER
p_CID3941 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID3941/sc.h5ad")
p_CID3948 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID3948/sc.h5ad")
p_CID4040 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID4040/sc.h5ad")
p_CID4067 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID4067/sc.h5ad")

p_CID4290A = scanpy.read_h5ad("/data03/WTG/spascer/49/raw/sc_cellphonedb/CID4290A/sc.h5ad")

p_CID4398 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID4398/sc.h5ad")
p_CID4461 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID4461/sc.h5ad")
p_CID4463 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID4463/sc.h5ad")
p_CID4471 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID4471/sc.h5ad")

p_CID4535 = scanpy.read_h5ad("/data03/WTG/spascer/49/raw/sc_cellphonedb/CID4535/sc.h5ad")

p_CID4530N = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID4530N/sc.h5ad")

p_CID4066 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID4066/sc.h5ad")
p_CID4515 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID4515/sc.h5ad")
p_CID4523 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID4523/sc.h5ad")
p_CID3586 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID3586/sc.h5ad")
p_CID3838 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID3838/sc.h5ad")
p_CID3921 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID3921/sc.h5ad")
p_CID45171 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/sc_cellphonedb/CID45171/sc.h5ad")

# TNBC sample
adata = p_CID3946.copy()
adata = p_CID3963.copy()
adata = p_CID4513.copy()
adata = p_CID4515.copy()

adata = p_CID4465.copy()

adata = p_CID4495.copy()
adata = p_CID44041.copy()

adata = p_CID44971.copy()

adata = p_CID44991.copy()

# ER
adata = p_CID3941.copy()
adata = p_CID3948.copy()
adata = p_CID4040.copy()
adata = p_CID4067.copy()

adata = p_CID4290A.copy()

adata = p_CID4398.copy()
adata = p_CID4461.copy()
adata = p_CID4463.copy()
adata = p_CID4471.copy()

adata = p_CID4535.copy()

adata = p_CID4530N.copy()

adata = p_CID4066.copy()
adata = p_CID4523.copy()
adata = p_CID3586.copy()
adata = p_CID3838.copy()
adata = p_CID3921.copy()
adata = p_CID45171.copy()

# TNBC spatial
CID_1142243F_spa = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/spatial/1142243F_spa_TNBC.h5ad")
CID_1160920F_spa = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/spatial/1160920F_spa_TNBC.h5ad")

CID_4465_spa = scanpy.read_h5ad("/data03/WTG/spascer/49/raw/spatial/CID4465_spa_TNBC.h5ad")

CID_44971_spa = scanpy.read_h5ad("/data03/WTG/spascer/49/raw/spatial/CID44971_spa.h5ad")

adata_vis = CID_1142243F_spa.copy()
adata_vis = CID_1160920F_spa.copy()

adata_vis = CID_4465_spa.copy()

adata_vis = CID_44971_spa.copy()

# ER spatial
CID4290_spa = scanpy.read_h5ad("/data03/WTG/spascer/49/raw/spatial/CID4290_spa.h5ad")
adata_vis = CID4290_spa.copy()

CID_4535_spa = scanpy.read_h5ad("/data03/WTG/spascer/49/raw/spatial/CID4535_spa.h5ad")
adata_vis = CID_4535_spa.copy()

# preprocess
# scanpy.pl.highest_expr_genes(adata, n_top=20, )
# scanpy.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
#              jitter=0.4, multi_panel=True)
# scanpy.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
# scanpy.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

# adata = adata[adata.obs.n_genes_by_counts < 3000, :]
# adata = adata[adata.obs.pct_counts_mt < 3, :]
scanpy.pp.filter_cells(adata, min_genes=2938)


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
    scanpy.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    scanpy.pp.scale(adata, max_value=10)

    # scanpy.tl.pca(adata, svd_solver='arpack')
    # scanpy.pl.pca(adata, color='celltype_major')
    scanpy.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    scanpy.tl.leiden(adata)
    # scanpy.tl.paga(adata)
    # scanpy.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
    # scanpy.tl.umap(adata, init_pos='paga')
    # scanpy.pl.umap(adata, color='celltype_major')

    scanpy.tl.rank_genes_groups(adata, 'leiden', method='t-test')
    # scanpy.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
    rank_genes = []
    for i in range(len(adata.uns.data['rank_genes_groups']['names'].dtype)):
        rank_genes.append(adata.uns.data['rank_genes_groups']['names'][str(i)][0:25])
    return rank_genes


rank_genes = preprocess(adata)
# spatial preprocessing
# CID_1142243F_spa = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/spatial/1142243F_spa_TNBC.h5ad")
# adata_vis = CID_1142243F_spa.copy()
train_genes = [gene for cluster in rank_genes for gene in cluster]
spa_gene_name = list(adata_vis.var_names)
train_gene = list(set(train_genes).intersection(spa_gene_name))

train_gene = pd.read_csv('/data03/WTG/spascer/49/raw/44971/train_gene.txt', sep='\t', index_col='Unnamed: 0')
train_gene = train_gene.iloc[:, 0].to_list()
# TNBC
adata = p_CID3946.copy()
adata = p_CID3963.copy()
adata = p_CID4513.copy()
adata = p_CID4515.copy()
adata = p_CID4465.copy()
adata = p_CID4495.copy()
adata = p_CID44041.copy()

# ER
adata = p_CID4290A.copy()
adata = p_CID4535.copy()

adata = adata[:, train_gene].copy()
adata_vis = adata_vis[:, train_gene].copy()
adata_vis = read_visium("/data2/WTG/spascer_data/49/raw/spatial/spatial/1142243F_spatial/", "1142243F", adata_vis,
                        False)
adata_vis = read_visium("/data2/WTG/spascer_data/49/raw/spatial/spatial/1160920F_spatial/", "1160920F", adata_vis,
                        False)
adata_vis = read_visium("/data03/WTG/spascer/49/raw/spatial/spatial/CID4290_spatial/", "4290", adata_vis, False)
adata_vis = read_visium("/data03/WTG/spascer/49/raw/spatial/spatial/CID4465_spatial/", "4465", adata_vis, False)
adata_vis = read_visium("/data03/WTG/spascer/49/raw/spatial/spatial/CID4535_spatial/", "4535", adata_vis, False)
adata_vis = read_visium("/data03/WTG/spascer/49/raw/spatial/spatial/CID44971_spatial/", "44971", adata_vis, False)

# drop spatial na
a = pd.DataFrame(adata_vis.obs['Classification'])
a.dropna(inplace=True)
adata_vis = adata_vis[a.index, :]
adata_vis = adata_vis.copy()
del a

# adata = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/trained_model/p_CID3946_1142243F/p_CID3946_for_train.h5ad")
# adata_vis = scanpy.read_h5ad("/data2/WTG/spascer_data/49/raw/trained_model/p_CID3946_1142243F/1142243F_for_training_spa.h5ad")
GIMVI.setup_anndata(adata, labels_key='celltype_major')
GIMVI.setup_anndata(adata_vis, labels_key='Classification')
model = GIMVI(adata, adata_vis, generative_distributions=['zinb', 'zinb'], model_library_size=[True, True])
# model.load_model("/data2/WTG/spascer_data/49/raw/trained_model/p_CID3946_1142243F/")

model.train(250,
            use_gpu="cuda:0",
            kappa=2.5
            )

latent_seq, latent_spatial = model.get_latent_representation()
# latents = [latent_seq, latent_spatial]
latent_representation = np.concatenate([latent_seq, latent_spatial])
batch_a = np.zeros(len(latent_seq))
batch_b = np.ones(len(latent_spatial))
batch = np.concatenate([batch_a, batch_b])


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


score = batch_entropy_mixing_score(latent_representation, batch)  # score = 0.5123512570756076

lr_file = pd.read_csv("/data2/WTG/spascer_data/49/data03/WTG/spascer/49/raw/sc_cellphonedb/CID3946_TNBC/out/means.txt",
                      sep='\t')
lr_file = lr_file.dropna(subset=["gene_a"])
lr_file = lr_file.dropna(subset=["gene_b"])
lr_list = lr_file[['gene_a', 'gene_b']]
ligand_gene = lr_list.loc[:, 'gene_a'].tolist()
receptor_gene = lr_list.loc[:, 'gene_b'].tolist()

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

mapping = pd.DataFrame(rf_pred_dist, index=adata.obs_names, columns=adata_vis.obs_names)
index_top10 = np.argsort(rf_pred_dist, axis=1)[:, :10]
index_max = np.argmax(-rf_pred_dist, axis=1)
# make a function here
x_psu_coord = np.zeros(len(index_max))
y_psu_coord = np.zeros(len(index_max))

x_coord = np.array(adata_vis.obs['array_row'])
y_coord = np.array(adata_vis.obs['array_col'])
x_pxl_coord = adata_vis.obsm['spatial'][:, 0]
y_pxl_coord = adata_vis.obsm['spatial'][:, 1]

# region = adata_vis.obs['regions']
spa_celltype = adata_vis.obs['Classification']
x_psu_coord = np.take(x_coord, index_max)
y_psu_coord = np.take(y_coord, index_max)
x_psu_pxl_coord = np.take(x_pxl_coord, index_max)
y_psu_pxl_coord = np.take(y_pxl_coord, index_max)
psu_celltype = np.take(spa_celltype, index_max)
psu_celltype = psu_celltype.to_numpy()
# psu_region = np.take(region, index_max)


adata.obs['array_row'] = x_psu_coord
adata.obs['array_col'] = y_psu_coord
adata.obs['spadata_celltype'] = psu_celltype
adata.obs['pxl_row_in_fullres'] = x_psu_pxl_coord
adata.obs['pxl_col_in_fullres'] = y_psu_pxl_coord

sc_coord_4465 = pd.DataFrame()
sc_coord_4465['array_row'] = adata.obs['array_row']
sc_coord_4465['array_col'] = adata.obs['array_col']
sc_coord_4465['sc_celltype'] = adata.obs['celltype_major']
sc_coord_4465['spa_celltype'] = adata.obs['spadata_celltype']
sc_coord_4465['sc_celltype_subset'] = adata.obs['celltype_subset']
sc_coord_4465['sc_celltype_minor'] = adata.obs['celltype_minor']

sc_coord_44971 = pd.DataFrame()
sc_coord_44971['array_row'] = adata.obs['array_row']
sc_coord_44971['array_col'] = adata.obs['array_col']
sc_coord_44971['sc_celltype'] = adata.obs['celltype_major']
sc_coord_44971['spa_celltype'] = adata.obs['spadata_celltype']
sc_coord_44971['sc_celltype_subset'] = adata.obs['celltype_subset']
sc_coord_44971['sc_celltype_minor'] = adata.obs['celltype_minor']

sc_coord_4290 = pd.DataFrame()
sc_coord_4290['array_row'] = adata.obs['array_row']
sc_coord_4290['array_col'] = adata.obs['array_col']
sc_coord_4290['sc_celltype'] = adata.obs['celltype_major']
sc_coord_4290['spa_celltype'] = adata.obs['spadata_celltype']
sc_coord_4290['sc_celltype_subset'] = adata.obs['celltype_subset']
sc_coord_4290['sc_celltype_minor'] = adata.obs['celltype_minor']

sc_coord_4535 = pd.DataFrame()
sc_coord_4535['array_row'] = adata.obs['array_row']
sc_coord_4535['array_col'] = adata.obs['array_col']
sc_coord_4535['sc_celltype'] = adata.obs['celltype_major']
sc_coord_4535['spa_celltype'] = adata.obs['spadata_celltype']
sc_coord_4535['sc_celltype_subset'] = adata.obs['celltype_subset']
sc_coord_4535['sc_celltype_minor'] = adata.obs['celltype_minor']

scanpy.pl.spatial()
subset = sc_coord_4465[(sc_coord_4465['sc_celltype'] == 'Cancer Epithelial') &
                       (sc_coord_4465['spa_celltype'].isin(['Invasive cancer + stroma + lymphocytes']))]
subset = sc_coord_44971[(sc_coord_44971['sc_celltype'] == 'Cancer Epithelial') &
                        (sc_coord_44971['spa_celltype'].isin(['Invasive cancer + lymphocytes', 'DCIS']))]
subset = sc_coord_4290[(sc_coord_4290['sc_celltype'] == 'Cancer Epithelial') &
                       (sc_coord_4290['spa_celltype'].isin(
                           ['Invasive cancer + stroma', 'Invasive cancer + stroma + lymphocytes']))]
subset = sc_coord_4535[(sc_coord_4535['sc_celltype'] == 'Cancer Epithelial') &
                       (sc_coord_4535['spa_celltype'].isin(
                           ['Invasive cancer + adipose tissue + lymphocytes', 'Invasive cancer + lymphocytes',
                            'Uncertain', 'Invasive cancer']))]
whole_subset = sc_coord_4465['sc_celltype'].value_counts()['Cancer Epithelial']
whole_subset = sc_coord_44971['sc_celltype'].value_counts()['Cancer Epithelial']
whole_subset = sc_coord_4290['sc_celltype'].value_counts()['Cancer Epithelial']
whole_subset = sc_coord_4535['sc_celltype'].value_counts()['Cancer Epithelial']

accuracy = len(subset) / whole_subset
# TPR sensitive
FN_subset = sc_coord_44971[(sc_coord_44971['sc_celltype'] == 'Cancer Epithelial') &
                           (~sc_coord_44971['spa_celltype'].isin(['Invasive cancer + lymphocytes', 'DCIS']))]
TP = len(FN_subset)
FN = whole_subset - TP
TPR = TP / (TP + FN)

exhausted_T_gene = ['INPP5F']
exhausted_T = pd.read_csv(
    '/data03/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/benchmarking/our_method/breast_cancer_data/exhausted_cd8_T_wu_etc.txt',
    header=None)
exhausted_T.iloc[0] = exhausted_T.iloc[0].astype(str)
exhausted_T_gene = exhausted_T.iloc[0].tolist()
exhausted_T_gene.append('INPP5F')

exhausted_T = pd.read_csv(
    '/data03/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/benchmarking/our_method/breast_cancer_data/exhausted_cd8_T_wu_etc.txt',
    header=None).T

exhausted_T.rename({0: 'member'}, inplace=True, axis=1)
exhausted_T['description'] = 'predefined'
exhausted_T['name'] = 'Exhausted CD8 T pathway'
exhausted_T.sort_index(axis=1, inplace=True)

exp = pd.DataFrame(p_CID44971.X.toarray(), index=p_CID44971.obs_names, columns=p_CID44971.var_names)
import gseapy as gp
from gseapy.plot import barplot, dotplot
from gseapy import ssgsea

scanpy.pl.spatial()
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from GSVA import gsva

dis_mat = cdist(sc_coord_44971[['array_row', 'array_col']], sc_coord_44971[['array_row', 'array_col']])
celltype_list = sc_coord_44971['sc_celltype_subset'].unique()
cancer_idx = np.where(sc_coord_44971['sc_celltype'] == 'Cancer Epithelial')[0]
celltype_distances = {}
for celltype in celltype_list:
    celltype_idx = np.where(sc_coord_44971['sc_celltype_subset'] == celltype)[0]
    distances_to_cancer = dis_mat[celltype_idx[:, None], cancer_idx]
    celltype_distances[celltype] = distances_to_cancer

T_celltype_idx = np.where(sc_coord_44971['sc_celltype_subset'] == 'T_cells_c8_CD8+_LAG3')[0]
p_CID44971_T_sub = p_CID44971[T_celltype_idx, :].copy()

# 将 T-cells 距离癌症细胞的距离数据分为 proximal 和 distal 两组
from tqdm import tqdm

tcell_distances = celltype_distances['T_cells_c8_CD8+_LAG3']
tcell_distances_min = np.min(tcell_distances, axis=1)
tcell_distances_mean = np.mean(tcell_distances, axis=1)
tcell_distances_meansorted_indices = np.argsort(tcell_distances_mean)

tcell_distances_min_unique = np.unique(tcell_distances_min)
score = []
tcell_distances_min_unique[0] = 1.5
for i in tqdm(tcell_distances_min_unique):
    tcell_distances_class = np.where(tcell_distances_min < i, 'proximal', 'distal')

    # 将分组标签存储在 p_CID44971_T_sub.obs['distance_classification'] 中
    p_CID44971_T_sub.obs['distance_classification'] = pd.Categorical(
        values=tcell_distances_class, categories=['proximal', 'distal']
    )
    p_CID44971_T_sub = p_CID44971_T_sub[:, exhausted_T_gene].copy()
    scanpy.tl.rank_genes_groups(p_CID44971_T_sub, 'distance_classification', groups=['proximal', 'distal'],
                                method='t-test_overestim_var')
    de_result = p_CID44971_T_sub.uns['rank_genes_groups']
    de_groups = de_result['names'].dtype.names
    proxi_genes = de_result['names']['proximal']  # 假设你只想获得前10个基因
    distal_genes = de_result['names']['distal']  # 假设你只想获得前10个基因
    de_score_proximal = pd.DataFrame({'Gene': proxi_genes, 'Score': de_result['scores']['proximal']})
    de_score_distal = pd.DataFrame({'Gene': distal_genes, 'Score': de_result['scores']['distal']})

    prerank_gene = gp.prerank(rnk=de_score_proximal,
                              gene_sets='/data03/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/benchmarking/our_method/breast_cancer_data/predefined_exhausted_cd8_T_geneset_Zheng.gmt',
                              outdir='/data03/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/benchmarking/our_method/breast_cancer_data/gsea_results',
                              permutation_num=1000, method='signal_to_noise')
    score.append(prerank_gene.res2d['ES'])
score = [x.to_numpy() for x in score]
score = np.squeeze(np.array(score)).astype(float)
score = np.delete(score, 3)
tcell_distances_min_unique = np.delete(tcell_distances_min_unique, 3)
from matplotlib.collections import LineCollection

cmap = plt.cm.get_cmap('RdYlBu')
norm = plt.Normalize(vmin=score.min(), vmax=score.max())
colors = cmap(norm(score))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axhline(y=0, color='black', linestyle='-')
ax.scatter(tcell_distances_min_unique, score, c=colors, cmap=cmap, edgecolor='black')
points = np.array([tcell_distances_min_unique, score]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=cmap, norm=norm, colors=colors)
lc.set_array(-score)
lc.set_linewidth(15)
lc.set_alpha(1)

# 每个点垂直向下的直线
for i in range(len(score)):
    color = colors[i]
    ax.plot([tcell_distances_min_unique[i], tcell_distances_min_unique[i]], [0, score[i]], color=color, linestyle='--',
            linewidth=0.5)

# 设置图形属性
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
ax.set_axisbelow(True)
ax.grid(axis='y', color='white')
ax.set_xlabel('From proximal distance -> distal distance')
ax.set_ylabel('Enrichment scores(NES)')

# 去掉背景颜色
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

plt.show()

dis_mat = cdist(sc_coord_44971[['array_row', 'array_col']], sc_coord_44971[['array_row', 'array_col']])
celltype_list = sc_coord_44971['sc_celltype'].unique()
cancer_idx = np.where(sc_coord_44971['sc_celltype'] == 'Cancer Epithelial')[0]
celltype_distances = {}
for celltype in celltype_list:
    celltype_idx = np.where(sc_coord_44971['sc_celltype'] == celltype)[0]
    distances_to_cancer = dis_mat[celltype_idx[:, None], cancer_idx]
    celltype_distances[celltype] = distances_to_cancer

T_celltype_idx = np.where(sc_coord_44971['sc_celltype'] == 'T-cells')[0]
p_CID44971_T_sub = p_CID44971[T_celltype_idx, :].copy()

# 将 T-cells 距离癌症细胞的距离数据分为 proximal 和 distal 两组
from tqdm import tqdm

tcell_distances = celltype_distances['T-cells']
tcell_distances_min = np.min(tcell_distances, axis=1)

tcell_distances_min_unique = np.unique(tcell_distances_min)
score = []
tcell_distances_min_unique[0] = 1
for i in tqdm(tcell_distances_min_unique):
    tcell_distances_class = np.where(tcell_distances_min < i, 'proximal', 'distal')

    # 将分组标签存储在 p_CID44971_T_sub.obs['distance_classification'] 中
    p_CID44971_T_sub.obs['distance_classification'] = pd.Categorical(
        values=tcell_distances_class, categories=['proximal', 'distal']
    )
    p_CID44971_T_sub = p_CID44971_T_sub[:, exhausted_T_gene].copy()
    scanpy.tl.rank_genes_groups(p_CID44971_T_sub, 'distance_classification', groups=['proximal', 'distal'],
                                method='t-test_overestim_var')
    de_result = p_CID44971_T_sub.uns['rank_genes_groups']
    de_groups = de_result['names'].dtype.names
    proxi_genes = de_result['names']['proximal']  # 假设你只想获得前10个基因
    distal_genes = de_result['names']['distal']  # 假设你只想获得前10个基因
    de_score_proximal = pd.DataFrame({'Gene': proxi_genes, 'Score': de_result['scores']['proximal']})
    de_score_distal = pd.DataFrame({'Gene': distal_genes, 'Score': de_result['scores']['distal']})

    prerank_gene = gp.prerank(rnk=de_score_proximal,
                              gene_sets='/data03/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/benchmarking/our_method/breast_cancer_data/predefined_exhausted_cd8_T_geneset_Zheng.gmt',
                              outdir='/data03/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/benchmarking/our_method/breast_cancer_data/gsea_results',
                              permutation_num=1000, method='signal_to_noise')
    score.append(prerank_gene.res2d['ES'])
score = [x.to_numpy() for x in score]
score = np.squeeze(np.array(score)).astype(float)
# score = np.delete(score, 3)
tcell_distances_min_unique = np.delete(tcell_distances_min_unique, 43)
from matplotlib.collections import LineCollection

cmap = plt.cm.get_cmap('RdYlBu')
norm = plt.Normalize(vmin=score.min(), vmax=score.max())
colors = cmap(norm(score))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axhline(y=0, color='black', linestyle='-')
ax.scatter(tcell_distances_min_unique, score, c=colors, cmap=cmap, edgecolor='black')
points = np.array([tcell_distances_min_unique, score]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=cmap, norm=norm, colors=colors)
lc.set_array(-score)
lc.set_linewidth(15)
lc.set_alpha(1)

# 每个点垂直向下的直线
for i in range(len(score)):
    color = colors[i]
    ax.plot([tcell_distances_min_unique[i], tcell_distances_min_unique[i]], [0, score[i]], color=color, linestyle='--',
            linewidth=0.5)

# 设置图形属性
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
ax.set_axisbelow(True)
ax.grid(axis='y', color='white')
ax.set_xlabel('From proximal distance -> distal distance')
ax.set_ylabel('Enrichment scores(NES)')

# 去掉背景颜色
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

plt.show()

dis_mat = cdist(sc_coord_44971[['array_row', 'array_col']], sc_coord_44971[['array_row', 'array_col']])
celltype_list = sc_coord_44971['sc_celltype_minor'].unique()
cancer_idx = np.where(sc_coord_44971['sc_celltype'] == 'Cancer Epithelial')[0]
celltype_distances = {}
for celltype in celltype_list:
    celltype_idx = np.where(sc_coord_44971['sc_celltype_minor'] == celltype)[0]
    distances_to_cancer = dis_mat[celltype_idx[:, None], cancer_idx]
    celltype_distances[celltype] = distances_to_cancer

T_celltype_idx = np.where(sc_coord_44971['sc_celltype_minor'] == 'T cells CD8+')[0]
p_CID44971_T_sub = p_CID44971[T_celltype_idx, :].copy()

# 将 T-cells 距离癌症细胞的距离数据分为 proximal 和 distal 两组
from tqdm import tqdm

tcell_distances = celltype_distances['T cells CD8+']
tcell_distances_min = np.min(tcell_distances, axis=1)

tcell_distances_min_unique = np.unique(tcell_distances_min)
score = []
tcell_distances_min_unique[0] = 1.5
tcell_distances_min_unique[1] = 1.8
for i in tqdm(tcell_distances_min_unique):
    tcell_distances_class = np.where(tcell_distances_min < i, 'proximal', 'distal')

    # 将分组标签存储在 p_CID44971_T_sub.obs['distance_classification'] 中
    p_CID44971_T_sub.obs['distance_classification'] = pd.Categorical(
        values=tcell_distances_class, categories=['proximal', 'distal']
    )
    p_CID44971_T_sub = p_CID44971_T_sub[:, exhausted_T_gene].copy()
    scanpy.tl.rank_genes_groups(p_CID44971_T_sub, 'distance_classification', groups=['proximal', 'distal'],
                                method='t-test_overestim_var')
    de_result = p_CID44971_T_sub.uns['rank_genes_groups']
    de_groups = de_result['names'].dtype.names
    proxi_genes = de_result['names']['proximal']  # 假设你只想获得前10个基因
    distal_genes = de_result['names']['distal']  # 假设你只想获得前10个基因
    de_score_proximal = pd.DataFrame({'Gene': proxi_genes, 'Score': de_result['scores']['proximal']})
    de_score_distal = pd.DataFrame({'Gene': distal_genes, 'Score': de_result['scores']['distal']})

    prerank_gene = gp.prerank(rnk=de_score_proximal,
                              gene_sets='/data03/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/benchmarking/our_method/breast_cancer_data/predefined_exhausted_cd8_T_geneset_Zheng.gmt',
                              outdir='/data03/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/benchmarking/our_method/breast_cancer_data/gsea_results',
                              permutation_num=1000, method='signal_to_noise')
    score.append(prerank_gene.res2d['ES'])
score = [x.to_numpy() for x in score]
score = np.squeeze(np.array(score)).astype(float)
# score = np.delete(score, 3)
# tcell_distances_min_unique = np.delete(tcell_distances_min_unique,43)
from matplotlib.collections import LineCollection

cmap = plt.cm.get_cmap('RdYlBu')
norm = plt.Normalize(vmin=score.min(), vmax=score.max())
colors = cmap(norm(score))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axhline(y=0, color='black', linestyle='-')
ax.scatter(tcell_distances_min_unique, score, c=colors, cmap=cmap, edgecolor='black')
points = np.array([tcell_distances_min_unique, score]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=cmap, norm=norm, colors=colors)
lc.set_array(-score)
lc.set_linewidth(15)
lc.set_alpha(1)

# 每个点垂直向下的直线
for i in range(len(score)):
    color = colors[i]
    ax.plot([tcell_distances_min_unique[i], tcell_distances_min_unique[i]], [0, score[i]], color=color, linestyle='--',
            linewidth=0.5)

# 设置图形属性
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
ax.set_axisbelow(True)
ax.grid(axis='y', color='white')
ax.set_xlabel('From proximal distance -> distal distance')
ax.set_ylabel('Enrichment scores(NES)')

# 去掉背景颜色
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

plt.show()

adata.uns['spatial'] = adata_vis.uns['spatial']
adata.obsm['spatial'] = adata.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
scanpy.pl.spatial(adata, color='celltype_minor')

########################################################################################################################
########################################################################################################################
####################################### violin plot cell type distance to tumor   ######################################
########################################################################################################################
########################################################################################################################

dis_mat = cdist(sc_coord_44971[['array_row', 'array_col']], sc_coord_44971[['array_row', 'array_col']])
celltype_list = sc_coord_44971['sc_celltype'].unique()
cancer_idx = np.where(sc_coord_44971['sc_celltype'] == 'Cancer Epithelial')[0]
celltype_distances = {}
for celltype in celltype_list:
    celltype_idx = np.where(sc_coord_44971['sc_celltype'] == celltype)[0]
    distances_to_cancer = dis_mat[celltype_idx[:, None], cancer_idx]
    celltype_distances[celltype] = distances_to_cancer
cell_2_tumor_distance_mean = {}
for celltype in celltype_list:
    cell_2_tumor_distance_mean[celltype] = np.mean(celltype_distances[celltype], axis=1)

# 创建包含8个子图的Figure对象
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))

# 在每个子图上画出对应细胞类型的小提琴图
for ax, celltype in zip(axes.flat, celltype_list):
    if celltype == 'Cancer Epithelial':  # 跳过癌症细胞类型
        continue
    distances = cell_2_tumor_distance_mean[celltype]
    parts = ax.violinplot(distances, showmeans=False, showmedians=True)
    ax.set_title(celltype)
    ax.set_xticks([])
    ax.set_ylabel('Distance')
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    for line in ['cbars', 'cmaxes', 'cmins', 'cmedians']:
        parts[line].set_color('black')
    parts['cmedians'].set_linewidth(2)

# 去掉背景颜色和网格线
fig.patch.set_facecolor('white')
for ax in axes.flat:
    ax.set_facecolor('white')
    ax.grid(False)

# 调整子图之间的间距和边距
plt.subplots_adjust(hspace=0.5, wspace=0.3)
fig.tight_layout()

# 显示图形
plt.show()

########################################################################################################################
########################################################################################################################
################################## violin plot T cell subcluster distance to tumor   ###################################
########################################################################################################################
########################################################################################################################
# 画T细胞的10种亚群
import re

T_subset = sc_coord_44971[sc_coord_44971['sc_celltype_subset'].str.startswith('T_cells')][
    'sc_celltype_subset'].drop_duplicates().to_list()
T_subset = sorted(T_subset, key=lambda x: (int(re.findall(r'T_cells_c(\d+)_', x)[0]),
                                           'CD4+' in x,
                                           re.findall(r'_([A-Z0-9]+)$', x)[0]))
dis_mat = cdist(sc_coord_44971[['array_row', 'array_col']], sc_coord_44971[['array_row', 'array_col']])
cancer_idx = np.where(sc_coord_44971['sc_celltype'] == 'Cancer Epithelial')[0]
T_sub_celltype_distances = {}
for celltype in T_subset:
    celltype_idx = np.where(sc_coord_44971['sc_celltype_subset'] == celltype)[0]
    distances_to_cancer = dis_mat[celltype_idx[:, None], cancer_idx]
    T_sub_celltype_distances[celltype] = distances_to_cancer
T_sub_cell_2_tumor_distance_mean = {}
for celltype in T_subset:
    T_sub_cell_2_tumor_distance_mean[celltype] = np.mean(T_sub_celltype_distances[celltype], axis=1)

# 创建包含8个子图的Figure对象
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 8))
# 在每个子图上画出对应细胞类型的小提琴图
for ax, celltype in zip(axes.flat, T_subset):
    distances = T_sub_cell_2_tumor_distance_mean[celltype]
    parts = ax.violinplot(distances, showmeans=False, showmedians=True)
    ax.set_title(celltype)
    ax.set_xticks([])
    ax.set_ylabel('Distance')
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    for line in ['cbars', 'cmaxes', 'cmins', 'cmedians']:
        parts[line].set_color('black')
    parts['cmedians'].set_linewidth(2)
# 去掉背景颜色和网格线
fig.patch.set_facecolor('white')
for ax in axes.flat:
    ax.set_facecolor('white')
    ax.grid(False)
# 调整子图之间的间距和边距
plt.subplots_adjust(hspace=0.5, wspace=0.3)
fig.tight_layout()
# 显示图形
plt.show()

########################################################################################################################
########################################################################################################################
################################## violin plot CAF cell subcluster distance to tumor   ###################################
########################################################################################################################
########################################################################################################################
import re

CAF_subset = sc_coord_44971[sc_coord_44971['sc_celltype_subset'].str.startswith('CAFs')][
    'sc_celltype_subset'].drop_duplicates().to_list()
dis_mat = cdist(sc_coord_44971[['array_row', 'array_col']], sc_coord_44971[['array_row', 'array_col']])
cancer_idx = np.where(sc_coord_44971['sc_celltype'] == 'Cancer Epithelial')[0]

CAF_sub_celltype_distances = {}
for celltype in CAF_subset:
    celltype_idx = np.where(sc_coord_44971['sc_celltype_subset'] == celltype)[0]
    distances_to_cancer = dis_mat[celltype_idx[:, None], cancer_idx]
    CAF_sub_celltype_distances[celltype] = distances_to_cancer
CAF_sub_cell_2_tumor_distance_mean = {}
for celltype in CAF_subset:
    CAF_sub_cell_2_tumor_distance_mean[celltype] = np.mean(CAF_sub_celltype_distances[celltype], axis=1)

# 创建包含8个子图的Figure对象
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(16, 8))
# 在每个子图上画出对应细胞类型的小提琴图
for ax, celltype in zip(axes.flat, CAF_subset):
    distances = CAF_sub_cell_2_tumor_distance_mean[celltype]
    parts = ax.violinplot(distances, showmeans=False, showmedians=True)
    ax.set_title(celltype)
    ax.set_xticks([])
    ax.set_ylabel('Distance')
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    for line in ['cbars', 'cmaxes', 'cmins', 'cmedians']:
        parts[line].set_color('black')
    parts['cmedians'].set_linewidth(2)
# 去掉背景颜色和网格线
fig.patch.set_facecolor('white')
for ax in axes.flat:
    ax.set_facecolor('white')
    ax.grid(False)
# 调整子图之间的间距和边距
plt.subplots_adjust(hspace=0.5, wspace=0.3)
fig.tight_layout()
# 显示图形
plt.show()

########################################################################################################################
########################################################################################################################
################################## violin plot CAF cell subcluster distance to tumor   ###################################
########################################################################################################################
########################################################################################################################
import re

Myeloid_subset = sc_coord_44971[sc_coord_44971['sc_celltype_subset'].str.startswith('Myeloid_')][
    'sc_celltype_subset'].drop_duplicates().to_list()
Myeloid_subset = sorted(Myeloid_subset, key=lambda x: (int(re.findall(r'Myeloid_c(\d+)_', x)[0]),
                                                       re.findall(r'_([A-Z0-9]+)$', x)[0]))

dis_mat = cdist(sc_coord_44971[['array_row', 'array_col']], sc_coord_44971[['array_row', 'array_col']])
cancer_idx = np.where(sc_coord_44971['sc_celltype'] == 'Cancer Epithelial')[0]

Myeloid_sub_celltype_distances = {}
for celltype in Myeloid_subset:
    celltype_idx = np.where(sc_coord_44971['sc_celltype_subset'] == celltype)[0]
    distances_to_cancer = dis_mat[celltype_idx[:, None], cancer_idx]
    Myeloid_sub_celltype_distances[celltype] = distances_to_cancer
Myeloid_sub_cell_2_tumor_distance_mean = {}
for celltype in Myeloid_subset:
    Myeloid_sub_cell_2_tumor_distance_mean[celltype] = np.mean(Myeloid_sub_celltype_distances[celltype], axis=1)

# 创建包含8个子图的Figure对象
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 8))
# 在每个子图上画出对应细胞类型的小提琴图
for ax, celltype in zip(axes.flat, Myeloid_subset):
    distances = Myeloid_sub_cell_2_tumor_distance_mean[celltype]
    parts = ax.violinplot(distances, showmeans=False, showmedians=True)
    ax.set_title(celltype)
    ax.set_xticks([])
    ax.set_ylabel('Distance')
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    for line in ['cbars', 'cmaxes', 'cmins', 'cmedians']:
        parts[line].set_color('black')
    parts['cmedians'].set_linewidth(2)
# 去掉背景颜色和网格线
fig.patch.set_facecolor('white')
for ax in axes.flat:
    ax.set_facecolor('white')
    ax.grid(False)
# 调整子图之间的间距和边距
plt.subplots_adjust(hspace=0.5, wspace=0.3)
fig.tight_layout()
# 显示图形
plt.show()

########################################################################################################################
########################################################################################################################
################################## spatial plot of cell type major minor subset   ######################################
########################################################################################################################
########################################################################################################################


adata.uns['spatial'] = adata_vis.uns['spatial']
adata.obsm['spatial'] = adata.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy()
idx = adata.obs['celltype_subset'].isin(
    ['CAFs MSC iCAF-like s1', 'CAFs MSC iCAF-like s2', 'B cells Memory', ' B cells Naive'] + T_subset)
scanpy.pl.spatial(adata[idx], color='celltype_subset', cmap='viridis')
scanpy.pl.spatial(adata_vis, color='Classification', cmap='cividis', alpha_img=0.3)

idx = adata.obs['celltype_subset'].isin(Myeloid_subset + T_subset)
scanpy.pl.spatial(adata[idx], color='celltype_subset', cmap='viridis')

idx = adata.obs['celltype_subset'].isin(['Myeloid', 'T-cells'])
scanpy.pl.spatial(adata[idx], color='celltype_subset', cmap='viridis')

########################################################################################################################
########################################################################################################################
################################## spatial plot of one spot contain ten cells   ########################################
########################################################################################################################
########################################################################################################################
index_top10 = np.argsort(rf_pred_dist, axis=1)[:, :10]
# 构建一个空的数据框
df = pd.DataFrame(
    columns=['psu_cell', 'x_psu_coord', 'y_psu_coord', 'x_psu_pxl_coord', 'y_psu_pxl_coord', 'psu_celltype'])
# 遍历索引
for i in range(len(index_top10)):
    # 对于每个索引，提取前十个映射的坐标和标签
    x_psu_coord = np.take(x_coord, index_top10[i])
    y_psu_coord = np.take(y_coord, index_top10[i])
    x_psu_pxl_coord = np.take(x_pxl_coord, index_top10[i])
    y_psu_pxl_coord = np.take(y_pxl_coord, index_top10[i])
    psu_celltype = np.take(spa_celltype, index_top10[i]).to_numpy()

    # 对于每个数据，构建一个字典，将提取的信息存储到字典中
    d = {'psu_cell': [i] * 10, 'x_psu_coord': x_psu_coord, 'y_psu_coord': y_psu_coord,
         'x_psu_pxl_coord': x_psu_pxl_coord, 'y_psu_pxl_coord': y_psu_pxl_coord, 'psu_celltype': psu_celltype}
    # 将字典转换为数据框
    df_temp = pd.DataFrame(d)
    # 将数据框添加到主数据框中
    df = df.append(df_temp, ignore_index=True)
celltype_major = adata.obs['celltype_major'].repeat(10).to_frame('celltype_major')
celltype_minor = adata.obs['celltype_minor'].repeat(10).to_frame('celltype_minor')
celltype_subset = adata.obs['celltype_subset'].repeat(10).to_frame('celltype_subset')
celltype_major.reset_index(inplace=True, drop=True)
celltype_minor.reset_index(inplace=True, drop=True)
celltype_subset.reset_index(inplace=True, drop=True)
df = pd.concat([df, celltype_major, celltype_minor, celltype_subset], axis=1, ignore_index=True)
df.columns = ['psu_cell', 'x_psu_coord', 'y_psu_coord', 'x_psu_pxl_coord', 'y_psu_pxl_coord', 'psu_celltype', 'celltype_major',
              'celltype_minor', 'celltype_subset']
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import partial
from scanpy.plotting._utils import circles
# 获取图像数据
img_data = adata_vis.uns['spatial']['count-44971']['images']['hires']
# 将颜色标签对应到 df['celltype_major'] 的标签
# 将颜色标签对应到 df['celltype_major'] 的标签
cmap = plt.cm.get_cmap('Set1', len(df['celltype_major'].cat.categories))
colors = [cmap(i) for i in range(len(df['celltype_major'].cat.categories))]
color_dict = dict(zip(df['celltype_major'].cat.categories, colors))

# 创建画布和子图对象
fig, ax = plt.subplots(figsize=(10, 10))
size = 1
scale_factor = adata_vis.uns['spatial']['count-44971']['scalefactors']['tissue_hires_scalef']
spot_size = adata_vis.uns['spatial']['count-44971']['scalefactors']['spot_diameter_fullres']
circle_radius = size * scale_factor * spot_size * 0.5

# # 创建颜色条
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax)

# 绘制散点图，并指定颜色和大小
for cat, color in color_dict.items():
    x = df.loc[df['celltype_major'] == cat, 'x_psu_pxl_coord']
    y = df.loc[df['celltype_major'] == cat, 'y_psu_pxl_coord']
    ax.scatter(x, y, s=circle_radius, c=color, label=cat)
cur_coords = np.concatenate([ax.get_xlim(), ax.get_ylim()])
# 绘制图片
# im = ax.imshow(img_data, cmap='gray', extent=[0, img_data.shape[1], 0, img_data.shape[0]])
ax.set_xlim(cur_coords[0], cur_coords[1])
ax.set_ylim(cur_coords[3], cur_coords[2])


# 添加颜色标签
ax.legend(loc="upper right", title="celltype_major", frameon=True)

# 显示图形
plt.show()

########################################################################################################################
########################################################################################################################
################################## spatial plot of with celltype probability on each spot   ########################################
########################################################################################################################
########################################################################################################################
mapping_frac = pd.DataFrame(rf_pred_dist, index=adata.obs['celltype_major'], columns=adata_vis.obs_names).T

# Two stratgies, one is using max value, the other is using mean value
spa_plot = adata_vis.copy()

scanpy.pl.spatial()
size = 1
scale_factor = adata_vis.uns['spatial']['count-44971']['scalefactors']['tissue_hires_scalef']
spot_size = adata_vis.uns['spatial']['count-44971']['scalefactors']['spot_diameter_fullres']
circle_radius = size * scale_factor * spot_size * 0.5

spa_coord = pd.DataFrame(adata_vis.obs[['array_row', 'array_col']], index=adata_vis.obs_names)


endothelial_col = mapping_frac['Endothelial']
endothelial_mean = endothelial_col.mean(axis=1)
def max_min_norm(data):
    return (data - data.min()) / (data.max() - data.min())
endothelial_mean = max_min_norm(endothelial_mean)
merged_data = pd.concat([spa_coord, endothelial_mean], axis=1)
merged_data.columns = ['array_row', 'array_col', 'Endothelial_mean']

fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(merged_data['array_col'], merged_data['array_row'], c=merged_data['Endothelial_mean'], cmap='viridis')
plt.colorbar(scatter, ax=ax, label='Endothelial Mean')
plt.show()

def spatial_plot_frac(mapping_frac, celltype, spa_coord, cmap='viridis'):
    col_ = mapping_frac[celltype]
    mean_ = col_.mean(axis=1)

    def max_min_norm(data):
        return (data - data.min()) / (data.max() - data.min())
    mean_ = max_min_norm(mean_)
    merged_data = pd.concat([spa_coord, mean_], axis=1)
    merged_data.columns = ['array_row', 'array_col', celltype+'_mean']

    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(merged_data['array_col'], merged_data['array_row'], c=merged_data[celltype+'_mean'],
                         cmap=cmap)
    plt.colorbar(scatter, ax=ax, label=celltype+'fraction')
    plt.show()
'RdBu_r'