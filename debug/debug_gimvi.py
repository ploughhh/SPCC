import matplotlib.pyplot as plt
import dotplot
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
from scvi.data import smfish, cortex
from debug.gimvi_data.preprocess import load_smfish
from src.utilities import preprocess_graph
from src.validation_test import get_train_set
from src.utilities import get_spatial_metric, fast_correlation, corrcoef
# from src.sinkhorn_pointcloud import sinkhorn_loss
# from src.visiualize import sigmoid
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.neighbors import NearestNeighbors
from src.utilities import knn_weight
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from spa_decom.spatial_decomposition import spatial_range_limtation, get_view_information
from src.build_cell_communication import phi_exp

# spatial_data = smfish()
spatial_data = load_smfish()
seq_data = cortex()



train_size = 0.8


def get_spatial_metric(sc_adata, spa_adata):
    position = pd.DataFrame([spa_adata.obs['x_coord'], spa_adata.obs['y_coord']])
    dis = pdist(position.T, metric='euclidean')
    spa_dis = squareform(dis)
    spa_adata.spatial_metric = spa_dis
    sc_adata.spatial_metric = np.zeros([sc_adata.n_obs, sc_adata.n_vars])

    return spa_adata, sc_adata


seq_data = seq_data[:, spatial_data.var_names].copy()

seq_gene_names = seq_data.var_names
n_genes = seq_data.n_vars
n_train_genes = int(n_genes*train_size)

#randomly select training_genes
rand_train_gene_idx = np.random.choice(range(n_genes), n_train_genes, replace = False)
rand_test_gene_idx = sorted(set(range(n_genes)) - set(rand_train_gene_idx))
rand_train_genes = seq_gene_names[rand_train_gene_idx]
rand_test_genes = seq_gene_names[rand_test_gene_idx]

#spatial_data_partial has a subset of the genes to train on
spatial_data_partial = spatial_data[:,rand_train_genes].copy()

#remove cells with no counts
# scanpy.pp.filter_cells(spatial_data_partial, min_counts= 1)
# scanpy.pp.filter_cells(seq_data, min_counts = 1)


# spatial_data_partial, seq_data = get_spatial_metric(seq_data, spatial_data_partial)
# spa_nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=1000, algorithm='auto').fit(spatial_data_partial.spatial_metric)
# spa_adj = spa_nbrs.kneighbors_graph(spatial_data_partial.spatial_metric).toarray()
# spa_adj = preprocess_graph(spa_adj)
# spatial_data_partial.spatial_metric = spa_adj

# seq_data.spatial_metric = None
#setup_anndata for spatial and sequencing data
seq_data.obs['cell_type'].replace({'pyramidal SS': 'Pyramidals', 'pyramidal CA1': 'Pyramidals', 'oligodendrocytes': 'Oligodendrocytes', 'microglia': 'Microglias', 'endothelial-mural': 'Endothelials',
                                'astrocytes_ependymal': 'Astrocytes', 'interneurons': 'Inhibitory'}, inplace=True
                               )
# GIMVI.setup_anndata(spatial_data_partial, labels_key='labels', batch_key='batch')
# GIMVI.setup_anndata(seq_data, labels_key='labels')
GIMVI.setup_anndata(spatial_data_partial, labels_key='str_labels', batch_key='batch')
GIMVI.setup_anndata(seq_data, labels_key='cell_type')

#spatial_data should use the same cells as our training data
#cells may have been removed by scanpy.pp.filter_cells()
spatial_data = spatial_data[spatial_data_partial.obs_names]

model = GIMVI(seq_data, spatial_data_partial)

# train for 200 epochs
model.train(200,
            use_gpu="cuda:0",
            celltype_classifier=True,
            kappa=2.5
            )
# model.train(200,
#             use_gpu="cpu")


latent_seq, latent_spatial = model.get_latent_representation()
latents = [latent_seq, latent_spatial]

#concatenate to one latent representation
latent_representation = np.concatenate([latent_seq, latent_spatial])
batch_a = np.zeros(len(latent_seq))
batch_b = np.ones(len(latent_spatial))
batch = np.concatenate([batch_a, batch_b])

a = pd.DataFrame(batch)




latent_adata = anndata.AnnData(latent_representation)

# Adversarial classifier True -> False + discriminative loss = 0.5312186
from src.utilities import batch_entropy_mixing_score


score = batch_entropy_mixing_score(latent_representation, batch)

lr_file = pd.read_csv("/data2/WTG/Sc-Spatial-transformer/data/osmfish_brain_data/out/means.txt", sep='\t')
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
spa_coord = spatial_data.obs[['x_coord', 'y_coord']]
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
formula = 'Multivar(x_coord, y_coord) ~ .'
distance = 'all'
formula = robjects.Formula(formula)

rf_train = r_rfsrc.rfsrc(formula=formula, data=r_st_train, block_size=5, ntree=1000)
rf_pred = r_rfsrc.predict_rfsrc(rf_train, newdata=r_test_data, distance='all')
distance = rf_pred.rx2('distance')
rf_pred_dist = distance[0:len(latent_seq), len(latent_seq):len(latent_representation)]

index_top10 = np.argsort(rf_pred_dist, axis=1)[:, :10]
index_max = np.argmax(-rf_pred_dist, axis=1)
adata = seq_data.copy()
adata_vis = spatial_data.copy()
x_psu_coord = np.zeros(len(index_max))
y_psu_coord = np.zeros(len(index_max))

x_coord = np.array(adata_vis.obs['x_coord'])
y_coord = np.array(adata_vis.obs['y_coord'])

region = adata_vis.obs['regions']
x_psu_coord = np.take(x_coord, index_max)
y_psu_coord = np.take(y_coord, index_max)
psu_region = np.take(region, index_max)
adata.obs['x_coord'] = x_psu_coord
adata.obs['y_coord'] = y_psu_coord

sc_coord = pd.DataFrame()
sc_coord['x_coord'] = adata.obs['x_coord']
sc_coord['y_coord'] = adata.obs['y_coord']
spa_coord = pd.DataFrame()
spa_coord['x_coord'] = adata_vis.obs['x_coord']
spa_coord['y_coord'] = adata_vis.obs['y_coord']
spa_coord['str_labels'] = adata_vis.obs['str_labels']
sc_coord['labels'] = adata.obs.cell_type
adata.obs['regions'] = region

plt.grid(b=False)
sns.scatterplot(x=sc_coord['x_coord'], y=sc_coord['y_coord'], hue=adata.obs.cell_type.tolist(),style=adata.obs.cell_type.tolist(),
                palette=sns.color_palette("Set1",6), markers='o',
                data=sc_coord).set(title="single cell data cell type distribution", yticklabels=[], xticklabels=[])
plt.show()


















# spatial visualization based on pseudo-distance
adata_vis, adata = get_spatial_metric(seq_data, spatial_data)
weight_matrix, index_max = knn_weight(latent_seq, latent_spatial, metric='correlation')
distance_matrix = np.zeros((len(index_max), len(index_max)))
is_dmat = adata_vis.spatial_metric
for i in range(len(index_max)):
    distance_matrix[i, :] = np.take(is_dmat[i, :], index_max)
from sklearn import preprocessing
min_max_normalizer = preprocessing.MinMaxScaler(feature_range=(0,1))
def max_min_norm(data):
    return (data - data.min()) / (data.max() - data.min())
distance_matrix_scale = min_max_normalizer.fit_transform(distance_matrix)
adata.obsm['spatial_metric'] = distance_matrix_scale
adata.spatial_metric = distance_matrix_scale
scanpy.pp.neighbors(adata, use_rep="spatial_metric")
scanpy.tl.umap(adata, min_dist=0.2)
scanpy.pl.umap(adata, color="cell_type")

x_psu_coord = np.zeros(len(index_max))
y_psu_coord = np.zeros(len(index_max))

x_coord = np.array(adata_vis.obs['x_coord'])
y_coord = np.array(adata_vis.obs['y_coord'])

region = adata_vis.obs['regions']
x_psu_coord = np.take(x_coord, index_max)
y_psu_coord = np.take(y_coord, index_max)
psu_region = np.take(region, index_max)
adata.obs['x_coord'] = x_psu_coord
adata.obs['y_coord'] = y_psu_coord

sc_coord = pd.DataFrame()
sc_coord['x_coord'] = adata.obs['x_coord']
sc_coord['y_coord'] = adata.obs['y_coord']
spa_coord = pd.DataFrame()
spa_coord['x_coord'] = adata_vis.obs['x_coord']
spa_coord['y_coord'] = adata_vis.obs['y_coord']
sc_coord['labels'] = adata.obs.cell_type
adata.obs['regions'] = region

plt.grid(b=None)
sns.scatterplot(x=sc_coord['x_coord'], y=sc_coord['y_coord'], hue=adata.obs.cell_type.tolist(),style=adata.obs.cell_type.tolist(),
                palette=sns.color_palette("Set1",7), markers='o',
                data=sc_coord).set(title="single cell data cell type distribution", yticklabels=[], xticklabels=[])
plt.show()

fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(11.7, 8.27)
plt.rcParams['figure.figsize']=(6.4, 4.8)
sns.scatterplot(x=sc_coord['x_coord'], y=sc_coord['y_coord'], hue=adata.obs['regions'].tolist(), style=adata.obs['regions'].tolist() ,
                palette=sns.color_palette("Set1",11), markers='o',
                data=sc_coord, ax=ax).set(title="single cell data region distribution", yticklabels=[], xticklabels=[])
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
plt.show()

sns.scatterplot(x=adata_vis.obs['x_coord'], y=adata_vis.obs['y_coord'], hue=adata_vis.obs['regions'].tolist(), style=adata_vis.obs['regions'].tolist() ,
                palette=sns.color_palette("Set1",11), markers='o',
                data=spa_coord).set(title="spatial data region", yticklabels=[], xticklabels=[])
plt.show()

sns.scatterplot(x=sc_coord['x_coord'], y=sc_coord['y_coord'], hue=adata.obs['regions'].tolist(), style=adata.obs['regions'].tolist() ,
                palette=sns.color_palette("Set1",11), markers='o',
                data=sc_coord).set(title="single cell data cell type distribution", yticklabels=[], xticklabels=[])
plt.show()

region_type = ["Pia Layer 1", "Layer 2-3 medial", "Layer 2-3 lateral", "Layer 3-4", "Layer 4", "Layer 5", "Layer 6", "Internal Capsule Caudoputamen", "White matter", "Hippocampus", "Ventricle"]
region_index = []
for subtype in region_type:
    idx = np.where(region == subtype)
    region_index.append(idx)
spa_region_index_re = []
for i in range(len(region_index)):
    spa_region_index_re.append(region_index[i][0])
spa_cell_num_region = []
for i in region_index:
    spa_cell_num_region.append(len(i[0]))

sc_region = adata.obs['regions']
sc_region_index = []
for subtype in region_type:
    idx = np.where(sc_region == subtype)
    sc_region_index.append(idx)
sc_region_index_re = []
for i in range(len(sc_region_index)):
    sc_region_index_re.append(sc_region_index[i][0])
sc_cell_num_region = []
for i in sc_region_index:
    sc_cell_num_region.append(len(i[0]))

# violin_y = [12000, 11000, 10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000]
# violin_y = np.array(violin_y) + np.array(sc_cell_num_region)
#
# # 取第一个区域的单细胞索引
# sc_reg_sub = sc_region_index_re[0]
# spa_sub_index = index_max[sc_reg_sub]
# # spa data region
# spa_reg_sub_type = adata_vis.obs['regions'][spa_sub_index]
# sub_type = spa_reg_sub_type.drop_duplicates()
# sub_num = []
# for subtype in region_type:
#     idx = np.where(spa_reg_sub_type == subtype)
#     sub_num.append(len(idx[0]))



def my_custom_sort(region):
    region_type = ["Pia Layer 1", "Layer 2-3 medial", "Layer 2-3 lateral", "Layer 3-4", "Layer 4", "Layer 5", "Layer 6", "Internal Capsule Caudoputamen", "White matter", "Hippocampus", "Ventricle"]
    if region in region_type:
        return region_type.index(region)
    else:
        return len(region_type)

mapping_num = []
mapping_region = []
sc_region = []
for i in range(len(region_type)):
    sub_num = []
    sc_reg_sub = sc_region_index_re[i]
    spa_sub_index = index_max[sc_reg_sub]
    # spa data region
    spa_reg_sub_type = adata_vis.obs['regions'][spa_sub_index]
    sub_type = list(spa_reg_sub_type.drop_duplicates())
    sub_type.sort(key=my_custom_sort)
    mapping_region.append(sub_type)

    for j in sub_type:
        idx = np.where(spa_reg_sub_type == j)
        sub_num.append(len(idx[0]))
        sc_region.append(region_type[i])
    mapping_num.append(sub_num)



# sc_reg_index = [item for reg in sc_region_index_re for item in reg]
# spa_reg_index = [item for reg in spa_region_index_re for item in reg]
# sc_region_order = adata.obs['regions'][sc_reg_index]
# spa_region_order = adata_vis.obs['regions'][spa_reg_index]


mapping_num_re = [item for sub_mapping in mapping_num for item in sub_mapping]
mapping_region_re = [item for sub_mapping_region in mapping_region for item in sub_mapping_region]
violin_ = {'sc_region': sc_region,
           'mapping_num': mapping_num_re,
           'spa_mapping_region': mapping_region_re}
violin_ = pd.DataFrame(violin_)

new_keys = {'item_key': 'spa_mapping_region', 'group_key': 'sc_region', 'sizes_key': 'mapping_num'}
dp = dotplot.DotPlot.parse_from_tidy_data(violin_, **new_keys)
sct = dp.plot(cmap='Reds', size_factor=1)
plt.show()
violin_.to_csv("/data2/WTG/Sc-Spatial-transformer/data/region_mapping.csv", sep='\t')

# 绘制分组小提琴图
sns.violinplot(x = "sc_region", # 指定x轴的数据
               y = "spa_mapping_region", # 指定y轴的数据
               # hue = "sex", # 指定分组变量
               data = violin_, # 指定绘图的数据集
               order = region_type, # 指定x轴刻度标签的顺序
               scale = 'count', # 以男女客户数调节小提琴图左右的宽度
               split = True, # 将小提琴图从中间割裂开，形成不同的密度曲线；
               palette = 'RdBu' # 指定不同性别对应的颜色（因为hue参数为设置为性别变量）
              )
# 添加图形标题
plt.title('region mapping proportion') #每天不同性别客户的酒吧消费额情况
# 设置图例
plt.legend(loc = 'upper left', ncol = 2)
# 显示图形
plt.show()

sns.scatterplot(x="sc_region", y="spa_mapping_region", hue=region_type, style=region_type ,
                palette=sns.color_palette("Set1",11), markers='o', s='mapping_num',
                data=violin_).set(title="single cell data cell type distribution", yticklabels=[], xticklabels=[])
plt.show()


##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
# raw data cell cel修改行的顺序l interaction
raw = cortex()
raw.var_names = raw.var_names.str.upper()
ligand = raw[:, ligand_gene]
receptor = raw[:, receptor_gene]
scanpy.pp.normalize_total(ligand)
scanpy.pp.normalize_total(receptor)

ligand_expression = ligand.X
receptor_expression = receptor.X
# alpha_1 = np.array((phi_exp(ligand_expression, 1.0, 15, -1)))
# beta_1 = np.array((phi_exp(receptor_expression, 1.0, 15, -1)))
ligand_expression = np.log1p(ligand_expression)
receptor_expression = np.log1p(receptor_expression)
alpha_1 = np.array(np.exp(ligand_expression))
beta_1 = np.array(np.exp(receptor_expression))
# score_1 = np.outer(alpha_1, beta_1)
score = np.zeros((len(alpha_1), len(alpha_1)))
from tqdm import tqdm
for i in tqdm(range(len(ligand_gene))):
    score_1 = np.outer(alpha_1[:, i], beta_1[:, i])
    score = score + score_1
from sklearn import preprocessing
min_max_normalizer = preprocessing.MinMaxScaler(feature_range=(0,1))
score_scale = min_max_normalizer.fit_transform(score)
cell_type_list = list(raw.obs.cell_type)
# index = [i for i, val in enumerate(cell_type_list) if val=="Pericytes"]
cell_type = list(pd.DataFrame(seq_data.obs.cell_type).drop_duplicates().iloc[:, 0])

cell_type_index_list = []
for i in range(len(cell_type)):
    index = [j for j, val in enumerate(cell_type_list) if val == cell_type[i]]
    cell_type_index_list.append(index)

cluster_intera = []
cluster_communication = []
for i in range(len(cell_type)):
    cluster_intera.append(np.take(score_scale, cell_type_index_list[i], axis=0))
    for j in range(len(cell_type)):
        cluster_communication.append(np.take(cluster_intera[i], cell_type_index_list[j], axis=1))

cluster = []
for x in cluster_communication:
    cluster.append(np.mean(x))
cluster = np.array((cluster))
cluster = cluster.reshape((7, 7))
# from sklearn import preprocessing
# min_max_normalizer = preprocessing.MinMaxScaler(feature_range=(0,1))
# cluster_scale = min_max_normalizer.fit_transform(cluster)
def max_min_norm(data):
    return (data - data.min()) / (data.max() - data.min())
cluster_scale = max_min_norm(cluster)
f, ax = plt.subplots(figsize=(9, 9))
ax = sns.heatmap(cluster_scale, xticklabels=cell_type, yticklabels=cell_type)
plt.show()
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################




##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
# spatial range decomposition and find the best spatial range
importance = []


views = spatial_range_limtation(adata, adata_vis, latents, [0, 50, 100])
expr_view = get_view_information(adata, views)

intra_pred_list = []
juxta_pred_list = []
para_pred_list = []
intra_importance_list = []
juxta_importance_list = []
para_importance_list = []
regression_coe = []


for i in tqdm(range(len(adata.var))):
    intra_vector = expr_view[0][:, i]
    juxta_vector = expr_view[1][:, i]
    para_vector = expr_view[2][:, i]

    intra_regressor = RandomForestRegressor(n_estimators=33, random_state=0)
    juxta_regressor = RandomForestRegressor(n_estimators=33, random_state=0)
    para_regressor = RandomForestRegressor(n_estimators=33, random_state=0)
    intra_regressor.fit(expr_view[0], intra_vector)
    juxta_regressor.fit(expr_view[1], juxta_vector)
    para_regressor.fit(expr_view[2], para_vector)

    intra_pred = intra_regressor.predict(expr_view[0])
    intra_importance = intra_regressor.feature_importances_
    intra_pred_list.append(intra_pred)
    intra_importance_list.append(intra_importance)

    juxta_pred = juxta_regressor.predict(expr_view[1])
    juxta_importance = juxta_regressor.feature_importances_
    juxta_pred_list.append(juxta_pred)
    juxta_importance_list.append(juxta_importance)

    para_pred = para_regressor.predict(expr_view[2])
    para_importance = para_regressor.feature_importances_
    para_pred_list.append(para_pred)
    para_importance_list.append(para_importance)

    pred_combine = np.vstack((intra_pred, juxta_pred, para_pred)).T
    ridge_reg = Ridge(alpha=1.0, fit_intercept=True)
    ridge_reg.fit(pred_combine, expr_view[0])
    regression_coe.append(ridge_reg.coef_)
importance.append(intra_importance_list)
importance.append(juxta_importance_list)
importance.append(para_importance_list)

spatial_effect = []
for i in importance:
    view = np.array(i)
    view_mean = np.mean(view, axis=0)
    view_mean = max_min_norm(view_mean)
    view_mean = np.mean(view_mean)
    spatial_effect.append(view_mean)

x = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900,2996]
plt.plot(x, spatial_effect, label='spatial range effect')
plt.xlabel('spatial neighbor')
plt.ylabel('spatial range effect')
plt.legend()
plt.grid(visible=False)
plt.show()
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################



##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
# cell cell interaction with spatial limitation
views = spatial_range_limtation(raw, adata_vis, latents, [1100])
expr_view = get_view_information(raw, views)
view_spatial_limitation = expr_view[0]
adata_spa_lim = anndata.AnnData(view_spatial_limitation)
adata_spa_lim.var_names = raw.var_names

ligand_lim = adata_spa_lim[:, ligand_gene]
receptor_lim = adata_spa_lim[:, receptor_gene]
scanpy.pp.normalize_total(ligand_lim)
scanpy.pp.normalize_total(receptor_lim)

ligand_expression_lim = ligand_lim.X
receptor_expression_lim = receptor_lim.X
# alpha_1 = np.array((phi_exp(ligand_expression, 1.0, 15, -1)))
# beta_1 = np.array((phi_exp(receptor_expression, 1.0, 15, -1)))
ligand_expression_lim = np.log1p(ligand_expression_lim)
receptor_expression_lim = np.log1p(receptor_expression_lim)
alpha_1_lim = np.array(np.exp(ligand_expression_lim))
beta_1_lim = np.array(np.exp(receptor_expression_lim))
# score_1 = np.outer(alpha_1, beta_1)
score_lim = np.zeros((len(alpha_1_lim), len(alpha_1_lim)))
from tqdm import tqdm
for i in tqdm(range(len(ligand_gene))):
    score_1_lim = np.outer(alpha_1_lim[:, i], beta_1_lim[:, i])
    score_lim = score_lim + score_1_lim
from sklearn import preprocessing
min_max_normalizer = preprocessing.MinMaxScaler(feature_range=(0,1))
score_scale_lim = min_max_normalizer.fit_transform(score_lim)
cell_type_list = list(raw.obs.cell_type)
# index = [i for i, val in enumerate(cell_type_list) if val=="Pericytes"]
cell_type = list(pd.DataFrame(seq_data.obs.cell_type).drop_duplicates().iloc[:, 0])

cell_type_index_list_lim = []
for i in range(len(cell_type)):
    index = [j for j, val in enumerate(cell_type_list) if val == cell_type[i]]
    cell_type_index_list_lim.append(index)

cluster_intera_lim = []
cluster_communication_lim = []
for i in range(len(cell_type)):
    cluster_intera_lim.append(np.take(score_scale_lim, cell_type_index_list_lim[i], axis=0))
    for j in range(len(cell_type)):
        cluster_communication_lim.append(np.take(cluster_intera_lim[i], cell_type_index_list_lim[j], axis=1))

cluster_lim = []
for x in cluster_communication_lim:
    cluster_lim.append(np.mean(x))
cluster_lim = np.array((cluster_lim))
cluster_lim = cluster_lim.reshape((7, 7))
# from sklearn import preprocessing
# min_max_normalizer = preprocessing.MinMaxScaler(feature_range=(0,1))
# cluster_scale = min_max_normalizer.fit_transform(cluster)
def max_min_norm(data):
    return (data - data.min()) / (data.max() - data.min())
cluster_scale_lim = max_min_norm(cluster_lim)
f, ax = plt.subplots(figsize=(9, 9))
ax = sns.heatmap(cluster_scale_lim, xticklabels=cell_type, yticklabels=cell_type)
plt.show()
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################







##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
# ligand-receptor contribution
contribution = []
for i in tqdm(range(len(ligand_gene))):
    score_1 = np.outer(alpha_1[:, i], beta_1[:, i])
    # score_1 = score_1 * views[0]
    contribution.append(np.mean(score_1))
contribution = np.array(contribution)
contri_rank = contribution.argsort()[-9:]
contribution_per = []
contribution_gene = []
for i in list(contri_rank):
    contribution_per.append(contribution[i] / np.sum(contribution))
    contribution_gene.append(str(ligand_gene[i]+ "-" +receptor_gene[i]))
plt.pie(np.array(contribution_per), labels=np.array(contribution_gene))
plt.show()

contribution = np.array(contribution)
contri_rank = (-contribution).argsort()  # 逆序排序
contribution_per = []
contribution_gene = []
for i in list(contri_rank):
    contribution_per.append(contribution[i] / np.sum(contribution))
    contribution_gene.append(str(ligand_gene[i] + "-" + receptor_gene[i]))
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################



##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
# ligand receptor contribution with spatial limitation
contribution_lim = []
for i in tqdm(range(len(ligand_gene))):
    score_1_lim = np.outer(alpha_1_lim[:, i], beta_1_lim[:, i])
    # score_1 = score_1 * views[0]
    contribution_lim.append(np.mean(score_1_lim))
contribution_lim = np.array(contribution_lim)
contri_rank_lim = contribution_lim.argsort()[-9:]
contribution_per_lim = []
contribution_gene_lim = []
for i in list(contri_rank_lim):
    contribution_per_lim.append(contribution_lim[i] / np.sum(contribution_lim))
    contribution_gene_lim.append(str(ligand_gene[i]+ "-" +receptor_gene[i]))
plt.pie(np.array(contribution_per_lim), labels=np.array(contribution_gene_lim))
plt.show()
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################








##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
# spatial pattern
def get_morans(W, X):
    n, g = X.shape
    X = np.array(X)
    S0 = np.sum(W)

    Z = (X - np.mean(X, 0))
    sZs = (Z ** 2).sum(0)
    I = n / S0 * (Z.T @ (W @ Z)) / sZs
    return np.diagonal(I)

W = adata.spatial_metric
raw = raw[adata.obs_name, :]
X = raw.X
n, g = X.shape
X = np.array(X)
npermut = 100
pI = np.zeros((npermut, g))
idx = np.arange(n)

for i in tqdm(np.arange(npermut)):
    pidx = np.random.permutation(idx)
    pI[i, :] = get_morans(W, X[pidx, :]).reshape((1, -1))

I = get_morans(W, X)
EI = np.mean(pI, 0)
seI_norm = np.std(pI, 0)
z_norm = (I - EI) / seI_norm
import scipy.stats as stats
is_neg_mI = z_norm < 0
p_norm = 1 - stats.norm.cdf(z_norm)
p_norm[is_neg_mI] = stats.norm.cdf(z_norm)[is_neg_mI]
mI = I
pvals = p_norm
mI[np.isnan(mI)] = -np.inf
results = pd.DataFrame({'genes': raw.var_names, 'mI': mI, 'pval': pvals})
results = results.sort_values(by=['mI'], ascending=False)
spatial_data_gene = pd.DataFrame(adata_vis.var_names.str.upper())
spatial_pattern_gene = pd.merge(spatial_data_gene, results, left_on=0, right_on='genes')

def plot_gene_spatial(model, adata_vis, gene):
    data_seq = model.adatas[0]
    data_fish = adata_vis

    # fig, ax = plt.figure()

    if type(gene) == str:
        gene_id = list(data_seq.var_names).index(gene)
    else:
        gene_id = gene

    x_coord = data_fish.obs["x_coord"]
    y_coord = data_fish.obs["y_coord"]

    def order_by_strenght(x, y, z):
        ind = np.argsort(z)
        return x[ind], y[ind], z[ind]

    s = 20

    def transform(data):
        return np.log(1 + 100 * data)

    # Plot groundtruth
    x, y, z = order_by_strenght(
        x_coord, y_coord, data_fish.X[:, gene_id] / (adata_vis.X.sum(axis=1) + 1)
    )
    plt.scatter(x, y, c=transform(z), s=s, edgecolors="none", marker="s", cmap="Greens")
    # plt.set_title(gene_id)
    plt.title(gene)
    plt.axis("off")

    # _, imputed = model.get_imputed_values(normalized=True)
    # x, y, z = order_by_strenght(x_coord, y_coord, imputed[:, gene_id])
    # ax.scatter(x, y, c=transform(z), s=s, edgecolors="none", marker="s", cmap="Reds")
    # ax.set_title("Imputed")
    # ax.axis("off")
    plt.tight_layout()
    plt.show()

# adata_vis cell type spatial visualization
spatial_ct = pd.DataFrame()
spatial_ct['labels'] = adata_vis.obs.str_labels
spatial_ct['x_coord'] = adata_vis.obs['x_coord']
spatial_ct['y_coord'] = adata_vis.obs['y_coord']
plt.grid(b=None)
sns.scatterplot(x=spatial_ct.x_coord.to_list(), y=spatial_ct.y_coord.to_list(),hue=spatial_ct.labels.tolist(),style=spatial_ct.labels.tolist(),
                palette=sns.color_palette("Set1",6), markers='o',
                data=spatial_ct).set(title="spatial data cell type distribution", yticklabels=[], xticklabels=[])
plt.show()

plot_gene_spatial(model, adata_vis, 'LAMP5')
plot_gene_spatial(model, adata_vis, 'SLC32A1')
plot_gene_spatial(model, adata_vis, 'RORB')
plot_gene_spatial(model, adata_vis, 'GAD2')
plot_gene_spatial(model, adata_vis, 'TMEM2')
plot_gene_spatial(model, adata_vis, 'TBR1')
plot_gene_spatial(model, adata_vis, 'ITPR2')
plot_gene_spatial(model, adata_vis, 'CTPS')
plot_gene_spatial(model, adata_vis, 'CRH')





















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


