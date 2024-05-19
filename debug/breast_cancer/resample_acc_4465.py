import os
print(os.getcwd())
os.chdir('/data03/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer')
import pandas as pd
import anndata
from imblearn.over_sampling import RandomOverSampler
import scanpy
import numpy as np

from gimvi import GIMVI

adata = scanpy.read_h5ad('/data03/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/benchmarking/our_method/breast_cancer_data/4465_result/adata.h5ad')
adata_vis = scanpy.read_h5ad('/data03/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/benchmarking/our_method/breast_cancer_data/4465_result/adata_vis.h5ad')
distance = np.load('/data03/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/benchmarking/our_method/breast_cancer_data/44971_result/distance.npy', allow_pickle=True)
p_CID4465 = scanpy.read_h5ad("/data03/WTG/spascer/49/raw/sc_cellphonedb/CID4465/sc.h5ad")
CID_4465_spa = scanpy.read_h5ad("/data03/WTG/spascer/49/raw/spatial/CID4465_spa_TNBC.h5ad")


def resample(adata, classication):
    celltype_major = adata.obs[classication]
    celltype_counts = celltype_major.value_counts()
    # 计算目标每个细胞类型应该有的数量
    target_count = celltype_counts.max()
    target_counts = {celltype: target_count for celltype in celltype_counts.index}
    # 对每个细胞类型进行上采样
    X = adata.X
    y = celltype_major.values
    ros = RandomOverSampler(sampling_strategy=target_counts, random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    adata_new = anndata.AnnData(X_resampled)
    adata_new.obs[classication] = y_resampled
    adata_new.var_names = adata.var_names
    adata_new.var = adata.var
    return X_resampled, y_resampled, adata_new

_, _, adata_new = resample(adata, 'celltype_major')
_, _, adata_vis_new = resample(adata_vis, 'Classification')

GIMVI.setup_anndata(adata_new, labels_key='celltype_major')
GIMVI.setup_anndata(adata_vis_new, labels_key='Classification')
model = GIMVI(adata_new, adata_vis_new, generative_distributions=['zinb', 'zinb'], model_library_size=[True, True])
# model.load_model("/data2/WTG/spascer_data/49/raw/trained_model/p_CID3946_1142243F/")

model.train(250,
            use_gpu="cuda:0",
            kappa=2.5
            )

latent_seq, latent_spatial = model.get_latent_representation()
latent_representation = np.concatenate([latent_seq, latent_spatial])
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

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

mapping = pd.DataFrame(rf_pred_dist, index=adata_new.obs_names, columns=adata_vis_new.obs_names)
index_max = np.argmax(-rf_pred_dist, axis=1)
spa_celltype = adata_vis_new.obs['Classification']
psu_celltype = np.take(spa_celltype, index_max)
psu_celltype = psu_celltype.to_numpy()
# psu_region = np.take(region, index_max)
adata_new.obs['spadata_celltype'] = psu_celltype

sc_coord_4465 = pd.DataFrame()
sc_coord_4465['sc_celltype'] = adata_new.obs['celltype_major']
sc_coord_4465['spa_celltype'] = adata_new.obs['spadata_celltype']


subset = sc_coord_4465[(sc_coord_4465['sc_celltype'] == 'Cancer Epithelial') &
                       (sc_coord_4465['spa_celltype'].isin(['Invasive cancer + stroma + lymphocytes']))]
whole_subset = sc_coord_4465['sc_celltype'].value_counts()['Cancer Epithelial']
accuracy = len(subset) / whole_subset
# TPR sensitive
FN_subset = sc_coord_4465[(sc_coord_4465['sc_celltype'] == 'Cancer Epithelial') &
                          (~sc_coord_4465['spa_celltype'].isin(['Invasive cancer + stroma + lymphocytes']))]
TP = len(FN_subset)
FN = whole_subset - TP
TPR = TP / (TP + FN)