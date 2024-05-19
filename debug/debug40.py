import numpy
import scanpy
import json
import matplotlib.pyplot as plt
import pandas as pd

import scvi
from scvi_build._Sc_Spa_model import SC_SPA_model

sc_data = scanpy.read_h5ad("/data03/WTG/spascer/40_done/annotation/40_Human_intestinal_embryo_annoated.h5ad")
# sc_data.X = sc_data.raw.X

# vis1 = scanpy.read_visium("/data03/WTG/spascer/40_done/spatial/A1", count_file="/data03/WTG/spascer/40_done/spatial/A1/40_A1_spatial.h5ad")
path = "/data03/WTG/spascer/40_done/spatial/A1/"
adata_vis = scanpy.read_10x_mtx(path=path + "raw_feature_bc_matrix")
library_id = "count-A1"

adata_vis.uns["spatial"] = dict()
adata_vis.uns["spatial"][library_id] = dict()
files = dict(
    tissue_positions_file=path + 'spatial/tissue_positions_list.csv',
    scalefactors_json_file=path + 'spatial/scalefactors_json.json',
    hires_image=path + 'tissue_hires_image.png',
    lowres_image=path + 'tissue_lowres_image.png',
)

adata_vis.uns["spatial"][library_id]['images'] = dict()
adata_vis.uns["spatial"][library_id]['images']['hires'] = plt.imread(
    path + 'spatial/tissue_hires_image.png'
)
adata_vis.uns["spatial"][library_id]['images']['lowres'] = plt.imread(
    path + 'spatial/tissue_lowres_image.png'
)

with open(path + "spatial/scalefactors_json.json", 'r') as f:
    str = f.read()
    scalefactor = json.loads(str)


adata_vis.uns["spatial"][library_id]['scalefactors'] = scalefactor


positions = pd.read_csv(files['tissue_positions_file'], header=None)
positions.columns = [
    'barcode',
    'in_tissue',
    'array_row',
    'array_col',
    'pxl_col_in_fullres',
    'pxl_row_in_fullres',
]

positions.index = positions['barcode']
adata_vis.obs = adata_vis.obs.join(positions, how="left")
adata_vis.obsm['spatial'] = adata_vis.obs[
    ['pxl_row_in_fullres', 'pxl_col_in_fullres']
].to_numpy()
adata_vis.obs.drop(
    columns=['barcode', 'pxl_row_in_fullres', 'pxl_col_in_fullres'],
    inplace=True,
)

adata_vis.var_names_make_unique()
spa_data = adata_vis

var_names = sc_data.var_names.intersection(spa_data.var_names)
sc_data_common = sc_data[:, var_names]
spa_data.var_names_make_unique()
spa_data.obs_names_make_unique()
spa_common = spa_data[:, var_names]
spa_common = spa_common.copy()

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

position = spa_data.obs
dis = pdist(position, metric='euclidean')
spa_dis = squareform(dis)
spa_common.spatial_metric = spa_dis
spa_common.obs.spatial_metric = spa_dis
sc_data.spatial_metric = numpy.zeros([sc_data.n_obs, sc_data.n_vars])




SC_SPA_model.setup_anndata(sc_data)
SC_SPA_model.setup_anndata(spa_common, spatial_metric='spatial_metric')
gim_vae = SC_SPA_model(sc_data, spa_common)

gim_vae.train(max_epochs=1,
              use_gpu=True,
              batch_size=64)



print("hello")


from cell2location.models import RegressionModel
RegressionModel