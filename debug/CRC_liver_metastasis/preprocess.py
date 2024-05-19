import scanpy
import pandas as pd
import numpy as np

meta = pd.read_csv('/data03/WTG/spascer/38_output/raw/Zhangzemin_CRC_liver_metastasis/GSE164522_CRLM_metadata.csv', sep='\t')
adata_MT = scanpy.read_csv("/data03/WTG/spascer/38_output/raw/Zhangzemin_CRC_liver_metastasis/GSE164522_CRLM_MT_expression.csv").T
selected_rows = meta[meta['Unnamed: 0'].isin(adata_MT.obs_names)]
adata_MT.obs = selected_rows.set_index('Unnamed: 0')
adata_MT.write_h5ad('/data03/WTG/spascer/38_output/raw/Zhangzemin_CRC_liver_metastasis/MT/sc.h5ad')

adata_PT = scanpy.read_csv("/data03/WTG/spascer/38_output/raw/Zhangzemin_CRC_liver_metastasis/GSE164522_CRLM_PT_expression.csv").T
selected_rows = meta[meta['Unnamed: 0'].isin(adata_PT.obs_names)]
adata_PT.obs = selected_rows.set_index('Unnamed: 0')
adata_PT.write_h5ad('/data03/WTG/spascer/38_output/raw/Zhangzemin_CRC_liver_metastasis/PT/sc.h5ad')

adata_LN = scanpy.read_csv("/data03/WTG/spascer/38_output/raw/Zhangzemin_CRC_liver_metastasis/GSE164522_CRLM_LN_expression.csv").T
selected_rows = meta[meta['Unnamed: 0'].isin(adata_LN.obs_names)]
adata_LN.obs = selected_rows.set_index('Unnamed: 0')
adata_LN.write_h5ad('/data03/WTG/spascer/38_output/raw/Zhangzemin_CRC_liver_metastasis/LN/sc.h5ad')
del adata_LN

adata_MN = scanpy.read_csv("/data03/WTG/spascer/38_output/raw/Zhangzemin_CRC_liver_metastasis/GSE164522_CRLM_MN_expression.csv").T
selected_rows = meta[meta['Unnamed: 0'].isin(adata_MN.obs_names)]
adata_MN.obs = selected_rows.set_index('Unnamed: 0')
adata_MN.write_h5ad('/data03/WTG/spascer/38_output/raw/Zhangzemin_CRC_liver_metastasis/MN/sc.h5ad')
del adata_MN

adata_PN = scanpy.read_csv("/data03/WTG/spascer/38_output/raw/Zhangzemin_CRC_liver_metastasis/GSE164522_CRLM_PN_expression.csv").T
selected_rows = meta[meta['Unnamed: 0'].isin(adata_PN.obs_names)]
adata_PN.obs = selected_rows.set_index('Unnamed: 0')
adata_PN.write_h5ad('/data03/WTG/spascer/38_output/raw/Zhangzemin_CRC_liver_metastasis/PN/sc.h5ad')
del adata_PN

adata_vis_colon1 = scanpy.read_h5ad('/data03/WTG/spascer/38_output/raw/ST/ST-colon1/spa.h5ad')
adata_vis_colon2 = scanpy.read_visium('/data03/WTG/spascer/38_output/raw/ST/ST-colon2')
adata_vis_colon2.write_h5ad('/data03/WTG/spascer/38_output/raw/ST/ST-colon2/adata_vis.h5ad')
adata_vis_colon3 = scanpy.read_visium('/data03/WTG/spascer/38_output/raw/ST/ST-colon3')
adata_vis_colon3.write_h5ad('/data03/WTG/spascer/38_output/raw/ST/ST-colon3/adata_vis.h5ad')
adata_vis_colon4 = scanpy.read_visium('/data03/WTG/spascer/38_output/raw/ST/ST-colon4')
adata_vis_colon4.write_h5ad('/data03/WTG/spascer/38_output/raw/ST/ST-colon4/adata_vis.h5ad')

adata_vis_liver1 = scanpy.read_visium('/data03/WTG/spascer/38_output/raw/ST/ST-liver1/')
adata_vis_liver1.write_h5ad('/data03/WTG/spascer/38_output/raw/ST/ST-liver1/adata_vis.h5ad')
adata_vis_liver2 = scanpy.read_visium('/data03/WTG/spascer/38_output/raw/ST/ST-liver2/')
adata_vis_liver2.write_h5ad('/data03/WTG/spascer/38_output/raw/ST/ST-liver2/adata_vis.h5ad')
adata_vis_liver3 = scanpy.read_visium('/data03/WTG/spascer/38_output/raw/ST/ST-liver3/')
adata_vis_liver3.write_h5ad('/data03/WTG/spascer/38_output/raw/ST/ST-liver3/adata_vis.h5ad')
adata_vis_liver4 = scanpy.read_visium('/data03/WTG/spascer/38_output/raw/ST/ST-liver4/')
adata_vis_liver4.write_h5ad('/data03/WTG/spascer/38_output/raw/ST/ST-liver4/adata_vis.h5ad')

primary_selected_rows = meta[(meta['patient'].isin(['patient08', 'patient11'])) & (meta['tissue'].isin(['primary normal', 'primary tumor']))]
adata_ = adata_PT[(adata_PT.obs['patient'].isin(['patient08', 'patient11', 'patient17'])) & (adata_PT.obs['tissue'].isin(['primary normal', 'primary tumor']))].copy()
adata_ = adata_MT[(adata_MT.obs['patient'].isin(['patient08', 'patient11', 'patient17'])) & (adata_MT.obs['tissue'].isin(['metastasis tumor']))].copy()

import scanpy as sc
scanpy.pl.highest_expr_genes(adata_, n_top=20, )
sc.pp.filter_cells(adata_, min_genes=200)
sc.pp.filter_genes(adata_, min_cells=3)
adata_.var['mt'] = adata_.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata_, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(adata_, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)
sc.pl.scatter(adata_, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata_, x='total_counts', y='n_genes_by_counts')
adata_ = adata_[adata_.obs.n_genes_by_counts < 4500, :]
adata_ = adata_[adata_.obs.pct_counts_mt < 2, :]
# adata = adata_.copy()
sc.pp.normalize_total(adata_, target_sum=1e4)
sc.pp.log1p(adata_)
sc.pp.highly_variable_genes(adata_, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pl.highly_variable_genes(adata_)
adata_ = adata_[:, adata_.var.highly_variable]




adata_vis_colon1 = scanpy.read_h5ad('/data03/WTG/spascer/38_output/raw/ST/ST-colon1/spa.h5ad')
rctd_result = pd.read_csv('/data03/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/benchmarking/our_method/CRC_liver_metastasis/RCTD_PT_results/results_df.csv')
rctd_result.index = rctd_result['Unnamed: 0']
rctd_result.index = rctd_result.index.str.replace('.', '-')
adata_vis_colon1.obs['celltype_major'] = rctd_result['second_class']
adata_vis_colon1.write_h5ad('/data03/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/benchmarking/our_method/CRC_liver_metastasis/PT_data/spa.h5ad')

adata_vis_liver1 = scanpy.read_h5ad('/data03/WTG/spascer/38_output/raw/ST/ST-liver1/adata_vis.h5ad')
position = pd.read_csv('/data03/WTG/spascer/38_output/raw/ST/ST-liver1/spatial/tissue_positions_list.csv')
position.columns = [
    'barcode',
    'in_tissue',
    'array_row',
    'array_col',
    'pxl_col_in_fullres',
    'pxl_row_in_fullres',
]
position.index = position['barcode']
position = position.loc[adata_vis_liver1.obs_names]
position = position[['pxl_col_in_fullres', 'pxl_row_in_fullres']]
adata_vis_liver1.var_names_make_unique()
sc_gene = adata.var_names.to_list()
spa_gene = adata_vis_liver1.var_names.to_list()
common_gene = list(set(sc_gene).intersection(spa_gene))
adata_vis_liver1 = adata_vis_liver1[:, common_gene]
x = pd.DataFrame(adata_vis_liver1.X.toarray(), index=adata_vis_liver1.obs_names, columns=adata_vis_liver1.var_names).T
x.to_csv('/data03/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/benchmarking/our_method/CRC_liver_metastasis/MT/RCTD_MT_results/RCTD_MT_input/MT_spa.csv', sep='\t')