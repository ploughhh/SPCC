import os
import numpy as np
import pandas as pd
import scanpy
from tqdm import tqdm
import subprocess
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

sc_data_path = "/data2/WTG/spascer_data/49/data03/WTG/spascer/49/raw/sc_cellphonedb/"
adata = scanpy.read_h5ad("/data03/WTG/spascer/49/raw/sc/whole_sc.h5ad")
patient_ID = adata.obs['Patient']
patient_cat = patient_ID.drop_duplicates()

p_CID4066 = adata[adata.obs['Patient']=="CID4066",:].copy()
p_CID4066_meta = pd.DataFrame(p_CID4066.obs['celltype_major'])
p_CID4066_meta.to_csv("/data03/WTG/spascer/49/raw/sc_cellphonedb/CID4066/meta.tsv", sep='\t')
p_CID4066.write_h5ad("/data03/WTG/spascer/49/raw/sc_cellphonedb/CID4066/CID4066.h5ad")
p_CID4066 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/data03/WTG/spascer/49/raw/sc_cellphonedb/CID4066/sc.h5ad")

p_CID3921 = adata[adata.obs['Patient']=="CID3921",:].copy()
p_CID3921_meta = pd.DataFrame(p_CID3921.obs['celltype_major'])
p_CID3921_meta.to_csv("/data03/WTG/spascer/49/raw/sc_cellphonedb/CID3921/meta.tsv", sep='\t')
p_CID3921.write_h5ad("/data03/WTG/spascer/49/raw/sc_cellphonedb/CID3921/CID4066.h5ad")

for i in tqdm(list(patient_cat)):
    path = "/data03/WTG/spascer/49/raw/sc_cellphonedb/" + i
    os.mkdir(path)
    patient = adata[adata.obs['Patient']==i,:].copy()
    meta = pd.DataFrame(patient.obs['celltype_major'])
    meta.to_csv(path + "/meta.tsv", sep='\t')
    patient.write_h5ad(path + "/sc.h5ad")

command = []
file_name = "cpdb_command.txt"
with open("/data03/WTG/spascer/49/raw/sc_cellphonedb/cpdb_command.txt", 'w') as f:
    for i in tqdm(list(patient_cat)):
        path = "/data2/WTG/spascer_data/49/data03/WTG/spascer/49/raw/sc_cellphonedb/"
        f.write("cd " + path + i + "\n")
        f.write("source ~/.bashrc\n")
        f.write("conda activate cpdb\n")
        f.write("cellphonedb method statistical_analysis " + '"' + path + i + "/meta.tsv" + '" ' + '"' + path + i + "/sc.h5ad" + '" ' + "--counts-data gene_name " + "--threads 100\n")


        # process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # output, err = process.communicate()
        # print(output.decode('gbk'), end='')
        # print(err.decode('gbk'))
        # print(process.poll())
        # print('Exit code:', process.returncode)


# trying to use high variable gene to train
p_CID4066 = scanpy.read_h5ad("/data2/WTG/spascer_data/49/data03/WTG/spascer/49/raw/sc_cellphonedb/CID4066/sc.h5ad")
scanpy.pl.highest_expr_genes(p_CID4066, n_top=20, )
scanpy.pp.filter_cells(p_CID4066, min_genes=400)
scanpy.pp.filter_genes(p_CID4066, min_cells=5)

p_CID4066.var['mt'] = p_CID4066.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
scanpy.pp.calculate_qc_metrics(p_CID4066, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
scanpy.pl.violin(p_CID4066, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)
scanpy.pl.scatter(p_CID4066, x='total_counts', y='pct_counts_mt')
scanpy.pl.scatter(p_CID4066, x='total_counts', y='n_genes_by_counts')

p_CID4066 = p_CID4066[p_CID4066.obs.n_genes_by_counts < 6800, :]
p_CID4066 = p_CID4066[p_CID4066.obs.pct_counts_mt < 3, :]

scanpy.pp.normalize_total(p_CID4066, target_sum=1e4)
scanpy.pp.log1p(p_CID4066)
scanpy.pp.highly_variable_genes(p_CID4066, min_mean=0.0125, max_mean=3, min_disp=0.5)
scanpy.pl.highly_variable_genes(p_CID4066)
p_CID4066.raw = p_CID4066
p_CID4066 = p_CID4066[:, p_CID4066.var.highly_variable]
scanpy.pp.regress_out(p_CID4066, ['total_counts', 'pct_counts_mt'])
scanpy.pp.scale(p_CID4066, max_value=10)

scanpy.tl.pca(p_CID4066, svd_solver='arpack')
scanpy.pl.pca(p_CID4066, color='celltype_major')
scanpy.pp.neighbors(p_CID4066, n_neighbors=10, n_pcs=40)
scanpy.tl.leiden(p_CID4066)
scanpy.tl.paga(p_CID4066)
scanpy.pl.paga(p_CID4066, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
scanpy.tl.umap(p_CID4066, init_pos='paga')

scanpy.tl.rank_genes_groups(p_CID4066, 'leiden', method='t-test')
scanpy.pl.rank_genes_groups(p_CID4066, n_genes=25, sharey=False)

# rank_genes = p_CID4066.uns.data['rank_genes_groups']['names']['0']
rank_genes = []
for i in range(18):
    rank_genes.append(p_CID4066.uns.data['rank_genes_groups']['names'][str(i)][0:25])

train_genes = [genes for cluster in rank_genes for genes in cluster]
pd.DataFrame(train_genes).to_csv("/data2/WTG/spascer_data/49/data03/WTG/spascer/49/raw/sc_cellphonedb/CID4066/top_25_marker_list.csv", sep='\t')




sc_sample = os.listdir("/data2/WTG/spascer_data/49/data03/WTG/spascer/49/raw/sc_cellphonedb/")
path = "/data2/WTG/spascer_data/49/data03/WTG/spascer/49/raw/sc_cellphonedb/"

cancer_type = []
for i in tqdm(list(patient_cat)):
    path = "/data03/WTG/spascer/49/raw/sc_cellphonedb/" + i
    patient = adata[adata.obs['Patient']==i,:].copy()
    cancer_type.append(patient.obs['subtype'].drop_duplicates())


for i in range(26):
    pd.read_csv(path + sc_sample[i] + "/meta.tsv", sep='\t')