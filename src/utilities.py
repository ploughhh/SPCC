import warnings
from typing import Tuple, Union, Optional

from numba import njit

import numpy as np
import pandas as pd
import torch
import sklearn
import scanpy
import json
import matplotlib.pyplot as plt
from anndata import AnnData

import scvi.external
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager

import scipy
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.neighbors import NearestNeighbors


def get_spatial_metric(spa_adata):
    position = pd.DataFrame([spa_adata.obs['array_row'], spa_adata.obs['array_col']])
    dis = pdist(position.T, metric='euclidean')
    spa_dis = squareform(dis)
    spa_adata.spatial_metric = spa_dis
    # sc_adata.spatial_metric = np.zeros([sc_adata.n_obs, sc_adata.n_vars])

    return spa_adata

def preprocess_graph(adj):
    adj_ = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return adj_normalized

def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net



def adjust_learning_rate(init_lr, optimizer, iteration, max_lr, adjust_epoch):
    lr = max(init_lr * (0.9 ** (iteration // adjust_epoch)), max_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr

def one_hot(index, category):
    onehot = torch.zeros(index.size(0), category, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)

def parse_use_gpu_arg(
    use_gpu: Optional[Union[str, int, bool]] = None,
    return_device=True,
):
    """
    Parses the use_gpu arg in codebase.

    Returned gpus are is compatible with PytorchLightning's gpus arg.
    If return_device is True, will also return the device.

    Parameters
    ----------
    use_gpu
        Use default GPU if available (if None or True), or index of GPU to use (if int),
        or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
    return_device
        If True, will return the torch.device of use_gpu.
    """
    gpu_available = torch.cuda.is_available()
    if (use_gpu is None and not gpu_available) or (use_gpu is False):
        gpus = 0
        device = torch.device("cpu")
    elif (use_gpu is None and gpu_available) or (use_gpu is True):
        current = torch.cuda.current_device()
        device = torch.device(current)
        gpus = [current]
    elif isinstance(use_gpu, int):
        device = torch.device(use_gpu)
        gpus = [use_gpu]
    elif isinstance(use_gpu, str):
        device = torch.device(use_gpu)
        # changes "cuda:0" to "0,"
        gpus = use_gpu.split(":")[-1] + ","
    else:
        raise ValueError("use_gpu argument not understood.")

    if return_device:
        return gpus, device
    else:
        return gpus

def get_library_size(adata_manager: AnnDataManager, n_batch) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes and returns library size.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object setup with :class:`~scvi.model.SCVI`.
    n_batch
        Number of batches.

    Returns
    -------
    type
        Tuple of two 1 x n_batch ``np.ndarray`` containing the means and variances
        of library size in each batch in adata.

        If a certain batch is not present in the adata, the mean defaults to 0,
        and the variance defaults to 1. These defaults are arbitrary placeholders which
        should not be used in any downstream computation.
    """
    data = adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
    batch_indices = adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)

    library_log_means = np.zeros(n_batch)
    library_log_vars = np.ones(n_batch)

    for i_batch in np.unique(batch_indices):
        idx_batch = np.squeeze(batch_indices == i_batch)
        batch_data = data[
            idx_batch.nonzero()[0]
        ]  # h5ad requires integer indexing arrays.
        sum_counts = batch_data.sum(axis=1)
        masked_log_sum = np.ma.log(sum_counts)
        if np.ma.is_masked(masked_log_sum):
            warnings.warn(
                "This dataset has some empty cells, this might fail inference."
                "Data should be filtered with `scanpy.pp.filter_cells()`"
            )

        log_counts = masked_log_sum.filled(0)
        library_log_means[i_batch] = np.mean(log_counts).astype(np.float32)
        library_log_vars[i_batch] = np.var(log_counts).astype(np.float32)

    return library_log_means.reshape(1, -1), library_log_vars.reshape(1, -1)

@njit()
def mean(input):
    n = len(input)
    input_mean = np.empty(n)
    for i in range(n):
        input_mean[i] = input[i].mean()
    return input_mean

@njit()
def std(input):
    n = len(input)
    input_std = np.empty(n)
    for i in range(n):
        input_std[i] = input[i].std()
    return input_std

@njit()
def fast_correlation(x, y):
    """
    fix the problem from url:https://www.45ma.com/post-37067.html
    Solve the question of speed when calculating correlation between cell and spot when
    the sample is too much
    :param x:
    :param y:
    :return:
    """
    n, k = x.shape
    m, k = y.shape
    mu_x = mean(x)
    mu_y = mean(y)
    std_x = std(x)
    std_y = std(y)

    out = np.empty((n,m))

    for i in range(n):
        for j in range(m):
            out[i,j] = (x[i] - mu_x[i]) @ (y[j] - mu_y[j]) / k / std_x[i] / std_y[j]
    return out



def corrcoef(x):
    """
    传入一个tensor格式的矩阵x(x.shape(m,n))，输出其相关系数矩阵
    adopted from https://www.zhihu.com/question/450669124
    """
    f = (x.shape[0] - 1) / x.shape[0]  # 方差调整系数
    x_reducemean = x - torch.mean(x, axis=0)
    numerator = torch.matmul(x_reducemean.T, x_reducemean) / x.shape[0]
    var_ = x.var(axis=0).reshape(x.shape[1], 1)
    denominator = torch.sqrt(torch.matmul(var_, var_.T)) * f
    corrcoef = numerator / denominator
    return corrcoef

def get_corr(fake_Y, Y):  # 计算两个向量person相关系数
    fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
    fake_Y_mean, Y_mean = torch.mean(fake_Y), torch.mean(Y)
    corr = (torch.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
            torch.sqrt(torch.sum((fake_Y - fake_Y_mean) ** 2)) * torch.sqrt(torch.sum((Y - Y_mean) ** 2)))
    return corr

from audtorch.metrics.functional import pearsonr



def Geary_C(data,weight):
    # Geary's C计算
    a_weight = 0
    s_0=a_weight*2
    n, m = len(data), len(data[0])
    x_hat = np.mean(data)
    up_sum = 0
    down_sum = 0
    for i in range(n):
        for j in range(m):
            # 下一轮
            loc = i * m + j
            for v in weight[loc]:

                up_sum += v[2] * (data[i][j] - data[v[0]][v[1]])**2
            down_sum += (data[i][j] - x_hat) ** 2
    return ((n*m-1)/(s_0))*(up_sum/down_sum)

def knn_weight(latent_seq, latent_spatial, knn_neighbors=55, return_mapping=True, metric='euclidean'):
    knn = NearestNeighbors(n_neighbors=knn_neighbors, n_jobs=10, metric=metric)
    knn.fit(latent_spatial)
    distance, index = knn.kneighbors(latent_seq)
    distance_ = np.power(distance, 2)
    sigma = distance_[:, knn_neighbors-1:knn_neighbors]
    weight = distance_ / sigma
    weight = np.exp(-weight)
    weight_matrix = np.zeros([len(latent_seq), len(latent_spatial)])
    for i in range(len(latent_seq)):
        np.put(weight_matrix[i,:], index[i,:], weight[i,:])
    index_max = np.argmax(weight_matrix, axis=1)

    if return_mapping:
        return weight_matrix, index_max
    else:
        return weight_matrix

def read_visium(path, slice: str = "A1"):
    adata_vis = scanpy.read_10x_mtx(path + "raw_feature_bc_matrix")
    library_id = "count-" + slice
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
    return adata_vis

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
