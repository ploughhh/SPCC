import numpy as np
import pandas as pd
import loompy
import h5py
import anndata
import os


_subtype_to_high_level_mapping = {
    "Astrocytes": ("Astrocyte Gfap", "Astrocyte Mfge8"),
    "Endothelials": ("Endothelial", "Endothelial 1"),
    "Inhibitory": (
        "Inhibitory Cnr1",
        "Inhibitory Kcnip2",
        "Inhibitory Pthlh",
        "Inhibitory Crhbp",
        "Inhibitory CP",
        "Inhibitory IC",
        "Inhibitory Vip",
    ),
    "Microglias": ("Perivascular Macrophages", "Microglia"),
    "Oligodendrocytes": (
        "Oligodendrocyte Precursor cells",
        "Oligodendrocyte COP",
        "Oligodendrocyte NF",
        "Oligodendrocyte Mature",
        "Oligodendrocyte MF",
    ),
    "Pyramidals": (
        "Pyramidal L2-3",
        "Pyramidal Cpne5",
        "Pyramidal L2-3 L5",
        "pyramidal L4",
        "Pyramidal L3-4",
        "Pyramidal Kcnip2",
        "Pyramidal L6",
        "Pyramidal L5",
        "Hippocampus",
    ),
}
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# url = "http://linnarssonlab.org/osmFISH/osmFISH_SScortex_mouse_all_cells.loom"
def load_smfish():
    # ds = loompy.connect("/data2/WTG/Sc-Spatial-transformer/data/osmFISH_SScortex_mouse_all_cell.loom")
    ds = loompy.connect("/home/WTG/Sc-Spatial-transformer/Sc-Spatial-transformer/data/osmFISH_SScortex_mouse_all_cell.loom")
    x_coord, y_coord = ds.ca["X"], ds.ca["Y"]
    regions = ds.ca["Region"]
    data = ds[:, :].T
    gene_names = ds.ra["Gene"].astype(np.str)
    labels = ds.ca["ClusterID"]
    str_labels = np.asarray(ds.ca["ClusterName"])
    labels_mapping = pd.Categorical(str_labels).categories

    for high_level_cluster, subtypes in _subtype_to_high_level_mapping.items():
        for subtype in subtypes:
            idx = np.where(str_labels == subtype)
            str_labels[idx] = high_level_cluster
    cell_types_to_keep = [
        "Astrocytes",
        "Endothelials",
        "Inhibitory",
        "Microglias",
        "Oligodendrocytes",
        "Pyramidals",
    ]
    row_indices = [
        i
        for i in range(data.shape[0])
        if ds.ca["ClusterName"][i] in cell_types_to_keep
    ]
    str_labels = str_labels[row_indices]
    data = data[row_indices, :]
    x_coord = x_coord[row_indices]
    y_coord = y_coord[row_indices]
    regions = regions[row_indices]
    str_labels = pd.Categorical(str_labels)
    labels = str_labels.codes
    labels_mapping = str_labels.categories

    adata_vis = anndata.AnnData(
        X=data,
        obs={
            "x_coord": x_coord,
            "y_coord": y_coord,
            "labels": labels,
            "str_labels": str_labels,
            "regions": regions
        },
        uns={"cell_types": labels_mapping},
    )
    adata_vis.var_names = gene_names
    adata_vis.obs["batch"] = np.zeros(adata_vis.shape[0], dtype=np.int64)
    return adata_vis