import warnings
from typing import Optional, Union
import rich

import numpy as np
import pandas as pd

import anndata
from anndata import AnnData
from scvi_build import _constants
from scvi.data._utils import (
    _check_nonnegative_integers,
    _verify_and_correct_data_format
)
from scvi.data.fields import BaseAnnDataField

def get_anndata_attribute(
    adata: anndata.AnnData, attr_name: str, attr_key: Optional[str]
) -> Union[np.ndarray, pd.DataFrame]:
    """Returns the requested data from a given AnnData object."""
    # get data from response field
    adata_attr = getattr(adata, attr_name)
    if attr_key is None:
        field = adata_attr
    else:
        if isinstance(adata_attr, pd.DataFrame):
            if attr_key not in adata_attr.columns:
                raise ValueError(
                    f"{attr_key} is not a valid column in adata.{attr_name}."
                )
            field = adata_attr.loc[:, attr_key]
        else:
            if attr_key not in adata_attr.keys():
                raise ValueError(f"{attr_key} is not a valid key in adata.{attr_name}.")
            field = adata_attr[attr_key]
    if isinstance(field, pd.Series):
        field = field.to_numpy().reshape(-1, 1)
    return field


def get_spatial_metric(
        adata: anndata.AnnData,
        attr_name: str
) -> Union[np.ndarray, pd.DataFrame]:
    adata_attr = getattr(adata, attr_name)
    return adata_attr

class SpatialField(BaseAnnDataField):
    N_CELLS_KEY = "n_cells"
    N_VARS_KEY = "n_cells_used_to_calculate_distance"

    def __init__(
            self,
            registry_key: str,
            is_count_data: bool = False,
            correct_data_format: bool = False
    ) -> None:
        super().__init__()
        self._registry_key = registry_key
        self._attr_name = (
            _constants._ADATA_ATTRS.SPATIAL_METRIC
        )
        self._attr_key = None
        self.is_count_data = is_count_data
        self.correct_data_format = correct_data_format

    @property
    def registry_key(self) -> str:
        return self._registry_key

    @property
    def attr_name(self) -> str:
        return self._attr_name

    @property
    def attr_key(self) -> Optional[str]:
        return self._attr_key

    @property
    def is_empty(self) -> bool:
        return False

    def validate_field(self, adata: AnnData) -> None:
        super().validate_field(adata)
        spatial = get_spatial_metric(adata, self.attr_name)
        if self.is_count_data and not _check_nonnegative_integers(spatial):
            logger_data_loc = (
                "adata.X" if self.attr_key is None else f"adata.layers[{self.attr_key}]"
            )
            warnings.warn(
                f"{logger_data_loc} does not contain unnormalized count data. "
                "Are you sure this is what you want?"
            )

    def register_field(self, adata: AnnData) -> dict:
        super().register_field(adata)
        if self.correct_data_format:
            pass
        return {
            self.N_CELLS_KEY: adata.n_obs,
            self.N_VARS_KEY: adata.n_obs
        }

    def transfer_field(
        self, state_registry: dict, adata_target: AnnData, **kwargs
    ) -> dict:
        super().transfer_field(state_registry, adata_target, **kwargs)
        n_vars = state_registry[self.N_VARS_KEY]
        target_n_vars = adata_target.n_vars
        if target_n_vars != n_vars:
            raise ValueError(
                "Number of vars in adata_target not the same as source. "
                + "Expected: {} Received: {}".format(target_n_vars, n_vars)
            )
        return self.register_field(adata_target)

    def get_summary_stats(self, state_registry: dict) -> dict:
        return state_registry.copy()

    def view_state_registry(self, state_registry: dict) -> Optional[rich.table.Table]:
        return None