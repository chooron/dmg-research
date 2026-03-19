"""GnannEnvironmentSplitter — splits dataset by Gnann et al. climate clusters."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray


# Gnann et al. 4-group mapping (0-indexed clusters)
_CLUSTER_TO_GROUP = {
    0: 1, 1: 1, 7: 1, 8: 1,   # Group 1: eastern low-seasonality
    2: 2, 3: 2,                 # Group 2: western snowmelt
    4: 3, 5: 3, 6: 3,           # Group 3: NW forested mountains
    9: 4,                       # Group 4: Appalachian
}


class GnannEnvironmentSplitter:
    """Split a dataset dict into per-environment sub-dicts using Gnann clusters.

    Parameters
    ----------
    cluster_csv : str
        Path to ``gauge_pos_cluster_climate.csv`` (columns: ``gauge_index``,
        ``gauge_cluster``).
    basin_ids : array-like of int
        Ordered list of basin IDs matching the basin dimension of the dataset.
    use_groups : bool
        If True (default), use the 4 coarse groups; otherwise use 10 fine clusters.
    holdout_group : int or None
        If set, this group is excluded from training environments (OOD test set).
    """

    def __init__(
        self,
        cluster_csv: str,
        basin_ids: NDArray,
        use_groups: bool = True,
        holdout_group: Optional[int] = None,
    ) -> None:
        import pandas as pd
        df = pd.read_csv(cluster_csv)
        df['gauge_index'] = df['gauge_index'].astype(int)

        self.use_groups    = use_groups
        self.holdout_group = holdout_group

        cluster_map = dict(zip(df['gauge_index'], df['gauge_cluster'].astype(int)))
        basin_ids   = np.asarray(basin_ids, dtype=int)

        if use_groups:
            env_labels = np.array(
                [_CLUSTER_TO_GROUP.get(cluster_map.get(b, -1), -1) for b in basin_ids]
            )
        else:
            env_labels = np.array([cluster_map.get(b, -1) for b in basin_ids])

        self.env_labels = env_labels
        self.basin_ids  = basin_ids

        unique_envs = sorted(set(env_labels.tolist()) - {-1})
        self.env_indices: dict[int, NDArray] = {
            e: np.where(env_labels == e)[0] for e in unique_envs
        }

        if holdout_group is not None and holdout_group in self.env_indices:
            self.holdout_indices    = self.env_indices[holdout_group]
            self.train_env_indices  = {
                e: idx for e, idx in self.env_indices.items() if e != holdout_group
            }
        else:
            self.holdout_indices   = np.array([], dtype=int)
            self.train_env_indices = self.env_indices

    def split_dataset(
        self,
        dataset: dict[str, torch.Tensor],
    ) -> dict[int, dict[str, torch.Tensor]]:
        return {
            env_id: self._index_dataset(dataset, idx)
            for env_id, idx in self.train_env_indices.items()
        }

    def holdout_dataset(
        self,
        dataset: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if len(self.holdout_indices) == 0:
            return dataset
        return self._index_dataset(dataset, self.holdout_indices)

    @staticmethod
    def _index_dataset(
        dataset: dict[str, torch.Tensor],
        idx: NDArray,
    ) -> dict[str, torch.Tensor]:
        out = {}
        for key, val in dataset.items():
            if isinstance(val, (torch.Tensor, np.ndarray)):
                if val.ndim == 3:
                    out[key] = val[:, idx, :]
                elif val.ndim == 2:
                    out[key] = val[idx, :]
                else:
                    out[key] = val
            else:
                out[key] = val
        return out
