"""GnannEnvironmentSplitter — split basins by merged Gnann behavior clusters."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray

# Original 10-cluster -> 7 effective-cluster mapping
_CLUSTER_MERGE = {
    0: "A",
    1: "B",
    2: "C",
    3: "C",
    4: "D",
    5: "D",
    6: "D",
    7: "E",
    8: "F",
    9: "G",
}

EFFECTIVE_CLUSTERS = ["A", "B", "C", "D", "E", "F", "G"]

EXPECTED_TEST_BASIN_COUNTS = {
    "A": 230,
    "B": 101,
    "C": 59,
    "D": 50,
    "E": 90,
    "F": 61,
    "G": 52,
}


def normalize_held_out_cluster(held_out_cluster: Optional[str]) -> Optional[str]:
    """Normalize and validate a held-out effective cluster label."""
    if held_out_cluster is None:
        return None

    normalized = str(held_out_cluster).strip().upper()
    if normalized not in EFFECTIVE_CLUSTERS:
        valid = ", ".join(EFFECTIVE_CLUSTERS)
        raise ValueError(
            f"Invalid held-out effective cluster '{held_out_cluster}'. "
            f"Use one of: {valid}."
        )
    return normalized


class GnannEnvironmentSplitter:
    """Split datasets into per-environment subsets using 7 effective clusters."""

    def __init__(
        self,
        cluster_csv: str,
        basin_ids: NDArray,
        holdout_cluster: Optional[str] = None,
    ) -> None:
        import pandas as pd

        df = pd.read_csv(cluster_csv).copy()
        df['gauge_index'] = df['gauge_index'].astype(int)
        df['gauge_cluster'] = df['gauge_cluster'].astype(int)
        df['effective_cluster'] = df['gauge_cluster'].map(_CLUSTER_MERGE)

        basin_ids = np.asarray(basin_ids, dtype=int)
        holdout_cluster = normalize_held_out_cluster(holdout_cluster)

        raw_cluster_map = dict(zip(df['gauge_index'], df['gauge_cluster']))
        effective_cluster_map = dict(zip(df['gauge_index'], df['effective_cluster']))

        raw_clusters = np.array([raw_cluster_map.get(b, -1) for b in basin_ids], dtype=int)
        effective_clusters = np.array(
            [effective_cluster_map.get(b, "") for b in basin_ids],
            dtype=object,
        )
        missing_mask = effective_clusters == ""

        self.cluster_df = df
        self.basin_ids = basin_ids
        self.raw_clusters = raw_clusters
        self.effective_clusters = effective_clusters
        self.holdout_cluster = holdout_cluster
        self.env_labels = effective_clusters
        self.unassigned_indices = np.where(missing_mask)[0]

        self.env_indices: dict[str, NDArray] = {
            cluster_id: np.where(effective_clusters == cluster_id)[0]
            for cluster_id in EFFECTIVE_CLUSTERS
            if np.any(effective_clusters == cluster_id)
        }

        if holdout_cluster is not None and holdout_cluster in self.env_indices:
            self.holdout_indices = self.env_indices[holdout_cluster]
            self.train_env_indices = {
                cluster_id: idx
                for cluster_id, idx in self.env_indices.items()
                if cluster_id != holdout_cluster
            }
        else:
            self.holdout_indices = np.array([], dtype=int)
            self.train_env_indices = self.env_indices

    def split_dataset(
        self,
        dataset: dict[str, torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor]]:
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

    def basin_metadata(self, idx: NDArray) -> dict[str, NDArray]:
        """Return basin-aligned metadata for an index selection."""
        return {
            'basin_id': self.basin_ids[idx].copy(),
            'gauge_cluster': self.raw_clusters[idx].copy(),
            'effective_cluster': self.effective_clusters[idx].copy(),
        }

    def fold_size_summary(self) -> list[dict[str, int | str | bool]]:
        """Return train/test basin counts for all 7 leave-one-cluster folds."""
        total_basins = int(sum(len(idx) for idx in self.env_indices.values()))
        summary: list[dict[str, int | str | bool]] = []
        for cluster_id in EFFECTIVE_CLUSTERS:
            test_basins = int(len(self.env_indices.get(cluster_id, np.array([], dtype=int))))
            train_basins = total_basins - test_basins
            summary.append(
                {
                    'held_out_cluster': cluster_id,
                    'test_basins': test_basins,
                    'train_basins': train_basins,
                    'train_environments': len(EFFECTIVE_CLUSTERS) - 1,
                    'matches_expected': test_basins == EXPECTED_TEST_BASIN_COUNTS[cluster_id],
                }
            )
        return summary

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
