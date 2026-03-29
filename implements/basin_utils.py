"""Utilities for loading basin ids and subsetting basin-major datasets."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray


def load_basin_ids(path: str | Path) -> NDArray[np.int64]:
    """Load basin ids from ``.npy``, plain-text numeric files, or Python-list txt."""
    basin_path = Path(path)
    if basin_path.suffix == ".npy":
        return np.load(basin_path, allow_pickle=True).astype(np.int64)

    text = basin_path.read_text().strip()
    if not text:
        return np.array([], dtype=np.int64)

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None

    if parsed is not None:
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return np.asarray(parsed, dtype=np.int64).reshape(-1)
        if isinstance(parsed, (int, np.integer)):
            return np.asarray([parsed], dtype=np.int64)

    return np.atleast_1d(np.loadtxt(basin_path, dtype=np.int64)).reshape(-1)


def basin_subset_indices(
    reference_basin_ids: NDArray[np.int64],
    subset_basin_ids: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Return indices into ``reference_basin_ids`` that match ``subset_basin_ids`` order."""
    reference = np.asarray(reference_basin_ids, dtype=np.int64).reshape(-1)
    subset = np.asarray(subset_basin_ids, dtype=np.int64).reshape(-1)

    if len(np.unique(subset)) != len(subset):
        raise ValueError("Subset basin ids contain duplicates.")

    basin_to_index = {int(basin_id): idx for idx, basin_id in enumerate(reference)}
    missing = [int(basin_id) for basin_id in subset if int(basin_id) not in basin_to_index]
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(
            f"{len(missing)} subset basin ids were not found in the reference basin list: {preview}"
        )

    return np.asarray([basin_to_index[int(basin_id)] for basin_id in subset], dtype=np.int64)


def subset_dataset_by_indices(
    dataset: dict[str, torch.Tensor | np.ndarray | object],
    basin_indices: NDArray[np.int64],
) -> dict[str, torch.Tensor | np.ndarray | object]:
    """Subset a dataset whose basin dimension is axis 1 for 3D and axis 0 for 2D tensors."""
    out: dict[str, torch.Tensor | np.ndarray | object] = {}
    for key, value in dataset.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            if value.ndim == 3:
                out[key] = value[:, basin_indices, :]
            elif value.ndim == 2:
                out[key] = value[basin_indices, :]
            else:
                out[key] = value
        else:
            out[key] = value
    return out


def subset_dataset_by_basin_ids(
    dataset: dict[str, torch.Tensor | np.ndarray | object],
    reference_basin_ids: NDArray[np.int64],
    subset_basin_ids: NDArray[np.int64],
) -> tuple[dict[str, torch.Tensor | np.ndarray | object], NDArray[np.int64]]:
    """Subset a basin-major dataset using explicit basin ids."""
    basin_indices = basin_subset_indices(reference_basin_ids, subset_basin_ids)

    basin_dim = None
    for value in dataset.values():
        if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 3:
            basin_dim = value.shape[1]
            break
        if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 2:
            basin_dim = value.shape[0]
            break

    if basin_dim is None:
        raise ValueError("Unable to infer basin dimension from dataset.")
    if basin_dim != len(reference_basin_ids):
        raise ValueError(
            "Reference basin ids length does not match dataset basin dimension: "
            f"{len(reference_basin_ids)} vs {basin_dim}."
        )

    return subset_dataset_by_indices(dataset, basin_indices), subset_basin_ids.copy()
