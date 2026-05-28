"""Basin-id utilities for CAMELS benchmark tasks."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np


def load_basin_ids(path: str | Path) -> np.ndarray:
    basin_path = Path(path)
    if basin_path.suffix == ".npy":
        return np.load(basin_path, allow_pickle=True).astype(np.int64).reshape(-1)

    text = basin_path.read_text().strip()
    if not text:
        return np.array([], dtype=np.int64)

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None

    if parsed is not None:
        return np.asarray(parsed, dtype=np.int64).reshape(-1)
    return np.atleast_1d(np.loadtxt(basin_path, dtype=np.int64)).reshape(-1)


def basin_index(reference_ids: np.ndarray, basin_id: int | str) -> int:
    target = int(basin_id)
    matches = np.where(np.asarray(reference_ids, dtype=np.int64) == target)[0]
    if len(matches) == 0:
        raise ValueError(f"Basin {target} was not found in reference basin ids.")
    return int(matches[0])
