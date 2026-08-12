"""Shared CAMELS array-axis metadata contract for dPL and IC loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np


FORCING_NAMES = ("P", "T", "PET")


def load_gage_ids(path: str | Path) -> list[str]:
    values = np.asarray(np.load(path, allow_pickle=False))
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"gage_id.npy must be a non-empty one-dimensional array, got {values.shape}")
    ids = [str(int(value)).zfill(8) for value in values]
    if len(set(ids)) != len(ids):
        raise ValueError("gage_id.npy contains duplicate basin IDs")
    return ids


def load_dates(path: str | Path) -> np.ndarray:
    dates = np.asarray(np.load(path, allow_pickle=False)).astype("datetime64[D]")
    if dates.ndim != 1 or dates.size == 0:
        raise ValueError(f"date axis must be a non-empty one-dimensional array, got {dates.shape}")
    if len(np.unique(dates)) != len(dates):
        raise ValueError("date axis contains duplicate dates")
    if len(dates) > 1 and not np.all(np.diff(dates.astype("int64")) == 1):
        raise ValueError("date axis must be contiguous daily dates")
    return dates
