from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]


def load_ids(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.is_absolute(): p = ROOT / p
    if p.suffix == ".npy": return np.load(p).astype(np.int64).reshape(-1)
    import ast
    return np.asarray(ast.literal_eval(p.read_text()), dtype=np.int64).reshape(-1)


def data_hash(paths: Iterable[str | Path]) -> str:
    h = hashlib.sha256()
    for raw in paths:
        p = Path(raw); p = p if p.is_absolute() else ROOT / p
        h.update(str(p.resolve()).encode()); h.update(str(p.stat().st_size).encode()); h.update(str(p.stat().st_mtime_ns).encode())
    return h.hexdigest()


def load_period(basin_ids: np.ndarray, config: dict, split: str, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Load one fixed split as [time, basin, forcing], [time, basin] in mm/day."""
    dcfg = config["data"]
    bundle = ROOT / dcfg["data_path"]
    with bundle.open("rb") as handle: forcings, target, attributes = pickle.load(handle)
    reference = load_ids(dcfg["reference_ids"])
    index = np.asarray([np.where(reference == int(b))[0][0] for b in basin_ids], dtype=np.int64)
    all_time = pd.date_range("1980-10-01", "2014-09-30", freq="D")
    dates = dcfg[split]
    left, right = all_time.get_loc(pd.Timestamp(dates["start_time"])), all_time.get_loc(pd.Timestamp(dates["end_time"])) + 1
    # Bundle forcing order is project-standard prcp, tmean, pet; source streamflow is ft3/s.
    x = forcings[index, left:right, :3]
    y = target[index, left:right, 0].copy()
    # Attribute layout is project benchmark standard; area_gages2 index 11 in COMMON_ATTRIBUTES.
    area = attributes[index, 11].astype(np.float64)
    y *= (0.0283168 * 86400 * 1e3 / (area * 1e6))[:, None]
    return (torch.as_tensor(np.transpose(x, (1, 0, 2)), dtype=torch.float32, device=device),
            torch.as_tensor(y.T, dtype=torch.float32, device=device), attributes[index], index)


def load_repeated_warmup_and_train(
    basin_ids: np.ndarray,
    config: dict,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Return repeated warm-up forcing followed by train forcing and train-only targets.

    The repeated forcing is deliberately excluded from the objective.  This
    keeps the 1989--1998 calibration target fixed while allowing model states
    to equilibrate from a deterministic 1980--1981 water-year cycle.
    """
    dcfg, wcfg = config["data"], config["warmup"]
    if wcfg.get("mode") != "repeat_forcing":
        raise ValueError("load_repeated_warmup_and_train requires warmup.mode=repeat_forcing")
    bundle = ROOT / dcfg["data_path"]
    with bundle.open("rb") as handle:
        forcings, target, attributes = pickle.load(handle)
    reference = load_ids(dcfg["reference_ids"])
    index = np.asarray([np.where(reference == int(b))[0][0] for b in basin_ids], dtype=np.int64)
    all_time = pd.date_range("1980-10-01", "2014-09-30", freq="D")

    def bounds(period: dict[str, str]) -> tuple[int, int]:
        left = all_time.get_loc(pd.Timestamp(period["start_time"]))
        right = all_time.get_loc(pd.Timestamp(period["end_time"])) + 1
        return int(left), int(right)

    warm_left, warm_right = bounds(wcfg["source"])
    train_left, train_right = bounds(dcfg["train"])
    source_days = warm_right - warm_left
    expected_source_days = int(wcfg["source_days"])
    repetitions = int(wcfg["repetitions"])
    if source_days != expected_source_days:
        raise ValueError(f"warm-up source has {source_days} days; expected {expected_source_days}")
    if warm_right > train_left:
        raise ValueError("warm-up source overlaps the training selection period")

    warm_forcing = forcings[index, warm_left:warm_right, :3]
    train_forcing = forcings[index, train_left:train_right, :3]
    warm_doy = all_time[warm_left:warm_right].dayofyear.to_numpy()
    train_doy = all_time[train_left:train_right].dayofyear.to_numpy()
    warm_with_doy = np.concatenate((warm_forcing, np.broadcast_to(warm_doy[None, :, None], (len(index), source_days, 1))), axis=2)
    train_with_doy = np.concatenate((train_forcing, np.broadcast_to(train_doy[None, :, None], (len(index), len(train_doy), 1))), axis=2)
    forcing = np.concatenate((np.concatenate([warm_with_doy] * repetitions, axis=1), train_with_doy), axis=1)

    y = target[index, train_left:train_right, 0].copy()
    area = attributes[index, 11].astype(np.float64)
    y *= (0.0283168 * 86400 * 1e3 / (area * 1e6))[:, None]
    metadata = {
        "warmup_source_days": source_days,
        "warmup_repetitions": repetitions,
        "warmup_total_days": source_days * repetitions,
        "train_days": train_right - train_left,
        "input_days": forcing.shape[1],
    }
    return (
        torch.as_tensor(np.transpose(forcing, (1, 0, 2)), dtype=torch.float32, device=device),
        torch.as_tensor(y.T, dtype=torch.float32, device=device),
        metadata,
    )


def select_pilot_basins(all_ids: np.ndarray, config: dict, count: int = 32) -> np.ndarray:
    """Deterministic stratification on available frac_snow/aridity; no test-flow metric is read."""
    dcfg = config["data"]
    with (ROOT / dcfg["data_path"]).open("rb") as handle: _f, _q, attributes = pickle.load(handle)
    reference = load_ids(dcfg["reference_ids"])
    index = np.asarray([np.where(reference == int(b))[0][0] for b in all_ids])
    # p_mean, pet_mean, p_seasonality, frac_snow, aridity in existing common attribute order.
    values = attributes[index][:, [3, 4]]
    finite = np.all(np.isfinite(values), axis=1)
    candidates = all_ids[finite]; values = values[finite]
    if len(candidates) <= count: return candidates
    ranks = np.column_stack([pd.qcut(values[:, j], 4, labels=False, duplicates="drop") for j in range(2)])
    picked: list[int] = []
    for group in sorted(set(map(tuple, ranks))):
        members = np.sort(candidates[np.all(ranks == group, axis=1)])
        quota = max(1, round(count / max(1, len(set(map(tuple, ranks))))))
        picked.extend(members[:quota].tolist())
    # Stable fill to exact count, avoiding test data or outcome information.
    return np.asarray(sorted(dict.fromkeys(picked))[:count] + [int(x) for x in sorted(candidates) if x not in set(picked)][:max(0, count-len(set(picked)))], dtype=np.int64)[:count]
