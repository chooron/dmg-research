from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

# canonical full300 evaluation helpers (restored) depend on the benchmark model
# registry and streaming-KGE objective; imported at module scope like the
# original evaluate_full300_metrics.py
from src.model_registry import build_model
from src.objective import streaming_kge


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


# ============================================================================
# Restored canonical CMA-ES full300 evaluation helpers.
#
# These reproduce, byte-for-byte in behaviour, the original benchmark
# evaluation semantics of the full300 independent-calibration run (source:
# experiments/cmaes_36models/scripts/evaluate_full300_metrics.py, which
# produced results/full300_final_36models_evaluation/):
#   * selection is ONLY the best checkpointed TRAINING KGE among the starts
#     (per basin, best of ``expected_starts`` independent CMA-ES starts taken
#     from the FINAL-generation checkpoint);
#   * test-period KGE is computed afterwards and is never used to choose a
#     start;
#   * evaluate_period re-applies the canonical repeat_forcing warm-up (source
#     period repeated ``repetitions`` times, excluded from scoring) and appends
#     the day-of-year channel to the forcing, so calendar models such as the
#     restored MOPEX4 consume ``doy`` exactly as in training.
# ============================================================================
def _bounds(dates: pd.DatetimeIndex, period: dict[str, str]) -> tuple[int, int]:
    return (
        int(dates.get_loc(pd.Timestamp(period["start_time"]))),
        int(dates.get_loc(pd.Timestamp(period["end_time"]))) + 1,
    )


def load_warmup_and_period(
    basin_ids: np.ndarray, config: dict, period_name: str, device: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Load repeated warm forcing plus one scored split; targets exclude warm-up."""
    dcfg, wcfg = config["data"], config["warmup"]
    with (ROOT / dcfg["data_path"]).open("rb") as handle:
        forcings, target, attributes = pickle.load(handle)
    reference = load_ids(dcfg["reference_ids"])
    index = np.asarray([np.where(reference == int(b))[0][0] for b in basin_ids], dtype=np.int64)
    dates = pd.date_range("1980-10-01", "2014-09-30", freq="D")
    warm_left, warm_right = _bounds(dates, wcfg["source"])
    period_left, period_right = _bounds(dates, dcfg[period_name])
    source_days, repetitions = warm_right - warm_left, int(wcfg["repetitions"])
    if source_days != int(wcfg["source_days"]):
        raise ValueError(f"warm-up source has {source_days} days, expected {wcfg['source_days']}")

    def with_doy(values: np.ndarray, day_values: np.ndarray) -> np.ndarray:
        doy = day_values.dayofyear.to_numpy()
        return np.concatenate((values, np.broadcast_to(doy[None, :, None], (*values.shape[:2], 1))), axis=2)

    warm = with_doy(forcings[index, warm_left:warm_right, :3], dates[warm_left:warm_right])
    scored = with_doy(forcings[index, period_left:period_right, :3], dates[period_left:period_right])
    forcing = np.concatenate(([warm] * repetitions) + [scored], axis=1)
    y = target[index, period_left:period_right, 0].copy()
    area = attributes[index, 11].astype(np.float64)
    y *= (0.0283168 * 86400.0 * 1e3 / (area * 1e6))[:, None]
    return (
        torch.as_tensor(forcing.transpose(1, 0, 2), dtype=torch.float32, device=device),
        torch.as_tensor(y.T, dtype=torch.float32, device=device),
        source_days * repetitions,
    )


def _final_checkpoints(model_dir: Path, generations: int) -> list[Path]:
    pieces = sorted(model_dir.glob(f"chunk_*_gen_{generations}.pt"), key=lambda p: int(p.name.split("_")[1]))
    if not pieces:
        raise FileNotFoundError(f"No generation-{generations} checkpoints in {model_dir}")
    return pieces


def frozen_parameters(model_dir: Path, generations: int, expected_starts: int) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
    """Return basin ids, selected latent parameters, and checkpoint train KGE.

    Each basin's maximum over independent starts is selected here. No score
    from the test period is observed by this function.
    """
    basin_parts: list[np.ndarray] = []
    latent_parts: list[torch.Tensor] = []
    score_parts: list[np.ndarray] = []
    for path in _final_checkpoints(model_dir, generations):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        basin_ids = np.asarray(payload["basin_ids"], dtype=np.int64)
        state = payload["solver"]["state"]
        best = state["best_fitness"].detach().cpu().numpy()
        latent = state["best_latent"].detach().cpu()
        if best.size != basin_ids.size * expected_starts:
            raise ValueError(f"{path}: {best.size} scores do not match {basin_ids.size} x {expected_starts} starts")
        dimension = latent.shape[-1]
        best = best.reshape(basin_ids.size, expected_starts)
        latent = latent.reshape(basin_ids.size, expected_starts, dimension)
        selected_start = best.argmax(axis=1)
        basin_parts.append(basin_ids)
        latent_parts.append(latent[torch.arange(basin_ids.size), torch.as_tensor(selected_start)])
        score_parts.append(best[np.arange(basin_ids.size), selected_start])
    basin_ids = np.concatenate(basin_parts)
    if np.unique(basin_ids).size != basin_ids.size:
        raise ValueError(f"{model_dir.name}: duplicate basin checkpoint pieces")
    return basin_ids, torch.cat(latent_parts, dim=0), np.concatenate(score_parts)


def evaluate_period(model_name: str, latent: torch.Tensor, basin_ids: np.ndarray, config: dict, period: str, device: str, backend: str) -> np.ndarray:
    x, y, warmup_days = load_warmup_and_period(basin_ids, config, period, device)
    model = build_model(model_name, device, warm_up=warmup_days, backend=backend)
    # HydrologyModel expects [basin, parameter, group]. One group is one
    # already-frozen best training start for each basin.
    raw = torch.sigmoid(latent.to(device=device, dtype=torch.float64)).unsqueeze(-1).float()
    with torch.inference_mode():
        q = model({"x_phy": x}, (None, raw))["streamflow"].reshape(-1, len(basin_ids), 1, 1)
        score, invalid = streaming_kge(q, y)
    if bool(invalid.any()):
        count = int(invalid.sum().item())
        raise RuntimeError(f"{model_name} {period}: {count} frozen solutions yielded invalid KGE")
    return score[:, 0, 0].detach().cpu().numpy()
