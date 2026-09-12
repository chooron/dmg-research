#!/usr/bin/env python3
"""Shared frozen-artifact helpers for the remaining seen-basin analyses.

This module only reads canonical artifacts.  It never constructs an optimizer,
performs a training step, calls backward, or writes a checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BENCHMARK = REPO / "project/benchmark"
sys.path[:0] = [str(REPO), str(BENCHMARK), str(BENCHMARK / "src")]

from dpl.attributes import CAMELS_35_ATTRIBUTES  # noqa: E402
from src.data_selection import load_ids  # noqa: E402
from src.model_registry import NPARAM_INFO_36, get_spec  # noqa: E402

FORMAL = BENCHMARK / "results/ic_dpl_seenbasin_formal_20260901"
DPL_ROOT = BENCHMARK / "results/dpl_canonical_v2_20260831"
DPL_RUNS = DPL_ROOT / "runs"
IC_ROOT = BENCHMARK / "results/ic_dpl_aligned_full300_20260819_final"
IC_BEST_ROOT = IC_ROOT / "best_training"
CARAVAN_PATH = REPO.parent / "dmg-research_replay_forensics/data_remote_contract/caravan_671_attributes.npy"
GAGE_PATH = REPO / "data/gage_id.npy"
IDS_PATH = REPO / "data/531sub_id.txt"
RESULTS = BENCHMARK / "results/seenbasin_remaining_analysis_20260901"
ANALYSIS_DIR = BENCHMARK / "analysis/seenbasin_remaining_20260901"
ALL_MODELS = tuple(NPARAM_INFO_36)
STRICT_FULL300 = {m for m in ALL_MODELS if m != "simhyd"}
CATEGORICAL = {"dom_land_cover", "geol_1st_class", "geol_2nd_class"}
ATTR_TYPES = {a: ("CATEGORICAL_CODE" if a in CATEGORICAL else "CONTINUOUS") for a in CAMELS_35_ATTRIBUTES}
SEED = 20260901
TEST_PERIOD = "1995-10-01..2010-09-30"


def canonical_id(value: Any) -> str:
    text = str(value)
    return text[:-2] if text.endswith(".0") else text.zfill(8)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10f")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def load_inputs() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    ids = np.asarray([int(x) for x in load_ids(str(IDS_PATH))], dtype=np.int64)
    paired = pd.read_csv(FORMAL / "02_BASIN_PAIRED_KGE_LONG.csv", dtype={"basin_id": str})
    paired["basin_id"] = paired["basin_id"].map(canonical_id)
    params = pd.read_csv(FORMAL / "05_PARAMETER_ESTIMATES_LONG.csv", dtype={"basin_id": str})
    params["basin_id"] = params["basin_id"].map(canonical_id)
    distance = pd.read_csv(FORMAL / "07_PARAMETER_DISTANCE_BY_BASIN.csv", dtype={"basin_id": str})
    distance["basin_id"] = distance["basin_id"].map(canonical_id)
    raw, normalized, attr_meta = canonical_attributes(ids)
    return ids, paired, params, distance, raw, normalized, attr_meta


def canonical_attributes(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    caravan = np.load(CARAVAN_PATH)
    gage = np.load(GAGE_PATH).astype(np.int64)
    if caravan.shape != (len(gage), len(CAMELS_35_ATTRIBUTES)):
        raise RuntimeError(f"unexpected Caravan contract shape: {caravan.shape}")
    lookup = {int(basin): i for i, basin in enumerate(gage)}
    if any(int(b) not in lookup for b in ids):
        raise RuntimeError("canonical basin IDs are not all present in gage_id.npy")
    raw = caravan[[lookup[int(b)] for b in ids]].astype(np.float64, copy=True)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    for index in (11, 23, 34):
        minimum = float(raw[:, index].min())
        raw[:, index] = np.log(raw[:, index] + (abs(minimum) + 1.0 if minimum < 0 else 1.0))
    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, keepdims=True) + 1e-6
    normalized = (raw - mean) / std
    meta = {
        "path": str(CARAVAN_PATH),
        "sha256": sha256_file(CARAVAN_PATH),
        "shape": list(caravan.shape),
        "gage_path": str(GAGE_PATH),
        "gage_sha256": sha256_file(GAGE_PATH),
        "selected_basin_count": int(len(ids)),
        "attribute_order": list(CAMELS_35_ATTRIBUTES),
        "normalization": "nan_to_num; log columns 11,23,34; zscore selected 531 rows; std + 1e-6",
    }
    return raw, normalized, meta


def load_status() -> dict[str, Any]:
    status = json.loads((IC_ROOT / "status_summary.json").read_text())
    return {model: status[model] for model in ALL_MODELS}


def generation_for(model: str, status: dict[str, Any]) -> int:
    item = status[model]
    if item.get("generation") is not None:
        return int(item["generation"])
    values = [int(v) for v in item.get("latest_generation_by_chunk", {}).values()]
    if not values:
        values = [int(re.search(r"_gen_(\d+)\.pt$", str(x)).group(1)) for x in item.get("final_checkpoint_files", []) if re.search(r"_gen_(\d+)\.pt$", str(x))]
    if not values or len(set(values)) != 1:
        raise RuntimeError(f"cannot determine IC generation for {model}")
    return values[0]


def load_ic_restart(model: str, ids: np.ndarray, status: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load archived ten-start IC candidates; no optimizer/replay is performed."""
    generation = generation_for(model, status)
    files = sorted((IC_BEST_ROOT / model).glob("chunk_*_best.pt"))
    if not files:
        raise FileNotFoundError(f"missing IC best-training files for {model}")
    spec = get_spec(model, device="cpu")
    id_parts, latent_parts, fitness_parts = [], [], []
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if str(payload.get("model")) != model or int(payload.get("generation")) != generation:
            raise RuntimeError(f"IC provenance mismatch in {path}")
        chunk_ids = np.asarray(payload["basin_ids"], dtype=np.int64)
        latent = payload["best_latent"].detach().cpu()
        fitness = payload["best_fitness"].detach().cpu().numpy()
        if latent.shape != (len(chunk_ids) * 10, spec.dimension) or fitness.shape != (len(chunk_ids) * 10,):
            raise RuntimeError(f"unexpected ten-start shape in {path}")
        id_parts.append(chunk_ids)
        latent_parts.append(latent)
        fitness_parts.append(fitness)
    stored = np.concatenate(id_parts)
    if stored.size != ids.size or set(stored.tolist()) != set(ids.tolist()) or len(np.unique(stored)) != ids.size:
        raise RuntimeError(f"IC basin IDs are not exactly canonical for {model}")
    order = np.asarray([int(np.where(stored == basin)[0][0]) for basin in ids])
    latent = torch.cat(latent_parts).reshape(len(stored), 10, spec.dimension)[order]
    fitness = np.concatenate(fitness_parts).reshape(len(stored), 10)[order]
    with torch.inference_mode():
        u = torch.sigmoid(latent).numpy()
    bounds = spec.bounds.numpy()
    physical = bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])
    return u, physical, fitness, {"generation": generation, "starts": 10, "files": [str(x) for x in files], "parameter_names": list(spec.parameter_names)}


def corr(x: Any, y: Any) -> tuple[float, float, int]:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 3 or np.unique(a[ok]).size < 2 or np.unique(b[ok]).size < 2:
        return float("nan"), float("nan"), n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(a[ok], b[ok])
    return float(result.statistic), float(result.pvalue), n


def bh_adjust(frame: pd.DataFrame, p_column: str, group_column: str | None = None) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    groups = [(None, frame.index)] if group_column is None else frame.groupby(group_column, sort=False).groups.items()
    for _, index in groups:
        valid = frame.loc[index, p_column].notna()
        if valid.any():
            out.loc[index[valid.to_numpy()]] = multipletests(frame.loc[index[valid], p_column].to_numpy(float), method="fdr_bh")[1]
    return out


def bootstrap_median(values: np.ndarray, n_boot: int = 5000, seed: int = SEED) -> tuple[float, float]:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if not len(valid):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    # A single model has 531 basins, so this remains a bounded ~21 MB array.
    medians = np.median(valid[rng.integers(0, len(valid), size=(n_boot, len(valid)))], axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def bootstrap_corr(x: np.ndarray, y: np.ndarray, n_boot: int = 5000, seed: int = SEED) -> tuple[float, float]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        index = rng.integers(0, len(x), len(x))
        values.append(corr(x[index], y[index])[0])
    values = np.asarray(values, dtype=float)
    return float(np.nanquantile(values, 0.025)), float(np.nanquantile(values, 0.975))


def median_iqr(values: Any) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan"), float("nan"), float("nan")
    return float(np.median(x)), float(np.quantile(x, .25)), float(np.quantile(x, .75))


def main_label() -> str:
    return "SEEN_BASIN_SHARED_MAPPING_FLEXIBILITY_GAP"
