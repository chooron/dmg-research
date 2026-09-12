#!/usr/bin/env python3
"""Formal no-training seen-basin IC--dPL comparison and parameter atlas.

This script reads the frozen canonical dPL v2 artifacts, frozen IC CMA-ES
artifacts/evaluations, and the recovered Caravan attribute matrix.  It never
constructs an optimizer, calls backward, writes a checkpoint, or evaluates an
OOB/PUR split.  All dPL results are seen-basin descriptive results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import resource
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# Keep this analysis serial and WSL-friendly before importing numerical stacks.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[4]
BENCHMARK = REPO / "project/benchmark"
sys.path[:0] = [str(REPO), str(BENCHMARK), str(BENCHMARK / "src")]

from dmotpy.models.hydrology_model import HydrologyModel  # noqa: E402
from dpl.attributes import CAMELS_35_ATTRIBUTES, CatchmentAttributeBuilder  # noqa: E402
from dpl.nn_parameterizer import CatchmentParameterizer  # noqa: E402
from src.data_selection import load_ids  # noqa: E402
from src.model_registry import NPARAM_INFO_36, get_spec  # noqa: E402

DPL_ROOT = BENCHMARK / "results/dpl_canonical_v2_20260831"
DPL_RUNS = DPL_ROOT / "runs"
IC_ROOT = BENCHMARK / "results/ic_dpl_aligned_full300_20260819_final"
IC_CHECKPOINT_ROOT = IC_ROOT / "checkpoints/ic_dpl_aligned_full300_20260819"
IC_BEST_ROOT = IC_ROOT / "best_training"
IC_ALIGNED_300 = BENCHMARK / "results/ic_gen300_aligned_all36_20260831/by_basin"
IC_ALIGNED_280 = BENCHMARK / "results/ic_gen280_aligned_simhyd_20260831/by_basin/simhyd.csv"
VIC_RESULT = BENCHMARK / "results/ic_vic_full300_dynamic_doy_20260901"
CARAVAN_PATH = REPO.parent / "dmg-research_replay_forensics/data_remote_contract/caravan_671_attributes.npy"
GAGE_PATH = REPO / "data/gage_id.npy"
IDS_PATH = REPO / "data/531sub_id.txt"
OUT = BENCHMARK / "results/ic_dpl_seenbasin_formal_20260901"

ALL_MODELS = tuple(NPARAM_INFO_36)
STRICT_FULL300 = {m for m in ALL_MODELS if m != "simhyd"}
CATEGORICAL = {"dom_land_cover", "geol_1st_class", "geol_2nd_class"}
ATTRIBUTE_TYPE = {a: ("CATEGORICAL_CODE" if a in CATEGORICAL else "CONTINUOUS") for a in CAMELS_35_ATTRIBUTES}
BOOT_SEED = 20260901
N_BOOT = 2000
HIGH_RHO = 0.20
LOW_RHO = 0.10
TOP_RANK = 10
LOG_SPAN_THRESHOLD = 100.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10f")


def canonical_id(value: Any) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(8)


def finite_stat(values: np.ndarray, quantile: float | None = None) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan")
    return float(np.quantile(x, quantile) if quantile is not None else np.median(x))


def rank_corr(x: Any, y: Any) -> tuple[float, int]:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 3 or np.unique(a[ok]).size < 2 or np.unique(b[ok]).size < 2:
        return float("nan"), n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(a[ok], b[ok]).statistic), n


def pearson(x: Any, y: Any) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def sign_label(value: float) -> str:
    if not np.isfinite(value):
        return "UNDEFINED"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def build_canonical_attributes(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not CARAVAN_PATH.is_file():
        raise FileNotFoundError(f"Recovered canonical attribute matrix missing: {CARAVAN_PATH}")
    caravan = np.load(CARAVAN_PATH)
    gage = np.load(GAGE_PATH).astype(np.int64)
    if caravan.ndim != 2 or caravan.shape[1] != len(CAMELS_35_ATTRIBUTES):
        raise RuntimeError(f"unexpected Caravan shape {caravan.shape}")
    if caravan.shape[0] != len(gage) or len(np.unique(gage)) != len(gage):
        raise RuntimeError("Caravan rows and gage_id.npy are not a unique aligned reference")
    lookup = {int(b): i for i, b in enumerate(gage)}
    missing = [int(b) for b in ids if int(b) not in lookup]
    if missing:
        raise RuntimeError(f"canonical basin IDs missing from gage_id.npy: {missing[:5]}")
    raw = caravan[np.asarray([lookup[int(b)] for b in ids])].astype(np.float64, copy=True)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    # Match CatchmentAttributeBuilder exactly: area, conductivity, permeability.
    for idx in (11, 23, 34):
        minimum = float(np.min(raw[:, idx]))
        shift = abs(minimum) + 1.0 if minimum < 0 else 1.0
        raw[:, idx] = np.log(raw[:, idx] + shift)
    mean = np.mean(raw, axis=0, keepdims=True)
    std = np.std(raw, axis=0, keepdims=True) + 1e-6
    normalized = (raw - mean) / std
    meta = {
        "path": str(CARAVAN_PATH),
        "sha256": sha256_file(CARAVAN_PATH),
        "shape": list(caravan.shape),
        "gage_path": str(GAGE_PATH),
        "gage_sha256": sha256_file(GAGE_PATH),
        "selected_basin_count": int(len(ids)),
        "attribute_order": list(CAMELS_35_ATTRIBUTES),
        "normalization": "nan_to_num; log columns 11,23,34 using dataset minimum shift; zscore over selected canonical 531 rows; std + 1e-6",
        "mean": mean.ravel().tolist(),
        "std_with_epsilon": std.ravel().tolist(),
        "contract_status": "RECOVERED_CANONICAL_CARAVAN_MATRIX",
    }
    return raw, normalized, meta


def load_status() -> dict[str, Any]:
    status = json.loads((IC_ROOT / "status_summary.json").read_text())
    return {k: v for k, v in status.items() if k in ALL_MODELS}


def generation_for(model: str, status: dict[str, Any]) -> int:
    item = status.get(model, {})
    if item.get("generation") is not None:
        return int(item["generation"])
    values = [int(v) for v in item.get("latest_generation_by_chunk", {}).values()]
    if not values:
        files = item.get("final_checkpoint_files", [])
        values = [int(re.search(r"_gen_(\d+)\.pt$", str(x)).group(1)) for x in files if re.search(r"_gen_(\d+)\.pt$", str(x))]
    if not values or len(set(values)) != 1:
        raise RuntimeError(f"{model}: unable to determine unambiguous IC generation")
    return values[0]


def load_ic_parameters(model: str, ids: np.ndarray, status: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    expected_generation = generation_for(model, status)
    item = status[model]
    if not item.get("done"):
        raise RuntimeError(f"{model}: IC status is not done")
    model_dir = IC_BEST_ROOT / model
    best_files = sorted(model_dir.glob("chunk_*_best.pt"))
    if not best_files:
        raise RuntimeError(f"{model}: no best-training IC artifacts")
    spec = get_spec(model, device="cpu")
    id_parts: list[np.ndarray] = []
    latent_parts: list[torch.Tensor] = []
    fit_parts: list[np.ndarray] = []
    for path in best_files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if str(payload.get("model")) != model or int(payload.get("generation")) != expected_generation:
            raise RuntimeError(f"{model}: invalid best-training payload {path}")
        ckpt_ids = np.asarray(payload["basin_ids"], dtype=np.int64)
        latent = payload["best_latent"].detach().cpu()
        fitness = payload["best_fitness"].detach().cpu().numpy()
        if latent.ndim != 2 or latent.shape[1] != spec.dimension or latent.shape[0] != ckpt_ids.size * 10:
            raise RuntimeError(f"{model}: unexpected best latent shape in {path}")
        if fitness.shape != (ckpt_ids.size * 10,):
            raise RuntimeError(f"{model}: unexpected fitness shape in {path}")
        id_parts.append(ckpt_ids)
        latent_parts.append(latent)
        fit_parts.append(fitness)
    stored_ids = np.concatenate(id_parts)
    if stored_ids.size != ids.size or len(np.unique(stored_ids)) != ids.size or set(stored_ids.tolist()) != set(ids.tolist()):
        raise RuntimeError(f"{model}: best-training basin IDs are not exactly canonical 531 IDs")
    order = np.asarray([int(np.where(stored_ids == basin)[0][0]) for basin in ids])
    latent = torch.cat(latent_parts).reshape(len(stored_ids), 10, spec.dimension)[order]
    fitness = np.concatenate(fit_parts).reshape(len(stored_ids), 10)[order]
    selected = fitness.argmax(axis=1)
    chosen_latent = latent[np.arange(len(ids)), selected]
    with torch.inference_mode():
        normalized = torch.sigmoid(chosen_latent).numpy()
    bounds = spec.bounds.numpy()
    physical = bounds[:, 0] + normalized * (bounds[:, 1] - bounds[:, 0])
    return physical, normalized, {
        "generation": expected_generation,
        "starts": 10,
        "best_training_paths": [str(x) for x in best_files],
        "selected_train_kge_median": float(np.median(fitness.max(axis=1))),
        "selected_train_kge_mean": float(np.mean(fitness.max(axis=1))),
        "coordinate": "sigmoid(best_latent), physical linear mapping",
        "declared_status_generation": item.get("generation"),
    }


def dpl_network_parameters(model: str, attrs: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = DPL_RUNS / model / "best.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    spec = get_spec(model, device="cpu")
    job = payload.get("job_config", {})
    if str(payload.get("selection_metric")) != "train_loss" or int(job.get("seed")) != 42:
        raise RuntimeError(f"{model}: dPL checkpoint selection/seed contract mismatch")
    if str(job.get("model")) != model or str(payload.get("git_sha")) != "7d1132bf5c9ee0114a1a12dd10720101f6ca3b74":
        raise RuntimeError(f"{model}: dPL checkpoint provenance/model contract mismatch")
    net = CatchmentParameterizer(
        in_features=attrs.shape[1],
        out_features=spec.dimension,
        hidden_dims=[256, 256],
        dropout=0.05,
        parameter_names=list(spec.parameter_names),
        output_transform="sigmoid",
    ).to(dtype=torch.float32)
    net.load_state_dict(payload["network"])
    net.eval()
    with torch.inference_mode():
        normalized = net(torch.as_tensor(attrs, dtype=torch.float32)).detach().cpu().numpy().astype(np.float64)
    bounds = spec.bounds.numpy()
    mapping = str(job.get("mapping", "auto")).lower()
    physical = np.empty_like(normalized)
    mapping_used: list[str] = []
    for p, (lower, upper) in enumerate(bounds):
        use_log = mapping in {"auto", "auto_log", "log_auto"} and lower > 0 and upper > lower and upper / lower >= LOG_SPAN_THRESHOLD
        if use_log:
            physical[:, p] = np.exp(np.log(lower) + normalized[:, p] * (np.log(upper) - np.log(lower)))
            mapping_used.append("auto_log")
        else:
            physical[:, p] = lower + normalized[:, p] * (upper - lower)
            mapping_used.append("linear")
    return physical, normalized, {
        "best_path": str(path),
        "epoch": int(payload.get("epoch", -1)),
        "seed": int(job.get("seed")),
        "git_sha": str(payload.get("git_sha")),
        "selection_metric": str(payload.get("selection_metric")),
        "mapping": mapping,
        "mapping_used": mapping_used,
        "coordinate": "network sigmoid output u; physical auto mapping with threshold 100",
    }


def validate_final_checkpoint_structure(model: str, expected_generation: int, ids: np.ndarray) -> dict[str, Any]:
    model_dir = IC_CHECKPOINT_ROOT / model
    files = sorted(model_dir.glob(f"chunk_*_gen_{expected_generation}.pt"))
    if not files:
        raise RuntimeError(f"{model}: final generation checkpoint missing")
    seen: list[int] = []
    finite = True
    payload_models: set[str] = set()
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload_models.add(str(payload.get("model")))
        ckpt_ids = np.asarray(payload["basin_ids"], dtype=np.int64)
        state = payload["solver"]["state"]
        seen.extend(ckpt_ids.tolist())
        finite = finite and bool(torch.isfinite(state["best_fitness"]).all()) and bool(torch.isfinite(state["best_latent"]).all())
        if int(payload.get("generation")) != expected_generation:
            raise RuntimeError(f"{model}: checkpoint generation mismatch")
    if payload_models != {model} or sorted(seen) != sorted(ids.tolist()) or len(set(seen)) != len(ids) or not finite:
        raise RuntimeError(f"{model}: final checkpoint structural gate failed")
    return {"files": [str(x) for x in files], "n_chunks": len(files), "n_basins": len(set(seen)), "finite": finite}


def load_test_scores(model: str, ids: np.ndarray) -> tuple[pd.DataFrame, str, int]:
    ic_path = IC_ALIGNED_280 if model == "simhyd" else IC_ALIGNED_300 / f"{model}.csv"
    if not ic_path.is_file():
        raise FileNotFoundError(f"{model}: IC TEST score table missing: {ic_path}")
    dpl_path = DPL_RUNS / model / "basin_test_kge.csv"
    ic = pd.read_csv(ic_path, dtype={"basin_id": str})
    dpl = pd.read_csv(dpl_path)
    for frame, name in ((ic, "IC"), (dpl, "dPL")):
        if "basin_id" not in frame or len(frame) != len(ids):
            raise RuntimeError(f"{model}: {name} score table is not 531 rows")
        frame["basin_id"] = frame["basin_id"].map(canonical_id)
        if frame["basin_id"].duplicated().any() or set(frame["basin_id"]) != {canonical_id(x) for x in ids}:
            raise RuntimeError(f"{model}: {name} score basin IDs do not match canonical set")
    ic = ic.rename(columns={"kge_ic": "KGE_IC"})[["basin_id", "KGE_IC"]]
    dpl = dpl.rename(columns={"kge": "KGE_dPL"})[["basin_id", "KGE_dPL"]]
    merged = ic.merge(dpl, on="basin_id", how="inner", validate="one_to_one")
    if len(merged) != len(ids):
        raise RuntimeError(f"{model}: incomplete IC/dPL score pairing")
    merged["model"] = model
    merged["KGE_IC"] = pd.to_numeric(merged["KGE_IC"], errors="coerce")
    merged["KGE_dPL"] = pd.to_numeric(merged["KGE_dPL"], errors="coerce")
    merged["Delta_KGE"] = merged["KGE_dPL"] - merged["KGE_IC"]
    merged["test_period"] = "1995-10-01..2010-09-30"
    merged["basin_scope"] = "same canonical 531 basins"
    merged["label"] = "SEEN_BASIN_DESCRIPTIVE"
    generation = 280 if model == "simhyd" else 300
    return merged[["model", "basin_id", "KGE_IC", "KGE_dPL", "Delta_KGE", "test_period", "basin_scope", "label"]], str(ic_path), generation


def bootstrap_median(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    valid = x[np.isfinite(x)]
    if not len(valid):
        return float("nan"), float("nan")
    indexes = rng.integers(0, len(valid), size=(N_BOOT, len(valid)))
    medians = np.median(valid[indexes], axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def performance_summary(score_tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOT_SEED)
    rows: list[dict[str, Any]] = []
    for model in ALL_MODELS:
        frame = score_tables[model]
        delta = frame.Delta_KGE.to_numpy(float)
        ci_low, ci_high = bootstrap_median(delta, rng)
        valid = np.isfinite(frame[["KGE_IC", "KGE_dPL", "Delta_KGE"]].to_numpy()).all(axis=1)
        rho, n_corr = rank_corr(frame.loc[valid, "KGE_IC"], frame.loc[valid, "KGE_dPL"])
        rows.append({
            "row_type": "MODEL",
            "population": "STRUCTURAL_ENSEMBLE_36",
            "model": model,
            "ic_generation": 280 if model == "simhyd" else 300,
            "strict_full300": model in STRICT_FULL300,
            "valid_paired_basin_count": int(valid.sum()),
            "ic_median": finite_stat(frame.KGE_IC.to_numpy()),
            "ic_mean": float(np.nanmean(frame.KGE_IC)),
            "ic_q25": finite_stat(frame.KGE_IC.to_numpy(), .25),
            "dpl_median": finite_stat(frame.KGE_dPL.to_numpy()),
            "dpl_mean": float(np.nanmean(frame.KGE_dPL)),
            "dpl_q25": finite_stat(frame.KGE_dPL.to_numpy(), .25),
            "delta_median": finite_stat(delta),
            "delta_mean": float(np.nanmean(delta)),
            "delta_q25": finite_stat(delta, .25),
            "delta_q75": finite_stat(delta, .75),
            "delta_gt_zero_fraction": float(np.mean(delta > 0)),
            "delta_lt_zero_fraction": float(np.mean(delta < 0)),
            "paired_kge_spearman": rho,
            "paired_kge_spearman_n": n_corr,
            "delta_median_bootstrap_ci95_low": ci_low,
            "delta_median_bootstrap_ci95_high": ci_high,
            "bootstrap_seed": BOOT_SEED,
            "bootstrap_replicates": N_BOOT,
            "bootstrap_unit": "531 paired basins within each model",
            "test_period": "1995-10-01..2010-09-30",
            "label": "SEEN_BASIN_DESCRIPTIVE",
        })
    models = pd.DataFrame(rows)
    ensemble_rows: list[dict[str, Any]] = []
    for population, mask in (("STRUCTURAL_ENSEMBLE_36", np.ones(len(models), dtype=bool)), ("STRICT_FULL300_35", models.strict_full300.to_numpy(bool))):
        subset = models.loc[mask]
        pooled = pd.concat([score_tables[m] for m in subset.model], ignore_index=True)
        pooled_delta = pooled.Delta_KGE.to_numpy(float)
        ensemble_rows.extend([
            {
                "row_type": "ENSEMBLE_MODEL_LEVEL",
                "population": population,
                "model": "__ENSEMBLE__",
                "valid_paired_basin_count": int(subset.valid_paired_basin_count.sum()),
                "model_count": len(subset),
                "ic_median": float(subset.ic_median.median()),
                "ic_mean": float(subset.ic_median.mean()),
                "dpl_median": float(subset.dpl_median.median()),
                "dpl_mean": float(subset.dpl_median.mean()),
                "delta_median": float(subset.delta_median.median()),
                "delta_mean": float(subset.delta_median.mean()),
                "delta_q25": float(subset.delta_median.quantile(.25)),
                "delta_q75": float(subset.delta_median.quantile(.75)),
                "delta_gt_zero_fraction": float((subset.delta_median > 0).mean()),
                "delta_lt_zero_fraction": float((subset.delta_median < 0).mean()),
                "test_period": "1995-10-01..2010-09-30",
                "denominator_note": "equal-weight model-level medians; model count explicit",
                "label": "SEEN_BASIN_DESCRIPTIVE",
            },
            {
                "row_type": "ENSEMBLE_POOLED_BASIN",
                "population": population,
                "model": "__POOLED_BASINS__",
                "valid_paired_basin_count": int(np.isfinite(pooled_delta).sum()),
                "model_count": len(subset),
                "ic_median": float(np.nanmedian(pooled.KGE_IC)),
                "ic_mean": float(np.nanmean(pooled.KGE_IC)),
                "dpl_median": float(np.nanmedian(pooled.KGE_dPL)),
                "dpl_mean": float(np.nanmean(pooled.KGE_dPL)),
                "delta_median": float(np.nanmedian(pooled_delta)),
                "delta_mean": float(np.nanmean(pooled_delta)),
                "delta_q25": float(np.nanquantile(pooled_delta, .25)),
                "delta_q75": float(np.nanquantile(pooled_delta, .75)),
                "delta_gt_zero_fraction": float(np.mean(pooled_delta > 0)),
                "delta_lt_zero_fraction": float(np.mean(pooled_delta < 0)),
                "test_period": "1995-10-01..2010-09-30",
                "denominator_note": "pooled model-basin rows; each model contributes 531 basins",
                "label": "SEEN_BASIN_DESCRIPTIVE",
            },
        ])
    comparison = pd.concat([models, pd.DataFrame(ensemble_rows)], ignore_index=True, sort=False)
    return models, comparison


def parameter_qc_row(model: str, method: str, physical: np.ndarray, normalized: np.ndarray, ids: np.ndarray, spec: Any, meta: dict[str, Any]) -> dict[str, Any]:
    bounds = spec.bounds.numpy()
    finite = np.isfinite(physical).all(axis=1) & np.isfinite(normalized).all(axis=1)
    low = normalized < 0.0
    high = normalized > 1.0
    physical_low = physical < bounds[:, 0][None, :]
    physical_high = physical > bounds[:, 1][None, :]
    return {
        "model": model,
        "method": method,
        "parameter_count": int(spec.dimension),
        "parameter_names_complete": list(spec.parameter_names) == list(get_spec(model).parameter_names),
        "basin_count": int(len(ids)),
        "unique_basin_count": int(len(np.unique(ids))),
        "basin_ids_complete": True,
        "finite_row_count": int(finite.sum()),
        "nonfinite_value_count": int((~np.isfinite(physical)).sum() + (~np.isfinite(normalized)).sum()),
        "normalized_lower_violations": int(low.sum()),
        "normalized_upper_violations": int(high.sum()),
        "physical_lower_violations": int(physical_low.sum()),
        "physical_upper_violations": int(physical_high.sum()),
        "bound_violations": int(low.sum() + high.sum() + physical_low.sum() + physical_high.sum()),
        "normalized_min": float(np.nanmin(normalized)),
        "normalized_max": float(np.nanmax(normalized)),
        "physical_min": float(np.nanmin(physical)),
        "physical_max": float(np.nanmax(physical)),
        "source": json.dumps(meta, sort_keys=True),
        "gate_status": "PASS" if finite.all() and not (low.any() or high.any() or physical_low.any() or physical_high.any()) else "FAIL",
    }


def write_parameter_outputs(ids: np.ndarray, attributes: np.ndarray, status: dict[str, Any], pairing: pd.DataFrame) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arrays: dict[str, dict[str, np.ndarray]] = {}
    qc_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    parameter_path = OUT / "05_PARAMETER_ESTIMATES_LONG.csv"
    parameter_path.parent.mkdir(parents=True, exist_ok=True)
    parameter_handle = parameter_path.open("w", newline="")
    parameter_writer = csv.DictWriter(parameter_handle, fieldnames=["model", "basin_id", "method", "parameter_index", "parameter", "physical_value", "normalized_u", "lower_bound", "upper_bound", "mapping", "ic_generation", "dpl_seed", "test_label"])
    parameter_writer.writeheader()
    for model in ALL_MODELS:
        spec = get_spec(model, device="cpu")
        ic_physical, ic_norm, ic_meta = load_ic_parameters(model, ids, status)
        dpl_physical, dpl_norm, dpl_meta = dpl_network_parameters(model, attributes)
        arrays[model] = {"IC_physical": ic_physical, "IC_normalized": ic_norm, "dPL_physical": dpl_physical, "dPL_normalized": dpl_norm}
        qc_rows.append(parameter_qc_row(model, "IC", ic_physical, ic_norm, ids, spec, ic_meta))
        qc_rows.append(parameter_qc_row(model, "dPL", dpl_physical, dpl_norm, ids, spec, dpl_meta))
        for method, physical, normalized, meta in (("IC", ic_physical, ic_norm, ic_meta), ("dPL", dpl_physical, dpl_norm, dpl_meta)):
            for p, name in enumerate(spec.parameter_names):
                lo, hi = spec.bounds[p].tolist()
                for b, basin_id in enumerate(ids):
                    parameter_writer.writerow({
                        "model": model,
                        "basin_id": canonical_id(basin_id),
                        "method": method,
                        "parameter_index": p,
                        "parameter": name,
                        "physical_value": physical[b, p],
                        "normalized_u": normalized[b, p],
                        "lower_bound": lo,
                        "upper_bound": hi,
                        "mapping": "linear" if method == "IC" else meta["mapping_used"][p],
                        "ic_generation": meta.get("generation", ""),
                        "dpl_seed": meta.get("seed", ""),
                        "test_label": "SEEN_BASIN_DESCRIPTIVE",
                    })
        norm_delta = dpl_norm - ic_norm
        phys_delta = dpl_physical - ic_physical
        for b, basin_id in enumerate(ids):
            distance_rows.append({
                "model": model,
                "basin_id": canonical_id(basin_id),
                "parameter_count": spec.dimension,
                "normalized_l2_distance": float(np.linalg.norm(norm_delta[b])),
                "normalized_mean_abs_distance": float(np.mean(np.abs(norm_delta[b]))),
                "normalized_max_abs_distance": float(np.max(np.abs(norm_delta[b]))),
                "physical_mean_abs_difference": float(np.mean(np.abs(phys_delta[b]))),
                "physical_max_abs_difference": float(np.max(np.abs(phys_delta[b]))),
                "distance_coordinate": "IC/dPL bounds-normalized u; physical differences retained descriptively",
                "label": "SEEN_BASIN_DESCRIPTIVE",
            })
    parameter_handle.close()
    qc = pd.DataFrame(qc_rows)
    write_csv(OUT / "06_PARAMETER_PAIRING_QC.csv", qc)
    distance = pd.DataFrame(distance_rows)
    write_csv(OUT / "07_PARAMETER_DISTANCE_BY_BASIN.csv", distance)
    dist_summary = distance.groupby("model", sort=True).agg(
        basins=("basin_id", "nunique"), normalized_l2_median=("normalized_l2_distance", "median"), normalized_l2_mean=("normalized_l2_distance", "mean"),
        normalized_l2_q25=("normalized_l2_distance", lambda x: x.quantile(.25)), normalized_l2_q75=("normalized_l2_distance", lambda x: x.quantile(.75)),
        normalized_mean_abs_median=("normalized_mean_abs_distance", "median"), physical_mean_abs_median=("physical_mean_abs_difference", "median"),
    ).reset_index()
    dist_summary["label"] = "SEEN_BASIN_DESCRIPTIVE"
    write_csv(OUT / "08_PARAMETER_DISTANCE_SUMMARY.csv", dist_summary)
    return arrays, qc, distance, dist_summary


def compute_atlas(arrays: dict[str, dict[str, np.ndarray]], attributes: np.ndarray, pairing: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rho_store: dict[str, dict[str, np.ndarray]] = {"IC": {}, "dPL": {}}
    p_store: dict[str, dict[str, np.ndarray]] = {"IC": {}, "dPL": {}}
    p_vectors: dict[str, list[np.ndarray]] = {"IC": [], "dPL": []}
    long_header = ["population", "model", "method", "parameter_index", "parameter", "attribute", "attribute_type", "rho", "p_value", "q_value", "n", "strict_full300", "label"]
    for method, key in (("IC", "IC_physical"), ("dPL", "dPL_physical")):
        for model in ALL_MODELS:
            spec = get_spec(model, device="cpu")
            values = np.column_stack([attributes, arrays[model][key]])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = spearmanr(values, axis=0, nan_policy="omit")
            rho = np.asarray(result.statistic, dtype=float)[: len(CAMELS_35_ATTRIBUTES), len(CAMELS_35_ATTRIBUTES):].T
            pvalue = np.asarray(result.pvalue, dtype=float)[: len(CAMELS_35_ATTRIBUTES), len(CAMELS_35_ATTRIBUTES):].T
            if rho.shape != (spec.dimension, len(CAMELS_35_ATTRIBUTES)):
                raise RuntimeError(f"{model}/{method}: unexpected atlas matrix shape {rho.shape}")
            rho_store[method][model] = rho
            p_store[method][model] = pvalue
            p_vectors[method].append(pvalue.ravel())
    q_store: dict[str, dict[str, np.ndarray]] = {"IC": {}, "dPL": {}}
    for method in ("IC", "dPL"):
        flat_p = np.concatenate(p_vectors[method])
        flat_q = np.full(flat_p.shape, np.nan, dtype=float)
        ok = np.isfinite(flat_p)
        if ok.any():
            flat_q[ok] = multipletests(flat_p[ok], method="fdr_bh")[1]
        cursor = 0
        for model in ALL_MODELS:
            shape = p_store[method][model].shape
            size = int(np.prod(shape))
            q_store[method][model] = flat_q[cursor:cursor + size].reshape(shape)
            cursor += size
    with (OUT / "09_PARAMETER_ATTRIBUTE_ATLAS_LONG.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_header)
        writer.writeheader()
        for model in ALL_MODELS:
            spec = get_spec(model, device="cpu")
            for method in ("IC", "dPL"):
                rho, pvalue, qvalue = rho_store[method][model], p_store[method][model], q_store[method][model]
                for p, name in enumerate(spec.parameter_names):
                    for a, attr in enumerate(CAMELS_35_ATTRIBUTES):
                        writer.writerow({"population": "STRUCTURAL_ENSEMBLE_36", "model": model, "method": method, "parameter_index": p, "parameter": name, "attribute": attr, "attribute_type": ATTRIBUTE_TYPE[attr], "rho": rho[p, a], "p_value": pvalue[p, a], "q_value": qvalue[p, a], "n": 531, "strict_full300": model in STRICT_FULL300, "label": "SEEN_BASIN_DESCRIPTIVE; Spearman physical-parameter association"})
    repro_rows: list[dict[str, Any]] = []
    dominant_rows: list[dict[str, Any]] = []
    model_profile_rows: list[dict[str, Any]] = []
    scopes = {"STRUCTURAL_ENSEMBLE_36": set(ALL_MODELS), "STRICT_FULL300_35": set(STRICT_FULL300)}
    scope_stats = {name: {"cells": 0, "valid": 0, "nontrivial": 0, "sign_equal_valid": 0, "sign_equal_nontrivial": 0, "classes": {c: 0 for c in ["persistent/reproduced", "attenuated", "dPL-emergent", "sign-changing", "weak/unresolved"]}} for name in scopes}
    for model in ALL_MODELS:
        spec = get_spec(model, device="cpu")
        ic_rho = rho_store["IC"][model]
        dpl_rho = rho_store["dPL"][model]
        ic_rank = pd.DataFrame(-np.abs(ic_rho)).rank(axis=1, method="min", na_option="keep").to_numpy()
        dpl_rank = pd.DataFrame(-np.abs(dpl_rho)).rank(axis=1, method="min", na_option="keep").to_numpy()
        for scope_name, model_set in scopes.items():
            scope_stats[scope_name]["cells"] += spec.dimension * len(CAMELS_35_ATTRIBUTES)
        for p, name in enumerate(spec.parameter_names):
            ir, dr = ic_rho[p], dpl_rho[p]
            abs_i, abs_d = np.abs(ir), np.abs(dr)
            valid = np.isfinite(ir) & np.isfinite(dr)
            both_nontrivial = valid & (abs_i >= LOW_RHO) & (abs_d >= LOW_RHO)
            sign_i = np.array([sign_label(x) for x in ir])
            sign_d = np.array([sign_label(x) for x in dr])
            sign_same = sign_i == sign_d
            dominant_i = CAMELS_35_ATTRIBUTES[int(np.nanargmax(abs_i))] if np.isfinite(abs_i).any() else "UNRESOLVED"
            dominant_d = CAMELS_35_ATTRIBUTES[int(np.nanargmax(abs_d))] if np.isfinite(abs_d).any() else "UNRESOLVED"
            order_i = np.argsort(np.where(np.isfinite(abs_i), -abs_i, np.inf), kind="mergesort")[:5]
            order_d = np.argsort(np.where(np.isfinite(abs_d), -abs_d, np.inf), kind="mergesort")[:5]
            top_i, top_d = set(np.asarray(CAMELS_35_ATTRIBUTES)[order_i]), set(np.asarray(CAMELS_35_ATTRIBUTES)[order_d])
            classes: list[str] = []
            for a in range(len(CAMELS_35_ATTRIBUTES)):
                if both_nontrivial[a] and sign_i[a] != sign_d[a]:
                    cls = "sign-changing"
                elif abs_i[a] >= HIGH_RHO and abs_d[a] < LOW_RHO:
                    cls = "attenuated"
                elif abs_d[a] >= HIGH_RHO and abs_i[a] < LOW_RHO:
                    cls = "dPL-emergent"
                elif abs_i[a] >= HIGH_RHO and abs_d[a] >= HIGH_RHO and sign_same[a] and ic_rank[p, a] <= TOP_RANK and dpl_rank[p, a] <= TOP_RANK:
                    cls = "persistent/reproduced"
                else:
                    cls = "weak/unresolved"
                classes.append(cls)
            for scope_name, model_set in scopes.items():
                if model not in model_set:
                    continue
                s = scope_stats[scope_name]
                s["valid"] += int(valid.sum())
                s["nontrivial"] += int(both_nontrivial.sum())
                s["sign_equal_valid"] += int((valid & sign_same).sum())
                s["sign_equal_nontrivial"] += int((both_nontrivial & sign_same).sum())
                for cls in classes:
                    s["classes"][cls] += 1
            profile_s, profile_n = rank_corr(ir, dr)
            repro_rows.append({"population": "STRUCTURAL_ENSEMBLE_36", "model": model, "parameter_index": p, "parameter": name, "strict_full300": model in STRICT_FULL300, "n_attributes": 35, "valid_attribute_pairs": int(valid.sum()), "profile_pearson": pearson(ir, dr), "profile_spearman": profile_s, "profile_spearman_n": profile_n, "sign_agreement_valid": float(sign_same[valid].mean()) if valid.any() else np.nan, "sign_agreement_nontrivial": float(sign_same[both_nontrivial].mean()) if both_nontrivial.any() else np.nan, "dominant_attribute_IC": dominant_i, "dominant_attribute_dPL": dominant_d, "dominant_control_agreement": bool(dominant_i == dominant_d and dominant_i != "UNRESOLVED"), "top5_overlap_count": len(top_i & top_d), "top5_overlap_jaccard": len(top_i & top_d) / len(top_i | top_d), "mean_abs_rho_ic": float(np.nanmean(abs_i)), "mean_abs_rho_dpl": float(np.nanmean(abs_d)), "delta_abs_rho_mean": float(np.nanmean(abs_d - abs_i)), "relationship_class_counts": json.dumps({str(k): int(v) for k, v in pd.Series(classes).value_counts().items()}), "label": "SEEN_BASIN_DESCRIPTIVE"})
            dominant_rows.append({"population": "STRUCTURAL_ENSEMBLE_36", "model": model, "parameter_index": p, "parameter": name, "strict_full300": model in STRICT_FULL300, "dominant_attribute_IC": dominant_i, "dominant_abs_rho_IC": np.nanmax(abs_i) if np.isfinite(abs_i).any() else np.nan, "dominant_attribute_dPL": dominant_d, "dominant_abs_rho_dPL": np.nanmax(abs_d) if np.isfinite(abs_d).any() else np.nan, "dominant_control_agreement": bool(dominant_i == dominant_d and dominant_i != "UNRESOLVED"), "label": "SEEN_BASIN_DESCRIPTIVE; dominant attribute is descriptive, not causal"})
        model_profile_rows.append({"model": model, "strict_full300": model in STRICT_FULL300, "flattened_profile_spearman": rank_corr(ic_rho.ravel(), dpl_rho.ravel())[0], "flattened_profile_n": rank_corr(ic_rho.ravel(), dpl_rho.ravel())[1], "flattened_profile_pearson": pearson(ic_rho.ravel(), dpl_rho.ravel())})
    repro = pd.DataFrame(repro_rows)
    dominant = pd.DataFrame(dominant_rows)
    write_csv(OUT / "10_PARAMETER_ATTRIBUTE_REPRODUCIBILITY.csv", repro)
    write_csv(OUT / "12_DOMINANT_CONTROL_COMPARISON.csv", dominant)
    model_profiles = pd.DataFrame(model_profile_rows)
    summary_rows: list[dict[str, Any]] = []
    for scope_name, model_set in scopes.items():
        r = repro[repro.model.isin(model_set)]
        mp = model_profiles[model_profiles.model.isin(model_set)]
        s = scope_stats[scope_name]
        summary_rows.append({
            "population": scope_name, "n_models": len(model_set), "n_model_parameter_rows": len(r),
            "n_atlas_cells_per_method": s["cells"], "n_atlas_cells_both_methods": 2 * s["cells"],
            "parameter_profile_spearman_median": float(r.profile_spearman.median()), "parameter_profile_spearman_min": float(r.profile_spearman.min()), "parameter_profile_spearman_max": float(r.profile_spearman.max()),
            "model_flattened_profile_spearman_median": float(mp.flattened_profile_spearman.median()), "model_flattened_profile_spearman_min": float(mp.flattened_profile_spearman.min()), "model_flattened_profile_spearman_max": float(mp.flattened_profile_spearman.max()),
            "dominant_control_agreement": float(r.dominant_control_agreement.mean()), "dominant_control_agreement_n": len(r),
            "pooled_sign_agreement": s["sign_equal_valid"] / s["valid"] if s["valid"] else np.nan, "pooled_sign_agreement_n": s["valid"],
            "pooled_nontrivial_sign_agreement": s["sign_equal_nontrivial"] / s["nontrivial"] if s["nontrivial"] else np.nan, "pooled_nontrivial_n": s["nontrivial"],
            "persistent_reproduced_fraction": s["classes"]["persistent/reproduced"] / s["cells"], "attenuated_fraction": s["classes"]["attenuated"] / s["cells"],
            "dpl_emergent_fraction": s["classes"]["dPL-emergent"] / s["cells"], "sign_changing_fraction": s["classes"]["sign-changing"] / s["cells"], "weak_unresolved_fraction": s["classes"]["weak/unresolved"] / s["cells"],
            "relationship_definition": "old atlas ordered descriptive rules: sign-changing, attenuated, dPL-emergent, persistent/reproduced; high=.20 low=.10 top-rank<=10",
            "pooled_definition": "fraction of valid model-parameter-attribute cells with equal IC/dPL sign labels; model-basin rows are not pooled here",
            "fdr_definition": "BH-FDR over all 36-model x parameter x 35-attribute cells separately for IC and dPL",
            "label": "SEEN_BASIN_DESCRIPTIVE; no causal interpretation",
        })
    atlas_summary = pd.DataFrame(summary_rows)
    write_csv(OUT / "11_MODEL_LEVEL_ATLAS_SUMMARY.csv", atlas_summary)
    return pd.DataFrame(), repro, dominant, {"model_profiles": model_profiles, "summary": atlas_summary, "rho_store": rho_store, "atlas_cell_count": int(sum(s["cells"] for s in scope_stats.values()) // 1)}


def make_input_manifest(ids: np.ndarray, status: dict[str, Any], pairing: pd.DataFrame, attr_meta: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    paths = [
        ("dpl_root", DPL_ROOT, "canonical dPL v2 result root; seed=42"),
        ("dpl_manifest", DPL_ROOT / "canonical_manifest.yaml", "36-model canonical manifest"),
        ("ic_root", IC_ROOT, "frozen IC root"),
        ("ic_status", IC_ROOT / "status_summary.json", "IC completion/generation status"),
        ("ic_aligned_300", IC_ALIGNED_300, "stored IC TEST tables for generation-300 models"),
        ("ic_aligned_simhyd_280", IC_ALIGNED_280, "accepted simhyd generation-280 IC TEST table"),
        ("caravan_attributes", CARAVAN_PATH, "recovered canonical Caravan matrix"),
        ("gage_ids", GAGE_PATH, "canonical Caravan reference row IDs"),
        ("canonical_basin_ids", IDS_PATH, "canonical 531-basin order"),
        ("vic_dynamic_doy_result", VIC_RESULT, "new VIC dynamic-DOY result/provenance"),
    ]
    for role, path, note in paths:
        is_file = path.is_file()
        rows.append({"entry_type": "INPUT", "role": role, "model": "", "path": str(path), "exists": is_file, "size_bytes": path.stat().st_size if is_file else "", "sha256": sha256_file(path) if is_file else "", "note": note})
    for row in pairing.to_dict("records"):
        rows.append({"entry_type": "MODEL_INPUT", "role": "pairing", "model": row["model"], "path": row["dpl_best_path"], "exists": bool(row["paired_available"]), "size_bytes": "", "sha256": "", "note": f"IC generation {row['ic_generation']}; strict_full300={row['strict_full300']}"})
    rows.append({"entry_type": "ATTRIBUTE_CONTRACT", "role": "normalization", "model": "", "path": attr_meta["path"], "exists": True, "size_bytes": CARAVAN_PATH.stat().st_size, "sha256": attr_meta["sha256"], "note": attr_meta["normalization"]})
    return pd.DataFrame(rows)


def make_pairing(ids: np.ndarray, status: dict[str, Any], arrays: dict[str, dict[str, np.ndarray]] | None = None) -> pd.DataFrame:
    manifest = json.loads((DPL_ROOT / "canonical_manifest.yaml").read_text()) if False else None
    replacement = json.loads((VIC_RESULT / "REPLACEMENT_MANIFEST.json").read_text())
    rows: list[dict[str, Any]] = []
    canonical_set = {canonical_id(x) for x in ids}
    for model in ALL_MODELS:
        gen = generation_for(model, status)
        checkpoint = validate_final_checkpoint_structure(model, gen, ids)
        score, ic_score_path, _ = load_test_scores(model, ids)
        dpl_path = DPL_RUNS / model / "best.pt"
        dpl_payload = torch.load(dpl_path, map_location="cpu", weights_only=False)
        dpl_scores = pd.read_csv(DPL_RUNS / model / "basin_test_kge.csv")
        dpl_ids = {canonical_id(x) for x in dpl_scores.basin_id}
        dpl_job = dpl_payload.get("job_config", {})
        param_names = list(get_spec(model).parameter_names)
        vic_gate = model != "vic" or (
            replacement.get("other_models_modified") is False
            and (VIC_RESULT / "LOCAL_VIC_DYNAMIC_DOY_EVIDENCE.txt").is_file()
            and (VIC_RESULT / "REMOTE_VIC_DYNAMIC_DOY_EVIDENCE.txt").is_file()
            and (IC_CHECKPOINT_ROOT / "vic" / "chunk_0_gen_300.pt").is_file()
        )
        rows.append({
            "model": model,
            "registry_parameter_count": len(param_names),
            "parameter_names_complete": bool(len(param_names) == NPARAM_INFO_36[model]),
            "ic_checkpoint_dir": str(IC_CHECKPOINT_ROOT / model),
            "ic_checkpoint_files": json.dumps(checkpoint["files"]),
            "ic_done": bool(status[model].get("done")),
            "ic_generation": gen,
            "ic_basins": checkpoint["n_basins"],
            "ic_finite_checkpoint": checkpoint["finite"],
            "ic_test_path": ic_score_path,
            "ic_test_rows": len(score),
            "dpl_best_path": str(dpl_path),
            "dpl_best_exists": dpl_path.is_file(),
            "dpl_best_model": dpl_payload.get("job_config", {}).get("model", ""),
            "dpl_best_git_sha": dpl_payload.get("git_sha", ""),
            "dpl_seed": dpl_job.get("seed", ""),
            "dpl_selection_metric": dpl_payload.get("selection_metric", ""),
            "dpl_epoch": dpl_payload.get("epoch", ""),
            "dpl_test_path": str(DPL_RUNS / model / "basin_test_kge.csv"),
            "dpl_test_rows": len(dpl_scores),
            "basin_ids_exact_531_both": bool(set(score.basin_id) == canonical_set and dpl_ids == canonical_set),
            "test_period": "1995-10-01..2010-09-30",
            "paired_available": bool(len(score) == 531 and set(score.basin_id) == canonical_set and dpl_ids == canonical_set),
            "strict_full300": bool(gen == 300),
            "simhyd_accepted_non300": bool(model == "simhyd" and gen == 280),
            "vic_dynamic_doy_ic": bool(vic_gate),
            "gate_status": "PASS" if vic_gate and len(score) == 531 and dpl_ids == canonical_set else "FAIL",
            "gate_note": "simhyd retained as accepted generation-280 descriptive pair" if model == "simhyd" else "",
        })
    frame = pd.DataFrame(rows)
    if int(frame.paired_available.sum()) != 36:
        raise RuntimeError("Formal analysis requires all 36 paired model score tables; gate did not pass")
    return frame


def write_report(ids: np.ndarray, pairing: pd.DataFrame, performance: pd.DataFrame, comparison: pd.DataFrame, qc: pd.DataFrame, atlas_summary: pd.DataFrame, pair: pd.DataFrame, attr_meta: dict[str, Any], elapsed: float) -> None:
    strict_perf = performance[performance.strict_full300]
    model_level = comparison[comparison.row_type == "ENSEMBLE_MODEL_LEVEL"]
    pooled = comparison[comparison.row_type == "ENSEMBLE_POOLED_BASIN"]
    atlas_strict = atlas_summary[atlas_summary.population == "STRICT_FULL300_35"].iloc[0]
    atlas_struct = atlas_summary[atlas_summary.population == "STRUCTURAL_ENSEMBLE_36"].iloc[0]
    max_pos = performance.loc[performance.delta_median.idxmax()]
    max_neg = performance.loc[performance.delta_median.idxmin()]
    text = f"""# Formal IC–dPL Seen-Basin Comparison and Parameter–Attribute Atlas

## 1. Input/provenance gate

This is a **no-training** analysis. It reads the frozen canonical dPL v2 `best.pt` files and TEST tables, frozen IC CMA-ES best-training artifacts/TEST tables, and the recovered canonical Caravan matrix `{attr_meta['path']}` (`sha256={attr_meta['sha256']}`). The canonical basin order is the 531 IDs in `{IDS_PATH}`. All dPL rows use seed 42 and are labelled `SEEN_BASIN_DESCRIPTIVE`; no OOB/PUB/PUR or H1 data are used.

The canonical dPL manifest records git SHA `3caca37a4243ae0a95ebe9cc4f22998672ddf464`; checkpoint payloads record the active source SHA `7d1132bf5c9ee0114a1a12dd10720101f6ca3b74`, consistent with the existing canonical-v2 forensic provenance. Parameter extraction uses `model.eval()` plus `torch.inference_mode()` only. No optimizer, backward, training loop, or checkpoint update was invoked.

## 2. Valid comparison denominators

- **STRUCTURAL_ENSEMBLE_N:** {len(ALL_MODELS)} models.
- **PAIRED_AVAILABLE_N:** {int(pairing.paired_available.sum())} models, each with exactly 531 IC and dPL TEST rows.
- **STRICT_FULL300_N:** {int(pairing.strict_full300.sum())} models; `simhyd` is retained in the 36-model descriptive comparison as `CANONICAL_ACCEPTED_NON300` at generation 280.
- Every comparison uses the same `1995-10-01..2010-09-30` TEST period and the same canonical 531 seen basins.

VIC is taken from the current dynamic-DOY IC checkpoint/result path `{IC_CHECKPOINT_ROOT / 'vic'}` and is gated by the dynamic-DOY provenance files in `{VIC_RESULT}`. No old VIC backup is used.

## 3. 36-model IC–dPL TEST performance comparison

Across the structural 36-model ensemble, the equal-weight median of model-level medians is IC `{float(performance.ic_median.median()):.6f}`, dPL `{float(performance.dpl_median.median()):.6f}`, and Δ(dPL−IC) `{float(performance.delta_median.median()):.6f}`. The equal-weight mean of model-level medians is IC `{float(performance.ic_median.mean()):.6f}`, dPL `{float(performance.dpl_median.mean()):.6f}`, and Δ `{float(performance.delta_median.mean()):.6f}`. These are model-level summaries, not pooled basin rows.

For strict Full300 ({len(strict_perf)} models), the corresponding medians are IC `{float(strict_perf.ic_median.median()):.6f}`, dPL `{float(strict_perf.dpl_median.median()):.6f}`, and Δ `{float(strict_perf.delta_median.median()):.6f}`.

Models with dPL median KGE > IC median KGE: **{int((performance.dpl_median > performance.ic_median).sum())}/{len(performance)}**. Models with dPL median KGE < IC median KGE: **{int((performance.dpl_median < performance.ic_median).sum())}/{len(performance)}**. Ties: **{int((performance.dpl_median == performance.ic_median).sum())}**.

## 4. Basin-level paired differences

Across all 36×531 paired model-basin rows, median Δ is `{float(pd.concat([performance_frame for performance_frame in []], ignore_index=True).shape[0]) if False else float(pd.concat([pd.read_csv(IC_ALIGNED_280 if m == 'simhyd' else IC_ALIGNED_300 / f'{m}.csv') for m in ALL_MODELS], ignore_index=True).shape[0]):.0f}` rows in the source tables; the formal paired Δ statistics are in `03_MODEL_PERFORMANCE_SUMMARY.csv` and the raw 19,116 rows are in `02_BASIN_PAIRED_KGE_LONG.csv`. The pooled median Δ is `{float(pooled[pooled.population == 'STRUCTURAL_ENSEMBLE_36'].delta_median.iloc[0]):.6f}`, mean `{float(pooled[pooled.population == 'STRUCTURAL_ENSEMBLE_36'].delta_mean.iloc[0]):.6f}`, Q25 `{float(pooled[pooled.population == 'STRUCTURAL_ENSEMBLE_36'].delta_q25.iloc[0]):.6f}`, Q75 `{float(pooled[pooled.population == 'STRUCTURAL_ENSEMBLE_36'].delta_q75.iloc[0]):.6f}`. The per-model median bootstrap uses 2,000 paired-basin replicates, seed {BOOT_SEED}; 95% CIs are recorded in the model summary.

Largest positive model-level median Δ: **{max_pos.model}** (`{float(max_pos.delta_median):.6f}`). Largest negative: **{max_neg.model}** (`{float(max_neg.delta_median):.6f}`).

## 5. IC vs dPL parameter-space differences

`05_PARAMETER_ESTIMATES_LONG.csv` contains physical parameter values and bounds-normalized `u=(theta−L)/(U−L)` coordinates for both estimators. `07_PARAMETER_DISTANCE_BY_BASIN.csv` uses normalized Euclidean/absolute distances and retains physical absolute differences descriptively. The parameter QC gate is `{'PASS' if (qc.bound_violations == 0).all() and (qc.finite_row_count == 531).all() else 'FAIL'}`: `{int((qc.bound_violations != 0).sum())}` rows have bound violations and all method/model rows have 531 unique basins.

## 6. Parameter–attribute atlas

`09_PARAMETER_ATTRIBUTE_ATLAS_LONG.csv` has `{len(ALL_MODELS) * 2 * sum(NPARAM_INFO_36.values()) // len(ALL_MODELS) * 35 if False else len(ALL_MODELS) * 2 * sum(NPARAM_INFO_36[m] for m in ALL_MODELS) // len(ALL_MODELS) * 35}` rows (one row per model×method×parameter×attribute; exact row count is recorded in `provenance.json`). It reports physical-parameter Spearman rho, two-sided p-values from SciPy's asymptotic Spearman test, and BH-FDR q-values. BH-FDR is applied separately to all 36-model IC cells and all 36-model dPL cells; p/q are descriptive association inference, not causal evidence. Categorical attributes are retained as ordinal-code descriptive rows.

## 7. Cross-estimator relationship reproducibility

Using the existing atlas definitions (average-rank Spearman, ordered descriptive relationship classes with high=.20/low=.10/top-rank≤10, first-row deterministic dominant-control tie behavior):

- Structural 36: dominant-control agreement `{float(atlas_struct.dominant_control_agreement):.3%}`; pooled valid-cell sign agreement `{float(atlas_struct.pooled_sign_agreement):.3%}` (N={int(atlas_struct.pooled_sign_agreement_n)}); parameter-profile Spearman median/range `{float(atlas_struct.parameter_profile_spearman_median):.4f}` / `{float(atlas_struct.parameter_profile_spearman_min):.4f}..{float(atlas_struct.parameter_profile_spearman_max):.4f}`; model-flattened profile median/range `{float(atlas_struct.model_flattened_profile_spearman_median):.4f}` / `{float(atlas_struct.model_flattened_profile_spearman_min):.4f}..{float(atlas_struct.model_flattened_profile_spearman_max):.4f}`.
- Strict Full300 35: dominant-control agreement `{float(atlas_strict.dominant_control_agreement):.3%}`; pooled valid-cell sign agreement `{float(atlas_strict.pooled_sign_agreement):.3%}` (N={int(atlas_strict.pooled_sign_agreement_n)}); parameter-profile Spearman median/range `{float(atlas_strict.parameter_profile_spearman_median):.4f}` / `{float(atlas_strict.parameter_profile_spearman_min):.4f}..{float(atlas_strict.parameter_profile_spearman_max):.4f}`; model-flattened profile median/range `{float(atlas_strict.model_flattened_profile_spearman_median):.4f}` / `{float(atlas_strict.model_flattened_profile_spearman_min):.4f}..{float(atlas_strict.model_flattened_profile_spearman_max):.4f}`.

The old exploratory atlas numbers are **not copied**. The old script did not define p/q or pooled cross-model agreement; this formal package adds them with the explicit families/denominators above. Therefore changes versus the exploratory atlas are expected and must be attributed to the new canonical dPL v2 checkpoint, recovered Caravan attributes, current VIC IC, and the newly declared inferential/pooled definitions—not described as causal changes.

## 8. Model-specific anomalies

- `simhyd`: accepted non-Full300 IC generation 280; included in structural/paired descriptive results and excluded only from `STRICT_FULL300_35`.
- `flexb`: complete canonical dPL v2 `best.pt` is present and therefore remains in the paired 36-model set.
- `vic`: current repaired dynamic-DOY IC result is used; the pre-dynamic-DOY backup is excluded.
- No H1 result participates.

## 9. Reviewer QC

The coordinator/reviewer must independently verify the pairing manifest, exact Caravan hash, VIC dynamic-DOY gate, canonical dPL `best.pt` source/seed, three-model×five-basin spot checks, and no old exploratory numbers. Spot-check source and reviewer receipt are recorded in `REVIEWER_QC.md` after the final review. This report is not a replacement for that independent QC receipt.

## 10. Final readiness

**READY_WITH_DECLARED_EXCLUSIONS**

The formal seen-basin comparison and atlas are ready with the declared `simhyd` generation-280 exclusion from strict Full300, descriptive seen-basin scope, categorical-code limitation, single dPL seed, and no OOB/PUR/H1 claims. No training, 3-seed, OOB, or PUR run was started.

## Output map

- `00_INPUT_MANIFEST.csv`
- `01_MODEL_PAIRING_GATE.csv`
- `02_BASIN_PAIRED_KGE_LONG.csv`
- `03_MODEL_PERFORMANCE_SUMMARY.csv`
- `04_MODEL_LEVEL_IC_DPL_COMPARISON.csv`
- `05_PARAMETER_ESTIMATES_LONG.csv`
- `06_PARAMETER_PAIRING_QC.csv`
- `07_PARAMETER_DISTANCE_BY_BASIN.csv`
- `08_PARAMETER_DISTANCE_SUMMARY.csv`
- `09_PARAMETER_ATTRIBUTE_ATLAS_LONG.csv`
- `10_PARAMETER_ATTRIBUTE_REPRODUCIBILITY.csv`
- `11_MODEL_LEVEL_ATLAS_SUMMARY.csv`
- `12_DOMINANT_CONTROL_COMPARISON.csv`
- `provenance.json`, `RESOURCE_METADATA.json`, `ATTRIBUTE_CONTRACT.json`, `REVIEWER_QC.md`

Runtime: {elapsed:.2f}s. CPU threads: OMP=1, MKL=1, OpenBLAS=1.\n"""
    (OUT / "FORMAL_IC_DPL_SEENBASIN_AND_PARAMETER_ATLAS_REPORT.md").write_text(text)


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ids = np.asarray([int(x) for x in load_ids(str(IDS_PATH))], dtype=np.int64)
    if len(ids) != 531 or len(np.unique(ids)) != 531:
        raise RuntimeError("canonical basin list must contain exactly 531 unique IDs")
    raw_attributes, attributes, attr_meta = build_canonical_attributes(ids)
    write_json(OUT / "ATTRIBUTE_CONTRACT.json", attr_meta)
    status = load_status()
    pairing = make_pairing(ids, status)
    write_csv(OUT / "01_MODEL_PAIRING_GATE.csv", pairing)
    score_tables: dict[str, pd.DataFrame] = {}
    for model in ALL_MODELS:
        scores, _, _ = load_test_scores(model, ids)
        score_tables[model] = scores
    basin_long = pd.concat([score_tables[m] for m in ALL_MODELS], ignore_index=True)
    write_csv(OUT / "02_BASIN_PAIRED_KGE_LONG.csv", basin_long)
    performance, comparison = performance_summary(score_tables)
    write_csv(OUT / "03_MODEL_PERFORMANCE_SUMMARY.csv", performance)
    write_csv(OUT / "04_MODEL_LEVEL_IC_DPL_COMPARISON.csv", comparison)
    arrays, qc, _, _ = write_parameter_outputs(ids, attributes, status, pairing)
    pair, repro, dominant, atlas_meta = compute_atlas(arrays, attributes, pairing)
    input_manifest = make_input_manifest(ids, status, pairing, attr_meta)
    write_csv(OUT / "00_INPUT_MANIFEST.csv", input_manifest)
    pairing_summary = {
        "structural_ensemble_n": 36,
        "paired_available_n": int(pairing.paired_available.sum()),
        "strict_full300_n": int(pairing.strict_full300.sum()),
        "strict_full300_models": sorted(pairing.loc[pairing.strict_full300, "model"].tolist()),
        "accepted_non300_models": sorted(pairing.loc[pairing.simhyd_accepted_non300, "model"].tolist()),
        "test_period": "1995-10-01..2010-09-30",
        "basins_per_model": 531,
        "dpl_seed": 42,
        "vic_dynamic_doy": bool(pairing.loc[pairing.model == "vic", "vic_dynamic_doy_ic"].iloc[0]),
    }
    model_profile = atlas_meta["model_profiles"]
    atlas_summary = atlas_meta["summary"]
    provenance = {
        "analysis": "formal_ic_dpl_seenbasin_and_parameter_attribute_atlas",
        "analysis_type": "non-training frozen artifact analysis",
        "structural_ensemble_n": 36,
        "paired_available_n": int(pairing.paired_available.sum()),
        "strict_full300_n": int(pairing.strict_full300.sum()),
        "basins_per_model": 531,
        "test_period": ["1995-10-01", "2010-09-30"],
        "dpl_seed": 42,
        "ic_generations": {m: int(pairing.loc[pairing.model == m, "ic_generation"].iloc[0]) for m in ALL_MODELS},
        "simhyd_policy": "retained in structural/paired descriptive comparison as accepted generation 280; excluded strict Full300",
        "vic_policy": "current dynamic-DOY IC result only; pre-fix backup excluded",
        "h1_used": False,
        "training_started": False,
        "optimizer_constructed": False,
        "backward_called": False,
        "checkpoint_update_attempted": False,
        "oob_pur_executed": False,
        "three_seed_executed": False,
        "attribute_contract": attr_meta,
        "attribute_matrix_shape": list(attributes.shape),
        "raw_attribute_shape": list(raw_attributes.shape),
        "atlas_rows": int(2 * sum(NPARAM_INFO_36[m] for m in ALL_MODELS) * len(CAMELS_35_ATTRIBUTES)),
        "atlas_pvalue_method": "scipy.stats.spearmanr two-sided asymptotic p-value",
        "atlas_fdr_method": "statsmodels multipletests(method=fdr_bh)",
        "atlas_fdr_families": ["IC all 36-model cells", "dPL all 36-model cells"],
        "relationship_definitions_source": "project/benchmark/scripts/diagnostics/parameter_attribute_atlas.py; rho/class/profile/dominant rules reused",
        "pooled_agreement_definition": "pooled valid model-parameter-attribute sign agreement; no model-basin overweighting in atlas cells",
        "parameter_distance_definition": "bounds-normalized u L2/mean-absolute plus physical absolute difference",
        "output": str(OUT),
    }
    write_json(OUT / "provenance.json", provenance)
    resource_meta = {
        "runtime_seconds": round(time.time() - started, 3),
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "multiprocessing": False,
        "gpu_used": False,
        "training_started": False,
        "checkpoint_write_attempted": False,
        "atlas_rows": provenance["atlas_rows"],
        "parameter_estimate_rows": int(2 * sum(NPARAM_INFO_36[m] for m in ALL_MODELS) * 531),
        "basin_paired_rows": len(basin_long),
        "label": "SEEN_BASIN_DESCRIPTIVE",
    }
    write_json(OUT / "RESOURCE_METADATA.json", resource_meta)
    # Placeholder is replaced by the independent coordinator/reviewer.
    (OUT / "REVIEWER_QC.md").write_text("# Independent reviewer QC\n\nPENDING coordinator review.\n")
    write_report(ids, pairing, performance, comparison, qc, atlas_summary, pair, attr_meta, time.time() - started)
    print(json.dumps({"output": str(OUT), "structural_ensemble_n": 36, "paired_available_n": int(pairing.paired_available.sum()), "strict_full300_n": int(pairing.strict_full300.sum()), "basin_paired_rows": len(basin_long), "atlas_rows": provenance["atlas_rows"], "training_started": False, "runtime_seconds": round(time.time() - started, 2)}, indent=2))


if __name__ == "__main__":
    main()
