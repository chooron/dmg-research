#!/usr/bin/env python3
"""Replay controlled Chapter 3.6 models and validate stored evaluation KGE.

The full-axis state export uses the production controlled model classes with
``return_states=True``. Evaluation replay uses the same Lite classes and test
warm-up slice as the training evaluator. This script never changes training
artifacts and requires a CUDA device for the requested full run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import median

import numpy as np
import torch

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
RESULTS = PROJECT / "results"
OUT = RESULTS / "ch3_6_cross_process"
IC_ROOT = RESULTS / "ic_phase0_controlled_531_v1"
DPL_ROOT = RESULTS / "dpl_controlled_531_v1"
DPL_N_ROOT = OUT / "dpl_controlled_n_531_v1"
DATA_ROOT = (PROJECT.parents[1] / "data").resolve()

sys.path.insert(0, str(PROJECT))
from ablation.ic_core.parameter_adapter import get_parameter_spec  # noqa: E402
from manuscript.scripts.r4.common import load_bundle, bundle_config  # noqa: E402
from models.controlled_composed import (  # noqa: E402
    XAJControlledNWithCemaNeige,
    XAJControlledNWithCemaNeigeLite,
    XAJDEWithCemaNeige,
    XAJDEWithCemaNeigeLite,
    XAJDRWithCemaNeige,
    XAJDRWithCemaNeigeLite,
    XAJGEWithCemaNeige,
    XAJGEWithCemaNeigeLite,
    XAJGRWithCemaNeige,
    XAJGRWithCemaNeigeLite,
)
from training.dpl.run_dpl_model import compute_kge_fp64  # noqa: E402

FULL_CLASSES = {
    "N": XAJControlledNWithCemaNeige,
    "D_E": XAJDEWithCemaNeige,
    "G_E": XAJGEWithCemaNeige,
    "D_R": XAJDRWithCemaNeige,
    "G_R": XAJGRWithCemaNeige,
}
LITE_CLASSES = {
    "N": XAJControlledNWithCemaNeigeLite,
    "D_E": XAJDEWithCemaNeigeLite,
    "G_E": XAJGEWithCemaNeigeLite,
    "D_R": XAJDRWithCemaNeigeLite,
    "G_R": XAJGRWithCemaNeigeLite,
}
DPL_TO_STRUCTURE = {
    "XAJ_CONTROLLED_N_CN": "N",
    "XAJ_D_E_CN": "D_E",
    "XAJ_G_E_CN": "G_E",
    "XAJ_D_R_CN": "D_R",
    "XAJ_G_R_CN": "G_R",
}

PERIODS = {
    "warmup": slice(0, 365),
    "train": slice(365, 5478),
    "test": slice(5478, 10957),
    "test_forcing": slice(5113, 10957),
}


def json_load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def canonical_ic_parameters(model: str, basin_ids: tuple[str, ...]) -> tuple[np.ndarray, list[str], dict[str, float]]:
    root = IC_ROOT / model
    records = []
    for path in sorted((root / "raw" / model.lower()).glob("*.json")):
        records.append(json_load(path))
    if len(records) != 5310:
        raise RuntimeError(f"IC {model}: expected 5310 raw records, got {len(records)}")
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(record["basin_id"]).zfill(8), []).append(record)
    if len(basin_ids) == 531:
        if set(groups) != set(basin_ids) or len(groups) != len(basin_ids):
            raise RuntimeError(f"IC {model}: basin IDs do not exactly match canonical bundle")
    elif not set(basin_ids).issubset(groups):
        raise RuntimeError(f"IC {model}: basin IDs do not match canonical bundle")
    names = list(records[0]["parameter_names"])
    if any(list(record.get("parameter_names", [])) != names or len(record.get("parameters", [])) != len(names) for record in records):
        raise RuntimeError(f"IC {model}: parameter schema differs across raw records")
    selected = {
        basin: sorted(group, key=lambda r: (-float(r["best_train_objective"]), int(r["start"]))) [0]
        for basin, group in groups.items()
    }
    matrix = np.asarray([selected[basin]["parameters"] for basin in basin_ids], dtype=np.float32)
    stored = {basin: float(selected[basin]["test_metrics"]["kge"]) for basin in basin_ids}
    return matrix, names, stored


def dpl_parameters(model: str, basin_ids: tuple[str, ...]) -> tuple[np.ndarray, list[str], dict[str, float], Path]:
    if model == "XAJ_CONTROLLED_N_CN":
        root = DPL_N_ROOT / model / "seed_42"
    else:
        root = DPL_ROOT / model / "seed_42"
    required = [root / "COMPLETE", root / "config.json", root / "basin_final_summary.csv", root / "best_parameters_physical.npz", root / "best_parameters_normalized.npz"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"dPL {model} missing required artifacts: {missing}")
    config = json_load(root / "config.json")
    names = list(config["parameter_names"])
    with np.load(root / "best_parameters_physical.npz", allow_pickle=False) as archive:
        matrix = np.asarray(archive["params"], dtype=np.float32)
    with np.load(root / "best_parameters_normalized.npz", allow_pickle=False) as archive:
        normalized = np.asarray(archive["params"], dtype=np.float32)
    if normalized.shape != matrix.shape:
        raise RuntimeError(f"dPL {model}: normalized/physical parameter shapes differ: {normalized.shape} vs {matrix.shape}")
    stored_rows = {}
    with (root / "basin_final_summary.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stored_rows[str(row["basin_id"]).zfill(8)] = float(row["val_kge"])
    if matrix.shape != (len(stored_rows), len(names)):
        raise RuntimeError(f"dPL {model}: parameter shape {matrix.shape} does not match summary {len(stored_rows)}x{len(names)}")
    summary_ids = list(stored_rows)
    if len(basin_ids) == 531:
        if set(summary_ids) != set(basin_ids) or len(summary_ids) != len(basin_ids):
            raise RuntimeError(f"dPL {model}: basin IDs do not exactly match canonical bundle")
        if summary_ids != list(basin_ids):
            raise RuntimeError(f"dPL {model}: summary basin order differs from canonical bundle; no silent remap")
    else:
        if not set(basin_ids).issubset(stored_rows):
            raise RuntimeError(f"dPL {model}: basin IDs do not match canonical bundle")
        if summary_ids[: len(basin_ids)] != list(basin_ids):
            raise RuntimeError(f"dPL {model}: summary basin order differs from canonical bundle; no silent remap")
    matrix = matrix[: len(basin_ids)]
    return matrix, names, stored_rows, root


def forcing_dict(forcing: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    tensor = torch.as_tensor(forcing, device=device, dtype=torch.float32)
    return {"precip": tensor[:, :, 0], "temp": tensor[:, :, 1], "pet": tensor[:, :, 2]}


def parameter_dict(matrix: np.ndarray, names: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: torch.as_tensor(matrix[:, i], device=device, dtype=torch.float32) for i, name in enumerate(names)}


def get_trace(aux: dict, *keys: str) -> torch.Tensor | None:
    for key in keys:
        if key in aux:
            return aux[key]
    return None


def extract_states(aux: dict, structure: str) -> dict[str, torch.Tensor]:
    """Select only the existing controlled diagnostics needed downstream."""
    selected: dict[str, torch.Tensor] = {}
    candidates = {
        "effective_precip": ("effective_precip",),
        "sca": ("cn_sca",),
        "rain": ("cn_rain",),
        "melt": ("cn_melt",),
        "evap": ("xaj_evap_total", "xaj_evap"),
        "wu": ("xaj_wu",),
        "wl": ("xaj_wl",),
        "wd": ("xaj_wd",),
        "s": ("xaj_s_next", "xaj_s"),
        "fr": ("xaj_fr",),
        "q_surface": ("xaj_rs_routed",),
        "q_surface_instant": ("xaj_rs_instant",),
        "qi": ("xaj_qi",),
        "qg": ("xaj_qg",),
        "q_subsurface": ("xaj_q_ss",),
        "z": ("xaj_z", "xaj_response_storage"),
        "z_available": ("xaj_z_available",),
        "response_input": ("xaj_r_ss",),
    }
    for name, keys in candidates.items():
        value = get_trace(aux, *keys)
        if value is not None and value.ndim == 2:
            selected[name] = value
    if "q_subsurface" not in selected and "qi" in selected and "qg" in selected:
        selected["q_subsurface"] = selected["qi"] + selected["qg"]
    return selected


def compute_split_kge(
    matrix: np.ndarray,
    names: list[str],
    structure: str,
    forcing: np.ndarray,
    observed: np.ndarray,
    device: torch.device,
    batch: int,
) -> np.ndarray:
    torch.cuda.reset_peak_memory_stats(device)
    model = LITE_CLASSES[structure]().to(device=device).eval()
    warmup = 365
    q_test = np.empty((forcing.shape[0], forcing.shape[1] - warmup), dtype=np.float64)
    for left in range(0, forcing.shape[0], batch):
        right = min(forcing.shape[0], left + batch)
        fc = forcing_dict(forcing[left:right], device)
        params = parameter_dict(matrix[left:right], names, device)
        with torch.no_grad():
            q, _ = model(forcings=fc, params=params, return_states=False)
        q_test[left:right] = q[:, warmup:].detach().cpu().numpy().astype(np.float64)
    result = np.asarray(
        [compute_kge_fp64(q_test[i], observed[i]) for i in range(len(q_test))], dtype=np.float64
    )
    result[result == -999.0] = np.nan
    del model, q_test
    torch.cuda.empty_cache()
    return result


def full_replay(
    matrix: np.ndarray,
    names: list[str],
    structure: str,
    forcing: np.ndarray,
    device: torch.device,
    batch: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    torch.cuda.reset_peak_memory_stats(device)
    model = FULL_CLASSES[structure]().to(device=device).eval()
    n, time = forcing.shape[:2]
    q_full = np.empty((n, time), dtype=np.float32)
    states_full: dict[str, np.ndarray] = {}
    peak = 0
    for left in range(0, n, batch):
        right = min(n, left + batch)
        fc = forcing_dict(forcing[left:right], device)
        params = parameter_dict(matrix[left:right], names, device)
        with torch.no_grad():
            q, aux = model(forcings=fc, params=params, return_states=True)
        q_np = q.detach().cpu().numpy().astype(np.float32)
        q_full[left:right] = q_np
        selected = extract_states(aux, structure)
        if not states_full:
            states_full = {key: np.empty((n, time), dtype=np.float32) for key in selected}
        if set(selected) != set(states_full):
            raise RuntimeError(f"{structure}: diagnostic key set changed across batches: {set(selected)} vs {set(states_full)}")
        for key, value in selected.items():
            states_full[key][left:right] = value.detach().cpu().numpy().astype(np.float32)
        if torch.cuda.is_available():
            peak = max(peak, int(torch.cuda.max_memory_allocated(device) / (1024 * 1024)))
        del q, aux, selected, fc, params
    del model
    torch.cuda.empty_cache()
    return q_full, states_full, peak


def write_replay(
    regime: str,
    model_label: str,
    basin_ids: tuple[str, ...],
    dates: np.ndarray,
    q_full: np.ndarray,
    states: dict[str, np.ndarray],
    stored_kge: np.ndarray,
    split_kge: np.ndarray,
    eval_valid_mask: np.ndarray,
    root: Path,
    peak_mb: int,
) -> dict:
    out_dir = OUT / "02_replay" / ("IC" if regime == "IC" else "dPL")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_label}_full_replay.npz"
    arrays = {
        "basin_ids": np.asarray(basin_ids),
        "dates": np.asarray(dates, dtype="datetime64[D]"),
        "qsim": q_full,
        "stored_eval_kge": stored_kge.astype(np.float64),
        "split_replay_eval_kge": split_kge.astype(np.float64),
        "evaluation_valid_mask": np.asarray(eval_valid_mask, dtype=bool),
    }
    arrays.update(states)
    np.savez_compressed(path, **arrays)
    metadata = {
        "regime": regime,
        "model": model_label,
        "structure": DPL_TO_STRUCTURE.get(model_label, model_label),
        "parameter_root": str(root),
        "n_basins": len(basin_ids),
        "n_days": len(dates),
        "periods": {key: {"start": value.start, "stop": value.stop} for key, value in PERIODS.items()},
        "stored_eval_kge_source": "IC raw selected by max best_train_objective then min start; dPL basin_final_summary.csv",
        "split_replay_definition": "test_forcing indices 5113:10957, discard first 365 warmup days",
        "full_replay_definition": "full 12418-day axis from zero initial states; process metrics use test slice 5478:10957",
        "state_keys": sorted(states),
        "cuda_peak_allocated_mb": peak_mb,
        "qsim_finite_fraction": float(np.isfinite(q_full).mean()),
    }
    (path.with_suffix(".json")).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-basins", type=int)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()
    if args.max_basins is not None and not 1 <= args.max_basins <= 531:
        raise ValueError("--max-basins must be between 1 and 531")
    if args.batch > 32:
        raise ValueError("--batch above 32 is disabled by the low-memory replay policy")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Chapter 3.6 replay; refusing CPU fallback")
    if args.batch < 1:
        raise ValueError("--batch must be positive")
    device = torch.device("cuda")
    bundle = load_bundle(args.data_root)
    all_ids = tuple(str(value).zfill(8) for value in bundle.basin_ids)
    n = args.max_basins or len(all_ids)
    if n != len(all_ids):
        # Smoke mode is explicitly non-canonical and writes under a separate suffix.
        basin_ids = all_ids[:n]
    else:
        basin_ids = all_ids
    forcing = np.asarray(bundle.forcing[:n], dtype=np.float32)
    observed = np.asarray(bundle.target_mm_day[:n, PERIODS["test"]], dtype=np.float64)
    evaluation_valid_mask = np.asarray(bundle.valid_target_mask[:n, PERIODS["test"]], dtype=bool)
    dates = np.asarray(bundle.dates)
    labels = args.models or ["N", "D_E", "G_E", "D_R", "G_R", "XAJ_CONTROLLED_N_CN", "XAJ_D_E_CN", "XAJ_G_E_CN", "XAJ_D_R_CN", "XAJ_G_R_CN"]
    if args.max_basins:
        replay_out = OUT / "02_replay" / "smoke"
        replay_out.mkdir(parents=True, exist_ok=True)
    validation_rows = []
    for label in labels:
        if label in FULL_CLASSES:
            regime = "IC"
            structure = label
            matrix, names, stored_map = canonical_ic_parameters(label, basin_ids)
            parameter_root = IC_ROOT / label
        elif label in DPL_TO_STRUCTURE:
            regime = "dPL"
            structure = DPL_TO_STRUCTURE[label]
            matrix, names, stored_map, parameter_root = dpl_parameters(label, basin_ids)
        else:
            raise ValueError(f"unknown model label {label}")
        stored = np.asarray([stored_map[basin] for basin in basin_ids], dtype=np.float64)
        stored[stored == -999.0] = np.nan
        split_forcing = forcing[:, PERIODS["test_forcing"], :]
        split_kge = compute_split_kge(matrix, names, structure, split_forcing, observed, device, args.batch)
        q_full, states, peak_mb = full_replay(matrix, names, structure, forcing, device, args.batch)
        if args.max_basins:
            out_dir = replay_out / ("IC" if regime == "IC" else "dPL")
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{label}_full_replay_smoke.npz"
            np.savez_compressed(path, basin_ids=np.asarray(basin_ids), dates=dates, qsim=q_full, **states)
            metadata = {"smoke": True, "model": label, "n_basins": n, "state_keys": sorted(states), "peak_mb": peak_mb}
            path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        else:
            metadata = write_replay(regime, label, basin_ids, dates, q_full, states, stored, split_kge, evaluation_valid_mask, parameter_root, peak_mb)
        mismatch = split_kge - stored
        row = {
            "regime": regime,
            "model": label,
            "structure": structure,
            "basin_n": n,
            "stored_eval_kge_median": float(np.nanmedian(stored)),
            "split_replay_eval_kge_median": float(np.nanmedian(split_kge)),
            "mismatch_median": float(np.nanmedian(mismatch)),
            "mismatch_abs_median": float(np.nanmedian(np.abs(mismatch))),
            "mismatch_abs_p95": float(np.nanpercentile(np.abs(mismatch), 95)),
            "mismatch_abs_max": float(np.nanmax(np.abs(mismatch))),
            "mismatch_abs_gt_1e-5": int((np.abs(mismatch) > 1e-5).sum()),
            "mismatch_abs_gt_1e-4": int((np.abs(mismatch) > 1e-4).sum()),
            "qsim_nonfinite": int((~np.isfinite(q_full)).sum()),
            "cuda_peak_allocated_mb": peak_mb,
            "artifact": str((OUT / "02_replay" / ("IC" if regime == "IC" else "dPL") / f"{label}_full_replay.npz").relative_to(PROJECT)) if not args.max_basins else str(path.relative_to(PROJECT)),
        }
        validation_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del q_full, states, split_kge, matrix
        torch.cuda.empty_cache()
    if not args.max_basins:
        fields = list(validation_rows[0]) if validation_rows else []
        with (OUT / "02_replay" / "replay_validation.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(validation_rows)
        (OUT / "02_replay" / "replay_validation_note.md").write_text(
            "# Replay validation\n\n"
            "Stored KGE is compared with an evaluation-style replay using the same test-forcing warm-up. "
            "No frozen numerical tolerance was found, so mismatch columns are reported descriptively and are not silently passed/failed.\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
