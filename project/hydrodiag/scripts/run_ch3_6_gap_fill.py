#!/usr/bin/env python3
"""Evaluation-only Chapter 3.6 gap-fill numerical gate.

This script writes only under results/ch3_6_cross_process/07_gap_fill and does
not alter canonical replay assets.  It validates the existing full replay basin
ordering, then compares an aligned evaluation-only Full forward with Lite on the
same test-forcing slice and target mask.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
RESULTS = PROJECT / "results"
OUT = RESULTS / "ch3_6_cross_process" / "07_gap_fill"
DATA_ROOT = (PROJECT.parents[1] / "data").resolve()

sys.path.insert(0, str(PROJECT))
from manuscript.scripts.r4.common import load_bundle  # noqa: E402
from scripts.run_ch3_6_replay import (  # noqa: E402
    DPL_TO_STRUCTURE,
    FULL_CLASSES,
    LITE_CLASSES,
    PERIODS,
    canonical_ic_parameters,
    dpl_parameters,
    forcing_dict,
    parameter_dict,

)
from training.dpl.run_dpl_model import compute_kge_fp64  # noqa: E402

MODEL_LABELS = [
    "N",
    "D_E",
    "G_E",
    "D_R",
    "G_R",
    "XAJ_CONTROLLED_N_CN",
    "XAJ_D_E_CN",
    "XAJ_G_E_CN",
    "XAJ_D_R_CN",
    "XAJ_G_R_CN",
]


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.nanpercentile(values, q)) if values.size else float("nan")


def model_parameters(label: str, basin_ids: tuple[str, ...]):
    if label in LITE_CLASSES:
        matrix, names, _stored = canonical_ic_parameters(label, basin_ids)
        regime = "IC"
    else:
        matrix, names, _stored, _root = dpl_parameters(label, basin_ids)
        regime = "dPL"
    return regime, DPL_TO_STRUCTURE.get(label, label), matrix, names


def run_model(label: str, forcing: np.ndarray, observed: np.ndarray,
              valid_mask: np.ndarray, basin_ids: tuple[str, ...],
              device: torch.device, batch: int) -> tuple[list[dict], dict]:
    regime, structure, matrix, names = model_parameters(label, basin_ids)
    full_path = OUT.parent / "02_replay" / ("IC" if regime == "IC" else "dPL") / f"{label}_full_replay.npz"
    if not full_path.exists():
        raise FileNotFoundError(full_path)
    with np.load(full_path, allow_pickle=False) as archive:
        replay_ids = tuple(str(x).zfill(8) for x in archive["basin_ids"])
        if replay_ids != basin_ids:
            raise RuntimeError(f"{label}: full replay basin order differs from canonical bundle")

    split_forcing = forcing[:, PERIODS["test_forcing"], :]
    n_eval = PERIODS["test"].stop - PERIODS["test"].start
    q_lite = np.empty((len(basin_ids), n_eval), dtype=np.float64)
    q_full = np.empty_like(q_lite)
    warmup = PERIODS["test"].start - PERIODS["test_forcing"].start
    if warmup != 365 or n_eval != 5479:
        raise RuntimeError(f"unexpected test slice/warmup for {label}")
    lite_model = LITE_CLASSES[structure]().to(device=device).eval()
    full_model = FULL_CLASSES[structure]().to(device=device).eval()
    for left in range(0, len(basin_ids), batch):
        right = min(len(basin_ids), left + batch)
        fc = forcing_dict(split_forcing[left:right], device)
        params = parameter_dict(matrix[left:right], names, device)
        with torch.no_grad():
            q, _ = lite_model(forcings=fc, params=params, return_states=False)
        q_lite[left:right] = q[:, warmup:].detach().cpu().numpy().astype(np.float64)
        del q
        with torch.no_grad():
            q, _ = full_model(forcings=fc, params=params, return_states=False)
        q_full[left:right] = q[:, warmup:].detach().cpu().numpy().astype(np.float64)
        del q, fc, params


    rows: list[dict] = []
    for i, basin_id in enumerate(basin_ids):
        ql = q_lite[i]
        qf = q_full[i]
        obs = observed[i]
        mask = valid_mask[i] & np.isfinite(ql) & np.isfinite(qf) & (ql >= 0) & (qf >= 0)
        diff = qf[mask] - ql[mask]
        kge_lite = compute_kge_fp64(ql, obs)
        kge_full = compute_kge_fp64(qf, obs)
        rows.append({
            "regime": regime,
            "model": label,
            "structure": structure,
            "basin_id": basin_id,
            "n_valid_days": int(mask.sum()),
            "kge_lite": float(kge_lite) if kge_lite != -999.0 else float("nan"),
            "kge_full": float(kge_full) if kge_full != -999.0 else float("nan"),
            "delta_kge_full_minus_lite": float(kge_full - kge_lite) if kge_lite != -999.0 and kge_full != -999.0 else float("nan"),
            "rmse_q_full_minus_lite": float(np.sqrt(np.mean(diff * diff))) if diff.size else float("nan"),
            "mae_q_full_minus_lite": float(np.mean(np.abs(diff))) if diff.size else float("nan"),
            "max_abs_q_full_minus_lite": float(np.max(np.abs(diff))) if diff.size else float("nan"),
            "relative_rmse": float(np.sqrt(np.mean(diff * diff)) / (np.std(ql[mask]) + 1e-12)) if diff.size else float("nan"),
        })
    del lite_model, full_model, q_lite, q_full, matrix
    torch.cuda.empty_cache()

    delta = np.asarray([r["delta_kge_full_minus_lite"] for r in rows], dtype=np.float64)
    abs_delta = np.abs(delta)
    rmse = np.asarray([r["rmse_q_full_minus_lite"] for r in rows], dtype=np.float64)
    mae = np.asarray([r["mae_q_full_minus_lite"] for r in rows], dtype=np.float64)
    max_abs = np.asarray([r["max_abs_q_full_minus_lite"] for r in rows], dtype=np.float64)
    rel = np.asarray([r["relative_rmse"] for r in rows], dtype=np.float64)
    finite = np.isfinite(delta)
    summary = {
        "regime": regime,
        "model": label,
        "structure": structure,
        "basin_n": len(rows),
        "finite_kge_pairs": int(finite.sum()),
        "delta_kge_median": float(np.nanmedian(delta)),
        "delta_kge_p95_abs": percentile(abs_delta, 95),
        "delta_kge_p99_abs": percentile(abs_delta, 99),
        "delta_kge_max_abs": float(np.nanmax(abs_delta)),
        "delta_kge_positive_fraction": float(np.mean(delta > 0)) if finite.any() else float("nan"),
        "count_abs_delta_gt_1e-4": int(np.sum(abs_delta > 1e-4)),
        "count_abs_delta_gt_5e-4": int(np.sum(abs_delta > 5e-4)),
        "count_abs_delta_gt_1e-3": int(np.sum(abs_delta > 1e-3)),
        "count_abs_delta_gt_5e-3": int(np.sum(abs_delta > 5e-3)),
        "count_abs_delta_gt_1e-2": int(np.sum(abs_delta > 1e-2)),
        "rmse_q_median": float(np.nanmedian(rmse)),
        "rmse_q_p95": percentile(rmse, 95),
        "mae_q_median": float(np.nanmedian(mae)),
        "max_abs_q_median": float(np.nanmedian(max_abs)),
        "max_abs_q_p95": percentile(max_abs, 95),
        "relative_rmse_median": float(np.nanmedian(rel)),
        "relative_rmse_p95": percentile(rel, 95),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--models", nargs="*", default=MODEL_LABELS)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Lite-Full gate")
    if args.batch < 1 or args.batch > 32:
        raise ValueError("batch must be in [1, 32]")

    OUT.mkdir(parents=True, exist_ok=True)
    gate_dir = OUT / "01_lite_full_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    bundle = load_bundle(args.data_root)
    basin_ids = tuple(str(value).zfill(8) for value in bundle.basin_ids)
    forcing = np.asarray(bundle.forcing, dtype=np.float32)
    observed = np.asarray(bundle.target_mm_day[:, PERIODS["test"]], dtype=np.float64)
    valid_mask = np.asarray(bundle.valid_target_mask[:, PERIODS["test"]], dtype=bool)
    if len(basin_ids) != 531:
        raise RuntimeError(f"expected 531 canonical basins, got {len(basin_ids)}")

    all_rows: list[dict] = []
    summaries: list[dict] = []
    for label in args.models:
        rows, summary = run_model(label, forcing, observed, valid_mask, basin_ids, device, args.batch)
        all_rows.extend(rows)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    basin_fields = list(all_rows[0])
    with (gate_dir / "lite_full_basin.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=basin_fields)
        writer.writeheader()
        writer.writerows(all_rows)
    summary_fields = list(summaries[0])
    with (gate_dir / "lite_full_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    (gate_dir / "lite_full_gate_metadata.json").write_text(json.dumps({
        "n_models": len(summaries),
        "models": args.models,
        "n_basins": len(basin_ids),
        "forcing_slice": {"start": PERIODS["test_forcing"].start, "stop": PERIODS["test_forcing"].stop},
        "warmup_days": 365,
        "evaluation_slice": {"start": PERIODS["test"].start, "stop": PERIODS["test"].stop},
        "evaluation_mask": "bundle.valid_target_mask[:, 5478:10957] plus finite/nonnegative Lite and Full Q",
        "full_comparator": "FULL_CLASSES evaluation-only forward on the identical 5113:10957 forcing slice; existing 12418-day full replay is retained for diagnostics but has a different initial-history path",
        "compiled_execution": "production model step kernels are torch.compile(fullgraph=True); CUDA only; no CPU fallback",
        "device": torch.cuda.get_device_name(device),
        "dtype": "float32 forward; float64 summary metrics",
        "bootstrap": None,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
