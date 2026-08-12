#!/usr/bin/env python3
"""Diagnose a specific CAMELS basin's data for NaN/Inf/validity issues.

Usage:
    python3 scripts/diagnose_camels_basin.py --basin 12043000
    python3 scripts/diagnose_camels_basin.py --all
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

CAMELS_PATH = _PROJECT.parent.parent / "data" / "camels_dataset"
GAGE_ID_PATH = _PROJECT.parent.parent / "data" / "gage_id.npy"
OUTPUT_DIR = _PROJECT / "validation_results" / "static_router_camels_calibrated_pilot"


def flow_to_mmd(flow_ft3s, area_km2):
    return flow_ft3s * 2.446575 / max(area_km2, 1.0)


def check_basin(basin_id: int) -> dict:
    diag = {"basin_id": basin_id}

    gage_ids = np.load(GAGE_ID_PATH)
    match = np.where(gage_ids == basin_id)[0]
    if len(match) == 0:
        diag["error"] = f"Basin {basin_id} not found in gage_id.npy"
        return diag
    idx = int(match[0])
    diag["index"] = idx

    with open(CAMELS_PATH, "rb") as f:
        forcings, target, attributes = pickle.load(f)

    if idx >= forcings.shape[0]:
        diag["error"] = f"Index {idx} out of bounds for forcings ({forcings.shape[0]})"
        return diag

    n_timesteps = forcings.shape[1]
    diag["total_timesteps"] = n_timesteps
    diag["num_attr"] = int(attributes.shape[1])

    area = float(attributes[idx, 11])
    diag["area_km2"] = area
    diag["mean_precip"] = float(forcings[idx, :, 0].mean())
    diag["mean_temp"] = float(forcings[idx, :, 1].mean())
    diag["mean_pet"] = float(forcings[idx, :, 2].mean())

    # Forcing checks
    forc = forcings[idx]
    for i, name in enumerate(["prcp", "tmean", "pet"]):
        vals = forc[:, i]
        diag[f"forcing_{name}_nan_count"] = int(np.isnan(vals).sum())
        diag[f"forcing_{name}_inf_count"] = int(np.isinf(vals).sum())
        diag[f"forcing_{name}_min"] = float(np.nanmin(vals)) if not np.all(np.isnan(vals)) else float("nan")
        diag[f"forcing_{name}_max"] = float(np.nanmax(vals)) if not np.all(np.isnan(vals)) else float("nan")
        diag[f"forcing_{name}_all_zero"] = bool(np.all(vals == 0))
        n_valid = np.isfinite(vals).sum()
        diag[f"forcing_{name}_finite_ratio"] = float(n_valid / len(vals))

    # Target checks
    targ = target[idx, :, 0]
    diag["target_nan_count"] = int(np.isnan(targ).sum())
    diag["target_inf_count"] = int(np.isinf(targ).sum())
    diag["target_min"] = float(np.nanmin(targ)) if not np.all(np.isnan(targ)) else float("nan")
    diag["target_max"] = float(np.nanmax(targ)) if not np.all(np.isnan(targ)) else float("nan")
    diag["target_all_zero"] = bool(np.nanmax(np.abs(targ)) < 1e-10)

    # After warmup (365 days)
    targ_eval = targ[365:]
    valid_eval = np.isfinite(targ_eval)
    diag["n_valid_eval"] = int(valid_eval.sum())
    diag["n_total_eval"] = len(targ_eval)
    diag["valid_eval_ratio"] = float(valid_eval.sum() / max(len(targ_eval), 1))

    q_valid = targ_eval[valid_eval]
    diag["target_eval_mean"] = float(np.mean(q_valid)) if len(q_valid) > 0 else float("nan")
    diag["target_eval_std"] = float(np.std(q_valid)) if len(q_valid) > 0 else float("nan")
    diag["target_eval_cv"] = float(np.std(q_valid) / (np.mean(q_valid) + 1e-10)) if len(q_valid) > 1 else float("nan")

    # Convert to mm/d
    targ_mmd = flow_to_mmd(targ, area)
    targ_mmd_eval = targ_mmd[365:]
    mmd_valid = np.isfinite(targ_mmd_eval)
    diag["discharge_mmd_mean"] = float(np.mean(targ_mmd_eval[mmd_valid])) if mmd_valid.any() else float("nan")
    diag["discharge_mmd_std"] = float(np.std(targ_mmd_eval[mmd_valid])) if mmd_valid.any() else float("nan")

    # NSE/KGE feasibility
    if len(q_valid) >= 2:
        obs_mean = np.mean(q_valid)
        obs_std = np.std(q_valid)
        diag["obs_mean"] = float(obs_mean)
        diag["obs_std"] = float(obs_std)
        diag["obs_cv"] = float(obs_std / (obs_mean + 1e-10))
        diag["nse_denom_zero"] = bool(obs_std < 1e-12)
        diag["kge_std_zero"] = bool(obs_std < 1e-12)
    else:
        diag["obs_mean"] = float("nan")
        diag["obs_std"] = float("nan")
        diag["nse_denom_zero"] = True
        diag["kge_std_zero"] = True

    # Check if basin is workable
    issues = []
    if diag.get("forcing_prcp_all_zero", False):
        issues.append("prcp_all_zero")
    if diag.get("target_all_zero", False):
        issues.append("target_all_zero")
    if diag.get("nse_denom_zero", False):
        issues.append("nse_denom_zero")
    if diag.get("valid_eval_ratio", 0) < 0.1:
        issues.append(f"low_valid_ratio={diag['valid_eval_ratio']:.3f}")
    if diag.get("forcing_prcp_finite_ratio", 0) < 0.5:
        issues.append("prcp_low_finite_ratio")
    if diag.get("n_valid_eval", 0) < 10:
        issues.append("too_few_valid")
    diag["issues"] = "; ".join(issues) if issues else "none"
    diag["workable"] = len(issues) == 0 and diag.get("error") is None

    return diag


def main():
    ap = argparse.ArgumentParser(description="Diagnose CAMELS basin data")
    ap.add_argument("--basin", type=int, default=None, help="Basin ID to diagnose")
    ap.add_argument("--all", action="store_true", help="Check all basins")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    out = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    if args.basin:
        basins = [args.basin]
    elif args.all:
        gage_ids = np.load(GAGE_ID_PATH)
        basins = [int(gid) for gid in gage_ids]
    else:
        print("Specify --basin BASIN_ID or --all")
        sys.exit(1)

    all_rows = []
    for basin_id in basins:
        diag = check_basin(basin_id)
        all_rows.append(diag)
        if "error" in diag:
            print(f"\nBasin {basin_id}: ERROR — {diag['error']}")
        else:
            print(f"\nBasin {basin_id} (idx {diag['index']}, area={diag.get('area_km2','?')} km2):")
            print(f"  Forcing: prcp [{diag['forcing_prcp_min']:.1f}, {diag['forcing_prcp_max']:.1f}], "
                  f"nan={diag['forcing_prcp_nan_count']}, inf={diag['forcing_prcp_inf_count']}")
            print(f"  Target:  [{diag['target_min']:.1f}, {diag['target_max']:.1f}], "
                  f"nan={diag['target_nan_count']}, inf={diag['target_inf_count']}")
            print(f"  Eval (after 365d): {diag['n_valid_eval']}/{diag['n_total_eval']} valid "
                  f"({diag['valid_eval_ratio']:.1%})")
            if diag.get("discharge_mmd_mean") is not None and not np.isnan(diag.get("discharge_mmd_mean", float("nan"))):
                print(f"  Discharge mm/d: mean={diag['discharge_mmd_mean']:.4f}, std={diag['discharge_mmd_std']:.4f}")
            print(f"  Issues: {diag['issues']}")
            print(f"  Workable: {diag['workable']}")

    # Write CSV
    if all_rows:
        fields = [k for k in all_rows[0].keys()]
        csv_path = out / f"basin_diagnostics_{args.basin if args.basin else 'all'}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)

        # Write MD report
        md_lines = [
            f"# Basin Diagnostics: {args.basin if args.basin else 'all'}",
            "",
            f"Total basins checked: {len(all_rows)}",
            f"Workable: {sum(1 for r in all_rows if r.get('workable', False))}",
            f"Not workable: {sum(1 for r in all_rows if not r.get('workable', True))}",
            "",
        ]
        for diag in all_rows:
            if not diag.get("workable", False):
                md_lines.append(f"## Basin {diag['basin_id']}: NOT WORKABLE")
                if "error" in diag:
                    md_lines.append(f"  Error: {diag['error']}")
                md_lines.append(f"  Issues: {diag.get('issues', 'unknown')}")
                md_lines.append(f"  Valid eval: {diag.get('n_valid_eval', 0)}/{diag.get('n_total_eval', 0)}")
                md_lines.append(f"  Target std: {diag.get('obs_std', 'nan')}")
                md_lines.append("")

        md_path = out / f"basin_diagnostics_{args.basin if args.basin else 'all'}.md"
        md_path.write_text("\n".join(md_lines))

        print(f"\nOutput: {csv_path}")
        print(f"Output: {md_path}")


if __name__ == "__main__":
    main()
