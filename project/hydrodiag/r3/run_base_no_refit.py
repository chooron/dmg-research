#!/usr/bin/env python3
"""Phase 3: raw structural knockout -- ``Base-no-refit``.

Runs the XAJ Base structure with the 15 shared XAJ parameters taken
verbatim from ``theta*`` (the generating XAJ-CN truth), with the CemaNeige
snow module fully removed and no re-calibration.  Base receives the forcing
exactly as its normal definition prescribes (P, T, PET); no ``rainonly`` or
other intermediate control is introduced.

Outputs per basin (results/r3_base_no_refit_v1/):

- discharge metrics vs Q* (KGE, NSE, PBIAS) for train and test periods;
- spring timing / snow-season diagnostics reused from the repository's R1
  water-year signature definitions (CT center-of-timing and AMJJ seasonal
  volume errors; ``r3/common.water_year_errors``);
- ``frac_snow`` and descriptive relationships (no hard thresholds).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[0]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from r3.common import (  # noqa: E402
    COMMON_XAJ,
    DEFAULT_DATA_ROOT,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_RESULTS_ROOT,
    frac_snow_series,
    git_commit,
    nse,
    pbias,
    period_indices,
    standard_kge,
    water_year_errors,
    write_json,
)
from r3.recorded_forward import build_forcing_dict, recorded_base_forward  # noqa: E402


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--truth-run-id", default="r3_synthetic_truth_v1")
    parser.add_argument("--run-id", default="r3_base_no_refit_v1")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-basins", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    truth_dir = args.results_root / args.truth_run_id
    output_dir = args.results_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    from models import XAJLite

    bundle, _config = load_bundle(args.project_root, args.data_root)
    theta = np.load(truth_dir / "theta_star.npz")
    theta_star = theta["parameters"]
    names = [str(n) for n in theta["parameter_names"]]
    shared_indices = [names.index(n) for n in COMMON_XAJ]
    shared = theta_star[:, shared_indices]  # [531, 15] in XAJ spec order

    # verify no hidden optimization: parameters are the verbatim shared slice
    # of the stored generating truth
    np.savez_compressed(
        output_dir / "base_no_refit_parameters.npz",
        parameters=shared,
        parameter_names=np.asarray(COMMON_XAJ),
        basin_ids=np.asarray(bundle.basin_ids),
        provenance="theta_star[npz][:, shared XAJ columns]; no calibration",
    )

    dates = pd.to_datetime(bundle.dates)
    pi = period_indices(bundle)
    forcing = bundle.forcing.astype(np.float32)
    n_basins, n_time = forcing.shape[:2]
    model = XAJLite().to(device).eval()
    dtype = torch.float32
    params_t = torch.from_numpy(shared).to(device=device, dtype=dtype)

    # full-axis simulation (needed for the exact warm-up convention)
    q_full = np.empty((n_basins, n_time), dtype=np.float32)
    for left in range(0, n_basins, args.batch_basins):
        right = min(n_basins, left + args.batch_basins)
        fc = build_forcing_dict(forcing[left:right], device, dtype)
        p = {name: params_t[left:right, i] for i, name in enumerate(COMMON_XAJ)}
        qsim, _stores, _fs = recorded_base_forward(model, fc, p, device, dtype)
        q_full[left:right] = qsim.detach().cpu().numpy()
        print(f"chunk {left}:{right} done", flush=True)

    q_star = np.load(truth_dir / "q_star.npz")["target_mm_day"]
    q_star = np.asarray(q_star, dtype=np.float64)

    rows = []
    for b, basin in enumerate(bundle.basin_ids):
        row = {"basin_id": basin}
        for period, (si, ei) in (("train", pi["train"]), ("test", pi["test"])):
            sim = q_full[b, si : ei + 1]
            obs = q_star[b, si : ei + 1]
            row[f"kge_{period}"] = standard_kge(sim, obs)
            row[f"nse_{period}"] = nse(sim, obs)
            row[f"pbias_{period}"] = pbias(sim, obs)
            sig = water_year_errors(dates[si : ei + 1], sim, obs)
            row[f"ct_error_abs_{period}"] = sig["ct_error_absolute"]
            row[f"amjj_error_abs_{period}"] = sig["amjj_error_absolute"]
        rows.append(row)
    df = pd.DataFrame(rows)
    snow = frac_snow_series(bundle)
    df = df.merge(snow, on="basin_id", how="left")
    df.to_csv(output_dir / "base_no_refit_basin_metrics.csv", index=False)

    summary = {
        "protocol": "r3_base_no_refit_v1",
        "created_at": _utcnow(),
        "code": git_commit(args.project_root),
        "structure": "XAJ (Base), snow module fully removed",
        "parameters": "shared XAJ columns of theta* verbatim; no optimization",
        "no_refit_guard": "parameters loaded directly from theta_star.npz; script contains no optimizer call",
        "forcing_definition": "Base normal forcing input (P, T, PET); no rainonly control",
        "metrics": "KGE/NSE/PBIAS vs Q*; CT/AMJJ water-year signature errors (R1 definitions)",
        "n_basins": n_basins,
        "q_finite": bool(np.isfinite(q_full).all()),
        "q_nonnegative": bool((q_full >= 0).all()),
        "summary_train": {
            "median_kge": float(df["kge_train"].median()),
            "median_nse": float(df["nse_train"].median()),
            "median_pbias": float(df["pbias_train"].median()),
            "median_ct_error": float(df["ct_error_abs_train"].median()),
            "median_amjj_error": float(df["amjj_error_abs_train"].median()),
        },
        "summary_test": {
            "median_kge": float(df["kge_test"].median()),
            "median_nse": float(df["nse_test"].median()),
            "median_pbias": float(df["pbias_test"].median()),
            "median_ct_error": float(df["ct_error_abs_test"].median()),
            "median_amjj_error": float(df["amjj_error_abs_test"].median()),
        },
        "snow_relationship": {
            "spearman_kge_train_vs_frac_snow": float(
                df[["kge_train", "frac_snow"]]
                .dropna()
                .corr(method="spearman")
                .iloc[0, 1]
            ),
            "spearman_ct_error_train_vs_frac_snow": float(
                df[["ct_error_abs_train", "frac_snow"]]
                .dropna()
                .corr(method="spearman")
                .iloc[0, 1]
            ),
            "note": "descriptive only; frac_snow is an environmental diagnostic axis",
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(f"COMPLETE Base-no-refit -> {output_dir}", flush=True)


def load_bundle(project_root: Path, data_root: Path):
    from r3.common import load_bundle as _load

    return _load(project_root, data_root)


if __name__ == "__main__":
    main()
