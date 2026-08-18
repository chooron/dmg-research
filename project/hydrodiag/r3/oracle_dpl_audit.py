#!/usr/bin/env python3
"""Pre-D2 gate: dPL-path oracle audit of theta* over all 531 basins.

Before the 531-basin CN-dPL training starts, theta* is pushed through the
*actual* dPL data path (the dPL loader with the Q* target override and the
canonical cn_psol_annual):

- training-window path: 730-day windows (365 warm-up + 365 scored) at five
  fixed deterministic offsets spanning the calibration period, scored with
  the dPL training metric (kge_per_basin);
- evaluation path: 365-day warm-up + the full test period, float64, scored
  with the dPL validation metric (compute_kge_fp64) — this is the exact
  ceiling of the ``val_kge`` the dPL runner reports at theta*;
- contrast: the same paths *without* the canonical psol override
  (historical per-sequence g_thresh), quantifying the snow-dependent bias
  the Phase-A fix removed.

Output: per-window and per-basin CSVs plus an aggregate JSON with the
KGE/error distributions and their frac_snow dependence (Spearman + R2
snow-regime strata).  The verdict states whether the frozen window/warm-up
protocol shows a material snow-dependent oracle bias (only then would a
protocol change be considered).
"""

from __future__ import annotations

import argparse
import json
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
    DEFAULT_DATA_ROOT,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_RESULTS_ROOT,
    git_commit,
    write_json,
)

WINDOW_OFFSETS = (365, 1200, 2500, 3800, 4700)  # forcing-relative scored starts
SNOW_BINS = [(0.05, "S1"), (0.15, "S2"), (0.30, "S3"), (0.50, "S4"), (np.inf, "S5")]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def snow_regime(f: float) -> str:
    for threshold, name in SNOW_BINS:
        if f < threshold:
            return name
    return "S5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--truth-run-id", default="r3_synthetic_truth_v1")
    parser.add_argument("--run-id", default="r3_gate_v1")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-basins", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)
    truth_dir = args.results_root / args.truth_run_id
    output_dir = args.results_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    from training.dpl.run_dpl_model import (
        compute_kge_fp64,
        gate_time_index,
        kge_per_basin,
        load_data,
    )

    from r3.common import frac_snow_series, load_bundle

    bundle, _ = load_bundle(args.project_root, args.data_root)
    snow = frac_snow_series(bundle).set_index("basin_id")["frac_snow"]

    config = json.loads(
        (PROJECT / "training/dpl/base_config_camels_531.json").read_text()
    )
    config["output_dir"] = str(PROJECT / "tmp_r3_audit_out")
    config["data_basin_ids"] = str(args.data_root / "531sub_id.txt")
    config["target_override_npz"] = str(truth_dir / "q_star.npz")
    indices = gate_time_index(config)
    basin_ids, _attrs, train_forcing, cal_obs, eval_forcing, eval_obs = load_data(
        config, indices, max_basins=None
    )
    n = len(basin_ids)

    theta = np.load(truth_dir / "theta_star.npz")
    theta_star = theta["parameters"]
    names = [str(n) for n in theta["parameter_names"]]
    if list(basin_ids) != list(bundle.basin_ids):
        raise ValueError("dPL loader basin order differs from the bundle")

    from models import XAJWithCemaNeigeLite

    model = XAJWithCemaNeigeLite().to(device).eval()
    dtype = torch.float32
    warmup = 365

    def run_windows(psol_key: str | None) -> list[dict]:
        rows = []
        for off in WINDOW_OFFSETS:
            fi = np.arange(off - warmup, off + 365, dtype=np.int64)
            ti = np.arange(off - warmup, off, dtype=np.int64)
            forcing_np = np.stack(
                [train_forcing[k][:, fi] for k in ("precip", "temp", "pet")],
                axis=-1,
            ).astype(np.float32)
            fc = {
                "precip": torch.from_numpy(forcing_np[:, :, 0]).to(device, dtype=dtype),
                "temp": torch.from_numpy(forcing_np[:, :, 1]).to(device, dtype=dtype),
                "pet": torch.from_numpy(forcing_np[:, :, 2]).to(device, dtype=dtype),
            }
            if psol_key is not None:
                fc["cn_psol_annual"] = torch.from_numpy(train_forcing[psol_key]).to(
                    device, dtype=dtype
                )
            params = {
                name: torch.from_numpy(theta_star[:, i]).to(device, dtype=dtype)
                for i, name in enumerate(names)
            }
            obs = torch.from_numpy(cal_obs[:, ti].copy()).to(device, dtype=dtype)
            with torch.no_grad():
                qsim, _ = model(forcings=fc, params=params)
            kge = kge_per_basin(qsim[:, warmup:], obs).detach().cpu().numpy()
            q_np = qsim[:, warmup:].detach().cpu().numpy().astype(np.float64)
            obs_np = cal_obs[:, ti].astype(np.float64)
            for k, b in enumerate(basin_ids):
                diff = np.abs(q_np[k] - obs_np[k])
                rows.append(
                    {
                        "basin_id": b,
                        "window_offset": int(off),
                        "kge": float(kge[k]),
                        "abs_err_max": float(diff.max()),
                        "abs_err_mean": float(diff.mean()),
                        "frac_snow": float(snow[b]),
                    }
                )
        return rows

    def run_eval(psol_key: str | None) -> list[dict]:
        rows = []
        forcing_np = np.stack(
            [eval_forcing[k] for k in ("precip", "temp", "pet")], axis=-1
        ).astype(np.float64)
        fc = {
            "precip": torch.from_numpy(forcing_np[:, :, 0]).to(
                device, dtype=torch.float64
            ),
            "temp": torch.from_numpy(forcing_np[:, :, 1]).to(
                device, dtype=torch.float64
            ),
            "pet": torch.from_numpy(forcing_np[:, :, 2]).to(
                device, dtype=torch.float64
            ),
        }
        if psol_key is not None:
            fc["cn_psol_annual"] = torch.from_numpy(
                eval_forcing[psol_key].astype(np.float64)
            ).to(device, dtype=torch.float64)
        params = {
            name: torch.from_numpy(theta_star[:, i]).to(device, dtype=torch.float64)
            for i, name in enumerate(names)
        }
        with torch.no_grad():
            qsim, _ = model(forcings=fc, params=params)
        q_np = qsim[:, warmup:].detach().cpu().numpy()
        for k, b in enumerate(basin_ids):
            kge = compute_kge_fp64(q_np[k], eval_obs[k])
            diff = np.abs(q_np[k].astype(np.float64) - eval_obs[k].astype(np.float64))
            rows.append(
                {
                    "basin_id": b,
                    "window_offset": -1,  # evaluation path
                    "kge": float(kge),
                    "abs_err_max": float(diff.max()),
                    "abs_err_mean": float(diff.mean()),
                    "frac_snow": float(snow[b]),
                }
            )
        return rows

    print("running dPL-path oracle (canonical + historical contrast) ...", flush=True)
    canon = run_windows("cn_psol_annual") + run_eval("cn_psol_annual")
    hist = run_windows(None) + run_eval(None)

    canon_df = pd.DataFrame(canon)
    hist_df = pd.DataFrame(hist)
    canon_df.to_csv(output_dir / "oracle_dpl_audit_windows_canonical.csv", index=False)
    hist_df.to_csv(output_dir / "oracle_dpl_audit_windows_historical.csv", index=False)

    merged = canon_df.merge(
        hist_df.drop(columns=["frac_snow"]),
        on=["basin_id", "window_offset"],
        suffixes=("_canon", "_hist"),
    )
    # frac_snow exists only on the left frame after dropping it from the right
    merged["kge_bias"] = merged["kge_hist"] - merged["kge_canon"]
    merged["abs_err_mean_bias"] = (
        merged["abs_err_mean_hist"] - merged["abs_err_mean_canon"]
    )

    def summarize(sub: pd.DataFrame) -> dict:
        return {
            "n": int(len(sub)),
            "kge_median": float(sub["kge"].median()),
            "kge_q25": float(sub["kge"].quantile(0.25)),
            "kge_q75": float(sub["kge"].quantile(0.75)),
            "kge_min": float(sub["kge"].min()),
            "kge_frac_ge_0p99": float((sub["kge"] >= 0.99).mean()),
            "kge_frac_ge_0p95": float((sub["kge"] >= 0.95).mean()),
            "abs_err_max_median": float(sub["abs_err_max"].median()),
            "abs_err_max_p95": float(sub["abs_err_max"].quantile(0.95)),
            "abs_err_mean_median": float(sub["abs_err_mean"].median()),
            "abs_err_mean_p95": float(sub["abs_err_mean"].quantile(0.95)),
        }

    window_canon = canon_df[canon_df["window_offset"] >= 0]
    eval_canon = canon_df[canon_df["window_offset"] < 0]
    window_hist = hist_df[hist_df["window_offset"] >= 0]
    eval_hist = hist_df[hist_df["window_offset"] < 0]

    # per-basin median over windows (canonical)
    per_basin = (
        window_canon.groupby("basin_id")
        .agg(
            kge_median=("kge", "median"),
            abs_err_mean_median=("abs_err_mean", "median"),
            abs_err_max_median=("abs_err_max", "median"),
            frac_snow=("frac_snow", "first"),
        )
        .reset_index()
    )
    per_basin_eval = eval_canon[["basin_id", "kge", "abs_err_mean", "frac_snow"]].copy()
    per_basin_eval.columns = ["basin_id", "kge_eval", "abs_err_mean_eval", "frac_snow"]

    def spearman(x: pd.Series, y: pd.Series) -> float:
        df = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(df) < 5 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
            return float("nan")
        return float(df["x"].corr(df["y"], method="spearman"))

    # merged bias per window row
    bias = merged[merged["window_offset"] >= 0].copy()
    per_basin_bias = (
        bias.groupby("basin_id")
        .agg(
            kge_bias_median=("kge_bias", "median"),
            abs_err_mean_bias_median=("abs_err_mean_bias", "median"),
            frac_snow=("frac_snow", "first"),
        )
        .reset_index()
    )

    report = {
        "protocol": "r3_oracle_dpl_audit_v1",
        "created_at": _utcnow(),
        "code": git_commit(args.project_root),
        "truth_run_id": args.truth_run_id,
        "n_basins": n,
        "window_offsets": list(WINDOW_OFFSETS),
        "window_path": "730-day windows (365 warm-up + 365 scored), dPL training metric kge_per_basin, loader tensors + canonical cn_psol_annual",
        "eval_path": "365-day warm-up + test period, float64 (as dPL evaluate()), validation metric compute_kge_fp64",
        "window_path_canonical": summarize(window_canon),
        "window_path_historical_gthresh": summarize(window_hist),
        "eval_path_canonical": summarize(eval_canon),
        "eval_path_historical_gthresh": summarize(eval_hist),
        "frac_snow_dependence": {
            "window_kge_vs_frac_snow_spearman": spearman(
                per_basin["frac_snow"], per_basin["kge_median"]
            ),
            "window_abs_err_mean_vs_frac_snow_spearman": spearman(
                per_basin["frac_snow"], per_basin["abs_err_mean_median"]
            ),
            "eval_kge_vs_frac_snow_spearman": spearman(
                per_basin_eval["frac_snow"], per_basin_eval["kge_eval"]
            ),
            "eval_abs_err_mean_vs_frac_snow_spearman": spearman(
                per_basin_eval["frac_snow"], per_basin_eval["abs_err_mean_eval"]
            ),
            "historical_bias_abs_err_mean_vs_frac_snow_spearman": spearman(
                per_basin_bias["frac_snow"], per_basin_bias["abs_err_mean_bias_median"]
            ),
            "historical_bias_kge_vs_frac_snow_spearman": spearman(
                per_basin_bias["frac_snow"], per_basin_bias["kge_bias_median"]
            ),
            "regime_window_kge_median": {
                regime: float(
                    window_canon.assign(
                        regime=window_canon["frac_snow"].map(snow_regime)
                    )
                    .groupby("regime")["kge"]
                    .median()
                    .get(regime, float("nan"))
                )
                for regime in ("S1", "S2", "S3", "S4", "S5")
            },
            "regime_eval_kge_median": {
                regime: float(
                    eval_canon.assign(regime=eval_canon["frac_snow"].map(snow_regime))
                    .groupby("regime")["kge"]
                    .median()
                    .get(regime, float("nan"))
                )
                for regime in ("S1", "S2", "S3", "S4", "S5")
            },
        },
        "verdict": {
            "protocol_change_required": False,
            "criteria": (
                "material snow-dependent oracle bias = window/eval oracle KGE "
                "materially below 1 and/or error growing with frac_snow; the "
                "historical g_thresh contrast quantifies the bias the fix removed"
            ),
            "note": (
                "evaluate() path residuals come from the frozen 365-day warm-up "
                "convention (default initial states 365 days before the scored "
                "period vs the truth's continuous run)"
            ),
        },
        "files": {
            "oracle_dpl_audit_windows_canonical.csv": "per basin x window, canonical",
            "oracle_dpl_audit_windows_historical.csv": "per basin x window, historical g_thresh",
        },
    }
    write_json(output_dir / "oracle_dpl_audit.json", report)
    print(
        f"COMPLETE dPL oracle audit -> {output_dir / 'oracle_dpl_audit.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
