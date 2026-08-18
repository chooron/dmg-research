#!/usr/bin/env python3
"""R3 Figure 6 data preparation (read-only reshaping of canonical post-hoc outputs).

Consumes canonical R3 post-hoc outputs (results/r3_misspec_analysis_v1/)
and produces the tidy basin-level table, seed-median aggregated table, and
figure-facing summary JSON for Figure 6 (TGD2 matched-surrogate mitigation
and residual CN explicit-process advantage).

Inputs (frozen, read-only; results/r3_misspec_analysis_v1/):
  posthoc_basin_table.csv          per basin x regime x seed x period KGE rows
                                   (kge_base, kge_tgd2, kge_cn, G_tgd2, F_tgd2, frac_snow)
  posthoc_validation_tgd2_reduction.csv  R_theta_tgd2, R_state_tgd2 per basin x regime x seed
  posthoc_validation_residual.csv  G_CN_over_TGD2, F_explicit_residual per basin x regime x seed
  posthoc_process_errors.csv       process-conditioned RMSE (snow_active, no_snow_active, melt_active)
  posthoc_summary.json             frozen F_tgd2 / G_tgd2 medians + CIs
  posthoc_validation_summary.json  frozen V3 (TGD2 reduction) + V4 (residual CN advantage)

Outputs (manuscript/results/R3/, generated artifacts):
  figure6_basin_table.csv          tidy long table (per-seed rows kept)
  figure6_basin_seedmedian.csv     dPL aggregated to per-basin seed median (IC passthrough)
  figure6_summary.json             panel-level summary statistics and frozen anchors

Usage: python manuscript/scripts/r3/prepare_figure6_data.py
       [--results-root RES] [--run-id r3_misspec_analysis_v1]
       [--manuscript-root PROJECT/manuscript] [--n-boot 2000] [--seed 20260730]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]  # manuscript/scripts -> project/hydrodiag
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from manuscript.r3.common import (  # noqa: E402
    DEFAULT_RESULTS_ROOT,
    git_commit,
    write_json,
)

SEEDS = (42, 123, 2026)
REGIMES = ("IC", "dPL")
MANUSCRIPT_R3_REL = Path("results") / "R3"


def boot_ci(values: np.ndarray, stat_fn, n_boot: int, seed: int, alpha: float = 0.05):
    """Paired basin-level bootstrap CI — same protocol as r3/posthoc_stats.py."""
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = stat_fn(values[idx])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def norm_seed(series: pd.Series) -> pd.Series:
    """Normalize seed columns: IC -> "", dPL -> "42"/"123"/"2026"."""
    return series.apply(lambda v: "" if pd.isna(v) else str(int(v)))


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"refusing: Figure 6 input missing ({label}): {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", default="r3_misspec_analysis_v1")
    parser.add_argument("--manuscript-root", type=Path, default=PROJECT / "manuscript")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    src = args.results_root / args.run_id
    out_dir = args.manuscript_root / MANUSCRIPT_R3_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. Inputs (frozen) ----------------
    p_basin = src / "posthoc_basin_table.csv"
    p_red = src / "posthoc_validation_tgd2_reduction.csv"
    p_res = src / "posthoc_validation_residual.csv"
    p_proc = src / "posthoc_process_errors.csv"
    p_sum = src / "posthoc_summary.json"
    p_val = src / "posthoc_validation_summary.json"

    for p, label in [
        (p_basin, "basin table"),
        (p_red, "TGD2 reduction"),
        (p_res, "CN residual"),
        (p_proc, "process errors"),
        (p_sum, "posthoc summary"),
        (p_val, "validation summary"),
    ]:
        require(p, label)

    bt = pd.read_csv(p_basin)
    red = pd.read_csv(p_red)
    res = pd.read_csv(p_res)
    proc = pd.read_csv(p_proc)
    frozen_sum = json.loads(p_sum.read_text())
    frozen_val = json.loads(p_val.read_text())

    for df in (bt, red, res, proc):
        df["basin_id"] = df["basin_id"].astype(str).str.zfill(8)
    bt["seed"] = norm_seed(bt["seed"])
    red["seed"] = norm_seed(red["seed"])
    res["seed"] = norm_seed(res["seed"])

    n_basins = int(bt["basin_id"].nunique())
    if n_basins != 531:
        raise SystemExit(f"refusing: expected 531 basins, found {n_basins}")

    # ---------------- 2. Process error pivoting ----------------
    proc_test = proc[proc["period"] == "test"]
    proc_rows = []
    # IC fits: CN_IC, TGD2_IC
    for b in bt["basin_id"].unique():
        row = {"basin_id": b, "paradigm": "IC", "seed": ""}
        for cond in ["snow_active", "no_snow_active", "melt_active"]:
            cn_r = proc_test[
                (proc_test["basin_id"] == b)
                & (proc_test["fit"] == "CN_IC")
                & (proc_test["condition"] == cond)
            ]
            tg_r = proc_test[
                (proc_test["basin_id"] == b)
                & (proc_test["fit"] == "TGD2_IC")
                & (proc_test["condition"] == cond)
            ]
            if len(cn_r) == 1 and len(tg_r) == 1:
                row[f"delta_rmse_{cond}"] = float(
                    tg_r["rmse"].iloc[0] - cn_r["rmse"].iloc[0]
                )
                row[f"rmse_cn_{cond}"] = float(cn_r["rmse"].iloc[0])
                row[f"rmse_tgd2_{cond}"] = float(tg_r["rmse"].iloc[0])
            else:
                row[f"delta_rmse_{cond}"] = np.nan
                row[f"rmse_cn_{cond}"] = np.nan
                row[f"rmse_tgd2_{cond}"] = np.nan
        proc_rows.append(row)
    # dPL fits: CN_dPL_s*, TGD2_dPL_s*
    for s in SEEDS:
        for b in bt["basin_id"].unique():
            row = {"basin_id": b, "paradigm": "dPL", "seed": str(s)}
            for cond in ["snow_active", "no_snow_active", "melt_active"]:
                cn_r = proc_test[
                    (proc_test["basin_id"] == b)
                    & (proc_test["fit"] == f"CN_dPL_s{s}")
                    & (proc_test["condition"] == cond)
                ]
                tg_r = proc_test[
                    (proc_test["basin_id"] == b)
                    & (proc_test["fit"] == f"TGD2_dPL_s{s}")
                    & (proc_test["condition"] == cond)
                ]
                if len(cn_r) == 1 and len(tg_r) == 1:
                    row[f"delta_rmse_{cond}"] = float(
                        tg_r["rmse"].iloc[0] - cn_r["rmse"].iloc[0]
                    )
                    row[f"rmse_cn_{cond}"] = float(cn_r["rmse"].iloc[0])
                    row[f"rmse_tgd2_{cond}"] = float(tg_r["rmse"].iloc[0])
                else:
                    row[f"delta_rmse_{cond}"] = np.nan
                    row[f"rmse_cn_{cond}"] = np.nan
                    row[f"rmse_tgd2_{cond}"] = np.nan
            proc_rows.append(row)
    df_proc_piv = pd.DataFrame(proc_rows)

    # ---------------- 3. Tidy long table (test period focus) ----------------
    tidy = (
        bt[bt["period"] == "test"]
        .merge(
            red[["basin_id", "paradigm", "seed", "R_theta_tgd2", "R_state_tgd2"]],
            on=["basin_id", "paradigm", "seed"],
            how="left",
        )
        .merge(
            res[res["period"] == "test"][
                [
                    "basin_id",
                    "paradigm",
                    "seed",
                    "G_CN_over_TGD2",
                    "F_explicit_residual",
                ]
            ],
            on=["basin_id", "paradigm", "seed"],
            how="left",
        )
        .merge(df_proc_piv, on=["basin_id", "paradigm", "seed"], how="left")
    )
    tidy = tidy.sort_values(["paradigm", "seed", "basin_id"]).reset_index(drop=True)
    tidy.to_csv(out_dir / "figure6_basin_table.csv", index=False)

    # ---------------- 4. dPL seed-median aggregation ----------------
    agg_rows = []
    for reg in REGIMES:
        sub = tidy[tidy["paradigm"] == reg]
        if reg == "IC":
            agg_rows.append(sub.assign(seed=""))
        else:
            g = sub.groupby("basin_id", as_index=False)
            agg = g.agg(
                frac_snow=("frac_snow", "first"),
                kge_base_no_refit=("kge_base_no_refit", "first"),
                kge_base=("kge_base", "median"),
                kge_tgd2=("kge_tgd2", "median"),
                kge_cn=("kge_cn", "median"),
                G_base=("G_base", "median"),
                F_close=("F_close", "median"),
                G_tgd2=("G_tgd2", "median"),
                F_tgd2=("F_tgd2", "median"),
                R_theta_tgd2=("R_theta_tgd2", "median"),
                R_state_tgd2=("R_state_tgd2", "median"),
                G_CN_over_TGD2=("G_CN_over_TGD2", "median"),
                F_explicit_residual=("F_explicit_residual", "median"),
                delta_rmse_snow_active=("delta_rmse_snow_active", "median"),
                delta_rmse_no_snow_active=("delta_rmse_no_snow_active", "median"),
                delta_rmse_melt_active=("delta_rmse_melt_active", "median"),
            )
            agg["paradigm"] = "dPL"
            agg["seed"] = ""
            agg_rows.append(agg)
    seedmed = pd.concat(agg_rows, ignore_index=True)
    seedmed.to_csv(out_dir / "figure6_basin_seedmedian.csv", index=False)

    # ---------------- 5. Sanity assertions vs frozen results ----------------
    check_errors = []
    # F_tgd2 test median
    f_ic = float(tidy[tidy["paradigm"] == "IC"]["F_tgd2"].dropna().median())
    f_ic_frozen = float(frozen_sum["IC_test"]["F_tgd2"]["median"])
    if abs(f_ic - f_ic_frozen) > 1e-9:
        check_errors.append(f"F_tgd2 IC: {f_ic} != {f_ic_frozen}")

    # R_theta_tgd2 and R_state_tgd2 IC medians
    rth_ic = float(tidy[tidy["paradigm"] == "IC"]["R_theta_tgd2"].dropna().median())
    rth_ic_frozen = float(frozen_val["V3"]["IC"]["R_theta_tgd2"]["median"])
    if abs(rth_ic - rth_ic_frozen) > 1e-9:
        check_errors.append(f"R_theta IC: {rth_ic} != {rth_ic_frozen}")

    rst_ic = float(tidy[tidy["paradigm"] == "IC"]["R_state_tgd2"].dropna().median())
    rst_ic_frozen = float(frozen_val["V3"]["IC"]["R_state_tgd2"]["median"])
    if abs(rst_ic - rst_ic_frozen) > 1e-9:
        check_errors.append(f"R_state IC: {rst_ic} != {rst_ic_frozen}")

    # G_CN_over_TGD2 IC test median
    gcn_ic = float(tidy[tidy["paradigm"] == "IC"]["G_CN_over_TGD2"].dropna().median())
    gcn_ic_frozen = float(frozen_val["V4"]["IC_test"]["G_CN_over_TGD2"]["median"])
    if abs(gcn_ic - gcn_ic_frozen) > 1e-9:
        check_errors.append(f"G_CN_over_TGD2 IC: {gcn_ic} != {gcn_ic_frozen}")

    if check_errors:
        raise SystemExit(
            "Figure 6 sanity check FAILED:\n  " + "\n  ".join(check_errors)
        )
    print(
        f"[check] Figure 6 recomputed IC medians match frozen values exactly (1e-9)",
        flush=True,
    )

    # ---------------- 6. Figure-facing summary JSON ----------------
    summary: dict = {
        "protocol": "figure6_prepare_v1",
        "source_run": args.run_id,
        "inputs": [
            str(p.relative_to(args.results_root))
            for p in [p_basin, p_red, p_res, p_proc, p_sum, p_val]
        ],
        "code": git_commit(PROJECT),
        "n_basins": n_basins,
        "n_boot": args.n_boot,
        "boot_seed": args.seed,
    }

    # -- Panel (a): Structural ladder test KGEs --
    pa = {}
    for reg in REGIMES:
        sub = seedmed[seedmed["paradigm"] == reg]
        pa[reg] = {
            "kge_base_median": float(sub["kge_base"].median()),
            "kge_tgd2_median": float(sub["kge_tgd2"].median()),
            "kge_cn_median": float(sub["kge_cn"].median()),
            "kge_base_no_refit_median": float(sub["kge_base_no_refit"].median()),
            "G_tgd2_median": float(sub["G_tgd2"].median()),
        }
    summary["panel_a_ladder"] = pa

    # -- Panel (b): Mitigation fraction F_tgd2 --
    pb = {}
    for reg in REGIMES:
        sub = seedmed[seedmed["paradigm"] == reg]
        vals = sub["F_tgd2"].dropna().to_numpy()
        ci = boot_ci(vals, np.median, args.n_boot, args.seed + 1)
        entry = {
            "median": float(np.median(vals)),
            "q25": float(np.quantile(vals, 0.25)),
            "q75": float(np.quantile(vals, 0.75)),
            "boot_ci_median_display": list(ci),
            "n_valid": int(len(vals)),
            "frac_gt_0": float((vals > 0).mean()),
            "frac_ge_0p5": float((vals >= 0.5).mean()),
            "frac_outside_display_window": float(((vals < -0.4) | (vals > 1.4)).mean()),
        }
        if reg == "dPL":
            entry["seed_medians"] = [
                float(frozen_sum[f"dPL_{s}_test"]["F_tgd2"]["median"]) for s in SEEDS
            ]
        else:
            entry["frozen_median"] = f_ic_frozen
        pb[reg] = entry
    summary["panel_b_f_tgd2"] = pb

    # -- Panel (c): Parameter burden reduction R_theta_tgd2 --
    pc = {}
    for reg in REGIMES:
        sub = seedmed[seedmed["paradigm"] == reg]
        vals = sub["R_theta_tgd2"].dropna().to_numpy()
        ci = boot_ci(vals, np.median, args.n_boot, args.seed + 2)
        entry = {
            "median": float(np.median(vals)),
            "q25": float(np.quantile(vals, 0.25)),
            "q75": float(np.quantile(vals, 0.75)),
            "boot_ci_median_display": list(ci),
            "n_valid": int(len(vals)),
            "frac_gt_0": float((vals > 0).mean()),
            "frac_outside_display_window": float(
                ((vals < -0.03) | (vals > 0.07)).mean()
            ),
        }
        if reg == "IC":
            entry["boot_ci_frozen"] = frozen_val["V3"]["IC"]["R_theta_tgd2"][
                "boot_ci_median"
            ]
            entry["frac_gt_0_frozen"] = frozen_val["V3"]["IC"]["R_theta_tgd2"][
                "frac_gt_0"
            ]
        else:
            entry["seed_medians"] = [
                float(frozen_val["V3"][f"dPL_{s}"]["R_theta_tgd2"]["median"])
                for s in SEEDS
            ]
        pc[reg] = entry
    summary["panel_c_r_theta"] = pc

    # -- Panel (d): State burden reduction R_state_tgd2 --
    pd_ = {}
    for reg in REGIMES:
        sub = seedmed[seedmed["paradigm"] == reg]
        vals = sub["R_state_tgd2"].dropna().to_numpy()
        ci = boot_ci(vals, np.median, args.n_boot, args.seed + 3)
        entry = {
            "median": float(np.median(vals)),
            "q25": float(np.quantile(vals, 0.25)),
            "q75": float(np.quantile(vals, 0.75)),
            "boot_ci_median_display": list(ci),
            "n_valid": int(len(vals)),
            "frac_gt_0": float((vals > 0).mean()),
            "frac_outside_display_window": float(((vals < -0.3) | (vals > 0.9)).mean()),
        }
        if reg == "IC":
            entry["boot_ci_frozen"] = frozen_val["V3"]["IC"]["R_state_tgd2"][
                "boot_ci_median"
            ]
            entry["frac_gt_0_frozen"] = frozen_val["V3"]["IC"]["R_state_tgd2"][
                "frac_gt_0"
            ]
        else:
            entry["seed_medians"] = [
                float(frozen_val["V3"][f"dPL_{s}"]["R_state_tgd2"]["median"])
                for s in SEEDS
            ]
        pd_[reg] = entry
    summary["panel_d_r_state"] = pd_

    # -- Panel (e): Residual CN-over-TGD2 advantage vs frac_snow --
    pe = {}
    fs_all = np.sort(seedmed[seedmed["paradigm"] == "IC"]["frac_snow"].to_numpy())
    q_bins = np.quantile(fs_all, [0.25, 0.5, 0.75])
    bins = [
        (-np.inf, q_bins[0]),
        (q_bins[0], q_bins[1]),
        (q_bins[1], q_bins[2]),
        (q_bins[2], np.inf),
    ]

    for reg in REGIMES:
        sub = seedmed[seedmed["paradigm"] == reg]
        x = sub["frac_snow"].to_numpy()
        y = sub["G_CN_over_TGD2"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]

        bin_entries = []
        for k, (lo, hi) in enumerate(bins):
            m = (x > lo) & (x <= hi)
            if m.sum() < 10:
                continue
            bm = y[m]
            bin_entries.append(
                {
                    "bin": k + 1,
                    "frac_snow_range": [float(lo), float(hi)],
                    "n": int(m.sum()),
                    "frac_snow_median": float(np.median(x[m])),
                    "median": float(np.median(bm)),
                    "boot_ci_median_display": list(
                        boot_ci(bm, np.median, args.n_boot, args.seed + 20 + k)
                    ),
                }
            )
        sp_frozen = (
            frozen_val["V4"]["IC_test"]["G_CN_over_TGD2"]["spearman_vs_frac_snow"]
            if reg == "IC"
            else [
                frozen_val["V4"][f"dPL_{s}_test"]["G_CN_over_TGD2"][
                    "spearman_vs_frac_snow"
                ]
                for s in SEEDS
            ]
        )
        pe[reg] = {
            "median": float(np.median(y)),
            "spearman_frozen": sp_frozen
            if isinstance(sp_frozen, list)
            else [float(sp_frozen)],
            "quartile_bins": bin_entries,
            "y_display_limits": [-0.05, 0.35],
            "frac_beyond_y_display": float((y > 0.35).mean()),
            "frac_below_y_display": float((y < -0.05).mean()),
            "n": int(len(y)),
        }
    summary["panel_e_residual_vs_frac_snow"] = pe

    # -- Panel (f): Process-conditioned residual errors (snow_active vs no_snow) --
    pf = {}
    for reg in REGIMES:
        sub = seedmed[seedmed["paradigm"] == reg]
        pf[reg] = {}
        for cond in ["snow_active", "no_snow_active", "melt_active"]:
            d = sub[f"delta_rmse_{cond}"].dropna().to_numpy()
            ci = boot_ci(d, np.median, args.n_boot, args.seed + 30)
            pf[reg][cond] = {
                "median": float(np.median(d)),
                "q25": float(np.quantile(d, 0.25)),
                "q75": float(np.quantile(d, 0.75)),
                "boot_ci_median_display": list(ci),
                "n_valid": int(len(d)),
                "frac_gt_0": float((d > 0).mean()),
            }
    summary["panel_f_process_errors"] = pf

    write_json(out_dir / "figure6_summary.json", summary)

    print(f"COMPLETE Figure 6 data -> {out_dir}", flush=True)
    print(
        f"  figure6_basin_table.csv      (tidy, per seed: {len(tidy)} rows)", flush=True
    )
    print(
        f"  figure6_basin_seedmedian.csv (dPL seed-aggregated: {len(seedmed)} rows)",
        flush=True,
    )
    print(f"  figure6_summary.json         (panel-level numbers)", flush=True)


if __name__ == "__main__":
    main()
