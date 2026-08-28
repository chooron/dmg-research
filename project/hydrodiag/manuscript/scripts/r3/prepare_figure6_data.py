#!/usr/bin/env python3
"""R3 Figure 6 data preparation — canonical internal recovery evidence.

Consumes canonical R3 post-hoc outputs:
  - paired_parameters.csv (15 shared parameters truth errors)
  - state_excess.csv (common state/flux NRMSE excess)
  - posthoc_basin_table.csv (outlet recovery G_base, G_TGD)
  - r1_snow_attributes.csv (S1-S5 strata)
  - fig6_seasonal/ (monthly high-snow process trajectories)
  - r3_synthetic_truth_v1/ (generating truth seasonal profiles)

Produces (manuscript/results/R3/):
  figure6_basin_table.csv          tidy long table (per seed)
  figure6_basin_seedmedian.csv     dPL aggregated to per-basin seed median (IC passthrough)
  figure6_summary.json             panel-level summary statistics, CIs, and seasonal ensembles

Usage: python manuscript/scripts/r3/prepare_figure6_data.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]  # manuscript/scripts -> project/hydrodiag
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from manuscript.scripts.r3.common import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_RESULTS_ROOT,
    git_commit,
    load_bundle,
    period_indices,
    write_json,
)

SEEDS = (42, 123, 2026)
REGIMES = ("IC", "dPL")
MANUSCRIPT_R3_REL = Path("results") / "R3"
COMMON_XAJ = [
    "xaj_k", "xaj_b", "xaj_im", "xaj_um", "xaj_lm", "xaj_dm", "xaj_c",
    "xaj_sm", "xaj_ex", "xaj_ki", "xaj_kg", "xaj_ci", "xaj_cg",
    "xaj_a", "xaj_theta",
]
MONTH_LABELS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]


def boot_ci(values: np.ndarray, stat_fn, n_boot: int, seed: int, alpha: float = 0.05):
    """Paired basin-level bootstrap CI."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    n = len(vals)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = stat_fn(vals[idx])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return [float(lo), float(hi)]


def boot_ci_12m(arr_2d: np.ndarray, n_boot: int = 2000, seed: int = 20260730, alpha: float = 0.05):
    """Compute bootstrap CI of the median for each of the 12 months."""
    rng = np.random.default_rng(seed)
    n, m = arr_2d.shape
    ci_lo = np.empty(m)
    ci_hi = np.empty(m)
    for j in range(m):
        col = arr_2d[:, j]
        col_v = col[np.isfinite(col)]
        draws = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, len(col_v), len(col_v))
            draws[b] = np.median(col_v[idx])
        ci_lo[j], ci_hi[j] = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return [float(x) for x in ci_lo], [float(x) for x in ci_hi]


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    v = np.isfinite(x) & np.isfinite(y)
    if v.sum() < 5 or x[v].std() == 0 or y[v].std() == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x[v]))
    ry = np.argsort(np.argsort(y[v]))
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_spearman(x, y, z):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    v = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if v.sum() < 8:
        return float("nan")

    def rank(u):
        return np.argsort(np.argsort(u[v])).astype(float) + 1.0

    def resid(u, c):
        A = np.vstack([c, np.ones_like(c)]).T
        coef, *_ = np.linalg.lstsq(A, u, rcond=None)
        return u - A @ coef

    rx, ry, rz = rank(x), rank(y), rank(z)
    ex, ey = resid(rx, rz), resid(ry, rz)
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def quant(v, q):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, q)) if len(v) else float("nan")


def norm_seed(series: pd.Series) -> pd.Series:
    return series.apply(lambda v: "" if pd.isna(v) or v == "" else str(int(float(v))))


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"refusing: Figure 6 input missing ({label}): {path}")


def get_ensemble_stats(arr: np.ndarray, seed: int = 20260730) -> dict[str, list[float]]:
    """Compute per-month cross-basin median, Q25, Q75, and bootstrap 95% CI of median."""
    ci_lo, ci_hi = boot_ci_12m(arr, seed=seed)
    return {
        "median": [float(x) for x in np.median(arr, axis=0)],
        "q25": [float(x) for x in np.percentile(arr, 25, axis=0)],
        "q75": [float(x) for x in np.percentile(arr, 75, axis=0)],
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-id", default="r3_misspec_analysis_v1")
    parser.add_argument("--manuscript-root", type=Path, default=PROJECT / "manuscript")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    src = args.results_root / args.run_id
    truth_dir = args.results_root / "r3_synthetic_truth_v1"
    r1_dir = args.manuscript_root / "results" / "R1"
    out_dir = args.manuscript_root / MANUSCRIPT_R3_REL
    seasonal_dir = out_dir / "fig6_seasonal"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. Inputs ----------------
    p_params = src / "paired_parameters.csv"
    p_state = src / "state_excess.csv"
    p_basin = src / "posthoc_basin_table.csv"
    p_strata = r1_dir / "r1_snow_attributes.csv"
    p_seas_inp = seasonal_dir / "fig6_seasonal_input.npz"
    p_seas_st = seasonal_dir / "fig6_seasonal_state.npz"
    p_truth_snow = truth_dir / "snow_star.npz"
    p_truth_x = truth_dir / "x_star.npz"
    p_truth_theta = truth_dir / "theta_star.npz"

    for p, label in [
        (p_params, "paired parameters"),
        (p_state, "state excess"),
        (p_basin, "basin table"),
        (p_strata, "snow attributes"),
        (p_seas_inp, "seasonal input"),
        (p_seas_st, "seasonal state"),
        (p_truth_snow, "truth snow"),
        (p_truth_x, "truth states"),
        (p_truth_theta, "truth theta"),
    ]:
        require(p, label)

    # ---------------- 2. Parameter truth errors ----------------
    pp = pd.read_csv(p_params)
    pp["basin_id"] = pp["basin_id"].astype(str).str.zfill(8)
    pp["seed_key"] = pp["seed"].fillna(-1).astype(int)

    # Verify exactly 15 COMMON_XAJ
    assert sorted(pp["parameter"].unique()) == sorted(COMMON_XAJ)

    agg_p = pp.groupby(["basin_id", "paradigm", "seed_key", "structure"]).agg(
        E_param=("e", lambda x: np.median(np.abs(x))),
        C15=("delta_e", lambda x: np.median(np.abs(x))),
    ).reset_index()

    cn_p = pp.groupby(["basin_id", "paradigm", "seed_key"]).agg(
        E_param_cn=("e_cn", lambda x: np.median(np.abs(x)))
    ).reset_index()

    wide_p = agg_p.pivot(index=["basin_id", "paradigm", "seed_key"], columns="structure").reset_index()
    wide_p.columns = [f"{a}_{b}" if b else a for a, b in wide_p.columns]
    wide_p = wide_p.rename(columns={
        "E_param_Base": "E_param_base",
        "E_param_TGD2": "E_param_tgd",
        "C15_Base": "C15_base",
        "C15_TGD2": "C15_tgd",
    })
    param_df = wide_p.merge(cn_p, on=["basin_id", "paradigm", "seed_key"])
    param_df["seed"] = param_df["seed_key"].map(lambda k: "" if k == -1 else str(k))
    param_df = param_df.drop(columns=["seed_key"])
    param_df["E_param_excess_base"] = param_df["E_param_base"] - param_df["E_param_cn"]
    param_df["E_param_excess_tgd"] = param_df["E_param_tgd"] - param_df["E_param_cn"]

    # ---------------- 3. State excess (NRMSE, test period) ----------------
    se = pd.read_csv(p_state)
    se["basin_id"] = se["basin_id"].astype(str).str.zfill(8)
    se["seed_key"] = se["seed"].fillna(-1).astype(int)
    se_test = se[(se["metric"] == "nrmse") & (se["period"] == "test")]

    se_pivot = se_test.pivot_table(
        index=["basin_id", "paradigm", "seed_key"],
        columns=["structure", "variable"],
        values="delta_E",
        aggfunc="first"
    ).reset_index()
    se_pivot.columns = [f"delta_E_{v}_{s.lower().replace('2', '')}" if v else s for s, v in se_pivot.columns]
    se_pivot["seed"] = se_pivot["seed_key"].map(lambda k: "" if k == -1 else str(k))
    se_pivot = se_pivot.drop(columns=["seed_key"])

    # ---------------- 4. Posthoc basin table (outlet recovery, test) ----------------
    bt = pd.read_csv(p_basin)
    bt["basin_id"] = bt["basin_id"].astype(str).str.zfill(8)
    bt["seed"] = norm_seed(bt["seed"])
    bt_test = bt[bt["period"] == "test"].copy()
    bt_test["G_TGD"] = bt_test["kge_tgd2"] - bt_test["kge_base_no_refit"]

    # ---------------- 5. Strata merge ----------------
    strata = pd.read_csv(p_strata)
    strata["basin_id"] = strata["basin_id"].astype(str).str.zfill(8)

    # ---------------- 6. Tidy table assembly ----------------
    tidy = param_df.merge(se_pivot, on=["basin_id", "paradigm", "seed"], how="left")
    tidy = tidy.merge(
        bt_test[["basin_id", "paradigm", "seed", "kge_base_no_refit", "kge_base", "kge_tgd2", "kge_cn", "G_base", "G_TGD", "frac_snow"]],
        on=["basin_id", "paradigm", "seed"], how="left"
    )
    tidy = tidy.merge(strata[["basin_id", "snow_stratum"]], on="basin_id", how="left")
    tidy = tidy.sort_values(["paradigm", "seed", "basin_id"]).reset_index(drop=True)
    tidy.to_csv(out_dir / "figure6_basin_table.csv", index=False)

    # ---------------- 7. dPL seed-median aggregation ----------------
    agg_rows = []
    for reg in REGIMES:
        sub = tidy[tidy["paradigm"] == reg]
        if reg == "IC":
            agg_rows.append(sub.assign(seed=""))
        else:
            g = sub.groupby(["basin_id"], as_index=False)
            agg_dict = {
                "frac_snow": ("frac_snow", "first"),
                "snow_stratum": ("snow_stratum", "first"),
                "kge_base_no_refit": ("kge_base_no_refit", "first"),
                "kge_base": ("kge_base", "median"),
                "kge_tgd2": ("kge_tgd2", "median"),
                "kge_cn": ("kge_cn", "median"),
                "G_base": ("G_base", "median"),
                "G_TGD": ("G_TGD", "median"),
                "E_param_base": ("E_param_base", "median"),
                "E_param_tgd": ("E_param_tgd", "median"),
                "E_param_cn": ("E_param_cn", "median"),
                "E_param_excess_base": ("E_param_excess_base", "median"),
                "E_param_excess_tgd": ("E_param_excess_tgd", "median"),
                "C15_base": ("C15_base", "median"),
                "C15_tgd": ("C15_tgd", "median"),
            }
            # Add all state excess variables
            for v in ["wt", "wu", "wl", "wd", "s", "fr", "qi", "qg"]:
                agg_dict[f"delta_E_{v}_base"] = (f"delta_E_{v}_base", "median")
                agg_dict[f"delta_E_{v}_tgd"] = (f"delta_E_{v}_tgd", "median")
            agg = g.agg(**agg_dict)
            agg["paradigm"] = "dPL"
            agg["seed"] = ""
            agg_rows.append(agg)
    seedmed = pd.concat(agg_rows, ignore_index=True)
    seedmed.to_csv(out_dir / "figure6_basin_seedmedian.csv", index=False)

    # ---------------- 8. Seasonal trajectory processing (basin-first ensemble) ----------------
    seas_inp = np.load(p_seas_inp)
    seas_st = np.load(p_seas_st)
    high_ids = list(seas_inp["basin_ids"])

    # Load bundle for date indices and truth arrays
    os.environ["HYDRODIAG_DATA_ROOT"] = str(args.data_root)
    bundle, _ = load_bundle(PROJECT, args.data_root)
    pi = period_indices(bundle)
    test_start, test_end = pi["test"]
    tids = [str(b).zfill(8) for b in np.load(p_truth_theta)["basin_ids"]]
    high_idx = [tids.index(b) for b in high_ids]

    months = np.asarray([int(str(d)[5:7]) for d in bundle.dates[test_start : test_end + 1]])
    wy_month = ((months - 10) % 12).astype(np.int64)

    # Truth effective input per basin x month [133, 12]
    sn = np.load(p_truth_snow)["effective_precip"][high_idx, test_start : test_end + 1]
    truth_inp_monthly = np.empty((len(high_ids), 12), dtype=np.float32)
    for m in range(12):
        truth_inp_monthly[:, m] = sn[:, wy_month == m].mean(axis=1)

    # Truth wt storage per basin x month [133, 12]
    xn = np.load(p_truth_x)
    wt = (xn["wu"] + xn["wl"] + xn["wd"])[high_idx, test_start : test_end + 1]
    truth_wt_monthly = np.empty((len(high_ids), 12), dtype=np.float32)
    for m in range(12):
        truth_wt_monthly[:, m] = wt[:, wy_month == m].mean(axis=1)

    # ---------------- 9. Summary structure for plot_figure6.py ----------------
    summary: dict = {
        "protocol": "figure6_prepare_v2",
        "source_run": args.run_id,
        "code": git_commit(PROJECT),
        "n_basins": 531,
        "n_boot": args.n_boot,
        "boot_seed": args.seed,
    }

    # Panel (a): Parameter truth distance across strata
    pa = {}
    for reg in REGIMES:
        sub = seedmed[seedmed["paradigm"] == reg]
        fs = sub["frac_snow"].to_numpy()
        strata_dict = {}
        for st in ("S1", "S2", "S3", "S4", "S5"):
            m = sub["snow_stratum"] == st
            strata_dict[st] = {
                "n": int(m.sum()),
                "Base": {
                    "median": quant(sub.loc[m, "E_param_base"], 0.5),
                    "ci": list(boot_ci(sub.loc[m, "E_param_base"], np.median, args.n_boot, args.seed + 10)),
                },
                "TGD": {
                    "median": quant(sub.loc[m, "E_param_tgd"], 0.5),
                    "ci": list(boot_ci(sub.loc[m, "E_param_tgd"], np.median, args.n_boot, args.seed + 11)),
                },
                "CN": {
                    "median": quant(sub.loc[m, "E_param_cn"], 0.5),
                    "ci": list(boot_ci(sub.loc[m, "E_param_cn"], np.median, args.n_boot, args.seed + 12)),
                },
            }
        pa[reg] = {
            "strata": strata_dict,
            "spearman_frac_snow": {
                "Base": float(spearman(fs, sub["E_param_base"])),
                "TGD": float(spearman(fs, sub["E_param_tgd"])),
                "CN": float(spearman(fs, sub["E_param_cn"])),
            },
            "overall_medians": {
                "Base": float(sub["E_param_base"].median()),
                "TGD": float(sub["E_param_tgd"].median()),
                "CN": float(sub["E_param_cn"].median()),
            }
        }
    summary["panel_a_param_distance"] = pa

    # Panel (b): Parameter excess error across strata
    pb = {}
    for reg in REGIMES:
        sub = seedmed[seedmed["paradigm"] == reg]
        fs = sub["frac_snow"].to_numpy()
        strata_dict = {}
        for st in ("S1", "S2", "S3", "S4", "S5"):
            m = sub["snow_stratum"] == st
            strata_dict[st] = {
                "n": int(m.sum()),
                "Base": {
                    "median": quant(sub.loc[m, "E_param_excess_base"], 0.5),
                    "ci": list(boot_ci(sub.loc[m, "E_param_excess_base"], np.median, args.n_boot, args.seed + 20)),
                },
                "TGD": {
                    "median": quant(sub.loc[m, "E_param_excess_tgd"], 0.5),
                    "ci": list(boot_ci(sub.loc[m, "E_param_excess_tgd"], np.median, args.n_boot, args.seed + 21)),
                },
            }
        pb[reg] = {
            "strata": strata_dict,
            "spearman_frac_snow": {
                "Base": float(spearman(fs, sub["E_param_excess_base"])),
                "TGD": float(spearman(fs, sub["E_param_excess_tgd"])),
            },
            "overall_medians": {
                "Base": float(sub["E_param_excess_base"].median()),
                "TGD": float(sub["E_param_excess_tgd"].median()),
            },
            "frac_lt_0": {
                "Base": float((sub["E_param_excess_base"] < 0).mean()),
                "TGD": float((sub["E_param_excess_tgd"] < 0).mean()),
            }
        }
    summary["panel_b_param_excess"] = pb

    # Panel (c): Common-state / flux excess (PRIMARY)
    pc = {}
    for reg in REGIMES:
        sub = seedmed[seedmed["paradigm"] == reg]
        vars_dict = {}
        for var, is_flux, display_name in [
            ("wt", False, "Total storage $W_t$ (headline)"),
            ("wu", False, "Upper tension $W_u$"),
            ("wl", False, "Lower tension $W_l$"),
            ("qi", True, "Interflow flux $Q_i$"),
            ("qg", True, "Baseflow flux $Q_g$"),
        ]:
            col_b = f"delta_E_{var}_base"
            col_t = f"delta_E_{var}_tgd"
            vb = sub[col_b].dropna().to_numpy()
            vt = sub[col_t].dropna().to_numpy()

            vars_dict[var] = {
                "display_name": display_name,
                "is_flux": is_flux,
                "Base": {
                    "median": float(np.median(vb)),
                    "q25": float(np.quantile(vb, 0.25)),
                    "q75": float(np.quantile(vb, 0.75)),
                    "ci": list(boot_ci(vb, np.median, args.n_boot, args.seed + 30)),
                },
                "TGD": {
                    "median": float(np.median(vt)),
                    "q25": float(np.quantile(vt, 0.25)),
                    "q75": float(np.quantile(vt, 0.75)),
                    "ci": list(boot_ci(vt, np.median, args.n_boot, args.seed + 31)),
                },
            }
        pc[reg] = vars_dict
    summary["panel_c_state_excess"] = pc

    # Panel (d): Association audit (Raw vs Partial Spearman)
    pd_assoc = {}
    for reg in REGIMES:
        sub = seedmed[seedmed["paradigm"] == reg]
        fs = sub["frac_snow"].to_numpy()
        gb = sub["G_base"].to_numpy()
        gt = sub["G_TGD"].to_numpy()
        eb = sub["E_param_excess_base"].to_numpy()
        et = sub["E_param_excess_tgd"].to_numpy()
        ewt_b = sub["delta_E_wt_base"].to_numpy()
        ewt_t = sub["delta_E_wt_tgd"].to_numpy()

        pairs = [
            ("G_Base <-> E_param_excess_Base", gb, eb, "$G_{\\mathrm{Base}} \\leftrightarrow E^{\\mathrm{param,excess}}_{\\mathrm{Base}}$"),
            ("G_TGD <-> E_param_excess_TGD",   gt, et, "$G_{\\mathrm{TGD}} \\leftrightarrow E^{\\mathrm{param,excess}}_{\\mathrm{TGD}}$"),
            ("G_Base <-> Delta E_state(Wt)",   gb, ewt_b, "$G_{\\mathrm{Base}} \\leftrightarrow \\Delta E(W_t)$"),
            ("G_TGD <-> Delta E_state(Wt)",    gt, ewt_t, "$G_{\\mathrm{TGD}} \\leftrightarrow \\Delta E(W_t)$"),
        ]

        pair_dict = {}
        for pname, x, y, latex_lbl in pairs:
            raw_rho = float(spearman(x, y))
            part_rho = float(partial_spearman(x, y, fs))
            pair_dict[pname] = {
                "label": latex_lbl,
                "raw_spearman": raw_rho,
                "partial_spearman": part_rho,
            }
        pd_assoc[reg] = pair_dict
    summary["panel_d_associations"] = pd_assoc

    # Panel (e): Seasonal liquid-water delivery (cross-basin ensemble: median + 95% bootstrap CI)
    summary["panel_e_seasonal_input"] = {
        "months": MONTH_LABELS,
        "n_high_snow_basins": len(high_ids),
        "Truth": get_ensemble_stats(truth_inp_monthly, seed=args.seed + 50),
        "Base_IC": get_ensemble_stats(seas_inp["Base_IC"], seed=args.seed + 51),
        "TGD_IC": get_ensemble_stats(seas_inp["TGD2_IC"], seed=args.seed + 52),
        "CN_IC": get_ensemble_stats(seas_inp["CN_IC"], seed=args.seed + 53),
        "Base_dPL": get_ensemble_stats(seas_inp["Base_dPL"], seed=args.seed + 54),
        "TGD_dPL": get_ensemble_stats(seas_inp["TGD2_dPL"], seed=args.seed + 55),
        "CN_dPL": get_ensemble_stats(seas_inp["CN_dPL"], seed=args.seed + 56),
    }

    # Panel (f): Truth-relative seasonal deviation Delta Wt heatmap matrix & IQR heterogeneity
    heatmap_row_keys = [
        ("Base_IC", "Base (IC)"),
        ("Base_dPL", "Base (dPL)"),
        ("TGD2_IC", "TGD (IC)"),
        ("TGD2_dPL", "TGD (dPL)"),
        ("CN_IC", "CN refit (IC)"),
        ("CN_dPL", "CN refit (dPL)"),
    ]

    matrix_median = []
    matrix_iqr = []
    series_dict = {}

    for k, lbl in heatmap_row_keys:
        delta_m = seas_st[k] - truth_wt_monthly
        med_12 = [float(x) for x in np.median(delta_m, axis=0)]
        q25_12 = [float(x) for x in np.percentile(delta_m, 25, axis=0)]
        q75_12 = [float(x) for x in np.percentile(delta_m, 75, axis=0)]
        iqr_12 = [float(q75_12[j] - q25_12[j]) for j in range(12)]
        ci_lo_12, ci_hi_12 = boot_ci_12m(delta_m, seed=args.seed + 70)

        matrix_median.append(med_12)
        matrix_iqr.append(iqr_12)
        series_dict[k] = {
            "label": lbl,
            "median": med_12,
            "q25": q25_12,
            "q75": q75_12,
            "iqr": iqr_12,
            "ci_lo": ci_lo_12,
            "ci_hi": ci_hi_12,
        }

    summary["panel_f_seasonal_storage_heatmap"] = {
        "months": MONTH_LABELS,
        "n_high_snow_basins": len(high_ids),
        "row_labels": [r[1] for r in heatmap_row_keys],
        "median_matrix": matrix_median,
        "iqr_matrix": matrix_iqr,
        "row_iqr_medians": [float(np.median(iqr_row)) for iqr_row in matrix_iqr],
        "series": series_dict,
    }

    write_json(out_dir / "figure6_summary.json", summary)

    print(f"COMPLETE Figure 6 data -> {out_dir}", flush=True)
    print(f"  figure6_basin_table.csv      (tidy, per seed: {len(tidy)} rows)", flush=True)
    print(f"  figure6_basin_seedmedian.csv (dPL seed-aggregated: {len(seedmed)} rows)", flush=True)
    print(f"  figure6_summary.json         (panel-level numbers)", flush=True)


if __name__ == "__main__":
    main()
