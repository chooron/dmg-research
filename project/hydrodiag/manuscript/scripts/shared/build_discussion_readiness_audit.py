#!/usr/bin/env python3
"""Build canonical Discussion-readiness audit tables.

This script audits and freezes canonical statistics across Results 3.1–3.5
(R1–R5) for use in Discussion 4.1–4.4. It is strictly read-only with respect to
underlying model outputs, calibrations, and training checkpoints.

Outputs (manuscript/results/discussion_audit/):
  - r3_gap_recovery_ratio_audit.csv
  - r3_denominator_sensitivity_audit.csv
  - r3_conditional_association_audit.csv
  - r1_endpoint_timing_audit.csv
  - r4_tgd_spring_timing_audit.csv
  - r5_coherence_estimand_audit.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "manuscript" / "results" / "discussion_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOOT_SEED = 20260730
N_BOOT = 2000


def boot_ci_median(v: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float]:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(v)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = np.median(v[idx])
    return (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    v = np.isfinite(x) & np.isfinite(y)
    if v.sum() < 5 or x[v].std() == 0 or y[v].std() == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x[v]))
    ry = np.argsort(np.argsort(y[v]))
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    v = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if v.sum() < 8:
        return float("nan")

    def rank(u: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(u[v])).astype(float) + 1.0

    def resid(u: np.ndarray, c: np.ndarray) -> np.ndarray:
        a = np.vstack([c, np.ones_like(c)]).T
        coef, *_ = np.linalg.lstsq(a, u, rcond=None)
        return u - a @ coef

    rx, ry, rz = rank(x), rank(y), rank(z)
    ex, ey = resid(rx, rz), resid(ry, rz)
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def boot_ci_corr(
    x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if z is None:
            draws[b] = spearman(x[idx], y[idx])
        else:
            draws[b] = partial_spearman(x[idx], y[idx], z[idx])
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        return (np.nan, np.nan)
    return (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))


def main() -> None:
    # ---------------- 1. R3 Gap Recovery Ratio ----------------
    f5_path = PROJECT_ROOT / "manuscript" / "results" / "R3" / "figure5_basin_seedmedian.csv"
    df_f5 = pd.read_csv(f5_path)
    t1_rows = []
    for period in ("test", "train"):
        for reg in ("IC", "dPL"):
            sub = df_f5[(df_f5["paradigm"] == reg) & (df_f5["period"] == period)]
            for st in ("Full", "S1", "S2", "S3", "S4", "S5", "S4+S5"):
                if st == "Full":
                    s = sub
                elif st == "S4+S5":
                    s = sub[sub["snow_stratum"].isin(["S4", "S5"])]
                else:
                    s = sub[sub["snow_stratum"] == st]

                n_tot = len(s)
                d = s["D"].to_numpy()
                gb = s["G_base"].to_numpy()
                gt = s["G_TGD"].to_numpy()

                m_val = d > 1e-6
                n_val = int(m_val.sum())
                val_rate = n_val / n_tot if n_tot > 0 else 0

                d_med = np.median(d)
                d_iqr = np.percentile(d, 75) - np.percentile(d, 25)
                d_ci = boot_ci_median(d)
                d_le0 = int((d <= 0).sum())
                d_near0 = int(((d > 0) & (d <= 0.01)).sum())

                gb_med = np.median(gb)
                gb_iqr = np.percentile(gb, 75) - np.percentile(gb, 25)
                gb_ci = boot_ci_median(gb)
                gb_gt0 = float((gb > 0).mean())

                gt_med = np.median(gt)
                gt_iqr = np.percentile(gt, 75) - np.percentile(gt, 25)
                gt_ci = boot_ci_median(gt)
                gt_gt0 = float((gt > 0).mean())

                if n_val > 0:
                    fc = gb[m_val] / d[m_val]
                    ft = gt[m_val] / d[m_val]
                    df_diff = ft - fc

                    fc_med = np.median(fc)
                    fc_iqr = np.percentile(fc, 75) - np.percentile(fc, 25)
                    fc_ci = boot_ci_median(fc)
                    fc_lt0 = float((fc < 0).mean())
                    fc_01 = float(((fc >= 0) & (fc <= 1)).mean())
                    fc_gt1 = float((fc > 1).mean())

                    ft_med = np.median(ft)
                    ft_iqr = np.percentile(ft, 75) - np.percentile(ft, 25)
                    ft_ci = boot_ci_median(ft)
                    ft_lt0 = float((ft < 0).mean())
                    ft_01 = float(((ft >= 0) & (ft <= 1)).mean())
                    ft_gt1 = float((ft > 1).mean())

                    df_med = np.median(df_diff)
                    df_ci = boot_ci_median(df_diff)
                    df_gt0 = float((df_diff > 0).mean())
                else:
                    fc_med, fc_iqr, fc_ci, fc_lt0, fc_01, fc_gt1 = [np.nan] * 6
                    ft_med, ft_iqr, ft_ci, ft_lt0, ft_01, ft_gt1 = [np.nan] * 6
                    df_med, df_ci, df_gt0 = np.nan, (np.nan, np.nan), np.nan

                t1_rows.append(
                    {
                        "paradigm": reg,
                        "period": period,
                        "snow_stratum": st,
                        "n_total": n_tot,
                        "n_valid": n_val,
                        "valid_rate": val_rate,
                        "D_median": d_med,
                        "D_iqr": d_iqr,
                        "D_ci_low": d_ci[0],
                        "D_ci_high": d_ci[1],
                        "D_le0_count": d_le0,
                        "D_near0_count": d_near0,
                        "G_base_median": gb_med,
                        "G_base_iqr": gb_iqr,
                        "G_base_ci_low": gb_ci[0],
                        "G_base_ci_high": gb_ci[1],
                        "G_base_gt0_prop": gb_gt0,
                        "G_TGD_median": gt_med,
                        "G_TGD_iqr": gt_iqr,
                        "G_TGD_ci_low": gt_ci[0],
                        "G_TGD_ci_high": gt_ci[1],
                        "G_TGD_gt0_prop": gt_gt0,
                        "F_close_median": fc_med,
                        "F_close_iqr": fc_iqr,
                        "F_close_ci_low": fc_ci[0],
                        "F_close_ci_high": fc_ci[1],
                        "F_close_lt0_prop": fc_lt0,
                        "F_close_0_to_1_prop": fc_01,
                        "F_close_gt1_prop": fc_gt1,
                        "F_TGD_median": ft_med,
                        "F_TGD_iqr": ft_iqr,
                        "F_TGD_ci_low": ft_ci[0],
                        "F_TGD_ci_high": ft_ci[1],
                        "F_TGD_lt0_prop": ft_lt0,
                        "F_TGD_0_to_1_prop": ft_01,
                        "F_TGD_gt1_prop": ft_gt1,
                        "delta_F_median": df_med,
                        "delta_F_ci_low": df_ci[0],
                        "delta_F_ci_high": df_ci[1],
                        "delta_F_gt0_prop": df_gt0,
                    }
                )
    pd.DataFrame(t1_rows).to_csv(OUT_DIR / "r3_gap_recovery_ratio_audit.csv", index=False)

    # ---------------- 2. R3 Denominator Sensitivity ----------------
    t2_rows = []
    thresholds = [1e-6, 1e-4, 1e-3, 0.01, 0.02, 0.05, 0.10]
    for reg in ("IC", "dPL"):
        sub = df_f5[(df_f5["paradigm"] == reg) & (df_f5["period"] == "test")]
        for th in thresholds:
            m = sub["D"] > th
            n_val = int(m.sum())
            d_val = sub.loc[m, "D"]
            fc = sub.loc[m, "G_base"] / d_val
            ft = sub.loc[m, "G_TGD"] / d_val
            df_diff = ft - fc
            t2_rows.append(
                {
                    "paradigm": reg,
                    "period": "test",
                    "threshold": th,
                    "n_valid": n_val,
                    "valid_rate": n_val / 531,
                    "F_close_median": np.median(fc),
                    "F_close_iqr": np.percentile(fc, 75) - np.percentile(fc, 25),
                    "F_close_p5": np.percentile(fc, 5),
                    "F_close_p95": np.percentile(fc, 95),
                    "F_TGD_median": np.median(ft),
                    "F_TGD_iqr": np.percentile(ft, 75) - np.percentile(ft, 25),
                    "F_TGD_p5": np.percentile(ft, 5),
                    "F_TGD_p95": np.percentile(ft, 95),
                    "delta_F_median": np.median(df_diff),
                    "delta_F_gt0_prop": float((df_diff > 0).mean()),
                }
            )
    pd.DataFrame(t2_rows).to_csv(OUT_DIR / "r3_denominator_sensitivity_audit.csv", index=False)

    # ---------------- 3. R3 Conditional Associations ----------------
    f6_path = PROJECT_ROOT / "manuscript" / "results" / "R3" / "figure6_basin_seedmedian.csv"
    df_f6 = pd.read_csv(f6_path)
    t3_rows = []
    for reg in ("IC", "dPL"):
        sub = df_f6[df_f6["paradigm"] == reg].copy()
        fs = sub["frac_snow"].to_numpy()
        pairs = [
            ("G_Base vs E_param_excess_Base", sub["G_base"].to_numpy(), sub["E_param_excess_base"].to_numpy()),
            ("G_TGD vs E_param_excess_TGD", sub["G_TGD"].to_numpy(), sub["E_param_excess_tgd"].to_numpy()),
            ("G_Base vs delta_E_state_Wt", sub["G_base"].to_numpy(), sub["delta_E_wt_base"].to_numpy()),
            ("G_TGD vs delta_E_state_Wt", sub["G_TGD"].to_numpy(), sub["delta_E_wt_tgd"].to_numpy()),
        ]
        for name, x, y in pairs:
            valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(fs)
            n_val = int(valid_mask.sum())
            xv, yv, fsv = x[valid_mask], y[valid_mask], fs[valid_mask]
            raw_r = spearman(xv, yv)
            raw_ci = boot_ci_corr(xv, yv)
            part_r = partial_spearman(xv, yv, fsv)
            part_ci = boot_ci_corr(xv, yv, fsv)

            strata_res = {}
            for st in ("S1", "S2", "S3", "S4", "S5"):
                mst = sub["snow_stratum"] == st
                strata_res[st] = spearman(x[mst], y[mst])

            t3_rows.append(
                {
                    "paradigm": reg,
                    "pair_name": name,
                    "n_valid": n_val,
                    "raw_spearman": raw_r,
                    "raw_ci_low": raw_ci[0],
                    "raw_ci_high": raw_ci[1],
                    "partial_spearman_ctrl_snow": part_r,
                    "partial_ci_low": part_ci[0],
                    "partial_ci_high": part_ci[1],
                    "S1_raw": strata_res["S1"],
                    "S2_raw": strata_res["S2"],
                    "S3_raw": strata_res["S3"],
                    "S4_raw": strata_res["S4"],
                    "S5_raw": strata_res["S5"],
                }
            )
    pd.DataFrame(t3_rows).to_csv(OUT_DIR / "r3_conditional_association_audit.csv", index=False)

    # ---------------- 4. R1 Endpoint Timing ----------------
    r1_paired = PROJECT_ROOT / "manuscript" / "results" / "R1" / "r1_paired_effects_summary.csv"
    df_paired = pd.read_csv(r1_paired)
    ct = df_paired[df_paired["metric"] == "ct_error_abs"].copy()
    t4_rows = []
    for _, r in ct.iterrows():
        t4_rows.append(
            {
                "paradigm": r["paradigm"],
                "effect": r["effect"],
                "period": r["period"],
                "snow_stratum": str(r["snow_stratum"]),
                "valid_basin_count": int(r["valid_basin_count"]),
                "median_days": float(r["median"]),
                "ci_low_days": float(r["bootstrap_ci_low"]),
                "ci_high_days": float(r["bootstrap_ci_high"]),
            }
        )
    pd.DataFrame(t4_rows).to_csv(OUT_DIR / "r1_endpoint_timing_audit.csv", index=False)

    # ---------------- 5. R4 Spring Timing ----------------
    r4_path = PROJECT_ROOT / "results" / "r4_phase1_soil_official" / "three_structure_timing_metrics_basin_summary.csv"
    df_r4_t3 = pd.read_csv(r4_path)
    t5_rows = []
    for reg in ("IC_fused", "dPL_seed42", "dPL_seed123", "dPL_seed2026", "dPL_seed_median"):
        if reg == "dPL_seed_median":
            sub = (
                df_r4_t3[df_r4_t3["regime"].str.startswith("dPL_")]
                .groupby(["structure", "basin_id"])
                .median(numeric_only=True)
                .reset_index()
            )
        else:
            sub = df_r4_t3[df_r4_t3["regime"] == reg]
        for struct in ("Base", "TGD2", "CN"):
            s = sub[sub["structure"] == struct]
            n_basins = len(s)
            ws = s["median_wetup_error_days"].to_numpy()
            wa = s["median_abs_wetup_error_days"].to_numpy()
            ps = s["median_peak_error_days"].to_numpy()
            pa = s["median_abs_peak_error_days"].to_numpy()
            t5_rows.append(
                {
                    "regime": reg,
                    "structure": struct,
                    "n_basins": n_basins,
                    "wetup_signed_median": np.median(ws),
                    "wetup_signed_ci_low": boot_ci_median(ws)[0],
                    "wetup_signed_ci_high": boot_ci_median(ws)[1],
                    "wetup_abs_median": np.median(wa),
                    "wetup_abs_ci_low": boot_ci_median(wa)[0],
                    "wetup_abs_ci_high": boot_ci_median(wa)[1],
                    "peak_signed_median": np.median(ps),
                    "peak_signed_ci_low": boot_ci_median(ps)[0],
                    "peak_signed_ci_high": boot_ci_median(ps)[1],
                    "peak_abs_median": np.median(pa),
                    "peak_abs_ci_low": boot_ci_median(pa)[0],
                    "peak_abs_ci_high": boot_ci_median(pa)[1],
                }
            )
    pd.DataFrame(t5_rows).to_csv(OUT_DIR / "r4_tgd_spring_timing_audit.csv", index=False)

    # ---------------- 6. R5 Coherence Estimands ----------------
    r5_can_path = PROJECT_ROOT / "manuscript" / "results" / "R5" / "r5_figure9_primary_agreement.csv"
    r5_old_path = PROJECT_ROOT / "manuscript" / "cache" / "r5_cross_model_agreement_table.csv"
    df_r5_can = pd.read_csv(r5_can_path)
    df_r5_old = pd.read_csv(r5_old_path)
    t6_rows = []
    for _, r in df_r5_can.iterrows():
        t6_rows.append(
            {
                "estimand_framework": "Canonical Timing (Base vs CN delta_|CT| > 0)",
                "regime": r["regime"],
                "snow_stratum": r["stratum"],
                "n_basins": int(r["N"]),
                "P_all_3_positive": float(r["P_3_of_3"]),
                "P_exactly_2_positive": float(r["P_exactly_2_of_3"]),
                "P_majority_positive": float(r["P_at_least_2"]),
                "P_majority_ci_low": float(r["P_at_least_2_ci_low"]),
                "P_majority_ci_high": float(r["P_at_least_2_ci_high"]),
            }
        )
    for _, r in df_r5_old.iterrows():
        t6_rows.append(
            {
                "estimand_framework": "Old KGE (CN vs TGD2 KGE > 0)",
                "regime": r["regime"],
                "snow_stratum": str(r["stratum"]),
                "n_basins": int(r["N"]),
                "P_all_3_positive": float(r["P(A=3) [All 3 agree CN>TGD2]"]),
                "P_exactly_2_positive": np.nan,
                "P_majority_positive": float(r["P(A>=2) [Majority agree]"]),
                "P_majority_ci_low": np.nan,
                "P_majority_ci_high": np.nan,
            }
        )
    pd.DataFrame(t6_rows).to_csv(OUT_DIR / "r5_coherence_estimand_audit.csv", index=False)
    print("Discussion readiness audit tables successfully generated in:", OUT_DIR)


if __name__ == "__main__":
    main()
