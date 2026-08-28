"""Stage 5: Primary Explanatory — All 15 Signed Parameter Shifts.

Calculates canonical signed shifts delta_j = z_Base,j - z_CN,j across all 15 shared parameters:
  - Full sample (531 basins): median, IQR, 95% bootstrap CI, positive/negative/near-zero fractions,
    Spearman rho, OLS slope beta + 95% CI.
  - S1-S5 strata: median, IQR, 95% bootstrap CI.
  - S5-S1 endpoint activity contrast: Delta_S5_S1 = median(S5) - median(S1) + 95% CI.
  - ExcludeS5 (476 basins): slope beta + 95% CI, Spearman rho + 95% CI.
  - Leave-one-stratum-out sensitivity across all 15 parameters.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from r2_config import (
    BASE_SEED,
    DEFAULT_DRAWS,
    PARADIGMS,
    RESULTS_DIR,
    STRATA,
    STRATA_COUNTS,
    TOTAL_BASINS,
)
from canonical_vectors import build_canonical_parameter_vectors
from parameter_ledger import load_canonical_snow_metadata
from shared_parameter_specs import PARAMETER_METADATA, SHARED_15_PARAMETERS


def bootstrap_stat_ci(
    values: np.ndarray,
    stat_fn,
    seed: int,
    draws: int = DEFAULT_DRAWS,
) -> Tuple[float, float, float]:
    """Compute bootstrap CI for a scalar statistic."""
    val = np.asarray(values, dtype=np.float64)
    val = val[np.isfinite(val)]
    n = len(val)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    pt = float(stat_fn(val))
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(draws, n))
    boot_stats = np.array([stat_fn(val[boot_idx[i]]) for i in range(draws)])
    ci_l = float(np.quantile(boot_stats, 0.025))
    ci_h = float(np.quantile(boot_stats, 0.975))
    return pt, ci_l, ci_h


def bootstrap_slope_and_spearman(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    draws: int = DEFAULT_DRAWS,
) -> Dict[str, float]:
    """Compute OLS slope and Spearman rho with paired bootstrap CIs."""
    mask = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    n = len(xv)
    if n < 3:
        return {
            "slope": float("nan"), "slope_ci_low": float("nan"), "slope_ci_high": float("nan"),
            "spearman_rho": float("nan"), "spearman_ci_low": float("nan"), "spearman_ci_high": float("nan"),
        }

    slope_pt = float(np.polyfit(xv, yv, 1)[0])
    rho_pt = float(spearmanr(xv, yv)[0])

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(draws, n))

    b_slopes = np.zeros(draws, dtype=np.float64)
    b_rhos = np.zeros(draws, dtype=np.float64)

    for i in range(draws):
        bx, by = xv[boot_idx[i]], yv[boot_idx[i]]
        b_slopes[i] = np.polyfit(bx, by, 1)[0]
        b_rhos[i] = spearmanr(bx, by)[0]

    return {
        "slope": slope_pt,
        "slope_ci_low": float(np.quantile(b_slopes, 0.025)),
        "slope_ci_high": float(np.quantile(b_slopes, 0.975)),
        "spearman_rho": rho_pt,
        "spearman_ci_low": float(np.quantile(b_rhos, 0.025)),
        "spearman_ci_high": float(np.quantile(b_rhos, 0.975)),
    }


def analyze_parameter_shifts_all15(
    canonical_rows: List[Dict[str, Any]] | None = None,
    output_dir: Path | None = None,
    draws: int = DEFAULT_DRAWS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Compute all 15 signed parameter shifts across Full, Strata, ExcludeS5, and Leave-one-stratum-out.

    Returns:
        (basin_shifts, full_summaries, strata_summaries, robustness_summaries, audit_meta)
    """
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if canonical_rows is None:
        canonical_rows, _ = build_canonical_parameter_vectors(output_dir=out_dir)

    snow_meta = load_canonical_snow_metadata()
    basins = sorted(snow_meta.keys())

    canon_df = pd.DataFrame(canonical_rows)
    basin_shifts: List[Dict[str, Any]] = []

    # 1. Compute per-basin paired shifts delta = z_Base - z_CN
    for paradigm in PARADIGMS:
        p_df = canon_df[canon_df["paradigm"] == paradigm]
        base_map = p_df[p_df["structure"] == "Base"].set_index("basin_id")
        cn_map = p_df[p_df["structure"] == "CN"].set_index("basin_id")

        for b_id in basins:
            frac_snow, stratum = snow_meta[b_id]
            for p_name in SHARED_15_PARAMETERS:
                z_base = float(base_map.loc[b_id, f"z_{p_name}"])
                z_cn = float(cn_map.loc[b_id, f"z_{p_name}"])
                delta = z_base - z_cn

                phys_base = float(base_map.loc[b_id, f"phys_{p_name}"])
                phys_cn = float(cn_map.loc[b_id, f"phys_{p_name}"])
                delta_phys = phys_base - phys_cn

                basin_shifts.append({
                    "basin_id": b_id,
                    "paradigm": paradigm,
                    "regime": paradigm,
                    "contrast": "Base-CN",
                    "parameter": p_name,
                    "symbol": PARAMETER_METADATA[p_name]["symbol"],
                    "display_name": PARAMETER_METADATA[p_name]["display"],
                    "z_base": z_base,
                    "z_cn": z_cn,
                    "delta_base_minus_cn": delta,
                    "physical_base": phys_base,
                    "physical_cn": phys_cn,
                    "delta_physical": delta_phys,
                    "frac_snow": frac_snow,
                    "snow_stratum": stratum,
                    "snow_regime": stratum,
                })

    shift_df = pd.DataFrame(basin_shifts)

    # 2. Full Sample Summaries (30 rows: 2 paradigms x 15 parameters)
    full_summaries: List[Dict[str, Any]] = []
    for paradigm in PARADIGMS:
        p_sub = shift_df[shift_df["paradigm"] == paradigm]
        xs = p_sub.drop_duplicates("basin_id").set_index("basin_id").loc[basins, "frac_snow"].to_numpy()

        for p_name in SHARED_15_PARAMETERS:
            p_rows = p_sub[p_sub["parameter"] == p_name].set_index("basin_id").loc[basins]
            deltas = p_rows["delta_base_minus_cn"].to_numpy(dtype=np.float64)

            seed_p = BASE_SEED + 6000 + len(full_summaries)
            med_pt, ci_l, ci_h = bootstrap_stat_ci(deltas, np.median, seed=seed_p, draws=draws)
            q25 = float(np.quantile(deltas, 0.25))
            q75 = float(np.quantile(deltas, 0.75))

            reg = bootstrap_slope_and_spearman(xs, deltas, seed=seed_p + 50, draws=draws)

            pos_frac = float(np.mean(deltas > 0))
            neg_frac = float(np.mean(deltas < 0))
            zero_tol_frac = float(np.mean(np.abs(deltas) <= 0.01))

            full_summaries.append({
                "paradigm": paradigm,
                "contrast": "Base-CN",
                "parameter": p_name,
                "symbol": PARAMETER_METADATA[p_name]["symbol"],
                "display_name": PARAMETER_METADATA[p_name]["display"],
                "process": PARAMETER_METADATA[p_name]["process"],
                "n": len(deltas),
                "median_shift": med_pt,
                "q25": q25,
                "q75": q75,
                "iqr": q75 - q25,
                "ci95_low": ci_l,
                "ci95_high": ci_h,
                "slope_beta": reg["slope"],
                "beta": reg["slope"],
                "slope_ci_low": reg["slope_ci_low"],
                "slope_ci_high": reg["slope_ci_high"],
                "spearman_rho": reg["spearman_rho"],
                "spearman_ci_low": reg["spearman_ci_low"],
                "spearman_ci_high": reg["spearman_ci_high"],
                "positive_fraction": pos_frac,
                "negative_fraction": neg_frac,
                "near_zero_fraction_0p01": zero_tol_frac,
            })

    # 3. Stratified S1-S5 Summaries & Endpoint Contests (150 rows)
    strata_summaries: List[Dict[str, Any]] = []
    for paradigm in PARADIGMS:
        p_sub = shift_df[shift_df["paradigm"] == paradigm]
        for p_name in SHARED_15_PARAMETERS:
            p_rows = p_sub[p_sub["parameter"] == p_name]

            s1_deltas = p_rows[p_rows["snow_stratum"] == "S1"]["delta_base_minus_cn"].to_numpy()
            s5_deltas = p_rows[p_rows["snow_stratum"] == "S5"]["delta_base_minus_cn"].to_numpy()
            seed_ep = BASE_SEED + 7000 + len(strata_summaries)

            def endpoint_diff(idx):
                return float(np.median(s5_deltas) - np.median(s1_deltas))

            d_act_pt = float(np.median(s5_deltas) - np.median(s1_deltas))
            rng = np.random.default_rng(seed_ep)
            boot_ep = np.zeros(draws)
            for i in range(draws):
                b_s1 = s1_deltas[rng.integers(0, len(s1_deltas), size=len(s1_deltas))]
                b_s5 = s5_deltas[rng.integers(0, len(s5_deltas), size=len(s5_deltas))]
                boot_ep[i] = np.median(b_s5) - np.median(b_s1)
            ep_ci_l = float(np.quantile(boot_ep, 0.025))
            ep_ci_h = float(np.quantile(boot_ep, 0.975))

            for s_name in STRATA:
                s_deltas = p_rows[p_rows["snow_stratum"] == s_name]["delta_base_minus_cn"].to_numpy()
                seed_st = BASE_SEED + 8000 + len(strata_summaries)
                med, ci_l, ci_h = bootstrap_stat_ci(s_deltas, np.median, seed=seed_st, draws=draws)
                q25 = float(np.quantile(s_deltas, 0.25))
                q75 = float(np.quantile(s_deltas, 0.75))

                strata_summaries.append({
                    "paradigm": paradigm,
                    "contrast": "Base-CN",
                    "parameter": p_name,
                    "symbol": PARAMETER_METADATA[p_name]["symbol"],
                    "snow_stratum": s_name,
                    "n_basins": len(s_deltas),
                    "median_shift": med,
                    "q25": q25,
                    "q75": q75,
                    "iqr": q75 - q25,
                    "ci95_low": ci_l,
                    "ci95_high": ci_h,
                    "D_activity_S5_minus_S1": d_act_pt if s_name == "S5" else float("nan"),
                    "D_activity_ci_low": ep_ci_l if s_name == "S5" else float("nan"),
                    "D_activity_ci_high": ep_ci_h if s_name == "S5" else float("nan"),
                })

    # 4. Robustness Summaries: ExcludeS5 & Leave-One-Stratum-Out
    robustness_summaries: List[Dict[str, Any]] = []
    for paradigm in PARADIGMS:
        p_sub = shift_df[shift_df["paradigm"] == paradigm]

        for p_name in SHARED_15_PARAMETERS:
            p_rows = p_sub[p_sub["parameter"] == p_name].set_index("basin_id").loc[basins]

            # 4a. ExcludeS5
            excl_mask = p_rows["snow_stratum"] != "S5"
            xs_excl = p_rows.loc[excl_mask, "frac_snow"].to_numpy()
            ys_excl = p_rows.loc[excl_mask, "delta_base_minus_cn"].to_numpy()

            seed_rob = BASE_SEED + 9000 + len(robustness_summaries)
            reg_excl = bootstrap_slope_and_spearman(xs_excl, ys_excl, seed=seed_rob, draws=draws)
            med_excl, mci_l, mci_h = bootstrap_stat_ci(ys_excl, np.median, seed=seed_rob + 1, draws=draws)

            robustness_summaries.append({
                "paradigm": paradigm,
                "parameter": p_name,
                "symbol": PARAMETER_METADATA[p_name]["symbol"],
                "subset": "exclude_S5",
                "omitted_stratum": "S5",
                "n_basins": len(ys_excl),
                "slope": reg_excl["slope"],
                "slope_ci_low": reg_excl["slope_ci_low"],
                "slope_ci_high": reg_excl["slope_ci_high"],
                "spearman_rho": reg_excl["spearman_rho"],
                "spearman_ci_low": reg_excl["spearman_ci_low"],
                "spearman_ci_high": reg_excl["spearman_ci_high"],
                "median_shift": med_excl,
                "median_ci_low": mci_l,
                "median_ci_high": mci_h,
            })

            # 4b. Leave-one-stratum-out for each S1..S5
            for s_omit in STRATA:
                loso_mask = p_rows["snow_stratum"] != s_omit
                xs_loso = p_rows.loc[loso_mask, "frac_snow"].to_numpy()
                ys_loso = p_rows.loc[loso_mask, "delta_base_minus_cn"].to_numpy()

                seed_loso = BASE_SEED + 10000 + len(robustness_summaries)
                reg_loso = bootstrap_slope_and_spearman(xs_loso, ys_loso, seed=seed_loso, draws=draws)
                med_loso, lci_l, lci_h = bootstrap_stat_ci(ys_loso, np.median, seed=seed_loso + 1, draws=draws)

                robustness_summaries.append({
                    "paradigm": paradigm,
                    "parameter": p_name,
                    "symbol": PARAMETER_METADATA[p_name]["symbol"],
                    "subset": f"leave_out_{s_omit}",
                    "omitted_stratum": s_omit,
                    "n_basins": len(ys_loso),
                    "slope": reg_loso["slope"],
                    "slope_ci_low": reg_loso["slope_ci_low"],
                    "slope_ci_high": reg_loso["slope_ci_high"],
                    "spearman_rho": reg_loso["spearman_rho"],
                    "spearman_ci_low": reg_loso["spearman_ci_low"],
                    "spearman_ci_high": reg_loso["spearman_ci_high"],
                    "median_shift": med_loso,
                    "median_ci_low": lci_l,
                    "median_ci_high": lci_h,
                })

    # Write output CSV files
    pd.DataFrame(basin_shifts).to_csv(out_dir / "r2_paired_shifts_basin_level.csv", index=False, float_format="%.17g")
    pd.DataFrame(full_summaries).to_csv(out_dir / "r2_parameter_shifts_full_summary.csv", index=False, float_format="%.17g")
    pd.DataFrame(full_summaries).to_csv(out_dir / "r2_snow_gradients_summary.csv", index=False, float_format="%.17g")
    pd.DataFrame(strata_summaries).to_csv(out_dir / "r2_parameter_shifts_strata_summary.csv", index=False, float_format="%.17g")
    pd.DataFrame(robustness_summaries).to_csv(out_dir / "r2_snow_gradient_robustness.csv", index=False, float_format="%.17g")

    audit_meta = {
        "status": "PASS",
        "total_basin_shift_rows": len(basin_shifts),
        "total_full_summary_rows": len(full_summaries),
        "total_strata_summary_rows": len(strata_summaries),
        "total_robustness_rows": len(robustness_summaries),
        "parameters_evaluated": list(SHARED_15_PARAMETERS),
        "paradigms": list(PARADIGMS),
    }

    with (out_dir / "parameter_shifts_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit_meta, f, indent=2)

    return basin_shifts, full_summaries, strata_summaries, robustness_summaries, audit_meta


if __name__ == "__main__":
    b_shifts, f_sum, s_sum, rob_sum, meta = analyze_parameter_shifts_all15()
    print("Parameter shifts analysis complete:")
    print(f"  Basin shift rows: {len(b_shifts)}")
    print(f"  Full summary rows: {len(f_sum)}")
    print("\nKey Signatures (Full sample OLS slope beta):")
    for r in f_sum:
        if r["parameter"] in ["xaj_um", "xaj_ki", "xaj_ci", "xaj_im"]:
            print(f"  [{r['paradigm']}] {r['parameter']:8s} ({r['symbol']:5s}): slope = {r['slope_beta']:+.3f} [{r['slope_ci_low']:+.3f}, {r['slope_ci_high']:+.3f}], rho = {r['spearman_rho']:+.3f}")
