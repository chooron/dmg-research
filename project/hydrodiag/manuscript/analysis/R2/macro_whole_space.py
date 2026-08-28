"""Stage 4: Primary Macro Whole-Parameter-Space Base-CN Response.

Computes:
  4A. Canonical-Vector 15-D Displacement (Descriptive sensitivity):
        D_rms = sqrt(1/15 * sum( (z_Base - z_CN)^2 ))
        D_euclidean = sqrt(sum( (z_Base - z_CN)^2 ))
  4B. Ensemble-Level Within-Adjusted Structural Separation (Primary identification macro for Figure 3):
        - IC: 10 restarts -> 45 within_Base pairs, 45 within_CN pairs, 100 between_all pairs
        - dPL: 3 seeds -> 3 within_Base pairs, 3 within_CN pairs, 9 between_all pairs
        - within_pooled = (within_Base + within_CN) / 2
        - excess = between_all - within_pooled
        - prevalence: fraction(between_all > within_pooled)
        - regressions on frac_snow for within_pooled, between_all, and excess
"""
from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from r2_config import (
    BASE_SEED,
    DEFAULT_DRAWS,
    DPL_SEEDS,
    IC_STARTS,
    PARADIGMS,
    RESULTS_DIR,
    STRATA,
    STRATA_COUNTS,
    TOTAL_BASINS,
)
from canonical_vectors import build_canonical_parameter_vectors
from parameter_ledger import build_raw_parameter_ledger, load_canonical_snow_metadata
from shared_parameter_specs import SHARED_15_PARAMETERS


def rms_distance(z1: np.ndarray, z2: np.ndarray) -> float:
    """Compute root-mean-square distance over 15 normalized parameters."""
    return float(np.sqrt(np.mean((z1 - z2) ** 2)))


def euclidean_distance(z1: np.ndarray, z2: np.ndarray) -> float:
    """Compute Euclidean distance over 15 normalized parameters."""
    return float(np.sqrt(np.sum((z1 - z2) ** 2)))


def bootstrap_median_ci_cpu(
    values: np.ndarray,
    seed: int,
    draws: int = DEFAULT_DRAWS,
) -> Tuple[float, float, float, float, float]:
    """Bootstrap median CI on 1D array on CPU."""
    val = np.asarray(values, dtype=np.float64)
    val = val[np.isfinite(val)]
    n = len(val)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    pt_med = float(np.median(val))
    q25 = float(np.quantile(val, 0.25))
    q75 = float(np.quantile(val, 0.75))

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(draws, n))
    boot_medians = np.median(val[boot_indices], axis=1)

    ci_l = float(np.quantile(boot_medians, 0.025))
    ci_h = float(np.quantile(boot_medians, 0.975))
    return pt_med, ci_l, ci_h, q25, q75


def bootstrap_mean_ci_cpu(
    values: np.ndarray,
    seed: int,
    draws: int = DEFAULT_DRAWS,
) -> Tuple[float, float, float]:
    """Bootstrap mean CI for proportions on CPU."""
    val = np.asarray(values, dtype=np.float64)
    val = val[np.isfinite(val)]
    n = len(val)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    pt_mean = float(np.mean(val))
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(draws, n))
    boot_means = np.mean(val[boot_indices], axis=1)

    ci_l = float(np.quantile(boot_means, 0.025))
    ci_h = float(np.quantile(boot_means, 0.975))
    return pt_mean, ci_l, ci_h


def bootstrap_regression_cpu(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    draws: int = DEFAULT_DRAWS,
) -> Dict[str, float]:
    """OLS slope and Spearman rank correlation with paired bootstrap CIs."""
    mask = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    n = len(xv)
    if n < 3:
        return {
            "slope": float("nan"), "slope_ci_low": float("nan"), "slope_ci_high": float("nan"),
            "spearman_rho": float("nan"), "spearman_ci_low": float("nan"), "spearman_ci_high": float("nan"),
        }

    # Point estimates
    slope_pt = float(np.polyfit(xv, yv, 1)[0])
    from scipy.stats import spearmanr
    rho_pt = float(spearmanr(xv, yv)[0])

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(draws, n))

    boot_slopes = np.zeros(draws, dtype=np.float64)
    boot_rhos = np.zeros(draws, dtype=np.float64)

    for i in range(draws):
        bx, by = xv[boot_idx[i]], yv[boot_idx[i]]
        boot_slopes[i] = np.polyfit(bx, by, 1)[0]
        boot_rhos[i] = spearmanr(bx, by)[0]

    return {
        "slope": slope_pt,
        "slope_ci_low": float(np.quantile(boot_slopes, 0.025)),
        "slope_ci_high": float(np.quantile(boot_slopes, 0.975)),
        "spearman_rho": rho_pt,
        "spearman_ci_low": float(np.quantile(boot_rhos, 0.025)),
        "spearman_ci_high": float(np.quantile(boot_rhos, 0.975)),
    }


def analyze_macro_whole_space(
    canonical_rows: List[Dict[str, Any]] | None = None,
    ledger_rows: List[Dict[str, Any]] | None = None,
    output_dir: Path | None = None,
    draws: int = DEFAULT_DRAWS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Compute 4A (canonical D) and 4B (ensemble excess & regressions).

    Returns:
        (canonical_d_basin, canonical_d_summary, ensemble_basin, ensemble_summary, audit_meta)
    """
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if canonical_rows is None:
        canonical_rows, _ = build_canonical_parameter_vectors(output_dir=out_dir)
    if ledger_rows is None:
        ledger_rows, _ = build_raw_parameter_ledger(output_dir=out_dir)

    snow_meta = load_canonical_snow_metadata()
    basins = sorted(snow_meta.keys())

    # =========================================================================
    # 4A. Canonical-Vector 15-D Displacement
    # =========================================================================
    canon_df = pd.DataFrame(canonical_rows)
    canon_d_basin: List[Dict[str, Any]] = []

    for paradigm in PARADIGMS:
        p_df = canon_df[canon_df["paradigm"] == paradigm]
        base_df = p_df[p_df["structure"] == "Base"].set_index("basin_id")
        cn_df = p_df[p_df["structure"] == "CN"].set_index("basin_id")

        for b_id in basins:
            z_base = np.array([base_df.loc[b_id, f"z_{p}"] for p in SHARED_15_PARAMETERS], dtype=np.float64)
            z_cn = np.array([cn_df.loc[b_id, f"z_{p}"] for p in SHARED_15_PARAMETERS], dtype=np.float64)
            d_rms = rms_distance(z_base, z_cn)
            d_euc = euclidean_distance(z_base, z_cn)
            frac_snow, stratum = snow_meta[b_id]

            canon_d_basin.append({
                "basin_id": b_id,
                "paradigm": paradigm,
                "regime": paradigm,
                "contrast": "Base-CN",
                "D_rms": d_rms,
                "D_euclidean": d_euc,
                "frac_snow": frac_snow,
                "snow_stratum": stratum,
            })

    # Summaries for 4A
    canon_d_summary: List[Dict[str, Any]] = []
    cd_df = pd.DataFrame(canon_d_basin)

    for paradigm in PARADIGMS:
        sub_p = cd_df[cd_df["paradigm"] == paradigm]
        for metric_col in ["D_rms", "D_euclidean"]:
            # 1. Subsets: Full531 and ExcludeS5
            for subset_name, mask in [("Full531", np.ones(len(sub_p), dtype=bool)), ("ExcludeS5", sub_p["snow_stratum"] != "S5")]:
                vals = sub_p.loc[mask, metric_col].to_numpy()
                xs = sub_p.loc[mask, "frac_snow"].to_numpy()

                seed_sub = BASE_SEED + 1000 + len(canon_d_summary)
                med, cil, cih, q25, q75 = bootstrap_median_ci_cpu(vals, seed=seed_sub, draws=draws)
                reg_res = bootstrap_regression_cpu(xs, vals, seed=seed_sub + 10, draws=draws)

                canon_d_summary.append({
                    "table": "canonical_vector_15D_displacement",
                    "paradigm": paradigm,
                    "contrast": "Base-CN",
                    "metric": metric_col,
                    "stratum": subset_name,
                    "n_basins": int(len(vals)),
                    "median": med,
                    "q25": q25,
                    "q75": q75,
                    "iqr": q75 - q25,
                    "ci_low": cil,
                    "ci_high": cih,
                    "slope_beta": reg_res["slope"],
                    "slope_ci_low": reg_res["slope_ci_low"],
                    "slope_ci_high": reg_res["slope_ci_high"],
                    "spearman_rho": reg_res["spearman_rho"],
                    "spearman_ci_low": reg_res["spearman_ci_low"],
                    "spearman_ci_high": reg_res["spearman_ci_high"],
                })

            # 2. S1-S5 Strata
            for s_name in STRATA:
                mask = sub_p["snow_stratum"] == s_name
                vals = sub_p.loc[mask, metric_col].to_numpy()
                seed_s = BASE_SEED + 2000 + len(canon_d_summary)
                med, cil, cih, q25, q75 = bootstrap_median_ci_cpu(vals, seed=seed_s, draws=draws)

                canon_d_summary.append({
                    "table": "canonical_vector_15D_displacement_strata",
                    "paradigm": paradigm,
                    "contrast": "Base-CN",
                    "metric": metric_col,
                    "stratum": s_name,
                    "n_basins": int(len(vals)),
                    "median": med,
                    "q25": q25,
                    "q75": q75,
                    "iqr": q75 - q25,
                    "ci_low": cil,
                    "ci_high": cih,
                    "slope_beta": float("nan"),
                    "slope_ci_low": float("nan"),
                    "slope_ci_high": float("nan"),
                    "spearman_rho": float("nan"),
                    "spearman_ci_low": float("nan"),
                    "spearman_ci_high": float("nan"),
                })

    # =========================================================================
    # 4B. Ensemble-Level Within-Adjusted Structural Separation (Figure 3 Primary)
    # =========================================================================
    ledger_df = pd.DataFrame(ledger_rows)
    ensemble_basin: List[Dict[str, Any]] = []

    # Process IC
    ic_df = ledger_df[ledger_df["paradigm"] == "IC"]
    ic_dict: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {b: {"Base": {}, "CN": {}} for b in basins}

    for (struct, b_id, start_idx), g in ic_df[ic_df["structure"].isin(["Base", "CN"])].groupby(["structure", "basin_id", "start_or_seed"]):
        z_vec = g.set_index("parameter").loc[list(SHARED_15_PARAMETERS)]["normalized_value"].to_numpy(dtype=np.float64)
        ic_dict[b_id][struct][int(start_idx)] = z_vec

    for b_id in basins:
        b_starts = ic_dict[b_id]["Base"]
        c_starts = ic_dict[b_id]["CN"]
        if len(b_starts) != 10 or len(c_starts) != 10:
            raise RuntimeError(f"Basin {b_id} incomplete IC restarts: Base={len(b_starts)}, CN={len(c_starts)}")

        w_base = float(np.median([rms_distance(b_starts[s1], b_starts[s2]) for s1, s2 in combinations(range(10), 2)]))
        w_cn = float(np.median([rms_distance(c_starts[s1], c_starts[s2]) for s1, s2 in combinations(range(10), 2)]))
        w_pooled = (w_base + w_cn) / 2.0
        b_all = float(np.median([rms_distance(b_starts[s1], c_starts[s2]) for s1 in range(10) for s2 in range(10)]))
        excess = b_all - w_pooled

        frac_snow, stratum = snow_meta[b_id]
        ensemble_basin.append({
            "basin_id": b_id,
            "paradigm": "IC",
            "regime": "IC",
            "contrast": "Base-CN",
            "within_base": w_base,
            "within_cn": w_cn,
            "within_pooled": w_pooled,
            "between_all": b_all,
            "excess": excess,
            "prop_between_gt_within": bool(b_all > w_pooled),
            "prop_excess_gt_0": bool(excess > 0),
            "frac_snow": frac_snow,
            "snow_stratum": stratum,
        })

    # Process dPL
    dpl_df = ledger_df[ledger_df["paradigm"] == "dPL"]
    dpl_dict: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {b: {"Base": {}, "CN": {}} for b in basins}

    for (struct, b_id, seed_val), g in dpl_df[dpl_df["structure"].isin(["Base", "CN"])].groupby(["structure", "basin_id", "start_or_seed"]):
        z_vec = g.set_index("parameter").loc[list(SHARED_15_PARAMETERS)]["normalized_value"].to_numpy(dtype=np.float64)
        dpl_dict[b_id][struct][int(seed_val)] = z_vec

    for b_id in basins:
        b_seeds = dpl_dict[b_id]["Base"]
        c_seeds = dpl_dict[b_id]["CN"]
        if len(b_seeds) != 3 or len(c_seeds) != 3:
            raise RuntimeError(f"Basin {b_id} incomplete dPL seeds: Base={len(b_seeds)}, CN={len(c_seeds)}")

        w_base = float(np.median([rms_distance(b_seeds[s1], b_seeds[s2]) for s1, s2 in combinations(DPL_SEEDS, 2)]))
        w_cn = float(np.median([rms_distance(c_seeds[s1], c_seeds[s2]) for s1, s2 in combinations(DPL_SEEDS, 2)]))
        w_pooled = (w_base + w_cn) / 2.0
        b_all = float(np.median([rms_distance(b_seeds[s1], c_seeds[s2]) for s1 in DPL_SEEDS for s2 in DPL_SEEDS]))
        excess = b_all - w_pooled

        frac_snow, stratum = snow_meta[b_id]
        ensemble_basin.append({
            "basin_id": b_id,
            "paradigm": "dPL",
            "regime": "dPL",
            "contrast": "Base-CN",
            "within_base": w_base,
            "within_cn": w_cn,
            "within_pooled": w_pooled,
            "between_all": b_all,
            "excess": excess,
            "prop_between_gt_within": bool(b_all > w_pooled),
            "prop_excess_gt_0": bool(excess > 0),
            "frac_snow": frac_snow,
            "snow_stratum": stratum,
        })

    # Summaries for 4B (Ensemble distances, excess, prevalence, and regressions)
    ensemble_summary: List[Dict[str, Any]] = []
    regression_summary: List[Dict[str, Any]] = []
    ens_df = pd.DataFrame(ensemble_basin)

    strata_splits = [
        ("Full531", ens_df),
        ("ExcludeS5", ens_df[ens_df["snow_stratum"] != "S5"]),
        ("S1", ens_df[ens_df["snow_stratum"] == "S1"]),
        ("S2", ens_df[ens_df["snow_stratum"] == "S2"]),
        ("S3", ens_df[ens_df["snow_stratum"] == "S3"]),
        ("S4", ens_df[ens_df["snow_stratum"] == "S4"]),
        ("S5", ens_df[ens_df["snow_stratum"] == "S5"]),
    ]

    metric_cols = ["within_base", "within_cn", "within_pooled", "between_all", "excess"]

    for paradigm in PARADIGMS:
        for st_name, st_frame in strata_splits:
            sub = st_frame[st_frame["paradigm"] == paradigm]
            n_b = len(sub)

            # Numeric distance metrics
            for m in metric_cols:
                vals = sub[m].to_numpy()
                seed_m = BASE_SEED + 3000 + len(ensemble_summary)
                med, cil, cih, q25, q75 = bootstrap_median_ci_cpu(vals, seed=seed_m, draws=draws)
                ensemble_summary.append({
                    "paradigm": paradigm,
                    "contrast": "Base-CN",
                    "stratum": st_name,
                    "n_basins": n_b,
                    "metric": m,
                    "estimate": med,
                    "q25": q25,
                    "q75": q75,
                    "iqr": q75 - q25,
                    "ci_lower": cil,
                    "ci_upper": cih,
                })

            # Proportions: prop_between_gt_within & prop_excess_gt_0
            for p_metric, col in [("prop_between_gt_within", "prop_between_gt_within"), ("prop_excess_gt_0", "prop_excess_gt_0")]:
                prop_vals = sub[col].astype(float).to_numpy()
                seed_p = BASE_SEED + 4000 + len(ensemble_summary)
                mean_p, cil, cih = bootstrap_mean_ci_cpu(prop_vals, seed=seed_p, draws=draws)
                ensemble_summary.append({
                    "paradigm": paradigm,
                    "contrast": "Base-CN",
                    "stratum": st_name,
                    "n_basins": n_b,
                    "metric": p_metric,
                    "estimate": mean_p,
                    "q25": float("nan"),
                    "q75": float("nan"),
                    "iqr": float("nan"),
                    "ci_lower": cil,
                    "ci_upper": cih,
                })

        # Regressions on frac_snow (for Full531 and ExcludeS5)
        for st_name in ["Full531", "ExcludeS5"]:
            mask = np.ones(TOTAL_BASINS, dtype=bool) if st_name == "Full531" else ens_df[ens_df["paradigm"] == paradigm]["snow_stratum"] != "S5"
            sub_p = ens_df[ens_df["paradigm"] == paradigm][mask]
            xs = sub_p["frac_snow"].to_numpy()

            for dep_var in ["within_pooled", "between_all", "excess"]:
                ys = sub_p[dep_var].to_numpy()
                seed_reg = BASE_SEED + 5000 + len(regression_summary)
                res = bootstrap_regression_cpu(xs, ys, seed=seed_reg, draws=draws)
                regression_summary.append({
                    "paradigm": paradigm,
                    "contrast": "Base-CN",
                    "stratum": st_name,
                    "dependent_var": dep_var,
                    "n_basins": len(ys),
                    "slope": res["slope"],
                    "slope_ci_lower": res["slope_ci_low"],
                    "slope_ci_upper": res["slope_ci_high"],
                    "spearman_rho": res["spearman_rho"],
                    "spearman_ci_lower": res["spearman_ci_low"],
                    "spearman_ci_upper": res["spearman_ci_high"],
                })

    # Write output CSV files
    cd_basin_path = out_dir / "r2_canonical_15D_displacement_basin_level.csv"
    pd.DataFrame(canon_d_basin).to_csv(cd_basin_path, index=False, float_format="%.17g")

    cd_summary_path = out_dir / "r2_canonical_15D_displacement_summary.csv"
    pd.DataFrame(canon_d_summary).to_csv(cd_summary_path, index=False, float_format="%.17g")

    ens_basin_path = out_dir / "r2_within_structure_basin_level.csv"
    pd.DataFrame(ensemble_basin).to_csv(ens_basin_path, index=False, float_format="%.17g")

    ens_summary_path = out_dir / "r2_within_structure_summary.csv"
    pd.DataFrame(ensemble_summary).to_csv(ens_summary_path, index=False, float_format="%.17g")

    reg_summary_path = out_dir / "r2_macro_regressions.csv"
    pd.DataFrame(regression_summary).to_csv(reg_summary_path, index=False, float_format="%.17g")

    audit_meta = {
        "status": "PASS",
        "total_basins": TOTAL_BASINS,
        "prevalence_between_gt_within": {
            "IC_Full531": [r for r in ensemble_summary if r["paradigm"] == "IC" and r["stratum"] == "Full531" and r["metric"] == "prop_between_gt_within"][0]["estimate"],
            "dPL_Full531": [r for r in ensemble_summary if r["paradigm"] == "dPL" and r["stratum"] == "Full531" and r["metric"] == "prop_between_gt_within"][0]["estimate"],
        },
        "excess_slope": {
            "IC_Full531": [r for r in regression_summary if r["paradigm"] == "IC" and r["stratum"] == "Full531" and r["dependent_var"] == "excess"][0]["slope"],
            "dPL_Full531": [r for r in regression_summary if r["paradigm"] == "dPL" and r["stratum"] == "Full531" and r["dependent_var"] == "excess"][0]["slope"],
        },
    }

    with (out_dir / "macro_whole_space_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit_meta, f, indent=2)

    return canon_d_basin, canon_d_summary, ensemble_basin, ensemble_summary, audit_meta


if __name__ == "__main__":
    _, cd_sum, _, ens_sum, meta = analyze_macro_whole_space()
    print("Macro whole-space analysis complete:")
    print("  Prevalence (between > within):", meta["prevalence_between_gt_within"])
    print("  Excess OLS slope:", meta["excess_slope"])
