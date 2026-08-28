"""Stage 6: TGD Attribution Control at Macro Level and Paired Delta_beta Bootstrap.

Computes:
  1. Ensemble-level within/between/excess for Base-CN, Base-TGD, and TGD-CN.
  2. OLS regressions of excess ~ frac_snow for Base-CN and Base-TGD across Full531 and ExcludeS5.
  3. Paired basin-bootstrap for Delta_beta = beta_CN - beta_TGD:
       - Sample basin IDs with replacement.
       - Simultaneously refit beta_CN and beta_TGD on the same resampled basins.
       - Compute Delta_beta = beta_CN - beta_TGD per draw.
       - Obtain 95% percentile confidence interval.
"""
from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from r2_config import (
    BASE_SEED,
    DEFAULT_DRAWS,
    DPL_SEEDS,
    PARADIGMS,
    RESULTS_DIR,
    STRATA,
    STRATA_COUNTS,
    STRUCTURES,
    TOTAL_BASINS,
)
from parameter_ledger import build_raw_parameter_ledger, load_canonical_snow_metadata
from shared_parameter_specs import SHARED_15_PARAMETERS


def rms_dist(z1: np.ndarray, z2: np.ndarray) -> float:
    return float(np.sqrt(np.mean((z1 - z2) ** 2)))


def analyze_tgd_attribution_control(
    ledger_rows: List[Dict[str, Any]] | None = None,
    output_dir: Path | None = None,
    draws: int = DEFAULT_DRAWS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Perform macro TGD attribution control and paired Delta_beta bootstrap."""
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if ledger_rows is None:
        ledger_rows, _ = build_raw_parameter_ledger(output_dir=out_dir)

    snow_meta = load_canonical_snow_metadata()
    basins = sorted(snow_meta.keys())

    ledger_df = pd.DataFrame(ledger_rows)
    contrasts = [("Base", "CN"), ("Base", "TGD"), ("TGD", "CN")]

    tgd_basin_rows: List[Dict[str, Any]] = []

    # 1. Process IC (10 restarts per structure)
    ic_df = ledger_df[ledger_df["paradigm"] == "IC"]
    ic_dict: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {b: {s: {} for s in STRUCTURES} for b in basins}

    for (struct, b_id, start_idx), g in ic_df.groupby(["structure", "basin_id", "start_or_seed"]):
        z_vec = g.set_index("parameter").loc[list(SHARED_15_PARAMETERS)]["normalized_value"].to_numpy(dtype=np.float64)
        ic_dict[b_id][struct][int(start_idx)] = z_vec

    for b_id in basins:
        frac_snow, stratum = snow_meta[b_id]
        for a_name, b_name in contrasts:
            a_starts = ic_dict[b_id][a_name]
            b_starts = ic_dict[b_id][b_name]
            if len(a_starts) != 10 or len(b_starts) != 10:
                raise RuntimeError(f"IC basin {b_id} incomplete starts for {a_name} or {b_name}")

            w_a = float(np.median([rms_dist(a_starts[s1], a_starts[s2]) for s1, s2 in combinations(range(10), 2)]))
            w_b = float(np.median([rms_dist(b_starts[s1], b_starts[s2]) for s1, s2 in combinations(range(10), 2)]))
            w_pool = (w_a + w_b) / 2.0
            b_all = float(np.median([rms_dist(a_starts[s1], b_starts[s2]) for s1 in range(10) for s2 in range(10)]))
            excess = b_all - w_pool

            tgd_basin_rows.append({
                "basin_id": b_id,
                "paradigm": "IC",
                "regime": "IC",
                "contrast": f"{a_name}-{b_name}",
                "within_a": w_a,
                "within_b": w_b,
                "within_pooled": w_pool,
                "between_all": b_all,
                "excess": excess,
                "prop_between_gt_within": bool(b_all > w_pool),
                "prop_excess_gt_0": bool(excess > 0),
                "frac_snow": frac_snow,
                "snow_stratum": stratum,
                "snow_regime": stratum,
            })

    # 2. Process dPL (3 seeds per structure)
    dpl_df = ledger_df[ledger_df["paradigm"] == "dPL"]
    dpl_dict: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {b: {s: {} for s in STRUCTURES} for b in basins}

    for (struct, b_id, seed_val), g in dpl_df.groupby(["structure", "basin_id", "start_or_seed"]):
        z_vec = g.set_index("parameter").loc[list(SHARED_15_PARAMETERS)]["normalized_value"].to_numpy(dtype=np.float64)
        dpl_dict[b_id][struct][int(seed_val)] = z_vec

    for b_id in basins:
        frac_snow, stratum = snow_meta[b_id]
        for a_name, b_name in contrasts:
            a_seeds = dpl_dict[b_id][a_name]
            b_seeds = dpl_dict[b_id][b_name]
            if len(a_seeds) != 3 or len(b_seeds) != 3:
                raise RuntimeError(f"dPL basin {b_id} incomplete seeds for {a_name} or {b_name}")

            w_a = float(np.median([rms_dist(a_seeds[s1], a_seeds[s2]) for s1, s2 in combinations(DPL_SEEDS, 2)]))
            w_b = float(np.median([rms_dist(b_seeds[s1], b_seeds[s2]) for s1, s2 in combinations(DPL_SEEDS, 2)]))
            w_pool = (w_a + w_b) / 2.0
            b_all = float(np.median([rms_dist(a_seeds[s1], b_seeds[s2]) for s1 in DPL_SEEDS for s2 in DPL_SEEDS]))
            excess = b_all - w_pool

            tgd_basin_rows.append({
                "basin_id": b_id,
                "paradigm": "dPL",
                "regime": "dPL",
                "contrast": f"{a_name}-{b_name}",
                "within_a": w_a,
                "within_b": w_b,
                "within_pooled": w_pool,
                "between_all": b_all,
                "excess": excess,
                "prop_between_gt_within": bool(b_all > w_pool),
                "prop_excess_gt_0": bool(excess > 0),
                "frac_snow": frac_snow,
                "snow_stratum": stratum,
                "snow_regime": stratum,
            })

    tgd_df = pd.DataFrame(tgd_basin_rows)

    # 3. Specificity Summaries across Strata
    specificity_summaries: List[Dict[str, Any]] = []
    strata_splits = [
        ("Full531", tgd_df),
        ("ExcludeS5", tgd_df[tgd_df["snow_stratum"] != "S5"]),
        ("S1", tgd_df[tgd_df["snow_stratum"] == "S1"]),
        ("S2", tgd_df[tgd_df["snow_stratum"] == "S2"]),
        ("S3", tgd_df[tgd_df["snow_stratum"] == "S3"]),
        ("S4", tgd_df[tgd_df["snow_stratum"] == "S4"]),
        ("S5", tgd_df[tgd_df["snow_stratum"] == "S5"]),
    ]

    for paradigm in PARADIGMS:
        for a_name, b_name in contrasts:
            c_label = f"{a_name}-{b_name}"
            for st_name, st_frame in strata_splits:
                sub = st_frame[(st_frame["paradigm"] == paradigm) & (st_frame["contrast"] == c_label)]
                n_b = len(sub)

                for m in ["within_a", "within_b", "within_pooled", "between_all", "excess"]:
                    vals = sub[m].to_numpy()
                    seed_m = BASE_SEED + 11000 + len(specificity_summaries)
                    rng = np.random.default_rng(seed_m)
                    pt = float(np.median(vals))
                    b_idx = rng.integers(0, len(vals), size=(draws, len(vals)))
                    b_meds = np.median(vals[b_idx], axis=1)
                    ci_l = float(np.quantile(b_meds, 0.025))
                    ci_h = float(np.quantile(b_meds, 0.975))

                    specificity_summaries.append({
                        "paradigm": paradigm,
                        "contrast": c_label,
                        "stratum": st_name,
                        "n_basins": n_b,
                        "metric": m,
                        "median": pt,
                        "ci_lower": ci_l,
                        "ci_upper": ci_h,
                    })

                for p_metric, col in [("prop_between_gt_within", "prop_between_gt_within"), ("prop_excess_gt_0", "prop_excess_gt_0")]:
                    prop_vals = sub[col].astype(float).to_numpy()
                    seed_p = BASE_SEED + 12000 + len(specificity_summaries)
                    rng = np.random.default_rng(seed_p)
                    pt_mean = float(np.mean(prop_vals))
                    b_idx = rng.integers(0, len(prop_vals), size=(draws, len(prop_vals)))
                    b_means = np.mean(prop_vals[b_idx], axis=1)
                    ci_l = float(np.quantile(b_means, 0.025))
                    ci_h = float(np.quantile(b_means, 0.975))

                    specificity_summaries.append({
                        "paradigm": paradigm,
                        "contrast": c_label,
                        "stratum": st_name,
                        "n_basins": n_b,
                        "metric": p_metric,
                        "median": pt_mean,
                        "ci_lower": ci_l,
                        "ci_upper": ci_h,
                    })

    # 4. Regressions on frac_snow for within_pooled, between_all, excess
    regression_rows: List[Dict[str, Any]] = []
    for paradigm in PARADIGMS:
        for a_name, b_name in contrasts:
            c_label = f"{a_name}-{b_name}"
            for st_name in ["Full531", "ExcludeS5"]:
                mask = np.ones(TOTAL_BASINS, dtype=bool) if st_name == "Full531" else tgd_df[(tgd_df["paradigm"] == paradigm) & (tgd_df["contrast"] == c_label)]["snow_stratum"] != "S5"
                sub_c = tgd_df[(tgd_df["paradigm"] == paradigm) & (tgd_df["contrast"] == c_label)][mask]
                xs = sub_c["frac_snow"].to_numpy()

                for dep_var in ["within_pooled", "between_all", "excess"]:
                    ys = sub_c[dep_var].to_numpy()
                    slope_pt = float(np.polyfit(xs, ys, 1)[0])
                    rho_pt = float(spearmanr(xs, ys)[0])

                    seed_r = BASE_SEED + 13000 + len(regression_rows)
                    rng = np.random.default_rng(seed_r)
                    b_idx = rng.integers(0, len(xs), size=(draws, len(xs)))
                    b_slopes = np.array([np.polyfit(xs[b_idx[i]], ys[b_idx[i]], 1)[0] for i in range(draws)])
                    b_rhos = np.array([spearmanr(xs[b_idx[i]], ys[b_idx[i]])[0] for i in range(draws)])

                    regression_rows.append({
                        "paradigm": paradigm,
                        "contrast": c_label,
                        "stratum": st_name,
                        "dependent_var": dep_var,
                        "n_basins": len(ys),
                        "slope": slope_pt,
                        "slope_ci_lower": float(np.quantile(b_slopes, 0.025)),
                        "slope_ci_upper": float(np.quantile(b_slopes, 0.975)),
                        "spearman_rho": rho_pt,
                        "spearman_ci_lower": float(np.quantile(b_rhos, 0.025)),
                        "spearman_ci_upper": float(np.quantile(b_rhos, 0.975)),
                    })

    # 5. Paired Delta_beta Bootstrap: Delta_beta = beta(Base-CN) - beta(Base-TGD)
    slope_diff_rows: List[Dict[str, Any]] = []
    for paradigm in PARADIGMS:
        p_df = tgd_df[tgd_df["paradigm"] == paradigm]
        cn_sub = p_df[p_df["contrast"] == "Base-CN"].set_index("basin_id").loc[basins]
        tgd_sub = p_df[p_df["contrast"] == "Base-TGD"].set_index("basin_id").loc[basins]

        xs_all = cn_sub["frac_snow"].to_numpy()
        y_cn_all = cn_sub["excess"].to_numpy()
        y_tgd_all = tgd_sub["excess"].to_numpy()

        for st_name in ["Full531", "ExcludeS5"]:
            mask = np.ones(TOTAL_BASINS, dtype=bool) if st_name == "Full531" else cn_sub["snow_stratum"] != "S5"
            xs = xs_all[mask]
            y_cn = y_cn_all[mask]
            y_tgd = y_tgd_all[mask]
            n_b = len(xs)

            beta_cn = float(np.polyfit(xs, y_cn, 1)[0])
            beta_tgd = float(np.polyfit(xs, y_tgd, 1)[0])
            delta_beta = beta_cn - beta_tgd

            seed_db = BASE_SEED + 14000 + len(slope_diff_rows)
            rng = np.random.default_rng(seed_db)
            b_idx = rng.integers(0, n_b, size=(draws, n_b))

            b_diffs = np.zeros(draws)
            b_cn_slopes = np.zeros(draws)
            b_tgd_slopes = np.zeros(draws)

            for i in range(draws):
                bx = xs[b_idx[i]]
                b_ycn = y_cn[b_idx[i]]
                b_ytgd = y_tgd[b_idx[i]]
                s_cn = np.polyfit(bx, b_ycn, 1)[0]
                s_tgd = np.polyfit(bx, b_ytgd, 1)[0]
                b_cn_slopes[i] = s_cn
                b_tgd_slopes[i] = s_tgd
                b_diffs[i] = s_cn - s_tgd

            slope_diff_rows.append({
                "paradigm": paradigm,
                "stratum": st_name,
                "n_basins": n_b,
                "beta_Base_CN": beta_cn,
                "beta_Base_CN_ci_lower": float(np.quantile(b_cn_slopes, 0.025)),
                "beta_Base_CN_ci_upper": float(np.quantile(b_cn_slopes, 0.975)),
                "beta_Base_TGD": beta_tgd,
                "beta_Base_TGD_ci_lower": float(np.quantile(b_tgd_slopes, 0.025)),
                "beta_Base_TGD_ci_upper": float(np.quantile(b_tgd_slopes, 0.975)),
                "delta_beta": delta_beta,
                "delta_beta_ci_lower": float(np.quantile(b_diffs, 0.025)),
                "delta_beta_ci_upper": float(np.quantile(b_diffs, 0.975)),
                "paired_bootstrap": True,
            })

    # Write output CSV files
    pd.DataFrame(tgd_basin_rows).to_csv(out_dir / "r2_tgd2_specificity_basin_level.csv", index=False, float_format="%.17g")
    pd.DataFrame(specificity_summaries).to_csv(out_dir / "r2_tgd2_specificity_summary.csv", index=False, float_format="%.17g")
    pd.DataFrame(regression_rows).to_csv(out_dir / "r2_tgd2_specificity_regressions.csv", index=False, float_format="%.17g")
    pd.DataFrame(slope_diff_rows).to_csv(out_dir / "r2_tgd2_slope_difference_summary.csv", index=False, float_format="%.17g")

    audit_meta = {
        "status": "PASS",
        "total_basin_contrast_rows": len(tgd_basin_rows),
        "total_specificity_summary_rows": len(specificity_summaries),
        "total_regression_rows": len(regression_rows),
        "slope_difference_rows": len(slope_diff_rows),
        "delta_beta_results": {
            f"{r['paradigm']}_{r['stratum']}": f"{r['delta_beta']:+.3f} [{r['delta_beta_ci_lower']:+.3f}, {r['delta_beta_ci_upper']:+.3f}]"
            for r in slope_diff_rows
        },
    }

    with (out_dir / "tgd_attribution_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit_meta, f, indent=2)

    return tgd_basin_rows, specificity_summaries, regression_rows, slope_diff_rows, audit_meta


if __name__ == "__main__":
    b_rows, s_rows, r_rows, d_rows, meta = analyze_tgd_attribution_control()
    print("TGD attribution control complete:")
    print("Delta_beta results (Paired bootstrap):")
    for k, v in meta["delta_beta_results"].items():
        print(f"  {k}: {v}")
