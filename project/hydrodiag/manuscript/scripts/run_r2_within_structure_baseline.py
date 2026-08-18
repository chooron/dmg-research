#!/usr/bin/env python3
"""Reproducible R2 Within-Structure Baseline & Directional Re-analysis.

Establishes the within-structure parameter baseline (restart/seed variability)
for Base and CN, compares against between-structure separation distance,
and evaluates whether 15D parameter separation exceeds restart/seed dispersion.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parents[2]
MANUSCRIPT = PROJECT / "manuscript"
DATA = PROJECT.parents[1] / "data"
RESULTS_R2 = MANUSCRIPT / "results" / "R2"

BASIN_FILE = DATA / "531sub_id.txt"
BOUNDS_FILE = (
    MANUSCRIPT / "supplement" / "results" / "s2_parameter_bounds_from_code.csv"
)
SNOW_FILE = MANUSCRIPT / "results" / "R1" / "r1_snow_attributes.csv"

COMMON_XAJ = [
    "xaj_k",
    "xaj_b",
    "xaj_im",
    "xaj_um",
    "xaj_lm",
    "xaj_dm",
    "xaj_c",
    "xaj_sm",
    "xaj_ex",
    "xaj_ki",
    "xaj_kg",
    "xaj_ci",
    "xaj_cg",
    "xaj_a",
    "xaj_theta",
]


def assign_regime(f: float) -> str:
    if f < 0.05:
        return "S1"
    elif f < 0.15:
        return "S2"
    elif f < 0.30:
        return "S3"
    elif f < 0.50:
        return "S4"
    else:
        return "S5"


def rms_dist(z1: np.ndarray, z2: np.ndarray) -> float:
    return float(np.sqrt(np.mean((z1 - z2) ** 2)))


def bootstrap_stat(
    values: np.ndarray, stat_func, n_boot: int = 10000, seed: int = 20260730
) -> tuple[float, float, float]:
    point_val = float(stat_func(values))
    n = len(values)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_boot, n))
    boot_vals = np.array([stat_func(values[boot_idx[i]]) for i in range(n_boot)])
    ci_low = float(np.percentile(boot_vals, 2.5))
    ci_high = float(np.percentile(boot_vals, 97.5))
    return point_val, ci_low, ci_high


def bootstrap_regression(
    x: np.ndarray, y: np.ndarray, n_boot: int = 10000, seed: int = 20260730
) -> dict:
    slope_point = float(np.polyfit(x, y, 1)[0])
    rho_point, _ = spearmanr(x, y)

    n = len(x)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_boot, n))

    boot_slopes = np.zeros(n_boot)
    boot_rhos = np.zeros(n_boot)
    for i in range(n_boot):
        bx, by = x[boot_idx[i]], y[boot_idx[i]]
        boot_slopes[i] = np.polyfit(bx, by, 1)[0]
        r, _ = spearmanr(bx, by)
        boot_rhos[i] = r

    s_low, s_high = (
        float(np.percentile(boot_slopes, 2.5)),
        float(np.percentile(boot_slopes, 97.5)),
    )
    r_low, r_high = (
        float(np.percentile(boot_rhos, 2.5)),
        float(np.percentile(boot_rhos, 97.5)),
    )

    return {
        "slope": slope_point,
        "slope_ci_lower": s_low,
        "slope_ci_upper": s_high,
        "spearman_rho": float(rho_point),
        "spearman_rho_ci_lower": r_low,
        "spearman_rho_ci_upper": r_high,
    }


def main() -> None:
    print("Starting R2 within-structure baseline analysis...")
    RESULTS_R2.mkdir(parents=True, exist_ok=True)

    # 1. Load Basins, Bounds, Snow
    basins = [str(x).zfill(8) for x in json.loads(BASIN_FILE.read_text().strip())]
    if len(basins) != 531:
        raise ValueError(f"Expected 531 basins, found {len(basins)}")

    b_df = pd.read_csv(BOUNDS_FILE)
    bounds = (
        b_df[b_df["active_model_key"] == "XAJ"]
        .drop_duplicates("code_name")
        .set_index("code_name")
    )
    lowers = np.array([bounds.loc[n, "lower_bound"] for n in COMMON_XAJ], dtype=float)
    uppers = np.array([bounds.loc[n, "upper_bound"] for n in COMMON_XAJ], dtype=float)

    snow = pd.read_csv(SNOW_FILE)
    snow["basin_id"] = snow["basin_id"].astype(str).str.zfill(8)
    snow["snow_regime"] = snow["frac_snow"].apply(assign_regime)
    snow_map = snow.set_index("basin_id")

    # 2. Process IC Data (10 restarts per structure)
    ic_base_raw = (
        PROJECT / "results" / "xaj_base_cmaes_531_batched_paired_v2" / "raw" / "xaj"
    )
    ic_cn_raw = (
        PROJECT / "results" / "xaj_cn_cmaes_531_batched_paired_v2" / "raw" / "xaj_cn"
    )

    ic_z = {b: {"Base": {}, "CN": {}} for b in basins}
    for p in ic_base_raw.glob("*.json"):
        d = json.loads(p.read_text())
        b = str(d.get("basin_id", "")).zfill(8)
        if b not in ic_z:
            continue
        s = int(d.get("start"))
        train_kge = float(d.get("train_metrics", {}).get("kge", np.nan))
        p_dict = dict(zip(d["parameter_names"], d["parameters"]))
        p_vals = np.array([p_dict[n] for n in COMMON_XAJ], dtype=float)
        z = (p_vals - lowers) / (uppers - lowers)
        ic_z[b]["Base"][s] = (z, train_kge)

    for p in ic_cn_raw.glob("*.json"):
        d = json.loads(p.read_text())
        b = str(d.get("basin_id", "")).zfill(8)
        if b not in ic_z:
            continue
        s = int(d.get("start"))
        train_kge = float(d.get("train_metrics", {}).get("kge", np.nan))
        p_dict = dict(zip(d["parameter_names"], d["parameters"]))
        p_vals = np.array([p_dict[n] for n in COMMON_XAJ], dtype=float)
        z = (p_vals - lowers) / (uppers - lowers)
        ic_z[b]["CN"][s] = (z, train_kge)

    ic_basin_rows = []
    ic_kge_rows = []

    for b in basins:
        base_dict = ic_z[b]["Base"]
        cn_dict = ic_z[b]["CN"]
        b_starts = sorted(base_dict.keys())
        c_starts = sorted(cn_dict.keys())

        if len(b_starts) != 10 or len(c_starts) != 10:
            raise ValueError(
                f"Basin {b} IC restarts incomplete: Base={len(b_starts)}, CN={len(c_starts)}"
            )

        w_base_dists = [
            rms_dist(base_dict[s1][0], base_dict[s2][0])
            for s1, s2 in combinations(b_starts, 2)
        ]
        w_cn_dists = [
            rms_dist(cn_dict[s1][0], cn_dict[s2][0])
            for s1, s2 in combinations(c_starts, 2)
        ]

        w_base = float(np.median(w_base_dists))
        w_cn = float(np.median(w_cn_dists))
        w_pooled = (w_base + w_cn) / 2.0

        b_all_dists = [
            rms_dist(base_dict[s1][0], cn_dict[s2][0])
            for s1 in b_starts
            for s2 in c_starts
        ]
        b_all = float(np.median(b_all_dists))

        excess = b_all - w_pooled
        ratio = b_all / w_pooled if w_pooled > 1e-12 else np.nan

        matched_dists = [
            rms_dist(base_dict[s][0], cn_dict[s][0]) for s in b_starts if s in cn_dict
        ]
        matched_median = float(np.median(matched_dists))

        best_s_base = max(b_starts, key=lambda s: base_dict[s][1])
        best_s_cn = max(c_starts, key=lambda s: cn_dict[s][1])
        canonical_best_d_rms = rms_dist(
            base_dict[best_s_base][0], cn_dict[best_s_cn][0]
        )

        ic_basin_rows.append(
            {
                "basin_id": b,
                "paradigm": "IC",
                "within_base": w_base,
                "within_cn": w_cn,
                "within_pooled": w_pooled,
                "between_all": b_all,
                "excess": excess,
                "ratio": ratio,
                "matched_d_rms": matched_median,
                "canonical_best_d_rms": canonical_best_d_rms,
            }
        )

        # KGE Audit & Top 3 / Top 5 sensitivity
        b_kges = np.array([base_dict[s][1] for s in b_starts])
        c_kges = np.array([cn_dict[s][1] for s in c_starts])

        b_best_minus_med = float(np.max(b_kges) - np.median(b_kges))
        b_iqr = float(np.percentile(b_kges, 75) - np.percentile(b_kges, 25))
        c_best_minus_med = float(np.max(c_kges) - np.median(c_kges))
        c_iqr = float(np.percentile(c_kges, 75) - np.percentile(c_kges, 25))

        top5_b = sorted(b_starts, key=lambda s: base_dict[s][1], reverse=True)[:5]
        top5_c = sorted(c_starts, key=lambda s: cn_dict[s][1], reverse=True)[:5]
        w5_b = float(
            np.median(
                [
                    rms_dist(base_dict[s1][0], base_dict[s2][0])
                    for s1, s2 in combinations(top5_b, 2)
                ]
            )
        )
        w5_c = float(
            np.median(
                [
                    rms_dist(cn_dict[s1][0], cn_dict[s2][0])
                    for s1, s2 in combinations(top5_c, 2)
                ]
            )
        )
        w5_pool = (w5_b + w5_c) / 2.0
        b5_all = float(
            np.median(
                [
                    rms_dist(base_dict[s1][0], cn_dict[s2][0])
                    for s1 in top5_b
                    for s2 in top5_c
                ]
            )
        )

        top3_b = sorted(b_starts, key=lambda s: base_dict[s][1], reverse=True)[:3]
        top3_c = sorted(c_starts, key=lambda s: cn_dict[s][1], reverse=True)[:3]
        w3_b = float(
            np.median(
                [
                    rms_dist(base_dict[s1][0], base_dict[s2][0])
                    for s1, s2 in combinations(top3_b, 2)
                ]
            )
        )
        w3_c = float(
            np.median(
                [
                    rms_dist(cn_dict[s1][0], cn_dict[s2][0])
                    for s1, s2 in combinations(top3_c, 2)
                ]
            )
        )
        w3_pool = (w3_b + w3_c) / 2.0
        b3_all = float(
            np.median(
                [
                    rms_dist(base_dict[s1][0], cn_dict[s2][0])
                    for s1 in top3_b
                    for s2 in top3_c
                ]
            )
        )

        ic_kge_rows.append(
            {
                "basin_id": b,
                "base_best_minus_median_kge": b_best_minus_med,
                "base_kge_iqr": b_iqr,
                "cn_best_minus_median_kge": c_best_minus_med,
                "cn_kge_iqr": c_iqr,
                "within_pooled_all10": w_pooled,
                "within_pooled_top5": w5_pool,
                "within_pooled_top3": w3_pool,
                "between_all_all10": b_all,
                "between_all_top5": b5_all,
                "between_all_top3": b3_all,
                "excess_all10": excess,
                "excess_top5": b5_all - w5_pool,
                "excess_top3": b3_all - w3_pool,
            }
        )

    # 3. Process dPL Data (3 seeds per structure)
    df_seed = pd.read_csv(RESULTS_R2 / "r2_parameter_values_seed_level.csv")
    df_seed["basin_id"] = df_seed["basin_id"].astype(str).str.zfill(8)
    dpl_seed = df_seed[
        (df_seed["paradigm"] == "dPL") & (df_seed["structure"].isin(["Base", "CN"]))
    ].copy()

    dpl_z = {b: {"Base": {}, "CN": {}} for b in basins}
    for b, sub in dpl_seed.groupby("basin_id"):
        for (struct, s), s_sub in sub.groupby(["structure", "seed"]):
            s_sub = s_sub.set_index("parameter").loc[COMMON_XAJ]
            dpl_z[b][struct][str(s)] = s_sub["z"].to_numpy(dtype=float)

    dpl_basin_rows = []
    for b in basins:
        b_dict = dpl_z[b]["Base"]
        c_dict = dpl_z[b]["CN"]
        b_seeds = sorted(b_dict.keys())
        c_seeds = sorted(c_dict.keys())

        if len(b_seeds) != 3 or len(c_seeds) != 3:
            raise ValueError(
                f"Basin {b} dPL seeds incomplete: Base={len(b_seeds)}, CN={len(c_seeds)}"
            )

        w_base = float(
            np.median(
                [
                    rms_dist(b_dict[s1], b_dict[s2])
                    for s1, s2 in combinations(b_seeds, 2)
                ]
            )
        )
        w_cn = float(
            np.median(
                [
                    rms_dist(c_dict[s1], c_dict[s2])
                    for s1, s2 in combinations(c_seeds, 2)
                ]
            )
        )
        w_pooled = (w_base + w_cn) / 2.0

        b_all = float(
            np.median(
                [rms_dist(b_dict[s1], c_dict[s2]) for s1 in b_seeds for s2 in c_seeds]
            )
        )
        excess = b_all - w_pooled
        ratio = b_all / w_pooled if w_pooled > 1e-12 else np.nan

        matched_dists = [rms_dist(b_dict[s], c_dict[s]) for s in b_seeds if s in c_dict]
        matched_median = float(np.median(matched_dists))

        z_base_canon = np.median(np.array([b_dict[s] for s in b_seeds]), axis=0)
        z_cn_canon = np.median(np.array([c_dict[s] for s in c_seeds]), axis=0)
        canonical_best_d_rms = rms_dist(z_base_canon, z_cn_canon)

        dpl_basin_rows.append(
            {
                "basin_id": b,
                "paradigm": "dPL",
                "within_base": w_base,
                "within_cn": w_cn,
                "within_pooled": w_pooled,
                "between_all": b_all,
                "excess": excess,
                "ratio": ratio,
                "matched_d_rms": matched_median,
                "canonical_best_d_rms": canonical_best_d_rms,
            }
        )

    df_ic_basin = pd.DataFrame(ic_basin_rows).merge(snow, on="basin_id")
    df_dpl_basin = pd.DataFrame(dpl_basin_rows).merge(snow, on="basin_id")

    df_basin_all = pd.concat([df_ic_basin, df_dpl_basin], ignore_index=True)
    df_basin_all.to_csv(
        RESULTS_R2 / "r2_within_structure_basin_level.csv",
        index=False,
        float_format="%.17g",
    )

    df_ic_kge = pd.DataFrame(ic_kge_rows).merge(snow, on="basin_id")
    df_ic_kge.to_csv(
        RESULTS_R2 / "r2_ic_restart_quality_audit.csv",
        index=False,
        float_format="%.17g",
    )

    # 4. Compute Stratum Summaries & Bootstrap CIs
    summary_rows = []
    strata_list = [
        ("Full531", df_basin_all),
        ("S1", df_basin_all[df_basin_all["snow_regime"] == "S1"]),
        ("S2", df_basin_all[df_basin_all["snow_regime"] == "S2"]),
        ("S3", df_basin_all[df_basin_all["snow_regime"] == "S3"]),
        ("S4", df_basin_all[df_basin_all["snow_regime"] == "S4"]),
        ("S5", df_basin_all[df_basin_all["snow_regime"] == "S5"]),
        ("ExcludeS5", df_basin_all[df_basin_all["snow_regime"] != "S5"]),
    ]

    metrics = [
        "within_base",
        "within_cn",
        "within_pooled",
        "between_all",
        "excess",
        "ratio",
        "prop_excess_gt_0",
        "prop_between_gt_within",
    ]

    for paradigm in ["IC", "dPL"]:
        for st_name, st_df in strata_list:
            sub = st_df[st_df["paradigm"] == paradigm]
            n_b = len(sub)

            for m in [
                "within_base",
                "within_cn",
                "within_pooled",
                "between_all",
                "excess",
                "ratio",
            ]:
                vals = sub[m].dropna().to_numpy()
                pt, low, high = bootstrap_stat(
                    vals, np.median, n_boot=10000, seed=20260730
                )
                summary_rows.append(
                    {
                        "paradigm": paradigm,
                        "stratum": st_name,
                        "n_basins": n_b,
                        "metric": m,
                        "median": pt,
                        "ci_lower": low,
                        "ci_upper": high,
                    }
                )

            # Proportions
            exc_vals = (sub["excess"] > 0).astype(float).to_numpy()
            pt, low, high = bootstrap_stat(
                exc_vals, np.mean, n_boot=10000, seed=20260730
            )
            summary_rows.append(
                {
                    "paradigm": paradigm,
                    "stratum": st_name,
                    "n_basins": n_b,
                    "metric": "prop_excess_gt_0",
                    "median": pt,
                    "ci_lower": low,
                    "ci_upper": high,
                }
            )

            bet_vals = (
                (sub["between_all"] > sub["within_pooled"]).astype(float).to_numpy()
            )
            pt, low, high = bootstrap_stat(
                bet_vals, np.mean, n_boot=10000, seed=20260730
            )
            summary_rows.append(
                {
                    "paradigm": paradigm,
                    "stratum": st_name,
                    "n_basins": n_b,
                    "metric": "prop_between_gt_within",
                    "median": pt,
                    "ci_lower": low,
                    "ci_upper": high,
                }
            )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(
        RESULTS_R2 / "r2_within_structure_summary.csv",
        index=False,
        float_format="%.17g",
    )

    # 5. Compute Snow Regressions & Bootstrap CIs
    reg_rows = []
    reg_strata = [
        ("Full531", df_basin_all),
        ("ExcludeS5", df_basin_all[df_basin_all["snow_regime"] != "S5"]),
    ]

    for paradigm in ["IC", "dPL"]:
        for st_name, st_df in reg_strata:
            sub = st_df[st_df["paradigm"] == paradigm]
            x = sub["frac_snow"].to_numpy()
            for dep in ["within_pooled", "between_all", "excess"]:
                y = sub[dep].to_numpy()
                res = bootstrap_regression(x, y, n_boot=10000, seed=20260730)
                reg_rows.append(
                    {
                        "paradigm": paradigm,
                        "stratum": st_name,
                        "dependent_var": dep,
                        **res,
                    }
                )

    df_reg = pd.DataFrame(reg_rows)
    df_reg.to_csv(
        RESULTS_R2 / "r2_within_structure_regressions.csv",
        index=False,
        float_format="%.17g",
    )

    # 6. Boundary Contribution Audit
    df_canon = pd.read_csv(RESULTS_R2 / "r2_parameter_values_canonical.csv")
    df_canon["basin_id"] = df_canon["basin_id"].astype(str).str.zfill(8)

    thresholds = [0.01, 0.02, 0.05]
    bnd_rows = []

    for paradigm in ["IC", "dPL"]:
        p_base_z = df_canon[
            (df_canon["structure"] == "Base") & (df_canon["paradigm"] == paradigm)
        ].pivot_table(index="basin_id", columns="parameter", values="z")[COMMON_XAJ]
        p_cn_z = df_canon[
            (df_canon["structure"] == "CN") & (df_canon["paradigm"] == paradigm)
        ].pivot_table(index="basin_id", columns="parameter", values="z")[COMMON_XAJ]

        diff2 = (p_base_z - p_cn_z) ** 2
        sum_diff2 = diff2.sum(axis=1)

        for eps in thresholds:
            is_bnd = (
                (p_base_z <= eps)
                | (p_base_z >= 1.0 - eps)
                | (p_cn_z <= eps)
                | (p_cn_z >= 1.0 - eps)
            )
            bnd_diff2 = (diff2 * is_bnd).sum(axis=1)
            frac = np.where(sum_diff2 > 1e-12, bnd_diff2 / sum_diff2, 0.0)

            df_frac = pd.DataFrame({"basin_id": p_base_z.index, "frac": frac}).merge(
                snow, on="basin_id"
            )

            bnd_strata = [
                ("Full531", df_frac),
                ("S1", df_frac[df_frac["snow_regime"] == "S1"]),
                ("S2", df_frac[df_frac["snow_regime"] == "S2"]),
                ("S3", df_frac[df_frac["snow_regime"] == "S3"]),
                ("S4", df_frac[df_frac["snow_regime"] == "S4"]),
                ("S5", df_frac[df_frac["snow_regime"] == "S5"]),
            ]

            for st_name, st_sub in bnd_strata:
                med_pct = float(st_sub["frac"].median() * 100.0)
                bnd_rows.append(
                    {
                        "paradigm": paradigm,
                        "threshold": eps,
                        "stratum": st_name,
                        "median_boundary_contribution_percent": med_pct,
                    }
                )

    df_bnd = pd.DataFrame(bnd_rows)
    df_bnd.to_csv(
        RESULTS_R2 / "r2_boundary_contribution_audit.csv",
        index=False,
        float_format="%.17g",
    )

    print("Finished successfully. All CSV files written to manuscript/results/R2/")


if __name__ == "__main__":
    main()
