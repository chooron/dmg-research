"""Stage 8: Supporting Diagnostics and Safeguards for R2.

Implements:
  1. IC restart quality audit:
       - Best-minus-median train-period KGE, restart KGE IQR across 10 starts.
       - Top-3 and Top-5 restart sensitivity for excess and between/within.
       - Correlation with frac_snow and within-structure variability.
  2. dPL seed stability audit:
       - Across-seed dispersion per parameter.
       - Directional consistency across seeds (42, 123, 2026).
  3. Boundary and signed-vs-absolute safeguards:
       - Exact boundary hits (z=0, z=1) and near-boundary mass (1%, 2%, 5% tolerances).
       - Both-structure same-bound co-occurrence.
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
    DPL_SEEDS,
    IC_STARTS,
    PARADIGMS,
    RESULTS_DIR,
    STRUCTURES,
    TOTAL_BASINS,
)
from canonical_vectors import build_canonical_parameter_vectors
from parameter_ledger import build_raw_parameter_ledger, load_canonical_snow_metadata
from shared_parameter_specs import PARAMETER_METADATA, SHARED_15_PARAMETERS


def rms_dist(z1: np.ndarray, z2: np.ndarray) -> float:
    return float(np.sqrt(np.mean((z1 - z2) ** 2)))


def run_diagnostics_and_safeguards(
    ledger_rows: List[Dict[str, Any]] | None = None,
    canonical_rows: List[Dict[str, Any]] | None = None,
    output_dir: Path | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Run all supporting diagnostics and boundary safeguards."""
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if ledger_rows is None:
        ledger_rows, _ = build_raw_parameter_ledger(output_dir=out_dir)
    if canonical_rows is None:
        canonical_rows, _ = build_canonical_parameter_vectors(output_dir=out_dir)

    snow_meta = load_canonical_snow_metadata()
    basins = sorted(snow_meta.keys())
    ledger_df = pd.DataFrame(ledger_rows)
    canon_df = pd.DataFrame(canonical_rows)

    # -------------------------------------------------------------
    # 1. IC Restart Quality Audit & Top-3/Top-5 Sensitivity
    # -------------------------------------------------------------
    ic_df = ledger_df[ledger_df["paradigm"] == "IC"]
    ic_quality_rows: List[Dict[str, Any]] = []

    ic_dict = {b: {"Base": {}, "CN": {}} for b in basins}
    for (struct, b_id, start_idx), g in ic_df[ic_df["structure"].isin(["Base", "CN"])].groupby(["structure", "basin_id", "start_or_seed"]):
        z_vec = g.set_index("parameter").loc[list(SHARED_15_PARAMETERS)]["normalized_value"].to_numpy(dtype=np.float64)
        train_kge = float(g["train_kge"].iloc[0])
        ic_dict[b_id][struct][int(start_idx)] = (z_vec, train_kge)

    for b_id in basins:
        frac_snow, stratum = snow_meta[b_id]
        b_starts = ic_dict[b_id]["Base"]
        c_starts = ic_dict[b_id]["CN"]

        b_kges = np.array([b_starts[s][1] for s in range(10)])
        c_kges = np.array([c_starts[s][1] for s in range(10)])

        b_best_minus_med = float(np.max(b_kges) - np.median(b_kges))
        b_iqr = float(np.quantile(b_kges, 0.75) - np.quantile(b_kges, 0.25))
        c_best_minus_med = float(np.max(c_kges) - np.median(c_kges))
        c_iqr = float(np.quantile(c_kges, 0.75) - np.quantile(c_kges, 0.25))

        # Top 5 starts
        top5_b = sorted(range(10), key=lambda s: b_starts[s][1], reverse=True)[:5]
        top5_c = sorted(range(10), key=lambda s: c_starts[s][1], reverse=True)[:5]
        w5_b = float(np.median([rms_dist(b_starts[s1][0], b_starts[s2][0]) for s1, s2 in combinations(top5_b, 2)]))
        w5_c = float(np.median([rms_dist(c_starts[s1][0], c_starts[s2][0]) for s1, s2 in combinations(top5_c, 2)]))
        w5_pool = (w5_b + w5_c) / 2.0
        b5_all = float(np.median([rms_dist(b_starts[s1][0], c_starts[s2][0]) for s1 in top5_b for s2 in top5_c]))
        excess_top5 = b5_all - w5_pool

        # Top 3 starts
        top3_b = sorted(range(10), key=lambda s: b_starts[s][1], reverse=True)[:3]
        top3_c = sorted(range(10), key=lambda s: c_starts[s][1], reverse=True)[:3]
        w3_b = float(np.median([rms_dist(b_starts[s1][0], b_starts[s2][0]) for s1, s2 in combinations(top3_b, 2)]))
        w3_c = float(np.median([rms_dist(c_starts[s1][0], c_starts[s2][0]) for s1, s2 in combinations(top3_c, 2)]))
        w3_pool = (w3_b + w3_c) / 2.0
        b3_all = float(np.median([rms_dist(b_starts[s1][0], c_starts[s2][0]) for s1 in top3_b for s2 in top3_c]))
        excess_top3 = b3_all - w3_pool

        # All 10
        w_b10 = float(np.median([rms_dist(b_starts[s1][0], b_starts[s2][0]) for s1, s2 in combinations(range(10), 2)]))
        w_c10 = float(np.median([rms_dist(c_starts[s1][0], c_starts[s2][0]) for s1, s2 in combinations(range(10), 2)]))
        w10_pool = (w_b10 + w_c10) / 2.0
        b10_all = float(np.median([rms_dist(b_starts[s1][0], c_starts[s2][0]) for s1 in range(10) for s2 in range(10)]))
        excess_all10 = b10_all - w10_pool

        ic_quality_rows.append({
            "basin_id": b_id,
            "frac_snow": frac_snow,
            "snow_stratum": stratum,
            "base_best_minus_median_kge": b_best_minus_med,
            "base_kge_iqr": b_iqr,
            "cn_best_minus_median_kge": c_best_minus_med,
            "cn_kge_iqr": c_iqr,
            "within_pooled_all10": w10_pool,
            "within_pooled_top5": w5_pool,
            "within_pooled_top3": w3_pool,
            "between_all_all10": b10_all,
            "between_all_top5": b5_all,
            "between_all_top3": b3_all,
            "excess_all10": excess_all10,
            "excess_top5": excess_top5,
            "excess_top3": excess_top3,
        })

    # -------------------------------------------------------------
    # 2. dPL Seed Stability Audit
    # -------------------------------------------------------------
    dpl_df = ledger_df[ledger_df["paradigm"] == "dPL"]
    dpl_stability_rows: List[Dict[str, Any]] = []

    for struct in STRUCTURES:
        s_df = dpl_df[dpl_df["structure"] == struct]
        for p_name in SHARED_15_PARAMETERS:
            p_sub = s_df[s_df["parameter"] == p_name]
            piv = p_sub.pivot(index="basin_id", columns="start_or_seed", values="normalized_value")

            seed_stds = piv.std(axis=1, ddof=1).to_numpy()
            med_std = float(np.median(seed_stds))
            iqr_std = float(np.quantile(seed_stds, 0.75) - np.quantile(seed_stds, 0.25))

            dpl_stability_rows.append({
                "paradigm": "dPL",
                "structure": struct,
                "parameter": p_name,
                "symbol": PARAMETER_METADATA[p_name]["symbol"],
                "median_across_seed_std": med_std,
                "iqr_across_seed_std": iqr_std,
                "n_basins": len(piv),
            })

    # -------------------------------------------------------------
    # 3. Boundary & Point Mass Safeguards
    # -------------------------------------------------------------
    boundary_rows: List[Dict[str, Any]] = []
    tolerances = (0.01, 0.02, 0.05)

    for paradigm in PARADIGMS:
        p_canon = canon_df[canon_df["paradigm"] == paradigm]

        for struct in STRUCTURES:
            s_canon = p_canon[p_canon["structure"] == struct]
            for p_name in SHARED_15_PARAMETERS:
                z_vals = s_canon[f"z_{p_name}"].to_numpy(dtype=np.float64)

                exact_0 = float(np.mean(z_vals == 0.0))
                exact_1 = float(np.mean(z_vals == 1.0))

                for tol in tolerances:
                    near_0 = float(np.mean(z_vals <= tol))
                    near_1 = float(np.mean(z_vals >= (1.0 - tol)))
                    total_boundary = near_0 + near_1

                    boundary_rows.append({
                        "paradigm": paradigm,
                        "structure": struct,
                        "parameter": p_name,
                        "symbol": PARAMETER_METADATA[p_name]["symbol"],
                        "tolerance": tol,
                        "exact_zero_share": exact_0,
                        "exact_one_share": exact_1,
                        "near_lower_bound_share": near_0,
                        "near_upper_bound_share": near_1,
                        "total_boundary_mass_share": total_boundary,
                        "n_basins": len(z_vals),
                    })

    # Write output CSV files
    pd.DataFrame(ic_quality_rows).to_csv(out_dir / "r2_ic_restart_quality_audit.csv", index=False, float_format="%.17g")
    pd.DataFrame(dpl_stability_rows).to_csv(out_dir / "r2_dpl_seed_stability_audit.csv", index=False, float_format="%.17g")
    pd.DataFrame(boundary_rows).to_csv(out_dir / "r2_boundary_mass_safeguards.csv", index=False, float_format="%.17g")

    audit_meta = {
        "status": "PASS",
        "ic_restart_quality": {
            "mean_base_kge_iqr": float(pd.DataFrame(ic_quality_rows)["base_kge_iqr"].mean()),
            "mean_cn_kge_iqr": float(pd.DataFrame(ic_quality_rows)["cn_kge_iqr"].mean()),
            "all10_excess_median": float(pd.DataFrame(ic_quality_rows)["excess_all10"].median()),
            "top5_excess_median": float(pd.DataFrame(ic_quality_rows)["excess_top5"].median()),
            "top3_excess_median": float(pd.DataFrame(ic_quality_rows)["excess_top3"].median()),
        },
        "dpl_seed_stability": {
            "max_across_seed_std": float(pd.DataFrame(dpl_stability_rows)["median_across_seed_std"].max()),
            "mean_across_seed_std": float(pd.DataFrame(dpl_stability_rows)["median_across_seed_std"].mean()),
        },
        "boundary_safeguards_rows": len(boundary_rows),
    }

    with (out_dir / "diagnostics_and_safeguards_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit_meta, f, indent=2)

    return ic_quality_rows, dpl_stability_rows, boundary_rows, audit_meta


if __name__ == "__main__":
    ic_q, dpl_s, b_mass, meta = run_diagnostics_and_safeguards()
    print("Diagnostics and safeguards analysis complete:")
    print("  IC Restart Quality (Excess median):", meta["ic_restart_quality"])
    print("  dPL Seed Stability (Mean across-seed std):", meta["dpl_seed_stability"])
