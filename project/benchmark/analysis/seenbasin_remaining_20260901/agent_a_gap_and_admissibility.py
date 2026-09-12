#!/usr/bin/env python3
"""Agent A: frozen G_seen heterogeneity and admissibility analysis."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    ALL_MODELS,
    RESULTS,
    SEED,
    bootstrap_median,
    corr,
    load_inputs,
    main_label,
    write_csv,
)

N_BOOT = 5000
THRESHOLDS = (0.02, 0.05, 0.10)


def main() -> None:
    ids, paired, _, _, _, _, _ = load_inputs()
    paired = paired.copy()
    paired["G_seen"] = paired["KGE_IC"] - paired["KGE_dPL"]
    paired["gap_definition"] = main_label()
    write_csv(RESULTS / "agent_A/A02_BASIN_GAP_LONG.csv", paired)

    model_rows = []
    rng = np.random.default_rng(SEED)
    for model in ALL_MODELS:
        frame = paired[paired.model == model].sort_values("basin_id")
        g = frame.G_seen.to_numpy(float)
        ci_low, ci_high = bootstrap_median(g, n_boot=N_BOOT, seed=int(rng.integers(0, 2**32 - 1)))
        rho, p, n = corr(frame.KGE_IC, frame.KGE_dPL)
        model_rows.append({
            "model": model,
            "n_basins": int(len(frame)),
            "G_seen_median": float(np.median(g)),
            "G_seen_mean": float(np.mean(g)),
            "G_seen_q25": float(np.quantile(g, .25)),
            "G_seen_q75": float(np.quantile(g, .75)),
            "G_seen_bootstrap_ci95_low": ci_low,
            "G_seen_bootstrap_ci95_high": ci_high,
            "fraction_G_seen_positive": float(np.mean(g > 0)),
            "fraction_G_seen_negative": float(np.mean(g < 0)),
            "fraction_G_seen_zero": float(np.mean(g == 0)),
            "spearman_KGE_IC_KGE_dPL": rho,
            "spearman_p_value": p,
            "spearman_n": n,
            "bootstrap_seed": SEED,
            "bootstrap_replicates": N_BOOT,
            "test_period": "1995-10-01..2010-09-30",
            "label": main_label(),
        })
    model_summary = pd.DataFrame(model_rows)
    write_csv(RESULTS / "agent_A/A01_MODEL_GAP_HETEROGENEITY.csv", model_summary)

    basin_rows = []
    for basin, frame in paired.groupby("basin_id", sort=True):
        g = frame.G_seen.to_numpy(float)
        basin_rows.append({
            "basin_id": basin,
            "n_models": int(len(frame)),
            "median_G_seen": float(np.median(g)),
            "mean_G_seen": float(np.mean(g)),
            "fraction_models_G_positive": float(np.mean(g > 0)),
            "fraction_models_G_negative": float(np.mean(g < 0)),
            "G_seen_q25": float(np.quantile(g, .25)),
            "G_seen_q75": float(np.quantile(g, .75)),
            "G_seen_min": float(np.min(g)),
            "G_seen_max": float(np.max(g)),
            "G_seen_spread_max_minus_min": float(np.max(g) - np.min(g)),
            "label": main_label(),
        })
    basin_summary = pd.DataFrame(basin_rows)
    write_csv(RESULTS / "agent_A/A03_BASIN_SHARED_MAPPING_SUSCEPTIBILITY.csv", basin_summary)

    admissible_rows, model_admissible_rows, sensitivity_rows = [], [], []
    for tau in THRESHOLDS:
        for basin, frame in paired.groupby("basin_id", sort=True):
            best_ic = float(frame.KGE_IC.max())
            eligible = frame[frame.KGE_IC >= best_ic - tau].copy()
            g = eligible.G_seen.to_numpy(float)
            ordered = eligible.sort_values(["G_seen", "model"])
            lowest = ordered.iloc[0]
            highest = ordered.iloc[-1]
            names = ";".join(eligible.sort_values("model").model.tolist())
            admissible_rows.append({
                "tau": tau,
                "basin_id": basin,
                "best_IC_KGE": best_ic,
                "admissible_model_count": int(len(eligible)),
                "admissible_models": names,
                "admissible_G_seen_median": float(np.median(g)),
                "admissible_G_seen_mean": float(np.mean(g)),
                "admissible_G_seen_q25": float(np.quantile(g, .25)),
                "admissible_G_seen_q75": float(np.quantile(g, .75)),
                "admissible_G_seen_spread": float(np.max(g) - np.min(g)),
                "admissible_lowest_G_seen": float(lowest.G_seen),
                "admissible_lowest_G_model": lowest.model,
                "admissible_highest_G_seen": float(highest.G_seen),
                "admissible_highest_G_model": highest.model,
                "best_worst_G_seen_difference": float(highest.G_seen - lowest.G_seen),
                "label": "PREDICTIVELY_ADMISSIBLE_SEEN_BASIN",
            })
        current = pd.DataFrame([r for r in admissible_rows if r["tau"] == tau])
        model_count = paired["model"].map(paired["model"].value_counts()).astype(float)
        for model in ALL_MODELS:
            vals = []
            for basin, frame in paired.groupby("basin_id", sort=True):
                best_ic = frame.KGE_IC.max()
                hit = frame[(frame.model == model) & (frame.KGE_IC >= best_ic - tau)]
                if len(hit):
                    vals.append(float(hit.G_seen.iloc[0]))
            model_admissible_rows.append({
                "tau": tau,
                "model": model,
                "admissible_basin_count": len(vals),
                "admissible_basin_fraction": len(vals) / len(ids),
                "median_G_seen_when_admissible": float(np.median(vals)) if vals else np.nan,
                "mean_G_seen_when_admissible": float(np.mean(vals)) if vals else np.nan,
                "fraction_admissible_G_positive": float(np.mean(np.asarray(vals) > 0)) if vals else np.nan,
                "label": "PREDICTIVELY_ADMISSIBLE_SEEN_BASIN",
            })
        sensitivity_rows.append({
            "tau": tau,
            "primary_threshold": tau == 0.05,
            "median_admissible_model_count": float(current.admissible_model_count.median()),
            "mean_admissible_model_count": float(current.admissible_model_count.mean()),
            "fraction_basins_with_multiple_admissible_models": float(np.mean(current.admissible_model_count > 1)),
            "median_admissible_G_seen_spread": float(current.admissible_G_seen_spread.median()),
            "median_best_worst_G_seen_difference": float(current.best_worst_G_seen_difference.median()),
            "fraction_basins_with_positive_and_negative_G": float(np.mean((current.admissible_lowest_G_seen < 0) & (current.admissible_highest_G_seen > 0))),
            "label": "ADMISSIBILITY_THRESHOLD_SENSITIVITY",
        })
    write_csv(RESULTS / "agent_A/A04_PREDICTIVELY_ADMISSIBLE_BASIN_LEVEL.csv", pd.DataFrame(admissible_rows))
    write_csv(RESULTS / "agent_A/A05_ADMISSIBLE_MODEL_GAP_SUMMARY.csv", pd.DataFrame(model_admissible_rows))
    write_csv(RESULTS / "agent_A/A06_ADMISSIBILITY_THRESHOLD_SENSITIVITY.csv", pd.DataFrame(sensitivity_rows))
    (RESULTS / "agent_A/A_CONFIG.json").write_text(json.dumps({"thresholds": THRESHOLDS, "primary_tau": .05, "bootstrap_seed": SEED, "bootstrap_replicates": N_BOOT, "G_seen": "KGE_IC-KGE_dPL"}, indent=2) + "\n")
    print("Agent A complete", len(model_summary), len(basin_summary), len(admissible_rows))


if __name__ == "__main__":
    main()
