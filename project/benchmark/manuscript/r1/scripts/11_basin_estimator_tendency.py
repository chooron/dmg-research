#!/usr/bin/env python3
"""R1X Analyses B/C: basin cross-model tendency and tolerance sensitivity."""
from __future__ import annotations

import numpy as np
import pandas as pd

from R1X_common import canonical_ids, load_full_table
from r1_config import TABLES_DIR

TAUS = (0.00, 0.01, 0.02, 0.05)


def main() -> None:
    frame = load_full_table()
    rows: list[dict[str, object]] = []
    for basin in canonical_ids():
        values = frame.loc[frame["basin_id"] == basin, "delta_KGE"].to_numpy(float)
        rows.append({
            "basin_id": basin,
            "N_models": int(values.size),
            "delta_crossmodel_median": float(np.median(values)),
            "delta_crossmodel_mean": float(np.mean(values)),
            "delta_crossmodel_Q25": float(np.quantile(values, 0.25)),
            "delta_crossmodel_Q75": float(np.quantile(values, 0.75)),
            "delta_crossmodel_IQR": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
            "delta_crossmodel_min": float(np.min(values)),
            "delta_crossmodel_max": float(np.max(values)),
            "frac_models_dPL_gt_IC": float(np.mean(values > 0.0)),
            "frac_models_IC_gt_dPL": float(np.mean(values < 0.0)),
            "frac_models_equal": float(np.mean(values == 0.0)),
        })
    tendency = pd.DataFrame(rows)
    tendency.to_csv(TABLES_DIR / "R1X_basin_estimator_tendency.csv", index=False, float_format="%.10f")

    sensitivity_rows: list[dict[str, object]] = []
    for tau in TAUS:
        fractions = []
        for basin in canonical_ids():
            values = frame.loc[frame["basin_id"] == basin, "delta_KGE"].to_numpy(float)
            fractions.append(float(np.mean(values >= -tau)))
        x = np.asarray(fractions, dtype=float)
        sensitivity_rows.append({
            "tau": tau,
            "N_basins": int(x.size),
            "median_C_b": float(np.median(x)),
            "C_b_Q10": float(np.quantile(x, 0.10)),
            "C_b_Q25": float(np.quantile(x, 0.25)),
            "C_b_Q75": float(np.quantile(x, 0.75)),
            "C_b_Q90": float(np.quantile(x, 0.90)),
            "frac_basins_C_gt_0.5": float(np.mean(x > 0.5)),
            "frac_basins_C_ge_0.75": float(np.mean(x >= 0.75)),
            "frac_basins_C_ge_0.90": float(np.mean(x >= 0.90)),
        })
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(TABLES_DIR / "R1X_basin_comparable_or_better_sensitivity.csv", index=False, float_format="%.10f")

    population_rows = []
    for tau in TAUS:
        row = sensitivity.loc[sensitivity["tau"] == tau].iloc[0]
        population_rows.append({
            "tau": tau,
            "metric": "C_b_population_summary",
            "N_basins": int(row["N_basins"]),
            "median_C_b": row["median_C_b"],
            "C_b_Q10": row["C_b_Q10"], "C_b_Q25": row["C_b_Q25"],
            "C_b_Q75": row["C_b_Q75"], "C_b_Q90": row["C_b_Q90"],
            "frac_basins_C_gt_0.5": row["frac_basins_C_gt_0.5"],
            "frac_basins_C_ge_0.75": row["frac_basins_C_ge_0.75"],
            "frac_basins_C_ge_0.90": row["frac_basins_C_ge_0.90"],
            "notes": "Descriptive prevalence only; tau=0 is primary and tau>0 are sensitivity thresholds.",
        })
    pd.DataFrame(population_rows).to_csv(TABLES_DIR / "R1X_comparable_or_better_population_summary.csv", index=False, float_format="%.10f")
    print(f"PASS: wrote basin tendency ({len(tendency)} basins) and comparable-or-better sensitivity ({len(sensitivity)} thresholds)")
    print(f"RESULT: P_b+ median={tendency.frac_models_dPL_gt_IC.median():.6f}; range={tendency.frac_models_dPL_gt_IC.min():.6f}..{tendency.frac_models_dPL_gt_IC.max():.6f}")


if __name__ == "__main__":
    main()
