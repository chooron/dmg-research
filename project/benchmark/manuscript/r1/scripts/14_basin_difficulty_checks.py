#!/usr/bin/env python3
"""R1X Analysis F: descriptive basin difficulty and ceiling-effect checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from R1X_common import load_full_table
from r1_config import TABLES_DIR
from r1_utils import pearson, spearman


def main() -> None:
    frame = load_full_table()
    basin = frame.groupby("basin_id", observed=True).agg(
        baseline_IC_KGE_median=("KGE_IC", "median"),
        baseline_IC_KGE_Q25=("KGE_IC", lambda x: np.quantile(x, 0.25)),
        baseline_IC_KGE_Q75=("KGE_IC", lambda x: np.quantile(x, 0.75)),
    ).reset_index()
    basin["baseline_IC_KGE_IQR"] = basin["baseline_IC_KGE_Q75"] - basin["baseline_IC_KGE_Q25"]
    tendency = pd.read_csv(TABLES_DIR / "R1X_basin_estimator_tendency.csv", dtype={"basin_id": str})
    merged = basin.merge(tendency, on="basin_id", how="inner", validate="one_to_one")
    if len(merged) != 531:
        raise RuntimeError(f"difficulty-check join expected 531 basins, got {len(merged)}")
    checks = [
        ("baseline_IC_KGE_median", "delta_crossmodel_median"),
        ("baseline_IC_KGE_median", "frac_models_dPL_gt_IC"),
        ("baseline_IC_KGE_IQR", "delta_crossmodel_median"),
        ("baseline_IC_KGE_IQR", "frac_models_dPL_gt_IC"),
    ]
    rows = []
    for covariate, outcome in checks:
        x = merged[covariate].to_numpy(float)
        y = merged[outcome].to_numpy(float)
        rows.append({
            "covariate": covariate,
            "outcome": outcome,
            "spearman": spearman(x, y),
            "pearson": pearson(x, y),
            "N": len(merged),
            "notes": "Descriptive robustness check only; no attributes, p-values, or causal interpretation.",
        })
    output = pd.DataFrame(rows)
    output.to_csv(TABLES_DIR / "R1X_basin_difficulty_checks.csv", index=False, float_format="%.10f")
    print(f"PASS: wrote {TABLES_DIR / 'R1X_basin_difficulty_checks.csv'} ({len(output)} checks)")
    print(f"RESULT: max_abs_Spearman={output.spearman.abs().max():.6f}")


if __name__ == "__main__":
    main()
