#!/usr/bin/env python3
"""R1X Analysis E: basin-level temporal persistence."""
from __future__ import annotations

import numpy as np
import pandas as pd

from R1X_common import canonical_ids, load_temporal_table
from r1_config import TABLES_DIR
from r1_utils import pearson, spearman


def main() -> None:
    frame = load_temporal_table()
    rows: list[dict[str, object]] = []
    for basin in canonical_ids():
        row: dict[str, object] = {"basin_id": basin}
        for partition in ("A", "B"):
            values = frame.loc[(frame["basin_id"] == basin) & (frame["partition"] == partition), "delta_KGE"].to_numpy(float)
            suffix = partition
            row[f"N_models_{suffix}"] = int(values.size)
            row[f"delta_crossmodel_median_{suffix}"] = float(np.median(values))
            row[f"delta_crossmodel_mean_{suffix}"] = float(np.mean(values))
            row[f"frac_models_dPL_gt_IC_{suffix}"] = float(np.mean(values > 0.0))
            row[f"frac_models_IC_gt_dPL_{suffix}"] = float(np.mean(values < 0.0))
            row[f"frac_models_equal_{suffix}"] = float(np.mean(values == 0.0))
        rows.append(row)
    output = pd.DataFrame(rows)
    output.to_csv(TABLES_DIR / "R1X_temporal_basin_tendency.csv", index=False, float_format="%.10f")

    median_a = output["delta_crossmodel_median_A"].to_numpy(float)
    median_b = output["delta_crossmodel_median_B"].to_numpy(float)
    fraction_a = output["frac_models_dPL_gt_IC_A"].to_numpy(float)
    fraction_b = output["frac_models_dPL_gt_IC_B"].to_numpy(float)
    neutral = (median_a == 0.0) | (median_b == 0.0)
    same_sign_numerator = int(np.sum(np.sign(median_a[~neutral]) == np.sign(median_b[~neutral])))
    same_sign_denominator = int((~neutral).sum())
    summary_rows = [
        {"metric": "rho_M_spearman", "value": spearman(median_a, median_b), "N": len(output), "notes": "Primary basin-level temporal persistence diagnostic."},
        {"metric": "rho_M_pearson_secondary", "value": pearson(median_a, median_b), "N": len(output), "notes": "Secondary descriptive correlation."},
        {"metric": "same_sign_numerator", "value": same_sign_numerator, "N": same_sign_denominator, "notes": "Exact-zero median in either partition is neutral and excluded."},
        {"metric": "same_sign_denominator", "value": same_sign_denominator, "N": len(output), "notes": "Non-neutral basin pairs."},
        {"metric": "same_sign_fraction", "value": same_sign_numerator / same_sign_denominator if same_sign_denominator else np.nan, "N": same_sign_denominator, "notes": "Descriptive fraction among non-neutral basin pairs."},
        {"metric": "neutral_count", "value": int(neutral.sum()), "N": len(output), "notes": "Basin pairs with an exact-zero median in A or B."},
        {"metric": "rho_P_spearman", "value": spearman(fraction_a, fraction_b), "N": len(output), "notes": "Secondary persistence of cross-model dPL-positive fractions."},
        {"metric": "rho_P_pearson_secondary", "value": pearson(fraction_a, fraction_b), "N": len(output), "notes": "Secondary descriptive correlation."},
    ]
    pd.DataFrame(summary_rows).to_csv(TABLES_DIR / "R1X_temporal_basin_persistence_summary.csv", index=False, float_format="%.10f")
    print(f"PASS: wrote {TABLES_DIR / 'R1X_temporal_basin_tendency.csv'} ({len(output)} basins)")
    print(f"RESULT: rho_M={spearman(median_a, median_b):.6f}; same_sign={same_sign_numerator}/{same_sign_denominator}; rho_P={spearman(fraction_a, fraction_b):.6f}")


if __name__ == "__main__":
    main()
