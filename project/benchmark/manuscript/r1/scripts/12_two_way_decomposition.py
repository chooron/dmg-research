#!/usr/bin/env python3
"""R1X Analysis D: descriptive additive two-way decomposition."""
from __future__ import annotations

import numpy as np
import pandas as pd

from R1X_common import canonical_ids, load_full_table
from r1_config import MODEL_REGISTRY, TABLES_DIR


def main() -> None:
    frame = load_full_table()
    matrix = frame.pivot(index="model", columns="basin_id", values="delta_KGE").loc[list(MODEL_REGISTRY), canonical_ids()]
    values = matrix.to_numpy(dtype=float)
    n_models, n_basins = values.shape
    grand_mean = float(values.mean())
    model_effect = values.mean(axis=1) - grand_mean
    basin_effect = values.mean(axis=0) - grand_mean
    residual = values - grand_mean - model_effect[:, None] - basin_effect[None, :]
    ss_total = float(np.square(values - grand_mean).sum())
    ss_model = float(n_basins * np.square(model_effect).sum())
    ss_basin = float(n_models * np.square(basin_effect).sum())
    ss_residual = float(np.square(residual).sum())
    ss_sum_error = abs(ss_total - ss_model - ss_basin - ss_residual)
    if ss_total == 0.0:
        raise RuntimeError("two-way decomposition has zero total variation")
    rows = [
        {"component": "total", "SS": ss_total, "fraction_of_total": 1.0, "N_models": n_models, "N_basins": n_basins, "notes": "DESCRIPTIVE ONLY; NO P-VALUES; no inferential interaction claim."},
        {"component": "model_main_effect", "SS": ss_model, "fraction_of_total": ss_model / ss_total, "N_models": n_models, "N_basins": n_basins, "notes": "DESCRIPTIVE ONLY; model main effect from row means."},
        {"component": "basin_main_effect", "SS": ss_basin, "fraction_of_total": ss_basin / ss_total, "N_models": n_models, "N_basins": n_basins, "notes": "DESCRIPTIVE ONLY; basin main effect from column means."},
        {"component": "residual_model_basin_specificity", "SS": ss_residual, "fraction_of_total": ss_residual / ss_total, "N_models": n_models, "N_basins": n_basins, "notes": "DESCRIPTIVE ONLY; residual model-basin specificity, not inferential interaction variance."},
    ]
    output = pd.DataFrame(rows)
    output["additive_identity_abs_error"] = ss_sum_error
    output.to_csv(TABLES_DIR / "R1X_two_way_decomposition.csv", index=False, float_format="%.12f")
    print(f"PASS: wrote {TABLES_DIR / 'R1X_two_way_decomposition.csv'}")
    print(f"RESULT: model={ss_model / ss_total:.6f}; basin={ss_basin / ss_total:.6f}; residual_specificity={ss_residual / ss_total:.6f}; identity_error={ss_sum_error:.3g}")


if __name__ == "__main__":
    main()
