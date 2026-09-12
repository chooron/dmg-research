#!/usr/bin/env python3
"""R1X Analysis A: model-level estimator-relative tendency."""
from __future__ import annotations

import numpy as np
import pandas as pd

from R1X_common import load_full_table
from r1_config import MODEL_REGISTRY, TABLES_DIR


def main() -> None:
    frame = load_full_table()
    rows: list[dict[str, object]] = []
    for model in MODEL_REGISTRY:
        values = frame.loc[frame["model"] == model, "delta_KGE"].to_numpy(float)
        rows.append({
            "model": model,
            "N_basins": int(values.size),
            "delta_median": float(np.median(values)),
            "delta_mean": float(np.mean(values)),
            "delta_Q10": float(np.quantile(values, 0.10)),
            "delta_Q25": float(np.quantile(values, 0.25)),
            "delta_Q75": float(np.quantile(values, 0.75)),
            "delta_Q90": float(np.quantile(values, 0.90)),
            "frac_dPL_gt_IC": float(np.mean(values > 0.0)),
            "frac_IC_gt_dPL": float(np.mean(values < 0.0)),
            "frac_equal": float(np.mean(values == 0.0)),
        })
    output = pd.DataFrame(rows)
    output.to_csv(TABLES_DIR / "R1X_model_estimator_tendency.csv", index=False, float_format="%.10f")
    print(f"PASS: wrote {TABLES_DIR / 'R1X_model_estimator_tendency.csv'} ({len(output)} models)")
    print(f"RESULT: P_m+ median={output.frac_dPL_gt_IC.median():.6f}; range={output.frac_dPL_gt_IC.min():.6f}..{output.frac_dPL_gt_IC.max():.6f}")


if __name__ == "__main__":
    main()
