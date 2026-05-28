"""Evidence-table interfaces reserved for later merge stages."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INDEPENDENT_CALIBRATION_COLUMNS = [
    "basin_id",
    "model_id",
    "objective",
    "random_start_id",
    "optimized_parameters",
    "train_NSE",
    "validation_NSE",
    "test_NSE",
    "train_logNSE",
    "validation_logNSE",
    "test_logNSE",
    "KGE",
    "KGE_r",
    "KGE_alpha",
    "KGE_beta",
    "high_flow_bias",
    "low_flow_bias",
    "rmse_high_flow",
    "rmse_low_flow",
    "boundary_flag",
    "optimization_success_flag",
    "runtime",
]


def collect_independent_calibration_results(root: str | Path) -> pd.DataFrame:
    """Collect per-task CSV files into one long evidence table."""
    frames = [pd.read_csv(path) for path in Path(root).rglob("results.csv")]
    if not frames:
        return pd.DataFrame(columns=INDEPENDENT_CALIBRATION_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def write_evidence_table(root: str | Path, output_path: str | Path) -> Path:
    table = collect_independent_calibration_results(root)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        table.to_parquet(path, index=False)
    else:
        table.to_csv(path, index=False)
    return path
