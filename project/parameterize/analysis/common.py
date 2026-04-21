"""Common utilities for seed/loss stability analysis.

This module centralizes:
- config/output-root resolution
- run discovery and table loading
- parameter normalization by physical bounds
- CSV/NPZ/Markdown helpers

All scripts under ``project/parameterize/analysis`` use these helpers so the
analysis outputs stay consistent across steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from project.parameterize.figures.data_loading import ANALYSIS_PARAMETER_BOUNDS, discover_runs
from project.parameterize.figures.common import resolve_analysis_output_root as resolve_figures_analysis_root


DEFAULT_TOP_K = 10
DEFAULT_CORR_METHODS = ("spearman", "pearson", "kendall")
DEFAULT_METRICS = ("nse", "kge", "bias_abs")


def resolve_analysis_root(config_path: str, analysis_root: str | None = None) -> Path:
    if analysis_root is not None:
        root = Path(analysis_root).resolve()
    else:
        root = resolve_figures_analysis_root(config_path).parent / "stability_stats"
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_output_dirs(analysis_root: Path) -> dict[str, Path]:
    dirs = {
        "root": analysis_root,
        "tables": analysis_root / "tables",
        "parameter_variance": analysis_root / "parameter_variance",
        "correlation_matrices": analysis_root / "correlation_matrices",
        "correlation_summaries": analysis_root / "correlation_summaries",
        "reports": analysis_root / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_analysis_data(
    config_path: str,
    outputs_root: str | None = None,
    analysis_root: str | None = None,
    device: str = "cpu",
    parameter_csv: str | None = None,
    attribute_csv: str | None = None,
) -> dict[str, Any]:
    root = resolve_analysis_root(config_path, analysis_root)
    outputs_root_path = Path(outputs_root).resolve() if outputs_root else resolve_figures_analysis_root(config_path).parent.parent
    runs = discover_runs(outputs_root_path)
    metrics_long = load_metrics_long_from_outputs(runs)
    payload: dict[str, Any] = {
        "config_path": config_path,
        "outputs_root": outputs_root_path,
        "runs": runs,
        "metrics_long": metrics_long,
        "stability_analysis_root": root,
        "stability_output_dirs": ensure_output_dirs(root),
        "device": device,
        "parameter_bounds": ANALYSIS_PARAMETER_BOUNDS,
        "model_order": sorted(metrics_long["model"].dropna().unique().tolist()),
        "loss_order": sorted(metrics_long["loss"].dropna().unique().tolist()),
        "seed_order": sorted(metrics_long["seed"].dropna().unique().tolist()),
    }
    if parameter_csv:
        payload["params_long"] = load_parameter_long_from_csv(parameter_csv)
    if attribute_csv:
        payload["attributes"] = load_attributes_from_csv(attribute_csv)
    return payload


def parse_corr_methods(raw_value: str | None) -> tuple[str, ...]:
    if not raw_value:
        return DEFAULT_CORR_METHODS
    methods = tuple(item.strip().lower() for item in raw_value.split(",") if item.strip())
    supported = {"spearman", "pearson", "kendall"}
    unknown = [method for method in methods if method not in supported]
    if unknown:
        raise ValueError(f"Unsupported correlation methods: {unknown}")
    return methods


def parse_metric_names(raw_value: str | None) -> tuple[str, ...]:
    if not raw_value:
        return DEFAULT_METRICS
    return tuple(item.strip().lower() for item in raw_value.split(",") if item.strip())


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default="project/parameterize/conf/config_param_paper.yaml",
        help="Paper-stack config file used to resolve outputs and attributes.",
    )
    parser.add_argument(
        "--outputs-root",
        default=None,
        help="Optional override for the run outputs root.",
    )
    parser.add_argument(
        "--analysis-root",
        default=None,
        help="Optional override for the analysis output root.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "mps"),
        help="Reserved for compatibility; this analysis package does not reconstruct models.",
    )
    parser.add_argument(
        "--parameter-csv",
        default=None,
        help="Existing parameter csv for parameter/correlation analysis. Supports long or *_mean wide format.",
    )
    parser.add_argument(
        "--attribute-csv",
        default=None,
        help="Existing basin attribute csv for correlation analysis.",
    )
    return parser


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metrics_long_from_outputs(runs: list[Any]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for run in runs:
        results_path = run.run_dir / f"results_seed{run.seed}.csv"
        metrics_path = run.run_dir / "metrics_avg.json"
        if not metrics_path.exists():
            metrics_path = run.run_dir / "metrics.json"
        if not results_path.exists() or not metrics_path.exists():
            continue
        frame = pd.read_csv(results_path)
        metrics = read_json(metrics_path)
        for metric in ("nse", "kge", "bias", "pbias_abs"):
            values = metrics.get(metric)
            if values is not None and len(values) == len(frame):
                frame[metric] = values
        if "bias_abs" not in frame.columns:
            if "bias" in frame.columns:
                frame["bias_abs"] = frame["bias"].abs()
            elif "pbias_abs" in frame.columns:
                frame["bias_abs"] = frame["pbias_abs"]
            else:
                frame["bias_abs"] = np.nan
        frame["model"] = run.model
        frame["loss"] = run.loss
        frame["seed"] = run.seed
        rows.append(frame)
    if not rows:
        raise FileNotFoundError("No metrics json/results csv files were found under the outputs root.")
    return pd.concat(rows, ignore_index=True)


def load_parameter_long_from_csv(parameter_csv: str) -> pd.DataFrame:
    frame = pd.read_csv(parameter_csv)
    required_long = {"model", "loss", "seed", "basin_id", "parameter", "mean"}
    if required_long.issubset(frame.columns):
        if "sample_count" not in frame.columns:
            frame["sample_count"] = np.nan
        if "std" not in frame.columns:
            frame["std"] = np.nan
        return frame

    meta_cols = {"model", "loss", "seed", "basin_id", "sample_count"}
    value_cols = [column for column in frame.columns if column not in meta_cols]
    mean_cols = [column for column in value_cols if column.endswith("_mean")]
    std_cols = {column[: -len("_std")]: column for column in value_cols if column.endswith("_std")}
    if not mean_cols:
        raise ValueError(
            f"Could not interpret parameter csv '{parameter_csv}'. Expected long format or *_mean wide columns."
        )

    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        for mean_col in mean_cols:
            parameter = mean_col[: -len("_mean")]
            std_col = std_cols.get(parameter)
            records.append(
                {
                    "model": row["model"],
                    "loss": row["loss"],
                    "seed": row["seed"],
                    "basin_id": row["basin_id"],
                    "sample_count": row.get("sample_count", np.nan),
                    "parameter": parameter,
                    "mean": row[mean_col],
                    "std": row[std_col] if std_col is not None else np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def load_attributes_from_csv(attribute_csv: str) -> pd.DataFrame:
    frame = pd.read_csv(attribute_csv)
    if "basin_id" not in frame.columns:
        raise ValueError(f"Attribute csv '{attribute_csv}' must include a basin_id column.")
    return frame


def normalize_parameters_to_unit_interval(
    params_long: pd.DataFrame,
    parameter_bounds: dict[str, tuple[float, float]],
    value_column: str = "mean",
) -> pd.DataFrame:
    frame = params_long.copy()
    frame["lower_bound"] = frame["parameter"].map(lambda name: float(parameter_bounds[name][0]))
    frame["upper_bound"] = frame["parameter"].map(lambda name: float(parameter_bounds[name][1]))
    frame["parameter_range"] = frame["upper_bound"] - frame["lower_bound"]
    frame[f"{value_column}_unit"] = (
        frame[value_column] - frame["lower_bound"]
    ) / frame["parameter_range"]
    frame[f"{value_column}_unit"] = frame[f"{value_column}_unit"].clip(0.0, 1.0)
    return frame


def pairwise_mean_abs_diff(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[~np.isnan(array)]
    if array.size < 2:
        return 0.0
    diffs = []
    for idx in range(array.size):
        for jdx in range(idx + 1, array.size):
            diffs.append(abs(array[idx] - array[jdx]))
    return float(np.mean(diffs))


def correlation_value(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float]:
    valid = ~(x.isna() | y.isna())
    x_valid = x.loc[valid]
    y_valid = y.loc[valid]
    if len(x_valid) < 2:
        return np.nan, np.nan
    if method == "spearman":
        rho, p_value = spearmanr(x_valid, y_valid, nan_policy="omit")
    elif method == "pearson":
        rho, p_value = pearsonr(x_valid, y_valid)
    elif method == "kendall":
        rho, p_value = kendalltau(x_valid, y_valid, nan_policy="omit")
    else:
        raise ValueError(f"Unsupported correlation method '{method}'.")
    return float(rho), float(p_value)


def save_frame(frame: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index)
    return path


def save_npz(matrix: np.ndarray, row_names: list[str], col_names: list[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, matrix=matrix, row_names=np.asarray(row_names), col_names=np.asarray(col_names))
    return path


def frame_to_markdown(frame: pd.DataFrame) -> str:
    render = frame.copy()
    for column in render.columns:
        render[column] = render[column].map(
            lambda value: f"{value:.6f}" if isinstance(value, (float, np.floating)) and np.isfinite(value) else value
        )
    headers = [str(column) for column in render.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in render.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in render.columns) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, title: str, sections: list[tuple[str, str]]) -> Path:
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", "", body, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
