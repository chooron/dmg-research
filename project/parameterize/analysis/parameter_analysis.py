"""Parameter-level stability analysis across seeds and losses.

This module focuses on:
- per-basin per-parameter variance across seeds
- mean absolute parameter differences across seeds
- pooled or seed-first variance across losses
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from project.parameterize.analysis.common import (
    frame_to_markdown,
    normalize_parameters_to_unit_interval,
    pairwise_mean_abs_diff,
    save_frame,
    write_markdown,
)


def compute_seed_parameter_variance(
    params_long: pd.DataFrame,
    parameter_bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    normalized = normalize_parameters_to_unit_interval(params_long, parameter_bounds, value_column="mean")
    grouped = (
        normalized.groupby(["model", "loss", "basin_id", "parameter"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            variance_unit=("mean_unit", lambda values: float(np.var(values, ddof=0))),
            mean_abs_seed_diff=("mean_unit", pairwise_mean_abs_diff),
            mean_unit=("mean_unit", "mean"),
            std_unit=("mean_unit", lambda values: float(np.std(values, ddof=0))),
        )
    )
    return grouped


def variance_long_to_wide(variance_long: pd.DataFrame, value_column: str) -> dict[tuple[str, str], pd.DataFrame]:
    wide_tables: dict[tuple[str, str], pd.DataFrame] = {}
    for (model, loss), subset in variance_long.groupby(["model", "loss"]):
        wide = subset.pivot(index="basin_id", columns="parameter", values=value_column).reset_index()
        wide_tables[(model, loss)] = wide
    return wide_tables


def summarize_seed_parameter_variance(variance_long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by_model_loss = (
        variance_long.groupby(["model", "loss"], as_index=False)
        .agg(
            mean_variance_unit=("variance_unit", "mean"),
            median_variance_unit=("variance_unit", "median"),
            p90_variance_unit=("variance_unit", lambda values: float(np.nanpercentile(values, 90))),
            mean_abs_seed_diff=("mean_abs_seed_diff", "mean"),
        )
    )
    by_parameter = (
        variance_long.groupby(["model", "loss", "parameter"], as_index=False)
        .agg(
            mean_variance_unit=("variance_unit", "mean"),
            mean_abs_seed_diff=("mean_abs_seed_diff", "mean"),
        )
    )
    by_model = (
        variance_long.groupby(["model"], as_index=False)
        .agg(
            mean_variance_unit=("variance_unit", "mean"),
            median_variance_unit=("variance_unit", "median"),
            mean_abs_seed_diff=("mean_abs_seed_diff", "mean"),
        )
    )
    return {
        "seed_parameter_variance_by_model_loss": by_model_loss,
        "seed_parameter_variance_by_parameter": by_parameter,
        "seed_parameter_variance_by_model": by_model,
    }


def compute_cross_loss_parameter_variance(
    params_long: pd.DataFrame,
    parameter_bounds: dict[str, tuple[float, float]],
    mode: str = "pooled",
) -> pd.DataFrame:
    normalized = normalize_parameters_to_unit_interval(params_long, parameter_bounds, value_column="mean")
    if mode == "pooled":
        return (
            normalized.groupby(["model", "basin_id", "parameter"], as_index=False)
            .agg(
                sample_count=("mean_unit", "size"),
                pooled_loss_variance_unit=("mean_unit", lambda values: float(np.var(values, ddof=0))),
                pooled_loss_mean_abs_diff=("mean_unit", pairwise_mean_abs_diff),
            )
        )

    if mode == "seed-first":
        seed_level = (
            normalized.groupby(["model", "seed", "basin_id", "parameter"], as_index=False)
            .agg(
                seed_loss_variance_unit=("mean_unit", lambda values: float(np.var(values, ddof=0))),
                seed_loss_mean_abs_diff=("mean_unit", pairwise_mean_abs_diff),
            )
        )
        return (
            seed_level.groupby(["model", "basin_id", "parameter"], as_index=False)
            .agg(
                sample_count=("seed", "size"),
                pooled_loss_variance_unit=("seed_loss_variance_unit", "mean"),
                pooled_loss_mean_abs_diff=("seed_loss_mean_abs_diff", "mean"),
            )
        )
    raise ValueError(f"Unsupported cross-loss mode '{mode}'.")


def summarize_cross_loss_parameter_variance(cross_loss_long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by_model = (
        cross_loss_long.groupby(["model"], as_index=False)
        .agg(
            mean_pooled_loss_variance_unit=("pooled_loss_variance_unit", "mean"),
            median_pooled_loss_variance_unit=("pooled_loss_variance_unit", "median"),
            mean_pooled_loss_abs_diff=("pooled_loss_mean_abs_diff", "mean"),
        )
    )
    by_parameter = (
        cross_loss_long.groupby(["model", "parameter"], as_index=False)
        .agg(
            mean_pooled_loss_variance_unit=("pooled_loss_variance_unit", "mean"),
            mean_pooled_loss_abs_diff=("pooled_loss_mean_abs_diff", "mean"),
        )
    )
    return {
        "cross_loss_parameter_variance_by_model": by_model,
        "cross_loss_parameter_variance_by_parameter": by_parameter,
    }


def write_parameter_outputs(
    variance_long: pd.DataFrame,
    variance_summary: dict[str, pd.DataFrame],
    cross_loss_outputs: dict[str, pd.DataFrame],
    output_dir: Path,
) -> dict[str, Path]:
    paths = {
        "seed_parameter_variance_long": save_frame(
            variance_long, output_dir / "seed_parameter_variance_long.csv"
        ),
    }
    for (model, loss), wide in variance_long_to_wide(variance_long, "variance_unit").items():
        stem = f"seed_parameter_variance__{model}__{loss}.csv"
        paths[stem] = save_frame(wide, output_dir / stem)
    for (model, loss), wide in variance_long_to_wide(variance_long, "mean_abs_seed_diff").items():
        stem = f"seed_parameter_absdiff__{model}__{loss}.csv"
        paths[stem] = save_frame(wide, output_dir / stem)

    for name, frame in variance_summary.items():
        paths[name] = save_frame(frame, output_dir / f"{name}.csv")

    for name, frame in cross_loss_outputs.items():
        paths[name] = save_frame(frame, output_dir / f"{name}.csv")

    report = write_markdown(
        output_dir / "parameter_variance_report.md",
        title="Parameter Variance Summary",
        sections=[
            ("Seed Variance by Model and Loss", frame_to_markdown(variance_summary["seed_parameter_variance_by_model_loss"])),
            ("Seed Variance by Model", frame_to_markdown(variance_summary["seed_parameter_variance_by_model"])),
            ("Cross-loss Pooled Variance by Model", frame_to_markdown(cross_loss_outputs["cross_loss_parameter_variance_by_model"])),
        ],
    )
    paths["report"] = report
    return paths

