"""Accuracy statistics for seed-wise HBV parameterization runs.

This module summarizes basin-level performance metrics such as NSE and KGE
across model, loss, and seed combinations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from project.parameterize.analysis.common import frame_to_markdown, save_frame, write_markdown


def summarize_metrics(metrics_long: pd.DataFrame, metric_names: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    available_metrics = [metric for metric in metric_names if metric in metrics_long.columns]
    if not available_metrics:
        raise ValueError(f"None of the requested metrics are present: {metric_names}")

    by_run = (
        metrics_long.groupby(["model", "loss", "seed"], as_index=False)[available_metrics]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
    by_run.columns = [
        "_".join(str(part) for part in column if part)
        for column in by_run.columns.to_flat_index()
    ]
    by_run = by_run.rename(columns={"model_": "model", "loss_": "loss", "seed_": "seed"})

    by_model_loss = (
        metrics_long.groupby(["model", "loss"], as_index=False)[available_metrics]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
    by_model_loss.columns = [
        "_".join(str(part) for part in column if part)
        for column in by_model_loss.columns.to_flat_index()
    ]
    by_model_loss = by_model_loss.rename(columns={"model_": "model", "loss_": "loss"})

    by_model = (
        metrics_long.groupby(["model"], as_index=False)[available_metrics]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
    by_model.columns = [
        "_".join(str(part) for part in column if part)
        for column in by_model.columns.to_flat_index()
    ]
    by_model = by_model.rename(columns={"model_": "model"})

    return {
        "metrics_long": metrics_long,
        "metrics_by_run": by_run,
        "metrics_by_model_loss": by_model_loss,
        "metrics_by_model": by_model,
    }


def write_metric_outputs(summary: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, Path]:
    paths = {
        "metrics_long": save_frame(summary["metrics_long"], output_dir / "basin_metrics_long.csv"),
        "metrics_by_run": save_frame(summary["metrics_by_run"], output_dir / "metrics_by_run.csv"),
        "metrics_by_model_loss": save_frame(
            summary["metrics_by_model_loss"], output_dir / "metrics_by_model_loss.csv"
        ),
        "metrics_by_model": save_frame(summary["metrics_by_model"], output_dir / "metrics_by_model.csv"),
    }
    report = write_markdown(
        output_dir / "metric_accuracy_report.md",
        title="Metric Accuracy Summary",
        sections=[
            ("By Model and Loss", frame_to_markdown(summary["metrics_by_model_loss"])),
            ("By Model", frame_to_markdown(summary["metrics_by_model"])),
        ],
    )
    paths["report"] = report
    return paths

