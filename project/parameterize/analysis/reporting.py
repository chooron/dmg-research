"""Markdown reporting for seed/loss stability analysis scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from project.parameterize.analysis.common import frame_to_markdown, write_markdown


def build_master_report(
    analysis_root: Path,
    metric_summary: dict[str, pd.DataFrame],
    parameter_summary: dict[str, pd.DataFrame],
    cross_loss_parameter_summary: dict[str, pd.DataFrame],
    seed_corr_summary: dict[str, pd.DataFrame],
    loss_corr_summary: dict[str, pd.DataFrame],
    aggregate_corr_summary: dict[str, pd.DataFrame],
) -> Path:
    return write_markdown(
        analysis_root / "reports" / "analysis_results.md",
        title="Seed/Loss Stability Analysis",
        sections=[
            ("Metric Accuracy", frame_to_markdown(metric_summary["metrics_by_model_loss"])),
            ("Seed Parameter Variance", frame_to_markdown(parameter_summary["seed_parameter_variance_by_model_loss"])),
            ("Cross-loss Parameter Variance", frame_to_markdown(cross_loss_parameter_summary["cross_loss_parameter_variance_by_model"])),
            ("Seed Correlation Stability", frame_to_markdown(seed_corr_summary["correlation_seed_stability_summary"])),
            ("Loss Correlation Stability", frame_to_markdown(loss_corr_summary["correlation_loss_stability_summary"])),
            ("Top Correlation Relationships", frame_to_markdown(aggregate_corr_summary["correlation_top_relationships"].head(40))),
        ],
    )

