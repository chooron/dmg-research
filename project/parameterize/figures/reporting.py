"""Markdown reporting for the WRR figure suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from project.parameterize.figures.common import (
    pretty_loss_name,
    pretty_model_name,
    pretty_parameter_name,
    reference_loss_only,
    top_pairs_by_abs_rho,
)


def _reference_metric_summary(data_dict: dict[str, Any]) -> pd.DataFrame:
    metrics = reference_loss_only(data_dict["metrics_long"], data_dict)
    grouped = (
        metrics.groupby("model", as_index=False)
        .agg(
            basin_count=("basin_id", "nunique"),
            seed_count=("seed", "nunique"),
            nse_mean=("nse", "mean"),
            nse_median=("nse", "median"),
            kge_mean=("kge", "mean"),
            kge_median=("kge", "median"),
            bias_abs_mean=("bias_abs", "mean"),
        )
        .sort_values("model")
        .reset_index(drop=True)
    )
    grouped["model"] = grouped["model"].map(pretty_model_name)
    return grouped


def _reference_stability_summary(data_dict: dict[str, Any]) -> pd.DataFrame:
    params = reference_loss_only(data_dict["params_long"], data_dict)
    grouped = params.groupby(["model", "basin_id", "parameter"], as_index=False).agg(
        mean_of_seed=("mean", "mean"),
        raw_variance=("mean", "var"),
    )
    grouped["parameter_range"] = grouped["parameter"].map(
        lambda name: float(data_dict["parameter_bounds"][name][1] - data_dict["parameter_bounds"][name][0])
    )
    grouped["range_normalized_variance"] = grouped["raw_variance"] / (grouped["parameter_range"] ** 2)
    summary = (
        grouped.groupby("model", as_index=False)
        .agg(
            mean_range_normalized_variance=("range_normalized_variance", "mean"),
            median_range_normalized_variance=("range_normalized_variance", "median"),
            mean_raw_variance=("raw_variance", "mean"),
        )
        .sort_values("model")
        .reset_index(drop=True)
    )
    summary["model"] = summary["model"].map(pretty_model_name)
    return summary


def _reference_corr_stability_summary(data_dict: dict[str, Any]) -> pd.DataFrame:
    corr = reference_loss_only(data_dict["corr_long"], data_dict)
    grouped = (
        corr.groupby(["model", "parameter", "attribute"], as_index=False)
        .agg(
            mean_abs_rho=("abs_rho", "mean"),
            variance_r2=("spearman_r2", "var"),
            range_r2=("spearman_r2", lambda values: float(np.nanmax(values) - np.nanmin(values))),
        )
    )
    summary = (
        grouped.groupby("model", as_index=False)
        .agg(
            mean_abs_rho=("mean_abs_rho", "mean"),
            mean_variance_r2=("variance_r2", "mean"),
            mean_range_r2=("range_r2", "mean"),
        )
        .sort_values("model")
        .reset_index(drop=True)
    )
    summary["model"] = summary["model"].map(pretty_model_name)
    return summary


def _cross_loss_retention_summary(data_dict: dict[str, Any]) -> pd.DataFrame:
    corr = data_dict["corr_long"]
    rows = []
    for model in data_dict["model_order"]:
        subset = corr.loc[corr["model"] == model]
        grouped = (
            subset.groupby(["loss", "parameter", "attribute"], as_index=False)
            .agg(mean_abs_rho=("abs_rho", "mean"), mean_rho=("spearman_rho", "mean"))
            .sort_values(["loss", "parameter", "mean_abs_rho"], ascending=[True, True, False])
            .groupby(["loss", "parameter"], as_index=False)
            .head(1)
            .reset_index(drop=True)
        )

        overlap_scores = []
        parameters = sorted(grouped["parameter"].unique())
        losses = [loss for loss in data_dict["loss_order"] if loss in set(grouped["loss"])]
        for parameter in parameters:
            parameter_subset = grouped.loc[grouped["parameter"] == parameter]
            attrs = {row["loss"]: row["attribute"] for _, row in parameter_subset.iterrows()}
            for idx, loss_a in enumerate(losses):
                for loss_b in losses[idx + 1 :]:
                    if loss_a in attrs and loss_b in attrs:
                        overlap_scores.append(float(attrs[loss_a] == attrs[loss_b]))

        rows.append(
            {
                "model": pretty_model_name(model),
                "mean_dominant_attr_retention": float(np.mean(overlap_scores)) if overlap_scores else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _distributional_dominant_attr_changes(data_dict: dict[str, Any]) -> pd.DataFrame:
    corr = data_dict["corr_long"]
    subset = corr.loc[corr["model"] == "distributional"]
    dominant = (
        subset.groupby(["loss", "parameter", "attribute"], as_index=False)
        .agg(mean_abs_rho=("abs_rho", "mean"), mean_rho=("spearman_rho", "mean"))
        .sort_values(["loss", "parameter", "mean_abs_rho"], ascending=[True, True, False])
        .groupby(["loss", "parameter"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    rows = []
    for parameter in sorted(dominant["parameter"].unique()):
        param_subset = dominant.loc[dominant["parameter"] == parameter]
        attrs = []
        for loss in data_dict["loss_order"]:
            value = param_subset.loc[param_subset["loss"] == loss, "attribute"]
            attrs.append(str(value.iloc[0]) if not value.empty else "n/a")
        rows.append(
            {
                "parameter": pretty_parameter_name(parameter),
                "NSE": attrs[0] if len(attrs) > 0 else "n/a",
                "LogNSE": attrs[1] if len(attrs) > 1 else "n/a",
                "HybridNSE": attrs[2] if len(attrs) > 2 else "n/a",
                "all_same": len(set(attrs)) == 1,
            }
        )
    return pd.DataFrame(rows)


def _top_distributional_relationships(data_dict: dict[str, Any], value_column: str) -> pd.DataFrame:
    corr = reference_loss_only(data_dict["corr_long"], data_dict)
    subset = corr.loc[corr["model"] == "distributional"]
    top_pairs = top_pairs_by_abs_rho(subset, top_k=8)
    rows = []
    for parameter, attribute in top_pairs:
        match = subset.loc[
            (subset["parameter"] == parameter) & (subset["attribute"] == attribute),
            "spearman_rho",
        ]
        if match.empty:
            continue
        rows.append(
            {
                "parameter": pretty_parameter_name(parameter),
                "attribute": attribute,
                "mean_spearman_rho": float(match.mean()),
            }
        )
    return pd.DataFrame(rows)


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    render = frame.copy()
    for column in render.columns:
        render[column] = render[column].map(
            lambda value: f"{value:.4f}" if isinstance(value, (float, np.floating)) and np.isfinite(value) else value
        )
    headers = [str(column) for column in render.columns]
    separator = ["---"] * len(headers)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(separator) + " |"]
    for _, row in render.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in render.columns) + " |")
    return "\n".join(rows)


def render_markdown_report(
    data_dict: dict[str, Any],
    generated_figures: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    output_path = Path(output_dir) / "analysis_results.md"
    reference_loss = pretty_loss_name(data_dict["reference_loss"])
    metric_summary = _reference_metric_summary(data_dict)
    stability_summary = _reference_stability_summary(data_dict)
    corr_summary = _reference_corr_stability_summary(data_dict)
    loss_retention = _cross_loss_retention_summary(data_dict)
    top_relationships = _top_distributional_relationships(data_dict, "mean")
    dominant_attr_changes = _distributional_dominant_attr_changes(data_dict)

    lines = [
        "# WRR Figure Analysis Results",
        "",
        "## Run Scope",
        "",
        f"- Outputs root: `{data_dict['outputs_root']}`",
        f"- Analysis root: `{data_dict['analysis_root']}`",
        f"- Models: {', '.join(pretty_model_name(model) for model in data_dict['model_order'])}",
        f"- Losses: {', '.join(pretty_loss_name(loss) for loss in data_dict['loss_order'])}",
        f"- Seeds: {', '.join(str(seed) for seed in data_dict['seed_order'])}",
        f"- Reference loss for Figs 1-5 and 9-13: `{reference_loss}`",
        f"- Configured loss in YAML: `{pretty_loss_name(data_dict['configured_reference_loss'])}`",
        f"- Parameter-capable runs: `{len(data_dict['parameter_runs'])}`",
        f"- Metric-capable runs: `{len(data_dict['metric_runs'])}`",
        "",
    ]

    if data_dict["skipped_metric_runs"]:
        lines.extend(
            [
                "## Incomplete Metric Runs",
                "",
                "The following runs had checkpoints but were missing evaluation artifacts, so they were excluded from metric-based figures and summaries:",
                "",
            ]
        )
        for run in data_dict["skipped_metric_runs"]:
            lines.append(f"- `{run.model}/{run.loss}/seed_{run.seed}`")
        lines.append("")

    lines.extend(
        [
            "## Overall Predictive Performance",
            "",
            _frame_to_markdown(metric_summary),
            "",
            "## Cross-seed Parameter Stability",
            "",
            _frame_to_markdown(stability_summary),
            "",
            "## Cross-seed Correlation Stability",
            "",
            _frame_to_markdown(corr_summary),
            "",
            "## Cross-loss Dominant-Attribute Retention",
            "",
            _frame_to_markdown(loss_retention),
            "",
            "## Distributional Dominant Attribute by Loss",
            "",
            _frame_to_markdown(dominant_attr_changes),
            "",
            "## Top Distributional Attribute-Parameter Relationships",
            "",
            _frame_to_markdown(top_relationships),
            "",
            "## Generated Figures",
            "",
        ]
    )

    for key in sorted(item for item in generated_figures if item.isdigit()):
        payload = generated_figures[key]
        figure_name = payload["name"]
        png_path = payload["outputs"].get("png")
        pdf_path = payload["outputs"].get("pdf")
        lines.extend(
            [
                f"- `{key}` `{figure_name}`",
                f"  png: `{png_path}`",
                f"  pdf: `{pdf_path}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Manifest",
            "",
            f"- `{generated_figures['manifest']}`",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
