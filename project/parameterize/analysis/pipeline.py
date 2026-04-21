"""Orchestrated execution helpers for the numbered analysis scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from project.parameterize.analysis.common import (
    DEFAULT_CORR_METHODS,
    DEFAULT_METRICS,
    DEFAULT_TOP_K,
    load_analysis_data,
    save_frame,
)
from project.parameterize.analysis.correlation_analysis import (
    aggregate_correlation_exports,
    build_correlation_long,
    compute_loss_correlation_stability,
    compute_seed_correlation_stability,
    write_correlation_outputs,
)
from project.parameterize.analysis.metrics_analysis import summarize_metrics, write_metric_outputs
from project.parameterize.analysis.parameter_analysis import (
    compute_cross_loss_parameter_variance,
    compute_seed_parameter_variance,
    summarize_cross_loss_parameter_variance,
    summarize_seed_parameter_variance,
    write_parameter_outputs,
)
from project.parameterize.analysis.reporting import build_master_report


def run_collect_run_tables(data: dict[str, Any]) -> dict[str, Path]:
    out_dir = data["stability_output_dirs"]["tables"]
    manifest = [
        {"model": run.model, "loss": run.loss, "seed": run.seed, "run_dir": str(run.run_dir)}
        for run in data["runs"]
    ]
    paths = {
        "run_manifest": save_frame(
            data["metrics_long"][["model", "loss", "seed"]].drop_duplicates().reset_index(drop=True),
            out_dir / "metric_run_manifest.csv",
        ),
        "run_inventory": save_frame(
            pd.DataFrame(manifest), out_dir / "run_inventory.csv"
        ),
        "metrics_long": save_frame(data["metrics_long"], out_dir / "metrics_long.csv"),
    }
    if "params_long" in data:
        paths["parameter_run_manifest"] = save_frame(
            data["params_long"][["model", "loss", "seed", "sample_count"]].drop_duplicates().reset_index(drop=True),
            out_dir / "parameter_run_manifest.csv",
        )
        paths["params_long"] = save_frame(data["params_long"], out_dir / "params_long.csv")
    if "attributes" in data:
        paths["attributes"] = save_frame(data["attributes"], out_dir / "basin_attributes.csv")
    return paths


def run_metric_accuracy(data: dict[str, Any], metric_names: tuple[str, ...] = DEFAULT_METRICS):
    summary = summarize_metrics(data["metrics_long"], metric_names=metric_names)
    paths = write_metric_outputs(summary, data["stability_output_dirs"]["tables"])
    return summary, paths


def run_parameter_seed_variance(data: dict[str, Any]):
    variance_long = compute_seed_parameter_variance(data["params_long"], data["parameter_bounds"])
    variance_summary = summarize_seed_parameter_variance(variance_long)
    cross_loss_pooled = compute_cross_loss_parameter_variance(
        data["params_long"], data["parameter_bounds"], mode="pooled"
    )
    cross_loss_seedfirst = compute_cross_loss_parameter_variance(
        data["params_long"], data["parameter_bounds"], mode="seed-first"
    )
    cross_loss_outputs = summarize_cross_loss_parameter_variance(cross_loss_pooled)
    cross_loss_outputs["cross_loss_parameter_variance_seedfirst_long"] = cross_loss_seedfirst
    cross_loss_outputs["cross_loss_parameter_variance_pooled_long"] = cross_loss_pooled
    paths = write_parameter_outputs(
        variance_long=variance_long,
        variance_summary=variance_summary,
        cross_loss_outputs=cross_loss_outputs,
        output_dir=data["stability_output_dirs"]["parameter_variance"],
    )
    return variance_long, variance_summary, cross_loss_outputs, paths


def run_correlation_analysis(
    data: dict[str, Any],
    corr_methods: tuple[str, ...] = DEFAULT_CORR_METHODS,
    top_k: int = DEFAULT_TOP_K,
):
    base_corr_tables = build_correlation_long(data["params_long"], data["attributes"], methods=corr_methods)
    corr_tables = {
        method: table.assign(method=method)
        for method, table in base_corr_tables.items()
    }
    combined = pd.concat(corr_tables.values(), ignore_index=True)
    seed_outputs = compute_seed_correlation_stability(combined, top_k=top_k)
    loss_outputs = compute_loss_correlation_stability(combined, top_k=top_k)
    aggregate_outputs = aggregate_correlation_exports(combined)
    paths = write_correlation_outputs(
        corr_tables=corr_tables,
        seed_outputs=seed_outputs,
        loss_outputs=loss_outputs,
        aggregate_outputs=aggregate_outputs,
        output_dir=data["stability_output_dirs"]["correlation_summaries"],
    )
    return corr_tables, seed_outputs, loss_outputs, aggregate_outputs, paths


def run_all(
    config_path: str,
    outputs_root: str | None = None,
    analysis_root: str | None = None,
    device: str = "cpu",
    metric_names: tuple[str, ...] = DEFAULT_METRICS,
    corr_methods: tuple[str, ...] = DEFAULT_CORR_METHODS,
    top_k: int = DEFAULT_TOP_K,
    parameter_csv: str | None = None,
    attribute_csv: str | None = None,
):
    data = load_analysis_data(
        config_path=config_path,
        outputs_root=outputs_root,
        analysis_root=analysis_root,
        device=device,
        parameter_csv=parameter_csv,
        attribute_csv=attribute_csv,
    )
    collect_paths = run_collect_run_tables(data)
    metric_summary, metric_paths = run_metric_accuracy(data, metric_names=metric_names)
    parameter_paths = {}
    corr_paths = {}
    variance_long = None
    corr_tables = None
    report_path = None

    if "params_long" in data:
        variance_long, variance_summary, cross_loss_outputs, parameter_paths = run_parameter_seed_variance(data)
        if "attributes" in data:
            corr_tables, seed_outputs, loss_outputs, aggregate_outputs, corr_paths = run_correlation_analysis(
                data,
                corr_methods=corr_methods,
                top_k=top_k,
            )
            report_path = build_master_report(
                analysis_root=data["stability_analysis_root"],
                metric_summary=metric_summary,
                parameter_summary=variance_summary,
                cross_loss_parameter_summary=cross_loss_outputs,
                seed_corr_summary=seed_outputs,
                loss_corr_summary=loss_outputs,
                aggregate_corr_summary=aggregate_outputs,
            )
    return {
        "data": data,
        "collect_paths": collect_paths,
        "metric_paths": metric_paths,
        "parameter_paths": parameter_paths,
        "correlation_paths": corr_paths,
        "report_path": report_path,
        "variance_long": variance_long,
        "corr_tables": corr_tables,
    }
