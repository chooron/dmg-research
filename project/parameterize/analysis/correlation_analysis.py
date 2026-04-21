"""Correlation-matrix construction and stability analysis.

This module produces:
- per-run correlation matrices in csv/npz
- per-method long tables
- cross-seed and cross-loss stability summaries for strong relationships
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from project.parameterize.analysis.common import (
    correlation_value,
    frame_to_markdown,
    pairwise_mean_abs_diff,
    save_frame,
    save_npz,
    write_markdown,
)


def build_correlation_long(
    params_long: pd.DataFrame,
    attributes: pd.DataFrame,
    methods: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    merged = params_long.merge(attributes, on="basin_id", how="inner")
    attribute_columns = [column for column in attributes.columns if column != "basin_id"]
    rows_by_method: dict[str, list[dict[str, object]]] = {method: [] for method in methods}

    for (model, loss, seed, parameter), subset in merged.groupby(["model", "loss", "seed", "parameter"]):
        for attribute in attribute_columns:
            for method in methods:
                corr_value, p_value = correlation_value(subset["mean"], subset[attribute], method=method)
                rows_by_method[method].append(
                    {
                        "model": model,
                        "loss": loss,
                        "seed": int(seed),
                        "parameter": parameter,
                        "attribute": attribute,
                        "corr": corr_value,
                        "p_value": p_value,
                        "abs_corr": float(abs(corr_value)) if not np.isnan(corr_value) else np.nan,
                    }
                )

    return {method: pd.DataFrame(rows) for method, rows in rows_by_method.items()}


def export_correlation_matrices(
    corr_tables: dict[str, pd.DataFrame],
    output_dir: Path,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    manifest_rows: list[dict[str, object]] = []
    for method, table in corr_tables.items():
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        paths[f"{method}_long"] = save_frame(table, method_dir / f"correlation_long__{method}.csv")
        for (model, loss, seed), subset in table.groupby(["model", "loss", "seed"]):
            matrix = subset.pivot(index="attribute", columns="parameter", values="corr")
            stem = f"{model}__{loss}__seed_{seed}__{method}"
            csv_path = method_dir / f"{stem}.csv"
            npz_path = method_dir / f"{stem}.npz"
            save_frame(matrix.reset_index(), csv_path)
            save_npz(
                matrix=matrix.to_numpy(),
                row_names=list(matrix.index),
                col_names=list(matrix.columns),
                path=npz_path,
            )
            manifest_rows.append(
                {
                    "method": method,
                    "model": model,
                    "loss": loss,
                    "seed": int(seed),
                    "csv_path": str(csv_path),
                    "npz_path": str(npz_path),
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    paths["manifest"] = save_frame(manifest, output_dir / "correlation_matrix_manifest.csv")
    return paths


def _topk_pairs_per_parameter(
    corr_long: pd.DataFrame,
    top_k: int,
    group_cols: tuple[str, ...],
) -> pd.DataFrame:
    ranked = (
        corr_long.groupby(list(group_cols) + ["parameter", "attribute"], as_index=False)
        .agg(mean_abs_corr=("abs_corr", "mean"))
        .sort_values(list(group_cols) + ["parameter", "mean_abs_corr"], ascending=[True] * len(group_cols) + [True, False])
    )
    top = (
        ranked.groupby(list(group_cols) + ["parameter"], as_index=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    return top


def compute_seed_correlation_stability(
    corr_long: pd.DataFrame,
    top_k: int,
) -> dict[str, pd.DataFrame]:
    top_pairs = _topk_pairs_per_parameter(corr_long, top_k=top_k, group_cols=("method", "model", "loss"))
    selected = corr_long.merge(
        top_pairs[["method", "model", "loss", "parameter", "attribute"]],
        on=["method", "model", "loss", "parameter", "attribute"],
        how="inner",
    )
    pair_stats = (
        selected.groupby(["method", "model", "loss", "parameter", "attribute"], as_index=False)
        .agg(
            mean_corr=("corr", "mean"),
            variance_corr=("corr", lambda values: float(np.var(values, ddof=0))),
            range_corr=("corr", lambda values: float(np.nanmax(values) - np.nanmin(values))),
            mean_abs_seed_diff=("corr", pairwise_mean_abs_diff),
            mean_abs_corr=("abs_corr", "mean"),
        )
    )
    summary = (
        pair_stats.groupby(["method", "model", "loss"], as_index=False)
        .agg(
            mean_variance_corr=("variance_corr", "mean"),
            mean_range_corr=("range_corr", "mean"),
            mean_abs_seed_diff=("mean_abs_seed_diff", "mean"),
            mean_abs_corr=("mean_abs_corr", "mean"),
        )
    )
    return {
        "correlation_seed_stability_pairs": pair_stats,
        "correlation_seed_stability_summary": summary,
    }


def compute_loss_correlation_stability(
    corr_long: pd.DataFrame,
    top_k: int,
) -> dict[str, pd.DataFrame]:
    top_pairs = _topk_pairs_per_parameter(corr_long, top_k=top_k, group_cols=("method", "model"))
    selected = corr_long.merge(
        top_pairs[["method", "model", "parameter", "attribute"]],
        on=["method", "model", "parameter", "attribute"],
        how="inner",
    )

    pooled = (
        selected.groupby(["method", "model", "parameter", "attribute"], as_index=False)
        .agg(
            pooled_variance_corr=("corr", lambda values: float(np.var(values, ddof=0))),
            pooled_range_corr=("corr", lambda values: float(np.nanmax(values) - np.nanmin(values))),
            pooled_abs_diff=("corr", pairwise_mean_abs_diff),
            mean_abs_corr=("abs_corr", "mean"),
        )
    )

    seed_first = (
        selected.groupby(["method", "model", "seed", "parameter", "attribute"], as_index=False)
        .agg(seed_mean_corr=("corr", "mean"))
        .groupby(["method", "model", "parameter", "attribute"], as_index=False)
        .agg(
            seedfirst_variance_corr=("seed_mean_corr", lambda values: float(np.var(values, ddof=0))),
            seedfirst_range_corr=("seed_mean_corr", lambda values: float(np.nanmax(values) - np.nanmin(values))),
            seedfirst_abs_diff=("seed_mean_corr", pairwise_mean_abs_diff),
        )
    )

    summary = (
        pooled.groupby(["method", "model"], as_index=False)
        .agg(
            mean_pooled_variance_corr=("pooled_variance_corr", "mean"),
            mean_pooled_range_corr=("pooled_range_corr", "mean"),
            mean_pooled_abs_diff=("pooled_abs_diff", "mean"),
            mean_abs_corr=("mean_abs_corr", "mean"),
        )
    )
    return {
        "correlation_loss_stability_pooled": pooled,
        "correlation_loss_stability_seedfirst": seed_first,
        "correlation_loss_stability_summary": summary,
    }


def aggregate_correlation_exports(corr_long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    mean_table = (
        corr_long.groupby(["method", "model", "loss", "parameter", "attribute"], as_index=False)
        .agg(
            mean_corr=("corr", "mean"),
            std_corr=("corr", lambda values: float(np.std(values, ddof=0))),
            variance_corr=("corr", lambda values: float(np.var(values, ddof=0))),
            mean_abs_corr=("abs_corr", "mean"),
        )
    )
    top_relationships = (
        mean_table.sort_values(["method", "model", "loss", "mean_abs_corr"], ascending=[True, True, True, False])
        .groupby(["method", "model", "loss"], as_index=False)
        .head(20)
        .reset_index(drop=True)
    )
    return {
        "correlation_mean_std_var": mean_table,
        "correlation_top_relationships": top_relationships,
    }


def write_correlation_outputs(
    corr_tables: dict[str, pd.DataFrame],
    seed_outputs: dict[str, pd.DataFrame],
    loss_outputs: dict[str, pd.DataFrame],
    aggregate_outputs: dict[str, pd.DataFrame],
    output_dir: Path,
) -> dict[str, Path]:
    paths = export_correlation_matrices(corr_tables, output_dir / "matrices")
    for name, frame in seed_outputs.items():
        paths[name] = save_frame(frame, output_dir / f"{name}.csv")
    for name, frame in loss_outputs.items():
        paths[name] = save_frame(frame, output_dir / f"{name}.csv")
    for name, frame in aggregate_outputs.items():
        paths[name] = save_frame(frame, output_dir / f"{name}.csv")

    report = write_markdown(
        output_dir / "correlation_stability_report.md",
        title="Correlation Stability Summary",
        sections=[
            ("Seed Stability Summary", frame_to_markdown(seed_outputs["correlation_seed_stability_summary"])),
            ("Loss Stability Summary", frame_to_markdown(loss_outputs["correlation_loss_stability_summary"])),
        ],
    )
    paths["report"] = report
    return paths

