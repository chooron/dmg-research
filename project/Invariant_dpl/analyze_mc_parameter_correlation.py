#!/usr/bin/env python3
"""Generate Spearman rankings and heatmaps for attribute-parameter analysis."""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ID_COLUMNS = {"basin_id", "effective_cluster", "gauge_cluster", "dataset_split"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Path to mc_attribute_parameter_table_*.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for ranking CSVs and heatmaps. Defaults to the input directory.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of strongest absolute-Spearman pairs to save separately.",
    )
    parser.add_argument(
        "--figsize",
        default="12,10",
        help="Heatmap figure size as 'width,height'.",
    )
    return parser.parse_args()


def _parse_figsize(value: str) -> tuple[float, float]:
    width, height = value.split(",", maxsplit=1)
    return float(width), float(height)


def _attribute_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    excluded = set()
    for col in df.columns:
        if col in ID_COLUMNS:
            excluded.add(col)
        if col.startswith("norm_"):
            excluded.add(col)
        if col.endswith("_mean") or col.endswith("_var"):
            excluded.add(col)
    return sorted(numeric_cols - excluded)


def _parameter_columns(df: pd.DataFrame, suffix: str) -> list[str]:
    cols = []
    for col in df.columns:
        if not col.endswith(suffix):
            continue
        if col.startswith("norm_"):
            continue
        if col.startswith("par") or col.startswith("route_"):
            cols.append(col)
    return sorted(cols)


def _spearman_matrix(
    df: pd.DataFrame,
    attribute_cols: list[str],
    parameter_cols: list[str],
) -> pd.DataFrame:
    if not attribute_cols or not parameter_cols:
        raise ValueError("Attribute columns and parameter columns must both be non-empty.")
    corr = df[attribute_cols + parameter_cols].corr(method="spearman")
    return corr.loc[attribute_cols, parameter_cols]


def _ranking_from_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    records = []
    for attribute in matrix.index:
        for parameter in matrix.columns:
            rho = float(matrix.loc[attribute, parameter])
            if np.isnan(rho):
                continue
            records.append(
                {
                    "attribute": attribute,
                    "parameter": parameter,
                    "spearman_rho": rho,
                    "abs_spearman_rho": abs(rho),
                }
            )
    ranking = pd.DataFrame(records).sort_values(
        ["abs_spearman_rho", "spearman_rho"],
        ascending=[False, False],
        ignore_index=True,
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking[["rank", "attribute", "parameter", "spearman_rho", "abs_spearman_rho"]]


def _topk_per_parameter(matrix: pd.DataFrame, top_k: int) -> pd.DataFrame:
    frames = []
    for parameter in matrix.columns:
        series = matrix[parameter].dropna()
        ordered = series.reindex(series.abs().sort_values(ascending=False).index).head(top_k)
        frame = pd.DataFrame(
            {
                "parameter": parameter,
                "attribute": ordered.index,
                "spearman_rho": ordered.values,
                "abs_spearman_rho": np.abs(ordered.values),
                "parameter_rank": np.arange(1, len(ordered) + 1),
            }
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=["parameter", "attribute", "spearman_rho", "abs_spearman_rho", "parameter_rank"]
        )
    return pd.concat(frames, ignore_index=True)


def _plot_heatmap(
    matrix: pd.DataFrame,
    output_path: str,
    title: str,
    figsize: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    ax.set_xlabel("Model parameters")
    ax.set_ylabel("Static attributes")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Spearman rho")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_outputs(
    matrix: pd.DataFrame,
    output_dir: str,
    stem: str,
    top_k: int,
    figsize: tuple[float, float],
) -> None:
    matrix_path = os.path.join(output_dir, f"{stem}_matrix.csv")
    ranking_path = os.path.join(output_dir, f"{stem}_ranking.csv")
    topk_path = os.path.join(output_dir, f"{stem}_top{top_k}.csv")
    per_parameter_path = os.path.join(output_dir, f"{stem}_top{top_k}_per_parameter.csv")
    heatmap_path = os.path.join(output_dir, f"{stem}_heatmap.png")

    ranking = _ranking_from_matrix(matrix)
    topk = ranking.head(top_k)
    per_parameter = _topk_per_parameter(matrix, top_k)

    matrix.to_csv(matrix_path)
    ranking.to_csv(ranking_path, index=False)
    topk.to_csv(topk_path, index=False)
    per_parameter.to_csv(per_parameter_path, index=False)
    _plot_heatmap(matrix, heatmap_path, stem.replace("_", " ").title(), figsize)


def _print_preview(title: str, ranking: pd.DataFrame, top_k: int) -> None:
    print(title)
    preview = ranking.head(top_k)
    if preview.empty:
        print("No valid correlations found.")
        return
    print(preview.to_string(index=False))
    print()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    output_dir = args.output_dir or os.path.dirname(args.input) or "."
    os.makedirs(output_dir, exist_ok=True)

    attribute_cols = _attribute_columns(df)
    mean_cols = _parameter_columns(df, "_mean")
    var_cols = _parameter_columns(df, "_var")
    figsize = _parse_figsize(args.figsize)

    mean_matrix = _spearman_matrix(df, attribute_cols, mean_cols)
    var_matrix = _spearman_matrix(df, attribute_cols, var_cols)

    _write_outputs(mean_matrix, output_dir, "spearman_parameter_mean", args.top_k, figsize)
    _write_outputs(var_matrix, output_dir, "spearman_parameter_variance", args.top_k, figsize)

    mean_ranking = _ranking_from_matrix(mean_matrix)
    var_ranking = _ranking_from_matrix(var_matrix)

    print(f"Input table: {args.input}")
    print(f"Attributes: {len(attribute_cols)} | mean parameters: {len(mean_cols)} | variance parameters: {len(var_cols)}")
    print()
    _print_preview("Top mean correlations", mean_ranking, min(args.top_k, 10))
    _print_preview("Top variance correlations", var_ranking, min(args.top_k, 10))
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
