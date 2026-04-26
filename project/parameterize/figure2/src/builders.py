"""Revised figure builders and table exports for the figure2 manuscript suite."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from sklearn.metrics.pairwise import cosine_similarity

from project.parameterize.figure2.src.data_registry import FigureDataRegistry, MissingFigureDataError
from project.parameterize.figure2.src.figure_utils import (
    QcCollector,
    apply_panel_letters,
    basin_polygon_map,
    build_asymmetric_gridspec,
    categorize_stability,
    clean_attribute_name,
    clean_model_name,
    clean_parameter_name,
    compact_table_panel,
    label_axes,
    make_figure,
    make_shared_colorbar,
    save_figure,
    write_table_outputs,
)
from project.parameterize.figure2.src.metadata import GRADIENT_ATTRIBUTES, MODEL_ORDER


def _model_display(model: str) -> str:
    return clean_model_name(model)


def _model_color(registry: FigureDataRegistry, model: str) -> str:
    return registry.palette["models"][_model_display(model)]


def _diverging_cmap(registry: FigureDataRegistry):
    return mcolors.LinearSegmentedColormap.from_list("revised_diverging", registry.palette["diverging_signed"])


def _stability_cmap(registry: FigureDataRegistry):
    return mcolors.LinearSegmentedColormap.from_list("revised_stability", registry.palette["sequential_stability"])


def _uncertainty_cmap(registry: FigureDataRegistry):
    return mcolors.LinearSegmentedColormap.from_list("revised_uncertainty", registry.palette["sequential_uncertainty"])


def _figure_size(registry: FigureDataRegistry, figure_id: str) -> tuple[float, float]:
    width_mm, height_mm = registry.style["figure"]["figure_dimensions_mm"][figure_id]
    return float(width_mm), float(height_mm)


def _reference_metrics(registry: FigureDataRegistry) -> pd.DataFrame:
    frame = registry.require_columns(
        "metrics_long",
        ["basin_id", "model", "loss", "seed", "nse", "kge", "bias_abs", "pbias_abs"],
    )
    return registry.filter_reference(frame)


def _reference_params(registry: FigureDataRegistry) -> pd.DataFrame:
    frame = registry.require_columns(
        "params_long",
        ["basin_id", "model", "loss", "seed", "parameter", "mean", "std"],
    )
    return registry.filter_reference(frame)


def _attributes(registry: FigureDataRegistry) -> pd.DataFrame:
    return registry.require_columns("basin_attributes", ["basin_id"] + registry.attribute_order)


def _focus_parameters(registry: FigureDataRegistry) -> list[str]:
    return [value for value in ["parBETA", "parFC", "parPERC", "parUZL", "parCFR", "parCWH"] if value in registry.parameter_order]


def _focus_attributes(registry: FigureDataRegistry) -> list[str]:
    return [value for value in ["aridity", "frac_snow", "slope_mean", "pet_mean", "soil_conductivity", "soil_depth_pelletier"] if value in registry.attribute_order]


def _family_for_parameter(parameter: str) -> str:
    if parameter in {"parFC", "parLP", "parBETA", "parPERC"}:
        return "storage_recharge"
    if parameter in {"parK0", "parK1", "parK2", "parUZL", "route_a", "route_b"}:
        return "routing_runoff"
    return "snow_cold"


def _robust_quantile_limit(values: np.ndarray | pd.Series, quantile: float = 0.98, floor: float = 0.05) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return floor
    return max(float(np.nanquantile(np.abs(arr), quantile)), floor)


def _clean_output_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    if "model" in cleaned.columns:
        cleaned["model"] = cleaned["model"].map(clean_model_name)
    for column in list(cleaned.columns):
        if "parameter" in column:
            cleaned[column] = cleaned[column].map(clean_parameter_name)
        if "attribute" in column:
            cleaned[column] = cleaned[column].map(clean_attribute_name)
    renamed = {}
    for column in cleaned.columns:
        if isinstance(column, str) and len(column) > 3 and column.startswith("par") and column[3].isupper() and column[3:] not in cleaned.columns:
            renamed[column] = clean_parameter_name(column)
    if renamed:
        cleaned = cleaned.rename(columns=renamed)
    return cleaned


def _complete_figure(
    registry: FigureDataRegistry,
    figure_id: str,
    fig: plt.Figure,
    output_dir: Path,
    qc: QcCollector,
    formats: tuple[str, ...],
) -> dict[str, Path]:
    stem = FIGURE_SPECS[figure_id]["stem"]
    outputs = save_figure(fig, stem, output_dir, registry.style, formats=formats)
    qc.add_entry(figure_id, FIGURE_SPECS[figure_id]["title"], fig, outputs)
    plt.close(fig)
    return outputs


def _corr_focus_matrix(registry: FigureDataRegistry, model: str, value_column: str) -> pd.DataFrame:
    corr = registry.require_columns(
        "correlation_mean_std_var",
        ["method", "model", "loss", "parameter", "attribute", value_column],
    )
    corr = corr.loc[
        (corr["method"] == "spearman")
        & (corr["loss"] == registry.reference_loss)
        & (corr["model"] == model)
        & (corr["parameter"].isin(_focus_parameters(registry)))
        & (corr["attribute"].isin(_focus_attributes(registry)))
    ].copy()
    matrix = corr.pivot_table(index="attribute", columns="parameter", values=value_column, aggfunc="mean")
    return matrix.reindex(index=_focus_attributes(registry), columns=_focus_parameters(registry))


def _stacked_focus_matrix(registry: FigureDataRegistry, source: pd.DataFrame, value_column: str) -> pd.DataFrame:
    blocks = []
    for model in MODEL_ORDER:
        subset = source.loc[source["model"] == model].pivot_table(
            index="attribute",
            columns="parameter",
            values=value_column,
            aggfunc="mean",
        )
        subset = subset.reindex(index=_focus_attributes(registry), columns=_focus_parameters(registry))
        subset.index = pd.MultiIndex.from_product([[clean_model_name(model)], subset.index])
        blocks.append(subset)
    return pd.concat(blocks, axis=0)


def _distributional_mean_table(registry: FigureDataRegistry) -> pd.DataFrame:
    params = _reference_params(registry)
    params = params.loc[params["model"] == "distributional"].copy()
    grouped = params.groupby(["basin_id", "parameter"], as_index=False)["mean"].mean()
    return grouped.pivot_table(index="basin_id", columns="parameter", values="mean", aggfunc="mean").reset_index()


def _distributional_std_table(registry: FigureDataRegistry) -> pd.DataFrame:
    params = _reference_params(registry)
    params = params.loc[params["model"] == "distributional"].copy()
    grouped = params.groupby(["basin_id", "parameter"], as_index=False)["std"].mean()
    return grouped.pivot_table(index="basin_id", columns="parameter", values="std", aggfunc="mean").reset_index()


def _set_tick_text(
    ax: plt.Axes,
    xlabels: list[str] | None = None,
    ylabels: list[str] | None = None,
    xrotation: float = 0,
    yrotation: float = 0,
    ysize: float | None = None,
) -> None:
    if xlabels is not None:
        ax.set_xticks(np.arange(len(xlabels)) + 0.5)
        ax.set_xticklabels(xlabels, rotation=xrotation, ha="right" if xrotation else "center")
    if ylabels is not None:
        ax.set_yticks(np.arange(len(ylabels)) + 0.5)
        ax.set_yticklabels(ylabels, rotation=yrotation)
        if ysize is not None:
            ax.tick_params(axis="y", labelsize=ysize)


def _text_rows_panel(ax: plt.Axes, title: str, lines: list[str], style: dict[str, Any]) -> None:
    ax.axis("off")
    label_axes(ax, title, None, style)
    y = 0.92
    for line in lines:
        ax.text(0.0, y, line, ha="left", va="top", fontsize=style["figure"]["table_text_size_pt"])
        y -= 0.11


def generate_table_outputs(registry: FigureDataRegistry, table_dir: Path) -> dict[str, dict[str, Path]]:
    outputs: dict[str, dict[str, Path]] = {}

    metrics = _reference_metrics(registry)
    table01_rows = []
    for model in MODEL_ORDER:
        subset = metrics.loc[metrics["model"] == model]
        table01_rows.append(
            {
                "model": clean_model_name(model),
                "NSE median": round(float(subset["nse"].median()), 3),
                "NSE IQR": round(float(subset["nse"].quantile(0.75) - subset["nse"].quantile(0.25)), 3),
                "KGE median": round(float(subset["kge"].median()), 3),
                "KGE IQR": round(float(subset["kge"].quantile(0.75) - subset["kge"].quantile(0.25)), 3),
                "|bias| median": round(float(subset["bias_abs"].median()), 3),
                "|pbias| median": round(float(subset["pbias_abs"].median()), 3),
            }
        )
    outputs["Table01_performance_summary"] = write_table_outputs(
        "Table01_performance_summary",
        pd.DataFrame(table01_rows),
        table_dir,
        index=False,
    )

    relationships = registry.require_columns(
        "relationship_classes",
        ["model", "parameter", "attribute", "relationship_class", "mean_abs_corr", "seed_stable", "loss_stable"],
    )
    robust = (
        relationships.loc[relationships["relationship_class"] == "robust"]
        .sort_values(["mean_abs_corr", "model"], ascending=[False, True])
        .head(18)
        .loc[:, ["model", "parameter", "attribute", "mean_abs_corr", "seed_stable", "loss_stable"]]
    )
    robust = _clean_output_frame(robust)
    robust["mean_abs_corr"] = robust["mean_abs_corr"].round(2)
    outputs["Table02_top_robust_relationships"] = write_table_outputs(
        "Table02_top_robust_relationships",
        robust,
        table_dir,
        index=False,
    )

    stability = registry.require_columns(
        "seed_parameter_variance_by_parameter",
        ["model", "loss", "parameter", "mean_variance_unit", "mean_abs_seed_diff"],
    )
    stability = stability.loc[stability["loss"] == registry.reference_loss].copy()
    stability["stability_class"] = categorize_stability(stability["mean_variance_unit"])
    stability["parameter_family"] = stability["parameter"].map(_family_for_parameter)
    stability = stability.sort_values(["parameter_family", "parameter", "model"]).copy()
    stability = _clean_output_frame(stability)
    stability["mean_variance_unit"] = stability["mean_variance_unit"].round(4)
    stability["mean_abs_seed_diff"] = stability["mean_abs_seed_diff"].round(3)
    outputs["Table03_parameter_stability_classes"] = write_table_outputs(
        "Table03_parameter_stability_classes",
        stability,
        table_dir,
        index=False,
    )

    similarity = registry.require_columns(
        "results332_matrix_similarity",
        ["model_a", "model_b", "same_model", "same_loss", "same_seed", "matrix_corr_spearman"],
    )
    table04_rows = [
        {"group": "Within model", "mean_similarity": similarity.loc[similarity["same_model"], "matrix_corr_spearman"].mean()},
        {"group": "Cross model", "mean_similarity": similarity.loc[~similarity["same_model"], "matrix_corr_spearman"].mean()},
        {"group": "Within loss", "mean_similarity": similarity.loc[similarity["same_loss"], "matrix_corr_spearman"].mean()},
        {"group": "Cross loss", "mean_similarity": similarity.loc[~similarity["same_loss"], "matrix_corr_spearman"].mean()},
        {"group": "Within seed", "mean_similarity": similarity.loc[similarity["same_seed"], "matrix_corr_spearman"].mean()},
        {"group": "Cross seed", "mean_similarity": similarity.loc[~similarity["same_seed"], "matrix_corr_spearman"].mean()},
    ]
    by_pair = (
        similarity.groupby(["model_a", "model_b"], as_index=False)["matrix_corr_spearman"]
        .mean()
        .assign(
            model_a=lambda df: df["model_a"].map(clean_model_name),
            model_b=lambda df: df["model_b"].map(clean_model_name),
            matrix_corr_spearman=lambda df: df["matrix_corr_spearman"].round(2),
        )
    )
    table04 = pd.concat(
        [
            pd.DataFrame(table04_rows).assign(mean_similarity=lambda df: df["mean_similarity"].round(2)),
            pd.DataFrame([{}]),
            by_pair.rename(columns={"matrix_corr_spearman": "mean_similarity", "model_a": "model A", "model_b": "model B"}),
        ],
        ignore_index=True,
    )
    outputs["Table04_matrix_similarity_summary"] = write_table_outputs(
        "Table04_matrix_similarity_summary",
        table04,
        table_dir,
        index=False,
    )

    representative = registry.require_columns(
        "results343_representative_basins",
        ["basin_id", "group_label", "gradient_attribute", "aridity", "frac_snow", "slope_mean"] +
        [f"{parameter}_mean_unit" for parameter in _focus_parameters(registry)],
    ).copy()
    representative = _clean_output_frame(representative)
    rename_map = {old: clean_parameter_name(old.replace("_mean_unit", "")) for old in representative.columns if old.endswith("_mean_unit")}
    representative = representative.rename(columns=rename_map)
    mean_columns = list(rename_map.values())
    keep_columns = ["basin_id", "group_label", "gradient_attribute", "aridity", "frac_snow", "slope_mean"] + mean_columns
    representative = representative.loc[:, keep_columns].round(3)
    outputs["Table05_basin_archetypes"] = write_table_outputs(
        "Table05_basin_archetypes",
        representative,
        table_dir,
        index=False,
    )

    return outputs


def build_fig01_performance_merged(
    registry: FigureDataRegistry,
    output_dir: Path,
    qc: QcCollector,
    formats: tuple[str, ...],
) -> dict[str, Path]:
    metrics = _reference_metrics(registry)
    spatial = registry.spatial.copy()
    width_mm, height_mm = _figure_size(registry, "01")
    fig = make_figure(width_mm, height_mm)
    outer = build_asymmetric_gridspec(fig, 3, 2, width_ratios=[1.8, 0.95], height_ratios=[0.82, 0.72, 0.58], wspace=0.18, hspace=0.14)
    panel_a = outer[:, 0].subgridspec(1, 2, wspace=0.02)
    ax_a1 = fig.add_subplot(panel_a[0, 0])
    ax_a2 = fig.add_subplot(panel_a[0, 1])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[1, 1])
    ax_d = fig.add_subplot(outer[2, 1])

    basin_means = metrics.groupby(["basin_id", "model"], as_index=False)[["nse", "kge", "bias_abs", "pbias_abs"]].mean()
    nse_pivot = basin_means.pivot_table(index="basin_id", columns="model", values="nse", aggfunc="mean").reset_index()
    diff_gdf = spatial.merge(nse_pivot, on="basin_id", how="inner")
    diff_gdf["dist_vs_det"] = diff_gdf["distributional"] - diff_gdf["deterministic"]
    diff_gdf["dist_vs_mcd"] = diff_gdf["distributional"] - diff_gdf["mc_dropout"]
    vmax = _robust_quantile_limit(diff_gdf[["dist_vs_det", "dist_vs_mcd"]].to_numpy(), floor=0.02)
    cmap = _diverging_cmap(registry)
    plotted = basin_polygon_map(
        ax_a1,
        diff_gdf,
        "dist_vs_det",
        cmap,
        "δdtb - δdtm",
        registry.palette,
        vmin=-vmax,
        vmax=vmax,
        line_width=registry.style["figure"]["map_line_width"],
    )
    basin_polygon_map(
        ax_a2,
        diff_gdf,
        "dist_vs_mcd",
        cmap,
        "δdtb - δmcd",
        registry.palette,
        vmin=-vmax,
        vmax=vmax,
        line_width=registry.style["figure"]["map_line_width"],
    )
    label_axes(ax_a1, "Paired performance-difference maps", "Primary evidence panel", registry.style)
    make_shared_colorbar(fig, plotted, [ax_a1, ax_a2], "Δ NSE", orientation="horizontal", fraction=0.05, pad=0.03)

    metric_specs = [("nse", "NSE", True), ("kge", "KGE", True), ("bias_abs", "|bias|", False)]
    summary_rows = []
    annotations = []
    for metric_name, metric_label, higher_is_better in metric_specs:
        grouped = metrics.groupby("model")[metric_name]
        medians = grouped.median()
        rank = medians.rank(ascending=not higher_is_better, method="dense")
        summary_rows.append([len(MODEL_ORDER) + 1 - rank[model] for model in MODEL_ORDER])
        annotations.append([f"{medians[model]:.2f}" for model in MODEL_ORDER])
    summary = pd.DataFrame(summary_rows, index=[label for _, label, _ in metric_specs], columns=[_model_display(model) for model in MODEL_ORDER])
    sns.heatmap(
        summary,
        ax=ax_b,
        cmap=_stability_cmap(registry),
        annot=np.array(annotations),
        fmt="",
        cbar=False,
        linewidths=0.5,
        linecolor=registry.palette["neutrals"]["divider"],
        square=False,
    )
    label_axes(ax_b, "Model x metric summary", "Exact medians/IQRs moved to Table01", registry.style)
    ax_b.set_xlabel("")
    ax_b.set_ylabel("")

    ordered_basins = _attributes(registry).sort_values(["aridity", "frac_snow"])["basin_id"].tolist()
    carpet_nse = basin_means.pivot_table(index="basin_id", columns="model", values="nse", aggfunc="mean").reindex(ordered_basins)
    carpet_kge = basin_means.pivot_table(index="basin_id", columns="model", values="kge", aggfunc="mean").reindex(ordered_basins)
    carpet_matrix = np.vstack(
        [
            carpet_nse["distributional"] - carpet_nse["deterministic"],
            carpet_nse["distributional"] - carpet_nse["mc_dropout"],
            carpet_kge["distributional"] - carpet_kge["deterministic"],
            carpet_kge["distributional"] - carpet_kge["mc_dropout"],
        ]
    )
    sns.heatmap(
        carpet_matrix,
        ax=ax_c,
        cmap=cmap,
        center=0.0,
        cbar=False,
        xticklabels=False,
        yticklabels=["NSE δdtb-δdtm", "NSE δdtb-δmcd", "KGE δdtb-δdtm", "KGE δdtb-δmcd"],
    )
    label_axes(ax_c, "Environment-ordered carpet", "Basins ordered by aridity then snow fraction", registry.style)
    ax_c.tick_params(axis="y", labelsize=7.2)

    dominance_rows = []
    for metric_name, metric_label, higher_is_better in metric_specs:
        pivot = basin_means.pivot_table(index="basin_id", columns="model", values=metric_name, aggfunc="mean").reindex(ordered_basins)
        winner = pivot.idxmax(axis=1) if higher_is_better else pivot.idxmin(axis=1)
        counts = winner.value_counts(normalize=True)
        dominance_rows.append([counts.get(model, 0.0) for model in MODEL_ORDER])
    dominance = pd.DataFrame(dominance_rows, index=[label for _, label, _ in metric_specs], columns=[_model_display(model) for model in MODEL_ORDER])
    sns.heatmap(
        dominance,
        ax=ax_d,
        cmap=_stability_cmap(registry),
        annot=True,
        fmt=".2f",
        cbar=False,
        linewidths=0.5,
        linecolor=registry.palette["neutrals"]["divider"],
    )
    label_axes(ax_d, "Dominance summary", "Tile fractions of basin-level wins", registry.style)
    ax_d.set_xlabel("")
    ax_d.set_ylabel("")

    apply_panel_letters([ax_a1, ax_b, ax_c, ax_d], registry.style)
    return _complete_figure(registry, "01", fig, output_dir, qc, formats)


def build_fig02_cross_seed_param_stability(
    registry: FigureDataRegistry,
    output_dir: Path,
    qc: QcCollector,
    formats: tuple[str, ...],
) -> dict[str, Path]:
    summary = registry.require_columns(
        "seed_parameter_variance_by_parameter",
        ["model", "loss", "parameter", "mean_variance_unit", "mean_abs_seed_diff"],
    )
    summary = summary.loc[summary["loss"] == registry.reference_loss].copy()
    summary["stability_score"] = 1.0 - (summary["mean_variance_unit"] / summary["mean_variance_unit"].max())
    summary["stability_class"] = categorize_stability(summary["mean_variance_unit"])

    width_mm, height_mm = _figure_size(registry, "02")
    fig = make_figure(width_mm, height_mm)
    outer = build_asymmetric_gridspec(fig, 2, 2, width_ratios=[1.55, 0.95], height_ratios=[0.9, 0.72], wspace=0.22, hspace=0.18)
    ax_a = fig.add_subplot(outer[:, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[1, 1])

    heat = summary.pivot_table(index="parameter", columns="model", values="stability_score", aggfunc="mean")
    heat = heat.reindex(index=registry.parameter_order, columns=MODEL_ORDER)
    sns.heatmap(
        heat,
        ax=ax_a,
        cmap=_stability_cmap(registry),
        vmin=0.0,
        vmax=1.0,
        linewidths=0.35,
        linecolor=registry.palette["neutrals"]["divider"],
        cbar_kws={"label": "Cross-seed stability"},
    )
    _set_tick_text(ax_a, [_model_display(value) for value in heat.columns], [clean_parameter_name(value) for value in heat.index])
    label_axes(ax_a, "Parameter x model cross-seed stability", "Primary evidence panel", registry.style)
    family_breaks = [4, 10]
    for value in family_breaks:
        ax_a.hlines(value, *ax_a.get_xlim(), colors="#8A817C", linewidth=1.0)

    ordered_basins = _attributes(registry).sort_values(["aridity", "frac_snow"])["basin_id"].tolist()
    carpet_rows = []
    carpet_labels = []
    for model in MODEL_ORDER:
        path = registry.analysis_root / "parameter_variance" / f"seed_parameter_variance__{model}__{registry.reference_loss}.csv"
        frame = pd.read_csv(path).set_index("basin_id").reindex(ordered_basins)
        for family, parameters in {
            "Storage": ["parFC", "parBETA", "parPERC"],
            "Routing": ["parK0", "parK1", "parK2", "parUZL", "route_a", "route_b"],
            "Snow": ["parTT", "parCFMAX", "parCFR", "parCWH"],
        }.items():
            available = [parameter for parameter in parameters if parameter in frame.columns]
            values = 1.0 - frame[available].mean(axis=1) / frame[available].mean(axis=1).max()
            carpet_rows.append(values.to_numpy())
            carpet_labels.append(f"{_model_display(model)} {family}")
    sns.heatmap(
        np.vstack(carpet_rows),
        ax=ax_b,
        cmap=_stability_cmap(registry),
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        xticklabels=False,
        yticklabels=carpet_labels,
    )
    label_axes(ax_b, "Basin-ordered family stability carpet", "Secondary overview", registry.style)
    ax_b.tick_params(axis="y", labelsize=7.0)

    class_code = {"stable": 0, "intermediate": 1, "sensitive": 2}
    class_matrix = (
        summary.assign(code=summary["stability_class"].map(class_code))
        .pivot_table(index="parameter", columns="model", values="code", aggfunc="mean")
        .reindex(index=registry.parameter_order, columns=MODEL_ORDER)
    )
    sns.heatmap(
        class_matrix,
        ax=ax_c,
        cmap=mcolors.ListedColormap([
            registry.palette["classes"]["stable"],
            registry.palette["classes"]["intermediate"],
            registry.palette["classes"]["unstable"],
        ]),
        cbar=False,
        linewidths=0.35,
        linecolor="white",
    )
    _set_tick_text(ax_c, [_model_display(value) for value in class_matrix.columns], [clean_parameter_name(value) for value in class_matrix.index])
    label_axes(ax_c, "Stability classes", "Exact values moved to Table03", registry.style)

    apply_panel_letters([ax_a, ax_b, ax_c], registry.style)
    return _complete_figure(registry, "02", fig, output_dir, qc, formats)


def build_fig03_cross_seed_corr_stability(
    registry: FigureDataRegistry,
    output_dir: Path,
    qc: QcCollector,
    formats: tuple[str, ...],
) -> dict[str, Path]:
    width_mm, height_mm = _figure_size(registry, "03")
    fig = make_figure(width_mm, height_mm)
    outer = build_asymmetric_gridspec(fig, 2, 2, width_ratios=[1.45, 0.95], height_ratios=[1.0, 0.78], wspace=0.22, hspace=0.18)
    ax_a = fig.add_subplot(outer[:, 0])
    right_top = outer[0, 1].subgridspec(2, 1, hspace=0.18)
    ax_b = fig.add_subplot(right_top[0, 0])
    ax_c = fig.add_subplot(right_top[1, 0])
    bottom = outer[1, 1].subgridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.28)
    ax_d = fig.add_subplot(bottom[0, 0])
    ax_e = fig.add_subplot(bottom[0, 1])

    cmap = _diverging_cmap(registry)
    corr_vlim = registry.style["figure"]["correlation_vlim"]
    matrix_main = _corr_focus_matrix(registry, "distributional", "mean_corr")
    hm_main = sns.heatmap(
        matrix_main,
        ax=ax_a,
        cmap=cmap,
        center=0.0,
        vmin=-corr_vlim,
        vmax=corr_vlim,
        linewidths=0.4,
        linecolor=registry.palette["neutrals"]["divider"],
        cbar=False,
    )
    _set_tick_text(ax_a, [clean_parameter_name(value) for value in matrix_main.columns], [clean_attribute_name(value) for value in matrix_main.index])
    label_axes(ax_a, "δdtb correlation matrix", None, registry.style)

    for ax, model in [(ax_b, "deterministic"), (ax_c, "mc_dropout")]:
        matrix = _corr_focus_matrix(registry, model, "mean_corr")
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            center=0.0,
            vmin=-corr_vlim,
            vmax=corr_vlim,
            linewidths=0.35,
            linecolor=registry.palette["neutrals"]["divider"],
            cbar=False,
        )
        _set_tick_text(ax, [clean_parameter_name(value) for value in matrix.columns], [""] * len(matrix.index))
        label_axes(ax, _model_display(model), None, registry.style)

    make_shared_colorbar(fig, hm_main.collections[0], [ax_a, ax_b, ax_c], "Spearman ρ", orientation="horizontal", fraction=0.05, pad=0.03)

    relationships = registry.require_columns(
        "relationship_classes",
        ["model", "parameter", "attribute", "relationship_class", "mean_abs_corr"],
    )
    relationships = relationships.loc[
        relationships["parameter"].isin(_focus_parameters(registry)) &
        relationships["attribute"].isin(_focus_attributes(registry))
    ].copy()
    code_map = {"robust": 2, "loss-sensitive": 1, "model-sensitive": 0}
    mask = relationships.assign(code=relationships["relationship_class"].map(code_map))
    stacked_mask = (
        mask.pivot_table(index=["model", "attribute"], columns="parameter", values="code", aggfunc="mean")
        .reindex(
            index=pd.MultiIndex.from_product([MODEL_ORDER, _focus_attributes(registry)]),
            columns=_focus_parameters(registry),
        )
    )
    stacked_mask.index.names = [None, None]
    sns.heatmap(
        stacked_mask,
        ax=ax_d,
        cmap=mcolors.ListedColormap([
            registry.palette["classes"]["model-sensitive"],
            registry.palette["classes"]["loss-sensitive"],
            registry.palette["classes"]["robust"],
        ]),
        cbar=False,
        linewidths=0.3,
        linecolor="white",
    )
    ax_d.set_ylabel("")
    _set_tick_text(
        ax_d,
        [clean_parameter_name(value) for value in stacked_mask.columns],
        [f"{clean_model_name(idx[0])} / {clean_attribute_name(idx[1])}" for idx in stacked_mask.index],
        ysize=7.0,
    )
    label_axes(ax_d, "Cross-seed reliability mask", "Robust cells shown darkest", registry.style)

    robust_table = (
        relationships.loc[relationships["relationship_class"] == "robust", ["model", "parameter", "attribute", "mean_abs_corr"]]
        .sort_values(["mean_abs_corr", "model"], ascending=[False, True])
        .head(6)
    )
    robust_table = _clean_output_frame(robust_table)
    lines = [
        f"{row['model']}  {row['parameter']} <- {row['attribute']}  |ρ|={row['mean_abs_corr']:.2f}"
        for _, row in robust_table.iterrows()
    ]
    _text_rows_panel(ax_e, "Top robust relationships", lines, registry.style)

    apply_panel_letters([ax_a, ax_b, ax_c, ax_d, ax_e], registry.style)
    return _complete_figure(registry, "03", fig, output_dir, qc, formats)


def build_fig04_cross_loss_corr_stability(
    registry: FigureDataRegistry,
    output_dir: Path,
    qc: QcCollector,
    formats: tuple[str, ...],
) -> dict[str, Path]:
    loss_stability = registry.require_columns(
        "pair_loss_stability",
        ["model", "parameter", "attribute", "loss_std_rho", "sign_consistency_loss"],
    )
    loss_stability = loss_stability.loc[
        loss_stability["parameter"].isin(_focus_parameters(registry)) &
        loss_stability["attribute"].isin(_focus_attributes(registry))
    ].copy()
    loss_stability["loss_stability_score"] = 1.0 - (loss_stability["loss_std_rho"] / loss_stability["loss_std_rho"].max())

    width_mm, height_mm = _figure_size(registry, "04")
    fig = make_figure(width_mm, height_mm)
    outer = build_asymmetric_gridspec(fig, 2, 2, width_ratios=[1.45, 0.95], height_ratios=[1.0, 0.78], wspace=0.22, hspace=0.18)
    ax_a = fig.add_subplot(outer[:, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    right_bottom = outer[1, 1].subgridspec(2, 1, hspace=0.25)
    ax_c = fig.add_subplot(right_bottom[0, 0])
    ax_d = fig.add_subplot(right_bottom[1, 0])

    stacked_stability = (
        loss_stability.pivot_table(index=["model", "attribute"], columns="parameter", values="loss_stability_score", aggfunc="mean")
        .reindex(index=pd.MultiIndex.from_product([MODEL_ORDER, _focus_attributes(registry)]), columns=_focus_parameters(registry))
    )
    stacked_stability.index.names = [None, None]
    hm_a = sns.heatmap(
        stacked_stability,
        ax=ax_a,
        cmap=_stability_cmap(registry),
        vmin=0.0,
        vmax=1.0,
        linewidths=0.35,
        linecolor=registry.palette["neutrals"]["divider"],
        cbar=False,
    )
    ax_a.set_ylabel("")
    _set_tick_text(
        ax_a,
        [clean_parameter_name(value) for value in stacked_stability.columns],
        [f"{clean_model_name(idx[0])} / {clean_attribute_name(idx[1])}" for idx in stacked_stability.index],
        ysize=7.0,
    )
    label_axes(ax_a, "Cross-loss robustness matrix", "Primary evidence panel", registry.style)
    make_shared_colorbar(fig, hm_a.collections[0], [ax_a], "Loss robustness", orientation="horizontal", fraction=0.05, pad=0.03)

    variability = (
        loss_stability.pivot_table(index="attribute", columns="parameter", values="loss_std_rho", aggfunc="mean")
        .reindex(index=_focus_attributes(registry), columns=_focus_parameters(registry))
    )
    hm_b = sns.heatmap(
        variability,
        ax=ax_b,
        cmap=_stability_cmap(registry).reversed(),
        linewidths=0.35,
        linecolor=registry.palette["neutrals"]["divider"],
        cbar=False,
    )
    _set_tick_text(ax_b, [clean_parameter_name(value) for value in variability.columns], [clean_attribute_name(value) for value in variability.index])
    label_axes(ax_b, "Cross-loss variability", "Smaller panel, same ordering", registry.style)
    make_shared_colorbar(fig, hm_b.collections[0], [ax_b], "Loss std(ρ)", orientation="horizontal", fraction=0.12, pad=0.10)

    classes = registry.require_columns(
        "relationship_classes",
        ["model", "parameter", "attribute", "seed_stable", "loss_stable"],
    )
    classes = classes.loc[
        classes["parameter"].isin(_focus_parameters(registry)) &
        classes["attribute"].isin(_focus_attributes(registry))
    ].copy()
    classes["class_name"] = np.select(
        [
            classes["seed_stable"] & classes["loss_stable"],
            classes["seed_stable"] & ~classes["loss_stable"],
            ~classes["seed_stable"] & classes["loss_stable"],
        ],
        ["both", "seed only", "loss only"],
        default="sensitive",
    )
    class_counts = (
        classes.groupby(["model", "class_name"], as_index=False)
        .size()
        .pivot_table(index="model", columns="class_name", values="size", aggfunc="sum")
        .reindex(index=MODEL_ORDER, columns=["both", "seed only", "loss only", "sensitive"])
        .fillna(0)
    )
    sns.heatmap(
        class_counts,
        ax=ax_c,
        cmap=_stability_cmap(registry),
        annot=True,
        fmt=".0f",
        cbar=False,
        linewidths=0.35,
        linecolor="white",
    )
    _set_tick_text(ax_c, ["both", "seed only", "loss only", "sensitive"], [_model_display(value) for value in class_counts.index], xrotation=20)
    label_axes(ax_c, "Seed vs loss classes", None, registry.style)

    family_summary = (
        classes.assign(parameter_family=classes["parameter"].map(_family_for_parameter))
        .groupby(["parameter_family", "class_name"], as_index=False)
        .size()
        .pivot_table(index="parameter_family", columns="class_name", values="size", aggfunc="sum")
        .reindex(index=["storage_recharge", "routing_runoff", "snow_cold"], columns=["both", "seed only", "loss only", "sensitive"])
        .fillna(0)
    )
    sns.heatmap(
        family_summary,
        ax=ax_d,
        cmap=_stability_cmap(registry),
        annot=True,
        fmt=".0f",
        cbar=False,
        linewidths=0.35,
        linecolor="white",
    )
    _set_tick_text(ax_d, ["both", "seed", "loss", "sensitive"], ["Storage", "Routing", "Snow"])
    label_axes(ax_d, "Parameter-family summary", None, registry.style)

    apply_panel_letters([ax_a, ax_b, ax_c, ax_d], registry.style)
    return _complete_figure(registry, "04", fig, output_dir, qc, formats)


def build_fig05_shared_dominant_core(
    registry: FigureDataRegistry,
    output_dir: Path,
    qc: QcCollector,
    formats: tuple[str, ...],
) -> dict[str, Path]:
    dominant = registry.require_columns(
        "results331_dominant_attribute_summary",
        [
            "parameter",
            "deterministic_attribute",
            "mc_dropout_attribute",
            "distributional_attribute",
            "overall_relationship_class",
        ],
    )
    relationships = registry.require_columns(
        "relationship_classes",
        ["model", "parameter", "attribute", "mean_abs_corr"],
    )
    width_mm, height_mm = _figure_size(registry, "05")
    fig = make_figure(width_mm, height_mm)
    outer = build_asymmetric_gridspec(fig, 1, 2, width_ratios=[1.7, 0.92], hspace=0.0, wspace=0.18)
    ax_a = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(2, 1, height_ratios=[0.9, 1.1], hspace=0.22)
    ax_b = fig.add_subplot(right[0, 0])
    ax_c = fig.add_subplot(right[1, 0])

    consensus_edges = []
    for _, row in dominant.iterrows():
        candidates = [row["deterministic_attribute"], row["mc_dropout_attribute"], row["distributional_attribute"]]
        consensus_attribute = Counter(candidates).most_common(1)[0][0]
        weight_frame = relationships.loc[
            (relationships["parameter"] == row["parameter"]) &
            (relationships["attribute"] == consensus_attribute)
        ]
        consensus_edges.append(
            {
                "parameter": row["parameter"],
                "attribute": consensus_attribute,
                "class": row["overall_relationship_class"],
                "weight": float(weight_frame["mean_abs_corr"].mean()) if not weight_frame.empty else 0.25,
            }
        )
    edge_frame = pd.DataFrame(consensus_edges)
    left_nodes = sorted(edge_frame["attribute"].unique(), key=lambda item: clean_attribute_name(item))
    right_nodes = [value for value in registry.parameter_order if value in edge_frame["parameter"].unique()]
    y_left = np.linspace(0.90, 0.10, len(left_nodes))
    y_right = np.linspace(0.90, 0.10, len(right_nodes))
    positions = {value: (0.12, y) for value, y in zip(left_nodes, y_left)}
    positions.update({value: (0.86, y) for value, y in zip(right_nodes, y_right)})
    class_colors = {
        "shared dominant controls": registry.palette["classes"]["robust"],
        "partially shared controls": registry.palette["classes"]["loss-sensitive"],
        "model-sensitive controls": registry.palette["classes"]["model-sensitive"],
    }

    for attribute in left_nodes:
        x, y = positions[attribute]
        ax_a.scatter(x, y, s=150, color=registry.palette["families"][registry.attribute_family_lookup()[attribute]], edgecolor="white", linewidth=0.8, zorder=3)
        ax_a.text(x - 0.035, y, clean_attribute_name(attribute), ha="right", va="center", fontsize=7.7)
    for parameter in right_nodes:
        x, y = positions[parameter]
        ax_a.scatter(x, y, s=150, color=registry.palette["families"][_family_for_parameter(parameter)], edgecolor="white", linewidth=0.8, zorder=3)
        ax_a.text(x + 0.035, y, clean_parameter_name(parameter), ha="left", va="center", fontsize=7.7)
    for _, row in edge_frame.iterrows():
        (x0, y0), (x1, y1) = positions[row["attribute"]], positions[row["parameter"]]
        ax_a.plot(
            [x0 + 0.018, x1 - 0.018],
            [y0, y1],
            color=class_colors[row["class"]],
            linewidth=registry.style["figure"]["network_edge_width_min"] + row["weight"] * 2.0,
            alpha=0.85,
            zorder=1,
        )
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_ylim(0.0, 1.0)
    ax_a.axis("off")
    label_axes(ax_a, "Shared dominant-control core", "Primary network panel", registry.style)

    summary = (
        edge_frame.assign(parameter_family=edge_frame["parameter"].map(_family_for_parameter))
        .groupby(["parameter_family", "class"], as_index=False)
        .size()
        .pivot_table(index="parameter_family", columns="class", values="size", aggfunc="sum")
        .reindex(index=["storage_recharge", "routing_runoff", "snow_cold"])
        .fillna(0)
    )
    sns.heatmap(
        summary,
        ax=ax_b,
        cmap=_stability_cmap(registry),
        annot=True,
        fmt=".0f",
        cbar=False,
        linewidths=0.4,
        linecolor="white",
    )
    _set_tick_text(ax_b, ["shared", "partial", "sensitive"], ["Storage", "Routing", "Snow"])
    label_axes(ax_b, "Relation-class summary", None, registry.style)

    callouts = edge_frame.loc[:, ["parameter", "attribute", "class", "weight"]].copy()
    callouts = callouts.sort_values("weight", ascending=False).head(6)
    callouts = _clean_output_frame(callouts)
    lines = [
        f"{row['parameter']} <- {row['attribute']}  {row['class']}  |ρ|={row['weight']:.2f}"
        for _, row in callouts.iterrows()
    ]
    _text_rows_panel(ax_c, "Representative relations", lines, registry.style)

    apply_panel_letters([ax_a, ax_b, ax_c], registry.style)
    return _complete_figure(registry, "05", fig, output_dir, qc, formats)


def build_fig06_matrix_similarity(
    registry: FigureDataRegistry,
    output_dir: Path,
    qc: QcCollector,
    formats: tuple[str, ...],
) -> dict[str, Path]:
    similarity = registry.require_columns(
        "results332_matrix_similarity",
        ["run_id_a", "run_id_b", "model_a", "model_b", "loss_a", "loss_b", "same_model", "same_loss", "same_seed", "matrix_corr_spearman"],
    )
    embedding = registry.require_columns(
        "results332_matrix_embedding",
        ["run_id", "model", "loss", "seed", "mds_x", "mds_y"],
    )
    parquet = registry.optional_table("results332_spearman_matrices")
    if parquet is None:
        raise MissingFigureDataError("Missing results332_spearman_matrices.parquet.")

    width_mm, height_mm = _figure_size(registry, "06")
    fig = make_figure(width_mm, height_mm)
    outer = build_asymmetric_gridspec(fig, 2, 2, width_ratios=[1.3, 0.92], height_ratios=[1.0, 0.75], wspace=0.22, hspace=0.18)
    ax_a = fig.add_subplot(outer[:, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    bottom = outer[1, 1].subgridspec(2, 1, height_ratios=[0.52, 0.48], hspace=0.28)
    ax_c = fig.add_subplot(bottom[0, 0])
    ax_d = fig.add_subplot(bottom[1, 0])

    order = embedding.sort_values(["model", "loss", "seed"])["run_id"].tolist()
    matrix = similarity.pivot_table(index="run_id_a", columns="run_id_b", values="matrix_corr_spearman", aggfunc="mean").reindex(index=order, columns=order)
    hm = sns.heatmap(
        matrix,
        ax=ax_a,
        cmap=_stability_cmap(registry),
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
    )
    label_axes(ax_a, "Matrix-level similarity heatmap", None, registry.style)
    make_shared_colorbar(fig, hm.collections[0], [ax_a], "Similarity", orientation="horizontal", fraction=0.05, pad=0.03)

    markers = {"NseBatchLoss": "o", "LogNseBatchLoss": "s", "HybridNseBatchLoss": "^"}
    for model in MODEL_ORDER:
        subset = embedding.loc[embedding["model"] == model]
        for loss in registry.loss_order:
            loss_subset = subset.loc[subset["loss"] == loss]
            ax_b.scatter(
                loss_subset["mds_x"],
                loss_subset["mds_y"],
                s=22,
                marker=markers[loss],
                color=_model_color(registry, model),
                alpha=0.9,
            )
    label_axes(ax_b, "2D embedding", None, registry.style)
    ax_b.set_xlabel("MDS-1")
    ax_b.set_ylabel("MDS-2")
    model_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=_model_color(registry, model), label=_model_display(model), markersize=6)
        for model in MODEL_ORDER
    ]
    loss_handles = [
        Line2D([0], [0], marker=marker, linestyle="", color="#6E7783", label=label, markersize=6)
        for label, marker in [("NSE", "o"), ("LogNSE", "s"), ("Hybrid", "^")]
    ]
    legend1 = ax_b.legend(handles=model_handles, frameon=False, fontsize=7.0, loc="upper left", title="Model")
    ax_b.add_artist(legend1)
    ax_b.legend(handles=loss_handles, frameon=False, fontsize=7.0, loc="lower right", title="Loss")

    block_rows = pd.DataFrame(
        [
            [similarity.loc[similarity["same_model"], "matrix_corr_spearman"].mean(), similarity.loc[~similarity["same_model"], "matrix_corr_spearman"].mean(), similarity.loc[similarity["same_loss"], "matrix_corr_spearman"].mean()],
            [similarity.loc[~similarity["same_loss"], "matrix_corr_spearman"].mean(), similarity.loc[similarity["same_seed"], "matrix_corr_spearman"].mean(), similarity.loc[~similarity["same_seed"], "matrix_corr_spearman"].mean()],
        ],
        index=["Model / loss", "Seed"],
        columns=["Within model", "Cross model", "Within loss"],
    )
    block_rows.loc["Seed", "Within model"] = similarity.loc[~similarity["same_loss"], "matrix_corr_spearman"].mean()
    block_rows.loc["Seed", "Cross model"] = similarity.loc[similarity["same_seed"], "matrix_corr_spearman"].mean()
    block_rows.loc["Seed", "Within loss"] = similarity.loc[~similarity["same_seed"], "matrix_corr_spearman"].mean()
    sns.heatmap(
        block_rows,
        ax=ax_c,
        cmap=_stability_cmap(registry),
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        cbar=False,
        linewidths=0.4,
        linecolor="white",
    )
    label_axes(ax_c, "Block summary", None, registry.style)
    ax_c.tick_params(axis="x", labelrotation=20)

    meta = embedding.set_index("run_id").loc[order].reset_index()
    parquet["family"] = parquet["parameter"].map(_family_for_parameter)
    subsystem_rows = []
    for family_key, title in [("storage_recharge", "Storage"), ("routing_runoff", "Routing"), ("snow_cold", "Snow")]:
        vector = parquet.loc[parquet["family"] == family_key].pivot_table(index="run_id", columns=["attribute", "parameter"], values="corr", aggfunc="mean").reindex(order)
        family_similarity = cosine_similarity(vector.fillna(0.0))
        pair_values = {}
        for left, right in [("deterministic", "mc_dropout"), ("deterministic", "distributional"), ("mc_dropout", "distributional")]:
            idx_a = meta.index[meta["model"] == left]
            idx_b = meta.index[meta["model"] == right]
            pair_values[f"{_model_display(left)}-{_model_display(right)}"] = float(family_similarity[np.ix_(idx_a, idx_b)].mean())
        subsystem_rows.append({"family": title, **pair_values})
    subsystem = pd.DataFrame(subsystem_rows).set_index("family")
    sns.heatmap(
        subsystem,
        ax=ax_d,
        cmap=_stability_cmap(registry),
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        cbar=False,
        linewidths=0.35,
        linecolor="white",
    )
    ax_d.set_yticklabels(list(subsystem.index), rotation=0)
    ax_d.set_xticklabels(list(subsystem.columns), rotation=20, ha="right")
    label_axes(ax_d, "Subsystem decomposition", None, registry.style)

    apply_panel_letters([ax_a, ax_b, ax_c, ax_d], registry.style)
    return _complete_figure(registry, "06", fig, output_dir, qc, formats)


def build_fig07_explainability_support(
    registry: FigureDataRegistry,
    output_dir: Path,
    qc: QcCollector,
    formats: tuple[str, ...],
) -> dict[str, Path]:
    alignment = registry.require_columns(
        "results333_importance_alignment",
        ["model", "parameter", "dominant_attribute_top3_rate"],
    )
    overlap = registry.require_columns(
        "results333_importance_overlap",
        ["parameter", "model_a", "model_b", "topk", "jaccard_overlap"],
    )
    importance = registry.require_columns(
        "results333_parameter_feature_importance",
        ["model", "loss", "parameter", "attribute", "top3_rate"],
    )

    width_mm, height_mm = _figure_size(registry, "07")
    fig = make_figure(width_mm, height_mm)
    outer = build_asymmetric_gridspec(fig, 2, 2, width_ratios=[1.2, 0.95], height_ratios=[0.95, 0.8], wspace=0.22, hspace=0.18)
    ax_a = fig.add_subplot(outer[:, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    fingerprints = outer[1, 1].subgridspec(2, 1, hspace=0.28)
    fp_axes = [fig.add_subplot(fingerprints[idx, 0]) for idx in range(2)]

    agreement = alignment.pivot_table(index="parameter", columns="model", values="dominant_attribute_top3_rate", aggfunc="mean")
    agreement = agreement.reindex(index=registry.parameter_order, columns=MODEL_ORDER)
    sns.heatmap(
        agreement,
        ax=ax_a,
        cmap=_stability_cmap(registry),
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Top-3 agreement"},
        linewidths=0.35,
        linecolor=registry.palette["neutrals"]["divider"],
    )
    _set_tick_text(ax_a, [_model_display(value) for value in agreement.columns], [clean_parameter_name(value) for value in agreement.index])
    label_axes(ax_a, "Explainability / correlation agreement", None, registry.style)

    overlap = overlap.loc[overlap["topk"] == 3].copy()
    overlap["pair"] = overlap["model_a"].map(clean_model_name) + " vs " + overlap["model_b"].map(clean_model_name)
    overlap_matrix = overlap.pivot_table(index="parameter", columns="pair", values="jaccard_overlap", aggfunc="mean")
    overlap_matrix = overlap_matrix.reindex(index=registry.parameter_order)
    sns.heatmap(
        overlap_matrix,
        ax=ax_b,
        cmap=_stability_cmap(registry),
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        cbar=False,
        linewidths=0.35,
        linecolor="white",
    )
    _set_tick_text(ax_b, list(overlap_matrix.columns), [clean_parameter_name(value) for value in overlap_matrix.index], xrotation=20)
    label_axes(ax_b, "Top-feature overlap", None, registry.style)

    selected_params = [value for value in ["parFC", "parPERC"] if value in importance["parameter"].unique()]
    reference_importance = importance.loc[importance["loss"] == registry.reference_loss].copy()
    for ax, parameter in zip(fp_axes, selected_params):
        subset = reference_importance.loc[reference_importance["parameter"] == parameter]
        top_attrs = (
            subset.loc[subset["model"] == "distributional"]
            .sort_values("top3_rate", ascending=False)["attribute"]
            .head(4)
            .tolist()
        )
        pivot = subset.loc[subset["attribute"].isin(top_attrs)].pivot_table(index="attribute", columns="model", values="top3_rate", aggfunc="mean")
        pivot = pivot.reindex(index=top_attrs, columns=MODEL_ORDER)
        sns.heatmap(
            pivot,
            ax=ax,
            cmap=_uncertainty_cmap(registry),
            vmin=0.0,
            vmax=1.0,
            cbar=False,
            linewidths=0.3,
            linecolor="white",
        )
        _set_tick_text(ax, [_model_display(value) for value in pivot.columns], [clean_attribute_name(value) for value in pivot.index])
        label_axes(ax, clean_parameter_name(parameter), None, registry.style)

    apply_panel_letters([ax_a, ax_b, fp_axes[0]], registry.style)
    return _complete_figure(registry, "07", fig, output_dir, qc, formats)


def build_fig08_parameter_mean_gradients(
    registry: FigureDataRegistry,
    output_dir: Path,
    qc: QcCollector,
    formats: tuple[str, ...],
) -> dict[str, Path]:
    relationships = registry.require_columns(
        "results341_distributional_mean_relationships",
        ["parameter", "attribute", "mean_spearman_corr"],
    )
    gradient_stats = registry.require_columns(
        "results341_gradient_group_stats",
        ["gradient_attribute", "parameter", "high_minus_low_median_unit"],
    )
    spatial = registry.spatial.copy()
    params = _distributional_mean_table(registry)

    width_mm, height_mm = _figure_size(registry, "08")
    fig = make_figure(width_mm, height_mm)
    outer = build_asymmetric_gridspec(fig, 2, 2, width_ratios=[1.45, 0.95], height_ratios=[1.0, 0.72], wspace=0.22, hspace=0.18)
    ax_a = fig.add_subplot(outer[:, 0])
    maps = outer[0, 1].subgridspec(2, 2, hspace=0.05, wspace=0.04)
    map_axes = [fig.add_subplot(maps[idx // 2, idx % 2]) for idx in range(4)]
    bottom = outer[1, 1].subgridspec(2, 1, height_ratios=[0.32, 0.68], hspace=0.10)
    ax_b = fig.add_subplot(bottom[0, 0])
    ax_d = fig.add_subplot(bottom[1, 0])

    matrix = relationships.pivot_table(index="attribute", columns="parameter", values="mean_spearman_corr", aggfunc="mean")
    matrix = matrix.reindex(index=_focus_attributes(registry), columns=_focus_parameters(registry))
    hm = sns.heatmap(
        matrix,
        ax=ax_a,
        cmap=_diverging_cmap(registry),
        center=0.0,
        vmin=-registry.style["figure"]["correlation_vlim"],
        vmax=registry.style["figure"]["correlation_vlim"],
        linewidths=0.4,
        linecolor=registry.palette["neutrals"]["divider"],
        cbar=False,
    )
    _set_tick_text(ax_a, [clean_parameter_name(value) for value in matrix.columns], [clean_attribute_name(value) for value in matrix.index])
    label_axes(ax_a, "Large-scale gradients in parameter means", "Primary heatmap panel", registry.style)
    make_shared_colorbar(fig, hm.collections[0], [ax_a], "Mean Spearman ρ", orientation="horizontal", fraction=0.05, pad=0.03)
    for boundary in [3, 4]:
        ax_a.vlines(boundary, *ax_a.get_ylim(), colors="#8A817C", linewidth=1.0)

    merged = spatial.merge(params, on="basin_id", how="inner")
    selected = [value for value in ["parFC", "parPERC", "parCFR", "parBETA"] if value in merged.columns]
    for ax, parameter in zip(map_axes, selected):
        plotted = basin_polygon_map(
            ax,
            merged,
            parameter,
            _stability_cmap(registry),
            clean_parameter_name(parameter),
            registry.palette,
            line_width=registry.style["figure"]["map_line_width"],
        )
        ax.set_title("")
        ax.title.set_visible(False)
        ax.text(0.03, 0.96, clean_parameter_name(parameter), transform=ax.transAxes, ha="left", va="top", fontsize=7.4)
    make_shared_colorbar(fig, plotted, map_axes, "Parameter mean", orientation="horizontal", fraction=0.06, pad=0.05)

    ax_b.axis("off")
    label_axes(ax_b, "Family strip", None, registry.style)
    family_blocks = [
        ("Storage / recharge", ["parBETA", "parFC", "parPERC"], registry.palette["families"]["storage_recharge"]),
        ("Routing", ["parUZL"], registry.palette["families"]["routing_runoff"]),
        ("Snow", ["parCFR", "parCWH"], registry.palette["families"]["snow_cold"]),
    ]
    x = 0.02
    for label, parameters, color in family_blocks:
        width = 0.13 * len(parameters)
        ax_b.add_patch(Rectangle((x, 0.28), width, 0.28, facecolor=color, edgecolor="white"))
        ax_b.text(x + width / 2.0, 0.64, label, ha="center", va="bottom", fontsize=7.6)
        for idx, parameter in enumerate(parameters):
            ax_b.text(x + 0.065 + idx * 0.13, 0.18, clean_parameter_name(parameter), ha="center", va="top", fontsize=7.4)
        x += width + 0.04

    direction = gradient_stats.pivot_table(index="gradient_attribute", columns="parameter", values="high_minus_low_median_unit", aggfunc="mean")
    direction = direction.reindex(index=GRADIENT_ATTRIBUTES, columns=_focus_parameters(registry))
    sns.heatmap(
        direction,
        ax=ax_d,
        cmap=_diverging_cmap(registry),
        center=0.0,
        linewidths=0.35,
        linecolor="white",
        annot=True,
        fmt=".2f",
        cbar=False,
    )
    _set_tick_text(ax_d, [clean_parameter_name(value) for value in direction.columns], [clean_attribute_name(value) for value in direction.index])
    label_axes(ax_d, "Direction summary", None, registry.style)

    apply_panel_letters([ax_a, map_axes[0], ax_b, ax_d], registry.style)
    return _complete_figure(registry, "08", fig, output_dir, qc, formats)


def build_fig09_representative_mean_gradients(
    registry: FigureDataRegistry,
    output_dir: Path,
    qc: QcCollector,
    formats: tuple[str, ...],
) -> dict[str, Path]:
    attributes = _attributes(registry)
    params = _distributional_mean_table(registry)
    representative = registry.require_columns(
        "results343_representative_basins",
        ["basin_id", "group_label", "gradient_attribute"] + [f"{parameter}_mean_unit" for parameter in _focus_parameters(registry)],
    )
    spatial = registry.spatial.copy()
    merged = attributes.merge(params, on="basin_id", how="inner")

    width_mm, height_mm = _figure_size(registry, "09")
    fig = make_figure(width_mm, height_mm)
    outer = build_asymmetric_gridspec(fig, 2, 1, height_ratios=[0.95, 0.88], wspace=0.0, hspace=0.20)
    top = outer[0, 0].subgridspec(1, 3, wspace=0.20)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(top[0, 2])
    bottom = outer[1, 0].subgridspec(1, 2, width_ratios=[0.72, 1.18], wspace=0.20)
    ax_d_map = fig.add_subplot(bottom[0, 0])
    ax_d_table = fig.add_subplot(bottom[0, 1])

    response_specs = [
        ("aridity", "parPERC", "Aridity response"),
        ("frac_snow", "parCWH", "Snow response"),
        ("slope_mean", "parBETA", "Topography response"),
    ]
    for ax, (attribute, parameter, title) in zip([ax_a, ax_b, ax_c], response_specs):
        hb = ax.hexbin(
            merged[attribute],
            merged[parameter],
            gridsize=28,
            cmap=_stability_cmap(registry),
            mincnt=1,
            linewidths=0.0,
        )
        ax.set_xlabel(clean_attribute_name(attribute))
        ax.set_ylabel(clean_parameter_name(parameter))
        label_axes(ax, title, None, registry.style)
        make_shared_colorbar(fig, hb, [ax], "Basin count", orientation="vertical", fraction=0.08, pad=0.02)

    rep_points = spatial.merge(representative[["basin_id", "group_label"]].drop_duplicates(), on="basin_id", how="inner")
    group_palette = {
        group: color for group, color in zip(
            sorted(rep_points["group_label"].unique()),
            ["#3B5B7A", "#4C8C7A", "#B66A45", "#7C6BB0", "#A27B43", "#6AA7C8"],
        )
    }
    for group, subset in rep_points.groupby("group_label"):
        subset.plot(ax=ax_d_map, color=group_palette[group], linewidth=0.1, edgecolor="white")
    ax_d_map.set_axis_off()
    label_axes(ax_d_map, "Archetype locator groups", None, registry.style)
    handles = [Line2D([0], [0], marker="s", linestyle="", color=color, label=group, markersize=7) for group, color in group_palette.items()]
    ax_d_map.legend(handles=handles, frameon=False, fontsize=7.0, loc="lower left")

    group_summary = registry.require_columns(
        "results343_basin_group_summary",
        ["group_label", "parameter", "mean_median_unit"],
    )
    group_summary = group_summary.pivot_table(index="group_label", columns="parameter", values="mean_median_unit", aggfunc="mean")
    group_summary = group_summary.reindex(columns=_focus_parameters(registry))
    sns.heatmap(
        group_summary,
        ax=ax_d_table,
        cmap=_stability_cmap(registry),
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Mean median (unit interval)"},
    )
    _set_tick_text(ax_d_table, [clean_parameter_name(value) for value in group_summary.columns], list(group_summary.index))
    label_axes(ax_d_table, "Archetype fingerprints", "Exact metadata moved to Table05", registry.style)

    apply_panel_letters([ax_a, ax_b, ax_c, ax_d_map], registry.style)
    return _complete_figure(registry, "09", fig, output_dir, qc, formats)


def build_fig10_parameter_uncertainty_gradients(
    registry: FigureDataRegistry,
    output_dir: Path,
    qc: QcCollector,
    formats: tuple[str, ...],
) -> dict[str, Path]:
    std_relationships = registry.require_columns(
        "results342_distributional_std_relationships",
        ["parameter", "attribute", "mean_spearman_corr"],
    )
    caution = registry.require_columns(
        "finalcheck_uncertainty_interpretation_flags",
        ["parameter", "attribute", "interpretation_flag"],
    )
    mean_relationships = registry.require_columns(
        "results341_distributional_mean_relationships",
        ["parameter", "attribute", "mean_spearman_corr"],
    )
    spatial = registry.spatial.copy()
    std_params = _distributional_std_table(registry)

    width_mm, height_mm = _figure_size(registry, "10")
    fig = make_figure(width_mm, height_mm)
    outer = build_asymmetric_gridspec(fig, 2, 2, width_ratios=[1.45, 0.95], height_ratios=[1.0, 0.72], wspace=0.22, hspace=0.18)
    ax_a = fig.add_subplot(outer[:, 0])
    maps = outer[0, 1].subgridspec(2, 2, hspace=0.05, wspace=0.04)
    map_axes = [fig.add_subplot(maps[idx // 2, idx % 2]) for idx in range(4)]
    bottom = outer[1, 1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.20)
    ax_b = fig.add_subplot(bottom[0, 0])
    ax_c = fig.add_subplot(bottom[0, 1])

    matrix = std_relationships.pivot_table(index="attribute", columns="parameter", values="mean_spearman_corr", aggfunc="mean")
    matrix = matrix.reindex(index=_focus_attributes(registry), columns=_focus_parameters(registry))
    hm = sns.heatmap(
        matrix,
        ax=ax_a,
        cmap=_uncertainty_cmap(registry),
        linewidths=0.4,
        linecolor=registry.palette["neutrals"]["divider"],
        cbar=False,
    )
    _set_tick_text(ax_a, [clean_parameter_name(value) for value in matrix.columns], [clean_attribute_name(value) for value in matrix.index])
    label_axes(ax_a, "Large-scale gradients in organized uncertainty", "Primary heatmap panel", registry.style)
    make_shared_colorbar(fig, hm.collections[0], [ax_a], "Uncertainty structure", orientation="horizontal", fraction=0.05, pad=0.03)

    merged = spatial.merge(std_params, on="basin_id", how="inner")
    selected = [value for value in ["parBETA", "parPERC", "parUZL", "parCWH"] if value in merged.columns]
    for ax, parameter in zip(map_axes, selected):
        plotted = basin_polygon_map(
            ax,
            merged,
            parameter,
            _uncertainty_cmap(registry),
            f"{clean_parameter_name(parameter)} std",
            registry.palette,
            line_width=registry.style["figure"]["map_line_width"],
        )
        ax.set_title("")
        ax.title.set_visible(False)
        ax.text(0.03, 0.96, clean_parameter_name(parameter), transform=ax.transAxes, ha="left", va="top", fontsize=7.4)
    make_shared_colorbar(fig, plotted, map_axes, "Parameter std", orientation="horizontal", fraction=0.06, pad=0.05)

    caution["flag_code"] = caution["interpretation_flag"].factorize()[0]
    caution_matrix = caution.pivot_table(index="attribute", columns="parameter", values="flag_code", aggfunc="mean")
    caution_matrix = caution_matrix.reindex(index=_focus_attributes(registry), columns=_focus_parameters(registry)).fillna(0.0)
    sns.heatmap(
        caution_matrix,
        ax=ax_b,
        cmap=mcolors.ListedColormap(["#F4F1EC", "#D9DEE5", "#8A817C"]),
        cbar=False,
        linewidths=0.35,
        linecolor="white",
    )
    _set_tick_text(ax_b, [clean_parameter_name(value) for value in caution_matrix.columns], [clean_attribute_name(value) for value in caution_matrix.index])
    label_axes(ax_b, "Caution mask", None, registry.style)
    for row_idx, row_name in enumerate(caution_matrix.index):
        for col_idx, col_name in enumerate(caution_matrix.columns):
            if caution_matrix.loc[row_name, col_name] > 0:
                ax_b.add_patch(Rectangle((col_idx, row_idx), 1, 1, fill=False, hatch="///", edgecolor="#8A817C", linewidth=0.0))

    joint = mean_relationships.merge(
        std_relationships.rename(columns={"mean_spearman_corr": "std_corr"}),
        on=["parameter", "attribute"],
        how="inner",
    )
    joint["agreement"] = np.sign(joint["mean_spearman_corr"] * joint["std_corr"])
    joint_matrix = joint.pivot_table(index="attribute", columns="parameter", values="agreement", aggfunc="mean")
    joint_matrix = joint_matrix.reindex(index=_focus_attributes(registry), columns=_focus_parameters(registry))
    sns.heatmap(
        joint_matrix,
        ax=ax_c,
        cmap=_diverging_cmap(registry),
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        annot=True,
        fmt=".0f",
        cbar=False,
        linewidths=0.35,
        linecolor="white",
    )
    _set_tick_text(ax_c, [clean_parameter_name(value) for value in joint_matrix.columns], [clean_attribute_name(value) for value in joint_matrix.index])
    label_axes(ax_c, "Mean-uncertainty organization", None, registry.style)

    apply_panel_letters([ax_a, map_axes[0], ax_b, ax_c], registry.style)
    return _complete_figure(registry, "10", fig, output_dir, qc, formats)


def build_fig11_conceptual_synthesis(
    registry: FigureDataRegistry,
    output_dir: Path,
    qc: QcCollector,
    formats: tuple[str, ...],
) -> dict[str, Path]:
    width_mm, height_mm = _figure_size(registry, "11")
    fig = make_figure(width_mm, height_mm)
    ax = fig.add_subplot(111)
    ax.axis("off")

    def add_box(x: float, y: float, width: float, height: float, title: str, body: str, facecolor: str) -> None:
        ax.add_patch(Rectangle((x, y), width, height, facecolor=facecolor, edgecolor="white", linewidth=1.0))
        ax.text(x + 0.02, y + height - 0.045, title, fontsize=8.2, fontweight="bold", ha="left", va="top", color="white")
        ax.text(x + 0.02, y + height - 0.12, body, fontsize=7.2, ha="left", va="top", color="white", linespacing=1.2)

    add_box(0.04, 0.20, 0.18, 0.56, "Formulations", "δdtm\nδmcd\nδdtb\n\nSanity check", registry.palette["models"]["δdtm"])
    add_box(0.29, 0.56, 0.17, 0.20, "Reliability", "Cross-seed\nCross-loss", registry.palette["models"]["δdtb"])
    add_box(0.29, 0.24, 0.17, 0.20, "Shared core", "Common control\nscaffold", registry.palette["models"]["δmcd"])
    add_box(0.52, 0.56, 0.17, 0.20, "Mean gradients", "Organized means", "#4C6A92")
    add_box(0.52, 0.24, 0.17, 0.20, "Organized uncertainty", "Structured\nuncertainty", "#4C8C7A")
    add_box(0.77, 0.30, 0.17, 0.34, "Interpretation", "Interpretable\nRegionalization\nUngauged", "#6E7783")

    arrows = [
        ((0.19, 0.62), (0.27, 0.65)),
        ((0.19, 0.38), (0.27, 0.35)),
        ((0.44, 0.65), (0.52, 0.65)),
        ((0.44, 0.35), (0.52, 0.35)),
        ((0.70, 0.65), (0.78, 0.57)),
        ((0.70, 0.35), (0.78, 0.43)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=10, linewidth=1.0, color="#48515A"))

    for x0, y0 in [(0.35, 0.58), (0.35, 0.26), (0.59, 0.58), (0.59, 0.26)]:
        ax.add_patch(Rectangle((x0, y0), 0.055, 0.055, facecolor="#F4F1EC", edgecolor="white", linewidth=0.8))
    ax.add_patch(Rectangle((0.362, 0.594), 0.014, 0.014, facecolor=registry.palette["diverging_signed"][0], edgecolor="none"))
    ax.add_patch(Rectangle((0.382, 0.594), 0.014, 0.014, facecolor=registry.palette["diverging_signed"][-1], edgecolor="none"))
    ax.plot([0.363, 0.400], [0.318, 0.284], color=registry.palette["classes"]["loss-sensitive"], linewidth=2.0)
    ax.plot([0.363, 0.400], [0.340, 0.340], color=registry.palette["classes"]["robust"], linewidth=2.0)
    ax.add_patch(Rectangle((0.602, 0.594), 0.014, 0.014, facecolor=registry.palette["diverging_signed"][0], edgecolor="none"))
    ax.add_patch(Rectangle((0.622, 0.594), 0.014, 0.014, facecolor=registry.palette["diverging_signed"][-1], edgecolor="none"))
    ax.add_patch(Rectangle((0.602, 0.284), 0.014, 0.014, facecolor=registry.palette["sequential_uncertainty"][1], edgecolor="none"))
    ax.add_patch(Rectangle((0.622, 0.284), 0.014, 0.014, facecolor=registry.palette["sequential_uncertainty"][-1], edgecolor="none"))

    apply_panel_letters([ax], registry.style, x=-0.01, y=1.00)
    return _complete_figure(registry, "11", fig, output_dir, qc, formats)


FIGURE_SPECS: dict[str, dict[str, Callable[..., dict[str, Path]] | str]] = {
    "01": {"title": "Performance merged", "stem": "Fig01_performance_merged_revised", "builder": build_fig01_performance_merged},
    "02": {"title": "Cross-seed stability of inferred parameter values", "stem": "Fig02_cross_seed_param_stability_revised", "builder": build_fig02_cross_seed_param_stability},
    "03": {"title": "Cross-seed stability of attribute-parameter correlations", "stem": "Fig03_cross_seed_corr_stability_revised", "builder": build_fig03_cross_seed_corr_stability},
    "04": {"title": "Cross-loss stability of attribute-parameter correlations", "stem": "Fig04_cross_loss_corr_stability_revised", "builder": build_fig04_cross_loss_corr_stability},
    "05": {"title": "Shared dominant-control core across models", "stem": "Fig05_shared_dominant_core_revised", "builder": build_fig05_shared_dominant_core},
    "06": {"title": "Matrix-level similarity of relationship structures", "stem": "Fig06_matrix_similarity_revised", "builder": build_fig06_matrix_similarity},
    "07": {"title": "Post-hoc explainability as supporting evidence", "stem": "Fig07_explainability_support_revised", "builder": build_fig07_explainability_support},
    "08": {"title": "Large-scale gradients in parameter means", "stem": "Fig08_parameter_mean_gradients_revised", "builder": build_fig08_parameter_mean_gradients},
    "09": {"title": "Representative mean gradients and basin archetypes", "stem": "Fig09_representative_mean_gradients_revised", "builder": build_fig09_representative_mean_gradients},
    "10": {"title": "Large-scale gradients in parameter uncertainty", "stem": "Fig10_parameter_uncertainty_gradients_revised", "builder": build_fig10_parameter_uncertainty_gradients},
    "11": {"title": "Conceptual synthesis", "stem": "Fig11_conceptual_synthesis_revised", "builder": build_fig11_conceptual_synthesis},
}
