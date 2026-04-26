"""Standalone analysis for Results 3.3.1 and 3.4.1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

from project.parameterize.analysis.common import (
    build_parser,
    frame_to_markdown,
    load_analysis_data,
    normalize_parameters_to_unit_interval,
    save_frame,
    save_json,
    write_markdown,
)
from project.parameterize.figures.common import (
    COLORS,
    MODEL_ORDER,
    apply_wrr_style,
    pretty_model_name,
    pretty_parameter_name,
    save_figure,
)


RESULTS341_PARAMETERS = ["parBETA", "parFC", "parPERC", "parUZL", "parCFR", "parCWH"]
RESULTS341_ATTRIBUTES = ["aridity", "frac_snow", "slope_mean", "pet_mean", "soil_conductivity", "soil_depth_pelletier"]
GRADIENT_ATTRIBUTES = ["aridity", "frac_snow", "slope_mean"]

ATTRIBUTE_TYPE_COLORS = {
    "climate": "#D95F02",
    "snow": "#1B9E77",
    "topography": "#7570B3",
    "vegetation": "#66A61E",
    "soil": "#A6761D",
    "geology": "#E7298A",
    "other": "#666666",
}


@dataclass(frozen=True)
class Results331Outputs:
    dominant_attribute_summary: pd.DataFrame
    relationship_classes: pd.DataFrame
    model_agreement_summary: dict[str, object]
    model_agreement_summary_md: str


@dataclass(frozen=True)
class Results341Outputs:
    distributional_mean_relationships: pd.DataFrame
    gradient_group_stats: pd.DataFrame


def attribute_type(name: str) -> str:
    if name in {"p_mean", "pet_mean", "p_seasonality", "aridity", "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur"}:
        return "climate"
    if name in {"frac_snow"}:
        return "snow"
    if name in {"elev_mean", "slope_mean", "area_gages2"}:
        return "topography"
    if name.startswith("lai") or name.startswith("gvf") or "forest" in name or "land_cover" in name:
        return "vegetation"
    if name.startswith("soil_") or name in {"root_depth_50", "max_water_content", "sand_frac", "silt_frac", "clay_frac"}:
        return "soil"
    if name.startswith("geol_") or name.startswith("glim_") or "carbonate" in name:
        return "geology"
    return "other"


def short_attribute_label(name: str) -> str:
    labels = {
        "slope_mean": "slope",
        "clay_frac": "clay",
        "pet_mean": "PET",
        "frac_snow": "snow",
        "soil_conductivity": "soil cond.",
        "soil_depth_pelletier": "soil depth",
        "soil_depth_statsgo": "soil depth",
        "aridity": "aridity",
        "elev_mean": "elev",
        "high_prec_dur": "high prec dur",
        "high_prec_freq": "high prec freq",
        "low_prec_freq": "low prec freq",
        "low_prec_dur": "low prec dur",
        "lai_diff": "LAI diff",
        "p_mean": "P",
    }
    return labels.get(name, name.replace("_", " "))


def evidence_label(row: pd.Series, focused_lookup: dict[tuple[str, str], str]) -> str:
    key = (row["parameter"], row["attribute"])
    focused = focused_lookup.get(key)
    if row["relationship_class"] == "robust":
        return "robust"
    if focused == "headline evidence":
        return "supportive"
    if focused == "supportive but not decisive":
        return "supportive"
    if focused == "not supportive":
        return "exploratory"
    if row["relationship_class"] == "loss-sensitive":
        return "supportive"
    return "exploratory"


def build_results331_outputs(
    relationship_classes: pd.DataFrame,
    parameter_level_consistency: pd.DataFrame,
    focused_pair_classes: pd.DataFrame,
) -> Results331Outputs:
    dominant = relationship_classes.loc[relationship_classes["core_rank"] == 1].copy()
    focused_lookup = {
        (row["parameter"], row["attribute"]): row["evidence_class"]
        for _, row in focused_pair_classes.iterrows()
    }
    dominant["previous_evidence"] = dominant.apply(lambda row: evidence_label(row, focused_lookup), axis=1)
    dominant["direction"] = dominant["mean_corr"].map(lambda value: "positive" if value > 0 else "negative")

    summary_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []

    for _, consistency_row in parameter_level_consistency.sort_values("parameter").iterrows():
        parameter = consistency_row["parameter"]
        subset = dominant.loc[dominant["parameter"] == parameter].set_index("model")
        attr_consistency = consistency_row["dominant_attribute_consistency_across_models"]
        direction_consistency = consistency_row["direction_consistency_across_models"]

        if attr_consistency == "all_same" and direction_consistency == "all_same":
            relationship_class = "shared dominant controls"
        elif direction_consistency == "sign_flip_present" or attr_consistency == "all_different":
            relationship_class = "model-sensitive controls"
        else:
            relationship_class = "partially shared controls"

        agreement_models = []
        attr_values = {model: subset.loc[model, "attribute"] for model in subset.index}
        counts = pd.Series(attr_values).value_counts()
        if not counts.empty and int(counts.iloc[0]) > 1:
            shared_attribute = str(counts.index[0])
            agreement_models = sorted([model for model, attr in attr_values.items() if attr == shared_attribute])
        else:
            shared_attribute = ""

        summary_rows.append(
            {
                "parameter": parameter,
                "deterministic_attribute": subset.loc["deterministic", "attribute"],
                "deterministic_direction": subset.loc["deterministic", "direction"],
                "deterministic_previous_evidence": subset.loc["deterministic", "previous_evidence"],
                "mc_dropout_attribute": subset.loc["mc_dropout", "attribute"],
                "mc_dropout_direction": subset.loc["mc_dropout", "direction"],
                "mc_dropout_previous_evidence": subset.loc["mc_dropout", "previous_evidence"],
                "distributional_attribute": subset.loc["distributional", "attribute"],
                "distributional_direction": subset.loc["distributional", "direction"],
                "distributional_previous_evidence": subset.loc["distributional", "previous_evidence"],
                "dominant_attribute_consistency": attr_consistency,
                "direction_consistency": direction_consistency,
                "overall_relationship_class": relationship_class,
                "comment": consistency_row["comments"],
            }
        )

        class_rows.append(
            {
                "parameter": parameter,
                "relationship_class": relationship_class,
                "dominant_attribute_consistency": attr_consistency,
                "direction_consistency": direction_consistency,
                "shared_attribute_if_any": shared_attribute,
                "agreeing_models": ", ".join(agreement_models),
                "deterministic_attribute": subset.loc["deterministic", "attribute"],
                "mc_dropout_attribute": subset.loc["mc_dropout", "attribute"],
                "distributional_attribute": subset.loc["distributional", "attribute"],
                "comment": consistency_row["comments"],
            }
        )

    dominant_summary = pd.DataFrame(summary_rows)
    relationship_class_table = pd.DataFrame(class_rows)

    total = len(relationship_class_table)
    class_counts = relationship_class_table["relationship_class"].value_counts().to_dict()
    attr_counts = relationship_class_table["dominant_attribute_consistency"].value_counts().to_dict()
    sign_flip = relationship_class_table.loc[
        relationship_class_table["direction_consistency"] == "sign_flip_present", "parameter"
    ].tolist()
    all_same_sign = relationship_class_table.loc[
        (relationship_class_table["dominant_attribute_consistency"] == "all_same")
        & (relationship_class_table["direction_consistency"] == "all_same"),
        "parameter",
    ].tolist()

    summary_dict = {
        "parameter_count": total,
        "relationship_class_counts": class_counts,
        "relationship_class_proportions": {key: value / total for key, value in class_counts.items()},
        "dominant_attribute_consistency_counts": attr_counts,
        "dominant_attribute_consistency_proportions": {key: value / total for key, value in attr_counts.items()},
        "all_same_and_direction_consistent_parameters": all_same_sign,
        "sign_flip_present_parameters": sign_flip,
        "headline_common_controls": relationship_class_table.loc[
            relationship_class_table["relationship_class"] == "shared dominant controls", "parameter"
        ].tolist(),
        "model_sensitive_controls": relationship_class_table.loc[
            relationship_class_table["relationship_class"] == "model-sensitive controls", "parameter"
        ].tolist(),
    }

    md_lines = [
        "# Results 3.3.1 Model Agreement Summary",
        "",
        f"- Total parameters: `{total}`",
        f"- Shared dominant controls: `{class_counts.get('shared dominant controls', 0)}`",
        f"- Partially shared controls: `{class_counts.get('partially shared controls', 0)}`",
        f"- Model-sensitive controls: `{class_counts.get('model-sensitive controls', 0)}`",
        f"- All-same dominant attribute + direction: `{', '.join(all_same_sign)}`",
        f"- Sign flips present: `{', '.join(sign_flip) if sign_flip else 'none'}`",
        "",
        "## Parameter Classes",
        "",
        frame_to_markdown(relationship_class_table),
        "",
    ]

    return Results331Outputs(
        dominant_attribute_summary=dominant_summary,
        relationship_classes=relationship_class_table,
        model_agreement_summary=summary_dict,
        model_agreement_summary_md="\n".join(md_lines),
    )


def build_results341_outputs(
    params_long: pd.DataFrame,
    attributes: pd.DataFrame,
    correlation_mean_std_var: pd.DataFrame,
    relationship_classes: pd.DataFrame,
    focused_pair_classes: pd.DataFrame,
    parameter_bounds: dict[str, tuple[float, float]],
) -> tuple[Results341Outputs, pd.DataFrame]:
    dist_corr = correlation_mean_std_var.loc[
        (correlation_mean_std_var["model"] == "distributional")
        & (correlation_mean_std_var["method"] == "spearman")
        & (correlation_mean_std_var["parameter"].isin(RESULTS341_PARAMETERS))
        & (correlation_mean_std_var["attribute"].isin(RESULTS341_ATTRIBUTES))
    ].copy()
    dist_corr = (
        dist_corr.groupby(["parameter", "attribute"], as_index=False)
        .agg(
            mean_corr=("mean_corr", "mean"),
            mean_abs_corr=("mean_abs_corr", "mean"),
            std_corr=("std_corr", "mean"),
        )
    )

    pearson = correlation_mean_std_var.loc[
        (correlation_mean_std_var["model"] == "distributional")
        & (correlation_mean_std_var["method"] == "pearson")
        & (correlation_mean_std_var["parameter"].isin(RESULTS341_PARAMETERS))
        & (correlation_mean_std_var["attribute"].isin(RESULTS341_ATTRIBUTES))
    ].groupby(["parameter", "attribute"], as_index=False)["mean_corr"].mean().rename(columns={"mean_corr": "mean_pearson_corr"})
    kendall = correlation_mean_std_var.loc[
        (correlation_mean_std_var["model"] == "distributional")
        & (correlation_mean_std_var["method"] == "kendall")
        & (correlation_mean_std_var["parameter"].isin(RESULTS341_PARAMETERS))
        & (correlation_mean_std_var["attribute"].isin(RESULTS341_ATTRIBUTES))
    ].groupby(["parameter", "attribute"], as_index=False)["mean_corr"].mean().rename(columns={"mean_corr": "mean_kendall_corr"})

    support_lookup = relationship_classes.loc[relationship_classes["model"] == "distributional", [
        "parameter", "attribute", "relationship_class", "core_rank", "seed_stable", "loss_stable"
    ]].copy()
    focused_lookup = {
        (row["parameter"], row["attribute"]): row["evidence_class"]
        for _, row in focused_pair_classes.iterrows()
    }

    dist_corr = dist_corr.merge(pearson, on=["parameter", "attribute"], how="left").merge(
        kendall, on=["parameter", "attribute"], how="left"
    )
    dist_corr["sign"] = dist_corr["mean_corr"].map(lambda value: "positive" if value > 0 else "negative")

    role_lookup = {}
    for _, row in support_lookup.iterrows():
        key = (row["parameter"], row["attribute"])
        if int(row["core_rank"]) == 1:
            role_lookup[key] = "dominant"
        elif int(row["core_rank"]) <= 3:
            role_lookup[key] = "supportive"
        else:
            role_lookup[key] = "secondary"

    dist_corr["relationship_role"] = dist_corr.apply(
        lambda row: role_lookup.get((row["parameter"], row["attribute"]), "secondary"),
        axis=1,
    )
    dist_corr["previous_analysis_class"] = dist_corr.apply(
        lambda row: support_lookup.loc[
            (support_lookup["parameter"] == row["parameter"]) & (support_lookup["attribute"] == row["attribute"]),
            "relationship_class",
        ].iloc[0]
        if not support_lookup.loc[
            (support_lookup["parameter"] == row["parameter"]) & (support_lookup["attribute"] == row["attribute"])
        ].empty
        else "not_selected",
        axis=1,
    )
    dist_corr["robust_flag"] = dist_corr["previous_analysis_class"].eq("robust")
    dist_corr["focused_evidence"] = dist_corr.apply(
        lambda row: focused_lookup.get((row["parameter"], row["attribute"]), "not_focused"),
        axis=1,
    )
    distributional_mean_relationships = dist_corr.rename(
        columns={
            "mean_corr": "mean_spearman_corr",
            "mean_abs_corr": "abs_spearman_corr",
            "std_corr": "seed_loss_std_spearman_corr",
        }
    )[
        [
            "parameter",
            "attribute",
            "mean_spearman_corr",
            "abs_spearman_corr",
            "sign",
            "relationship_role",
            "previous_analysis_class",
            "robust_flag",
            "focused_evidence",
            "mean_pearson_corr",
            "mean_kendall_corr",
        ]
    ].sort_values(["parameter", "abs_spearman_corr"], ascending=[True, False]).reset_index(drop=True)

    distributional_params = params_long.loc[
        (params_long["model"] == "distributional") & (params_long["parameter"].isin(RESULTS341_PARAMETERS))
    ].copy()
    distributional_params = (
        distributional_params.groupby(["basin_id", "parameter"], as_index=False)["mean"]
        .mean()
    )
    distributional_params = normalize_parameters_to_unit_interval(
        distributional_params,
        parameter_bounds,
        value_column="mean",
    )
    averaged = distributional_params.merge(attributes, on="basin_id", how="inner")

    gradient_rows = []
    plot_rows = []
    for gradient in GRADIENT_ATTRIBUTES:
        bins = pd.qcut(averaged[gradient], q=3, labels=["low", "mid", "high"], duplicates="drop")
        averaged[f"{gradient}_group"] = bins
        for parameter in RESULTS341_PARAMETERS:
            subset = averaged.loc[averaged["parameter"] == parameter].dropna(subset=[f"{gradient}_group"])
            if subset.empty:
                continue
            low_values = subset.loc[subset[f"{gradient}_group"] == "low", "mean_unit"]
            high_values = subset.loc[subset[f"{gradient}_group"] == "high", "mean_unit"]
            high_low_p = np.nan
            high_low_diff = np.nan
            if len(low_values) > 0 and len(high_values) > 0:
                high_low_p = float(mannwhitneyu(high_values, low_values, alternative="two-sided").pvalue)
                high_low_diff = float(high_values.median() - low_values.median())
            for group_name, group_df in subset.groupby(f"{gradient}_group", observed=False):
                gradient_rows.append(
                    {
                        "gradient_attribute": gradient,
                        "gradient_group": str(group_name),
                        "parameter": parameter,
                        "sample_count": int(group_df.shape[0]),
                        "mean_parameter_raw": float(group_df["mean"].mean()),
                        "median_parameter_raw": float(group_df["mean"].median()),
                        "std_parameter_raw": float(group_df["mean"].std(ddof=0)),
                        "mean_parameter_unit": float(group_df["mean_unit"].mean()),
                        "median_parameter_unit": float(group_df["mean_unit"].median()),
                        "std_parameter_unit": float(group_df["mean_unit"].std(ddof=0)),
                        "high_minus_low_median_unit": high_low_diff,
                        "high_vs_low_mannwhitney_p": high_low_p,
                    }
                )
            plot_rows.append(
                subset[["basin_id", "parameter", "mean", "mean_unit", gradient, f"{gradient}_group"]].rename(
                    columns={f"{gradient}_group": "gradient_group"}
                ).assign(gradient_attribute=gradient)
            )

    gradient_group_stats = pd.DataFrame(gradient_rows).sort_values(
        ["gradient_attribute", "parameter", "gradient_group"]
    ).reset_index(drop=True)
    plot_data = pd.concat(plot_rows, ignore_index=True)
    return Results341Outputs(
        distributional_mean_relationships=distributional_mean_relationships,
        gradient_group_stats=gradient_group_stats,
    ), plot_data


def figure_results331_heatmap(dominant_summary: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    apply_wrr_style()
    order = dominant_summary["parameter"].tolist()
    plot = dominant_summary.set_index("parameter").loc[order]
    matrix = np.zeros((len(order), len(MODEL_ORDER)), dtype=int)
    annotations = np.empty_like(matrix, dtype=object)
    category_order = list(ATTRIBUTE_TYPE_COLORS)
    category_to_code = {name: idx for idx, name in enumerate(category_order)}

    for row_idx, parameter in enumerate(order):
        for col_idx, model in enumerate(MODEL_ORDER):
            attr = plot.loc[parameter, f"{model}_attribute"]
            direction = plot.loc[parameter, f"{model}_direction"]
            category = attribute_type(attr)
            matrix[row_idx, col_idx] = category_to_code[category]
            annotations[row_idx, col_idx] = f"{short_attribute_label(attr)}\n{'+' if direction == 'positive' else '-'}"

    fig, ax = plt.subplots(figsize=(8.8, max(6.5, 0.42 * len(order))))
    cmap = matplotlib.colors.ListedColormap([ATTRIBUTE_TYPE_COLORS[name] for name in category_order])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(category_order) - 0.5)
    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_xticklabels([pretty_model_name(model) for model in MODEL_ORDER], rotation=0)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([pretty_parameter_name(parameter) for parameter in order])
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, annotations[row_idx, col_idx], ha="center", va="center", fontsize=7)
    ax.set_title("Dominant Basin Controls Across Models")
    legend_handles = [Patch(facecolor=color, label=category.title()) for category, color in ATTRIBUTE_TYPE_COLORS.items()]
    ax.legend(handles=legend_handles, frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.08), loc="upper center")
    return save_figure(fig, "results331_dominant_attribute_heatmap", output_dir, formats=("png",))


def figure_results331_network(
    dominant_summary: pd.DataFrame,
    relationship_classes: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    apply_wrr_style()
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    ax.axis("off")

    params = relationship_classes["parameter"].tolist()
    dominant_lookup = dominant_summary.set_index("parameter")
    unique_attrs = sorted(
        {
            dominant_summary[f"{model}_attribute"].iloc[idx]
            for model in ("deterministic", "mc_dropout", "distributional")
            for idx in range(len(dominant_summary))
        }
    )

    attr_y = {attr: value for attr, value in zip(unique_attrs, np.linspace(0.95, 0.05, len(unique_attrs)))}
    param_y = {param: value for param, value in zip(params, np.linspace(0.95, 0.05, len(params)))}

    for attr, y in attr_y.items():
        ax.scatter(0.15, y, s=180, color=ATTRIBUTE_TYPE_COLORS[attribute_type(attr)], edgecolor="black", linewidth=0.6)
        ax.text(0.13, y, short_attribute_label(attr), ha="right", va="center", fontsize=8)
    for parameter, y in param_y.items():
        row = relationship_classes.loc[relationship_classes["parameter"] == parameter].iloc[0]
        if row["relationship_class"] == "shared dominant controls":
            color = "#222222"
        elif row["relationship_class"] == "partially shared controls":
            color = "#666666"
        else:
            color = "#BDBDBD"
        ax.scatter(0.85, y, s=180, color=color, edgecolor="black", linewidth=0.6)
        ax.text(0.87, y, pretty_parameter_name(parameter), ha="left", va="center", fontsize=8)

    for parameter in params:
        class_row = relationship_classes.loc[relationship_classes["parameter"] == parameter].iloc[0]
        dom_row = dominant_lookup.loc[parameter]
        param_yval = param_y[parameter]
        attr_values = {model: dom_row[f"{model}_attribute"] for model in MODEL_ORDER}
        counts = pd.Series(attr_values).value_counts()

        if class_row["relationship_class"] == "shared dominant controls":
            attr = dom_row["distributional_attribute"]
            ax.plot([0.2, 0.8], [attr_y[attr], param_yval], color="#222222", linewidth=3.0, alpha=0.9)
        elif class_row["relationship_class"] == "partially shared controls" and counts.iloc[0] >= 2:
            shared_attr = str(counts.index[0])
            ax.plot([0.2, 0.8], [attr_y[shared_attr], param_yval], color="#555555", linewidth=2.2, alpha=0.9)
            ax.text(0.5, (attr_y[shared_attr] + param_yval) / 2.0 + 0.012, "2 models", fontsize=7, color="#444444")
            for model in MODEL_ORDER:
                attr = attr_values[model]
                if attr == shared_attr:
                    continue
                ax.plot([0.2, 0.8], [attr_y[attr], param_yval], color=COLORS[model], linewidth=1.5, linestyle="--", alpha=0.75)
        else:
            offsets = {"deterministic": -0.012, "mc_dropout": 0.0, "distributional": 0.012}
            for model in MODEL_ORDER:
                attr = attr_values[model]
                ax.plot(
                    [0.2, 0.8],
                    [attr_y[attr], param_yval + offsets[model]],
                    color=COLORS[model],
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.75,
                )

    legend_handles = [
        Line2D([0], [0], color="#222222", linewidth=3.0, label="Shared dominant control"),
        Line2D([0], [0], color="#555555", linewidth=2.2, label="Partially shared control"),
        Line2D([0], [0], color="#999999", linewidth=1.5, linestyle="--", label="Model-specific edge"),
        Line2D([0], [0], color=COLORS["deterministic"], linewidth=1.5, linestyle="--", label="Deterministic-specific"),
        Line2D([0], [0], color=COLORS["mc_dropout"], linewidth=1.5, linestyle="--", label="MC-dropout-specific"),
        Line2D([0], [0], color=COLORS["distributional"], linewidth=1.5, linestyle="--", label="Distributional-specific"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2)
    ax.set_title("Shared vs Model-Sensitive Dominant Controls")
    return save_figure(fig, "results331_shared_relationship_network", output_dir, formats=("png",))


def figure_results341_heatmap(distributional_mean_relationships: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    apply_wrr_style()
    matrix = distributional_mean_relationships.pivot(index="attribute", columns="parameter", values="mean_spearman_corr")
    matrix = matrix.loc[[attr for attr in RESULTS341_ATTRIBUTES if attr in matrix.index], [param for param in RESULTS341_PARAMETERS if param in matrix.columns]]
    robust_flags = distributional_mean_relationships.pivot(index="attribute", columns="parameter", values="robust_flag").reindex_like(matrix)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    annot = matrix.copy()
    for column in annot.columns:
        annot[column] = annot[column].map(lambda value: f"{value:.2f}")
    sns.heatmap(
        matrix,
        cmap="coolwarm",
        center=0.0,
        annot=annot,
        fmt="",
        linewidths=0.5,
        cbar_kws={"label": "Mean Spearman correlation"},
        ax=ax,
    )
    for row_idx, attr in enumerate(matrix.index):
        for col_idx, param in enumerate(matrix.columns):
            if bool(robust_flags.loc[attr, param]):
                ax.text(col_idx + 0.9, row_idx + 0.15, "*", ha="center", va="center", fontsize=10, color="black")
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Attribute")
    ax.set_xticklabels([pretty_parameter_name(param) for param in matrix.columns], rotation=20, ha="right")
    ax.set_yticklabels([short_attribute_label(attr) for attr in matrix.index], rotation=0)
    ax.set_title("Distributional Parameter-Mean Gradients")
    return save_figure(fig, "results341_distributional_mean_heatmap", output_dir, formats=("png",))


def figure_results341_gradient_boxplots(plot_data: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    apply_wrr_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
    hue_order = ["low", "mid", "high"]
    palette = {"low": "#4C78A8", "mid": "#BEBEBE", "high": "#F58518"}
    for ax, gradient in zip(axes, GRADIENT_ATTRIBUTES):
        subset = plot_data.loc[plot_data["gradient_attribute"] == gradient].copy()
        sns.boxplot(
            data=subset,
            x="parameter",
            y="mean_unit",
            hue="gradient_group",
            order=RESULTS341_PARAMETERS,
            hue_order=hue_order,
            palette=palette,
            showfliers=False,
            ax=ax,
        )
        ax.set_title(short_attribute_label(gradient).title())
        ax.set_xlabel("")
        ax.set_ylabel("Normalized parameter mean" if gradient == GRADIENT_ATTRIBUTES[0] else "")
        ax.set_xticks(range(len(RESULTS341_PARAMETERS)))
        ax.set_xticklabels([pretty_parameter_name(param) for param in RESULTS341_PARAMETERS], rotation=20, ha="right")
        if ax is not axes[0]:
            ax.get_legend().remove()
    axes[0].legend(title="Gradient group", frameon=False, loc="upper left")
    return save_figure(fig, "results341_gradient_boxplots", output_dir, formats=("png",))


def render_report(
    report_path: Path,
    results331: Results331Outputs,
    results341: Results341Outputs,
    focused_pair_classes: pd.DataFrame,
    figure_paths: dict[str, dict[str, Path]],
) -> Path:
    dominant = results331.dominant_attribute_summary
    shared = results331.relationship_classes.loc[
        results331.relationship_classes["relationship_class"] == "shared dominant controls"
    ]["parameter"].tolist()
    headline_common = dominant.loc[
        (dominant["overall_relationship_class"] == "shared dominant controls")
        & (dominant["distributional_previous_evidence"] != "exploratory")
    ]["parameter"].tolist()
    partially = results331.relationship_classes.loc[
        results331.relationship_classes["relationship_class"] == "partially shared controls"
    ]["parameter"].tolist()
    sensitive = results331.relationship_classes.loc[
        results331.relationship_classes["relationship_class"] == "model-sensitive controls"
    ]["parameter"].tolist()

    dist_strengthened = focused_pair_classes.loc[
        focused_pair_classes["evidence_class"].isin(["headline evidence", "supportive but not decisive"])
    ]["pair_label"].tolist()

    relationship_table = results341.distributional_mean_relationships.copy()
    strongest_gradients = relationship_table.sort_values("abs_spearman_corr", ascending=False).head(10)
    headline_gradients = relationship_table.loc[
        relationship_table["relationship_role"] == "dominant"
    ].sort_values("abs_spearman_corr", ascending=False)
    support_gradients = relationship_table.loc[
        relationship_table["relationship_role"] == "supportive"
    ].sort_values("abs_spearman_corr", ascending=False)
    headline_gradient_labels = (headline_gradients["parameter"] + "–" + headline_gradients["attribute"]).tolist()
    support_gradient_labels = (support_gradients["parameter"] + "–" + support_gradients["attribute"]).tolist()
    strongest_gradient_labels = (strongest_gradients["parameter"] + "–" + strongest_gradients["attribute"]).tolist()
    supportive_gradients = strongest_gradients.loc[
        ~strongest_gradients.index.isin(headline_gradients.index)
    ]

    results_lines = [
        "Objective",
        "Results 3.3.1 summary",
        "Results 3.4.1 summary",
        "Headline results suitable for the main paper",
        "Supporting but non-headline results",
        "Suggested wording for Results paragraphs",
        "Which findings should be moved into Discussion",
    ]

    sections = [
        (
            "Objective",
            "\n".join(
                [
                    "This analysis shifts from method feasibility to interpretation.",
                    "Results 3.3.1 asks whether the three model families recover shared dominant basin controls.",
                    "Results 3.4.1 then examines what large-scale parameter-mean gradients are learned by the distributional model once the relationship structure has been shown to be reliable.",
                ]
            ),
        ),
        (
            "Results 3.3.1 summary",
            "\n\n".join(
                [
                    f"Shared dominant controls: {', '.join(shared)}.",
                    f"Partially shared controls: {', '.join(partially)}.",
                    f"Model-sensitive controls: {', '.join(sensitive)}.",
                    frame_to_markdown(dominant),
                    f"Figure: `{figure_paths['results331_heatmap']['png']}`",
                    f"Figure: `{figure_paths['results331_network']['png']}`",
                ]
            ),
        ),
        (
            "Results 3.4.1 summary",
            "\n\n".join(
                [
                    "Distributional-only large-scale gradients are summarized below for a small set of high-value parameters and attributes.",
                    frame_to_markdown(results341.distributional_mean_relationships),
                    frame_to_markdown(results341.gradient_group_stats.head(30)),
                    f"Figure: `{figure_paths['results341_heatmap']['png']}`",
                    f"Figure: `{figure_paths['results341_boxplots']['png']}`",
                ]
            ),
        ),
        (
            "Headline results suitable for the main paper",
            "\n".join(
                [
                    f"- Strongest shared dominant controls across all three models: {', '.join(headline_common)}.",
                    f"- Distributional strengthens but does not contradict shared hydrologic signals on: {', '.join(dist_strengthened)}.",
                    f"- Distributional mean gradients most suitable for the paper body: {', '.join(headline_gradient_labels)}.",
                ]
            ),
        ),
        (
            "Supporting but non-headline results",
            "\n".join(
                [
                    f"- Partially shared controls that help support common structure without being universal: {', '.join(partially)}.",
                    f"- Additional gradient relationships that are interpretable but weaker: {', '.join(support_gradient_labels)}.",
                ]
            ),
        ),
        (
            "Suggested wording for Results paragraphs",
            "\n".join(
                [
                    "Results 3.3.1: Across the three model families, several dominant basin controls were conserved, indicating that the learned parameter structures are not arbitrary model artifacts. In particular, the same dominant controls emerged for the clearest parameters, while the remaining disagreements were concentrated in a smaller set of model-sensitive controls.",
                    "Results 3.4.1: Within the distributional model, parameter means varied systematically along large-scale hydroclimatic and physiographic gradients. The strongest gradients aligned with the dominant-control analysis, showing that the recovered controls are expressed not only as pairwise correlations but also as coherent shifts in basin-scale parameter distributions.",
                ]
            ),
        ),
        (
            "Which findings should be moved into Discussion",
            "\n".join(
                [
                    f"- Model-sensitive controls that likely reflect process compensation: {', '.join(sensitive)}.",
                    "- Pair-level focused evidence that supports distributional over deterministic but is not universal should be interpreted cautiously rather than framed as blanket superiority.",
                    "- Snow- and routing-related relationships that change dominant attribute across models should be discussed as alternative but plausible control structures.",
                ]
            ),
        ),
        (
            "Answers to the Required Questions",
            "\n".join(
                [
                    f"1. The strongest shared dominant attribute–parameter relationships are {', '.join(headline_common)}.",
                    f"2. Distributional strengthens structure without contradicting common controls most clearly on {', '.join(dist_strengthened)}.",
                    f"3. Distributional parameter means vary most clearly along {', '.join(sorted(set(headline_gradients['attribute'])))}; the broader top-gradient set is {', '.join(strongest_gradient_labels)}.",
                    f"4. Findings strong enough for Results are the shared dominant controls and the strongest distributional gradient pairs; model-sensitive controls and ambiguous snow/routing cases are better reserved for Discussion.",
                ]
            ),
        ),
    ]
    return write_markdown(report_path, title="Results 3.3.1 and 3.4.1 Analysis", sections=sections)


def run_results331_results341(
    params_long: pd.DataFrame,
    attributes: pd.DataFrame,
    parameter_bounds: dict[str, tuple[float, float]],
    analysis_root: Path,
    relationship_classes: pd.DataFrame,
    parameter_level_consistency: pd.DataFrame,
    focused_pair_classes: pd.DataFrame,
    correlation_mean_std_var: pd.DataFrame,
) -> dict[str, Path]:
    figures_dir = analysis_root / "figures"
    reports_dir = analysis_root / "reports"
    tables_dir = analysis_root / "correlation_summaries"

    results331 = build_results331_outputs(
        relationship_classes=relationship_classes,
        parameter_level_consistency=parameter_level_consistency,
        focused_pair_classes=focused_pair_classes,
    )
    results341, plot_data = build_results341_outputs(
        params_long=params_long,
        attributes=attributes,
        correlation_mean_std_var=correlation_mean_std_var,
        relationship_classes=relationship_classes,
        focused_pair_classes=focused_pair_classes,
        parameter_bounds=parameter_bounds,
    )

    paths = {
        "results331_dominant_attribute_summary": save_frame(
            results331.dominant_attribute_summary, tables_dir / "results331_dominant_attribute_summary.csv"
        ),
        "results331_relationship_classes": save_frame(
            results331.relationship_classes, tables_dir / "results331_relationship_classes.csv"
        ),
        "results331_model_agreement_summary_json": save_json(
            results331.model_agreement_summary, tables_dir / "results331_model_agreement_summary.json"
        ),
        "results331_model_agreement_summary_md": (tables_dir / "results331_model_agreement_summary.md"),
        "results341_distributional_mean_relationships": save_frame(
            results341.distributional_mean_relationships, tables_dir / "results341_distributional_mean_relationships.csv"
        ),
        "results341_gradient_group_stats": save_frame(
            results341.gradient_group_stats, tables_dir / "results341_gradient_group_stats.csv"
        ),
    }
    paths["results331_model_agreement_summary_md"].write_text(results331.model_agreement_summary_md, encoding="utf-8")

    fig331_heatmap = figure_results331_heatmap(results331.dominant_attribute_summary, figures_dir)
    fig331_network = figure_results331_network(results331.dominant_attribute_summary, results331.relationship_classes, figures_dir)
    fig341_heatmap = figure_results341_heatmap(results341.distributional_mean_relationships, figures_dir)
    fig341_boxplots = figure_results341_gradient_boxplots(plot_data, figures_dir)

    paths["report"] = render_report(
        report_path=reports_dir / "results331_results341_report.md",
        results331=results331,
        results341=results341,
        focused_pair_classes=focused_pair_classes,
        figure_paths={
            "results331_heatmap": fig331_heatmap,
            "results331_network": fig331_network,
            "results341_heatmap": fig341_heatmap,
            "results341_boxplots": fig341_boxplots,
        },
    )
    return paths


def main() -> None:
    parser = build_parser("Run Results 3.3.1 and 3.4.1 analyses.")
    parser.add_argument(
        "--relationship-classes-csv",
        default="project/parameterize/outputs/analysis/stability_stats/correlation_summaries/relationship_classes.csv",
    )
    parser.add_argument(
        "--parameter-level-consistency-csv",
        default="project/parameterize/outputs/analysis/stability_stats/correlation_summaries/parameter_level_consistency.csv",
    )
    parser.add_argument(
        "--focused-pair-classes-csv",
        default="project/parameterize/outputs/analysis/stability_stats/correlation_summaries/focused_pair_classes.csv",
    )
    parser.add_argument(
        "--correlation-mean-std-var-csv",
        default="project/parameterize/outputs/analysis/stability_stats/correlation_summaries/correlation_mean_std_var.csv",
    )
    args = parser.parse_args()

    data = load_analysis_data(
        config_path=args.config,
        outputs_root=args.outputs_root,
        analysis_root=args.analysis_root,
        device=args.device,
        parameter_csv=args.parameter_csv,
        attribute_csv=args.attribute_csv,
    )
    if "params_long" not in data or "attributes" not in data:
        raise ValueError("--parameter-csv and --attribute-csv are required.")

    paths = run_results331_results341(
        params_long=data["params_long"],
        attributes=data["attributes"],
        parameter_bounds=data["parameter_bounds"],
        analysis_root=data["stability_analysis_root"],
        relationship_classes=pd.read_csv(args.relationship_classes_csv),
        parameter_level_consistency=pd.read_csv(args.parameter_level_consistency_csv),
        focused_pair_classes=pd.read_csv(args.focused_pair_classes_csv),
        correlation_mean_std_var=pd.read_csv(args.correlation_mean_std_var_csv),
    )
    print(paths["report"])


if __name__ == "__main__":
    main()
