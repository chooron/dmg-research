"""Figure 3: cross-seed parameter stability overview."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from project.parameterize.figures.common import (
    COLORS,
    add_panel_labels,
    apply_wrr_style,
    pretty_model_name,
    reference_loss_only,
    save_figure,
)


def _variance_frame(params_long: pd.DataFrame) -> pd.DataFrame:
    grouped = params_long.groupby(["model", "loss", "basin_id", "parameter"], as_index=False).agg(
        mean_of_seed=("mean", "mean"),
        raw_variance=("mean", "var"),
    )
    return grouped


def _add_range_normalized_variance(grouped: pd.DataFrame, parameter_bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    grouped = grouped.copy()
    grouped["parameter_range"] = grouped["parameter"].map(
        lambda name: float(parameter_bounds[name][1] - parameter_bounds[name][0])
    )
    grouped["normalized_variance"] = grouped["raw_variance"] / (grouped["parameter_range"] ** 2)
    return grouped


def generate_figure(data_dict: dict, output_dir, formats=("png", "pdf")):
    apply_wrr_style()
    params = reference_loss_only(data_dict["params_long"], data_dict)
    variance_frame = _add_range_normalized_variance(_variance_frame(params), data_dict["parameter_bounds"])

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    sns.boxplot(
        data=variance_frame,
        x="model",
        y="normalized_variance",
        order=data_dict["model_order"],
        palette=COLORS,
        showfliers=False,
        ax=axes[0],
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Variance / parameter_range²")
    axes[0].set_xticklabels([pretty_model_name(name) for name in data_dict["model_order"]], rotation=15)

    sns.boxplot(
        data=variance_frame,
        x="model",
        y="raw_variance",
        order=data_dict["model_order"],
        palette=COLORS,
        showfliers=False,
        ax=axes[1],
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Raw variance")
    axes[1].set_xticklabels([pretty_model_name(name) for name in data_dict["model_order"]], rotation=15)

    per_parameter = (
        variance_frame.groupby(["model", "parameter"], as_index=False)["normalized_variance"]
        .mean()
    )
    det_order = (
        per_parameter.loc[per_parameter["model"] == "deterministic"]
        .sort_values("normalized_variance", ascending=False)["parameter"]
        .tolist()
    )
    sns.barplot(
        data=per_parameter,
        x="parameter",
        y="normalized_variance",
        hue="model",
        order=det_order,
        hue_order=data_dict["model_order"],
        palette=COLORS,
        ax=axes[2],
    )
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Mean range-normalized variance")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].legend(title="")

    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    add_panel_labels(axes)
    return save_figure(fig, "fig03_cross_seed_parameter_stability", output_dir, formats=formats)
