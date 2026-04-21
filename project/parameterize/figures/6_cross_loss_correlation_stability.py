"""Figure 6: cross-loss stability of dominant parameter-attribute relationships."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from project.parameterize.figures.common import (
    COLORS,
    LOSS_MARKERS,
    add_panel_labels,
    apply_wrr_style,
    pretty_loss_name,
    pretty_model_name,
    pretty_parameter_name,
    save_figure,
)


def _dominant_attribute_table(corr_long: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        corr_long.groupby(["model", "loss", "parameter", "attribute"], as_index=False)
        .agg(
            mean_abs_rho=("abs_rho", "mean"),
            mean_rho=("spearman_rho", "mean"),
        )
    )
    dominant = (
        grouped.sort_values(
            ["model", "loss", "parameter", "mean_abs_rho"],
            ascending=[True, True, True, False],
        )
        .groupby(["model", "loss", "parameter"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return dominant


def _parameter_level_loss_stability(dominant: pd.DataFrame, loss_order: list[str]) -> pd.DataFrame:
    rows = []
    for (model, parameter), subset in dominant.groupby(["model", "parameter"]):
        attr_by_loss = {row["loss"]: row["attribute"] for _, row in subset.iterrows()}
        rho_by_loss = {row["loss"]: row["mean_rho"] for _, row in subset.iterrows()}
        abs_by_loss = {row["loss"]: row["mean_abs_rho"] for _, row in subset.iterrows()}

        available_losses = [loss for loss in loss_order if loss in attr_by_loss]
        pairwise_matches = []
        for idx, loss_a in enumerate(available_losses):
            for loss_b in available_losses[idx + 1 :]:
                pairwise_matches.append(float(attr_by_loss[loss_a] == attr_by_loss[loss_b]))

        rows.append(
            {
                "model": model,
                "parameter": parameter,
                "dominant_attr_retention": float(np.mean(pairwise_matches)) if pairwise_matches else np.nan,
                "dominant_rho_variance": float(np.nanvar([rho_by_loss[loss] for loss in available_losses])),
                "dominant_abs_rho_mean": float(np.nanmean([abs_by_loss[loss] for loss in available_losses])),
                "attribute_sequence": " -> ".join(attr_by_loss[loss] for loss in available_losses),
            }
        )
    return pd.DataFrame(rows)


def generate_figure(data_dict: dict, output_dir, formats=("png", "pdf")):
    apply_wrr_style()
    corr_long = data_dict["corr_long"]
    dominant = _dominant_attribute_table(corr_long)
    stability = _parameter_level_loss_stability(dominant, data_dict["loss_order"])

    fig = plt.figure(figsize=(15.5, 4.8))
    grid = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.35])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c_host = fig.add_subplot(grid[0, 2])
    ax_c_host.axis("off")
    inner_grid = grid[0, 2].subgridspec(3, 1, hspace=0.30)
    ax_c = [fig.add_subplot(inner_grid[idx, 0]) for idx in range(3)]

    retention = (
        stability.groupby("model", as_index=False)["dominant_attr_retention"]
        .mean()
        .rename(columns={"dominant_attr_retention": "retention"})
    )
    ax_a.bar(
        [pretty_model_name(model) for model in retention["model"]],
        retention["retention"],
        color=[COLORS[model] for model in retention["model"]],
    )
    ax_a.set_ylabel("Dominant-attribute retention")
    ax_a.set_ylim(0.0, 1.0)
    ax_a.set_title("Per-parameter dominant attribute agreement")

    import seaborn as sns

    sns.boxplot(
        data=stability,
        x="model",
        y="dominant_rho_variance",
        order=data_dict["model_order"],
        palette=COLORS,
        showfliers=False,
        ax=ax_b,
    )
    ax_b.set_xlabel("")
    ax_b.set_ylabel("Var of dominant ρ across losses")
    ax_b.set_xticklabels([pretty_model_name(name) for name in data_dict["model_order"]], rotation=15)
    ax_b.set_title("Per-parameter dominant-correlation variance")

    for model, axis in zip(data_dict["model_order"], ax_c):
        dominant_subset = dominant.loc[dominant["model"] == model]
        stable_subset = (
            stability.loc[stability["model"] == model]
            .sort_values(["dominant_attr_retention", "dominant_abs_rho_mean"], ascending=[False, False])
            .head(5)
        )
        x = np.arange(len(data_dict["loss_order"]))
        for _, stable_row in stable_subset.iterrows():
            parameter = stable_row["parameter"]
            pair_series = dominant_subset.loc[dominant_subset["parameter"] == parameter]
            y = [
                float(pair_series.loc[pair_series["loss"] == loss, "mean_rho"].iloc[0])
                if not pair_series.loc[pair_series["loss"] == loss].empty
                else np.nan
                for loss in data_dict["loss_order"]
            ]
            labels = [
                str(pair_series.loc[pair_series["loss"] == loss, "attribute"].iloc[0])
                if not pair_series.loc[pair_series["loss"] == loss].empty
                else "n/a"
                for loss in data_dict["loss_order"]
            ]
            axis.plot(
                x,
                y,
                marker="o",
                linewidth=1.4,
                label=f"{pretty_parameter_name(parameter)} ({' → '.join(labels)})",
            )
        axis.set_title(pretty_model_name(model), loc="left")
        axis.set_xticks(x)
        axis.set_xticklabels([pretty_loss_name(loss) for loss in data_dict["loss_order"]], rotation=15)
        axis.set_ylabel("Dominant Spearman ρ")
        axis.grid(axis="y", linestyle="--", alpha=0.25)
    ax_c[0].legend(frameon=False, fontsize=7, ncol=1, bbox_to_anchor=(1.02, 1.02), loc="upper left")

    add_panel_labels([ax_a, ax_b, ax_c[0]], labels=["A", "B", "C"])
    return save_figure(fig, "fig06_cross_loss_correlation_stability", output_dir, formats=formats)
