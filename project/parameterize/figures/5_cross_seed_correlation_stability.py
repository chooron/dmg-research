"""Figure 5: cross-seed correlation stability."""

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
    pretty_parameter_name,
    reference_loss_only,
    save_figure,
    top_pairs_by_abs_rho,
)


def _pair_stability(corr_long: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        corr_long.groupby(["model", "parameter", "attribute"], as_index=False)
        .agg(
            variance_r2=("spearman_r2", "var"),
            range_r2=("spearman_r2", lambda values: float(np.nanmax(values) - np.nanmin(values))),
            sign_consistency=("spearman_rho", lambda values: float(max((np.sign(values) > 0).sum(), (np.sign(values) < 0).sum()) / max((np.sign(values) != 0).sum(), 1))),
            abs_rho=("abs_rho", "mean"),
        )
    )
    return grouped


def generate_figure(data_dict: dict, output_dir, formats=("png", "pdf")):
    apply_wrr_style()
    corr_long = reference_loss_only(data_dict["corr_long"], data_dict)
    pair_frame = _pair_stability(corr_long)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    sns.boxplot(
        data=pair_frame,
        x="model",
        y="variance_r2",
        order=data_dict["model_order"],
        palette=COLORS,
        showfliers=False,
        ax=axes[0],
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Var(Spearman R²)")
    axes[0].set_xticklabels([pretty_model_name(name) for name in data_dict["model_order"]], rotation=15)

    sns.boxplot(
        data=pair_frame,
        x="model",
        y="range_r2",
        order=data_dict["model_order"],
        palette=COLORS,
        showfliers=False,
        ax=axes[1],
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Range of Spearman R²")
    axes[1].set_xticklabels([pretty_model_name(name) for name in data_dict["model_order"]], rotation=15)

    top_pairs = top_pairs_by_abs_rho(corr_long, top_k=15)
    heatmap_rows = []
    for model in data_dict["model_order"]:
        row = []
        for parameter, attribute in top_pairs:
            value = pair_frame.loc[
                (pair_frame["model"] == model)
                & (pair_frame["parameter"] == parameter)
                & (pair_frame["attribute"] == attribute),
                "sign_consistency",
            ]
            row.append(float(value.iloc[0]) if not value.empty else np.nan)
        heatmap_rows.append(row)

    labels = [f"{pretty_parameter_name(parameter)}\n{attribute}" for parameter, attribute in top_pairs]
    sns.heatmap(
        np.asarray(heatmap_rows),
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        ax=axes[2],
        cbar_kws={"label": "Sign consistency"},
    )
    axes[2].set_yticklabels([pretty_model_name(model) for model in data_dict["model_order"]], rotation=0)
    axes[2].set_xticklabels(labels, rotation=45, ha="right")
    axes[2].set_xlabel("Top |R| parameter-attribute pairs")

    add_panel_labels(axes)
    return save_figure(fig, "fig05_cross_seed_correlation_stability", output_dir, formats=formats)

