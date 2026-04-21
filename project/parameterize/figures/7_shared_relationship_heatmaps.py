"""Figure 7: shared attribute-parameter relationships across models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from project.parameterize.figures.common import (
    add_panel_labels,
    apply_wrr_style,
    overlay_shared_cell_borders,
    pretty_model_name,
    pretty_parameter_name,
    save_figure,
    symmetric_vlim,
)


def _mean_matrix(corr_long: pd.DataFrame, model: str, key_attributes: list[str]) -> pd.DataFrame:
    subset = corr_long.loc[(corr_long["model"] == model) & (corr_long["attribute"].isin(key_attributes))]
    matrix = subset.pivot_table(
        index="attribute",
        columns="parameter",
        values="spearman_rho",
        aggfunc="mean",
    )
    return matrix.reindex(index=key_attributes)


def generate_figure(data_dict: dict, output_dir, formats=("png", "pdf")):
    apply_wrr_style()
    key_attributes = data_dict["key_attributes"]
    matrices = {
        model: _mean_matrix(data_dict["corr_long"], model, key_attributes)
        for model in data_dict["model_order"]
    }
    common_columns = [column for column in data_dict["param_names"] if column in set.intersection(*(set(matrix.columns) for matrix in matrices.values()))]
    for model in matrices:
        matrices[model] = matrices[model][common_columns]

    shared_mask = np.ones_like(matrices[data_dict["model_order"][0]].to_numpy(), dtype=bool)
    for matrix in matrices.values():
        shared_mask &= np.abs(matrix.to_numpy()) > 0.3

    vmax = symmetric_vlim(pd.concat([matrix.stack() for matrix in matrices.values()], ignore_index=True))
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), sharey=True)
    image = None
    for ax, model in zip(axes, data_dict["model_order"]):
        matrix = matrices[model]
        image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(pretty_model_name(model))
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels([pretty_parameter_name(name) for name in matrix.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(matrix.index)
        overlay_shared_cell_borders(ax, shared_mask)
    fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, label="Mean Spearman ρ")
    add_panel_labels(axes)
    return save_figure(fig, "fig07_shared_relationship_heatmaps", output_dir, formats=formats)

