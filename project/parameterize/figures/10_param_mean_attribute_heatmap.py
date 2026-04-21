"""Figure 10: distributional parameter-mean versus attribute correlations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from project.parameterize.figures.common import (
    bonferroni_threshold,
    cluster_order,
    distributional_correlation_tables,
    pretty_parameter_name,
    save_figure,
    symmetric_vlim,
    apply_wrr_style,
)


def generate_figure(data_dict: dict, output_dir, formats=("png", "pdf")):
    apply_wrr_style()
    rho_table, p_table = distributional_correlation_tables(data_dict, "mean")
    ordered_rows = cluster_order(rho_table.fillna(0.0))
    rho_table = rho_table.loc[ordered_rows]
    p_table = p_table.loc[ordered_rows]
    threshold = bonferroni_threshold(rho_table.size)
    vmax = symmetric_vlim(rho_table.stack())

    fig, ax = plt.subplots(figsize=(12.5, max(8.0, 0.28 * len(ordered_rows))))
    sns.heatmap(
        rho_table,
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        cbar_kws={"label": "Spearman ρ"},
    )
    ax.set_xticklabels([pretty_parameter_name(name) for name in rho_table.columns], rotation=45, ha="right")
    ax.set_xlabel("Parameter mean")
    ax.set_ylabel("Basin attribute")

    for row_idx, attribute in enumerate(rho_table.index):
        for col_idx, parameter in enumerate(rho_table.columns):
            if float(p_table.loc[attribute, parameter]) < threshold:
                ax.text(col_idx + 0.5, row_idx + 0.5, "*", ha="center", va="center", color="black", fontsize=7)
    return save_figure(fig, "fig10_param_mean_attribute_heatmap", output_dir, formats=formats)
