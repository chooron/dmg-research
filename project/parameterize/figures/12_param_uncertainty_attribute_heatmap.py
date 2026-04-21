"""Figure 12: distributional parameter-uncertainty versus attribute correlations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from project.parameterize.figures.common import (
    apply_wrr_style,
    bonferroni_threshold,
    cluster_order,
    distributional_correlation_tables,
    pretty_parameter_name,
    save_figure,
    symmetric_vlim,
)


def generate_figure(data_dict: dict, output_dir, formats=("png", "pdf")):
    apply_wrr_style()
    rho_mean, p_mean = distributional_correlation_tables(data_dict, "mean")
    rho_std, p_std = distributional_correlation_tables(data_dict, "std")
    ordered_rows = cluster_order(rho_std.fillna(0.0))
    rho_mean = rho_mean.loc[ordered_rows]
    p_mean = p_mean.loc[ordered_rows]
    rho_std = rho_std.loc[ordered_rows]
    p_std = p_std.loc[ordered_rows]

    threshold = bonferroni_threshold(rho_std.size)
    vmax = symmetric_vlim(rho_std.stack())
    fig, ax = plt.subplots(figsize=(12.5, max(8.0, 0.28 * len(ordered_rows))))
    sns.heatmap(
        rho_std,
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        ax=ax,
        cbar_kws={"label": "Spearman ρ"},
    )
    ax.set_xticklabels([pretty_parameter_name(name) for name in rho_std.columns], rotation=45, ha="right")
    ax.set_xlabel("Parameter std")
    ax.set_ylabel("Basin attribute")

    for row_idx, attribute in enumerate(rho_std.index):
        for col_idx, parameter in enumerate(rho_std.columns):
            std_sig = float(p_std.loc[attribute, parameter]) < threshold
            mean_sig = float(p_mean.loc[attribute, parameter]) < threshold
            if std_sig:
                marker = "★" if mean_sig and abs(float(rho_mean.loc[attribute, parameter])) > 0.3 else "*"
                ax.text(col_idx + 0.5, row_idx + 0.5, marker, ha="center", va="center", color="black", fontsize=7)
    return save_figure(fig, "fig12_param_uncertainty_attribute_heatmap", output_dir, formats=formats)

