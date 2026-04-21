"""Figure 1: overall predictive performance."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from project.parameterize.figures.common import (
    COLORS,
    MODEL_ORDER,
    add_panel_labels,
    add_significance_bracket,
    apply_wrr_style,
    paired_wilcoxon,
    pretty_model_name,
    reference_loss_only,
    save_figure,
)


def generate_figure(data_dict: dict, output_dir, formats=("png", "pdf")):
    apply_wrr_style()
    metrics = reference_loss_only(data_dict["metrics_long"], data_dict)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    metric_specs = [
        ("nse", "NSE"),
        ("kge", "KGE"),
        ("bias_abs", "|Bias|"),
    ]

    for ax, (metric, ylabel) in zip(axes, metric_specs):
        sns.violinplot(
            data=metrics,
            x="model",
            y=metric,
            order=data_dict["model_order"],
            palette=COLORS,
            inner="quartile",
            cut=0,
            linewidth=1.0,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_xticklabels([pretty_model_name(name) for name in data_dict["model_order"]], rotation=15)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        p_value = paired_wilcoxon(metrics, metric, "deterministic", "distributional")
        y_max = float(metrics[metric].max())
        add_significance_bracket(ax, 0, 2, y_max * 1.02 if y_max > 0 else 0.05, f"dist vs det\np={p_value:.3g}")

    add_panel_labels(axes)
    return save_figure(fig, "fig01_overall_predictive_performance", output_dir, formats=formats)

