"""Figure 2: basin-wise paired NSE differences."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from project.parameterize.figures.common import add_panel_labels, apply_wrr_style, reference_loss_only, save_figure


def _paired_difference(metrics: pd.DataFrame, left: str, right: str) -> pd.Series:
    pivot = metrics.pivot_table(
        index=["basin_id", "seed", "loss"],
        columns="model",
        values="nse",
        aggfunc="mean",
    ).dropna(subset=[left, right])
    return pivot[left] - pivot[right]


def generate_figure(data_dict: dict, output_dir, formats=("png", "pdf")):
    apply_wrr_style()
    metrics = reference_loss_only(data_dict["metrics_long"], data_dict)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    specs = [
        ("distributional", "deterministic", "Distributional - Deterministic NSE"),
        ("mc_dropout", "deterministic", "MC Dropout - Deterministic NSE"),
    ]

    for ax, (left, right, title) in zip(axes, specs):
        diff = _paired_difference(metrics, left, right)
        sns.kdeplot(x=diff, fill=True, linewidth=1.6, ax=ax)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        win_rate = float((diff > 0).mean() * 100.0)
        ax.set_title(title)
        ax.set_xlabel("Paired basin difference")
        ax.set_ylabel("Density")
        ax.text(0.03, 0.95, f"{win_rate:.1f}% basins > 0", transform=ax.transAxes, va="top")

    add_panel_labels(axes)
    return save_figure(fig, "fig02_basinwise_accuracy_differences", output_dir, formats=formats)

