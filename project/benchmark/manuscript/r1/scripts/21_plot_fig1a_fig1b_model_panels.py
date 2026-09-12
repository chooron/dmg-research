#!/usr/bin/env python3
"""Render standalone formal Figure 1 panels (a) and (b)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from R1X_fig_utils import draw_panel_a, draw_panel_b, save_figure
from r1_config import FIGURES_DIR, TABLES_DIR


def main() -> None:
    data_a = pd.read_csv(TABLES_DIR / "Fig1a_model_aggregate_performance.csv").sort_values("plot_order")
    data_b = pd.read_csv(TABLES_DIR / "Fig1b_model_delta_distribution.csv").sort_values("plot_order")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.4, 6.8))
    draw_panel_a(ax, data_a, label="")
    ax.text(-0.18, 1.10, "(a)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
    ax.text(1.02, 0.98, f"IC ensemble median={data_a.IC_median_KGE.median():.3f}\ndPL ensemble median={data_a.dPL_median_KGE.median():.3f}\nN=36 models", transform=ax.transAxes, fontsize=6.5, va="top", ha="left", clip_on=False)
    fig.tight_layout()
    save_figure(fig, "Fig1a_aggregate_performance")

    fig, ax = plt.subplots(figsize=(4.4, 6.8))
    draw_panel_b(ax, data_b, show_labels=True)
    fig.tight_layout()
    save_figure(fig, "Fig1b_model_delta_distribution")
    print(f"PASS: wrote standalone panels (a) and (b) under {FIGURES_DIR}")


if __name__ == "__main__":
    main()
