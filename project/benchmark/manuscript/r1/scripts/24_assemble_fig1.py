#!/usr/bin/env python3
"""Assemble the five-panel publication Figure 1 without touching legacy Fig1 files."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

from R1X_fig_utils import draw_panel_a, draw_panel_b, draw_panel_d, draw_panel_e, save_figure, setup
from R1X_gis import draw_tendency_map, load_conus_boundaries
from r1_config import FIGURES_DIR, TABLES_DIR


def main() -> None:
    setup()
    data_a = pd.read_csv(TABLES_DIR / "Fig1a_model_aggregate_performance.csv").sort_values("plot_order")
    data_b = pd.read_csv(TABLES_DIR / "Fig1b_model_delta_distribution.csv").sort_values("plot_order")
    data_c = pd.read_csv(TABLES_DIR / "Fig1c_basin_crossmodel_tendency.csv")
    data_d = pd.read_csv(TABLES_DIR / "Fig1d_two_way_decomposition.csv")
    data_e = pd.read_csv(TABLES_DIR / "Fig1e_temporal_model_persistence.csv").sort_values("plot_order")
    states, national = load_conus_boundaries()

    fig = plt.figure(figsize=(14.2, 8.0))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1.12, 1.12, 1.35, 1.35], height_ratios=[1.55, 1.0], hspace=0.30, wspace=0.36)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[0, 2:])
    ax_d = fig.add_subplot(gs[1, 2])
    ax_e = fig.add_subplot(gs[1, 3])

    draw_panel_a(ax_a, data_a, label="(a)", show_labels=True)
    draw_panel_b(ax_b, data_b, label="(b)", show_labels=False)
    mappable = draw_tendency_map(ax_c, data_c, states, national, title="Basin-level cross-model estimator tendency", panel_label="(c)")
    cbar = fig.colorbar(mappable, ax=ax_c, fraction=0.026, pad=0.012, shrink=0.82)
    cbar.set_ticks([0.0, 0.5, 1.0]); cbar.set_ticklabels(["0 (IC)", "0.5 (mixed)", "1 (dPL)"])
    cbar.set_label("Fraction of models with dPL KGE > IC KGE", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5, length=2)
    draw_panel_d(ax_d, data_d, label="(d)")
    draw_panel_e(ax_e, data_e, label="(e)")
    fig.suptitle("IC–dPL aggregate performance and model–basin heterogeneity", fontsize=11, y=0.995)
    fig.savefig(FIGURES_DIR / "Fig1_R1_main.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(FIGURES_DIR / "Fig1_R1_main.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"PASS: wrote {FIGURES_DIR / 'Fig1_R1_main.png'} and vector PDF")


if __name__ == "__main__":
    main()
