#!/usr/bin/env python3
"""Step 11: render the combined three-panel Fig. 1."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from r1_config import FIGURES_DIR, TABLES_DIR
from r1_plot_utils import configure_matplotlib, draw_fig1a, draw_fig1b, draw_fig1c, save_figure


def main() -> None:
    configure_matplotlib()
    pair = pd.read_csv(TABLES_DIR / "R1_model_basin_delta_kge.csv")
    summary = pd.read_csv(TABLES_DIR / "R1_model_performance_summary.csv")
    temporal = pd.read_csv(TABLES_DIR / "R1_temporal_AB_model_summary.csv")
    fig = plt.figure(figsize=(14.2, 7.2))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.18, 1.0], wspace=0.50)
    ax_a, ax_b, ax_c = [fig.add_subplot(grid[0, i]) for i in range(3)]
    draw_fig1a(ax_a, pair, summary, label="a")
    draw_fig1b(ax_b, summary, label="b")
    draw_fig1c(ax_c, temporal, label="c")
    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.09, top=0.94, wspace=0.55)
    save_figure(fig, FIGURES_DIR / "Fig1_R1_combined")
    print(f"PASS: wrote {FIGURES_DIR / 'Fig1_R1_combined.png'} and .pdf")


if __name__ == "__main__":
    main()
