#!/usr/bin/env python3
"""Render standalone formal Figure 1 panels (d) and (e)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from R1X_fig_utils import draw_panel_d, draw_panel_e, save_figure, setup
from r1_config import FIGURES_DIR, TABLES_DIR


def main() -> None:
    setup()
    data_d = pd.read_csv(TABLES_DIR / "Fig1d_two_way_decomposition.csv")
    data_e = pd.read_csv(TABLES_DIR / "Fig1e_temporal_model_persistence.csv").sort_values("plot_order")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.7, 2.5))
    draw_panel_d(ax, data_d)
    fig.tight_layout()
    save_figure(fig, "Fig1d_two_way_decomposition")

    fig, ax = plt.subplots(figsize=(4.3, 4.0))
    draw_panel_e(ax, data_e)
    fig.tight_layout()
    save_figure(fig, "Fig1e_temporal_persistence")
    print(f"PASS: wrote standalone panels (d) and (e) under {FIGURES_DIR}")


if __name__ == "__main__":
    main()
