#!/usr/bin/env python3
"""Render standalone formal Figure 1 panel (c), the CONUS GIS map."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from R1X_fig_utils import save_figure, setup
from R1X_gis import draw_tendency_map, load_conus_boundaries
from r1_config import FIGURES_DIR, TABLES_DIR


def main() -> None:
    setup()
    data = pd.read_csv(TABLES_DIR / "Fig1c_basin_crossmodel_tendency.csv")
    states, national = load_conus_boundaries()
    fig, ax = plt.subplots(figsize=(7.1, 5.0))
    mappable = draw_tendency_map(ax, data, states, national, title="Basin-level cross-model estimator tendency", panel_label="(c)")
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.015, shrink=0.84)
    cbar.set_ticks([0.0, 0.5, 1.0]); cbar.set_ticklabels(["0 (IC)", "0.5 (mixed)", "1 (dPL)"])
    cbar.set_label("Fraction of models with dPL KGE > IC KGE", fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2)
    fig.tight_layout()
    save_figure(fig, "Fig1c_CAMELS_crossmodel_tendency")
    print(f"PASS: wrote GIS panel under {FIGURES_DIR}; points={len(data)}; matched states={len(states)}")


if __name__ == "__main__":
    main()
