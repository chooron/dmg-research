#!/usr/bin/env python3
"""Step 9: render Fig. 1a."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from r1_config import FIGURES_DIR, TABLES_DIR
from r1_plot_utils import configure_matplotlib, draw_fig1a, save_figure


def main() -> None:
    configure_matplotlib()
    pair = pd.read_csv(TABLES_DIR / "R1_model_basin_delta_kge.csv")
    summary = pd.read_csv(TABLES_DIR / "R1_model_performance_summary.csv")
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    draw_fig1a(ax, pair, summary, label="a")
    fig.tight_layout()
    save_figure(fig, FIGURES_DIR / "Fig1a_aggregate_IC_vs_dPL")
    print(f"PASS: wrote {FIGURES_DIR / 'Fig1a_aggregate_IC_vs_dPL.png'} and .pdf")


if __name__ == "__main__":
    main()
