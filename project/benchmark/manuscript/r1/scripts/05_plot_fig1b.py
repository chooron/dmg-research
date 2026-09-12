#!/usr/bin/env python3
"""Step 9: render Fig. 1b."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from r1_config import FIGURES_DIR, TABLES_DIR
from r1_plot_utils import configure_matplotlib, draw_fig1b, save_figure


def main() -> None:
    configure_matplotlib()
    summary = pd.read_csv(TABLES_DIR / "R1_model_performance_summary.csv")
    fig, ax = plt.subplots(figsize=(6.1, 8.6))
    draw_fig1b(ax, summary, label="b")
    fig.tight_layout()
    save_figure(fig, FIGURES_DIR / "Fig1b_deltaKGE_heterogeneity")
    print(f"PASS: wrote {FIGURES_DIR / 'Fig1b_deltaKGE_heterogeneity.png'} and .pdf")


if __name__ == "__main__":
    main()
