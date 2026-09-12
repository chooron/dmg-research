#!/usr/bin/env python3
"""Step 10: render Fig. 1c."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from r1_config import FIGURES_DIR, TABLES_DIR
from r1_plot_utils import configure_matplotlib, draw_fig1c, save_figure


def main() -> None:
    configure_matplotlib()
    temporal = pd.read_csv(TABLES_DIR / "R1_temporal_AB_model_summary.csv")
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    draw_fig1c(ax, temporal, label="c")
    fig.tight_layout()
    save_figure(fig, FIGURES_DIR / "Fig1c_temporal_AB_reproducibility")
    print(f"PASS: wrote {FIGURES_DIR / 'Fig1c_temporal_AB_reproducibility.png'} and .pdf")


if __name__ == "__main__":
    main()
