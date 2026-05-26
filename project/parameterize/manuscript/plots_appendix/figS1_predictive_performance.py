"""Fig. S1 — Full predictive performance across all models × losses.
Style: identical to Fig01 (boxplot + CDF panels from fig01_predictive_performance.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common_appendix import (
    setup_style, clean_axes, add_panel_label,
    MM, DPI, MODEL_ORDER, MODEL_COLORS, TABLE_ROOT,
    math_model_labels, APP_FIG_DIR, save_fig,
)

OUT_STEM = "figS1_predictive_performance_full"

LOSS_ORDER  = ["NseBatchLoss", "LogNseBatchLoss", "HybridNseBatchLoss"]
LOSS_LABELS = {"NseBatchLoss": "NSE loss", "LogNseBatchLoss": "logNSE loss",
               "HybridNseBatchLoss": "Hybrid loss"}
METRICS = ["nse", "kge"]
METRIC_LABELS = {"nse": "NSE", "kge": "KGE"}
FIG_FONT_FAMILY = "Times New Roman"


def setup_fig_s1_style() -> None:
    setup_style()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FIG_FONT_FAMILY],
            "mathtext.fontset": "custom",
            "mathtext.rm": FIG_FONT_FAMILY,
            "mathtext.it": f"{FIG_FONT_FAMILY}:italic",
            "mathtext.bf": f"{FIG_FONT_FAMILY}:bold",
        }
    )


def _boxplot_panel(ax, data_by_model: dict, metric: str, panel: str) -> None:
    """Identical boxplot style to fig01."""
    labels = math_model_labels()
    positions = np.arange(1, len(MODEL_ORDER) + 1)
    data = [data_by_model.get(m, np.array([])) for m in MODEL_ORDER]
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]

    bp = ax.boxplot(
        data, positions=positions, widths=0.42, patch_artist=True,
        showfliers=True,
        medianprops={"color": "#2A2A2A", "lw": 1.0},
        whiskerprops={"color": "#666666", "lw": 0.8},
        capprops={"color": "#666666", "lw": 0.8},
        boxprops={"edgecolor": "#666666", "lw": 0.8},
        flierprops={"marker": "o", "markersize": 2.2,
                    "markerfacecolor": "#6F6F6F", "markeredgecolor": "#6F6F6F",
                    "markeredgewidth": 0.0, "alpha": 0.5},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    medians = [float(np.median(d)) if len(d) else np.nan for d in data]
    for x, med, color in zip(positions, medians, colors):
        ax.scatter([x], [med], s=18, facecolor="white", edgecolor=color,
                   linewidth=0.9, zorder=4)
        ax.text(x, -0.92, f"Median\n{med:.2f}", ha="center", va="bottom",
                fontsize=7.5, color=color)

    for y in (0.0, 0.5):
        ax.axhline(y, color="#D8D8D8", lw=0.75, ls=(0, (3, 3)), zorder=0)

    ax.set_xlim(0.45, len(MODEL_ORDER) + 0.55)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks(np.linspace(-1.0, 1.0, 5))
    ax.set_xticks(positions)
    ax.set_xticklabels([labels[m] for m in MODEL_ORDER])
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.tick_params(axis="x", length=0)
    add_panel_label(ax, f"({panel})", x=0.98, y=0.98,
                    ha="right", va="top", fontweight="normal", fontsize=10.5)
    clean_axes(ax)


def main() -> None:
    setup_fig_s1_style()
    metrics = pd.read_csv(TABLE_ROOT / "metrics_long.csv")

    # 3 losses × 2 metrics = 6 columns, 1 row of boxplots
    fig_w = 200 * MM
    fig_h = 90 * MM
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(len(METRICS), len(LOSS_ORDER),
                          left=0.07, right=0.99, top=0.88, bottom=0.22,
                          hspace=0.55, wspace=0.42)

    panel_idx = 0
    for row, metric in enumerate(METRICS):
        for col, loss in enumerate(LOSS_ORDER):
            ax = fig.add_subplot(gs[row, col])
            sub = metrics[metrics["loss"] == loss]
            basin_agg = (sub.groupby(["basin_id", "model"], as_index=False)[metric]
                         .median())
            data_by_model = {m: basin_agg.loc[basin_agg["model"] == m, metric]
                             .dropna().clip(-1, 1).values
                             for m in MODEL_ORDER}
            panel_lbl = chr(ord("a") + panel_idx)
            _boxplot_panel(ax, data_by_model, metric, panel_lbl)
            if row == 0:
                ax.set_title(LOSS_LABELS[loss], fontsize=8.5, pad=4)
            panel_idx += 1

    save_fig(fig, OUT_STEM)
    print(f"Saved {APP_FIG_DIR / OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
