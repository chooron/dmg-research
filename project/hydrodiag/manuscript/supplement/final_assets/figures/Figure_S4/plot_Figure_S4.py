#!/usr/bin/env python3
"""Figure S4 (R3 Supplement): Seasonal process delivery and storage deviation diagnostics.

Contains the seasonal process and storage dynamics formerly in main Figure 6:
  Panel (a): Seasonal liquid-water delivery (effective core input, high-snow ensemble N=133)
  Panel (b): Seasonal storage deviation from truth (Delta Wt green-white-orange heatmap + row-wise IQR heterogeneity)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np

# Path setup
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
MANUSCRIPT = PROJECT / "manuscript"
RESULTS_R3 = MANUSCRIPT / "results" / "R3"
FIG_DIR = HERE
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Add shared styles
sys.path.insert(0, str(MANUSCRIPT / "scripts" / "shared"))
from r1_plot_style import (
    COLOR_BASE,
    COLOR_TGD,
    COLOR_CN,
    COLOR_DARK_NEUTRAL,
    COLOR_LIGHT_REF,
    COLOR_ZERO_LINE,
    apply_clean_spines,
    setup_publication_style,
)

OUT_NAME = "Figure_S4"

COLOR_TRUTH = "#303438"
MONTH_LABELS = ["O", "N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S"]

CMAP_STORAGE_DEV = LinearSegmentedColormap.from_list(
    "green_white_orange",
    ["#007248", "#009E73", "#d0ede3", "#FFFFFF", "#fbe0ce", "#D55E00", "#9e3d00"],
    N=256,
)


def load_data():
    summary = json.loads((RESULTS_R3 / "figure6_summary.json").read_text())
    return summary


def build_figure(summary, out_dir: Path | None = None) -> Path:
    setup_publication_style()

    target_dir = out_dir or FIG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9.0, 4.5))

    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[0.90, 1.10],
        wspace=0.25,
        left=0.08,
        right=0.98,
        top=0.88,
        bottom=0.12,
    )

    # ── Panel (a): Seasonal Liquid-Water Delivery ──
    ax_a = fig.add_subplot(gs[0])
    apply_clean_spines(ax_a)
    ax_a.set_title("(a) Seasonal liquid-water delivery", weight="bold", loc="left", pad=6, fontsize=8.8)

    pe = summary["panel_e_seasonal_input"]
    xs = np.arange(12)

    # Truth line
    t_med = np.array(pe["Truth"]["median"])
    t_ci_lo = np.array(pe["Truth"]["ci_lo"])
    t_ci_hi = np.array(pe["Truth"]["ci_hi"])
    ax_a.plot(xs, t_med, color=COLOR_TRUTH, linestyle="-", linewidth=1.8, label="Truth", zorder=6)
    ax_a.fill_between(xs, t_ci_lo, t_ci_hi, color=COLOR_TRUTH, alpha=0.14, zorder=1)

    series = [
        ("Base_IC", COLOR_BASE, "-", 1.3, 3),
        ("Base_dPL", COLOR_BASE, "--", 1.3, 3),
        ("TGD_IC", COLOR_TGD, "-", 1.3, 4),
        ("TGD_dPL", COLOR_TGD, "--", 1.3, 4),
        ("CN_IC", COLOR_CN, "-", 1.3, 5),
        ("CN_dPL", COLOR_CN, "--", 1.3, 5),
    ]

    for key, col, ls, lw, zo in series:
        med = np.array(pe[key]["median"])
        ci_lo = np.array(pe[key]["ci_lo"])
        ci_hi = np.array(pe[key]["ci_hi"])
        ax_a.plot(xs, med, color=col, linestyle=ls, linewidth=lw, zorder=zo)
        ax_a.fill_between(xs, ci_lo, ci_hi, color=col, alpha=0.10, zorder=zo - 2)

    ax_a.set_xlim(-0.3, 11.3)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(MONTH_LABELS, fontsize=7.2)
    ax_a.set_ylim(0.0, 8.4)
    ax_a.set_xlabel("Water-year month (Oct–Sep)", labelpad=2, fontsize=8.0)
    ax_a.set_ylabel("Effective input (mm d$^{-1}$)", labelpad=2, fontsize=8.0)
    ax_a.grid(True, axis="both", linestyle=":", alpha=0.35, color=COLOR_LIGHT_REF)
    ax_a.text(0.04, 0.92, "High-snow ensemble, $N = 133$\nLines = median, bands = 95% CI",
              transform=ax_a.transAxes, va="top", ha="left", fontsize=6.5, color="#555555", style="italic")

    # ── Panel (b): Storage Deviation Heatmap + Row IQR ──
    hf = summary["panel_f_seasonal_storage_heatmap"]
    med_mat = np.array(hf["median_matrix"])
    row_lbls = hf["row_labels"]
    row_iqr_meds = hf["row_iqr_medians"]

    gs_f = gridspec.GridSpecFromSubplotSpec(
        2, 2,
        subplot_spec=gs[1],
        width_ratios=[1.0, 0.30],
        height_ratios=[1.0, 0.055],
        wspace=0.16,
        hspace=0.38,
    )

    ax1 = fig.add_subplot(gs_f[0, 0])
    apply_clean_spines(ax1)
    norm1 = TwoSlopeNorm(vmin=-35, vcenter=0, vmax=90)
    im1 = ax1.imshow(med_mat, cmap=CMAP_STORAGE_DEV, norm=norm1, aspect="auto", interpolation="nearest")
    ax1.set_xticks(np.arange(12))
    ax1.set_xticklabels(MONTH_LABELS, fontsize=7.0)
    ax1.set_yticks(np.arange(6))
    ax1.set_yticklabels(row_lbls, fontsize=6.8)
    ax1.set_title("(b) Storage deviation from truth", weight="bold", loc="left", pad=4, fontsize=8.2)
    ax1.set_xlabel("Water-year month (Oct–Sep)", labelpad=1.5, fontsize=7.5)

    cax = fig.add_subplot(gs_f[1, 0])
    ax_blank = fig.add_subplot(gs_f[1, 1])
    ax_blank.axis("off")

    cb1 = fig.colorbar(im1, cax=cax, orientation="horizontal")
    cb1.ax.tick_params(labelsize=6.2)
    cb1.set_label("Median $\\Delta W_t = W_{t,M} - W_{t,\\mathrm{truth}}$ (mm)", fontsize=6.6, labelpad=2)

    ax2 = fig.add_subplot(gs_f[0, 1], sharey=ax1)
    apply_clean_spines(ax2)
    y_pos = np.arange(6)
    row_colors = [COLOR_BASE, COLOR_BASE, COLOR_TGD, COLOR_TGD, COLOR_CN, COLOR_CN]
    ax2.barh(y_pos, row_iqr_meds, height=0.60, color=row_colors, alpha=0.75, edgecolor="#444444", linewidth=0.7)
    ax2.set_xlim(0, 110)
    ax2.set_xticks([0, 50, 100])
    ax2.set_xticklabels(["0", "50", "100"], fontsize=6.5)
    ax2.set_xlabel("IQR (mm)", labelpad=2, fontsize=7.0)
    ax2.set_title("Heterogeneity", fontsize=7.4, pad=4, weight="bold")
    ax2.grid(True, axis="x", linestyle=":", alpha=0.35, color=COLOR_LIGHT_REF)

    row_lbls_abbr = ["Base IC", "Base dPL", "TGD IC", "TGD dPL", "CN IC", "CN dPL"]
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(row_lbls_abbr, fontsize=6.0)
    ax2.yaxis.tick_right()
    ax2.tick_params(axis="y", left=False, labelleft=False, right=False, labelright=True, length=0, pad=2)
    ax2.set_ylim(ax1.get_ylim())

    for i, v in enumerate(row_iqr_meds):
        ax2.text(v + 3, i, f"{v:.0f}", va="center", ha="left", fontsize=6.3, family="monospace", color=COLOR_DARK_NEUTRAL)

    # Global legend
    struct_handles = [
        Line2D([0], [0], color=COLOR_BASE, lw=1.5, marker="o", markersize=4.2, markerfacecolor=COLOR_BASE, markeredgecolor="white", label="Base"),
        Line2D([0], [0], color=COLOR_TGD, lw=1.5, marker="^", markersize=4.2, markerfacecolor=COLOR_TGD, markeredgecolor="white", label="TGD"),
        Line2D([0], [0], color=COLOR_CN, lw=1.5, marker="s", markersize=4.2, markerfacecolor=COLOR_CN, markeredgecolor="white", label="CN"),
        Line2D([0], [0], color=COLOR_TRUTH, lw=1.6, ls="-", label="CN truth"),
        Line2D([0], [0], color=COLOR_DARK_NEUTRAL, lw=1.3, ls="-", label="IC (solid)"),
        Line2D([0], [0], color=COLOR_DARK_NEUTRAL, lw=1.3, ls="--", label="dPL (dashed)"),
    ]
    fig.legend(
        handles=struct_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=6,
        frameon=False,
        fontsize=7.2,
        columnspacing=1.3,
        handletextpad=0.4,
    )

    out_path = target_dir / f"{OUT_NAME}.png"
    plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved SI Figure S4 -> {out_path} ({file_size_mb:.2f} MB)", flush=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Render R3 Supplement Figure S4.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory.")
    args = parser.parse_args()
    summary = load_data()
    build_figure(summary, args.out_dir)


if __name__ == "__main__":
    main()
