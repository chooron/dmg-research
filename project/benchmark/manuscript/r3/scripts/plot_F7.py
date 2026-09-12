#!/usr/bin/env python3
"""Plot Figure 7 (F7) for Journal of Hydrology manuscript R3.

Scientific Question:
    How far can same-coordinate specificity be interpreted?
    1. Same-coordinate specificity remains beyond raw parameter-rank similarity.
    2. But this specificity does not extend to broad hydrological functional roles.

Two Core Panels:
    (a) Beyond rank similarity (HERO scatter: Same coordinate vs Rank-matched alternative, n=270)
    (b) Functional-role boundary (Paired-dot/line plot: Same role vs Different role, n=29 models)

Style:
    - Minimal-text, standard scientific result graphics (no tables, no boxes, no third panel)
    - Clean color palette: Deep Teal/Green for observed points, Purple/Slate for role comparison
    - Resolution: 600 DPI, PNG only
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
R3_DIR = SCRIPT_DIR.parent
TABLES_DIR = R3_DIR / "tables"
FIGURES_DIR = R3_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = FIGURES_DIR / "F7.png"
OUTPUT_MAIN_PNG = R3_DIR / "F7_main.png"

# Source tables for F7
MATCHING_CSV = R3_DIR.parents[3] / "project/benchmark/results/joh_direct_parameter_change_diagnostic_20260905/r2_parameter_axis_audit_20260906/agent_D_r2_r3_rank_linkage/tables/matching_coordinate_results.csv"
ROLE_CSV = R3_DIR.parents[3] / "project/benchmark/results/joh_functional_role_diagnostic_20260905/tables/13_ROLE_OFFDIAGONAL_ADVANTAGE_BY_MODEL.csv"

# Colors
COLOR_POINT = "#1e3a5f"      # Deep Navy / Slate Blue for Panel a scatter points
COLOR_ROLE = "#7b1fa2"       # Purple for Panel b median role markers
COLOR_DARK = "#0f172a"       # Dark Slate / Charcoal for text and titles
COLOR_MUTED = "#64748b"      # Muted Slate Grey
GRID_COLOR = "#f1f5f9"
BORDER_COLOR = "#cbd5e1"
LINE_REF = "#94a3b8"


def setup_matplotlib():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.0,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11.2,
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.2,
        "legend.fontsize": 9.0,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.60,
        "ytick.major.width": 0.60,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def load_panel_a_data():
    df_match = pd.read_csv(MATCHING_CSV)
    sub = df_match[(df_match["spec"] == "primary_k3_no_caliper") & (df_match["matched"] == True)].copy()
    assert len(sub) == 270, f"Expected 270 matched parameter coordinates, got {len(sub)}"
    return sub


def load_panel_b_data():
    df_role = pd.read_csv(ROLE_CSV)
    sub = df_role[(df_role["confidence_filter"] == "high") & (df_role["valid_role_parameter_count"] > 0)].copy()
    assert len(sub) == 29, f"Expected 29 models with valid role contrasts, got {len(sub)}"
    return sub


def render_figure(df_a: pd.DataFrame, df_b: pd.DataFrame):
    setup_matplotlib()

    # Panel a statistics (Rank-matched residual advantage)
    # Model-equal median is 0.295238, 95% CI [0.254386, 0.429073], p = 0.000400
    delta_val = 0.295
    ci_low, ci_high = 0.254, 0.429
    p_val_str = "p < 0.001"

    # Panel b statistics (Functional-role boundary)
    # same_role median = -0.0278195... -> -0.028
    # cross_role median = 0.005263... -> 0.005
    # A_role = -0.119, p = 0.806
    same_med = float(df_b["same_role_offdiag_median"].median())
    cross_med = float(df_b["cross_role_median"].median())
    a_role_med = float(df_b["role_offdiag_advantage_median"].median())

    # Figure Layout: 2 Panels (Panel a HERO ~64% width, Panel b ~36% width)
    fig = plt.figure(figsize=(11.5, 5.2), dpi=600)

    gs = GridSpec(
        nrows=1, ncols=2,
        width_ratios=[1.72, 1.0],
        wspace=0.22,
        left=0.075, right=0.965, top=0.91, bottom=0.11
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])


    # =========================================================
    # PANEL (a): Beyond Rank Similarity (HERO Scatter, n=270)
    # =========================================================
    ax_a.set_title(r"$\mathbf{(a)}$ Beyond rank similarity",
                   loc="left", fontsize=11.2, pad=8, fontweight="bold", color=COLOR_DARK)

    x_a = df_a["matched_offdiag_Q_info_mean"].to_numpy()
    y_a = df_a["Q_info_diag"].to_numpy()

    lim_min, lim_max = -0.85, 1.05

    # 1:1 Identity Reference Line (y = x)
    ax_a.plot([lim_min, lim_max], [lim_min, lim_max], color=LINE_REF, lw=0.9, linestyle="--", zorder=1)

    # Shaded region above 1:1 line to emphasize positive residual advantage
    ax_a.fill_between([lim_min, lim_max], [lim_min, lim_max], [lim_max, lim_max], facecolor="#f8fafc", zorder=0)

    # Observed parameter points (n = 270)
    ax_a.scatter(x_a, y_a, s=26, color=COLOR_POINT, alpha=0.60, edgecolor="none", zorder=3)

    ax_a.set_xlim(lim_min, lim_max)
    ax_a.set_ylim(lim_min, lim_max)
    ax_a.set_xlabel("Rank-matched alternative", fontsize=10.5, labelpad=4, color=COLOR_DARK)
    ax_a.set_ylabel("Same coordinate", fontsize=10.5, labelpad=4, color=COLOR_DARK)
    ax_a.grid(color=GRID_COLOR, lw=0.5, linestyle="--", zorder=0)

    # Core statistical annotation (clean unboxed text)
    stat_text_a = rf"$\Delta = {delta_val:.3f}\ [{ci_low:.3f},\ {ci_high:.3f}],\ {p_val_str}$"
    ax_a.text(0.04, 0.94, stat_text_a, transform=ax_a.transAxes,
              fontsize=9.6, color=COLOR_DARK, fontweight="bold", va="top", ha="left")


    # =========================================================
    # PANEL (b): Functional-Role Boundary (Paired-Dot/Line, n=29 models)
    # =========================================================
    ax_b.set_title(r"$\mathbf{(b)}$ Functional-role boundary",
                   loc="left", fontsize=11.2, pad=8, fontweight="bold", color=COLOR_DARK)

    y_same = df_b["same_role_offdiag_median"].to_numpy()
    y_diff = df_b["cross_role_median"].to_numpy()
    n_models_b = len(df_b)

    x_same = np.zeros(n_models_b)
    x_diff = np.ones(n_models_b)

    # Horizontal Zero Reference Line (y = 0)
    ax_b.axhline(0, color=LINE_REF, lw=0.75, linestyle="--", zorder=1)

    # Draw paired connecting lines for each model (thin light grey)
    for i in range(n_models_b):
        ax_b.plot([0, 1], [y_same[i], y_diff[i]], color="#cbd5e1", lw=0.75, alpha=0.85, zorder=2)

    # Draw individual model points
    ax_b.scatter(x_same, y_same, s=20, color="#64748b", alpha=0.55, edgecolor="none", zorder=3)
    ax_b.scatter(x_diff, y_diff, s=20, color="#64748b", alpha=0.55, edgecolor="none", zorder=3)

    # Superimpose Model-Equal Median Markers (Purple Diamonds)
    ax_b.scatter([0], [same_med], s=60, color=COLOR_ROLE, marker="D", edgecolor="white", lw=0.6, zorder=5)
    ax_b.scatter([1], [cross_med], s=60, color=COLOR_ROLE, marker="D", edgecolor="white", lw=0.6, zorder=5)

    # Direct short numeric labels beside median markers
    ax_b.text(-0.08, same_med, f"{same_med:.3f}", color=COLOR_ROLE, fontsize=9.2, fontweight="bold", va="center", ha="right")
    ax_b.text(1.08, cross_med, f"{cross_med:.3f}", color=COLOR_ROLE, fontsize=9.2, fontweight="bold", va="center", ha="left")

    ax_b.set_xlim(-0.35, 1.35)
    ax_b.set_ylim(-1.05, 1.05)
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["Same role", "Different role"], fontsize=10.0, fontweight="bold", color=COLOR_DARK)
    ax_b.set_ylabel("Correspondence", fontsize=10.5, labelpad=4, color=COLOR_DARK)
    ax_b.grid(axis="y", color=GRID_COLOR, lw=0.5, linestyle="--", zorder=0)

    # Core statistical annotation (clean unboxed text)
    stat_text_b = r"$A_{\mathrm{role}} = -0.119,\ p = 0.806$"
    ax_b.text(0.50, 0.94, stat_text_b, transform=ax_b.transAxes,
              fontsize=9.6, color=COLOR_DARK, fontweight="bold", va="top", ha="center")

    # Save PNG only (600 DPI, no PDF)
    plt.savefig(OUTPUT_PNG, dpi=600)
    plt.savefig(OUTPUT_MAIN_PNG, dpi=600)
    plt.close()

    print(f"[SUCCESS] Saved F7.png to: {OUTPUT_PNG} (600 DPI)")
    print(f"[SUCCESS] Saved F7_main.png to: {OUTPUT_MAIN_PNG}")


if __name__ == "__main__":
    df_a = load_panel_a_data()
    df_b = load_panel_b_data()
    render_figure(df_a, df_b)
