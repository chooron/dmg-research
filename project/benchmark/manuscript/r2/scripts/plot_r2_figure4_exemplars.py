#!/usr/bin/env python3
"""Render the refined R2 Figure 4 parameter-exemplar composite (Flat Pale Green & Purple Edition).

Figure 4 provides concrete, visually interpretable coordinate-level exemplars
of how normalized parameters change across 531 catchments when moving from
IC to dPL across 12 distinct hydrological models (3x4 small-multiples layout).

Refined design features:
- Flat pale green (IC, top) and flat pale purple (dPL, right) marginal histograms
  with uniform dark charcoal/black-gray outlines (no gradient).
- Integrated 2-line in-scatter badge combining panel title and statistics:
  Line 1: (letter) model — parameter
  Line 2: M = ..., R = ...
  Placed in the optimal empty corner of each central scatter panel.
- Central paired scatter points: homogeneous deep charcoal/black-gray (alpha=0.60).
- Highly compressed inter-panel spacing (wspace=0.13, hspace=0.14) for a tight, cohesive composite.
- Concise mathematical axis labels: $\\bar{\\theta}_{\\mathrm{IC}}$ and $\\bar{\\theta}_{\\mathrm{dPL}}$.
- Standardized LaTeX mathematical typography for all parameter names.
- Systematically enlarged typography across titles, axes, ticks, and statistics.
- Single high-resolution PNG output at 600 DPI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.offsetbox import TextArea, VPacker, AnchoredOffsetbox
import numpy as np
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
CACHE = ROOT / "cache"
BENCHMARK = ROOT.parents[1]
RESULT_ROOT = BENCHMARK / "results/joh_direct_parameter_change_diagnostic_20260905"
OUTPUT = FIGURES / "Figure4_R2_parameter_exemplars_final.png"

# 12 Exemplar coordinates from 12 distinct models across 3 scientific rows
EXEMPLARS = [
    # Row 1: Large displacement with substantial rank preservation
    {"model": "alpine1", "param": "Smax", "tex_name": r"alpine1 — $\boldsymbol{S}_{\boldsymbol{\max}}$", "letter": "a", "row": 0, "col": 0},
    {"model": "gr4j", "param": "x1", "tex_name": r"gr4j — $\boldsymbol{x}_1$", "letter": "b", "row": 0, "col": 1},
    {"model": "newzealand1", "param": "s1max", "tex_name": r"newzealand1 — $\boldsymbol{s}_{1,\boldsymbol{\max}}$", "letter": "c", "row": 0, "col": 2},
    {"model": "mopex2", "param": "s2max", "tex_name": r"mopex2 — $\boldsymbol{s}_{2,\boldsymbol{\max}}$", "letter": "d", "row": 0, "col": 3},
    # Row 2: Large displacement with rank reorganization / weak rank preservation
    {"model": "vic", "param": "ishift", "tex_name": r"vic — $\boldsymbol{i}_{\mathbf{shift}}$", "letter": "e", "row": 1, "col": 0},
    {"model": "mopex3", "param": "s3max", "tex_name": r"mopex3 — $\boldsymbol{s}_{3,\boldsymbol{\max}}$", "letter": "f", "row": 1, "col": 1},
    {"model": "flexi", "param": "imax", "tex_name": r"flexi — $\boldsymbol{i}_{\boldsymbol{\max}}$", "letter": "g", "row": 1, "col": 2},
    {"model": "modhydrolog", "param": "k3", "tex_name": r"modhydrolog — $\boldsymbol{k}_3$", "letter": "h", "row": 1, "col": 3},
    # Row 3: Strong directional / recurrent displacement examples
    {"model": "simhyd", "param": "smsc", "tex_name": r"simhyd — $\boldsymbol{s}_{\mathbf{msc}}$", "letter": "i", "row": 2, "col": 0},
    {"model": "hymod", "param": "smax", "tex_name": r"hymod — $\boldsymbol{S}_{\boldsymbol{\max}}$", "letter": "j", "row": 2, "col": 1},
    {"model": "wetland", "param": "swmax", "tex_name": r"wetland — $\boldsymbol{s}_{w,\boldsymbol{\max}}$", "letter": "k", "row": 2, "col": 2},
    {"model": "ihacres", "param": "d", "tex_name": r"ihacres — $\boldsymbol{d}$", "letter": "l", "row": 2, "col": 3},
]

# Uniform flat palette (no gradient)
IC_HIST_FILL = "#c2dfd3"       # Soft pale sage/teal green fill (flat)
DPL_HIST_FILL = "#d8cae4"      # Soft pale lavender/mauve purple fill (flat)

# Core styling constants
HIST_BORDER_COLOR = "#383838"  # Dark charcoal outline for histogram bars
SCATTER_COLOR = "#242424"      # Homogeneous dark charcoal/black-gray scatter points
REF_LINE_COLOR = "#383838"     # 1:1 reference diagonal dashed line
GRID_COLOR = "#ebebeb"         # Dotted background grid
TEXT_COLOR = "#111111"         # Crisp black text


def configure_typography() -> None:
    """Configure matplotlib rcParams for journal publication standards."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 11.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def compute_optimal_stat_position(ic: np.ndarray, dpl: np.ndarray) -> tuple[str, tuple[float, float]]:
    """Determine the quadrant inside the scatter area with minimum point overlap."""
    br = np.sum((ic > 0.55) & (dpl < 0.40))
    tl = np.sum((ic < 0.45) & (dpl > 0.60))
    bl = np.sum((ic < 0.45) & (dpl < 0.40))
    tr = np.sum((ic > 0.55) & (dpl > 0.60))

    counts = {"br": br, "tl": tl, "bl": bl, "tr": tr}
    best = min(counts, key=counts.get)

    if best == "br":
        return "lower right", (0.97, 0.04)
    elif best == "tl":
        return "upper left", (0.03, 0.96)
    elif best == "bl":
        return "lower left", (0.03, 0.04)
    else:
        return "upper right", (0.97, 0.96)

def render_figure4() -> Path:
    """Render and save the refined Figure 4 publication composite."""
    configure_typography()

    # Load diagnostic table
    diag_path = TABLES / "F4_ALL_COORDINATE_DIAGNOSTICS.csv"
    if not diag_path.exists():
        raise FileNotFoundError(f"Missing F4 diagnostics table: {diag_path}")
    df_diag = pd.read_csv(diag_path)

    # Double-column format: 7.5 inches wide, 5.6 inches high
    fig = plt.figure(figsize=(7.5, 5.6), dpi=600)

    # Outer 3x4 grid with ultra-tight panel spacing
    outer = GridSpec(
        3, 4, figure=fig,
        wspace=0.05, hspace=0.05,
        left=0.055, right=0.990, top=0.990, bottom=0.055
    )
    # Uniform binning for marginal histograms across [0, 1]
    bins = np.linspace(0.0, 1.0, 21)

    for item in EXEMPLARS:
        m = item["model"]
        p = item["param"]
        tex_title = item["tex_name"]
        letter = item["letter"]
        r_idx = item["row"]
        c_idx = item["col"]

        # Extract diagnostic values
        match = df_diag[(df_diag.model_id == m) & (df_diag.parameter_name == p)]
        if len(match) != 1:
            raise ValueError(f"No unique diagnostic match for {m}:{p}")
        r_info = match.iloc[0]
        p_idx = int(r_info["coordinate_id"])
        m_disp = float(r_info["median_abs_displacement"])
        r_rank = float(r_info["R_rank"])

        # Load per-basin normalized vectors
        npz_path = RESULT_ROOT / f"r2/cache/{m}_normalized_parameter_matrices.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing npz cache: {npz_path}")
        npz = np.load(npz_path)
        ic = npz["IC"][:, p_idx]
        dpl = npz["dPL"][:, p_idx]

        # Inner grid for central scatter + top/right marginal histograms
        inner = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[r_idx, c_idx],
            width_ratios=[4.5, 1.0], height_ratios=[1.0, 4.5],
            wspace=0.02, hspace=0.02
        )

        ax_main = fig.add_subplot(inner[1, 0])
        ax_top = fig.add_subplot(inner[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(inner[1, 1], sharey=ax_main)

        # 1. 1:1 Reference diagonal (dashed line)
        ax_main.plot([0.0, 1.0], [0.0, 1.0], color=REF_LINE_COLOR, linestyle="--", linewidth=0.8, zorder=2)

        # 2. Central Paired Scatter Points (Homogeneous dark charcoal/black-gray)
        ax_main.scatter(
            ic, dpl,
            color=SCATTER_COLOR,
            s=7.5, alpha=0.60,
            edgecolors="none", zorder=3
        )

        # 3. Top Marginal Histogram (IC, Flat Pale Green strictly on [0, 1])
        ax_top.hist(
            ic, bins=bins,
            color=IC_HIST_FILL, edgecolor=HIST_BORDER_COLOR,
            linewidth=0.5, zorder=2
        )
        ax_top.axhline(0.0, color=HIST_BORDER_COLOR, linewidth=0.55, zorder=3)

        # 4. Right Marginal Histogram (dPL, Flat Pale Purple strictly on [0, 1], horizontal bars)
        ax_right.hist(
            dpl, bins=bins, orientation="horizontal",
            color=DPL_HIST_FILL, edgecolor=HIST_BORDER_COLOR,
            linewidth=0.5, zorder=2
        )
        ax_right.axvline(0.0, color=HIST_BORDER_COLOR, linewidth=0.55, zorder=3)

        # 5. Coordinate Limits, Ticks & Grid
        ax_main.set_xlim(-0.02, 1.02)
        ax_main.set_ylim(-0.02, 1.02)
        ax_main.set_xticks([0.0, 0.5, 1.0])
        ax_main.set_yticks([0.0, 0.5, 1.0])
        ax_main.set_xticklabels(["0", "0.5", "1.0"])
        ax_main.set_yticklabels(["0", "0.5", "1.0"])
        ax_main.grid(True, linestyle=":", color=GRID_COLOR, linewidth=0.55, zorder=1)

        # Clean marginal plot frames
        ax_top.set_xlim(-0.02, 1.02)
        ax_top.set_ylim(bottom=0.0)
        ax_top.axis("off")

        ax_right.set_ylim(-0.02, 1.02)
        ax_right.set_xlim(left=0.0)
        ax_right.axis("off")

        # 6. Integrated 2-Line In-Scatter Badge (Bold Line 1 + Enlarged Font)
        loc_str, bbox_anchor = compute_optimal_stat_position(ic, dpl)
        t1 = TextArea(f"({letter}) {tex_title}", textprops=dict(fontsize=8.5, fontweight="bold", color=TEXT_COLOR))
        t2 = TextArea(f"$M = {m_disp:.3f}, \\; R = {r_rank:.3f}$", textprops=dict(fontsize=7.8, color=TEXT_COLOR))
        vbox = VPacker(children=[t1, t2], align="left", pad=0, sep=1.8)

        box = AnchoredOffsetbox(
            loc=loc_str, child=vbox, pad=0.25, borderpad=0.0,
            frameon=True,
            bbox_to_anchor=bbox_anchor,
            bbox_transform=ax_main.transAxes
        )
        box.patch.set_boxstyle("square,pad=0.25")
        box.patch.set_facecolor("white")
        box.patch.set_edgecolor("#c8c8c8")
        box.patch.set_linewidth(0.45)
        box.patch.set_alpha(0.92)
        box.set_zorder(5)
        ax_main.add_artist(box)

        # 7. Concise Mathematical Axis Labels ($\bar{\theta}_{\mathrm{IC}}$ and $\bar{\theta}_{\mathrm{dPL}}$)
        if c_idx == 0:
            ax_main.set_ylabel(r"$\bar{\theta}_{\mathrm{dPL}}$", fontsize=11.0, color=TEXT_COLOR)
        else:
            ax_main.set_yticklabels([])

        if r_idx == 2:
            ax_main.set_xlabel(r"$\bar{\theta}_{\mathrm{IC}}$", fontsize=11.0, color=TEXT_COLOR)
        else:
            ax_main.set_xticklabels([])

    # Save final publication-ready composite
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600)
    plt.close(fig)
    print(f"Successfully rendered refined Figure 4 (Flat Pale Green & Purple Edition): {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    render_figure4()
