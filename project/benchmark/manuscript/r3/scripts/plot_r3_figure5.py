#!/usr/bin/env python3
"""Plot Figure 5: Retention and asymmetry of catchment-parameter associations.

Journal of Hydrology Manuscript R3
Figure 5: Cross-paradigm association field overlap and bidirectional asymmetry.

Panels:
  (a) HERO: Nested association ledger (5,420 -> 902 -> 712 -> 692, with 20 sign-flipped)
  (b) Bidirectional conditioning asymmetry (78.9% vs 32.9%)
  (c) Retention threshold progression ladder (94.1% -> 87.7% -> 76.7% -> 57.3%, n=902)
  (d) Hydrological domain distribution of IC-stable associations (n=902)
"""
from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
OUTPUT_PNG = ROOT / "F5_main.png"

# Color Palette (Publication Grade, Colorblind Friendly, JoH Aesthetic)
INK = "#1a202c"
MUTED_TEXT = "#4a5568"
LIGHT_BORDER = "#cbd5e0"
GRID_COLOR = "#f0f2f5"

# Semantic Colors
IC_BLUE = "#1f5f9e"          # Independent Calibration
DPL_ORANGE = "#c45a1c"       # Parameter Learning
OVERLAP_GREEN = "#276738"    # Retained / Concordant (same sign)
SIGN_FLIP_RED = "#c53030"    # Sign flipped
NEUTRAL_GRAY = "#718096"     # Inactive / background
BG_TINT_BLUE = "#edf3f9"
BG_TINT_GREEN = "#edf7ef"
BG_TINT_ORANGE = "#fdf5ee"

def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.8,
        "figure.titlesize": 10.5,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

def verify_inputs() -> dict:
    """Verify frozen numbers against tables."""
    df_nested = pd.read_csv(TABLES / "R3_F5_NESTED_LEDGER.csv")
    df_bidi = pd.read_csv(TABLES / "R3_F5_BIDIRECTIONAL_CONDITIONING.csv")
    df_ret = pd.read_csv(TABLES / "R3_IC_TO_DPL_MAGNITUDE_RETENTION_SUMMARY.csv")
    df_diff = pd.read_csv(TABLES / "R3_CELL_SET_IDENTITY_DIFF.csv")
    
    # Assert exact ledger counts
    assert len(df_diff) == 902, f"Expected 902 IC stable cells, got {len(df_diff)}"
    assert int(df_nested.loc[df_nested.ledger_stage == "all_cells", "n_cells"].iloc[0]) == 5420
    assert int(df_nested.loc[df_nested.ledger_stage == "ic_absrho20", "n_cells"].iloc[0]) == 902
    assert int(df_nested.loc[df_nested.ledger_stage == "both_absrho20", "n_cells"].iloc[0]) == 712
    assert int(df_nested.loc[df_nested.ledger_stage == "both_same_sign", "n_cells"].iloc[0]) == 692
    assert int(df_nested.loc[df_nested.ledger_stage == "both_sign_flip", "n_cells"].iloc[0]) == 20
    
    # Assert conditioning probabilities
    p_dpl_given_ic = float(df_bidi.loc[df_bidi.conditioning == "dPL_strong_given_IC_strong", "probability"].iloc[0])
    p_ic_given_dpl = float(df_bidi.loc[df_bidi.conditioning == "IC_strong_given_dPL_strong", "probability"].iloc[0])
    assert np.isclose(p_dpl_given_ic, 712 / 902, atol=1e-6)
    assert np.isclose(p_ic_given_dpl, 712 / 2163, atol=1e-6)
    
    return {
        "df_nested": df_nested,
        "df_bidi": df_bidi,
        "df_ret": df_ret,
        "df_diff": df_diff
    }

def plot_figure5() -> None:
    data = verify_inputs()
    configure_matplotlib()
    
    fig = plt.figure(figsize=(11.0, 7.0), dpi=600)
    
    # Grid layout: left column for Panel A (Hero Nested Ledger), right column for Panels B, C, D
    gs = fig.add_gridspec(3, 2, width_ratios=[1.22, 1.0], height_ratios=[1.0, 1.0, 0.92],
                           left=0.065, right=0.975, bottom=0.07, top=0.94,
                           wspace=0.26, hspace=0.36)
    
    ax_a = fig.add_subplot(gs[:, 0])      # Panel a: Hero Nested Ledger
    ax_b = fig.add_subplot(gs[0, 1])      # Panel b: Bidirectional Conditioning
    ax_c = fig.add_subplot(gs[1, 1])      # Panel c: Retention Progression Ladder
    ax_d = fig.add_subplot(gs[2, 1])      # Panel d: Hydrological Domain Breakdown
    
    # =============================================================
    # PANEL (a): HERO — Nested Association Ledger
    # =============================================================
    ax_a.set_title("(a) Nested Association Ledger: Cross-Paradigm Evidence Shrinkage",
                   loc="left", pad=10, fontsize=9.8, fontweight="bold", color=INK)
    ax_a.axis("off")
    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(0, 100)
    
    # Outer box: All relationship cells (5420)
    rect_all = FancyBboxPatch((3, 5), 94, 89, boxstyle="round,pad=0.5,rounding_size=2.0",
                              facecolor="#f8fafc", edgecolor="#cbd5e0", linewidth=1.1, zorder=1)
    ax_a.add_patch(rect_all)
    
    ax_a.text(6, 90.8, "Stage 0: Total Relationship Candidate Space", fontsize=8.5, fontweight="bold", color=INK, zorder=5)
    ax_a.text(6, 87.6, r"N = 5,420 candidate cells (36 models $\times$ 20 information clusters)",
              fontsize=7.8, color=MUTED_TEXT, zorder=5)
    ax_a.text(94, 87.6, "IC Inactive / Weak: 4,518 (83.4%)", fontsize=7.4, fontstyle="italic", color=NEUTRAL_GRAY, ha="right", zorder=5)
    
    # Box 1: IC Strong / Stable (902 cells)
    rect_ic = FancyBboxPatch((6, 9), 88, 75, boxstyle="round,pad=0.5,rounding_size=1.8",
                             facecolor=BG_TINT_BLUE, edgecolor=IC_BLUE, linewidth=1.3, zorder=2)
    ax_a.add_patch(rect_ic)
    
    ax_a.text(9, 80.5, "Stage 1: IC Strong & Sign-Stable Baseline", fontsize=8.5, fontweight="bold", color=IC_BLUE, zorder=5)
    ax_a.text(9, 77.0, r"n = 902 cells (16.6% of total candidate pool) with $|\rho_{\mathrm{IC}}| \geq 0.20$ and $P(\mathrm{sign}) \geq 0.95$",
              fontsize=7.8, color=INK, zorder=5)
    ax_a.text(91, 77.0, r"Jaccard = 1.000 vs $|\rho_{\mathrm{IC}}| \geq 0.20$", fontsize=7.2, color=IC_BLUE, ha="right", zorder=5)
    
    # Annotation for IC-only strong
    ax_a.text(79, 69.5, "IC-only strong:\n190 cells (21.1% of IC)", fontsize=7.4, color=NEUTRAL_GRAY, ha="center", zorder=5)
    
    # Box 2: Both Strong (712 cells)
    rect_both = FancyBboxPatch((9, 13), 57, 59, boxstyle="round,pad=0.5,rounding_size=1.5",
                               facecolor="#eef6f0", edgecolor=OVERLAP_GREEN, linewidth=1.3, zorder=3)
    ax_a.add_patch(rect_both)
    
    ax_a.text(12, 68.2, "Stage 2: Strong in Both Paradigms", fontsize=8.3, fontweight="bold", color=OVERLAP_GREEN, zorder=5)
    ax_a.text(12, 64.6, r"n = 712 cells (78.9% of IC strong baseline)",
              fontsize=8.0, fontweight="bold", color=INK, zorder=5)
    ax_a.text(12, 61.2, r"$|\rho_{\mathrm{IC}}| \geq 0.20 \quad \mathrm{and} \quad |\rho_{\mathrm{dPL}}| \geq 0.20$",
              fontsize=7.5, color=MUTED_TEXT, zorder=5)
    
    # Box 3a: Both strong + Same Sign (692 cells) - HERO
    rect_same = FancyBboxPatch((11, 16), 53, 41.5, boxstyle="round,pad=0.5,rounding_size=1.2",
                              facecolor="#d8eedc", edgecolor="#1b5e20", linewidth=1.5, zorder=4)
    ax_a.add_patch(rect_same)
    
    ax_a.text(14, 53.5, "Stage 3: Sign-Concordant Overlap (Retained)", fontsize=8.5, fontweight="bold", color="#134e1b", zorder=5)
    ax_a.text(14, 48.8, r"n = 692 cells (97.2% of strong-both)",
              fontsize=8.8, fontweight="bold", color="#134e1b", zorder=5)
    ax_a.text(14, 44.8, "76.7% of all 902 IC-stable associations retained",
              fontsize=8.0, fontweight="bold", color="#134e1b", zorder=5)
    ax_a.text(14, 40.8, r"$\mathrm{sign}(\rho_{\mathrm{IC}}) = \mathrm{sign}(\rho_{\mathrm{dPL}})$ across 36 models",
              fontsize=7.5, color=MUTED_TEXT, zorder=5)
    
    ax_a.text(14, 30.5, "• Positive concordant associations: 334 cells (48.3%)\n• Negative concordant associations: 358 cells (51.7%)\n• Replicated across all conceptual model architectures",
              fontsize=7.2, color=INK, zorder=5, linespacing=1.35)
    
    # Box 3b: Sign-flipped among strong-both (20 cells)
    rect_flip = FancyBboxPatch((68, 16), 23, 38, boxstyle="round,pad=0.5,rounding_size=1.2",
                               facecolor="#fff5f5", edgecolor=SIGN_FLIP_RED, linewidth=1.2, zorder=4)
    ax_a.add_patch(rect_flip)
    
    ax_a.text(70, 50.5, "Sign Reversal", fontsize=8.0, fontweight="bold", color=SIGN_FLIP_RED, zorder=5)
    ax_a.text(70, 45.8, "n = 20 cells", fontsize=8.5, fontweight="bold", color=SIGN_FLIP_RED, zorder=5)
    ax_a.text(70, 41.5, "2.8% of strong-both", fontsize=7.5, fontweight="bold", color=INK, zorder=5)
    ax_a.text(70, 37.5, "(0.37% of all cells)", fontsize=7.2, color=MUTED_TEXT, zorder=5)
    ax_a.text(70, 29.5, "Strong in both,\nbut opposite signs:\n" + r"$\mathrm{sign}(\rho_{\mathrm{IC}}) \neq \mathrm{sign}(\rho_{\mathrm{dPL}})$",
              fontsize=7.0, color=MUTED_TEXT, zorder=5, linespacing=1.2)
    
    # Connecting dashed arrow from stage 2 to sign flip box
    ax_a.annotate("", xy=(68, 35), xytext=(62, 35),
                  arrowprops=dict(arrowstyle="->", color=SIGN_FLIP_RED, lw=1.2, ls="--"))
    
    # Bottom summary ledger banner
    ax_a.text(50, 7.2, "Nested Ledger: 5,420  " + r"$\to$" + "  902 (16.6%)  " + r"$\to$" + "  712 (78.9%)  " + r"$\to$" + "  692 (97.2%) Retained Overlap",
              ha="center", va="center", fontsize=8.0, fontweight="bold", color=INK,
              bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=LIGHT_BORDER, lw=0.8), zorder=6)
    
    # =============================================================
    # PANEL (b): Bidirectional Conditioning Asymmetry
    # =============================================================
    ax_b.set_title("(b) Bidirectional Association Conditioning Asymmetry",
                   loc="left", pad=6, fontsize=9.2, fontweight="bold", color=INK)
    
    categories = [
        "P(IC strong | dPL strong)\n" + r"$(712 \,/\, 2{,}163)$",
        "P(dPL strong | IC strong)\n" + r"$(712 \,/\, 902)$"
    ]
    vals = [32.917, 78.936]
    colors = [DPL_ORANGE, IC_BLUE]
    
    y_pos = np.arange(len(categories))
    bars = ax_b.barh(y_pos, vals, height=0.48, color=colors, edgecolor=INK, linewidth=0.7, zorder=3)
    
    ax_b.set_xlim(0, 100)
    ax_b.set_xlabel("Conditional Overlap Probability (%)", labelpad=3, fontsize=8.2)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(categories, fontsize=7.8)
    ax_b.xaxis.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax_b.set_axisbelow(True)
    
    ax_b.text(vals[1] + 2.0, y_pos[1], "78.9% (712 / 902)", va="center", ha="left", fontsize=8.0, fontweight="bold", color=IC_BLUE)
    ax_b.text(vals[0] + 2.0, y_pos[0], "32.9% (712 / 2,163)", va="center", ha="left", fontsize=8.0, fontweight="bold", color=DPL_ORANGE)
    
    ax_b.text(0.98, 0.08, "Asymmetry note: dPL attribute mapping induces a denser\nstrong field (2,163 vs 902 cells). Overlap asymmetry reflects\nfield density differences, not parameter estimation validity.",
              transform=ax_b.transAxes, fontsize=6.8, fontstyle="italic", color=MUTED_TEXT, ha="right", va="bottom",
              bbox=dict(boxstyle="square,pad=0.25", facecolor=BG_TINT_ORANGE, edgecolor="#fed7aa", lw=0.6))
    
    # =============================================================
    # PANEL (c): Retention Threshold Progression Ladder
    # =============================================================
    ax_c.set_title(r"(c) Retention Progression under Tightening dPL Thresholds ($n = 902$)",
                   loc="left", pad=6, fontsize=9.2, fontweight="bold", color=INK)
    
    steps = [
        "Same sign\n(any magnitude)",
        "Same sign\n" + r"$\wedge \; |\rho_{\mathrm{dPL}}| \geq 0.10$",
        "Same sign\n" + r"$\wedge \; |\rho_{\mathrm{dPL}}| \geq 0.20$",
        "Same sign\n" + r"$\wedge \; |\rho_{\mathrm{dPL}}| \geq 0.30$"
    ]
    rates = [94.124, 87.694, 76.718, 57.317]
    counts = [849, 791, 692, 517]
    x_pos = np.arange(len(steps))
    
    ax_c.plot(x_pos, rates, color=OVERLAP_GREEN, lw=1.6, marker="o", markersize=5.5,
              markerfacecolor="white", markeredgecolor=OVERLAP_GREEN, markeredgewidth=1.6, zorder=4)
    ax_c.bar(x_pos, rates, width=0.46, color="#e2f0e6", edgecolor=OVERLAP_GREEN, linewidth=0.75, alpha=0.85, zorder=3)
    
    ax_c.set_ylim(0, 112)
    ax_c.set_ylabel("Retention in dPL (% of n=902)", fontsize=8.0)
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(steps, fontsize=7.2)
    ax_c.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax_c.set_axisbelow(True)
    
    for i, (r, c) in enumerate(zip(rates, counts)):
        ax_c.text(i, r + 3.0, f"{r:.1f}%\n({c}/902)", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#1b5e20")
    
    # =============================================================
    # PANEL (d): Hydrological Process Domain Context Strip
    # =============================================================
    ax_d.set_title(r"(d) Hydrological Domain Distribution of Stable Associations ($n = 902$)",
                   loc="left", pad=6, fontsize=9.2, fontweight="bold", color=INK)
    
    domains = [
        "Hydrogeology\n& Storage",
        "Vegetation\n& Canopy",
        "Topography\n& Scale",
        "Climate &\nAridity/Snow",
        "Soil Moisture\n& Texture"
    ]
    domain_counts = [32, 43, 103, 358, 366]
    domain_shares = [c / 902 * 100 for c in domain_counts]
    domain_colors = ["#7c3aed", "#16a34a", "#ea580c", "#2563eb", "#475569"]
    
    y_d = np.arange(len(domains))
    bars_d = ax_d.barh(y_d, domain_counts, height=0.55, color="#cbd5e0", edgecolor=INK, linewidth=0.6, zorder=3)
    
    for bar, col in zip(bars_d, domain_colors):
        bar.set_facecolor(col)
        bar.set_alpha(0.85)
    
    ax_d.set_xlim(0, 430)
    ax_d.set_xlabel("Number of Stable Association Cells (out of 902)", labelpad=3, fontsize=8.0)
    ax_d.set_yticks(y_d)
    ax_d.set_yticklabels(domains, fontsize=7.2)
    ax_d.xaxis.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax_d.set_axisbelow(True)
    
    for i, (c, s) in enumerate(zip(domain_counts, domain_shares)):
        ax_d.text(c + 6, y_d[i], f"{c} ({s:.1f}%)", va="center", ha="left", fontsize=7.2, fontweight="bold", color=INK)
    
    # Save PNG at 600 DPI (PNG only, no PDF, no intermediate files)
    plt.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated {OUTPUT_PNG}")

if __name__ == "__main__":
    plot_figure5()
