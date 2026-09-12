#!/usr/bin/env python3
"""Plot Figure 5 (F5) for Journal of Hydrology manuscript R3.

Scientific Question:
    How does the catchment–parameter association field change from IC to dPL?

Key Evidence:
    1. Full 5,420-cell 5-category decomposition:
       - Neither strong: 3067 (56.6%)
       - dPL-only strong: 1451 (26.8%)
       - IC-only strong: 190 (3.5%)
       - Both strong, same sign: 692 (12.8%)
       - Both strong, opposite sign: 20 (0.37%)
    2. Bidirectional conditioning asymmetry:
       - P(dPL strong | IC strong) = 712 / 902 = 78.9%
       - P(IC strong | dPL strong) = 712 / 2163 = 32.9%
       - Both same sign / Both strong = 692 / 712 = 97.2%
    3. Hydrological information organization across 20 clusters in 5 domains.
"""
from __future__ import annotations

import os
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

OUTPUT_PNG = FIGURES_DIR / "F5.png"
OUTPUT_MAIN_PNG = R3_DIR / "F5_main.png"

# Color Palette: Green - Purple - Neutral System
COLOR_NEITHER = "#e2e8f0"     # Light Neutral Grey
COLOR_DPL_ONLY = "#805ad5"    # Purple / Plum (dPL)
COLOR_IC_ONLY = "#2f855a"     # Forest / Sage Green (IC)
COLOR_BOTH_SAME = "#134e4a"   # Deep Dark Teal-Green (Both same sign)
COLOR_BOTH_FLIP = "#334155"   # Dark Slate / Charcoal (Both sign flipped)

TEXT_DARK = "#1e293b"
TEXT_MUTED = "#64748b"
GRID_COLOR = "#f1f5f9"
BORDER_COLOR = "#cbd5e1"


def setup_matplotlib():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 9.2,
        "axes.titlesize": 9.8,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def load_and_classify_data():
    # Load relationship matrices
    rel_path = TABLES_DIR / "R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv"
    df = pd.read_csv(rel_path)

    df_ic = df[(df["space"] == "information_cluster") & (df["method"] == "IC")].rename(columns={"rho": "rho_ic"})
    df_dpl = df[(df["space"] == "information_cluster") & (df["method"] == "dPL")].rename(columns={"rho": "rho_dpl"})

    merged = pd.merge(
        df_ic[["model", "parameter_index", "parameter", "feature_index", "feature", "rho_ic"]],
        df_dpl[["model", "parameter_index", "parameter", "feature_index", "feature", "rho_dpl"]],
        on=["model", "parameter_index", "parameter", "feature_index", "feature"]
    )

    strong_ic = merged["rho_ic"].abs() >= 0.20
    strong_dpl = merged["rho_dpl"].abs() >= 0.20
    same_sign = np.sign(merged["rho_ic"]) == np.sign(merged["rho_dpl"])

    merged["category"] = "neither"
    merged.loc[(~strong_ic) & (strong_dpl), "category"] = "dpl_only"
    merged.loc[(strong_ic) & (~strong_dpl), "category"] = "ic_only"
    merged.loc[(strong_ic) & (strong_dpl) & (same_sign), "category"] = "both_same"
    merged.loc[(strong_ic) & (strong_dpl) & (~same_sign), "category"] = "both_flip"

    return merged


def get_cluster_definitions():
    # 20 information clusters mapped to hydrological domains & concise representative names
    # Grouped and ordered by physical domain
    clusters = [
        # Domain: Climate & Aridity / Snow (5)
        {"id": "absrho_ge_0.70_C014", "domain": "Climate", "label": r"Mean precip. / aridity ($p_{\mathrm{mean}}$, aridity)", "short": "Precip. / aridity"},
        {"id": "absrho_ge_0.70_C016", "domain": "Climate", "label": r"PET / snow fraction ($\mathrm{pet}_{\mathrm{mean}}$, $f_{\mathrm{snow}}$)", "short": "PET / snow fraction"},
        {"id": "absrho_ge_0.70_C015", "domain": "Climate", "label": r"Precip. seasonality ($p_{\mathrm{seasonality}}$)", "short": "Precip. seasonality"},
        {"id": "absrho_ge_0.70_C010", "domain": "Climate", "label": r"High/low precip. freq. ($f_{\mathrm{prec}}$)", "short": "Precip. frequency"},
        {"id": "absrho_ge_0.70_C012", "domain": "Climate", "label": r"Precip. duration / greenness ($d_{\mathrm{prec}}$, $\Delta\mathrm{GVF}$)", "short": "Precip. duration"},
        
        # Domain: Soil Moisture & Texture (5)
        {"id": "absrho_ge_0.70_C019", "domain": "Soil", "label": r"Elevation, slope & soil depth ($z_{\mathrm{mean}}$, $\theta_{\mathrm{slope}}$)", "short": "Elevation / slope / depth"},
        {"id": "absrho_ge_0.70_C013", "domain": "Soil", "label": r"Max water content & depth ($W_{\mathrm{max}}$, $d_{\mathrm{soil}}$)", "short": "Max water content"},
        {"id": "absrho_ge_0.70_C017", "domain": "Soil", "label": r"Sand & clay fraction ($f_{\mathrm{sand}}$, $f_{\mathrm{clay}}$)", "short": "Sand / clay fraction"},
        {"id": "absrho_ge_0.70_C018", "domain": "Soil", "label": r"Silt fraction ($f_{\mathrm{silt}}$)", "short": "Silt fraction"},
        {"id": "absrho_ge_0.70_C020", "domain": "Soil", "label": r"Soil porosity & conductivity ($\phi_{\mathrm{soil}}$, $K_{\mathrm{soil}}$)", "short": "Soil porosity / cond."},

        # Domain: Vegetation & Canopy (4)
        {"id": "absrho_ge_0.70_C005", "domain": "Vegetation", "label": r"Forest fraction ($f_{\mathrm{forest}}$)", "short": "Forest fraction"},
        {"id": "absrho_ge_0.70_C011", "domain": "Vegetation", "label": r"Max LAI & rooting depth ($\mathrm{LAI}_{\mathrm{max}}$, $d_{\mathrm{root}}$)", "short": "Max LAI / root depth"},
        {"id": "absrho_ge_0.70_C004", "domain": "Vegetation", "label": r"Dominant land cover fraction ($f_{\mathrm{land}}$)", "short": "Land cover fraction"},
        {"id": "absrho_ge_0.70_C003", "domain": "Vegetation", "label": r"Dominant land cover class ($\mathrm{code}_{\mathrm{land}}$)", "short": "Land cover class"},

        # Domain: Hydrogeology & Lithology (5)
        {"id": "absrho_ge_0.70_C008", "domain": "Geology", "label": r"Subsurface permeability ($k_{\mathrm{geol}}$)", "short": "Subsurface permeability"},
        {"id": "absrho_ge_0.70_C002", "domain": "Geology", "label": r"Carbonate rock fraction ($f_{\mathrm{carb}}$)", "short": "Carbonate rocks"},
        {"id": "absrho_ge_0.70_C006", "domain": "Geology", "label": r"Bedrock class & porosity ($\phi_{\mathrm{geol}}$)", "short": "Bedrock class / porosity"},
        {"id": "absrho_ge_0.70_C007", "domain": "Geology", "label": r"Secondary lithology class ($\mathrm{code}_{\mathrm{lith2}}$)", "short": "2nd lithology class"},
        {"id": "absrho_ge_0.70_C009", "domain": "Geology", "label": r"Lithology class fractions ($f_{\mathrm{glim}}$)", "short": "Lithology fractions"},

        # Domain: Topography & Drainage Scale (1)
        {"id": "absrho_ge_0.70_C001", "domain": "Topography", "label": r"Catchment drainage area ($A_{\mathrm{basin}}$)", "short": "Drainage area"},
    ]
    return clusters


def render_figure(df: pd.DataFrame):
    setup_matplotlib()

    # Total counts
    n_total = len(df)
    counts = df["category"].value_counts().to_dict()
    n_neither = counts.get("neither", 0)
    n_dpl_only = counts.get("dpl_only", 0)
    n_ic_only = counts.get("ic_only", 0)
    n_both_same = counts.get("both_same", 0)
    n_both_flip = counts.get("both_flip", 0)

    # Assert exact frozen values
    assert n_total == 5420, f"Expected 5420 total, got {n_total}"
    assert n_neither == 3067, f"Expected 3067 neither, got {n_neither}"
    assert n_dpl_only == 1451, f"Expected 1451 dpl_only, got {n_dpl_only}"
    assert n_ic_only == 190, f"Expected 190 ic_only, got {n_ic_only}"
    assert n_both_same == 692, f"Expected 692 both_same, got {n_both_same}"
    assert n_both_flip == 20, f"Expected 20 both_flip, got {n_both_flip}"

    # Create figure: 2-column layout (Left: Hero Panel a; Right: Panels b and c)
    fig = plt.figure(figsize=(11.8, 7.0), dpi=600)
    gs = GridSpec(
        nrows=2, ncols=2,
        width_ratios=[1.06, 1.0],
        height_ratios=[1.8, 1.0],
        wspace=0.30, hspace=0.36,
        left=0.065, right=0.865, top=0.91, bottom=0.07
    )

    ax_a = fig.add_subplot(gs[:, 0])  # Hero spans entire left column
    ax_b = fig.add_subplot(gs[0, 1])  # Top-right
    ax_c = fig.add_subplot(gs[1, 1])  # Bottom-right
    # ==========================================
    # PANEL (a): HERO — Association-field composition
    # ==========================================
    ax_a.set_title(r"$\mathbf{(a)}$ Cross-paradigm composition of catchment–parameter associations",
                   loc="left", fontsize=9.8, pad=9, fontweight="bold", color=TEXT_DARK)

    # Sub-elements in Panel a:
    # 1. Top summary 100% stacked bar (All 5,420 candidate cells)
    # 2. Paradigm-specific breakdown bars (IC Strong: 902 vs dPL Strong: 2,163)
    # 3. Direct count and percentage cards / breakdown bars for all 5 mutually exclusive classes

    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(0, 100)
    ax_a.axis("off")

    # --- Bar 1: Full 5,420 cells composition ---
    bar_y = 86
    bar_h = 7.5

    pct_neither = (n_neither / n_total) * 100
    pct_dpl_only = (n_dpl_only / n_total) * 100
    pct_both_same = (n_both_same / n_total) * 100
    pct_ic_only = (n_ic_only / n_total) * 100
    pct_both_flip = (n_both_flip / n_total) * 100

    lefts = [0,
             pct_neither,
             pct_neither + pct_dpl_only,
             pct_neither + pct_dpl_only + pct_both_same,
             pct_neither + pct_dpl_only + pct_both_same + pct_ic_only]
    widths = [pct_neither, pct_dpl_only, pct_both_same, pct_ic_only, pct_both_flip]
    colors = [COLOR_NEITHER, COLOR_DPL_ONLY, COLOR_BOTH_SAME, COLOR_IC_ONLY, COLOR_BOTH_FLIP]

    ax_a.text(0, bar_y + bar_h + 1.8, r"$\mathbf{All\ 5{,}420\ Candidate\ Association\ Cells\ (100\%)}$",
              fontsize=8.6, color=TEXT_DARK, va="bottom")

    for left, width, col in zip(lefts, widths, colors):
        rect = mpatches.Rectangle((left, bar_y), width, bar_h, facecolor=col, edgecolor="white", linewidth=0.8)
        ax_a.add_patch(rect)

    # Annotations on the total bar
    ax_a.text(lefts[0] + widths[0]/2, bar_y + bar_h/2, f"Neither strong\n{n_neither} ({pct_neither:.1f}%)",
              ha="center", va="center", fontsize=7.6, color=TEXT_MUTED, fontweight="bold")
    ax_a.text(lefts[1] + widths[1]/2, bar_y + bar_h/2, f"dPL-only\n{n_dpl_only}\n({pct_dpl_only:.1f}%)",
              ha="center", va="center", fontsize=7.2, color="white", fontweight="bold")
    ax_a.text(lefts[2] + widths[2]/2, bar_y + bar_h/2, f"Both\n(same sign)\n{n_both_same}\n({pct_both_same:.1f}%)",
              ha="center", va="center", fontsize=6.8, color="white", fontweight="bold")
    # Annotate IC-only and Both-flip outside with callout lines due to narrow width
    ic_center = lefts[3] + widths[3]/2
    ax_a.annotate(f"IC-only: {n_ic_only} ({pct_ic_only:.1f}%)",
                  xy=(ic_center, bar_y), xytext=(ic_center - 10, bar_y - 7.5),
                  ha="center", va="top", fontsize=7.3, color=COLOR_IC_ONLY, fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color=COLOR_IC_ONLY, lw=0.7, shrinkA=2, shrinkB=2))

    flip_center = lefts[4] + widths[4]/2
    ax_a.annotate(f"Both (sign-flipped): {n_both_flip} ({pct_both_flip:.2f}%)",
                  xy=(flip_center, bar_y + bar_h), xytext=(flip_center, bar_y + bar_h + 5.5),
                  ha="right", va="bottom", fontsize=7.0, color=COLOR_BOTH_FLIP, fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color=COLOR_BOTH_FLIP, lw=0.7, shrinkA=2, shrinkB=2))

    # --- Section 2: Detailed Category Decomposition Cards ---
    ax_a.text(0, 68, r"$\mathbf{Mutually\ Exclusive\ Category\ Breakdown}$", fontsize=8.6, color=TEXT_DARK)

    cat_data = [
        {"name": "Neither strong in IC nor dPL", "count": n_neither, "pct": pct_neither, "color": COLOR_NEITHER, "tc": TEXT_DARK, "sub": r"$|\rho_{\mathrm{IC}}| < 0.20\ \mathrm{and}\ |\rho_{\mathrm{dPL}}| < 0.20$"},
        {"name": "dPL-only strong", "count": n_dpl_only, "pct": pct_dpl_only, "color": COLOR_DPL_ONLY, "tc": "white", "sub": r"$|\rho_{\mathrm{IC}}| < 0.20\ \mathrm{and}\ |\rho_{\mathrm{dPL}}| \geq 0.20$ (denser mapping field)"},
        {"name": "Strong in both paradigms, same sign", "count": n_both_same, "pct": pct_both_same, "color": COLOR_BOTH_SAME, "tc": "white", "sub": r"$\mathrm{sign}(\rho_{\mathrm{IC}}) = \mathrm{sign}(\rho_{\mathrm{dPL}})\ \mathrm{and}\ |\rho| \geq 0.20$ (robust retention)"},
        {"name": "IC-only strong", "count": n_ic_only, "pct": pct_ic_only, "color": COLOR_IC_ONLY, "tc": "white", "sub": r"$|\rho_{\mathrm{IC}}| \geq 0.20\ \mathrm{and}\ |\rho_{\mathrm{dPL}}| < 0.20$ (associations lost in dPL)"},
        {"name": "Strong in both paradigms, opposite sign", "count": n_both_flip, "pct": pct_both_flip, "color": COLOR_BOTH_FLIP, "tc": "white", "sub": r"$\mathrm{sign}(\rho_{\mathrm{IC}}) \neq \mathrm{sign}(\rho_{\mathrm{dPL}})\ \mathrm{and}\ |\rho| \geq 0.20$ (rare sign reversals)"},
    ]

    card_y_start = 63.5
    card_h = 7.0
    card_spacing = 8.6

    for i, c in enumerate(cat_data):
        cy = card_y_start - i * card_spacing
        
        # Left color chip / mini bar
        chip_rect = mpatches.Rectangle((0, cy), 3.5, card_h, facecolor=c["color"], edgecolor=BORDER_COLOR, linewidth=0.5)
        ax_a.add_patch(chip_rect)

        # Category title & math subtext
        ax_a.text(5.0, cy + card_h - 1.8, c["name"], fontsize=8.0, color=TEXT_DARK, fontweight="bold", va="top")
        ax_a.text(5.0, cy + 1.2, c["sub"], fontsize=6.8, color=TEXT_MUTED, va="bottom")

        # Count and percentage on right
        ax_a.text(80.0, cy + card_h/2 + 0.5, f"{c['count']:,}", fontsize=8.6, color=TEXT_DARK, fontweight="bold", ha="right", va="center")
        ax_a.text(99.0, cy + card_h/2 + 0.5, f"{c['pct']:.1f}%", fontsize=8.6, color=c["color"] if c["color"] != COLOR_NEITHER else TEXT_MUTED, fontweight="bold", ha="right", va="center")

        # Bottom subtle separator line
        ax_a.plot([0, 100], [cy - 1.0, cy - 1.0], color=GRID_COLOR, lw=0.6)

    # --- Section 3: Paradigm Totals Comparison (IC Total vs dPL Total) ---
    comp_y = 15.0
    ax_a.text(0, comp_y + 4.5, r"$\mathbf{Paradigm\ Strong\ Association\ Totals\ (|\rho| \geq 0.20)}$", fontsize=8.4, color=TEXT_DARK)

    # IC Total bar (902)
    # Breakdown: IC-only (190, 21.1%), Both same (692, 76.7%), Both flip (20, 2.2%)
    ax_a.text(0, comp_y + 0.8, r"$\mathbf{IC\ Strong\ Total:}\ n = 902\ (100\%)$", fontsize=7.6, color=TEXT_DARK)
    ic_pct_same = (692 / 902) * 100
    ic_pct_only = (190 / 902) * 100
    ic_pct_flip = (20 / 902) * 100

    rect_ic1 = mpatches.Rectangle((0, comp_y - 3.8), ic_pct_same, 3.2, facecolor=COLOR_BOTH_SAME, edgecolor="white", lw=0.5)
    rect_ic2 = mpatches.Rectangle((ic_pct_same, comp_y - 3.8), ic_pct_only, 3.2, facecolor=COLOR_IC_ONLY, edgecolor="white", lw=0.5)
    rect_ic3 = mpatches.Rectangle((ic_pct_same + ic_pct_only, comp_y - 3.8), ic_pct_flip, 3.2, facecolor=COLOR_BOTH_FLIP, edgecolor="white", lw=0.5)
    ax_a.add_patch(rect_ic1); ax_a.add_patch(rect_ic2); ax_a.add_patch(rect_ic3)

    ax_a.text(ic_pct_same / 2, comp_y - 2.2, f"Retained in dPL: 692 ({ic_pct_same:.1f}%)", color="white", fontsize=6.8, ha="center", va="center", fontweight="bold")
    ax_a.text(ic_pct_same + ic_pct_only / 2, comp_y - 2.2, f"Lost: 190", color="white", fontsize=6.6, ha="center", va="center", fontweight="bold")

    # dPL Total bar (2163)
    # Breakdown: dPL-only (1451, 67.1%), Both same (692, 32.0%), Both flip (20, 0.9%)
    dpl_y = comp_y - 9.0
    ax_a.text(0, dpl_y + 4.2, r"$\mathbf{dPL\ Strong\ Total:}\ n = 2{,}163\ (100\%)$", fontsize=7.6, color=TEXT_DARK)
    dpl_pct_only = (1451 / 2163) * 100
    dpl_pct_same = (692 / 2163) * 100
    dpl_pct_flip = (20 / 2163) * 100

    rect_dpl1 = mpatches.Rectangle((0, dpl_y - 0.5), dpl_pct_only, 3.2, facecolor=COLOR_DPL_ONLY, edgecolor="white", lw=0.5)
    rect_dpl2 = mpatches.Rectangle((dpl_pct_only, dpl_y - 0.5), dpl_pct_same, 3.2, facecolor=COLOR_BOTH_SAME, edgecolor="white", lw=0.5)
    rect_dpl3 = mpatches.Rectangle((dpl_pct_only + dpl_pct_same, dpl_y - 0.5), dpl_pct_flip, 3.2, facecolor=COLOR_BOTH_FLIP, edgecolor="white", lw=0.5)
    ax_a.add_patch(rect_dpl1); ax_a.add_patch(rect_dpl2); ax_a.add_patch(rect_dpl3)

    ax_a.text(dpl_pct_only / 2, dpl_y + 1.1, f"dPL-only (constructive / unconstrained): 1,451 ({dpl_pct_only:.1f}%)", color="white", fontsize=6.8, ha="center", va="center", fontweight="bold")
    ax_a.text(dpl_pct_only + dpl_pct_same / 2, dpl_y + 1.1, f"Shared: 692 ({dpl_pct_same:.1f}%)", color="white", fontsize=6.8, ha="center", va="center", fontweight="bold")


    # ==========================================
    # PANEL (b): Hydrological Organization across 20 Info Clusters
    # ==========================================
    ax_b.set_title(r"$\mathbf{(b)}$ Association categories across catchment-information domains",
                   loc="left", fontsize=9.4, pad=7, fontweight="bold", color=TEXT_DARK)

    clusters = get_cluster_definitions()
    c_df = pd.crosstab(df["feature"], df["category"])

    # Build matrix in the order specified in clusters (reversed for top-to-bottom barh)
    clusters_rev = list(reversed(clusters))
    y_pos = np.arange(len(clusters_rev))

    # Pre-calculate counts per cluster
    neither_counts = [c_df.loc[c["id"], "neither"] if "neither" in c_df.columns else 0 for c in clusters_rev]
    dpl_counts = [c_df.loc[c["id"], "dpl_only"] if "dpl_only" in c_df.columns else 0 for c in clusters_rev]
    both_same_counts = [c_df.loc[c["id"], "both_same"] if "both_same" in c_df.columns else 0 for c in clusters_rev]
    ic_counts = [c_df.loc[c["id"], "ic_only"] if "ic_only" in c_df.columns else 0 for c in clusters_rev]
    both_flip_counts = [c_df.loc[c["id"], "both_flip"] if "both_flip" in c_df.columns else 0 for c in clusters_rev]

    totals = np.array(neither_counts) + np.array(dpl_counts) + np.array(both_same_counts) + np.array(ic_counts) + np.array(both_flip_counts)
    assert np.all(totals == 271), "Each cluster must have exactly 271 cells"

    # Convert to percentages for 100% stacked bar
    p_neither = np.array(neither_counts) / 271 * 100
    p_dpl = np.array(dpl_counts) / 271 * 100
    p_same = np.array(both_same_counts) / 271 * 100
    p_ic = np.array(ic_counts) / 271 * 100
    p_flip = np.array(both_flip_counts) / 271 * 100

    bar_height = 0.65

    # Draw stacked bars
    ax_b.barh(y_pos, p_neither, height=bar_height, left=0, color=COLOR_NEITHER, edgecolor="white", linewidth=0.3, label="Neither strong")
    ax_b.barh(y_pos, p_dpl, height=bar_height, left=p_neither, color=COLOR_DPL_ONLY, edgecolor="white", linewidth=0.3, label="dPL-only strong")
    ax_b.barh(y_pos, p_same, height=bar_height, left=p_neither + p_dpl, color=COLOR_BOTH_SAME, edgecolor="white", linewidth=0.3, label="Both, same sign")
    ax_b.barh(y_pos, p_ic, height=bar_height, left=p_neither + p_dpl + p_same, color=COLOR_IC_ONLY, edgecolor="white", linewidth=0.3, label="IC-only strong")
    ax_b.barh(y_pos, p_flip, height=bar_height, left=p_neither + p_dpl + p_same + p_ic, color=COLOR_BOTH_FLIP, edgecolor="white", linewidth=0.3, label="Both, opposite sign")

    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(-0.6, len(clusters_rev) - 0.4)
    ax_b.set_xlabel("Proportion of model parameter cells (%)", fontsize=8.0, labelpad=2)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([c["short"] for c in clusters_rev], fontsize=6.8)
    ax_b.grid(axis="x", color=GRID_COLOR, linewidth=0.5, linestyle="--", zorder=0)

    # Add Domain bracket indicators / annotations on the right y-axis or left margin
    domain_boundaries = [
        ("Topography (1)", 0, 0),
        ("Hydrogeology (5)", 1, 5),
        ("Vegetation (4)", 6, 9),
        ("Soil (5)", 10, 14),
        ("Climate (5)", 15, 19)
    ]
    for d_name, start_idx, end_idx in domain_boundaries:
        mid_y = (start_idx + end_idx) / 2
        ax_b.text(102, mid_y, d_name, va="center", ha="left", fontsize=6.6, color=TEXT_MUTED, fontweight="bold")
        # Draw small bracket line
        ax_b.plot([100.8, 101.4, 101.4, 100.8], [start_idx - 0.25, start_idx - 0.25, end_idx + 0.25, end_idx + 0.25],
                  color=BORDER_COLOR, lw=0.6, clip_on=False)

    # Compact Legend above Panel b
    handles, labels = ax_b.get_legend_handles_labels()
    ax_b.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 1.10),
                ncol=3, frameon=False, fontsize=6.6, handlelength=0.9, handleheight=0.7, columnspacing=0.8)


    # ==========================================
    # PANEL (c): Bidirectional Overlap Asymmetry
    # ==========================================
    ax_c.set_title(r"$\mathbf{(c)}$ Bidirectional overlap asymmetry of strong association fields",
                   loc="left", fontsize=9.4, pad=7, fontweight="bold", color=TEXT_DARK)

    # 2 horizontal bars comparing P(dPL strong | IC strong) vs P(IC strong | dPL strong)
    y_c = np.array([1, 0])
    probs = [712 / 902 * 100, 712 / 2163 * 100]
    labels_c = [
        r"$P(\mathrm{dPL\ strong} \mid \mathrm{IC\ strong})$",
        r"$P(\mathrm{IC\ strong} \mid \mathrm{dPL\ strong})$"
    ]
    bar_colors_c = [COLOR_BOTH_SAME, COLOR_DPL_ONLY]

    bars = ax_c.barh(y_c, probs, height=0.45, color=bar_colors_c, edgecolor="white", linewidth=0.5, zorder=3)
    ax_c.set_xlim(0, 100)
    ax_c.set_ylim(-0.5, 1.6)
    ax_c.set_xlabel("Conditional association recovery rate (%)", fontsize=8.0, labelpad=2)
    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels(labels_c, fontsize=8.2, fontweight="bold")
    ax_c.grid(axis="x", color=GRID_COLOR, linewidth=0.5, linestyle="--", zorder=0)

    # Direct value annotations
    ax_c.text(probs[0] + 1.5, y_c[0], r"$\mathbf{78.9\%}$ ($712 / 902$ IC-stable cells recovered)",
              va="center", ha="left", fontsize=7.8, color=COLOR_BOTH_SAME, fontweight="bold")
    ax_c.text(probs[1] + 1.5, y_c[1], r"$\mathbf{32.9\%}$ ($712 / 2{,}163$ dPL-strong cells recovered)",
              va="center", ha="left", fontsize=7.8, color=COLOR_DPL_ONLY, fontweight="bold")

    # Short footnote annotation
    ax_c.text(0, -0.42,
              r"$\ast\ 692/712$ ($97.2\%$) retain sign concordance. Asymmetry reflects dPL association field density ($n=2{,}163$ vs $902$), not estimation validity.",
              fontsize=6.8, color=TEXT_MUTED, va="top")

    # Save to both paths
    plt.savefig(OUTPUT_PNG, dpi=600)
    plt.savefig(OUTPUT_MAIN_PNG, dpi=600)
    plt.close()

    print(f"[SUCCESS] Saved F5.png to: {OUTPUT_PNG}")
    print(f"[SUCCESS] Saved F5_main.png to: {OUTPUT_MAIN_PNG}")


if __name__ == "__main__":
    df = load_and_classify_data()
    render_figure(df)
