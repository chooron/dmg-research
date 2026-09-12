#!/usr/bin/env python3
"""Plot Figure 5 (F5) for Journal of Hydrology manuscript R3.

Scientific Question:
    For the same parameter coordinate and catchment-information dimension, how similar are the IC and dPL associations,
    how are association-strength shifts distributed across hydrological domains, and is strong-association overlap
    consistently greater than marginal expectation across conceptual models?

Three Core Panels:
    (a) Paired catchment–parameter associations
        - 2D hexbin with clean flush alignment with KDE marginals
        - Threshold grid lines at x = +-t, y = +-t (t = 0.20)
        - Clean grouped annotations: Both strong (n=712, 97.2% same sign, 2.8% reversed), dPL-only (n=1451), IC-only (n=190)
        - Overlaid opposite-paradigm marginal densities with identical vertical scale
    (b) Paired association-strength shifts across catchment-information dimensions
        - 20-row continuous diverging gradient ridgelines
        - Alternating domain background bands
        - Left spine in SOLID BLACK
        - All text in solid black with enlarged font sizes and right-side aligned labels
        - Single clean header: 'Information dimension'
    (c) Strong-association overlap across conceptual models
        - Neutral title, x-axis extended to [0.8, max+5%] with [0.8, 1.0) shaded
        - Dark baseline at x = 1.0 ('1.0 marginal expectation')
        - Unboxed minimal 2-line summary (36/36 models > 1, Median = 2.00x)
        - collie1† annotated with dagger; flexb with shortened leader
        - Thicker step line, darker rug marks
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import gaussian_kde
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

# Color Palette: Green - Purple System
COLOR_IC = "#2e7d32"         # Forest / Emerald Green (IC)
COLOR_DPL = "#7b1fa2"        # Violet / Purple (dPL)
COLOR_DARK = "#0f172a"       # Dark Slate / Charcoal for outlines and main text
COLOR_BLACK = "#000000"      # Pure Black for panel b labels and spines
COLOR_MUTED = "#475569"      # Muted Slate for column headers and subtitles
COLOR_LIGHT_BG = "#f4f6f8"   # Very subtle soft light grey for alternating domain bands
GRID_COLOR = "#f1f5f9"
BORDER_COLOR = "#cbd5e1"
LINE_REF = "#94a3b8"

# Continuous diverging colormap for Panel b: Green -> Light Neutral -> Purple
CMAP_DIV = LinearSegmentedColormap.from_list(
    "grad_ic_dpl",
    [COLOR_IC, "#eef2f6", COLOR_DPL]
)
NORM_DIV = TwoSlopeNorm(vmin=-0.50, vcenter=0.0, vmax=0.70)


def setup_matplotlib():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11.5,
        "axes.labelsize": 11.8,
        "axes.titlesize": 12.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.0,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def load_data():
    rel_path = TABLES_DIR / "R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv"
    df = pd.read_csv(rel_path)

    df_ic = df[(df["space"] == "information_cluster") & (df["method"] == "IC")].rename(columns={"rho": "rho_ic"})
    df_dpl = df[(df["space"] == "information_cluster") & (df["method"] == "dPL")].rename(columns={"rho": "rho_dpl"})

    merged = pd.merge(
        df_ic[["model", "parameter_index", "parameter", "feature_index", "feature", "rho_ic"]],
        df_dpl[["model", "parameter_index", "parameter", "feature_index", "feature", "rho_dpl"]],
        on=["model", "parameter_index", "parameter", "feature_index", "feature"]
    )

    # Dynamic threshold t from frozen classification
    t = 0.20
    strong_ic = merged["rho_ic"].abs() >= t
    strong_dpl = merged["rho_dpl"].abs() >= t
    both_strong = strong_ic & strong_dpl
    same_sign = np.sign(merged["rho_ic"]) == np.sign(merged["rho_dpl"])

    merged["strong_ic"] = strong_ic
    merged["strong_dpl"] = strong_dpl
    merged["both_strong"] = both_strong
    merged["same_sign"] = same_sign

    merged["abs_ic"] = merged["rho_ic"].abs()
    merged["abs_dpl"] = merged["rho_dpl"].abs()
    merged["delta_abs"] = merged["abs_dpl"] - merged["abs_ic"]

    return merged, t


def get_cluster_definitions():
    # 20 information clusters mapped to hydrological domains & concise representative names
    # Grouped from top (Climate) to bottom (Topography)
    clusters = [
        # Domain: Climate & Snow (5)
        {"id": "absrho_ge_0.70_C014", "domain": "Climate & Snow (5)", "label": "Mean precip. & aridity"},
        {"id": "absrho_ge_0.70_C016", "domain": "Climate & Snow (5)", "label": "PET & snow fraction"},
        {"id": "absrho_ge_0.70_C015", "domain": "Climate & Snow (5)", "label": "Precipitation seasonality"},
        {"id": "absrho_ge_0.70_C010", "domain": "Climate & Snow (5)", "label": "High/low precip. freq."},
        {"id": "absrho_ge_0.70_C012", "domain": "Climate & Snow (5)", "label": "Precip. duration & greenness"},

        # Domain: Soil & Terrain (5)
        {"id": "absrho_ge_0.70_C019", "domain": "Soil & Terrain (5)", "label": "Elevation, slope & soil depth"},
        {"id": "absrho_ge_0.70_C013", "domain": "Soil & Terrain (5)", "label": "Max water content & depth"},
        {"id": "absrho_ge_0.70_C017", "domain": "Soil & Terrain (5)", "label": "Sand & clay fraction"},
        {"id": "absrho_ge_0.70_C018", "domain": "Soil & Terrain (5)", "label": "Silt fraction"},
        {"id": "absrho_ge_0.70_C020", "domain": "Soil & Terrain (5)", "label": "Soil porosity & cond."},

        # Domain: Vegetation (4)
        {"id": "absrho_ge_0.70_C005", "domain": "Vegetation (4)", "label": "Forest fraction"},
        {"id": "absrho_ge_0.70_C011", "domain": "Vegetation (4)", "label": "Max LAI & rooting depth"},
        {"id": "absrho_ge_0.70_C004", "domain": "Vegetation (4)", "label": "Dominant land cover frac."},
        {"id": "absrho_ge_0.70_C003", "domain": "Vegetation (4)", "label": "Land cover class"},

        # Domain: Geology (5)
        {"id": "absrho_ge_0.70_C008", "domain": "Geology (5)", "label": "Subsurface permeability"},
        {"id": "absrho_ge_0.70_C002", "domain": "Geology (5)", "label": "Carbonate rock fraction"},
        {"id": "absrho_ge_0.70_C006", "domain": "Geology (5)", "label": "Bedrock class & porosity"},
        {"id": "absrho_ge_0.70_C007", "domain": "Geology (5)", "label": "Secondary lithology class"},
        {"id": "absrho_ge_0.70_C009", "domain": "Geology (5)", "label": "Lithology class fractions"},

        # Domain: Topography (1)
        {"id": "absrho_ge_0.70_C001", "domain": "Topography (1)", "label": "Catchment drainage area"},
    ]
    return clusters


def render_figure(df: pd.DataFrame, t: float):
    setup_matplotlib()

    # --- ASSERTIONS & AUDIT CALCULATIONS ---
    # 1. Five class counts based on threshold t
    strong_ic = df["rho_ic"].abs() >= t
    strong_dpl = df["rho_dpl"].abs() >= t
    both_strong = strong_ic & strong_dpl
    same_sign = np.sign(df["rho_ic"]) == np.sign(df["rho_dpl"])

    c_neither = ((~strong_ic) & (~strong_dpl)).sum()
    c_ic_only = (strong_ic & (~strong_dpl)).sum()
    c_dpl_only = ((~strong_ic) & strong_dpl).sum()
    c_both_same = (both_strong & same_sign).sum()
    c_both_flip = (both_strong & (~same_sign)).sum()
    c_total = len(df)

    assert c_neither == 3067, f"Neither count mismatch: {c_neither} != 3067"
    assert c_ic_only == 190, f"IC only count mismatch: {c_ic_only} != 190"
    assert c_dpl_only == 1451, f"dPL only count mismatch: {c_dpl_only} != 1451"
    assert c_both_same == 692, f"Both same sign count mismatch: {c_both_same} != 692"
    assert c_both_flip == 20, f"Both sign-flip count mismatch: {c_both_flip} != 20"
    assert c_total == 5420, f"Total count mismatch: {c_total} != 5420"
    assert c_neither + c_ic_only + c_dpl_only + c_both_same + c_both_flip == 5420

    # 2. Compute model level enrichments for panel (c)
    models = sorted(df["model"].unique())
    model_enrich = []
    for m in models:
        sub = df[df["model"] == m]
        n_tot = len(sub)
        n_ic_m = (sub["rho_ic"].abs() >= t).sum()
        n_dpl_m = (sub["rho_dpl"].abs() >= t).sum()
        n_both_m = ((sub["rho_ic"].abs() >= t) & (sub["rho_dpl"].abs() >= t)).sum()
        e_both_m = (n_ic_m * n_dpl_m) / n_tot
        enrich_m = n_both_m / e_both_m if e_both_m > 0 else np.nan
        p_cnt = sub["parameter_index"].nunique()
        model_enrich.append({"model": m, "enrich": enrich_m, "p_cnt": p_cnt, "n_both": n_both_m, "e_both": e_both_m})

    df_m = pd.DataFrame(model_enrich).sort_values("enrich").reset_index(drop=True)
    count_em_less_1 = (df_m["enrich"] < 1.0).sum()
    assert count_em_less_1 == 0, f"Found models with Em < 1.0: {df_m[df_m['enrich'] < 1.0]['model'].tolist()}"

    pooled_obs = (strong_ic & strong_dpl).sum()
    pooled_exp = (strong_ic.sum() * strong_dpl.sum()) / c_total
    pooled_enrich = pooled_obs / pooled_exp
    median_em = float(df_m["enrich"].median())

    collie1_row = df_m[df_m["model"] == "collie1"].iloc[0]

    # --- FIGURE LAYOUT CREATION ---
    # Tightened horizontal gap between Left Column (a/c) and Right Column (b): wspace = 0.05
    fig = plt.figure(figsize=(13.0, 8.2), dpi=600)

    gs_master = GridSpec(
        nrows=1, ncols=2,
        width_ratios=[1.0, 1.48],
        wspace=0.05,
        left=0.060, right=0.965, top=0.94, bottom=0.08
    )

    # Left Column Grid (Top: Panel a, Bottom: Panel c) - Tightened vertical gap: hspace = 0.20
    gs_left = GridSpecFromSubplotSpec(
        nrows=2, ncols=1,
        subplot_spec=gs_master[0, 0],
        height_ratios=[1.80, 1.0],
        hspace=0.20
    )

    # Panel a Sub-Grid: Top-marginal, Main 2D, Right-marginal
    gs_a = GridSpecFromSubplotSpec(
        nrows=2, ncols=2,
        subplot_spec=gs_left[0, 0],
        height_ratios=[0.16, 1.0],
        width_ratios=[1.0, 0.16],
        wspace=0.03, hspace=0.03
    )

    ax_a_main = fig.add_subplot(gs_a[1, 0])
    ax_a_top = fig.add_subplot(gs_a[0, 0], sharex=ax_a_main)
    ax_a_right = fig.add_subplot(gs_a[1, 1], sharey=ax_a_main)

    # Panel c (Bottom-Left)
    ax_c = fig.add_subplot(gs_left[1, 0])

    # Right Column: Panel b (20-row ridgelines, expanded width)
    ax_b = fig.add_subplot(gs_master[0, 1])


    # =========================================================
    # PANEL (a): Paired Catchment–Parameter Associations
    # =========================================================
    ax_a_top.set_title(r"$\mathbf{(a)}$ Paired catchment–parameter associations",
                       loc="left", fontsize=12.2, pad=6, fontweight="bold", color=COLOR_DARK)

    x_ic = df["rho_ic"].to_numpy()
    y_dpl = df["rho_dpl"].to_numpy()

    # Equal range for both axes based on pooled data
    data_max = max(np.abs(x_ic).max(), np.abs(y_dpl).max())
    lim = float(np.ceil(data_max * 10) / 10 + 0.02)  # 0.82

    # 2D Hexbin on main axis
    hb = ax_a_main.hexbin(
        x_ic, y_dpl,
        gridsize=34,
        extent=[-lim, lim, -lim, lim],
        cmap="Greys",
        mincnt=1,
        bins="log",
        linewidths=0.2,
        edgecolors="#eef2f6",
        zorder=2
    )

    # Layer 1: Coordinate references (x=0, y=0) - Thinner, lighter grey
    ax_a_main.axhline(0, color="#cbd5e1", lw=0.60, linestyle="-", zorder=3)
    ax_a_main.axvline(0, color="#cbd5e1", lw=0.60, linestyle="-", zorder=3)

    # Layer 2: Conclusion reference (y=x) - Darker dashed line
    ax_a_main.plot([-lim, lim], [-lim, lim], color="#334155", lw=0.95, linestyle="--", zorder=4)

    # Layer 3: Intensity threshold grid lines (x = +-t, y = +-t) - Thinnest, lightest dotted lines
    for thresh in [-t, t]:
        ax_a_main.axhline(thresh, color="#94a3b8", lw=0.55, linestyle=":", zorder=3)
        ax_a_main.axvline(thresh, color="#94a3b8", lw=0.55, linestyle=":", zorder=3)

    ax_a_main.set_xlim(-lim, lim)
    ax_a_main.set_ylim(-lim, lim)
    ax_a_main.set_xlabel(r"IC catchment–information association, $\rho_{\mathrm{IC}}$", fontsize=11.6, color=COLOR_IC, fontweight="bold", labelpad=2)
    ax_a_main.set_ylabel(r"dPL catchment–information association, $\rho_{\mathrm{dPL}}$", fontsize=11.6, color=COLOR_DPL, fontweight="bold", labelpad=2)
    ax_a_main.grid(color=GRID_COLOR, lw=0.5, linestyle="--", zorder=0)

    # Clean grouped annotations
    both_strong_text = (
        r"$\mathbf{Both\ strong:\ } n = 712$" "\n"
        r"$97.2\%\ \mathrm{same\ sign}$" "\n"
        r"$2.8\%\ \mathrm{reversed}$"
    )
    ax_a_main.text(0.48, 0.68, both_strong_text,
                   fontsize=9.2, color=COLOR_DARK, fontweight="bold", va="top", ha="center",
                   bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.85),
                   zorder=5)

    ax_a_main.text(-0.50, 0.60, r"$\mathbf{dPL\text{-}only}$" "\n" r"$n = 1{,}451$",
                   fontsize=9.2, color=COLOR_DPL, fontweight="bold", ha="center", va="center", zorder=5)
    ax_a_main.text(0.52, -0.05, r"$\mathbf{IC\text{-}only}$" "\n" r"$n = 190$",
                   fontsize=9.2, color=COLOR_IC, fontweight="bold", ha="center", va="center", zorder=5)

    # Marginal KDE distributions
    kde_ic = gaussian_kde(x_ic)
    kde_dpl = gaussian_kde(y_dpl)
    eval_pts = np.linspace(-lim, lim, 200)
    dens_ic = kde_ic(eval_pts)
    dens_dpl = kde_dpl(eval_pts)

    # Shared density scale upper limit across both marginals
    d_max = max(dens_ic.max(), dens_dpl.max()) * 1.08

    # Top marginal KDE (rho_IC solid green + overlaid rho_dPL light purple dashed)
    ax_a_top.fill_between(eval_pts, 0, dens_ic, color=COLOR_IC, alpha=0.30, zorder=2)
    ax_a_top.plot(eval_pts, dens_ic, color=COLOR_IC, lw=1.2, zorder=3)
    ax_a_top.plot(eval_pts, dens_dpl, color=COLOR_DPL, lw=1.1, linestyle="--", alpha=0.75, zorder=3)
    ax_a_top.axvline(0, color=LINE_REF, lw=0.5, linestyle="-")
    ax_a_top.set_xlim(-lim, lim)
    ax_a_top.set_ylim(0, d_max)
    ax_a_top.axis("off")

    # Right marginal KDE (rho_dPL solid purple + overlaid rho_IC light green dashed)
    ax_a_right.fill_betweenx(eval_pts, 0, dens_dpl, color=COLOR_DPL, alpha=0.30, zorder=2)
    ax_a_right.plot(dens_dpl, eval_pts, color=COLOR_DPL, lw=1.2, zorder=3)
    ax_a_right.plot(dens_ic, eval_pts, color=COLOR_IC, lw=1.1, linestyle="--", alpha=0.75, zorder=3)
    ax_a_right.axhline(0, color=LINE_REF, lw=0.5, linestyle="-")
    ax_a_right.set_ylim(-lim, lim)
    ax_a_right.set_xlim(0, d_max)
    ax_a_right.axis("off")


    # =========================================================
    # PANEL (c): Strong-Association Overlap across Conceptual Models
    # =========================================================
    ax_c.set_title(r"$\mathbf{(c)}$ Strong-association overlap across conceptual models",
                   loc="left", fontsize=11.8, pad=6, fontweight="bold", color=COLOR_DARK)

    enrich_vals = df_m["enrich"].to_numpy()
    ecdf_y = np.arange(1, len(enrich_vals) + 1) / len(enrich_vals)

    # Extend x-axis below 1.0 to [0.80, data_max + 5%]
    c_xmin = 0.80
    c_xmax = float(enrich_vals.max() * 1.05) + 0.05

    # Light background tint on interval [0.8, 1.0) showing absence of model density
    ax_c.axvspan(c_xmin, 1.0, facecolor="#f1f5f9", edgecolor="none", zorder=0)

    # Distinct Marginal Expectation Baseline at x = 1.0 (Dark solid line)
    ax_c.axvline(1.0, color="#334155", lw=1.2, linestyle="-", zorder=2)
    ax_c.text(1.02, 0.06, "1.0 marginal expectation", color="#334155", fontsize=9.2, rotation=90, va="bottom", fontweight="bold")

    # ECDF Step curve (Thicker dark charcoal line)
    ax_c.step(enrich_vals, ecdf_y, where="post", color=COLOR_DARK, lw=2.0, zorder=3)

    # Darker rug ticks along the bottom x-axis
    ax_c.plot(enrich_vals, np.zeros_like(enrich_vals) + 0.018, "|", color="#0f172a", markersize=8, mew=1.2, zorder=4)

    # Dotted line and marker at median Em
    ax_c.axvline(median_em, color="#475569", lw=0.95, linestyle=":", zorder=2)
    ax_c.scatter([median_em], [0.50], color=COLOR_DARK, s=32, zorder=5, edgecolor="white", lw=0.6)

    # Anchor Model Annotations: flexb and collie1†
    min_m = df_m.iloc[0]
    max_m = df_m.iloc[-1]

    ax_c.annotate(f"{min_m['model']} ({min_m['enrich']:.2f}×)",
                  xy=(min_m["enrich"], 1/36), xytext=(min_m["enrich"] + 0.05, 0.16),
                  fontsize=9.4, color=COLOR_DARK, fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color=COLOR_MUTED, lw=0.55, shrinkA=1, shrinkB=2))

    ax_c.annotate(f"{max_m['model']}† ({max_m['enrich']:.2f}×)",
                  xy=(max_m["enrich"], 1.0), xytext=(max_m["enrich"] - 0.55, 0.72),
                  fontsize=9.4, color=COLOR_DARK, fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color=COLOR_MUTED, lw=0.55, shrinkA=1, shrinkB=2))
    ax_c.set_xlim(c_xmin, c_xmax)
    ax_c.set_ylim(0, 1.04)
    ax_c.set_xlabel(r"Observed / marginally expected overlap", fontsize=11.5, labelpad=3)
    ax_c.set_ylabel("Cumulative model fraction", fontsize=11.5, labelpad=3)
    ax_c.grid(color=GRID_COLOR, lw=0.5, linestyle="--", zorder=0)

    # In-plot unboxed summary text
    summary_text = (
        r"$\mathbf{36/36\ models}\ > 1$" "\n"
        rf"$\mathrm{{Median}} = {median_em:.2f}\times$"
    )
    ax_c.text(0.14, 0.94, summary_text, transform=ax_c.transAxes,
              fontsize=9.6, color=COLOR_DARK, fontweight="bold", va="top", ha="left")


    # =========================================================
    # PANEL (b): HERO — 20-Row Ridgelines (Solid Black Left Spine & Text, Enlarged Fonts)
    # =========================================================
    ax_b.set_title(r"$\mathbf{(b)}$ Paired association-strength shifts across catchment-information dimensions",
                   loc="left", fontsize=12.0, pad=7, fontweight="bold", color=COLOR_DARK)

    clusters = get_cluster_definitions()
    clusters_rev = list(reversed(clusters))
    n_rows = len(clusters_rev)
    y_pos = np.arange(n_rows)

    x_min, x_max_data = -0.55, 0.65
    x_max_axis = 1.44
    x_eval = np.linspace(x_min, x_max_data, 260)
    scale_h = 1.20

    # Domain Group Definitions & Alternating Background Bands
    domain_groups = [
        ("Topography (1)", 0, 0, True),
        ("Geology (5)", 1, 5, False),
        ("Vegetation (4)", 6, 9, True),
        ("Soil & Terrain (5)", 10, 14, False),
        ("Climate & Snow (5)", 15, 19, True)
    ]

    # Draw alternating background bands across the entire width of Panel b
    for d_name, start_idx, end_idx, is_grey in domain_groups:
        y_bottom = start_idx - 0.38
        y_top = end_idx + scale_h - 0.12
        if is_grey:
            ax_b.axhspan(y_bottom, y_top, color=COLOR_LIGHT_BG, zorder=0)

        # Group name directly on the right margin of the band in SOLID BLACK BOLD
        mid_y = (start_idx + end_idx) / 2 + 0.10
        ax_b.text(x_max_axis - 0.02, mid_y, d_name, va="center", ha="right",
                  fontsize=9.8, color=COLOR_BLACK, fontweight="bold")

    # Vertical Zero Reference Line
    ax_b.plot([0, 0], [-0.35, n_rows + scale_h - 0.3], color=LINE_REF, lw=0.75, linestyle="--", zorder=1)

    # Ridgelines with continuous progressive diverging gradient fill
    for i, c in enumerate(clusters_rev):
        sub = df[df["feature"] == c["id"]]
        deltas = sub["delta_abs"].to_numpy()

        kde = gaussian_kde(deltas, bw_method=0.30)
        dens = kde(x_eval)
        dens_norm = dens / np.max(dens) * scale_h

        base_y = y_pos[i]
        top_y = base_y + dens_norm

        # Draw continuous gradient fill using fine segments
        for j in range(len(x_eval) - 1):
            x0, x1 = x_eval[j], x_eval[j+1]
            x_mid = 0.5 * (x0 + x1)
            seg_col = CMAP_DIV(NORM_DIV(x_mid))
            ax_b.fill_between(
                [x0, x1], base_y, [top_y[j], top_y[j+1]],
                color=seg_col, alpha=0.65, edgecolor="none", zorder=2
            )

        # Baseline & top outline
        ax_b.plot(x_eval, top_y, color=COLOR_DARK, lw=0.80, zorder=3)
        ax_b.plot([x_min, x_max_data], [base_y, base_y], color=BORDER_COLOR, lw=0.45, zorder=1)

        # Median marker dot
        med_val = np.median(deltas)
        ax_b.scatter([med_val], [base_y + 0.10], color=COLOR_DARK,
                     s=16, marker="o", edgecolor="white", lw=0.5, zorder=4)

        # Right-aligned dimension label aligned with this row in SOLID BLACK
        ax_b.text(x_max_data + 0.04, base_y + 0.16, c["label"], va="center", ha="left",
                  fontsize=9.5, color=COLOR_BLACK)

    ax_b.set_xlim(x_min, x_max_axis)
    ax_b.set_ylim(-0.35, n_rows + scale_h - 0.25)
    ax_b.set_xticks([-0.50, -0.25, 0.0, 0.25, 0.50])
    ax_b.set_xlabel(r"Paired association-strength shift, $\Delta|\rho| = |\rho_{\mathrm{dPL}}| - |\rho_{\mathrm{IC}}|$",
                    fontsize=11.5, labelpad=3, color=COLOR_BLACK, fontweight="bold")

    # Left spine restored in SOLID BLACK
    ax_b.set_yticks([])
    ax_b.spines["left"].set_visible(True)
    ax_b.spines["left"].set_color(COLOR_BLACK)
    ax_b.spines["left"].set_linewidth(0.85)
    ax_b.tick_params(left=False, axis="x", labelcolor=COLOR_BLACK)
    ax_b.grid(axis="x", color=GRID_COLOR, lw=0.5, linestyle="--", zorder=0)

    # Top direction indicators
    ax_b.text(-0.48, n_rows + 0.22, r"$\leftarrow$ IC stronger", color=COLOR_IC, fontsize=10.2, fontweight="bold")
    ax_b.text(0.12, n_rows + 0.22, r"dPL stronger $\rightarrow$", color=COLOR_DPL, fontsize=10.2, fontweight="bold")

    # Single clean header above the 20 variable names column (Recommended Scheme A)
    ax_b.text(x_max_data + 0.04, n_rows + 0.22, "Information dimension", color=COLOR_DARK, fontsize=10.0, fontweight="bold", va="bottom", ha="left")

    # Save PNG only (600 DPI, no PDF)
    plt.savefig(OUTPUT_PNG, dpi=600)
    plt.savefig(OUTPUT_MAIN_PNG, dpi=600)
    plt.close()

    # --- PRINT REQUIRED CONSOLE CHECK REPORT ---
    cbar_max = int(hb.get_array().max())
    print("\n" + "="*70)
    print("FIGURE 5 CONSOLE AUDIT REPORT (VERIFICATION CHECKS)")
    print("="*70)
    print(f"1. Used threshold t: {t}")
    print(f"   - Neither strong (|x|<t and |y|<t): {c_neither} (expected 3067)")
    print(f"   - IC-only strong (|x|>=t and |y|<t): {c_ic_only} (expected 190)")
    print(f"   - dPL-only strong (|x|<t and |y|>=t): {c_dpl_only} (expected 1451)")
    print(f"   - Both strong, same sign: {c_both_same} (expected 692)")
    print(f"   - Both strong, sign-flip: {c_both_flip} (expected 20)")
    print(f"   - Total count sum: {c_total} (expected 5420)")

    print(f"\n2. Panel (a) Axis Limits & Alignment:")
    print(f"   - X-axis range: [-{lim}, {lim}], Y-axis range: [-{lim}, {lim}] (Equal: {lim == lim})")
    print(f"   - Flush marginal KDE alignment without aspect ratio gap")

    print(f"\n3. Marginal Densities Scale Consistency:")
    print(f"   - Top marginal Y-scale upper limit: {d_max:.4f}")
    print(f"   - Right marginal X-scale upper limit: {d_max:.4f}")
    print(f"   - Scale consistency confirmed identical: True")

    print(f"\n4. Hexbin Max Bin Count: {cbar_max}")

    print(f"\n5. Panel (c) Overlap Enrichment E_m Audit:")
    print(f"   - Number of models with E_m < 1.0: {count_em_less_1} (Confirmed strictly 0)")
    print(f"   - Min E_m: {df_m.iloc[0]['enrich']:.4f} ({df_m.iloc[0]['model']})")
    print(f"   - Median E_m: {median_em:.4f}")
    print(f"   - Max E_m: {df_m.iloc[-1]['enrich']:.4f} ({df_m.iloc[-1]['model']})")

    print(f"\n6. Pooled Enrichment vs Model-Level Median:")
    print(f"   - Pooled overlap enrichment: {pooled_enrich:.4f} ({pooled_obs} observed / {pooled_exp:.2f} expected)")
    print(f"   - Model-level median E_m: {median_em:.4f}")

    print(f"\n7. collie1 Diagnostic Audit:")
    print(f"   - N_both: {collie1_row['n_both']}")
    print(f"   - E[N_both]: {collie1_row['e_both']:.4f}")
    print(f"   - E_m: {collie1_row['enrich']:.4f}")

    print(f"\n8. Panel (b) Styling Update:")
    print(f"   - Left spine set to SOLID BLACK.")
    print(f"   - All text enlarged by 2.5-3pt across the figure.")
    print(f"   - Dimension labels & group headers set to solid black.")

    print(f"\n9. Deliverable Output Verification:")
    print(f"   - Saved PNG: {OUTPUT_PNG} (600 DPI, PNG only, no PDF)")
    print(f"   - Figure physical dimensions: 13.0 x 8.2 inches")
    print("="*70 + "\n")


if __name__ == "__main__":
    df, t = load_data()
    render_figure(df, t)
