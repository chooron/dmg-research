"""Appendix shared helpers — imports style directly from plots/common.py."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PLOTS_ROOT = ROOT / "project" / "parameterize" / "manuscript" / "plots"
sys.path.insert(0, str(PLOTS_ROOT))

# Re-export everything from the main common module
from common import (  # noqa: F401
    setup_style, clean_axes, add_panel_label,
    MM, DPI,
    MODEL_ORDER, MODEL_LABELS, MODEL_COLORS,
    PARAM_ORDER, ATTR_LABELS,
    TABLE_ROOT, CORR_ROOT, VAR_ROOT, ANALYSIS_ROOT,
    muted_diverging, muted_seq,
    p_label, a_label,
    math_model_labels,
)

PARAM_ROOT = ROOT / "project" / "parameterize"
MANUSCRIPT_ROOT = PARAM_ROOT / "manuscript"
APP_FIG_DIR = MANUSCRIPT_ROOT / "figures" / "appendix"
APP_PLOTS_DIR = MANUSCRIPT_ROOT / "plots_appendix"

STABILITY_ROOT = PARAM_ROOT / "outputs" / "analysis" / "stability_stats"
ANALYSIS_OUT = MANUSCRIPT_ROOT / "analysis"

SEED_LOSS_DIR  = ANALYSIS_OUT / "02_seed_loss_sensitivity" / "data"
SPATIAL_MEAN_DIR = ANALYSIS_OUT / "03_distributional_parameter_spatial_data" / "data"
MEAN_ATTR_DIR  = ANALYSIS_OUT / "04_mean_attribute_relationships" / "data"
GRADIENT_DIR   = ANALYSIS_OUT / "05_environmental_gradient_groups" / "data"
SPATIAL_STD_DIR = ANALYSIS_OUT / "06_uncertainty_spatial_data" / "data"
STD_ATTR_DIR   = ANALYSIS_OUT / "07_uncertainty_attribute_relationships" / "data"

CONUS_CLIPPED_DIR = ROOT / "data" / "camels_loc" / "conus_clipped"
BASIN_SHAPEFILE = CONUS_CLIPPED_DIR / "camels_671_loc_conus_clipped.shp"
STATE_SHAPEFILE = CONUS_CLIPPED_DIR / "s_18mr25_conus.shp"
BASIN_LIST = ROOT / "data" / "531sub_id.txt"
PARAMETER_TABLE = ANALYSIS_OUT / "figure2" / "data" / "parameter_estimates_by_run_long.csv"

REFERENCE_LOSS = "HybridNseBatchLoss"

# Process group colours — identical to fig10/fig11
GROUPS = [
    "Snow / seasonality",
    "Aridity / ET",
    "Terrain / topography",
    "Soil / storage",
    "Routing / extremes",
]
GROUP_COLORS = {
    "Snow / seasonality":   "#56B4E9",
    "Aridity / ET":         "#E69F00",
    "Terrain / topography": "#009E73",
    "Soil / storage":       "#CC79A7",
    "Routing / extremes":   "#0072B2",
}

ATTR_SHORT = {
    "frac_snow":            "Snow fraction",
    "elev_mean":            "Mean elevation",
    "pet_mean":             "Mean PET",
    "aridity":              "Aridity index",
    "p_seasonality":        "Precip. seasonality",
    "p_mean":               "Mean precip.",
    "slope_mean":           "Mean slope",
    "soil_conductivity":    "Soil conductivity",
    "clay_frac":            "Clay fraction",
    "frac_forest":          "Forest fraction",
    "soil_depth_pelletier": "Soil depth",
    "lai_diff":             "LAI seasonality",
    "high_prec_dur":        "High-prec. dur.",
    "high_prec_freq":       "High-prec. freq.",
    "low_prec_dur":         "Low-prec. dur.",
    "low_prec_freq":        "Low-prec. freq.",
    "area_gages2":          "Drainage area",
    "carbonate_rocks_frac": "Carbonate frac.",
}

N_BINS = 8  # same as fig10/fig11


def attr_label(attr: str) -> str:
    return ATTR_SHORT.get(attr, attr.replace("_", " "))


def strip_par(name: str) -> str:
    return str(name).replace("par", "")


def compute_binned(x, y, n_bins: int = N_BINS):
    """Identical to fig10/fig11 binning."""
    import numpy as np
    edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    bin_idx = np.digitize(x, edges[1:-1])
    centres, medians, q25s, q75s = [], [], [], []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        if mask.sum() < 4:
            continue
        centres.append(np.median(x[mask]))
        medians.append(np.median(y[mask]))
        q25s.append(np.percentile(y[mask], 25))
        q75s.append(np.percentile(y[mask], 75))
    import numpy as np
    return (np.array(centres), np.array(medians),
            np.array(q25s), np.array(q75s))


def plot_gradient_panel(ax, x, y, attr: str, param: str, rho: float,
                        color: str, show_xlabel: bool, show_ylabel: bool,
                        ylabel_suffix: str = "mean", seed_sd: float | None = None):
    """Single gradient panel — identical visual language to fig10/fig11."""
    import numpy as np
    import matplotlib as mpl

    xlo, xhi = np.percentile(x, 1), np.percentile(x, 99)
    ylo, yhi = np.percentile(y, 1), np.percentile(y, 99)
    xpad = max((xhi - xlo) * 0.05, 1e-6)
    ypad = max((yhi - ylo) * 0.08, 1e-6)

    ax.scatter(x, y, s=3.5, alpha=0.22, color="#BBBBBB",
               linewidths=0, rasterized=True, zorder=1)

    bx, bmed, bq25, bq75 = compute_binned(x, y)
    if len(bx) >= 2:
        ax.fill_between(bx, bq25, bq75, alpha=0.25, color=color,
                        linewidth=0, zorder=2)
        ax.plot(bx, bmed, color=color, linewidth=1.5, zorder=3)
        ax.plot(bx, bmed, "o", color=color, markersize=3.0,
                markeredgewidth=0, zorder=4)

    ax.set_xlim(xlo - xpad, xhi + xpad)
    ax.set_ylim(ylo - ypad, yhi + ypad)

    sign_char = "+" if rho >= 0 else "−"
    ax.text(0.96, 0.96, f"ρ = {sign_char}{abs(rho):.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.8, color="#222222", zorder=5)
    if seed_sd is not None:
        ax.text(0.96, 0.83, f"SD = {seed_sd:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6.2, color="#555555", zorder=5)

    clean_axes(ax, grid_axis="y")
    ax.tick_params(labelsize=6.2, length=2.5, pad=2)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune="both"))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune="both"))

    if show_xlabel:
        ax.set_xlabel(attr_label(attr), fontsize=7.2, labelpad=3)
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel(f"{strip_par(param)} {ylabel_suffix}", fontsize=7.2, labelpad=3)
    else:
        ax.set_ylabel("")


def add_gradient_legend(fig, groups, group_colors, ylabel_suffix="mean"):
    """Two-row legend identical to fig10/fig11."""
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    leg_dot = mlines.Line2D([], [], marker="o", color="none",
                            markerfacecolor="#BBBBBB", markeredgewidth=0,
                            markersize=5.5, label="Basins (n = 531)")
    leg_line = mlines.Line2D([], [], color="#555555", linewidth=1.5,
                             marker="o", markersize=3.5,
                             markerfacecolor="#555555", markeredgewidth=0,
                             label="Binned median")
    leg_band = mpatches.Patch(facecolor="#888888", alpha=0.28,
                              edgecolor="none", label="IQR (25–75%)")
    grp_handles = [
        mpatches.Patch(facecolor=group_colors[g], alpha=0.80,
                       edgecolor="none", label=g)
        for g in groups
    ]
    leg1 = fig.legend(
        handles=[leg_dot, leg_line, leg_band],
        loc="lower center", ncol=3,
        fontsize=7.6, frameon=False,
        bbox_to_anchor=(0.5, 0.055),
        handlelength=1.8, handletextpad=0.5, columnspacing=1.8,
    )
    fig.add_artist(leg1)
    fig.legend(
        handles=grp_handles,
        loc="lower center", ncol=5,
        fontsize=7.2, frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=1.2, handletextpad=0.4, columnspacing=1.2,
    )


def save_fig(fig, stem: str):
    import matplotlib.pyplot as plt
    APP_FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = APP_FIG_DIR / f"{stem}.png"
    pdf = APP_FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf
