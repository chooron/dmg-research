"""
R1 Publication Style Module for HESS / Copernicus Manuscripts.
Provides fixed visual grammar, model color palettes, typography, and axes styling.
"""

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 3.1 Nature Research Figure Guide / Wong (2011) / Okabe-Ito Colorblind-Safe Palette
# Base = Vermillion / deep orange, TGD = Bluish green / teal, CN = Blue
COLOR_BASE = "#D55E00"           # Okabe-Ito vermillion / deep orange; omitted-process baseline
COLOR_TGD = "#009E73"            # Okabe-Ito bluish green / teal; matched generic control
COLOR_CN = "#0072B2"             # Okabe-Ito blue; explicit snow-process structure
COLOR_OBSERVATION = "#303438"    # dark neutral; observation / truth / reference
COLOR_DARK_NEUTRAL = "#303438"   # text, primary ticks, axis spines
COLOR_SECONDARY_NEUTRAL = "#C8CDD1" # light neutral reference, bounds
COLOR_ZERO_LINE = "#70767B"      # mid grey; zero lines, reference guides
COLOR_LIGHT_REF = "#C8CDD1"      # grid and secondary bounds
COLOR_TOLERANCE_BAND = "#B3B3B3" # tolerance shading (±15 d band, low alpha)

MODEL_COLORS = {
    "Base": COLOR_BASE,
    "TGD2": COLOR_TGD,
    "TGD": COLOR_TGD,
    "CN": COLOR_CN,
    "HBV": "#6F6F6F",  # neutral grey legacy benchmark
    "XAJ-Base": COLOR_BASE,
    "XAJ-TGD": COLOR_TGD,
    "XAJ-TGD2": COLOR_TGD,
    "XAJ-CN": COLOR_CN,
}

MODEL_LABELS = {
    "Base": "Base",
    "TGD2": "TGD",
    "TGD": "TGD",
    "CN": "CN",
    "HBV": "HBV benchmark",
    "XAJ-Base": "Base",
    "XAJ-TGD": "TGD",
    "XAJ-CN": "CN",
}

# 3.2 Model markers for redundant encoding in dot/interval panels
MODEL_MARKERS = {
    "Base": "o",
    "TGD2": "^",
    "TGD": "^",
    "CN": "s",
    "HBV": "D",
    "XAJ-Base": "o",
    "XAJ-TGD": "^",
    "XAJ-CN": "s",
}

# 3.2b Cross-host model palette (Tol vibrant; used by R5 Figure 9 only)
HOST_COLORS = {
    "XAJ": COLOR_CN,
    "GR4J": COLOR_TGD,
    "SIMHYD": "#78A79F",
}
HOST_LABELS = {
    "XAJ": "XAJ",
    "GR4J": "GR4J",
    "SIMHYD": "SIMHYD",
}
HOST_MARKERS = {
    "XAJ": "o",
    "GR4J": "s",
    "SIMHYD": "^",
}

# 3.2c Sequential snow strata palette (lightness progression)
SNOW_STRATA_PALETTE = {
    "S1": "#E8F0F4",
    "S2": "#D2E2EB",
    "S3": "#B7D1E0",
    "S4": "#91B7CD",
    "S5": "#6597B6",
}

# 3.2d Seven-level diverging palette (zero-centered variables)
DIVERGING_PALETTE_7 = [
    "#B87555",  # negative (orange)
    "#D19A7C",
    "#E7C9B9",
    "#F2F1EE",  # zero (warm light grey)
    "#C8DAE6",
    "#91B5CD",
    "#5E8DB0",  # positive (blue)
]

# 3.3 Train/test and IC/dPL line encodings
PERIOD_STYLES = {
    "train": {"linestyle": "-", "linewidth": 1.4, "alpha": 0.85},
    "test": {"linestyle": (0, (4.0, 2.0)), "linewidth": 1.6, "alpha": 1.00},
}

PARADIGM_STYLES = {
    "IC": {"linestyle": "-", "linewidth": 1.4},
    "IC-CMA-ES": {"linestyle": "-", "linewidth": 1.4},
    "dPL": {"linestyle": (0, (4.0, 2.0)), "linewidth": 1.5},
    "dPL-MLP": {"linestyle": (0, (4.0, 2.0)), "linewidth": 1.5},
}

# 3.4 Future heatmap and GIS colormaps (cmcrameri map names)
SIGNED_EFFECT_CMAP = "vik_r"
NONNEGATIVE_CMAP = "batlow"

# Preferred sans-serif font family hierarchy
FONT_PREFERENCE = ["Arial", "Liberation Sans", "DejaVu Sans"]


def resolve_font_family():
    """Identify the actual resolved sans-serif font family installed on the system."""
    system_fonts = {f.name for f in fm.fontManager.ttflist}
    for font in FONT_PREFERENCE:
        if font in system_fonts:
            return font
    return "sans-serif"


RESOLVED_FONT = resolve_font_family()


def setup_publication_style():
    """Apply global matplotlib rcParams following HESS/Copernicus guidelines."""
    font_family = resolve_font_family()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_family],
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "figure.titlesize": 10.0,
            # Line & patch width
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "patch.linewidth": 0.8,
            # Ticks
            "xtick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.major.size": 3.5,
            "ytick.major.width": 0.8,
            "ytick.direction": "out",
            # Saving
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def apply_clean_spines(ax):
    """Apply white background, remove top/right spines, and set 0.8 pt spine width."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_facecolor("#FFFFFF")
