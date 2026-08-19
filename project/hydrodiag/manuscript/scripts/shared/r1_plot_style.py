"""
R1 Publication Style Module for HESS / Copernicus Manuscripts.
Provides fixed visual grammar, model color palettes, typography, and axes styling.
"""

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# 3.1 Categorical model palette (Tol vibrant + neutral grey for HBV)
# Canonical structure palette shared by every manuscript figure.
# TGD2 is the matched generic temperature-memory control. The legacy key
# ``TGD`` remains as a compatibility alias only; plotting code must not use it
# to relabel legacy inputs as TGD2.
MODEL_COLORS = {
    "Base": "#EE7733",  # warm orange; omitted-process baseline
    "TGD2": "#009988",  # teal; matched generic temperature-memory control
    "CN": "#0077BB",  # deep blue; explicit snow-process structure
    "HBV": "#6F6F6F",  # neutral grey; independent benchmark
}
MODEL_COLORS["TGD"] = MODEL_COLORS["TGD2"]  # legacy source compatibility
MODEL_LABELS = {
    "Base": "Base",
    "TGD2": "TGD2",
    "TGD": "TGD (legacy; not TGD2)",
    "CN": "CN",
    "HBV": "HBV benchmark",
}

# 3.2 Model markers for redundant encoding in dot/interval panels
MODEL_MARKERS = {
    "Base": "o",
    "TGD2": "^",
    "CN": "s",
    "HBV": "D",
}
MODEL_MARKERS["TGD"] = MODEL_MARKERS["TGD2"]

# 3.3 Train/test line encoding
PERIOD_STYLES = {
    "train": {"linestyle": "-", "linewidth": 1.65, "alpha": 0.92},
    "test": {"linestyle": (0, (4.0, 2.0)), "linewidth": 1.75, "alpha": 1.00},
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
