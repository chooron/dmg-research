#!/usr/bin/env python3
"""Prototype script for F6: PyGraphviz Topology Layout + Matplotlib Publication Rendering."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pygraphviz as pgv
from scipy.stats import gaussian_kde

# -------------------------------------------------------------
# Matplotlib Publication Configuration
# -------------------------------------------------------------
def configure_mpl():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "figure.titlesize": 9.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

# -------------------------------------------------------------
# Color Palettes
# -------------------------------------------------------------
# Specificity Colormap: Sequential Teal / Charcoal
# adv ~ 0 -> light neutral (#e2e8f0), adv ~ 1.5 -> deep dark teal (#042f2e)
def get_adv_color(adv: float, vmin: float = 0.0, vmax: float = 1.5):
    val = np.clip((adv - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = mpl.colormaps["YlGnBu"] # smooth yellow-green-teal-blue or teal
    # Custom high-contrast publication teal colormap
    cdict = [
        (0.0, "#e2e8f0"),  # 0.0: neutral light slate/gray
        (0.2, "#99f6e4"),  # 0.3: soft cyan-teal
        (0.5, "#14b8a6"),  # 0.75: vibrant teal
        (0.8, "#0f766e"),  # 1.2: deep sea teal
        (1.0, "#042f2e"),  # 1.5: dark charcoal teal
    ]
    # Linear interpolation
    positions = [p[0] for p in cdict]
    for i in range(len(positions) - 1):
        if positions[i] <= val <= positions[i+1]:
            t = (val - positions[i]) / (positions[i+1] - positions[i])
            c1 = mpl.colors.to_rgb(cdict[i][1])
            c2 = mpl.colors.to_rgb(cdict[i+1][1])
            rgb = tuple((1 - t) * a + t * b for a, b in zip(c1, c2))
            return rgb
    return mpl.colors.to_rgb(cdict[-1][1])

# Color constants
COLOR_STORE_BG = "#f8fafc"
COLOR_STORE_BORDER = "#334155"
COLOR_FLUX_WATER = "#475569"
COLOR_FLUX_ET = "#b45309"
COLOR_OUTLET = "#1e293b"
COLOR_STRICT = "#0f766e"
COLOR_STANDARD = "#64748b"
COLOR_CORAL = "#c2593f"
COLOR_CORAL_DARK = "#9c3b24"

print("Helper setup complete!")
