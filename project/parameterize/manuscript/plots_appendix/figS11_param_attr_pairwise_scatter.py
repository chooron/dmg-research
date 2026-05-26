"""
Fig S11a–e: Lower-triangle pairwise scatter plots of basin attributes,
coloured by parameter value, for the 5 parameters with the strongest
attribute correlations.

  S11a  parCWH   – frac_snow      (ρ = -0.90)
  S11b  parPERC  – aridity        (ρ = -0.59)
  S11c  parBETA  – slope_mean     (ρ = -0.58)
  S11d  parUZL   – soil_cond.     (ρ =  0.57)
  S11e  parFC    – pet_mean       (ρ =  0.51)

Each panel is a lower-triangle scatter matrix of the attributes most
correlated with that parameter (top-5 by |ρ|).  Scatter point colour
encodes the normalised parameter value using the 'mako' colormap.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec

from common import PARAM_LABELS, setup_style

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/workspace/autoresearch")
PARAM_ROOT = ROOT / "project" / "parameterize"
MANUSCRIPT_ROOT = PARAM_ROOT / "manuscript"

MEAN_MAP_FILE = (
    MANUSCRIPT_ROOT
    / "analysis"
    / "03_distributional_parameter_spatial_data"
    / "data"
    / "distributional_parameter_mean_maps_long.csv"
)
BASIN_ATTR_FILE = (
    PARAM_ROOT
    / "outputs"
    / "analysis"
    / "stability_stats"
    / "tables"
    / "basin_attributes.csv"
)
CORR_FILE = (
    MANUSCRIPT_ROOT
    / "analysis"
    / "04_mean_attribute_relationships"
    / "data"
    / "distributional_mean_attribute_correlations.csv"
)
APP_FIG_DIR = MANUSCRIPT_ROOT / "figures" / "appendix"

DPI = 600
MM = 1 / 25.4

# ---------------------------------------------------------------------------
# Panel definitions: (parameter, primary_attribute, panel_label)
# ---------------------------------------------------------------------------
PANELS = [
    ("parCWH", "frac_snow", "S11a"),
    ("parPERC", "aridity", "S11b"),
    ("parBETA", "slope_mean", "S11c"),
    ("parUZL", "soil_conductivity", "S11d"),
    ("parFC", "pet_mean", "S11e"),
]

# Number of attributes to include in each scatter matrix (top-N by |ρ|)
N_ATTRS = 4

ATTR_LABELS = {
    "aridity": "Aridity",
    "frac_snow": "Snow fraction",
    "slope_mean": "Mean slope",
    "pet_mean": "PET",
    "p_mean": "Precip.",
    "p_seasonality": "Precip. seasonality",
    "elev_mean": "Elevation",
    "soil_conductivity": "Soil cond.",
    "soil_depth_pelletier": "Soil depth",
    "clay_frac": "Clay frac.",
    "sand_frac": "Sand frac.",
    "low_prec_freq": "Low-prec. freq.",
    "low_prec_dur": "Low-prec. dur.",
    "high_prec_freq": "High-prec. freq.",
    "high_prec_dur": "High-prec. dur.",
    "max_water_content": "Max water cont.",
    "frac_forest": "Forest frac.",
    "area_gages2": "Basin area",
}


def _attr_label(col: str) -> str:
    return ATTR_LABELS.get(col, col.replace("_", " "))


def _panel_letter(panel_label: str) -> str:
    return panel_label.replace("S11", "").lower()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mean_df = pd.read_csv(MEAN_MAP_FILE)
    attrs = pd.read_csv(BASIN_ATTR_FILE)
    corr = pd.read_csv(CORR_FILE)
    return mean_df, attrs, corr


def build_merged(
    mean_df: pd.DataFrame,
    attrs: pd.DataFrame,
    parameter: str,
) -> pd.DataFrame:
    param_df = mean_df[mean_df["parameter"] == parameter][
        ["basin_id", "seed_mean"]
    ].rename(columns={"seed_mean": "param_value"})
    merged = param_df.merge(attrs, on="basin_id", how="inner")
    return merged


def top_attrs(corr: pd.DataFrame, parameter: str, n: int) -> list[str]:
    sub = corr[corr["parameter"] == parameter].nlargest(n, "abs_rho")
    return sub["attribute"].tolist()


# ---------------------------------------------------------------------------
# Single-panel figure
# ---------------------------------------------------------------------------
def make_panel(
    merged: pd.DataFrame,
    attrs_list: list[str],
    parameter: str,
    panel_label: str,
    primary_attr: str,
) -> None:
    n = len(attrs_list)
    param_vals = merged["param_value"].values
    vmin, vmax = (
        np.nanpercentile(param_vals, 2),
        np.nanpercentile(param_vals, 98),
    )
    norm = Normalize(vmin=vmin, vmax=vmax)
    # mako palette (seaborn) reconstructed from its key hex stops
    cmap = LinearSegmentedColormap.from_list(
        "mako",
        ["#0B0405", "#382A54", "#395D9C", "#29A39E", "#7CCBA2", "#DEF5E5"],
        N=256,
    )

    fig_w = 140 * MM
    fig_h = 130 * MM
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Grid: n×n scatter cells + right colorbar column
    gs = GridSpec(
        n,
        n + 1,
        figure=fig,
        width_ratios=[1] * n + [0.06],
        hspace=0.10,
        wspace=0.10,
        left=0.12,
        right=0.91,
        top=0.96,
        bottom=0.12,
    )

    scatter_axes: dict[tuple[int, int], mpl.axes.Axes] = {}

    for row in range(n):
        for col in range(n):
            ax = fig.add_subplot(gs[row, col])
            scatter_axes[(row, col)] = ax

            if col >= row:
                # upper triangle and diagonal — hide
                ax.set_visible(False)
                continue

            x_col = attrs_list[col]
            y_col = attrs_list[row]
            x = merged[x_col].values
            y = merged[y_col].values
            c = param_vals

            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
            sc = ax.scatter(
                x[mask],
                y[mask],
                c=c[mask],
                cmap=cmap,
                norm=norm,
                s=4,
                alpha=0.65,
                linewidths=0,
                rasterized=True,
            )

            ax.tick_params(labelsize=6.5, length=2, pad=2)

            # x-axis labels only on bottom row
            if row == n - 1:
                ax.set_xlabel(_attr_label(x_col), fontsize=7.5, labelpad=3)
            else:
                ax.set_xticklabels([])

            # y-axis labels only on left column
            if col == 0:
                ax.set_ylabel(_attr_label(y_col), fontsize=7.5, labelpad=3)
            else:
                ax.set_yticklabels([])

    label_ax = scatter_axes.get((1, 0))
    if label_ax is not None:
        label_ax.text(
            -0.48,
            1.18,
            f"({_panel_letter(panel_label)})",
            transform=label_ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12.0,
            color="#111111",
            clip_on=False,
        )

    # Colorbar: span only the active scatter rows (1 → n-1) so it matches plot height
    cax = fig.add_subplot(gs[1:, n])
    pos = cax.get_position()
    cax.set_position(
        [
            pos.x0 - 0.17,
            pos.y0,
            pos.width,
            pos.height,
        ]
    )
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    param_tex = PARAM_LABELS.get(parameter, parameter.replace("par", ""))
    cbar.set_label(f"{param_tex} value", fontsize=8.0, labelpad=4)
    cbar.ax.tick_params(labelsize=7.0, length=2)

    APP_FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"fig{panel_label}_param_attr_scatter"
    fig.savefig(APP_FIG_DIR / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {stem}.png / .pdf")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    setup_style()
    mean_df, attrs, corr = load_data()

    for parameter, primary_attr, panel_label in PANELS:
        attrs_list = top_attrs(corr, parameter, N_ATTRS)
        # Ensure primary attribute is first
        if primary_attr in attrs_list:
            attrs_list.remove(primary_attr)
        attrs_list = [primary_attr] + attrs_list[: N_ATTRS - 1]

        merged = build_merged(mean_df, attrs, parameter)
        print(f"[{panel_label}] {parameter} | attrs: {attrs_list}")
        make_panel(merged, attrs_list, parameter, panel_label, primary_attr)

    print("Done.")


if __name__ == "__main__":
    main()
