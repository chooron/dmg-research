"""
Fig. S6a — Parameter-mean spatial maps for delta_base
Fig. S6b — Parameter-mean spatial maps for delta_mcd
Fig. S6c — Parameter-std (seed spread) spatial maps for delta_mcd

Directly cloned from plot_fig05_parameter_spatial_maps.py.
Same layout (4×4), same colormap (mako), same draw_map_panel logic.
S6c uses seed SD across seeds as the "spread" metric for delta_mcd.
"""
from __future__ import annotations

import ast
import logging
import string
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# ---------------------------------------------------------------------------
# Paths — identical to fig05
# ---------------------------------------------------------------------------
ROOT = Path("/workspace/autoresearch")
PARAM_ROOT = ROOT / "project" / "parameterize"
PARAMETER_TABLE = (
    PARAM_ROOT / "manuscript" / "analysis" / "figure2" / "data"
    / "parameter_estimates_by_run_long.csv"
)
CONUS_CLIPPED_DIR = ROOT / "data" / "camels_loc" / "conus_clipped"
BASIN_SHAPEFILE = CONUS_CLIPPED_DIR / "camels_671_loc_conus_clipped.shp"
STATE_SHAPEFILE = CONUS_CLIPPED_DIR / "s_18mr25_conus.shp"
BASIN_LIST = ROOT / "data" / "531sub_id.txt"
APP_FIG_DIR = PARAM_ROOT / "manuscript" / "figures" / "appendix"

REFERENCE_LOSS = "HybridNseBatchLoss"

# ---------------------------------------------------------------------------
# Constants — identical to fig05
# ---------------------------------------------------------------------------
PARAMETER_ORDER = [
    "parBETA", "parFC", "parLP", "parPERC", "parUZL",
    "parK0", "parK1", "parK2", "route_a", "route_b",
    "parTT", "parCFMAX", "parCFR", "parCWH",
]
PARAMETER_LABELS = {
    "parBETA": "BETA", "parFC": "FC", "parLP": "LP",
    "parPERC": "PERC", "parUZL": "UZL", "parK0": "K0",
    "parK1": "K1", "parK2": "K2", "route_a": "route_a",
    "route_b": "route_b", "parTT": "TT", "parCFMAX": "CFMAX",
    "parCFR": "CFR", "parCWH": "CWH",
}
PARAMETER_BOUNDS = {
    "parBETA": (1.0, 6.0), "parFC": (50.0, 1000.0), "parLP": (0.2, 1.0),
    "parPERC": (0.0, 10.0), "parUZL": (0.0, 100.0),
    "parK0": (0.05, 0.9), "parK1": (0.01, 0.5), "parK2": (0.001, 0.2),
    "route_a": (0.0, 2.9), "route_b": (0.0, 6.5),
    "parTT": (-2.5, 2.5), "parCFMAX": (0.5, 10.0),
    "parCFR": (0.0, 0.1), "parCWH": (0.0, 0.2),
}
# Std bounds: 10 % of physical range as upper bound (reasonable for seed spread)
PARAMETER_STD_BOUNDS = {
    p: (0.0, (PARAMETER_BOUNDS[p][1] - PARAMETER_BOUNDS[p][0]) * 0.15)
    for p in PARAMETER_ORDER
}

DPI = 600
MM = 1 / 25.4


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def setup_style() -> None:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.size": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": DPI,
    })


def get_colormap() -> mpl.colors.Colormap:
    try:
        return sns.color_palette("mako", as_cmap=True)
    except Exception:
        return plt.get_cmap("viridis")


def get_std_colormap() -> mpl.colors.Colormap:
    """Sequential colormap for std maps — use mako_r so low=light, high=dark."""
    try:
        return sns.color_palette("rocket", as_cmap=True)
    except Exception:
        return plt.get_cmap("YlOrRd")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_basin_ids() -> list[int]:
    text = BASIN_LIST.read_text(encoding="utf-8").strip()
    ids = ast.literal_eval(text)
    return [int(x) for x in ids]


def load_spatial_data(basin_ids: list[int]) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    basins = gpd.read_file(BASIN_SHAPEFILE)
    basins["basin_id"] = basins["gage_id"].astype(int)
    basins = basins.loc[basins["basin_id"].isin(basin_ids),
                        ["basin_id", "lat", "lon", "geometry"]].copy()
    basins = basins.set_geometry(
        gpd.points_from_xy(basins["lon"], basins["lat"], crs="EPSG:4269")
    )
    conus = gpd.read_file(STATE_SHAPEFILE)
    return basins, conus


def load_parameter_means(basin_ids: list[int], model: str) -> pd.DataFrame:
    """Mean of estimate_physical across seeds — identical to fig05."""
    usecols = ["model_raw", "loss", "seed", "basin_id", "parameter",
               "estimate_physical"]
    raw = pd.read_csv(PARAMETER_TABLE, usecols=usecols)
    sub = raw.loc[
        raw["model_raw"].eq(model)
        & raw["loss"].eq(REFERENCE_LOSS)
        & raw["basin_id"].isin(basin_ids)
        & raw["parameter"].isin(PARAMETER_ORDER)
    ].copy()
    means = (sub.groupby(["basin_id", "parameter"], as_index=False)
             .agg(parameter_value=("estimate_physical", "mean")))
    return means


def load_parameter_stds(basin_ids: list[int], model: str) -> pd.DataFrame:
    """Seed SD of estimate_physical — used for the spread map."""
    usecols = ["model_raw", "loss", "seed", "basin_id", "parameter",
               "estimate_physical"]
    raw = pd.read_csv(PARAMETER_TABLE, usecols=usecols)
    sub = raw.loc[
        raw["model_raw"].eq(model)
        & raw["loss"].eq(REFERENCE_LOSS)
        & raw["basin_id"].isin(basin_ids)
        & raw["parameter"].isin(PARAMETER_ORDER)
    ].copy()
    stds = (sub.groupby(["basin_id", "parameter"], as_index=False)
            .agg(parameter_value=("estimate_physical", "std")))
    stds["parameter_value"] = stds["parameter_value"].fillna(0.0)
    return stds


# ---------------------------------------------------------------------------
# Drawing — identical to fig05's draw_map_panel
# ---------------------------------------------------------------------------
def format_bound(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.1f}"
    return f"{value:.3g}"


def panel_label(index: int) -> str:
    return f"({string.ascii_lowercase[index]})"


def draw_map_panel(
    ax: plt.Axes,
    conus: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    parameter: str,
    panel_index: int,
    cmap: mpl.colors.Colormap,
    extent: tuple[float, float, float, float],
    bounds: tuple[float, float],
) -> None:
    """Exact copy of fig05's draw_map_panel, with configurable bounds."""
    conus.boundary.plot(ax=ax, color="#D2D2D2", linewidth=0.34, zorder=1)
    data = points.loc[points["parameter"].eq(parameter)]
    value_min, value_max = bounds
    norm = Normalize(vmin=value_min, vmax=value_max)
    data.plot(
        ax=ax,
        column="parameter_value",
        cmap=cmap,
        norm=norm,
        markersize=7.2,
        marker="o",
        linewidth=0.12,
        edgecolor="#F7F7F7",
        alpha=0.96,
        zorder=3,
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1.2, adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Panel label — bottom-right, identical to fig05
    ax.text(0.965, 0.035, panel_label(panel_index),
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8.4, fontweight="bold", color="#111111")
    # Parameter label — bottom-left below colorbar, identical to fig05
    ax.text(0.045, -0.112, PARAMETER_LABELS[parameter],
            transform=ax.transAxes, ha="left", va="center",
            fontsize=8.2, color="#111111", clip_on=False)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = ax.inset_axes([0.32, -0.135, 0.50, 0.045])
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([value_min, value_max])
    cbar.set_ticklabels([format_bound(value_min), format_bound(value_max)])
    cbar.outline.set_linewidth(0.45)
    cbar.outline.set_edgecolor("#777777")
    cbar.ax.tick_params(length=1.8, width=0.45, labelsize=5.8,
                        pad=1.0, colors="#222222")
    cbar.ax.set_clip_on(False)


# ---------------------------------------------------------------------------
# Figure builder — identical layout to fig05's make_model_figure
# ---------------------------------------------------------------------------
def make_figure(
    stem: str,
    points: gpd.GeoDataFrame,
    conus: gpd.GeoDataFrame,
    extent: tuple[float, float, float, float],
    cmap: mpl.colors.Colormap,
    bounds_dict: dict[str, tuple[float, float]],
    title: str,
    figure_label: str,
) -> None:
    fig, axes = plt.subplots(
        4, 4,
        figsize=(188 * MM, 124 * MM),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.018, right=0.992, bottom=0.052, top=0.992,
                        wspace=0.055, hspace=-0.08)
    fig.text(0.006, 0.982, figure_label, ha="left", va="top",
             fontsize=10.8, fontweight="normal", color="#111111")

    flat_axes = axes.ravel()
    for idx, parameter in enumerate(PARAMETER_ORDER):
        draw_map_panel(flat_axes[idx], conus, points, parameter, idx,
                       cmap, extent, bounds_dict[parameter])
    for ax in flat_axes[len(PARAMETER_ORDER):]:
        ax.set_axis_off()

    APP_FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = APP_FIG_DIR / f"{stem}.png"
    pdf = APP_FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    setup_style()
    basin_ids = load_basin_ids()
    basins, conus = load_spatial_data(basin_ids)
    raw_bounds = conus.total_bounds
    extent = (
        float(raw_bounds[0] - 0.7), float(raw_bounds[2] + 0.45),
        float(raw_bounds[1] - 0.25), float(raw_bounds[3] + 0.25),
    )
    cmap_mean = get_colormap()
    cmap_std  = get_std_colormap()

    # ── S6a: delta_base parameter means ──────────────────────────────────────
    means_base = load_parameter_means(basin_ids, "deterministic")
    pts_base   = basins.merge(means_base, on="basin_id", how="inner")
    make_figure("figS6a_spatial_maps_base_mean", pts_base, conus, extent,
                cmap_mean, PARAMETER_BOUNDS,
                r"$\delta_{base}$ — parameter means", "(1)")

    # ── S6b: delta_mcd parameter means ───────────────────────────────────────
    means_mcd = load_parameter_means(basin_ids, "mc_dropout")
    pts_mcd   = basins.merge(means_mcd, on="basin_id", how="inner")
    make_figure("figS6b_spatial_maps_mcd_mean", pts_mcd, conus, extent,
                cmap_mean, PARAMETER_BOUNDS,
                r"$\delta_{mcd}$ — parameter means", "(2)")

    # ── S6c: delta_mcd parameter std (seed spread) ───────────────────────────
    stds_mcd  = load_parameter_stds(basin_ids, "mc_dropout")
    pts_mcd_s = basins.merge(stds_mcd, on="basin_id", how="inner")
    make_figure("figS6c_spatial_maps_mcd_std", pts_mcd_s, conus, extent,
                cmap_std, PARAMETER_STD_BOUNDS,
                r"$\delta_{mcd}$ — parameter spread (seed SD)", "(3)")


if __name__ == "__main__":
    main()
