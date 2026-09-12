"""GIS helpers following the hydrodiag CONUS plotting convention."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point

from r1_config import CACHE_DIR

STATE_CACHE = CACHE_DIR / "gis" / "us_states.geojson"
MAP_CRS = "EPSG:5070"


def load_conus_boundaries() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if not STATE_CACHE.is_file():
        raise RuntimeError(f"missing cached state boundary: {STATE_CACHE}")
    states = gpd.read_file(STATE_CACHE)
    conus = states[~states["name"].isin(["Alaska", "Hawaii", "Puerto Rico"])].copy().to_crs(MAP_CRS)
    national = conus.dissolve()
    return conus, national


def basin_point_geodataframe(data: pd.DataFrame) -> gpd.GeoDataFrame:
    geometry = [Point(xy) for xy in zip(data["longitude"], data["latitude"])]
    return gpd.GeoDataFrame(data.copy(), geometry=geometry, crs="EPSG:4326").to_crs(MAP_CRS)


def tendency_cmap() -> tuple[mcolors.Colormap, mcolors.TwoSlopeNorm]:
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "ic_dpl_tendency", ["#2166AC", "#F7F7F7", "#D6604D"], N=256,
    )
    return cmap, mcolors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)


def draw_tendency_map(
    ax: plt.Axes,
    data: pd.DataFrame,
    states: gpd.GeoDataFrame,
    national: gpd.GeoDataFrame,
    *,
    add_histogram: bool = True,
    title: str = "Basin-level cross-model estimator tendency",
    panel_label: str | None = None,
) -> plt.cm.ScalarMappable:
    points = basin_point_geodataframe(data)
    cmap, norm = tendency_cmap()
    ax.set_aspect("equal")
    ax.axis("off")
    bounds = national.total_bounds
    dx = bounds[2] - bounds[0]; dy = bounds[3] - bounds[1]
    ax.set_xlim(bounds[0] - 0.015 * dx, bounds[2] + 0.015 * dx)
    ax.set_ylim(bounds[1] - 0.015 * dy, bounds[3] + 0.015 * dy)
    states.plot(ax=ax, facecolor="#FFFFFF", edgecolor="#D8DEE4", linewidth=0.35, zorder=1)
    national.plot(ax=ax, facecolor="none", edgecolor="#5F6A75", linewidth=0.65, zorder=2)
    points.plot(ax=ax, column="P_b_dPL", cmap=cmap, norm=norm, markersize=12, edgecolor="white", linewidth=0.22, zorder=3)
    ax.set_title(title, pad=7)
    if panel_label:
        ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
    if add_histogram:
        inset = ax.inset_axes([0.69, 0.05, 0.27, 0.26])
        inset.hist(data["P_b_dPL"], bins=11, range=(0, 1), color="#9E9E9E", edgecolor="white", linewidth=0.25)
        inset.axvline(0.5, color="#4D4D4D", ls="--", lw=0.6)
        inset.set_xlim(0, 1); inset.set_xlabel(r"$P_b^{dPL}$", fontsize=6); inset.set_ylabel("Basins", fontsize=6)
        inset.tick_params(labelsize=5, length=2)
        inset.set_title("Population", fontsize=6, pad=2)
        for spine in inset.spines.values(): spine.set_linewidth(0.45)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)
