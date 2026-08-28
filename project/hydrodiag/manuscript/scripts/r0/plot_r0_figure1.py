"""
Plotting Script for Main-Text Figure 1 (R0 Analysis):
Spatial distribution of the 531 CAMELS-US catchments and prespecified snow-activity strata.

Panel layout (2-panel asymmetric horizontal layout):
  (a) Spatial distribution of the 531 CAMELS-US catchments
      Enlarged CONUS map showing gauge locations sorted by frac_snow ascending, colored by
      a layered sequential blue-teal colormap, with external bottom-aligned colorbar.
  (b) Snow-activity distribution and prespecified strata
      Continuous [0, 1] frac_snow histogram with vertical dashed strata boundaries,
      stratum S1-S5 in-chart labels at interval tops, a concise upper-right summary
      annotation box (ranges + sample counts), and an internal bottom rug plot.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
from shapely.geometry import Point

# Ensure shared plotting style is accessible
HERE = Path(__file__).resolve().parent
SHARED_DIR = HERE.parent / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from r1_plot_style import (
    COLOR_DARK_NEUTRAL,
    COLOR_SECONDARY_NEUTRAL,
    COLOR_ZERO_LINE,
    apply_clean_spines,
    setup_publication_style,
)


def load_canonical_data(project_root: Path) -> pd.DataFrame:
    """Load canonical 531-basin snow attributes and join with station coordinates."""
    snow_path = project_root / "manuscript" / "results" / "R1" / "r1_snow_attributes.csv"
    if not snow_path.exists():
        raise FileNotFoundError(f"Missing canonical snow attributes at {snow_path}")

    df_snow = pd.read_csv(snow_path)
    df_snow["basin_id"] = df_snow["basin_id"].astype(str).str.zfill(8)

    dbf_path = project_root.parents[1] / "data" / "camels_loc" / "camels_671_loc.dbf"
    if not dbf_path.exists():
        dbf_path = Path("/home/jingxin/code/dmg-research/data/camels_loc/camels_671_loc.dbf")

    if not dbf_path.exists():
        raise FileNotFoundError(f"Missing CAMELS location database at {dbf_path}")

    df_dbf = pyogrio.read_dataframe(dbf_path)
    df_dbf["basin_id"] = df_dbf["gage_id"].astype(str).str.zfill(8)

    df_merged = pd.merge(df_snow, df_dbf[["basin_id", "lat", "lon"]], on="basin_id", how="inner")

    # Integrity assertions
    if len(df_merged) != 531:
        raise ValueError(f"Expected 531 catchments, got {len(df_merged)}")

    expected_counts = {"S1": 165, "S2": 156, "S3": 121, "S4": 34, "S5": 55}
    actual_counts = df_merged["snow_stratum"].value_counts().to_dict()
    for stratum, expected_n in expected_counts.items():
        if actual_counts.get(stratum, 0) != expected_n:
            raise ValueError(
                f"Stratum {stratum} count mismatch: expected {expected_n}, got {actual_counts.get(stratum, 0)}"
            )

    # Sort ascending by frac_snow so high-snow catchments plot on top
    df_sorted = df_merged.sort_values("frac_snow", ascending=True).reset_index(drop=True)
    return df_sorted


def load_conus_boundaries(cache_dir: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load and reproject CONUS state boundaries and national outline."""
    geojson_path = cache_dir / "gis" / "us_states.geojson"
    if not geojson_path.exists():
        geojson_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
        urllib.request.urlretrieve(url, geojson_path)

    gdf = gpd.read_file(geojson_path)
    conus = gdf[~gdf["name"].isin(["Alaska", "Hawaii", "Puerto Rico"])].copy()
    conus_5070 = conus.to_crs(epsg=5070)
    national_5070 = conus_5070.dissolve()

    return conus_5070, national_5070


def plot_figure1(
    df: pd.DataFrame,
    conus_states: gpd.GeoDataFrame,
    conus_national: gpd.GeoDataFrame,
    out_file: Path,
) -> None:
    """Generate and export Main-Text Figure 1."""
    setup_publication_style()

    # Reproject station points to Albers Equal Area (EPSG:5070)
    geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    gdf_stations = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326").to_crs(epsg=5070)

    # Muted, layered sequential blue-teal colormap
    cmap_colors = ["#E4EDF2", "#BED8E5", "#8AB7CE", "#5393B3", "#2B6F95", "#144F72", "#0A334E"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("snow_slate_teal", cmap_colors, N=256)

    # Discrete stratum summary definitions
    strata_defs = [
        ("S1", 0.00, 0.05, 165, "[0.00, 0.05)"),
        ("S2", 0.05, 0.15, 156, "[0.05, 0.15)"),
        ("S3", 0.15, 0.30, 121, "[0.15, 0.30)"),
        ("S4", 0.30, 0.50, 34, "[0.30, 0.50)"),
        ("S5", 0.50, 1.00, 55, "[0.50, 1.00]"),
    ]

    # Create figure: 7.5 in x 3.3 in (190 mm x 84 mm, 600 DPI)
    fig = plt.figure(figsize=(7.5, 3.3), dpi=600)
    gs = gridspec.GridSpec(
        1,
        2,
        width_ratios=[1.38, 1.0],
        wspace=0.16,
        left=0.025,
        right=0.975,
        top=0.92,
        bottom=0.10,
    )

    # =========================================================================
    # Column 0: Panel a (Enlarged Map + Lowered Bottom Colorbar)
    # =========================================================================
    gs_a = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=gs[0],
        height_ratios=[0.87, 0.13],
        hspace=0.02,
    )

    ax_map = fig.add_subplot(gs_a[0])
    ax_map.set_aspect("equal")
    ax_map.axis("off")

    # Tight bounding box on CONUS to maximize map display size
    tot_bounds = conus_national.total_bounds  # [minx, miny, maxx, maxy]
    dx = tot_bounds[2] - tot_bounds[0]
    dy = tot_bounds[3] - tot_bounds[1]
    ax_map.set_xlim(tot_bounds[0] - 0.012 * dx, tot_bounds[2] + 0.012 * dx)
    ax_map.set_ylim(tot_bounds[1] - 0.012 * dy, tot_bounds[3] + 0.012 * dy)

    # State lines (very light, thin) and national boundary
    conus_states.plot(ax=ax_map, facecolor="#FFFFFF", edgecolor="#D8DEE4", linewidth=0.38, zorder=1)
    conus_national.plot(ax=ax_map, facecolor="none", edgecolor="#5F6A75", linewidth=0.68, zorder=2)

    # Scatter points (sorted ascending: dark points plotted on top)
    x_pts = [p.x for p in gdf_stations.geometry]
    y_pts = [p.y for p in gdf_stations.geometry]
    sc = ax_map.scatter(
        x_pts,
        y_pts,
        c=gdf_stations["frac_snow"],
        cmap=custom_cmap,
        vmin=0.0,
        vmax=1.0,
        s=17,
        edgecolors="#1A2836",
        linewidths=0.25,
        alpha=0.92,
        zorder=5,
    )

    # Unified Title
    ax_map.text(
        -0.005,
        1.02,
        "(a) Spatial distribution of the 531 CAMELS-US catchments",
        transform=ax_map.transAxes,
        fontsize=7.0,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=COLOR_DARK_NEUTRAL,
    )

    # Dedicated colorbar axis placed below the map
    ax_cbar = fig.add_subplot(gs_a[1])
    ax_cbar.axis("off")

    cax = ax_cbar.inset_axes([0.05, 0.12, 0.90, 0.36])
    cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.set_label(
        r"Fraction of precipitation falling as snow, $f_{\mathrm{snow}}$",
        fontsize=6.2,
        labelpad=2.5,
        color=COLOR_DARK_NEUTRAL,
    )
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    cbar.ax.tick_params(labelsize=5.6, length=2.0, pad=1.2, color=COLOR_DARK_NEUTRAL)
    cbar.outline.set_linewidth(0.38)

    # Strata boundary cut marks inside colorbar
    for b_val in [0.05, 0.15, 0.30, 0.50]:
        cbar.ax.axvline(b_val, color="#1A2836", linestyle=":", linewidth=0.60, zorder=10)

    # =========================================================================
    # Column 1: Panel b (Snow-activity distribution and prespecified strata)
    # =========================================================================
    ax_b = fig.add_subplot(gs[1])
    apply_clean_spines(ax_b)

    # Unified Title
    ax_b.text(
        -0.10,
        1.02,
        "(b) Snow-activity distribution and prespecified strata",
        transform=ax_b.transAxes,
        fontsize=7.0,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=COLOR_DARK_NEUTRAL,
    )

    bin_edges = np.linspace(0, 1.0, 36)
    counts, _ = np.histogram(df["frac_snow"], bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bw = bin_edges[1] - bin_edges[0]

    # Plot histogram bars
    for bc, cnt in zip(bin_centers, counts):
        col = custom_cmap(bc)
        ax_b.bar(
            bc,
            cnt,
            width=bw * 0.88,
            color=col,
            edgecolor="#1E2B37",
            linewidth=0.28,
            alpha=0.92,
            zorder=3,
        )

    # Vertical dashed lines at true strata boundaries
    y_max = max(counts) * 1.15
    ax_b.set_ylim(0, y_max)
    ax_b.set_xlim(-0.02, 1.02)

    for name, x0, x1, count, interval in strata_defs:
        if x0 > 0:
            ax_b.axvline(x0, color="#6C757D", linestyle="--", linewidth=0.50, zorder=2)
        # Stratum name label only at the top of each interval
        x_mid = 0.5 * (x0 + x1)
        # Shift S1 slightly left to center naturally over the near-zero peak and give more clearance to S2
        x_pos = x_mid - 0.007 if name == "S1" else x_mid
        ax_b.text(
            x_pos,
            y_max * 0.94,
            name,
            ha="center",
            va="top",
            fontsize=6.5,
            fontweight="bold",
            color="#0A334E",
            zorder=6,
        )

    # Subtle rug plot strictly inside plot area at bottom
    rug_h = y_max * 0.035
    ax_b.vlines(
        df["frac_snow"],
        0,
        rug_h,
        color="#64748B",
        linewidth=0.28,
        alpha=0.45,
        zorder=4,
    )

    # Axis labels and ticks
    ax_b.set_xlabel(
        r"Fraction of precipitation falling as snow, $f_{\mathrm{snow}}$",
        fontsize=6.2,
        color=COLOR_DARK_NEUTRAL,
    )
    ax_b.set_ylabel("Catchment count", fontsize=6.2, color=COLOR_DARK_NEUTRAL)
    ax_b.set_xticks([0.0, 0.05, 0.15, 0.30, 0.50, 1.0])
    ax_b.set_xticklabels(["0", ".05", ".15", ".30", ".50", "1.0"], fontsize=5.6)
    ax_b.tick_params(axis="both", which="both", labelsize=5.6, length=2.2, color=COLOR_DARK_NEUTRAL)

    # Unified concise legend / annotation box in upper right whitespace of panel b
    annot_text = (
        r"$\mathbf{S1}$: [0.00, 0.05), $n=165$"
        + "\n"
        + r"$\mathbf{S2}$: [0.05, 0.15), $n=156$"
        + "\n"
        + r"$\mathbf{S3}$: [0.15, 0.30), $n=121$"
        + "\n"
        + r"$\mathbf{S4}$: [0.30, 0.50), $n=34$"
        + "\n"
        + r"$\mathbf{S5}$: [0.50, 1.00], $n=55$"
    )
    ax_b.text(
        0.52,
        0.74,
        annot_text,
        transform=ax_b.transAxes,
        fontsize=5.6,
        ha="left",
        va="top",
        linespacing=1.35,
        color=COLOR_DARK_NEUTRAL,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#FFFFFF",
            edgecolor="#CBD5E1",
            linewidth=0.40,
            alpha=0.92,
        ),
        zorder=10,
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=600)
    print(f"Saved: {out_file}")
    plt.close(fig)


def main(out_dir: Path | None = None) -> Path:
    project_root = HERE.parents[2]
    cache_dir = project_root / "manuscript" / "cache"
    fig_dir = out_dir or (project_root / "manuscript" / "figures")
    out_file = fig_dir / "Figure1_final.png"

    print("Loading canonical CAMELS-US dataset and coordinates...")
    df = load_canonical_data(project_root)
    print(f"Loaded {len(df)} catchments (sorted by frac_snow).")

    print("Loading CONUS state boundaries...")
    conus_states, conus_national = load_conus_boundaries(cache_dir)

    print("Plotting Main-Text Figure 1...")
    plot_figure1(df, conus_states, conus_national, out_file)
    print("Done!")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Main-Text Figure 1.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for generated figure.")
    args = parser.parse_args()
    main(args.out_dir)
