from __future__ import annotations

import ast
import logging
import string
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


ROOT = Path("/workspace/autoresearch")
OUT_DIR = ROOT / "project" / "parameterize" / "manuscript" / "analysis" / "spatial_parameter_maps"
PARAMETER_TABLE = (
    ROOT
    / "project"
    / "parameterize"
    / "manuscript"
    / "analysis"
    / "figure2"
    / "data"
    / "parameter_estimates_by_run_long.csv"
)
CONUS_CLIPPED_DIR = ROOT / "data" / "camels_loc" / "conus_clipped"
BASIN_SHAPEFILE = CONUS_CLIPPED_DIR / "camels_671_loc_conus_clipped.shp"
STATE_SHAPEFILE = CONUS_CLIPPED_DIR / "s_18mr25_conus.shp"
BASIN_LIST = ROOT / "data" / "531sub_id.txt"

REFERENCE_MODEL = "distributional"
REFERENCE_LOSS = "HybridNseBatchLoss"
PARAMETER_ORDER = [
    "parBETA",
    "parFC",
    "parLP",
    "parPERC",
    "parUZL",
    "parK0",
    "parK1",
    "parK2",
    "route_a",
    "route_b",
    "parTT",
    "parCFMAX",
    "parCFR",
    "parCWH",
]
PARAMETER_LABELS = {
    "parBETA": "BETA",
    "parFC": "FC",
    "parLP": "LP",
    "parPERC": "PERC",
    "parUZL": "UZL",
    "parK0": "K0",
    "parK1": "K1",
    "parK2": "K2",
    "route_a": "route_a",
    "route_b": "route_b",
    "parTT": "TT",
    "parCFMAX": "CFMAX",
    "parCFR": "CFR",
    "parCWH": "CWH",
}

DPI = 600
MM = 1 / 25.4


def setup_style() -> None:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams.update(
        {
            "font.family": "Calibri",
            "font.sans-serif": ["Calibri", "Carlito", "Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": DPI,
        }
    )


def load_basin_ids() -> list[int]:
    text = BASIN_LIST.read_text(encoding="utf-8").strip()
    basin_ids = ast.literal_eval(text)
    if len(basin_ids) != 531 or len(set(basin_ids)) != 531:
        raise ValueError(f"Expected 531 unique basin IDs, found {len(basin_ids)} rows and {len(set(basin_ids))} unique IDs.")
    return [int(x) for x in basin_ids]


def load_parameter_means(basin_ids: list[int]) -> pd.DataFrame:
    usecols = ["model_raw", "loss", "seed", "basin_id", "parameter", "estimate_norm"]
    raw = pd.read_csv(PARAMETER_TABLE, usecols=usecols)
    selected = raw.loc[
        raw["model_raw"].eq(REFERENCE_MODEL)
        & raw["loss"].eq(REFERENCE_LOSS)
        & raw["basin_id"].isin(basin_ids)
        & raw["parameter"].isin(PARAMETER_ORDER)
    ].copy()
    expected_rows = len(basin_ids) * len(PARAMETER_ORDER) * selected["seed"].nunique()
    if len(selected) != expected_rows:
        raise ValueError(f"Unexpected parameter row count: found {len(selected)}, expected {expected_rows}.")
    if selected["seed"].nunique() < 1:
        raise ValueError("No seed-level distributional parameter estimates were found.")

    means = (
        selected.groupby(["basin_id", "parameter"], as_index=False)
        .agg(parameter_value=("estimate_norm", "mean"), n_seeds=("seed", "nunique"))
    )
    if means.shape[0] != len(basin_ids) * len(PARAMETER_ORDER):
        raise ValueError(f"Expected {len(basin_ids) * len(PARAMETER_ORDER)} basin-parameter means, found {means.shape[0]}.")
    if not means["parameter_value"].between(0.0, 1.0).all():
        bad = means.loc[~means["parameter_value"].between(0.0, 1.0)].head()
        raise ValueError(f"Normalized parameter values outside [0, 1]:\n{bad}")
    return means


def load_spatial_data(basin_ids: list[int]) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    basins = gpd.read_file(BASIN_SHAPEFILE)
    basins["basin_id"] = basins["gage_id"].astype(int)
    basins = basins.loc[basins["basin_id"].isin(basin_ids), ["basin_id", "lat", "lon", "geometry"]].copy()
    if basins.shape[0] != 531 or basins["basin_id"].nunique() != 531:
        raise ValueError(f"Expected 531 mapped basins, found {basins.shape[0]} rows and {basins['basin_id'].nunique()} unique IDs.")
    basins = basins.set_geometry(gpd.points_from_xy(basins["lon"], basins["lat"], crs="EPSG:4269"))

    conus = gpd.read_file(STATE_SHAPEFILE)
    if conus.empty:
        raise ValueError("CONUS state boundary selection is empty.")
    return basins, conus


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
) -> None:
    conus.boundary.plot(ax=ax, color="#D2D2D2", linewidth=0.34, zorder=1)
    data = points.loc[points["parameter"].eq(parameter)]
    value_min = float(data["parameter_value"].min())
    value_max = float(data["parameter_value"].max())
    if value_min == value_max:
        value_max = value_min + 1e-6
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
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        0.015,
        0.985,
        panel_label(panel_index),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        fontweight="bold",
        color="#111111",
    )
    ax.text(
        0.035,
        0.075,
        PARAMETER_LABELS[parameter],
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#111111",
    )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = ax.inset_axes([0.36, 0.075, 0.48, 0.045])
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([value_min, value_max])
    cbar.set_ticklabels([f"{value_min:.2f}", f"{value_max:.2f}"])
    cbar.outline.set_linewidth(0.45)
    cbar.outline.set_edgecolor("#777777")
    cbar.ax.tick_params(length=1.8, width=0.45, labelsize=5.8, pad=1.0, colors="#222222")


def make_figure() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    basin_ids = load_basin_ids()
    means = load_parameter_means(basin_ids)
    basins, conus = load_spatial_data(basin_ids)

    points = basins.merge(means, on="basin_id", how="inner")
    if points.shape[0] != 531 * 14:
        raise ValueError(f"Expected 7434 plotted basin-parameter rows, found {points.shape[0]}.")

    bounds = conus.total_bounds
    extent = (float(bounds[0] - 1.0), float(bounds[2] + 0.7), float(bounds[1] - 0.4), float(bounds[3] + 0.4))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("muted_teal", ["#F4F7F7", "#A8C6C0", "#2A9D8F"], N=256)

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(188 * MM, 116 * MM),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.018, right=0.992, bottom=0.018, top=0.992, wspace=0.055, hspace=-0.18)

    flat_axes = axes.ravel()
    for idx, parameter in enumerate(PARAMETER_ORDER):
        draw_map_panel(flat_axes[idx], conus, points, parameter, idx, cmap, extent)
    for ax in flat_axes[len(PARAMETER_ORDER) :]:
        ax.set_axis_off()

    png_path = OUT_DIR / "figure_parameter_spatial_maps.png"
    pdf_path = OUT_DIR / "figure_parameter_spatial_maps.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    write_notes(means, points)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def write_notes(means: pd.DataFrame, points: gpd.GeoDataFrame) -> None:
    seeds = sorted(means["n_seeds"].unique().tolist())
    lines = [
        "# Spatial parameter maps notes",
        "",
        "## Inputs",
        "",
        f"- Parameter table: `{PARAMETER_TABLE}`",
        f"- Basin shapefile: `{BASIN_SHAPEFILE}`",
        f"- State boundary shapefile: `{STATE_SHAPEFILE}`",
        f"- Basin list: `{BASIN_LIST}`",
        "",
        "## Extraction",
        "",
        f"- Formulation: `{REFERENCE_MODEL}`",
        f"- Loss: `{REFERENCE_LOSS}`",
        "- Parameter statistic: `estimate_norm`, the normalized distributional parameter mean from the Figure 2 analysis table.",
        f"- Seed aggregation: arithmetic mean across seeds; each basin-parameter mean used {seeds[0]} seed-level estimates.",
        "- Parameter uncertainty columns and stochastic samples were not used.",
        "",
        "## Checks",
        "",
        f"- Basins plotted: {points['basin_id'].nunique()} unique CAMELS-US basins.",
        f"- Basin-parameter rows plotted: {points.shape[0]}.",
        f"- Parameters plotted: {means['parameter'].nunique()}.",
        "- Layout: 4 rows x 4 columns with compressed row spacing; panels (a)-(n) are parameter maps and the final two panels are intentionally blank.",
        "- Color scale: each parameter panel has an independent colorbar using that parameter's normalized mean range.",
        "- Colormap: muted teal sequential scale consistent with the manuscript plotting style.",
        "- Font request: Calibri, with runtime fallback to Liberation Sans or DejaVu Sans if Calibri is not installed.",
        "",
        "## Parameter order",
        "",
    ]
    lines.extend([f"{idx}. `{parameter}`" for idx, parameter in enumerate(PARAMETER_ORDER, start=1)])
    (OUT_DIR / "figure_parameter_spatial_maps_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    make_figure()
