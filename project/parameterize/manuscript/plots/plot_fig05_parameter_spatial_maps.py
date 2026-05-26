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


ROOT = Path("/workspace/autoresearch")
PARAM_ROOT = ROOT / "project" / "parameterize"
FIGURE5_ROOT = PARAM_ROOT / "manuscript" / "analysis" / "figure5"
REPORT_DIR = FIGURE5_ROOT / "reports"
DATA_DIR = FIGURE5_ROOT / "data"
MAIN_FIG_DIR = PARAM_ROOT / "manuscript" / "figures" / "main"
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

MODEL_ORDER = ["deterministic", "mc_dropout", "distributional"]
MODEL_LABELS = {
    "deterministic": "delta_base",
    "mc_dropout": "delta_mcd",
    "distributional": "delta_dist",
}
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
    "parTT",
    "parCFMAX",
    "parCFR",
    "parCWH",
    "route_a",
    "route_b",
]
PARAMETER_LABELS = {
    "parBETA": r"$\mathrm{BETA}$",
    "parFC": r"$\mathrm{FC}$",
    "parLP": r"$\mathrm{LP}$",
    "parPERC": r"$\mathrm{PERC}$",
    "parUZL": r"$\mathrm{UZL}$",
    "parK0": r"$\mathrm{K}_0$",
    "parK1": r"$\mathrm{K}_1$",
    "parK2": r"$\mathrm{K}_2$",
    "parTT": r"$\mathrm{TT}$",
    "parCFMAX": r"$\mathrm{CFMAX}$",
    "parCFR": r"$\mathrm{CFR}$",
    "parCWH": r"$\mathrm{CWH}$",
    "route_a": r"$\mathrm{UH}_a$",
    "route_b": r"$\mathrm{UH}_b$",
}
PARAMETER_BOUNDS = {
    "parBETA": (1.0, 6.0),
    "parFC": (50.0, 1000.0),
    "parLP": (0.2, 1.0),
    "parPERC": (0.0, 10.0),
    "parUZL": (0.0, 100.0),
    "parK0": (0.05, 0.9),
    "parK1": (0.01, 0.5),
    "parK2": (0.001, 0.2),
    "route_a": (0.0, 2.9),
    "route_b": (0.0, 6.5),
    "parTT": (-2.5, 2.5),
    "parCFMAX": (0.5, 10.0),
    "parCFR": (0.0, 0.1),
    "parCWH": (0.0, 0.2),
}

DPI = 600
MM = 1 / 25.4


def setup_style() -> None:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 10.5,
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


def load_parameter_means(basin_ids: list[int], model: str) -> pd.DataFrame:
    usecols = [
        "model_raw",
        "loss",
        "seed",
        "basin_id",
        "parameter",
        "estimate_physical",
        "parameter_lower_bound",
        "parameter_upper_bound",
    ]
    raw = pd.read_csv(PARAMETER_TABLE, usecols=usecols)
    selected = raw.loc[
        raw["model_raw"].eq(model)
        & raw["loss"].eq(REFERENCE_LOSS)
        & raw["basin_id"].isin(basin_ids)
        & raw["parameter"].isin(PARAMETER_ORDER)
    ].copy()
    expected_rows = len(basin_ids) * len(PARAMETER_ORDER) * selected["seed"].nunique()
    if len(selected) != expected_rows:
        raise ValueError(f"Unexpected parameter row count for {model}: found {len(selected)}, expected {expected_rows}.")
    if selected["seed"].nunique() < 1:
        raise ValueError(f"No seed-level parameter estimates were found for {model}.")

    means = (
        selected.groupby(["basin_id", "parameter"], as_index=False)
        .agg(
            parameter_value=("estimate_physical", "mean"),
            parameter_lower_bound=("parameter_lower_bound", "first"),
            parameter_upper_bound=("parameter_upper_bound", "first"),
            n_seeds=("seed", "nunique"),
        )
    )
    if means.shape[0] != len(basin_ids) * len(PARAMETER_ORDER):
        raise ValueError(f"Expected {len(basin_ids) * len(PARAMETER_ORDER)} basin-parameter means for {model}, found {means.shape[0]}.")
    for parameter, (lower, upper) in PARAMETER_BOUNDS.items():
        rows = means.loc[means["parameter"].eq(parameter)]
        if not rows["parameter_value"].between(lower, upper).all():
            bad = rows.loc[~rows["parameter_value"].between(lower, upper)].head()
            raise ValueError(f"Physical parameter values outside configured bounds for {parameter}:\n{bad}")
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
    value_min, value_max = PARAMETER_BOUNDS[parameter]
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

    # ax.text(
    #     0.965,
    #     0.105,
    #     panel_label(panel_index),
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="bottom",
    #     fontsize=8.4,
    #     fontweight="normal",
    #     color="#111111",
    # )
    ax.text(
        0.045,
        -0.112,
        PARAMETER_LABELS[parameter],
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10.5,
        color="#111111",
        clip_on=False,
    )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = ax.inset_axes([0.35, -0.145, 0.52, 0.052])
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([value_min, value_max])
    cbar.set_ticklabels([format_bound(value_min), format_bound(value_max)])
    cbar.outline.set_linewidth(0.45)
    cbar.outline.set_edgecolor("#777777")
    cbar.ax.tick_params(length=1.8, width=0.45, labelsize=9.0, pad=1.0, colors="#222222")
    cbar.ax.set_clip_on(False)


def format_bound(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.1f}"
    return f"{value:.3g}"


def make_model_figure(
    model: str,
    basin_ids: list[int],
    basins: gpd.GeoDataFrame,
    conus: gpd.GeoDataFrame,
    extent: tuple[float, float, float, float],
    cmap: mpl.colors.Colormap,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame, Path]:
    means = load_parameter_means(basin_ids, model)
    points = basins.merge(means, on="basin_id", how="inner")
    if points.shape[0] != 531 * 14:
        raise ValueError(f"Expected 7434 plotted basin-parameter rows for {model}, found {points.shape[0]}.")

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(210 * MM, 150 * MM),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.018, right=0.992, bottom=0.080, top=0.992, wspace=0.070, hspace=0.035)

    flat_axes = axes.ravel()
    for idx, parameter in enumerate(PARAMETER_ORDER):
        draw_map_panel(flat_axes[idx], conus, points, parameter, idx, cmap, extent)
    for ax in flat_axes[len(PARAMETER_ORDER) :]:
        ax.set_axis_off()

    png_path = MAIN_FIG_DIR / f"Fig05_parameter_spatial_maps_{model}.png"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    if model == "distributional":
        fig.savefig(MAIN_FIG_DIR / "Fig05_parameter_spatial_maps.png", dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return means, points, png_path


def make_figure() -> None:
    setup_style()
    MAIN_FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    basin_ids = load_basin_ids()
    basins, conus = load_spatial_data(basin_ids)
    bounds = conus.total_bounds
    extent = (float(bounds[0] - 0.7), float(bounds[2] + 0.45), float(bounds[1] - 0.25), float(bounds[3] + 0.25))
    cmap = get_colormap()

    outputs: list[Path] = []
    model_payloads: dict[str, tuple[pd.DataFrame, gpd.GeoDataFrame]] = {}
    for model in MODEL_ORDER:
        means, points, png_path = make_model_figure(model, basin_ids, basins, conus, extent, cmap)
        outputs.append(png_path)
        model_payloads[model] = (means, points)

    write_notes(model_payloads, outputs)
    for path in outputs:
        print(f"Wrote {path}")
    print(f"Wrote {MAIN_FIG_DIR / 'Fig05_parameter_spatial_maps.png'}")


def get_colormap() -> mpl.colors.Colormap:
    try:
        return sns.color_palette("mako", as_cmap=True)
    except Exception:
        return plt.get_cmap("viridis")


def write_notes(model_payloads: dict[str, tuple[pd.DataFrame, gpd.GeoDataFrame]], outputs: list[Path]) -> None:
    dist_means, dist_points = model_payloads["distributional"]
    seeds = sorted(dist_means["n_seeds"].unique().tolist())
    lines = [
        "# Fig05 parameter spatial maps notes",
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
        f"- Formulations: {', '.join(f'`{model}` ({MODEL_LABELS[model]})' for model in MODEL_ORDER)}",
        f"- Loss: `{REFERENCE_LOSS}`",
        "- Parameter statistic: `estimate_physical`, the physical-scale parameter mean from the Figure 2 analysis table.",
        f"- Seed aggregation: arithmetic mean across seeds; each basin-parameter mean used {seeds[0]} seed-level estimates.",
        "- Parameter uncertainty columns and stochastic samples were not used.",
        "",
        "## Checks",
        "",
        f"- Basins plotted per model: {dist_points['basin_id'].nunique()} unique CAMELS-US basins.",
        f"- Basin-parameter rows plotted per model: {dist_points.shape[0]}.",
        f"- Parameters plotted per model: {dist_means['parameter'].nunique()}.",
        "- Layout: 4 rows x 4 columns with compressed row spacing; panels (a)-(n) are parameter maps and the final two panels are intentionally blank.",
        "- Color scale: each parameter panel uses the physical HBV/routing search bounds listed below.",
        "- Parameter label placement: labels are offset left of the colorbar to reduce overlap.",
        "- Colormap: `mako` from seaborn when available; otherwise `viridis`.",
        "- Font request: Times New Roman for all text, including math labels.",
        "",
        "## Outputs",
        "",
        *[f"- `{path}`" for path in outputs],
        f"- `{MAIN_FIG_DIR / 'Fig05_parameter_spatial_maps.png'}` retained as the distributional compatibility filename.",
        "",
        "## Parameter order",
        "",
    ]
    lines.extend([f"{idx}. `{parameter}`" for idx, parameter in enumerate(PARAMETER_ORDER, start=1)])
    lines.extend(["", "## Parameter bounds", ""])
    lines.extend([f"- `{parameter}`: {PARAMETER_BOUNDS[parameter][0]} to {PARAMETER_BOUNDS[parameter][1]}" for parameter in PARAMETER_ORDER])
    (REPORT_DIR / "figure5_plot_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    make_figure()
