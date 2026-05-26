from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from shapely.geometry import box


ROOT = Path("/workspace/autoresearch")
PARAM_ROOT = ROOT / "project" / "parameterize"
OUTPUT_ROOT = PARAM_ROOT / "outputs"
STABILITY_TABLE_DIR = OUTPUT_ROOT / "analysis" / "stability_stats" / "tables"
FIG4_ROOT = PARAM_ROOT / "manuscript" / "analysis" / "figure4"
DATA_DIR = FIG4_ROOT / "data"
REPORT_DIR = FIG4_ROOT / "reports"
FIG_DIR = PARAM_ROOT / "manuscript" / "figures" / "main"
PLOTS_DIR = PARAM_ROOT / "manuscript" / "plots"

OUT_STEM = FIG_DIR / "Fig04_spatial_robustness_revised"
REPORT_PATH = REPORT_DIR / "Fig04_spatial_robustness_revised_report.txt"
PANEL_D_VALUES_PATH = DATA_DIR / "figure4_panel_d_parameter_values_revised.csv"
PANEL_DE_HEATMAP_PATH = DATA_DIR / "figure4_panel_de_heatmap_summary_revised.csv"

CAMELS_REGION_SHP = ROOT / "data" / "camels_loc" / "s_18mr25.shp"
CAMELS_LOC_SHP = ROOT / "data" / "camels_loc" / "camels_671_loc.shp"
MAP_CACHE_DIR = DATA_DIR / "map_cache"
CLIPPED_REGION_SHP = MAP_CACHE_DIR / "camels_regions_clipped_to_531.shp"

MODEL_ORDER = ["deterministic", "mc_dropout", "distributional"]
MODEL_LABELS = {
    "deterministic": "delta_base",
    "mc_dropout": "delta_mcd",
    "distributional": "delta_dist",
}
MODEL_MATH_LABELS = {
    "deterministic": r"$\delta_{\mathrm{base}}$",
    "mc_dropout": r"$\delta_{\mathrm{mcd}}$",
    "distributional": r"$\delta_{\mathrm{dist}}$",
}
MODEL_COLORS = {
    "deterministic": "#2F6DB5",
    "mc_dropout": "#E6862E",
    "distributional": "#2A9D8F",
}

GROUP_NAMES = {
    "G1": "Humid-steep",
    "G2": "Low-snow humid-lowland",
    "G3": "Arid-lowland",
    "G4": "Arid-seasonal",
    "G5": "Low-snow arid-steep",
    "G6": "Snow arid-steep",
    "G7": "Snow humid-steep",
}
GROUP_COLORS = {
    "G1": "#4E79A7",
    "G2": "#59A14F",
    "G3": "#F28E2B",
    "G4": "#E15759",
    "G5": "#76B7B2",
    "G6": "#B07AA1",
    "G7": "#9C755F",
}

PARAM_ORDER = [
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
PARAM_DISPLAY = {
    "parBETA": "BETA",
    "parFC": "FC",
    "parLP": "LP",
    "parPERC": "PERC",
    "parUZL": "UZL",
    "parK0": r"$\mathrm{K}_0$",
    "parK1": r"$\mathrm{K}_1$",
    "parK2": r"$\mathrm{K}_2$",
    "parTT": "TT",
    "parCFMAX": "CFMAX",
    "parCFR": "CFR",
    "parCWH": "CWH",
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
    "parTT": (-2.5, 2.5),
    "parCFMAX": (0.5, 10.0),
    "parCFR": (0.0, 0.1),
    "parCWH": (0.0, 0.2),
    "route_a": (0.0, 2.9),
    "route_b": (0.0, 6.5),
}


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.65,
            "font.size": 14.2,
            "axes.labelsize": 15.5,
            "axes.titlesize": 15.8,
            "xtick.labelsize": 14.0,
            "ytick.labelsize": 14.0,
            "legend.fontsize": 13.2,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
        }
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def ensure_dirs() -> None:
    for path in (DATA_DIR, REPORT_DIR, FIG_DIR, MAP_CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def normalize_basin_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = re.sub(r"\D", "", text)
    return text.zfill(8) if text else ""


def group_number(group_id: Any) -> int:
    match = re.search(r"\d+", str(group_id))
    return int(match.group(0)) if match else 10**9


def first_existing(candidates: list[Path], search_root: Path, pattern: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(search_root.rglob(pattern)) if search_root.exists() else []
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not locate {pattern}; checked {candidates}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    paths = {
        "basin_groups": first_existing(
            [
                DATA_DIR / "basin_group_assignment_531.csv",
                DATA_DIR / "balanced_basin_groups_531.csv",
                DATA_DIR / "official_basin_groups_11_17_531.csv",
            ],
            FIG4_ROOT,
            "*basin*group*.csv",
        ),
        "relationship_stability": first_existing(
            [DATA_DIR / "figure4_panel_b_data.csv", DATA_DIR / "groupwise_relationship_variability.csv"],
            FIG4_ROOT,
            "*panel_b*.csv",
        ),
        "top5_overlap": first_existing(
            [DATA_DIR / "figure4_panel_c_data.csv", DATA_DIR / "groupwise_topk_overlap.csv"],
            FIG4_ROOT,
            "*panel_c*.csv",
        ),
        "parameters": first_existing(
            [
                STABILITY_TABLE_DIR / "params_long.csv",
                OUTPUT_ROOT / "analysis" / "stability_stats" / "clean" / "params_long_clean.csv",
            ],
            OUTPUT_ROOT,
            "params_long*.csv",
        ),
    }
    assignment = pd.read_csv(paths["basin_groups"])
    variability = pd.read_csv(paths["relationship_stability"])
    topk = pd.read_csv(paths["top5_overlap"])
    params = pd.read_csv(paths["parameters"])

    assignment["basin_id"] = assignment["basin_id"].map(normalize_basin_id)
    params["basin_id"] = params["basin_id"].map(normalize_basin_id)
    for frame in (assignment, variability, topk):
        if "group_id" in frame.columns:
            frame["group_id"] = frame["group_id"].astype(str)
    return assignment, variability, topk, params, paths


def group_order(groups: pd.Series | list[str]) -> list[str]:
    return sorted(pd.Series(groups).dropna().astype(str).unique(), key=group_number)


def group_counts(assignment: pd.DataFrame) -> pd.DataFrame:
    counts = assignment.groupby("group_id", as_index=False).agg(n_basins=("basin_id", "nunique"))
    counts["_order"] = counts["group_id"].map(group_number)
    return counts.sort_values("_order").drop(columns="_order")


def model_column(frame: pd.DataFrame) -> str:
    if "model_raw" in frame.columns:
        return "model_raw"
    if "model" in frame.columns:
        return "model"
    if "formulation" in frame.columns:
        return "formulation"
    raise ValueError(f"No model/formulation column found in {frame.columns.tolist()}")


def summarize_metric(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    mcol = model_column(frame)
    data = frame.loc[frame[mcol].isin(MODEL_ORDER)].copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    summary = (
        data.groupby(["group_id", mcol], as_index=False)
        .agg(
            median=(metric, "median"),
            q25=(metric, lambda values: values.quantile(0.25)),
            q75=(metric, lambda values: values.quantile(0.75)),
            n_values=(metric, "count"),
        )
        .rename(columns={mcol: "model_raw"})
    )
    return summary


def prepare_parameter_values(params: pd.DataFrame, assignment: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_cols = {"basin_id", "model", "loss", "seed", "parameter", "mean"}
    missing_cols = sorted(required_cols - set(params.columns))
    if missing_cols:
        raise ValueError(f"Parameter table missing required columns: {missing_cols}")

    data = params.loc[params["model"].isin(MODEL_ORDER) & params["parameter"].isin(PARAM_ORDER)].copy()
    data["mean"] = pd.to_numeric(data["mean"], errors="coerce")
    grouped = (
        data.groupby(["basin_id", "model", "loss", "seed", "parameter"], as_index=False)
        .agg(value=("mean", "mean"), raw_rows=("mean", "count"), max_sample_count=("sample_count", "max"))
    )
    grouped = grouped.merge(assignment[["basin_id", "group_id"]], on="basin_id", how="inner")
    grouped["group_id"] = grouped["group_id"].astype(str)
    grouped["model_raw"] = grouped["model"]
    grouped["model_plot_label"] = grouped["model_raw"].map(MODEL_LABELS)
    grouped["parameter_label"] = grouped["parameter"].map(PARAM_DISPLAY)

    lows = grouped["parameter"].map(lambda param: PARAMETER_BOUNDS[param][0])
    highs = grouped["parameter"].map(lambda param: PARAMETER_BOUNDS[param][1])
    grouped["normalized_value_raw"] = (grouped["value"] - lows) / (highs - lows)
    grouped["normalized_value"] = grouped["normalized_value_raw"].clip(0, 1)
    grouped = grouped.replace([np.inf, -np.inf], np.nan).dropna(subset=["normalized_value"])

    expected_rows_per_run = assignment["basin_id"].nunique() * len(PARAM_ORDER)
    run_completeness = (
        grouped.groupby(["model_raw", "loss", "seed"], as_index=False)
        .agg(n_basins=("basin_id", "nunique"), n_parameters=("parameter", "nunique"), n_rows=("normalized_value", "count"))
    )
    run_completeness["is_complete"] = run_completeness["n_rows"].eq(expected_rows_per_run)
    out_of_range = int(
        ((grouped["normalized_value_raw"] < 0) | (grouped["normalized_value_raw"] > 1)).sum()
    )
    diagnostics = {
        "expected_rows_per_run": int(expected_rows_per_run),
        "run_completeness": run_completeness,
        "n_runs": int(run_completeness.shape[0]),
        "n_complete_runs": int(run_completeness["is_complete"].sum()),
        "out_of_range_values_clipped": out_of_range,
        "n_parameter_rows_after_aggregation": int(grouped.shape[0]),
        "n_raw_parameter_rows": int(data.shape[0]),
        "losses": sorted(grouped["loss"].dropna().astype(str).unique().tolist()),
        "seeds": sorted(grouped["seed"].dropna().astype(int).unique().tolist()),
    }
    grouped.to_csv(PANEL_D_VALUES_PATH, index=False)
    return grouped, diagnostics


def clean_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color="#ECECEC", linewidth=0.45)
        ax.set_axisbelow(True)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.02, y: float = 1.02) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=18.0,
        fontweight="normal",
        color="#111111",
        clip_on=False,
    )


def basin_points_5070(assignment: pd.DataFrame) -> gpd.GeoDataFrame:
    if {"longitude", "latitude"}.issubset(assignment.columns) and assignment[["longitude", "latitude"]].notna().all(axis=None):
        points = assignment[["basin_id", "group_id", "longitude", "latitude"]].copy()
        points = gpd.GeoDataFrame(
            points,
            geometry=gpd.points_from_xy(points["longitude"], points["latitude"]),
            crs="EPSG:4326",
        )
    else:
        loc = gpd.read_file(CAMELS_LOC_SHP)[["gage_id", "lat", "lon"]].copy()
        loc["basin_id"] = loc["gage_id"].map(normalize_basin_id)
        points = loc.merge(assignment[["basin_id", "group_id"]], on="basin_id", how="inner")
        points = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points["lon"], points["lat"]), crs="EPSG:4326")
    return points.to_crs("EPSG:5070")


def clipped_regions_for_points(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
    if CLIPPED_REGION_SHP.exists():
        return gpd.read_file(CLIPPED_REGION_SHP)
    if not CAMELS_REGION_SHP.exists():
        return None
    regions = gpd.read_file(CAMELS_REGION_SHP)
    if regions.crs is None:
        regions = regions.set_crs("EPSG:4326")
    regions = regions.to_crs("EPSG:5070")
    regions["geometry"] = regions.geometry.make_valid().buffer(0)
    regions = regions.loc[regions.geometry.notna() & ~regions.geometry.is_empty].copy()
    minx, miny, maxx, maxy = points.total_bounds
    pad_x = (maxx - minx) * 0.055
    pad_y = (maxy - miny) * 0.055
    bbox = gpd.GeoDataFrame(geometry=[box(minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)], crs=regions.crs)
    clipped = gpd.clip(regions, bbox)
    clipped["geometry"] = clipped.geometry.make_valid().buffer(0)
    clipped = clipped.loc[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()
    clipped.to_file(CLIPPED_REGION_SHP, encoding="utf-8")
    return clipped


def plot_panel_a(ax: plt.Axes, assignment: pd.DataFrame, counts: pd.DataFrame) -> None:
    points = basin_points_5070(assignment)
    regions = clipped_regions_for_points(points)
    if regions is not None and not regions.empty:
        regions.plot(ax=ax, facecolor="#F8F8F8", edgecolor="#B8B8B8", linewidth=0.36)
        bounds = regions.total_bounds
    else:
        bounds = points.total_bounds

    for gid in group_order(counts["group_id"]):
        subset = points.loc[points["group_id"].eq(gid)]
        ax.scatter(
            subset.geometry.x,
            subset.geometry.y,
            s=27,
            c=GROUP_COLORS.get(gid, "#777777"),
            alpha=0.9,
            edgecolors="white",
            linewidths=0.25,
            rasterized=True,
            zorder=3,
        )

    minx, miny, maxx, maxy = bounds
    pad_x = (maxx - minx) * 0.01
    pad_y = (maxy - miny) * 0.01
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    count_lookup = counts.set_index("group_id")["n_basins"].to_dict()
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=GROUP_COLORS.get(gid, "#777777"),
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=7.3,
            label=f"{gid}: {GROUP_NAMES.get(gid, gid)} (n={int(count_lookup.get(gid, 0))})",
        )
        for gid in group_order(counts["group_id"])
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.035),
        frameon=True,
        facecolor="white",
        framealpha=0.72,
        edgecolor="none",
        ncol=1,
        handletextpad=0.35,
        borderaxespad=0.2,
        labelspacing=0.18,
        columnspacing=0.7,
        fontsize=13.5,
    )
    add_panel_label(ax, "(a)", x=0.0, y=1.02)


def plot_metric_points(
    ax: plt.Axes,
    summary: pd.DataFrame,
    groups: list[str],
    metric_label: str,
    title: str,
    xlim: tuple[float, float] | None,
    show_ylabel: bool,
) -> None:
    offsets = {"deterministic": -0.18, "mc_dropout": 0.0, "distributional": 0.18}
    ypos = {gid: idx for idx, gid in enumerate(groups)}
    for model in MODEL_ORDER:
        sub = summary.loc[summary["model_raw"].eq(model)].copy()
        ys = sub["group_id"].map(ypos).astype(float) + offsets[model]
        x = sub["median"].to_numpy(dtype=float)
        xerr = np.vstack(
            [
                np.maximum(0, x - sub["q25"].to_numpy(dtype=float)),
                np.maximum(0, sub["q75"].to_numpy(dtype=float) - x),
            ]
        )
        ax.errorbar(
            x,
            ys,
            xerr=xerr,
            fmt="o",
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            elinewidth=1.0,
            capsize=2.1,
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.35,
            alpha=0.95,
            label=MODEL_MATH_LABELS[model],
        )
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_ylim(-0.55, len(groups) - 0.45)
    ax.set_xlabel(metric_label, labelpad=3)
    if show_ylabel:
        ax.set_ylabel("Hydroclimatic stratum", labelpad=4)
    else:
        ax.tick_params(axis="y", labelleft=False)
    if xlim:
        ax.set_xlim(*xlim)
    ax.tick_params(axis="both", length=2.5, width=0.65, pad=2)
    clean_axes(ax, "x")


def iqr(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return np.nan
    return float(vals.quantile(0.75) - vals.quantile(0.25))


def summarize_parameter_heatmaps(values: pd.DataFrame) -> pd.DataFrame:
    summary = (
        values.groupby(["parameter", "parameter_label", "group_id", "model_raw", "model_plot_label"], as_index=False)
        .agg(
            median_normalized_value=("normalized_value", "median"),
            iqr_normalized_value=("normalized_value", iqr),
            n_values=("normalized_value", "count"),
        )
    )
    summary.to_csv(PANEL_DE_HEATMAP_PATH, index=False)
    return summary


def heatmap_matrix(summary: pd.DataFrame, model: str, metric: str, groups: list[str]) -> np.ndarray:
    subset = summary.loc[summary["model_raw"].eq(model)]
    table = subset.pivot(index="parameter", columns="group_id", values=metric)
    table = table.reindex(index=PARAM_ORDER, columns=groups)
    return table.to_numpy(dtype=float)


def draw_parameter_heatmap_module(
    fig: plt.Figure,
    spec: Any,
    summary: pd.DataFrame,
    metric: str,
    title: str,
    panel_label: str,
    cmap: str,
    vlim: tuple[float, float],
    colorbar_label: str,
    groups: list[str],
    colorbar_title_x: float,
) -> list[plt.Axes]:
    inner = spec.subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.055], wspace=0.08)
    axes: list[plt.Axes] = []
    image = None
    for idx, model in enumerate(MODEL_ORDER):
        ax = fig.add_subplot(inner[0, idx])
        matrix = heatmap_matrix(summary, model, metric, groups)
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vlim[0], vmax=vlim[1], interpolation="nearest")
        ax.set_title(MODEL_MATH_LABELS[model], pad=5.5, fontsize=16.2)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(groups)
        ax.set_yticks(np.arange(len(PARAM_ORDER)))
        if idx == 0:
            ax.set_yticklabels([PARAM_DISPLAY[param] for param in PARAM_ORDER])
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.set_xticks(np.arange(-0.5, len(groups), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(PARAM_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.55)
        ax.tick_params(axis="both", which="major", length=0, pad=2.8, labelsize=14.0)
        ax.tick_params(axis="both", which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        axes.append(ax)

    cax = fig.add_subplot(inner[0, 3])
    if image is not None:
        cbar = fig.colorbar(image, cax=cax)
        cbar.ax.text(
            colorbar_title_x,
            1.035,
            colorbar_label,
            transform=cbar.ax.transAxes,
            fontsize=14.2,
            linespacing=1.05,
            ha="left",
            va="bottom",
            clip_on=False,
        )
        cbar.ax.tick_params(labelsize=13.8, length=3.0, width=0.65)

    add_panel_label(axes[0], panel_label, x=-0.18, y=1.01)
    return axes


def plot_figure(
    assignment: pd.DataFrame,
    variability: pd.DataFrame,
    topk: pd.DataFrame,
    parameter_values: pd.DataFrame,
) -> None:
    setup_style()
    counts = group_counts(assignment)
    groups = group_order(counts["group_id"])
    panel_b = summarize_metric(variability, "seed_sd_rho")
    panel_c = summarize_metric(topk, "top5_overlap")

    fig = plt.figure(figsize=(19.0, 10.8), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.28],
        left=0.055,
        right=0.965,
        bottom=0.07,
        top=0.945,
        hspace=0.30,
    )
    top = outer[0].subgridspec(1, 3, width_ratios=[1.55, 1.0, 1.0], wspace=0.28)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.14)

    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(top[0, 2], sharey=ax_b)

    plot_panel_a(ax_a, assignment, counts)
    b_max = float(np.nanpercentile(variability["seed_sd_rho"], 98) * 1.12)
    plot_metric_points(
        ax_b,
        panel_b,
        groups,
        "Seed SD of Spearman ρ",
        "Cross-seed relationship stability",
        # (0, max(0.02, b_max)),
        (0, 0.25),
        True,
    )
    add_panel_label(ax_b, "(b)")
    plot_metric_points(
        ax_c,
        panel_c,
        groups,
        "Top-5 overlap (Jaccard)",
        "Top-5 relationship overlap",
        (0.2, 1.0),
        False,
    )
    add_panel_label(ax_c, "(c)")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=MODEL_COLORS[model],
            markerfacecolor=MODEL_COLORS[model],
            markeredgecolor="white",
            markeredgewidth=0.35,
            linestyle="-",
            linewidth=1.0,
            markersize=6.2,
            label=MODEL_MATH_LABELS[model],
        )
        for model in MODEL_ORDER
    ]
    legend_kwargs = {
        "handles": handles,
        "ncol": 1,
        "frameon": False,
        "handlelength": 1.35,
        "borderaxespad": 0.1,
        "labelspacing": 0.25,
        "columnspacing": 0.9,
    }
    ax_b.legend(
        loc="lower right",
        bbox_to_anchor=(0.985, 0.035),
        **legend_kwargs,
    )
    ax_c.legend(
        loc="lower right",
        bbox_to_anchor=(0.985, 0.035),
        **legend_kwargs,
    )

    heatmap_summary = summarize_parameter_heatmaps(parameter_values)
    iqr_vmax = float(np.nanpercentile(heatmap_summary["iqr_normalized_value"], 98))
    iqr_vmax = max(0.05, min(1.0, iqr_vmax))
    draw_parameter_heatmap_module(
        fig,
        bottom[0, 0],
        heatmap_summary,
        "median_normalized_value",
        "Median normalized parameter value",
        "(d)",
        "YlGnBu",
        (0.0, 1.0),
        "Median normalized\nparameter value",
        groups,
        -3.45,
    )
    draw_parameter_heatmap_module(
        fig,
        bottom[0, 1],
        heatmap_summary,
        "iqr_normalized_value",
        "Within-group IQR (normalized value)",
        "(e)",
        "YlOrRd",
        (0.0, iqr_vmax),
        "Within-group IQR\n(normalized value)",
        groups,
        -6.1,
    )

    fig.savefig(f"{OUT_STEM}.png", dpi=600)
    plt.close(fig)


def write_report(
    paths: dict[str, Path],
    assignment: pd.DataFrame,
    variability: pd.DataFrame,
    topk: pd.DataFrame,
    parameter_values: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> None:
    counts = group_counts(assignment)
    groups = group_order(counts["group_id"])
    represented = set(parameter_values["basin_id"].unique())
    assigned = set(assignment["basin_id"].unique())
    missing_param_basins = sorted(assigned - represented)
    panel_b_groups = set(variability["group_id"].astype(str).unique())
    panel_c_groups = set(topk["group_id"].astype(str).unique())
    incomplete_runs = diagnostics["run_completeness"].loc[
        ~diagnostics["run_completeness"]["is_complete"]
    ]
    run_summary = diagnostics["run_completeness"].groupby("model_raw", as_index=False).agg(
        runs=("seed", "count"),
        complete_runs=("is_complete", "sum"),
        min_basins=("n_basins", "min"),
        max_basins=("n_basins", "max"),
    )

    lines = [
        "Figure 4 spatial robustness revised report",
        "",
        "Input files used:",
        f"- Basin metadata/groups: {paths['basin_groups']}",
        f"- Relationship stability summary: {paths['relationship_stability']}",
        f"- Top-5 overlap summary: {paths['top5_overlap']}",
        f"- Parameter table: {paths['parameters']}",
        f"- Map outline: {CAMELS_REGION_SHP if CAMELS_REGION_SHP.exists() else 'not available'}",
        "",
        "Basin counts by hydroclimatic stratum:",
    ]
    for row in counts.itertuples(index=False):
        gid = str(row.group_id)
        lines.append(f"- {gid}: {GROUP_NAMES.get(gid, gid)}; n={int(row.n_basins)}")
    lines.extend(
        [
            "",
            "Run inclusion:",
            f"- Runs included for panels (d) and (e): {diagnostics['n_runs']} model/loss/seed combinations.",
            f"- Complete runs after model/loss/seed/basin/parameter aggregation: {diagnostics['n_complete_runs']}/{diagnostics['n_runs']}.",
            f"- Seeds: {', '.join(map(str, diagnostics['seeds']))}.",
            f"- Losses: {', '.join(diagnostics['losses'])}.",
            f"- Panels (d) and (e) used all losses pooled across basins, seeds, and losses.",
            "- Panel (d) cells show median normalized parameter value by parameter, stratum, and formulation.",
            "- Panel (e) cells show within-group IQR of normalized parameter value by parameter, stratum, and formulation.",
            f"- Parameter values were normalized with HBV search bounds and clipped to [0, 1] for display; clipped values: {diagnostics['out_of_range_values_clipped']}.",
            f"- Aggregated parameter rows used for panels (d) and (e): {diagnostics['n_parameter_rows_after_aggregation']} from {diagnostics['n_raw_parameter_rows']} raw rows.",
            "",
            "Missing data and filtering decisions:",
            f"- Basin group rows: {assignment['basin_id'].nunique()} unique basins.",
            f"- Basins represented in panel (d): {len(represented)} unique basins.",
            f"- Missing panel (d) basins: {missing_param_basins if missing_param_basins else 'none'}.",
            f"- Groups in panel (b): {', '.join(sorted(panel_b_groups, key=group_number))}; missing: {sorted(set(groups) - panel_b_groups, key=group_number) or 'none'}.",
            f"- Groups in panel (c): {', '.join(sorted(panel_c_groups, key=group_number))}; missing: {sorted(set(groups) - panel_c_groups, key=group_number) or 'none'}.",
            f"- Incomplete runs: {'none' if incomplete_runs.empty else incomplete_runs.to_string(index=False)}.",
            f"- Revised panel (d/e) parameter values saved to: {PANEL_D_VALUES_PATH}.",
            f"- Revised panel (d/e) heatmap summary saved to: {PANEL_DE_HEATMAP_PATH}.",
            "",
            "Output files:",
            f"- {OUT_STEM}.png",
            f"- {OUT_STEM}.pdf",
            f"- {PLOTS_DIR / 'plot_fig04_spatial_robustness.py'}",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Report written to %s", REPORT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create revised WRR-style Figure 4 spatial robustness plot.")
    parser.add_argument("--report-only", action="store_true", help="Prepare data/report without saving the figure.")
    args = parser.parse_args()

    ensure_dirs()
    setup_logging()
    assignment, variability, topk, params, paths = load_inputs()
    parameter_values, diagnostics = prepare_parameter_values(params, assignment)
    if not args.report_only:
        plot_figure(assignment, variability, topk, parameter_values)
        logging.info("Figure written to %s.png and %s.pdf", OUT_STEM, OUT_STEM)
    write_report(paths, assignment, variability, topk, parameter_values, diagnostics)


if __name__ == "__main__":
    main()
