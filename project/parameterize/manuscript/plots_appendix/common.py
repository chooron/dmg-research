from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[4]
PARAM_ROOT = ROOT / "project" / "parameterize"
ANALYSIS_ROOT = PARAM_ROOT / "outputs" / "analysis" / "stability_stats"
TABLE_ROOT = ANALYSIS_ROOT / "tables"
CORR_ROOT = ANALYSIS_ROOT / "correlation_summaries"
VAR_ROOT = ANALYSIS_ROOT / "parameter_variance"
MANUSCRIPT_ROOT = PARAM_ROOT / "manuscript"
PLOTS_ROOT = MANUSCRIPT_ROOT / "plots"
MAIN_FIG_DIR = MANUSCRIPT_ROOT / "figures" / "main"
APP_FIG_DIR = MANUSCRIPT_ROOT / "figures" / "appendix"
REPORT_DIR = MANUSCRIPT_ROOT / "reports"

MM = 1 / 25.4
DPI = 600
FONT_FAMILY = "Times New Roman"
MODEL_ORDER = ["deterministic", "mc_dropout", "distributional"]
MODEL_LABELS = {
    "deterministic": "Deterministic",
    "mc_dropout": "MC dropout",
    "distributional": "Distributional",
}
MODEL_COLORS = {
    "deterministic": "#4C78A8",
    "mc_dropout": "#F58518",
    "distributional": "#2A9D8F",
}
CLASS_COLORS = {
    "shared dominant controls": "#2A9D8F",
    "partially shared controls": "#8AA6B8",
    "model-sensitive controls": "#B55A4A",
    "loss-sensitive": "#B55A4A",
    "robust": "#2A9D8F",
    "supportive": "#8AA6B8",
    "secondary": "#D0D4D8",
    "headline": "#2A9D8F",
}
ATTR_LABELS = {
    "aridity": "Aridity",
    "frac_snow": "Snow fraction",
    "slope_mean": "Mean slope",
    "pet_mean": "PET",
    "p_mean": "Precip.",
    "clay_frac": "Clay",
    "soil_depth_pelletier": "Soil depth",
    "soil_conductivity": "Soil cond.",
    "elev_mean": "Elevation",
    "area_gages2": "Area",
    "frac_forest": "Forest",
    "lai_diff": "LAI diff.",
    "gvf_diff": "GVF diff.",
    "p_seasonality": "Precip. seasonality",
    "high_prec_dur": "High-precip. dur.",
    "high_prec_freq": "High-precip. freq.",
    "low_prec_dur": "Low-precip. dur.",
    "low_prec_freq": "Low-precip. freq.",
    "carbonate_rocks_frac": "Carbonate",
    "dom_land_cover": "Land cover",
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
FOCUS_PARAMS = ["parFC", "parUZL", "parCFR"]
KEY_ATTRS = ["aridity", "frac_snow", "slope_mean", "pet_mean", "soil_depth_pelletier", "soil_conductivity"]
SHORT_ATTR_LABELS = {
    "aridity": "Aridity",
    "frac_snow": "Snow\nfrac.",
    "slope_mean": "Slope",
    "pet_mean": "PET",
    "soil_depth_pelletier": "Soil\ndepth",
    "soil_conductivity": "Soil\ncond.",
}


@dataclass(frozen=True)
class FigureSpec:
    stem: str
    directory: Path
    width_mm: float = 180.0
    height_mm: float = 125.0


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FONT_FAMILY],
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_FAMILY,
            "mathtext.it": f"{FONT_FAMILY}:italic",
            "mathtext.bf": f"{FONT_FAMILY}:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.85,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 9.2,
            "axes.grid": False,
            "savefig.dpi": DPI,
            "savefig.facecolor": "white",
        }
    )


def muted_diverging() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "muted_blue_white_brown",
        ["#4C78A8", "#F7F7F4", "#B55A4A"],
        N=256,
    )


def muted_seq() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("muted_teal", ["#F4F7F7", "#A8C6C0", "#2A9D8F"], N=256)


def read_csv(name: str, root: Path = CORR_ROOT) -> pd.DataFrame:
    return pd.read_csv(root / name)


PARAM_LABELS = {
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


def p_label(parameter: str) -> str:
    return PARAM_LABELS.get(str(parameter), str(parameter).replace("par", ""))


def a_label(attribute: str) -> str:
    return ATTR_LABELS.get(str(attribute), str(attribute).replace("_", " "))


def pair_label(parameter: str, attribute: str) -> str:
    return f"{p_label(parameter)} - {a_label(attribute)}"


def pair_label_from_key(pair_key: str) -> str:
    if "__" not in str(pair_key):
        return str(pair_key).replace("par", "")
    parameter, attribute = str(pair_key).split("__", 1)
    return pair_label(parameter, attribute)


def ordered(items: list[str], preferred: list[str]) -> list[str]:
    seen = [x for x in preferred if x in set(items)]
    seen.extend([x for x in items if x not in seen])
    return seen


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.08,
    y: float = 1.05,
    *,
    ha: str = "left",
    va: str = "bottom",
    fontweight: str = "normal",
    fontsize: float = 13.5,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight=fontweight,
        color="#111111",
    )


def clean_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color="#E6E6E6", linewidth=0.55)
        ax.set_axisbelow(True)


def wrap_labels(labels: list[str], width: int = 14) -> list[str]:
    return ["\n".join(wrap(str(label), width=width, break_long_words=False)) for label in labels]


def figure(spec: FigureSpec, *, rows: int = 1, cols: int = 1, **kwargs):
    figsize = (spec.width_mm * MM, spec.height_mm * MM)
    return plt.subplots(rows, cols, figsize=figsize, constrained_layout=True, **kwargs)


def save(fig: plt.Figure, spec: FigureSpec, *, formats: tuple[str, ...] = ("png",)) -> None:
    spec.directory.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(spec.directory / f"{spec.stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def horizontal_bars(ax: plt.Axes, data: pd.DataFrame, value: str, label: str, color: str, xlab: str) -> None:
    d = data.sort_values(value)
    ax.barh(np.arange(len(d)), d[value], color=color, alpha=0.9, height=0.66)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d[label])
    ax.set_xlabel(xlab)
    clean_axes(ax, "x")


def heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    cmap,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    cbar_label: str = "",
    text: bool = False,
    fmt: str = "{:.2f}",
) -> None:
    values = matrix.to_numpy(dtype=float)
    norm = None
    if center is not None:
        bound = np.nanmax(np.abs(values)) if vmin is None or vmax is None else max(abs(vmin), abs(vmax))
        norm = TwoSlopeNorm(vmin=-bound, vcenter=center, vmax=bound)
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(wrap_labels([a_label(c) for c in matrix.columns], 11), rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([p_label(r) for r in matrix.index])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if text:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=8.6, color="#222222")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.ax.tick_params(labelsize=8.8, length=2)
    cbar.set_label(cbar_label, fontsize=9.2)


def shorten_heatmap_xticks(ax: plt.Axes, labels: list[str], *, rotation: float = 50, fontsize: float = 8.9) -> None:
    ax.set_xticklabels(labels, rotation=rotation, ha="right", fontsize=fontsize)


def parameter_family(parameter: str) -> str:
    p = str(parameter)
    if p in {"parBETA", "parFC", "parLP"}:
        return "Soil storage"
    if p in {"parPERC", "parUZL", "parK0", "parK1", "parK2"}:
        return "Runoff response"
    if p in {"parCFMAX", "parTT", "parCFR", "parCWH"}:
        return "Snow/cold process"
    return "Routing"


def metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    out = (
        metrics.groupby("model", as_index=False)
        .agg(nse_median=("nse", "median"), kge_median=("kge", "median"), bias_abs_median=("bias_abs", "median"))
        .set_index("model")
        .loc[MODEL_ORDER]
    )
    return out


def math_model_labels() -> dict[str, str]:
    return {
        "deterministic": r"$\it{\delta}_{base}$",
        "mc_dropout": r"$\it{\delta}_{mcd}$",
        "distributional": r"$\it{\delta}_{dist}$",
    }
