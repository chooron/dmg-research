"""Shared plotting utilities for WRR figure generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, wilcoxon

from project.parameterize.train_dmotpy import _normalize_runtime_paths


COLORS = {
    "deterministic": "#4878CF",
    "mc_dropout": "#6ACC65",
    "distributional": "#D65F5F",
}
FONT = "DejaVu Serif"
DPI = 300
MODEL_ORDER = ["deterministic", "mc_dropout", "distributional"]
FIGURE_PREFIX = "fig"
TITLE_SIZE = 12
LABEL_SIZE = 10
TICK_SIZE = 8
PANEL_SIZE = 13
LOSS_LABELS = {
    "NseBatchLoss": "NSE",
    "LogNseBatchLoss": "LogNSE",
    "HybridNseBatchLoss": "HybridNSE",
}
LOSS_ORDER = ["NseBatchLoss", "LogNseBatchLoss", "HybridNseBatchLoss"]
LOSS_MARKERS = {
    "NseBatchLoss": "o",
    "LogNseBatchLoss": "s",
    "HybridNseBatchLoss": "^",
}
PARAM_LABELS = {
    "parFC": "FC",
    "parLP": "LP",
    "parBETA": "BETA",
    "parK0": "K0",
    "parK1": "K1",
    "parK2": "K2",
    "parUZL": "UZL",
    "parPERC": "PERC",
    "parTT": "TT",
    "parCFMAX": "CFMAX",
    "parCFR": "CFR",
    "parCWH": "CWH",
    "route_a": "route_a",
    "route_b": "route_b",
}
KEY_PARAM_ALIASES = {
    "FC": "parFC",
    "BETA": "parBETA",
    "K1": "parK1",
    "TT": "parTT",
}
KEY_ATTRIBUTE_CANDIDATES = [
    "aridity",
    "frac_snow",
    "slope_mean",
    "frac_forest",
    "clay_frac",
    "p_mean",
    "pet_mean",
    "area_gages2",
]
CONCEPT_POINTS = {
    "deterministic": {"xy": (0.78, 0.56), "width": 0.12, "height": 0.10},
    "mc_dropout": {"xy": (0.58, 0.48), "width": 0.14, "height": 0.12},
    "distributional": {"xy": (0.55, 0.82), "width": 0.12, "height": 0.10},
}


def apply_wrr_style() -> None:
    """Apply one consistent style across all figures."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": FONT,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": TICK_SIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.alpha": 0.25,
        }
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(
    fig: Figure,
    stem: str,
    output_dir: Path,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = DPI,
) -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    fig.tight_layout()
    saved: dict[str, Path] = {}
    for suffix in formats:
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved[suffix] = path
    return saved


def add_panel_labels(
    axes: Sequence[Axes] | np.ndarray,
    labels: Sequence[str] | None = None,
    x: float = -0.14,
    y: float = 1.06,
) -> None:
    axes_list = list(np.ravel(axes))
    labels = labels or [chr(ord("A") + idx) for idx in range(len(axes_list))]
    for ax, label in zip(axes_list, labels):
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=PANEL_SIZE,
            fontweight="bold",
            va="bottom",
            ha="left",
        )


def pretty_model_name(name: str) -> str:
    return name.replace("_", " ").title()


def pretty_loss_name(name: str) -> str:
    return LOSS_LABELS.get(name, name)


def pretty_parameter_name(name: str) -> str:
    return PARAM_LABELS.get(name, name)


def normalize_parameter_name(name: str) -> str:
    return KEY_PARAM_ALIASES.get(name, name)


def format_pvalue(value: float) -> str:
    if np.isnan(value):
        return "p=n/a"
    if value < 1e-3:
        return "p<0.001"
    return f"p={value:.3f}"


def add_significance_bracket(
    ax: Axes,
    x0: float,
    x1: float,
    y: float,
    text: str,
    bar_height: float | None = None,
) -> None:
    bar_height = bar_height if bar_height is not None else max(abs(y) * 0.02, 0.01)
    ax.plot([x0, x0, x1, x1], [y, y + bar_height, y + bar_height, y], color="black", lw=1.0)
    ax.text((x0 + x1) / 2.0, y + bar_height * 1.15, text, ha="center", va="bottom")


def paired_wilcoxon(frame: pd.DataFrame, metric: str, left: str, right: str) -> float:
    pivot = (
        frame.pivot_table(
            index=["basin_id", "seed", "loss"],
            columns="model",
            values=metric,
            aggfunc="mean",
        )
        .dropna(subset=[left, right])
        .reset_index(drop=True)
    )
    if pivot.empty:
        return np.nan
    if np.allclose(pivot[left].to_numpy(), pivot[right].to_numpy()):
        return 1.0
    statistic = wilcoxon(pivot[left], pivot[right], zero_method="wilcox", alternative="two-sided")
    return float(statistic.pvalue)


def cluster_order(frame: pd.DataFrame) -> list[str]:
    if frame.shape[0] <= 2:
        return list(frame.index)
    distances = pdist(frame.to_numpy(), metric="euclidean")
    if np.allclose(distances, 0.0):
        return list(frame.index)
    order = leaves_list(linkage(distances, method="average"))
    return [frame.index[idx] for idx in order]


def resolve_analysis_output_root(config_path: str) -> Path:
    raw_config = OmegaConf.load(config_path)
    _normalize_runtime_paths(raw_config)
    output_dir = Path(str(raw_config["output_dir"]).rstrip("/"))
    if len(output_dir.parents) < 3:
        raise ValueError(f"Could not infer outputs root from '{output_dir}'.")
    return ensure_dir(output_dir.parents[2] / "analysis" / "wrr_figures")


def reference_loss_only(frame: pd.DataFrame, data_dict: dict) -> pd.DataFrame:
    if "loss" not in frame.columns:
        return frame.copy()
    return frame.loc[frame["loss"] == data_dict["reference_loss"]].copy()


def top_pairs_by_abs_rho(
    corr_long: pd.DataFrame,
    top_k: int,
    group_cols: Sequence[str] = ("parameter", "attribute"),
) -> list[tuple[str, str]]:
    ranked = (
        corr_long.groupby(list(group_cols), as_index=False)["abs_rho"]
        .mean()
        .sort_values("abs_rho", ascending=False)
        .head(top_k)
    )
    return [(row["parameter"], row["attribute"]) for _, row in ranked.iterrows()]


def correlation_vector_table(corr_long: pd.DataFrame) -> pd.DataFrame:
    vector_table = corr_long.pivot_table(
        index=["model", "loss", "seed"],
        columns=["parameter", "attribute"],
        values="spearman_rho",
    )
    return vector_table.sort_index(axis=1)


def write_manifest(output_dir: Path, manifest: dict) -> Path:
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def overlay_shared_cell_borders(
    ax: Axes,
    mask: np.ndarray,
    linewidth: float = 1.6,
) -> None:
    for row_idx, col_idx in np.argwhere(mask):
        rect = Rectangle(
            (col_idx - 0.5, row_idx - 0.5),
            1.0,
            1.0,
            fill=False,
            edgecolor="black",
            linewidth=linewidth,
        )
        ax.add_patch(rect)


def bonferroni_threshold(num_tests: int, alpha: float = 0.05) -> float:
    if num_tests < 1:
        return alpha
    return alpha / float(num_tests)


def percentile_band(values: Iterable[float], low: float = 5, high: float = 95) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return float(np.nanpercentile(array, low)), float(np.nanpercentile(array, high))


def choose_available_attributes(attribute_names: Sequence[str]) -> list[str]:
    available = []
    for name in KEY_ATTRIBUTE_CANDIDATES:
        if name in attribute_names:
            available.append(name)
    return available


def seed_error_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["model", "loss", "basin_id", "parameter"], as_index=False).agg(
        mean_seed=("mean", "mean"),
        std_seed=("mean", "std"),
    )


def symmetric_vlim(values: pd.Series | np.ndarray, floor: float = 0.30) -> float:
    maximum = float(np.nanmax(np.abs(np.asarray(values, dtype=float))))
    return max(floor, maximum)


def distributional_correlation_tables(
    data_dict: dict,
    value_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = reference_loss_only(data_dict["params_long"], data_dict)
    params = params.loc[params["model"] == "distributional"]
    parameter_matrix = params.groupby(["basin_id", "parameter"], as_index=False)[value_column].mean()
    parameter_matrix = parameter_matrix.pivot(index="basin_id", columns="parameter", values=value_column)
    merged = data_dict["attributes"].merge(parameter_matrix, on="basin_id", how="inner")
    attributes = [column for column in data_dict["attribute_names"]]

    rho_table = pd.DataFrame(index=attributes, columns=data_dict["param_names"], dtype=float)
    p_table = pd.DataFrame(index=attributes, columns=data_dict["param_names"], dtype=float)
    for attribute in attributes:
        for parameter in data_dict["param_names"]:
            rho, p_value = spearmanr(merged[attribute], merged[parameter], nan_policy="omit")
            rho_table.loc[attribute, parameter] = float(rho)
            p_table.loc[attribute, parameter] = float(p_value)
    return rho_table, p_table
