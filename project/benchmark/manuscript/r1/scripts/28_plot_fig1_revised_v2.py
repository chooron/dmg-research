#!/usr/bin/env python3
"""Render the second revised R1 Figure 1 as one PNG-only composite.

The renderer consumes the existing formal R1/R1X figure tables. It adds no
analysis, produces no panel files, and does not write PDF output.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from r1_config import FIGURES_DIR, MODEL_REGISTRY, TABLES_DIR
from R1X_gis import basin_point_geodataframe, load_conus_boundaries, tendency_cmap

OUT_PATH = FIGURES_DIR / "Fig1_R1_main_revised_v2.png"
BLUE = "#2166AC"
ORANGE = "#D6604D"
DARK = "#243447"
MID = "#7B8794"
LIGHT = "#D8DEE4"
ROW_DPL = "#FEF7F4"
GRAY_LIGHT = "#D9D9D9"
GRAY_MID = "#969696"
GRAY_DARK = "#4D4D4D"


def configure_serif_style() -> str:
    """Use a fixed installed serif family for stable rerendering."""
    installed = {font.name for font in fm.fontManager.ttflist}
    family = "DejaVu Serif" if "DejaVu Serif" in installed else "serif"
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [family],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.1,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.72,
            "xtick.major.size": 3.0,
            "xtick.major.width": 0.68,
            "ytick.major.size": 3.0,
            "ytick.major.width": 0.68,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
        }
    )
    return family


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.72)
        ax.spines[side].set_color(DARK)
    ax.tick_params(colors=DARK, pad=2.2)


def heading(ax: plt.Axes, label: str, title: str, *, fontsize: float = 8.8) -> None:
    ax.text(
        0.0,
        1.025,
        f"{label} {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color=DARK,
        clip_on=False,
    )


def add_row_backdrops(ax: plt.Axes, y: np.ndarray, p_positive: np.ndarray) -> None:
    """Highlight only rows where more than half of basins favor dPL."""
    for yi, p_value in zip(y, p_positive):
        if float(p_value) > 0.5:
            ax.axhspan(yi - 0.46, yi + 0.46, color=ROW_DPL, zorder=0)


def draw_panel_a(ax: plt.Axes, data: pd.DataFrame, p_data: pd.DataFrame) -> None:
    y = np.arange(len(data))
    x_ic = data["IC_median_KGE"].to_numpy(float)
    x_dpl = data["dPL_median_KGE"].to_numpy(float)
    p_positive = p_data["P_m_positive"].to_numpy(float)
    lo = float(min(x_ic.min(), x_dpl.min()) - 0.025)
    hi = float(max(x_ic.max(), x_dpl.max()) + 0.025)
    add_row_backdrops(ax, y, p_positive)
    for yi, ic, dpl in zip(y, x_ic, x_dpl):
        ax.plot([ic, dpl], [yi - 0.07, yi + 0.07], color="#AEB8C2", lw=0.82, zorder=1)
    ax.scatter(x_ic, y - 0.07, s=27, facecolor="white", edgecolor=BLUE, linewidth=0.95, marker="o", label="IC", zorder=3)
    ax.scatter(x_dpl, y + 0.07, s=27, facecolor=ORANGE, edgecolor="white", linewidth=0.45, marker="s", label="dPL", zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.55, len(data) - 0.45)
    ax.set_yticks(y, data["model"])
    ax.invert_yaxis()
    ax.set_xlabel("Median KGE across basins")
    clean_axes(ax)
    heading(ax, "(a)", "Aggregate outlet performance")
    med_ic = float(data["IC_median_KGE"].median())
    med_dpl = float(data["dPL_median_KGE"].median())
    med_delta = float(data["delta_model_median"].median())
    ax.text(
        0.98,
        1.005,
        f"Across-model median: IC {med_ic:.3f} · dPL {med_dpl:.3f}\nmedian model-level $\\Delta KGE$ = {med_delta:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.9,
        color=MID,
        clip_on=False,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.985), frameon=True, framealpha=0.90, edgecolor=LIGHT, borderpad=0.32, handletextpad=0.32)


def draw_panel_b(ax: plt.Axes, data: pd.DataFrame, p_data: pd.DataFrame) -> None:
    y = np.arange(len(data))
    q10 = data["delta_Q10"].to_numpy(float)
    q25 = data["delta_Q25"].to_numpy(float)
    med = data["delta_median"].to_numpy(float)
    q75 = data["delta_Q75"].to_numpy(float)
    q90 = data["delta_Q90"].to_numpy(float)
    p_positive = p_data["P_m_positive"].to_numpy(float)
    lo = float(q10.min() - 0.018)
    hi = float(q90.max() + 0.065)
    add_row_backdrops(ax, y, p_positive)
    ax.hlines(y, q10, q90, color=MID, lw=1.0, zorder=1)
    box_colors = np.where(med >= 0.0, ORANGE, BLUE)
    for yi, left, right, color in zip(y, q25, q75, box_colors):
        ax.hlines(yi, left, right, color=color, lw=4.2, alpha=0.88, zorder=2)
    ax.scatter(med, y, s=24, color=DARK, marker="o", edgecolor="white", linewidth=0.35, zorder=3)
    ax.axvline(0.0, color=DARK, lw=0.75, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.55, len(data) - 0.45)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta KGE = KGE_{dPL} - KGE_{IC}$")
    clean_axes(ax)
    heading(ax, "(b)", r"Basin-level $\Delta KGE$ distribution")
    ax.text(0.985, 1.005, r"$P_m^+$", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.9, color=MID, clip_on=False)
    for yi, p_value in zip(y, p_data["P_m_positive"]):
        ax.text(hi - 0.006, yi, f"{100 * float(p_value):.0f}%", ha="right", va="center", fontsize=5.5, color=DARK, zorder=4)


def draw_panel_c(fig: plt.Figure, ax: plt.Axes, data: pd.DataFrame, states: gpd.GeoDataFrame, national: gpd.GeoDataFrame) -> None:
    points = basin_point_geodataframe(data)
    cmap, norm = tendency_cmap()
    ax.set_aspect("equal")
    ax.axis("off")
    bounds = national.total_bounds
    dx = bounds[2] - bounds[0]
    dy = bounds[3] - bounds[1]
    ax.set_xlim(bounds[0] - 0.014 * dx, bounds[2] + 0.014 * dx)
    ax.set_ylim(bounds[1] - 0.014 * dy, bounds[3] + 0.014 * dy)
    states.plot(ax=ax, facecolor="#F8FAFC", edgecolor="#CBD3DC", linewidth=0.30, zorder=1)
    national.plot(ax=ax, facecolor="none", edgecolor="#4F5D6B", linewidth=0.70, zorder=2)
    points.plot(ax=ax, column="P_b_dPL", cmap=cmap, norm=norm, markersize=15, edgecolor="white", linewidth=0.24, alpha=0.94, zorder=3)
    heading(ax, "(c)", "Basin-level cross-model estimator tendency", fontsize=8.7)
    ax.text(0.012, 0.018, "Full-period descriptive tendency", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.7, color=MID, clip_on=False)
    inset = ax.inset_axes([0.704, 0.055, 0.245, 0.245], zorder=5)
    inset.set_facecolor("white")
    inset.hist(data["P_b_dPL"], bins=11, range=(0, 1), color="#9AA0A6", edgecolor="white", linewidth=0.35)
    inset.axvline(0.5, color=DARK, ls=(0, (3, 2)), lw=0.72)
    inset.set_xlim(0, 1)
    inset.set_xticks([0, 0.5, 1.0])
    inset.set_xlabel(r"$P_b^{dPL}$", fontsize=5.9, labelpad=1)
    inset.set_ylabel("Basins", fontsize=5.9, labelpad=1)
    inset.set_title("Basin distribution", loc="left", fontsize=6.0, pad=2)
    inset.tick_params(labelsize=5.1, length=2, pad=1)
    for spine in inset.spines.values():
        spine.set_linewidth(0.48)
        spine.set_color(DARK)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.024, pad=0.010, shrink=0.76)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["IC", "mixed", "dPL"])
    cbar.ax.tick_params(labelsize=5.9, length=2.0, width=0.55, pad=2)
    cbar.set_label("Fraction of models with dPL > IC", fontsize=6.5, labelpad=4)
    cbar.outline.set_linewidth(0.42)


def draw_panel_d(ax: plt.Axes, data: pd.DataFrame) -> None:
    order = ["model_main_effect", "basin_main_effect", "residual_model_basin_specificity"]
    labels = ["Model\nmain effect", "Basin\nmain effect", "Residual\nmodel–basin\nspecificity"]
    colors = [GRAY_LIGHT, GRAY_MID, GRAY_DARK]
    values = [100 * float(data.loc[data.component == key, "fraction_of_total"].iloc[0]) for key in order]
    y = np.arange(3)
    ax.barh(y, values, color=colors, height=0.52, edgecolor="white", linewidth=0.45, zorder=2)
    for yi, value in zip(y, values):
        ax.text(value + 1.8, yi, f"{value:.1f}%", va="center", ha="left", fontsize=6.9, color=DARK, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_ylim(-0.55, 2.55)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"])
    ax.set_xlabel(r"Share of $\Delta KGE$ variation (%)", fontsize=7.0)
    ax.grid(axis="x", color=LIGHT, lw=0.5, alpha=0.72)
    clean_axes(ax)
    heading(ax, "(d)", r"Descriptive decomposition of $\Delta KGE$ variation", fontsize=7.9)


def draw_panel_e(ax: plt.Axes, data: pd.DataFrame) -> None:
    x = data["delta_median_A"].to_numpy(float)
    y = data["delta_median_B"].to_numpy(float)
    same = data["same_sign_non_neutral"].astype(str).str.lower().eq("true").to_numpy()
    bound = max(0.045, float(np.max(np.abs(np.r_[x, y]))) * 1.14)
    ax.scatter(x[same], y[same], s=24, color=DARK, edgecolor="white", linewidth=0.4, marker="o", label="same sign", zorder=3)
    ax.scatter(x[~same], y[~same], s=30, facecolor="white", edgecolor=DARK, linewidth=0.9, marker="D", label="sign-changing", zorder=3)
    ax.plot([-bound, bound], [-bound, bound], color=DARK, lw=0.82, ls=(0, (4, 2)), zorder=1)
    ax.axhline(0, color=MID, lw=0.55, zorder=1)
    ax.axvline(0, color=MID, lw=0.55, zorder=1)
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Median $\Delta KGE$, A", fontsize=7.5)
    ax.set_ylabel(r"Median $\Delta KGE$, B", fontsize=7.5)
    ax.grid(color=LIGHT, lw=0.42, alpha=0.42)
    clean_axes(ax)
    heading(ax, "(e)", "Temporal A/B persistence", fontsize=8.3)
    rank_a = pd.Series(x).rank(method="average").to_numpy()
    rank_b = pd.Series(y).rank(method="average").to_numpy()
    rho = float(np.corrcoef(rank_a, rank_b)[0, 1])
    same_count = int(np.sum(same))
    ax.text(0.045, 0.955, f"Spearman $\\rho$={rho:.3f}\nsame sign={same_count}/36\n$N$=36 models", transform=ax.transAxes, ha="left", va="top", fontsize=6.1, bbox={"facecolor": "white", "edgecolor": LIGHT, "pad": 2.3, "alpha": 0.92})
    ax.legend(loc="upper right", fontsize=5.7, frameon=False, handletextpad=0.3, borderpad=0.05)
    changing = np.where(~same)[0]
    if changing.size:
        extreme = changing[np.argmax(np.maximum(np.abs(x[changing]), np.abs(y[changing])))]
        ax.annotate(str(data.iloc[extreme]["model"]), (x[extreme], y[extreme]), xytext=(-4, 5), textcoords="offset points", ha="right", va="bottom", fontsize=5.6, color=DARK)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a = pd.read_csv(TABLES_DIR / "Fig1a_model_aggregate_performance.csv").sort_values("plot_order").reset_index(drop=True)
    b = pd.read_csv(TABLES_DIR / "Fig1b_model_delta_distribution.csv").sort_values("plot_order").reset_index(drop=True)
    c = pd.read_csv(TABLES_DIR / "Fig1c_basin_crossmodel_tendency.csv").sort_values("plot_order").reset_index(drop=True)
    d = pd.read_csv(TABLES_DIR / "Fig1d_two_way_decomposition.csv")
    e = pd.read_csv(TABLES_DIR / "Fig1e_temporal_model_persistence.csv").sort_values("plot_order").reset_index(drop=True)
    if tuple(a["model"]) != MODEL_REGISTRY or tuple(b["model"]) != MODEL_REGISTRY or tuple(e["model"]) != MODEL_REGISTRY:
        raise ValueError("Figure 1 model order is not the canonical registry order")
    if len(c) != 531 or c["basin_id"].nunique() != 531 or c["n_models"].ne(36).any():
        raise ValueError("Figure 1 GIS table does not have exact 531-basin/36-model coverage")
    if not np.isclose(d.loc[d.component != "total", "fraction_of_total"].sum(), 1.0):
        raise ValueError("Figure 1 decomposition shares do not sum to one")
    if not np.isfinite(b["P_m_positive"].to_numpy(float)).all():
        raise ValueError("Figure 1 model tendency values contain non-finite values")
    if not np.array_equal(a["model"].to_numpy(), b["model"].to_numpy()) or not np.array_equal(a["model"].to_numpy(), e["model"].to_numpy()):
        raise ValueError("Panels (a), (b), and (e) are not aligned")
    return a, b, c, d, e


def main() -> None:
    font_family = configure_serif_style()
    a, b, c, d, e = load_tables()
    states, national = load_conus_boundaries()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14.3, 8.35), facecolor="white")
    gs = gridspec.GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[1.06, 1.06, 1.34, 1.16],
        height_ratios=[1.42, 0.94],
        left=0.055,
        right=0.955,
        bottom=0.085,
        top=0.945,
        wspace=0.28,
        hspace=0.25,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[0, 2:])
    ax_d = fig.add_subplot(gs[1, 2])
    ax_e = fig.add_subplot(gs[1, 3])
    draw_panel_a(ax_a, a, b)
    draw_panel_b(ax_b, b, b)
    draw_panel_c(fig, ax_c, c, states, national)
    draw_panel_d(ax_d, d)
    draw_panel_e(ax_e, e)
    fig.savefig(OUT_PATH, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"PASS: wrote {OUT_PATH}")
    print(f"font_family={font_family}; png_only=True; dpi=600; panels=5; highlighted_rule=P_m_positive>0.5")


if __name__ == "__main__":
    main()
