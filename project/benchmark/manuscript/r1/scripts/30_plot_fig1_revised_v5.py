#!/usr/bin/env python3
"""Render the fifth revised R1 Figure 1 as one PNG-only composite.

The renderer consumes the existing formal R1/R1X figure tables. It adds no
analysis, produces no panel files, and does not write PDF output.
"""
from __future__ import annotations

import os
import json

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import geopandas as gpd
import matplotlib

import matplotlib.colors as mcolors
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from r1_config import CACHE_DIR, FIGURES_DIR, MODEL_REGISTRY, TABLES_DIR
from R1X_gis import basin_point_geodataframe, load_conus_boundaries

OUT_PATH = FIGURES_DIR / "Fig1_R1_main_revised_v5.png"
BLUE = "#2166AC"
ORANGE = "#D6604D"
DARK = "#243447"
MID = "#7B8794"
LIGHT = "#D8DEE4"
SLATE_LIGHT = "#B8C5D2"
SLATE_MID = "#718096"
SLATE_DARK = "#2F4858"


def configure_serif_style() -> str:
    """Use a fixed installed serif family for stable rerendering."""
    installed = {font.name for font in fm.fontManager.ttflist}
    family = "DejaVu Serif" if "DejaVu Serif" in installed else "serif"
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [family],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9.4,
            "axes.labelsize": 10.4,
            "axes.titlesize": 10.8,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.2,
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


def heading(ax: plt.Axes, label: str, title: str, *, fontsize: float = 10.8) -> None:
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


def draw_panel_a(ax: plt.Axes, data: pd.DataFrame) -> None:
    y = np.arange(len(data))
    x_ic = data["IC_median_KGE"].to_numpy(float)
    x_dpl = data["dPL_median_KGE"].to_numpy(float)
    lo = float(min(x_ic.min(), x_dpl.min()) - 0.025)
    hi = float(max(x_ic.max(), x_dpl.max()) + 0.025)
    for yi, ic, dpl in zip(y, x_ic, x_dpl):
        ax.plot([ic, dpl], [yi, yi], color="#AEB8C2", lw=0.82, zorder=1)
    ax.scatter(x_ic, y, s=27, facecolor="white", edgecolor=BLUE, linewidth=0.95, marker="o", label="IC", zorder=3)
    ax.scatter(x_dpl, y, s=27, facecolor=ORANGE, edgecolor="white", linewidth=0.45, marker="s", label="dPL", zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.55, len(data) - 0.45)
    ax.set_yticks(y, data["model"])
    ax.invert_yaxis()
    ax.set_xlabel("Median KGE across basins")
    clean_axes(ax)
    heading(ax, "(a)", "Aggregate outlet performance")
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.985), frameon=True, framealpha=0.90, edgecolor=LIGHT, borderpad=0.32, handletextpad=0.32)



def draw_panel_b(ax: plt.Axes, data: pd.DataFrame) -> None:
    y = np.arange(len(data))
    q10 = data["delta_Q10"].to_numpy(float)
    q25 = data["delta_Q25"].to_numpy(float)
    med = data["delta_median"].to_numpy(float)
    q75 = data["delta_Q75"].to_numpy(float)
    q90 = data["delta_Q90"].to_numpy(float)
    lo = float(q10.min() - 0.018)
    hi = float(q90.max() + 0.065)
    ax.set_xlim(lo, hi)
    ax.axvspan(lo, 0.0, color=BLUE, alpha=0.045, zorder=0)
    ax.axvspan(0.0, hi, color=ORANGE, alpha=0.045, zorder=0)
    ax.hlines(y, q10, q90, color=MID, lw=1.0, zorder=1)
    box_colors = np.where(med >= 0.0, ORANGE, BLUE)
    for yi, left, right, color in zip(y, q25, q75, box_colors):
        ax.hlines(yi, left, right, color=color, lw=4.2, alpha=0.88, zorder=2)
    ax.scatter(med, y, s=27, color=DARK, marker="o", edgecolor="white", linewidth=0.40, zorder=3)
    ax.axvline(0.0, color=DARK, lw=1.05, ls=(0, (4, 2)), zorder=2)
    ax.set_ylim(-0.55, len(data) - 0.45)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta KGE = KGE_{dPL} - KGE_{IC}$")
    clean_axes(ax)
    heading(ax, "(b)", r"Basin-level $\Delta KGE$ distribution")
    ax.text(0.02, 1.005, "IC higher  ←", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.8, color=BLUE, clip_on=False)
    ax.text(0.50, 1.005, r"$\Delta KGE = 0$", transform=ax.transAxes, ha="center", va="bottom", fontsize=8.7, color=DARK, clip_on=False)
    ax.text(0.68, 1.005, "→ dPL higher", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.8, color=ORANGE, clip_on=False)
    ax.text(0.985, 1.005, r"$P_m^+$", transform=ax.transAxes, ha="right", va="bottom", fontsize=9.2, color=DARK, clip_on=False)
    for yi, p_value in zip(y, data["P_m_positive"]):
        ax.text(hi - 0.008, yi, f"{100 * float(p_value):.0f}%", ha="right", va="center", fontsize=8.5, color=DARK, zorder=4)



def draw_panel_c(fig: plt.Figure, ax: plt.Axes, data: pd.DataFrame, states: gpd.GeoDataFrame, national: gpd.GeoDataFrame, map_metadata: dict) -> None:
    points = basin_point_geodataframe(data.sort_values("abs_M_b", kind="stable"))
    limit = float(map_metadata["selected_color_limit_L"])
    cmap = mcolors.LinearSegmentedColormap.from_list("median_delta_kge", [BLUE, "#F7F7F7", ORANGE], N=256)
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    ax.axis("off")
    heading(ax, "(c)", "Basin-level estimator effect and cross-model spread", fontsize=10.8)

    map_ax = ax.inset_axes([0.0, 0.0, 0.72, 0.92], zorder=1)
    map_ax.set_aspect("equal")
    map_ax.axis("off")
    bounds = national.total_bounds
    dx = bounds[2] - bounds[0]
    dy = bounds[3] - bounds[1]
    map_ax.set_xlim(bounds[0] - 0.014 * dx, bounds[2] + 0.014 * dx)
    map_ax.set_ylim(bounds[1] - 0.014 * dy, bounds[3] + 0.014 * dy)
    states.plot(ax=map_ax, facecolor="#F8FAFC", edgecolor="#CBD3DC", linewidth=0.30, zorder=1)
    national.plot(ax=map_ax, facecolor="none", edgecolor="#4F5D6B", linewidth=0.70, zorder=2)
    points.plot(ax=map_ax, column="M_b_median_deltaKGE", cmap=cmap, norm=norm, markersize=17, edgecolor="#F8F9FA", linewidth=0.24, alpha=0.94, zorder=3)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = ax.inset_axes([0.855, 0.53, 0.055, 0.30], zorder=4)
    cbar = fig.colorbar(sm, cax=cbar_ax, extend="both")
    cbar.set_ticks([-limit, 0.0, limit])
    cbar.set_ticklabels(["IC higher", "0", "dPL higher"])
    cbar.ax.tick_params(labelsize=8.0, length=2.0, width=0.55, pad=2)
    cbar.set_label("Median ΔKGE\nacross 36 models", fontsize=8.0, labelpad=8)
    cbar.outline.set_linewidth(0.42)

    inset = ax.inset_axes([0.73, 0.035, 0.26, 0.40], zorder=5)
    inset.set_facecolor("white")
    inset.scatter(data["M_b_median_deltaKGE"], data["S_b_MAD_deltaKGE"], s=10, color=SLATE_DARK, edgecolor="white", linewidth=0.25, alpha=0.72)
    xs = np.linspace(-0.50, 0.50, 200)
    inset.plot(xs, np.abs(xs), color=MID, ls=(0, (3, 2)), lw=0.72)
    inset.axvline(0.0, color=LIGHT, lw=0.60)
    inset.set_xlim(-0.50, 0.50)
    inset.set_ylim(0.0, 0.50)
    inset.set_xticks([-0.4, 0.0, 0.4])
    inset.set_yticks([0.0, 0.2, 0.4])
    inset.set_xlabel(r"$M_b$", fontsize=8.2, labelpad=1)
    inset.set_ylabel(r"$S_b$ (MAD)", fontsize=8.2, labelpad=1)
    inset.set_title("Effect vs spread", loc="left", fontsize=8.2, pad=2)
    inset.tick_params(labelsize=7.2, length=2, pad=1)
    for spine in inset.spines.values():
        spine.set_linewidth(0.52)
        spine.set_color(DARK)



def draw_panel_d(ax: plt.Axes, data: pd.DataFrame) -> None:
    order = ["model_main_effect", "basin_main_effect", "residual_model_basin_specificity"]
    labels = ["Model\neffect", "Basin\neffect", "Residual\nmodel–basin\nspecificity"]
    colors = [SLATE_LIGHT, SLATE_MID, SLATE_DARK]
    values = [100 * float(data.loc[data.component == key, "fraction_of_total"].iloc[0]) for key in order]
    y = np.arange(3)
    ax.axis("off")
    bar_ax = ax.inset_axes([0.30, 0.0, 0.70, 0.90], zorder=1)
    bar_ax.barh(y, values, color=colors, height=0.52, edgecolor="white", linewidth=0.45, zorder=2)
    for yi, value in zip(y, values):
        bar_ax.text(value + 1.8, yi, f"{value:.1f}%", va="center", ha="left", fontsize=9.0, color=DARK, fontweight="bold")
    bar_ax.set_xlim(0, 105)
    bar_ax.set_ylim(-0.55, 2.55)
    bar_ax.set_yticks(y, labels)
    bar_ax.invert_yaxis()
    bar_ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"])
    bar_ax.set_xlabel(r"Share of $\Delta KGE$ variation (%)", fontsize=9.4, labelpad=4)
    bar_ax.tick_params(axis="y", labelsize=9.0, length=0, pad=3)
    bar_ax.tick_params(axis="x", labelsize=8.8)
    bar_ax.grid(axis="x", color=LIGHT, lw=0.5, alpha=0.72)
    clean_axes(bar_ax)
    heading(ax, "(d)", "Descriptive decomposition of\nΔKGE variation", fontsize=10.2)



def draw_panel_e(ax: plt.Axes, data: pd.DataFrame) -> None:
    x = data["delta_median_A"].to_numpy(float)
    y = data["delta_median_B"].to_numpy(float)
    same = data["same_sign_non_neutral"].astype(str).str.lower().eq("true").to_numpy()
    bound = max(0.045, float(np.max(np.abs(np.r_[x, y]))) * 1.14)
    ax.scatter(x[same], y[same], s=30, color=SLATE_DARK, edgecolor="white", linewidth=0.45, marker="o", label="same sign", zorder=3)
    ax.scatter(x[~same], y[~same], s=36, facecolor="white", edgecolor=SLATE_DARK, linewidth=1.0, marker="D", label="sign-changing", zorder=3)
    ax.plot([-bound, bound], [-bound, bound], color=SLATE_DARK, lw=0.88, ls=(0, (4, 2)), zorder=1)
    ax.axhline(0, color=MID, lw=0.60, zorder=1)
    ax.axvline(0, color=MID, lw=0.60, zorder=1)
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Median $\Delta KGE$, A", fontsize=10.0)
    ax.set_ylabel(r"Median $\Delta KGE$, B", fontsize=10.0)
    ax.grid(color=LIGHT, lw=0.42, alpha=0.42)
    clean_axes(ax)
    heading(ax, "(e)", "Temporal A/B persistence", fontsize=10.8)
    rank_a = pd.Series(x).rank(method="average").to_numpy()
    rank_b = pd.Series(y).rank(method="average").to_numpy()
    rho = float(np.corrcoef(rank_a, rank_b)[0, 1])
    same_count = int(np.sum(same))
    ax.text(0.045, 0.955, f"Spearman $\\rho$={rho:.3f}\nsame sign={same_count}/36", transform=ax.transAxes, ha="left", va="top", fontsize=8.0, bbox={"facecolor": "white", "edgecolor": LIGHT, "pad": 2.6, "alpha": 0.92})
    ax.legend(loc="upper right", fontsize=7.7, frameon=False, handletextpad=0.3, borderpad=0.05)
    changing = np.where(~same)[0]
    if changing.size:
        extreme = changing[np.argmax(np.maximum(np.abs(x[changing]), np.abs(y[changing])))]
        ax.annotate(str(data.iloc[extreme]["model"]), (x[extreme], y[extreme]), xytext=(-4, 5), textcoords="offset points", ha="right", va="bottom", fontsize=7.6, color=DARK)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    a = pd.read_csv(TABLES_DIR / "Fig1a_model_aggregate_performance.csv").sort_values("plot_order").reset_index(drop=True)
    b = pd.read_csv(TABLES_DIR / "Fig1b_model_delta_distribution.csv").sort_values("plot_order").reset_index(drop=True)
    c = pd.read_csv(TABLES_DIR / "Fig1c_basin_effect_spread.csv").sort_values("plot_order").reset_index(drop=True)
    d = pd.read_csv(TABLES_DIR / "Fig1d_two_way_decomposition.csv")
    e = pd.read_csv(TABLES_DIR / "Fig1e_temporal_model_persistence.csv").sort_values("plot_order").reset_index(drop=True)
    map_metadata = json.loads((CACHE_DIR / "Fig1c_map_metadata.json").read_text())
    if tuple(a["model"]) != MODEL_REGISTRY or tuple(b["model"]) != MODEL_REGISTRY or tuple(e["model"]) != MODEL_REGISTRY:
        raise ValueError("Figure 1 model order is not the canonical registry order")
    if len(c) != 531 or c["basin_id"].nunique() != 531 or c["N_models"].ne(36).any():
        raise ValueError("Figure 1 GIS table does not have exact 531-basin/36-model coverage")
    required_c = {"M_b_median_deltaKGE", "S_b_MAD_deltaKGE", "P_b_positive", "abs_M_b", "color_clipped_flag", "latitude", "longitude"}
    if not required_c.issubset(c.columns) or not np.isfinite(c[["M_b_median_deltaKGE", "S_b_MAD_deltaKGE", "P_b_positive", "latitude", "longitude"]].to_numpy(float)).all():
        raise ValueError("Figure 1 v5 GIS table is incomplete or non-finite")
    if int(map_metadata["n_basins"]) != 531 or int(map_metadata["n_models"]) != 36:
        raise ValueError("Figure 1 v5 map metadata has invalid coverage")
    if not np.isclose(d.loc[d.component != "total", "fraction_of_total"].sum(), 1.0):
        raise ValueError("Figure 1 decomposition shares do not sum to one")
    if not np.isfinite(b["P_m_positive"].to_numpy(float)).all():
        raise ValueError("Figure 1 model tendency values contain non-finite values")
    if not np.array_equal(a["model"].to_numpy(), b["model"].to_numpy()) or not np.array_equal(a["model"].to_numpy(), e["model"].to_numpy()):
        raise ValueError("Panels (a), (b), and (e) are not aligned")
    return a, b, c, d, e, map_metadata


def main() -> None:
    font_family = configure_serif_style()
    a, b, c, d, e, map_metadata = load_tables()
    states, national = load_conus_boundaries()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14.0, 8.5), facecolor="white")
    gs = gridspec.GridSpec(
        2,
        12,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08],
        height_ratios=[1.12, 0.88],
        left=0.040,
        right=0.965,
        bottom=0.085,
        top=0.935,
        wspace=0.10,
        hspace=0.16,
    )
    ax_a = fig.add_subplot(gs[:, 0:3])
    ax_b = fig.add_subplot(gs[:, 3:6])
    ax_c = fig.add_subplot(gs[0, 6:12])
    ax_d = fig.add_subplot(gs[1, 6:9])
    ax_e = fig.add_subplot(gs[1, 9:12])
    draw_panel_a(ax_a, a)
    draw_panel_b(ax_b, b)
    draw_panel_c(fig, ax_c, c, states, national, map_metadata)
    draw_panel_d(ax_d, d)
    draw_panel_e(ax_e, e)
    fig.savefig(OUT_PATH, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"PASS: wrote {OUT_PATH}")
    print(f"font_family={font_family}; png_only=True; dpi=600; panels=5; map=M_b; inset=S_b_MAD")


if __name__ == "__main__":
    main()
