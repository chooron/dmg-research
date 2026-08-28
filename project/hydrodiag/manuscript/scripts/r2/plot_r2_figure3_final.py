#!/usr/bin/env python3
"""Render the revised manuscript-facing canonical R2 Figure 3 (F3).

Layout (6-panel composite in 3 rows x 2 columns):
    Row 0: (a) Parameter-space geometry (IC)       | (b) Parameter-space geometry (dPL)
    Row 1: (c) Snow-organized macro separation (IC)| (d) Snow-organized macro separation (dPL)
    Row 2: (e) Paired Base–CN vs Base–TGD contrast | (f) Combined separation decomposition

All values are read from frozen canonical R2 result tables in `manuscript/analysis/R2/results`.
No upstream analyses are recomputed or modified.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT = Path(__file__).resolve().parents[3]
MANUSCRIPT = PROJECT / "manuscript"
RESULTS_R2 = MANUSCRIPT / "analysis" / "R2" / "results"
DEFAULT_OUTPUT_DIR = MANUSCRIPT / "figures"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from r1_plot_style import (  # noqa: E402
    apply_clean_spines,
    setup_publication_style,
)

OUTPUT_NAME = "Figure3_R2_final.png"
REGIMES = ["S1", "S2", "S3", "S4", "S5"]
REGIME_N = {"S1": 165, "S2": 156, "S3": 121, "S4": 34, "S5": 55}
PARADIGMS = ["IC", "dPL"]

# Color palette: HESS / R1 publication style (matching Figure 1 Okabe-Ito Palette)
COLOR_TEXT = "#303438"
COLOR_REF = "#70767B"
COLOR_GRID = "#E5E8EB"
COLOR_WITHIN = "#555D65"

# Saturated structural contrast palettes
# Base–CN: Blue family (primary estimand)
COLOR_CN = "#0072B2"
COLOR_CN_DARK = "#0B4C79"
COLOR_CN_LIGHT = "#D4E6F1"

# Base–TGD: Teal/Green family (matched control)
COLOR_TGD = "#009E73"
COLOR_TGD_DARK = "#005E44"
COLOR_TGD_LIGHT = "#D5F5E3"

# Distinct Colors for IC and dPL in Panel (e)
COLOR_IC = "#0072B2"   # Okabe-Ito Blue
COLOR_DPL = "#D55E00"  # Okabe-Ito Vermillion / Orange


def _read(name: str) -> pd.DataFrame:
    path = RESULTS_R2 / name
    if not path.exists():
        raise FileNotFoundError(f"canonical R2 source missing: {path}")
    return pd.read_csv(path)


def load_data() -> dict[str, pd.DataFrame]:
    data = {
        "spec_basin": _read("r2_tgd2_specificity_basin_level.csv"),
        "trajectory": _read("r2_s1_s5_macro_trajectory.csv"),
        "prevalence": _read("r2_canonical_prevalence_summary.csv"),
        "delta_summary": _read("r2_paired_cn_tgd_delta_excess_summary.csv"),
        "delta_basin": _read("r2_paired_cn_tgd_delta_excess_basin_level.csv"),
        "slope_diff": _read("r2_tgd2_slope_difference_summary.csv"),
    }
    _assert_schema(data)
    return data


def _assert_schema(data: dict[str, pd.DataFrame]) -> None:
    spec_basin = data["spec_basin"]
    assert set(spec_basin["paradigm"]) == {"IC", "dPL"}
    assert {"Base-CN", "Base-TGD"}.issubset(set(spec_basin["contrast"]))
    assert {"within_pooled", "between_all", "excess"}.issubset(spec_basin.columns)

    traj = data["trajectory"]
    assert set(traj["paradigm"]) == {"IC", "dPL"}
    assert {"Base-CN", "Base-TGD"}.issubset(set(traj["contrast"]))
    assert {"excess_median", "excess_ci_lower", "excess_ci_upper", "between_all_median", "within_pooled_median"}.issubset(traj.columns)

    prev = data["prevalence"]
    assert set(prev["paradigm"]) == {"IC", "dPL"}
    assert {"base_cn_prevalence", "base_tgd_prevalence"}.issubset(prev.columns)

    delta = data["delta_summary"]
    assert set(delta["paradigm"]) == {"IC", "dPL"}
    assert {"median_delta_excess", "ci_lower", "ci_upper", "prop_positive"}.issubset(delta.columns)


# ---------------------------------------------------------------------------
# Panels (a) & (b): Parameter-space geometry (IC & dPL)
# ---------------------------------------------------------------------------
def _plot_geometry_panel(
    ax: plt.Axes,
    data: dict[str, pd.DataFrame],
    paradigm: str,
    letter: str,
    show_ylabel: bool = True,
) -> None:
    df_b = data["spec_basin"]
    prev = data["prevalence"]

    sub_cn = df_b[(df_b["paradigm"] == paradigm) & (df_b["contrast"] == "Base-CN")]
    sub_tg = df_b[(df_b["paradigm"] == paradigm) & (df_b["contrast"] == "Base-TGD")]

    is_ic = (paradigm == "IC")
    if is_ic:
        lim = (0.10, 0.72)
        ticks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    else:
        lim = (0.0, 0.62)
        ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    # 1:1 reference line
    ax.plot([lim[0], lim[1]], [lim[0], lim[1]], color=COLOR_REF, linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)

    # Scatter points: Base-TGD (green triangles, secondary)
    ax.scatter(
        sub_tg["within_pooled"], sub_tg["between_all"],
        s=18, marker="^",
        facecolors="white", edgecolors=COLOR_TGD,
        linewidths=1.2, alpha=0.75,
        rasterized=True, zorder=2,
        label="Base–TGD",
    )

    # Scatter points: Base-CN (blue filled circles, primary)
    ax.scatter(
        sub_cn["within_pooled"], sub_cn["between_all"],
        s=18, marker="o",
        facecolors=COLOR_CN, edgecolors="white",
        linewidths=0.7, alpha=0.75,
        rasterized=True, zorder=3,
        label="Base–CN",
    )

    # Inset prevalence annotation badge
    row_prev = prev[(prev["paradigm"] == paradigm) & (prev["stratum"] == "Full531")].iloc[0]
    p_cn = float(row_prev["base_cn_prevalence"])
    p_tg = float(row_prev["base_tgd_prevalence"])

    badge_text = f"Base–CN:  {p_cn:.1%} above 1:1\nBase–TGD: {p_tg:.1%} above 1:1"
    ax.text(
        0.04, 0.94, badge_text,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8.5, color=COLOR_TEXT,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#CBD5E1", alpha=0.95, linewidth=0.6),
        zorder=5,
    )

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t:.1f}" if t != 0 else "0" for t in ticks], fontsize=9.2)
    if show_ylabel:
        ax.set_yticklabels([f"{t:.1f}" if t != 0 else "0" for t in ticks], fontsize=9.2)
        ax.set_ylabel(r"Between-structure separation, $D_{\mathrm{between}}$", fontsize=10.2)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")

    ax.set_xlabel(r"Within-structure variability, $D_{\mathrm{within}}$", fontsize=10.2)
    ax.grid(True, linestyle=":", linewidth=0.6, color=COLOR_GRID, alpha=0.75)
    ax.set_title(f"({letter}) Parameter-space geometry ({paradigm})", loc="left", weight="bold", fontsize=11.2)
    ax.legend(loc="lower right", ncol=2, frameon=True, facecolor="white", edgecolor="#CBD5E1",
              fontsize=8.5, handlelength=1.2, columnspacing=1.0, framealpha=0.95)


# ---------------------------------------------------------------------------
# Panels (c) & (d): Snow-Organized Macro Separation (Boxplots + Distributions)
# ---------------------------------------------------------------------------
def _plot_macro_separation_boxplots(
    ax: plt.Axes,
    data: dict[str, pd.DataFrame],
    paradigm: str,
    letter: str,
    show_ylabel: bool = True,
) -> None:
    df_b = data["spec_basin"]
    b_sub = df_b[df_b["paradigm"] == paradigm]

    x_pos = np.arange(len(REGIMES))
    width = 0.20  # Reduced width for slimmer, cleaner boxplots
    dx = 0.13     # Reduced dx offset

    # Horizontal zero line
    ax.axhline(0.0, color=COLOR_REF, linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)

    cn_data = []
    tg_data = []

    for i, reg in enumerate(REGIMES):
        cn_pts = b_sub[(b_sub["contrast"] == "Base-CN") & (b_sub["snow_stratum"] == reg)]["excess"].to_numpy(float)
        tg_pts = b_sub[(b_sub["contrast"] == "Base-TGD") & (b_sub["snow_stratum"] == reg)]["excess"].to_numpy(float)
        cn_data.append(cn_pts)
        tg_data.append(tg_pts)

        # Light jittered background points
        np.random.seed(42 + i)
        jit_cn = np.random.uniform(-0.04, 0.04, len(cn_pts))
        jit_tg = np.random.uniform(-0.04, 0.04, len(tg_pts))
        ax.scatter(i - dx + jit_cn, cn_pts, s=4.5, color=COLOR_CN, alpha=0.18, edgecolors="none", rasterized=True, zorder=2)
        ax.scatter(i + dx + jit_tg, tg_pts, s=4.5, color=COLOR_TGD, alpha=0.18, edgecolors="none", rasterized=True, zorder=2)

    # Base-CN Boxplots (Rich Blue)
    ax.boxplot(
        cn_data,
        positions=x_pos - dx,
        widths=width,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=COLOR_CN, edgecolor=COLOR_CN_DARK, alpha=0.62, linewidth=1.3),
        medianprops=dict(color=COLOR_CN_DARK, linewidth=2.2),
        whiskerprops=dict(color=COLOR_CN_DARK, linewidth=1.3),
        capprops=dict(color=COLOR_CN_DARK, linewidth=1.3),
        zorder=3,
    )

    # Base-TGD Boxplots (Rich Teal/Green)
    ax.boxplot(
        tg_data,
        positions=x_pos + dx,
        widths=width,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=COLOR_TGD, edgecolor=COLOR_TGD_DARK, alpha=0.58, linewidth=1.3),
        medianprops=dict(color=COLOR_TGD_DARK, linewidth=2.2),
        whiskerprops=dict(color=COLOR_TGD_DARK, linewidth=1.3),
        capprops=dict(color=COLOR_TGD_DARK, linewidth=1.3),
        zorder=3,
    )

    ax.set_xticks(x_pos)
    xtick_labels = [f"{r}\n(n={REGIME_N[r]})" for r in REGIMES]
    ax.set_xticklabels(xtick_labels, fontsize=9.2)
    ax.set_xlim(-0.5, len(REGIMES) - 0.5)

    if paradigm == "IC":
        ax.set_ylim(-0.08, 0.36)
    else:
        ax.set_ylim(-0.08, 0.44)
    ax.set_xlabel("Snow activity stratum", fontsize=10.2)
    if show_ylabel:
        ax.set_ylabel(r"Excess distance, $D_{\mathrm{between}} - D_{\mathrm{within}}$", fontsize=10.2)
        ax.tick_params(axis="y", labelsize=9.2)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.tick_params(axis="y", labelsize=9.2)

    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, color=COLOR_GRID, alpha=0.75)
    ax.set_title(f"({letter}) Snow-organized macro separation ({paradigm})", loc="left", weight="bold", fontsize=11.2)

    # Legend in panel: Panel (c) at lower right, Panel (d) at upper left
    handles = [
        Patch(facecolor=COLOR_CN, edgecolor=COLOR_CN_DARK, alpha=0.62, label="Base–CN"),
        Patch(facecolor=COLOR_TGD, edgecolor=COLOR_TGD_DARK, alpha=0.58, label="Base–TGD"),
    ]
    if paradigm == "IC":
        ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.98, 0.08), frameon=True, facecolor="white", edgecolor="#CBD5E1",
                  fontsize=8.5, handlelength=1.2, columnspacing=1.0, framealpha=0.95)
    else:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.04, 0.96), frameon=True, facecolor="white", edgecolor="#CBD5E1",
                  fontsize=8.5, handlelength=1.2, columnspacing=1.0, framealpha=0.95)

# ---------------------------------------------------------------------------
# Panel (e): Paired Base–CN vs Base–TGD Contrast
# ---------------------------------------------------------------------------
def _plot_paired_contrast_forest_panel(
    ax: plt.Axes,
    data: dict[str, pd.DataFrame],
) -> None:
    df_delta = data["delta_summary"]

    # All 14 required entries: dPL on top (Orange), IC on bottom (Blue)
    rows_def = [
        ("dPL", "Full531", "dPL Full", COLOR_DPL, "s", True),
        ("dPL", "ExcludeS5", "dPL Excl. S5", COLOR_DPL, "s", False),
        ("dPL", "S5", "dPL S5", COLOR_DPL, "o", False),
        ("dPL", "S4", "dPL S4", COLOR_DPL, "o", False),
        ("dPL", "S3", "dPL S3", COLOR_DPL, "o", False),
        ("dPL", "S2", "dPL S2", COLOR_DPL, "o", False),
        ("dPL", "S1", "dPL S1", COLOR_DPL, "o", False),
        # Gap between dPL and IC
        ("IC", "Full531", "IC Full", COLOR_IC, "s", True),
        ("IC", "ExcludeS5", "IC Excl. S5", COLOR_IC, "s", False),
        ("IC", "S5", "IC S5", COLOR_IC, "o", False),
        ("IC", "S4", "IC S4", COLOR_IC, "o", False),
        ("IC", "S3", "IC S3", COLOR_IC, "o", False),
        ("IC", "S2", "IC S2", COLOR_IC, "o", False),
        ("IC", "S1", "IC S1", COLOR_IC, "o", False),
    ]

    y_positions = [
        14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0,
        6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0,
    ]

    ax.axvline(0.0, color=COLOR_REF, linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    ax.axhline(7.0, color="#CBD5E1", linestyle=":", linewidth=0.8, zorder=1)
    ax.text(0.128, 14.95, r"$\mathrm{Prop}>0$", fontsize=8.8, fontweight="bold", color=COLOR_TEXT, ha="center", va="bottom")

    for y_p, (p, st, label, col, mk, is_filled) in zip(y_positions, rows_def):
        sub = df_delta[(df_delta["paradigm"] == p) & (df_delta["stratum"] == st)]
        assert len(sub) == 1, f"missing {p} {st}"
        r = sub.iloc[0]
        med = float(r["median_delta_excess"])
        lo = float(r["ci_lower"])
        hi = float(r["ci_upper"])
        prop = float(r["prop_positive"])

        is_full = ("Full" in label)
        ax.errorbar(
            med, y_p,
            xerr=[[med - lo], [hi - med]],
            fmt=mk, color=col, ecolor=col,
            markersize=5.8 if is_full else 4.8,
            markerfacecolor=col if is_filled else "white",
            markeredgecolor=col, markeredgewidth=1.3,
            elinewidth=1.5, capsize=3.0, capthick=1.2,
            zorder=3,
        )

        ax.text(
            0.128, y_p, f"{prop:.1%}",
            fontsize=8.5, color=COLOR_TEXT,
            ha="center", va="center", zorder=4,
        )

    labels = [r[2] for r in rows_def]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9.0)
    ax.set_xlim(-0.035, 0.160)
    ax.set_ylim(-0.8, 15.8)
    ax.tick_params(axis="x", labelsize=9.2)
    ax.set_xlabel(r"Paired contrast, $\delta_{\mathrm{excess}}$ (95% CI)", fontsize=10.2)
    ax.set_ylabel("Subset", fontsize=10.2)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, color=COLOR_GRID, alpha=0.75)
    ax.set_title(r"(e) Paired Base–CN vs Base–TGD contrast", loc="left", weight="bold", fontsize=11.2)

    # Inset Legend in Panel (e) for dPL (Orange) vs IC (Blue)
    leg_e = [
        Line2D([0], [0], color=COLOR_DPL, marker="s", markersize=5.5, markerfacecolor=COLOR_DPL, markeredgecolor=COLOR_DPL, lw=1.5, label="dPL (neural)"),
        Line2D([0], [0], color=COLOR_IC, marker="s", markersize=5.5, markerfacecolor=COLOR_IC, markeredgecolor=COLOR_IC, lw=1.5, label="IC (CMA-ES)"),
    ]
    ax.legend(handles=leg_e, loc="lower left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.0, framealpha=0.95)


# ---------------------------------------------------------------------------
# Panel (f): Combined Separation Decomposition
# ---------------------------------------------------------------------------
def _plot_combined_decomposition_panel(
    ax: plt.Axes,
    data: dict[str, pd.DataFrame],
) -> None:
    traj = data["trajectory"]
    x_idx = np.arange(len(REGIMES))
    dx = 0.15

    for p, sign, ls, alpha_line in [("IC", -dx, "-", 1.0), ("dPL", +dx, "--", 0.90)]:
        t_cn = traj[(traj["paradigm"] == p) & (traj["contrast"] == "Base-CN")].set_index("snow_stratum").loc[REGIMES].reset_index()
        t_tg = traj[(traj["paradigm"] == p) & (traj["contrast"] == "Base-TGD")].set_index("snow_stratum").loc[REGIMES].reset_index()

        w_pts = t_cn["within_pooled_median"].to_numpy(float)
        b_cn_pts = t_cn["between_all_median"].to_numpy(float)
        b_tg_pts = t_tg["between_all_median"].to_numpy(float)

        x_coords = x_idx + sign

        # Connect medians across strata
        ax.plot(x_coords, b_cn_pts, color=COLOR_CN, linestyle=ls, linewidth=1.8, alpha=alpha_line, zorder=2)
        ax.plot(x_coords, b_tg_pts, color=COLOR_TGD, linestyle=ls, linewidth=1.8, alpha=alpha_line, zorder=2)
        ax.plot(x_coords, w_pts, color=COLOR_WITHIN, linestyle=ls, linewidth=1.5, alpha=alpha_line * 0.85, zorder=2)

        # Vertical dumbbell lines within each stratum
        for xc, w, b_cn, b_tg in zip(x_coords, w_pts, b_cn_pts, b_tg_pts):
            ax.plot([xc, xc], [w, max(b_cn, b_tg)], color="#A0AAB2", linewidth=1.2, zorder=1)
            ax.plot(xc, w, marker="o", markersize=5.0, color=COLOR_WITHIN, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            ax.plot(xc, b_cn, marker="s", markersize=5.6, color=COLOR_CN, markeredgecolor=COLOR_CN_DARK, markeredgewidth=0.9, zorder=4)
            ax.plot(xc, b_tg, marker="^", markersize=5.8, color=COLOR_TGD, markeredgecolor=COLOR_TGD_DARK, markeredgewidth=0.9, zorder=4)

    ax.set_xticks(x_idx)
    ax.set_xticklabels(REGIMES, fontsize=9.2)
    ax.set_xlim(-0.5, len(REGIMES) - 0.5)
    ax.set_ylim(0.0, 0.80)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8"], fontsize=9.2)
    ax.tick_params(axis="both", labelsize=9.2)

    ax.set_xlabel("Snow stratum", fontsize=10.2)
    ax.set_ylabel("Distance (RMS)", fontsize=10.2)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, color=COLOR_GRID, alpha=0.75)
    ax.set_title("(f) Combined separation decomposition", loc="left", weight="bold", fontsize=11.2)

    # Consolidated legend
    handles = [
        Line2D([0], [0], marker="s", color=COLOR_CN, markerfacecolor=COLOR_CN, markeredgecolor=COLOR_CN_DARK, markersize=5.2, linestyle="-", linewidth=1.8, label="Between (Base–CN)"),
        Line2D([0], [0], marker="^", color=COLOR_TGD, markerfacecolor=COLOR_TGD, markeredgecolor=COLOR_TGD_DARK, markersize=5.4, linestyle="-", linewidth=1.8, label="Between (Base–TGD)"),
        Line2D([0], [0], marker="o", color=COLOR_WITHIN, markerfacecolor=COLOR_WITHIN, markeredgecolor="white", markersize=4.8, linestyle="-", linewidth=1.5, label=r"Within baseline ($D_{\mathrm{within}}$)"),
        Line2D([0], [0], color=COLOR_TEXT, linestyle="-", linewidth=1.4, label="IC (solid, left)"),
        Line2D([0], [0], color=COLOR_TEXT, linestyle="--", linewidth=1.4, label="dPL (dashed, right)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.0, ncol=2, framealpha=0.95)


# ---------------------------------------------------------------------------
# Figure Assembly (Clean 3 Rows x 2 Columns Composite)
# ---------------------------------------------------------------------------
def build_figure(data: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    fig = plt.figure(figsize=(13.2, 11.4))

    # GridSpec with generous vertical spacing and tighter horizontal gaps
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1.0, 1.05, 1.15],
        hspace=0.38,
        wspace=0.14,
        left=0.075,
        right=0.975,
        top=0.965,
        bottom=0.055,
    )

    # Row 0: Panels (a) & (b) Geometry
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    apply_clean_spines(ax_a)
    apply_clean_spines(ax_b)

    _plot_geometry_panel(ax_a, data, "IC", "a", show_ylabel=True)
    _plot_geometry_panel(ax_b, data, "dPL", "b", show_ylabel=False)

    # Row 1: Panels (c) & (d) Macro separation
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    apply_clean_spines(ax_c)
    apply_clean_spines(ax_d)

    _plot_macro_separation_boxplots(ax_c, data, "IC", "c", show_ylabel=True)
    _plot_macro_separation_boxplots(ax_d, data, "dPL", "d", show_ylabel=False)

    # Row 2: Panels (e) & (f)
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])
    apply_clean_spines(ax_e)
    apply_clean_spines(ax_f)

    _plot_paired_contrast_forest_panel(ax_e, data)
    _plot_combined_decomposition_panel(ax_f, data)

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / OUTPUT_NAME
    fig.savefig(output, dpi=600, facecolor="white")
    plt.close(fig)
    return output


def render(out_dir: Path | None = None) -> Path:
    setup_publication_style()
    data = load_data()
    return build_figure(data, out_dir or DEFAULT_OUTPUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    print(render(args.out_dir))


if __name__ == "__main__":
    main()
