#!/usr/bin/env python3
"""Supplementary Figure S1: R4 Multi-Catchment External-State Consistency Validation.

Generates a publication-grade 7-panel composite figure:
- Panels (a–f): 6-basin small-multiple trajectories (3 rows x 2 columns) across
  Low, Middle, and High external SWE burden terciles.
- Panel (g): Full-width, well-proportioned population-level anchor panel evaluating
  paired anomaly correlation contrasts (CN–Base and TGD–Base) across eligible
  catchments under the identical snowiest-year snow-active episode sampling protocol.

All data are read from pre-computed canonical audit files:
- `manuscript/supplement/figures/FigureS1_R4_population_audit.csv`
- `manuscript/supplement/figures/FigureS1_R4_selection_audit.json`

Outputs:
  manuscript/supplement/figures/FigureS1_R4_multibasin_validation.png (300 DPI PNG)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SHARED = PROJECT_ROOT / "manuscript" / "scripts" / "shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from r1_plot_style import (
    COLOR_BASE,
    COLOR_CN,
    COLOR_DARK_NEUTRAL,
    COLOR_LIGHT_REF,
    COLOR_OBSERVATION,
    COLOR_TGD,
    COLOR_ZERO_LINE,
    apply_clean_spines,
    setup_publication_style,
)
from manuscript.scripts.r4.soil_analysis import calendar_month_anomaly

RESULTS_ROOT = PROJECT_ROOT / "results"
SUPPLEMENT_FIG_DIR = PROJECT_ROOT / "manuscript" / "supplement" / "figures"
SUPPLEMENT_FIG_DIR.mkdir(parents=True, exist_ok=True)

MONTH_LABELS = ["O", "N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S"]
EXAMPLE_BASIN_IDS = ["02472000", "07195800", "05495000", "03473000", "12167000", "08377900"]


def calc_standardized_anomaly(arr: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Compute standardized calendar-month anomalies along axis 1."""
    months = dates.month.to_numpy()
    anom = np.empty_like(arr, dtype=float)
    for m in range(1, 13):
        m_mask = (months == m)
        anom[:, m_mask] = arr[:, m_mask] - np.nanmean(arr[:, m_mask], axis=1, keepdims=True)
    stds = np.nanstd(anom, axis=1, keepdims=True)
    stds[stds < 1e-9] = 1.0
    return (anom - np.nanmean(anom, axis=1, keepdims=True)) / stds


def load_raw_arrays():
    caravan = np.load(RESULTS_ROOT / "r4_caravan_soil_reference_v1" / "caravan_soil_ensemble.npz")
    test = slice(int(caravan["test_slice_start"]), int(caravan["test_slice_stop"]))
    dates = pd.to_datetime(caravan["dates"][test])
    basin_ids = [str(x).zfill(8) for x in caravan["basin_ids"]]
    sm100 = caravan["SM100"][:, test].astype(float)
    swe = np.load(RESULTS_ROOT / "r4_swe_reference_v1" / "swe_ensemble.npz")["swe_median"][:, test].astype(float)

    tgd_dpl_path = RESULTS_ROOT / "r4_replay_dpl_XAJ_TGD2_seed42/reconstructed_dpl_XAJ_TGD2_seed42_full_arrays.npz"
    if not tgd_dpl_path.exists():
        tgd_dpl_path = RESULTS_ROOT / "r4_official_dpl_XAJ_TGD2_seed42/official_dpl_XAJ_TGD2_seed42_full_arrays.npz"

    npz_base = np.load(RESULTS_ROOT / "r4_official_dpl_XAJ_seed42/official_dpl_XAJ_seed42_full_arrays.npz")
    npz_tgd = np.load(tgd_dpl_path)
    npz_cn = np.load(RESULTS_ROOT / "r4_official_dpl_XAJ_CN_seed42/official_dpl_XAJ_CN_seed42_full_arrays.npz")

    w_base = (npz_base["wu"][:, test] + npz_base["wl"][:, test] + npz_base["wd"][:, test]).astype(float)
    w_tgd = (npz_tgd["wu"][:, test] + npz_tgd["wl"][:, test] + npz_tgd["wd"][:, test]).astype(float)
    w_cn = (npz_cn["wu"][:, test] + npz_cn["wl"][:, test] + npz_cn["wd"][:, test]).astype(float)

    return basin_ids, dates, sm100, swe, {"Base": w_base, "TGD": w_tgd, "CN": w_cn}


def main():
    setup_publication_style()

    # Load audit files
    audit_json_path = SUPPLEMENT_FIG_DIR / "FigureS1_R4_selection_audit.json"
    audit_csv_path = SUPPLEMENT_FIG_DIR / "FigureS1_R4_population_audit.csv"
    if not audit_json_path.exists() or not audit_csv_path.exists():
        from manuscript.scripts.r4.build_figure_s1_population_audit import build_audit_data
        build_audit_data()

    audit_json = json.loads(audit_json_path.read_text(encoding="utf-8"))
    df_pop = pd.read_csv(audit_csv_path)
    df_pop["basin_id"] = df_pop["basin_id"].astype(str).str.zfill(8)

    example_configs = audit_json["selected_example_basins"]
    pop_summaries = audit_json["population_summaries"]

    basin_ids, dates, sm100, swe, models = load_raw_arrays()
    d = pd.DatetimeIndex(dates)
    wy = np.where(d.month >= 10, d.year + 1, d.year).astype(int)

    # Standardize anomalies across full evaluation period
    anom_sm = calc_standardized_anomaly(sm100, d)
    anom_base = calc_standardized_anomaly(models["Base"], d)
    anom_tgd = calc_standardized_anomaly(models["TGD"], d)
    anom_cn = calc_standardized_anomaly(models["CN"], d)

    # ---------------------------------------------------------------------------
    # Figure Layout: 17.8 cm width x 21.8 cm height
    # Two-level GridSpec:
    # - Top block: 3 rows x 2 cols (Panels a–f)
    # - Bottom block: 1 row x 1 col (Panel g)
    # - Generous vertical separation hspace=0.38 between Top and Bottom blocks
    # ---------------------------------------------------------------------------
    fig_w = 17.8 / 2.54
    fig_h = 21.8 / 2.54
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs_outer = gridspec.GridSpec(
        2, 1,
        height_ratios=[3.0, 1.10],
        hspace=0.28,
        top=0.915,
        bottom=0.055,
        left=0.08,
        right=0.97,
    )

    gs_top = gs_outer[0].subgridspec(3, 2, hspace=0.28, wspace=0.18)
    gs_bot = gs_outer[1].subgridspec(1, 1)

    letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    # --- Panels (a–f): 6 Trajectory Subplots ---
    for i, cfg in enumerate(example_configs):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs_top[row, col])
        apply_clean_spines(ax)

        b_id = cfg["basin_id"]
        b_idx = basin_ids.index(b_id)
        group = cfg["group"]
        sel_wy = cfg["selected_water_year"]
        pk_swe = cfg["water_year_peak_swe_mm"]

        wy_mask = (wy == sel_wy)
        wy_doy = np.arange(1, wy_mask.sum() + 1)
        wy_swe = swe[b_idx, wy_mask]

        ref_anom = anom_sm[b_idx, wy_mask]
        base_anom = anom_base[b_idx, wy_mask]
        tgd_anom = anom_tgd[b_idx, wy_mask]
        cn_anom = anom_cn[b_idx, wy_mask]

        # Shading for snow-active window (SWE >= 5.0 mm)
        snow_active = wy_swe >= 5.0
        if snow_active.any():
            active_doy = wy_doy[snow_active]
            ax.axvspan(active_doy[0], active_doy[-1], color="#EBF5FB", alpha=0.80, zorder=0)

        # Zero reference line
        ax.axhline(0, color=COLOR_LIGHT_REF, lw=0.6, ls=":", zorder=1)

        # Trajectory lines with clean dash patterns and line widths
        ax.plot(wy_doy, ref_anom, color="#2B2B2B", lw=1.25, ls="-", zorder=3)
        ax.plot(wy_doy, base_anom, color=COLOR_BASE, lw=1.25, ls=(0, (4, 2)), zorder=4)
        ax.plot(wy_doy, tgd_anom, color=COLOR_TGD, lw=1.25, ls=(0, (4, 1.5, 1, 1.5)), zorder=5)
        ax.plot(wy_doy, cn_anom, color=COLOR_CN, lw=1.35, ls="-", zorder=6)

        # 1. Panel letter placed OUTSIDE top-left of axes (8.8 pt, bold, unified with (g))
        ax.text(
            -0.07, 1.04, letters[i],
            transform=ax.transAxes,
            fontsize=8.8, fontweight="bold", color=COLOR_DARK_NEUTRAL,
            va="bottom", ha="left", clip_on=False
        )

        # 2. Unified, light, two-tier basin metadata box INSIDE top-left (no panel letter)
        basin_txt = f"Basin {b_id} ({group} SWE)"
        sub_txt = f"WY {sel_wy} | Peak: {pk_swe:.1f} mm"
        ax.text(
            0.035, 0.935, f"{basin_txt}\n{sub_txt}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=6.8, color=COLOR_DARK_NEUTRAL,
            linespacing=1.18,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", edgecolor="#D5DBDF", alpha=0.85, lw=0.35),
            zorder=8,
        )

        ax.set_xlim(1, 365)
        ax.set_ylim(-3.0, 3.2)
        ax.set_yticks([-2.0, 0.0, 2.0])
        ax.tick_params(labelsize=7.5)

        if col == 0:
            ax.set_ylabel("Standardized anomaly", fontsize=8.0)
        if row == 2:
            month_mid_doy = [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]
            ax.set_xticks(month_mid_doy)
            ax.set_xticklabels(MONTH_LABELS, fontsize=7.5)
            ax.set_xlabel("Day of water year (Oct–Sep)", fontsize=8.0)
        else:
            ax.set_xticklabels([])

    # --- Panel (g): Full-Width Population Summary Panel ---
    ax_pop = fig.add_subplot(gs_bot[0, 0])
    apply_clean_spines(ax_pop)

    groups = ["Low", "Middle", "High"]
    group_labels = [
        f"Low SWE\n(N = {pop_summaries['Low']['eligible_basins']})",
        f"Middle SWE\n(N = {pop_summaries['Middle']['eligible_basins']})",
        f"High SWE\n(N = {pop_summaries['High']['eligible_basins']})",
    ]

    x_centers = np.array([0, 1, 2], dtype=float)
    dx = 0.16  # offset for CN vs TGD
    rng = np.random.default_rng(20260730)

    ax_pop.axhline(0, color=COLOR_ZERO_LINE, lw=0.75, ls="--", zorder=1)

    for i, grp in enumerate(groups):
        sub_el = df_pop[(df_pop["swe_burden_group"] == grp) & (df_pop["eligible"])]
        d_cn = sub_el["delta_r_CN_Base"].dropna().values
        d_tgd = sub_el["delta_r_TGD_Base"].dropna().values

        # Light background jitter points
        jit_cn = rng.uniform(-0.05, 0.05, len(d_cn))
        jit_tgd = rng.uniform(-0.05, 0.05, len(d_tgd))

        ax_pop.scatter(
            x_centers[i] - dx + jit_cn, d_cn,
            s=7.5, color=COLOR_CN, alpha=0.18, edgecolors="none", rasterized=True, zorder=2
        )
        ax_pop.scatter(
            x_centers[i] + dx + jit_tgd, d_tgd,
            s=7.5, color=COLOR_TGD, alpha=0.18, edgecolors="none", rasterized=True, zorder=2
        )

        # Median + 95% Bootstrap CI
        cn_stats = pop_summaries[grp]["delta_r_CN_Base"]
        tgd_stats = pop_summaries[grp]["delta_r_TGD_Base"]

        med_c = cn_stats["median"]
        lo_c, hi_c = cn_stats["ci_95"]
        med_t = tgd_stats["median"]
        lo_t, hi_t = tgd_stats["ci_95"]

        ax_pop.errorbar(
            x_centers[i] - dx, med_c,
            yerr=[[med_c - lo_c], [hi_c - med_c]],
            fmt="o", color=COLOR_CN, mfc=COLOR_CN, mec="white", mew=0.9,
            ecolor=COLOR_CN, elinewidth=1.5, capsize=3.0, capthick=1.1, markersize=5.8, zorder=5,
            label="CN − Base (population median + 95% CI)" if i == 0 else ""
        )
        ax_pop.errorbar(
            x_centers[i] + dx, med_t,
            yerr=[[med_t - lo_t], [hi_t - med_t]],
            fmt="^", color=COLOR_TGD, mfc=COLOR_TGD, mec="white", mew=0.9,
            ecolor=COLOR_TGD, elinewidth=1.5, capsize=3.0, capthick=1.1, markersize=5.8, zorder=5,
            label="TGD − Base (population median + 95% CI)" if i == 0 else ""
        )

        # Neat numerical badge positioned above each group column
        badge_txt = (
            f"CN − Base:  {med_c:+.3f} [{lo_c:+.2f}, {hi_c:+.2f}]\n"
            f"TGD − Base: {med_t:+.3f} [{lo_t:+.2f}, {hi_t:+.2f}]"
        )
        ax_pop.text(
            x_centers[i], 1.25, badge_txt,
            ha="center", va="top", fontsize=6.5, family="sans-serif", color=COLOR_DARK_NEUTRAL,
            linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="#FAFBFC", edgecolor="#D5DBDF", alpha=0.90, lw=0.35),
            zorder=6,
        )

    # Highlight the 6 example basins with distinct outlined markers
    for cfg in example_configs:
        bid = cfg["basin_id"]
        grp = cfg["group"]
        grp_idx = groups.index(grp)
        row = df_pop[df_pop["basin_id"] == bid].iloc[0]
        if row["eligible"]:
            d_c_val = row["delta_r_CN_Base"]
            d_t_val = row["delta_r_TGD_Base"]
            ax_pop.scatter(
                grp_idx - dx, d_c_val,
                s=34, facecolors="none", edgecolors=COLOR_DARK_NEUTRAL, linewidths=1.15, zorder=7
            )
            ax_pop.scatter(
                grp_idx + dx, d_t_val,
                s=34, facecolors="none", edgecolors=COLOR_DARK_NEUTRAL, linewidths=1.15, zorder=7
            )

    ax_pop.set_xticks(x_centers)
    ax_pop.set_xticklabels(group_labels, fontsize=7.8)
    ax_pop.set_xlim(-0.5, 2.5)
    ax_pop.set_ylim(-1.45, 1.45)
    ax_pop.set_ylabel(r"$\Delta r$ (vs. $\mathrm{SM}_{100}$)", fontsize=8.0)
    ax_pop.set_xlabel("External Snow-17 SWE burden tercile", fontsize=8.0)

    # 1. Place "(g)" outside top-left of axes (8.8 pt, bold, unified with (a)–(f))
    ax_pop.text(
        -0.07, 1.04, "(g)",
        transform=ax_pop.transAxes,
        fontsize=8.8, fontweight="bold", color=COLOR_DARK_NEUTRAL,
        va="bottom", ha="left", clip_on=False
    )

    # 2. Place shortened descriptive title alongside (g) outside top
    ax_pop.text(
        -0.005, 1.04, f"External-state consistency across snowiest episodes ($N = {int(df_pop['eligible'].sum())}$)",
        transform=ax_pop.transAxes,
        fontsize=8.0, fontweight="bold", color=COLOR_DARK_NEUTRAL,
        va="bottom", ha="left", clip_on=False
    )

    # 3. Clean note explaining outlined markers
    ax_pop.text(
        0.015, 0.05, "Outlined markers: example catchments from panels (a–f)",
        transform=ax_pop.transAxes, fontsize=6.6, fontstyle="italic", color="#555555"
    )

    # Unified Legend on Top of Figure
    handles = [
        Line2D([0], [0], color="#2B2B2B", lw=1.25, ls="-", label=r"ERA5-Land $\mathrm{SM}_{100}$ (external reference)"),
        Line2D([0], [0], color=COLOR_BASE, lw=1.25, ls="--", label=r"Base $W_{\mathrm{total}}$ (knockout)"),
        Line2D([0], [0], color=COLOR_TGD, lw=1.25, ls="-.", label=r"TGD $W_{\mathrm{total}}$ (generic control)"),
        Line2D([0], [0], color=COLOR_CN, lw=1.35, ls="-", label=r"CN $W_{\mathrm{total}}$ (explicit snow)"),
        Patch(facecolor="#EBF5FB", edgecolor="#BCE2F5", lw=0.5, label=r"Snow-active period ($\mathrm{SWE} \geq 5$ mm)"),
        Line2D([0], [0], marker="o", color="none", markeredgecolor=COLOR_DARK_NEUTRAL, markersize=5.0, label="Example catchment anchor"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.525, 0.988),
        ncol=3,
        frameon=False,
        fontsize=7.2,
        columnspacing=1.0,
        handlelength=1.5,
    )

    out_png = SUPPLEMENT_FIG_DIR / "FigureS1_R4_multibasin_validation.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()

    print(f"Canonical Figure S1 updated successfully:\n  {out_png}")


if __name__ == "__main__":
    main()
