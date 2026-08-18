"""Generate Figure 8: Spring soil-water recharge timing in real basins.

Figure 7 localizes the Base--CN internal-state separation in high-snow environments
and during active melt. Figure 8 demonstrates how this separation manifests on the
temporal axis: snow storage -> melt delivery -> soil-water response -> wet-up / peak timing.

Layout (6-panel asymmetric composite layout following HESS / WRR guidelines):
  - (a) Snow accumulation and depletion (illustrative high-snow water year, Oct–Sep)
  - (b) Standardized soil-water trajectories (same water year, shared x-axis with a)
  - (c) MAIN: Spring process zoom with event timing rails (Jan–Jun)
  - (d) Basin-level spring wet-up timing distribution (Q3 subset, n=133)
  - (e) Basin-level soil-water peak timing distribution (Q3 subset, n=133)
  - (f) Timing-definition sensitivity synthesis rail (Base MAE − CN MAE [days])

Outputs:
    manuscript/figures/figure8_r4_soil_timing.png (300 DPI, PNG only - no PDF)
    manuscript/scripts/figure8_r4_selection_audit.json (audit of illustrative case)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript.scripts.r1_plot_style import (  # noqa: E402
    apply_clean_spines,
    setup_publication_style,
)
from r4.common import default_results_root  # noqa: E402
from r4.soil_analysis import calendar_month_anomaly  # noqa: E402

FIGURES_DIR = HERE.parents[0] / "figures"
CANONICAL_DPL = "dPL_seed42"
REGIMES = [CANONICAL_DPL, "IC_fused"]
REG_SHORT = ["dPL-42", "IC fused"]
REGIME_CFG = {
    CANONICAL_DPL: {
        "label": "dPL (seed 42)",
        "color": "#882E72",
        "marker": "o",
        "ls": "-",
    },
    "IC_fused": {"label": "IC (fused)", "color": "#333333", "marker": "D", "ls": "-."},
}
STATE_CFG = {
    "Reference": {
        "label": "ERA5-Land SM$_{100}$ reference",
        "color": "#555555",
        "ls": "-",
        "lw": 1.2,
    },
    "Base": {"label": "Base $W_{total}$", "color": "#2878B5", "ls": "--", "lw": 1.2},
    "CN": {"label": "CN $W_{total}$", "color": "#D95F02", "ls": "-", "lw": 1.3},
}
Q3_QUANTILE = 0.75


def _standardized_month_anomaly(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    months = pd.to_datetime(dates).month.to_numpy()
    anomaly = calendar_month_anomaly(values.astype(float), months)
    return (anomaly - np.nanmean(anomaly)) / (np.nanstd(anomaly) + 1e-12)


def _water_year_doy_to_date(water_year: int, doy: float) -> pd.Timestamp:
    return pd.Timestamp(year=int(water_year) - 1, month=10, day=1) + pd.Timedelta(
        days=float(doy) - 1
    )


def _select_illustrative_basin_year(results_root: Path) -> tuple[str, int, dict]:
    official = results_root / "r4_phase1_soil_official"
    summary = pd.read_csv(
        official / "timing_metrics_basin_summary.csv", dtype={"basin_id": str}
    )
    state = pd.read_csv(
        official / "basin_state_consistency.csv", dtype={"basin_id": str}
    )
    basin_swe = state[
        (state["regime"] == CANONICAL_DPL) & (state["structure"] == "Base")
    ]
    basin_swe = basin_swe.drop_duplicates("basin_id")[
        ["basin_id", "snow_burden_swe_mm"]
    ]
    q3_threshold = float(basin_swe["snow_burden_swe_mm"].quantile(Q3_QUANTILE))
    q3_n = int((basin_swe["snow_burden_swe_mm"] >= q3_threshold).sum())

    fields = [
        "n_valid_snow_years",
        "median_abs_wetup_error_days",
        "median_abs_peak_error_days",
    ]
    pivot = summary[summary["regime"] == CANONICAL_DPL].pivot(
        index="basin_id", columns="structure", values=fields
    )
    candidates = pd.DataFrame(index=pivot.index)
    for field in fields:
        candidates[f"Base_{field}"] = pivot[(field, "Base")]
        candidates[f"CN_{field}"] = pivot[(field, "CN")]
    candidates = candidates.reset_index().merge(
        basin_swe, on="basin_id", validate="one_to_one"
    )
    candidates["wetup_improvement_days"] = (
        candidates["Base_median_abs_wetup_error_days"]
        - candidates["CN_median_abs_wetup_error_days"]
    )
    candidates["peak_improvement_days"] = (
        candidates["Base_median_abs_peak_error_days"]
        - candidates["CN_median_abs_peak_error_days"]
    )
    candidates = candidates[
        (candidates["snow_burden_swe_mm"] >= q3_threshold)
        & (candidates["Base_n_valid_snow_years"] >= 10)
        & (candidates["CN_n_valid_snow_years"] >= 10)
    ].copy()
    target = candidates[
        ["wetup_improvement_days", "peak_improvement_days", "snow_burden_swe_mm"]
    ].median()
    scale = (
        candidates[
            ["wetup_improvement_days", "peak_improvement_days", "snow_burden_swe_mm"]
        ]
        .std()
        .replace(0, 1)
    )
    candidates["selection_score"] = sum(
        ((candidates[col] - target[col]).abs() / scale[col])
        for col in [
            "wetup_improvement_days",
            "peak_improvement_days",
            "snow_burden_swe_mm",
        ]
    )
    selected = candidates.sort_values(["selection_score", "basin_id"]).iloc[0]
    basin_id = str(selected["basin_id"]).zfill(8)

    year = pd.read_csv(
        official / "timing_metrics_basin_year.csv", dtype={"basin_id": str}
    )
    year = year[
        (year["regime"] == CANONICAL_DPL)
        & (year["basin_id"] == basin_id)
        & year["is_snow_year"]
    ]
    year_rows = []
    for water_year, group in year.groupby("water_year"):
        if set(group["structure"]) != {"Base", "CN"}:
            continue
        g = group.set_index("structure")
        year_rows.append(
            {
                "water_year": int(water_year),
                "annual_swe_max_mm": float(g.loc["Base", "annual_swe_max_mm"]),
                "wetup_improvement_days": abs(
                    float(g.loc["Base", "wetup_timing_error_days"])
                )
                - abs(float(g.loc["CN", "wetup_timing_error_days"])),
                "peak_improvement_days": abs(
                    float(g.loc["Base", "peak_timing_error_days"])
                )
                - abs(float(g.loc["CN", "peak_timing_error_days"])),
            }
        )
    years = pd.DataFrame(year_rows)
    year_target = np.array(
        [selected["wetup_improvement_days"], selected["peak_improvement_days"]],
        dtype=float,
    )
    year_scale = (
        years[["wetup_improvement_days", "peak_improvement_days"]]
        .std()
        .replace(0, 1)
        .to_numpy()
    )
    years["selection_score"] = (
        years["wetup_improvement_days"] - year_target[0]
    ).abs() / year_scale[0] + (
        years["peak_improvement_days"] - year_target[1]
    ).abs() / year_scale[1]
    selected_year = int(
        years.sort_values(["selection_score", "water_year"]).iloc[0]["water_year"]
    )
    audit = {
        "case_description": "Illustrative Q3 high-snow basin-year",
        "selection_rule": "Among Q3 basins with >=10 valid snow years, minimize standardized distance from Q3 medians of wet-up improvement, peak improvement, and basin SWE burden; then select the year minimizing distance from that basin's median improvements.",
        "q3_threshold_mm": q3_threshold,
        "q3_basin_count": q3_n,
        "q3_population_medians": {k: float(v) for k, v in target.items()},
        "selected_basin_id": basin_id,
        "selected_basin_swe_burden_mm": float(selected["snow_burden_swe_mm"]),
        "selected_basin_valid_snow_years": int(selected["Base_n_valid_snow_years"]),
        "selected_basin_wetup_improvement_days": float(
            selected["wetup_improvement_days"]
        ),
        "selected_basin_peak_improvement_days": float(
            selected["peak_improvement_days"]
        ),
        "selected_water_year": selected_year,
        "selected_year_annual_swe_max_mm": float(
            years.loc[years.water_year.eq(selected_year), "annual_swe_max_mm"].iloc[0]
        ),
        "selected_year_wetup_improvement_days": float(
            years.loc[
                years.water_year.eq(selected_year), "wetup_improvement_days"
            ].iloc[0]
        ),
        "selected_year_peak_improvement_days": float(
            years.loc[years.water_year.eq(selected_year), "peak_improvement_days"].iloc[
                0
            ]
        ),
        "timing_protocol_source": "r4/soil_analysis.py::compute_basin_year_timing; 14-day Jan-Jun wet-up and annual W_total/SM100 peak",
    }
    return basin_id, selected_year, audit


def _load_illustrative_series(results_root: Path, basin_id: str, water_year: int):
    caravan = np.load(
        results_root / "r4_caravan_soil_reference_v1" / "caravan_soil_ensemble.npz"
    )
    dates_full = caravan["dates"]
    test_slice = slice(
        int(caravan["test_slice_start"]), int(caravan["test_slice_stop"])
    )
    dates = pd.to_datetime(dates_full[test_slice])
    basin_ids = [str(x).zfill(8) for x in caravan["basin_ids"]]
    idx = basin_ids.index(basin_id)
    ref = caravan["SM100"][idx, test_slice].astype(float)
    swe = np.load(results_root / "r4_swe_reference_v1" / "swe_ensemble.npz")[
        "swe_median"
    ][idx, test_slice].astype(float)

    base_npz = np.load(
        results_root
        / "r4_official_dpl_XAJ_seed42"
        / "official_dpl_XAJ_seed42_full_arrays.npz"
    )
    cn_npz = np.load(
        results_root
        / "r4_official_dpl_XAJ_CN_seed42"
        / "official_dpl_XAJ_CN_seed42_full_arrays.npz"
    )
    base = (
        base_npz["wu"][idx, test_slice]
        + base_npz["wl"][idx, test_slice]
        + base_npz["wd"][idx, test_slice]
    ).astype(float)
    cn = (
        cn_npz["wu"][idx, test_slice]
        + cn_npz["wl"][idx, test_slice]
        + cn_npz["wd"][idx, test_slice]
    ).astype(float)
    timing = pd.read_csv(
        results_root / "r4_phase1_soil_official" / "timing_metrics_basin_year.csv",
        dtype={"basin_id": str},
    )
    timing = timing[
        (timing["regime"] == CANONICAL_DPL)
        & (timing["basin_id"] == basin_id)
        & (timing["water_year"] == water_year)
    ]
    return dates, ref, base, cn, swe, timing


def generate_figure8(results_root: Path, out_dir: Path) -> Path:
    setup_publication_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    official = results_root / "r4_phase1_soil_official"
    basin_id, water_year, audit = _select_illustrative_basin_year(results_root)
    (HERE / "figure8_r4_selection_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    dates, ref, base, cn, swe, timing = _load_illustrative_series(
        results_root, basin_id, water_year
    )

    ref_z = _standardized_month_anomaly(ref, dates.to_numpy())
    base_z = _standardized_month_anomaly(base, dates.to_numpy())
    cn_z = _standardized_month_anomaly(cn, dates.to_numpy())

    df = pd.DataFrame(
        {"date": dates, "ref_z": ref_z, "base_z": base_z, "cn_z": cn_z, "swe": swe}
    )
    df["wy"] = df["date"].apply(lambda d: d.year if d.month < 10 else d.year + 1)
    df_wy = df[df["wy"] == water_year].copy()

    # Timing markers for illustrative year
    row_base = timing[timing["structure"] == "Base"].iloc[0]
    row_cn = timing[timing["structure"] == "CN"].iloc[0]

    d_ref_w = _water_year_doy_to_date(water_year, row_base["wetup_doy_ref"])
    d_base_w = _water_year_doy_to_date(water_year, row_base["wetup_doy_model"])
    d_cn_w = _water_year_doy_to_date(water_year, row_cn["wetup_doy_model"])

    d_ref_p = _water_year_doy_to_date(water_year, row_base["peak_doy_ref"])
    d_base_p = _water_year_doy_to_date(water_year, row_base["peak_doy_model"])
    d_cn_p = _water_year_doy_to_date(water_year, row_cn["peak_doy_model"])

    # Q3 Population
    summary = pd.read_csv(
        official / "timing_metrics_basin_summary.csv", dtype={"basin_id": str}
    )
    paired = pd.read_csv(
        official / "paired_structural_effects.csv", dtype={"basin_id": str}
    )
    swe_q3 = paired[paired["regime"] == CANONICAL_DPL].drop_duplicates("basin_id")
    q3_threshold = float(swe_q3["snow_burden_swe_mm"].quantile(Q3_QUANTILE))
    q3_basins = set(swe_q3[swe_q3["snow_burden_swe_mm"] >= q3_threshold]["basin_id"])
    q3_n = len(q3_basins)

    # Build wide composite figure (7.2 x 6.2 inches -> aspect ratio ~1.16 : 1)
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[3.6, 1.4, 0.75],
        hspace=0.36,
        left=0.075,
        right=0.965,
        top=0.96,
        bottom=0.06,
    )

    # ---------------------------------------------------------------------------
    # Tier 1: Process Dynamics (a, b, c) ~63% area
    # ---------------------------------------------------------------------------
    gs_top = gs[0].subgridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.26)
    gs_left = gs_top[0].subgridspec(2, 1, height_ratios=[1.0, 1.45], hspace=0.20)

    # (a) Snow accumulation and depletion (illustrative high-snow water year)
    ax_a = fig.add_subplot(gs_left[0])
    apply_clean_spines(ax_a)
    ax_a.axvspan(
        pd.Timestamp(f"{water_year}-03-01"),
        pd.Timestamp(f"{water_year}-06-01"),
        color="#EAF1F8",
        alpha=0.55,
        zorder=0,
    )
    ax_a.plot(df_wy["date"], df_wy["swe"], color="#4C78A8", lw=1.1, zorder=3)
    ax_a.fill_between(
        df_wy["date"], 0, df_wy["swe"], color="#9ECAE1", alpha=0.25, zorder=2
    )
    ax_a.set_ylabel("SWE [mm]", fontsize=7.2)
    ax_a.set_title(
        "(a) Snow accumulation and depletion",
        loc="left",
        fontweight="bold",
        fontsize=8.0,
    )
    ax_a.tick_params(axis="x", labelbottom=False)
    ax_a.set_xlim(
        pd.Timestamp(f"{water_year - 1}-10-01"), pd.Timestamp(f"{water_year}-09-30")
    )
    ax_a.text(
        0.97,
        0.88,
        "Active melt period",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color="#4A6FA5",
    )

    # (b) Soil-water trajectories (same water year, shared x-axis)
    ax_b = fig.add_subplot(gs_left[1], sharex=ax_a)
    apply_clean_spines(ax_b)
    ax_b.axvspan(
        pd.Timestamp(f"{water_year}-03-01"),
        pd.Timestamp(f"{water_year}-06-01"),
        color="#EAF1F8",
        alpha=0.55,
        zorder=0,
    )
    ax_b.axhline(0, color="#CCCCCC", ls=":", lw=0.7, zorder=1)
    for key in ["Reference", "Base", "CN"]:
        cfg = STATE_CFG[key]
        col_name = (
            "ref_z" if key == "Reference" else ("base_z" if key == "Base" else "cn_z")
        )
        ax_b.plot(
            df_wy["date"],
            df_wy[col_name],
            color=cfg["color"],
            ls=cfg["ls"],
            lw=cfg["lw"],
            label=cfg["label"],
            zorder=4,
        )
    ax_b.set_ylabel("Standardized anomaly", fontsize=7.2)
    ax_b.set_title(
        "(b) Soil-water trajectories", loc="left", fontweight="bold", fontsize=8.0
    )
    ax_b.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_b.set_xlabel(f"Water year {water_year} (Basin {basin_id})", fontsize=7.0)
    ax_b.legend(loc="upper left", fontsize=5.5, frameon=True, framealpha=0.92)

    # (c) MAIN spring zoom + timing rails
    gs_c = gs_top[1].subgridspec(2, 1, height_ratios=[2.5, 0.95], hspace=0.15)
    ax_c_top = fig.add_subplot(gs_c[0])
    ax_c_bot = fig.add_subplot(gs_c[1], sharex=ax_c_top)
    apply_clean_spines(ax_c_top)
    apply_clean_spines(ax_c_bot)

    zoom_start = pd.Timestamp(f"{water_year}-01-01")
    zoom_end = pd.Timestamp(f"{water_year}-06-30")
    df_zoom = df_wy[(df_wy["date"] >= zoom_start) & (df_wy["date"] <= zoom_end)]

    # Trajectory zoom
    ax_c_top.axvspan(
        pd.Timestamp(f"{water_year}-03-01"),
        pd.Timestamp(f"{water_year}-06-01"),
        color="#EAF1F8",
        alpha=0.55,
        zorder=0,
    )
    ax_c_top.axhline(0, color="#CCCCCC", ls=":", lw=0.7, zorder=1)
    for key in ["Reference", "Base", "CN"]:
        cfg = STATE_CFG[key]
        col_name = (
            "ref_z" if key == "Reference" else ("base_z" if key == "Base" else "cn_z")
        )
        ax_c_top.plot(
            df_zoom["date"],
            df_zoom[col_name],
            color=cfg["color"],
            ls=cfg["ls"],
            lw=cfg["lw"],
            zorder=4,
        )
    ax_c_top.set_ylabel("Standardized anomaly", fontsize=7.2)
    ax_c_top.set_title(
        "(c) Spring recharge timing markers",
        loc="left",
        fontweight="bold",
        fontsize=8.0,
    )
    ax_c_top.tick_params(axis="x", labelbottom=False)
    ax_c_top.set_xlim(zoom_start, zoom_end)

    # Timing rails
    ax_c_bot.axvspan(
        pd.Timestamp(f"{water_year}-03-01"),
        pd.Timestamp(f"{water_year}-06-01"),
        color="#EAF1F8",
        alpha=0.55,
        zorder=0,
    )
    ax_c_bot.axhline(1.0, color="#E0E0E0", ls="-", lw=0.8, zorder=1)
    ax_c_bot.axhline(0.0, color="#E0E0E0", ls="-", lw=0.8, zorder=1)

    # Wet-up markers (Reference as open circle so CN filled diamond inside/atop is visible)
    ax_c_bot.plot(
        [d_base_w, d_ref_w],
        [1.0, 1.0],
        color="#2878B5",
        ls=":",
        lw=1.0,
        alpha=0.8,
        zorder=2,
    )
    ax_c_bot.plot(
        d_ref_w,
        1.0,
        marker="o",
        color="none",
        markeredgecolor="#555555",
        markeredgewidth=1.6,
        ms=6.8,
        zorder=4,
    )
    ax_c_bot.plot(d_base_w, 1.0, marker="^", color="#2878B5", ms=5.2, zorder=5)
    ax_c_bot.plot(d_cn_w, 1.0, marker="D", color="#D95F02", ms=4.2, zorder=6)
    ax_c_bot.text(
        d_base_w,
        1.22,
        f"{int(row_base['wetup_timing_error_days'])} d",
        ha="center",
        va="bottom",
        fontsize=5.5,
        color="#2878B5",
    )
    ax_c_bot.text(
        d_cn_w,
        0.78,
        f"{int(row_cn['wetup_timing_error_days'])} d",
        ha="center",
        va="top",
        fontsize=5.5,
        color="#D95F02",
    )

    # Peak markers
    ax_c_bot.plot(
        [d_base_p, d_ref_p],
        [0.0, 0.0],
        color="#2878B5",
        ls=":",
        lw=1.0,
        alpha=0.8,
        zorder=2,
    )
    ax_c_bot.plot(
        d_ref_p,
        0.0,
        marker="o",
        color="none",
        markeredgecolor="#555555",
        markeredgewidth=1.6,
        ms=6.8,
        zorder=4,
    )
    ax_c_bot.plot(d_base_p, 0.0, marker="^", color="#2878B5", ms=5.2, zorder=5)
    ax_c_bot.plot(d_cn_p, 0.0, marker="D", color="#D95F02", ms=4.2, zorder=6)
    ax_c_bot.text(
        d_base_p,
        0.22,
        f"{int(row_base['peak_timing_error_days'])} d",
        ha="center",
        va="bottom",
        fontsize=5.5,
        color="#2878B5",
    )
    ax_c_bot.text(
        d_cn_p,
        -0.22,
        f"{int(row_cn['peak_timing_error_days'])} d",
        ha="center",
        va="top",
        fontsize=5.5,
        color="#D95F02",
    )

    ax_c_bot.set_yticks([0.0, 1.0])
    ax_c_bot.set_yticklabels(["Peak", "Wet-up"], fontsize=6.8)
    ax_c_bot.set_ylim(-0.48, 1.48)
    ax_c_bot.xaxis.set_major_locator(mdates.MonthLocator())
    ax_c_bot.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_c_bot.set_xlabel(f"Spring {water_year}", fontsize=7.0)

    # Neutral legend (Reference, Base, CN)
    leg_c = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markeredgecolor="#555555",
            markeredgewidth=1.5,
            markerfacecolor="none",
            ms=5.5,
            label="Reference",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="#2878B5",
            ms=5.0,
            label="Base",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="#D95F02",
            ms=4.5,
            label="CN",
        ),
    ]
    ax_c_top.legend(
        handles=leg_c, loc="upper left", fontsize=5.6, frameon=True, framealpha=0.92
    )

    # ---------------------------------------------------------------------------
    # Tier 2: Basin-level Timing Populations (d, e) ~24% area
    # ---------------------------------------------------------------------------
    gs_mid = gs[1].subgridspec(1, 2, wspace=0.24)
    ax_d = fig.add_subplot(gs_mid[0])
    ax_e = fig.add_subplot(gs_mid[1])

    # (d) Spring wet-up timing ECDF
    apply_clean_spines(ax_d)
    ax_d.axvline(0, color="#999999", ls="--", lw=0.8, zorder=1)
    for reg, reg_name in [("dPL_seed42", "dPL-42"), ("IC_fused", "IC fused")]:
        cfg = REGIME_CFG[reg]
        sub = summary[
            (summary["regime"] == reg) & (summary["basin_id"].isin(q3_basins))
        ].copy()
        for struct, col, struct_name in [
            ("Base", "#2878B5", "Base"),
            ("CN", "#D95F02", "CN"),
        ]:
            vals = (
                sub[sub["structure"] == struct]["median_wetup_error_days"]
                .dropna()
                .to_numpy()
            )
            sx = np.sort(vals)
            sy = np.linspace(0, 1, len(sx))
            ax_d.step(
                sx,
                sy,
                where="post",
                color=col,
                ls=cfg["ls"],
                lw=1.2,
                alpha=0.9,
                label=f"{struct_name} ({reg_name})",
            )

    ax_d.set_title(
        "(d) Basin-level spring wet-up timing",
        loc="left",
        fontweight="bold",
        fontsize=8.0,
    )
    ax_d.set_xlabel("Signed wet-up error [days] ($t_{model} - t_{ref}$)", fontsize=7.0)
    ax_d.set_ylabel(f"Cumulative fraction (Q3, n = {q3_n})", fontsize=7.0)
    ax_d.set_xlim(-140, 110)
    ax_d.set_ylim(-0.02, 1.02)
    ax_d.legend(loc="upper left", fontsize=5.4, frameon=True, framealpha=0.92)
    ax_d.text(
        0.97,
        0.08,
        "Median error (dPL-42):\nBase +20 d, CN 0 d\n(displacement reduced)",
        transform=ax_d.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.2,
        color="#444444",
        bbox=dict(
            boxstyle="square,pad=0.25",
            facecolor="#FAFAFA",
            edgecolor="#E0E0E0",
            alpha=0.9,
        ),
    )

    # (e) Soil-water peak timing ECDF
    apply_clean_spines(ax_e)
    ax_e.axvline(0, color="#999999", ls="--", lw=0.8, zorder=1)
    for reg, reg_name in [("dPL_seed42", "dPL-42"), ("IC_fused", "IC fused")]:
        cfg = REGIME_CFG[reg]
        sub = summary[
            (summary["regime"] == reg) & (summary["basin_id"].isin(q3_basins))
        ].copy()
        for struct, col, struct_name in [
            ("Base", "#2878B5", "Base"),
            ("CN", "#D95F02", "CN"),
        ]:
            vals = (
                sub[sub["structure"] == struct]["median_peak_error_days"]
                .dropna()
                .to_numpy()
            )
            sx = np.sort(vals)
            sy = np.linspace(0, 1, len(sx))
            ax_e.step(
                sx,
                sy,
                where="post",
                color=col,
                ls=cfg["ls"],
                lw=1.2,
                alpha=0.9,
                label=f"{struct_name} ({reg_name})",
            )

    ax_e.set_title(
        "(e) Basin-level soil-water peak timing",
        loc="left",
        fontweight="bold",
        fontsize=8.0,
    )
    ax_e.set_xlabel("Signed peak error [days] ($t_{model} - t_{ref}$)", fontsize=7.0)
    ax_e.set_ylabel(f"Cumulative fraction (Q3, n = {q3_n})", fontsize=7.0)
    ax_e.set_xlim(-210, 30)
    ax_e.set_ylim(-0.02, 1.02)
    ax_e.legend(loc="upper left", fontsize=5.4, frameon=True, framealpha=0.92)
    ax_e.text(
        0.97,
        0.08,
        "Median error (dPL-42):\nBase −58 d, CN −28 d\n(early bias reduced)",
        transform=ax_e.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.2,
        color="#444444",
        bbox=dict(
            boxstyle="square,pad=0.25",
            facecolor="#FAFAFA",
            edgecolor="#E0E0E0",
            alpha=0.9,
        ),
    )

    # ---------------------------------------------------------------------------
    # Tier 3: Definition Sensitivity (f) ~13% area
    # ---------------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[2])
    apply_clean_spines(ax_f)
    ax_f.axvline(0, color="#999999", ls="--", lw=0.8, zorder=1)

    sens = pd.read_csv(official / "robustness_timing_sensitivity.csv")
    sens_rows = [
        (
            "Wet-up 7 d",
            "Peak_Annual_FullWY",
            "Wetup_07d_Spring",
            "wetup_abs_error_improvement_days",
        ),
        (
            "Wet-up 14 d",
            "Peak_Annual_FullWY",
            "Wetup_14d_Spring",
            "wetup_abs_error_improvement_days",
        ),
        (
            "Wet-up 21 d",
            "Peak_Annual_FullWY",
            "Wetup_21d_Spring",
            "wetup_abs_error_improvement_days",
        ),
        (
            "Peak full WY",
            "Peak_Annual_FullWY",
            "Wetup_14d_Spring",
            "peak_abs_error_improvement_days",
        ),
        (
            "Peak Mar–Aug",
            "Peak_SpringSummer_MarAug",
            "Wetup_14d_Spring",
            "peak_abs_error_improvement_days",
        ),
    ]
    y_pos = np.array([4, 3, 2, 0.6, -0.4])

    for y, (label, peak_def, wet_def, value_col) in zip(y_pos, sens_rows):
        for reg in REGIMES:
            row = sens[
                (sens["regime"] == reg)
                & (sens["peak_definition"] == peak_def)
                & (sens["wetup_definition"] == wet_def)
            ].iloc[0]
            ax_f.plot(
                row[value_col],
                y,
                marker=REGIME_CFG[reg]["marker"],
                color=REGIME_CFG[reg]["color"],
                ms=4.5,
                zorder=3,
            )

    ax_f.set_yticks(y_pos)
    ax_f.set_yticklabels([r[0] for r in sens_rows], fontsize=6.5)
    ax_f.axhline(1.3, color="#E5E5E5", lw=0.7)
    ax_f.set_xlabel("Base MAE − CN MAE [days]", fontsize=7.0)
    ax_f.set_xlim(-5, 45)
    ax_f.set_title(
        "(f) Timing-definition sensitivity", loc="left", fontweight="bold", fontsize=8.0
    )
    ax_f.text(
        0.01,
        0.90,
        "Wet-up definitions",
        transform=ax_f.transAxes,
        fontsize=5.5,
        color="#666666",
    )
    ax_f.text(
        0.01,
        0.25,
        "Peak windows",
        transform=ax_f.transAxes,
        fontsize=5.5,
        color="#666666",
    )

    leg_f = [
        Line2D(
            [0],
            [0],
            marker=REGIME_CFG[r]["marker"],
            color=REGIME_CFG[r]["color"],
            ls="none",
            ms=4.5,
            label=REGIME_CFG[r]["label"],
        )
        for r in REGIMES
    ]
    ax_f.legend(
        handles=leg_f,
        loc="lower right",
        fontsize=5.6,
        ncol=2,
        frameon=True,
        framealpha=0.92,
    )

    out = out_dir / "figure8_r4_soil_timing.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(
        f"Generated Figure 8 (PNG only, 300 dpi):\n  {out}\n  basin={basin_id}, water_year={water_year}"
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    generate_figure8(args.results_root or default_results_root(), args.out_dir)
