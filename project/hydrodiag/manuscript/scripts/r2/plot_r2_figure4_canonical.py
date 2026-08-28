#!/usr/bin/env python3
"""Render the manuscript-facing, canonical R2 Figure 4 with single-row paired ridgelines.

Layout (8 panels, normal lettering (a)–(h)):
    Top row (wspace=0.12, shared y-axis):
        (a) All-15 slope response map: Base–CN
        (b) All-15 slope response map: Base–TGD
    Middle row (wspace=0.12):
        (c) Endpoint fingerprint: Base–CN
        (d) Endpoint fingerprint: Base–TGD
    Bottom row (wspace=0.06, shared y-axis across e, f, g, h):
        (e) um (with S1-S5 labels retained)
        (f) ki
        (g) ci
        (h) im (with IC/dPL legend aligned to the height of the Base-CN text badge)

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

OUTPUT_NAME = "Figure4_R2_final.png"
REGIMES = ["S1", "S2", "S3", "S4", "S5"]
REGIME_N = {"S1": 165, "S2": 156, "S3": 121, "S4": 34, "S5": 55}
PARADIGMS = ["IC", "dPL"]
KEY_PARAMS = ["xaj_um", "xaj_ki", "xaj_ci", "xaj_im"]

# Color palette: HESS / R1 publication style
COLOR_TEXT = "#303438"
COLOR_REF = "#70767B"
COLOR_GRID = "#E5E8EB"
COLOR_BASELINE = "#D0D5DA"

# Structural contrast palettes (Okabe-Ito / Nature aligned)
# Base–CN: Blue family
COLOR_CN = "#0072B2"
COLOR_CN_DARK = "#0B4C79"
COLOR_CN_LIGHT = "#D4E6F1"
COLOR_CN_DPL = "#5B9BD5"

# Base–TGD: Teal/Green family
COLOR_TGD = "#009E73"
COLOR_TGD_DARK = "#005E44"
COLOR_TGD_LIGHT = "#D5F5E3"
COLOR_TGD_DPL = "#52B788"

# Ridgeline parameters
DZ_GRID = np.linspace(-1.0, 1.0, 240)
DZ_BW = 0.075
RIDGE_HEIGHT = 0.38
IC_FILL_ALPHA = 0.22   # Soft light background tint
DPL_FILL_ALPHA = 0.08  # Soft light background tint
DPL_LINESTYLE = (0, (4.5, 2.5))


def _read(name: str) -> pd.DataFrame:
    path = RESULTS_R2 / name
    if not path.exists():
        raise FileNotFoundError(f"canonical R2 source missing: {path}")
    return pd.read_csv(path)


def load_data() -> dict[str, pd.DataFrame]:
    data = {
        "specs": _read("authoritative_15_parameter_specs.csv"),
        "gradients": _read("r2_snow_gradients_summary.csv"),
        "strata": _read("r2_parameter_shifts_strata_summary.csv"),
        "values": _read("r2_parameter_values_canonical.csv"),
    }
    _assert_schema(data)
    return data


def _assert_schema(data: dict[str, pd.DataFrame]) -> None:
    specs = data["specs"].sort_values("shared_index")
    assert len(specs) == 15
    assert specs["parameter_name"].is_unique
    assert set(KEY_PARAMS).issubset(set(specs["parameter_name"]))

    values = data["values"]
    assert set(values["paradigm"]) == set(PARADIGMS)
    assert set(values["structure"]) == {"Base", "CN", "TGD"}
    z_columns = [f"z_{p}" for p in specs["parameter_name"]]
    assert set(z_columns).issubset(values.columns)
    z_values = values[z_columns].to_numpy(float)
    assert np.isfinite(z_values).all() and ((z_values >= -1e-9) & (z_values <= 1 + 1e-9)).all()


def _parameter_order(data: dict[str, pd.DataFrame]) -> tuple[list[str], dict[str, str]]:
    specs = data["specs"].sort_values("shared_index")
    order = specs["parameter_name"].tolist()
    display = dict(zip(specs["parameter_name"], specs["symbol"]))
    return order, display


def ridge_density(vals: np.ndarray, grid: np.ndarray, h: float) -> np.ndarray:
    """Fixed-bandwidth Gaussian KDE on [-1, 1] with boundary reflection."""
    v = np.asarray(vals, dtype=float)
    v = v[(v >= -1.0) & (v <= 1.0)]
    if len(v) < 2:
        d = np.zeros_like(grid)
        d[np.argmin(np.abs(grid - float(np.median(v))))] = 1.0
        return d
    refl = np.concatenate([-2.0 - v, v, 2.0 - v])
    d = np.exp(-0.5 * ((grid[:, None] - refl[None, :]) / h) ** 2).sum(axis=1)
    d[grid < -1.0] = 0.0
    d[grid > 1.0] = 0.0
    if d.max() > 0:
        d = d / d.max()
    return d


def compute_all_slopes(data: dict[str, pd.DataFrame], order: list[str]) -> pd.DataFrame:
    """Compute snow-gradient slopes (beta) for Base-CN and Base-TGD across IC and dPL."""
    values = data["values"]
    base = values[values["structure"] == "Base"].set_index(["paradigm", "basin_id"])
    cn = values[values["structure"] == "CN"].set_index(["paradigm", "basin_id"])
    tgd = values[values["structure"] == "TGD"].set_index(["paradigm", "basin_id"])
    frac_snow = base["frac_snow"]

    rows = []
    for p in PARADIGMS:
        x = frac_snow.loc[p].to_numpy(float)
        for par in order:
            y_cn = (base.loc[p, f"z_{par}"] - cn.loc[p, f"z_{par}"]).to_numpy(float)
            y_tg = (base.loc[p, f"z_{par}"] - tgd.loc[p, f"z_{par}"]).to_numpy(float)
            rows.append({
                "paradigm": p,
                "parameter": par,
                "beta_cn": float(np.polyfit(x, y_cn, 1)[0]),
                "beta_tgd": float(np.polyfit(x, y_tg, 1)[0]),
            })
    return pd.DataFrame(rows)


def compute_endpoint_shifts(data: dict[str, pd.DataFrame], order: list[str]) -> pd.DataFrame:
    """Compute S5 - S1 net median endpoint shifts with 95% bootstrap intervals."""
    values = data["values"]
    strata_cn = data["strata"].set_index(["paradigm", "parameter", "snow_stratum"])

    base = values[values["structure"] == "Base"].set_index(["paradigm", "snow_stratum", "basin_id"]).sort_index()
    tgd = values[values["structure"] == "TGD"].set_index(["paradigm", "snow_stratum", "basin_id"]).sort_index()

    rng = np.random.default_rng(20260730)
    n_boot = 2000

    rows = []
    for p in PARADIGMS:
        s1_b = base.loc[(p, "S1")]
        s1_tg = tgd.loc[(p, "S1")]
        s5_b = base.loc[(p, "S5")]
        s5_tg = tgd.loc[(p, "S5")]
        n_s1 = len(s1_b)
        n_s5 = len(s5_b)

        idx1 = rng.integers(0, n_s1, size=(n_boot, n_s1))
        idx5 = rng.integers(0, n_s5, size=(n_boot, n_s5))

        for par in order:
            row_cn_s5 = strata_cn.loc[(p, par, "S5")]
            cn_val = float(row_cn_s5["D_activity_S5_minus_S1"])
            cn_lo = float(row_cn_s5["D_activity_ci_low"])
            cn_hi = float(row_cn_s5["D_activity_ci_high"])

            v1_tg = (s1_b[f"z_{par}"] - s1_tg[f"z_{par}"]).to_numpy(float)
            v5_tg = (s5_b[f"z_{par}"] - s5_tg[f"z_{par}"]).to_numpy(float)
            tg_val = float(np.median(v5_tg) - np.median(v1_tg))

            b1 = np.median(v1_tg[idx1], axis=1)
            b5 = np.median(v5_tg[idx5], axis=1)
            diffs = b5 - b1
            tg_lo, tg_hi = np.percentile(diffs, [2.5, 97.5])

            rows.append({
                "paradigm": p,
                "parameter": par,
                "cn_val": cn_val, "cn_lo": cn_lo, "cn_hi": cn_hi,
                "tg_val": tg_val, "tg_lo": float(tg_lo), "tg_hi": float(tg_hi),
            })
    return pd.DataFrame(rows)


def build_basin_shifts(data: dict[str, pd.DataFrame], order: list[str]) -> pd.DataFrame:
    """Build long-format basin-level signed shifts for Base-CN and Base-TGD."""
    values = data["values"]
    base = values[values["structure"] == "Base"].set_index(["paradigm", "basin_id"])
    cn = values[values["structure"] == "CN"].set_index(["paradigm", "basin_id"])
    tgd = values[values["structure"] == "TGD"].set_index(["paradigm", "basin_id"])

    rows = []
    for p in PARADIGMS:
        b_sub = base.loc[p]
        cn_sub = cn.loc[p]
        tg_sub = tgd.loc[p]

        for par in order:
            d_cn = (b_sub[f"z_{par}"] - cn_sub[f"z_{par}"]).to_numpy(float)
            d_tg = (b_sub[f"z_{par}"] - tg_sub[f"z_{par}"]).to_numpy(float)
            strata = b_sub["snow_stratum"].to_numpy()

            for s, v_cn, v_tg in zip(strata, d_cn, d_tg):
                rows.append({
                    "paradigm": p,
                    "snow_stratum": s,
                    "parameter": par,
                    "delta_cn": v_cn,
                    "delta_tgd": v_tg,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Panels (a) & (b): All-15 Parameter Response Maps (Base–CN & Base–TGD)
# ---------------------------------------------------------------------------
def _plot_response_map_sub(
    ax: plt.Axes,
    df_slopes: pd.DataFrame,
    order: list[str],
    display: dict[str, str],
    contrast_key: str,  # 'beta_cn' or 'beta_tgd'
    main_color: str,
    panel_title: str,
    show_ylabel: bool = True,
) -> None:
    ic_map = df_slopes[df_slopes["paradigm"] == "IC"].set_index("parameter")[contrast_key]
    dpl_map = df_slopes[df_slopes["paradigm"] == "dPL"].set_index("parameter")[contrast_key]

    # Light quadrant shading
    ax.axvspan(0, 0.70, 0.5, 1.0, color="#F0F4F8", alpha=0.45, zorder=0)
    ax.axvspan(-0.70, 0, 0.0, 0.5, color="#F5F0F4", alpha=0.45, zorder=0)

    # Reference lines
    ax.axvline(0.0, color=COLOR_REF, linestyle="--", linewidth=0.85, zorder=1)
    ax.axhline(0.0, color=COLOR_REF, linestyle="--", linewidth=0.85, zorder=1)
    ax.plot([-0.70, 0.70], [-0.70, 0.70], color=COLOR_REF, linestyle=":", linewidth=0.75, alpha=0.6, zorder=1)

    label_offsets = {
        "xaj_um": (-0.022, 0.022, "right", "bottom"),
        "xaj_ki": (-0.022, -0.026, "right", "top"),
        "xaj_ci": (0.022, -0.026, "left", "top"),
        "xaj_im": (0.022, -0.024, "left", "top"),
        "xaj_dm": (-0.022, 0.022, "right", "bottom"),
        "xaj_ex": (-0.022, 0.022, "right", "bottom"),
        "xaj_kg": (-0.022, 0.022, "right", "bottom"),
        "xaj_lm": (0.022, 0.022, "left", "bottom"),
        "xaj_c": (-0.022, 0.022, "right", "bottom"),
        "xaj_sm": (0.022, 0.022, "left", "bottom"),
        "xaj_cg": (0.022, -0.026, "left", "top"),
        "xaj_theta": (-0.022, 0.024, "right", "bottom"),
        "xaj_b": (-0.022, 0.022, "right", "bottom"),
        "xaj_k": (-0.022, -0.026, "right", "top"),
        "xaj_a": (0.022, -0.026, "left", "top"),
    }

    for p in order:
        bx = float(ic_map.loc[p])
        by = float(dpl_map.loc[p])
        sym = display[p]
        is_key = p in KEY_PARAMS

        if is_key:
            ax.plot(
                bx, by, marker="o", markersize=7.8,
                markerfacecolor=main_color, markeredgecolor="white", markeredgewidth=1.0,
                zorder=4
            )
            dx, dy, ha, va = label_offsets.get(p, (0.02, 0.02, "left", "bottom"))
            ax.text(
                bx + dx, by + dy, sym,
                fontsize=9.8, fontweight="bold", color=COLOR_TEXT,
                ha=ha, va=va, zorder=5
            )
        else:
            ax.plot(
                bx, by, marker="s", markersize=5.4,
                markerfacecolor="#8B949E", markeredgecolor="white", markeredgewidth=0.7,
                zorder=3
            )
            dx, dy, ha, va = label_offsets.get(p, (0.02, 0.02, "left", "bottom"))
            ax.text(
                bx + dx, by + dy, sym,
                fontsize=8.6, color="#50555A",
                ha=ha, va=va, zorder=4
            )

    # Quadrant annotations
    ax.text(
        0.67, 0.67, "Positive gradient in both regimes",
        ha="right", va="top", fontsize=8.0, color="#3E5C76", style="italic"
    )
    ax.text(
        -0.67, -0.67, "Negative gradient in both regimes",
        ha="left", va="bottom", fontsize=8.0, color="#6D435F", style="italic"
    )

    ax.set_xlim(-0.70, 0.70)
    ax.set_ylim(-0.70, 0.70)
    ticks = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t:.1f}" if t != 0 else "0" for t in ticks], fontsize=9.2)
    if show_ylabel:
        ax.set_yticklabels([f"{t:.1f}" if t != 0 else "0" for t in ticks], fontsize=9.2)
        ax.set_ylabel(r"Snow-gradient slope under dPL, $\beta_{\mathrm{dPL}}$", fontsize=10.2)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")

    ax.set_xlabel(r"Snow-gradient slope under IC, $\beta_{\mathrm{IC}}$", fontsize=10.2)
    ax.set_title(panel_title, loc="left", weight="bold", fontsize=11.2, color=COLOR_TEXT, pad=6)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.35)


# ---------------------------------------------------------------------------
# Panels (c) & (d): All-15 Endpoint Fingerprints (Base–CN & Base–TGD)
# ---------------------------------------------------------------------------
def _plot_endpoint_sub(
    ax: plt.Axes,
    df_ep: pd.DataFrame,
    order: list[str],
    display: dict[str, str],
    val_key: str,
    lo_key: str,
    hi_key: str,
    main_color: str,
    dark_edge: str,
    panel_title: str,
    show_ylabel: bool = True,
) -> None:
    y_pos = np.arange(len(order))
    bar_height = 0.30
    offset = 0.17

    # Subtle row background bands for key parameters
    for i, p in enumerate(order):
        if p in KEY_PARAMS:
            ax.axhspan(i - 0.5, i + 0.5, color="#F4F6F8", linewidth=0, zorder=0)

    ep_ic = df_ep[df_ep["paradigm"] == "IC"].set_index("parameter")
    ep_dpl = df_ep[df_ep["paradigm"] == "dPL"].set_index("parameter")

    for i, p in enumerate(order):
        y = y_pos[i]
        # IC (upper bar)
        ic_val = float(ep_ic.loc[p, val_key])
        ic_lo = float(ep_ic.loc[p, lo_key])
        ic_hi = float(ep_ic.loc[p, hi_key])
        ax.barh(
            y + offset, ic_val, height=bar_height,
            color=main_color, edgecolor=dark_edge, linewidth=0.7, alpha=0.85,
            zorder=2
        )
        if abs(ic_hi - ic_lo) > 1e-4:
            ax.errorbar(
                ic_val, y + offset,
                xerr=[[ic_val - ic_lo], [ic_hi - ic_val]],
                fmt="none", ecolor=dark_edge, elinewidth=0.8, capsize=1.5, alpha=0.85, zorder=3
            )

        # dPL (lower bar)
        dpl_val = float(ep_dpl.loc[p, val_key])
        dpl_lo = float(ep_dpl.loc[p, lo_key])
        dpl_hi = float(ep_dpl.loc[p, hi_key])
        ax.barh(
            y - offset, dpl_val, height=bar_height,
            color=main_color, edgecolor=dark_edge, linewidth=0.7, alpha=0.45,
            zorder=2
        )
        if abs(dpl_hi - dpl_lo) > 1e-4:
            ax.errorbar(
                dpl_val, y - offset,
                xerr=[[dpl_val - dpl_lo], [dpl_hi - dpl_val]],
                fmt="none", ecolor=dark_edge, elinewidth=0.8, capsize=1.5, alpha=0.85, zorder=3
            )

    ax.axvline(0.0, color=COLOR_REF, linestyle="--", linewidth=0.85, zorder=1)
    ax.set_yticks(y_pos)
    if show_ylabel:
        labels = [display[p] for p in order]
        ax.set_yticklabels(labels, fontsize=9.2)
        for tick, p in zip(ax.get_yticklabels(), order):
            if p in KEY_PARAMS:
                tick.set_fontweight("bold")
                tick.set_color(COLOR_TEXT)
        ax.set_ylabel("Shared parameter", fontsize=10.2)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")

    ax.invert_yaxis()
    ax.set_xlim(-1.0, 1.0)
    ax.tick_params(axis="x", labelsize=9.2)
    ax.set_xlabel(r"Net signed shift endpoint, $\Delta z_{\mathrm{S5}} - \Delta z_{\mathrm{S1}}$", fontsize=10.2)
    ax.set_title(panel_title, loc="left", weight="bold", fontsize=11.2, color=COLOR_TEXT, pad=6)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.35)


# ---------------------------------------------------------------------------
# Panels (e)–(h): Single-Row Paired Parameter Ridgelines
# ---------------------------------------------------------------------------
def _plot_paired_parameter_panel(
    ax: plt.Axes,
    df_shifts: pd.DataFrame,
    parameter: str,
    letter: str,
    display: dict[str, str],
    is_panel_e: bool = False,
    is_panel_h: bool = False,
) -> None:
    """One single panel containing S1–S5 baselines with Base–CN above and Base–TGD below."""
    y_pos = np.arange(len(REGIMES))

    for i, reg in enumerate(REGIMES):
        y = y_pos[i]
        ax.axhline(y, color=COLOR_BASELINE, linewidth=0.65, zorder=1)

        # -------------------------------------------------------------------
        # Upper half: Base–CN (Blue family)
        # -------------------------------------------------------------------
        # Base–CN, IC (upper, solid) - Soft light fill
        vals_cn_ic = df_shifts[
            (df_shifts["paradigm"] == "IC")
            & (df_shifts["parameter"] == parameter)
            & (df_shifts["snow_stratum"] == reg)
        ]["delta_cn"].to_numpy(float)
        d_cn_ic = ridge_density(vals_cn_ic, DZ_GRID, DZ_BW)
        curve_cn_ic = y + RIDGE_HEIGHT * d_cn_ic
        ax.fill_between(DZ_GRID, y, curve_cn_ic, color=COLOR_CN, alpha=IC_FILL_ALPHA, linewidth=0, zorder=2)
        ax.plot(DZ_GRID, curve_cn_ic, color=COLOR_CN_DARK, linewidth=1.5, zorder=3)

        med_cn_ic = float(np.median(vals_cn_ic))
        med_idx_cn_ic = np.argmin(np.abs(DZ_GRID - med_cn_ic))
        ax.plot([med_cn_ic], [y + RIDGE_HEIGHT * d_cn_ic[med_idx_cn_ic]], marker="o", color=COLOR_CN_DARK, markersize=3.8, markeredgecolor="white", markeredgewidth=0.5, zorder=5)
        q1_cn_ic, q3_cn_ic = np.percentile(vals_cn_ic, [25, 75])
        ax.plot([q1_cn_ic, q3_cn_ic], [y + 0.015, y + 0.015], color=COLOR_CN_DARK, linewidth=1.1, solid_capstyle="butt", zorder=5)

        # Base–CN, dPL (upper, dashed) - Soft light fill
        vals_cn_dpl = df_shifts[
            (df_shifts["paradigm"] == "dPL")
            & (df_shifts["parameter"] == parameter)
            & (df_shifts["snow_stratum"] == reg)
        ]["delta_cn"].to_numpy(float)
        d_cn_dpl = ridge_density(vals_cn_dpl, DZ_GRID, DZ_BW)
        curve_cn_dpl = y + RIDGE_HEIGHT * d_cn_dpl
        ax.fill_between(DZ_GRID, y, curve_cn_dpl, color=COLOR_CN, alpha=DPL_FILL_ALPHA, linewidth=0, zorder=2)
        ax.plot(DZ_GRID, curve_cn_dpl, color=COLOR_CN_DARK, linestyle=DPL_LINESTYLE, linewidth=1.4, zorder=3)

        med_cn_dpl = float(np.median(vals_cn_dpl))
        med_idx_cn_dpl = np.argmin(np.abs(DZ_GRID - med_cn_dpl))
        ax.plot([med_cn_dpl], [y + RIDGE_HEIGHT * d_cn_dpl[med_idx_cn_dpl]], marker="^", color="white", markeredgecolor=COLOR_CN_DARK, markeredgewidth=1.3, markersize=4.2, zorder=5)

        # -------------------------------------------------------------------
        # Lower half: Base–TGD (Teal/Green family)
        # -------------------------------------------------------------------
        # Base–TGD, IC (lower, solid) - Soft light fill
        vals_tg_ic = df_shifts[
            (df_shifts["paradigm"] == "IC")
            & (df_shifts["parameter"] == parameter)
            & (df_shifts["snow_stratum"] == reg)
        ]["delta_tgd"].to_numpy(float)
        d_tg_ic = ridge_density(vals_tg_ic, DZ_GRID, DZ_BW)
        curve_tg_ic = y - RIDGE_HEIGHT * d_tg_ic
        ax.fill_between(DZ_GRID, y, curve_tg_ic, color=COLOR_TGD, alpha=IC_FILL_ALPHA, linewidth=0, zorder=2)
        ax.plot(DZ_GRID, curve_tg_ic, color=COLOR_TGD_DARK, linewidth=1.5, zorder=3)

        med_tg_ic = float(np.median(vals_tg_ic))
        med_idx_tg_ic = np.argmin(np.abs(DZ_GRID - med_tg_ic))
        ax.plot([med_tg_ic], [y - RIDGE_HEIGHT * d_tg_ic[med_idx_tg_ic]], marker="o", color=COLOR_TGD_DARK, markersize=3.8, markeredgecolor="white", markeredgewidth=0.5, zorder=5)
        q1_tg_ic, q3_tg_ic = np.percentile(vals_tg_ic, [25, 75])
        ax.plot([q1_tg_ic, q3_tg_ic], [y - 0.015, y - 0.015], color=COLOR_TGD_DARK, linewidth=1.1, solid_capstyle="butt", zorder=5)

        # Base–TGD, dPL (lower, dashed) - Soft light fill
        vals_tg_dpl = df_shifts[
            (df_shifts["paradigm"] == "dPL")
            & (df_shifts["parameter"] == parameter)
            & (df_shifts["snow_stratum"] == reg)
        ]["delta_tgd"].to_numpy(float)
        d_tg_dpl = ridge_density(vals_tg_dpl, DZ_GRID, DZ_BW)
        curve_tg_dpl = y - RIDGE_HEIGHT * d_tg_dpl
        ax.fill_between(DZ_GRID, y, curve_tg_dpl, color=COLOR_TGD, alpha=DPL_FILL_ALPHA, linewidth=0, zorder=2)
        ax.plot(DZ_GRID, curve_tg_dpl, color=COLOR_TGD_DARK, linestyle=DPL_LINESTYLE, linewidth=1.4, zorder=3)

        med_tg_dpl = float(np.median(vals_tg_dpl))
        med_idx_tg_dpl = np.argmin(np.abs(DZ_GRID - med_tg_dpl))
        ax.plot([med_tg_dpl], [y - RIDGE_HEIGHT * d_tg_dpl[med_idx_tg_dpl]], marker="^", color="white", markeredgecolor=COLOR_TGD_DARK, markeredgewidth=1.3, markersize=4.2, zorder=5)

    ax.axvline(0.0, color=COLOR_REF, linestyle="--", linewidth=0.85, zorder=1)
    ax.set_yticks(y_pos)

    if is_panel_e:
        # Retain S1..S5 stratum labels on panel (e)
        ytick_labels = [f"{r}\n(n={REGIME_N[r]})" for r in REGIMES]
        ax.set_yticklabels(ytick_labels, fontsize=8.6)
        ax.tick_params(axis="y", left=True, labelleft=True, pad=2)
        ax.set_ylabel(r"Snow stratum $\uparrow$", fontsize=10.2)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_yticklabels([])
        ax.set_ylabel("")

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.55, len(REGIMES) - 0.45)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xticklabels(["-1", "-0.5", "0", "0.5", "1"], fontsize=9.2)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.3)
    ax.set_xlabel(r"$\Delta z = z_{\mathrm{Base}} - z_{\mathrm{struct}}$", fontsize=9.8)
    ax.set_title(f"({letter}) {display[parameter]}", loc="left", weight="bold", fontsize=11.2, color=COLOR_TEXT, pad=6)

    # In-panel subtle tags
    ax.text(
        0.03, 0.94, "▲ Base–CN",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8.0, fontweight="bold", color=COLOR_CN_DARK,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=COLOR_CN, alpha=0.90, linewidth=0.5),
        zorder=6
    )
    ax.text(
        0.03, 0.06, "▼ Base–TGD",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=8.0, fontweight="bold", color=COLOR_TGD_DARK,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor=COLOR_TGD, alpha=0.90, linewidth=0.5),
        zorder=6
    )

    # Panel (h): Dedicated IC vs dPL legend placed outside above panel (h), aligned with the title
    if is_panel_h:
        ridge_legend_handles = [
            Line2D([0], [0], color=COLOR_TEXT, linestyle="-", linewidth=1.5, marker="o", markersize=4.0, markerfacecolor=COLOR_TEXT, markeredgecolor="white", label="IC (solid, ●)"),
            Line2D([0], [0], color=COLOR_TEXT, linestyle=DPL_LINESTYLE, linewidth=1.5, marker="^", markersize=4.5, markerfacecolor="white", markeredgecolor=COLOR_TEXT, markeredgewidth=1.2, label="dPL (dashed, △)"),
        ]
        ax.legend(
            handles=ridge_legend_handles,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.01),
            frameon=True,
            facecolor="white",
            edgecolor="#CBD5E1",
            fontsize=8.0,
            ncol=2,
            columnspacing=0.8,
            handlelength=1.4,
            framealpha=0.95,
        )


# ---------------------------------------------------------------------------
# Figure Builder (8 Panels: Top 2, Middle 2, Bottom 4 with clean in-panel legend)
# ---------------------------------------------------------------------------
def build_figure(data: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    order, display = _parameter_order(data)
    df_slopes = compute_all_slopes(data, order)
    df_ep = compute_endpoint_shifts(data, order)
    df_shifts = build_basin_shifts(data, order)

    fig = plt.figure(figsize=(13.2, 11.2))

    # GridSpec with standard margins and generous vertical spacing
    gs_main = gridspec.GridSpec(
        3, 1,
        figure=fig,
        height_ratios=[1.0, 1.10, 1.25],
        hspace=0.36,
        left=0.065,
        right=0.975,
        top=0.965,
        bottom=0.055,
    )

    # -----------------------------------------------------------------------
    # Top Row: Panels (a) & (b) - Shared y-axis, wspace=0.12
    # -----------------------------------------------------------------------
    gs_row0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0], wspace=0.12)
    ax_a = fig.add_subplot(gs_row0[0, 0])
    ax_b = fig.add_subplot(gs_row0[0, 1], sharey=ax_a)
    apply_clean_spines(ax_a)
    apply_clean_spines(ax_b)

    _plot_response_map_sub(ax_a, df_slopes, order, display, "beta_cn", COLOR_CN, "(a) All-15 slope response map: Base–CN", show_ylabel=True)
    _plot_response_map_sub(ax_b, df_slopes, order, display, "beta_tgd", COLOR_TGD, "(b) All-15 slope response map: Base–TGD", show_ylabel=False)

    marker_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_TEXT, markeredgecolor="white", markersize=7.0, label="Recurring signatures (um, ki, ci, im)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#8B949E", markeredgecolor="white", markersize=5.5, label="Other shared parameters"),
    ]
    ax_a.legend(handles=marker_handles, loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.5, framealpha=0.95)

    # -----------------------------------------------------------------------
    # Middle Row: Panels (c) & (d) - wspace=0.12
    # -----------------------------------------------------------------------
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[1], wspace=0.12)
    ax_c = fig.add_subplot(gs_row1[0, 0])
    ax_d = fig.add_subplot(gs_row1[0, 1])
    apply_clean_spines(ax_c)
    apply_clean_spines(ax_d)

    _plot_endpoint_sub(ax_c, df_ep, order, display, "cn_val", "cn_lo", "cn_hi", COLOR_CN, COLOR_CN_DARK, "(c) Endpoint fingerprint: Base–CN", show_ylabel=True)
    _plot_endpoint_sub(ax_d, df_ep, order, display, "tg_val", "tg_lo", "tg_hi", COLOR_TGD, COLOR_TGD_DARK, "(d) Endpoint fingerprint: Base–TGD", show_ylabel=False)

    c_handles = [
        Patch(facecolor=COLOR_CN, edgecolor=COLOR_CN_DARK, label="IC (upper bar, solid)"),
        Patch(facecolor=COLOR_CN_DPL, edgecolor=COLOR_CN_DARK, alpha=0.55, label="dPL (lower bar, light)"),
    ]
    ax_c.legend(handles=c_handles, loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8.5, framealpha=0.95)

    # -----------------------------------------------------------------------
    # Bottom Row: Panels (e), (f), (g), (h) - Shared y-axis across 4 panels, wspace=0.06
    # -----------------------------------------------------------------------
    gs_row2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[2], wspace=0.06)
    params = ["xaj_um", "xaj_ki", "xaj_ci", "xaj_im"]
    letters = ["e", "f", "g", "h"]

    ax_params = []
    for col, (p, let) in enumerate(zip(params, letters)):
        if col == 0:
            ax_p = fig.add_subplot(gs_row2[0, col])
        else:
            ax_p = fig.add_subplot(gs_row2[0, col], sharey=ax_params[0])
        apply_clean_spines(ax_p)
        ax_params.append(ax_p)

    for col, (p, let, ax_p) in enumerate(zip(params, letters, ax_params)):
        _plot_paired_parameter_panel(ax_p, df_shifts, p, let, display, is_panel_e=(col == 0), is_panel_h=(col == 3))

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
