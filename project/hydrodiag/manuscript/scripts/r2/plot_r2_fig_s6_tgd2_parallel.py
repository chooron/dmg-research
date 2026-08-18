#!/usr/bin/env python3
"""Supplementary Figure S6 (Fig. S6): Base-TGD2 parallel to main-text Figure 4.

Matched-control counterpart of F4 for the parameter-count-matched structural
control XAJ_TGD2 (canonical structure tag 'GD'). Same parameter-level quantities as
F4, with the paired shift redefined as dZ = z_Base - z_TGD2:

  (a)  Snow gradients of paired shifts (Base-TGD2), split into IC (top) and
       dPL (bottom) regimes, all 15 shared parameters (teal triangles; um/ki/ci
       rows shaded + bold as in F4 because they are expanded in (b-d)).
  (b-d) um / ki / ci  ridgeline distributions of dZ = z_Base - z_TGD2 across the
       snow regimes S1-S5 (IC deep-blue ridge above / dPL orange ridge below the
       shared baseline, identical grammar to F4).

No GIS panel: the Base-TGD2 paired shifts are small and diffuse with only weak
snow-gradient spatial organization (|corr(dZ, frac_snow)| < 0.3 for the key
parameters, vs strong Base-CN signals), so maps would be noisy/redundant; they are
deliberately omitted.

Data (all read-only, frozen):
  r2_snow_gradients_base_tgd2_summary.csv - verified Base-TGD2 gradients (a)
  r2_parameter_values_canonical.csv       - canonical z; dZ = z_Base - z_GD (b-d),
                                            GD == XAJ_TGD2 per the R2 pipeline.

Visual grammar reuses the F3/F4 module (plot_r2_figure4): fixed-bandwidth
boundary-reflected KDE, peak-normalized, mirrored IC/dPL ridges, thin zero line,
regime n in the y-axis labels.

Output: manuscript/supplement/figures/Fig_S6_R2_tgd2_parallel_figure4.png (600 DPI)
and vector PDF, consistent with the supplement figure convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_r2_figure4 as F4  # noqa: E402  (read-only reuse of grammar/constants)

from manuscript.scripts.shared.r1_plot_style import (  # noqa: E402
    MODEL_COLORS,
    apply_clean_spines,
    setup_publication_style,
)

PROJECT = Path(__file__).resolve().parents[3]
MANUSCRIPT = PROJECT / "manuscript"
RESULTS_R2 = MANUSCRIPT / "results" / "R2"
SUPP_FIG_DIR = MANUSCRIPT / "supplement" / "figures"
SUPP_FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_NAME = "Fig_S6_R2_tgd2_parallel_figure4"

# Reuse the F4 grammar constants
PARAM_ORDER = F4.PARAM_ORDER
DISPLAY = F4.DISPLAY
KEY_PARAMS = F4.KEY_PARAMS
REGIMES = F4.REGIMES
REGIME_N = F4.REGIME_N
COLOR_CN = F4.COLOR_CN
COLOR_BASE = F4.COLOR_BASE
COLOR_TGD2 = MODEL_COLORS["TGD"]  # #009988  teal (Base-TGD2 contrast)
COLOR_REF = F4.COLOR_REF
RIDGE_EDGE = F4.RIDGE_EDGE
KEY_ROW_SHADE = F4.KEY_ROW_SHADE
IC_FILL_ALPHA = F4.IC_FILL_ALPHA
DPL_FILL_ALPHA = F4.DPL_FILL_ALPHA
DPL_LINESTYLE = F4.DPL_LINESTYLE
BASELINE_COLOR = F4.BASELINE_COLOR
ZERO_LINE_COLOR = F4.ZERO_LINE_COLOR
GRID_ALPHA = F4.GRID_ALPHA
DZ_XTICKS = F4.DZ_XTICKS
DZ_GRID = F4.DZ_GRID
DZ_BW = F4.DZ_BW
RIDGE_HEIGHT = F4.RIDGE_HEIGHT


def load_tgd2_gradients() -> pd.DataFrame:
    df_tgd2 = pd.read_csv(RESULTS_R2 / "r2_snow_gradients_base_tgd2_summary.csv")
    assert len(df_tgd2) == 30 and set(df_tgd2["contrast"]) == {"Base-TGD2"}
    assert float(df_tgd2["validation_base_cn_max_abs_diff"].max()) == 0.0
    return df_tgd2


def load_base_tgd2_paired() -> pd.DataFrame:
    """dZ = z_Base - z_TGD2 per (paradigm, basin, parameter), with snow_regime.

    Derived in-script from the canonical z table (GD == XAJ_TGD2); no upstream file
    is written or modified. dPL canonical z is the within-basin 3-seed median, as in
    the R2 pipeline.
    """
    canon = pd.read_csv(RESULTS_R2 / "r2_parameter_values_canonical.csv")
    canon["basin_id"] = canon["basin_id"].astype(str).str.zfill(8)
    base = canon[canon["structure"] == "Base"].rename(columns={"z": "z_base"})
    gd = canon[canon["structure"] == "GD"].rename(columns={"z": "z_tgd2"})
    df = base.merge(
        gd[["paradigm", "basin_id", "parameter", "z_tgd2"]],
        on=["paradigm", "basin_id", "parameter"],
        how="inner",
    )
    assert len(df) == 531 * 15 * 2
    df["delta_base_minus_tgd2"] = df["z_base"] - df["z_tgd2"]
    return df


# ---------------------------------------------------------------------------
# Panel (a): snow gradients of paired shifts (Base-TGD2 only)
# ---------------------------------------------------------------------------
def panel_a_gradients(ax, df_tgd2, paradigm, shared_lim=None):
    y_pos = np.arange(len(PARAM_ORDER))
    for i, p in enumerate(PARAM_ORDER):
        if p in KEY_PARAMS:
            ax.axhspan(i - 0.5, i + 0.5, color=KEY_ROW_SHADE, linewidth=0, zorder=0)
    for i, p in enumerate(PARAM_ORDER):
        slope, lo, hi = F4.slope_ci_param(df_tgd2, paradigm, p)
        y = y_pos[i]
        ax.errorbar(
            slope,
            y,
            xerr=[[slope - lo], [hi - slope]],
            fmt="^",
            color=COLOR_TGD2,
            ecolor=COLOR_TGD2,
            elinewidth=1.1,
            capsize=2.4,
            capthick=0.9,
            markersize=5.2,
            markerfacecolor=COLOR_TGD2,
            markeredgecolor="none",
            linestyle="none",
            zorder=3,
        )
    ax.axvline(0, color=COLOR_REF, linestyle="--", linewidth=0.85, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([DISPLAY[p] for p in PARAM_ORDER], fontsize=7.5)
    for tick, p in zip(ax.get_yticklabels(), PARAM_ORDER):
        if p in KEY_PARAMS:
            tick.set_fontweight("bold")
    ax.set_ylim(-0.6, len(PARAM_ORDER) + 0.4)
    ax.set_xlabel("Snow gradient of paired shift, \u03b2", labelpad=3)
    ax.grid(True, axis="x", linestyle=":", alpha=0.12)
    if shared_lim is not None:
        lo_r, hi_r, ticks = shared_lim
        ax.set_xlim(lo_r, hi_r)
        ax.set_xticks(ticks)


# ---------------------------------------------------------------------------
# Panels (b)-(d): ridgeline distributions of dZ = z_Base - z_TGD2
# ---------------------------------------------------------------------------
def _dz_ridge_pair(ax, df_dz, parameter, xlabel=False):
    y_pos = np.arange(len(REGIMES))
    for i, reg in enumerate(REGIMES):
        y = y_pos[i]
        ax.axhline(y, color=BASELINE_COLOR, linewidth=0.6, zorder=1)
        for paradigm, sign, alpha, fill, edge, ls in [
            ("IC", +1.0, IC_FILL_ALPHA, COLOR_CN, RIDGE_EDGE, "-"),
            ("dPL", -1.0, DPL_FILL_ALPHA, COLOR_BASE, COLOR_BASE, DPL_LINESTYLE),
        ]:
            vals = df_dz[
                (df_dz["paradigm"] == paradigm)
                & (df_dz["parameter"] == parameter)
                & (df_dz["snow_regime"] == reg)
            ]["delta_base_minus_tgd2"].to_numpy(float)
            d = F4.ridge_density(vals, DZ_GRID, DZ_BW)
            curve = y + sign * RIDGE_HEIGHT * d
            ax.fill_between(
                DZ_GRID, y, curve, color=fill, alpha=alpha, linewidth=0, zorder=2
            )
            ax.plot(DZ_GRID, curve, color=edge, linestyle=ls, linewidth=0.7, zorder=3)
            med = float(np.median(vals))
            ax.plot(
                [med],
                [y + sign * RIDGE_HEIGHT * d[np.argmin(np.abs(DZ_GRID - med))]],
                marker="o",
                color=edge,
                markersize=2.4,
                markerfacecolor=edge,
                markeredgecolor="none",
                zorder=5,
            )
            q1, q3 = np.percentile(vals, [25, 75])
            ax.plot(
                [q1, q3],
                [y + sign * 0.015, y + sign * 0.015],
                color=edge,
                linewidth=0.9,
                solid_capstyle="butt",
                zorder=5,
            )
    ax.axvline(0, color=ZERO_LINE_COLOR, linestyle="--", linewidth=0.8, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{r} (n={n})" for r, n in zip(REGIMES, REGIME_N)], fontsize=7.0
    )
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.5, float(len(REGIMES)) + 0.5)
    ax.set_xticks(DZ_XTICKS)
    ax.set_xticklabels(["-1", "-0.5", "0", "0.5", "1"], fontsize=7.0)
    ax.grid(True, axis="x", linestyle=":", alpha=GRID_ALPHA)
    if xlabel:
        ax.set_xlabel(r"$\Delta z = z_{\mathrm{Base}} - z_{\mathrm{TGD2}}$", labelpad=3)


def panel_ridge(ax, df_dz, parameter, letter, xlabel=False, snow_cue=False):
    ax.set_title(
        f"({letter}) {DISPLAY[parameter]}",
        weight="bold",
        loc="left",
        pad=5,
        fontsize=9.0,
    )
    _dz_ridge_pair(ax, df_dz, parameter, xlabel=xlabel)
    if snow_cue:
        ax.text(
            0.02,
            0.985,
            "Increasing snow influence \u2191",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.4,
            color="#999999",
        )


# ---------------------------------------------------------------------------
# Figure assembly (two columns: (a) | (b-d); no GIS)
# ---------------------------------------------------------------------------
def build_figure(df_tgd2, df_dz) -> None:
    fig = plt.figure(figsize=(10.0, 9.6))
    TOP, BOT = 0.955, 0.060
    c1 = (0.070, 0.465)  # panel (a)
    c2 = (0.560, 0.985)  # panels (b)(c)(d)
    gsL = gridspec.GridSpec(
        2, 1, left=c1[0], right=c1[1], top=TOP, bottom=BOT, hspace=0.30
    )
    gsM = gridspec.GridSpec(
        3, 1, left=c2[0], right=c2[1], top=TOP, bottom=BOT, hspace=0.26
    )

    # (a) shared x-axis across both regimes (Base-TGD2 only)
    ax_a1 = fig.add_subplot(gsL[0, 0])
    apply_clean_spines(ax_a1)
    ax_a2 = fig.add_subplot(gsL[1, 0])
    apply_clean_spines(ax_a2)
    ax_a1.set_title(
        "(a) Snow gradients of paired shifts (Base\u2013TGD2)",
        weight="bold",
        loc="left",
        pad=5,
        fontsize=9.0,
    )
    cilo = min(
        F4.slope_ci_param(df_tgd2, p_, par)[1]
        for p_ in ["IC", "dPL"]
        for par in PARAM_ORDER
    )
    cihi = max(
        F4.slope_ci_param(df_tgd2, p_, par)[2]
        for p_ in ["IC", "dPL"]
        for par in PARAM_ORDER
    )
    pad = max(0.05, (cihi - cilo) * 0.08)
    lo_r = np.floor((cilo - pad) / 0.5) * 0.5
    hi_r = np.ceil((cihi + pad) / 0.5) * 0.5
    shared_lim = (lo_r, hi_r + 0.10, np.arange(lo_r, hi_r + 0.5 + 1e-9, 0.5))
    panel_a_gradients(ax_a1, df_tgd2, "IC", shared_lim=shared_lim)
    panel_a_gradients(ax_a2, df_tgd2, "dPL", shared_lim=shared_lim)
    ax_a1.set_ylabel("Parameter", labelpad=3)
    ax_a1.text(
        0.02,
        0.97,
        "IC regime",
        transform=ax_a1.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        fontweight="bold",
        color="#333333",
    )
    ax_a2.text(
        0.02,
        0.97,
        "dPL regime",
        transform=ax_a2.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        fontweight="bold",
        color="#333333",
    )

    # (b)(c)(d) ridgelines
    for row, (param, letter) in enumerate(zip(KEY_PARAMS, ["b", "c", "d"])):
        ax = fig.add_subplot(gsM[row, 0])
        apply_clean_spines(ax)
        panel_ridge(ax, df_dz, param, letter, xlabel=(row == 2), snow_cue=(row == 0))

    # shared IC/dPL ridge legend
    handles = [
        Patch(
            facecolor=COLOR_CN,
            alpha=IC_FILL_ALPHA,
            edgecolor=RIDGE_EDGE,
            linewidth=0.8,
            label="IC",
        ),
        Patch(
            facecolor=COLOR_BASE,
            alpha=DPL_FILL_ALPHA,
            edgecolor=COLOR_BASE,
            linewidth=0.7,
            linestyle=DPL_LINESTYLE,
            label="dPL",
        ),
    ]
    leg = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.028),
        ncol=2,
        frameon=False,
        fontsize=7.0,
        columnspacing=1.4,
        handlelength=1.8,
    )
    for t in leg.get_texts():
        t.set_fontsize(7.0)

    out_png = SUPP_FIG_DIR / f"{OUT_NAME}.png"
    out_pdf = SUPP_FIG_DIR / f"{OUT_NAME}.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf, format="pdf")
    print("saved:", out_png)
    print("saved:", out_pdf)
    plt.close(fig)


def main() -> None:
    setup_publication_style()
    df_tgd2 = load_tgd2_gradients()
    df_dz = load_base_tgd2_paired()
    # sanity: delta bounded in [-1, 1]; 531 basins per (paradigm, parameter)
    dz = df_dz[df_dz["parameter"].isin(KEY_PARAMS)]["delta_base_minus_tgd2"]
    assert float(dz.min()) >= -1.0 - 1e-9 and float(dz.max()) <= 1.0 + 1e-9
    assert (
        df_dz[df_dz["parameter"].isin(KEY_PARAMS)]
        .groupby(["paradigm", "parameter"])
        .size()
        == 531
    ).all()
    build_figure(df_tgd2, df_dz)
    print("Figure S6 (Base-TGD2 parallel to F4) generated.")


if __name__ == "__main__":
    main()
