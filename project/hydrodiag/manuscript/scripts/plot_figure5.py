#!/usr/bin/env python3
"""Final R3 Figure 5 (F5): 6-panel composite on limited compensation and
snow-organized internal consequences.

Scientific line (frozen, unchanged):
    Known snow-process omission can only be partially absorbed by parameter
    adjustment, while the remaining parameter and state deviations are
    systematically organized by snow-process activity (frac_snow).
    No implication that output recovery independently causes internal
    distortion; IC and dPL are distinct calibration regimes, not a benchmark;
    TGD2 is not part of the Figure 5 question (it belongs to Figure 6).

Layout (three-level asymmetric composite)
-----------------------------------------------------------------
  Row 1 (reference + observable consequence):
      (a) Correct-CN baseline (approx. 28 % width)
      (b) Output-level structural gap (approx. 72 %, IC and dPL facets)
  Row 2 (quantitative compensation diagnostics):
      (c) Gap-closure fraction F_close (approx. 44 % width)
      (d) Train-to-test compensation decay (approx. 56 % width)
  Row 3 (main R3 evidence, hero panels, approx. 43 % of figure height):
      (e) Parameter excess error C_theta vs. frac_snow (50 % width, IC+dPL on single axis)
      (f) State excess error C_state vs. frac_snow (50 % width, IC+dPL on single axis)

Visual grammar (inherited from r1_plot_style.py / F1-F3):
  * Base fitted  -> Base orange family (#EE7733 standard, #C2410C darker tone)
  * Base no-refit-> neutral grey #A0A0A0 (raw knockout reference)
  * CN           -> CN blue    #0077BB
  * Regime encoding in (e)/(f):
      - IC:  darker tone #C2410C, circle marker 'o', solid line '-'
      - dPL: warm orange #EE7733, triangle marker '^', dashed line '--'
      - Redundant marker + tone encoding within the Base orange palette
  * Descriptive snow gradient:
      - Raw basin points (alpha=0.18)
      - frac_snow-quartile medians + bootstrap 95 % CI
      - Simple connected line (no OLS regression, no parametric claim)
      - Frozen Spearman rho annotations
  * PNG only, 600 DPI, saved to manuscript/figures/ and manuscript/plots/figures/

Statistics: all headline numbers are read from the frozen post-hoc summaries
via manuscript/results/R3/figure5_summary.json (prepared by
prepare_figure5_data.py, which asserts equality with the frozen values).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_plot_style import (  # noqa: E402
    MODEL_COLORS,
    apply_clean_spines,
    setup_publication_style,
)

PROJECT = Path(__file__).resolve().parents[2]
MANUSCRIPT = PROJECT / "manuscript"
RESULTS_R3 = MANUSCRIPT / "results" / "R3"
FIG_DIR = MANUSCRIPT / "figures"
PLOTS_FIG_DIR = MANUSCRIPT / "plots" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_NAME = "Figure5_R3_final"

# ---------------------------------------------------------------------------
# Frozen visual grammar (r1_plot_style.py / F1-F3 system)
# ---------------------------------------------------------------------------
C_CN = MODEL_COLORS["CN"]  # #0077BB  correct snow-process structure
C_BASE = MODEL_COLORS["Base"]  # #EE7733  omitted-process baseline (fitted, dPL)
C_BASE_IC = "#C2410C"  # darker tone in Base family for IC regime
C_NOREFIT = "#A0A0A0"  # neutral grey: Base without recalibration
C_REF = "#999999"  # reference lines
C_TEXT = "#333333"  # annotation text
TEXT_BG = dict(
    boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.88
)


# ---------------------------------------------------------------------------
# Data loading (frozen Figure 5 package, read-only)
# ---------------------------------------------------------------------------
def load_data():
    summary = json.loads((RESULTS_R3 / "figure5_summary.json").read_text())
    tidy = pd.read_csv(RESULTS_R3 / "figure5_basin_table.csv")
    seedmed = pd.read_csv(RESULTS_R3 / "figure5_basin_seedmedian.csv")
    for df in (tidy, seedmed):
        df["basin_id"] = df["basin_id"].astype(str).str.zfill(8)
        df["seed"] = df["seed"].fillna("").astype(str)
    return tidy, seedmed, summary


def ecdf(vals: np.ndarray):
    v = np.sort(np.asarray(vals, dtype=np.float64))
    v = v[np.isfinite(v)]
    return v, np.arange(1, len(v) + 1) / len(v)


# ---------------------------------------------------------------------------
# Panel (a): correct-CN baseline reference (compact)
# ---------------------------------------------------------------------------
def panel_a(ax, seedmed, summary):
    pa = summary["panel_a_cn_deficit"]
    ax.set_title("(a) Correct-CN baseline", weight="bold", loc="left", pad=5)
    styles = [  # (regime, period, linestyle, linewidth, alpha, zorder)
        ("IC", "test", "-", 1.8, 1.00, 5),
        ("dPL", "test", (0, (4.0, 2.0)), 1.8, 1.00, 5),
        ("IC", "train", "-", 1.0, 0.45, 4),
        ("dPL", "train", (0, (4.0, 2.0)), 1.0, 0.45, 4),
    ]
    for reg, period, ls, lw, alpha, zo in styles:
        sub = seedmed[(seedmed["paradigm"] == reg) & (seedmed["period"] == period)]
        d = 1.0 - sub["kge_cn"].to_numpy()
        x, y = ecdf(d)
        ax.step(
            x,
            y,
            where="post",
            color=C_CN,
            linestyle=ls,
            linewidth=lw,
            alpha=alpha,
            zorder=zo,
        )
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.0)
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    ax.set_xticklabels(["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$10^{0}$"])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Correct-CN deficit, $1 - \\mathrm{KGE}_{CN}$", labelpad=2)
    ax.set_ylabel("Cumulative probability", labelpad=2)
    ax.grid(True, axis="y", linestyle=":", alpha=0.25)
    handles = [
        Line2D([0], [0], color=C_CN, linestyle="-", linewidth=1.6, label="IC"),
        Line2D(
            [0], [0], color=C_CN, linestyle=(0, (4.0, 2.0)), linewidth=1.6, label="dPL"
        ),
        Line2D(
            [0],
            [0],
            color=C_CN,
            linestyle="-",
            linewidth=1.0,
            alpha=0.45,
            label="train",
        ),
        Line2D([0], [0], color=C_CN, linestyle="-", linewidth=1.8, label="test"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=True,
        framealpha=0.90,
        edgecolor="none",
        fontsize=7.6,
        ncol=2,
        columnspacing=0.8,
        handlelength=1.4,
    )
    ax.text(
        0.97,
        0.95,
        f"test median deficit:\n{pa['IC_test']['median']:.4f} (IC) / {pa['dPL_test']['median']:.4f} (dPL)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        color=C_TEXT,
        linespacing=1.3,
        bbox=TEXT_BG,
    )


# ---------------------------------------------------------------------------
# Panel (b): output-level structural gap (test) — wide panel, IC/dPL facets
# ---------------------------------------------------------------------------
def _facet_gap_ecdf(ax, seedmed, reg):
    te = seedmed[(seedmed["paradigm"] == reg) & (seedmed["period"] == "test")]
    series = [
        (
            "Base no-refit",
            te["kge_base_no_refit"].to_numpy(),
            C_NOREFIT,
            (0, (4.0, 2.0)),
            1.5,
        ),
        ("Base fitted", te["kge_base"].to_numpy(), C_BASE, "-", 1.6),
        ("CN", te["kge_cn"].to_numpy(), C_CN, "-", 1.7),
    ]
    for label, vals, color, ls, lw in series:
        x, y = ecdf(vals)
        ax.step(x, y, where="post", color=color, linestyle=ls, linewidth=lw, zorder=4)
        med = float(np.median(vals))
        ax.axvline(med, color=color, linestyle=":", linewidth=0.7, alpha=0.6, zorder=2)
    ax.set_xlim(-0.30, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([-0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, axis="y", linestyle=":", alpha=0.25)
    # y-axis is shared with panel (a) (sharey); keep the tick marks and
    # labels visible on the (b) facets too
    ax.text(
        0.02,
        0.94,
        f"{reg} regime",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=C_TEXT,
    )
    ax.text(
        0.98,
        0.97,
        "median 0.898 | 0.899 | 0.993"
        if reg == "IC"
        else "median 0.898 | 0.908 | 0.995",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        color=C_TEXT,
        bbox=TEXT_BG,
    )


def panel_b(ax_ic, ax_dpl, seedmed):
    ax_ic.set_title(
        "(b) Output-level structural gap (test)", weight="bold", loc="left", pad=5
    )
    _facet_gap_ecdf(ax_ic, seedmed, "IC")
    _facet_gap_ecdf(ax_dpl, seedmed, "dPL")
    # y-axis label lives on (a) only (shared axis)
    ax_ic.set_xlabel("Test KGE vs. $Q^*$", labelpad=2)
    ax_dpl.set_xlabel("Test KGE vs. $Q^*$", labelpad=2)
    handles = [
        Line2D(
            [0],
            [0],
            color=C_NOREFIT,
            linestyle=(0, (4.0, 2.0)),
            linewidth=1.5,
            label="Base no-refit",
        ),
        Line2D(
            [0], [0], color=C_BASE, linestyle="-", linewidth=1.6, label="Base fitted"
        ),
        Line2D([0], [0], color=C_CN, linestyle="-", linewidth=1.7, label="CN"),
    ]
    ax_ic.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.04),
        frameon=True,
        framealpha=0.95,
        edgecolor="none",
        fontsize=7.8,
    )


# ---------------------------------------------------------------------------
# Panel (c): gap-closure fraction F_close
# ---------------------------------------------------------------------------
def _group_summary(ax, xi, vals, entry, reg, period, rng):
    """Jittered basin cloud + IQR box + median + bootstrap CI for one group."""
    valid = vals[np.isfinite(vals)]
    win = valid[(valid >= -0.5) & (valid <= 1.75)]  # documented display window
    jx = xi + rng.uniform(-0.16, 0.16, len(win))
    marker = "o" if reg == "IC" else "^"
    face = C_BASE if period == "train" else "white"
    ax.scatter(
        jx,
        win,
        s=7,
        alpha=0.28,
        marker=marker,
        color=C_BASE,
        facecolors=face,
        edgecolors=C_BASE,
        linewidths=0.4,
        zorder=2,
    )
    med, q25, q75 = entry["median"], entry["q25"], entry["q75"]
    lo, hi = entry["boot_ci_median_display"]
    bw = 0.26
    ax.add_patch(
        plt.Rectangle(
            (xi - bw / 2, q25),
            bw,
            q75 - q25,
            facecolor=C_BASE,
            edgecolor=C_BASE,
            alpha=0.35,
            linewidth=0.8,
            zorder=3,
        )
    )
    ax.plot(
        [xi - bw / 2, xi + bw / 2], [med, med], color=C_BASE, linewidth=1.6, zorder=4
    )
    ax.errorbar(
        [xi],
        [med],
        yerr=[[med - lo], [hi - med]],
        fmt="none",
        ecolor=C_TEXT,
        elinewidth=1.2,
        capsize=2.6,
        capthick=1.1,
        zorder=5,
    )
    return med


def panel_c(ax, seedmed, summary):
    pc = summary["panel_c_f_close"]
    ax.set_title(
        "(c) Gap-closure fraction $F_{close}$", weight="bold", loc="left", pad=5
    )
    groups = [("IC", "train"), ("IC", "test"), ("dPL", "train"), ("dPL", "test")]
    xs = np.arange(len(groups))
    rng = np.random.default_rng(20260730)
    for xi, (reg, period) in enumerate(groups):
        entry = pc[f"{reg}_{period}"]
        sub = seedmed[(seedmed["paradigm"] == reg) & (seedmed["period"] == period)]
        vals = sub["F_close"].to_numpy()
        med = _group_summary(ax, xi, vals, entry, reg, period, rng)
        # per-seed dPL medians (frozen) as small open diamonds
        if reg == "dPL":
            for s, smed in enumerate(entry["seed_medians"]):
                ax.plot(
                    xi + 0.24 + 0.03 * s,
                    smed,
                    marker="D",
                    markersize=2.6,
                    color=C_BASE_IC,
                    markerfacecolor="none",
                    markeredgewidth=0.7,
                    zorder=5,
                )
        ax.text(
            xi,
            0.98,
            f"{med:.3f}",
            ha="center",
            va="top",
            fontsize=8.0,
            color=C_TEXT,
            fontweight="bold",
        )
    # reference lines: no closure (0) and full closure (1)
    ax.axhline(0.0, color=C_REF, linestyle="-", linewidth=0.8, alpha=0.6, zorder=1)
    ax.axhline(1.0, color=C_REF, linestyle=":", linewidth=0.8, alpha=0.8, zorder=1)
    ax.text(
        0.01, 1.01, "full closure", ha="left", va="bottom", fontsize=7.2, color=C_REF
    )
    ax.set_xlim(-0.55, 3.75)
    ax.set_ylim(-0.80, 1.30)
    ax.set_xticks(xs)
    n_vals = [str(pc[f"{r}_{p}"]["n_valid"]) for r, p in groups]
    ax.set_xticklabels(
        [
            f"IC train\nn = {n_vals[0]}",
            f"IC test\nn = {n_vals[1]}",
            f"dPL train\nn = {n_vals[2]}",
            f"dPL test\nn = {n_vals[3]}",
        ],
        fontsize=8.1,
    )
    ax.set_ylabel("Gap-closure fraction $F_{close}$", labelpad=2)
    ax.grid(True, axis="y", linestyle=":", alpha=0.25)
    # small legend: train/test fill convention
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=C_BASE,
            linestyle="none",
            markerfacecolor=C_BASE,
            markersize=4.5,
            label="train",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=C_BASE,
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=C_BASE,
            markersize=4.5,
            label="test",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.97),
        frameon=True,
        framealpha=0.92,
        edgecolor="none",
        fontsize=7.2,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    ax.text(
        0.99,
        0.02,
        "unclipped $F_{close}$; $\\approx$3–10 % of basins beyond window",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#666666",
    )


# ---------------------------------------------------------------------------
# Panel (d): train-to-test compensation decay (direct distribution)
# ---------------------------------------------------------------------------
def panel_d(ax, seedmed, summary):
    pd_ = summary["panel_d_decay"]
    ax.set_title(
        "(d) Train-to-test compensation decay", weight="bold", loc="left", pad=5
    )
    rng = np.random.default_rng(20260730)
    for xi, reg_name in enumerate(["IC", "dPL"]):
        agg = pd_[f"{reg_name}_agg"]
        sub = seedmed[seedmed["paradigm"] == reg_name].drop_duplicates("basin_id")
        d = sub["decay_G_base"].to_numpy()
        d = d[np.isfinite(d)]
        # documented display window (unclipped statistic; ~1-4 % beyond)
        win = d[(d >= -0.1) & (d <= 0.15)]
        jx = xi + rng.uniform(-0.16, 0.16, len(win))
        marker = "o" if reg_name == "IC" else "^"
        col = C_BASE_IC if reg_name == "IC" else C_BASE
        ax.scatter(
            jx,
            win,
            s=7,
            alpha=0.28,
            marker=marker,
            color=col,
            facecolors="white",
            edgecolors=col,
            linewidths=0.4,
            zorder=2,
        )
        q25, q75 = float(np.quantile(d, 0.25)), float(np.quantile(d, 0.75))
        med = agg["median"]
        # IC: frozen bootstrap CI; dPL: display bootstrap CI (seed-aggregated)
        lo, hi = (
            pd_["IC"]["boot_ci_median"]
            if reg_name == "IC"
            else agg["boot_ci_median_display"]
        )
        bw = 0.26
        ax.add_patch(
            plt.Rectangle(
                (xi - bw / 2, q25),
                bw,
                q75 - q25,
                facecolor=col,
                edgecolor=col,
                alpha=0.35,
                linewidth=0.8,
                zorder=3,
            )
        )
        ax.plot(
            [xi - bw / 2, xi + bw / 2], [med, med], color=col, linewidth=1.6, zorder=4
        )
        ax.errorbar(
            [xi],
            [med],
            yerr=[[med - lo], [hi - med]],
            fmt="none",
            ecolor=C_TEXT,
            elinewidth=1.2,
            capsize=2.6,
            capthick=1.1,
            zorder=5,
        )
        ax.text(
            xi,
            0.180,
            f"+{med:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=C_TEXT,
            fontweight="bold",
        )
        ax.text(
            xi,
            0.152,
            f"{agg['frac_gt_0'] * 100:.0f} % > 0",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#666666",
        )
    ax.axhline(0.0, color=C_REF, linestyle="--", linewidth=1.0, zorder=1)
    ax.text(
        1.42, 0.008, "train = test", ha="right", va="bottom", fontsize=7.2, color=C_REF
    )
    ax.set_xlim(-0.55, 1.6)
    ax.set_ylim(-0.18, 0.22)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["IC", "dPL"], fontsize=9.0)
    ax.set_ylabel("Compensation decay $\\Delta G_{base}$", labelpad=2)
    ax.grid(True, axis="y", linestyle=":", alpha=0.25)
    ax.text(
        0.99,
        0.02,
        "$\\approx$1–4 % of basins beyond window",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#666666",
    )


# ---------------------------------------------------------------------------
# Panels (e)/(f): excess errors vs frac_snow (Base) — Hero Panels (Merged IC+dPL)
# ---------------------------------------------------------------------------
def _plot_merged_excess(ax, seedmed, metric, pef_ic, pef_dpl, show_legend=False):
    """Plot IC and dPL on one shared axis with restrained redundant encoding."""
    col = "C_theta_base" if metric == "C_theta" else "C_state_base"

    # Regimes to overlay: (regime_key, entry, marker, color, linestyle, x_offset, label)
    regimes = [
        ("IC", pef_ic, "o", C_BASE_IC, "-", -0.008, "IC (CMA-ES)"),
        ("dPL", pef_dpl, "^", C_BASE, "--", +0.008, "dPL (neural)"),
    ]

    for reg_name, entry, marker, color, ls, x_off, label in regimes:
        sub = seedmed[(seedmed["paradigm"] == reg_name) & (seedmed["period"] == "test")]
        x = sub["frac_snow"].to_numpy()
        y = sub[col].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]

        # Raw basin points (low alpha)
        ax.scatter(
            x,
            y,
            s=7,
            alpha=0.18,
            marker=marker,
            color=color,
            edgecolors="none",
            zorder=2,
        )

        # Descriptive gradient: quartile medians + bootstrap 95 % CI
        bins = entry["quartile_bins"]
        bx = np.asarray([b["frac_snow_median"] for b in bins]) + x_off
        bm = np.asarray([b["median"] for b in bins])
        blo = np.asarray([b["boot_ci_median_display"][0] for b in bins])
        bhi = np.asarray([b["boot_ci_median_display"][1] for b in bins])

        ax.plot(bx, bm, color=color, linestyle=ls, linewidth=1.6, alpha=0.95, zorder=4)
        ax.errorbar(
            bx,
            bm,
            yerr=[bm - blo, bhi - bm],
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.3,
            capsize=2.6,
            capthick=1.1,
            markersize=5.5 if marker == "o" else 6.0,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=5,
        )

    y_lo, y_hi = pef_ic["y_display_limits"]
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Basin snow fraction, $f_{snow}$", labelpad=2)
    ax.grid(True, axis="y", linestyle=":", alpha=0.25)

    # Compact Spearman rho box
    rho_ic = pef_ic["spearman_frozen"][0]
    rho_dpl = pef_dpl["spearman_frozen"]
    rho_txt = (
        f"IC:  $\\rho$ = +{rho_ic:.2f}\n"
        f"dPL: $\\rho$ = +{min(rho_dpl):.2f} .. +{max(rho_dpl):.2f}"
    )
    ax.text(
        0.98,
        0.04,
        rho_txt,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color=C_TEXT,
        linespacing=1.3,
        bbox=TEXT_BG,
    )

    # Shared legend in panel (e) only
    if show_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color=C_BASE_IC,
                linestyle="-",
                linewidth=1.5,
                markersize=5.0,
                markerfacecolor=C_BASE_IC,
                markeredgecolor="white",
                markeredgewidth=0.5,
                label="IC (CMA-ES)",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color=C_BASE,
                linestyle="--",
                linewidth=1.5,
                markersize=5.5,
                markerfacecolor=C_BASE,
                markeredgecolor="white",
                markeredgewidth=0.5,
                label="dPL (neural)",
            ),
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.97),
            frameon=True,
            framealpha=0.92,
            edgecolor="none",
            fontsize=7.8,
        )

    # Unobtrusive out-of-display note if any
    frac_out = max(pef_ic["frac_beyond_y_display"], pef_dpl["frac_beyond_y_display"])
    if frac_out > 0:
        ax.text(
            0.98,
            0.17,
            f"{frac_out * 100:.1f} % of basins beyond axis",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.8,
            color="#666666",
        )


def panels_ef(ax_e, ax_f, seedmed, summary):
    pef = summary["panels_ef_excess_vs_frac_snow"]
    ax_e.set_title(
        "(e) Parameter excess error vs. snow activity", weight="bold", loc="left", pad=5
    )
    ax_f.set_title(
        "(f) State excess error vs. snow activity", weight="bold", loc="left", pad=5
    )

    _plot_merged_excess(
        ax_e,
        seedmed,
        "C_theta",
        pef["C_theta_IC"],
        pef["C_theta_dPL"],
        show_legend=True,
    )
    _plot_merged_excess(
        ax_f,
        seedmed,
        "C_state",
        pef["C_state_IC"],
        pef["C_state_dPL"],
        show_legend=False,
    )

    ax_e.set_ylabel("Parameter excess error $C_\\theta$", labelpad=2)
    ax_f.set_ylabel("State excess error $C_S$", labelpad=2)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def build_figure(tidy, seedmed, summary) -> None:
    # Three rows with approximately equal heights; Row 1 = reference +
    # structural gap, Row 2-3 = the balanced near-square 2x2 lower block
    # (c)(d)(e)(f).  The canvas is compact in width relative to height so the
    # lower-block panels are less flat and closer to square.
    fig = plt.figure(figsize=(11.4, 13.6))
    gs = gridspec.GridSpec(
        3,
        1,
        height_ratios=[1, 1, 1],
        hspace=0.28,
        left=0.055,
        right=0.99,
        top=0.965,
        bottom=0.05,
    )

    # Row 1: (a) ~28 % | (b) ~72 %
    gs1 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[0], width_ratios=[0.28, 0.72], wspace=0.16
    )
    ax_a = fig.add_subplot(gs1[0, 0])
    apply_clean_spines(ax_a)
    gsb = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs1[0, 1], wspace=0.14)
    # (a) and the two (b) facets share one y-axis (cumulative probability);
    # ticks/labels stay visible on the (b) facets (sharey keeps scales linked).
    ax_b1 = fig.add_subplot(gsb[0, 0], sharey=ax_a)
    apply_clean_spines(ax_b1)
    ax_b2 = fig.add_subplot(gsb[0, 1], sharey=ax_a)
    apply_clean_spines(ax_b2)
    # with sharey, matplotlib hides the secondary axes' y tick labels by default;
    # re-enable them so the shared scale is readable on all three panels
    ax_b1.tick_params(axis="y", labelleft=True)
    ax_b2.tick_params(axis="y", labelleft=True)
    panel_a(ax_a, seedmed, summary)
    panel_b(ax_b1, ax_b2, seedmed)

    # Row 2: (c) 50 % | (d) 50 % (matching geometry with Row 3)
    gs2 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[1], width_ratios=[0.50, 0.50], wspace=0.12
    )
    ax_c = fig.add_subplot(gs2[0, 0])
    apply_clean_spines(ax_c)
    ax_d = fig.add_subplot(gs2[0, 1])
    apply_clean_spines(ax_d)
    panel_c(ax_c, seedmed, summary)
    panel_d(ax_d, seedmed, summary)

    # Row 3: (e) 50 % | (f) 50 % (Single axis each, merged IC+dPL panels)
    gs3 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[2], width_ratios=[0.50, 0.50], wspace=0.12
    )
    ax_e = fig.add_subplot(gs3[0, 0])
    apply_clean_spines(ax_e)
    ax_f = fig.add_subplot(gs3[0, 1])
    apply_clean_spines(ax_f)
    panels_ef(ax_e, ax_f, seedmed, summary)

    for out_dir in (FIG_DIR, PLOTS_FIG_DIR):
        plt.savefig(out_dir / f"{OUT_NAME}.png", dpi=600)
        print("saved:", out_dir / f"{OUT_NAME}.png")
    plt.close()


def main() -> None:
    setup_publication_style()
    # Moderate font up-scaling local to Figure 5 (manuscript readability).
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
        }
    )
    tidy, seedmed, summary = load_data()
    build_figure(tidy, seedmed, summary)
    print("Final Figure 5 (6-panel composite) generated successfully.")


if __name__ == "__main__":
    main()
