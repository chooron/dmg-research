#!/usr/bin/env python3
"""Final R2 Figure 3 (F3): 5-panel compact structure-level parameter-space diagnosis.

Role: structure-level parameter-space diagnosis (not a parameter-specific figure, not an
IC-vs-dPL benchmark). Frozen R2 positioning:
  1. A missing explicit snow process leaves systematic reorganization traces in the
     shared parameter space (Base–CN is the primary structural estimand).
  2. The reorganization strengthens with snow influence.
  3. IC and dPL are contrasting parameter-constraint regimes, not competitors.
  4. TGD2 (XAJ_TGD2) is the parameter-count-matched generic temperature-memory control
     (both CN and TGD2 are D = 17).
  5. Which parameters carry the reorganization is left to Figure 4.

Layout (2x3 composite, near-square equal cells)
-----------------------------------------------
  Row 1: (a) Structural separation under IC | (b) Structural separation under dPL
         | (d) Sources of excess separation (merged: IC blue / dPL orange colour
         families, dark shade = between, light shade = within)
  Row 2: (c) HERO — Snow-dependent structural separation (two facets: left IC, right
         dPL; continuous frac_snow relationship: basin-level scatter + regression
         line & slope-CI wedge + S1–S5 binned markers) | (e) Snow-gradient summary
         (slopes with CIs)

The paired Δβ difference is no longer a main panel; it is reported in the caption and
the execution report.

Visual language matches F1/F2 (r1_plot_style.py + plot_r1_figure2.py):
  * Base–CN  -> CN  blue #0077BB, square marker, solid line
  * Base–TGD2-> TGD teal #009988, triangle marker, dashed line
  * within baseline = neutral grey #A0A0A0; references = #999999
  * Regime (IC/dPL) carried by panel position / facet; colour encodes structural
    contrast (or quantity/neutrality in (d)).
  * Colour-vision-deficiency friendly: contrast also encoded by marker shape and line
    style; zero orange/blue IC-dPL colour system.

Statistics: all medians / CIs / slopes read from the frozen-style TGD2 specificity CSVs.
The only locally computed quantities are (i) OLS intercepts for drawing the regression
lines (deterministic; slopes reproduce the frozen values exactly, asserted), (ii)
descriptive above-1:1 prevalences, and (iii) bin-median frac_snow positions for the
binned markers. No upstream analysis is modified.

Output: high-resolution PNG only (600 DPI), manuscript convention.
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
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_plot_style import (  # noqa: E402
    MODEL_COLORS,
    MODEL_MARKERS,
    apply_clean_spines,
    setup_publication_style,
)

PROJECT = Path(__file__).resolve().parents[2]
MANUSCRIPT = PROJECT / "manuscript"
RESULTS_R2 = MANUSCRIPT / "results" / "R2"
PLOTS_FIG_DIR = MANUSCRIPT / "plots" / "figures"
PLOTS_FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_NAME = "Figure3_R2_final"

# ---------------------------------------------------------------------------
# Frozen canonical definitions
# ---------------------------------------------------------------------------
REGIMES = ["S1", "S2", "S3", "S4", "S5"]
REGIME_N = [165, 156, 121, 34, 55]
REGIME_XTICK_LABELS = [f"S{i + 1}\n(n={n})" for i, n in enumerate(REGIME_N)]
REGIME_XTICK_SHORT = [
    f"S{i + 1}" for i in range(len(REGIMES))
]  # sample sizes in caption
REGIME_BOUNDS = [0.05, 0.15, 0.30, 0.50]  # S1–S5 bin boundaries (frac_snow)

CN_PRIMARY = "Base-CN"
CN_TGD2 = "Base-TGD2"

# ---------------------------------------------------------------------------
# Visual grammar (F1/F2 system)
# ---------------------------------------------------------------------------
COLOR_BETWEEN = MODEL_COLORS["CN"]  # #0077BB  Base–CN contrast
COLOR_TGD2 = MODEL_COLORS["TGD"]  # #009988  Base–TGD2 contrast
COLOR_WITHIN = "#A0A0A0"  # within-structure baseline (neutral)
COLOR_REF = "#999999"  # reference lines (F2 grey)
COLOR_NEUTRAL = "#333333"  # F2 neutral for dark annotation text
# (d) merged-panel colour families: IC = blue family, dPL = orange family;
# within each regime the between series uses the dark shade (solid) and the within
# series the light shade (dashed) — same-colour-family light/dark encoding.
COLOR_IC_DARK = MODEL_COLORS["CN"]  # #0077BB deep blue (IC between)
COLOR_IC_LIGHT = "#9CC4E4"  # light blue (IC within)
COLOR_DPL_DARK = MODEL_COLORS["Base"]  # #EE7733 deep orange (dPL between)
COLOR_DPL_LIGHT = "#F4C29E"  # light orange (dPL within)

CONTRAST_STYLE = {
    CN_PRIMARY: {"color": COLOR_BETWEEN, "mk": "s", "ls": "-"},
    CN_TGD2: {"color": COLOR_TGD2, "mk": "^", "ls": "--"},
}


# ---------------------------------------------------------------------------
# Data loading (frozen-style TGD2 specificity outputs, read-only)
# ---------------------------------------------------------------------------
def load_data():
    df_b = pd.read_csv(RESULTS_R2 / "r2_tgd2_specificity_basin_level.csv")
    df_b["basin_id"] = df_b["basin_id"].astype(str).str.zfill(8)
    df_s = pd.read_csv(RESULTS_R2 / "r2_tgd2_specificity_summary.csv")
    df_r = pd.read_csv(RESULTS_R2 / "r2_tgd2_specificity_regressions.csv")
    return df_b, df_s, df_r


def regime_series(df_s, paradigm, contrast, metric):
    meds, lows, highs = [], [], []
    for reg in REGIMES:
        row = df_s[
            (df_s["paradigm"] == paradigm)
            & (df_s["contrast"] == contrast)
            & (df_s["stratum"] == reg)
            & (df_s["metric"] == metric)
        ]
        assert len(row) == 1, f"missing {paradigm} {contrast} {reg} {metric}"
        meds.append(float(row["median"].iloc[0]))
        lows.append(float(row["ci_lower"].iloc[0]))
        highs.append(float(row["ci_upper"].iloc[0]))
    return np.asarray(meds), np.asarray(lows), np.asarray(highs)


def slope_ci(df_r, paradigm, contrast, stratum="Full531"):
    row = df_r[
        (df_r["paradigm"] == paradigm)
        & (df_r["contrast"] == contrast)
        & (df_r["stratum"] == stratum)
        & (df_r["dependent_var"] == "excess")
    ]
    assert len(row) == 1, f"missing slope {paradigm} {contrast} {stratum}"
    return (
        float(row["slope"].iloc[0]),
        float(row["slope_ci_lower"].iloc[0]),
        float(row["slope_ci_upper"].iloc[0]),
    )


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def panel_ab_scatter(ax, df_b, paradigm, frac_cn, frac_tg, title, lim):
    """(a)/(b): basin-level separation vs within-structure variability for BOTH
    structural contrasts (Base–CN blue squares, Base–TGD2 teal triangles)."""
    ax.set_title(title, weight="bold", loc="left", pad=6)
    for contrast, alpha in [(CN_PRIMARY, 0.34), (CN_TGD2, 0.26)]:
        st = CONTRAST_STYLE[contrast]
        sub = df_b[(df_b["paradigm"] == paradigm) & (df_b["contrast"] == contrast)]
        ax.scatter(
            sub["within_pooled"],
            sub["between_all"],
            s=8,
            alpha=alpha,
            color=st["color"],
            marker=st["mk"],
            edgecolors="none",
            linewidths=0,
            zorder=3,
        )
    ax.plot(
        [0, lim], [0, lim], color=COLOR_REF, linestyle="--", linewidth=1.0, zorder=1
    )
    ticks = [0.0, 0.25, 0.5, 0.75] if lim >= 0.75 else [0.0, 0.25, 0.5, 0.65]
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Within-structure variability (RMS)", labelpad=3)
    ax.set_ylabel("Between-structure distance (RMS)", labelpad=3)
    ax.grid(True, linestyle=":", alpha=0.25)
    ax.text(
        0.98,
        0.03,
        f"Base–CN: {frac_cn * 100:.1f} % above 1:1\n"
        f"Base–TGD2: {frac_tg * 100:.1f} % above 1:1",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        fontweight="normal",
        linespacing=1.5,
    )


def _facet_excess(ax, df_b, df_s, df_r, paradigm):
    """Draw one (c) facet: scatter + regression (slope CI wedge) + S1–S5 binned markers."""
    x_idx = 0.01  # horizontal offset between the two contrasts' binned markers
    for contrast, sgn in [(CN_PRIMARY, -x_idx), (CN_TGD2, +x_idx)]:
        st = CONTRAST_STYLE[contrast]
        sub = df_b[(df_b["paradigm"] == paradigm) & (df_b["contrast"] == contrast)]
        x, y = sub["frac_snow"].to_numpy(), sub["excess"].to_numpy()
        # basin-level continuous relationship (low-alpha scatter)
        # basin-level continuous relationship (very low-alpha scatter; the regression
        # line, CI wedge and binned markers carry the main visual weight)
        ax.scatter(
            x,
            y,
            s=4,
            alpha=0.11,
            color=st["color"],
            marker=st["mk"],
            edgecolors="none",
            linewidths=0,
            zorder=2,
        )
        # regression line: frozen slope + locally derived OLS intercept (asserted equal)
        slope, lo, hi = slope_ci(df_r, paradigm, contrast)
        b, a = np.polyfit(x, y, 1)
        assert abs(b - slope) < 1e-9, f"{paradigm} {contrast} local slope mismatch"
        x_line = np.linspace(0.0, 1.0, 100)
        ax.plot(
            x_line,
            a + slope * x_line,
            color=st["color"],
            linestyle=st["ls"],
            linewidth=1.4,
            zorder=4,
        )
        # slope-uncertainty wedge anchored at the sample centroid
        mx, my = float(x.mean()), float(y.mean())
        ax.fill_between(
            x_line,
            my + lo * (x_line - mx),
            my + hi * (x_line - mx),
            color=st["color"],
            alpha=0.15,
            zorder=1,
            linewidth=0,
        )
        # S1–S5 binned summary (median + 95% CI from frozen summary) as larger markers
        bmeds, blows, bhighs = regime_series(df_s, paradigm, contrast, "excess")
        bfrac = [
            float(
                df_b[(df_b["paradigm"] == paradigm) & (df_b["snow_regime"] == reg)][
                    "frac_snow"
                ].median()
            )
            for reg in REGIMES
        ]
        bx = np.asarray(bfrac) + sgn
        ax.errorbar(
            bx,
            bmeds,
            yerr=[bmeds - blows, bhighs - bmeds],
            fmt=st["mk"],
            color=st["color"],
            ecolor=st["color"],
            elinewidth=1.3,
            capsize=2.5,
            capthick=1.0,
            markersize=7.0,
            markerfacecolor=st["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=5,
        )
    ax.axhline(0, color=COLOR_REF, linestyle="--", linewidth=0.9, zorder=1)
    for bnd in REGIME_BOUNDS:  # subtle S1–S5 bin boundary guides
        ax.axvline(
            bnd, color=COLOR_REF, linestyle=":", linewidth=0.6, alpha=0.4, zorder=1
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Basin snow fraction, $f_{\\mathrm{snow}}$", labelpad=3)
    ax.grid(True, axis="y", linestyle=":", alpha=0.25)
    # small S1–S5 bin labels at the top edge
    for reg, bfrac in zip(REGIMES, [0.025, 0.10, 0.225, 0.40, 0.75]):
        ax.text(
            bfrac,
            0.965,
            reg,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=5.5,
            color="#777777",
        )


def panel_c_hero(ax_ic, ax_dpl, df_b, df_s, df_r):
    """(c) HERO: continuous snow-dependent structural separation (IC left, dPL right)."""
    ax_ic.set_title(
        "(c) Snow-dependent structural separation", weight="bold", loc="left", pad=6
    )
    ylims = {"IC": None, "dPL": None}
    for ax, paradigm in [(ax_ic, "IC"), (ax_dpl, "dPL")]:
        _facet_excess(ax, df_b, df_s, df_r, paradigm)
        # facet y-limits from the basin-level excess range with margin
        sub = df_b[df_b["paradigm"] == paradigm]
        ymin = float(sub["excess"].min()) - 0.05
        ymax = float(sub["excess"].max()) + 0.06
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel("Excess distance (RMS)", labelpad=3)
        ax.text(
            0.02,
            0.93,
            f"{paradigm} regime",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            fontweight="bold",
            color="#333333",
        )
        ylims[paradigm] = (ymin, ymax)
    # one shared hero legend (IC facet, below the facet header)
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color=COLOR_BETWEEN,
            linestyle="-",
            markersize=5.0,
            label="Base–CN",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color=COLOR_TGD2,
            linestyle="--",
            markersize=5.0,
            label="Base–TGD2",
        ),
    ]
    ax_ic.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.78),
        frameon=True,
        framealpha=0.90,
        edgecolor="none",
        fontsize=7.0,
    )


def panel_d_merged(ax, df_s):
    """(d) MERGED: Base–CN between vs within for IC and dPL on one shared axis.
    IC = blue colour family, dPL = orange colour family; within each regime the
    between series uses the dark shade (solid) and the within series the light
    shade (dashed) — same-colour-family light/dark encoding per regime."""
    x_idx = np.arange(len(REGIMES))
    all_vals = []
    for paradigm, dark, light in [
        ("IC", COLOR_IC_DARK, COLOR_IC_LIGHT),
        ("dPL", COLOR_DPL_DARK, COLOR_DPL_LIGHT),
    ]:
        b_med, b_lo, b_hi = regime_series(df_s, paradigm, CN_PRIMARY, "between_all")
        w_med, w_lo, w_hi = regime_series(df_s, paradigm, CN_PRIMARY, "within_pooled")
        all_vals += [
            float(b_lo.min()),
            float(b_hi.max()),
            float(w_lo.min()),
            float(w_hi.max()),
        ]
        ax.errorbar(
            x_idx,
            b_med,
            yerr=[b_med - b_lo, b_hi - b_med],
            marker="s",
            linestyle="-",
            color=dark,
            ecolor=dark,
            elinewidth=1.4,
            capsize=3.0,
            capthick=1.1,
            markersize=5.0,
            markerfacecolor=dark,
            zorder=3,
            label=f"{paradigm} between",
        )
        ax.errorbar(
            x_idx,
            w_med,
            yerr=[w_med - w_lo, w_hi - w_med],
            marker="s",
            linestyle="--",
            color=light,
            ecolor=light,
            elinewidth=1.0,
            capsize=2.0,
            capthick=0.9,
            markersize=3.5,
            markerfacecolor=light,
            zorder=2,
            label=f"{paradigm} within",
        )
    ax.set_title("(d) Sources of excess separation", weight="bold", loc="left", pad=6)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(REGIME_XTICK_SHORT, fontsize=6.8)
    ax.set_xlabel("Snow regime", labelpad=3)
    ax.set_ylabel("RMS distance", labelpad=3)
    ylo = max(0.0, float(np.min(all_vals)) - 0.03)
    yhi = float(np.max(all_vals)) + 0.03
    ax.set_ylim(ylo, yhi)
    ax.grid(True, axis="y", linestyle=":", alpha=0.25)
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="-",
            color=COLOR_IC_DARK,
            markersize=5.0,
            label="IC between",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="--",
            color=COLOR_IC_LIGHT,
            markersize=3.5,
            label="IC within",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="-",
            color=COLOR_DPL_DARK,
            markersize=5.0,
            label="dPL between",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="--",
            color=COLOR_DPL_LIGHT,
            markersize=3.5,
            label="dPL within",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="none",
        fontsize=6.2,
    )


def panel_e_slopes(ax, df_r):
    """(e) Gradient summary: Base–CN and Base–TGD2 slopes with CIs across
    IC/dPL x Full/Excl-S5."""
    ax.set_title("(e) Snow-gradient summary", weight="bold", loc="left", pad=6)
    ax.axvline(0, color=COLOR_REF, linestyle="--", linewidth=0.9, zorder=1)
    rows = [  # (row centre y, paradigm, stratum, row label)
        (3.0, "IC", "Full531", "IC Full"),
        (2.0, "IC", "ExcludeS5", "IC Excl. S5"),
        (1.0, "dPL", "Full531", "dPL Full"),
        (0.0, "dPL", "ExcludeS5", "dPL Excl. S5"),
    ]
    dy = 0.13  # sub-row offset between the two contrasts
    for yy, paradigm, stratum, label in rows:
        for contrast, sgn in [(CN_PRIMARY, +1.0), (CN_TGD2, -1.0)]:
            st = CONTRAST_STYLE[contrast]
            slope, lo, hi = slope_ci(df_r, paradigm, contrast, stratum)
            y = yy + sgn * dy
            ax.errorbar(
                slope,
                y,
                xerr=[[slope - lo], [hi - slope]],
                fmt=st["mk"],
                color=st["color"],
                ecolor=st["color"],
                elinewidth=1.3,
                capsize=2.5,
                capthick=1.0,
                markersize=5.5,
                markerfacecolor=st["color"],
                markeredgewidth=1.0,
                zorder=3,
            )
            ax.text(
                hi + 0.015,
                y,
                f"{slope:+.3f}",
                ha="left",
                va="center",
                fontsize=5.5,
                color="#333333",
            )
    ax.axhline(1.5, color=COLOR_REF, linestyle=":", linewidth=0.8, alpha=0.7, zorder=1)
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[3] for r in rows], fontsize=7.5)
    ax.set_xlim(-0.08, 0.90)
    ax.set_ylim(-0.6, 3.7)
    ax.set_xlabel("Snow-gradient slope, β", labelpad=3)
    ax.grid(True, axis="x", linestyle=":", alpha=0.25)
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color=COLOR_BETWEEN,
            markersize=5.5,
            linestyle="none",
            label="Base–CN",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color=COLOR_TGD2,
            markersize=5.5,
            linestyle="none",
            label="Base–TGD2",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        frameon=True,
        framealpha=0.90,
        edgecolor="none",
        fontsize=6.4,
    )


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def _probe_layout(axes_by_name):
    """Programmatic layout validation: six near-square equal cells in a 2x3 grid."""
    plt.gcf().canvas.draw()
    boxes = {k: ax.get_position() for k, ax in axes_by_name.items()}
    widths = [b.width for b in boxes.values()]
    heights = [b.height for b in boxes.values()]
    assert max(widths) - min(widths) <= 0.02 * max(widths), f"unequal widths {widths}"
    assert max(heights) - min(heights) <= 0.02 * max(heights), (
        f"unequal heights {heights}"
    )
    for name, b in boxes.items():
        # physical aspect: bbox fractions are normalized to the figure, so
        # convert with the figure size in inches before judging square-ness.
        fw, fh = plt.gcf().get_size_inches()
        aspect = (b.height * fh) / (b.width * fw)
        assert 0.8 <= aspect <= 1.25, f"{name} aspect {aspect:.3f} not near-square"
    for col, below in [("a", "c1"), ("b", "c2"), ("d", "e")]:
        assert boxes[col].y0 >= boxes[below].y1 - 1e-6, f"{col} not above {below}"
    fw, fh = plt.gcf().get_size_inches()
    phys = (min(heights) * fh) / (min(widths) * fw)
    print(
        f"LAYOUT PROBE OK: 6 equal cells ({min(widths) * fw:.3f} x "
        f"{min(heights) * fh:.3f} in, aspect {phys:.3f}), 2x3 aligned."
    )


def build_figure(df_b, df_s, df_r) -> None:
    # 2x3 composite of near-square equal cells:
    #   Row 0: (a) | (b) | (d) merged
    #   Row 1: (c1) | (c2) | (e)
    fig = plt.figure(figsize=(10.8, 7.3))
    gs = gridspec.GridSpec(
        2, 3, hspace=0.35, wspace=0.28, left=0.08, right=0.97, top=0.96, bottom=0.06
    )

    ax_a = fig.add_subplot(gs[0, 0])
    apply_clean_spines(ax_a)
    ax_b = fig.add_subplot(gs[0, 1])
    apply_clean_spines(ax_b)
    ax_d = fig.add_subplot(gs[0, 2])
    apply_clean_spines(ax_d)  # merged (d)
    ax_c1 = fig.add_subplot(gs[1, 0])
    apply_clean_spines(ax_c1)
    ax_c2 = fig.add_subplot(gs[1, 1])
    apply_clean_spines(ax_c2)
    ax_e = fig.add_subplot(gs[1, 2])
    apply_clean_spines(ax_e)

    def frac_above(paradigm, contrast):
        sub = df_b[(df_b["paradigm"] == paradigm) & (df_b["contrast"] == contrast)]
        return float((sub["between_all"] > sub["within_pooled"]).mean())

    f_ic_cn, f_ic_tg = frac_above("IC", CN_PRIMARY), frac_above("IC", CN_TGD2)
    f_dp_cn, f_dp_tg = frac_above("dPL", CN_PRIMARY), frac_above("dPL", CN_TGD2)

    panel_ab_scatter(
        ax_a,
        df_b,
        "IC",
        f_ic_cn,
        f_ic_tg,
        "(a) Structural separation under IC",
        lim=0.75,
    )
    panel_ab_scatter(
        ax_b,
        df_b,
        "dPL",
        f_dp_cn,
        f_dp_tg,
        "(b) Structural separation under dPL",
        lim=0.65,
    )
    panel_c_hero(ax_c1, ax_c2, df_b, df_s, df_r)
    panel_d_merged(ax_d, df_s)
    panel_e_slopes(ax_e, df_r)

    # shared (a)/(b) legend in (a) upper-left
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color=COLOR_BETWEEN,
            markersize=5.5,
            linestyle="none",
            label="Base–CN",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color=COLOR_TGD2,
            markersize=5.5,
            linestyle="none",
            label="Base–TGD2",
        ),
    ]
    ax_a.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        framealpha=0.85,
        edgecolor="none",
        fontsize=7.0,
    )

    _probe_layout(
        {"a": ax_a, "b": ax_b, "d": ax_d, "c1": ax_c1, "c2": ax_c2, "e": ax_e}
    )

    plt.savefig(PLOTS_FIG_DIR / f"{OUT_NAME}.png", dpi=600)
    print("saved:", PLOTS_FIG_DIR / f"{OUT_NAME}.png")
    plt.close()


def main() -> None:
    setup_publication_style()
    df_b, df_s, df_r = load_data()
    # sanity: descriptive prevalences reproduced from the frozen basin CSV
    for paradigm, contrast, exp in [
        ("IC", "Base-CN", 0.6309),
        ("IC", "Base-TGD2", 0.6064),
        ("dPL", "Base-CN", 0.8380),
        ("dPL", "Base-TGD2", 0.8230),
    ]:
        sub = df_b[(df_b["paradigm"] == paradigm) & (df_b["contrast"] == contrast)]
        frac = float((sub["between_all"] > sub["within_pooled"]).mean())
        assert abs(frac - exp) < 0.005, f"{paradigm} {contrast} {frac:.4f} != {exp:.4f}"
    build_figure(df_b, df_s, df_r)
    print("Final Figure 3 (5-panel, compact) generated successfully.")


if __name__ == "__main__":
    main()
