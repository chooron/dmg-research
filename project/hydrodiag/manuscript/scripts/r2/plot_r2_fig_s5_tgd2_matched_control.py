#!/usr/bin/env python3
"""Supplementary Figure S5 (Fig. S5): parameter-level matched-control context for TGD2.

Compact dot-whisker / forest figure showing snow-gradient estimates of the canonical
paired shifts for the 15 shared XAJ parameters, for the two structural contrasts
Base-CN (primary estimand, frozen values from `r2_snow_gradients_summary.csv`) and
Base-TGD2 (matched control, `r2_snow_gradients_base_tgd2_summary.csv`), under the
IC and dPL parameter-constraint regimes.

The figure is Supplement-only context for Figure 4: it reports the same
parameter-level quantity (OLS slope beta of delta_p = z_Base - z_CN/TGD2 against
basin snow fraction, with 95 % basin-level bootstrap CI) without modifying or
recomputing any main-text result. It deliberately contains no third contrast
(TGD2-CN), no subset rows, and no interpretation annotation.

Visual grammar matches F3/F4 (r1_plot_style):
  * Base-CN  -> deep blue #0077BB, square marker, solid
  * Base-TGD2-> teal #009988, triangle marker, dashed
  * vertical zero reference line; regime carried by facet position.
Output: manuscript/supplement/figures/Fig_S5_R2_tgd2_parameter_matched_control.png
(+ vector PDF), 600 DPI, manuscript convention.
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

HERE = Path(__file__).resolve().parent
PLOTS_DIR = HERE.parents[3] / "plots"
if str(PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PLOTS_DIR))

from manuscript.scripts.shared.r1_plot_style import (  # noqa: E402
    MODEL_COLORS,
    apply_clean_spines,
    setup_publication_style,
)

MANUSCRIPT = HERE.parent
RESULTS_R2 = MANUSCRIPT / "results" / "R2"
SUPP_FIG_DIR = MANUSCRIPT / "supplement" / "figures"

OUT_NAME = "Fig_S5_R2_tgd2_parameter_matched_control"

# Canonical parameter order and labels, identical to Figure 4 (panel a).
PARAM_ORDER = [
    "xaj_k",
    "xaj_b",
    "xaj_im",
    "xaj_um",
    "xaj_lm",
    "xaj_dm",
    "xaj_c",
    "xaj_sm",
    "xaj_ex",
    "xaj_ki",
    "xaj_kg",
    "xaj_ci",
    "xaj_cg",
    "xaj_a",
    "xaj_theta",
]
DISPLAY = {
    "xaj_k": "k",
    "xaj_b": "b",
    "xaj_im": "im",
    "xaj_um": "um",
    "xaj_lm": "lm",
    "xaj_dm": "dm",
    "xaj_c": "c",
    "xaj_sm": "sm",
    "xaj_ex": "ex",
    "xaj_ki": "ki",
    "xaj_kg": "kg",
    "xaj_ci": "ci",
    "xaj_cg": "cg",
    "xaj_a": "a",
    "xaj_theta": "\u03b8",
}

CN_PRIMARY = "Base-CN"
CN_TGD2 = "Base-TGD2"

COLOR_BETWEEN = MODEL_COLORS["CN"]  # #0077BB  Base-CN
COLOR_TGD2 = MODEL_COLORS["TGD"]  # #009988  Base-TGD2
COLOR_REF = "#999999"
DY = 0.22  # sub-row offset between the two contrasts


def load_gradients() -> dict:
    df_cn = pd.read_csv(RESULTS_R2 / "r2_snow_gradients_summary.csv")
    df_tg = pd.read_csv(RESULTS_R2 / "r2_snow_gradients_base_tgd2_summary.csv")
    return {CN_PRIMARY: df_cn, CN_TGD2: df_tg}


def gradient_map(df, paradigm, parameter):
    row = df[(df["paradigm"] == paradigm) & (df["parameter"] == parameter)]
    assert len(row) == 1, f"missing {paradigm} {parameter}"
    r = row.iloc[0]
    return float(r["beta"]), float(r["ci95_low"]), float(r["ci95_high"])


def draw_facet(ax, gradients, paradigm):
    """One dot-whisker facet: 15 parameters, Base-CN vs Base-TGD2 slopes + 95% CI."""
    ax.axvline(0.0, color=COLOR_REF, linestyle="--", linewidth=0.85, zorder=1)
    y_pos = np.arange(len(PARAM_ORDER))
    for i, p in enumerate(PARAM_ORDER):
        y = y_pos[i]
        for contrast, sgn, style in [
            (CN_PRIMARY, +1.0, {"color": COLOR_BETWEEN, "mk": "s", "ls": "-"}),
            (CN_TGD2, -1.0, {"color": COLOR_TGD2, "mk": "^", "ls": (0, (4.0, 2.0))}),
        ]:
            beta, lo, hi = gradient_map(gradients[contrast], paradigm, p)
            yy = y + sgn * DY
            ax.errorbar(
                beta,
                yy,
                xerr=[[beta - lo], [hi - beta]],
                fmt=style["mk"],
                color=style["color"],
                ecolor=style["color"],
                elinewidth=1.1,
                capsize=2.2,
                capthick=0.9,
                markersize=4.6,
                markerfacecolor=style["color"],
                markeredgecolor="none",
                linestyle="none",
                zorder=3,
            )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([DISPLAY[p] for p in PARAM_ORDER], fontsize=7.5)
    ax.set_ylim(-0.75, len(PARAM_ORDER) - 0.25)
    ax.set_xlabel("Snow gradient of paired shift, \u03b2", labelpad=3)
    ax.grid(True, axis="x", linestyle=":", alpha=0.18)


def build_figure(gradients) -> None:
    setup_publication_style()
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = gridspec.GridSpec(
        2,
        1,
        height_ratios=[1.0, 1.0],
        hspace=0.32,
        left=0.10,
        right=0.97,
        top=0.94,
        bottom=0.10,
    )
    ax_ic = fig.add_subplot(gs[0, 0])
    ax_dpl = fig.add_subplot(gs[1, 0])
    apply_clean_spines(ax_ic)
    apply_clean_spines(ax_dpl)

    # Shared x-axis limits across the two facets (union of both contrasts).
    lo_all = min(
        gradient_map(gradients[c], par_, p_)[1]
        for c in (CN_PRIMARY, CN_TGD2)
        for par_ in ("IC", "dPL")
        for p_ in PARAM_ORDER
    )
    hi_all = max(
        gradient_map(gradients[c], par_, p_)[2]
        for c in (CN_PRIMARY, CN_TGD2)
        for par_ in ("IC", "dPL")
        for p_ in PARAM_ORDER
    )
    pad = max(0.05, (hi_all - lo_all) * 0.06)
    xlim = (lo_all - pad, hi_all + pad)

    draw_facet(ax_ic, gradients, "IC")
    draw_facet(ax_dpl, gradients, "dPL")
    for ax in (ax_ic, ax_dpl):
        ax.set_xlim(xlim)
    ax_ic.set_xticklabels([])
    ax_ic.set_xlabel("")
    ax_ic.set_title("IC regime", loc="left", fontsize=8.5, fontweight="bold", pad=5)
    ax_dpl.set_title("dPL regime", loc="left", fontsize=8.5, fontweight="bold", pad=5)

    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color=COLOR_BETWEEN,
            markerfacecolor=COLOR_BETWEEN,
            markersize=5.2,
            linestyle="none",
            label="Base\u2013CN",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color=COLOR_TGD2,
            markerfacecolor=COLOR_TGD2,
            markersize=5.2,
            linestyle="none",
            label="Base\u2013TGD2",
        ),
    ]
    ax_ic.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        framealpha=0.90,
        edgecolor="none",
        fontsize=7.0,
    )

    SUPP_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SUPP_FIG_DIR / f"{OUT_NAME}.png", dpi=600)
    fig.savefig(SUPP_FIG_DIR / f"{OUT_NAME}.pdf", bbox_inches="tight", pad_inches=0.03)
    print("saved:", SUPP_FIG_DIR / f"{OUT_NAME}.png")
    print("saved:", SUPP_FIG_DIR / f"{OUT_NAME}.pdf")
    plt.close(fig)


def main() -> None:
    gradients = load_gradients()
    # sanity: 15 params x 2 paradigms x 2 contrasts, finite values
    for c, df in gradients.items():
        assert len(df) == 30
        for par_ in ("IC", "dPL"):
            for p_ in PARAM_ORDER:
                b, lo, hi = gradient_map(df, par_, p_)
                assert np.isfinite(b) and lo <= b <= hi, f"{c} {par_} {p_}"
    build_figure(gradients)
    print("Supplementary Figure S5 generated successfully.")


if __name__ == "__main__":
    main()
