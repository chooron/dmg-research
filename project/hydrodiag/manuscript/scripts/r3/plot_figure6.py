#!/usr/bin/env python3
"""Figure 6: Internal parameter and state recovery under structural omission and generic control.

Hierarchical 4-panel layout matching Results 3.3 revised evidence architecture:
  Row 1 (Parameter Evidence):
    (a) Parameter distance to generating truth (Base, TGD, CN across S1-S5)
    (b) Parameter excess beyond CN refit (Base, TGD across S1-S5; CN baseline = 0)
    Both panels share S1-S5 sequential blue background bands.
  Row 2 (State Excess & Association Audit):
    (c) Shared-state and flux excess (~60% width, Scatter + Boxplot style)
        Wt (total), Wu (upper), Wl (lower) states and Qi, Qg fluxes; Base & TGD (IC vs dPL).
    (d) Recovery–excess-error association (~40% width, Horizontal Dumbbell)
        Raw vs Partial Spearman rho on frac_snow across 8 parameter and state associations.

Visual grammar (matching Figure 1 Okabe-Ito frozen standards):
  - Base refit:       #D55E00 (Okabe-Ito vermillion / deep orange)
  - TGD refit:        #009E73 (Okabe-Ito bluish green / teal)
  - CN refit:         #0072B2 (Okabe-Ito blue)
  - Truth / Neutral:  #303438
  - S1-S5 Background: Light sequential blues
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# Path setup
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
MANUSCRIPT = PROJECT / "manuscript"
RESULTS_R3 = MANUSCRIPT / "results" / "R3"
FIG_DIR = MANUSCRIPT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Add shared styles
sys.path.insert(0, str(MANUSCRIPT / "scripts" / "shared"))
from r1_plot_style import (
    COLOR_BASE,
    COLOR_TGD,
    COLOR_CN,
    COLOR_DARK_NEUTRAL,
    COLOR_LIGHT_REF,
    COLOR_ZERO_LINE,
    apply_clean_spines,
    setup_publication_style,
)

OUT_NAME = "Figure6_R3_final"

# Sequential blue background palette for snow activity strata (S1 -> S5)
BLUE_BANDS = ["#F4F8FA", "#EBF3F8", "#DFEDF5", "#D2E5F0", "#C4DDEB"]
STRATA_LABELS = ["S1\n(low)", "S2", "S3", "S4", "S5\n(high)"]


def load_data():
    summary = json.loads((RESULTS_R3 / "figure6_summary.json").read_text())
    tidy = pd.read_csv(RESULTS_R3 / "figure6_basin_table.csv")
    seedmed = pd.read_csv(RESULTS_R3 / "figure6_basin_seedmedian.csv")
    for df in (tidy, seedmed):
        df["basin_id"] = df["basin_id"].astype(str).str.zfill(8)
        df["seed"] = df["seed"].fillna("").astype(str)
    return tidy, seedmed, summary


def build_figure(tidy, seedmed, summary, out_dir: Path | None = None) -> Path:
    setup_publication_style()

    target_dir = out_dir or FIG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    fig_w_in = 11.5
    fig_h_in = 8.6
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # Outer layout: 2 rows with reduced vertical spacing (hspace=0.25)
    gs_main = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.15],
        hspace=0.25,
        top=0.925,
        bottom=0.08,
        left=0.075,
        right=0.985,
    )

    # Row 1: Panels (a) and (b)
    gs_top = gs_main[0].subgridspec(1, 2, width_ratios=[1.08, 0.92], wspace=0.18)
    ax_a1 = fig.add_subplot(gs_top[0, 0])
    ax_a2 = fig.add_subplot(gs_top[0, 1])

    # Row 2: Panels (c) and (d)
    gs_bot = gs_main[1].subgridspec(1, 2, width_ratios=[1.50, 1.0], wspace=0.22)
    ax_b = fig.add_subplot(gs_bot[0, 0])
    ax_c = fig.add_subplot(gs_bot[0, 1])

    xs = np.arange(5)

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL (a) & PANEL (b): Parameter Evidence
    # ═════════════════════════════════════════════════════════════════════════
    apply_clean_spines(ax_a1)
    apply_clean_spines(ax_a2)

    # Distinct panel titles with (a) and (b)
    ax_a1.set_title("(a) Parameter distance to generating truth", loc="left", fontsize=11.2, fontweight="bold", pad=6)
    ax_a2.set_title("(b) Parameter excess beyond CN refit", loc="left", fontsize=11.2, fontweight="bold", pad=6)

    # Background sequential blue strata bands
    for ax in [ax_a1, ax_a2]:
        for i in range(5):
            ax.axvspan(i - 0.5, i + 0.5, color=BLUE_BANDS[i], alpha=0.55, zorder=0)
        for i in range(4):
            ax.axvline(i + 0.5, color="#CBD5E1", lw=0.6, ls=":", zorder=1)

    # ── Panel (a): Parameter Distance to Generating Truth ──
    pa = summary["panel_a_param_distance"]
    for reg, ls, marker in [("IC", "-", "o"), ("dPL", "--", "^")]:
        sub_strata = pa[reg]["strata"]
        b_vals = [sub_strata[st]["Base"]["median"] for st in ("S1", "S2", "S3", "S4", "S5")]
        t_vals = [sub_strata[st]["TGD"]["median"] for st in ("S1", "S2", "S3", "S4", "S5")]
        c_vals = [sub_strata[st]["CN"]["median"] for st in ("S1", "S2", "S3", "S4", "S5")]

        b_ci = [sub_strata[st]["Base"]["ci"] for st in ("S1", "S2", "S3", "S4", "S5")]
        t_ci = [sub_strata[st]["TGD"]["ci"] for st in ("S1", "S2", "S3", "S4", "S5")]
        c_ci = [sub_strata[st]["CN"]["ci"] for st in ("S1", "S2", "S3", "S4", "S5")]

        off = -0.06 if reg == "IC" else +0.06

        # Base
        yerr_b = [[b_vals[i] - b_ci[i][0] for i in range(5)], [b_ci[i][1] - b_vals[i] for i in range(5)]]
        if reg == "IC":
            ax_a1.errorbar(xs + off, b_vals, yerr=yerr_b, fmt=marker, color=COLOR_BASE, linestyle=ls, linewidth=1.5,
                           markersize=5.2, markerfacecolor=COLOR_BASE, markeredgecolor="white", markeredgewidth=1.0, capsize=2.5, elinewidth=1.2, zorder=3)
        else:
            ax_a1.errorbar(xs + off, b_vals, yerr=yerr_b, fmt=marker, color=COLOR_BASE, linestyle=ls, linewidth=1.5,
                           markersize=5.2, markerfacecolor="white", markeredgecolor=COLOR_BASE, markeredgewidth=1.3, capsize=2.5, elinewidth=1.2, zorder=3)

        # TGD
        yerr_t = [[t_vals[i] - t_ci[i][0] for i in range(5)], [t_ci[i][1] - t_vals[i] for i in range(5)]]
        if reg == "IC":
            ax_a1.errorbar(xs + off, t_vals, yerr=yerr_t, fmt=marker, color=COLOR_TGD, linestyle=ls, linewidth=1.5,
                           markersize=5.2, markerfacecolor=COLOR_TGD, markeredgecolor="white", markeredgewidth=1.0, capsize=2.5, elinewidth=1.2, zorder=4)
        else:
            ax_a1.errorbar(xs + off, t_vals, yerr=yerr_t, fmt=marker, color=COLOR_TGD, linestyle=ls, linewidth=1.5,
                           markersize=5.2, markerfacecolor="white", markeredgecolor=COLOR_TGD, markeredgewidth=1.3, capsize=2.5, elinewidth=1.2, zorder=4)

        # CN
        yerr_c = [[c_vals[i] - c_ci[i][0] for i in range(5)], [c_ci[i][1] - c_vals[i] for i in range(5)]]
        if reg == "IC":
            ax_a1.errorbar(xs + off, c_vals, yerr=yerr_c, fmt=marker, color=COLOR_CN, linestyle=ls, linewidth=1.5,
                           markersize=5.2, markerfacecolor=COLOR_CN, markeredgecolor="white", markeredgewidth=1.0, capsize=2.5, elinewidth=1.2, zorder=5)
        else:
            ax_a1.errorbar(xs + off, c_vals, yerr=yerr_c, fmt=marker, color=COLOR_CN, linestyle=ls, linewidth=1.5,
                           markersize=5.2, markerfacecolor="white", markeredgecolor=COLOR_CN, markeredgewidth=1.3, capsize=2.5, elinewidth=1.2, zorder=5)

    ax_a1.set_xlim(-0.5, 4.5)
    ax_a1.set_xticks(xs)
    ax_a1.set_xticklabels(STRATA_LABELS, fontsize=9.2)
    ax_a1.set_ylim(-0.02, 0.48)
    ax_a1.tick_params(axis="y", labelsize=9.2)
    ax_a1.set_xlabel("Snow activity stratum ($f_{\\mathrm{snow}}$)", labelpad=2, fontsize=10.2)
    ax_a1.set_ylabel("Truth distance $E^{\\mathrm{param}} = \\mathrm{med}_p |e_p|$", labelpad=2, fontsize=10.2)
    ax_a1.text(0.04, 0.92, "15 shared parameters", transform=ax_a1.transAxes, va="top", ha="left", fontsize=8.6, color="#555555", style="italic", zorder=6)
    ax_a1.grid(True, axis="y", linestyle=":", alpha=0.35, color=COLOR_LIGHT_REF)

    # ── Panel (b): Parameter Excess Beyond CN Refit ──
    ax_a2.axhline(0.0, color=COLOR_ZERO_LINE, linestyle="--", linewidth=0.8, zorder=1)
    ax_a2.text(0.98, 0.02, "CN-refit baseline = 0", transform=ax_a2.get_yaxis_transform(), va="bottom", ha="right", fontsize=8.6, color="#555555", style="italic", zorder=6)

    pb = summary["panel_b_param_excess"]
    for reg, ls, marker in [("IC", "-", "o"), ("dPL", "--", "^")]:
        sub_strata = pb[reg]["strata"]
        b_vals = [sub_strata[st]["Base"]["median"] for st in ("S1", "S2", "S3", "S4", "S5")]
        t_vals = [sub_strata[st]["TGD"]["median"] for st in ("S1", "S2", "S3", "S4", "S5")]

        b_ci = [sub_strata[st]["Base"]["ci"] for st in ("S1", "S2", "S3", "S4", "S5")]
        t_ci = [sub_strata[st]["TGD"]["ci"] for st in ("S1", "S2", "S3", "S4", "S5")]

        off = -0.06 if reg == "IC" else +0.06

        # Base
        yerr_b = [[b_vals[i] - b_ci[i][0] for i in range(5)], [b_ci[i][1] - b_vals[i] for i in range(5)]]
        if reg == "IC":
            ax_a2.errorbar(xs + off, b_vals, yerr=yerr_b, fmt=marker, color=COLOR_BASE, linestyle=ls, linewidth=1.5,
                           markersize=5.2, markerfacecolor=COLOR_BASE, markeredgecolor="white", markeredgewidth=1.0, capsize=2.5, elinewidth=1.2, zorder=3)
        else:
            ax_a2.errorbar(xs + off, b_vals, yerr=yerr_b, fmt=marker, color=COLOR_BASE, linestyle=ls, linewidth=1.5,
                           markersize=5.2, markerfacecolor="white", markeredgecolor=COLOR_BASE, markeredgewidth=1.3, capsize=2.5, elinewidth=1.2, zorder=3)

        # TGD
        yerr_t = [[t_vals[i] - t_ci[i][0] for i in range(5)], [t_ci[i][1] - t_vals[i] for i in range(5)]]
        if reg == "IC":
            ax_a2.errorbar(xs + off, t_vals, yerr=yerr_t, fmt=marker, color=COLOR_TGD, linestyle=ls, linewidth=1.5,
                           markersize=5.2, markerfacecolor=COLOR_TGD, markeredgecolor="white", markeredgewidth=1.0, capsize=2.5, elinewidth=1.2, zorder=4)
        else:
            ax_a2.errorbar(xs + off, t_vals, yerr=yerr_t, fmt=marker, color=COLOR_TGD, linestyle=ls, linewidth=1.5,
                           markersize=5.2, markerfacecolor="white", markeredgecolor=COLOR_TGD, markeredgewidth=1.3, capsize=2.5, elinewidth=1.2, zorder=4)

    ax_a2.set_xlim(-0.5, 4.5)
    ax_a2.set_xticks(xs)
    ax_a2.set_xticklabels(STRATA_LABELS, fontsize=9.2)
    ax_a2.set_ylim(-0.06, 0.38)
    ax_a2.tick_params(axis="y", labelsize=9.2)
    ax_a2.set_xlabel("Snow activity stratum ($f_{\\mathrm{snow}}$)", labelpad=2, fontsize=10.2)
    ax_a2.set_ylabel("Excess $E_M^{\\mathrm{param}} - E_{\\mathrm{CN}}^{\\mathrm{param}}$", labelpad=2, fontsize=10.2)
    ax_a2.grid(True, axis="y", linestyle=":", alpha=0.35, color=COLOR_LIGHT_REF)

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL (c): Shared-state and flux excess (Scatter + Boxplot)
    # ═════════════════════════════════════════════════════════════════════════
    apply_clean_spines(ax_b)
    ax_b.set_title("(c) Shared-state and flux excess", loc="left", fontsize=11.2, fontweight="bold", pad=6)
    ax_b.axhline(0.0, color=COLOR_ZERO_LINE, linestyle="--", linewidth=0.8, zorder=1)
    ax_b.text(0.98, 0.02, "CN-refit baseline = 0", transform=ax_b.get_yaxis_transform(), va="bottom", ha="right", fontsize=8.6, color="#555555", style="italic", zorder=6)

    vars_order = [
        ("delta_E_wt_base", "delta_E_wt_tgd", "$W_t$\n(total)"),
        ("delta_E_wu_base", "delta_E_wu_tgd", "$W_u$\n(upper)"),
        ("delta_E_wl_base", "delta_E_wl_tgd", "$W_l$\n(lower)"),
        ("delta_E_qi_base", "delta_E_qi_tgd", "$Q_i$\n(flux)"),
        ("delta_E_qg_base", "delta_E_qg_tgd", "$Q_g$\n(flux)"),
    ]

    x_vars = np.arange(len(vars_order))
    sub_ic = seedmed[seedmed["paradigm"] == "IC"]
    sub_dpl = seedmed[seedmed["paradigm"] == "dPL"]

    offsets_cfg = [
        ("Base", "IC", sub_ic, -0.27, COLOR_BASE, True),
        ("Base", "dPL", sub_dpl, -0.09, COLOR_BASE, False),
        ("TGD", "IC", sub_ic, +0.09, COLOR_TGD, True),
        ("TGD", "dPL", sub_dpl, +0.27, COLOR_TGD, False),
    ]

    for vi, (col_b, col_t, vlabel) in enumerate(vars_order):
        if vi > 0:
            ax_b.axvline(vi - 0.5, color="#E2E8F0", lw=0.6, ls=":", zorder=1)

        for struct, reg, df_sub, dx, col, is_ic in offsets_cfg:
            col_name = col_b if struct == "Base" else col_t
            vals = df_sub[col_name].dropna().values

            # Jittered background scatter
            np.random.seed(42 + vi * 10 + (0 if is_ic else 1))
            jit = np.random.uniform(-0.035, 0.035, len(vals))
            ax_b.scatter(vi + dx + jit, vals, s=6, color=col, alpha=0.15, edgecolors="none", rasterized=True, zorder=2)

            # Boxplot showing quartile distributions
            ax_b.boxplot(
                vals,
                positions=[vi + dx],
                widths=0.13,
                patch_artist=True,
                showfliers=False,
                whis=[5, 95],
                zorder=4,
                boxprops=dict(
                    facecolor=col if is_ic else "white",
                    edgecolor=col,
                    linewidth=1.2,
                    linestyle="-" if is_ic else "--",
                    alpha=0.45 if is_ic else 0.95,
                ),
                whiskerprops=dict(color=col, linewidth=1.1, linestyle="-" if is_ic else "--"),
                capprops=dict(color=col, linewidth=1.1),
                medianprops=dict(color=COLOR_DARK_NEUTRAL if not is_ic else "#1E293B", linewidth=1.6),
            )

    ax_b.set_xlim(-0.55, len(vars_order) - 0.45)
    ax_b.set_xticks(x_vars)
    ax_b.set_xticklabels([v[2] for v in vars_order], fontsize=9.2)
    ax_b.set_ylim(-0.25, 1.85)
    ax_b.tick_params(axis="y", labelsize=9.2)
    ax_b.set_ylabel("NRMSE excess vs. CN refit ($\\Delta E^{\\mathrm{state}}$)", labelpad=2, fontsize=10.2)
    ax_b.text(0.03, 0.92, "$W_t = W_u + W_l + W_d$ headline storage\nBoxes = [q25, q75], whiskers = [5%, 95%]", transform=ax_b.transAxes, va="top", ha="left", fontsize=8.6, color="#555555", style="italic", zorder=6)
    ax_b.grid(True, axis="y", linestyle=":", alpha=0.35, color=COLOR_LIGHT_REF)

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL (d): Recovery–excess-error association (Horizontal Dumbbell)
    # ═════════════════════════════════════════════════════════════════════════
    apply_clean_spines(ax_c)
    ax_c.set_title("(d) Recovery–excess-error association", loc="left", fontsize=11.2, fontweight="bold", pad=6)
    ax_c.axvline(0.0, color=COLOR_ZERO_LINE, linestyle="--", linewidth=0.8, zorder=1)

    pd_assoc = summary["panel_d_associations"]
    rows = [
        ("IC",  "G_Base <-> E_param_excess_Base", 7.0, "Base param (IC)",  COLOR_BASE),
        ("dPL", "G_Base <-> E_param_excess_Base", 6.0, "Base param (dPL)", COLOR_BASE),
        ("IC",  "G_TGD <-> E_param_excess_TGD",   5.0, "TGD param (IC)",   COLOR_TGD),
        ("dPL", "G_TGD <-> E_param_excess_TGD",   4.0, "TGD param (dPL)",  COLOR_TGD),
        ("IC",  "G_Base <-> Delta E_state(Wt)",   3.0, "Base $W_t$ (IC)",  COLOR_BASE),
        ("dPL", "G_Base <-> Delta E_state(Wt)",   2.0, "Base $W_t$ (dPL)", COLOR_BASE),
        ("IC",  "G_TGD <-> Delta E_state(Wt)",    1.0, "TGD $W_t$ (IC)",   COLOR_TGD),
        ("dPL", "G_TGD <-> Delta E_state(Wt)",    0.0, "TGD $W_t$ (dPL)",  COLOR_TGD),
    ]

    for reg, pname, ypos, lbl, col in rows:
        entry = pd_assoc[reg][pname]
        raw_r = entry["raw_spearman"]
        part_r = entry["partial_spearman"]

        # Dumbbell connecting line
        ax_c.plot([raw_r, part_r], [ypos, ypos], color="#888888", linestyle="-", linewidth=1.2, zorder=2)
        # Raw Spearman
        ax_c.scatter(raw_r, ypos, color=col, s=36, zorder=4, edgecolors="#333333", linewidths=0.7)
        # Partial Spearman
        ax_c.scatter(part_r, ypos, color="white", s=38, zorder=5, edgecolors=col, linewidths=1.4, marker="D")

        # Text with zorder=7 so it sits on top of all lines
        ax_c.text(0.98, ypos, f"{raw_r:+.2f} → {part_r:+.2f}", transform=ax_c.get_yaxis_transform(),
                  va="center", ha="right", fontsize=8.6, family="monospace", color=COLOR_DARK_NEUTRAL, zorder=7)

    ax_c.set_ylim(-0.7, 7.7)
    ax_c.set_yticks([r[2] for r in rows])
    ax_c.set_yticklabels([r[3] for r in rows], fontsize=8.8)
    ax_c.set_xlim(-0.35, 1.05)
    ax_c.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
    ax_c.set_xticklabels(["-0.2", "0.0", "+0.2", "+0.4", "+0.6", "+0.8"], fontsize=9.2)
    ax_c.set_xlabel("Spearman $\\rho$: Raw (●) vs. Partial on $f_{\\mathrm{snow}}$ (◇)", labelpad=2, fontsize=10.2)
    ax_c.grid(True, axis="x", linestyle=":", alpha=0.35, color=COLOR_LIGHT_REF)

    # ═════════════════════════════════════════════════════════════════════════
    # GLOBAL LEGENDS (Positioned closely at top of figure)
    # ═════════════════════════════════════════════════════════════════════════
    struct_handles = [
        Line2D([0], [0], color=COLOR_BASE, lw=1.6, marker="o", markersize=5.0, markerfacecolor=COLOR_BASE, markeredgecolor="white", label="Base refit"),
        Line2D([0], [0], color=COLOR_TGD, lw=1.6, marker="^", markersize=5.0, markerfacecolor=COLOR_TGD, markeredgecolor="white", label="TGD refit"),
        Line2D([0], [0], color=COLOR_CN, lw=1.6, marker="s", markersize=5.0, markerfacecolor=COLOR_CN, markeredgecolor="white", label="CN refit"),
    ]
    regime_handles = [
        Line2D([0], [0], color=COLOR_DARK_NEUTRAL, ls="-", lw=1.4, marker="o", markersize=4.5, markerfacecolor=COLOR_DARK_NEUTRAL, markeredgecolor="white", label="IC (filled)"),
        Line2D([0], [0], color=COLOR_DARK_NEUTRAL, ls="--", lw=1.4, marker="^", markersize=4.5, markerfacecolor="white", markeredgecolor=COLOR_DARK_NEUTRAL, markeredgewidth=1.2, label="dPL (hollow)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555", markeredgecolor="#222222", markersize=4.5, label="Raw $\\rho$"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="white", markeredgecolor="#555555", markeredgewidth=1.2, markersize=4.5, label="Partial $\\rho$ (adj. $f_{\\mathrm{snow}}$)"),
    ]

    fig.legend(handles=struct_handles, loc="upper left", bbox_to_anchor=(0.075, 0.985), ncol=3, frameon=False, fontsize=8.5, handlelength=1.5, columnspacing=1.0)
    fig.legend(handles=regime_handles, loc="upper right", bbox_to_anchor=(0.985, 0.985), ncol=4, frameon=False, fontsize=8.5, handlelength=1.5, columnspacing=0.9)

    out_path = target_dir / f"{OUT_NAME}.png"
    plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved Figure 6 -> {out_path} ({file_size_mb:.2f} MB)", flush=True)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Render R3 Figure 6.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for generated figure.")
    args = parser.parse_args()
    tidy, seedmed, summary = load_data()
    build_figure(tidy, seedmed, summary, args.out_dir)


if __name__ == "__main__":
    main()
