#!/usr/bin/env python3
"""Final R3 Figure 6 (F6): 6-panel composite on TGD2 matched-surrogate
mitigation and residual CN explicit-process advantage.

Scientific line (frozen):
    Generic temperature-conditioned storage/memory (TGD2) provides
    substantial (~50 %) but incomplete output mitigation, and partially
    relieves internal parameter and state distortion; the residual advantage
    of the explicit snow-process structure (CN) strengthens with snow
    activity and is heavily concentrated during snow-active periods.

Layout (three-level asymmetric composite)
-----------------------------------------------------------------
  Row 1 (output structural ladder & mitigation):
      (a) Base -> TGD2 -> CN structural ladder (approx. 70 % width, IC & dPL facets)
      (b) TGD2 gap-mitigation fraction F_tgd2 (approx. 30 % width)
  Row 2 (deterministic process trajectories, matched pair):
      (c) Seasonal liquid-water input timing (approx. 50 % width)
      (d) Seasonal shared total tension storage response (approx. 50 % width)
  Row 3 (process-specific adjudication, hero row, approx. 44 % height):
      (e) Residual CN advantage vs. frac_snow (approx. 68 % width, IC+dPL on single axis)
      (f) Process-conditioned residual: snow-active vs. non-snow (approx. 32 % width)

The process panels use compact water-year-month summaries from a deterministic
recorded-forward replay of frozen R3 fits; they do not retrain or recalibrate.

Visual grammar (inherited from r1_plot_style.py / F1-F5):
  * Base fitted  -> Base orange #EE7733
  * TGD2         -> TGD teal   #009988 (#007766 for darker IC tone)
  * CN           -> CN blue    #0077BB (#005588 for darker IC tone)
  * Base no-refit-> neutral grey #A0A0A0 (raw knockout reference)
  * Regime encoding in (e):
      - IC:  darker blue #005588, circle marker 'o', solid line '-'
      - dPL: CN blue    #0077BB, triangle marker '^', dashed line '--'
  * PNG only, 600 DPI, saved to manuscript/figures/ and manuscript/plots/figures/

Statistics: all headline numbers are read from the frozen post-hoc summaries
via manuscript/results/R3/figure6_summary.json (prepared by
prepare_figure6_data.py, which asserts equality with the frozen values).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_plot_style import (  # noqa: E402
    MODEL_COLORS,
    setup_publication_style,
    apply_clean_spines,
)

PROJECT = Path(__file__).resolve().parents[2]
MANUSCRIPT = PROJECT / "manuscript"
RESULTS_R3 = MANUSCRIPT / "results" / "R3"
FIG_DIR = MANUSCRIPT / "figures"
SEASONAL_DIR = RESULTS_R3 / "fig6_seasonal"
PLOTS_FIG_DIR = MANUSCRIPT / "plots" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_NAME = "Figure6_R3_final"

# ---------------------------------------------------------------------------
# Frozen visual grammar (r1_plot_style.py / F1-F5 system)
# ---------------------------------------------------------------------------
C_BASE = MODEL_COLORS["Base"]       # #EE7733  omitted-process baseline (fitted)
C_TGD = MODEL_COLORS["TGD"]         # #009988  generic temperature-memory control
C_TGD_IC = "#007766"               # darker teal tone for IC
C_CN = MODEL_COLORS["CN"]           # #0077BB  correct snow-process structure
C_CN_IC = "#005588"                # darker blue tone for IC
C_NOREFIT = "#A0A0A0"               # neutral grey: Base without recalibration
C_REF = "#999999"                   # reference lines
C_TEXT = "#333333"                  # annotation text
TEXT_BG = dict(boxstyle="round,pad=0.2", facecolor="white",
               edgecolor="none", alpha=0.88)

# ---------------------------------------------------------------------------
# Data loading (frozen Figure 6 package, read-only)
# ---------------------------------------------------------------------------
def load_data():
    summary = json.loads((RESULTS_R3 / "figure6_summary.json").read_text())
    tidy = pd.read_csv(RESULTS_R3 / "figure6_basin_table.csv")
    seedmed = pd.read_csv(RESULTS_R3 / "figure6_basin_seedmedian.csv")
    for df in (tidy, seedmed):
        df["basin_id"] = df["basin_id"].astype(str).str.zfill(8)
        df["seed"] = df["seed"].fillna("").astype(str)
    return tidy, seedmed, summary


def ecdf(vals: np.ndarray):
    v = np.sort(np.asarray(vals, dtype=np.float64))
    v = v[np.isfinite(v)]
    return v, np.arange(1, len(v) + 1) / len(v)


# ---------------------------------------------------------------------------
# Panel (a): Base -> TGD2 -> CN structural ladder (test) — wide overview
# ---------------------------------------------------------------------------
def _facet_ladder_ecdf(ax, seedmed, reg):
    te = seedmed[seedmed["paradigm"] == reg]
    series = [
        ("Base no-refit", te["kge_base_no_refit"].to_numpy(), C_NOREFIT, (0, (1.5, 2.0)), 1.2),
        ("Base fitted", te["kge_base"].to_numpy(), C_BASE, (0, (4.0, 2.0)), 1.5),
        ("TGD2", te["kge_tgd2"].to_numpy(), C_TGD, "-", 1.6),
        ("CN", te["kge_cn"].to_numpy(), C_CN, "-", 1.7),
    ]
    for label, vals, color, ls, lw in series:
        x, y = ecdf(vals)
        ax.step(x, y, where="post", color=color, linestyle=ls, linewidth=lw,
                zorder=4)
        med = float(np.median(vals))
        ax.axvline(med, color=color, linestyle=":", linewidth=0.7, alpha=0.6,
                   zorder=2)
    ax.set_xlim(-0.30, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([-0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, axis="y", linestyle=":", alpha=0.25)
    ax.text(0.02, 0.94, f"{reg} regime", transform=ax.transAxes, ha="left",
            va="top", fontsize=9.0, fontweight="bold", color=C_TEXT)
    ax.text(0.98, 0.97, "median 0.899 | 0.934 | 0.993" if reg == "IC"
            else "median 0.908 | 0.944 | 0.995",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.4,
            color=C_TEXT, bbox=TEXT_BG)


def panel_a(ax_ic, ax_dpl, seedmed):
    ax_ic.set_title("(a) Base–TGD2–CN ladder",
                    weight="bold", loc="left", pad=5)
    _facet_ladder_ecdf(ax_ic, seedmed, "IC")
    _facet_ladder_ecdf(ax_dpl, seedmed, "dPL")
    ax_ic.set_ylabel("Cumulative probability", labelpad=2)
    ax_ic.set_xlabel(r"Test KGE vs. $Q^*$", labelpad=2)
    ax_dpl.set_xlabel(r"Test KGE vs. $Q^*$", labelpad=2)
    handles = [
        Line2D([0], [0], color=C_NOREFIT, linestyle=(0, (1.5, 2.0)),
               linewidth=1.2, label="Base no-refit"),
        Line2D([0], [0], color=C_BASE, linestyle=(0, (4.0, 2.0)),
               linewidth=1.5, label="Base fitted"),
        Line2D([0], [0], color=C_TGD, linestyle="-", linewidth=1.6,
               label="TGD2"),
        Line2D([0], [0], color=C_CN, linestyle="-", linewidth=1.7,
               label="CN"),
    ]
    ax_ic.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.04),
                 frameon=True, framealpha=0.95, edgecolor="none", fontsize=7.6,
                 ncol=2, columnspacing=0.8, handlelength=1.4)


# ---------------------------------------------------------------------------
# Panel (b): TGD2 gap-mitigation fraction F_tgd2
# ---------------------------------------------------------------------------
def panel_b(ax, seedmed, summary):
    pb = summary["panel_b_f_tgd2"]
    ax.set_title("(b) Mitigation",
                 weight="bold", loc="left", pad=5)
    rng = np.random.default_rng(20260730)
    for xi, reg in enumerate(["IC", "dPL"]):
        entry = pb[reg]
        sub = seedmed[seedmed["paradigm"] == reg]
        vals = sub["F_tgd2"].dropna().to_numpy()
        win = vals[(vals >= -0.4) & (vals <= 1.4)]
        jx = xi + rng.uniform(-0.16, 0.16, len(win))
        marker = "o" if reg == "IC" else "^"
        col = C_TGD_IC if reg == "IC" else C_TGD
        ax.scatter(jx, win, s=7, alpha=0.28, marker=marker, color=col,
                   facecolors="white", edgecolors=col, linewidths=0.4, zorder=2)
        med, q25, q75 = entry["median"], entry["q25"], entry["q75"]
        lo, hi = entry["boot_ci_median_display"]
        bw = 0.26
        ax.add_patch(plt.Rectangle((xi - bw / 2, q25), bw, q75 - q25,
                                   facecolor=col, edgecolor=col,
                                   alpha=0.35, linewidth=0.8, zorder=3))
        ax.plot([xi - bw / 2, xi + bw / 2], [med, med], color=col,
                linewidth=1.6, zorder=4)
        ax.errorbar([xi], [med], yerr=[[med - lo], [hi - med]], fmt="none",
                    ecolor=C_TEXT, elinewidth=1.2, capsize=2.6, capthick=1.1,
                    zorder=5)
        if reg == "dPL":
            for s, smed in enumerate(entry["seed_medians"]):
                ax.plot(xi + 0.24 + 0.03 * s, smed, marker="D", markersize=2.6,
                        color=C_TGD_IC, markerfacecolor="none", markeredgewidth=0.7,
                        zorder=5)
        ax.text(xi, 1.15, f"{med:.3f}", ha="center", va="bottom", fontsize=8.0,
                color=C_TEXT, fontweight="bold")
    ax.axhline(0.0, color=C_REF, linestyle="-", linewidth=0.8, alpha=0.6, zorder=1)
    ax.axhline(0.5, color=C_REF, linestyle="--", linewidth=0.8, alpha=0.8, zorder=1)
    ax.text(1.48, 0.51, "half closure", ha="right", va="bottom", fontsize=7.0,
            color=C_REF)
    ax.axhline(1.0, color=C_REF, linestyle=":", linewidth=0.8, alpha=0.8, zorder=1)
    ax.text(1.48, 1.01, "full closure", ha="right", va="bottom", fontsize=7.0,
            color=C_REF)
    ax.set_xlim(-0.55, 1.6)
    ax.set_ylim(-0.55, 1.45)
    ax.set_xticks([0, 1])
    n_ic = pb["IC"]["n_valid"]
    n_dpl = pb["dPL"]["n_valid"]
    ax.set_xticklabels([f"IC\nn = {n_ic}", f"dPL\nn = {n_dpl}"], fontsize=8.4)
    ax.set_ylabel("Mitigation fraction", labelpad=2)
    ax.grid(True, axis="y", linestyle=":", alpha=0.25)
    ax.text(0.99, 0.02,
            r"unclipped; $\approx$4–8 % beyond window",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8,
            color="#666666")


# ---------------------------------------------------------------------------
# Panels (c)/(d): deterministic seasonal process summaries
# ---------------------------------------------------------------------------
def _load_seasonal(quantity: str) -> dict[str, np.ndarray]:
    path = SEASONAL_DIR / f"fig6_seasonal_{quantity}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"missing Figure 6 process export: {path}; run export_figure6_process_data.py"
        )
    with np.load(path) as data:
        return {key: np.asarray(data[key], dtype=float) for key in data.files}


def _plot_seasonal(ax, quantity: str, ylabel: str, title: str) -> None:
    data = _load_seasonal(quantity)
    styles = {
        "Base": (C_BASE, "o"),
        "TGD2": (C_TGD, "^"),
        "CN": (C_CN, "s"),
    }
    x = np.arange(12)
    for reg, ls, alpha in (("IC", "-", 1.0), ("dPL", "--", 0.72)):
        for structure, (color, marker) in styles.items():
            values = data[f"{structure}_{reg}"]
            median = np.nanmedian(values, axis=0)
            q25, q75 = np.nanpercentile(values, [25, 75], axis=0)
            ax.fill_between(x, q25, q75, color=color, alpha=0.06 if reg == "IC" else 0.035,
                            linewidth=0, zorder=1)
            ax.plot(x, median, color=color, linestyle=ls, linewidth=1.6,
                    marker=marker, markersize=3.4, markevery=1, alpha=alpha, zorder=3)
    ax.set_title(title, weight="bold", loc="left", pad=5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"],
                       fontsize=8.0)
    ax.set_xlabel("Water-year month", labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.grid(True, axis="y", linestyle=":", alpha=0.25)
    ax.legend([
        Line2D([0], [0], color=C_BASE, marker="o", linestyle="-", linewidth=1.5, markersize=3.5),
        Line2D([0], [0], color=C_TGD, marker="^", linestyle="-", linewidth=1.5, markersize=3.5),
        Line2D([0], [0], color=C_CN, marker="s", linestyle="-", linewidth=1.5, markersize=3.5),
        Line2D([0], [0], color=C_TEXT, linestyle="-", linewidth=1.3),
        Line2D([0], [0], color=C_TEXT, linestyle="--", linewidth=1.3),
    ], ["Base", "TGD2", "CN", "IC", "dPL"], loc="best", fontsize=7.0,
       frameon=True, framealpha=0.92, edgecolor="none", ncol=2)


def panel_c(ax):
    _plot_seasonal(ax, "input", r"Effective core input (mm d$^{-1}$)",
                   "(c) Seasonal release timing")


def panel_d(ax):
    _plot_seasonal(ax, "state", r"Total tension storage $w_t$ (mm)",
                   "(d) Shared-state response")



# ---------------------------------------------------------------------------
# Panel (e): Residual CN-over-TGD2 advantage vs. frac_snow (Hero Panel, merged)
# ---------------------------------------------------------------------------
def panel_e(ax, seedmed, summary):
    pe = summary["panel_e_residual_vs_frac_snow"]
    ax.set_title("(e) Residual CN advantage",
                 weight="bold", loc="left", pad=5)

    regimes = [
        ("IC", pe["IC"], "o", C_CN_IC, "-", -0.008, "IC (CMA-ES)"),
        ("dPL", pe["dPL"], "^", C_CN, "--", +0.008, "dPL (neural)"),
    ]

    for reg_name, entry, marker, color, ls, x_off, label in regimes:
        sub = seedmed[seedmed["paradigm"] == reg_name]
        x = sub["frac_snow"].to_numpy()
        y = sub["G_CN_over_TGD2"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]

        # Raw points
        ax.scatter(x, y, s=7, alpha=0.18, marker=marker, color=color,
                   edgecolors="none", zorder=2)

        # Quartile medians + bootstrap 95 % CI
        bins = entry["quartile_bins"]
        bx = np.asarray([b["frac_snow_median"] for b in bins]) + x_off
        bm = np.asarray([b["median"] for b in bins])
        blo = np.asarray([b["boot_ci_median_display"][0] for b in bins])
        bhi = np.asarray([b["boot_ci_median_display"][1] for b in bins])

        ax.plot(bx, bm, color=color, linestyle=ls, linewidth=1.6, alpha=0.95,
                zorder=4)
        ax.errorbar(bx, bm, yerr=[bm - blo, bhi - bm], fmt=marker, color=color,
                    ecolor=color, elinewidth=1.3, capsize=2.6, capthick=1.1,
                    markersize=5.5 if marker == "o" else 6.0,
                    markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=0.7, zorder=5)

    ax.axhline(0.0, color=C_REF, linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 0.35)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(r"Basin snow fraction, $f_{\mathrm{snow}}$", labelpad=2)
    ax.set_ylabel(r"$\Delta\mathrm{KGE}_{\mathrm{CN}-\mathrm{TGD2}}$", labelpad=2)
    ax.grid(True, axis="y", linestyle=":", alpha=0.25)

    # Spearman rho text box
    rho_ic = pe["IC"]["spearman_frozen"][0]
    rho_dpl = pe["dPL"]["spearman_frozen"]
    rho_txt = (f"IC:  $\\rho$ = +{rho_ic:.2f}\n"
               f"dPL: $\\rho$ = +{min(rho_dpl):.2f} .. +{max(rho_dpl):.2f}")
    ax.text(0.98, 0.04, rho_txt, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8.0, color=C_TEXT, linespacing=1.3, bbox=TEXT_BG)

    # Shared legend
    handles = [
        Line2D([0], [0], marker="o", color=C_CN_IC, linestyle="-",
               linewidth=1.5, markersize=5.0, markerfacecolor=C_CN_IC,
               markeredgecolor="white", markeredgewidth=0.5, label="IC (CMA-ES)"),
        Line2D([0], [0], marker="^", color=C_CN, linestyle="--",
               linewidth=1.5, markersize=5.5, markerfacecolor=C_CN,
               markeredgecolor="white", markeredgewidth=0.5, label="dPL (neural)"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.97),
              frameon=True, framealpha=0.92, edgecolor="none", fontsize=7.8)


# ---------------------------------------------------------------------------
# Panel (f): Process-conditioned residual (snow_active vs. non-snow)
# ---------------------------------------------------------------------------
def panel_f(ax, seedmed, summary):
    pf = summary["panel_f_process_errors"]
    ax.set_title(r"(f) Process-conditioned residual",
                 weight="bold", loc="left", pad=5)
    rng = np.random.default_rng(20260730)

    groups = [
        ("IC", "snow_active", 0, "IC\nsnow", C_CN_IC, "o"),
        ("IC", "no_snow_active", 1, "IC\nno snow", C_CN_IC, "o"),
        ("dPL", "snow_active", 2, "dPL\nsnow", C_CN, "^"),
        ("dPL", "no_snow_active", 3, "dPL\nno snow", C_CN, "^"),
    ]

    for reg, cond, xi, lbl, col, marker in groups:
        entry = pf[reg][cond]
        sub = seedmed[seedmed["paradigm"] == reg]
        vals = sub[f"delta_rmse_{cond}"].dropna().to_numpy()
        win = vals[(vals >= -0.3) & (vals <= 2.2)]
        jx = xi + rng.uniform(-0.16, 0.16, len(win))
        face = col if "snow_active" == cond else "white"
        ax.scatter(jx, win, s=7, alpha=0.28, marker=marker, color=col,
                   facecolors=face, edgecolors=col, linewidths=0.4, zorder=2)
        med, q25, q75 = entry["median"], entry["q25"], entry["q75"]
        lo, hi = entry["boot_ci_median_display"]
        bw = 0.26
        ax.add_patch(plt.Rectangle((xi - bw / 2, q25), bw, q75 - q25,
                                   facecolor=col, edgecolor=col,
                                   alpha=0.35, linewidth=0.8, zorder=3))
        ax.plot([xi - bw / 2, xi + bw / 2], [med, med], color=col,
                linewidth=1.6, zorder=4)
        ax.errorbar([xi], [med], yerr=[[med - lo], [hi - med]], fmt="none",
                    ecolor=C_TEXT, elinewidth=1.2, capsize=2.6, capthick=1.1,
                    zorder=5)
        ax.text(xi, 1.85 if "snow_active" == cond else 0.40,
                f"+{med:.3f}", ha="center", va="bottom", fontsize=7.8,
                color=C_TEXT, fontweight="bold")

    ax.axhline(0.0, color=C_REF, linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xlim(-0.55, 3.65)
    ax.set_ylim(-0.45, 2.35)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([g[3] for g in groups], fontsize=8.2)
    ax.set_ylabel(r"Residual RMSE gap (mm d$^{-1}$)", labelpad=2)
    ax.grid(True, axis="y", linestyle=":", alpha=0.25)
    # compact legend: fill convention (x tick labels already identify IC/dPL groups)
    handles = [
        Line2D([0], [0], marker="o", color=C_CN_IC, linestyle="none",
               markerfacecolor=C_CN_IC, markersize=4.5, label="snow active"),
        Line2D([0], [0], marker="o", color=C_CN_IC, linestyle="none",
               markerfacecolor="white", markeredgecolor=C_CN_IC, markersize=4.5,
               label="no snow"),
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.97),
              frameon=True, framealpha=0.92, edgecolor="none", fontsize=6.8)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def build_figure(tidy, seedmed, summary) -> None:
    # Three rows: (a)|(b), process (c)|(d), then (e)|(f).
    fig = plt.figure(figsize=(13.4, 10.4))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.30,
                           left=0.055, right=0.99, top=0.965, bottom=0.05)

    # Row 1: (a) wide structural ladder | (b) compact mitigation.
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0],
                                           width_ratios=[0.70, 0.30], wspace=0.16)
    gsa = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs1[0],
                                           width_ratios=[0.50, 0.50], wspace=0.10)
    ax_a1 = fig.add_subplot(gsa[0, 0]); apply_clean_spines(ax_a1)
    ax_a2 = fig.add_subplot(gsa[0, 1]); apply_clean_spines(ax_a2)
    ax_b = fig.add_subplot(gs1[1]); apply_clean_spines(ax_b)
    panel_a(ax_a1, ax_a2, seedmed)
    panel_b(ax_b, seedmed, summary)

    # Row 2: equal-width seasonal process panels.
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1],
                                           width_ratios=[0.50, 0.50], wspace=0.14)
    ax_c = fig.add_subplot(gs2[0]); apply_clean_spines(ax_c)
    ax_d = fig.add_subplot(gs2[1]); apply_clean_spines(ax_d)
    panel_c(ax_c)
    panel_d(ax_d)

    # Row 3: (e) wide residual advantage | (f) compact process concentration.
    gs3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2],
                                           width_ratios=[0.68, 0.32], wspace=0.14)
    ax_e = fig.add_subplot(gs3[0]); apply_clean_spines(ax_e)
    ax_f = fig.add_subplot(gs3[1]); apply_clean_spines(ax_f)
    panel_e(ax_e, seedmed, summary)
    panel_f(ax_f, seedmed, summary)


    for out_dir in (FIG_DIR, PLOTS_FIG_DIR):
        plt.savefig(out_dir / f"{OUT_NAME}.png", dpi=600)
        print("saved:", out_dir / f"{OUT_NAME}.png")
    plt.close()


def main() -> None:
    setup_publication_style()
    # Moderate font up-scaling local to Figure 6 (manuscript readability).
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.0,
    })
    tidy, seedmed, summary = load_data()
    build_figure(tidy, seedmed, summary)
    print("Final Figure 6 (6-panel composite) generated successfully.")


if __name__ == "__main__":
    main()
