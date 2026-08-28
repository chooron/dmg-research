#!/usr/bin/env python3
"""Figure 5: Known-truth limits of Base parameter compensation and TGD generic control.

Hierarchical 5-panel layout matching Results 3.3 outlet recovery architecture:
  Row 1 (Recoverability Reference & Outlet Ladder):
    (a) CN-refit recoverability (context log-ECDF, 1 - KGE_CN)
    (b) Outlet recovery ladder (unified ECDF for Base no-refit, Base refit, TGD, CN refit across IC and dPL)
  Row 2 (Quantitative Recovery Evidence):
    (c) Recovery from the imposed knockout (D vs G_Base and G_TGD across 531 basins)
    (d) Normalized gap recovery (F_close and F_TGD on common D > 1e-6 set)
  Row 3 (Generalization Robustness Strip):
    (e) Train-to-test recovery attenuation (compact horizontal forest distribution)

Visual grammar (matching Figure 1 Okabe-Ito standard):
  - Base refit:       #D55E00 (Okabe-Ito vermillion / orange)
  - TGD refit:        #009E73 (Okabe-Ito bluish green / teal)
  - CN refit:         #0072B2 (Okabe-Ito blue)
  - Base no-refit:    #7A7A7A (neutral grey, dashed)
  - Reference lines:  #888888 / #C8CDD1 (neutral grey)
  - Dark neutral:     #2B2B2B
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from r1_plot_style import apply_clean_spines, setup_publication_style  # noqa: E402

PROJECT = Path(__file__).resolve().parents[3]
MANUSCRIPT = PROJECT / "manuscript"
RESULTS_R3 = MANUSCRIPT / "results" / "R3"
FIG_DIR = MANUSCRIPT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_NAME = "Figure5_R3_final"

# ---------------------------------------------------------------------------
# Color Palette (Figure 1 Okabe-Ito Standard)
# ---------------------------------------------------------------------------
COLOR_BASE = "#D55E00"       # Okabe-Ito vermillion / deep orange: Base refit
COLOR_TGD = "#009E73"        # Okabe-Ito bluish green / teal: TGD refit
COLOR_CN = "#0072B2"         # Okabe-Ito blue: CN refit
COLOR_NOREFIT = "#7A7A7A"    # Neutral grey: Base no-refit knockout reference
COLOR_TRUTH = "#2B2B2B"      # Dark neutral: Truth / generating
COLOR_ZERO_LINE = "#888888"  # Mid grey reference lines
COLOR_LIGHT_REF = "#C8CDD1"  # Light grid
COLOR_DARK_NEUTRAL = "#2B2B2B"


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
    if len(v) == 0:
        return np.array([]), np.array([])
    return v, np.arange(1, len(v) + 1) / len(v)


# ---------------------------------------------------------------------------
# Panel (a): CN-refit recoverability (context ECDF)
# ---------------------------------------------------------------------------
def panel_a(ax, seedmed, summary):
    ax.set_title("(a) CN-refit recoverability", weight="bold", loc="left", pad=4, fontsize=8.8)

    styles = [
        ("IC", "test", "-", 1.5, 0.95, 5, "IC test"),
        ("dPL", "test", "--", 1.5, 0.95, 5, "dPL test"),
        ("IC", "train", "-", 0.9, 0.35, 4, "IC train"),
        ("dPL", "train", "--", 0.9, 0.35, 4, "dPL train"),
    ]

    for reg, period, ls, lw, alpha, zo, lbl in styles:
        sub = seedmed[(seedmed["paradigm"] == reg) & (seedmed["period"] == period)]
        d = 1.0 - sub["kge_cn"].to_numpy()
        x, y = ecdf(d)
        if len(x) > 0:
            ax.step(x, y, where="post", color=COLOR_CN, linestyle=ls, linewidth=lw, alpha=alpha, zorder=zo, label=lbl)

    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.0)
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    ax.set_xticklabels(["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$10^{0}$"], fontsize=7.2)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Deficit, $1 - \\mathrm{KGE}_{\\mathrm{CN}}$", labelpad=2, fontsize=8.0)
    ax.set_ylabel("Cumulative prob.", labelpad=2, fontsize=8.0)
    ax.grid(True, axis="both", linestyle=":", alpha=0.35, color=COLOR_LIGHT_REF)

    # Context note
    ax.text(
        0.04, 0.92, "CN refit is the recoverability reference,\nnot generating truth",
        transform=ax.transAxes, va="top", ha="left", fontsize=6.5, color="#555555", style="italic"
    )

    # Local legend for train vs test line style
    legend_elements = [
        Line2D([0], [0], color=COLOR_CN, ls="-", lw=1.4, label="Test"),
        Line2D([0], [0], color=COLOR_CN, ls="-", lw=0.9, alpha=0.45, label="Train"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=6.8, frameon=False, handlelength=1.5)


# ---------------------------------------------------------------------------
# Panel (b): Unified outlet recovery ladder (IC and dPL together)
# ---------------------------------------------------------------------------
def panel_b(ax, seedmed, summary):
    ax.set_title("(b) Outlet recovery ladder", weight="bold", loc="left", pad=4, fontsize=8.8)

    te_ic = seedmed[(seedmed["paradigm"] == "IC") & (seedmed["period"] == "test")]
    te_dpl = seedmed[(seedmed["paradigm"] == "dPL") & (seedmed["period"] == "test")]

    # 1. Base no-refit knockout (neutral grey dashed line)
    x_nr, y_nr = ecdf(te_ic["kge_base_no_refit"].to_numpy())
    ax.step(x_nr, y_nr, where="post", color=COLOR_NOREFIT, linestyle="--", linewidth=1.4, alpha=0.95, zorder=3, label="Base no-refit")

    # 2. Base refit (orange: IC solid, dPL dashed)
    x_bic, y_bic = ecdf(te_ic["kge_base"].to_numpy())
    x_bdpl, y_bdpl = ecdf(te_dpl["kge_base"].to_numpy())
    ax.step(x_bic, y_bic, where="post", color=COLOR_BASE, linestyle="-", linewidth=1.6, alpha=0.95, zorder=4, label="Base refit (IC)")
    ax.step(x_bdpl, y_bdpl, where="post", color=COLOR_BASE, linestyle="--", linewidth=1.6, alpha=0.95, zorder=4, label="Base refit (dPL)")

    # 3. TGD refit (green: IC solid, dPL dashed)
    x_tic, y_tic = ecdf(te_ic["kge_tgd2"].to_numpy())
    x_tdpl, y_tdpl = ecdf(te_dpl["kge_tgd2"].to_numpy())
    ax.step(x_tic, y_tic, where="post", color=COLOR_TGD, linestyle="-", linewidth=1.6, alpha=0.95, zorder=5, label="TGD (IC)")
    ax.step(x_tdpl, y_tdpl, where="post", color=COLOR_TGD, linestyle="--", linewidth=1.6, alpha=0.95, zorder=5, label="TGD (dPL)")

    # 4. CN refit (blue: IC solid, dPL dashed)
    x_cic, y_cic = ecdf(te_ic["kge_cn"].to_numpy())
    x_cdpl, y_cdpl = ecdf(te_dpl["kge_cn"].to_numpy())
    ax.step(x_cic, y_cic, where="post", color=COLOR_CN, linestyle="-", linewidth=1.8, alpha=0.95, zorder=6, label="CN refit (IC)")
    ax.step(x_cdpl, y_cdpl, where="post", color=COLOR_CN, linestyle="--", linewidth=1.8, alpha=0.95, zorder=6, label="CN refit (dPL)")

    # Summary median markers on curves at y = 0.50
    med_nr = te_ic["kge_base_no_refit"].median()
    med_bic = te_ic["kge_base"].median()
    med_bdpl = te_dpl["kge_base"].median()
    med_tic = te_ic["kge_tgd2"].median()
    med_tdpl = te_dpl["kge_tgd2"].median()
    med_cic = te_ic["kge_cn"].median()
    med_cdpl = te_dpl["kge_cn"].median()

    ax.scatter([med_bic, med_tic, med_cic], [0.5, 0.5, 0.5], color=[COLOR_BASE, COLOR_TGD, COLOR_CN], marker="o", s=22, zorder=7, edgecolors="white", linewidths=0.5)
    ax.scatter([med_bdpl, med_tdpl, med_cdpl], [0.5, 0.5, 0.5], color=[COLOR_BASE, COLOR_TGD, COLOR_CN], marker="^", s=24, zorder=7, edgecolors="white", linewidths=0.5)

    ax.set_xlim(0.35, 1.01)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0.4", "0.6", "0.8", "1.0"], fontsize=7.2)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Evaluation KGE vs. $Q^*$", labelpad=2, fontsize=8.0)
    ax.set_ylabel("Cumulative prob.", labelpad=2, fontsize=8.0)
    ax.grid(True, axis="both", linestyle=":", alpha=0.35, color=COLOR_LIGHT_REF)

    # Aligned concise median text box
    med_text = (
        f"Test KGE medians (IC / dPL):\n"
        f" Base no-refit: {med_nr:.3f} / {med_nr:.3f}\n"
        f" Base refit:    {med_bic:.3f} / {med_bdpl:.3f}\n"
        f" TGD:           {med_tic:.3f} / {med_tdpl:.3f}\n"
        f" CN refit:      {med_cic:.3f} / {med_cdpl:.3f}"
    )
    ax.text(0.04, 0.92, med_text, transform=ax.transAxes, va="top", ha="left", fontsize=6.6, family="monospace", color=COLOR_DARK_NEUTRAL)


# ---------------------------------------------------------------------------
# Panel (c): Recovery from the imposed knockout (PRIMARY)
# ---------------------------------------------------------------------------
def panel_c(ax, seedmed, summary):
    ax.set_title("(c) Recovery from the imposed knockout", weight="bold", loc="left", pad=4, fontsize=8.8)

    # Reference lines: y = 0 and y = x
    ax.axhline(0, color=COLOR_ZERO_LINE, linestyle="--", linewidth=0.8, zorder=1)
    ax.plot([0, 1.15], [0, 1.15], color=COLOR_ZERO_LINE, linestyle=":", linewidth=0.8, zorder=1)
    ax.text(0.88, 0.93, "$G = D$ (1:1 closure)", color="#666666", fontsize=6.8, rotation=33, transform=ax.transAxes)

    # Scatter points and binned summary points with vertical 95% CIs
    # Summary markers are hollow / white-filled with colored edges for clean visibility
    for reg, marker in [("IC", "o"), ("dPL", "^")]:
        sub = seedmed[(seedmed["paradigm"] == reg) & (seedmed["period"] == "test")]
        d_vals = sub["D"].to_numpy()
        gb_vals = sub["G_base"].to_numpy()
        gt_vals = sub["G_TGD"].to_numpy()

        # Light raw scatter points
        ax.scatter(d_vals, gb_vals, color=COLOR_BASE, marker=marker, s=8, alpha=0.10, edgecolors="none", zorder=2)
        ax.scatter(d_vals, gt_vals, color=COLOR_TGD, marker=marker, s=8, alpha=0.10, edgecolors="none", zorder=2)

        # Binned summary points + vertical 95% CIs
        binned = summary["panel_c_raw_recovery"][reg]["binned"]
        bin_d = [b["D_median"] for b in binned if b["D_median"] > 0]
        bin_gb = [b["G_base_median"] for b in binned if b["D_median"] > 0]
        bin_gt = [b["G_TGD_median"] for b in binned if b["D_median"] > 0]

        gb_ci = [b["G_base_ci"] for b in binned if b["D_median"] > 0]
        gt_ci = [b["G_TGD_ci"] for b in binned if b["D_median"] > 0]

        yerr_gb = [[bin_gb[i] - gb_ci[i][0] for i in range(len(bin_gb))],
                   [gb_ci[i][1] - bin_gb[i] for i in range(len(bin_gb))]]
        yerr_gt = [[bin_gt[i] - gt_ci[i][0] for i in range(len(bin_gt))],
                   [gt_ci[i][1] - bin_gt[i] for i in range(len(bin_gt))]]

        reg_lbl = "IC" if reg == "IC" else "dPL"
        off = -0.007 if reg == "IC" else +0.007

        # Discrete summary points + vertical CIs (hollow/white-filled with coloured outline)
        ax.errorbar(
            np.array(bin_d) + off, bin_gb, yerr=yerr_gb, fmt=marker,
            color=COLOR_BASE, mfc="white", mec=COLOR_BASE, mew=1.3,
            ecolor=COLOR_BASE, elinewidth=1.4, capsize=2.4, capthick=1.1, markersize=5.0, zorder=6,
            label=f"$G_{{\\mathrm{{Base}}}}$ ({reg_lbl})"
        )
        ax.errorbar(
            np.array(bin_d) + off, bin_gt, yerr=yerr_gt, fmt=marker,
            color=COLOR_TGD, mfc="white", mec=COLOR_TGD, mew=1.3,
            ecolor=COLOR_TGD, elinewidth=1.4, capsize=2.4, capthick=1.1, markersize=5.0, zorder=6,
            label=f"$G_{{\\mathrm{{TGD}}}}$ ({reg_lbl})"
        )

    ax.set_xlim(-0.06, 1.15)
    ax.set_ylim(-0.15, 1.15)
    ax.set_xlabel("Imposed knockout deficit, $D = \\mathrm{KGE}_{\\mathrm{CN}} - \\mathrm{KGE}_{\\mathrm{Base,no\\text{-}refit}}$", labelpad=2, fontsize=8.0)
    ax.set_ylabel("Raw recovery, $G = \\mathrm{KGE}_{\\mathrm{fit}} - \\mathrm{KGE}_{\\mathrm{Base,no\\text{-}refit}}$", labelpad=2, fontsize=8.0)
    ax.grid(True, axis="both", linestyle=":", alpha=0.35, color=COLOR_LIGHT_REF)

    # Concise correlation and summary text in upper left
    ax.text(
        0.03, 0.94,
        "Base recovery: med +0.003 (IC), +0.007 (dPL);  $\\rho(D, G) = +0.80 / +0.85$\n"
        "TGD recovery:  med +0.039 (IC), +0.036 (dPL);  $\\rho(D, G) = +0.88 / +0.87$",
        transform=ax.transAxes, va="top", ha="left", fontsize=6.6, family="monospace", color=COLOR_DARK_NEUTRAL
    )

    ax.legend(loc="lower right", fontsize=6.6, frameon=False, ncol=2, handlelength=1.4, columnspacing=0.8)


# ---------------------------------------------------------------------------
# Panel (d): Normalized gap recovery (F_close & F_TGD*)
# ---------------------------------------------------------------------------
# Panel (d): Normalized gap recovery (F_close & F_TGD)
# ---------------------------------------------------------------------------
def panel_d(ax, seedmed, summary):
    ax.set_title("(d) Normalized gap recovery", weight="bold", loc="left", pad=4, fontsize=8.8)

    # Reference lines at F = 0 and F = 1
    ax.axhline(0.0, color=COLOR_ZERO_LINE, linestyle="--", linewidth=0.8, zorder=1)
    ax.axhline(1.0, color=COLOR_ZERO_LINE, linestyle=":", linewidth=0.8, zorder=1)

    pd_frac = summary["panel_d_fractions"]

    groups = [
        ("IC_test", 0.0, "IC\ntest"),
        ("dPL_test", 1.0, "dPL\ntest"),
        ("IC_train", 2.0, "IC\ntrain"),
        ("dPL_train", 3.0, "dPL\ntrain"),
    ]

    rng = np.random.default_rng(20260730)
    w_off = 0.16
    bar_w = 0.07  # width of thick horizontal median bar

    for key, xc, xlbl in groups:
        reg, period = key.split("_")
        entry = pd_frac[key]
        sub = seedmed[(seedmed["paradigm"] == reg) & (seedmed["period"] == period)]

        fc = sub["F_close"].dropna().to_numpy()
        ft_col = "F_TGD" if "F_TGD" in sub.columns else "F_TGD_star"
        ft = sub[ft_col].dropna().to_numpy()

        # F_close points & thick median bar + CI whisker
        fc_win = fc[(fc >= -0.5) & (fc <= 1.5)]
        jx_fc = xc - w_off + rng.uniform(-0.04, 0.04, len(fc_win))
        ax.scatter(jx_fc, fc_win, color=COLOR_BASE, s=6, alpha=0.12, edgecolors="none", zorder=2)

        fc_med = entry["F_close"]["median"]
        fc_ci = entry["F_close"]["ci"]
        # Thin vertical 95% CI whisker
        ax.errorbar(xc - w_off, fc_med, yerr=[[fc_med - fc_ci[0]], [fc_ci[1] - fc_med]],
                    fmt="none", ecolor=COLOR_BASE, elinewidth=1.2, capsize=2.2, capthick=1.0, zorder=4)
        # Thick horizontal median bar
        ax.plot([xc - w_off - bar_w, xc - w_off + bar_w], [fc_med, fc_med], color=COLOR_BASE, linewidth=2.5, solid_capstyle="butt", zorder=5)

        # F_TGD points & thick median bar + CI whisker
        ft_win = ft[(ft >= -0.5) & (ft <= 1.5)]
        jx_ft = xc + w_off + rng.uniform(-0.04, 0.04, len(ft_win))
        ax.scatter(jx_ft, ft_win, color=COLOR_TGD, s=6, alpha=0.12, edgecolors="none", zorder=2)

        ft_entry = entry.get("F_TGD", entry.get("F_TGD_star"))
        ft_med = ft_entry["median"]
        ft_ci = ft_entry["ci"]
        # Thin vertical 95% CI whisker
        ax.errorbar(xc + w_off, ft_med, yerr=[[ft_med - ft_ci[0]], [ft_ci[1] - ft_med]],
                    fmt="none", ecolor=COLOR_TGD, elinewidth=1.2, capsize=2.2, capthick=1.0, zorder=4)
        # Thick horizontal median bar
        ax.plot([xc + w_off - bar_w, xc + w_off + bar_w], [ft_med, ft_med], color=COLOR_TGD, linewidth=2.5, solid_capstyle="butt", zorder=5)

        # dPL per-seed medians as small open diamonds
        if reg == "dPL":
            if "seed_medians" in entry["F_close"]:
                for sm in entry["F_close"]["seed_medians"]:
                    ax.plot(xc - w_off, sm, marker="d", markersize=2.8, color=COLOR_BASE, fillstyle="none", zorder=6)
            if "seed_medians" in ft_entry:
                for sm in ft_entry["seed_medians"]:
                    ax.plot(xc + w_off, sm, marker="d", markersize=2.8, color=COLOR_TGD, fillstyle="none", zorder=6)

        # Valid N annotation below axis
        n_val = entry["n_valid"]
        ax.text(xc, -0.44, f"$N={n_val}$", ha="center", va="bottom", fontsize=6.6, color="#666666")

    # Range for data columns x in 0..3
    ax.set_xlim(-0.45, 3.45)
    ax.set_xticks([g[1] for g in groups])
    ax.set_xticklabels([g[2] for g in groups], fontsize=7.2)
    ax.set_ylim(-0.48, 1.48)
    ax.set_ylabel("Normalized fraction ($G / D$)", labelpad=2, fontsize=8.0)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35, color=COLOR_LIGHT_REF)

    # Legend placed compactly in upper right of panel (d)
    legend_elements = [
        Line2D([0], [0], color=COLOR_BASE, lw=2.2, label="$F_{\\mathrm{close}} = G_{\\mathrm{Base}} / D$"),
        Line2D([0], [0], color=COLOR_TGD, lw=2.2, label="$F_{\\mathrm{TGD}} = G_{\\mathrm{TGD}} / D$"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        fontsize=6.8,
        frameon=False,
    )

    # Concise note in top left
    ax.text(
        0.04, 0.92, "Common $D > 10^{-6}$ set\n$F_{\\mathrm{close}}$ test: ~0.10\n$F_{\\mathrm{TGD}}$ test: ~0.52",
        transform=ax.transAxes, va="top", ha="left", fontsize=6.6, family="monospace", color=COLOR_DARK_NEUTRAL
    )


# ---------------------------------------------------------------------------
# Panel (e): Train-to-test recovery attenuation (Compact Forest Distribution)
# ---------------------------------------------------------------------------
def panel_e(ax, seedmed, summary):
    ax.set_title("(e) Train-to-test recovery attenuation", weight="bold", loc="left", pad=4, fontsize=8.8)

    ax.axvline(0.0, color=COLOR_ZERO_LINE, linestyle="--", linewidth=0.8, zorder=1)

    pe_decay = summary["panel_e_decay"]

    rows = [
        ("IC", "decay_G_base", 3.0, "IC Base ($G_{\\mathrm{Base}}$)", COLOR_BASE, "o"),
        ("IC", "decay_G_tgd",  2.0, "IC TGD ($G_{\\mathrm{TGD}}$)",   COLOR_TGD,  "o"),
        ("dPL", "decay_G_base", 1.0, "dPL Base ($G_{\\mathrm{Base}}$)", COLOR_BASE, "^"),
        ("dPL", "decay_G_tgd",  0.0, "dPL TGD ($G_{\\mathrm{TGD}}$)",  COLOR_TGD,  "^"),
    ]

    for reg, metric, ypos, lbl, col, marker in rows:
        sub = seedmed[seedmed["paradigm"] == reg].drop_duplicates("basin_id")
        d_col = "decay_G_base" if metric == "decay_G_base" else "decay_G_tgd2"
        d_vals = sub[d_col].dropna().to_numpy()

        p10 = pe_decay[reg][metric]["p10"]
        q25 = pe_decay[reg][metric]["q25"]
        med = pe_decay[reg][metric]["median"]
        q75 = pe_decay[reg][metric]["q75"]
        p90 = pe_decay[reg][metric]["p90"]
        ci = pe_decay[reg][metric]["ci"]
        p_gt0 = pe_decay[reg][metric]["p_gt_0"]

        # 1. Thin colored line: P10 to P90
        ax.plot([p10, p90], [ypos, ypos], color=col, linestyle="-", linewidth=1.1, zorder=2)

        # 2. Thicker colored line: Q25 to Q75 (IQR)
        ax.plot([q25, q75], [ypos, ypos], color=col, linestyle="-", linewidth=3.2, solid_capstyle="butt", zorder=3)

        # 3. Median marker in dark neutral
        ax.scatter([med], [ypos], color=COLOR_DARK_NEUTRAL, marker=marker, s=28, zorder=5, edgecolors="none")

        # 4. Bootstrap 95% CI error bar on median
        ax.errorbar(
            med, ypos, xerr=[[med - ci[0]], [ci[1] - med]],
            fmt="none", ecolor=COLOR_DARK_NEUTRAL, elinewidth=1.6, capsize=2.6, capthick=1.2, zorder=6
        )

        # 5. Right-side prevalence cue: P(Delta G > 0)
        ax.text(
            0.138, ypos, f"med = +{med:.4f}  |  P(>0) = {p_gt0*100:.0f}%",
            va="center", ha="right", fontsize=6.6, family="monospace", color=COLOR_DARK_NEUTRAL
        )

    ax.set_ylim(-0.6, 3.8)
    ax.set_yticks([r[2] for r in rows])
    ax.set_yticklabels([r[3] for r in rows], fontsize=7.5)
    ax.set_xlim(-0.06, 0.145)
    ax.set_xticks([-0.05, 0.0, 0.05, 0.10])
    ax.set_xticklabels(["-0.05", "0.00", "+0.05", "+0.10"], fontsize=7.2)
    ax.set_xlabel("Recovery attenuation, $\\Delta G = G_{\\mathrm{train}} - G_{\\mathrm{test}}$", labelpad=2, fontsize=8.0)
    ax.grid(True, axis="x", linestyle=":", alpha=0.35, color=COLOR_LIGHT_REF)

    # Forest legend placed inside panel (e), single row starting around x = +0.06 shifted upward
    legend_elements = [
        Line2D([0], [0], color=COLOR_DARK_NEUTRAL, lw=1.1, label="P10–P90 (thin)"),
        Line2D([0], [0], color=COLOR_DARK_NEUTRAL, lw=3.2, label="Q25–Q75 (thick)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_DARK_NEUTRAL, markersize=4.5, label="Med. ± 95% CI"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.58, 1.06),
        ncol=3,
        frameon=False,
        fontsize=6.4,
        handlelength=1.4,
        columnspacing=0.8,
        handletextpad=0.4,
        borderaxespad=0.0,
    )


# ---------------------------------------------------------------------------
# Figure Assembly (Non-uniform GridSpec with Figure 1 Top Legends)
# ---------------------------------------------------------------------------
def build_figure(tidy, seedmed, summary) -> None:
    # 7.8 x 8.8 inches canvas for compact HESS publication standard
    fig = plt.figure(figsize=(7.8, 8.8))
    setup_publication_style()

    # Outer layout: 3 rows with non-uniform height ratios
    gs = gridspec.GridSpec(
        3, 1,
        height_ratios=[1.00, 1.15, 0.44],
        hspace=0.30,
        left=0.080,
        right=0.985,
        top=0.935,
        bottom=0.050,
    )

    # Row 1: (a) ~36% context (0.36) | (b) ~64% Unified recovery ladder (0.64)
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], width_ratios=[0.36, 0.64], wspace=0.22)
    ax_a = fig.add_subplot(gs1[0, 0])
    apply_clean_spines(ax_a)
    ax_b = fig.add_subplot(gs1[0, 1])
    apply_clean_spines(ax_b)

    panel_a(ax_a, seedmed, summary)
    panel_b(ax_b, seedmed, summary)

    # Row 2: (c) PRIMARY raw recovery ~64% (0.64) | (d) Normalized fractions ~36% (0.36)
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], width_ratios=[0.64, 0.36], wspace=0.22)
    ax_c = fig.add_subplot(gs2[0, 0])
    apply_clean_spines(ax_c)
    ax_d = fig.add_subplot(gs2[0, 1])
    apply_clean_spines(ax_d)

    panel_c(ax_c, seedmed, summary)
    panel_d(ax_d, seedmed, summary)

    # Row 3: (e) Full-width horizontal forest distribution footer strip
    gs3 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[2])
    ax_e = fig.add_subplot(gs3[0, 0])
    apply_clean_spines(ax_e)

    panel_e(ax_e, seedmed, summary)

    # Shared Legends at top (matching Figure 1 design)
    struct_handles = [
        Line2D([0], [0], color=COLOR_BASE, lw=1.5, marker="o", markersize=4.2, markerfacecolor=COLOR_BASE, markeredgecolor="white", label="Base refit"),
        Line2D([0], [0], color=COLOR_TGD, lw=1.5, marker="^", markersize=4.2, markerfacecolor=COLOR_TGD, markeredgecolor="white", label="TGD"),
        Line2D([0], [0], color=COLOR_CN, lw=1.5, marker="s", markersize=4.2, markerfacecolor=COLOR_CN, markeredgecolor="white", label="CN refit"),
        Line2D([0], [0], color=COLOR_NOREFIT, lw=1.3, ls="--", label="Base no-refit (knockout)"),
    ]
    regime_handles = [
        Line2D([0], [0], color=COLOR_DARK_NEUTRAL, ls="-", lw=1.3, marker="o", markersize=4.0, markerfacecolor=COLOR_DARK_NEUTRAL, markeredgecolor="white", label="IC (CMA-ES)"),
        Line2D([0], [0], color=COLOR_DARK_NEUTRAL, ls="--", lw=1.3, marker="^", markersize=4.0, markerfacecolor=COLOR_DARK_NEUTRAL, markeredgecolor="white", label="dPL (neural)"),
    ]

    fig.legend(
        handles=struct_handles,
        loc="upper left",
        bbox_to_anchor=(0.080, 0.985),
        ncol=4,
        frameon=False,
        fontsize=7.5,
        handlelength=1.5,
        columnspacing=0.9,
    )
    fig.legend(
        handles=regime_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=2,
        frameon=False,
        fontsize=7.5,
        handlelength=1.5,
        columnspacing=0.9,
    )

    # Save outputs
    out_path = FIG_DIR / f"{OUT_NAME}.png"
    plt.savefig(out_path, dpi=600, facecolor="white", edgecolor="none")
    plt.close(fig)

    print(f"[plot] Saved Figure 5 -> {out_path}", flush=True)


def main():
    tidy, seedmed, summary = load_data()
    build_figure(tidy, seedmed, summary)


if __name__ == "__main__":
    main()
