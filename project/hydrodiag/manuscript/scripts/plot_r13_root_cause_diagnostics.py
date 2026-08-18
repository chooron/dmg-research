"""Generate Figure R13: Flex-MOPEX Root-Cause Diagnostics of Shared-Head Conflict.

Publication-ready 4-panel figure following HESS / Copernicus guidelines:
- Panel A: Hypothesis A — Representation Probe OOF ROC-AUC Trajectory (ep0..ep10) vs Raw Attributes X
- Panel B: Hypothesis C — Gradient Coherence & Counterfactual Decompositions (Canonical, No-Bias, Centered)
- Panel C: Hypothesis B — Parameter Compensation Dynamics vs Finite Interception Benefit (Fixed 103 cohort)
- Panel D: Hypothesis D — Initialization & Cross-Head Gradient Interference Trajectories

Outputs:
    manuscript/figures/figure_r13_root_cause_diagnostics.png (600 DPI)
    manuscript/figures/figure_r13_root_cause_diagnostics.pdf (vector)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from manuscript.scripts.r1_plot_style import (
        apply_clean_spines,
        setup_publication_style,
    )
except ImportError:
    def setup_publication_style():
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "figure.titlesize": 10.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "patch.linewidth": 0.8,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

    def apply_clean_spines(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.set_facecolor('#FFFFFF')


DEFAULT_R13_DIR = Path("/home/jingxin/orca/workspaces/dmg-research/flex-mopex/project/flexmopex/results/root_cause_r13")
DEFAULT_OUT_DIR = HERE.parents[0] / "figures"


def generate_figure_r13(r13_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    setup_publication_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load Data
    # -----------------------------------------------------------------------
    df_rep = pd.read_csv(r13_dir / "representation_probe_table.csv")
    df_grad = pd.read_csv(r13_dir / "gradient_coherence_table.csv")
    df_init = pd.read_csv(r13_dir / "initialization_audit_table.csv")
    with open(r13_dir / "initialization_audit.json") as f:
        init_json = json.load(f)
    with open(r13_dir / "compensation_audit_summary.json") as f:
        comp_summary = json.load(f)
    df_comp_param = pd.read_csv(r13_dir / "compensation_parameter_trajectory.csv")
    df_comp_ben = pd.read_csv(r13_dir / "compensation_benefit_trajectory.csv")

    # -----------------------------------------------------------------------
    # Create 2x2 multi-panel layout (width 7.2 in for 2-column Copernicus journals)
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.28, left=0.08, right=0.96, top=0.94, bottom=0.07)

    # Palette
    c_blue = "#0077BB"
    c_orange = "#EE7733"
    c_teal = "#009988"
    c_red = "#CC3311"
    c_grey = "#666666"
    c_dark = "#222222"

    # =======================================================================
    # Panel A: Hypothesis A — Representation Probe OOF ROC-AUC Trajectory
    # =======================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    apply_clean_spines(ax_a)

    target_filter = "sensitivity_r8_ep2"
    # Raw Attributes reference
    raw_row = df_rep[(df_rep["run"] == "Raw_Attributes_X") & (df_rep["target"] == target_filter)]
    raw_auc = raw_row["oof_roc_auc"].values[0] if len(raw_row) > 0 else 0.6393

    # Filter h_final_128d
    runs_meta = [
        ("Baseline", "Baseline", c_orange, "o", "-"),
        ("R8_AICDelay", "R8 (AIC Delay)", c_blue, "s", "-"),
        ("R10B_ReweightDelay", "R10-B (Reweight Delay)", c_teal, "^", "-."),
    ]

    for run_name, label, color, marker, ls in runs_meta:
        sub = df_rep[
            (df_rep["run"] == run_name)
            & (df_rep["feature_set"] == "h_final_128d")
            & (df_rep["target"] == target_filter)
        ].sort_values("epoch")
        if not sub.empty:
            epochs = sub["epoch"].values
            aucs = sub["oof_roc_auc"].values
            stds = sub["fold_roc_auc_std"].values
            ax_a.plot(epochs, aucs, label=label, color=color, marker=marker, markersize=4.5, linestyle=ls, linewidth=1.4, zorder=3)
            ax_a.fill_between(epochs, aucs - stds, aucs + stds, color=color, alpha=0.12, zorder=2)

    # Reference line for Raw Attributes X
    ax_a.axhline(raw_auc, color=c_grey, linestyle="--", linewidth=1.1, label=f"Raw Attributes $X$ ({raw_auc:.3f})", zorder=1)

    ax_a.set_xlabel("Training Epoch")
    ax_a.set_ylabel("OOF ROC-AUC (Interception Requirement)")
    ax_a.set_title("(a) H-A: Representation Discrimination in Backbone $h$", loc="left", fontweight="bold", fontsize=8.5)
    ax_a.set_ylim(0.52, 0.88)
    ax_a.set_xlim(-0.5, 10.5)
    ax_a.set_xticks([0, 1, 2, 3, 4, 5, 10])

    # Annotation box for H-A verdict
    ax_a.text(
        0.04, 0.12,
        r"$\mathbf{h}$ discrimination rises to $\mathbf{0.787}$" + "\n" + r"$\rightarrow$ $\mathbf{H\text{-}A\ Rejected}$ (No Repr. Loss)",
        transform=ax_a.transAxes,
        fontsize=7.0,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F4F8", edgecolor=c_blue, alpha=0.9),
        verticalalignment="bottom",
    )
    ax_a.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", framealpha=0.9, fontsize=6.8)

    # =======================================================================
    # Panel B: Hypothesis C — Gradient Coherence & Counterfactual Decomposition
    # =======================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    apply_clean_spines(ax_b)

    # Focus on R8 ep2 primary state
    df_g_ep2 = df_grad[df_grad["checkpoint"] == "r8_ep2_primary"].copy()
    procs = ["w_phen", "w_snow", "w_sub", "w_int"]
    proc_labels = ["Phenology\n($w_{\\mathrm{phen}}$)", "Snow\n($w_{\\mathrm{snow}}$)", "Subsurface\n($w_{\\mathrm{sub}}$)", "Interception\n($w_{\\mathrm{int}}$)"]

    x_pos = np.arange(len(procs))
    w = 0.24

    can_vals = []
    nobias_vals = []
    cent_vals = []

    for p in procs:
        row = df_g_ep2[df_g_ep2["process"] == p].iloc[0]
        can_vals.append(row["can_cos_pos_full"])
        nobias_vals.append(row["nobias_cos_pos_full"])
        cent_vals.append(row["cent_cos_pos_full"])

    b1 = ax_b.bar(x_pos - w, can_vals, width=w, label=r"Canonical ($\cos(G_{\mathrm{pos}}, G_{\mathrm{full}})$)", color=c_blue, alpha=0.9)
    b2 = ax_b.bar(x_pos, nobias_vals, width=w, label=r"No-Bias Decomp.", color=c_teal, alpha=0.9)
    b3 = ax_b.bar(x_pos + w, cent_vals, width=w, label=r"Centered / No-DC Decomp.", color=c_orange, alpha=0.9)

    ax_b.axhline(0, color=c_grey, linestyle="--", linewidth=0.8, zorder=1)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(proc_labels, fontsize=7.2)
    ax_b.set_ylabel(r"Gradient Alignment with Full Update $\cos(G_{\mathrm{pos}}, G_{\mathrm{full}})$")
    ax_b.set_title(r"(b) H-C: Gradient Alignment & Head Decomposition (ep2)", loc="left", fontweight="bold", fontsize=8.5)
    ax_b.set_ylim(-1.15, 1.25)

    # Highlight w_int negative collision
    ax_b.annotate(
        "Severe Opposing Collision\n" + r"$\cos \approx -0.83$ (Norm ratio $2.58\times$)" + "\n" + r"$\rightarrow \mathbf{H\text{-}C\ Confirmed\ (Primary)}$",
        xy=(3, -0.85),
        xytext=(2.6, -0.1),
        ha="center",
        fontsize=6.8,
        fontweight="bold",
        color=c_red,
        arrowprops=dict(arrowstyle="->", color=c_red, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF0F0", edgecolor=c_red, alpha=0.9),
    )
    ax_b.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", framealpha=0.9, fontsize=6.5)

    # =======================================================================
    # Panel C: Hypothesis B — Parameter Compensation Dynamics vs Finite Benefit
    # =======================================================================
    ax_c = fig.add_subplot(gs[1, 0])
    apply_clean_spines(ax_c)

    traj = comp_summary["trajectory_by_epoch"]
    eps = [int(k) for k in traj.keys()]
    w_med = [traj[str(k)]["learned_w_median"] for k in eps]
    dnse_med = [traj[str(k)]["dNSE_max_median"] for k in eps]
    frac_dnse_gt01 = [traj[str(k)]["frac_dNSE_gt001"] * 100 for k in eps]

    ax_c.plot(eps, w_med, color=c_red, marker="o", linewidth=1.5, label=r"Learned $w_{\mathrm{int}}$ Median", zorder=3)
    ax_c.set_xlabel("R8 Training Epoch")
    ax_c.set_ylabel(r"Learned Gate $w_{\mathrm{int}}$ Median", color=c_red)
    ax_c.tick_params(axis="y", labelcolor=c_red)
    ax_c.set_ylim(-0.05, 1.08)
    ax_c.set_xticks(eps)
    ax_c.set_title(r"(c) H-B: Gate Collapse vs Finite Benefit Retention (N=103)", loc="left", fontweight="bold", fontsize=8.5)

    # Twin axis for finite benefit
    ax_c_twin = ax_c.twinx()
    ax_c_twin.spines['top'].set_visible(False)
    ax_c_twin.spines['left'].set_visible(False)
    ax_c_twin.spines['right'].set_linewidth(0.8)
    ax_c_twin.plot(eps, dnse_med, color=c_blue, marker="s", linestyle="--", linewidth=1.4, label=r"Finite $\Delta\mathrm{NSE}_{\max}$ Median", zorder=3)
    ax_c_twin.set_ylabel(r"Finite $\Delta\mathrm{NSE}_{\max}$ Median", color=c_blue)
    ax_c_twin.tick_params(axis="y", labelcolor=c_blue)
    ax_c_twin.set_ylim(0.015, 0.065)

    # Annotation box for H-B
    ax_c.text(
        0.05, 0.22,
        r"Collapse ep2$\rightarrow$ep3 ($0.998 \rightarrow 0.005$)" + "\n" +
        r"Finite benefit remains ($83.5\%$ retain $\Delta\mathrm{NSE}>0.01$ at ep10)" + "\n" +
        r"$S_e$ shift ($r=+0.335, p=0.001$) $\rightarrow \mathbf{H\text{-}B\ Secondary}$",
        transform=ax_c.transAxes,
        fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor=c_grey, alpha=0.9),
        verticalalignment="bottom",
    )

    # Combined legend
    lines_c1, labels_c1 = ax_c.get_legend_handles_labels()
    lines_c2, labels_c2 = ax_c_twin.get_legend_handles_labels()
    ax_c.legend(lines_c1 + lines_c2, labels_c1 + labels_c2, loc="upper right", frameon=True, facecolor="#FFFFFF", fontsize=6.8)

    # =======================================================================
    # Panel D: Hypothesis D — Initialization & Cross-Head Alignment Trajectories
    # =======================================================================
    ax_d = fig.add_subplot(gs[1, 1])
    apply_clean_spines(ax_d)

    epochs_d = [1, 2, 3, 5, 10]
    cos_wint_full = []
    for ep in epochs_d:
        row = df_init[(df_init["epoch"] == ep) & (df_init["process"] == "w_int")].iloc[0]
        cos_wint_full.append(row["cos_pos_full"])

    ax_d.plot(epochs_d, cos_wint_full, color=c_red, marker="D", linewidth=1.6, label=r"$w_{\mathrm{int}}$ $\cos(G_{\mathrm{pos}}, G_{\mathrm{full}})$", zorder=4)

    # Cross-head alignments
    cross_snow = [init_json[f"ep{ep}"]["cross_process_alignment"]["w_int_vs_w_snow"] for ep in epochs_d]
    cross_sub = [init_json[f"ep{ep}"]["cross_process_alignment"]["w_int_vs_w_sub"] for ep in epochs_d]
    cross_phen = [init_json[f"ep{ep}"]["cross_process_alignment"]["w_int_vs_w_phen"] for ep in epochs_d]

    ax_d.plot(epochs_d, cross_snow, color=c_blue, marker="s", linestyle="--", linewidth=1.2, label=r"Cross-Head: $w_{\mathrm{int}}$ vs $w_{\mathrm{snow}}$", zorder=3)
    ax_d.plot(epochs_d, cross_sub, color=c_teal, marker="^", linestyle="-.", linewidth=1.2, label=r"Cross-Head: $w_{\mathrm{int}}$ vs $w_{\mathrm{sub}}$", zorder=3)
    ax_d.plot(epochs_d, cross_phen, color=c_orange, marker="o", linestyle=":", linewidth=1.2, label=r"Cross-Head: $w_{\mathrm{int}}$ vs $w_{\mathrm{phen}}$", zorder=3)

    ax_d.axhline(0, color=c_grey, linestyle="--", linewidth=0.8, zorder=1)
    ax_d.set_xlabel("R8 Training Epoch")
    ax_d.set_ylabel(r"Gradient Alignment Cosine")
    ax_d.set_title(r"(d) H-D: Dynamic Polarization & Cross-Head Fluctuation", loc="left", fontweight="bold", fontsize=8.5)
    ax_d.set_xticks(epochs_d)
    ax_d.set_ylim(-1.05, 1.15)

    # Annotation for H-D
    ax_d.text(
        0.04, 0.08,
        r"ep1 $\cos=+0.921$ excludes poor init" + "\n" +
        r"Cross-head alignment swings $\pm 0.6$" + "\n" +
        r"$\rightarrow \mathbf{H\text{-}D\ Secondary\ Amplifier}$",
        transform=ax_d.transAxes,
        fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFBEA", edgecolor="#E6AB02", alpha=0.9),
        verticalalignment="bottom",
    )
    ax_d.legend(loc="upper right", frameon=True, facecolor="#FFFFFF", framealpha=0.9, fontsize=6.5)

    # -----------------------------------------------------------------------
    # Save Figures
    # -----------------------------------------------------------------------
    png_path = out_dir / "figure_r13_root_cause_diagnostics.png"
    pdf_path = out_dir / "figure_r13_root_cause_diagnostics.pdf"

    plt.savefig(png_path, dpi=600)
    plt.savefig(pdf_path)
    plt.close()

    print(f"[R13 Figure Complete]\n  PNG: {png_path}\n  PDF: {pdf_path}")
    return png_path, pdf_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r13-dir", type=Path, default=DEFAULT_R13_DIR, help="Path to R13 root-cause results directory")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Path to output figures directory")
    args = parser.parse_args()

    generate_figure_r13(args.r13_dir, args.out_dir)
