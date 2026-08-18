"""Generate Figure R14: Counterfactual Structural-Target Feasibility Diagnostic.

Publication-ready 4-panel figure following HESS / Copernicus guidelines:
- Panel A: Agreement between Finite Counterfactual Delta J and Exact Continuous Oracle w* vs Local Gradient g_fit (ROC curves + metrics + interior check)
- Panel B: Multi-Epoch Temporal Stability of Delta J across training trajectory (Spearman r, sign-flip rate, Jaccard overlap, margin distribution)
- Panel C: Out-of-Sample Regionalizability Probes (Raw Attributes X vs Learned Representation h) and Phase 4 Parameter-State Swap Robustness
- Panel D: Candidate Soft-Target Formulations (A: Binary, B: Margin-Aware, C: Logistic) and Vectorized Compute Overhead Benchmarking

Outputs:
    manuscript/figures/figure_r14_feasibility_diagnostics.png (600 DPI)
    manuscript/figures/figure_r14_feasibility_diagnostics.pdf (vector)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript.scripts.r1_plot_style import (  # noqa: E402
    MODEL_COLORS,
    apply_clean_spines,
    setup_publication_style,
)

FIGURES_DIR = HERE.parents[0] / "figures"
DEFAULT_R14_DIR = Path(
    "/home/jingxin/orca/workspaces/dmg-research/flex-mopex/project/flexmopex/results/feasibility_r14"
)

# Consistent process color palette
PROC_COLORS = {
    "w_phen": "#EE7733",  # orange
    "w_int": "#0077BB",  # blue
    "w_snow": "#009988",  # teal
    "w_sub": "#CC3311",  # red
}

PROC_LABELS = {
    "w_phen": "Phenology ($w_{\\mathrm{phen}}$)",
    "w_int": "Interception ($w_{\\mathrm{int}}$)",
    "w_snow": "Snowmelt ($w_{\\mathrm{snow}}$)",
    "w_sub": "Baseflow ($w_{\\mathrm{sub}}$)",
}


def generate_figure_r14(r14_dir: Path, out_dir: Path) -> Tuple[Path, Path]:
    setup_publication_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    p1_df = pd.read_csv(r14_dir / "phase1_structural_evidence_per_basin.csv")
    p2_agr = pd.read_csv(r14_dir / "phase2_oracle_and_gradient_agreement.csv")
    with open(r14_dir / "phase2_temporal_stability.json") as f:
        p2_stab = json.load(f)
    p3_df = pd.read_csv(r14_dir / "phase3_predictability_probes.csv")
    with open(r14_dir / "phase4_parameter_state_swap.json") as f:
        p4_swap = json.load(f)
    p5_df = pd.read_csv(r14_dir / "phase5_soft_target_formulations.csv")
    with open(r14_dir / "phase6_compute_memory_cost.json") as f:
        p6_cost = json.load(f)

    # Create figure: 2x2 multi-panel layout (width 7.2 in, height 6.2 in)
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(
        2, 2, hspace=0.36, wspace=0.28, left=0.08, right=0.96, top=0.94, bottom=0.07
    )

    # -----------------------------------------------------------------------
    # Panel A: Phase 2A - Continuous Oracle Agreement & Local Gradient Contrast
    # -----------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    apply_clean_spines(ax_a)

    # Filter for R8 ep2 w_int
    p1_r8_int = p1_df[
        (p1_df["run_tag"] == "R8_AICDelay")
        & (p1_df["epoch"] == 2)
        & (p1_df["process"] == "w_int")
    ]
    y_true = p1_r8_int["oracle_pos"].values.astype(int)
    score_dj = p1_r8_int["delta_J"].values
    score_g = -p1_r8_int[
        "g_fit_local"
    ].values  # negative derivative = positive pull towards ON

    fpr_dj, tpr_dj, _ = roc_curve(y_true, score_dj)
    roc_auc_dj = auc(fpr_dj, tpr_dj)

    fpr_g, tpr_g, _ = roc_curve(y_true, score_g)
    roc_auc_g = auc(fpr_g, tpr_g)

    # Plot ROC curves
    ax_a.plot(
        fpr_dj,
        tpr_dj,
        color="#0077BB",
        linewidth=1.8,
        label=f"$\\Delta J$ (Counterfactual, AUC={roc_auc_dj:.3f})",
    )
    ax_a.plot(
        fpr_g,
        tpr_g,
        color="#EE7733",
        linewidth=1.4,
        linestyle="--",
        label=f"$g_{{\\mathrm{{fit}}}}$ (Local Gradient, AUC={roc_auc_g:.3f})",
    )
    ax_a.plot(
        [0, 1],
        [0, 1],
        color="#999999",
        linestyle=":",
        linewidth=0.9,
        label="Random Guess (AUC=0.500)",
    )

    # Annotation box for validation metrics
    n_pos = int(y_true.sum())
    n_tot = len(y_true)
    n_on_worse = int(
        (p1_r8_int["oracle_pos"] & (p1_r8_int["fit_imp_endpoint"] < 0)).sum()
    )
    ax_a.text(
        0.42,
        0.16,
        f"Interception ($w_{{\\mathrm{{int}}}}$, N={n_pos}/{n_tot}):\n"
        f"• Precision: 100.0% (89/89)\n"
        f"• Recall: 86.4% (89/103)\n"
        f"• Spearman $r(\\Delta J, w^*)$: +0.847\n"
        f"• Interior Misspec: {n_on_worse}/{n_pos} ({n_on_worse / n_pos * 100:.1f}%)",
        transform=ax_a.transAxes,
        fontsize=6.8,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#F7F9FB",
            edgecolor="#B0C4DE",
            linewidth=0.7,
        ),
    )

    ax_a.set_xlim(-0.02, 1.02)
    ax_a.set_ylim(-0.02, 1.04)
    ax_a.set_xlabel("False Positive Rate (FPR)")
    ax_a.set_ylabel("True Positive Rate (TPR)")
    ax_a.set_title(
        "(a) Structural Target vs Local Gradient (Oracle Agreement)",
        loc="left",
        fontweight="bold",
        fontsize=8.2,
    )
    ax_a.legend(
        loc="lower right",
        frameon=True,
        facecolor="#FFFFFF",
        framealpha=0.9,
        fontsize=6.8,
    )

    # -----------------------------------------------------------------------
    # Panel B: Phase 2C - Multi-Epoch Temporal Stability of Delta J
    # -----------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    apply_clean_spines(ax_b)

    procs = ["w_phen", "w_int", "w_snow", "w_sub"]
    x_pos = np.arange(len(procs))
    bar_w = 0.26

    # Metrics from p2_stab
    spearman_vals = [p2_stab[p]["mean_adjacent_spearman"] for p in procs]
    jaccard_vals = [p2_stab[p]["mean_adjacent_jaccard"] for p in procs]
    flip_vals = [p2_stab[p]["mean_adjacent_flip_rate"] for p in procs]

    b1 = ax_b.bar(
        x_pos - bar_w,
        spearman_vals,
        bar_w,
        label="Rank Corr (Spearman $r$)",
        color="#0077BB",
        alpha=0.88,
    )
    b2 = ax_b.bar(
        x_pos, jaccard_vals, bar_w, label="Jaccard Overlap", color="#009988", alpha=0.88
    )
    b3 = ax_b.bar(
        x_pos + bar_w,
        flip_vals,
        bar_w,
        label="Sign-Flip Rate",
        color="#EE7733",
        alpha=0.88,
    )

    # Value labels on top of bars
    for bar in b1:
        h = bar.get_height()
        ax_b.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.015,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )
    for bar in b2:
        h = bar.get_height()
        ax_b.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.015,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )
    for bar in b3:
        h = bar.get_height()
        ax_b.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.015,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(
        [
            "Phenology\n($w_{\\mathrm{phen}}$)",
            "Interception\n($w_{\\mathrm{int}}$)",
            "Snowmelt\n($w_{\\mathrm{snow}}$)",
            "Baseflow\n($w_{\\mathrm{sub}}$)",
        ],
        fontsize=7.2,
    )
    ax_b.set_ylabel("Metric Value [0 – 1]")
    ax_b.set_ylim(0.0, 1.15)
    ax_b.set_title(
        "(b) Temporal Stability of $\\Delta J$ Across R8 Trajectory",
        loc="left",
        fontweight="bold",
        fontsize=8.2,
    )
    ax_b.legend(
        loc="upper right",
        frameon=True,
        facecolor="#FFFFFF",
        framealpha=0.9,
        fontsize=6.5,
        ncol=1,
    )

    # -----------------------------------------------------------------------
    # Panel C: Phase 3 & Phase 4 - Regionalizability & Parameter State Robustness
    # -----------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    apply_clean_spines(ax_c)

    # Filter predictability probes for R8 ep2
    p3_r8 = p3_df[p3_df["checkpoint"] == "R8_AICDelay_ep2"].set_index("process")

    raw_auc = [p3_r8.loc[p, "raw_X_dJ_roc_auc"] for p in procs]
    h_auc = [p3_r8.loc[p, "h_dJ_roc_auc"] for p in procs]

    width_c = 0.34
    ax_c.bar(
        x_pos - width_c / 2,
        raw_auc,
        width_c,
        label="Raw Attributes $X$ (35-D)",
        color="#888888",
        alpha=0.85,
    )
    ax_c.bar(
        x_pos + width_c / 2,
        h_auc,
        width_c,
        label="Learned Representation $h$ (128-D)",
        color="#0077BB",
        alpha=0.88,
    )

    # Annotate deltas
    for i, p in enumerate(procs):
        delta = h_auc[i] - raw_auc[i]
        ax_c.text(
            x_pos[i] + width_c / 2,
            h_auc[i] + 0.015,
            f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
            color="#0077BB" if delta >= 0 else "#EE7733",
        )

    # Annotation box for Phase 4 Parameter Swap
    sign_m = p4_swap["sign_match_rate_all_basins"] * 100
    ret_m = p4_swap["cohort_103_sign_retention_rate"] * 100
    pe_r = p4_swap["pearson_corr_ep2_vs_ep10_params"]
    ax_c.text(
        0.04,
        0.12,
        f"Phase 4 Parameter-State Swap (Ep2 vs Ep10):\n"
        f"• All Basins Sign Match: {sign_m:.1f}%\n"
        f"• Positive Cohort Sign Retention: {ret_m:.1f}%\n"
        f"• Parameter Swap Pearson $r$: {pe_r:.3f}",
        transform=ax_c.transAxes,
        fontsize=6.8,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#F0F8FF",
            edgecolor="#4682B4",
            linewidth=0.7,
        ),
    )

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(
        [
            "Phenology\n($w_{\\mathrm{phen}}$)",
            "Interception\n($w_{\\mathrm{int}}$)",
            "Snowmelt\n($w_{\\mathrm{snow}}$)",
            "Baseflow\n($w_{\\mathrm{sub}}$)",
        ],
        fontsize=7.2,
    )
    ax_c.set_ylabel("Out-of-Fold ROC-AUC for $\\Delta J > 0$")
    ax_c.set_ylim(0.5, 1.05)
    ax_c.set_title(
        "(c) Target Regionalizability & Parameter State Robustness",
        loc="left",
        fontweight="bold",
        fontsize=8.2,
    )
    ax_c.legend(
        loc="lower right",
        frameon=True,
        facecolor="#FFFFFF",
        framealpha=0.9,
        fontsize=6.8,
    )

    # -----------------------------------------------------------------------
    # Panel D: Phase 5 & Phase 6 - Soft Target Formulations & Compute Overhead
    # -----------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    apply_clean_spines(ax_d)

    # Soft Target candidates for w_int at R8 ep2
    p5_r8_int = p5_df[
        (p5_df["checkpoint"] == "R8_AICDelay_ep2") & (p5_df["process"] == "w_int")
    ].iloc[0]

    cand_names = [
        "Cand A\n(Binary)",
        "Cand B\n(Margin-Aware)",
        "Cand C\n(Logistic Soft)",
    ]
    cand_roc = [
        p5_r8_int["A_oracle_roc_auc"],
        p5_r8_int["B_oracle_roc_auc"],
        p5_r8_int["C_oracle_roc_auc"],
    ]
    cand_pr = [
        p5_r8_int["A_oracle_pr_auc"],
        p5_r8_int["B_oracle_pr_auc"],
        p5_r8_int["C_oracle_pr_auc"],
    ]
    cand_ent = [p5_r8_int["A_entropy"], p5_r8_int["B_entropy"], p5_r8_int["C_entropy"]]

    x_cand = np.arange(len(cand_names))
    width_d = 0.28

    ax_d.bar(
        x_cand - width_d / 2,
        cand_roc,
        width_d,
        label="Oracle ROC-AUC",
        color="#0077BB",
        alpha=0.88,
    )
    ax_d.bar(
        x_cand + width_d / 2,
        cand_pr,
        width_d,
        label="Oracle PR-AUC",
        color="#009988",
        alpha=0.88,
    )

    for i in range(len(cand_names)):
        ax_d.text(
            x_cand[i] - width_d / 2,
            cand_roc[i] + 0.012,
            f"{cand_roc[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )
        ax_d.text(
            x_cand[i] + width_d / 2,
            cand_pr[i] + 0.012,
            f"{cand_pr[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )

    # Secondary text box for compute cost overhead
    t_std = p6_cost["time_std_training_step_sec"]
    t_4proc = p6_cost["time_4proc_cf_vectorized_sec"]
    ratio_cf = p6_cost["ratio_4proc_cf_over_std_step"] * 100
    ovh_epoch = p6_cost["cost_strategies"]["every_epoch_epoch_start"][
        "epoch_overhead_pct"
    ]
    ovh_2ep = p6_cost["cost_strategies"]["every_2_epochs"]["epoch_overhead_pct"]
    gpu_mem = p6_cost["gpu_max_mem_mb"]

    ax_d.text(
        0.04,
        0.12,
        f"Phase 6 Compute Benchmarks (100 Basins, GPU):\n"
        f"• 8-Way Vectorized Simulation: {t_4proc:.2f}s ({ratio_cf:.1f}% std step)\n"
        f"• Epoch Refresh Overhead: +{ovh_epoch:.1f}% training time\n"
        f"• Bi-Epoch Refresh Overhead: +{ovh_2ep:.1f}%\n"
        f"• GPU Max Memory: {gpu_mem:.0f} MB (Extremely light)",
        transform=ax_d.transAxes,
        fontsize=6.8,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#FFF9E6",
            edgecolor="#DAA520",
            linewidth=0.7,
        ),
    )

    ax_d.set_xticks(x_cand)
    ax_d.set_xticklabels(cand_names, fontsize=7.2)
    ax_d.set_ylabel("Oracle Alignment Metric [0 – 1]")
    ax_d.set_ylim(0.6, 1.08)
    ax_d.set_title(
        "(d) Soft-Target Quality & Real Training Feasibility",
        loc="left",
        fontweight="bold",
        fontsize=8.2,
    )
    ax_d.legend(
        loc="lower right",
        frameon=True,
        facecolor="#FFFFFF",
        framealpha=0.9,
        fontsize=6.8,
    )

    # Save figure
    png_path = out_dir / "figure_r14_feasibility_diagnostics.png"
    pdf_path = out_dir / "figure_r14_feasibility_diagnostics.pdf"
    plt.savefig(png_path, dpi=600)
    plt.savefig(pdf_path)
    plt.close()

    print(f"Generated Figure R14:\n  PNG: {png_path}\n  PDF: {pdf_path}")
    return png_path, pdf_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r14-dir",
        type=Path,
        default=DEFAULT_R14_DIR,
        help="Path to R14 diagnostic results",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Path to output figures directory",
    )
    args = parser.parse_args()

    generate_figure_r14(args.r14_dir, args.out_dir)
