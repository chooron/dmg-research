#!/usr/bin/env python3
"""Generate the 4 publication-quality audit figures for the 531-basin structural consistency audit:
1. J(w) optimum location distribution across lambda and processes
2. Soft regret R_soft vs Hard regret R_hard
3. Predicted positive rate vs counterfactual positive rate
4. ROC-AUC / PR-AUC / Spearman correlation across lambda
"""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
audit_dir = PROJECT_DIR / "results" / "formal_531_parallel" / "structural_consistency_audit"
fig_dir = audit_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# Set publication style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "figure.autolayout": True,
    "figure.dpi": 300,
})

# Load summary CSVs
df_sweep = pd.read_csv(audit_dir / "continuous_gate_objective_sweep_summary.csv")
df_cal = pd.read_csv(audit_dir / "structural_calibration_audit_by_process.csv")
df_4p = pd.read_csv(audit_dir / "structural_calibration_audit_four_process_summary.csv")

lambdas = sorted(df_sweep["lambda"].unique())
lambda_labels = [f"{l:.3f}" if l < 0.01 else f"{l:.3f}".rstrip("0") if f"{l:.3f}".endswith("0") else f"{l:.3f}" for l in lambdas]
lambda_labels = [f"{l:g}" for l in lambdas]

proc_names = ["w_phen", "w_int", "w_snow", "w_sub"]
proc_titles = {
    "w_phen": "Vegetation Phenology (w_phen)",
    "w_int": "Canopy Interception (w_int)",
    "w_snow": "Snow Accum/Melt (w_snow)",
    "w_sub": "Subsurface Baseflow (w_sub)"
}
proc_colors = {
    "w_phen": "#2ca02c",  # Green
    "w_int": "#1f77b4",   # Blue
    "w_snow": "#9467bd",  # Purple
    "w_sub": "#d62728"    # Red
}

# =============================================================================
# FIGURE 1: Optimum Location Distribution across lambda and processes
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
axes = axes.flatten()

for idx, p in enumerate(proc_names):
    ax = axes[idx]
    sub = df_sweep[df_sweep["process"] == p].sort_values("lambda")
    
    x = np.arange(len(lambdas))
    w0 = sub["frac_opt_w0"].values
    w_int_loc = sub["frac_opt_interior"].values
    w1 = sub["frac_opt_w1"].values
    
    ax.bar(x, w0, label="Optimum w* = 0.0 (OFF)", color="#4575b4", alpha=0.85, width=0.6)
    ax.bar(x, w_int_loc, bottom=w0, label="Interior Optimum w* ∈ (0, 1)", color="#fdae61", alpha=0.85, width=0.6)
    ax.bar(x, w1, bottom=w0+w_int_loc, label="Optimum w* = 1.0 (ON)", color="#d73027", alpha=0.85, width=0.6)
    
    ax.set_title(proc_titles[p], fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Basin Fraction (%)" if idx in [0, 2] else "")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l:g}" for l in sub["lambda"]], rotation=30)
    if idx >= 2:
        ax.set_xlabel("Complexity Regularization $\lambda$ (aic_alpha)")
    if idx == 0:
        ax.legend(loc="upper right", framealpha=0.9)

plt.suptitle("Figure 1: Objective J(w) Optimum Location Distribution Across $\lambda$ (531 Basins)", fontsize=13, fontweight="bold", y=1.02)
fig1_path = fig_dir / "fig1_optimum_location_by_lambda_process.png"
plt.savefig(fig1_path, bbox_inches="tight")
plt.close()
print(f"Saved: {fig1_path}")

# =============================================================================
# FIGURE 2: Soft Regret R_soft vs Hard Regret R_hard
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
axes = axes.flatten()

for idx, p in enumerate(proc_names):
    ax = axes[idx]
    sub = df_sweep[df_sweep["process"] == p].sort_values("lambda")
    
    l_vals = sub["lambda"].values
    r_soft = sub["mean_R_soft"].values
    r_hard = sub["mean_R_hard"].values
    pct_soft_better = sub["frac_R_soft_lt_R_hard"].values
    
    ax.plot(l_vals, r_soft, marker="o", color="#1b9e77", label="Mean $R_{\\rm soft} = J(p) - \\min_w J(w)$", linewidth=2.0)
    ax.plot(l_vals, r_hard, marker="s", color="#d95f02", label="Mean $R_{\\rm hard} = J(\\mathbf{1}[p>0.5]) - \\min_w J(w)$", linewidth=2.0)
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(proc_titles[p], fontweight="bold")
    ax.set_xlabel("Complexity Regularization $\lambda$ (aic_alpha)")
    ax.set_ylabel("Objective Regret (loss scale)")
    
    # Highlight interior/continuous advantage
    if idx == 0 or idx == 3: # phenology and subsurface
        ax.text(0.05, 0.15, f"Soft < Hard in {np.mean(pct_soft_better):.0f}% basins\n(Continuous supported)",
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", fc="#e6f5d0", ec="#4dac26", alpha=0.8), fontsize=8.5)
    elif idx == 1:
        ax.text(0.05, 0.75, "Hard ≈ Soft at high $\lambda$\n(Sparsity dominated)",
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#999999", alpha=0.8), fontsize=8.5)
        
    ax.legend(loc="upper left" if idx != 1 else "lower right", framealpha=0.9)

plt.suptitle("Figure 2: Objective Regret Comparison: Soft Gate $R_{\\rm soft}$ vs Hard Gate $R_{\\rm hard}$", fontsize=13, fontweight="bold", y=1.02)
fig2_path = fig_dir / "fig2_regret_comparison_rsoft_vs_rhard.png"
plt.savefig(fig2_path, bbox_inches="tight")
plt.close()
print(f"Saved: {fig2_path}")

# =============================================================================
# FIGURE 3: Predicted Positive Rate vs Counterfactual Positive Rate
# =============================================================================
fig, ax = plt.subplots(figsize=(8.5, 6))

markers = ["o", "s", "^", "D"]
for idx, p in enumerate(proc_names):
    sub = df_cal[df_cal["process"] == p].sort_values("lambda")
    l_vals = sub["lambda"].values
    net_pos = sub["net_pos_rate_pct"].values
    cf_pos = sub["cf_pos_rate_pct"].values
    
    ax.plot(l_vals, net_pos, marker=markers[idx], color=proc_colors[p], linestyle="-", label=f"{p}: Network $P(p > 0.5)$")
    ax.plot(l_vals, cf_pos, marker=markers[idx], color=proc_colors[p], linestyle="--", alpha=0.6, label=f"{p}: Counterfactual $P(\\Delta J > 0)$")

ax.set_xscale("log")
ax.set_ylim(-2, 102)
ax.set_xlabel("Complexity Regularization $\lambda$ (aic_alpha)", fontsize=11)
ax.set_ylabel("Positive Activation Rate (%)", fontsize=11)
ax.set_title("Figure 3: Structural Network Fidelity: Predicted $P(p > 0.5)$ vs Ground-Truth $P(\\Delta J > 0)$", fontsize=12.5, fontweight="bold")
ax.legend(loc="center right", bbox_to_anchor=(1.45, 0.5), framealpha=0.9, fontsize=8.5)

fig3_path = fig_dir / "fig3_predicted_vs_counterfactual_positive_rate.png"
plt.savefig(fig3_path, bbox_inches="tight")
plt.close()
print(f"Saved: {fig3_path}")

# =============================================================================
# FIGURE 4: ROC-AUC / PR-AUC / Spearman across lambda
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

# 4A: 4-Process Overall Metrics
l_vals_4p = df_4p["lambda"].values
roc_4p = df_4p["overall_roc_auc"].values
pr_4p = df_4p["overall_pr_auc"].values
sp_4p = df_4p["overall_spearman_rho"].values
f1_4p = df_4p["overall_f1"].values

ax1.plot(l_vals_4p, roc_4p, marker="o", color="#e41a1c", label="Overall ROC-AUC", linewidth=2.0)
ax1.plot(l_vals_4p, pr_4p, marker="s", color="#377eb8", label="Overall PR-AUC", linewidth=2.0)
ax1.plot(l_vals_4p, sp_4p, marker="^", color="#4daf4a", label="Overall Spearman $\\rho(p, \\Delta J)$", linewidth=2.0)
ax1.plot(l_vals_4p, f1_4p, marker="D", color="#984ea3", label="Overall F1-Score", linewidth=2.0)

ax1.set_xscale("log")
ax1.set_ylim(0.85, 1.005)
ax1.set_xlabel("Complexity Regularization $\lambda$ (aic_alpha)")
ax1.set_ylabel("Calibration / Fidelity Metric")
ax1.set_title("(A) Overall 4-Process Structural Fidelity", fontweight="bold")
ax1.legend(loc="lower right", framealpha=0.9)

# 4B: Per-process Spearman correlation with DeltaJ
for idx, p in enumerate(proc_names):
    sub = df_cal[df_cal["process"] == p].sort_values("lambda")
    ax2.plot(sub["lambda"].values, sub["spearman_rho"].values, marker=markers[idx], color=proc_colors[p], label=p, linewidth=2.0)

ax2.set_xscale("log")
ax2.set_ylim(0.75, 1.005)
ax2.set_xlabel("Complexity Regularization $\lambda$ (aic_alpha)")
ax2.set_ylabel("Spearman Rank Correlation $\\rho(p, \\Delta J)$")
ax2.set_title("(B) Process-Wise Rank Fidelity $\\rho(p, \\Delta J)$", fontweight="bold")
ax2.legend(loc="lower right", framealpha=0.9)

plt.suptitle("Figure 4: Structural Network Calibration Curves Across Regularization Spectrum $\lambda$", fontsize=13, fontweight="bold", y=1.02)
fig4_path = fig_dir / "fig4_calibration_roc_pr_spearman_curves.png"
plt.savefig(fig4_path, bbox_inches="tight")
plt.close()
print(f"Saved: {fig4_path}")
