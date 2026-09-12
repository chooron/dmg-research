#!/usr/bin/env python3
"""Plot Figure 7 for Journal of Hydrology manuscript R3.

Scientific Question:
"How strong is cross-paradigm coordinate specificity relative to within-IC calibration
repeatability, and does that specificity extend to broad hydrological functional roles?"

Layout:
- Panel (a) HERO (Left Column, ~55% width): Model-level paired comparison for coordinate
  specificity A_diag (n=34 models, ordered by parameter count P_m, comparing IC-self vs matched cross).
- Panel (b) (Top Right): Profile-level correlation calibration reference R_paired (n=35 models).
- Panel (c) (Middle Right): Calibration-scale reference sensitivity across model subsets (All vs Strict R2).
- Panel (d) (Bottom Right): Hydrological functional-role interpretation boundary (Hierarchy, Permutation Null,
  and Broad Process Taxonomy).
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
from matplotlib.lines import Line2D

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
R3_DIR = SCRIPT_DIR.parent
TABLES_DIR = R3_DIR / "tables"
ROLE_TABLES_DIR = R3_DIR.parents[1] / "results/joh_functional_role_diagnostic_20260905/tables"
OUTPUT_PNG = R3_DIR / "F7_main.png"

# -------------------------------------------------------------------------
# Matplotlib Configuration (JoH Publication Quality)
# -------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 9.0,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.titlesize": 11.0,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# -------------------------------------------------------------------------
# Color Palette (Consistent with Manuscript R2 & R3 Standards)
# -------------------------------------------------------------------------
COLOR_IC_SELF = "#1f4e79"        # Deep Navy / Slate Blue (Within-IC Repeatability)
COLOR_CROSS = "#d95f02"          # Warm Terracotta / Amber (Cross-Paradigm IC-dPL)
COLOR_DELTA_POS = "#2a7b4c"      # Green (Positive Repeatability Advantage)
COLOR_DELTA_NEG = "#c53030"      # Coral Red (Negative Difference)
COLOR_NEUTRAL = "#718096"        # Slate Grey (Baseline / Secondary)
COLOR_LIGHT_BG = "#f8f9fa"       # Pale Neutral Background
COLOR_BOX_BORDER = "#cbd5e1"     # Border Grey
COLOR_ROLE_SAME = "#8c564b"      # Muted Brown (Same Role Off-diagonal)
COLOR_ROLE_CROSS = "#64748b"     # Mid Slate Grey (Cross Role Off-diagonal)
COLOR_ROLE_NULL = "#94a3b8"      # Light Neutral (Permutation Null)
GRID_COLOR = "#edf0f2"

# -------------------------------------------------------------------------
# Load Data
# -------------------------------------------------------------------------
def load_data():
    df_model = pd.read_csv(TABLES_DIR / "R3_F6_IC_SELF_REFERENCE_MODEL.csv")
    df_sum = pd.read_csv(TABLES_DIR / "R3_F6_IC_SELF_REFERENCE_SUMMARY.csv")
    df_role_mod = pd.read_csv(ROLE_TABLES_DIR / "13_ROLE_OFFDIAGONAL_ADVANTAGE_BY_MODEL.csv")
    df_role_perm = pd.read_csv(ROLE_TABLES_DIR / "15_ROLE_LABEL_PERMUTATION.csv")
    df_role_reg = pd.read_csv(ROLE_TABLES_DIR / "00_PARAMETER_FUNCTIONAL_ROLE_REGISTRY.csv")
    return df_model, df_sum, df_role_mod, df_role_perm, df_role_reg

# -------------------------------------------------------------------------
# Helper: Clean Model Names
# -------------------------------------------------------------------------
def clean_model_name(name: str) -> str:
    mapping = {
        "gr4j": "GR4J",
        "simhyd": "SIMHYD",
        "hbv96": "HBV-96",
        "ihacres": "IHACRES",
        "modhydrolog": "MODHYDROLOG",
        "xinanjiang": "Xinanjiang",
        "topmodel": "TOPMODEL",
        "hymod": "HYMOD",
        "gsfb": "GSFB",
        "vic": "VIC",
        "penman": "Penman",
        "tank": "Tank",
        "smar": "SMAR",
        "tcm": "TCM",
        "wetland": "Wetland",
        "plateau": "Plateau",
        "hillslope": "Hillslope",
        "australia": "Australia",
    }
    if name in mapping:
        return mapping[name]
    if name.startswith("mopex"):
        return f"MOPEX-{name[5:]}"
    if name.startswith("collie"):
        return f"Collie-{name[6:]}"
    if name.startswith("alpine"):
        return f"Alpine-{name[6:]}"
    if name.startswith("susannah"):
        return f"Susannah-{name[8:]}"
    if name.startswith("newzealand"):
        return f"NewZealand-{name[10:]}"
    if name.startswith("flex"):
        return f"FLEX-{name[4:].upper()}"
    return name.upper()


def main():
    df_model, df_sum, df_role_mod, df_role_perm, df_role_reg = load_data()
    
    # ---------------------------------------------------------------------
    # Prepare Data for Panel (a) - HERO: Model-level A_diag Paired Reference
    # ---------------------------------------------------------------------
    df_a = df_model[df_model["a_profile_estimable"] == True].copy()
    df_a = df_a.sort_values(by=["parameter_count", "model"], ascending=[True, True]).reset_index(drop=True)
    n_a = len(df_a)  # 34 models
    
    sum_a_all = df_sum[(df_sum["population"] == "all_model_available") & (df_sum["estimand"] == "A_diag_IC_self")].iloc[0]
    ic_self_a_med = sum_a_all["self_value"]          # 0.981203
    cross_a_med = sum_a_all["cross_value"]              # 0.603759
    delta_a_med = sum_a_all["difference_self_minus_cross"] # 0.339474
    ci_low_a = sum_a_all["ci_low"]                      # 0.202256
    ci_high_a = sum_a_all["ci_high"]                    # 0.428571
    n_pos_a = int(sum_a_all["n_positive"])              # 31
    
    # ---------------------------------------------------------------------
    # Prepare Data for Panel (b) - Profile-level R_paired Reference
    # ---------------------------------------------------------------------
    df_r = df_model[df_model["r_profile_estimable"] == True].copy()
    df_r = df_r.sort_values(by=["parameter_count", "model"], ascending=[True, True]).reset_index(drop=True)
    n_r = len(df_r)  # 35 models
    
    sum_r_all = df_sum[(df_sum["population"] == "all_model_available") & (df_sum["estimand"] == "R_paired_IC_self")].iloc[0]
    ic_self_r_med = sum_r_all["self_value"]          # 0.967669
    cross_r_med = sum_r_all["cross_value"]              # 0.731579
    delta_r_med = sum_r_all["difference_self_minus_cross"] # 0.217293
    ci_low_r = sum_r_all["ci_low"]                      # 0.156391
    ci_high_r = sum_r_all["ci_high"]                    # 0.284211
    n_pos_r = int(sum_r_all["n_positive"])              # 35
    
    # ---------------------------------------------------------------------
    # Prepare Data for Panel (c) - Subset Sensitivity
    # ---------------------------------------------------------------------
    sum_r_strict = df_sum[(df_sum["population"] == "r2_strict_23") & (df_sum["estimand"] == "R_paired_IC_self")].iloc[0]
    sum_a_strict = df_sum[(df_sum["population"] == "r2_strict_23") & (df_sum["estimand"] == "A_diag_IC_self")].iloc[0]
    
    # ---------------------------------------------------------------------
    # Prepare Data for Panel (d) - Functional Role Boundary
    # ---------------------------------------------------------------------
    role_perm_high = df_role_perm[(df_role_perm["model"] == "ALL_MODELS") & (df_role_perm["confidence_filter"] == "high")].iloc[0]
    a_role_obs = role_perm_high["observed_A_role"]      # -0.118797
    a_role_p = role_perm_high["empirical_p_ge_observed"] # 0.805719
    a_role_null_mean = role_perm_high["null_mean"]       # -0.043050
    a_role_null_ci_low = role_perm_high["null_ci_low"]   # -0.233835
    a_role_null_ci_high = role_perm_high["null_ci_high"] # 0.133835
    
    role_high_mod = df_role_mod[df_role_mod["confidence_filter"] == "high"].copy()
    valid_role_mods = role_high_mod["role_offdiag_advantage_median"].dropna()
    n_pos_role = (valid_role_mods > 0).sum()             # 12
    same_role_med = role_high_mod["same_role_offdiag_median"].dropna().median() # -0.027820
    cross_role_med = role_high_mod["cross_role_median"].dropna().median()       # +0.005263

    # ---------------------------------------------------------------------
    # Create Figure & GridSpec Layout
    # ---------------------------------------------------------------------
    fig = plt.figure(figsize=(11.8, 9.6), dpi=600)
    
    gs_outer = gridspec.GridSpec(1, 2, width_ratios=[1.18, 1.0], wspace=0.28, left=0.065, right=0.975, top=0.95, bottom=0.05)
    
    # Left: Panel (a) HERO
    ax_a = fig.add_subplot(gs_outer[0, 0])
    
    # Right: 3 stacked sub-panels for (b), (c), (d)
    gs_right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_outer[0, 1], height_ratios=[0.95, 0.8, 1.55], hspace=0.38)
    ax_b = fig.add_subplot(gs_right[0, 0])
    ax_c = fig.add_subplot(gs_right[1, 0])
    ax_d = fig.add_subplot(gs_right[2, 0])
    
    # =========================================================================
    # PANEL (a) HERO: Model-Level Paired Coordinate Specificity (A_diag)
    # =========================================================================
    ax_a.set_title(
        r"$\mathbf{a}$  Coordinate Specificity vs. Within-IC Repeatability Reference ($A_{\mathrm{diag}}$, $n=34$ models)",
        loc="left", fontsize=9.2, fontweight="bold", pad=8
    )
    
    y_pos = np.arange(n_a)
    
    # Zebra striping
    for i in range(n_a):
        if i % 2 == 1:
            ax_a.axhspan(i - 0.45, i + 0.45, color="#f8fafc", zorder=0)
    
    ax_a.grid(axis="x", color=GRID_COLOR, linestyle="-", linewidth=0.6, zorder=1)
    ax_a.axvline(0.0, color="#cbd5e1", linestyle="-", linewidth=0.8, zorder=2)
    
    # Ensemble Medians
    ax_a.axvline(cross_a_med, color=COLOR_CROSS, linestyle="--", linewidth=1.1, alpha=0.9, zorder=2)
    ax_a.axvline(ic_self_a_med, color=COLOR_IC_SELF, linestyle="--", linewidth=1.1, alpha=0.9, zorder=2)
    
    # Draw paired dumbbells
    for i, row in df_a.iterrows():
        y = i
        x_cross = row["A_m_cross_matched"]
        x_self = row["A_m_IC_self"]
        delta = row["delta_A_self_minus_cross"]
        
        line_color = COLOR_DELTA_POS if delta >= 0 else COLOR_DELTA_NEG
        line_lw = 1.35 if delta >= 0 else 1.15
        ax_a.plot([x_cross, x_self], [y, y], color=line_color, linewidth=line_lw, alpha=0.75, zorder=3)
        
        ax_a.scatter(x_cross, y, color=COLOR_CROSS, s=24, edgecolors="white", linewidth=0.4, zorder=5)
        ax_a.scatter(x_self, y, color=COLOR_IC_SELF, s=26, edgecolors="white", linewidth=0.4, zorder=6)
        
        delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        txt_color = "#1b6338" if delta > 0 else "#991b1b"
        ax_a.text(1.68, y, delta_str, va="center", ha="right", fontsize=6.8, color=txt_color, fontfamily="monospace")
    
    ax_a.set_yticks(y_pos)
    yticklabels = [f"{clean_model_name(r['model'])} (P={r['parameter_count']})" for _, r in df_a.iterrows()]
    ax_a.set_yticklabels(yticklabels, fontsize=7.2)
    
    # Set explicit Y limits to allow room at top and dedicated space at bottom for summary card
    ax_a.set_ylim(n_a + 4.2, -1.8)
    ax_a.set_xlabel(r"Coordinate specificity statistic, $A_m$ (matched common basins)", fontsize=8.5, labelpad=4)
    ax_a.set_xlim(-0.15, 1.74)
    
    ax_a.text(1.68, -0.9, r"$\Delta_{\mathrm{self}-\mathrm{cross}}$", va="bottom", ha="right", fontsize=7.5, fontweight="bold", color="#334155")
    
    summary_text = (
        r"$\mathbf{Within-IC\ Repeatability\ Reference\ Gap\ (n=34):}$" + "\n"
        rf"$\bullet$ IC-self repeatability median: $A_{{\mathrm{{diag}}}}^{{\mathrm{{self}}}} = {ic_self_a_med:.3f}$" + "\n"
        rf"$\bullet$ Matched cross-paradigm: $A_{{\mathrm{{diag}}}}^{{\mathrm{{cross}}}} = {cross_a_med:.3f}$" + "\n"
        rf"$\bullet$ Paired repeatability gap: $\Delta = +{delta_a_med:.3f}$ [95% CI: ${ci_low_a:.3f}, {ci_high_a:.3f}$]" + "\n"
        rf"$\bullet$ Model consistency: $\mathbf{{{n_pos_a}/34\ models\ (91.2\%)}}$ show $A^{{\mathrm{{self}}}} > A^{{\mathrm{{cross}}}}$" + "\n"
        r"$\bullet$ $\mathit{collie1}$ ($P=1$, no off-diag) & $\mathit{newzealand2}$ ($B<400$) excluded"
    )
    ax_a.text(
        0.02, 0.015, summary_text,
        transform=ax_a.transAxes, fontsize=7.2, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#ffffff", edgecolor="#94a3b8", alpha=0.95, linewidth=0.7)
    )
    
    legend_elements_a = [
        Line2D([0], [0], marker="o", color="w", label=r"Within-IC self reference ($A_m^{\mathrm{self}}$, matched $B_m$)", markerfacecolor=COLOR_IC_SELF, markersize=5.5),
        Line2D([0], [0], marker="o", color="w", label=r"Cross-paradigm matched ($A_m^{\mathrm{cross}}$, IC vs dPL)", markerfacecolor=COLOR_CROSS, markersize=5.5),
        Line2D([0], [0], color=COLOR_DELTA_POS, lw=1.5, label=r"Positive repeatability advantage ($\Delta > 0$, 31 models)"),
        Line2D([0], [0], color=COLOR_DELTA_NEG, lw=1.5, label=r"Negative difference ($\Delta < 0$, 3 models)"),
    ]
    ax_a.legend(handles=legend_elements_a, loc="upper right", frameon=True, facecolor="white", edgecolor=COLOR_BOX_BORDER, framealpha=0.92, fontsize=6.8, borderpad=0.35)

    # =========================================================================
    # PANEL (b): Profile-Level Calibration Reference (R_paired)
    # =========================================================================
    ax_b.set_title(
        r"$\mathbf{b}$  Profile-Level Correlation Reference ($R_{\mathrm{paired}}$, $n=35$ models)",
        loc="left", fontsize=9.0, fontweight="bold", pad=6
    )
    
    ax_b.grid(axis="x", color=GRID_COLOR, linestyle="-", linewidth=0.6, zorder=1)
    
    y_cats = [1, 0]
    labels_cats = [r"Within-IC Self Reference" + "\n" + r"($R_{\mathrm{paired}}^{\mathrm{self}} = 0.968$)", 
                   r"Cross-Paradigm Matched" + "\n" + r"($R_{\mathrm{paired}}^{\mathrm{cross}} = 0.732$)"]
    
    r_self_vals = df_r["R_model_IC_self"].values
    r_cross_vals = df_r["R_model_cross_matched"].values
    
    np.random.seed(42)
    jit_self = np.random.uniform(-0.09, 0.09, size=n_r)
    jit_cross = np.random.uniform(-0.09, 0.09, size=n_r)
    
    for i in range(n_r):
        ax_b.plot([r_cross_vals[i], r_self_vals[i]], [0 + jit_cross[i], 1 + jit_self[i]], 
                  color="#94a3b8", alpha=0.35, linewidth=0.6, zorder=2)
    
    ax_b.scatter(r_self_vals, 1 + jit_self, color=COLOR_IC_SELF, s=18, alpha=0.85, edgecolors="none", zorder=4)
    ax_b.scatter(r_cross_vals, 0 + jit_cross, color=COLOR_CROSS, s=18, alpha=0.85, edgecolors="none", zorder=4)
    
    q25_s, med_s, q75_s = np.percentile(r_self_vals, [25, 50, 75])
    q25_c, med_c, q75_c = np.percentile(r_cross_vals, [25, 50, 75])
    
    ax_b.errorbar([med_s], [1], xerr=[[med_s - q25_s], [q75_s - med_s]], fmt="D", color="#0f2b48", 
                  ecolor="#0f2b48", elinewidth=2.0, capsize=4, capthick=1.4, markersize=6.0, zorder=6)
    ax_b.errorbar([med_c], [0], xerr=[[med_c - q25_c], [q75_c - med_c]], fmt="D", color="#993d00", 
                  ecolor="#993d00", elinewidth=2.0, capsize=4, capthick=1.4, markersize=6.0, zorder=6)
    
    ax_b.set_yticks(y_cats)
    ax_b.set_yticklabels(labels_cats, fontsize=7.6)
    ax_b.set_xlim(0.2, 1.05)
    ax_b.set_ylim(-0.45, 1.45)
    ax_b.set_xlabel(r"Information-profile correlation, $R_m$ (Spearman $\rho$ across 20 clusters)", fontsize=8.2, labelpad=3)
    
    callout_b = (
        rf"$\mathbf{{All\ 35/35\ models\ show\ R^{{\mathrm{{self}}}} > R^{{\mathrm{{cross}}}}}}$" + "\n"
        rf"$\Delta = +{delta_r_med:.3f}$ [95% CI: ${ci_low_r:.3f}, {ci_high_r:.3f}$]"
    )
    ax_b.text(0.22, 0.48, callout_b, fontsize=7.2, va="center", ha="left",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#94a3b8", alpha=0.92, linewidth=0.6))

    # =========================================================================
    # PANEL (c): Sensitivity Across Subsets (All-model vs Strict R2)
    # =========================================================================
    ax_c.set_title(
        r"$\mathbf{c}$  Sensitivity across Model Subsets (All Models vs. Strict $R^2$ Ensemble)",
        loc="left", fontsize=9.0, fontweight="bold", pad=6
    )
    
    ax_c.grid(axis="x", color=GRID_COLOR, linestyle="-", linewidth=0.6, zorder=1)
    
    y_sens = [3, 2, 1, 0]
    sens_labels = [
        r"$R_{\mathrm{paired}}$: All Models ($n=35$)",
        r"$R_{\mathrm{paired}}$: Strict $R^2$ ($n=23$)",
        r"$A_{\mathrm{diag}}$: All Models ($n=34$)",
        r"$A_{\mathrm{diag}}$: Strict $R^2$ ($n=22$)",
    ]
    
    sens_self = [sum_r_all["self_value"], sum_r_strict["self_value"], sum_a_all["self_value"], sum_a_strict["self_value"]]
    sens_cross = [sum_r_all["cross_value"], sum_r_strict["cross_value"], sum_a_all["cross_value"], sum_a_strict["cross_value"]]
    sens_delta = [sum_r_all["difference_self_minus_cross"], sum_r_strict["difference_self_minus_cross"], 
                  sum_a_all["difference_self_minus_cross"], sum_a_strict["difference_self_minus_cross"]]
    
    for i in range(4):
        y = y_sens[i]
        xs = sens_self[i]
        xc = sens_cross[i]
        d = sens_delta[i]
        
        ax_c.plot([xc, xs], [y, y], color=COLOR_DELTA_POS, linewidth=1.8, alpha=0.7, zorder=2)
        ax_c.scatter(xc, y, color=COLOR_CROSS, s=28, edgecolors="white", linewidth=0.5, zorder=4)
        ax_c.scatter(xs, y, color=COLOR_IC_SELF, s=30, edgecolors="white", linewidth=0.5, zorder=5)
        
        ax_c.text(xs + 0.03, y, rf"$\Delta = +{d:.3f}$", va="center", ha="left", fontsize=7.0, color="#1e293b", fontweight="bold")
    
    ax_c.set_yticks(y_sens)
    ax_c.set_yticklabels(sens_labels, fontsize=7.5)
    ax_c.set_xlim(0.45, 1.25)
    ax_c.set_ylim(-0.6, 3.6)
    ax_c.set_xlabel(r"Statistic value (Self vs. Matched Cross)", fontsize=8.2, labelpad=3)

    # =========================================================================
    # PANEL (d): Hydrological Functional-Role Interpretation Boundary
    # =========================================================================
    ax_d.set_title(
        r"$\mathbf{d}$  Hydrological Functional-Role Interpretation Boundary ($n=36$ models)",
        loc="left", fontsize=9.0, fontweight="bold", pad=6
    )
    
    ax_d.grid(axis="x", color=GRID_COLOR, linestyle="-", linewidth=0.6, zorder=1)
    ax_d.axvline(0.0, color="#cbd5e1", linestyle="-", linewidth=0.8, zorder=2)
    
    y_hier = [2.2, 1.35, 0.5]
    hier_labels = [
        r"$\mathbf{1.\ Same\ Parameter\ Coordinate}$" + "\n" + r"(Exact diagonal correspondence)",
        r"$\mathbf{2.\ Same\ Hydrological\ Role}$" + "\n" + r"(Off-diagonal parameters, same role)",
        r"$\mathbf{3.\ Cross\ Hydrological\ Role}$" + "\n" + r"(Parameters in different roles)",
    ]
    
    val_coord = 0.615038
    val_same_role = same_role_med  # -0.027820
    val_cross_role = cross_role_med # +0.005263
    
    bars = ax_d.barh(y_hier, [val_coord, val_same_role, val_cross_role], height=0.45, 
                     color=[COLOR_CROSS, COLOR_ROLE_SAME, COLOR_ROLE_CROSS], 
                     edgecolor=["#8c3b00", "#5c3830", "#4b5563"], linewidth=0.7, alpha=0.85, zorder=3)
    
    ax_d.text(val_coord + 0.02, y_hier[0], rf"$+0.615$ (Preferential Specificity)", va="center", ha="left", fontsize=7.2, fontweight="bold", color="#8c3b00")
    ax_d.text(val_same_role - 0.02, y_hier[1], rf"${val_same_role:.3f}$", va="center", ha="right", fontsize=7.2, fontweight="bold", color="#5c3830")
    ax_d.text(val_cross_role + 0.02, y_hier[2], rf"$+{val_cross_role:.3f}$", va="center", ha="left", fontsize=7.2, fontweight="bold", color="#4b5563")
    
    ax_d.set_yticks(y_hier)
    ax_d.set_yticklabels(hier_labels, fontsize=7.4)
    ax_d.set_xlim(-0.25, 0.88)
    ax_d.set_ylim(-1.05, 2.75)
    ax_d.set_xlabel(r"Correspondence statistic ($C_{m,p,q}$)", fontsize=8.2, labelpad=3)
    
    role_box_text = (
        r"$\mathbf{Functional-Role\ Negative\ Boundary\ Evidence:}$" + "\n"
        rf"$\bullet$ Role off-diagonal advantage: $A_{{\mathrm{{role}}}} = {a_role_obs:.3f}$ ($p = {a_role_p:.3f}$ vs. label null)" + "\n"
        rf"$\bullet$ Permutation null mean: ${a_role_null_mean:.3f}$ [95% null range: ${a_role_null_ci_low:.3f}, +{a_role_null_ci_high:.3f}$]" + "\n"
        rf"$\bullet$ Model consistency: Only $\mathbf{{{n_pos_role}/36\ models\ (33\%)}}$ exhibit $A_{{\mathrm{{role}}}} > 0$" + "\n"
        r"$\bullet$ $\mathbf{Process\ Roles:}$ Soil storage, Runoff partitioning, Routing delay, ET/loss, Snow, Baseflow" + "\n"
        r"$\bullet$ $\mathbf{Boundary\ Verdict:}$ Specificity is coordinate-locked; broad roles share no advantage."
    )
    
    ax_d.text(
        0.02, 0.03, role_box_text,
        transform=ax_d.transAxes, fontsize=6.8, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef2f2", edgecolor="#f87171", alpha=0.95, linewidth=0.7)
    )

    # ---------------------------------------------------------------------
    # Save Figure
    # ---------------------------------------------------------------------
    plt.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Successfully generated {OUTPUT_PNG} at 600 DPI.")

if __name__ == "__main__":
    main()
