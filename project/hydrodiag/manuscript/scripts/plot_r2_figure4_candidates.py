#!/usr/bin/env python3
"""Generate R2 Figure 4 Layout Exploration Candidates (A and B).

Figure 4 is the parameter-layer space occupation & organization figure.
It answers: How do Base and CN occupy the shared 15D parameter space,
and what distinct parameter-space organization signatures do IC and dPL leave behind?

Candidate A — Actual Parameter Occupation Triptych (1x3 aligned triptych):
- (a) Global Base−CN paired shifts (Δz = Base - CN) for IC & dPL (median + 95% CI)
- (b) IC normalized parameter occupation (Base vs CN median + IQR + 5-95% range, x in [0,1])
- (c) dPL normalized parameter occupation (Base vs CN median + IQR + 5-95% range, x in [0,1])

Candidate B — Derived Organization Diagnostics Triptych (1x3 aligned triptych):
- (a) Global Base−CN paired shifts (same as A(a))
- (b) Boundary-concentration change (CN - Base boundary rate at threshold 0.01 + 95% CI)
- (c) Dispersion / IQR change (CN - Base normalized IQR + 95% CI)

Constraint: PNG ONLY (600 DPI). No PDF files generated!
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT = Path(__file__).resolve().parents[2]
MANUSCRIPT = PROJECT / "manuscript"
RESULTS_R2 = MANUSCRIPT / "results" / "R2"
FIG_DIR = MANUSCRIPT / "figures"
PLOTS_FIG_DIR = MANUSCRIPT / "plots" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Visual Style Constants (R1 / HESS Standard)
COLOR_IC = "#EE7733"      # warm orange (Tol vibrant Base/IC)
COLOR_DPL = "#0077BB"     # deep blue (Tol vibrant CN/dPL)
COLOR_BASE = "#EE7733"    # Base structure color
COLOR_CN = "#0077BB"      # CN structure color
COLOR_GREY = "#6F6F6F"    # neutral grey
COLOR_HL = "#D55E00"      # vermilion highlight

COMMON_XAJ = [
    "xaj_k", "xaj_b", "xaj_im", "xaj_um", "xaj_lm", "xaj_dm", "xaj_c",
    "xaj_sm", "xaj_ex", "xaj_ki", "xaj_kg", "xaj_ci", "xaj_cg", "xaj_a",
    "xaj_theta",
]

DISPLAY = {
    "xaj_k": "k", "xaj_b": "b", "xaj_im": "im", "xaj_um": "um",
    "xaj_lm": "lm", "xaj_dm": "dm", "xaj_c": "c", "xaj_sm": "sm",
    "xaj_ex": "ex", "xaj_ki": "ki", "xaj_kg": "kg", "xaj_ci": "ci",
    "xaj_cg": "cg", "xaj_a": "a", "xaj_theta": "θ",
}

# Fixed 15 Parameter order (bottom to top for y-axis plotting)
PARAM_ORDER = [
    "xaj_k", "xaj_b", "xaj_im", "xaj_um", "xaj_lm", "xaj_dm", "xaj_c",
    "xaj_sm", "xaj_ex", "xaj_ki", "xaj_kg", "xaj_ci", "xaj_cg",
    "xaj_a", "xaj_theta"
]
Y_PARAMS = list(reversed(PARAM_ORDER))

def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "patch.linewidth": 0.8,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
    })

def apply_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.set_facecolor('#FFFFFF')

def load_r2_data():
    df_shift = pd.read_csv(RESULTS_R2 / "r2_primary_shift_summary.csv")
    df_canon = pd.read_csv(RESULTS_R2 / "r2_parameter_values_canonical.csv")
    df_canon["basin_id"] = df_canon["basin_id"].astype(str).str.zfill(8)
    df_bnd = pd.read_csv(RESULTS_R2 / "r2_boundary_summary.csv")
    df_disp = pd.read_csv(RESULTS_R2 / "r2_dispersion_change_summary.csv")
    return df_shift, df_canon, df_bnd, df_disp

# =============================================================================
# CANDIDATE A: ACTUAL PARAMETER OCCUPATION TRIPTYCH
# (a) Global Shifts | (b) IC Base vs CN Occupation | (c) dPL Base vs CN Occupation
# =============================================================================
def build_candidate_a(df_shift, df_canon):
    fig = plt.figure(figsize=(11.0, 7.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.25)
    
    y_positions = np.arange(len(Y_PARAMS))
    offset = 0.18
    
    # -------------------------------------------------------------------------
    # Panel (a): Global Base - CN paired shifts (Δz)
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    apply_spines(ax_a)
    ax_a.set_title("(a) Global Base−CN paired shifts (Δz)", weight="bold", loc="left", pad=8)
    ax_a.axvline(0, color=COLOR_GREY, linestyle="--", linewidth=0.9, zorder=1)
    
    for idx, p in enumerate(Y_PARAMS):
        s_ic = df_shift[(df_shift.paradigm == "IC") & (df_shift.parameter == p)].iloc[0]
        s_dpl = df_shift[(df_shift.paradigm == "dPL") & (df_shift.parameter == p)].iloc[0]
        
        y_pos = y_positions[idx]
        
        # IC
        m_ic, l_ic, h_ic = s_ic["median_shift"], s_ic["ci95_low"], s_ic["ci95_high"]
        ax_a.errorbar(m_ic, y_pos + offset, xerr=[[m_ic - l_ic], [h_ic - m_ic]],
                      fmt="o", color=COLOR_IC, ecolor=COLOR_IC, elinewidth=1.3,
                      capsize=2.5, capthick=1.0, markersize=4.5, zorder=3, label="IC (CMA-ES)" if idx == 0 else "")
        
        # dPL
        m_dpl, l_dpl, h_dpl = s_dpl["median_shift"], s_dpl["ci95_low"], s_dpl["ci95_high"]
        ax_a.errorbar(m_dpl, y_pos - offset, xerr=[[m_dpl - l_dpl], [h_dpl - m_dpl]],
                      fmt="s", color=COLOR_DPL, ecolor=COLOR_DPL, elinewidth=1.3,
                      capsize=2.5, capthick=1.0, markersize=4.5, zorder=3, label="dPL (Neural Net)" if idx == 0 else "")

    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels([DISPLAY[p] for p in Y_PARAMS], fontweight="normal", fontsize=8.2)
    ax_a.set_xlabel("Normalized shift Δz = z_Base − z_CN", labelpad=4)
    ax_a.set_xlim(-0.25, 0.25)
    ax_a.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=7.8)
    ax_a.grid(True, axis="x", linestyle=":", alpha=0.4)

    for tick in ax_a.get_yticklabels():
        if tick.get_text() in ["um", "ki", "ci", "im"]:
            tick.set_color(COLOR_HL)
            tick.set_fontweight("bold")

    # -------------------------------------------------------------------------
    # Helper to plot parameter occupation (Base vs CN distribution)
    # -------------------------------------------------------------------------
    def plot_occupation(ax, df_c, paradigm_name, panel_letter, title_text):
        apply_spines(ax)
        ax.set_title(f"({panel_letter}) {title_text}", weight="bold", loc="left", pad=8)
        sub = df_c[df_c.paradigm == paradigm_name]
        
        for idx, p in enumerate(Y_PARAMS):
            b_vals = sub[(sub.structure == "Base") & (sub.parameter == p)]["z"].to_numpy()
            c_vals = sub[(sub.structure == "CN") & (sub.parameter == p)]["z"].to_numpy()
            
            b_med, b_q25, b_q75 = np.median(b_vals), np.percentile(b_vals, 25), np.percentile(b_vals, 75)
            b_p5, b_p95 = np.percentile(b_vals, 5), np.percentile(b_vals, 95)
            
            c_med, c_q25, c_q75 = np.median(c_vals), np.percentile(c_vals, 25), np.percentile(c_vals, 75)
            c_p5, c_p95 = np.percentile(c_vals, 5), np.percentile(c_vals, 95)
            
            y_pos = y_positions[idx]
            
            # Base Structure (Warm Orange)
            ax.plot([b_p5, b_p95], [y_pos + offset, y_pos + offset], color=COLOR_BASE, linewidth=0.7, alpha=0.6, zorder=2)
            ax.plot([b_q25, b_q75], [y_pos + offset, y_pos + offset], color=COLOR_BASE, linewidth=2.4, alpha=0.85, zorder=3)
            ax.plot(b_med, y_pos + offset, "o", color="white", markeredgecolor=COLOR_BASE, markeredgewidth=1.2, markersize=4.0, zorder=4, label="Base structure" if idx == 0 else "")
            
            # CN Structure (Deep Blue)
            ax.plot([c_p5, c_p95], [y_pos - offset, y_pos - offset], color=COLOR_CN, linewidth=0.7, alpha=0.6, zorder=2)
            ax.plot([c_q25, c_q75], [y_pos - offset, y_pos - offset], color=COLOR_CN, linewidth=2.4, alpha=0.85, zorder=3)
            ax.plot(c_med, y_pos - offset, "s", color="white", markeredgecolor=COLOR_CN, markeredgewidth=1.2, markersize=4.0, zorder=4, label="CN structure" if idx == 0 else "")

        ax.set_yticks(y_positions)
        ax.set_yticklabels([DISPLAY[p] for p in Y_PARAMS], fontweight="normal", fontsize=8.2)
        ax.set_xlabel("Normalized parameter value z ∈ [0, 1]", labelpad=4)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, axis="x", linestyle=":", alpha=0.4)
        ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=7.8)

        for tick in ax.get_yticklabels():
            if tick.get_text() in ["um", "ki", "ci", "im"]:
                tick.set_color(COLOR_HL)
                tick.set_fontweight("bold")

    # -------------------------------------------------------------------------
    # Panel (b): IC Base vs CN Occupation
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    plot_occupation(ax_b, df_canon, "IC", "b", "IC normalized occupation")

    # -------------------------------------------------------------------------
    # Panel (c): dPL Base vs CN Occupation
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    plot_occupation(ax_c, df_canon, "dPL", "c", "dPL normalized occupation")

    out_png1 = FIG_DIR / "Figure4_layout_A_occupation_triptych.png"
    out_png2 = PLOTS_FIG_DIR / "Figure4_layout_A_occupation_triptych.png"
    plt.savefig(out_png1, dpi=600)
    plt.savefig(out_png2, dpi=600)
    plt.close()
    print("Generated Candidate A PNG:", out_png1)

# =============================================================================
# CANDIDATE B: DERIVED ORGANIZATION DIAGNOSTICS TRIPTYCH
# (a) Global Shifts | (b) Boundary Rate Change | (c) Dispersion / IQR Change
# =============================================================================
def build_candidate_b(df_shift, df_bnd, df_disp):
    fig = plt.figure(figsize=(11.0, 7.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.25)
    
    y_positions = np.arange(len(Y_PARAMS))
    offset = 0.18
    
    # -------------------------------------------------------------------------
    # Panel (a): Global Base - CN paired shifts (Δz)
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    apply_spines(ax_a)
    ax_a.set_title("(a) Global Base−CN paired shifts (Δz)", weight="bold", loc="left", pad=8)
    ax_a.axvline(0, color=COLOR_GREY, linestyle="--", linewidth=0.9, zorder=1)
    
    for idx, p in enumerate(Y_PARAMS):
        s_ic = df_shift[(df_shift.paradigm == "IC") & (df_shift.parameter == p)].iloc[0]
        s_dpl = df_shift[(df_shift.paradigm == "dPL") & (df_shift.parameter == p)].iloc[0]
        
        y_pos = y_positions[idx]
        
        m_ic, l_ic, h_ic = s_ic["median_shift"], s_ic["ci95_low"], s_ic["ci95_high"]
        ax_a.errorbar(m_ic, y_pos + offset, xerr=[[m_ic - l_ic], [h_ic - m_ic]],
                      fmt="o", color=COLOR_IC, ecolor=COLOR_IC, elinewidth=1.3,
                      capsize=2.5, capthick=1.0, markersize=4.5, zorder=3, label="IC (CMA-ES)" if idx == 0 else "")
        
        m_dpl, l_dpl, h_dpl = s_dpl["median_shift"], s_dpl["ci95_low"], s_dpl["ci95_high"]
        ax_a.errorbar(m_dpl, y_pos - offset, xerr=[[m_dpl - l_dpl], [h_dpl - m_dpl]],
                      fmt="s", color=COLOR_DPL, ecolor=COLOR_DPL, elinewidth=1.3,
                      capsize=2.5, capthick=1.0, markersize=4.5, zorder=3, label="dPL (Neural Net)" if idx == 0 else "")

    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels([DISPLAY[p] for p in Y_PARAMS], fontweight="normal", fontsize=8.2)
    ax_a.set_xlabel("Normalized shift Δz = z_Base − z_CN", labelpad=4)
    ax_a.set_xlim(-0.25, 0.25)
    ax_a.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=7.8)
    ax_a.grid(True, axis="x", linestyle=":", alpha=0.4)

    for tick in ax_a.get_yticklabels():
        if tick.get_text() in ["um", "ki", "ci", "im"]:
            tick.set_color(COLOR_HL)
            tick.set_fontweight("bold")

    # -------------------------------------------------------------------------
    # Panel (b): Boundary-concentration change (CN - Base rate at threshold 0.01)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    apply_spines(ax_b)
    ax_b.set_title("(b) Boundary-rate change (CN − Base)", weight="bold", loc="left", pad=8)
    ax_b.axvline(0, color=COLOR_GREY, linestyle="--", linewidth=0.9, zorder=1)
    
    sub_bnd = df_bnd[(df_bnd.threshold == 0.01) & df_bnd.cn_minus_base.notna()]
    
    for idx, p in enumerate(Y_PARAMS):
        b_ic = sub_bnd[(sub_bnd.paradigm == "IC") & (sub_bnd.parameter == p)].iloc[0]
        b_dpl = sub_bnd[(sub_bnd.paradigm == "dPL") & (sub_bnd.parameter == p)].iloc[0]
        
        y_pos = y_positions[idx]
        
        m_ic, l_ic, h_ic = b_ic["cn_minus_base"], b_ic["ci95_low"], b_ic["ci95_high"]
        ax_b.errorbar(m_ic, y_pos + offset, xerr=[[m_ic - l_ic], [h_ic - m_ic]],
                      fmt="o", color=COLOR_IC, ecolor=COLOR_IC, elinewidth=1.3,
                      capsize=2.5, capthick=1.0, markersize=4.5, zorder=3, label="IC" if idx == 0 else "")
        
        m_dpl, l_dpl, h_dpl = b_dpl["cn_minus_base"], b_dpl["ci95_low"], b_dpl["ci95_high"]
        ax_b.errorbar(m_dpl, y_pos - offset, xerr=[[m_dpl - l_dpl], [h_dpl - m_dpl]],
                      fmt="s", color=COLOR_DPL, ecolor=COLOR_DPL, elinewidth=1.3,
                      capsize=2.5, capthick=1.0, markersize=4.5, zorder=3, label="dPL" if idx == 0 else "")

    ax_b.set_yticks(y_positions)
    ax_b.set_yticklabels([DISPLAY[p] for p in Y_PARAMS], fontweight="normal", fontsize=8.2)
    ax_b.set_xlabel("Δ Boundary fraction (ε = 0.01)", labelpad=4)
    ax_b.set_xlim(-0.25, 0.25)
    ax_b.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=7.8)
    ax_b.grid(True, axis="x", linestyle=":", alpha=0.4)

    for tick in ax_b.get_yticklabels():
        if tick.get_text() in ["um", "ki", "ci", "im"]:
            tick.set_color(COLOR_HL)
            tick.set_fontweight("bold")

    # -------------------------------------------------------------------------
    # Panel (c): Dispersion / IQR change (CN - Base IQR)
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    apply_spines(ax_c)
    ax_c.set_title("(c) Normalized IQR change (CN − Base)", weight="bold", loc="left", pad=8)
    ax_c.axvline(0, color=COLOR_GREY, linestyle="--", linewidth=0.9, zorder=1)
    
    for idx, p in enumerate(Y_PARAMS):
        d_ic = df_disp[(df_disp.paradigm == "IC") & (df_disp.parameter == p)].iloc[0]
        d_dpl = df_disp[(df_disp.paradigm == "dPL") & (df_disp.parameter == p)].iloc[0]
        
        y_pos = y_positions[idx]
        
        m_ic, l_ic, h_ic = d_ic["iqr_difference_cn_minus_base"], d_ic["ci95_low"], d_ic["ci95_high"]
        ax_c.errorbar(m_ic, y_pos + offset, xerr=[[m_ic - l_ic], [h_ic - m_ic]],
                      fmt="o", color=COLOR_IC, ecolor=COLOR_IC, elinewidth=1.3,
                      capsize=2.5, capthick=1.0, markersize=4.5, zorder=3, label="IC" if idx == 0 else "")
        
        m_dpl, l_dpl, h_dpl = d_dpl["iqr_difference_cn_minus_base"], d_dpl["ci95_low"], d_dpl["ci95_high"]
        ax_c.errorbar(m_dpl, y_pos - offset, xerr=[[m_dpl - l_dpl], [h_dpl - m_dpl]],
                      fmt="s", color=COLOR_DPL, ecolor=COLOR_DPL, elinewidth=1.3,
                      capsize=2.5, capthick=1.0, markersize=4.5, zorder=3, label="dPL" if idx == 0 else "")

    ax_c.set_yticks(y_positions)
    ax_c.set_yticklabels([DISPLAY[p] for p in Y_PARAMS], fontweight="normal", fontsize=8.2)
    ax_c.set_xlabel("Δ IQR (CN − Base)", labelpad=4)
    ax_c.set_xlim(-0.35, 0.35)
    ax_c.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=7.8)
    ax_c.grid(True, axis="x", linestyle=":", alpha=0.4)

    for tick in ax_c.get_yticklabels():
        if tick.get_text() in ["um", "ki", "ci", "im"]:
            tick.set_color(COLOR_HL)
            tick.set_fontweight("bold")

    out_png1 = FIG_DIR / "Figure4_layout_B_diagnostics_triptych.png"
    out_png2 = PLOTS_FIG_DIR / "Figure4_layout_B_diagnostics_triptych.png"
    plt.savefig(out_png1, dpi=600)
    plt.savefig(out_png2, dpi=600)
    plt.close()
    print("Generated Candidate B PNG:", out_png1)

def main():
    setup_style()
    df_shift, df_canon, df_bnd, df_disp = load_r2_data()
    build_candidate_a(df_shift, df_canon)
    build_candidate_b(df_shift, df_bnd, df_disp)
    print("Both Candidate A and Candidate B PNG figures generated successfully.")

if __name__ == "__main__":
    main()
