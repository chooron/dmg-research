#!/usr/bin/env python3
"""Generate Figure 6 for Journal of Hydrology (JoH) manuscript R3.

Scientific Question:
"Is the retained correspondence preferentially aligned with the same model parameter coordinate rather than alternative coordinates?"

Panels:
(a) HERO: Representative-model correspondence atlas (GR4J, SIMHYD, HBV96)
(b) Ensemble-level parameter-label permutation null test (1,000 permutations)
(c) Model-level specificity breadth (n=35 models sorted by parameter count P_m)
(d) Cumulative coordinate rank consequence (Top-1, Top-2, Top-3 vs random expectation)

Strictly outputs F6_main.png at DPI=600.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import rankdata, gaussian_kde

# Paths
ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
OUTPUT_PNG = ROOT / "F6_main.png"

# Color Palette (consistent across manuscript figures)
NAVY = "#1b3a57"
SLATE = "#335c67"
TEAL = "#2a7b4c"
AMBER = "#c05621"
CORAL = "#c2593f"
CORAL_DARK = "#9c3b24"
GREY_DARK = "#2d3748"
GREY_MID = "#64748b"
GREY_LIGHT = "#e2e8f0"
GRID_COLOR = "#f1f5f9"
STRICT_COLOR = "#1b3a57"
STANDARD_COLOR = "#64748b"


def configure_mpl() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "figure.titlesize": 10.0,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def rank_rows(x: np.ndarray) -> np.ndarray:
    return np.vstack([rankdata(row, method="average") for row in np.asarray(x, float)])


def cross_matrix(ic: np.ndarray, dpl: np.ndarray) -> np.ndarray:
    ir = rank_rows(ic)
    dr = rank_rows(dpl)
    ir = ir - ir.mean(axis=1, keepdims=True)
    dr = dr - dr.mean(axis=1, keepdims=True)
    denom = np.sqrt((ir * ir).sum(axis=1)[:, None] * (dr * dr).sum(axis=1)[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (ir @ dr.T) / denom
    return out


def load_data():
    df_cross = pd.read_csv(TABLES / "R3_PARAMETER_CROSS_CORRESPONDENCE_LONG.csv")
    df_cross_info = df_cross[df_cross["space"] == "information_cluster"]

    df_model = pd.read_csv(TABLES / "R3_PARAMETER_IDENTITY_MODEL_SUMMARY.csv")
    df_model_info = df_model[(df_model["space"] == "information_cluster") & (df_model["population"] == "all36")]

    df_ref = pd.read_csv(TABLES / "R3_F6_CROSS_MATCHED_REFERENCE_MODEL.csv")
    df_null_summary = pd.read_csv(TABLES / "R3_PARAMETER_LABEL_PERMUTATION_NULL.csv")

    return df_cross_info, df_model_info, df_ref, df_null_summary


def compute_null_distribution(n_perm=1000, seed=20260902):
    """Recompute exact null distribution of model-equal median A_diag under parameter-label permutation."""
    df_long = pd.read_csv(TABLES / "R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv")
    df_long_info = df_long[df_long["space"] == "information_cluster"]
    
    models = sorted(df_long_info["model"].unique())
    ic_by_model = {}
    dpl_by_model = {}
    for m in models:
        sub_ic = df_long_info[(df_long_info["model"] == m) & (df_long_info["method"] == "IC")]
        p_ic = sorted(sub_ic["parameter_index"].unique())
        f_ic = sorted(sub_ic["feature_index"].unique())
        ic_by_model[m] = sub_ic.pivot(index="parameter_index", columns="feature_index", values="rho").reindex(index=p_ic, columns=f_ic).to_numpy(float)

        sub_dpl = df_long_info[(df_long_info["model"] == m) & (df_long_info["method"] == "dPL")]
        p_dpl = sorted(sub_dpl["parameter_index"].unique())
        f_dpl = sorted(sub_dpl["feature_index"].unique())
        dpl_by_model[m] = sub_dpl.pivot(index="parameter_index", columns="feature_index", values="rho").reindex(index=p_dpl, columns=f_dpl).to_numpy(float)

    rng = np.random.default_rng(seed)
    null_vals = np.empty(n_perm, dtype=float)
    
    for b in range(n_perm):
        model_advs = []
        for m in models:
            dpl_p = dpl_by_model[m]
            ic_p = ic_by_model[m]
            if dpl_p.shape[0] == 1:
                continue
            perm = rng.permutation(dpl_p.shape[0])
            c_mat = cross_matrix(ic_p, dpl_p[perm])
            diag = np.diag(c_mat)
            advs = []
            for i in range(c_mat.shape[0]):
                others = np.delete(c_mat[i], i)
                advs.append(diag[i] - np.nanmedian(others))
            model_advs.append(np.nanmedian(advs))
        null_vals[b] = float(np.nanmedian(model_advs))
        
    return null_vals


def draw_matrix(ax, c_mat, param_names, title, subtitle="", annotate_threshold=7):
    """Draw a single correspondence matrix heatmap with diagonal outlines."""
    n_p = len(param_names)
    cmap = plt.cm.RdBu_r
    im = ax.imshow(c_mat, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="equal", origin="upper")

    ax.set_xticks(np.arange(n_p))
    ax.set_yticks(np.arange(n_p))
    ax.set_xticklabels(param_names, fontsize=6.8, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(param_names, fontsize=6.8)

    # Highlight diagonal boxes with dark border
    for i in range(n_p):
        rect = Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor="#0f172a", linewidth=1.2, linestyle="-", zorder=4
        )
        ax.add_patch(rect)

    # Annotate numeric values if matrix dimension is small/medium
    if n_p <= annotate_threshold:
        for i in range(n_p):
            for j in range(n_p):
                val = c_mat[i, j]
                color = "white" if abs(val) > 0.60 else "#0f172a"
                fontweight = "bold" if i == j else "normal"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center", color=color,
                    fontsize=6.2 if n_p > 5 else 6.8, fontweight=fontweight
                )

    ax.set_title(title, fontsize=8.6, fontweight="bold", pad=4)
    if subtitle:
        ax.text(0.5, -0.22, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=6.8, color="#4a5568")

    return im


def main():
    configure_mpl()
    df_cross_info, df_model_info, df_ref, df_null_summary = load_data()

    print("Computing permutation null distribution (1000 draws)...")
    null_vals = compute_null_distribution(1000, seed=20260902)
    print(f"Null mean = {np.mean(null_vals):.6f}, 95% CI = [{np.percentile(null_vals, 2.5):.6f}, {np.percentile(null_vals, 97.5):.6f}]")

    # Figure Layout: Width = 11.6 in, Height = 8.8 in
    fig = plt.figure(figsize=(11.6, 8.8), dpi=600)

    # Main Grid: Top half for Panel a (Hero matrices), Bottom half for (b, c, d)
    gs = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=[1.18, 1.0], hspace=0.32,
        left=0.06, right=0.96, top=0.94, bottom=0.06
    )

    # -------------------------------------------------------------
    # Top Section: Panel (a) Representative Model Correspondence Atlas
    # -------------------------------------------------------------
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=gs[0], width_ratios=[1.0, 1.4, 2.6, 0.12], wspace=0.28
    )

    ax_a1 = fig.add_subplot(gs_top[0])
    ax_a2 = fig.add_subplot(gs_top[1])
    ax_a3 = fig.add_subplot(gs_top[2])
    cax = fig.add_subplot(gs_top[3])

    # Representative Models
    # 1. GR4J (4 params)
    m_gr4j = "gr4j"
    sub_gr4j = df_cross_info[df_cross_info["model"] == m_gr4j]
    p_gr4j = sub_gr4j.drop_duplicates("ic_parameter").sort_values("ic_parameter_index")["ic_parameter"].tolist()
    mat_gr4j = sub_gr4j.pivot(index="ic_parameter", columns="dpl_parameter", values="correspondence").loc[p_gr4j, p_gr4j].to_numpy()

    # 2. SIMHYD (7 params)
    m_sim = "simhyd"
    sub_sim = df_cross_info[df_cross_info["model"] == m_sim]
    p_sim = sub_sim.drop_duplicates("ic_parameter").sort_values("ic_parameter_index")["ic_parameter"].tolist()
    mat_sim = sub_sim.pivot(index="ic_parameter", columns="dpl_parameter", values="correspondence").loc[p_sim, p_sim].to_numpy()

    # 3. HBV96 (15 params)
    m_hbv = "hbv96"
    sub_hbv = df_cross_info[df_cross_info["model"] == m_hbv]
    p_hbv = sub_hbv.drop_duplicates("ic_parameter").sort_values("ic_parameter_index")["ic_parameter"].tolist()
    mat_hbv = sub_hbv.pivot(index="ic_parameter", columns="dpl_parameter", values="correspondence").loc[p_hbv, p_hbv].to_numpy()

    # Plot representative heatmaps
    im1 = draw_matrix(ax_a1, mat_gr4j, p_gr4j, "GR4J ($P_m=4$)", subtitle="Low dimension (4 stores/fluxes)", annotate_threshold=4)
    im2 = draw_matrix(ax_a2, mat_sim, p_sim, "SIMHYD ($P_m=7$)", subtitle="Mid dimension (7 process params)", annotate_threshold=7)
    im3 = draw_matrix(ax_a3, mat_hbv, p_hbv, "HBV96 ($P_m=15$)", subtitle="High dimension (15 multi-zone params)", annotate_threshold=0)

    # Set axis labels
    ax_a1.set_ylabel("IC Parameter Coordinate ($p$)", fontsize=8.2, fontweight="bold")
    ax_a1.set_xlabel("dPL Parameter ($q$)", fontsize=7.8)
    ax_a2.set_xlabel("dPL Parameter ($q$)", fontsize=7.8)
    ax_a3.set_xlabel("dPL Parameter ($q$)", fontsize=7.8)

    # Colorbar
    cb = fig.colorbar(im3, cax=cax, orientation="vertical")
    cb.set_label(r"Spearman Profile Correspondence $C_{m,p,q}$", fontsize=7.8, fontweight="bold")
    cb.ax.tick_params(labelsize=7.0)

    # Panel (a) main title tag
    ax_a1.text(
        -0.35, 1.15, "(a) Representative-model correspondence atlas: Diagonal alignment across model complexities",
        transform=ax_a1.transAxes, fontsize=9.2, fontweight="bold", ha="left", va="bottom", color="#0f172a"
    )

    # -------------------------------------------------------------
    # Bottom Section: Panels (b), (c), (d)
    # -------------------------------------------------------------
    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[1], width_ratios=[1.1, 2.0, 1.2], wspace=0.30
    )

    ax_b = fig.add_subplot(gs_bottom[0])
    ax_c = fig.add_subplot(gs_bottom[1])
    ax_d = fig.add_subplot(gs_bottom[2])

    # -------------------------------------------------------------
    # Panel (b): Permutation Null Test
    # -------------------------------------------------------------
    ax_b.set_title("(b) Label-permutation null test", fontsize=8.8, fontweight="bold", loc="left", pad=8)
    
    # Histogram + KDE of null
    bins = np.linspace(-0.25, 0.25, 26)
    n_counts, bin_edges, _ = ax_b.hist(
        null_vals, bins=bins, density=True, color="#cbd5e1", edgecolor="white", linewidth=0.5, label="Null (1,000 perm.)"
    )
    kde = gaussian_kde(null_vals)
    xs = np.linspace(-0.25, 0.25, 200)
    ax_b.plot(xs, kde(xs), color=GREY_MID, linewidth=1.2)

    # Observed headline A_diag = 0.6150
    obs_adiag = 0.6150375940
    ax_b.axvline(obs_adiag, color=CORAL, linewidth=1.6, linestyle="-", label="Observed $A_{\\mathrm{diag}}$ = 0.615")
    ax_b.scatter([obs_adiag], [kde(0.0) * 0.5], color=CORAL, s=35, zorder=5, edgecolor="white", linewidth=0.5)

    # Annotations on Panel b
    ax_b.text(
        obs_adiag - 0.02, kde(0.0) * 0.75,
        f"Observed\n$A_{{\\mathrm{{diag}}}} = 0.615$\n($p = 0.000999$)",
        color=CORAL_DARK, fontsize=7.2, fontweight="bold", ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=CORAL, alpha=0.9, lw=0.6)
    )

    ax_b.text(
        0.0015, kde(0.0) * 0.88,
        f"Null mean $\\approx 0.0015$\n95% CI: [-0.123, 0.113]",
        color=GREY_DARK, fontsize=6.8, ha="center", va="bottom"
    )

    ax_b.set_xlabel("Ensemble Median Diagonal Advantage ($A_{\\mathrm{diag}}$)", fontsize=7.8)
    ax_b.set_ylabel("Probability Density", fontsize=7.8)
    ax_b.set_xlim(-0.30, 0.75)
    ax_b.grid(True, linestyle=":", color=GRID_COLOR, alpha=0.8, linewidth=0.5)
    ax_b.legend(loc="upper left", frameon=False, fontsize=6.8)

    # -------------------------------------------------------------
    # Panel (c): Model-level Specificity Breadth (35 models)
    # -------------------------------------------------------------
    ax_c.set_title("(c) Specificity breadth across conceptual models ($n=35$)", fontsize=8.8, fontweight="bold", loc="left", pad=8)

    # Prepare 35 models sorted by P_m, then model name
    m_dict = df_ref.set_index("model")[["parameter_count", "r2_strict_model", "A_m_cross_matched"]].to_dict(orient="index")
    rows_c = []
    for idx, r in df_model_info.iterrows():
        m = r["model"]
        if m == "collie1":
            continue
        p_cnt = m_dict[m]["parameter_count"]
        strict = m_dict[m]["r2_strict_model"]
        am_full = r["diagonal_advantage_median"]
        rows_c.append({
            "model": m, "P_m": p_cnt, "strict": strict, "A_m": am_full,
            "diag_med": r["diagonal_median"], "offdiag_med": r["offdiagonal_median"]
        })
    df_c = pd.DataFrame(rows_c).sort_values(by=["P_m", "model"], ascending=[True, True]).reset_index(drop=True)

    y_pos = np.arange(len(df_c))
    
    # Background shading for IQR [0.4508, 0.7643]
    q25, q75 = 0.4507518797, 0.76428571425
    med_val = 0.6150375940
    ax_c.axvspan(q25, q75, color="#e6f0fa", alpha=0.6, label="Ensemble IQR [0.451, 0.764]", zorder=0)
    ax_c.axvline(med_val, color=NAVY, linestyle="--", linewidth=1.0, label=f"Median $A_m = 0.615$", zorder=1)
    ax_c.axvline(0.0, color="#a0aec0", linestyle="-", linewidth=0.75, zorder=1)

    # Plot horizontal lollipops
    for i, r in df_c.iterrows():
        color = STRICT_COLOR if r["strict"] else STANDARD_COLOR
        ax_c.hlines(i, 0, r["A_m"], color=color, linewidth=0.9, alpha=0.7, zorder=2)
        marker = "o" if r["strict"] else "s"
        ax_c.scatter(r["A_m"], i, color=color, s=16, marker=marker, zorder=3, edgecolor="white", linewidth=0.3)

    # Format y-ticks with model name and parameter count
    ytick_labels = [f"{r['model']} ({r['P_m']}p)" for _, r in df_c.iterrows()]
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(ytick_labels, fontsize=5.8)
    ax_c.set_ylim(-0.8, len(df_c) - 0.2)
    ax_c.set_xlabel("Model Coordinate Specificity Advantage ($A_m$)", fontsize=7.8)
    ax_c.set_xlim(-0.1, 1.55)
    ax_c.grid(True, axis="x", linestyle=":", color=GRID_COLOR, alpha=0.8, linewidth=0.5)

    # Annotation box for 35/35 positive
    ax_c.text(
        0.98, 0.12,
        "35 / 35 models have $A_m > 0$\nMedian $A_m = 0.615$ (IQR 0.451–0.764)\n• Solid dot: R2 strict ($n=22$)\n▪ Square: Standard ($n=13$)",
        transform=ax_c.transAxes, fontsize=6.8, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cbd5e1", lw=0.6)
    )
    ax_c.legend(loc="upper right", frameon=False, fontsize=6.5)

    # -------------------------------------------------------------
    # Panel (d): Top-k Coordinate Rank Consequence
    # -------------------------------------------------------------
    ax_d.set_title("(d) Coordinate rank consequence", fontsize=8.8, fontweight="bold", loc="left", pad=8)

    ranks = ["Top 1", "Top 2", "Top 3"]
    obs_rates = [57.565, 71.218, 84.133]
    obs_counts = [156, 193, 228]
    null_rates = [13.284, 26.199, 39.114]
    
    y_d = np.array([2, 1, 0])

    for i, y in enumerate(y_d):
        # Dumbbell connecting line
        ax_d.hlines(y, null_rates[i], obs_rates[i], color="#94a3b8", linewidth=2.0, zorder=1)
        # Random null point
        ax_d.scatter(null_rates[i], y, color="#94a3b8", s=38, marker="D", zorder=2, edgecolor="white", linewidth=0.5, label="Random null" if i == 0 else "")
        # Observed point
        ax_d.scatter(obs_rates[i], y, color=CORAL, s=50, marker="o", zorder=3, edgecolor="white", linewidth=0.5, label="Observed" if i == 0 else "")

        # Text labels
        ax_d.text(
            null_rates[i] - 2.5, y, f"{null_rates[i]:.1f}%",
            ha="right", va="center", fontsize=6.8, color="#64748b"
        )
        ax_d.text(
            obs_rates[i] + 2.5, y, f"{obs_rates[i]:.1f}%\n({obs_counts[i]}/271)",
            ha="left", va="center", fontsize=6.8, color=CORAL_DARK, fontweight="bold"
        )

    ax_d.set_yticks(y_d)
    ax_d.set_yticklabels(ranks, fontsize=7.8, fontweight="bold")
    ax_d.set_xlabel("Cumulative Share of Same-Coordinate (%)", fontsize=7.8)
    ax_d.set_xlim(0, 105)
    ax_d.set_ylim(-0.6, 2.6)
    ax_d.grid(True, axis="x", linestyle=":", color=GRID_COLOR, alpha=0.8, linewidth=0.5)
    ax_d.legend(loc="lower right", frameon=False, fontsize=6.8)

    # Note on parameter dimension caveat
    ax_d.text(
        0.02, 0.05,
        "Caveat: Top-1 share correlates with\nparameter dimension ($\\rho = -0.564$,\n$p < 0.001$), but median advantage\n$A_m$ is robust ($\\rho = -0.283, p = 0.100$).",
        transform=ax_d.transAxes, fontsize=6.2, color="#4a5568", ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f8fafc", edgecolor="#e2e8f0", lw=0.5)
    )

    # Save PNG
    plt.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    print(f"Successfully saved Figure 6 to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
