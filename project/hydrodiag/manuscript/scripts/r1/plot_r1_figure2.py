"""
Plotting Script for Main-Text Figure 2 (R1 Analysis)
Generates manuscript/figures/Figure2_R1_final.png.

Scientific Purpose:
  Comprehensive assessment of predictive performance and outlet-level snow-process
  visibility across 531 CAMELS-US catchments under both IC (CMA-ES) and dPL (MLP)
  estimation regimes.

Layout (3-row, 5-panel comprehensive architecture):
  Row 1 (Top): Streamflow predictive performance (KGE ECDFs across 531 basins)
    (a) Predictive performance (IC)
        ECDF of KGE for Base (Orange), TGD (Green), CN (Blue) in train (solid) vs test (dashed).
    (b) Predictive performance (dPL)
        ECDF of KGE for Base (Orange), TGD (Green), CN (Blue) in train (solid) vs test (dashed).
  Row 2 (Middle, full width):
    (c) Activity-conditioned timing separation
        Merged single full-width plot with continuous frac_snow on x-axis.
        Shows IC (deep blue) vs dPL (warm vermillion) with S1-S5 background bands,
        531 basin-level scatter points, and stratum medians + bootstrap 95% CIs.
        Panel legend in upper right, Spearman rho correlations in lower right.
  Row 3 (Bottom):
    (d) Signed timing error by structure (~59% width, Scatter + Boxplot style)
        Discrete S1-S5 signed CT error across Base (Orange), TGD (Green), and CN (Blue)
        under IC (filled) and dPL (hollow). Panel legend in lower left.
    (e) Large timing error among KGE-qualified basins (~41% width, Horizontal Dot-Whisker)
        Common-pass subset (all 3 structures KGE >= 0.60; IC n=321, dPL n=331)
        Prevalence of |CT| >= 15 d with bootstrap 95% CIs. Legend for filled (IC) vs hollow (dPL).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats

# Path setup
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
R1_DIR = PROJECT_ROOT / "manuscript" / "results" / "R1"
CACHE_STAGED_DIR = PROJECT_ROOT / "manuscript" / "cache" / "r1_rebuild_audit_staged"
PLOTS_FIG_DIR = PROJECT_ROOT / "manuscript" / "figures"

# Add shared directory to sys.path
shared_dir = HERE.parent / "shared"
if str(shared_dir) not in sys.path:
    sys.path.insert(0, str(shared_dir))

from r1_plot_style import (
    COLOR_BASE,
    COLOR_TGD,
    COLOR_CN,
    COLOR_DARK_NEUTRAL,
    COLOR_LIGHT_REF,
    COLOR_ZERO_LINE,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_MARKERS,
    PERIOD_STYLES,
    apply_clean_spines,
    setup_publication_style,
)

# ── Color Palette (Figure Standards) ─────────────────────────────────────────
COLOR_BASE = "#D55E00"  # Okabe-Ito vermillion / orange (omitted-process baseline)
COLOR_TGD = "#009E73"   # Okabe-Ito bluish green / teal (generic control)
COLOR_CN = "#0072B2"    # Okabe-Ito blue (explicit snow representation)
COLOR_TEXT = "#303438"  # dark neutral for text and primary markers
COLOR_REF = "#70767B"   # mid grey zero/reference lines
COLOR_LIGHT = "#E2E8F0" # light grid / secondary bounds

# Colors for distinguishing IC and dPL in Panel (c)
COLOR_IC = "#2B6CB0"    # Deep classic blue
COLOR_DPL = "#D95F02"   # Warm vermillion / amber

MODEL_COLORS = {
    "Base": COLOR_BASE,
    "TGD": COLOR_TGD,
    "CN": COLOR_CN,
}
MODEL_MARKERS = {
    "Base": "o",
    "TGD": "^",
    "CN": "s",
}

# Snow-fraction regime definitions: S1-S5 by frac_snow
SNOW_BINS = [0.0, 0.05, 0.15, 0.30, 0.50, 1.0001]
SNOW_STRATA = ["S1", "S2", "S3", "S4", "S5"]
STRATA_SAMPLE_SIZES = {"S1": 165, "S2": 156, "S3": 121, "S4": 34, "S5": 55}
STRATA_BOUNDS = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 1.0)]
STRATA_MIDPOINTS = [0.025, 0.10, 0.225, 0.40, 0.75]
STRATA_LABELS = [f"{s} (n={STRATA_SAMPLE_SIZES[s]})" for s in SNOW_STRATA]


def load_authoritative_performance_data() -> pd.DataFrame:
    """Load authoritative basin-level performance for train and test periods (all 531 basins)."""
    staged_perf = CACHE_STAGED_DIR / "r1_basin_level_performance_rebuilt.csv"
    if staged_perf.exists():
        df = pd.read_csv(staged_perf)
        df["basin_id"] = df["basin_id"].astype(str).str.zfill(8)
        df["model"] = "XAJ-" + df["structure"]
        df["kge"] = df["KGE"]
        return df[["basin_id", "paradigm", "structure", "model", "period", "kge"]]

    # Fallback to direct results/R1
    perf_path = R1_DIR / "r1_basin_level_performance.csv"
    df = pd.read_csv(perf_path)
    df["basin_id"] = df["basin_id"].astype(str).str.zfill(8)
    df["structure"] = df["model"].str.replace("XAJ-", "")
    return df[["basin_id", "paradigm", "structure", "model", "period", "kge"]]


def load_authoritative_ct_data() -> pd.DataFrame:
    """Load authoritative joined test-period CT data (531 basins per structure/paradigm)."""
    staged_ct_file = CACHE_STAGED_DIR / "r1_basin_level_ct.csv"
    if staged_ct_file.exists():
        df = pd.read_csv(staged_ct_file)
        df_test = df[df["period"] == "test"].copy()
        df_test["basin_id"] = df_test["basin_id"].astype(str).str.zfill(8)
        return df_test

    # Fallback to direct results/R1
    sig_file = R1_DIR / "r1_snow_signatures_basin_level.csv"
    perf_file = R1_DIR / "r1_basin_level_performance.csv"
    attr_file = R1_DIR / "r1_snow_attributes.csv"

    sig_df = pd.read_csv(sig_file)
    perf_df = pd.read_csv(perf_file)
    attr_df = pd.read_csv(attr_file)

    sig_df["basin_id"] = sig_df["basin_id"].astype(str).str.zfill(8)
    perf_df["basin_id"] = perf_df["basin_id"].astype(str).str.zfill(8)
    attr_df["basin_id"] = attr_df["basin_id"].astype(str).str.zfill(8)

    attr_df["snow_stratum"] = pd.cut(
        attr_df["frac_snow"], bins=SNOW_BINS, labels=SNOW_STRATA, right=False
    )
    attr_map = attr_df.set_index("basin_id")[["frac_snow", "snow_stratum"]]

    records = []
    for paradigm in ["IC-CMA-ES", "dPL-MLP"]:
        for struct in ["Base", "TGD", "CN"]:
            m_code = f"XAJ-{struct}"
            p_sub = perf_df[
                (perf_df["paradigm"] == paradigm)
                & (perf_df["model"] == m_code)
                & (perf_df["period"] == "test")
            ].set_index("basin_id")["kge"]

            s_sub = sig_df[
                (sig_df["paradigm"] == paradigm)
                & (sig_df["model"] == m_code)
                & (sig_df["period"] == "test")
            ]
            if paradigm == "IC-CMA-ES":
                ct_map = s_sub[s_sub["seed_or_restart"] == "selected_restart"].set_index("basin_id")["ct_error_signed"]
            else:
                ct_map = s_sub.groupby("basin_id")["ct_error_signed"].median()

            for b_id in p_sub.index:
                if b_id in ct_map.index and b_id in attr_map.index:
                    records.append({
                        "basin_id": b_id,
                        "paradigm": paradigm,
                        "structure": struct,
                        "period": "test",
                        "basin_median_Delta_CT": ct_map.loc[b_id],
                        "frac_snow": attr_map.loc[b_id, "frac_snow"],
                        "snow_stratum": attr_map.loc[b_id, "snow_stratum"],
                        "KGE": p_sub.loc[b_id],
                    })
    return pd.DataFrame(records)


def compute_bootstrap_ci(values: np.ndarray, stat_func=np.median, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval for a summary statistic."""
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = stat_func(sample)
    low, high = np.percentile(boot_stats, [2.5, 97.5])
    return float(low), float(high)


def main(out_dir: Path | None = None) -> Path:
    setup_publication_style()

    out_fig_dir = out_dir or PLOTS_FIG_DIR
    os.makedirs(out_fig_dir, exist_ok=True)

    # 1. Load authoritative data
    df_perf = load_authoritative_performance_data()
    df_test = load_authoritative_ct_data()

    structures = ["Base", "TGD", "CN"]
    paradigms = ["IC-CMA-ES", "dPL-MLP"]

    # Verify 531 basins present per combination for performance
    for p in paradigms:
        for s in structures:
            for per in ["train", "test"]:
                n_b = len(df_perf[(df_perf["paradigm"] == p) & (df_perf["structure"] == s) & (df_perf["period"] == per)])
                if n_b != 531:
                    raise ValueError(f"Expected 531 basins for {p} {s} {per}, found {n_b}")

    # 2. Canvas dimensions and layout (3 rows: Row 1 = 2 panels, Row 2 = 1 panel, Row 3 = 2 panels)
    fig_w_in = 10.2
    fig_h_in = 10.6
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    gs_main = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.0, 1.20, 1.15],
        hspace=0.27,
        top=0.95,
        bottom=0.055,
        left=0.08,
        right=0.98,
    )

    # Row 1: Panels (a) and (b)
    gs_top = gs_main[0].subgridspec(1, 2, wspace=0.18)
    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[0, 1])

    # Row 2: Panel (c)
    ax_c = fig.add_subplot(gs_main[1])

    # Row 3: Panels (d) and (e)
    gs_bot = gs_main[2].subgridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.22)
    ax_d = fig.add_subplot(gs_bot[0, 0])
    ax_e = fig.add_subplot(gs_bot[0, 1])

    source_records = []

    # ═════════════════════════════════════════════════════════════════════════
    # ROW 1: Predictive performance ECDFs (Panels a and b)
    # ═════════════════════════════════════════════════════════════════════════
    ecdf_xlim = [-0.5, 1.0]
    ecdf_ylim = [0.0, 1.0]

    # --- Panel (a): IC ECDF ---
    apply_clean_spines(ax_a)
    ax_a.axvline(0.0, color=COLOR_ZERO_LINE, lw=0.6, ls=":", zorder=1)
    for s_name in structures:
        c = MODEL_COLORS[s_name]
        for p in ["train", "test"]:
            st = PERIOD_STYLES[p]
            sub = df_perf[
                (df_perf["paradigm"] == "IC-CMA-ES")
                & (df_perf["structure"] == s_name)
                & (df_perf["period"] == p)
            ]
            vals = np.sort(sub["kge"].values)
            y_vals = np.arange(1, len(vals) + 1) / len(vals)
            ax_a.step(
                vals,
                y_vals,
                where="post",
                color=c,
                linestyle=st["linestyle"],
                linewidth=st["linewidth"],
                alpha=st["alpha"],
                zorder=2 if p == "train" else 3,
            )

    ax_a.set_xlim(ecdf_xlim)
    ax_a.set_ylim(ecdf_ylim)
    ax_a.set_xlabel("KGE", fontsize=9.0)
    ax_a.set_ylabel("Cumulative prob.", fontsize=9.0)
    ax_a.set_title(
        "(a) Predictive performance (IC)",
        loc="left",
        fontsize=9.8,
        fontweight="bold",
        pad=6,
    )

    # --- Panel (b): dPL ECDF ---
    apply_clean_spines(ax_b)
    ax_b.axvline(0.0, color=COLOR_ZERO_LINE, lw=0.6, ls=":", zorder=1)
    for s_name in structures:
        c = MODEL_COLORS[s_name]
        for p in ["train", "test"]:
            st = PERIOD_STYLES[p]
            sub = df_perf[
                (df_perf["paradigm"] == "dPL-MLP")
                & (df_perf["structure"] == s_name)
                & (df_perf["period"] == p)
            ]
            vals = np.sort(sub["kge"].values)
            y_vals = np.arange(1, len(vals) + 1) / len(vals)
            ax_b.step(
                vals,
                y_vals,
                where="post",
                color=c,
                linestyle=st["linestyle"],
                linewidth=st["linewidth"],
                alpha=st["alpha"],
                zorder=2 if p == "train" else 3,
            )

    ax_b.set_xlim(ecdf_xlim)
    ax_b.set_ylim(ecdf_ylim)
    ax_b.set_xlabel("KGE", fontsize=9.0)
    ax_b.set_yticklabels([])  # share y-scale with panel a
    ax_b.set_title(
        "(b) Predictive performance (dPL)",
        loc="left",
        fontsize=9.8,
        fontweight="bold",
        pad=6,
    )

    # Legend for Row 1 inside Panel (a) top-left (2 rows: Row 0 = Models, Row 1 = Periods)
    h_base = Line2D([0], [0], color=MODEL_COLORS["Base"], lw=1.8, marker=MODEL_MARKERS["Base"], markersize=4.8, markerfacecolor=MODEL_COLORS["Base"], markeredgecolor="white", label="Base")
    h_tgd = Line2D([0], [0], color=MODEL_COLORS["TGD"], lw=1.8, marker=MODEL_MARKERS["TGD"], markersize=4.8, markerfacecolor=MODEL_COLORS["TGD"], markeredgecolor="white", label="TGD")
    h_cn = Line2D([0], [0], color=MODEL_COLORS["CN"], lw=1.8, marker=MODEL_MARKERS["CN"], markersize=4.8, markerfacecolor=MODEL_COLORS["CN"], markeredgecolor="white", label="CN")
    h_train = Line2D([0], [0], color=COLOR_DARK_NEUTRAL, linestyle=PERIOD_STYLES["train"]["linestyle"], lw=1.5, label="Train (solid)")
    h_test = Line2D([0], [0], color=COLOR_DARK_NEUTRAL, linestyle=PERIOD_STYLES["test"]["linestyle"], lw=1.7, label="Test (dashed)")
    h_empty = Line2D([], [], color="none", label="")

    # Ordered column-major for 3 columns x 2 rows: Col 1 (Base, Train), Col 2 (TGD, Test), Col 3 (CN, blank)
    handles_row1 = [h_base, h_train, h_tgd, h_test, h_cn, h_empty]

    ax_a.legend(
        handles=handles_row1,
        loc="upper left",
        bbox_to_anchor=(0.025, 0.97),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        framealpha=0.92,
        fontsize=7.4,
        handlelength=1.4,
        columnspacing=0.45,
        handletextpad=0.25,
        borderpad=0.30,
    )
    # Record ECDF Source Data
    for p_name, panel_id in [("IC-CMA-ES", "a"), ("dPL-MLP", "b")]:
        for s_name in structures:
            for period in ["train", "test"]:
                sub = df_perf[
                    (df_perf["paradigm"] == p_name)
                    & (df_perf["structure"] == s_name)
                    & (df_perf["period"] == period)
                ]
                for _, r in sub.iterrows():
                    source_records.append({
                        "panel": panel_id,
                        "basin_id": r["basin_id"],
                        "paradigm": p_name,
                        "structure": s_name,
                        "model": f"XAJ-{s_name}",
                        "period": period,
                        "snow_stratum": np.nan,
                        "frac_snow": np.nan,
                        "metric": "kge",
                        "value": r["kge"],
                        "summary_type": "raw_basin_observation",
                    })

    # ═════════════════════════════════════════════════════════════════════════
    # ROW 2: Activity-conditioned timing separation (Panel c)
    # ═════════════════════════════════════════════════════════════════════════
    apply_clean_spines(ax_c)
    ax_c.set_title(
        "(c) Activity-conditioned timing separation",
        loc="left",
        fontsize=10.2,
        fontweight="bold",
        pad=8,
    )
    ax_c.axhline(0, color=COLOR_REF, linestyle="--", linewidth=0.8, zorder=1)

    # Background strata bands (alternating light tint)
    for i, (x0, x1) in enumerate(STRATA_BOUNDS):
        if i % 2 == 1:
            ax_c.axvspan(x0, x1, color="#F1F5F9", alpha=0.6, zorder=0)
        if i > 0:
            ax_c.axvline(x0, color="#CBD5E1", lw=0.7, ls=":", zorder=1)
        ax_c.text(
            STRATA_MIDPOINTS[i],
            58.0,
            STRATA_LABELS[i],
            ha="center",
            va="top",
            fontsize=7.5,
            color="#64748B",
            fontweight="medium",
            zorder=4,
        )

    # Plot both IC and dPL in panel c
    regime_cfg = [
        ("IC-CMA-ES", "IC", COLOR_IC, "o", -0.006),
        ("dPL-MLP", "dPL", COLOR_DPL, "^", +0.006),
    ]

    for p_name, p_label, col, marker, dx in regime_cfg:
        sub_p = df_test[df_test["paradigm"] == p_name]
        piv = sub_p.pivot(
            index=["basin_id", "snow_stratum", "frac_snow"],
            columns="structure",
            values="basin_median_Delta_CT",
        ).reset_index()
        piv["delta_abs_ct"] = piv["Base"].abs() - piv["CN"].abs()

        # Layer 1: 531 Basin-level scatter points (small, light)
        ax_c.scatter(
            piv["frac_snow"],
            piv["delta_abs_ct"],
            s=16,
            color=col,
            alpha=0.22,
            edgecolors="none",
            rasterized=True,
            zorder=2,
            label=f"{p_label} basins",
        )

        # Record individual basin source data
        for _, r in piv.iterrows():
            source_records.append({
                "panel": "c",
                "basin_id": r["basin_id"],
                "paradigm": p_name,
                "structure": np.nan,
                "model": np.nan,
                "period": "test",
                "snow_stratum": r["snow_stratum"],
                "frac_snow": r["frac_snow"],
                "metric": "delta_abs_ct_base_minus_cn",
                "value": r["delta_abs_ct"],
                "summary_type": "basin_observation",
            })

        # Layer 2: Stratum summary medians + 95% bootstrap CIs
        for s, x_mid in zip(SNOW_STRATA, STRATA_MIDPOINTS):
            s_vals = piv[piv["snow_stratum"] == s]["delta_abs_ct"].values
            s_med = float(np.median(s_vals))
            ci_l, ci_h = compute_bootstrap_ci(s_vals, np.median, n_boot=2000, seed=42)

            ax_c.errorbar(
                x_mid + dx,
                s_med,
                yerr=[[s_med - ci_l], [ci_h - s_med]],
                fmt="none",
                ecolor=col,
                elinewidth=1.6,
                capsize=3.5,
                capthick=1.2,
                zorder=4,
            )
            ax_c.scatter(
                x_mid + dx,
                s_med,
                s=48,
                color=col,
                marker=marker,
                edgecolors="white",
                linewidths=1.1,
                zorder=5,
            )

            source_records.append({
                "panel": "c",
                "basin_id": "stratum_summary",
                "paradigm": p_name,
                "structure": np.nan,
                "model": np.nan,
                "period": "test",
                "snow_stratum": s,
                "frac_snow": x_mid,
                "metric": "delta_abs_ct_median",
                "value": s_med,
                "summary_type": "stratum_median_95ci",
            })

    # Correlation annotations in LOWER RIGHT
    sub_ic = df_test[df_test["paradigm"] == "IC-CMA-ES"].pivot(
        index=["basin_id", "snow_stratum", "frac_snow"],
        columns="structure",
        values="basin_median_Delta_CT",
    ).reset_index()
    sub_ic["delta_abs_ct"] = sub_ic["Base"].abs() - sub_ic["CN"].abs()
    r_ic, _ = stats.spearmanr(sub_ic["frac_snow"], sub_ic["delta_abs_ct"])

    sub_dpl = df_test[df_test["paradigm"] == "dPL-MLP"].pivot(
        index=["basin_id", "snow_stratum", "frac_snow"],
        columns="structure",
        values="basin_median_Delta_CT",
    ).reset_index()
    sub_dpl["delta_abs_ct"] = sub_dpl["Base"].abs() - sub_dpl["CN"].abs()
    r_dpl, _ = stats.spearmanr(sub_dpl["frac_snow"], sub_dpl["delta_abs_ct"])

    corr_box_text = (
        r"$\mathbf{IC:}$ Spearman $\rho = " + f"{r_ic:.3f}" + r"$ [$0.464$, $0.616$]" + "\n" +
        r"$\mathbf{dPL:}$ Spearman $\rho = " + f"{r_dpl:.3f}" + r"$ [$0.365$, $0.539$]"
    )
    ax_c.text(
        0.97,
        0.06,
        corr_box_text,
        transform=ax_c.transAxes,
        fontsize=8.2,
        color=COLOR_TEXT,
        ha="right",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#CBD5E1",
            alpha=0.95,
            lw=0.7,
        ),
        zorder=6,
    )

    # Panel (c) Legend in UPPER RIGHT
    leg_c_handles = [
        Line2D([0], [0], color=COLOR_IC, marker="o", markersize=6.0, markerfacecolor=COLOR_IC, markeredgecolor="white", markeredgewidth=1.0, lw=1.5, label="IC (CMA-ES)"),
        Line2D([0], [0], color=COLOR_DPL, marker="^", markersize=6.0, markerfacecolor=COLOR_DPL, markeredgecolor="white", markeredgewidth=1.0, lw=1.5, label="dPL (neural)"),
    ]
    ax_c.legend(
        handles=leg_c_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.94),
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=8.2,
        handlelength=1.4,
        framealpha=0.95,
    )

    ax_c.set_xlim(0.0, 1.0)
    ax_c.set_ylim(-10, 65)
    ax_c.set_xlabel(r"Snow activity ($\mathrm{frac}_{\mathrm{snow}}$)", fontsize=9.2)
    ax_c.set_ylabel(r"Reduction in $|CT|$ error, Base − CN (d)", fontsize=9.2)

    ax_c.text(0.025, 4.5, "near-zero in S1 (~0 d)", ha="center", va="bottom", fontsize=7.5, color="#64748B", style="italic", zorder=4)
    ax_c.text(0.75, 48.0, "~46–47 d in S5", ha="center", va="bottom", fontsize=7.5, color="#64748B", style="italic", zorder=4)

    # ═════════════════════════════════════════════════════════════════════════
    # ROW 3: Discrete Strata Timing Error & Large Error Prevalence (Panels d and e)
    # ═════════════════════════════════════════════════════════════════════════

    # --- PANEL (d): Signed timing error by structure (Scatter + Boxplot) ---
    apply_clean_spines(ax_d)
    ax_d.set_title(
        "(d) Signed timing error by structure",
        loc="left",
        fontsize=9.8,
        fontweight="bold",
        pad=6,
    )
    ax_d.axhline(0, color=COLOR_REF, linestyle="--", linewidth=0.8, zorder=1)

    box_cfgs = [
        ("Base", "IC-CMA-ES", -0.28, COLOR_BASE, True),
        ("Base", "dPL-MLP",   -0.17, COLOR_BASE, False),
        ("TGD",  "IC-CMA-ES", -0.05, COLOR_TGD,  True),
        ("TGD",  "dPL-MLP",   +0.05, COLOR_TGD,  False),
        ("CN",   "IC-CMA-ES", +0.17, COLOR_CN,   True),
        ("CN",   "dPL-MLP",   +0.28, COLOR_CN,   False),
    ]

    for i in range(len(SNOW_STRATA) - 1):
        ax_d.axvline(i + 0.5, color="#E2E8F0", lw=0.7, ls=":", zorder=1)

    for i, s in enumerate(SNOW_STRATA):
        for struct, p_name, dx, col, is_ic in box_cfgs:
            vals = df_test[
                (df_test["snow_stratum"] == s)
                & (df_test["structure"] == struct)
                & (df_test["paradigm"] == p_name)
            ]["basin_median_Delta_CT"].values
            med = float(np.median(vals))
            ci_l, ci_h = compute_bootstrap_ci(vals, np.median, n_boot=2000, seed=42)

            # Jittered background scatter
            np.random.seed(42 + i * 10 + (0 if is_ic else 1))
            jit = np.random.uniform(-0.022, 0.022, len(vals))
            ax_d.scatter(i + dx + jit, vals, s=6, color=col, alpha=0.18, edgecolors="none", rasterized=True, zorder=2)

            # Boxplot
            ax_d.boxplot(
                vals,
                positions=[i + dx],
                widths=0.085,
                patch_artist=True,
                showfliers=False,
                whis=[5, 95],
                zorder=4,
                boxprops=dict(
                    facecolor=col if is_ic else "white",
                    edgecolor=col,
                    linewidth=1.2,
                    linestyle="-" if is_ic else "--",
                    alpha=0.55 if is_ic else 0.95,
                ),
                whiskerprops=dict(color=col, linewidth=1.1, linestyle="-" if is_ic else "--"),
                capprops=dict(color=col, linewidth=1.1),
                medianprops=dict(color=COLOR_DARK_NEUTRAL if not is_ic else "#1E293B", linewidth=1.6),
            )

            source_records.append({
                "panel": "d",
                "basin_id": "stratum_summary",
                "paradigm": p_name,
                "structure": struct,
                "model": f"XAJ-{struct}",
                "period": "test",
                "snow_stratum": s,
                "frac_snow": np.nan,
                "metric": "signed_ct_error_median",
                "value": med,
                "summary_type": "stratum_median_95ci",
            })

    # Panel (d) Legend in LOWER LEFT
    leg_d_handles = [
        Patch(facecolor=COLOR_BASE, edgecolor=COLOR_BASE, label="Base (Orange)"),
        Patch(facecolor=COLOR_TGD,  edgecolor=COLOR_TGD,  label="TGD (Green)"),
        Patch(facecolor=COLOR_CN,   edgecolor=COLOR_CN,   label="CN (Blue)"),
        Line2D([0], [0], color=COLOR_DARK_NEUTRAL, ls="-", lw=1.2, label="IC (filled)"),
        Line2D([0], [0], color=COLOR_DARK_NEUTRAL, ls="--", lw=1.2, label="dPL (hollow)"),
    ]
    ax_d.legend(
        handles=leg_d_handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.04),
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=7.2,
        ncol=2,
        framealpha=0.92,
    )

    ax_d.set_xticks(range(5))
    ax_d.set_xticklabels(SNOW_STRATA, fontsize=8.8)
    ax_d.set_xlabel("Snow stratum", fontsize=9.0)
    ax_d.set_ylabel("Signed CT error (d)", fontsize=9.0)
    ax_d.set_xlim(-0.5, 4.5)
    ax_d.set_ylim(-75, 25)
    ax_d.grid(True, axis="y", linestyle=":", alpha=0.35, color=COLOR_LIGHT_REF)

    # --- PANEL (e): Common-pass subset prevalence of large timing error (|CT| >= 15 d) ---
    apply_clean_spines(ax_e)
    ax_e.set_title(
        "(e) Large timing error among KGE-qualified basins",
        loc="left",
        fontsize=9.8,
        fontweight="bold",
        pad=6,
    )
    ax_e.text(
        0.04,
        0.94,
        "Common-pass subset (all 3 structures KGE ≥ 0.60)\nIC n = 321; dPL n = 331",
        transform=ax_e.transAxes,
        fontsize=7.2,
        color="#4A5568",
        va="top",
        zorder=5,
    )

    y_map = {"CN": 0, "TGD": 1, "Base": 2}
    for y_idx in range(3):
        ax_e.axhline(y_idx, color="#F1F5F9", lw=14, zorder=0)
        ax_e.axhline(y_idx, color="#E2E8F0", lw=0.6, ls=":", zorder=1)

    for struct in ["Base", "TGD", "CN"]:
        y_base = y_map[struct]
        c = MODEL_COLORS[struct]
        m = MODEL_MARKERS[struct]

        for p_name, y_off, is_ic in [("IC-CMA-ES", 0.15, True), ("dPL-MLP", -0.15, False)]:
            sub_p = df_test[df_test["paradigm"] == p_name]
            piv_kge = sub_p.pivot(index="basin_id", columns="structure", values="KGE")
            piv_ct = sub_p.pivot(index="basin_id", columns="structure", values="basin_median_Delta_CT")

            # Common-pass criteria: all 3 structures with KGE >= 0.60
            cpass = piv_kge[
                (piv_kge["Base"] >= 0.60)
                & (piv_kge["TGD"] >= 0.60)
                & (piv_kge["CN"] >= 0.60)
            ].index
            n_cpass = len(cpass)
            ct_sub = piv_ct.loc[cpass, struct]
            n_large = int((ct_sub.abs() >= 15.0).sum())
            pct = (n_large / n_cpass) * 100.0

            # Bootstrap 95% CI on prevalence
            rng = np.random.default_rng(42)
            pcts = np.empty(2000, dtype=float)
            for b_idx in range(2000):
                sampled = rng.choice(ct_sub.values, size=n_cpass, replace=True)
                pcts[b_idx] = (np.abs(sampled) >= 15.0).sum() / n_cpass * 100.0
            ci_l, ci_h = np.percentile(pcts, [2.5, 97.5])

            y_pos = y_base + y_off
            ax_e.errorbar(
                pct,
                y_pos,
                xerr=[[pct - ci_l], [ci_h - pct]],
                fmt="none",
                ecolor=c,
                elinewidth=1.5,
                capsize=3.2,
                capthick=1.1,
                zorder=3,
            )
            if is_ic:
                ax_e.scatter(
                    pct,
                    y_pos,
                    s=45,
                    color=c,
                    marker=m,
                    edgecolors="white",
                    linewidths=1.0,
                    zorder=4,
                )
            else:
                ax_e.scatter(
                    pct,
                    y_pos,
                    s=45,
                    color="white",
                    marker=m,
                    edgecolors=c,
                    linewidths=1.5,
                    zorder=4,
                )

            ax_e.text(
                ci_h + 0.6,
                y_pos,
                f"{pct:.1f}%",
                va="center",
                ha="left",
                fontsize=7.8,
                fontweight="bold",
                color=c,
                zorder=5,
            )

            source_records.append({
                "panel": "e",
                "basin_id": "common_pass_summary",
                "paradigm": p_name,
                "structure": struct,
                "model": f"XAJ-{struct}",
                "period": "test",
                "snow_stratum": np.nan,
                "frac_snow": np.nan,
                "metric": "large_timing_error_prevalence_pct",
                "value": pct,
                "summary_type": "common_pass_prevalence_95ci",
            })

    # Panel (e) Legend for filled vs hollow
    leg_e_handles = [
        Line2D([0], [0], color=COLOR_DARK_NEUTRAL, marker="o", markersize=5.5, markerfacecolor=COLOR_DARK_NEUTRAL, markeredgecolor="white", lw=0, label="IC (filled)"),
        Line2D([0], [0], color=COLOR_DARK_NEUTRAL, marker="o", markersize=5.5, markerfacecolor="white", markeredgecolor=COLOR_DARK_NEUTRAL, markeredgewidth=1.3, lw=0, label="dPL (hollow)"),
    ]
    ax_e.legend(
        handles=leg_e_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.05),
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=7.5,
        framealpha=0.92,
    )

    ax_e.set_yticks([0, 1, 2])
    ax_e.set_yticklabels(["CN", "TGD", "Base"], fontsize=9.0)
    ax_e.set_xlim(0, 24)
    ax_e.set_ylim(-0.5, 2.5)
    ax_e.set_xlabel(r"Prevalence of $|CT| \geq 15\ \mathrm{d}$ (%)", fontsize=9.0)
    ax_e.grid(True, axis="x", linestyle=":", alpha=0.35, color=COLOR_LIGHT_REF)

    # 3. Save Figure PNG (600 dpi, bbox_inches='tight')
    canonical_png = out_fig_dir / "Figure2_R1_final.png"
    plt.savefig(canonical_png, dpi=600, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()

    # 4. Save Source Data CSV
    df_source = pd.DataFrame(source_records)
    source_csv = out_fig_dir / "Figure2_R1_source_data.csv"
    df_source.to_csv(source_csv, index=False)

    file_size_mb = os.path.getsize(canonical_png) / (1024 * 1024)
    print("Figure 2 restructured successfully (5-panel merged layout)!")
    print(f"  PNG: {canonical_png} ({file_size_mb:.2f} MB)")
    print(f"  Source data: {source_csv}")

    return canonical_png


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render R1 Figure 2.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for generated figure.")
    args = parser.parse_args()
    main(args.out_dir)
