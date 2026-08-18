"""Generate Figure 7: Environmental and Process Localization with Real-Basin Paired Evidence (R4).

6-panel asymmetric composite layout following HESS / WRR guidelines.
Main figure shows the canonical dPL seed (42) and the fused-IC sensitivity regime;
seed-123 is retained only as replicability evidence in the supplement, not in Figure 7.

- (a) Main full-width: Snow-burden dependence (D01–D10) — canonical dPL + IC fused
- (b)(c) Phase fingerprints by regime (canonical dPL, IC fused) — shared grammar
- (d)(e) Real-basin paired scatter: Base vs CN individual anomaly correlation
          (d) Active melt; (e) Summer dry-down, with Q3 subset emphasized
- (f) Thin full-width robustness rail: performance control, HUC02 omission, and SWE trimming

HUC02 region omission uses the authoritative CAMELS-US Daymet layout:
  G:\\Dataset\\CAMELS_US\\basin_mean_forcing\\daymet/<huc02>/<basin_id>.
All 531 current basins joined uniquely (0 missing, 0 conflicts), covering 18 HUC02
regions. No HUC02 value is inferred from a gauge-ID prefix. The join audit is saved
as huc02_daymet_join.json/csv beside the official R4 results.

The current 531-basin Snow-17 SWE-burden upper quartile is computed from the official
paired basin table (Q3 boundary ≈ 133.4 mm; n=133).

Outputs:
    manuscript/figures/figure7_r4_soil_consistency.png (300 DPI, PNG only - no PDF)

Data semantics (from results/r4_phase1_soil_official/):
  robustness_swe_decile_shape.csv — bootstrap 95% CI of decile medians
  robustness_process_phase_consistency.csv — per-basin per-phase anomaly correlation
  robustness_controlled_regressions.csv — beta1 with bootstrap 95% CI
  robustness_extreme_swe_trimming.csv — Spearman rho(SWE, Delta Anom.) for full/trimmed samples
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript.r4.common import default_results_root  # noqa: E402
from manuscript.r4.robustness_analysis import bootstrap_median_ci  # noqa: E402
from manuscript.scripts.shared.r1_plot_style import (  # noqa: E402
    apply_clean_spines,
    setup_publication_style,
)

FIGURES_DIR = HERE.parents[2] / "figures"

# ── Frozen visual encodings ──────────────────────────────────────────────────
# Canonical dPL seed + IC fused only; seed-123 stays in the supplement.
CANONICAL_DPL = "dPL_seed42"
REGIME_CFG = {
    CANONICAL_DPL: {
        "label": "dPL (seed 42)",
        "color": "#882E72",
        "marker": "o",
        "ls": "-",
        "lw": 1.5,
    },
    "IC_fused": {
        "label": "IC (fused)",
        "color": "#333333",
        "marker": "D",
        "ls": "-.",
        "lw": 1.4,
    },
}
REGIMES = [CANONICAL_DPL, "IC_fused"]
REG_SHORT = ["dPL-42", "IC fused"]

# Q3 is computed from the current 531-basin official paired table at plot time.
Q3_QUANTILE = 0.75

PHASE_ORDER = [
    "Phase_1_Snow_Accumulation",
    "Phase_2_Active_Melt_Recharge",
    "Phase_3_Post_Melt_Transition",
    "Phase_4_Summer_Dry_Down",
]
PHASE_LABELS = [
    "Accum.\n(Oct\u2013Feb)",
    "Active melt\n(Mar\u2013May)",
    "Post-melt\n(Jun\u2013Jul)",
    "Dry-down\n(Aug\u2013Sep)",
]

MINUS = "\u2212"
DELTA_LABEL = f"\u0394 anomaly correlation (CN {MINUS} Base)"
ABS_BASE_LABEL = "Base anomaly correlation with SM$_{100}$"
ABS_CN_LABEL = "CN anomaly correlation with SM$_{100}$"


def _phase_fingerprint(ax, med, lo, hi, regime_key):
    """Plot a dot-whisker phase fingerprint for one regime."""
    cfg = REGIME_CFG[regime_key]
    x = np.arange(4)
    y = np.array([med[(regime_key, p)] for p in PHASE_ORDER])
    ylo = np.array([lo[(regime_key, p)] for p in PHASE_ORDER])
    yhi = np.array([hi[(regime_key, p)] for p in PHASE_ORDER])

    ax.axhline(0, color="#999999", ls="--", lw=0.8, zorder=1)
    # Very light emphasis of the Active Melt column (shared grammar, no text)
    ax.axvspan(0.5, 1.5, color="#EAF1F8", alpha=0.45, zorder=0)
    ax.errorbar(
        x,
        y,
        yerr=[y - ylo, yhi - y],
        fmt="none",
        ecolor=cfg["color"],
        elinewidth=0.9,
        capsize=2.4,
        capthick=0.9,
        alpha=0.85,
        zorder=3,
    )
    ax.plot(
        x,
        y,
        color=cfg["color"],
        marker=cfg["marker"],
        markersize=5.0,
        ls="none",
        zorder=4,
    )


def generate_figure7(results_root: Path, out_dir: Path) -> Path:
    setup_publication_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    r4_dir = results_root / "r4_phase1_soil_official"

    # 1. Load data
    df_swe = pd.read_csv(r4_dir / "robustness_swe_decile_shape.csv")
    df_phase = pd.read_csv(r4_dir / "robustness_process_phase_consistency.csv")
    df_paired = pd.read_csv(r4_dir / "paired_structural_effects.csv")
    df_reg = pd.read_csv(r4_dir / "robustness_controlled_regressions.csv")
    df_loro = pd.read_csv(r4_dir / "robustness_leave_one_region_out.csv")
    df_trim = pd.read_csv(r4_dir / "robustness_extreme_swe_trimming.csv")

    # Use the actual upper quartile of the current 531-basin SWE burden.
    canonical_swe = df_paired[df_paired["regime"] == CANONICAL_DPL].drop_duplicates(
        "basin_id"
    )
    q3_swe_mm = float(canonical_swe["snow_burden_swe_mm"].quantile(Q3_QUANTILE))
    q3_n = int((canonical_swe["snow_burden_swe_mm"] >= q3_swe_mm).sum())

    # 2. Precompute phase bootstrap medians + CI for (b, c)
    pmed, plo, phi = {}, {}, {}
    for reg in REGIMES:
        df_r = df_phase[df_phase["regime"] == reg]
        for p in PHASE_ORDER:
            vals = df_r.loc[df_r["phase_name"] == p, "delta_anomaly_corr"].to_numpy(
                dtype=np.float64
            )
            med, lo, hi = bootstrap_median_ci(vals)
            pmed[(reg, p)] = med
            plo[(reg, p)] = lo
            phi[(reg, p)] = hi

    # 3. Setup asymmetric 4-tier GridSpec
    fig = plt.figure(figsize=(7.2, 9.0))
    gs = fig.add_gridspec(
        4,
        1,
        height_ratios=[2.0, 1.4, 2.1, 1.15],
        hspace=0.42,
        left=0.08,
        right=0.96,
        top=0.97,
        bottom=0.045,
    )

    # -----------------------------------------------------------------------
    # (a) Main: Snow-burden dependence (where?)
    # -----------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0])
    apply_clean_spines(ax_a)
    ax_a.axhline(0, color="#999999", ls="--", lw=0.8, zorder=1)
    ax_a.axvspan(7.5, 9.5, color="#EAF1F8", alpha=0.55, zorder=0)
    ax_a.text(
        8.5,
        0.222,
        "Upper SWE tail",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#4A6FA5",
        zorder=5,
    )

    x_dec = np.arange(10)
    for reg in REGIMES:
        cfg = REGIME_CFG[reg]
        df_sub = df_swe[df_swe["regime"] == reg].sort_values("decile")
        ym = df_sub["delta_anomaly_corr_median"].to_numpy()
        ylo = df_sub["delta_anomaly_corr_ci_lower"].to_numpy()
        yhi = df_sub["delta_anomaly_corr_ci_upper"].to_numpy()
        ci_handle = ax_a.errorbar(
            x_dec,
            ym,
            yerr=[ym - ylo, yhi - ym],
            fmt="none",
            ecolor=cfg["color"],
            elinewidth=0.8,
            capsize=2.2,
            capthick=0.8,
            alpha=0.85,
            zorder=3,
        )
        ax_a.plot(
            x_dec,
            ym,
            color=cfg["color"],
            marker=cfg["marker"],
            markersize=4.6,
            lw=cfg["lw"],
            ls=cfg["ls"],
            label=cfg["label"],
            zorder=4,
        )

    ax_a.set_xticks(x_dec)
    ax_a.set_xticklabels([f"D{i + 1:02d}" for i in range(10)], fontsize=7.8)
    ax_a.set_xlabel("Snow-17 SWE burden decile", fontsize=8.2)
    ax_a.set_ylabel(DELTA_LABEL, fontsize=8.5)
    ax_a.set_title(
        "(a) Snow-burden dependence", loc="left", fontweight="bold", fontsize=9.0
    )
    ax_a.set_ylim(-0.04, 0.25)
    ax_a.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])

    leg_a = [
        plt.Line2D(
            [0],
            [0],
            color=REGIME_CFG[r]["color"],
            marker=REGIME_CFG[r]["marker"],
            ls=REGIME_CFG[r]["ls"],
            lw=1.4,
            markersize=4.6,
            label=REGIME_CFG[r]["label"],
        )
        for r in REGIMES
    ]
    ci_handle.set_label("95 % bootstrap CI of the median")
    ax_a.legend(
        handles=leg_a + [ci_handle],
        loc="upper left",
        frameon=True,
        facecolor="#FFFFFF",
        framealpha=0.92,
        fontsize=7.4,
    )

    # -----------------------------------------------------------------------
    # (b, c) Phase fingerprints by regime (when?)
    # -----------------------------------------------------------------------
    gs_mid = gs[1].subgridspec(1, 2, wspace=0.22, width_ratios=[1.0, 1.0])
    fp_axes = [fig.add_subplot(gs_mid[i]) for i in range(2)]
    fp_subtitles = ["dPL seed 42", "IC fused"]

    # Shared y-limits across (b, c)
    all_lo_vals = [plo[(r, p)] for r in REGIMES for p in PHASE_ORDER]
    all_hi_vals = [phi[(r, p)] for r in REGIMES for p in PHASE_ORDER]
    fp_ylo = min(-0.06, min(all_lo_vals) * 1.15)
    fp_yhi = max(all_hi_vals) * 1.18

    for i, (ax, reg, sub) in enumerate(zip(fp_axes, REGIMES, fp_subtitles)):
        apply_clean_spines(ax)
        _phase_fingerprint(ax, pmed, plo, phi, reg)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(PHASE_LABELS, fontsize=6.0)
        ax.set_ylim(fp_ylo, fp_yhi)
        ax.set_yticks(np.arange(-0.05, 0.401, 0.10))
        if i == 0:
            ax.set_ylabel(DELTA_LABEL, fontsize=7.5)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(
            f"({chr(98 + i)}) {sub}", loc="left", fontweight="bold", fontsize=8.2
        )

    # -----------------------------------------------------------------------
    # (d, e) Real-basin paired evidence: Active melt vs Summer dry-down
    # -----------------------------------------------------------------------
    gs_scat = gs[2].subgridspec(1, 2, wspace=0.28, width_ratios=[1.0, 1.0])
    ax_d = fig.add_subplot(gs_scat[0])
    ax_e = fig.add_subplot(gs_scat[1])

    SCAT_LIM = (-0.45, 1.00)
    SCAT_TICKS = [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    scat_panels = []
    for ax, ph_key, letter, ph_title, note_col in [
        (ax_d, "Phase_2_Active_Melt_Recharge", "d", "Active melt", "#1B4F72"),
        (ax_e, "Phase_4_Summer_Dry_Down", "e", "Summer dry-down", "#555555"),
    ]:
        phase_subset = df_phase[df_phase["phase_name"] == ph_key]
        q3_parts = []
        for reg in REGIMES:
            q3 = phase_subset[
                (phase_subset["regime"] == reg)
                & (phase_subset["snow_burden_swe_mm"] >= q3_swe_mm)
            ]
            frac = 100.0 * (q3["cn_anomaly_corr"] > q3["base_anomaly_corr"]).mean()
            q3_parts.append(f"{REG_SHORT[REGIMES.index(reg)]} {frac:.0f}%")
        note_text = (
            f"Q3 (SWE ≥ {q3_swe_mm:.1f} mm, n = {q3_n})\n"
            f"CN > Base: {', '.join(q3_parts)}"
        )
        scat_panels.append((ax, ph_key, letter, ph_title, note_text, note_col))

    for ax, ph_key, letter, ph_title, note_text, note_col in scat_panels:
        apply_clean_spines(ax)
        ax.plot([-0.5, 1.05], [-0.5, 1.05], color="#888888", ls="--", lw=0.9, zorder=1)
        ax.axhline(0, color="#E8E8E8", ls=":", lw=0.7, zorder=0)
        ax.axvline(0, color="#E8E8E8", ls=":", lw=0.7, zorder=0)

        df_ph = df_phase[df_phase["phase_name"] == ph_key]

        for reg in REGIMES:
            sub = df_ph[df_ph["regime"] == reg]
            xb = sub["base_anomaly_corr"].to_numpy()
            yc = sub["cn_anomaly_corr"].to_numpy()
            hs = sub["snow_burden_swe_mm"].to_numpy() >= q3_swe_mm
            cfg = REGIME_CFG[reg]

            # Ordinary catchments: light transparency
            ax.scatter(
                xb[~hs],
                yc[~hs],
                marker=cfg["marker"],
                facecolors=cfg["color"],
                edgecolors="none",
                s=14,
                alpha=0.30,
                zorder=3,
            )
            # Upper snow-burden quartile (Q3): emphasize the high-snow evidence.
            if hs.sum() > 0:
                ax.scatter(
                    xb[hs],
                    yc[hs],
                    marker=cfg["marker"],
                    facecolors=cfg["color"],
                    edgecolors=cfg["color"],
                    s=30,
                    alpha=0.75,
                    zorder=4,
                )

        ax.set_xlim(*SCAT_LIM)
        ax.set_ylim(*SCAT_LIM)
        ax.set_xticks(SCAT_TICKS)
        ax.set_yticks(SCAT_TICKS)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(ABS_BASE_LABEL, fontsize=7.8)
        ax.set_ylabel(ABS_CN_LABEL, fontsize=7.8)
        ax.set_title(
            f"({letter}) {ph_title}", loc="left", fontweight="bold", fontsize=8.5
        )

        ax.text(
            0.03,
            0.96,
            note_text,
            transform=ax.transAxes,
            fontsize=6.4,
            va="top",
            ha="left",
            color=note_col,
            zorder=5,
        )

    # Single regime legend for scatter panels (panel e, lower-right)
    legend_scat = [
        Line2D(
            [0],
            [0],
            marker=REGIME_CFG[r]["marker"],
            color="w",
            markerfacecolor=REGIME_CFG[r]["color"],
            markersize=5.0,
            label=REGIME_CFG[r]["label"],
        )
        for r in REGIMES
    ]
    q3_patch = Patch(
        fc="none",
        ec="#555555",
        lw=1.0,
        label=f"Upper SWE quartile (Q3; SWE ≥ {q3_swe_mm:.1f} mm)",
    )
    ax_e.legend(
        handles=legend_scat + [q3_patch],
        loc="lower right",
        frameon=True,
        facecolor="#FFFFFF",
        framealpha=0.90,
        fontsize=6.6,
    )

    # -----------------------------------------------------------------------
    # (f) Robustness checks — thin full-width synthesis rail
    # -----------------------------------------------------------------------
    gs_rob_row = gs[3].subgridspec(2, 1, height_ratios=[0.38, 1.0], hspace=0.15)
    gs_rob_axes = gs_rob_row[1].subgridspec(1, 3, wspace=0.28)
    rail_y = np.arange(len(REGIMES))

    # (f1) Performance control
    ax_f1 = fig.add_subplot(gs_rob_axes[0])
    apply_clean_spines(ax_f1)
    ax_f1.axvline(0, color="#999999", ls="--", lw=0.8, zorder=1)
    df_ra = df_reg[df_reg["target_metric"] == "delta_anomaly_corr"]
    for idx, reg in enumerate(REGIMES):
        row = df_ra[df_ra["regime"] == reg].iloc[0]
        b1, lo, hi = (
            row["beta1_swe_burden_std"],
            row["beta1_ci_lower"],
            row["beta1_ci_upper"],
        )
        col = REGIME_CFG[reg]["color"]
        ax_f1.errorbar(
            b1,
            rail_y[idx],
            xerr=[[b1 - lo], [hi - b1]],
            fmt="o",
            color=col,
            ecolor=col,
            elinewidth=1.1,
            capsize=2.5,
            markersize=4.6,
            zorder=3,
        )
    ax_f1.set_yticks(rail_y)
    ax_f1.set_yticklabels(REG_SHORT, fontsize=6.8)
    ax_f1.invert_yaxis()
    ax_f1.set_xlabel("Controlled SWE $\\beta_1$ [std.]", fontsize=6.8)
    ax_f1.set_xlim(-0.03, 0.07)
    ax_f1.set_xticks([-0.02, 0.00, 0.02, 0.04, 0.06])
    ax_f1.set_title(
        "After controlling for Delta KGE", loc="left", fontsize=6.8, fontweight="bold"
    )

    # (f2) Authoritative leave-one-HUC02-out (18 regions)
    ax_f2 = fig.add_subplot(gs_rob_axes[1])
    apply_clean_spines(ax_f2)
    ax_f2.axvline(0, color="#999999", ls="--", lw=0.8, zorder=1)
    for idx, reg in enumerate(REGIMES):
        sub = df_loro[df_loro["regime"] == reg]
        full = sub[sub["dropped_region"] == "NONE (Full Sample)"][
            "rho_delta_anomaly_swe"
        ].iloc[0]
        loro = sub[sub["dropped_region"] != "NONE (Full Sample)"]
        rmin, rmax = (
            loro["rho_delta_anomaly_swe"].min(),
            loro["rho_delta_anomaly_swe"].max(),
        )
        col = REGIME_CFG[reg]["color"]
        ax_f2.plot(
            [rmin, rmax],
            [rail_y[idx], rail_y[idx]],
            color=col,
            lw=1.6,
            alpha=0.55,
            solid_capstyle="butt",
            zorder=2,
        )
        ax_f2.plot(
            [rmin, rmax],
            [rail_y[idx], rail_y[idx]],
            marker="|",
            color=col,
            ms=4.0,
            alpha=0.55,
            zorder=2,
        )
        ax_f2.plot(full, rail_y[idx], marker="*", color=col, ms=6.5, zorder=4)
    ax_f2.set_yticks(rail_y)
    ax_f2.set_yticklabels([])
    ax_f2.invert_yaxis()
    ax_f2.set_xlabel("Spearman $\\rho$(SWE, $\\Delta$Anom.)", fontsize=6.8)
    ax_f2.set_xlim(0.05, 0.45)
    ax_f2.set_xticks([0.10, 0.20, 0.30, 0.40])
    ax_f2.set_title(
        "After leaving out one HUC02 region",
        loc="left",
        fontsize=6.8,
        fontweight="bold",
    )
    legend_f2 = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#555555",
            ms=5.5,
            label="Full sample",
        ),
        Line2D([0], [0], color="#555555", lw=1.6, label="LORO range (18 regions)"),
    ]
    ax_f2.legend(
        handles=legend_f2,
        loc="upper left",
        frameon=True,
        facecolor="#FFFFFF",
        fontsize=5.6,
    )

    # (f3) Extreme-SWE trimming
    ax_f3 = fig.add_subplot(gs_rob_axes[2])
    apply_clean_spines(ax_f3)
    ax_f3.axvline(0, color="#999999", ls="--", lw=0.8, zorder=1)
    trim_offsets = [-0.16, 0.16]
    trim_markers = ["o", "^", "s"]
    trim_schemes = ["full_sample", "trim_top_1pct", "trim_top_5pct"]
    for idx, reg in enumerate(REGIMES):
        col = REGIME_CFG[reg]["color"]
        sub_trim = df_trim[df_trim["regime"] == reg]
        base_y = rail_y[idx]
        for t_idx, scheme in enumerate(trim_schemes):
            val = sub_trim[sub_trim["trimming_scheme"] == scheme][
                "rho_delta_anomaly_swe"
            ].iloc[0]
            # stack three markers vertically per regime row (small y offsets)
            yoff = (t_idx - 1) * 0.16
            ax_f3.plot(
                val,
                base_y + yoff,
                marker=trim_markers[t_idx],
                color=col,
                ms=4.2,
                zorder=3,
            )
    ax_f3.set_yticks(rail_y)
    ax_f3.set_yticklabels([])
    ax_f3.invert_yaxis()
    ax_f3.set_xlabel("Spearman $\\rho$(SWE, $\\Delta$Anom.)", fontsize=6.8)
    ax_f3.set_xlim(0.05, 0.40)
    ax_f3.set_xticks([0.10, 0.20, 0.30, 0.40])
    ax_f3.set_title(
        "After removing SWE extremes", loc="left", fontsize=6.8, fontweight="bold"
    )
    legend_f3 = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#555555",
            ms=4.0,
            label="Full",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="#555555",
            ms=4.0,
            label="Trim 1 %",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#555555",
            ms=4.0,
            label="Trim 5 %",
        ),
    ]
    ax_f3.legend(
        handles=legend_f3,
        loc="upper right",
        frameon=True,
        facecolor="#FFFFFF",
        fontsize=5.4,
    )

    # Unified title for panel (f) strip
    pos_top = gs_rob_row[0].get_position(fig)
    fig.text(
        0.08,
        pos_top.y1,
        "(f) Robustness checks",
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="top",
    )

    # 5. Save PNG only (300 dpi)
    png_path = out_dir / "figure7_r4_soil_consistency.png"
    plt.savefig(png_path, dpi=300)
    plt.close()

    print(f"Generated Figure 7 (PNG only, 300 dpi):\n  {png_path}")
    return png_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    results_root = args.results_root or default_results_root()
    generate_figure7(results_root, args.out_dir)
