"""Generate Figure 4: Real-Basin Shared Soil-Water State Consistency (R4).

Publication-ready 4-panel figure following HESS / Copernicus guidelines:
- Panel A: Response Shape across SWE Deciles & Theil-Sen regression with bootstrap 95% CI
- Panel B: Process-Phase Conditioned Soil Moisture Consistency (4 phases)
- Panel C: Real-catchment hydrograph snapshot for Colorado Rockies (09065500)
- Panel D: Timing error distributions (Spring wet-up & Soil-water peak timing)

Outputs:
    manuscript/figures/figure4_r4_soil_consistency.png
    manuscript/figures/figure4_r4_soil_consistency.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript.scripts.r1_plot_style import (  # noqa: E402
    MODEL_COLORS,
    apply_clean_spines,
    setup_publication_style,
)
from r4.common import default_results_root  # noqa: E402

FIGURES_DIR = HERE.parents[0] / "figures"


def generate_figure4(results_root: Path, out_dir: Path, regime: str = "dPL_seed42") -> Path:
    setup_publication_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    r4_dir = results_root / "r4_phase1_soil_official"
    caravan_dir = results_root / "r4_caravan_soil_reference_v1"

    # Load data
    df_dec = pd.read_csv(r4_dir / "robustness_swe_decile_shape.csv")
    df_dec_reg = df_dec[df_dec["regime"] == regime].sort_values("decile")

    df_paired = pd.read_csv(r4_dir / "paired_structural_effects.csv")
    df_p_reg = df_paired[df_paired["regime"] == regime]

    df_phase = pd.read_csv(r4_dir / "robustness_process_phase_consistency.csv")
    df_ph_reg = df_phase[df_phase["regime"] == regime]

    df_timing = pd.read_csv(r4_dir / "timing_metrics_basin_summary.csv")
    df_t_reg = df_timing[df_timing["regime"] == regime]

    caravan = np.load(caravan_dir / "caravan_soil_ensemble.npz")
    basin_ids = [str(b).zfill(8) for b in caravan["basin_ids"]]
    test_sl = slice(int(caravan["test_slice_start"]), int(caravan["test_slice_stop"]))
    test_dates = pd.to_datetime(caravan["dates"][test_sl])
    sm100_all = caravan["SM100"][:, test_sl]
    swe_all = caravan["caravan_swe"][:, test_sl]

    prefix = "official_dpl"
    seed = int(regime.split("seed")[-1]) if "seed" in regime else 42
    base_npz = np.load(results_root / f"r4_{prefix}_XAJ_seed{seed}" / f"{prefix}_XAJ_seed{seed}_full_arrays.npz")
    cn_npz = np.load(results_root / f"r4_{prefix}_XAJ_CN_seed{seed}" / f"{prefix}_XAJ_CN_seed{seed}_full_arrays.npz")

    w_base_all = base_npz["wu"][:, test_sl] + base_npz["wl"][:, test_sl] + base_npz["wd"][:, test_sl]
    w_cn_all = cn_npz["wu"][:, test_sl] + cn_npz["wl"][:, test_sl] + cn_npz["wd"][:, test_sl]
    g_cn_all = cn_npz["G"][:, test_sl]

    # Create 2x2 multi-panel layout (width 7.2 in for 2-column Copernicus journals)
    fig = plt.figure(figsize=(7.2, 5.8))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25, left=0.08, right=0.96, top=0.94, bottom=0.08)

    # -----------------------------------------------------------------------
    # Panel A: Response Shape Across SWE Deciles
    # -----------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    apply_clean_spines(ax_a)

    x_dec = np.arange(len(df_dec_reg))
    y_anom = df_dec_reg["delta_anomaly_corr_median"].values
    y_anom_lo = df_dec_reg["delta_anomaly_corr_ci_lower"].values
    y_anom_hi = df_dec_reg["delta_anomaly_corr_ci_upper"].values
    y_7d = df_dec_reg["delta_7d_corr_median"].values

    ax_a.axhline(0, color="#999999", linestyle="--", linewidth=0.8, zorder=1)

    # Shaded CI band for anomaly delta
    ax_a.fill_between(x_dec, y_anom_lo, y_anom_hi, color=MODEL_COLORS["CN"], alpha=0.18, label="95% Bootstrap CI")
    ax_a.plot(x_dec, y_anom, color=MODEL_COLORS["CN"], marker="s", markersize=4.5, linewidth=1.4, label=r"$\Delta$ Anomaly Corr (CN$-$Base)")
    ax_a.plot(x_dec, y_7d, color=MODEL_COLORS["TGD"], marker="^", markersize=4.5, linewidth=1.2, linestyle="-.", label=r"$\Delta$ 7-day Corr (CN$-$Base)")

    ax_a.set_xticks(x_dec)
    ax_a.set_xticklabels(df_dec_reg["decile"], fontsize=7.2)
    ax_a.set_xlabel("Snow-17 SWE Burden Decile (D01 .. D10)")
    ax_a.set_ylabel(r"Paired State Gain $\Delta C(\mathrm{CN}-\mathrm{Base})$")
    ax_a.set_title("(a) State Consistency Gain Across SWE Deciles", loc="left", fontweight="bold", fontsize=8.5)
    ax_a.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", framealpha=0.9, fontsize=7.0)

    # -----------------------------------------------------------------------
    # Panel B: 4-Phase Conditioned State Consistency
    # -----------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    apply_clean_spines(ax_b)

    phase_order = [
        "Phase_1_Snow_Accumulation",
        "Phase_2_Active_Melt_Recharge",
        "Phase_3_Post_Melt_Transition",
        "Phase_4_Summer_Dry_Down",
    ]
    phase_labels = ["1. Accumulation", "2. Active Melt", "3. Post-Melt", "4. Summer Dry"]

    p_base_vals = [df_ph_reg[df_ph_reg["phase_name"] == p]["base_anomaly_corr"].median() for p in phase_order]
    p_cn_vals = [df_ph_reg[df_ph_reg["phase_name"] == p]["cn_anomaly_corr"].median() for p in phase_order]

    x_p = np.arange(len(phase_labels))
    width = 0.32

    ax_b.bar(x_p - width / 2, p_base_vals, width, label="Base (Omitted Snow)", color=MODEL_COLORS["Base"], alpha=0.88, edgecolor="none")
    ax_b.bar(x_p + width / 2, p_cn_vals, width, label="CN (Explicit Snow)", color=MODEL_COLORS["CN"], alpha=0.88, edgecolor="none")

    # Annotate delta on active melt
    delta_melt = p_cn_vals[1] - p_base_vals[1]
    ax_b.annotate(
        f"+{delta_melt:.2f}\n(Gain)",
        xy=(1 + width / 2, p_cn_vals[1] + 0.02),
        xytext=(1, p_cn_vals[1] + 0.12),
        ha="center", fontsize=7.2, fontweight="bold", color=MODEL_COLORS["CN"],
        arrowprops=dict(arrowstyle="->", color=MODEL_COLORS["CN"], lw=0.9),
    )

    ax_b.set_xticks(x_p)
    ax_b.set_xticklabels(phase_labels, fontsize=7.2)
    ax_b.set_ylabel(r"Monthly Anomaly Corr vs $\mathrm{SM}_{100}$")
    ax_b.set_title("(b) State Consistency by External Process Phase", loc="left", fontweight="bold", fontsize=8.5)
    ax_b.set_ylim(0.0, 1.0)
    ax_b.legend(loc="lower left", frameon=True, facecolor="#FFFFFF", framealpha=0.9, fontsize=7.0)

    # -----------------------------------------------------------------------
    # Panel C: Real Catchment Hydrograph Snapshot (Colorado Rockies, 09065500)
    # -----------------------------------------------------------------------
    # 2-year window (WY 2005..2006: 2004-10-01 to 2006-09-30)
    b_target = "09065500"
    b_idx = basin_ids.index(b_target)
    snap_mask = (test_dates >= pd.Timestamp("2004-10-01")) & (test_dates <= pd.Timestamp("2006-09-30"))
    snap_dates = test_dates[snap_mask]

    ax_c = fig.add_subplot(gs[1, 0])
    apply_clean_spines(ax_c)

    # Normalize soil moisture series to [0, 1] relative range for visual hydrograph comparison
    def norm01(arr):
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        return (arr - mn) / (mx - mn + 1e-12)

    sm_ref_snap = norm01(sm100_all[b_idx, snap_mask])
    wb_snap = norm01(w_base_all[b_idx, snap_mask])
    wc_snap = norm01(w_cn_all[b_idx, snap_mask])
    swe_snap = swe_all[b_idx, snap_mask]

    # Plot soil moisture trajectories
    ax_c.plot(snap_dates, sm_ref_snap, color="#333333", linewidth=1.4, linestyle="-", label=r"Caravan $\mathrm{SM}_{100}$ (Ref)", zorder=3)
    ax_c.plot(snap_dates, wb_snap, color=MODEL_COLORS["Base"], linewidth=1.2, linestyle="--", label=r"Base $W_{\mathrm{tot}}$ (False Winter Peak)", zorder=2)
    ax_c.plot(snap_dates, wc_snap, color=MODEL_COLORS["CN"], linewidth=1.3, linestyle="-", label=r"CN $W_{\mathrm{tot}}$ (Snowmelt Pulse)", zorder=2)

    # Secondary twin axis for SWE
    ax_c_swe = ax_c.twinx()
    ax_c_swe.spines['top'].set_visible(False)
    ax_c_swe.fill_between(snap_dates, 0, swe_snap, color="#6BAED6", alpha=0.22, label="ERA5-Land SWE")
    ax_c_swe.set_ylabel("SWE [mm]", color="#3182BD", fontsize=7.5)
    ax_c_swe.tick_params(axis='y', labelcolor="#3182BD", labelsize=7.0)
    ax_c_swe.set_ylim(0, np.nanmax(swe_snap) * 2.8)

    ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax_c.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax_c.set_ylabel("Standardized Soil Storage [0–1]")
    ax_c.set_title(f"(c) Catchment 09065500 (Colorado Rockies, SWE 280 mm)", loc="left", fontweight="bold", fontsize=8.5)
    ax_c.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", framealpha=0.9, fontsize=6.8)

    # -----------------------------------------------------------------------
    # Panel D: Timing Error Distributions
    # -----------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    apply_clean_spines(ax_d)

    t_base = df_t_reg[df_t_reg["structure"] == "Base"]
    t_cn = df_t_reg[df_t_reg["structure"] == "CN"]

    # Valid timing records
    valid_mask = t_base["median_abs_wetup_error_days"].notna() & t_cn["median_abs_wetup_error_days"].notna()
    base_wet_err = t_base.loc[valid_mask, "median_abs_wetup_error_days"].values
    cn_wet_err = t_cn.loc[valid_mask, "median_abs_wetup_error_days"].values

    base_peak_err = t_base.loc[valid_mask, "median_abs_peak_error_days"].values
    cn_peak_err = t_cn.loc[valid_mask, "median_abs_peak_error_days"].values

    pos = [1, 2, 4, 5]
    bp_data = [base_wet_err, cn_wet_err, base_peak_err, cn_peak_err]
    bp = ax_d.boxplot(bp_data, positions=pos, widths=0.55, patch_artist=True, showfliers=False,
                      medianprops=dict(color="#111111", linewidth=1.4))

    colors = [MODEL_COLORS["Base"], MODEL_COLORS["CN"], MODEL_COLORS["Base"], MODEL_COLORS["CN"]]
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)

    ax_d.set_xticks([1.5, 4.5])
    ax_d.set_xticklabels(["Spring Wet-Up\nTiming Error", "Soil Peak\nTiming Error"], fontsize=7.5)
    ax_d.set_ylabel("Median Absolute Error [days]")
    ax_d.set_title("(d) Timing Error Reduction in Snow Catchments", loc="left", fontweight="bold", fontsize=8.5)

    # Custom legend for panel D
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=MODEL_COLORS["Base"], alpha=0.85, label="Base"),
        Patch(facecolor=MODEL_COLORS["CN"], alpha=0.85, label="CN"),
    ]
    ax_d.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="#FFFFFF", fontsize=7.2)

    # Save figure
    png_path = out_dir / "figure4_r4_soil_consistency.png"
    pdf_path = out_dir / "figure4_r4_soil_consistency.pdf"
    plt.savefig(png_path, dpi=600)
    plt.savefig(pdf_path)
    plt.close()

    print(f"Generated Figure 4:\n  {png_path}\n  {pdf_path}")
    return png_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--regime", default="dPL_seed42")
    args = parser.parse_args()

    results_root = args.results_root or default_results_root()
    generate_figure4(results_root, args.out_dir, args.regime)
