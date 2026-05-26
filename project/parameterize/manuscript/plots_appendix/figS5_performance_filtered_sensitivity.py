"""Fig. S5 — Performance-filtered sensitivity.
Style: identical panel layout to Fig10/Fig11 (plot_gradient_panel from common_appendix).
Shows full-sample vs NSE-filtered binned median curves overlaid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from common_appendix import (
    setup_style, clean_axes, add_panel_label,
    MM, DPI, TABLE_ROOT, SPATIAL_MEAN_DIR,
    APP_FIG_DIR, save_fig,
    GROUPS, GROUP_COLORS, attr_label, strip_par, compute_binned,
)

OUT_STEM = "figS5_performance_filtered_sensitivity"
NSE_THRESHOLD = 0.3
N_BINS = 8

# Same 20 pairs as Fig10, grouped by process group
PAIRS_BY_GROUP = {
    "Snow / seasonality": [
        ("parCWH", "frac_snow"),
        ("parCFR", "frac_snow"),
        ("parK0",  "frac_snow"),
        ("parCWH", "low_prec_freq"),
    ],
    "Aridity / ET": [
        ("parFC",   "pet_mean"),
        ("parPERC", "aridity"),
        ("parUZL",  "p_seasonality"),
        ("parCWH",  "high_prec_freq"),
    ],
    "Terrain / topography": [
        ("parBETA", "slope_mean"),
        ("parPERC", "slope_mean"),
        ("parUZL",  "slope_mean"),
        ("parBETA", "elev_mean"),
    ],
    "Soil / storage": [
        ("parUZL",  "soil_conductivity"),
        ("parPERC", "frac_forest"),
        ("parBETA", "soil_depth_pelletier"),
        ("parTT",   "high_prec_dur"),
    ],
    "Routing / extremes": [
        ("parK1",   "lai_diff"),
        ("route_b", "low_prec_freq"),
        ("parK2",   "high_prec_dur"),
        ("parFC",   "low_prec_dur"),
    ],
}

NCOLS = 5
NROWS = 4


def load_data():
    params_df = pd.read_csv(TABLE_ROOT / "params_long.csv")
    dist = params_df[params_df["model"] == "distributional"]
    param_means = (dist.groupby(["basin_id", "parameter"])["mean"]
                   .mean().reset_index().rename(columns={"mean": "param_mean"}))
    attrs = pd.read_csv(TABLE_ROOT / "basin_attributes.csv")

    metrics = pd.read_csv(TABLE_ROOT / "metrics_long.csv")
    dist_nse = (metrics[(metrics["model"] == "distributional") &
                        (metrics["loss"] == "HybridNseBatchLoss")]
                .groupby("basin_id")["nse"].median().reset_index())
    return param_means, attrs, dist_nse


def main() -> None:
    setup_style()
    param_means, attrs, dist_nse = load_data()

    wide = param_means.pivot(index="basin_id", columns="parameter", values="param_mean")
    all_attrs = list({a for grp in PAIRS_BY_GROUP.values() for _, a in grp
                      if a in attrs.columns})
    merged = wide.join(attrs.set_index("basin_id")[all_attrs], how="inner")
    merged = merged.join(dist_nse.set_index("basin_id")["nse"], how="left")

    fig_w = 200 * MM
    fig_h = NROWS * 44 * MM + 22 * MM
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(NROWS, NCOLS,
                          left=0.07, right=0.99, top=0.97, bottom=0.13,
                          hspace=0.55, wspace=0.42)
    axes = np.array([[fig.add_subplot(gs[r, c])
                      for c in range(NCOLS)] for r in range(NROWS)])

    for ci, grp in enumerate(GROUPS):
        color = GROUP_COLORS[grp]
        for ri, (param, attr) in enumerate(PAIRS_BY_GROUP[grp]):
            ax = axes[ri][ci]
            if param not in merged.columns or attr not in merged.columns:
                ax.set_visible(False)
                continue

            xy_all  = merged[[param, attr]].dropna()
            xy_filt = merged.loc[merged["nse"] >= NSE_THRESHOLD, [param, attr]].dropna()

            x_all, y_all   = xy_all[attr].values,  xy_all[param].values
            x_flt, y_flt   = xy_filt[attr].values, xy_filt[param].values

            # grey scatter (all basins)
            ax.scatter(x_all, y_all, s=3.5, alpha=0.18, color="#BBBBBB",
                       linewidths=0, rasterized=True, zorder=1)

            # full-sample binned median — dashed, muted
            bx, bm, bq25, bq75 = compute_binned(x_all, y_all)
            if len(bx) >= 2:
                ax.plot(bx, bm, color=color, linewidth=1.2,
                        linestyle="--", alpha=0.55, zorder=2)

            # filtered binned median — solid, full colour + IQR band
            if len(x_flt) >= 20:
                bx2, bm2, bq252, bq752 = compute_binned(x_flt, y_flt)
                if len(bx2) >= 2:
                    ax.fill_between(bx2, bq252, bq752, alpha=0.25,
                                    color=color, linewidth=0, zorder=3)
                    ax.plot(bx2, bm2, color=color, linewidth=1.5, zorder=4)
                    ax.plot(bx2, bm2, "o", color=color, markersize=3.0,
                            markeredgewidth=0, zorder=5)

            # ρ annotations
            rho_all, _ = spearmanr(x_all, y_all)
            rho_flt, _ = spearmanr(x_flt, y_flt) if len(x_flt) >= 10 else (np.nan, 1)
            sign_all = "+" if rho_all >= 0 else "−"
            sign_flt = "+" if rho_flt >= 0 else "−"
            ax.text(0.96, 0.96,
                    f"ρ={sign_all}{abs(rho_all):.2f} (all)\n"
                    f"ρ={sign_flt}{abs(rho_flt):.2f} (NSE≥{NSE_THRESHOLD})",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=5.8, color="#222222", zorder=6)

            import matplotlib as mpl
            clean_axes(ax, grid_axis="y")
            ax.tick_params(labelsize=6.2, length=2.5, pad=2)
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune="both"))
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune="both"))

            show_x = (ri == NROWS - 1)
            show_y = (ci == 0)
            ax.set_xlabel(attr_label(attr) if show_x else "", fontsize=7.2, labelpad=3)
            ax.set_ylabel(strip_par(param) if show_y else "", fontsize=7.2, labelpad=3)

    # Legend — two rows like fig10
    leg_dot  = mlines.Line2D([], [], marker="o", color="none",
                             markerfacecolor="#BBBBBB", markeredgewidth=0,
                             markersize=5.5, label="Basins (n = 531)")
    leg_full = mlines.Line2D([], [], color="#555555", linewidth=1.5,
                             marker="o", markersize=3.5,
                             markerfacecolor="#555555", markeredgewidth=0,
                             label=f"Filtered median (NSE ≥ {NSE_THRESHOLD})")
    leg_dash = mlines.Line2D([], [], color="#555555", linewidth=1.2,
                             linestyle="--", alpha=0.55,
                             label="Full-sample median")
    leg_band = mpatches.Patch(facecolor="#888888", alpha=0.28,
                              edgecolor="none", label="IQR (25–75 %, filtered)")
    grp_handles = [
        mpatches.Patch(facecolor=GROUP_COLORS[g], alpha=0.80,
                       edgecolor="none", label=g)
        for g in GROUPS
    ]
    leg1 = fig.legend(
        handles=[leg_dot, leg_full, leg_dash, leg_band],
        loc="lower center", ncol=4,
        fontsize=7.2, frameon=False,
        bbox_to_anchor=(0.5, 0.055),
        handlelength=1.8, handletextpad=0.5, columnspacing=1.5,
    )
    fig.add_artist(leg1)
    fig.legend(
        handles=grp_handles,
        loc="lower center", ncol=5,
        fontsize=7.2, frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=1.2, handletextpad=0.4, columnspacing=1.2,
    )

    save_fig(fig, OUT_STEM)
    print(f"Saved {APP_FIG_DIR / OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
