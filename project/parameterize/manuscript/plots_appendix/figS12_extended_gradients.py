"""Fig. S12a/b — Extended attribute–parameter gradient panels.
Style: exact clone of Fig10 (mean) and Fig11 (std) — same layout, same
plot_panel logic, same legend, same group colours.
S12a: parameter means (more pairs than Fig10)
S12b: parameter stds (more pairs than Fig11)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common_appendix import (
    setup_style, clean_axes, add_panel_label, MM, DPI,
    APP_FIG_DIR, save_fig, TABLE_ROOT,
    GROUPS, GROUP_COLORS, attr_label, strip_par, compute_binned,
)

OUT_STEM_MEAN = "figS12a_extended_mean_gradients"
OUT_STEM_STD  = "figS12b_extended_std_gradients"

NCOLS = 5
NROWS = 4

# ── Extended mean pairs (5 groups × 4 rows) ──────────────────────────────────
# Extends Fig10 with additional pairs not shown in the main figure
MEAN_PAIRS_BY_GROUP = {
    "Snow / seasonality": [
        ("parCFMAX", "frac_snow"),
        ("parTT",    "frac_snow"),
        ("parCFR",   "elev_mean"),
        ("parCWH",   "elev_mean"),
    ],
    "Aridity / ET": [
        ("parLP",   "aridity"),
        ("parK1",   "aridity"),
        ("parK2",   "aridity"),
        ("parPERC", "p_mean"),
    ],
    "Terrain / topography": [
        ("parK0",   "slope_mean"),
        ("parK1",   "slope_mean"),
        ("parK2",   "slope_mean"),
        ("route_a", "slope_mean"),
    ],
    "Soil / storage": [
        ("parFC",   "soil_depth_pelletier"),
        ("parLP",   "frac_forest"),
        ("parPERC", "clay_frac"),
        ("parUZL",  "clay_frac"),
    ],
    "Routing / extremes": [
        ("route_a", "lai_diff"),
        ("route_b", "area_gages2"),
        ("route_b", "low_prec_dur"),
        ("parK0",   "high_prec_dur"),
    ],
}

# ── Extended std pairs (5 groups × 4 rows) ───────────────────────────────────
STD_PAIRS_BY_GROUP = {
    "Snow / seasonality": [
        ("parCFMAX", "frac_snow"),
        ("parTT",    "frac_snow"),
        ("parCWH",   "frac_snow"),
        ("parCFMAX", "elev_mean"),
    ],
    "Aridity / ET": [
        ("parCWH",  "pet_mean"),
        ("parPERC", "aridity"),
        ("parUZL",  "p_seasonality"),
        ("parBETA", "aridity"),
    ],
    "Terrain / topography": [
        ("route_b",  "slope_mean"),
        ("parUZL",   "slope_mean"),
        ("parCFMAX", "slope_mean"),
        ("parCWH",   "slope_mean"),
    ],
    "Soil / storage": [
        ("parUZL",  "soil_conductivity"),
        ("parUZL",  "clay_frac"),
        ("parPERC", "frac_forest"),
        ("route_b", "soil_depth_pelletier"),
    ],
    "Routing / extremes": [
        ("route_a", "lai_diff"),
        ("route_a", "aridity"),
        ("parK2",   "high_prec_dur"),
        ("route_a", "high_prec_dur"),
    ],
}


def load_mean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    params_df = pd.read_csv(TABLE_ROOT / "params_long.csv")
    dist = params_df[params_df["model"] == "distributional"]
    param_means = (dist.groupby(["basin_id", "parameter"])["mean"]
                   .mean().reset_index().rename(columns={"mean": "param_mean"}))
    attrs = pd.read_csv(TABLE_ROOT / "basin_attributes.csv")
    return param_means, attrs


def load_std_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    params_df = pd.read_csv(TABLE_ROOT / "params_long.csv")
    dist = params_df[params_df["model"] == "distributional"]
    param_stds = (dist.groupby(["basin_id", "parameter"])["std"]
                  .mean().reset_index().rename(columns={"std": "param_std"}))
    attrs = pd.read_csv(TABLE_ROOT / "basin_attributes.csv")
    return param_stds, attrs


def load_stability() -> pd.DataFrame:
    from common_appendix import CORR_ROOT
    df = pd.read_csv(CORR_ROOT / "pair_seed_stability.csv")
    dist = df[df["model"] == "distributional"]
    return (dist.groupby(["parameter", "attribute"])
            .agg(mean_seed_std=("seed_std_rho", "mean"))
            .reset_index())


def make_gradient_figure(pairs_by_group: dict, wide: pd.DataFrame,
                         attrs_df: pd.DataFrame, stab: pd.DataFrame,
                         value_col: str, ylabel_suffix: str,
                         out_stem: str) -> None:
    all_attrs_needed = list({a for grp in pairs_by_group.values()
                             for _, a in grp if a in attrs_df.columns})
    merged = wide.join(attrs_df.set_index("basin_id")[all_attrs_needed],
                       how="inner")

    fig_w = 200 * MM
    fig_h = NROWS * 44 * MM + 22 * MM
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(NROWS, NCOLS,
                          left=0.07, right=0.99, top=0.97, bottom=0.13,
                          hspace=0.55, wspace=0.42)
    axes = np.array([[fig.add_subplot(gs[r, c])
                      for c in range(NCOLS)] for r in range(NROWS)])

    stab_idx = stab.set_index(["parameter", "attribute"]) if stab is not None else None

    for ci, grp in enumerate(GROUPS):
        color = GROUP_COLORS[grp]
        pairs = pairs_by_group.get(grp, [])
        for ri in range(NROWS):
            ax = axes[ri][ci]
            if ri >= len(pairs):
                ax.set_visible(False)
                continue
            param, attr = pairs[ri]
            if param not in merged.columns or attr not in merged.columns:
                ax.set_visible(False)
                continue

            xy = merged[[param, attr]].dropna()
            x, y = xy[attr].values, xy[param].values

            xlo, xhi = np.percentile(x, 1), np.percentile(x, 99)
            ylo, yhi = np.percentile(y, 1), np.percentile(y, 99)
            xpad = max((xhi - xlo) * 0.05, 1e-6)
            ypad = max((yhi - ylo) * 0.08, 1e-6)

            ax.scatter(x, y, s=3.5, alpha=0.22, color="#BBBBBB",
                       linewidths=0, rasterized=True, zorder=1)
            bx, bmed, bq25, bq75 = compute_binned(x, y)
            if len(bx) >= 2:
                ax.fill_between(bx, bq25, bq75, alpha=0.25, color=color,
                                linewidth=0, zorder=2)
                ax.plot(bx, bmed, color=color, linewidth=1.5, zorder=3)
                ax.plot(bx, bmed, "o", color=color, markersize=3.0,
                        markeredgewidth=0, zorder=4)

            ax.set_xlim(xlo - xpad, xhi + xpad)
            if ylabel_suffix == "std":
                ax.set_ylim(max(0, ylo - ypad), yhi + ypad)
            else:
                ax.set_ylim(ylo - ypad, yhi + ypad)

            rho, _ = stats.spearmanr(x, y)
            sign_char = "+" if rho >= 0 else "−"
            ax.text(0.96, 0.96, f"ρ = {sign_char}{abs(rho):.2f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.8, color="#222222", zorder=5)

            if stab_idx is not None and (param, attr) in stab_idx.index:
                sd = stab_idx.loc[(param, attr), "mean_seed_std"]
                ax.text(0.96, 0.83, f"SD = {sd:.3f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=6.2, color="#555555", zorder=5)

            clean_axes(ax, grid_axis="y")
            ax.tick_params(labelsize=6.2, length=2.5, pad=2)
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune="both"))
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune="both"))
            ax.set_xlabel(attr_label(attr) if ri == NROWS - 1 else "",
                          fontsize=7.2, labelpad=3)
            ax.set_ylabel(f"{strip_par(param)} {ylabel_suffix}" if ci == 0 else "",
                          fontsize=7.2, labelpad=3)

    # Legend — identical two-row layout to fig10/fig11
    leg_dot  = mlines.Line2D([], [], marker="o", color="none",
                             markerfacecolor="#BBBBBB", markeredgewidth=0,
                             markersize=5.5, label="Basins (n = 531)")
    leg_line = mlines.Line2D([], [], color="#555555", linewidth=1.5,
                             marker="o", markersize=3.5,
                             markerfacecolor="#555555", markeredgewidth=0,
                             label="Binned median")
    leg_band = mpatches.Patch(facecolor="#888888", alpha=0.28,
                              edgecolor="none", label="IQR (25–75%)")
    grp_handles = [
        mpatches.Patch(facecolor=GROUP_COLORS[g], alpha=0.80,
                       edgecolor="none", label=g)
        for g in GROUPS
    ]
    leg1 = fig.legend(
        handles=[leg_dot, leg_line, leg_band],
        loc="lower center", ncol=3, fontsize=7.6, frameon=False,
        bbox_to_anchor=(0.5, 0.055),
        handlelength=1.8, handletextpad=0.5, columnspacing=1.8,
    )
    fig.add_artist(leg1)
    fig.legend(
        handles=grp_handles,
        loc="lower center", ncol=5, fontsize=7.2, frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=1.2, handletextpad=0.4, columnspacing=1.2,
    )

    save_fig(fig, out_stem)
    print(f"Saved {APP_FIG_DIR / out_stem}.png / .pdf")


def main() -> None:
    setup_style()
    stab = load_stability()

    # S12a: means
    param_means, attrs = load_mean_data()
    wide_mean = param_means.pivot(index="basin_id", columns="parameter",
                                  values="param_mean")
    make_gradient_figure(MEAN_PAIRS_BY_GROUP, wide_mean, attrs, stab,
                         "param_mean", "mean", OUT_STEM_MEAN)

    # S12b: stds
    param_stds, attrs = load_std_data()
    wide_std = param_stds.pivot(index="basin_id", columns="parameter",
                                values="param_std")
    make_gradient_figure(STD_PAIRS_BY_GROUP, wide_std, attrs, stab,
                         "param_std", "std", OUT_STEM_STD)


if __name__ == "__main__":
    main()
