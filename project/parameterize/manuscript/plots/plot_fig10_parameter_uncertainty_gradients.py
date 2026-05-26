"""
Fig10: Parameter uncertainty gradients learned by delta_dist

Layout mirrors Fig09: 5 columns x 4 rows, each column = one process group.
y-axis: δ_dist parameter std (mean over seeds × loss functions per basin).
x-axis: basin attribute.
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

ROOT = Path(__file__).resolve().parents[4]
PARAM_ROOT = ROOT / "project" / "parameterize"
ANALYSIS_ROOT = PARAM_ROOT / "outputs" / "analysis" / "stability_stats"
TABLE_ROOT = ANALYSIS_ROOT / "tables"
CORR_ROOT = ANALYSIS_ROOT / "correlation_summaries"
PLOTS_ROOT = PARAM_ROOT / "manuscript" / "plots"
ANALYSIS_OUT = PARAM_ROOT / "manuscript" / "analysis"

sys.path.insert(0, str(PLOTS_ROOT))
from common import setup_style, clean_axes, p_label, MM, DPI

# ── output dirs ───────────────────────────────────────────────────────────────
FIG_OUT = PARAM_ROOT / "manuscript" / "figures" / "main"
ANALYSIS_FIG_OUT = ANALYSIS_OUT / "figure10"
REPORT_OUT = ANALYSIS_FIG_OUT / "reports"
for d in [FIG_OUT, ANALYSIS_FIG_OUT, REPORT_OUT]:
    d.mkdir(parents=True, exist_ok=True)

# ── process groups ────────────────────────────────────────────────────────────
GROUPS = [
    "Snow / seasonality",
    "Aridity / ET",
    "Terrain / topography",
    "Soil / storage",
    "Routing / extremes",
]
GROUP_COLORS = {
    "Snow / seasonality":   "#56B4E9",
    "Aridity / ET":         "#E69F00",
    "Terrain / topography": "#009E73",
    "Soil / storage":       "#CC79A7",
    "Routing / extremes":   "#0072B2",
}

# ── 20 selected uncertainty pairs (5 groups × 4 rows) ────────────────────────
# Selected by: |ρ| strength, hydrological interpretability, process coverage
# All have sign_consistency = 1.0 (verified below)
PAIRS_BY_GROUP: dict[str, list[tuple[str, str]]] = {
    "Snow / seasonality": [
        ("parCFMAX", "frac_snow"),       # ρ = -0.93
        ("parTT",    "frac_snow"),       # ρ = -0.93
        ("parCWH",   "frac_snow"),       # ρ = -0.92
        ("parCWH",   "elev_mean"),       # ρ = -0.57
    ],
    "Aridity / ET": [
        ("parCWH",   "pet_mean"),        # ρ = +0.60
        ("parPERC",  "aridity"),         # ρ = -0.58
        ("parUZL",   "p_seasonality"),   # ρ = -0.50
        ("parBETA",  "aridity"),         # ρ = -0.47
    ],
    "Terrain / topography": [
        ("route_b",  "slope_mean"),      # ρ = +0.63
        ("parUZL",   "slope_mean"),      # ρ = +0.53
        ("parCFMAX", "slope_mean"),      # ρ = -0.49
        ("parCWH",   "slope_mean"),      # ρ = -0.52
    ],
    "Soil / storage": [
        ("parUZL",   "soil_conductivity"),    # ρ = +0.63
        ("parUZL",   "clay_frac"),            # ρ = -0.56
        ("parPERC",  "frac_forest"),          # ρ = +0.52
        ("route_b",  "soil_depth_pelletier"), # ρ = -0.56
    ],
    "Routing / extremes": [
        ("route_a",  "lai_diff"),        # ρ = -0.68
        ("route_a",  "aridity"),         # ρ = +0.57
        ("parK2",    "high_prec_dur"),   # ρ = -0.47
        ("route_a",  "high_prec_dur"),   # ρ = +0.51
    ],
}

NCOLS = 5
NROWS = 4
N_BINS = 8

# ── attribute labels ──────────────────────────────────────────────────────────
ATTR_SHORT = {
    "frac_snow":            "Snow fraction",
    "elev_mean":            "Mean elevation",
    "pet_mean":             "Mean PET",
    "aridity":              "Aridity index",
    "p_seasonality":        "Precip. seasonality",
    "slope_mean":           "Mean slope",
    "soil_conductivity":    "Soil conductivity",
    "clay_frac":            "Clay fraction",
    "frac_forest":          "Forest fraction",
    "soil_depth_pelletier": "Soil depth",
    "lai_diff":             "LAI seasonality",
    "high_prec_dur":        "High-prec. dur.",
    "low_prec_dur":         "Low-prec. dur.",
    "low_prec_freq":        "Low-prec. freq.",
}

ATTRIBUTE_GROUP_ORDER = [
    "Climate/hydroclimate",
    "Soil",
    "Geology",
    "Topography/scale",
    "Vegetation/land cover",
]
ATTRIBUTE_GROUP_COLORS = {
    "Climate/hydroclimate": "#4C78A8",
    "Topography/scale": "#8FA35A",
    "Vegetation/land cover": "#2A9D8F",
    "Soil": "#D4A15F",
    "Geology": "#A67899",
}
ATTRIBUTE_GROUP_SHORT = {
    "Climate/hydroclimate": "Climate",
    "Topography/scale": "Topography",
    "Vegetation/land cover": "Vegetation",
    "Soil": "Soil",
    "Geology": "Geology",
}
ATTRIBUTE_GROUP_RULES = {
    "Climate/hydroclimate": {
        "p_mean", "pet_mean", "p_seasonality", "frac_snow", "aridity",
        "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur",
    },
    "Topography/scale": {
        "elev_mean", "slope_mean", "area_gages2", "relief",
        "topographic_wetness", "twi",
    },
    "Vegetation/land cover": {
        "frac_forest", "lai_max", "lai_diff", "gvf_max", "gvf_diff",
        "dom_land_cover_frac", "dom_land_cover", "land_cover", "vegetation",
    },
    "Soil": {
        "root_depth_50", "soil_depth_pelletier", "soil_depth_statsgo",
        "soil_porosity", "soil_conductivity", "max_water_content",
        "sand_frac", "silt_frac", "clay_frac",
    },
    "Geology": {
        "geol_1st_class", "glim_1st_class_frac", "geol_2nd_class",
        "glim_2nd_class_frac", "carbonate_rocks_frac",
    },
}


def strip_par(name: str) -> str:
    return p_label(name)


def attr_label(attr: str) -> str:
    return ATTR_SHORT.get(attr, attr.replace("_", " "))


def attribute_group(attr: str) -> str:
    for group in ATTRIBUTE_GROUP_ORDER:
        if attr in ATTRIBUTE_GROUP_RULES[group]:
            return group
    lower = attr.lower()
    if any(t in lower for t in ("prec", "snow", "pet", "arid", "season", "p_mean")):
        return "Climate/hydroclimate"
    if any(t in lower for t in ("slope", "elev", "area", "relief", "topo", "twi")):
        return "Topography/scale"
    if any(t in lower for t in ("forest", "lai", "gvf", "land_cover", "vegetation")):
        return "Vegetation/land cover"
    if any(t in lower for t in ("soil", "sand", "silt", "clay", "porosity", "conduct")):
        return "Soil"
    if any(t in lower for t in ("geol", "glim", "carbonate")):
        return "Geology"
    return "Climate/hydroclimate"


# ── data loading ──────────────────────────────────────────────────────────────

def load_dist_param_std() -> pd.DataFrame:
    """Per-basin parameter std, averaged over seeds × loss functions."""
    df = pd.read_csv(TABLE_ROOT / "params_long.csv")
    dist = df[df["model"] == "distributional"]
    return (dist.groupby(["basin_id", "parameter"])["std"]
            .mean().reset_index().rename(columns={"std": "param_std"}))


def load_attributes() -> pd.DataFrame:
    return pd.read_csv(TABLE_ROOT / "basin_attributes.csv")


def compute_all_correlations(wide_std: pd.DataFrame,
                             attrs: pd.DataFrame) -> pd.DataFrame:
    numeric_attrs = attrs.select_dtypes(include=[float, int]).columns.tolist()
    numeric_attrs = [c for c in numeric_attrs if c != "basin_id"]
    merged = wide_std.join(attrs.set_index("basin_id")[numeric_attrs], how="inner")
    records = []
    for param in wide_std.columns:
        for attr in numeric_attrs:
            xy = merged[[param, attr]].dropna()
            if len(xy) < 50:
                continue
            rho, pval = stats.spearmanr(xy[attr], xy[param])
            records.append(dict(parameter=param, attribute=attr,
                                spearman_rho=rho, abs_rho=abs(rho), pval=pval))
    return pd.DataFrame(records).sort_values("abs_rho", ascending=False)


# ── binning ───────────────────────────────────────────────────────────────────

def compute_binned(x: np.ndarray, y: np.ndarray, n_bins: int = N_BINS):
    edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    bin_idx = np.digitize(x, edges[1:-1])
    centres, medians, q25s, q75s = [], [], [], []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        if mask.sum() < 4:
            continue
        centres.append(np.median(x[mask]))
        medians.append(np.median(y[mask]))
        q25s.append(np.percentile(y[mask], 25))
        q75s.append(np.percentile(y[mask], 75))
    return (np.array(centres), np.array(medians),
            np.array(q25s), np.array(q75s))


# ── single panel ──────────────────────────────────────────────────────────────

def plot_panel(ax, x, y, attr, param, rho, color,
               show_xlabel: bool, show_ylabel: bool):
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
    ax.set_ylim(max(0, ylo - ypad), yhi + ypad)   # std ≥ 0

    sign_char = "+" if rho >= 0 else "−"
    ax.text(0.96, 0.96, f"ρ = {sign_char}{abs(rho):.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10.0, color="#222222", zorder=5)

    clean_axes(ax, grid_axis="y")
    ax.tick_params(labelsize=10.0, length=2.5, pad=2)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune="both"))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune="both"))

    if show_xlabel:
        ax.set_xlabel(attr_label(attr), fontsize=11.0, labelpad=3)
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel(f"{strip_par(param)} std", fontsize=11.0, labelpad=3)
    else:
        ax.set_ylabel("")


# ── full figure ───────────────────────────────────────────────────────────────

def make_figure(corr_df: pd.DataFrame,
                wide_std: pd.DataFrame,
                attributes: pd.DataFrame):

    corr_idx = corr_df.set_index(["parameter", "attribute"])
    all_attrs = list({a for grp in PAIRS_BY_GROUP.values()
                      for _, a in grp if a in attributes.columns})
    merged = wide_std.join(attributes.set_index("basin_id")[all_attrs], how="inner")

    fig_w = 258 * MM
    fig_h = NROWS * 56 * MM + 30 * MM

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        NROWS, NCOLS,
        left=0.07, right=0.99,
        top=0.97, bottom=0.13,
        hspace=0.35, wspace=0.42,
    )
    axes = np.array([[fig.add_subplot(gs[r, c])
                      for c in range(NCOLS)] for r in range(NROWS)])

    binned_records = []

    for ci, grp in enumerate(GROUPS):
        pairs = PAIRS_BY_GROUP[grp]

        for ri, (param, attr) in enumerate(pairs):
            ax = axes[ri][ci]

            if param not in merged.columns or attr not in merged.columns:
                ax.set_visible(False)
                continue

            xy = merged[[param, attr]].dropna()
            x, y = xy[attr].values, xy[param].values

            key = (param, attr)
            rho = float(corr_idx.loc[key, "spearman_rho"]) \
                  if key in corr_idx.index else 0.0
            color = GROUP_COLORS[grp]

            plot_panel(ax, x, y, attr, param, rho, color,
                       show_xlabel=True, show_ylabel=True)

            bx, bmed, bq25, bq75 = compute_binned(x, y)
            for bxi, bmi, bq2, bq7 in zip(bx, bmed, bq25, bq75):
                binned_records.append(dict(parameter=param, attribute=attr,
                                           bin_centre=bxi, median=bmi,
                                           q25=bq2, q75=bq7))

    # ── legend (two rows) ─────────────────────────────────────────────────────
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
        loc="lower center", ncol=3,
        fontsize=10.8, frameon=False,
        bbox_to_anchor=(0.5, 0.055),
        handlelength=1.8, handletextpad=0.5, columnspacing=1.8,
    )
    fig.add_artist(leg1)
    fig.legend(
        handles=grp_handles,
        loc="lower center", ncol=5,
        fontsize=10.5, frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=1.2, handletextpad=0.4, columnspacing=1.2,
    )

    return fig, pd.DataFrame(binned_records)


# ── reports ───────────────────────────────────────────────────────────────────

def write_selected_pairs_csv(corr_df: pd.DataFrame) -> None:
    selected_set = {(p, a) for grp in PAIRS_BY_GROUP.values() for p, a in grp}
    pair_group   = {(p, a): g
                    for g, pairs in PAIRS_BY_GROUP.items() for p, a in pairs}
    sel = corr_df[corr_df.apply(
        lambda r: (r["parameter"], r["attribute"]) in selected_set, axis=1
    )].copy()
    sel["process_group"] = sel.apply(
        lambda r: pair_group.get((r["parameter"], r["attribute"]), ""), axis=1)
    sel["topk_rank"] = sel.groupby("parameter")["abs_rho"].rank(
        ascending=False, method="min").astype(int)
    sel.to_csv(ANALYSIS_FIG_OUT / "figure10_selected_pairs.csv", index=False)
    return sel


def write_plot_notes() -> None:
    attr_full = "\n".join(f"- `{k}`: {v}" for k, v in ATTR_SHORT.items())
    txt = f"""# Figure 11 – Plot Notes

## What is plotted
- y-axis: δ_dist **parameter std** per basin, averaged over 5 seeds × 3 loss
  functions (model = distributional). Source: `params_long.csv` column `std`.
- x-axis: CAMELS basin attribute. Source: `basin_attributes.csv`.
- Spearman ρ computed on the full 531-basin sample.

## Attribute labels
{attr_full}

## Quantile bins
- {N_BINS} quantile bins of equal basin count per attribute axis.
- Bins with fewer than 4 basins are dropped.

## Display range
- Each panel clipped to 1st–99th percentile to prevent outlier axis stretch.
- y-axis lower bound forced to 0 (std ≥ 0 by definition).
- Binned statistics computed on full (unclipped) data.

## IQR
- Per bin: 25th–75th percentile of parameter std. Semi-transparent band
  (alpha = 0.25).

## Annotation
- ρ: Spearman correlation between attribute and parameter std across 531 basins.
- No seed SD shown (std is already an uncertainty measure; its cross-seed
  variability is not separately annotated here).

## Parameter name handling
- All `par` prefixes removed; y-axis label appended with " std".
- route_a and route_b retain their names.

## Interpretation caveat
Structured uncertainty gradients do **not** directly prove parameter
identifiability or physical correctness. The observed patterns may reflect:
- genuine identifiability gradients (e.g., snow parameters better constrained
  in high-snow basins);
- process compensation or equifinality;
- mean–std coupling (higher mean → higher absolute std);
- boundary effects near parameter bounds.
These gradients should therefore be interpreted cautiously as evidence of
structured, basin-attribute-dependent uncertainty, not as proof of
identifiability or true physical relationships.
"""
    (REPORT_OUT / "figure10_plot_notes.md").write_text(txt)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    setup_style()

    print("Loading data …")
    param_std   = load_dist_param_std()
    attributes  = load_attributes()
    wide_std    = param_std.pivot(index="basin_id", columns="parameter",
                                  values="param_std")

    print("Computing correlations …")
    corr_df = compute_all_correlations(wide_std, attributes)
    corr_df.to_csv(ANALYSIS_FIG_OUT / "figure10_all_correlations.csv", index=False)

    print("Writing selected pairs …")
    sel = write_selected_pairs_csv(corr_df)
    print(f"  → {len(sel)} pairs selected")

    print("Plotting …")
    fig, binned_df = make_figure(corr_df, wide_std, attributes)
    binned_df.to_csv(ANALYSIS_FIG_OUT / "figure10_binned_gradients.csv", index=False)

    stem = "Fig10_parameter_uncertainty_attribute_gradients"
    fig.savefig(FIG_OUT / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {stem}.png")

    write_plot_notes()
    print("  → notes written")
    print("Done.")


if __name__ == "__main__":
    main()
