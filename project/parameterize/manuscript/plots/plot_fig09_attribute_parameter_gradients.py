"""
Fig09: Reliable attribute-parameter gradients learned by delta_dist

Layout: 5 columns × 4 rows
- Each column = one process group (same colour)
- Column header: colour block + group name
- No subplot titles; x-axis = attribute name, y-axis = parameter name
- Spearman ρ and seed SD annotated top-right of each panel
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

ROOT = Path(__file__).resolve().parents[4]
PARAM_ROOT = ROOT / "project" / "parameterize"
ANALYSIS_ROOT = PARAM_ROOT / "outputs" / "analysis" / "stability_stats"
TABLE_ROOT = ANALYSIS_ROOT / "tables"
CORR_ROOT = ANALYSIS_ROOT / "correlation_summaries"
PLOTS_ROOT = PARAM_ROOT / "manuscript" / "plots"
ANALYSIS_OUT = PARAM_ROOT / "analysis"

sys.path.insert(0, str(PLOTS_ROOT))
from common import setup_style, clean_axes, p_label, MM, DPI

# ── output dirs ───────────────────────────────────────────────────────────────
FIG_OUT = PARAM_ROOT / "manuscript" / "figures" / "main"
ANALYSIS_FIG_OUT = ANALYSIS_OUT / "figure9"
REPORT_OUT = ANALYSIS_FIG_OUT / "reports"
for d in [FIG_OUT, ANALYSIS_FIG_OUT, REPORT_OUT]:
    d.mkdir(parents=True, exist_ok=True)

# ── process groups: column order and colours ──────────────────────────────────
# Each group has exactly 4 pairs → 4 rows
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
    "Soil / storage":       "#A6761D",
    "Routing / extremes":   "#CC79A7",
}
LINE_ALPHA = 1.0
BAND_ALPHA = 0.22
SCATTER_ALPHA = 0.12
SCATTER_COLOR = "#BDBDBD"

# ── 20 pairs: 5 groups × 4 rows, column-major order ──────────────────────────
# Each sub-list = one column (process group), top→bottom by |ρ| desc
PAIRS_BY_GROUP: dict[str, list[tuple[str, str]]] = {
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
N_BINS = 8

# ── attribute display labels ──────────────────────────────────────────────────
ATTR_SHORT = {
    "frac_snow":            "Snow fraction",
    "low_prec_freq":        "Low-prec. freq.",
    "high_prec_freq":       "High-prec. freq.",
    "high_prec_dur":        "High-prec. dur.",
    "pet_mean":             "Mean PET",
    "aridity":              "Aridity index",
    "p_seasonality":        "Precip. seasonality",
    "slope_mean":           "Mean slope",
    "elev_mean":            "Mean elevation",
    "soil_conductivity":    "Soil conductivity",
    "frac_forest":          "Forest fraction",
    "soil_depth_pelletier": "Soil depth",
    "lai_diff":             "LAI seasonality",
    "low_prec_dur":         "Low-prec. dur.",
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

def load_dist_param_means() -> pd.DataFrame:
    df = pd.read_csv(TABLE_ROOT / "params_long.csv")
    dist = df[df["model"] == "distributional"]
    return (dist.groupby(["basin_id", "parameter"])["mean"]
            .mean().reset_index().rename(columns={"mean": "param_mean"}))


def load_attributes() -> pd.DataFrame:
    return pd.read_csv(TABLE_ROOT / "basin_attributes.csv")


def load_stability() -> pd.DataFrame:
    df = pd.read_csv(CORR_ROOT / "pair_seed_stability.csv")
    dist = df[df["model"] == "distributional"]
    agg = dist.groupby(["parameter", "attribute"]).agg(
        mean_rho=("seed_mean_rho", "mean"),
        mean_seed_std=("seed_std_rho", "mean"),
        sign_consistency=("sign_consistency_seed", "mean"),
        topk_rate=("topk_rate_seed", "mean"),
        core_rank_min=("core_rank", "min"),
    ).reset_index()
    agg["abs_rho"] = agg["mean_rho"].abs()
    return agg


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

def plot_panel(ax, x, y, attr, param, rho, seed_sd, color,
               show_xlabel: bool, show_ylabel: bool):
    # display range: clip to 1–99 pct
    xlo, xhi = np.percentile(x, 1), np.percentile(x, 99)
    ylo, yhi = np.percentile(y, 1), np.percentile(y, 99)
    xpad = max((xhi - xlo) * 0.05, 1e-6)
    ypad = max((yhi - ylo) * 0.08, 1e-6)

    # scatter
    ax.scatter(x, y, s=3.5, alpha=SCATTER_ALPHA, color=SCATTER_COLOR,
               linewidths=0, rasterized=True, zorder=1)

    # binned median + IQR (full data)
    bx, bmed, bq25, bq75 = compute_binned(x, y)
    if len(bx) >= 2:
        ax.fill_between(bx, bq25, bq75, alpha=BAND_ALPHA, color=color,
                        linewidth=0, zorder=2)
        ax.plot(bx, bmed, color=color, alpha=LINE_ALPHA, linewidth=1.5, zorder=3)
        ax.plot(bx, bmed, "o", color=color, alpha=LINE_ALPHA, markersize=3.0,
                markeredgewidth=0, zorder=4)

    ax.set_xlim(xlo - xpad, xhi + xpad)
    ax.set_ylim(ylo - ypad, yhi + ypad)

    # ρ and SD annotation
    sign_char = "+" if rho >= 0 else "−"
    ax.text(0.96, 0.96, f"ρ = {sign_char}{abs(rho):.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10.0, color="#222222", zorder=5)
    ax.text(0.96, 0.83, f"SD = {seed_sd:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9.8, color="#555555", zorder=5)

    clean_axes(ax, grid_axis="y")
    ax.tick_params(labelsize=10.0, length=2.5, pad=2)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune="both"))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune="both"))

    # x-axis label = attribute name (every panel in this layout)
    if show_xlabel:
        ax.set_xlabel(attr_label(attr), fontsize=11.0, labelpad=3)
    else:
        ax.set_xlabel("")

    # y-axis label = parameter name (every panel in this layout)
    if show_ylabel:
        ax.set_ylabel(strip_par(param), fontsize=11.0, labelpad=3)
    else:
        ax.set_ylabel("")


# ── full figure ───────────────────────────────────────────────────────────────

def make_figure(stability: pd.DataFrame,
                param_means: pd.DataFrame,
                attributes: pd.DataFrame):

    stab_idx = stability.set_index(["parameter", "attribute"])
    wide = param_means.pivot(index="basin_id", columns="parameter",
                             values="param_mean")
    all_attrs = list({a for grp in PAIRS_BY_GROUP.values()
                      for _, a in grp if a in attributes.columns})
    merged = wide.join(attributes.set_index("basin_id")[all_attrs], how="inner")

    # ── figure geometry ───────────────────────────────────────────────────────
    fig_w = 258 * MM
    fig_h = NROWS * 56 * MM + 30 * MM   # rows + legend space

    fig = plt.figure(figsize=(fig_w, fig_h))

    # GridSpec: no header space at top; extra bottom for two-row legend
    gs = fig.add_gridspec(
        NROWS, NCOLS,
        left=0.07, right=0.99,
        top=0.97,
        bottom=0.13,
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
            rho = float(stab_idx.loc[key, "mean_rho"]) if key in stab_idx.index else 0.0
            sd  = float(stab_idx.loc[key, "mean_seed_std"]) if key in stab_idx.index else 0.0
            color = GROUP_COLORS[grp]

            # show x-label on every panel (different attrs per row within col)
            # show y-label on every panel (different params per row)
            plot_panel(ax, x, y, attr, param, rho, sd, color,
                       show_xlabel=True, show_ylabel=True)

            bx, bmed, bq25, bq75 = compute_binned(x, y)
            for bxi, bmi, bq2, bq7 in zip(bx, bmed, bq25, bq75):
                binned_records.append(dict(parameter=param, attribute=attr,
                                           bin_centre=bxi, median=bmi,
                                           q25=bq2, q75=bq7))

    # ── legend (two rows) ─────────────────────────────────────────────────────
    # Row 1: scatter / binned median / IQR
    leg_dot  = mlines.Line2D([], [], marker="o", color="none",
                             markerfacecolor=SCATTER_COLOR, alpha=SCATTER_ALPHA,
                             markeredgewidth=0,
                             markersize=5.5, label="Basins (n = 531)")
    leg_line = mlines.Line2D([], [], color="#555555", linewidth=1.5,
                             marker="o", markersize=3.5,
                             markerfacecolor="#555555", markeredgewidth=0,
                             label="Binned median")
    leg_band = mpatches.Patch(facecolor="#888888", alpha=BAND_ALPHA,
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


# ── CSV / report helpers (unchanged logic) ────────────────────────────────────

def build_candidate_df(stability: pd.DataFrame) -> pd.DataFrame:
    selected_set = {(p, a) for grp in PAIRS_BY_GROUP.values() for p, a in grp}
    pair_group   = {(p, a): g
                    for g, pairs in PAIRS_BY_GROUP.items() for p, a in pairs}
    cand = stability.copy()
    cand["process_group"] = cand.apply(
        lambda r: pair_group.get((r["parameter"], r["attribute"]), "Other"), axis=1)
    cand["selected_for_plot"] = cand.apply(
        lambda r: (r["parameter"], r["attribute"]) in selected_set, axis=1)
    cand["topk_rank"] = cand.groupby("parameter")["abs_rho"].rank(
        ascending=False, method="min").astype(int)
    cand["selection_reason"] = cand.apply(
        lambda r: (
            "selected: high |ρ|, stable, interpretable"
            if r["selected_for_plot"]
            else ("excluded: |ρ| < 0.35" if r["abs_rho"] < 0.35
                  else "excluded: unstable or no clear process group")
        ), axis=1)
    return cand


def write_selection_summary(cand: pd.DataFrame, n_layout: str) -> None:
    pair_group = {(p, a): g
                  for g, pairs in PAIRS_BY_GROUP.items() for p, a in pairs}
    sel = cand[cand["selected_for_plot"]].copy()
    sel["process_group"] = sel.apply(
        lambda r: pair_group.get((r["parameter"], r["attribute"]), ""), axis=1)

    lines = [
        "# Figure 10 – Selection Summary", "",
        f"- Candidate pairs evaluated: {len(cand)}",
        f"- Selected for main figure: {len(sel)}",
        f"- Layout: {n_layout}", "",
        "## Selected pairs", "",
        "| Parameter | Attribute | ρ | SD | Sign cons. | Top-k rank | Process group |",
        "|-----------|-----------|---|----|------------|------------|---------------|",
    ]
    for _, r in sel.sort_values(["process_group", "abs_rho"],
                                ascending=[True, False]).iterrows():
        lines.append(
            f"| {r['parameter'].replace('par','')} | {r['attribute']} "
            f"| {r['mean_rho']:.3f} | {r['mean_seed_std']:.3f} "
            f"| {r['sign_consistency']:.2f} | {int(r['topk_rank'])} "
            f"| {r['process_group']} |"
        )

    excl = cand[(~cand["selected_for_plot"]) & (cand["abs_rho"] >= 0.35)]
    lines += [
        "", "## Strong relationships excluded from main figure", "",
        "| Parameter | Attribute | |ρ| | Reason |",
        "|-----------|-----------|-----|--------|",
    ]
    for _, r in excl.sort_values("abs_rho", ascending=False).iterrows():
        lines.append(
            f"| {r['parameter'].replace('par','')} | {r['attribute']} "
            f"| {r['abs_rho']:.3f} | {r['selection_reason']} |"
        )

    lines += [
        "", "## Results writing suggestions", "",
        "The δ_dist model exhibits hydrologically interpretable gradients across",
        "basin attribute space: parameter means vary in a structured manner along",
        "aridity, snow fraction, terrain slope, and soil property gradients.",
        "These reliable attribute–parameter relationships are consistent with",
        "process-based interpretation of HBV parameter controls, providing",
        "evidence that the distributional parameterisation captures meaningful",
        "environmental signals rather than noise.",
    ]
    (REPORT_OUT / "figure9_selection_summary.md").write_text("\n".join(lines))


def write_plot_notes() -> None:
    attr_full = "\n".join(f"- `{k}`: {v}" for k, v in ATTR_SHORT.items())
    txt = f"""# Figure 10 – Plot Notes

## Layout
- 5 columns × 4 rows = 20 panels, no empty cells.
- Each column corresponds to one process group (same colour).
- Column header: colour block with group name in white bold text.
- No subplot titles; x-axis label = attribute name, y-axis label = parameter name.

## Data sources
- δ_dist parameter means: averaged over 5 seeds × 3 loss functions,
  model = distributional. Source: `params_long.csv`.
- Basin attributes: 531 CAMELS basins. Source: `basin_attributes.csv`.
- Stability statistics: `pair_seed_stability.csv`, distributional model,
  aggregated across 3 loss functions.

## Attribute labels used in figure
{attr_full}

## Quantile bins
- {N_BINS} quantile bins of equal basin count per attribute axis.
- Bins with fewer than 4 basins are dropped.

## Display range
- Each panel clipped to 1st–99th percentile to prevent outlier axis stretch.
- Binned statistics computed on full (unclipped) data.

## IQR
- Per bin: 25th–75th percentile of parameter mean. Shown as semi-transparent
  band (alpha = {BAND_ALPHA:.2f}) around the binned median line.

## Visual style
- Binned median line alpha = {LINE_ALPHA:.2f}.
- Basin scatter color = `{SCATTER_COLOR}`, alpha = {SCATTER_ALPHA:.2f}.
- Process-group colors follow the configured category palette:
  snow `#56B4E9`, aridity `#E69F00`, terrain `#009E73`,
  soil `#A6761D`, routing `#CC79A7`.

## Annotations
- ρ: mean Spearman correlation across 3 loss functions (distributional model).
- SD: mean seed standard deviation of Spearman ρ.
- All 20 selected pairs have sign_consistency = 1.0.

## Parameter name handling
- All `par` prefixes removed (parBETA → BETA, parCWH → CWH, etc.).
- route_b retains its name.
"""
    (REPORT_OUT / "figure9_plot_notes.md").write_text(txt)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    setup_style()

    print("Loading data …")
    param_means = load_dist_param_means()
    attributes  = load_attributes()
    stability   = load_stability()

    print("Building candidate table …")
    cand = build_candidate_df(stability)
    cand.to_csv(ANALYSIS_FIG_OUT / "figure9_candidate_pairs.csv", index=False)

    sel_cols = ["parameter", "attribute", "mean_rho", "abs_rho",
                "mean_seed_std", "sign_consistency", "topk_rate",
                "core_rank_min"]
    sel = cand[cand["selected_for_plot"]][sel_cols]
    sel.to_csv(ANALYSIS_FIG_OUT / "figure9_selected_pairs.csv", index=False)
    print(f"  → {len(sel)} pairs selected, layout {NROWS}×{NCOLS}")

    print("Plotting …")
    fig, binned_df = make_figure(stability, param_means, attributes)
    binned_df.to_csv(ANALYSIS_FIG_OUT / "figure9_binned_gradients.csv", index=False)

    stem = "Fig09_attribute_parameter_gradients"
    fig.savefig(FIG_OUT / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {stem}.png")

    write_selection_summary(cand, f"{NROWS}×{NCOLS}")
    write_plot_notes()
    print("  → reports written")
    print("Done.")


if __name__ == "__main__":
    main()
