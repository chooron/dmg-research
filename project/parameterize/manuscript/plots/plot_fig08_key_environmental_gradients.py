"""
Figure 7 — Key environmental gradients (mean + uncertainty).

Layout: 3 rows × 4 cols = 12 panels
  (a)–(f): parameter MEAN gradients (6 pairs)
  (g)–(i): parameter STD/uncertainty gradients (3 pairs)

Run:
    python plot_fig07_key_environmental_gradients.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

from common import p_label

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/workspace/autoresearch")
MANUSCRIPT_ROOT = ROOT / "project" / "parameterize" / "manuscript"
ANALYSIS_ROOT = MANUSCRIPT_ROOT / "analysis"

MEAN_MAP_FILE = (
    ANALYSIS_ROOT
    / "03_distributional_parameter_spatial_data"
    / "data"
    / "distributional_parameter_mean_maps_long.csv"
)
STD_MAP_FILE = (
    ANALYSIS_ROOT
    / "06_uncertainty_spatial_data"
    / "data"
    / "distributional_parameter_uncertainty_maps_long.csv"
)
GROUP_ASSIGN_FILE = (
    ANALYSIS_ROOT
    / "05_environmental_gradient_groups"
    / "data"
    / "gradient_group_assignments.csv"
)
MEAN_CORR_FILE = (
    ANALYSIS_ROOT
    / "04_mean_attribute_relationships"
    / "data"
    / "distributional_mean_attribute_correlations.csv"
)
STD_CORR_FILE = (
    ANALYSIS_ROOT
    / "07_uncertainty_attribute_relationships"
    / "data"
    / "distributional_std_attribute_correlations.csv"
)
BASIN_ATTR_FILE = (
    ROOT
    / "project"
    / "parameterize"
    / "outputs"
    / "analysis"
    / "stability_stats"
    / "tables"
    / "basin_attributes.csv"
)

OUT_DATA = ANALYSIS_ROOT / "figure7" / "data"
OUT_REPORT = ANALYSIS_ROOT / "figure7" / "reports"
OUT_FIG = MANUSCRIPT_ROOT / "figures" / "main"

OUT_PNG = OUT_FIG / "Fig08_key_environmental_gradients.png"
MEAN_STATS_FILE = OUT_DATA / "fig07_mean_gradient_group_stats.csv"
STD_STATS_FILE = OUT_DATA / "fig07_uncertainty_gradient_group_stats.csv"
SPEARMAN_FILE = OUT_DATA / "fig07_gradient_spearman_summary.csv"
NOTES_FILE = OUT_REPORT / "fig07_plot_notes.md"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DPI = 600
MM = 1 / 25.4

# Mean gradient pairs: (attribute, parameter, panel_label)
MEAN_RELS = [
    ("frac_snow",         "parCWH",  "a"),
    ("frac_snow",         "parCFR",  "b"),
    ("aridity",           "parPERC", "c"),
    ("slope_mean",        "parBETA", "d"),
    ("soil_conductivity", "parUZL",  "e"),
    ("pet_mean",          "parFC",   "f"),
]

# Uncertainty gradient pairs: (attribute, parameter, panel_label) — 6 strongest clean
STD_RELS = [
    ("frac_snow",  "parCFMAX", "g"),
    ("frac_snow",  "parTT",    "h"),
    ("elev_mean",  "parCFMAX", "i"),
    ("elev_mean",  "parTT",    "j"),
    ("pet_mean",   "parTT",    "k"),
    ("slope_mean", "parCFMAX", "l"),
]

GROUP_ORDER = ["low", "middle", "high"]
GROUP_LABELS = ["Low", "Middle", "High"]

COLORS = {
    "low":    "#0072B2",
    "middle": "#AAB7C4",
    "high":   "#6A3D9A",
}

ATTR_XLABEL = {
    "frac_snow":         "Snow fraction",
    "aridity":           "Aridity",
    "slope_mean":        "Slope",
    "soil_conductivity": "Soil conductivity",
    "pet_mean":          "PET",
    "elev_mean":         "Elevation",
}


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def setup_style() -> None:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": DPI,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mean_df = pd.read_csv(MEAN_MAP_FILE)
    std_df = pd.read_csv(STD_MAP_FILE)
    assign_df = pd.read_csv(GROUP_ASSIGN_FILE)
    basin_df = pd.read_csv(BASIN_ATTR_FILE)
    return mean_df, std_df, assign_df, basin_df


def get_tercile_group(assign_df: pd.DataFrame, basin_df: pd.DataFrame, attr: str) -> pd.DataFrame:
    """Return DataFrame with [basin_id, gradient_group] for the given attribute."""
    # gradient_group_assignments.csv covers: aridity, frac_snow, slope_mean, pet_mean,
    # soil_conductivity, p_seasonality — use it when available
    if attr in assign_df["gradient_attribute"].unique():
        grp = (
            assign_df[assign_df["gradient_attribute"] == attr][["basin_id", "gradient_group"]]
            .copy()
        )
        grp = grp[grp["gradient_group"].isin(GROUP_ORDER)]
        return grp
    # For other attributes (e.g. elev_mean) compute terciles from basin_attributes
    if attr not in basin_df.columns:
        raise ValueError(f"Attribute '{attr}' not found in basin_attributes.csv")
    sub = basin_df[["basin_id", attr]].dropna()
    labels = pd.qcut(sub[attr], 3, labels=["low", "middle", "high"])
    return pd.DataFrame({"basin_id": sub["basin_id"].values, "gradient_group": labels.values})


def build_basin_data(
    mean_df: pd.DataFrame,
    std_df: pd.DataFrame,
    assign_df: pd.DataFrame,
    basin_df: pd.DataFrame,
) -> tuple[dict, dict]:
    """Build basin-level dicts for mean and std panels."""
    mean_data: dict = {}
    for attr, par, _ in MEAN_RELS:
        pvals = mean_df[mean_df["parameter"] == par][["basin_id", "seed_mean"]].copy()
        pvals = pvals.rename(columns={"seed_mean": "param_val"})
        grp = get_tercile_group(assign_df, basin_df, attr)
        merged = pvals.merge(grp, on="basin_id", how="inner")
        mean_data[(attr, par)] = merged

    std_data: dict = {}
    for attr, par, _ in STD_RELS:
        pvals = std_df[std_df["parameter"] == par][["basin_id", "parameter_std_unit"]].copy()
        pvals = pvals.rename(columns={"parameter_std_unit": "param_val"})
        grp = get_tercile_group(assign_df, basin_df, attr)
        merged = pvals.merge(grp, on="basin_id", how="inner")
        std_data[(attr, par)] = merged

    return mean_data, std_data


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _group_stats(df: pd.DataFrame, attr: str, par: str, panel: str, kind: str) -> list[dict]:
    rows = []
    for grp in GROUP_ORDER:
        vals = df[df["gradient_group"] == grp]["param_val"].dropna().values
        rows.append({
            "panel": panel, "kind": kind, "attribute": attr, "parameter": par,
            "gradient_group": grp, "n": len(vals),
            "median": np.median(vals) if len(vals) > 0 else np.nan,
            "mean":   np.mean(vals)   if len(vals) > 0 else np.nan,
            "std":    np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
            "q25":    np.percentile(vals, 25) if len(vals) > 0 else np.nan,
            "q75":    np.percentile(vals, 75) if len(vals) > 0 else np.nan,
        })
    return rows


def compute_stats(
    mean_data: dict,
    std_data: dict,
    mean_corr: pd.DataFrame,
    std_corr: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # ---- Spearman from pre-computed tables ----
    all_rels = [(attr, par, panel, "mean") for attr, par, panel in MEAN_RELS] + \
               [(attr, par, panel, "std")  for attr, par, panel in STD_RELS]

    p_vals_raw: list[float] = []
    for attr, par, _, kind in all_rels:
        corr_df = mean_corr if kind == "mean" else std_corr
        sub = corr_df[(corr_df["attribute"] == attr) & (corr_df["parameter"] == par)]
        p_vals_raw.append(float(sub["p_value"].values[0]) if len(sub) > 0 else np.nan)

    valid_mask = [not np.isnan(p) for p in p_vals_raw]
    valid_ps = [p for p, m in zip(p_vals_raw, valid_mask) if m]
    if valid_ps:
        _, q_vals_valid, _, _ = multipletests(valid_ps, method="fdr_bh")
    else:
        q_vals_valid = []
    q_iter = iter(q_vals_valid)
    q_map: dict = {}
    for (attr, par, _, kind), m in zip(all_rels, valid_mask):
        q_map[(attr, par, kind)] = float(next(q_iter)) if m else np.nan

    spearman_rows: list[dict] = []
    mean_stat_rows: list[dict] = []
    std_stat_rows: list[dict] = []

    for attr, par, panel, kind in all_rels:
        corr_df = mean_corr if kind == "mean" else std_corr
        data_dict = mean_data if kind == "mean" else std_data
        df = data_dict[(attr, par)]

        sub = corr_df[(corr_df["attribute"] == attr) & (corr_df["parameter"] == par)]
        if len(sub) > 0:
            rho = float(sub["spearman_rho"].values[0])
            pv  = float(sub["p_value"].values[0])
        else:
            # compute on the fly
            xy = df[["param_val"]].join(
                df.set_index("basin_id").reindex(df["basin_id"])["gradient_group"]
            ).dropna()
            # use numeric rank of group as proxy
            rank_map = {"low": 0, "middle": 1, "high": 2}
            grp_num = df["gradient_group"].map(rank_map)
            valid = df["param_val"].notna() & grp_num.notna()
            if valid.sum() > 2:
                res = spearmanr(df.loc[valid, "param_val"].values, grp_num[valid].values)
                rho, pv = float(res.statistic), float(res.pvalue)
            else:
                rho, pv = np.nan, np.nan

        qv = q_map.get((attr, par, kind), np.nan)

        # high-low delta
        low_vals  = df[df["gradient_group"] == "low"]["param_val"].dropna().values
        high_vals = df[df["gradient_group"] == "high"]["param_val"].dropna().values
        if len(low_vals) > 0 and len(high_vals) > 0:
            delta_hl = float(np.median(high_vals) - np.median(low_vals))
            _, mw_p = mannwhitneyu(high_vals, low_vals, alternative="two-sided")
        else:
            delta_hl, mw_p = np.nan, np.nan

        spearman_rows.append({
            "panel": panel, "kind": kind, "attribute": attr, "parameter": par,
            "spearman_rho": rho, "p_value": pv, "q_value": qv,
            "high_minus_low_median": delta_hl, "mw_p_value": mw_p,
            "n_basins": len(df),
        })

        stat_rows = _group_stats(df, attr, par, panel, kind)
        if kind == "mean":
            mean_stat_rows.extend(stat_rows)
        else:
            std_stat_rows.extend(stat_rows)

    return (
        pd.DataFrame(spearman_rows),
        pd.DataFrame(mean_stat_rows),
        pd.DataFrame(std_stat_rows),
    )


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------
def _format_q(q: float) -> str:
    if np.isnan(q):
        return ""
    if q < 0.001:
        return "q < 0.001"
    if q < 0.01:
        return f"q = {q:.3f}"
    return f"q = {q:.2f}"


def draw_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    grp_stats: pd.DataFrame,
    spearman_row: pd.Series,
    attr: str,
    par: str,
    ylabel: str,
    panel_label: str,
) -> None:
    rng = np.random.default_rng(42)

    for xi, grp in enumerate(GROUP_ORDER):
        color = COLORS[grp]
        vals = df[df["gradient_group"] == grp]["param_val"].dropna().values
        if len(vals) == 0:
            continue

        # jitter scatter
        jx = rng.uniform(-0.22, 0.22, size=len(vals)) + xi
        ax.scatter(jx, vals, color=color, alpha=0.28, s=4, linewidths=0, zorder=2)

        # boxplot elements
        row = grp_stats[grp_stats["gradient_group"] == grp]
        if len(row) == 0:
            continue
        med = float(row["median"].values[0])
        q25 = float(row["q25"].values[0])
        q75 = float(row["q75"].values[0])
        iqr = q75 - q25
        lo_whisk = max(float(np.min(vals)), q25 - 1.5 * iqr)
        hi_whisk = min(float(np.max(vals)), q75 + 1.5 * iqr)

        box_w = 0.38
        rect = mpl.patches.FancyBboxPatch(
            (xi - box_w / 2, q25), box_w, iqr,
            boxstyle="square,pad=0",
            facecolor=mpl.colors.to_rgba(color, 0.55),
            edgecolor="#444444", linewidth=0.8, zorder=3,
        )
        ax.add_patch(rect)
        ax.plot([xi - box_w / 2, xi + box_w / 2], [med, med],
                color="#111111", linewidth=1.3, zorder=4)
        ax.plot([xi, xi], [q25, lo_whisk], color="#555555", linewidth=0.7, zorder=3)
        ax.plot([xi, xi], [q75, hi_whisk], color="#555555", linewidth=0.7, zorder=3)
        ax.plot([xi - 0.12, xi + 0.12], [lo_whisk, lo_whisk],
                color="#555555", linewidth=0.7, zorder=3)
        ax.plot([xi - 0.12, xi + 0.12], [hi_whisk, hi_whisk],
                color="#555555", linewidth=0.7, zorder=3)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Low", "Middle", "High"], fontsize=10.0)
    ax.set_xlim(-0.55, 2.55)
    ax.set_xlabel(ATTR_XLABEL.get(attr, attr), fontsize=10.5, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=10.5, labelpad=3)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=10.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)

    # annotation: rho, delta H-L, q
    rho = float(spearman_row["spearman_rho"])
    qv  = float(spearman_row["q_value"])
    dhl = float(spearman_row["high_minus_low_median"])
    q_str = _format_q(qv)
    ann_lines = [f"ρ = {rho:+.2f}", f"ΔH–L = {dhl:+.2f}"]
    if q_str:
        ann_lines.append(q_str)
    ax.text(0.97, 0.97, "\n".join(ann_lines),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9.2, color="#333333", linespacing=1.35,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.5))

    # panel label inside top-left
    ax.text(0.03, 0.97, f"({panel_label})",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12.5, fontweight="normal", color="#111111")


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def make_figure(
    mean_data: dict,
    std_data: dict,
    spearman_df: pd.DataFrame,
    mean_stats: pd.DataFrame,
    std_stats: pd.DataFrame,
) -> None:
    # 3 rows × 4 cols using GridSpec so we can control the gap between col-groups
    fig = plt.figure(figsize=(260 * MM, 210 * MM), constrained_layout=False)
    # Use a nested GridSpec: outer has 2 column groups separated by extra space
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.18)
    gs_left  = GridSpecFromSubplotSpec(3, 2, subplot_spec=outer[0], hspace=0.35, wspace=0.38)
    gs_right = GridSpecFromSubplotSpec(3, 2, subplot_spec=outer[1], hspace=0.35, wspace=0.38)

    mean_axes = [[fig.add_subplot(gs_left[r, c]) for c in range(2)] for r in range(3)]
    std_axes  = [[fig.add_subplot(gs_right[r, c]) for c in range(2)] for r in range(3)]

    mean_positions = [(r, c) for r in range(3) for c in range(2)]
    std_positions  = [(r, c) for r in range(3) for c in range(2)]

    for (r, c), (attr, par, panel_label) in zip(mean_positions, MEAN_RELS):
        ax = mean_axes[r][c]
        df = mean_data[(attr, par)]
        grp_stats = mean_stats[(mean_stats["attribute"] == attr) & (mean_stats["parameter"] == par)]
        sp_row = spearman_df[(spearman_df["attribute"] == attr) & (spearman_df["parameter"] == par)].iloc[0]
        ylabel = f"{p_label(par)} mean"
        draw_panel(ax, df, grp_stats, sp_row, attr, par, ylabel, panel_label)

    for (r, c), (attr, par, panel_label) in zip(std_positions, STD_RELS):
        ax = std_axes[r][c]
        df = std_data[(attr, par)]
        grp_stats = std_stats[(std_stats["attribute"] == attr) & (std_stats["parameter"] == par)]
        sp_row = spearman_df[(spearman_df["attribute"] == attr) & (spearman_df["parameter"] == par)].iloc[0]
        ylabel = f"{p_label(par)} std"
        draw_panel(ax, df, grp_stats, sp_row, attr, par, ylabel, panel_label)

    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.08, top=0.93)

    # vertical divider drawn after subplots_adjust so position is stable
    # midpoint between the two outer column groups in figure coords
    fig.add_artist(mpl.lines.Line2D(
        [0.505, 0.505], [0.06, 0.94],
        transform=fig.transFigure,
        color="#AAAAAA", linewidth=1.0, linestyle="--", zorder=10,
    ))

    # shared legend at top
    legend_handles = [
        Patch(facecolor=mpl.colors.to_rgba(COLORS[g], 0.55),
              edgecolor="#444444", linewidth=0.8, label=lbl)
        for g, lbl in zip(GROUP_ORDER, GROUP_LABELS)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        frameon=False,
        fontsize=10.0,
        handlelength=1.4,
        handletextpad=0.5,
        columnspacing=1.2,
    )

    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------
def write_notes(spearman_df: pd.DataFrame) -> None:
    lines = [
        "# Fig08 key environmental gradients - plot notes",
        "",
        "## 1. Figure structure",
        "",
        "- Layout: 3 rows × 4 cols = 12 panels.",
        "- Panels (a)–(f): parameter MEAN gradients (seed_mean column from distributional_parameter_mean_maps_long.csv).",
        "- Panels (g)–(i): parameter STD/uncertainty gradients (parameter_std_unit column from distributional_parameter_uncertainty_maps_long.csv).",
        "- Each panel shows a boxplot + jitter scatter across Low / Middle / High tercile groups.",
        "",
        "## 2. Mean gradient panels (a)–(f)",
        "",
        "- (a) frac_snow → parCWH mean",
        "- (b) frac_snow → parCFR mean",
        "- (c) aridity → parPERC mean",
        "- (d) slope_mean → parBETA mean",
        "- (e) soil_conductivity → parUZL mean",
        "- (f) pet_mean → parFC mean",
        "",
        "## 3. Uncertainty gradient panels (g)–(i)",
        "",
        "- (g) frac_snow → parCFMAX std  (ρ ≈ −0.919, strongest clean std gradient)",
        "- (h) elev_mean → parTT std     (ρ ≈ −0.590, second strongest clean std gradient)",
        "- (i) pet_mean → parCFMAX std   (ρ ≈ +0.477, third strongest clean std gradient)",
        "",
        "## 4. Interpretation note on uncertainty gradients",
        "",
        "- The std gradients in panels (g)–(i) represent structured diagnostic signal:",
        "  parameter uncertainty varies systematically with basin attributes.",
        "- This is NOT evidence of true parameter identifiability; it reflects that the",
        "  distributional model's seed-to-seed spread is geographically structured.",
        "- Interpretation should be cautious for parameters with strong mean–std coupling",
        "  or boundary effects (see Fig 8 panel c).",
        "",
        "## 5. Tercile group assignments",
        "",
        "- frac_snow, aridity, slope_mean, pet_mean, soil_conductivity: from gradient_group_assignments.csv.",
        "- elev_mean: computed directly from basin_attributes.csv using pd.qcut (equal-frequency terciles).",
        "",
        "## 6. Statistical annotations",
        "",
        "- ρ: Spearman correlation from pre-computed tables (distributional_mean_attribute_correlations.csv",
        "  or distributional_std_attribute_correlations.csv).",
        "- q: FDR-BH adjusted p-value across all 12 panels jointly.",
        "- ΔH–L: median(High) − median(Low).",
        "",
        "## 7. Key statistics per panel",
        "",
    ]
    for _, row in spearman_df.iterrows():
        lines.append(
            f"- ({row['panel']}) [{row['kind']}] {row['attribute']} → {row['parameter']}: "
            f"ρ = {row['spearman_rho']:+.3f}, q = {row['q_value']:.2e}, "
            f"ΔH–L = {row['high_minus_low_median']:+.3f}"
        )
    lines += [
        "",
        "## 8. Output files",
        "",
        f"- PNG: {OUT_PNG}",
        f"- Mean group stats: {MEAN_STATS_FILE}",
        f"- Std group stats: {STD_STATS_FILE}",
        f"- Spearman summary: {SPEARMAN_FILE}",
    ]
    NOTES_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    setup_style()
    for d in (OUT_DATA, OUT_REPORT, OUT_FIG):
        d.mkdir(parents=True, exist_ok=True)

    mean_df, std_df, assign_df, basin_df = load_data()
    mean_corr = pd.read_csv(MEAN_CORR_FILE)
    std_corr  = pd.read_csv(STD_CORR_FILE)

    mean_data, std_data = build_basin_data(mean_df, std_df, assign_df, basin_df)
    spearman_df, mean_stats, std_stats = compute_stats(
        mean_data, std_data, mean_corr, std_corr
    )

    mean_stats.to_csv(MEAN_STATS_FILE, index=False)
    std_stats.to_csv(STD_STATS_FILE, index=False)
    spearman_df.to_csv(SPEARMAN_FILE, index=False)

    make_figure(mean_data, std_data, spearman_df, mean_stats, std_stats)
    write_notes(spearman_df)

    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {MEAN_STATS_FILE}")
    print(f"Saved: {STD_STATS_FILE}")
    print(f"Saved: {SPEARMAN_FILE}")
    print(f"Saved: {NOTES_FILE}")


if __name__ == "__main__":
    main()
