"""
Case basin parameter regimes – δ_dist representative basin analysis.

Layout (3 panels, revised Fig11):
  Left column:  (a) CONUS representative-basin map + (c) merged parameter bubble heatmap
  Right column: (b) environmental percentile-rank curves arranged as 3 rows × 2 columns

Panel (c) merges the previous c/d panels:
  color = parameter mean deviation from all-basin median
  size  = within-parameter percentile rank of the parameter mean

Outputs
-------
  manuscript/analysis/case_study/case_basin_selection.csv
  manuscript/analysis/case_study/case_basin_attribute_profiles.csv
  manuscript/analysis/case_study/case_basin_parameter_regimes.csv
  manuscript/analysis/case_study/case_study_plot_notes.md
  manuscript/figures/main/Fig11_revised_layout.png
  manuscript/figures/main/Fig11_revised_layout.pdf
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
from matplotlib.patches import Circle
from scipy.stats import rankdata

from common import PARAM_LABELS

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT           = Path("/workspace/autoresearch")
PARAM_ROOT     = ROOT / "project" / "parameterize"
MANUSCRIPT_ROOT = PARAM_ROOT / "manuscript"

MEAN_MAP_FILE  = (MANUSCRIPT_ROOT / "analysis" / "03_distributional_parameter_spatial_data"
                  / "data" / "distributional_parameter_mean_maps_long.csv")
UNCERTAINTY_FILE = (MANUSCRIPT_ROOT / "analysis" / "06_uncertainty_spatial_data"
                    / "data" / "distributional_parameter_uncertainty_maps_long.csv")
BASIN_ATTR_FILE = (PARAM_ROOT / "outputs" / "analysis" / "stability_stats"
                   / "tables" / "basin_attributes.csv")
METRICS_FILE   = (PARAM_ROOT / "outputs" / "analysis" / "stability_stats"
                  / "tables" / "metrics_long.csv")
ANALYSIS_OUT   = MANUSCRIPT_ROOT / "analysis" / "case_study"
MAIN_FIG_DIR   = MANUSCRIPT_ROOT / "figures" / "main"
CONUS_DIR      = ROOT / "data" / "camels_loc" / "conus_clipped"
STATE_SHP      = CONUS_DIR / "s_18mr25_conus.shp"
OUT_PNG        = MAIN_FIG_DIR / "Fig11_revised_layout.png"
OUT_PDF        = MAIN_FIG_DIR / "Fig11_revised_layout.pdf"

DPI = 600
MM  = 1 / 25.4

# ---------------------------------------------------------------------------
# Parameter configuration (consistent with fig05/fig06/fig09)
# ---------------------------------------------------------------------------
PARAMETER_ORDER = [
    "parTT", "parCFMAX", "parCFR", "parCWH",
    "parBETA", "parFC", "parLP", "parPERC",
    "parUZL", "parK0", "parK1", "parK2",
    "route_a", "route_b",
]
PARAM_GROUP = {
    "parTT":"Snow","parCFMAX":"Snow","parCFR":"Snow","parCWH":"Snow",
    "parBETA":"Production/soil","parFC":"Production/soil",
    "parLP":"Production/soil","parPERC":"Production/soil",
    "parUZL":"Storage/recession","parK0":"Storage/recession",
    "parK1":"Storage/recession","parK2":"Storage/recession",
    "route_a":"Routing","route_b":"Routing",
}
# Colors matching fig09 GROUP_COLORS
PARAM_GROUP_COLORS = {
    "Snow":              "#56B4E9",
    "Production/soil":   "#E69F00",
    "Storage/recession": "#CC79A7",
    "Routing":           "#0072B2",
}
# Physical search ranges (from fig05 PARAMETER_BOUNDS)
PARAM_RANGES = {
    "parTT":   (-2.5, 2.5), "parCFMAX":(0.5,10.0),
    "parCFR":  (0.0,  0.1), "parCWH":  (0.0, 0.2),
    "parBETA": (1.0,  6.0), "parFC":   (50.0,1000.0),
    "parLP":   (0.2,  1.0), "parPERC": (0.0, 10.0),
    "parUZL":  (0.0,100.0), "parK0":   (0.05,0.9),
    "parK1":   (0.01, 0.5), "parK2":   (0.001,0.2),
    "route_a": (0.0,  2.9), "route_b": (0.0, 6.5),
}

# ---------------------------------------------------------------------------
# Attribute configuration
# ---------------------------------------------------------------------------
ATTR_COLS = [
    "aridity","frac_snow","p_seasonality","slope_mean",
    "pet_mean","soil_conductivity","area_gages2","p_mean",
]
ATTR_LABELS = {
    "aridity":"Aridity","frac_snow":"Snow frac.",
    "p_seasonality":"P season.","slope_mean":"Slope",
    "pet_mean":"PET","soil_conductivity":"Soil cond.",
    "area_gages2":"Area","p_mean":"P mean",
}
PANEL_B_ATTRS = [
    ("frac_snow", "Snow fraction"),
    ("aridity", "Aridity"),
    ("slope_mean", "Slope"),
    ("soil_conductivity", "Soil conductivity"),
    ("pet_mean", "PET"),
    ("area_gages2", "Area"),
]

# ---------------------------------------------------------------------------
# Case configuration
# ---------------------------------------------------------------------------
CASE_ORDER = ["snow","arid","humid","steep","soil_storage","routing"]
CASE_IDS   = {
    "snow":"S1","arid":"A1","humid":"H1",
    "steep":"T1","soil_storage":"G1","routing":"R1",
}
CASE_TYPE_LABELS = {
    "snow":"Snow","arid":"Arid","humid":"Humid",
    "steep":"Steep","soil_storage":"Soil","routing":"Routing",
}
PANEL_D_PARAM_LABELS = {
    "parTT": "TT",
    "parCFMAX": "CFMAX",
    "parCFR": "CFR",
    "parCWH": "CWH",
    "parBETA": "BETA",
    "parFC": "FC",
    "parLP": "LP",
    "parPERC": "PERC",
    "parUZL": "UZL",
    "parK0": "K\u2080",
    "parK1": "K\u2081",
    "parK2": "K\u2082",
    "route_a": "UH\u2090",
    "route_b": "UH\u1d66",
}
# Low-saturation colors aligned with fig09 group palette
CASE_COLORS = {
    "snow":        "#56B4E9",
    "arid":        "#E69F00",
    "humid":       "#0072B2",
    "steep":       "#009E73",
    "soil_storage":"#CC79A7",
    "routing":     "#6A3D9A",
}
PANEL_B_CASE_COLORS = CASE_COLORS
PANEL_D_CASE_LABELS = CASE_TYPE_LABELS

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def setup_style() -> None:
    mpl.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman"],
        "mathtext.fontset":  "custom",
        "mathtext.rm":       "Times New Roman",
        "mathtext.it":       "Times New Roman:italic",
        "mathtext.bf":       "Times New Roman:bold",
        "font.size":         10.5,
        "axes.labelsize":    10.5,
        "xtick.labelsize":   10.0,
        "ytick.labelsize":   10.0,
        "legend.fontsize":   10.0,
        "axes.linewidth":    0.6,
        "axes.edgecolor":    "#444444",
        "axes.facecolor":    "white",
        "figure.facecolor":  "white",
        "axes.grid":         False,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "savefig.dpi":       DPI,
        "savefig.facecolor": "white",
    })


def add_panel_label(ax: plt.Axes, label: str,
                    x: float = 0.02, y: float = 0.97) -> None:
    """Place panel label inside the axes (top-left by default)."""
    ax.text(x, y, label, transform=ax.transAxes,
            ha="left", va="top", fontsize=12.0,
            fontweight="normal", color="#111111",
            zorder=10)


def clean_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mean_df = pd.read_csv(MEAN_MAP_FILE)
    unc_df  = pd.read_csv(UNCERTAINTY_FILE)
    attrs   = pd.read_csv(BASIN_ATTR_FILE)
    metrics = pd.read_csv(METRICS_FILE)
    return mean_df, unc_df, attrs, metrics


def build_param_table(mean_df: pd.DataFrame,
                      unc_df:  pd.DataFrame) -> pd.DataFrame:
    """Wide table: basin_id × parameter, columns = *_mean_norm + *_std_norm."""
    mean_wide = mean_df.pivot(index="basin_id", columns="parameter", values="seed_mean")
    std_wide  = unc_df.pivot(index="basin_id",  columns="parameter", values="parameter_std_unit")

    mean_norm = mean_wide.copy()
    std_norm  = std_wide.copy()
    for p, (lo, hi) in PARAM_RANGES.items():
        rng = hi - lo
        if p in mean_norm.columns:
            mean_norm[p] = (mean_wide[p] - lo) / rng
            std_norm[p]  = std_wide[p] / rng

    mean_norm = mean_norm.clip(0, 1)
    std_norm  = std_norm.clip(0, None)

    # Compute deviation from all-basin median (in normalized space)
    dev = mean_norm - mean_norm.median()

    mean_norm.columns = [f"{c}_mean_norm" for c in mean_norm.columns]
    std_norm.columns  = [f"{c}_std_norm"  for c in std_norm.columns]
    dev.columns       = [f"{c}_dev"       for c in dev.columns]

    return pd.concat([mean_norm, std_norm, dev], axis=1).reset_index()


def build_perf(metrics: pd.DataFrame) -> pd.DataFrame:
    dist = metrics[metrics["model"] == "distributional"]
    return dist.groupby("basin_id", as_index=False).agg(
        nse=("nse","median"), kge=("kge","median")
    )


# ---------------------------------------------------------------------------
# Case basin selection  (NSE >= median AND KGE >= median)
# ---------------------------------------------------------------------------
def select_cases(attrs: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    df = attrs.merge(perf, on="basin_id", how="inner")
    n  = len(df)

    for col in ATTR_COLS + ["elev_mean"]:
        if col in df.columns:
            df[f"{col}_pct"] = (
                rankdata(df[col].fillna(df[col].median()), method="average") / n
            )

    nse_med = perf["nse"].median()
    kge_med = perf["kge"].median()
    # Primary filter: both NSE and KGE >= median
    ok = df[(df["nse"] >= nse_med) & (df["kge"] >= kge_med)].copy()
    used: set[int] = set()
    rows: list[dict] = []

    def pick(mask: pd.Series, ctype: str, reason: str,
             prefer_high: list[str],
             prefer_low:  list[str] | None = None) -> None:
        cands = ok[mask & ~ok["basin_id"].isin(used)].copy()
        if cands.empty:
            # Fallback: NSE >= 20th pct
            fallback = df[df["nse"] >= df["nse"].quantile(0.2)]
            cands = fallback[mask & ~fallback["basin_id"].isin(used)].copy()
        if cands.empty:
            return
        score = sum(
            cands[f"{a}_pct"] for a in prefer_high if f"{a}_pct" in cands.columns
        )
        if prefer_low:
            score = score - sum(
                cands[f"{a}_pct"] for a in prefer_low if f"{a}_pct" in cands.columns
            )
        cands = cands.copy()
        cands["_score"] = score
        best = cands.nlargest(1, "_score").iloc[0]
        used.add(int(best["basin_id"]))
        rows.append({
            "case_id":           CASE_IDS[ctype],
            "basin_id":          int(best["basin_id"]),
            "case_type":         ctype,
            "NSE":               round(best["nse"], 3),
            "KGE":               round(best["kge"], 3),
            "selected_reason":   reason,
            "aridity":           round(best.get("aridity",           np.nan), 3),
            "frac_snow":         round(best.get("frac_snow",         np.nan), 3),
            "p_seasonality":     round(best.get("p_seasonality",     np.nan), 3),
            "slope_mean":        round(best.get("slope_mean",        np.nan), 2),
            "pet_mean":          round(best.get("pet_mean",          np.nan), 3),
            "soil_conductivity": round(best.get("soil_conductivity", np.nan), 3),
            "area_gages2":       round(best.get("area_gages2",       np.nan), 1),
            "p_mean":            round(best.get("p_mean",            np.nan), 3),
        })

    pick(ok["frac_snow_pct"] >= 0.90,
         "snow", "top 10% frac_snow; NSE & KGE >= median",
         ["frac_snow"])

    pick(ok["aridity_pct"] >= 0.80,
         "arid", "top 20% aridity; NSE & KGE >= median",
         ["aridity", "kge"])

    pick(ok["aridity_pct"] <= 0.25,
         "humid", "bottom 25% aridity; lowest slope; NSE & KGE >= median",
         prefer_high=["p_mean"], prefer_low=["slope_mean"])

    pick(ok["slope_mean_pct"] >= 0.90,
         "steep", "top 10% slope_mean; NSE & KGE >= median",
         ["slope_mean"])

    pick(ok["soil_conductivity_pct"] >= 0.90,
         "soil_storage", "top 10% soil_conductivity; NSE & KGE >= median",
         ["soil_conductivity"])

    pick(ok["area_gages2_pct"] >= 0.90,
         "routing", "top 10% area_gages2; NSE & KGE >= median",
         ["area_gages2"])

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Panel (a): CONUS map
# ---------------------------------------------------------------------------
def draw_map(ax: plt.Axes, case_df: pd.DataFrame,
             all_locs: pd.DataFrame) -> None:
    conus = gpd.read_file(STATE_SHP)
    conus.plot(ax=ax, color="#F2F2F0", edgecolor="#C8C8C8", linewidth=0.35)

    ax.scatter(all_locs["longitude"], all_locs["latitude"],
               s=4, color="#CCCCCC", alpha=0.5, linewidths=0, zorder=2)

    label_offsets = {
        "S1": ( 5,  4), "A1": ( 5,  4), "H1": ( 5, -7),
        "T1": (-18,  4), "G1": ( 5,  4), "R1": ( 5, -7),
    }
    for _, row in case_df.iterrows():
        loc = all_locs[all_locs["basin_id"] == row["basin_id"]]
        if loc.empty:
            continue
        lon = float(loc.iloc[0]["longitude"])
        lat = float(loc.iloc[0]["latitude"])
        color = CASE_COLORS[row["case_type"]]
        cid   = row["case_id"]
        ax.scatter(lon, lat, s=100, color=color, edgecolors="white",
                   linewidths=0.9, zorder=5)
        dx, dy = label_offsets.get(cid, (5, 4))
        ax.annotate(cid, (lon, lat), xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=10.0, fontweight="bold",
                    color="#111111", zorder=6)

    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=CASE_COLORS[ct],
               markeredgecolor="white", markeredgewidth=0.5,
               markersize=7.2,
               label=f"{CASE_IDS[ct]} {CASE_TYPE_LABELS[ct]}")
        for ct in CASE_ORDER
    ]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=9.5,
                    frameon=True, framealpha=0.88, edgecolor="#CCCCCC",
                    handletextpad=0.3, labelspacing=0.25, borderpad=0.5,
                    ncol=3)
    leg.get_frame().set_linewidth(0.4)

    ax.set_xlim(-128, -65)
    ax.set_ylim(24, 50)
    ax.set_aspect(1.2, adjustable="box")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Panel (b): representative basins on full environmental distributions
# ---------------------------------------------------------------------------
def draw_attr_heatmap(ax: plt.Axes, case_df: pd.DataFrame,
                      attrs: pd.DataFrame) -> list[plt.Axes]:
    """Draw panel (b) as a 3 × 2 block for the right column."""
    fig = ax.figure
    parent_spec = ax.get_subplotspec()
    ax.remove()

    gs_b = GridSpecFromSubplotSpec(
        3, 2,
        subplot_spec=parent_spec,
        hspace=0.48,
        wspace=0.40,
    )
    case_rows = list(case_df.itertuples(index=False))
    label_offsets = {
        "S1": (-12, 8),
        "A1": (11, -9),
        "H1": (11, 8),
        "T1": (-12, 8),
        "G1": (-12, -9),
        "R1": (11, -9),
    }
    attr_label_offsets = {
        "frac_snow": {
            "S1": (-16, 12), "A1": (28, -18), "H1": (42, 34),
            "T1": (-18, 12), "G1": (-44, -30), "R1": (18, -10),
        },
        "slope_mean": {
            "S1": (16, 18), "A1": (22, -22), "H1": (18, 18),
            "T1": (-18, 14), "G1": (16, 12), "R1": (20, -18),
        },
        "soil_conductivity": {
            "S1": (22, 18), "A1": (22, -12), "H1": (-28, -20),
            "T1": (-24, 18), "G1": (-24, 20), "R1": (22, -20),
        },
        "pet_mean": {
            "S1": (-20, -12), "A1": (18, -24), "H1": (-24, 20),
            "T1": (-20, -16), "G1": (18, 18), "R1": (22, -18),
        },
        "area_gages2": {
            "S1": (18, -14), "A1": (36, 12), "H1": (18, 16),
            "T1": (-22, 18), "G1": (18, 20), "R1": (-22, 16),
        },
    }
    axes: list[plt.Axes] = []

    for idx, (attr, title) in enumerate(PANEL_B_ATTRS):
        # Right-column layout requested by reviewer/user: 3 rows × 2 columns.
        sub_ax = fig.add_subplot(gs_b[idx // 2, idx % 2])
        axes.append(sub_ax)
        if attr not in attrs.columns:
            sub_ax.axis("off")
            continue

        attr_df = (
            attrs[["basin_id", attr]]
            .dropna()
            .astype({"basin_id": int})
            .sort_values([attr, "basin_id"], kind="mergesort")
            .reset_index(drop=True)
        )
        n = len(attr_df)
        if n < 2:
            sub_ax.axis("off")
            continue

        rank_x = np.linspace(0, 1, n)
        values = attr_df[attr].to_numpy(dtype=float)
        rank_values = rankdata(values, method="average")
        pct_map = dict(zip(attr_df["basin_id"], (rank_values - 1) / (n - 1)))
        value_map = dict(zip(attr_df["basin_id"], values))

        sub_ax.plot(rank_x, values, color="#B8B8B8", linewidth=0.8, zorder=1)
        for case in case_rows:
            bid = int(case.basin_id)
            if bid not in pct_map:
                continue
            x = float(pct_map[bid])
            y = float(value_map[bid])
            color = PANEL_B_CASE_COLORS[case.case_type]
            sub_ax.vlines(x, ymin=np.nanmin(values), ymax=y,
                          color=color, linewidth=0.45, alpha=0.18, zorder=2)
            sub_ax.scatter(
                x, y,
                s=26,
                color=color,
                edgecolors="white",
                linewidths=0.45,
                zorder=4,
            )
            dx, dy = attr_label_offsets.get(attr, {}).get(
                case.case_id,
                label_offsets.get(case.case_id, (8, 6)),
            )
            sub_ax.annotate(
                case.case_id,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=9.0,
                color="#111111",
                ha="left" if dx >= 0 else "right",
                va="center",
                arrowprops=dict(
                    arrowstyle="-",
                    color="#777777",
                    linewidth=0.42,
                    shrinkA=1.0,
                    shrinkB=2.0,
                    alpha=0.75,
                ),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.25),
                annotation_clip=False,
                zorder=5,
            )

        sub_ax.set_ylabel(title, fontsize=9.0, color="#111111", labelpad=1.6)
        sub_ax.set_xlim(0, 1)
        pad = 0.04 * (np.nanmax(values) - np.nanmin(values))
        if not np.isfinite(pad) or pad == 0:
            pad = 0.05
        sub_ax.set_ylim(np.nanmin(values) - pad, np.nanmax(values) + pad)
        sub_ax.set_xticks([0, 0.5, 1.0])
        sub_ax.set_xticklabels(["0", "0.5", "1"], fontsize=9.0)
        sub_ax.tick_params(axis="y", labelsize=9.0, length=2, pad=1.2)
        sub_ax.tick_params(axis="x", length=2, pad=1.2)
        if idx // 2 == 2:
            sub_ax.set_xlabel("Percentile rank", fontsize=9.0, labelpad=1.5)
        sub_ax.grid(axis="x", color="#E8E8E8", linewidth=0.45)
        clean_ax(sub_ax)
        for spine in ["left", "bottom"]:
            sub_ax.spines[spine].set_linewidth(0.45)
            sub_ax.spines[spine].set_color("#777777")

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=PANEL_B_CASE_COLORS[case_type],
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=4.8,
            label=f"{CASE_IDS[case_type]} {PANEL_D_CASE_LABELS[case_type]}",
        )
        for case_type in CASE_ORDER
    ]
    left = min(a.get_position().x0 for a in axes) if axes else 0.50
    right = max(a.get_position().x1 for a in axes) if axes else 0.97
    bottom = min(a.get_position().y0 for a in axes) if axes else 0.40
    legend = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=((left + right) / 2, bottom - 0.038),
        ncol=2,
        frameon=True,
        framealpha=0.92,
        edgecolor="#CCCCCC",
        fontsize=9.0,
        handletextpad=0.25,
        borderpad=0.25,
        labelspacing=0.18,
        columnspacing=0.65,
    )
    legend.get_frame().set_linewidth(0.4)

    return axes


# ---------------------------------------------------------------------------
# Panel (c): merged bubble heatmap
# ---------------------------------------------------------------------------
def draw_param_heatmap(ax: plt.Axes, case_df: pd.DataFrame,
                       param_table: pd.DataFrame) -> plt.Axes:
    """Merged c/d panel.

    Color keeps the original panel-c meaning:
        parameter mean deviation from the all-basin median.
    Bubble size now carries the former panel-d value:
        within-parameter percentile rank of the parameter mean.
    """
    params = PARAMETER_ORDER
    n_p, n_c = len(params), len(case_df)

    dev_mat = np.full((n_c, n_p), np.nan)
    pct_mat = np.full((n_c, n_p), np.nan)

    # Former panel (d) information: for each parameter, rank all basins by
    # normalized parameter mean and then read the representative basin percentile.
    percentile_table = param_table[["basin_id"]].copy()
    for p in params:
        col = f"{p}_mean_norm"
        if col in param_table.columns:
            percentile_table[f"{p}_mean_pct"] = param_table[col].rank(
                pct=True, method="average"
            )

    for i, (_, row) in enumerate(case_df.iterrows()):
        bid = int(row["basin_id"])
        sub = param_table[param_table["basin_id"] == bid]
        pct_sub = percentile_table[percentile_table["basin_id"] == bid]
        if sub.empty:
            continue
        s = sub.iloc[0]
        pct_s = pct_sub.iloc[0] if not pct_sub.empty else pd.Series(dtype=float)
        for j, p in enumerate(params):
            if f"{p}_dev" in s.index:
                dev_mat[i, j] = s[f"{p}_dev"]
            if f"{p}_mean_pct" in pct_s.index:
                pct_mat[i, j] = pct_s[f"{p}_mean_pct"]

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "case_dev_purple_white_green",
        ["#6A3D9A", "#F7F7F7", "#1B9E77"],
        N=256,
    )

    abs_max = float(np.nanpercentile(np.abs(dev_mat[np.isfinite(dev_mat)]), 95))
    abs_max = max(abs_max, 0.15)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    ax.set_facecolor("#FBFBFB")
    ax.set_xlim(-0.5, n_p - 0.5)
    ax.set_ylim(n_c - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(n_p))
    ax.set_xticklabels([PARAM_LABELS.get(p, p) for p in params],
                       rotation=0, ha="center", fontsize=9.0, color="#111111")
    row_labels = [
        f"{row['case_id']} {CASE_TYPE_LABELS[row['case_type']]}"
        for _, row in case_df.iterrows()
    ]
    ax.set_yticks(np.arange(n_c))
    ax.set_yticklabels(row_labels, fontsize=10.0, color="#111111")
    ax.tick_params(axis="both", length=0, pad=2, colors="#111111")

    ax.set_xticks(np.arange(-0.5, n_p, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_c, 1), minor=True)
    ax.grid(which="minor", color="#D9D9D9", linewidth=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_edgecolor("#111111")

    prev_g = PARAM_GROUP.get(params[0], "")
    for j, p in enumerate(params[1:], 1):
        g = PARAM_GROUP.get(p, "")
        if g != prev_g:
            ax.axvline(j - 0.5, color="#888888", linewidth=0.55, zorder=3)
        prev_g = g

    # Size is directly tied to the former panel-d value.
    size_min = 22.0
    size_max = 200.0

    xs, ys, vals, sizes = [], [], [], []
    for i in range(n_c):
        for j in range(n_p):
            dv = dev_mat[i, j]
            pct = pct_mat[i, j]
            if np.isfinite(dv) and np.isfinite(pct):
                pct = float(np.clip(pct, 0.0, 1.0))
                xs.append(j)
                ys.append(i)
                vals.append(dv)
                sizes.append(size_min + (size_max - size_min) * pct)

    ax.scatter(xs, ys, s=sizes, c=vals, cmap=cmap, norm=norm,
               marker="o", edgecolors="#DDDDDD", linewidths=0.25,
               alpha=0.95, zorder=4)

    # One-line annotation above panel (c).
    # The explanatory text, legend title, and circles are packed into one row,
    # left-aligned with panel (c), with all elements vertically center-aligned.
    size_legend_values = [0.25, 0.50, 0.75, 1.00]

    def _circle_size_item(value: float) -> HPacker:
        marker_diameter = float(np.sqrt(size_min + (size_max - size_min) * value))
        box_size = max(16.0, marker_diameter + 2.0)
        da = DrawingArea(box_size, box_size, 0, 0)
        da.add_artist(Circle(
            (box_size / 2.0, box_size / 2.0),
            marker_diameter / 2.0,
            facecolor="#D9D9D9",
            edgecolor="#777777",
            linewidth=0.35,
        ))
        label = TextArea(f"{value:.2f}", textprops=dict(
            fontsize=9.0, color="#333333", va="center"
        ))
        return HPacker(
            children=[da, label],
            align="center",
            pad=0,
            sep=2.2,
        )

    note_row = HPacker(
        children=[
            TextArea(
                "Circle size = within-parameter percentile rank",
                textprops=dict(fontsize=9.4, color="#333333", va="center"),
            ),
            *[_circle_size_item(v) for v in size_legend_values],
        ],
        align="center",
        pad=0,
        sep=9.0,
    )
    note_box = AnchoredOffsetbox(
        loc="center left",
        child=note_row,
        bbox_to_anchor=(0.0, 1.085),
        bbox_transform=ax.transAxes,
        frameon=False,
        borderpad=0.0,
        pad=0.0,
    )
    note_box.set_clip_on(False)
    ax.add_artist(note_box)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Panel-c colorbar is horizontal and will be manually aligned below
    # panel (c) with panel (b)'s legend in make_figure().
    # Use a standalone colorbar axes so creating the colorbar does not shrink panel (c).
    cax = ax.figure.add_axes([0.10, 0.05, 0.10, 0.015])
    cbar = ax.figure.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=9.2, length=2, colors="#111111", pad=1.5)
    cbar.set_label("Deviation from all-basin median", fontsize=9.8, color="#111111", labelpad=2.0)
    cbar.outline.set_linewidth(0.5)
    plt.setp(cbar.ax.xaxis.get_ticklabels(), color="#111111")
    return cbar.ax



# ---------------------------------------------------------------------------
# Panel (d): compact full-parameter profile heatmap
# ---------------------------------------------------------------------------
def draw_panel_d_heatmap(fig: plt.Figure, gs_d: GridSpecFromSubplotSpec,
                         case_df: pd.DataFrame,
                         param_table: pd.DataFrame) -> tuple[plt.Axes, plt.Axes]:
    params = PARAMETER_ORDER
    matrix = np.full((len(case_df), len(params)), np.nan)
    row_labels: list[str] = []
    percentile_table = param_table[["basin_id"]].copy()
    for p in params:
        col = f"{p}_mean_norm"
        if col in param_table.columns:
            percentile_table[f"{p}_mean_pct"] = param_table[col].rank(
                pct=True, method="average"
            )

    for row_idx, (_, row) in enumerate(case_df.iterrows()):
        ctype = row["case_type"]
        cid = row["case_id"]
        bid = int(row["basin_id"])
        row_labels.append(f"{cid} {PANEL_D_CASE_LABELS[ctype]}")
        sub = percentile_table[percentile_table["basin_id"] == bid]
        if not sub.empty:
            s = sub.iloc[0]
            matrix[row_idx, :] = [
                s.get(f"{p}_mean_pct", np.nan) for p in params
            ]

    ax = fig.add_subplot(gs_d[0])
    cmap = LinearSegmentedColormap.from_list(
        "case_param_mean", ["#F6F7F2", "#AFCDB8", "#2F8F6B"], N=256
    )
    im = ax.imshow(matrix, aspect="equal", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_title(
        "Full parameter profile as within-parameter percentile ranks",
        fontsize=9.2,
        color="#111111",
        pad=5,
    )

    ax.set_xticks(np.arange(len(params)))
    ax.set_xticklabels(
        [PANEL_D_PARAM_LABELS[p] for p in params],
        rotation=0,
        ha="center",
        fontsize=9.0,
        color="#111111",
    )
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10.0, color="#111111")
    ax.tick_params(axis="both", length=0, pad=2, colors="#111111")

    ax.set_xticks(np.arange(-0.5, len(params), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="#F2F2F2", linewidth=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)

    for xpos in (3.5, 8.5, 11.5):
        ax.axvline(xpos, color="#8A8A8A", linewidth=0.6, zorder=3)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_edgecolor("#777777")

    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.012)
    cbar.set_label("Within-parameter percentile rank", fontsize=9.0, color="#111111")
    cbar.ax.tick_params(labelsize=9.0, length=2, colors="#111111")
    cbar.outline.set_linewidth(0.45)

    return ax, cbar.ax


# ---------------------------------------------------------------------------
# Save analysis CSVs + notes
# ---------------------------------------------------------------------------
def save_outputs(case_df: pd.DataFrame, attrs: pd.DataFrame,
                 param_table: pd.DataFrame) -> None:
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)
    case_df.to_csv(ANALYSIS_OUT / "case_basin_selection.csv", index=False)

    n = len(attrs)
    prof_rows = []
    for _, row in case_df.iterrows():
        bid = int(row["basin_id"])
        rec = {"case_id": row["case_id"], "basin_id": bid,
               "case_type": row["case_type"]}
        for col in ATTR_COLS:
            if col in attrs.columns:
                ranks = rankdata(attrs[col].fillna(attrs[col].median()),
                                 method="average") / n
                pmap = dict(zip(attrs["basin_id"].astype(int), ranks))
                rec[f"{col}_pct"] = round(pmap.get(bid, np.nan), 4)
                sub = attrs[attrs["basin_id"] == bid]
                rec[col] = round(float(sub.iloc[0][col]) if not sub.empty else np.nan, 4)
        prof_rows.append(rec)
    pd.DataFrame(prof_rows).to_csv(
        ANALYSIS_OUT / "case_basin_attribute_profiles.csv", index=False)

    reg_rows = []
    for _, row in case_df.iterrows():
        bid = int(row["basin_id"])
        sub = param_table[param_table["basin_id"] == bid]
        if sub.empty:
            continue
        s = sub.iloc[0]
        rec = {"case_id": row["case_id"], "basin_id": bid,
               "case_type": row["case_type"]}
        for p in PARAMETER_ORDER:
            rec[f"{p}_mean_norm"] = round(float(s.get(f"{p}_mean_norm", np.nan)), 4)
            rec[f"{p}_std_norm"]  = round(float(s.get(f"{p}_std_norm",  np.nan)), 4)
            rec[f"{p}_dev"]       = round(float(s.get(f"{p}_dev",       np.nan)), 4)
        reg_rows.append(rec)
    pd.DataFrame(reg_rows).to_csv(
        ANALYSIS_OUT / "case_basin_parameter_regimes.csv", index=False)

    notes = [
        "# Case Study Plot Notes - Fig11",
        "",
        "## 1. Case basin selection",
        "Basins selected data-driven from 531 CAMELS-US basins.",
        "Primary filter: NSE >= median AND KGE >= median of distributional model.",
        "Fallback: NSE >= 20th percentile if no candidate passes primary filter.",
        "- S1 Snow:         top 10% frac_snow",
        "- A1 Arid:         top 20% aridity, best KGE",
        "- H1 Humid:        bottom 25% aridity, lowest slope (humid lowland)",
        "- T1 Steep:        top 10% slope_mean",
        "- G1 Soil: top 10% soil_conductivity",
        "- R1 Routing:      top 10% area_gages2",
        "",
        "## 2. Attribute percentile ranks (panel b)",
        "Six small subplots show representative basins on full-basin ordered environmental distributions.",
        "Each subplot sorts all available basins by one core attribute and uses normalized rank position as the x-axis.",
        "The grey curve is the full-basin ordered distribution; colored points mark S1, A1, H1, T1, G1, and R1.",
        "Each highlighted point is labeled only with the short basin ID.",
        "Attributes: frac_snow, aridity, slope_mean, soil_conductivity, pet_mean, area_gages2.",
        "",
        "## 3. Parameter normalization",
        "Parameter means normalized to [0,1] using physical search ranges.",
        "Parameter stds also normalized by the same range width.",
        "Search ranges follow fig05 PARAMETER_BOUNDS.",
        "",
        "## 4. Panel (c) merged c/d encoding",
        "Color = deviation of normalized parameter mean from all-basin median.",
        "  Blue/green = below all-basin median; Red = above all-basin median.",
        "  Diverging colormap centered at 0.",
        "Circle size = within-parameter percentile rank of the parameter mean.",
        "  This is the information previously shown in panel (d).",
        "A size legend is placed above panel (c), on the same line as the circle-size note; the note, title, and circles are left-aligned and vertically centered.",
        "",
        "## 6. NSE/KGE source",
        f"Median across all seeds and losses for distributional model.",
        f"Source: {METRICS_FILE}",
        "",
        "## 7. Interpretation",
        "This figure is illustrative case analysis, not a new stability proof.",
        "Cases show how large-sample attribute–parameter gradients translate",
        "into basin-level parameter regimes.",
        "",
        "## 8. Results writing suggestions",
        "- Representative basins show distinct δ_dist parameter regimes aligned",
        "  with their hydroclimatic and physiographic attributes.",
        "- Snow-dominated, arid, humid, steep, soil, and routing-sensitive",
        "  basins exhibited different combinations of parameter means and uncertainties.",
        "- These cases illustrate how large-sample attribute–parameter gradients",
        "  translate into basin-level parameter regimes.",
        "- The cases are illustrative and should not be interpreted as proof of",
        "  basin-specific true physical parameters.",
        "",
        f"## Outputs",
        f"- PNG: {OUT_PNG}",
        f"- PDF: {OUT_PDF}",
    ]
    (ANALYSIS_OUT / "case_study_plot_notes.md").write_text(
        "\n".join(notes) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main figure assembly
# ---------------------------------------------------------------------------
def make_figure(case_df: pd.DataFrame, attrs: pd.DataFrame,
                param_table: pd.DataFrame,
                all_locs: pd.DataFrame) -> None:

    # New layout requested:
    #   left column  = panels (a) and (c)
    #   right column = panel (b), arranged as 3 rows × 2 columns
    #   horizontal width ratio left:right = 6:4
    #   figure width is increased to give the right-column 3 × 2 panel enough room.
    fig_w_mm = 292
    fig_h_mm = 204
    fig = plt.figure(figsize=(fig_w_mm * MM, fig_h_mm * MM))

    outer = GridSpec(
        1, 2,
        figure=fig,
        left=0.058, right=0.970,
        top=0.952, bottom=0.085,
        wspace=0.090,
        width_ratios=[6, 4],
    )
    left_col = GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=outer[0],
        hspace=0.34,
        height_ratios=[1.05, 1.00],
    )

    ax_map = fig.add_subplot(left_col[0])
    ax_heat = fig.add_subplot(left_col[1])
    ax_attr = fig.add_subplot(outer[1])

    draw_map(ax_map, case_df, all_locs)
    cbar_c = draw_param_heatmap(ax_heat, case_df, param_table)
    attr_axes = draw_attr_heatmap(ax_attr, case_df, attrs)

    fig.canvas.draw()

    # Move panel-c colorbar below panel (c), horizontally.
    # Its top edge is aligned with the top anchor used by panel (b)'s legend.
    c_pos = ax_heat.get_position()
    b_bottom = min((a.get_position().y0 for a in attr_axes), default=0.18)
    b_legend_top = b_bottom - 0.038
    cbar_h = 0.018
    cbar_c.set_position([
        c_pos.x0 + 0.080 * c_pos.width,
        b_legend_top - cbar_h,
        0.840 * c_pos.width,
        cbar_h,
    ])

    fig.canvas.draw()

    a_pos = ax_map.get_position()
    c_pos = ax_heat.get_position()
    b_left = min((a.get_position().x0 for a in attr_axes), default=0.73)
    b_top = max((a.get_position().y1 for a in attr_axes), default=0.95)

    label_specs = [
        ("(a)", a_pos.x0 + 0.004, a_pos.y1 - 0.002),
        ("(c)", c_pos.x0 + 0.004, c_pos.y1 - 0.002),
        ("(b)", b_left + 0.004, b_top - 0.002),
    ]
    for label, x, y in label_specs:
        fig.text(
            x,
            y,
            label,
            ha="left",
            va="top",
            fontsize=12.4,
            color="#111111",
            zorder=20,
        )

    MAIN_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    setup_style()
    mean_df, unc_df, attrs, metrics = load_data()
    perf        = build_perf(metrics)
    param_table = build_param_table(mean_df, unc_df)
    case_df     = select_cases(attrs, perf)

    all_locs = (
        mean_df[["basin_id","longitude","latitude"]]
        .drop_duplicates("basin_id")
        .astype({"basin_id": int})
    )

    print("Selected case basins:")
    print(case_df[["case_id","basin_id","case_type","NSE","KGE",
                   "selected_reason"]].to_string(index=False))

    save_outputs(case_df, attrs, param_table)
    make_figure(case_df, attrs, param_table, all_locs)

    print(f"\nWrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    print(f"Analysis outputs → {ANALYSIS_OUT}")


if __name__ == "__main__":
    main()
