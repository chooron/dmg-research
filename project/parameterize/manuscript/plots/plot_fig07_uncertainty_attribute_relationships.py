"""
Figure 8 — Uncertainty structure and diagnostics.

Layout:
  Top row:    (a) uncertainty–attribute circle heatmap  [full width + colorbar]
  Bottom row: (b) parameter-level uncertainty structure strength [dot+interval, vertical]
              (c) mean–std coupling vs boundary sensitivity scatter

figsize: 250*MM × 210*MM
outer GridSpec 2 rows, height_ratios=[1.7, 1.0], hspace=0.18
top:    GridSpecFromSubplotSpec 1×2, width_ratios=[8.2, 0.10], wspace=0.03
bottom: GridSpecFromSubplotSpec 1×3, width_ratios=[7.0, 2.6, 0.10], wspace=0.06
subplots_adjust: left=0.07, right=0.985, bottom=0.22, top=0.96

Run:
    python plot_fig08_uncertainty_attribute_relationships.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import spearmanr

from common import ATTR_LABELS, PARAM_LABELS, p_label

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/workspace/autoresearch")
MANUSCRIPT_ROOT = ROOT / "project" / "parameterize" / "manuscript"
UNCERTAINTY_SPATIAL_ROOT = (
    MANUSCRIPT_ROOT / "analysis" / "06_uncertainty_spatial_data"
)
UNCERTAINTY_REL_ROOT = (
    MANUSCRIPT_ROOT / "analysis" / "07_uncertainty_attribute_relationships"
)
FIGURE8_ROOT = MANUSCRIPT_ROOT / "analysis" / "figure8"
DATA_DIR = FIGURE8_ROOT / "data"
REPORT_DIR = FIGURE8_ROOT / "reports"
MAIN_FIG_DIR = MANUSCRIPT_ROOT / "figures" / "main"

UNCERTAINTY_MAP_FILE = (
    UNCERTAINTY_SPATIAL_ROOT
    / "data"
    / "distributional_parameter_uncertainty_maps_long.csv"
)
STD_CORR_FILE = (
    UNCERTAINTY_REL_ROOT
    / "data"
    / "distributional_std_attribute_correlations.csv"
)
STD_DOMINANT_FILE = (
    UNCERTAINTY_REL_ROOT / "data" / "std_dominant_relationships.csv"
)
GRADIENT_FLAGS_FILE = (
    UNCERTAINTY_REL_ROOT / "data" / "std_headline_gradient_flags.csv"
)
COUPLING_FILE = (
    UNCERTAINTY_SPATIAL_ROOT / "data" / "mean_std_coupling_diagnostics.csv"
)
BOUNDARY_FILE = (
    UNCERTAINTY_SPATIAL_ROOT / "data" / "boundary_uncertainty_diagnostics.csv"
)
BASIN_ATTRIBUTE_FILE = (
    ROOT
    / "project"
    / "parameterize"
    / "outputs"
    / "analysis"
    / "stability_stats"
    / "tables"
    / "basin_attributes.csv"
)

OUT_PNG = MAIN_FIG_DIR / "Fig07_uncertainty_attribute_relationships.png"
PANEL_A_FILE = DATA_DIR / "fig08_panel_a_uncertainty_heatmap_data.csv"
PANEL_B_FILE = DATA_DIR / "fig08_panel_b_uncertainty_structure_strength.csv"
PANEL_C_FILE = DATA_DIR / "fig08_panel_c_uncertainty_diagnostics.csv"
NOTES_FILE = REPORT_DIR / "fig08_plot_notes.md"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DPI = 600
MM = 1 / 25.4
STRONG_ABS_RHO = 0.5
Q_THRESHOLD = 0.05
MEAN_STD_COUPLING_THRESHOLD = 0.5
BOUNDARY_SENSITIVITY_THRESHOLD = 0.4

PARAMETER_ORDER = [
    "parTT",
    "parCFMAX",
    "parCFR",
    "parCWH",
    "parBETA",
    "parFC",
    "parLP",
    "parPERC",
    "parUZL",
    "parK0",
    "parK1",
    "parK2",
    "route_a",
    "route_b",
]
PARAMETER_LABELS = PARAM_LABELS
PARAMETER_GROUPS = {
    "parTT": "snow",
    "parCFMAX": "snow",
    "parCFR": "snow",
    "parCWH": "snow",
    "parBETA": "soil",
    "parFC": "soil",
    "parLP": "soil",
    "parPERC": "production",
    "parUZL": "production",
    "parK0": "production",
    "parK1": "production",
    "parK2": "production",
    "route_a": "routing",
    "route_b": "routing",
}
PARAMETER_GROUP_ORDER = ["snow", "soil", "production", "routing"]
PARAMETER_GROUP_COLORS = {
    "snow": "#3F8E72",
    "soil": "#8FA35A",
    "production": "#D4A15F",
    "routing": "#A67899",
}

ATTRIBUTE_GROUP_ORDER = [
    "Climate/hydroclimate",
    "Soil",
    "Topography/scale",
    "Geology",
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
    "Climate/hydroclimate": [
        "p_mean",
        "pet_mean",
        "p_seasonality",
        "frac_snow",
        "aridity",
        "high_prec_freq",
        "high_prec_dur",
        "low_prec_freq",
        "low_prec_dur",
    ],
    "Topography/scale": [
        "elev_mean",
        "slope_mean",
        "area_gages2",
        "relief",
        "topographic_wetness",
        "twi",
    ],
    "Vegetation/land cover": [
        "frac_forest",
        "lai_max",
        "lai_diff",
        "gvf_max",
        "gvf_diff",
        "dom_land_cover_frac",
        "dom_land_cover",
        "land_cover",
        "vegetation",
    ],
    "Soil": [
        "root_depth_50",
        "soil_depth_pelletier",
        "soil_depth_statsgo",
        "soil_porosity",
        "soil_conductivity",
        "max_water_content",
        "sand_frac",
        "silt_frac",
        "clay_frac",
    ],
    "Geology": [
        "geol_1st_class",
        "glim_1st_class_frac",
        "geol_2nd_class",
        "glim_2nd_class_frac",
        "carbonate_rocks_frac",
        "geol_porosity",
        "geol_permeability",
        "aquifer_frac",
    ],
}

# Parameters to annotate in panel (c)
PANEL_C_ANNOTATE = {
    "parCWH",
    "parPERC",
    "parUZL",
    "route_b",
    "parCFMAX",
    "parTT",
}


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def setup_style() -> None:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams.update(
        {
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
            "legend.fontsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": DPI,
        }
    )


def ensure_dirs() -> None:
    for path in (DATA_DIR, REPORT_DIR, MAIN_FIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def clean_label(name: str) -> str:
    return p_label(name)


def classify_attribute(attribute: str) -> str:
    for group in ATTRIBUTE_GROUP_ORDER:
        if attribute in ATTRIBUTE_GROUP_RULES[group]:
            return group
    lower = attribute.lower()
    if any(
        t in lower for t in ("prec", "snow", "pet", "arid", "season", "p_mean")
    ):
        return "Climate/hydroclimate"
    if any(
        t in lower for t in ("slope", "elev", "relief", "topo", "twi", "area")
    ):
        return "Topography/scale"
    if any(t in lower for t in ("lai", "gvf", "forest", "land", "veg")):
        return "Vegetation/land cover"
    if any(
        t in lower
        for t in ("soil", "sand", "silt", "clay", "root", "water_content")
    ):
        return "Soil"
    if any(
        t in lower
        for t in ("geol", "glim", "aquifer", "carbonate", "perm", "porosity")
    ):
        return "Geology"
    return "Climate/hydroclimate"


def fdr_bh(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return pd.Series(q, index=p_values.index)
    valid_indices = np.where(valid)[0]
    order = valid_indices[np.argsort(p[valid])]
    ranked_p = p[order]
    m = float(len(ranked_p))
    adjusted = ranked_p * m / np.arange(1, len(ranked_p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0, 1)
    return pd.Series(q, index=p_values.index)


def correlation_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "fig07_purple_white_green",
        ["#6A3D9A", "#F7F7F7", "#1B9E77"],
        N=256,
    )


def attribute_group_ranges(attributes: list[str]) -> list[tuple[str, int, int]]:
    ranges: list[tuple[str, int, int]] = []
    if not attributes:
        return ranges
    start = 0
    current = classify_attribute(attributes[0])
    for idx, attribute in enumerate(attributes[1:], start=1):
        group = classify_attribute(attribute)
        if group != current:
            ranges.append((current, start, idx - 1))
            start = idx
            current = group
    ranges.append((current, start, len(attributes) - 1))
    return ranges


def build_attribute_ordering(corr: pd.DataFrame) -> pd.DataFrame:
    parameter_group_rank = {g: i for i, g in enumerate(PARAMETER_GROUP_ORDER)}
    attribute_group_rank = {g: i for i, g in enumerate(ATTRIBUTE_GROUP_ORDER)}
    rows: list[dict] = []
    for attribute, sub in corr.groupby("attribute"):
        strongest = sub.sort_values(
            ["abs_rho", "parameter"], ascending=[False, True]
        ).iloc[0]
        rows.append(
            {
                "attribute": attribute,
                "attribute_group": classify_attribute(attribute),
                "max_abs_rho": float(strongest["abs_rho"]),
                "dominant_parameter": strongest["parameter"],
                "dominant_parameter_group": PARAMETER_GROUPS.get(
                    strongest["parameter"], "production"
                ),
            }
        )
    ordering = pd.DataFrame(rows)
    ordering["attribute_group_order"] = ordering["attribute_group"].map(
        attribute_group_rank
    )
    ordering["dominant_parameter_group_order"] = ordering[
        "dominant_parameter_group"
    ].map(parameter_group_rank)
    fixed_order = {
        attribute: idx
        for idx, attribute in enumerate(attribute_order(ordering["attribute"].tolist()))
    }
    ordering["fixed_attribute_order"] = ordering["attribute"].map(fixed_order)
    ordering = ordering.sort_values("fixed_attribute_order").reset_index(drop=True)
    ordering["attribute_order"] = np.arange(len(ordering))
    return ordering


def attribute_order(attributes: list[str]) -> list[str]:
    grouped: list[str] = []
    for group in ATTRIBUTE_GROUP_ORDER:
        preferred = [attr for attr in ATTRIBUTE_GROUP_RULES[group] if attr in attributes]
        leftovers = sorted(
            [attr for attr in attributes if attr not in preferred and classify_attribute(attr) == group]
        )
        grouped.extend(preferred)
        grouped.extend([attr for attr in leftovers if attr not in grouped])
    return grouped


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _expand_corr_from_maps() -> pd.DataFrame:
    """Compute Spearman correlations from raw uncertainty maps when corr file has < 35 attrs."""
    maps = pd.read_csv(UNCERTAINTY_MAP_FILE)
    attrs = pd.read_csv(BASIN_ATTRIBUTE_FILE)
    attribute_cols = [c for c in attrs.columns if c != "basin_id"]
    merged = maps[["basin_id", "parameter", "parameter_std_unit"]].merge(
        attrs[["basin_id", *attribute_cols]], on="basin_id", how="inner"
    )
    rows: list[dict] = []
    for parameter, sub in merged.groupby("parameter"):
        for attribute in attribute_cols:
            xy = pd.concat(
                [sub["parameter_std_unit"], sub[attribute]], axis=1
            ).dropna()
            n = len(xy)
            if n < 3 or xy.iloc[:, 0].std() == 0 or xy.iloc[:, 1].std() == 0:
                rho, pval = np.nan, np.nan
            else:
                res = spearmanr(
                    xy.iloc[:, 0].to_numpy(), xy.iloc[:, 1].to_numpy()
                )
                rho, pval = float(res.statistic), float(res.pvalue)
            rows.append(
                {
                    "parameter": parameter,
                    "attribute": attribute,
                    "spearman_rho": rho,
                    "p_value": pval,
                    "abs_rho": abs(rho) if np.isfinite(rho) else np.nan,
                    "n_basins": n,
                    "sign": "positive"
                    if np.isfinite(rho) and rho > 0
                    else "negative",
                }
            )
    df = pd.DataFrame(rows)
    df["q_value"] = fdr_bh(df["p_value"])
    # merge interpretation flags if available
    if STD_CORR_FILE.exists():
        flags = pd.read_csv(STD_CORR_FILE)
        flag_cols = [
            c
            for c in [
                "parameter",
                "attribute",
                "interpretation_flag",
                "mean_std_spearman",
                "boundary_distance_std_spearman",
                "near_boundary_share",
            ]
            if c in flags.columns
        ]
        df = df.merge(
            flags[flag_cols].drop_duplicates(),
            on=["parameter", "attribute"],
            how="left",
        )
    return df


def load_corr_data() -> pd.DataFrame:
    corr = pd.read_csv(STD_CORR_FILE)
    if (
        corr["attribute"].nunique() < 35
        and UNCERTAINTY_MAP_FILE.exists()
        and BASIN_ATTRIBUTE_FILE.exists()
    ):
        corr = _expand_corr_from_maps()
    else:
        corr = corr.copy()
    corr["attribute_group"] = corr["attribute"].map(classify_attribute)
    corr["parameter_group"] = corr["parameter"].map(PARAMETER_GROUPS)
    corr["parameter_label"] = corr["parameter"].map(clean_label)
    if "q_value" not in corr.columns:
        corr["q_value"] = fdr_bh(corr["p_value"])
    if "abs_rho" not in corr.columns:
        corr["abs_rho"] = corr["spearman_rho"].abs()
    corr["strong_flag"] = corr["abs_rho"].ge(STRONG_ABS_RHO) & corr[
        "q_value"
    ].lt(Q_THRESHOLD)
    corr["rank_abs_rho"] = corr.groupby("parameter")["abs_rho"].rank(
        method="first", ascending=False
    )
    corr["dominant_flag"] = corr["rank_abs_rho"].eq(1)
    if "interpretation_flag" in corr.columns:
        corr["caution_flag"] = (
            corr["interpretation_flag"]
            .str.lower()
            .str.contains("caution|coupled|boundary", na=False)
        )
    else:
        corr["caution_flag"] = False
    return corr


def load_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    coupling = pd.read_csv(COUPLING_FILE)
    boundary = pd.read_csv(BOUNDARY_FILE)
    return coupling, boundary


# ---------------------------------------------------------------------------
# Panel data preparation
# ---------------------------------------------------------------------------
def prepare_panel_a(corr: pd.DataFrame) -> pd.DataFrame:
    ordering = build_attribute_ordering(corr)
    panel_a = corr.merge(
        ordering[["attribute", "attribute_order"]], on="attribute", how="left"
    )
    parameters = [p for p in PARAMETER_ORDER if p in corr["parameter"].unique()]
    panel_a["parameter_order"] = panel_a["parameter"].map(
        {p: i for i, p in enumerate(parameters)}
    )
    panel_a = panel_a.sort_values(
        ["attribute_order", "parameter_order"]
    ).reset_index(drop=True)
    panel_a.to_csv(PANEL_A_FILE, index=False)
    return panel_a


def prepare_panel_b(corr: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    """For each parameter: top1, top3_mean, top5_mean |ρ|."""
    parameters = [p for p in PARAMETER_ORDER if p in corr["parameter"].unique()]
    diag_lookup = diagnostics.set_index("parameter")
    rows: list[dict] = []
    for i, par in enumerate(parameters):
        sub = corr[corr["parameter"] == par].sort_values(
            "abs_rho", ascending=False
        )
        abs_rhos = sub["abs_rho"].dropna().values
        top1 = float(abs_rhos[0]) if len(abs_rhos) >= 1 else np.nan
        top3 = (
            float(np.mean(abs_rhos[:3]))
            if len(abs_rhos) >= 3
            else float(np.mean(abs_rhos))
        )
        top5 = (
            float(np.mean(abs_rhos[:5]))
            if len(abs_rhos) >= 5
            else float(np.mean(abs_rhos))
        )
        dominant_attr = sub.iloc[0]["attribute"] if len(sub) > 0 else ""
        dominant_rho = (
            float(sub.iloc[0]["spearman_rho"]) if len(sub) > 0 else np.nan
        )
        diag_row = diag_lookup.loc[par] if par in diag_lookup.index else None
        mean_std_coupling = (
            float(diag_row["mean_std_coupling"])
            if diag_row is not None and pd.notna(diag_row["mean_std_coupling"])
            else np.nan
        )
        boundary_sensitivity = (
            float(diag_row["boundary_sensitivity"])
            if diag_row is not None and pd.notna(diag_row["boundary_sensitivity"])
            else np.nan
        )
        caution_flag = (
            np.isfinite(mean_std_coupling)
            and mean_std_coupling >= MEAN_STD_COUPLING_THRESHOLD
        ) or (
            np.isfinite(boundary_sensitivity)
            and boundary_sensitivity >= BOUNDARY_SENSITIVITY_THRESHOLD
        )
        rows.append(
            {
                "parameter": par,
                "parameter_label": clean_label(par),
                "parameter_group": PARAMETER_GROUPS.get(par, "production"),
                "parameter_order": i,
                "top1_abs_rho": top1,
                "top3_mean_abs_rho": top3,
                "top5_mean_abs_rho": top5,
                "dominant_attribute": dominant_attr,
                "dominant_rho": dominant_rho,
                "mean_std_coupling": mean_std_coupling,
                "boundary_sensitivity": boundary_sensitivity,
                "caution_flag": bool(caution_flag),
            }
        )
    panel_b = pd.DataFrame(rows)
    panel_b.to_csv(PANEL_B_FILE, index=False)
    return panel_b


def prepare_panel_c(
    corr: pd.DataFrame, coupling: pd.DataFrame, boundary: pd.DataFrame
) -> pd.DataFrame:
    parameters = [p for p in PARAMETER_ORDER if p in corr["parameter"].unique()]
    diag = pd.DataFrame({"parameter": parameters})
    diag["parameter_group"] = diag["parameter"].map(PARAMETER_GROUPS)
    diag = diag.merge(
        coupling[["parameter", "mean_std_spearman"]].rename(
            columns={"mean_std_spearman": "mean_std_coupling"}
        ),
        on="parameter",
        how="left",
    )
    diag["mean_std_coupling"] = diag["mean_std_coupling"].abs()
    diag = diag.merge(
        boundary[
            [
                "parameter",
                "boundary_distance_std_spearman",
                "near_boundary_share",
            ]
        ].rename(
            columns={"boundary_distance_std_spearman": "boundary_sensitivity"}
        ),
        on="parameter",
        how="left",
    )
    diag["boundary_sensitivity"] = diag["boundary_sensitivity"].abs()
    diag["interpretation_flag"] = np.where(
        (diag["mean_std_coupling"] >= MEAN_STD_COUPLING_THRESHOLD)
        | (diag["boundary_sensitivity"] >= BOUNDARY_SENSITIVITY_THRESHOLD),
        "interpret with caution",
        "clean",
    )
    diag.to_csv(PANEL_C_FILE, index=False)
    return diag


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_parameter_group_guides(
    ax: plt.Axes, parameters: list[str], x_left: float
) -> None:
    group_ranges: list[tuple[str, int, int]] = []
    start = 0
    current = PARAMETER_GROUPS[parameters[0]]
    for idx, parameter in enumerate(parameters[1:], start=1):
        group = PARAMETER_GROUPS[parameter]
        if group != current:
            group_ranges.append((current, start, idx - 1))
            start = idx
            current = group
    group_ranges.append((current, start, len(parameters) - 1))
    for group, start_idx, end_idx in group_ranges:
        y0 = start_idx - 0.48
        y1 = end_idx + 0.48
        ax.plot(
            [x_left, x_left],
            [y0, y1],
            color="#9E9E9E",
            linewidth=0.45,
            clip_on=False,
        )
        ax.plot(
            [x_left, x_left + 0.18],
            [y0, y0],
            color="#9E9E9E",
            linewidth=0.45,
            clip_on=False,
        )
        ax.plot(
            [x_left, x_left + 0.18],
            [y1, y1],
            color="#9E9E9E",
            linewidth=0.45,
            clip_on=False,
        )


def draw_attribute_group_guides(
    ax: plt.Axes, attributes: list[str], y_top: float
) -> None:
    # y_top negative = above heatmap (y-axis inverted); bracket ticks go upward (more negative)
    for group, start_idx, end_idx in attribute_group_ranges(attributes):
        x0 = start_idx - 0.48
        x1 = end_idx + 0.48
        ax.plot(
            [x0, x1],
            [y_top, y_top],
            color="#555555",
            linewidth=0.7,
            alpha=0.85,
            clip_on=False,
        )
        ax.plot(
            [x0, x0],
            [y_top, y_top - 0.14],
            color="#555555",
            linewidth=0.7,
            alpha=0.85,
            clip_on=False,
        )
        ax.plot(
            [x1, x1],
            [y_top, y_top - 0.14],
            color="#555555",
            linewidth=0.7,
            alpha=0.85,
            clip_on=False,
        )
        ax.text(
            (x0 + x1) / 2,
            y_top - 0.22,
            ATTRIBUTE_GROUP_SHORT[group],
            ha="center",
            va="bottom",
            fontsize=9.2,
            color="#111111",
            clip_on=False,
        )


# ---------------------------------------------------------------------------
# Panel (a): circle heatmap
# ---------------------------------------------------------------------------
def draw_panel_a(ax: plt.Axes, cax: plt.Axes, panel_a: pd.DataFrame) -> None:
    parameters = [
        p for p in PARAMETER_ORDER if p in panel_a["parameter"].unique()
    ]
    attributes = (
        panel_a[["attribute", "attribute_order"]]
        .drop_duplicates("attribute")
        .sort_values("attribute_order")["attribute"]
        .tolist()
    )
    cmap = correlation_cmap()
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    ax.set_xlim(-0.5, len(attributes) - 0.5)
    ax.set_ylim(len(parameters) - 0.5, -0.5)
    ax.set_aspect("auto")
    ax.set_facecolor("#FBFBFB")
    ax.set_xticks(np.arange(len(attributes)))
    ax.set_xticklabels(
        [ATTR_LABELS.get(a, a) for a in attributes],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=9.8,
    )
    ax.set_yticks(np.arange(len(parameters)))
    ax.set_yticklabels(
        [clean_label(p) for p in parameters],
        rotation=0,
        ha="right",
        fontsize=10.2,
    )
    ax.tick_params(axis="x", length=0, pad=2, colors="#111111")
    ax.tick_params(axis="y", length=0, pad=3, colors="#111111")
    ax.set_xticks(np.arange(-0.5, len(attributes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(parameters), 1), minor=True)
    ax.grid(which="minor", color="#D9D9D9", linewidth=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_edgecolor("#777777")

    lookup = panel_a.set_index(["attribute", "parameter"])
    xs, ys, values, sizes = [], [], [], []
    for y, parameter in enumerate(parameters):
        for x, attribute in enumerate(attributes):
            if (attribute, parameter) not in lookup.index:
                continue
            row = lookup.loc[(attribute, parameter)]
            rho = float(row["spearman_rho"])
            if np.isfinite(rho):
                xs.append(x)
                ys.append(y)
                values.append(rho)
                sizes.append(5.0 + 132.0 * (abs(rho) ** 1.55))
            if bool(row.get("dominant_flag", False)):
                ax.add_patch(
                    Rectangle(
                        (x - 0.49, y - 0.49),
                        0.98,
                        0.98,
                        fill=False,
                        edgecolor="#111111",
                        linewidth=1.35,
                        zorder=4,
                    )
                )
            if bool(row.get("strong_flag", False)):
                ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=3.6,
                    color="#111111",
                    linestyle="None",
                    zorder=5,
                )
            if bool(row.get("caution_flag", False)) and bool(
                row.get("strong_flag", False)
            ):
                ax.plot(
                    x,
                    y,
                    marker="^",
                    markersize=4.2,
                    color="#888888",
                    markerfacecolor="none",
                    markeredgewidth=0.8,
                    linestyle="None",
                    zorder=6,
                )

    ax.scatter(
        xs,
        ys,
        s=sizes,
        c=values,
        cmap=cmap,
        norm=norm,
        marker="o",
        edgecolors="#F7F7F7",
        linewidths=0.18,
        alpha=0.96,
        zorder=2,
    )

    prev_group = classify_attribute(attributes[0])
    for x, attribute in enumerate(attributes[1:], start=1):
        g = classify_attribute(attribute)
        if g != prev_group:
            ax.axvline(x - 0.5, color="#8A8A8A", linewidth=0.55)
            prev_group = g
    prev_pg = PARAMETER_GROUPS[parameters[0]]
    for y, parameter in enumerate(parameters[1:], start=1):
        g = PARAMETER_GROUPS[parameter]
        if g != prev_pg:
            ax.axhline(y - 0.5, color="#8A8A8A", linewidth=0.55)
            prev_pg = g

    for tick in ax.get_yticklabels():
        tick.set_color("#111111")
    draw_parameter_group_guides(ax, parameters, x_left=-2.8)
    draw_attribute_group_guides(ax, attributes, y_top=-0.52)

    ax.text(
        0.99,
        0.01,
        "(a)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12.5,
        fontweight="normal",
        color="#111111",
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#111111",
            markeredgecolor="#111111",
            markersize=4.0,
            label=r"$|\rho|\geq0.5$, $q<0.05$",
        ),
        Patch(
            facecolor="none",
            edgecolor="#111111",
            linewidth=1.35,
            label="dominant",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="none",
            markeredgecolor="#888888",
            markeredgewidth=0.8,
            markersize=4.5,
            label="caution (coupled/boundary)",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=1,
        frameon=True,
        framealpha=0.88,
        edgecolor="#CCCCCC",
        handlelength=1.2,
        handletextpad=0.45,
        labelspacing=0.35,
        borderaxespad=0.3,
        fontsize=9.0,
    )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label(r"Spearman $\rho$", fontsize=10.2)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.tick_params(labelsize=9.4, width=0.45, length=2.0, colors="#222222")
    cbar.outline.set_linewidth(0.5)


# ---------------------------------------------------------------------------
# Panel (b): parameter-level uncertainty structure strength (vertical)
# ---------------------------------------------------------------------------
def draw_panel_b(ax: plt.Axes, panel_b: pd.DataFrame) -> None:
    ordered = panel_b.sort_values("parameter_order").reset_index(drop=True)
    x = np.arange(len(ordered))
    colors = [PARAMETER_GROUP_COLORS[g] for g in ordered["parameter_group"]]

    # grey vline from top5 to top1
    ax.vlines(
        x,
        ordered["top5_mean_abs_rho"],
        ordered["top1_abs_rho"],
        color="#B8B8B8",
        linewidth=1.0,
        zorder=1,
    )
    ax.scatter(
        x,
        ordered["top3_mean_abs_rho"],
        s=30,
        c=colors,
        edgecolor="#222222",
        linewidth=0.35,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ordered["parameter_label"],
        rotation_mode="anchor",
        fontsize=10.0,
    )
    ax.set_ylabel(r"Top-$k$ uncertainty-attribute $|\rho|$", fontsize=10.5)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(axis="both", width=0.5, length=2.4, colors="#222222")
    ax.tick_params(axis="y", labelsize=10.0)

    ax.text(
        0.02,
        0.98,
        "(b)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="normal",
        color="#111111",
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PARAMETER_GROUP_COLORS[g],
            markeredgecolor="#222222",
            markeredgewidth=0.35,
            markersize=4.5,
            label=g,
        )
        for g in PARAMETER_GROUP_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        framealpha=0.88,
        edgecolor="#CCCCCC",
        handlelength=0.9,
        handletextpad=0.35,
        borderaxespad=0.3,
        labelspacing=0.25,
        fontsize=9.0,
    )


# ---------------------------------------------------------------------------
# Panel (c): diagnostic scatter
# ---------------------------------------------------------------------------
def draw_panel_c(ax: plt.Axes, panel_c: pd.DataFrame) -> None:
    # Per-point annotation offsets (in points) to avoid overlap
    # Keys are parameter names, values are (dx, dy) text offsets
    OFFSETS = {
        "parTT": (-28, 10),
        "parCFMAX": (8, 8),
        "parCWH": (-38, -12),
        "parPERC": (6, 6),
        "parUZL": (-38, -12),
        "route_b": (6, -12),
    }
    ax.axvspan(
        0,
        MEAN_STD_COUPLING_THRESHOLD,
        ymin=0,
        ymax=BOUNDARY_SENSITIVITY_THRESHOLD,
        facecolor="#E8F1EA",
        edgecolor="none",
        alpha=0.85,
        zorder=0,
    )
    ax.text(
        0.03,
        0.06,
        "clean",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        color="#3A6A46",
        zorder=1,
    )
    for _, row in panel_c.iterrows():
        x = (
            float(row["mean_std_coupling"])
            if pd.notna(row["mean_std_coupling"])
            else np.nan
        )
        y = (
            float(row["boundary_sensitivity"])
            if pd.notna(row["boundary_sensitivity"])
            else np.nan
        )
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        color = PARAMETER_GROUP_COLORS.get(row["parameter_group"], "#888888")
        ax.scatter(
            x,
            y,
            s=38,
            color=color,
            edgecolor="#222222",
            linewidth=0.4,
            zorder=3,
        )
        if row["parameter"] in PANEL_C_ANNOTATE:
            dx, dy = OFFSETS.get(row["parameter"], (5, 5))
            ax.annotate(
                clean_label(row["parameter"]),
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=9.0,
                color="#111111",
                arrowprops=dict(
                    arrowstyle="-",
                    color="#888888",
                    lw=0.6,
                    shrinkA=0,
                    shrinkB=2,
                ),
            )

    ax.axvline(
        MEAN_STD_COUPLING_THRESHOLD,
        color="#888888",
        linewidth=0.7,
        linestyle="--",
        zorder=1,
    )
    ax.axhline(
        BOUNDARY_SENSITIVITY_THRESHOLD,
        color="#888888",
        linewidth=0.7,
        linestyle="--",
        zorder=1,
    )
    ax.text(
        MEAN_STD_COUPLING_THRESHOLD + 0.015,
        0.97,
        r"$|\rho|=0.5$",
        ha="left",
        va="top",
        fontsize=9.0,
        color="#555555",
    )
    ax.text(
        0.98,
        BOUNDARY_SENSITIVITY_THRESHOLD + 0.015,
        r"$|\rho|=0.4$",
        ha="right",
        va="bottom",
        fontsize=9.0,
        color="#555555",
    )
    ax.set_xlabel(r"Mean–std coupling $|\rho|$", fontsize=10.5)
    ax.set_ylabel(r"Boundary sensitivity $|\rho|$", fontsize=10.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(
        axis="both", labelsize=9.8, width=0.5, length=2.4, colors="#222222"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.grid(color="#EEEEEE", linewidth=0.4)
    ax.set_axisbelow(True)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PARAMETER_GROUP_COLORS[g],
            markeredgecolor="#222222",
            markeredgewidth=0.4,
            markersize=4.5,
            label=g,
        )
        for g in PARAMETER_GROUP_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        framealpha=0.88,
        edgecolor="#CCCCCC",
        handlelength=0.9,
        handletextpad=0.35,
        labelspacing=0.25,
        fontsize=9.0,
    )
    ax.text(
        0.02,
        0.98,
        "(c)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="normal",
        color="#111111",
    )


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def make_figure(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(280 * MM, 240 * MM), constrained_layout=False)

    outer = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.55, 1.0],
        hspace=0.38,
    )

    top = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer[0],
        width_ratios=[1.0, 0.018],
        wspace=0.012,
    )

    bottom = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer[1],
        width_ratios=[5.2, 2.2],
        wspace=0.14,
    )

    ax_a = fig.add_subplot(top[0])
    cax_a = fig.add_subplot(top[1])
    ax_b = fig.add_subplot(bottom[0])
    ax_c = fig.add_subplot(bottom[1])

    draw_panel_a(ax_a, cax_a, panel_a)
    draw_panel_b(ax_b, panel_b)
    draw_panel_c(ax_c, panel_c)

    fig.subplots_adjust(left=0.07, right=0.955, bottom=0.30, top=0.96)

    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------
def write_notes(
    corr: pd.DataFrame, panel_b: pd.DataFrame, panel_c: pd.DataFrame
) -> None:
    parameters = [p for p in PARAMETER_ORDER if p in corr["parameter"].unique()]
    strong = corr[corr["strong_flag"]]
    caution_strong = corr[corr["strong_flag"] & corr["caution_flag"]]
    lines = [
        "# Fig07 uncertainty-attribute relationship plot notes",
        "",
        "## 1. Statistical object",
        "",
        "- Figure 8 analyses δ_dist parameter uncertainty (std across seeds) vs basin attributes.",
        "- Only the distributional model family is used; δ_mcd is excluded from the main figure.",
        "- Spearman ρ is computed between parameter_std_unit and each basin attribute across 531 CAMELS-US basins.",
        "",
        "## 2. Panel (a): uncertainty–attribute circle heatmap",
        "",
        f"- {len(parameters)} parameters × {corr['attribute'].nunique()} attributes.",
        "- Circle size = |ρ|, color = ρ (purple-white-green diverging colormap).",
        "- Black box = dominant (rank-1 |ρ| for that parameter).",
        "- Black dot = strong (|ρ| ≥ 0.5, q < 0.05).",
        "- Grey triangle = caution (coupled or boundary-sensitive).",
        f"- Strong relationships: {len(strong)} pairs.",
        f"- Strong + caution: {len(caution_strong)} pairs.",
        "",
        "## 3. Panel (b): parameter-level uncertainty structure strength",
        "",
        "- Vertical dot+interval plot: parameters on x-axis, top-k |ρ| on y-axis.",
        "- Grey vline spans top5_mean_abs_rho to top1_abs_rho.",
        "- Colored dot at top3_mean_abs_rho, colored by parameter group.",
        "- Parameters with higher top-k |ρ| have more structured uncertainty.",
        "",
        "## 4. Panel (c): mean–std coupling vs boundary sensitivity scatter",
        "",
        "- x: |mean–std coupling ρ| (from mean_std_coupling_diagnostics.csv).",
        "- y: |boundary distance–std ρ| (from boundary_uncertainty_diagnostics.csv).",
        f"- Parameters above x={MEAN_STD_COUPLING_THRESHOLD:.1f} or y={BOUNDARY_SENSITIVITY_THRESHOLD:.1f} dashed lines are flagged 'interpret with caution'.",
        "- The lower-left background marks the clean region below both thresholds.",
        f"- Caution parameters: {panel_c[panel_c['interpretation_flag'] == 'interpret with caution']['parameter'].map(clean_label).tolist()}",
        "",
        "## 5. Why δ_mcd is excluded",
        "",
        "- δ_mcd conflates model-choice uncertainty with parameter uncertainty.",
        "- Figure 8 focuses on within-model seed uncertainty (δ_dist) for a clean structural interpretation.",
        "",
        "## 6. Uncertainty gradients are structured diagnostic signal",
        "",
        "- The std gradients shown in Fig 7 panels (g)–(i) and the heatmap here represent",
        "  structured diagnostic signal: parameter uncertainty varies systematically with basin attributes.",
        "- This is NOT evidence of true parameter identifiability; it reflects that the",
        "  distributional model's seed-to-seed spread is geographically structured.",
        "- Interpretation should be cautious for parameters with strong mean–std coupling",
        "  or boundary effects (panel c).",
        "",
        "## 7. Output files",
        "",
        f"- PNG: {OUT_PNG}",
        f"- Panel A data: {PANEL_A_FILE}",
        f"- Panel B data: {PANEL_B_FILE}",
        f"- Panel C data: {PANEL_C_FILE}",
    ]
    NOTES_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    setup_style()
    ensure_dirs()
    corr = load_corr_data()
    coupling, boundary = load_diagnostics()
    panel_a = prepare_panel_a(corr)
    panel_c = prepare_panel_c(corr, coupling, boundary)
    panel_b = prepare_panel_b(corr, panel_c)
    make_figure(panel_a, panel_b, panel_c)
    write_notes(corr, panel_b, panel_c)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {PANEL_A_FILE}")
    print(f"Wrote {PANEL_B_FILE}")
    print(f"Wrote {PANEL_C_FILE}")
    print(f"Wrote {NOTES_FILE}")


if __name__ == "__main__":
    main()
