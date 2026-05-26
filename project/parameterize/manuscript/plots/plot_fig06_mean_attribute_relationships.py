from __future__ import annotations

import logging
import string
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import kendalltau, pearsonr, spearmanr

from common import ATTR_LABELS, PARAM_LABELS, p_label

ROOT = Path("/workspace/autoresearch")
MANUSCRIPT_ROOT = ROOT / "project" / "parameterize" / "manuscript"
RELATIONSHIP_ROOT = MANUSCRIPT_ROOT / "analysis" / "04_mean_attribute_relationships"
SENSITIVITY_ROOT = MANUSCRIPT_ROOT / "analysis" / "02_seed_loss_sensitivity"
FIGURE6_ROOT = MANUSCRIPT_ROOT / "analysis" / "figure6"
DATA_DIR = FIGURE6_ROOT / "data"
REPORT_DIR = FIGURE6_ROOT / "reports"
PLOT_DIR = MANUSCRIPT_ROOT / "plots"
MAIN_FIG_DIR = MANUSCRIPT_ROOT / "figures" / "main"

CORRELATION_FILE = RELATIONSHIP_ROOT / "data" / "distributional_mean_attribute_correlations.csv"
DOMINANT_FILE = RELATIONSHIP_ROOT / "data" / "distributional_dominant_mean_relationships.csv"
KEY_RELATIONSHIP_FILE = RELATIONSHIP_ROOT / "data" / "selected_key_relationships.csv"
MEAN_MAP_FILE = MANUSCRIPT_ROOT / "analysis" / "03_distributional_parameter_spatial_data" / "data" / "distributional_parameter_mean_maps_long.csv"
BASIN_ATTRIBUTE_FILE = ROOT / "project" / "parameterize" / "outputs" / "analysis" / "stability_stats" / "tables" / "basin_attributes.csv"
FOCUSED_STABILITY_FILE = SENSITIVITY_ROOT / "data" / "focused_pair_seed_loss_stability.csv"
CROSS_SEED_FILE = SENSITIVITY_ROOT / "data" / "cross_seed_relationship_sensitivity.csv"
CROSS_LOSS_FILE = SENSITIVITY_ROOT / "data" / "cross_loss_relationship_sensitivity.csv"

OUT_PNG = MAIN_FIG_DIR / "Fig06_mean_attribute_relationships.png"
PANEL_A_FILE = DATA_DIR / "fig06_panel_a_heatmap_data.csv"
ATTRIBUTE_ORDER_FILE = DATA_DIR / "fig06_attribute_ordering.csv"
PANEL_B_FILE = DATA_DIR / "fig06_panel_b_topk_strength.csv"
PANEL_C_FILE = DATA_DIR / "fig06_panel_c_group_summary.csv"
STABILITY_FILE = DATA_DIR / "fig06_stability_marker_table.csv"
NOTES_FILE = REPORT_DIR / "fig06_plot_notes.md"

DPI = 600
MM = 1 / 25.4
STRONG_ABS_RHO = 0.5
Q_THRESHOLD = 0.05

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
    "snow": "#56B4E9",
    "soil": "#CC79A7",
    "production": "#009E73",
    "routing": "#0072B2",
}

ATTRIBUTE_GROUP_ORDER = [
    "Climate/hydroclimate",
    "Soil",
    "Topography/scale",
    "Geology",
    "Vegetation/land cover",
]
ATTRIBUTE_GROUP_COLORS = {
    "Climate/hydroclimate": "#0072B2",
    "Topography/scale": "#009E73",
    "Vegetation/land cover": "#1B9E77",
    "Soil": "#CC79A7",
    "Geology": "#6A3D9A",
}
ATTRIBUTE_GROUP_SHORT = {
    "Climate/hydroclimate": "Climate",
    "Soil": "Soil",
    "Topography/scale": "Topography",
    "Geology": "Geology",
    "Vegetation/land cover": "Vegetation",
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
    "Topography/scale": ["elev_mean", "slope_mean", "area_gages2", "relief", "topographic_wetness", "twi"],
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
    for path in (DATA_DIR, REPORT_DIR, PLOT_DIR, MAIN_FIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def clean_label(name: str) -> str:
    return p_label(name)


def classify_attribute(attribute: str) -> str:
    for group in ATTRIBUTE_GROUP_ORDER:
        if attribute in ATTRIBUTE_GROUP_RULES[group]:
            return group
    lower = attribute.lower()
    if any(token in lower for token in ("prec", "snow", "pet", "arid", "season", "p_mean")):
        return "Climate/hydroclimate"
    if any(token in lower for token in ("slope", "elev", "relief", "topo", "twi", "area")):
        return "Topography/scale"
    if any(token in lower for token in ("lai", "gvf", "forest", "land", "veg")):
        return "Vegetation/land cover"
    if any(token in lower for token in ("soil", "sand", "silt", "clay", "root", "water_content")):
        return "Soil"
    if any(token in lower for token in ("geol", "glim", "aquifer", "carbonate", "perm", "porosity")):
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


def correlation_value(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float, int]:
    xy = pd.concat([x, y], axis=1).dropna()
    n = len(xy)
    if n < 3:
        return np.nan, np.nan, n
    x_val = xy.iloc[:, 0].to_numpy(dtype=float)
    y_val = xy.iloc[:, 1].to_numpy(dtype=float)
    if np.nanstd(x_val) == 0 or np.nanstd(y_val) == 0:
        return np.nan, np.nan, n
    if method == "spearman":
        result = spearmanr(x_val, y_val)
    elif method == "pearson":
        result = pearsonr(x_val, y_val)
    elif method == "kendall":
        result = kendalltau(x_val, y_val)
    else:
        raise ValueError(f"Unsupported correlation method: {method}")
    return float(result.statistic), float(result.pvalue), n


def load_expanded_correlation_data() -> pd.DataFrame:
    mean_maps = pd.read_csv(MEAN_MAP_FILE)
    attrs = pd.read_csv(BASIN_ATTRIBUTE_FILE)
    attribute_cols = [col for col in attrs.columns if col != "basin_id"]
    merged = mean_maps[["basin_id", "parameter", "parameter_mean_unit"]].merge(
        attrs[["basin_id", *attribute_cols]], on="basin_id", how="inner"
    )
    rows: list[dict[str, object]] = []
    for parameter, sub in merged.groupby("parameter"):
        for attribute in attribute_cols:
            rho, p_value, n = correlation_value(sub["parameter_mean_unit"], sub[attribute], "spearman")
            pearson_r, _, _ = correlation_value(sub["parameter_mean_unit"], sub[attribute], "pearson")
            kendall_tau, _, _ = correlation_value(sub["parameter_mean_unit"], sub[attribute], "kendall")
            rows.append(
                {
                    "parameter": parameter,
                    "attribute": attribute,
                    "spearman_rho": rho,
                    "pearson_r_optional": pearson_r,
                    "kendall_tau_optional": kendall_tau,
                    "p_value": p_value,
                    "abs_rho": abs(rho) if pd.notna(rho) else np.nan,
                    "sign": "positive" if pd.notna(rho) and rho > 0 else "negative" if pd.notna(rho) and rho < 0 else "zero",
                    "n_basins": n,
                }
            )
    corr = pd.DataFrame(rows)
    corr["q_value"] = fdr_bh(corr["p_value"])
    corr["rank_abs_rho"] = corr.groupby("parameter")["abs_rho"].rank(method="first", ascending=False)
    corr["relationship_role"] = np.where(
        corr["rank_abs_rho"].eq(1),
        "dominant",
        np.where((corr["abs_rho"] >= 0.3) | (corr["rank_abs_rho"] <= 5), "supportive", "weak"),
    )
    corr["source"] = "expanded_from_distributional_mean_maps"
    return corr


def build_attribute_ordering(corr: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    parameter_group_rank = {group: idx for idx, group in enumerate(PARAMETER_GROUP_ORDER)}
    attribute_group_rank = {group: idx for idx, group in enumerate(ATTRIBUTE_GROUP_ORDER)}
    for attribute, sub in corr.groupby("attribute"):
        strongest = sub.sort_values(["abs_rho", "parameter"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "attribute": attribute,
                "attribute_group": classify_attribute(attribute),
                "max_abs_rho": float(strongest["abs_rho"]),
                "dominant_parameter": strongest["parameter"],
                "dominant_parameter_label": clean_label(strongest["parameter"]),
                "dominant_parameter_group": PARAMETER_GROUPS[strongest["parameter"]],
                "dominant_rho": float(strongest["spearman_rho"]),
            }
        )
    ordering = pd.DataFrame(rows)
    ordering["attribute_group_order"] = ordering["attribute_group"].map(attribute_group_rank)
    ordering["dominant_parameter_group_order"] = ordering["dominant_parameter_group"].map(parameter_group_rank)
    fixed_order = {attribute: idx for idx, attribute in enumerate(attribute_order(ordering["attribute"].tolist()))}
    ordering["fixed_attribute_order"] = ordering["attribute"].map(fixed_order)
    ordering = ordering.sort_values("fixed_attribute_order").reset_index(drop=True)
    ordering["attribute_order"] = np.arange(len(ordering))
    return ordering


def attribute_order(corr_or_attributes: pd.DataFrame | list[str]) -> list[str]:
    if isinstance(corr_or_attributes, pd.DataFrame):
        return build_attribute_ordering(corr_or_attributes)["attribute"].tolist()
    attributes = corr_or_attributes
    grouped: list[str] = []
    for group in ATTRIBUTE_GROUP_ORDER:
        preferred = [attr for attr in ATTRIBUTE_GROUP_RULES[group] if attr in attributes]
        leftovers = sorted([attr for attr in attributes if attr not in preferred and classify_attribute(attr) == group])
        grouped.extend(preferred)
        grouped.extend([attr for attr in leftovers if attr not in grouped])
    return grouped


def load_relationship_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    supplied_corr = pd.read_csv(CORRELATION_FILE)
    if supplied_corr["attribute"].nunique() < 35 and MEAN_MAP_FILE.exists() and BASIN_ATTRIBUTE_FILE.exists():
        corr = load_expanded_correlation_data()
    else:
        corr = supplied_corr.copy()
        corr["source"] = "supplied_distributional_mean_attribute_correlations"
    dominant = pd.read_csv(DOMINANT_FILE)
    selected = pd.read_csv(KEY_RELATIONSHIP_FILE)
    required = {"parameter", "attribute", "spearman_rho", "abs_rho", "q_value"}
    missing = required.difference(corr.columns)
    if missing:
        raise ValueError(f"{CORRELATION_FILE} is missing required columns: {sorted(missing)}")
    corr = corr.copy()
    corr["attribute_group"] = corr["attribute"].map(classify_attribute)
    corr["parameter_group"] = corr["parameter"].map(PARAMETER_GROUPS)
    corr["parameter_label"] = corr["parameter"].map(clean_label)
    corr["strong_relationship"] = corr["abs_rho"].ge(STRONG_ABS_RHO) & corr["q_value"].lt(Q_THRESHOLD)
    return corr, dominant, selected


def consistency_to_rate(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().lower()
    if text.startswith("consistent_"):
        return 1.0
    if "flip" in text or "inconsistent" in text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return np.nan


def load_stability(corr: pd.DataFrame) -> pd.DataFrame:
    seed = pd.read_csv(CROSS_SEED_FILE)
    loss = pd.read_csv(CROSS_LOSS_FILE)
    seed = seed.loc[seed["model_raw"].eq("distributional")].copy()
    loss = loss.loc[loss["model_raw"].eq("distributional")].copy()

    if "loss" in seed.columns:
        seed = (
            seed.assign(seed_sign_rate=seed["sign_consistency_across_seeds"].map(consistency_to_rate))
            .groupby(["parameter", "attribute"], as_index=False)
            .agg(
                seed_mean_rho=("seed_mean_rho", "mean"),
                seed_sd_rho=("seed_sd_rho", "mean"),
                seed_min_sd_rho=("seed_sd_rho", "min"),
                sign_consistency_across_seeds=("seed_sign_rate", "mean"),
                topk_seed_rate=("topk_seed_rate", "max"),
                dominant_seed_rate=("dominant_seed_rate", "max"),
                n_seed_losses=("loss", "nunique"),
            )
        )
    else:
        seed["sign_consistency_across_seeds"] = seed["sign_consistency_across_seeds"].map(consistency_to_rate)

    loss = loss.copy()
    loss["sign_consistency_across_losses"] = loss["sign_consistency_across_losses"].map(consistency_to_rate)

    marker = corr[
        ["parameter", "attribute", "spearman_rho", "abs_rho", "q_value", "strong_relationship"]
    ].merge(
        seed[
            [
                "parameter",
                "attribute",
                "seed_mean_rho",
                "seed_sd_rho",
                "seed_min_sd_rho",
                "sign_consistency_across_seeds",
                "topk_seed_rate",
                "dominant_seed_rate",
            ]
        ],
        on=["parameter", "attribute"],
        how="left",
    )
    marker = marker.merge(
        loss[
            [
                "parameter",
                "attribute",
                "loss_mean_rho",
                "cross_loss_sd_rho",
                "sign_consistency_across_losses",
                "topk_loss_rate",
                "dominant_loss_rate",
            ]
        ],
        on=["parameter", "attribute"],
        how="left",
    )

    seed_sd_threshold = marker["seed_sd_rho"].median(skipna=True)
    loss_sd_threshold = marker["cross_loss_sd_rho"].median(skipna=True)
    marker["seed_sd_threshold"] = seed_sd_threshold
    marker["cross_loss_sd_threshold"] = loss_sd_threshold
    marker["stable_strong_relationship"] = (
        marker["strong_relationship"]
        & marker["sign_consistency_across_seeds"].ge(0.8)
        & marker["sign_consistency_across_losses"].ge(0.8)
        & marker["seed_sd_rho"].le(seed_sd_threshold)
        & marker["cross_loss_sd_rho"].le(loss_sd_threshold)
    )
    marker.to_csv(STABILITY_FILE, index=False)
    return marker


def prepare_panel_tables(corr: pd.DataFrame, stability: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parameters = [p for p in PARAMETER_ORDER if p in corr["parameter"].unique()]
    ordering = build_attribute_ordering(corr)
    ordering.to_csv(ATTRIBUTE_ORDER_FILE, index=False)
    attributes = ordering["attribute"].tolist()

    panel_a = corr.merge(
        stability[["parameter", "attribute", "stable_strong_relationship"]],
        on=["parameter", "attribute"],
        how="left",
    )
    panel_a["dominant_relationship"] = panel_a["rank_abs_rho"].eq(1)
    panel_a = panel_a.merge(
        ordering[
            [
                "attribute",
                "max_abs_rho",
                "dominant_parameter",
                "dominant_parameter_label",
                "dominant_parameter_group",
                "dominant_rho",
                "attribute_order",
            ]
        ].rename(columns={"max_abs_rho": "attribute_max_abs_rho", "dominant_rho": "attribute_dominant_rho"}),
        on="attribute",
        how="left",
    )
    panel_a["parameter_order"] = panel_a["parameter"].map({param: idx for idx, param in enumerate(parameters)})
    panel_a = panel_a.sort_values(["attribute_order", "parameter_order"]).reset_index(drop=True)
    panel_a.to_csv(PANEL_A_FILE, index=False)

    panel_b = (
        corr.sort_values(["parameter", "abs_rho"], ascending=[True, False])
        .groupby("parameter", as_index=False)
        .agg(
            max_abs_rho=("abs_rho", "max"),
            top1_abs_rho=("abs_rho", "max"),
            top3_mean_abs_rho=("abs_rho", lambda x: float(np.mean(np.sort(x.to_numpy())[-3:]))),
            top5_mean_abs_rho=("abs_rho", lambda x: float(np.mean(np.sort(x.to_numpy())[-5:]))),
            dominant_attribute=("attribute", "first"),
            dominant_rho=("spearman_rho", "first"),
            parameter_group=("parameter_group", "first"),
        )
    )
    panel_b["parameter_label"] = panel_b["parameter"].map(clean_label)
    panel_b["parameter_order"] = panel_b["parameter"].map({param: idx for idx, param in enumerate(parameters)})
    panel_b = panel_b.sort_values("parameter_order").reset_index(drop=True)
    panel_b.to_csv(PANEL_B_FILE, index=False)

    panel_c = (
        corr.groupby(["attribute_group", "parameter_group"], as_index=False)
        .agg(
            top3_mean_abs_rho=("abs_rho", lambda x: float(np.mean(np.sort(x.to_numpy())[-min(3, len(x)) :]))),
            mean_abs_rho=("abs_rho", "mean"),
            max_abs_rho=("abs_rho", "max"),
            n_cells=("abs_rho", "size"),
        )
    )
    panel_c["attribute_group_order"] = panel_c["attribute_group"].map(
        {group: idx for idx, group in enumerate(ATTRIBUTE_GROUP_ORDER)}
    )
    panel_c["parameter_group_order"] = panel_c["parameter_group"].map(
        {group: idx for idx, group in enumerate(PARAMETER_GROUP_ORDER)}
    )
    panel_c = panel_c.sort_values(["attribute_group_order", "parameter_group_order"]).reset_index(drop=True)
    panel_c.to_csv(PANEL_C_FILE, index=False)
    return panel_a, panel_b, panel_c


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.02) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.5,
        fontweight="normal",
        color="#111111",
        clip_on=False,
    )


def correlation_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "fig06_purple_white_green",
        ["#6A3D9A", "#F7F7F7", "#1B9E77"],
        N=256,
    )


def summary_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "fig06_teal_summary",
        ["#F7F7F7", "#DDEFEA", "#8CC7B5", "#1B9E77"],
        N=256,
    )


def draw_parameter_group_guides(ax: plt.Axes, parameters: list[str], x_left: float) -> None:
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
        ax.plot([x_left, x_left], [y0, y1], color="#9E9E9E", linewidth=0.45, clip_on=False)
        ax.plot([x_left, x_left + 0.18], [y0, y0], color="#9E9E9E", linewidth=0.45, clip_on=False)
        ax.plot([x_left, x_left + 0.18], [y1, y1], color="#9E9E9E", linewidth=0.45, clip_on=False)
        # group label text removed — shown via y-tick colors instead


def draw_attribute_group_bar(group_ax: plt.Axes, attributes: list[str]) -> None:
    rgb = np.array([mpl.colors.to_rgb(ATTRIBUTE_GROUP_COLORS[classify_attribute(attr)]) for attr in attributes])
    group_ax.imshow(rgb.reshape(len(attributes), 1, 3), aspect="auto", interpolation="nearest", extent=(0.72, 1.0, len(attributes) - 0.5, -0.5))
    group_ax.set_xlim(0, 1)
    group_ax.set_ylim(len(attributes) - 0.5, -0.5)
    group_ax.set_xticks([])
    group_ax.set_yticks([])
    for spine in group_ax.spines.values():
        spine.set_visible(False)
    for group in ATTRIBUTE_GROUP_ORDER:
        idx = [i for i, attr in enumerate(attributes) if classify_attribute(attr) == group]
        if not idx:
            continue
        y0 = min(idx) - 0.5
        y1 = max(idx) + 0.5
        center = (y0 + y1) / 2
        if y0 > -0.5:
            group_ax.axhline(y0, color="#777777", linewidth=0.42)
        group_ax.text(
            0.58,
            center,
            ATTRIBUTE_GROUP_SHORT[group],
            ha="right",
            va="center",
            fontsize=9.0,
            color="#333333",
            rotation=90,
            clip_on=False,
        )


def draw_attribute_labels(label_ax: plt.Axes, attributes: list[str]) -> None:
    label_ax.set_xlim(0, 1)
    label_ax.set_ylim(len(attributes) - 0.5, -0.5)
    label_ax.set_xticks([])
    label_ax.set_yticks([])
    for y, attribute in enumerate(attributes):
        label_ax.text(0.985, y, attribute, ha="right", va="center", fontsize=9.0, color="#111111")
    for spine in label_ax.spines.values():
        spine.set_visible(False)


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


def draw_attribute_group_guides(ax: plt.Axes, attributes: list[str], y_top: float) -> None:
    # y_top is in data coords (negative = above heatmap since y-axis is inverted top-to-bottom)
    # bracket tick goes upward (more negative), text sits above bracket
    for group, start_idx, end_idx in attribute_group_ranges(attributes):
        x0 = start_idx - 0.48
        x1 = end_idx + 0.48
        ax.plot([x0, x1], [y_top, y_top], color="#555555", linewidth=0.7, alpha=0.85, clip_on=False)
        ax.plot([x0, x0], [y_top, y_top - 0.14], color="#555555", linewidth=0.7, alpha=0.85, clip_on=False)
        ax.plot([x1, x1], [y_top, y_top - 0.14], color="#555555", linewidth=0.7, alpha=0.85, clip_on=False)
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


def draw_attribute_group_labels(label_ax: plt.Axes, attributes: list[str]) -> None:
    label_ax.set_xlim(0, 1)
    label_ax.set_ylim(0, 1)
    label_ax.set_xticks([])
    label_ax.set_yticks([])
    for spine in label_ax.spines.values():
        spine.set_visible(False)
    ranges = attribute_group_ranges(attributes)
    if not ranges:
        return
    y_positions = np.linspace(0.78, 0.22, len(ranges))
    for y, (group, _, _) in zip(y_positions, ranges):
        label_ax.text(
            0.02,
            y,
            ATTRIBUTE_GROUP_SHORT[group],
            ha="left",
            va="center",
            fontsize=9.2,
            color=ATTRIBUTE_GROUP_COLORS[group],
            fontweight="bold",
        )

def draw_panel_a(
    ax: plt.Axes,
    cax: plt.Axes,
    panel_a: pd.DataFrame,
) -> None:
    parameters = [p for p in PARAMETER_ORDER if p in panel_a["parameter"].unique()]
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
    ax.set_xticklabels([ATTR_LABELS.get(a, a) for a in attributes], rotation=55, ha="right", rotation_mode="anchor", fontsize=9.8)
    ax.set_yticks(np.arange(len(parameters)))
    ax.set_yticklabels([clean_label(param) for param in parameters], rotation=0, ha="right", fontsize=10.2)
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
    xs: list[int] = []
    ys: list[int] = []
    values: list[float] = []
    sizes: list[float] = []
    for y, parameter in enumerate(parameters):
        for x, attribute in enumerate(attributes):
            row = lookup.loc[(attribute, parameter)]
            rho = float(row["spearman_rho"])
            if np.isfinite(rho):
                xs.append(x)
                ys.append(y)
                values.append(rho)
                sizes.append(5.0 + 132.0 * (abs(rho) ** 1.55))
            if bool(row["dominant_relationship"]):
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
            if bool(row["strong_relationship"]):
                if bool(row["stable_strong_relationship"]):
                    ax.plot(x, y, marker="*", markersize=5.6, color="#111111", linestyle="None", zorder=5)
                else:
                    ax.plot(x, y, marker="o", markersize=3.6, color="#111111", linestyle="None", zorder=5)
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

    previous_group = classify_attribute(attributes[0])
    for x, attribute in enumerate(attributes[1:], start=1):
        group = classify_attribute(attribute)
        if group != previous_group:
            ax.axvline(x - 0.5, color="#8A8A8A", linewidth=0.55)
            previous_group = group
    previous_param_group = PARAMETER_GROUPS[parameters[0]]
    for y, parameter in enumerate(parameters[1:], start=1):
        group = PARAMETER_GROUPS[parameter]
        if group != previous_param_group:
            ax.axhline(y - 0.5, color="#8A8A8A", linewidth=0.55)
            previous_param_group = group

    # y-tick labels use default black (parameter group color shown via bracket guides only)
    for tick in ax.get_yticklabels():
        tick.set_color("#111111")
    # bracket guides on left — pushed far enough left to clear tick labels
    draw_parameter_group_guides(ax, parameters, x_left=-2.8)
    # y_top=-0.52 places the bracket line just below the x-axis, labels sit tight above it
    draw_attribute_group_guides(ax, attributes, y_top=-0.52)

    # panel label inside bottom-right of axes
    ax.text(
        0.99, 0.01, "(a)",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=12.5, fontweight="normal", color="#111111",
    )

    # legend inside panel a — top-right corner
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#111111", markeredgecolor="#111111", markersize=4.0, label=r"$|\rho|\geq0.5$, $q<0.05$"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#111111", markeredgecolor="#111111", markersize=7.0, label="stable strong"),
        Patch(facecolor="none", edgecolor="#111111", linewidth=1.35, label="dominant"),
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
        columnspacing=0.9,
        labelspacing=0.35,
        borderaxespad=0.3,
        fontsize=9.2,
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


def draw_panel_b(ax: plt.Axes, panel_b: pd.DataFrame) -> None:
    # vertical bars: parameters on x-axis (no rotation), top-k |rho| on y-axis
    ordered = panel_b.sort_values("parameter_order").reset_index(drop=True)
    x = np.arange(len(ordered))
    colors = [PARAMETER_GROUP_COLORS[group] for group in ordered["parameter_group"]]
    ax.vlines(x, ordered["top5_mean_abs_rho"], ordered["top1_abs_rho"], color="#B8B8B8", linewidth=1.0, zorder=1)
    ax.scatter(x, ordered["top3_mean_abs_rho"], s=30, c=colors, edgecolor="#222222", linewidth=0.35, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["parameter_label"], rotation_mode="anchor", fontsize=10.0)
    ax.set_ylabel(r"Top-$k$ mean $|\rho|$" + "\n(parameter means)", fontsize=10.5)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    ax.tick_params(axis="both", width=0.5, length=2.4, colors="#222222")
    ax.tick_params(axis="y", labelsize=10.0)
    # panel label inside top-left
    ax.text(
        0.02, 0.98, "(b)",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12.5, fontweight="normal", color="#111111",
    )
    handles = [
        Line2D(
            [0], [0],
            color="#B8B8B8",
            linewidth=1.0,
            label="top-5 mean to top-1",
        ),
    ] + [
        Line2D(
            [0], [0],
            marker="o", color="none",
            markerfacecolor=PARAMETER_GROUP_COLORS[group],
            markeredgecolor="#222222",
            markeredgewidth=0.35,
            markersize=4.5,
            label=group,
        )
        for group in PARAMETER_GROUP_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        framealpha=0.88,
        edgecolor="#CCCCCC",
        handlelength=1.1,
        handletextpad=0.35,
        borderaxespad=0.3,
        labelspacing=0.25,
        fontsize=9.0,
    )



def draw_panel_c(ax: plt.Axes, cax: plt.Axes, panel_c: pd.DataFrame) -> None:
    # transposed: rows = parameter groups (4), cols = attribute groups (5)
    attribute_group_order = [group for group in ATTRIBUTE_GROUP_ORDER if group in set(panel_c["attribute_group"])]
    matrix_df = (
        panel_c.pivot(index="parameter_group", columns="attribute_group", values="top3_mean_abs_rho")
        .reindex(index=PARAMETER_GROUP_ORDER, columns=attribute_group_order)
    )
    matrix = matrix_df.to_numpy()
    im = ax.imshow(matrix, cmap=summary_cmap(), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(attribute_group_order)))
    ax.set_xticklabels(
        [ATTRIBUTE_GROUP_SHORT.get(g, g) for g in attribute_group_order],
        rotation_mode="anchor", fontsize=9.6,
    )
    ax.set_yticks(np.arange(len(PARAMETER_GROUP_ORDER)))
    ax.set_yticklabels(PARAMETER_GROUP_ORDER, fontsize=9.8)
    ax.tick_params(axis="both", length=0, pad=3)
    ax.set_xticks(np.arange(-0.5, len(attribute_group_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(PARAMETER_GROUP_ORDER), 1), minor=True)
    ax.grid(which="minor", color="#D9D9D9", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=9.4, color="#222222")
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_edgecolor("#777777")
    # panel label inside top-left
    ax.text(
        0.02, 0.98, "(c)",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12.5, fontweight="normal", color="#111111",
    )
    cbar = plt.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label(r"Top-3 mean $|\rho|$", fontsize=10.0)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=9.4, width=0.45, length=2.0, colors="#222222")
    cbar.outline.set_linewidth(0.5)


def make_figure(panel_a: pd.DataFrame, panel_b: pd.DataFrame, panel_c: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(280 * MM, 240 * MM), constrained_layout=False)

    outer = GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[1.55, 1.0],
        hspace=0.44,
    )

    top = GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=outer[0],
        width_ratios=[1.0, 0.018],
        wspace=0.012,
    )

    bottom = GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=outer[1],
        width_ratios=[5.2, 2.2],
        wspace=0.14,
    )

    ax_a  = fig.add_subplot(top[0])
    cax_a = fig.add_subplot(top[1])
    ax_b  = fig.add_subplot(bottom[0])
    ax_c  = fig.add_subplot(bottom[1])
    # cax_c is placed manually after draw to sit tight against ax_c
    cax_c = fig.add_axes([0, 0, 0.01, 0.01])  # placeholder, repositioned below

    draw_panel_b(ax_b, panel_b)
    draw_panel_c(ax_c, cax_c, panel_c)
    draw_panel_a(ax_a, cax_a, panel_a)

    ax_c.set_zorder(1)
    cax_c.set_zorder(1)
    ax_b.set_zorder(1)
    ax_a.set_zorder(3)
    cax_a.set_zorder(3)
    ax_a.patch.set_alpha(1.0)
    cax_a.patch.set_alpha(1.0)

    fig.subplots_adjust(left=0.07, right=0.955, bottom=0.30, top=0.96)

    # Snap cax_c tight to ax_c right edge
    fig.canvas.draw()
    pc = ax_c.get_position()
    c_height = pc.height * 0.97
    ax_c.set_position([pc.x0, pc.y0, pc.width, c_height])
    pc = ax_c.get_position()
    cax_c.set_position([pc.x1 + 0.008, pc.y0, 0.012, pc.height])

    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def top_relationship_candidates(corr: pd.DataFrame, stability: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    merged = corr.merge(
        stability[["parameter", "attribute", "stable_strong_relationship"]],
        on=["parameter", "attribute"],
        how="left",
    )
    return merged.sort_values("abs_rho", ascending=False).head(n)[
        ["parameter", "parameter_label", "attribute", "spearman_rho", "q_value", "relationship_role", "stable_strong_relationship"]
    ]


def write_notes(corr: pd.DataFrame, panel_b: pd.DataFrame, panel_c: pd.DataFrame, stability: pd.DataFrame) -> None:
    ordering = build_attribute_ordering(corr)
    attributes = ordering["attribute"].tolist()
    parameters = [p for p in PARAMETER_ORDER if p in corr["parameter"].unique()]
    attribute_group_lines: list[str] = []
    for group in ATTRIBUTE_GROUP_ORDER:
        group_attrs = [attr for attr in attributes if classify_attribute(attr) == group]
        if group_attrs:
            attribute_group_lines.append(f"- {group}: " + ", ".join(f"`{attr}`" for attr in group_attrs))

    parameter_group_lines: list[str] = []
    for group in PARAMETER_GROUP_ORDER:
        group_params = [param for param in parameters if PARAMETER_GROUPS[param] == group]
        parameter_group_lines.append(f"- {group}: " + ", ".join(f"`{clean_label(param)}`" for param in group_params))

    candidates = top_relationship_candidates(corr, stability)
    candidate_lines = [
        f"- `{row.parameter_label}` - `{row.attribute}`: Spearman rho = {row.spearman_rho:.3f}, q = {row.q_value:.2e}, {row.relationship_role}; stable strong = {bool(row.stable_strong_relationship)}"
        for row in candidates.itertuples(index=False)
    ]

    strongest_groups = panel_c.sort_values("top3_mean_abs_rho", ascending=False).head(6)
    group_candidate_lines = [
        f"- {row.attribute_group} x {row.parameter_group}: top-3 mean |rho| = {row.top3_mean_abs_rho:.3f}"
        for row in strongest_groups.itertuples(index=False)
    ]

    strongest_parameters = panel_b.sort_values("top3_mean_abs_rho", ascending=False).head(6)
    parameter_candidate_lines = [
        f"- `{row.parameter_label}`: top-3 mean |rho| = {row.top3_mean_abs_rho:.3f}; dominant attribute = `{row.dominant_attribute}` (rho = {row.dominant_rho:.3f})"
        for row in strongest_parameters.itertuples(index=False)
    ]

    lines = [
        "# Fig06 mean-attribute relationship plot notes",
        "",
        "## 1. Input files",
        "",
        f"- `{CORRELATION_FILE}`",
        f"- `{DOMINANT_FILE}`",
        f"- `{KEY_RELATIONSHIP_FILE}`",
        f"- `{MEAN_MAP_FILE}`",
        f"- `{BASIN_ATTRIBUTE_FILE}`",
        f"- `{FOCUSED_STABILITY_FILE}` (listed input; not required after the full cross-seed and cross-loss tables are merged)",
        f"- `{CROSS_SEED_FILE}`",
        f"- `{CROSS_LOSS_FILE}`",
        "",
        "## 2. Attribute grouping",
        "",
        f"- Panel (a) contains {len(attributes)} basin attributes.",
        "- The supplied manuscript correlation table contains 18 core attributes. To satisfy the 35-attribute Figure 6 requirement, the script expands the table from existing upstream analysis data: `distributional_parameter_mean_maps_long.csv` and `basin_attributes.csv`. The expanded rows use the same Spearman/FDR/dominant-role definitions as the manuscript mean-attribute pipeline.",
        "- Attributes are sorted first by attribute group, then by the parameter group containing each attribute's strongest absolute correlation, and finally by descending max |rho| within that block.",
        *attribute_group_lines,
        "",
        "## 3. Parameter grouping",
        "",
        "- The 14 HBV parameters are reordered and regrouped for Figure 6 as `snow`, `soil`, `production`, and `routing` so the circle heatmap follows process logic rather than the previous storage/recession layout.",
        *parameter_group_lines,
        "",
        "## 4. Spearman rho definition",
        "",
        "- Spearman rho values use the upstream manuscript mean-attribute definition: basin-level Spearman correlations between seed-averaged distributional parameter means (`parameter_mean_unit`) and basin attributes over the available CAMELS-US basins.",
        "- The 18 core relationships supplied in `distributional_mean_attribute_correlations.csv` were used as the statistical-definition reference; the full 35-attribute matrix was regenerated from the existing upstream long tables with the same definition so panel (a) can show the requested 35 x 14 structure.",
        "",
        "## 5. FDR q-value use",
        "",
        f"- `q_value` is used only for marker eligibility. Strong relationships require `abs_rho >= {STRONG_ABS_RHO}` and `q_value < {Q_THRESHOLD}`.",
        "",
        "## 6. Dominant relationship definition",
        "",
        "- A dominant relationship is the row with the largest absolute Spearman rho within each parameter. The script uses `rank_abs_rho == 1` from the existing table and draws a thin black box around that cell.",
        "",
        "## 7. Stable strong relationship definition",
        "",
        "- Stability is used only as an additional marker in panel (a), not as a model comparison panel.",
        "- The full cross-seed and cross-loss sensitivity tables are filtered to `model_raw == distributional`.",
        "- Cross-seed rows are aggregated across losses by pair: mean sign consistency, mean seed SD, and maximum top-k rate are retained.",
        "- Text values `consistent_positive` and `consistent_negative` are treated as sign consistency 1.0; sign flips are treated as 0.0.",
        f"- Stable strong relationships require `abs_rho >= {STRONG_ABS_RHO}`, `q_value < {Q_THRESHOLD}`, seed sign consistency >= 0.8, loss sign consistency >= 0.8, seed SD no greater than the distributional-pair median, and cross-loss SD no greater than the distributional-pair median.",
        f"- Median seed SD threshold: {stability['seed_sd_threshold'].dropna().iloc[0]:.4f}. Median cross-loss SD threshold: {stability['cross_loss_sd_threshold'].dropna().iloc[0]:.4f}.",
        "",
        "## 8. Panel (b) top-3 mean |rho|",
        "",
        "- For each parameter mean, all attribute relationships are sorted by absolute Spearman rho. Panel (b) plots the arithmetic mean of the three largest absolute rho values. The grey segment spans top-5 mean to top-1 strength (`top5_mean_abs_rho` to `top1_abs_rho`), so it is a top-k range marker rather than a bootstrap or seed-variation error bar.",
        "",
        "## 9. Panel (c) group-level top-3 mean |rho|",
        "",
        "- For each attribute-group x parameter-group block, all cells in the block are sorted by absolute Spearman rho. Panel (c) uses the arithmetic mean of the three largest values in that block, or all values if fewer than three cells are available.",
        "",
        "## 10. Colors and symbols",
        "",
        "- Panel (a) is a circle heatmap: every 35 x 14 cell remains present as a square grid location, circle size encodes `abs_rho`, and circle color encodes signed Spearman rho.",
        "- Panel (a) correlation colors use a colorblind-friendly purple-white-green diverging scale, with negative rho on the purple side, near-zero rho near white, and positive rho on the green side.",
        "- Panel (a) markers: black dot = strong relationship; black star = stable strong relationship; black thick box = dominant attribute for that parameter.",
        "- Attribute group side bar uses colorblind-friendly group colors. Panel (b) dots are colored by the same parameter groups. Panel (c) uses a light grey-to-teal sequential scale.",
        "",
        "## 11. Main-text headline candidates",
        "",
        "- Strongest pair-level candidates:",
        *candidate_lines,
        "- Strongest parameter-level control candidates:",
        *parameter_candidate_lines,
        "- Strongest group-level candidates:",
        *group_candidate_lines,
        "",
        "## Output checks",
        "",
        "- No overall title or subplot titles are used.",
        "- Panels are labeled only as `(a)`, `(b)`, and `(c)`.",
        f"- Panel (a) is {len(attributes)} x {len(parameters)}.",
        "- No icons, network links, arrows, or explanatory boxes are used.",
        f"- PNG: `{OUT_PNG}`",
        "- Weak relationships are visually muted by the small circle size and pale colormap center; data values are unchanged.",
        f"- Panel data: `{PANEL_A_FILE}`, `{ATTRIBUTE_ORDER_FILE}`, `{PANEL_B_FILE}`, `{PANEL_C_FILE}`, `{STABILITY_FILE}`",
    ]
    NOTES_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    ensure_dirs()
    corr, _, _ = load_relationship_data()
    stability = load_stability(corr)
    panel_a, panel_b, panel_c = prepare_panel_tables(corr, stability)
    make_figure(panel_a, panel_b, panel_c)
    write_notes(corr, panel_b, panel_c, stability)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {PANEL_A_FILE}")
    print(f"Wrote {ATTRIBUTE_ORDER_FILE}")
    print(f"Wrote {PANEL_B_FILE}")
    print(f"Wrote {PANEL_C_FILE}")
    print(f"Wrote {STABILITY_FILE}")
    print(f"Wrote {NOTES_FILE}")


if __name__ == "__main__":
    main()
