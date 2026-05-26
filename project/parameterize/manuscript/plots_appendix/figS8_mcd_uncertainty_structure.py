"""Fig. S8 — MC-dropout variance structure.

This appendix figure mirrors the visual grammar of main-text Fig. 7:
  (a) variance-attribute circle heatmap
  (b) parameter-level top-k relationship strength
  (c) mean-variance coupling vs boundary sensitivity diagnostics
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

PLOTS_ROOT = Path(__file__).resolve().parents[1] / "plots"
sys.path.insert(0, str(PLOTS_ROOT))

from plot_fig07_uncertainty_attribute_relationships import (  # noqa: E402
    ATTRIBUTE_GROUP_ORDER,
    BASIN_ATTRIBUTE_FILE,
    BOUNDARY_SENSITIVITY_THRESHOLD,
    MEAN_STD_COUPLING_THRESHOLD,
    PARAMETER_GROUP_COLORS,
    PARAMETER_GROUP_ORDER,
    PARAMETER_GROUPS,
    PARAMETER_ORDER,
    Q_THRESHOLD,
    STRONG_ABS_RHO,
    build_attribute_ordering,
    classify_attribute,
    clean_label,
    draw_panel_a,
    draw_panel_b,
    fdr_bh,
    setup_style,
)

ROOT = Path("/workspace/autoresearch")
MANUSCRIPT_ROOT = ROOT / "project" / "parameterize" / "manuscript"
FIGURES8_ROOT = MANUSCRIPT_ROOT / "analysis" / "figureS8"
DATA_DIR = FIGURES8_ROOT / "data"
REPORT_DIR = FIGURES8_ROOT / "reports"
APP_FIG_DIR = MANUSCRIPT_ROOT / "figures" / "appendix"

PARAMETER_TABLE = MANUSCRIPT_ROOT / "analysis" / "figure2" / "data" / "parameter_estimates_by_run_long.csv"
OUT_PNG = APP_FIG_DIR / "figS8_mcd_uncertainty_structure.png"
OUT_PDF = APP_FIG_DIR / "figS8_mcd_uncertainty_structure.pdf"
PANEL_A_FILE = DATA_DIR / "figS8_panel_a_mcd_variance_heatmap_data.csv"
PANEL_B_FILE = DATA_DIR / "figS8_panel_b_mcd_variance_structure_strength.csv"
PANEL_C_FILE = DATA_DIR / "figS8_panel_c_mcd_variance_diagnostics.csv"
MCD_VARIANCE_MAP_FILE = DATA_DIR / "figS8_mcd_variance_maps_long.csv"
NOTES_FILE = REPORT_DIR / "figS8_plot_notes.md"

DPI = 600
MM = 1 / 25.4
PRIMARY_LOSS = "HybridNseBatchLoss"


def ensure_dirs() -> None:
    for path in (DATA_DIR, REPORT_DIR, APP_FIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def correlation_value(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    xy = pd.concat([x, y], axis=1).dropna()
    n = len(xy)
    if n < 3:
        return np.nan, np.nan, n
    xv = xy.iloc[:, 0].to_numpy(dtype=float)
    yv = xy.iloc[:, 1].to_numpy(dtype=float)
    if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
        return np.nan, np.nan, n
    result = spearmanr(xv, yv)
    return float(result.statistic), float(result.pvalue), n


def sign_label(value: float) -> str:
    if pd.isna(value):
        return "missing"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def interpretation_flag(abs_mean_variance: float, abs_boundary: float, near_share: float) -> str:
    flags: list[str] = []
    if pd.notna(abs_mean_variance) and abs_mean_variance >= MEAN_STD_COUPLING_THRESHOLD:
        flags.append("mean-coupled")
    if (pd.notna(abs_boundary) and abs_boundary >= BOUNDARY_SENSITIVITY_THRESHOLD) or (
        pd.notna(near_share) and near_share >= 0.25
    ):
        flags.append("boundary-sensitive")
    if not flags:
        return "clean"
    if len(flags) == 2:
        return "interpret with caution"
    return flags[0]


def load_mcd_variance_maps() -> pd.DataFrame:
    params = pd.read_csv(
        PARAMETER_TABLE,
        usecols=[
            "model_raw",
            "loss",
            "seed",
            "basin_id",
            "parameter",
            "estimate_norm",
            "sample_std_norm",
            "n_parameter_samples",
        ],
    )
    mcd = params.loc[
        params["model_raw"].eq("mc_dropout")
        & params["loss"].eq(PRIMARY_LOSS)
        & params["parameter"].isin(PARAMETER_ORDER)
    ].copy()
    if mcd.empty:
        raise ValueError(f"No mc_dropout rows found in {PARAMETER_TABLE} for {PRIMARY_LOSS}")
    mcd["parameter_variance_unit"] = mcd["sample_std_norm"].pow(2)
    by_basin = (
        mcd.groupby(["basin_id", "parameter"], as_index=False)
        .agg(
            parameter_mean_unit=("estimate_norm", "mean"),
            parameter_variance_unit=("parameter_variance_unit", "mean"),
            seed_sd_variance_unit=("parameter_variance_unit", "std"),
            n_seeds=("seed", "nunique"),
            n_parameter_samples=("n_parameter_samples", "median"),
        )
        .sort_values(["parameter", "basin_id"])
        .reset_index(drop=True)
    )
    by_basin["distance_to_boundary"] = np.minimum(
        by_basin["parameter_mean_unit"], 1.0 - by_basin["parameter_mean_unit"]
    )
    by_basin["near_boundary_flag"] = by_basin["distance_to_boundary"].le(0.05)
    by_basin.to_csv(MCD_VARIANCE_MAP_FILE, index=False)
    return by_basin


def load_diagnostics(maps: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for parameter, sub in maps.groupby("parameter"):
        mean_var, p_mean_var, n = correlation_value(
            sub["parameter_mean_unit"], sub["parameter_variance_unit"]
        )
        boundary_var, p_boundary, _ = correlation_value(
            sub["distance_to_boundary"], sub["parameter_variance_unit"]
        )
        rows.append(
            {
                "parameter": parameter,
                "parameter_group": PARAMETER_GROUPS.get(parameter, "production"),
                "mean_std_coupling": abs(mean_var) if pd.notna(mean_var) else np.nan,
                "mean_variance_spearman": mean_var,
                "mean_variance_p_value": p_mean_var,
                "boundary_sensitivity": abs(boundary_var) if pd.notna(boundary_var) else np.nan,
                "boundary_distance_variance_spearman": boundary_var,
                "boundary_distance_variance_p_value": p_boundary,
                "near_boundary_share": float(sub["near_boundary_flag"].mean()),
                "n_basins": n,
            }
        )
    diag = pd.DataFrame(rows)
    diag["interpretation_flag"] = diag.apply(
        lambda r: interpretation_flag(
            r["mean_std_coupling"], r["boundary_sensitivity"], r["near_boundary_share"]
        ),
        axis=1,
    )
    diag.to_csv(PANEL_C_FILE, index=False)
    return diag


def load_corr_data(maps: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    attrs = pd.read_csv(BASIN_ATTRIBUTE_FILE)
    attr_cols = [c for c in attrs.columns if c != "basin_id"]
    merged = maps.merge(attrs[["basin_id", *attr_cols]], on="basin_id", how="inner")

    rows: list[dict[str, object]] = []
    for parameter, sub in merged.groupby("parameter"):
        for attribute in attr_cols:
            rho, p_value, n = correlation_value(sub["parameter_variance_unit"], sub[attribute])
            rows.append(
                {
                    "parameter": parameter,
                    "attribute": attribute,
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "abs_rho": abs(rho) if pd.notna(rho) else np.nan,
                    "n_basins": n,
                    "sign": sign_label(rho),
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
    corr = corr.merge(diagnostics, on=["parameter", "n_basins"], how="left")
    corr["attribute_group"] = corr["attribute"].map(classify_attribute)
    corr["parameter_group"] = corr["parameter"].map(PARAMETER_GROUPS)
    corr["parameter_label"] = corr["parameter"].map(clean_label)
    corr["strong_flag"] = corr["abs_rho"].ge(STRONG_ABS_RHO) & corr["q_value"].lt(Q_THRESHOLD)
    corr["dominant_flag"] = corr["rank_abs_rho"].eq(1)
    corr["caution_flag"] = (
        corr["interpretation_flag"].str.lower().str.contains("caution|coupled|boundary", na=False)
    )
    return corr


def prepare_panel_a(corr: pd.DataFrame) -> pd.DataFrame:
    ordering = build_attribute_ordering(corr)
    panel_a = corr.merge(ordering[["attribute", "attribute_order"]], on="attribute", how="left")
    parameters = [p for p in PARAMETER_ORDER if p in corr["parameter"].unique()]
    panel_a["parameter_order"] = panel_a["parameter"].map({p: i for i, p in enumerate(parameters)})
    panel_a = panel_a.sort_values(["attribute_order", "parameter_order"]).reset_index(drop=True)
    panel_a.to_csv(PANEL_A_FILE, index=False)
    return panel_a


def prepare_panel_b(corr: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    parameters = [p for p in PARAMETER_ORDER if p in corr["parameter"].unique()]
    diag_lookup = diagnostics.set_index("parameter")
    rows: list[dict[str, object]] = []
    for idx, parameter in enumerate(parameters):
        sub = corr.loc[corr["parameter"].eq(parameter)].sort_values("abs_rho", ascending=False)
        abs_rhos = sub["abs_rho"].dropna().to_numpy()
        top1 = float(abs_rhos[0]) if len(abs_rhos) else np.nan
        top3 = float(np.mean(abs_rhos[: min(3, len(abs_rhos))])) if len(abs_rhos) else np.nan
        top5 = float(np.mean(abs_rhos[: min(5, len(abs_rhos))])) if len(abs_rhos) else np.nan
        diag_row = diag_lookup.loc[parameter] if parameter in diag_lookup.index else None
        rows.append(
            {
                "parameter": parameter,
                "parameter_label": clean_label(parameter),
                "parameter_group": PARAMETER_GROUPS.get(parameter, "production"),
                "parameter_order": idx,
                "top1_abs_rho": top1,
                "top3_mean_abs_rho": top3,
                "top5_mean_abs_rho": top5,
                "dominant_attribute": sub.iloc[0]["attribute"] if len(sub) else "",
                "dominant_rho": float(sub.iloc[0]["spearman_rho"]) if len(sub) else np.nan,
                "mean_std_coupling": float(diag_row["mean_std_coupling"]) if diag_row is not None else np.nan,
                "boundary_sensitivity": float(diag_row["boundary_sensitivity"]) if diag_row is not None else np.nan,
                "caution_flag": bool(diag_row["interpretation_flag"] != "clean") if diag_row is not None else False,
            }
        )
    panel_b = pd.DataFrame(rows)
    panel_b.to_csv(PANEL_B_FILE, index=False)
    return panel_b


def draw_panel_c_mcd(ax: plt.Axes, panel_c: pd.DataFrame) -> None:
    offsets = {
        "parTT": (-28, 10),
        "parCFMAX": (8, 8),
        "parCWH": (-38, -12),
        "parPERC": (6, 6),
        "parUZL": (-38, -12),
        "route_b": (6, -12),
    }
    annotate = {"parCWH", "parPERC", "parUZL", "route_b", "parCFMAX", "parTT"}
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
    ax.text(0.03, 0.06, "clean", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0, color="#3A6A46")
    for _, row in panel_c.iterrows():
        x = float(row["mean_std_coupling"]) if pd.notna(row["mean_std_coupling"]) else np.nan
        y = float(row["boundary_sensitivity"]) if pd.notna(row["boundary_sensitivity"]) else np.nan
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        color = PARAMETER_GROUP_COLORS.get(row["parameter_group"], "#888888")
        ax.scatter(x, y, s=38, color=color, edgecolor="#222222", linewidth=0.4, zorder=3)
        if row["parameter"] in annotate:
            dx, dy = offsets.get(row["parameter"], (5, 5))
            ax.annotate(
                clean_label(row["parameter"]),
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=6.8,
                color="#111111",
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6, shrinkA=0, shrinkB=2),
            )

    ax.axvline(MEAN_STD_COUPLING_THRESHOLD, color="#888888", linewidth=0.7, linestyle="--", zorder=1)
    ax.axhline(BOUNDARY_SENSITIVITY_THRESHOLD, color="#888888", linewidth=0.7, linestyle="--", zorder=1)
    ax.text(MEAN_STD_COUPLING_THRESHOLD + 0.015, 0.97, r"$|\rho|=0.5$", ha="left", va="top", fontsize=6.7, color="#555555")
    ax.text(0.98, BOUNDARY_SENSITIVITY_THRESHOLD + 0.015, r"$|\rho|=0.4$", ha="right", va="bottom", fontsize=6.7, color="#555555")
    ax.set_xlabel(r"Mean-variance coupling $|\rho|$", fontsize=8.5)
    ax.set_ylabel(r"Boundary sensitivity $|\rho|$", fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="both", labelsize=7.5, width=0.5, length=2.4, colors="#222222")
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
        fontsize=7.0,
    )
    ax.text(0.02, 0.98, "(c)", transform=ax.transAxes, ha="left", va="top", fontsize=11.5, color="#111111")


def make_figure(panel_a: pd.DataFrame, panel_b: pd.DataFrame, panel_c: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(250 * MM, 215 * MM), constrained_layout=False)
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.55, 1.0], hspace=0.38)
    top = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[1.0, 0.018], wspace=0.012)
    bottom = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], width_ratios=[5.2, 2.2], wspace=0.14)

    ax_a = fig.add_subplot(top[0])
    cax_a = fig.add_subplot(top[1])
    ax_b = fig.add_subplot(bottom[0])
    ax_c = fig.add_subplot(bottom[1])

    draw_panel_a(ax_a, cax_a, panel_a)
    draw_panel_b(ax_b, panel_b)
    draw_panel_c_mcd(ax_c, panel_c)

    fig.subplots_adjust(left=0.07, right=0.955, bottom=0.30, top=0.96)
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PDF, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_notes(corr: pd.DataFrame, panel_b: pd.DataFrame, panel_c: pd.DataFrame) -> None:
    parameters = [p for p in PARAMETER_ORDER if p in corr["parameter"].unique()]
    strong = corr.loc[corr["strong_flag"]]
    caution_strong = corr.loc[corr["strong_flag"] & corr["caution_flag"]]
    lines = [
        "# Fig. S8 MC-dropout variance structure plot notes",
        "",
        "## 1. Statistical object",
        "",
        "- Fig. S8 mirrors main-text Fig. 7 but replaces distributional parameter-scale std with MC-dropout sample variance.",
        f"- MC-dropout variance is `sample_std_norm^2`, filtered to `{PRIMARY_LOSS}`, then averaged across seeds for each basin x parameter.",
        "- Spearman rho is computed between this MC-dropout variance field and each basin attribute across 531 CAMELS-US basins.",
        "",
        "## 2. Panel (a): variance-attribute circle heatmap",
        "",
        f"- {len(parameters)} parameters x {corr['attribute'].nunique()} attributes.",
        "- Circle size = |rho|, color = rho, using the same purple-white-green diverging colormap and [-1, 1] range as Fig. 7.",
        "- Black box = dominant (rank-1 |rho| for that parameter).",
        f"- Black dot = strong (|rho| >= {STRONG_ABS_RHO}, q < {Q_THRESHOLD}).",
        "- Grey triangle = caution (mean-coupled or boundary-sensitive).",
        f"- Strong relationships: {len(strong)} pairs.",
        f"- Strong + caution: {len(caution_strong)} pairs.",
        "",
        "## 3. Panel (b): parameter-level variance structure strength",
        "",
        "- Vertical dot+interval plot follows Fig. 7 exactly: grey line spans top5 mean |rho| to top1 |rho|; colored dot is top3 mean |rho|.",
        "",
        "## 4. Panel (c): mean-variance coupling vs boundary sensitivity",
        "",
        "- x: |Spearman rho| between MC-dropout parameter mean and MC-dropout variance.",
        "- y: |Spearman rho| between normalized distance to boundary and MC-dropout variance.",
        f"- Dashed thresholds match Fig. 7: x={MEAN_STD_COUPLING_THRESHOLD:.1f}, y={BOUNDARY_SENSITIVITY_THRESHOLD:.1f}.",
        f"- Caution parameters: {panel_c.loc[panel_c['interpretation_flag'] != 'clean', 'parameter'].map(clean_label).tolist()}.",
        "",
        "## 5. Output files",
        "",
        f"- PNG: `{OUT_PNG}`",
        f"- PDF: `{OUT_PDF}`",
        f"- MCD variance maps: `{MCD_VARIANCE_MAP_FILE}`",
        f"- Panel A data: `{PANEL_A_FILE}`",
        f"- Panel B data: `{PANEL_B_FILE}`",
        f"- Panel C data: `{PANEL_C_FILE}`",
    ]
    NOTES_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    setup_style()
    ensure_dirs()
    maps = load_mcd_variance_maps()
    diagnostics = load_diagnostics(maps)
    corr = load_corr_data(maps, diagnostics)
    panel_a = prepare_panel_a(corr)
    panel_b = prepare_panel_b(corr, diagnostics)
    panel_c = diagnostics.sort_values(
        "parameter", key=lambda s: s.map({p: i for i, p in enumerate(PARAMETER_ORDER)})
    ).reset_index(drop=True)
    make_figure(panel_a, panel_b, panel_c)
    write_notes(corr, panel_b, panel_c)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {MCD_VARIANCE_MAP_FILE}")
    print(f"Wrote {PANEL_A_FILE}")
    print(f"Wrote {PANEL_B_FILE}")
    print(f"Wrote {PANEL_C_FILE}")
    print(f"Wrote {NOTES_FILE}")


if __name__ == "__main__":
    main()
