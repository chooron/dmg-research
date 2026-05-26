from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plots"))

from plot_fig06_mean_attribute_relationships import (
    ATTR_LABELS,
    ATTRIBUTE_GROUP_ORDER,
    ATTRIBUTE_GROUP_SHORT,
    PARAMETER_GROUPS,
    PARAMETER_ORDER,
    STRONG_ABS_RHO,
    classify_attribute,
    clean_label,
    correlation_cmap,
    draw_attribute_group_guides,
    setup_style,
)

ROOT = Path("/workspace/autoresearch")
MANUSCRIPT_ROOT = ROOT / "project" / "parameterize" / "manuscript"
STABILITY_ROOT = ROOT / "project" / "parameterize" / "outputs" / "analysis" / "stability_stats"
FIGURE6_DATA_DIR = MANUSCRIPT_ROOT / "analysis" / "figure6" / "data"
FIGURES7_ROOT = MANUSCRIPT_ROOT / "analysis" / "figureS7"
DATA_DIR = FIGURES7_ROOT / "data"
REPORT_DIR = FIGURES7_ROOT / "reports"
APPENDIX_FIG_DIR = MANUSCRIPT_ROOT / "figures" / "appendix"

SEED_LOSS_CORR_FILE = (
    MANUSCRIPT_ROOT
    / "analysis"
    / "01_model_consistency"
    / "data"
    / "seed_loss_correlation_matrix_long.csv"
)
PARAM_LONG_FILE = STABILITY_ROOT / "clean" / "params_long_clean.csv"
BASIN_ATTRIBUTE_FILE = STABILITY_ROOT / "tables" / "basin_attributes.csv"
FIG06_ATTRIBUTE_ORDER_FILE = FIGURE6_DATA_DIR / "fig06_attribute_ordering.csv"

OUT_PNG = APPENDIX_FIG_DIR / "figS7_mean_attr_matrix_base_mcd.png"
OUT_PDF = APPENDIX_FIG_DIR / "figS7_mean_attr_matrix_base_mcd.pdf"
HEATMAP_DATA_FILE = DATA_DIR / "figS7_base_mcd_heatmap_data.csv"
NOTES_FILE = REPORT_DIR / "figS7_plot_notes.md"

DPI = 600
MM = 1 / 25.4
MODEL_ORDER = ["deterministic", "mc_dropout"]
MODEL_LABELS = {
    "deterministic": "delta_base",
    "mc_dropout": "delta_mcd",
}
MODEL_TITLES = {
    "deterministic": r"$\delta_{\mathrm{base}}$",
    "mc_dropout": r"$\delta_{\mathrm{mcd}}$",
}


def ensure_dirs() -> None:
    for path in (DATA_DIR, REPORT_DIR, APPENDIX_FIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def fig06_attribute_order() -> list[str]:
    ordering = pd.read_csv(FIG06_ATTRIBUTE_ORDER_FILE)
    return ordering.sort_values("attribute_order")["attribute"].tolist()


def correlation_value(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    xy = pd.concat([x, y], axis=1).dropna()
    n = len(xy)
    if n < 3:
        return np.nan, np.nan, n
    x_val = xy.iloc[:, 0].to_numpy(dtype=float)
    y_val = xy.iloc[:, 1].to_numpy(dtype=float)
    if np.nanstd(x_val) == 0 or np.nanstd(y_val) == 0:
        return np.nan, np.nan, n
    result = spearmanr(x_val, y_val)
    return float(result.statistic), float(result.pvalue), n


def source_table_has_full_attribute_set(attributes: list[str]) -> bool:
    if not SEED_LOSS_CORR_FILE.exists():
        return False
    columns = pd.read_csv(SEED_LOSS_CORR_FILE, nrows=0).columns
    if not {"model_raw", "parameter", "attribute", "spearman_rho"}.issubset(columns):
        return False
    present = set(pd.read_csv(SEED_LOSS_CORR_FILE, usecols=["attribute"])["attribute"].unique())
    return set(attributes).issubset(present)


def load_from_seed_loss_table(attributes: list[str]) -> pd.DataFrame:
    raw = pd.read_csv(SEED_LOSS_CORR_FILE)
    raw = raw.loc[
        raw["model_raw"].isin(MODEL_ORDER)
        & raw["parameter"].isin(PARAMETER_ORDER)
        & raw["attribute"].isin(attributes)
    ].copy()
    grouped = (
        raw.groupby(["model_raw", "parameter", "attribute"], as_index=False)
        .agg(
            spearman_rho=("spearman_rho", "mean"),
            run_sd_rho=("spearman_rho", "std"),
            run_min_rho=("spearman_rho", "min"),
            run_max_rho=("spearman_rho", "max"),
            n_runs=("spearman_rho", "size"),
            n_basins=("n_basins", "median"),
        )
        .rename(columns={"model_raw": "model"})
    )
    grouped["source"] = "seed_loss_correlation_matrix_long"
    return grouped


def compute_full_matrix_from_parameter_means(attributes: list[str]) -> pd.DataFrame:
    params = pd.read_csv(
        PARAM_LONG_FILE,
        usecols=["basin_id", "model", "loss", "seed", "parameter", "mean"],
    )
    params = params.loc[
        params["model"].isin(MODEL_ORDER) & params["parameter"].isin(PARAMETER_ORDER)
    ].copy()
    attrs = pd.read_csv(BASIN_ATTRIBUTE_FILE, usecols=["basin_id", *attributes])
    merged = params.merge(attrs, on="basin_id", how="inner")

    rows: list[dict[str, object]] = []
    group_cols = ["model", "loss", "seed", "parameter"]
    for (model, loss, seed, parameter), sub in merged.groupby(group_cols, sort=False):
        for attribute in attributes:
            rho, p_value, n = correlation_value(sub["mean"], sub[attribute])
            rows.append(
                {
                    "model": model,
                    "loss": loss,
                    "seed": seed,
                    "parameter": parameter,
                    "attribute": attribute,
                    "run_spearman_rho": rho,
                    "run_p_value_optional": p_value,
                    "n_basins": n,
                }
            )

    run_level = pd.DataFrame(rows)
    grouped = (
        run_level.groupby(["model", "parameter", "attribute"], as_index=False)
        .agg(
            spearman_rho=("run_spearman_rho", "mean"),
            run_sd_rho=("run_spearman_rho", "std"),
            run_min_rho=("run_spearman_rho", "min"),
            run_max_rho=("run_spearman_rho", "max"),
            mean_p_value_optional=("run_p_value_optional", "mean"),
            n_runs=("run_spearman_rho", "count"),
            n_basins=("n_basins", "median"),
        )
    )
    grouped["source"] = "recomputed_from_params_long_clean_and_basin_attributes"
    return grouped


def load_relationship_matrix() -> pd.DataFrame:
    attributes = fig06_attribute_order()
    if source_table_has_full_attribute_set(attributes):
        corr = load_from_seed_loss_table(attributes)
        source_note = "seed_loss_correlation_matrix_long.csv"
    else:
        corr = compute_full_matrix_from_parameter_means(attributes)
        source_note = (
            "params_long_clean.csv + basin_attributes.csv; "
            "seed_loss_correlation_matrix_long.csv contains only the 18 core attributes"
        )

    attribute_order = {attribute: idx for idx, attribute in enumerate(attributes)}
    parameter_order = {parameter: idx for idx, parameter in enumerate(PARAMETER_ORDER)}
    corr["model_label"] = corr["model"].map(MODEL_LABELS)
    corr["parameter_label"] = corr["parameter"].map(clean_label)
    corr["parameter_group"] = corr["parameter"].map(PARAMETER_GROUPS)
    corr["attribute_group"] = corr["attribute"].map(classify_attribute)
    corr["attribute_group_short"] = corr["attribute_group"].map(ATTRIBUTE_GROUP_SHORT)
    corr["attribute_order"] = corr["attribute"].map(attribute_order)
    corr["parameter_order"] = corr["parameter"].map(parameter_order)
    corr["abs_rho"] = corr["spearman_rho"].abs()
    corr["strong_relationship"] = corr["abs_rho"].ge(STRONG_ABS_RHO)
    corr["dominant_relationship"] = corr.groupby(["model", "parameter"])["abs_rho"].rank(
        method="first", ascending=False
    ).eq(1)
    corr["strong_marker_rule"] = "abs_rho >= 0.5; no q_value available for model-specific full 35-attribute matrix"
    corr["stability_marker_rule"] = "not used; model-specific seed/loss stability was not recomputed"
    corr["source_note"] = source_note
    corr = corr.sort_values(["model", "parameter_order", "attribute_order"]).reset_index(drop=True)

    expected_rows = len(MODEL_ORDER) * len(PARAMETER_ORDER) * len(attributes)
    if len(corr) != expected_rows:
        raise ValueError(f"Expected {expected_rows} heatmap rows, found {len(corr)}")
    missing = corr[["spearman_rho", "attribute_order", "parameter_order"]].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"Heatmap data contains missing required values in {int(missing.sum())} rows")

    corr.to_csv(HEATMAP_DATA_FILE, index=False)
    return corr


def draw_circle_heatmap(
    ax: plt.Axes,
    panel: pd.DataFrame,
    *,
    panel_label: str,
    model_title: str,
) -> None:
    parameters = [param for param in PARAMETER_ORDER if param in set(panel["parameter"])]
    attributes = (
        panel[["attribute", "attribute_order"]]
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
    ax.set_xticklabels([ATTR_LABELS.get(a, a) for a in attributes], rotation=45, ha="right", rotation_mode="anchor", fontsize=6.2)
    ax.set_yticks(np.arange(len(parameters)))
    ax.set_yticklabels([clean_label(param) for param in parameters], rotation=0, ha="right", fontsize=7.6)
    ax.tick_params(axis="x", length=0, pad=2, colors="#111111")
    ax.tick_params(axis="y", length=0, pad=3, colors="#111111")
    ax.set_xticks(np.arange(-0.5, len(attributes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(parameters), 1), minor=True)
    ax.grid(which="minor", color="#D9D9D9", linewidth=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_edgecolor("#777777")

    lookup = panel.set_index(["attribute", "parameter"])
    xs: list[int] = []
    ys: list[int] = []
    values: list[float] = []
    sizes: list[float] = []
    for y, parameter in enumerate(parameters):
        for x, attribute in enumerate(attributes):
            row = lookup.loc[(attribute, parameter)]
            rho = float(row["spearman_rho"])
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

    draw_attribute_group_guides(ax, attributes, y_top=-0.30)
    ax.text(
        0.0,
        1.08,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.5,
        color="#111111",
        clip_on=False,
    )
    ax.text(
        0.5,
        1.08,
        model_title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#111111",
        clip_on=False,
    )


def make_figure(corr: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(335 * MM, 118 * MM), constrained_layout=False)
    gs = GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.12],
        wspace=0.08,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    side_ax = fig.add_subplot(gs[0, 2])
    side_ax.set_axis_off()
    cax = side_ax.inset_axes([0.02, 0.48, 0.20, 0.46])

    for ax, model, panel_label in zip(axes, MODEL_ORDER, ["(a)", "(b)"]):
        draw_circle_heatmap(
            ax,
            corr.loc[corr["model"].eq(model)].copy(),
            panel_label=panel_label,
            model_title=MODEL_TITLES[model],
        )

    sm = mpl.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), cmap=correlation_cmap())
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label(r"Spearman $\rho$", fontsize=8.0)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=7.0, width=0.45, length=2.0, colors="#222222")
    cbar.outline.set_linewidth(0.5)

    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#BDBDBD",
            markeredgecolor="#F7F7F7",
            markersize=np.sqrt(5.0 + 132.0 * (value**1.55)) / 1.65,
            label=f"{value:.2f}",
        )
        for value in (0.25, 0.50, 0.75)
    ]
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#111111",
            markeredgecolor="#111111",
            markersize=4.0,
            label=r"strong: $|\rho|\geq0.5$",
        ),
        Patch(facecolor="none", edgecolor="#111111", linewidth=1.35, label="dominant"),
    ]
    side_ax.legend(
        handles=size_handles + marker_handles,
        labels=[r"$|\rho|$ 0.25", r"$|\rho|$ 0.50", r"$|\rho|$ 0.75", "strong", "dominant"],
        loc="lower left",
        bbox_to_anchor=(0.0, 0.02),
        ncol=1,
        frameon=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        handlelength=1.2,
        handletextpad=0.45,
        columnspacing=1.05,
        labelspacing=0.3,
        fontsize=6.6,
    )

    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.27, top=0.84)
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PDF, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_notes(corr: pd.DataFrame) -> None:
    attributes = fig06_attribute_order()
    source_note = corr["source_note"].dropna().iloc[0]
    lines = [
        "# Fig. S7 baseline mean-attribute matrix notes",
        "",
        "1. Fig. S7 is the baseline supplement to main-text Fig. 6. It repeats the Fig. 6a circle-heatmap encoding for `delta_base` and `delta_mcd` only.",
        "2. Circle size represents `abs_rho`, the absolute value of the model-level mean Spearman rho.",
        "3. Circle color represents signed Spearman rho, using the same purple-white-green diverging scale and fixed range [-1, 1] as Fig. 6.",
        "4. The dominant cell is the single attribute with the largest `abs_rho` within each model x parameter row; it is marked with a black square outline.",
        f"5. A strong relationship is marked with a black dot when `abs_rho >= {STRONG_ABS_RHO}`. No model-specific full-matrix `q_value` is available, so the marker does not apply an FDR threshold.",
        "6. No stable strong marker is shown. Model-specific seed/loss stability was not recomputed for the full 35-attribute baseline matrices, so the Fig. 6 star marker is intentionally omitted to avoid confusing these baseline panels with the distributional stability analysis.",
        "",
        "## Inputs and ordering",
        "",
        f"- Full 35-attribute ordering is read from `{FIG06_ATTRIBUTE_ORDER_FILE}`.",
        f"- Run-level correlations are recomputed from `{PARAM_LONG_FILE}` and `{BASIN_ATTRIBUTE_FILE}` when the preferred long correlation table is incomplete.",
        f"- Source used for this run: {source_note}.",
        f"- The preferred table `{SEED_LOSS_CORR_FILE}` currently contains {pd.read_csv(SEED_LOSS_CORR_FILE, usecols=['attribute'])['attribute'].nunique()} attributes; Fig. 6/S7 require {len(attributes)} attributes.",
        "",
        "## Output checks",
        "",
        "- Panels included: `(a) delta_base`, `(b) delta_mcd`.",
        "- No Fig. 6b/c top-k or group-summary panels are included.",
        f"- Heatmap shape per model: {len(PARAMETER_ORDER)} parameters x {len(attributes)} attributes.",
        f"- Heatmap data: `{HEATMAP_DATA_FILE}`.",
        f"- PNG: `{OUT_PNG}`.",
        f"- PDF: `{OUT_PDF}`.",
    ]
    NOTES_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    setup_style()
    ensure_dirs()
    corr = load_relationship_matrix()
    make_figure(corr)
    write_notes(corr)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {HEATMAP_DATA_FILE}")
    print(f"Wrote {NOTES_FILE}")


if __name__ == "__main__":
    main()
