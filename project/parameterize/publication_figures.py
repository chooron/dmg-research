from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

from project.parameterize.manuscript.plots.common import (
    APP_FIG_DIR,
    ANALYSIS_ROOT,
    ATTR_LABELS,
    CLASS_COLORS,
    CORR_ROOT,
    DPI,
    FOCUS_PARAMS,
    FigureSpec,
    KEY_ATTRS,
    MAIN_FIG_DIR,
    MANUSCRIPT_ROOT,
    MM,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    PARAM_ORDER,
    PARAM_ROOT,
    REPORT_DIR,
    ROOT,
    SHORT_ATTR_LABELS,
    TABLE_ROOT,
    VAR_ROOT,
    a_label,
    add_panel_label,
    clean_axes,
    figure,
    heatmap,
    math_model_labels,
    metric_summary,
    muted_diverging,
    muted_seq,
    ordered,
    p_label,
    pair_label,
    pair_label_from_key,
    parameter_family,
    read_csv,
    save,
    setup_style,
    shorten_heatmap_xticks,
    wrap_labels,
)
try:
    from project.parameterize.manuscript.plots.fig01_predictive_performance import draw as fig01
except ModuleNotFoundError:
    from project.parameterize.manuscript.plots.plot_fig01_predictive_performance import draw as fig01

try:
    from project.parameterize.manuscript.plots.fig02_cross_seed_parameter_stability import draw as fig02
except ModuleNotFoundError:
    from project.parameterize.manuscript.plots.plot_fig02_parameter_stability import main as fig02


logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def muted_diverging() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "purple_white_green",
        ["#6A3D9A", "#F7F7F7", "#1B9E77"],
        N=256,
    )


def muted_seq() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("muted_teal", ["#F4F7F7", "#A8C6C0", "#2A9D8F"], N=256)


def read_csv(name: str, root: Path = CORR_ROOT) -> pd.DataFrame:
    return pd.read_csv(root / name)


def p_label(parameter: str) -> str:
    return str(parameter).replace("par", "")


def a_label(attribute: str) -> str:
    return ATTR_LABELS.get(str(attribute), str(attribute).replace("_", " "))


def pair_label(parameter: str, attribute: str) -> str:
    return f"{p_label(parameter)} - {a_label(attribute)}"


def pair_label_from_key(pair_key: str) -> str:
    if "__" not in str(pair_key):
        return str(pair_key).replace("par", "")
    parameter, attribute = str(pair_key).split("__", 1)
    return pair_label(parameter, attribute)


def ordered(items: list[str], preferred: list[str]) -> list[str]:
    seen = [x for x in preferred if x in set(items)]
    seen.extend([x for x in items if x not in seen])
    return seen


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.08,
    y: float = 1.05,
    *,
    ha: str = "left",
    va: str = "bottom",
    fontweight: str = "bold",
    fontsize: float = 13.5,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight=fontweight,
        color="#111111",
    )


def clean_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color="#E6E6E6", linewidth=0.55)
        ax.set_axisbelow(True)


def wrap_labels(labels: list[str], width: int = 14) -> list[str]:
    return ["\n".join(wrap(str(label), width=width, break_long_words=False)) for label in labels]


def figure(spec: FigureSpec, *, rows: int = 1, cols: int = 1, **kwargs):
    figsize = (spec.width_mm * MM, spec.height_mm * MM)
    return plt.subplots(rows, cols, figsize=figsize, constrained_layout=True, **kwargs)


def save(fig: plt.Figure, spec: FigureSpec) -> None:
    spec.directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(spec.directory / f"{spec.stem}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def horizontal_bars(ax: plt.Axes, data: pd.DataFrame, value: str, label: str, color: str, xlab: str) -> None:
    d = data.sort_values(value)
    ax.barh(np.arange(len(d)), d[value], color=color, alpha=0.9, height=0.66)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d[label])
    ax.set_xlabel(xlab)
    clean_axes(ax, "x")


def heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    cmap,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    cbar_label: str = "",
    text: bool = False,
    fmt: str = "{:.2f}",
) -> None:
    values = matrix.to_numpy(dtype=float)
    norm = None
    if center is not None:
        bound = np.nanmax(np.abs(values)) if vmin is None or vmax is None else max(abs(vmin), abs(vmax))
        norm = TwoSlopeNorm(vmin=-bound, vcenter=center, vmax=bound)
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(wrap_labels([a_label(c) for c in matrix.columns], 11), rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([p_label(r) for r in matrix.index])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if text:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=8.6, color="#222222")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.ax.tick_params(labelsize=8.8, length=2)
    cbar.set_label(cbar_label, fontsize=9.2)


def shorten_heatmap_xticks(ax: plt.Axes, labels: list[str], *, rotation: float = 50, fontsize: float = 8.9) -> None:
    ax.set_xticklabels(labels, rotation=rotation, ha="right", fontsize=fontsize)


def parameter_family(parameter: str) -> str:
    p = str(parameter)
    if p in {"parBETA", "parFC", "parLP"}:
        return "Soil storage"
    if p in {"parPERC", "parUZL", "parK0", "parK1", "parK2"}:
        return "Runoff response"
    if p in {"parCFMAX", "parTT", "parCFR", "parCWH"}:
        return "Snow/cold process"
    return "Routing"


def metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    out = (
        metrics.groupby("model", as_index=False)
        .agg(nse_median=("nse", "median"), kge_median=("kge", "median"), bias_abs_median=("bias_abs", "median"))
        .set_index("model")
        .loc[MODEL_ORDER]
    )
    return out


def fig03() -> None:
    df = pd.read_csv(VAR_ROOT / "seed_parameter_variance_by_parameter.csv")
    agg = df.groupby(["parameter", "model"], as_index=False)["mean_abs_seed_diff"].median()
    params = ordered(list(agg["parameter"].unique()), PARAM_ORDER)
    mat = agg.pivot(index="parameter", columns="model", values="mean_abs_seed_diff").reindex(params)[MODEL_ORDER]
    fam = agg.assign(family=agg["parameter"].map(parameter_family)).groupby(["family", "model"], as_index=False)[
        "mean_abs_seed_diff"
    ].median()
    spec = FigureSpec("Fig03_cross_seed_parameter_stability", MAIN_FIG_DIR, 180, 115)
    fig, axes = figure(spec, rows=1, cols=2, gridspec_kw={"width_ratios": [1.2, 1.0]})
    heatmap(
        axes[0],
        mat.rename(columns=MODEL_LABELS),
        cmap=muted_seq(),
        cbar_label="Median abs. seed diff.",
        text=False,
    )
    axes[0].set_title("Overall cross-seed parameter stability")
    add_panel_label(axes[0], "A")
    ax = axes[1]
    families = ["Soil storage", "Runoff response", "Snow/cold process", "Routing"]
    x = np.arange(len(families))
    width = 0.24
    for i, model in enumerate(MODEL_ORDER):
        values = [fam.loc[(fam.family == f) & (fam.model == model), "mean_abs_seed_diff"].median() for f in families]
        ax.bar(x + (i - 1) * width, values, width=width, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(wrap_labels(families, 12), rotation=0)
    ax.set_ylabel("Median abs. seed diff.")
    ax.set_title("Parameter-class summary")
    ax.legend(frameon=False, loc="upper left")
    add_panel_label(ax, "B")
    clean_axes(ax, "y")
    save(fig, spec)


def fig04() -> None:
    summary = read_csv("correlation_seed_stability_summary.csv")
    summary = summary.loc[summary["method"].eq("kendall")].copy()
    pairs = read_csv("relationship_classes.csv")
    seed_pairs = read_csv("pair_seed_stability.csv")

    spec = FigureSpec("Fig04_cross_seed_relationship_stability", MAIN_FIG_DIR, 185, 118)
    fig, axes = figure(spec, rows=1, cols=3, gridspec_kw={"width_ratios": [0.95, 1.15, 1.15]})

    ax = axes[0]
    d = summary.groupby("model", as_index=False)[["mean_variance_corr", "mean_range_corr"]].median().set_index("model").loc[MODEL_ORDER]
    x = np.arange(len(MODEL_ORDER))
    ax.bar(x - 0.15, d["mean_variance_corr"], width=0.3, color="#8AA6B8", label="Variance")
    ax.bar(x + 0.15, d["mean_range_corr"], width=0.3, color="#B55A4A", label="Range")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Mean across attribute-parameter pairs")
    ax.set_title("Cross-seed correlation spread")
    ax.legend(frameon=False, loc="upper right", fontsize=8.8)
    add_panel_label(ax, "A")
    clean_axes(ax, "y")

    ax = axes[1]
    sel = (
        pairs.loc[pairs["model"].eq("distributional")]
        .sort_values(["seed_range", "mean_abs_corr"], ascending=[True, False])
        .head(6)[["parameter", "attribute"]]
    )
    plot_rows = []
    for _, row in sel.iterrows():
        for model in MODEL_ORDER:
            sub = seed_pairs.loc[
                (seed_pairs["model"] == model)
                & (seed_pairs["parameter"] == row.parameter)
                & (seed_pairs["attribute"] == row.attribute)
            ]
            if len(sub):
                plot_rows.append(
                    {
                        "pair": pair_label(row.parameter, row.attribute),
                        "model": model,
                        "range": sub["seed_range_rho"].median(),
                    }
                )
    plot = pd.DataFrame(plot_rows)
    labels = list(sel.apply(lambda r: pair_label(r.parameter, r.attribute), axis=1))
    y = np.arange(len(labels))
    for i, model in enumerate(MODEL_ORDER):
        vals = [plot.loc[(plot.pair == lab) & (plot.model == model), "range"].median() for lab in labels]
        ax.scatter(vals, y + (i - 1) * 0.18, s=34, color=MODEL_COLORS[model], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(labels, 17))
    ax.invert_yaxis()
    ax.set_xlabel("Seed range in correlation")
    ax.set_title("Selected robust pairs")
    add_panel_label(ax, "B")
    clean_axes(ax, "x")

    ax = axes[2]
    cons = seed_pairs.groupby("model", as_index=False)[["sign_consistency_seed", "topk_rate_seed", "dominant_rate_seed"]].mean()
    cons = cons.set_index("model").loc[MODEL_ORDER].rename(index=MODEL_LABELS)
    heatmap(
        ax,
        cons.rename(columns={"sign_consistency_seed": "Sign", "topk_rate_seed": "Top-k", "dominant_rate_seed": "Dominant"}),
        cmap=muted_seq(),
        vmin=0,
        vmax=1,
        cbar_label="Mean consistency",
        text=True,
    )
    ax.set_title("Seed-consistency summary")
    add_panel_label(ax, "C")
    save(fig, spec)


def fig05() -> None:
    summary = read_csv("correlation_loss_stability_summary.csv")
    summary = summary.loc[summary["method"].eq("kendall")].set_index("model").loc[MODEL_ORDER]
    focus = read_csv("focused_pair_loss_stability.csv")
    loss_pairs = read_csv("pair_loss_stability.csv")

    spec = FigureSpec("Fig05_cross_loss_relationship_stability", MAIN_FIG_DIR, 185, 118)
    fig, axes = figure(spec, rows=1, cols=3, gridspec_kw={"width_ratios": [0.95, 1.15, 1.15]})

    ax = axes[0]
    x = np.arange(len(MODEL_ORDER))
    ax.bar(x - 0.15, summary["mean_pooled_variance_corr"], width=0.3, color="#8AA6B8", label="Variance")
    ax.bar(x + 0.15, summary["mean_pooled_range_corr"], width=0.3, color="#B55A4A", label="Range")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Mean across attribute-parameter pairs")
    ax.set_title("Cross-loss correlation spread")
    ax.legend(frameon=False, loc="upper right", fontsize=8.8)
    add_panel_label(ax, "A")
    clean_axes(ax, "y")

    ax = axes[1]
    d = focus.loc[focus["method"].eq("kendall")].drop_duplicates(["pair_label", "model"])
    selected_pairs = (
        d.loc[d["model"].eq("distributional")]
        .sort_values(["cross_loss_range", "abs_mean_rho"], ascending=[True, False])
        .head(6)["pair_label"]
        .tolist()
    )
    labels = [pair_label_from_key(pair) for pair in selected_pairs]
    y = np.arange(len(selected_pairs))
    for i, model in enumerate(MODEL_ORDER):
        vals = [d.loc[(d.pair_label == pair) & (d.model == model), "cross_loss_range"].median() for pair in selected_pairs]
        ax.scatter(vals, y + (i - 1) * 0.18, s=34, color=MODEL_COLORS[model])
    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(labels, 17))
    ax.invert_yaxis()
    ax.set_xlabel("Cross-loss range in correlation")
    ax.set_title("Focused robust pairs")
    add_panel_label(ax, "B")
    clean_axes(ax, "x")

    ax = axes[2]
    cons = loss_pairs.groupby("model", as_index=False)[["sign_consistency_loss", "topk_rate_loss", "dominant_rate_loss"]].mean()
    cons = cons.set_index("model").loc[MODEL_ORDER].rename(index=MODEL_LABELS)
    heatmap(
        ax,
        cons.rename(columns={"sign_consistency_loss": "Sign", "topk_rate_loss": "Top-k", "dominant_rate_loss": "Dominant"}),
        cmap=muted_seq(),
        vmin=0,
        vmax=1,
        cbar_label="Mean consistency",
        text=True,
    )
    ax.set_title("Loss-consistency summary")
    add_panel_label(ax, "C")
    save(fig, spec)


def fig06() -> None:
    df = read_csv("results331_dominant_attribute_summary.csv")
    class_order = ["shared dominant controls", "partially shared controls", "model-sensitive controls"]
    df["class_order"] = df["overall_relationship_class"].map({c: i for i, c in enumerate(class_order)})
    df = df.sort_values(["class_order", "parameter"])
    spec = FigureSpec("Fig06_shared_dominant_relationships", MAIN_FIG_DIR, 185, 125)
    fig, axes = figure(spec, rows=1, cols=3, gridspec_kw={"width_ratios": [1.15, 1.15, 0.65]})

    ax = axes[0]
    ax.axis("off")
    params = df["parameter"].tolist()
    attrs = ordered(sorted(set(df["distributional_attribute"]) | set(df["deterministic_attribute"]) | set(df["mc_dropout_attribute"])), KEY_ATTRS)
    y_params = np.linspace(0.92, 0.08, len(params))
    y_attrs = np.linspace(0.88, 0.12, len(attrs))
    attr_y = dict(zip(attrs, y_attrs))
    for y, (_, row) in zip(y_params, df.iterrows()):
        color = CLASS_COLORS[row["overall_relationship_class"]]
        ax.text(0.04, y, p_label(row["parameter"]), ha="left", va="center", fontsize=9.0, color="#111111")
        ax.scatter([0.27], [y], s=42, color=color, zorder=3)
        target = row["distributional_attribute"]
        arrow = FancyArrowPatch(
            (0.29, y),
            (0.72, attr_y[target]),
            arrowstyle="-",
            mutation_scale=1,
            lw=0.9,
            color=color,
            alpha=0.8,
            connectionstyle="arc3,rad=0.05",
        )
        ax.add_patch(arrow)
    for attr, y in attr_y.items():
        ax.scatter([0.75], [y], s=42, color="#E7ECEF", edgecolor="#666666", lw=0.6)
        ax.text(0.79, y, a_label(attr), ha="left", va="center", fontsize=8.8)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    ax.set_title("Simplified dominant-control map")
    add_panel_label(ax, "A", x=0.0)

    ax = axes[1]
    status = df[["parameter", "overall_relationship_class"]].copy()
    status["score"] = status["overall_relationship_class"].map(
        {"model-sensitive controls": 0, "partially shared controls": 1, "shared dominant controls": 2}
    )
    smat = status.set_index("parameter")[["score"]].reindex(df["parameter"])
    class_cmap = LinearSegmentedColormap.from_list("classes", ["#B55A4A", "#8AA6B8", "#2A9D8F"], N=3)
    ax.imshow(smat.to_numpy(), aspect="auto", cmap=class_cmap, vmin=0, vmax=2)
    ax.set_xticks([0])
    ax.set_xticklabels(["Class"])
    ax.set_yticks(np.arange(len(smat.index)))
    ax.set_yticklabels([p_label(p) for p in smat.index])
    ax.tick_params(length=0)
    for i, (_, row) in enumerate(df.iterrows()):
        short = {"shared dominant controls": "Shared", "partially shared controls": "Partial", "model-sensitive controls": "Sensitive"}[
            row["overall_relationship_class"]
        ]
        attr = a_label(row["distributional_attribute"])
        ax.text(0, i, f"{short}\n{attr}", ha="center", va="center", fontsize=7.8, color="#111111")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Dominant-control class by parameter")
    add_panel_label(ax, "B", x=0.0)

    ax = axes[2]
    counts = df["overall_relationship_class"].value_counts().reindex(class_order)
    ax.barh(np.arange(len(class_order)), counts.values, color=[CLASS_COLORS[c] for c in class_order], height=0.55)
    ax.set_yticks(np.arange(len(class_order)))
    ax.set_yticklabels(["Shared", "Partial", "Sensitive"])
    ax.set_xlabel("Count")
    ax.set_xlim(0, max(counts.values) + 1)
    for i, v in enumerate(counts.values):
        ax.text(v + 0.15, i, str(int(v)), va="center", fontsize=11, fontweight="bold")
    ax.set_title("7 / 4 / 3 structure")
    add_panel_label(ax, "C")
    clean_axes(ax, "x")
    save(fig, spec)


def fig07() -> None:
    sim = read_csv("results332_matrix_similarity.csv")
    emb = read_csv("results332_matrix_embedding.csv")
    spec = FigureSpec("Fig07_matrix_similarity", MAIN_FIG_DIR, 185, 122)
    fig, axes = figure(spec, rows=1, cols=3, gridspec_kw={"width_ratios": [0.9, 0.95, 0.95]})

    ax = axes[0]
    group = sim.groupby(["model_a", "model_b"], as_index=False)["matrix_corr_spearman"].mean()
    mat = group.pivot(index="model_a", columns="model_b", values="matrix_corr_spearman").reindex(MODEL_ORDER)[MODEL_ORDER]
    mat = mat.rename(index=MODEL_LABELS, columns=MODEL_LABELS)
    heatmap(ax, mat, cmap=muted_seq(), vmin=0.65, vmax=1.0, cbar_label="Mean matrix similarity", text=True)
    ax.set_title("Model-level similarity")
    add_panel_label(ax, "A")

    ax = axes[1]
    for model in MODEL_ORDER:
        d = emb.loc[emb["model"].eq(model)]
        ax.scatter(d["mds_x"], d["mds_y"], s=38, color=MODEL_COLORS[model], label=MODEL_LABELS[model], alpha=0.85)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title("Run embedding")
    ax.legend(frameon=False, loc="upper right", fontsize=8.8)
    add_panel_label(ax, "B")
    clean_axes(ax)

    ax = axes[2]
    ax.axis("off")
    within = (
        sim.loc[sim["same_model"] & (sim["run_id_a"] != sim["run_id_b"])]
        .groupby("model_a")["matrix_corr_spearman"]
        .median()
        .reindex(MODEL_ORDER)
    )
    rows = [["Model", "Within-model\nmedian"]]
    rows.extend([[MODEL_LABELS[m], f"{within.loc[m]:.2f}"] for m in MODEL_ORDER])
    table = ax.table(cellText=rows, loc="center", cellLoc="center", colWidths=[0.58, 0.42])
    table.auto_set_font_size(False)
    table.set_fontsize(9.4)
    table.scale(0.9, 1.45)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.6)
        if key[0] == 0:
            cell.set_facecolor("#F0F3F4")
            cell.set_text_props(fontweight="bold")
    ax.set_title("Compactness summary")
    add_panel_label(ax, "C", x=0.0)
    save(fig, spec)


def fig08() -> None:
    align = read_csv("results333_importance_alignment.csv")
    overlap = read_csv("results333_importance_overlap.csv")
    imp = read_csv("results333_parameter_feature_importance.csv")
    selected = FOCUS_PARAMS
    spec = FigureSpec("Fig08_explainability_support", MAIN_FIG_DIR, 185, 112)
    fig, axes = figure(spec, rows=1, cols=3, gridspec_kw={"width_ratios": [1.0, 0.78, 1.2]})

    ax = axes[0]
    d = align.loc[align["parameter"].isin(selected)]
    labels = [p_label(p) for p in selected]
    x = np.arange(len(labels))
    width = 0.24
    for i, model in enumerate(MODEL_ORDER):
        vals = [d.loc[(d.parameter == p) & (d.model == model), "dominant_attribute_top3_rate"].median() for p in selected]
        ax.bar(x + (i - 1) * width, vals, width=width, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Top-3 agreement rate")
    ax.set_title("Agreement summary")
    ax.legend(frameon=False, ncol=1, loc="lower left", fontsize=8.6)
    add_panel_label(ax, "A")
    clean_axes(ax, "y")

    ax = axes[1]
    ov = overlap.loc[overlap["parameter"].isin(selected)].groupby("parameter", as_index=False)["jaccard_overlap"].mean()
    ax.bar(np.arange(len(ov)), ov["jaccard_overlap"], color="#8AA6B8", width=0.6)
    ax.set_xticks(np.arange(len(ov)))
    ax.set_xticklabels([p_label(p) for p in ov["parameter"]])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean Jaccard")
    ax.set_title("Feature overlap")
    add_panel_label(ax, "B")
    clean_axes(ax, "y")

    ax = axes[2]
    rep_params = ["parFC", "parUZL"]
    d = imp.loc[(imp["model"].eq("distributional")) & (imp["parameter"].isin(rep_params))]
    d = d.groupby(["parameter", "attribute"], as_index=False)["mean_importance"].mean()
    bars = []
    for p in rep_params:
        bars.append(d.loc[d.parameter.eq(p)].nlargest(4, "mean_importance"))
    plot = pd.concat(bars)
    labels = [f"{p_label(r.parameter)}: {a_label(r.attribute)}" for r in plot.itertuples()]
    ax.barh(np.arange(len(plot)), plot["mean_importance"], color="#2A9D8F", alpha=0.85)
    ax.set_yticks(np.arange(len(plot)))
    ax.set_yticklabels(wrap_labels(labels, 20))
    ax.invert_yaxis()
    ax.set_xlabel("Mean importance")
    ax.set_title("Representative attribute importance")
    add_panel_label(ax, "C")
    clean_axes(ax, "x")
    save(fig, spec)


def fig09() -> None:
    rel = read_csv("results341_distributional_mean_relationships.csv")
    params = ordered(sorted(rel["parameter"].unique()), PARAM_ORDER)
    attrs = ordered(sorted(rel["attribute"].unique()), KEY_ATTRS)
    mat = rel.pivot(index="parameter", columns="attribute", values="mean_spearman_corr").reindex(params)[attrs]
    spec = FigureSpec("Fig09_parameter_mean_gradients", MAIN_FIG_DIR, 185, 124)
    fig, axes = figure(spec, rows=1, cols=3, gridspec_kw={"width_ratios": [1.35, 0.65, 0.58]})
    heatmap(axes[0], mat, cmap=muted_diverging(), center=0, cbar_label="Spearman rho")
    shorten_heatmap_xticks(axes[0], [SHORT_ATTR_LABELS.get(c, a_label(c)) for c in mat.columns], rotation=50)
    axes[0].set_title("Distributional parameter-mean gradients")
    add_panel_label(axes[0], "A")

    ax = axes[1]
    flag = rel.assign(score=rel["relationship_role"].map({"dominant": 2, "supportive": 1, "secondary": 0}).fillna(0))
    fmat = flag.pivot(index="parameter", columns="attribute", values="score").reindex(params)[attrs]
    role_cmap = LinearSegmentedColormap.from_list("role", ["#E7E8EA", "#8AA6B8", "#2A9D8F"], N=3)
    heatmap(ax, fmat, cmap=role_cmap, vmin=0, vmax=2, cbar_label="Role score")
    shorten_heatmap_xticks(ax, [SHORT_ATTR_LABELS.get(c, a_label(c)) for c in fmat.columns], rotation=55, fontsize=8.4)
    ax.set_title("Dominant/support flags")
    add_panel_label(ax, "B")

    ax = axes[2]
    counts = rel["relationship_role"].value_counts().reindex(["dominant", "supportive", "secondary"]).fillna(0)
    ax.barh(np.arange(len(counts)), counts.values, color=["#2A9D8F", "#8AA6B8", "#D0D4D8"], height=0.55)
    ax.set_yticks(np.arange(len(counts)))
    ax.set_yticklabels(["Dominant", "Supportive", "Secondary"])
    ax.set_xlabel("Count")
    ax.set_title("Support summary")
    for i, v in enumerate(counts.values):
        ax.text(v + 0.4, i, str(int(v)), va="center", fontweight="bold")
    add_panel_label(ax, "C")
    clean_axes(ax, "x")
    save(fig, spec)


def fig10() -> None:
    stats = read_csv("results341_gradient_group_stats.csv")
    group_summary = read_csv("results343_basin_group_summary.csv")
    spec = FigureSpec("Fig10_representative_mean_gradients", MAIN_FIG_DIR, 185, 125)
    fig, axes = figure(spec, rows=2, cols=2, gridspec_kw={"height_ratios": [1.0, 0.95]})
    axes = axes.ravel()
    gradients = [("aridity", ["parFC", "parPERC"]), ("frac_snow", ["parCWH", "parCFR"]), ("slope_mean", ["parBETA", "parUZL"])]
    for ax, (attr, params), panel in zip(axes[:3], gradients, ["A", "B", "C"]):
        d = stats.loc[(stats["gradient_attribute"].eq(attr)) & (stats["parameter"].isin(params))]
        x = np.arange(3)
        groups = ["low", "mid", "high"]
        for i, p in enumerate(params):
            vals = [d.loc[(d.parameter == p) & (d.gradient_group == g), "median_parameter_unit"].median() for g in groups]
            ax.plot(x, vals, marker="o", lw=1.3, color=MODEL_COLORS[MODEL_ORDER[i]], label=p_label(p))
        ax.set_xticks(x)
        ax.set_xticklabels(["Low", "Mid", "High"])
        ax.set_ylabel("Median unit value")
        ax.set_title(f"{a_label(attr)} gradient")
        ax.legend(frameon=False)
        add_panel_label(ax, panel)
        clean_axes(ax, "y")
    ax = axes[3]
    d = group_summary.pivot(index="parameter", columns="group_label", values="mean_median_vs_global")
    cols = [c for c in ["aridity_low", "aridity_high", "frac_snow_low", "frac_snow_high", "slope_mean_low", "slope_mean_high"] if c in d.columns]
    d = d.reindex([p for p in PARAM_ORDER if p in d.index])[cols]
    d.columns = [c.replace("_", "\n") for c in d.columns]
    heatmap(ax, d, cmap=muted_diverging(), center=0, cbar_label="Median vs global")
    ax.set_title("Archetype summary")
    add_panel_label(ax, "D")
    save(fig, spec)


def fig11() -> None:
    rel = read_csv("results342_distributional_std_relationships.csv")
    plot_data = read_csv("results342_std_plot_data.csv")
    flags = read_csv("finalcheck_uncertainty_interpretation_flags.csv")
    params = ordered(sorted(rel["parameter"].unique()), PARAM_ORDER)
    attrs = ordered(sorted(rel["attribute"].unique()), KEY_ATTRS)
    mat = rel.pivot(index="parameter", columns="attribute", values="mean_spearman_corr").reindex(params)[attrs]
    spec = FigureSpec("Fig11_parameter_uncertainty_gradients", MAIN_FIG_DIR, 190, 125)
    fig, axes = figure(spec, rows=1, cols=3, gridspec_kw={"width_ratios": [1.28, 1.08, 0.72]})
    heatmap(axes[0], mat, cmap=muted_diverging(), center=0, cbar_label="Spearman rho")
    shorten_heatmap_xticks(axes[0], [SHORT_ATTR_LABELS.get(c, a_label(c)) for c in mat.columns], rotation=50)
    axes[0].set_title("Parameter uncertainty-attribute gradients")
    add_panel_label(axes[0], "A")

    ax = axes[1]
    selected = [("parCWH", "frac_snow"), ("parPERC", "aridity"), ("parUZL", "slope_mean")]
    positions = []
    labels = []
    data = []
    pos = 0
    colors = []
    for p, attr in selected:
        for group in ["low", "mid", "high"]:
            vals = plot_data.loc[
                (plot_data["parameter"].eq(p))
                & (plot_data["gradient_attribute"].eq(attr))
                & (plot_data["gradient_group"].eq(group)),
                "std_unit",
            ].dropna()
            data.append(vals.to_numpy())
            positions.append(pos)
            labels.append(group[0].upper())
            colors.append("#E7ECEF" if group == "mid" else "#8AA6B8" if group == "low" else "#2A9D8F")
            pos += 1
        pos += 0.8
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False, medianprops={"color": "#222222", "lw": 0.9})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#555555")
        patch.set_linewidth(0.65)
    ax.set_xticks([1, 4.8, 8.6])
    ax.set_xticklabels([pair_label(*s) for s in selected], rotation=18, ha="right")
    ax.set_ylabel("Parameter std. (unit)")
    ax.set_title("Selected gradients")
    add_panel_label(ax, "B")
    clean_axes(ax, "y")

    ax = axes[2]
    flag_counts = flags["interpretation_flag"].value_counts()
    short_names = {
        "boundary-sensitive / interpret with caution": "Boundary\ncaution",
        "possibly mean-coupled": "Mean-coupled",
    }
    labels = [short_names.get(k, k) for k in flag_counts.index]
    colors = ["#6A3D9A" if "boundary" in k else "#8AA6B8" for k in flag_counts.index]
    ax.barh(np.arange(len(flag_counts)), flag_counts.values, color=colors, height=0.5)
    ax.set_yticks(np.arange(len(flag_counts)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Flagged pairs")
    for i, v in enumerate(flag_counts.values):
        ax.text(v + 0.08, i, str(int(v)), va="center", fontweight="bold")
    ax.set_xlim(0, max(flag_counts.values) + 0.7)
    ax.set_title("Interpretation flags")
    add_panel_label(ax, "C", x=0.0)
    clean_axes(ax, "x")
    save(fig, spec)


def fig12() -> None:
    spec = FigureSpec("Fig12_synthesis", MAIN_FIG_DIR, 180, 95)
    fig, ax = plt.subplots(figsize=(spec.width_mm * MM, spec.height_mm * MM), constrained_layout=True)
    ax.axis("off")
    columns = [
        ("Formulation", ["Deterministic", "MC dropout", "Distributional"], [MODEL_COLORS[m] for m in MODEL_ORDER]),
        ("Reliability / structure / gradients", ["Cross-seed stability", "Shared dominant controls", "Mean + uncertainty gradients"], ["#8AA6B8", "#009E73", "#6A3D9A"]),
        ("Interpretation", ["Robust relationships", "Partial/model-sensitive controls", "Caution for boundary-coupled uncertainty"], ["#009E73", "#8AA6B8", "#6A3D9A"]),
    ]
    x_positions = [0.12, 0.5, 0.86]
    for x, (title, items, colors) in zip(x_positions, columns):
        ax.text(x, 0.86, title, ha="center", va="center", fontsize=11.5, fontweight="bold")
        for i, (item, color) in enumerate(zip(items, colors)):
            y = 0.64 - i * 0.18
            ax.add_patch(Rectangle((x - 0.13, y - 0.045), 0.26, 0.09, facecolor="#F7F7F7", edgecolor=color, linewidth=1.0))
            ax.text(x, y, item, ha="center", va="center", fontsize=9.3, color="#222222")
    for x0, x1 in zip(x_positions[:-1], x_positions[1:]):
        ax.add_patch(FancyArrowPatch((x0 + 0.15, 0.5), (x1 - 0.15, 0.5), arrowstyle="->", mutation_scale=12, lw=0.9, color="#555555"))
    ax.text(0.5, 0.12, "Main claim: distributional parameter learning improves relationship reliability while preserving interpretable hydrologic structure.", ha="center", va="center", fontsize=10.2)
    save(fig, spec)


def fig_a01() -> None:
    metrics = pd.read_csv(TABLE_ROOT / "metrics_long.csv")
    spec = FigureSpec("FigA01_full_predictive_metrics", APP_FIG_DIR, 180, 125)
    fig, axes = figure(spec, rows=1, cols=3)
    for ax, metric, panel in zip(axes, ["nse", "kge", "bias_abs"], ["A", "B", "C"]):
        data = [metrics.loc[metrics.model.eq(m), metric].replace([np.inf, -np.inf], np.nan).dropna().clip(-2, 2 if metric != "bias_abs" else None).to_numpy() for m in MODEL_ORDER]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, medianprops={"color": "#222222", "lw": 0.9})
        for patch, model in zip(bp["boxes"], MODEL_ORDER):
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.75)
            patch.set_edgecolor("#555555")
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=20, ha="right")
        ax.set_ylabel(metric.upper() if metric != "bias_abs" else "|bias|")
        ax.set_title(metric.upper() if metric != "bias_abs" else "Absolute bias")
        add_panel_label(ax, panel)
        clean_axes(ax, "y")
    save(fig, spec)


def fig_a02_a03(which: str) -> None:
    df = read_csv("correlation_mean_std_var.csv")
    df = df.loc[(df["method"].eq("spearman")) & (df["model"].eq("distributional")) & (df["loss"].eq("HybridNseBatchLoss"))]
    value_col = "mean_corr" if which == "mean" else "std_corr"
    stem = "FigA02_full_mean_correlation_heatmap" if which == "mean" else "FigA03_full_std_correlation_heatmap"
    params = ordered(sorted(df["parameter"].unique()), PARAM_ORDER)
    attrs = ordered(sorted(df["attribute"].unique()), KEY_ATTRS)
    mat = df.pivot(index="parameter", columns="attribute", values=value_col).reindex(params)[attrs]
    spec = FigureSpec(stem, APP_FIG_DIR, 190, 145)
    fig, ax = figure(spec)
    heatmap(ax, mat, cmap=muted_diverging(), center=0, cbar_label="Spearman rho" if which == "mean" else "Seed std. of rho")
    ax.set_title("Full distributional parameter-mean relationship matrix" if which == "mean" else "Full distributional uncertainty relationship matrix")
    save(fig, spec)


def fig_a04() -> None:
    df = read_csv("finalcheck_attribute_collinearity.csv")
    attrs = ordered(sorted(set(df.attribute_a) | set(df.attribute_b)), KEY_ATTRS)
    mat = df.pivot(index="attribute_a", columns="attribute_b", values="spearman_rho").reindex(attrs)[attrs]
    spec = FigureSpec("FigA04_attribute_collinearity_heatmap", APP_FIG_DIR, 170, 135)
    fig, ax = figure(spec)
    heatmap(ax, mat, cmap=muted_diverging(), center=0, cbar_label="Spearman rho")
    ax.set_yticklabels(wrap_labels([a_label(a) for a in mat.index], 12))
    ax.set_title("Attribute collinearity check")
    save(fig, spec)


def fig_a05() -> None:
    imp = read_csv("results333_parameter_feature_importance.csv")
    d = imp.groupby(["model", "parameter"], as_index=False).agg(top3_rate=("top3_rate", "mean"), surrogate_r2=("surrogate_r2_mean", "mean"))
    params = ordered(sorted(d.parameter.unique()), PARAM_ORDER)
    spec = FigureSpec("FigA05_full_explainability_results", APP_FIG_DIR, 185, 135)
    fig, axes = figure(spec, rows=1, cols=2)
    for ax, value, title, panel in zip(axes, ["top3_rate", "surrogate_r2"], ["Mean top-3 feature rate", "Surrogate R2"], ["A", "B"]):
        mat = d.pivot(index="parameter", columns="model", values=value).reindex(params)[MODEL_ORDER].rename(columns=MODEL_LABELS)
        heatmap(ax, mat, cmap=muted_seq(), vmin=0, vmax=1, cbar_label=title, text=False)
        ax.set_title(title)
        add_panel_label(ax, panel)
    save(fig, spec)


def fig_a06() -> None:
    df = read_csv("results343_representative_basins.csv")
    params = ["parBETA", "parFC", "parPERC", "parUZL", "parCFR", "parCWH"]
    rows = []
    for r in df.itertuples():
        for p in params:
            rows.append(
                {
                    "basin": f"{r.group_label}\n{r.basin_id}",
                    "parameter": p,
                    "mean_unit": getattr(r, f"{p}_mean_unit"),
                    "std_unit": getattr(r, f"{p}_std_unit"),
                }
            )
    plot = pd.DataFrame(rows)
    spec = FigureSpec("FigA06_representative_basin_details", APP_FIG_DIR, 190, 145)
    fig, axes = figure(spec, rows=1, cols=2)
    for ax, value, title, panel in zip(axes, ["mean_unit", "std_unit"], ["Parameter mean", "Parameter uncertainty"], ["A", "B"]):
        mat = plot.pivot(index="parameter", columns="basin", values=value).reindex(params)
        heatmap(ax, mat, cmap=muted_seq(), cbar_label="Unit value")
        ax.set_title(title)
        add_panel_label(ax, panel)
    save(fig, spec)


def fig_a07() -> None:
    df = read_csv("focused_pair_loss_stability.csv")
    d = df.loc[df["method"].eq("kendall")].drop_duplicates(["pair_label", "model"])
    selected = d["pair_label"].drop_duplicates().head(12).tolist()
    spec = FigureSpec("FigA07_focused_pair_stability", APP_FIG_DIR, 185, 140)
    fig, axes = figure(spec, rows=1, cols=2)
    for ax, value, title, panel in zip(axes, ["cross_loss_range", "sign_consistency_across_losses"], ["Cross-loss range", "Sign consistency"], ["A", "B"]):
        mat = (
            d.loc[d["pair_label"].isin(selected)]
            .pivot(index="pair_label", columns="model", values=value)
            .reindex(index=selected, columns=MODEL_ORDER)
            .rename(columns=MODEL_LABELS)
        )
        mat.index = [idx.replace("__", " - ").replace("par", "") for idx in mat.index]
        heatmap(ax, mat, cmap=muted_seq(), vmin=0 if value != "cross_loss_range" else None, vmax=1 if value != "cross_loss_range" else None, cbar_label=title, text=True)
        ax.set_title(title)
        add_panel_label(ax, panel)
    save(fig, spec)


def write_style_guide() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    calibri_available = any("calibri" in f.name.lower() for f in mpl.font_manager.fontManager.ttflist)
    (REPORT_DIR / "figure_style_guide.md").write_text(
        f"""# Figure style guide

## Font

- Requested family: Calibri for all panel labels, axes, legends, colorbars, annotations, and table text.
- Panel labels: Calibri Bold.
- Runtime font availability in this container: {'Calibri found' if calibri_available else 'Calibri not found; Matplotlib requested Calibri and used configured sans-serif fallback for local rendering'}.

## Size

- Panel label: 13.5 pt.
- Subplot title: 11.5 pt.
- Axis label: 10.5 pt.
- Tick label: 9.2 pt.
- Legend: 9.2 pt.
- Annotation/table text: 8.8-10.2 pt; no intentionally sub-8.5 pt annotations.

## Palette

- deterministic: muted blue `#4C78A8`
- mc_dropout: muted orange `#F58518`
- distributional: muted teal `#2A9D8F`
- Shared/support classes use muted teal, grey-blue, and muted brown-red.
- Diverging heatmaps use a restrained blue-white-brown palette.

## Line Width

- Axis spines and primary lines: 0.85-1.0 pt.
- Secondary gridlines: 0.55 pt, only where they improve reading.
- Heatmap cell borders are omitted unless table structure requires them.

## Export Size

- Main double-column figures: 180-185 mm wide.
- Appendix/full heatmaps: up to 190 mm wide.
- PNG is exported for every figure.
- PNG resolution: 600 dpi.
- Editable source: `project/parameterize/publication_figures.py`.

## Panel Spacing

- Figures use constrained layout with two or three main panels.
- Main figures avoid dense multi-row inset grids; dense matrices and detailed tables are assigned to appendix figures.

## Colorbar Policy

- Heatmaps use one colorbar per panel.
- Diverging colorbars are centered at zero for correlations and signed gradients.
- Sequential colorbars are reserved for non-negative stability, importance, consistency, or summary metrics.
""",
        encoding="utf-8",
    )


def write_revision_log() -> None:
    rows = [
        ("Fig01", "Dense performance checks and small summary blocks.", "Reduced to paired differences plus compact metric heatmap.", "Clear sanity-check role; no overclaim."),
        ("Fig02", "Summary text and titles competed with paired differences.", "Kept NSE/KGE paired CDFs and a compact annotation block.", "Basin-level differences are easier to scan."),
        ("Fig03", "Basin-ordered carpet was too dense for main text.", "Kept overall stability heatmap and parameter-family summary.", "Main stability contrast is visible."),
        ("Fig04", "Dense pair-level panels obscured the main reliability evidence.", "Rebuilt as overall spread, selected robust pairs, and seed consistency.", "Distributional cross-seed advantage is the visual center."),
        ("Fig05", "Panel structure did not parallel Fig04 and the table was tight.", "Rebuilt as cross-loss spread, focused pairs, and loss consistency.", "Overall advantage plus sensitivity is clearer."),
        ("Fig06", "Network was visually busy.", "Simplified bipartite map, model-wise matrix, and 7/4/3 count strip.", "Shared/partial/sensitive structure is immediate."),
        ("Fig07", "Summary table was too small.", "Kept heatmap/embedding and enlarged compactness table.", "Within-model compactness is legible."),
        ("Fig08", "Too many support panels for a secondary evidence figure.", "Limited to parFC/parUZL/parCFR support summaries and representative importance.", "Reads as support rather than a new storyline."),
        ("Fig09", "Core heatmap competed with small side plots.", "Enlarged mean-gradient heatmap and limited side panels to flags/counts.", "Gradient structure is the focus."),
        ("Fig10", "Raw scatter panels were too fine-grained.", "Converted to three grouped gradient trends plus archetype heatmap.", "Representative gradients are easier to interpret."),
        ("Fig11", "Overlong vertical figure with mixed small panels.", "Compressed to heatmap, three selected gradients, and interpretation flags.", "Uncertainty structure is clear without losing cautions."),
        ("Fig12", "Schematic looked draft-like.", "Redrew with light boxes, consistent palette, and three-part logic.", "Synthesis is cleaner and publication-like."),
        ("FigA01-FigA07", "Appendix figures carried full-detail material with inconsistent typography.", "Regenerated full metrics, heatmaps, explainability, basin details, and focused pair stability in the same style.", "Appendix remains complete but orderly."),
    ]
    text = ["# Figure revision log\n"]
    for fig, problem, action, result in rows:
        text.append(f"## {fig}\n\n- Original problem: {problem}\n- Revision action: {action}\n- Clarity after revision: {result}\n")
    (REPORT_DIR / "figure_revision_log.md").write_text("\n".join(text), encoding="utf-8")


def write_role_summary() -> None:
    rows = [
        ("Fig01", "Main", "Do all formulations produce usable simulations?", "Main text only needs a compact predictive sanity check."),
        ("Fig02", "Main", "How large are basin-wise paired performance differences?", "Supports the secondary accuracy framing without dominating the paper."),
        ("Fig03", "Main", "How stable are inferred parameters across seeds?", "Introduces stability differences before relationship-level evidence."),
        ("Fig04", "Main", "Are recovered relationships stable across seeds?", "Primary reliability evidence."),
        ("Fig05", "Main", "Are recovered relationships stable across losses?", "Parallel robustness check with explicit caveat."),
        ("Fig06", "Main", "Which dominant controls are shared, partial, or model-sensitive?", "Central structural interpretation."),
        ("Fig07", "Main", "How similar are full relationship matrices?", "Connects local relationships to matrix-level structure."),
        ("Fig08", "Main", "Does post-hoc explainability support selected relationships?", "Secondary support, limited to representative parameters."),
        ("Fig09", "Main", "Do parameter means organize along basin gradients?", "Core gradient evidence."),
        ("Fig10", "Main", "What do representative gradients look like across basin groups?", "Translates Fig09 into interpretable archetypes."),
        ("Fig11", "Main", "Does uncertainty contain organized environmental structure?", "Core uncertainty evidence with caution flags."),
        ("Fig12", "Main", "How do formulation, reliability, structure, and interpretation connect?", "Conceptual synthesis."),
        ("FigA01", "Appendix", "What are the full predictive metric distributions?", "Expanded support for Fig01."),
        ("FigA02", "Appendix", "What is the full mean-correlation matrix?", "Full detail behind Fig09."),
        ("FigA03", "Appendix", "What is the full uncertainty-correlation matrix?", "Full detail behind Fig11."),
        ("FigA04", "Appendix", "Are attributes collinear?", "Defensive interpretability check."),
        ("FigA05", "Appendix", "What are the full explainability results?", "Complete support behind Fig08."),
        ("FigA06", "Appendix", "What are representative basin-level details?", "Detailed support behind Fig10/Fig11."),
        ("FigA07", "Appendix", "How do focused pairs behave across losses?", "Technical supplement for Fig05."),
    ]
    lines = ["# Figure role summary\n", "| Figure | Role | Question answered | Reason for placement |", "|---|---|---|---|"]
    lines.extend([f"| {fig} | {role} | {question} | {reason} |" for fig, role, question, reason in rows])
    (REPORT_DIR / "figure_role_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_all() -> None:
    setup_style()
    for fn in [fig04, fig05, fig09, fig11, fig06, fig07, fig08, fig10, fig01, fig02, fig03, fig12]:
        fn()
    fig_a01()
    fig_a02_a03("mean")
    fig_a02_a03("std")
    fig_a04()
    fig_a05()
    fig_a06()
    fig_a07()
    write_style_guide()
    write_revision_log()
    write_role_summary()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-style manuscript figures.")
    parser.add_argument("--all", action="store_true", help="Generate all manuscript figures.")
    args = parser.parse_args()
    if args.all:
        generate_all()
    else:
        parser.error("Only --all is currently supported.")


if __name__ == "__main__":
    main()
