"""Fig. S3 — Top-k sensitivity for relationship stability.

Recomputes top-k candidate relationships from the full seed/loss correlation
matrix rather than reusing a pre-filtered pair table.

For each model and top-k:
- seed_sd_rho: mean SD of Spearman rho across seeds for selected top-k pairs
- topk_retention: mean fraction of seeds in which selected pairs remain top-k
- dominant_consistency: mean fraction of seeds recovering the consensus dominant attribute
- loss_sd_rho: mean SD of seed-averaged Spearman rho across losses for selected top-k pairs
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common_appendix import (
    setup_style, clean_axes, add_panel_label,
    MM, MODEL_ORDER, MODEL_COLORS,
    math_model_labels, APP_FIG_DIR, save_fig,
)

OUT_STEM = "figS3_topk_sensitivity"

MODEL_LINESTYLES = {
    "deterministic": "-",
    "mc_dropout": "--",
    "distributional": "-.",
}

MODEL_MARKERS = {
    "deterministic": "o",
    "mc_dropout": "s",
    "distributional": "^",
}

TOPK_VALUES = [3, 5, 10]

CORR_TABLE_CANDIDATES = [
    Path("/workspace/autoresearch/project/parameterize/manuscript/analysis/01_model_consistency/data/seed_loss_correlation_matrix_long.csv"),
    Path("/workspace/autoresearch/project/parameterize/manuscript/analysis/figure6/data/fig06_panel_a_heatmap_data.csv"),
]

METRICS_CSV = Path(
    "/workspace/autoresearch/project/parameterize/manuscript/figures/appendix/figS3_topk_sensitivity_metrics.csv"
)


def _find_existing_path(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find a full correlation matrix table. Checked:\n"
        + "\n".join(str(p) for p in paths)
    )


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of these columns found: {candidates}")


def _standardize_corr_table(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and model labels."""

    model_col = None
    for cand in ["model", "model_raw", "model_name", "model_label"]:
        if cand in df.columns:
            model_col = cand
            break
    if model_col is None:
        raise KeyError("No model column found.")

    loss_col = _pick_col(df, ["loss", "loss_function", "loss_name"])
    seed_col = _pick_col(df, ["seed", "random_seed"])
    param_col = _pick_col(df, ["parameter", "parameter_name", "param"])
    attr_col = _pick_col(df, ["attribute", "attribute_name", "attr"])
    rho_col = _pick_col(df, ["spearman_rho", "rho", "spearman"])

    out = df[[model_col, loss_col, seed_col, param_col, attr_col, rho_col]].copy()
    out.columns = ["model", "loss", "seed", "parameter", "attribute", "rho"]

    model_map = {
        "delta_base": "deterministic",
        r"$\delta_{base}$": "deterministic",
        "δ_base": "deterministic",
        "base": "deterministic",
        "deterministic": "deterministic",

        "delta_mcd": "mc_dropout",
        r"$\delta_{mcd}$": "mc_dropout",
        "δ_mcd": "mc_dropout",
        "mcd": "mc_dropout",
        "mc_dropout": "mc_dropout",

        "delta_dist": "distributional",
        r"$\delta_{dist}$": "distributional",
        "δ_dist": "distributional",
        "dist": "distributional",
        "distributional": "distributional",
    }

    out["model"] = out["model"].astype(str).map(lambda x: model_map.get(x, x))
    out = out[out["model"].isin(MODEL_ORDER)].copy()

    out["rho"] = pd.to_numeric(out["rho"], errors="coerce")
    out = out.dropna(subset=["rho", "parameter", "attribute", "loss", "seed"])

    # Collapse any duplicate logical rows.
    out = (
        out.groupby(["model", "loss", "seed", "parameter", "attribute"], as_index=False)
        ["rho"].mean()
    )

    return out


def _top_attrs_by_mean_abs(g: pd.DataFrame, k: int) -> list[str]:
    rank = (
        g.assign(abs_rho=lambda x: x["rho"].abs())
        .groupby("attribute")["abs_rho"]
        .mean()
        .sort_values(ascending=False)
    )
    return rank.head(k).index.tolist()


def _seed_metrics_for_model(corr: pd.DataFrame, model: str, k: int) -> dict:
    """Compute seed-wise top-k stability metrics.

    Candidate top-k attributes are selected separately for each
    model × loss × parameter from the mean |rho| across seeds.
    """

    sds = []
    topk_rates = []
    dominant_rates = []
    n_pairs = 0

    mdf = corr[corr["model"] == model]

    for (loss, parameter), g in mdf.groupby(["loss", "parameter"]):
        top_attrs = _top_attrs_by_mean_abs(g, k)
        if not top_attrs:
            continue

        # Consensus dominant attribute based on mean |rho| across seeds.
        consensus_dom = _top_attrs_by_mean_abs(g, 1)[0]

        seed_topk_sets = {}
        seed_dom_attrs = {}

        for seed, sg in g.groupby("seed"):
            sg_ranked = (
                sg.assign(abs_rho=lambda x: x["rho"].abs())
                .sort_values("abs_rho", ascending=False)
            )
            seed_topk_sets[seed] = set(sg_ranked.head(k)["attribute"].tolist())
            seed_dom_attrs[seed] = sg_ranked.iloc[0]["attribute"]

        dominant_rates.append(
            np.mean([attr == consensus_dom for attr in seed_dom_attrs.values()])
        )

        for attr in top_attrs:
            vals = (
                g[g["attribute"] == attr]
                .groupby("seed")["rho"]
                .mean()
                .dropna()
            )
            if len(vals) >= 2:
                sds.append(float(vals.std(ddof=1)))

            topk_rates.append(
                np.mean([attr in attrs for attrs in seed_topk_sets.values()])
            )
            n_pairs += 1

    return {
        "n_seed_pairs": n_pairs,
        "seed_sd_rho": float(np.nanmean(sds)) if sds else np.nan,
        "topk_retention": float(np.nanmean(topk_rates)) if topk_rates else np.nan,
        "dominant_consistency": float(np.nanmean(dominant_rates)) if dominant_rates else np.nan,
    }


def _loss_metrics_for_model(corr: pd.DataFrame, model: str, k: int) -> dict:
    """Compute loss-wise stability metrics.

    Candidate top-k attributes are selected for each model × parameter
    from mean |rho| across all losses and seeds.
    """

    loss_sds = []
    n_pairs = 0

    mdf = corr[corr["model"] == model]

    for parameter, g in mdf.groupby("parameter"):
        top_attrs = _top_attrs_by_mean_abs(g, k)
        if not top_attrs:
            continue

        for attr in top_attrs:
            vals = (
                g[g["attribute"] == attr]
                .groupby("loss")["rho"]
                .mean()
                .dropna()
            )
            if len(vals) >= 2:
                loss_sds.append(float(vals.std(ddof=1)))
            n_pairs += 1

    return {
        "n_loss_pairs": n_pairs,
        "loss_sd_rho": float(np.nanmean(loss_sds)) if loss_sds else np.nan,
    }


def compute_topk_metrics(corr: pd.DataFrame) -> pd.DataFrame:
    records = []

    for model in MODEL_ORDER:
        for k in TOPK_VALUES:
            seed_metrics = _seed_metrics_for_model(corr, model, k)
            loss_metrics = _loss_metrics_for_model(corr, model, k)

            records.append({
                "model": model,
                "topk": k,
                **seed_metrics,
                **loss_metrics,
            })

    return pd.DataFrame(records)


def main() -> None:
    setup_style()

    corr_path = _find_existing_path(CORR_TABLE_CANDIDATES)
    corr_raw = pd.read_csv(corr_path)
    corr = _standardize_corr_table(corr_raw)

    metrics = compute_topk_metrics(corr)

    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_CSV, index=False)

    labels = math_model_labels()

    metric_cols = [
        ("seed_sd_rho", "Seed SD of ρ"),
        ("topk_retention", "Top-k retention"),
        ("dominant_consistency", "Dominant consistency"),
        ("loss_sd_rho", "Loss SD of ρ"),
    ]

    fig_w = 200 * MM
    fig_h = 60 * MM
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        1,
        len(metric_cols),
        left=0.075,
        right=0.99,
        top=0.88,
        bottom=0.26,
        hspace=0.0,
        wspace=0.48,
    )

    for col, (mcol, ylabel) in enumerate(metric_cols):
        ax = fig.add_subplot(gs[0, col])

        for model in MODEL_ORDER:
            sub = metrics[metrics["model"] == model].sort_values("topk")

            ax.plot(
                sub["topk"],
                sub[mcol],
                color=MODEL_COLORS[model],
                linestyle=MODEL_LINESTYLES[model],
                marker=MODEL_MARKERS[model],
                markersize=5.6,
                linewidth=1.55,
                label=labels[model],
            )

        ax.set_xticks(TOPK_VALUES)
        ax.tick_params(axis="both", labelsize=9.2, pad=2)
        ax.set_xlabel("Top-k", fontsize=9.5, labelpad=4)
        ax.set_ylabel(ylabel, fontsize=9.2, labelpad=4)

        if mcol in {"topk_retention", "dominant_consistency"}:
            ax.set_ylim(0.0, 1.05)

        clean_axes(ax, grid_axis="y")

        add_panel_label(
            ax,
            f"({'abcd'[col]})",
            x=0.98,
            y=0.98,
            ha="right",
            va="top",
            fontweight="normal",
            fontsize=11.5,
        )

    handles = [
        mlines.Line2D(
            [],
            [],
            color=MODEL_COLORS[m],
            linestyle=MODEL_LINESTYLES[m],
            marker=MODEL_MARKERS[m],
            markersize=5.6,
            linewidth=1.55,
            label=labels[m],
        )
        for m in MODEL_ORDER
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=9.2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
        handlelength=2.2,
        handletextpad=0.55,
        columnspacing=1.7,
    )

    save_fig(fig, OUT_STEM)

    print(f"Input correlation table: {corr_path}")
    print(f"Saved {APP_FIG_DIR / OUT_STEM}.png / .pdf")
    print(f"Saved metrics: {METRICS_CSV}")
    print(metrics)


if __name__ == "__main__":
    main()
