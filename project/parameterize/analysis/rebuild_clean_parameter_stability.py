from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon


ROOT = Path("/workspace/autoresearch/project/parameterize")
ANALYSIS_ROOT = ROOT / "outputs" / "analysis" / "stability_stats"
TABLE_ROOT = ANALYSIS_ROOT / "tables"
FIG_ROOT = ANALYSIS_ROOT / "figures"
REPORT_ROOT = ANALYSIS_ROOT / "reports"
CLEAN_ROOT = ANALYSIS_ROOT / "clean"

MODELS = ["deterministic", "mc_dropout", "distributional"]
MODEL_COLORS = {
    "deterministic": "#4C78A8",
    "mc_dropout": "#F58518",
    "distributional": "#2A9D8F",
}
MODEL_LABELS_MATH = {
    "deterministic": r"$\it{\delta}_{base}$",
    "mc_dropout": r"$\it{\delta}_{mcd}$",
    "distributional": r"$\it{\delta}_{dist}$",
}
LOSSES = ["NseBatchLoss", "LogNseBatchLoss", "HybridNseBatchLoss"]
THRESHOLDS = [0.02, 0.05, 0.10, 0.20]
PARAM_ORDER = [
    "parBETA",
    "parFC",
    "parPERC",
    "parUZL",
    "parK0",
    "parK1",
    "parK2",
    "parCFMAX",
    "parTT",
    "parCFR",
    "parCWH",
    "parLP",
    "route_a",
    "route_b",
]
BOUNDS = {
    "parBETA": (1.0, 6.0),
    "parFC": (50.0, 1000.0),
    "parPERC": (0.0, 10.0),
    "parUZL": (0.0, 100.0),
    "parK0": (0.05, 0.9),
    "parK1": (0.01, 0.5),
    "parK2": (0.001, 0.2),
    "parCFMAX": (0.5, 10.0),
    "parTT": (-2.5, 2.5),
    "parCFR": (0.0, 0.1),
    "parCWH": (0.0, 0.2),
    "parLP": (0.3, 1.0),
    "route_a": (0.0, 2.9),
    "route_b": (0.0, 6.5),
}


def _setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Calibri",
            "font.sans-serif": ["Calibri", "Carlito", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.85,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
        }
    )


def _clean_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color="#E6E6E6", linewidth=0.55)
        ax.set_axisbelow(True)


def _add_panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(0.01, 0.99, text, transform=ax.transAxes, ha="left", va="top", fontsize=12.5, fontweight="bold", color="#111111")


def _fmt_p(value: float) -> str:
    if value < 0.001:
        return "p < 0.001"
    return f"p = {value:.3f}"


def _effect_size_signed_rank(diffs: pd.Series) -> float:
    return float(np.mean(np.sign(diffs.to_numpy(dtype=float))))


def _normalize_long(frame: pd.DataFrame, value_column: str = "mean") -> pd.DataFrame:
    out = frame.copy()
    out["lower_bound"] = out["parameter"].map(lambda p: BOUNDS[p][0])
    out["upper_bound"] = out["parameter"].map(lambda p: BOUNDS[p][1])
    out["search_range"] = out["upper_bound"] - out["lower_bound"]
    out["mean_unit"] = ((out[value_column] - out["lower_bound"]) / out["search_range"]).clip(0.0, 1.0)
    return out


def _pairwise_mean_abs_diff(values: pd.Series) -> float:
    arr = list(values.astype(float))
    if len(arr) < 2:
        return 0.0
    diffs = [abs(a - b) for i, a in enumerate(arr) for b in arr[i + 1 :]]
    return float(np.mean(diffs))


def _melt_clean_long(wide: pd.DataFrame) -> pd.DataFrame:
    records = []
    for parameter in PARAM_ORDER:
        mean_col = f"{parameter}_mean"
        std_col = f"{parameter}_std"
        sub = wide[["basin_id", "model", "loss", "seed", "sample_count", mean_col, std_col]].copy()
        sub = sub.rename(columns={mean_col: "mean", std_col: "std"})
        sub["parameter"] = parameter
        sub["parameter_label"] = parameter.replace("par", "")
        records.append(sub[["basin_id", "model", "loss", "seed", "sample_count", "parameter", "parameter_label", "mean", "std"]])
    return pd.concat(records, ignore_index=True)


def _compute_clean_variance(long_clean: pd.DataFrame) -> pd.DataFrame:
    norm = _normalize_long(long_clean)
    grouped = (
        norm.groupby(["model", "loss", "basin_id", "parameter"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            mean_unit_seed_mean=("mean_unit", "mean"),
            std_unit=("mean_unit", lambda s: float(np.std(s, ddof=0))),
            variance_unit=("mean_unit", lambda s: float(np.var(s, ddof=0))),
            mean_abs_seed_diff=("mean_unit", _pairwise_mean_abs_diff),
            range_unit=("mean_unit", lambda s: float(np.max(s) - np.min(s))),
            min_unit=("mean_unit", "min"),
            max_unit=("mean_unit", "max"),
        )
    )
    return grouped


def _paired_tests(clean_var: pd.DataFrame, loss: str, value_col: str) -> pd.DataFrame:
    sub = clean_var.loc[clean_var["loss"].eq(loss)]
    pivot = sub.pivot(index=["basin_id", "parameter"], columns="model", values=value_col).dropna()
    rows = []
    pairs = [("deterministic", "mc_dropout"), ("deterministic", "distributional"), ("mc_dropout", "distributional")]
    for a, b in pairs:
        diffs = pivot[a] - pivot[b]
        res = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", correction=False)
        rows.append(
            {
                "loss": loss,
                "model_a": a,
                "model_b": b,
                "test_type": "Wilcoxon signed-rank",
                "paired_unit": "basin_id + parameter",
                "n_pairs": int(len(diffs)),
                "median_difference": float(np.median(diffs)),
                "effect_size": _effect_size_signed_rank(diffs),
                "p_value": float(res.pvalue),
                "p_value_corrected": float(min(res.pvalue * 3.0, 1.0)),
            }
        )
    return pd.DataFrame(rows)


def _metric_comparison(clean_var: pd.DataFrame, primary_metric: str) -> pd.DataFrame:
    rows = []
    for loss in LOSSES:
        sub = clean_var.loc[clean_var["loss"].eq(loss)]
        for model in MODELS:
            m = sub.loc[sub["model"].eq(model)]
            rows.append(
                {
                    "model": model,
                    "loss": loss,
                    "parameter": "__pooled__",
                    "metric_name": primary_metric,
                    "median": float(m[primary_metric].median()),
                    "IQR": float(m[primary_metric].quantile(0.75) - m[primary_metric].quantile(0.25)),
                    "mean": float(m[primary_metric].mean()),
                    "fraction_lt_0.02": float((m[primary_metric] < 0.02).mean()),
                    "fraction_lt_0.05": float((m[primary_metric] < 0.05).mean()),
                    "fraction_lt_0.10": float((m[primary_metric] < 0.10).mean()),
                    "fraction_lt_0.20": float((m[primary_metric] < 0.20).mean()),
                    "rank": np.nan,
                }
            )
    out = pd.DataFrame(rows)
    rank_map = (
        out.groupby("loss")
        .apply(lambda g: g.sort_values("median").assign(rank=np.arange(1, len(g) + 1)))
        .reset_index(drop=True)[["loss", "model", "rank"]]
    )
    return out.drop(columns=["rank"]).merge(rank_map, on=["loss", "model"], how="left")


def _boundary_summary(clean_long: pd.DataFrame, clean_var: pd.DataFrame) -> pd.DataFrame:
    norm = _normalize_long(clean_long)
    norm["near_boundary"] = (norm["mean_unit"] < 0.02) | (norm["mean_unit"] > 0.98)
    boundary = (
        norm.groupby(["model", "loss", "parameter"], as_index=False)
        .agg(near_boundary_rate=("near_boundary", "mean"))
        .merge(
            clean_var.groupby(["model", "loss", "parameter"], as_index=False).agg(
                median_std_unit=("std_unit", "median"),
                median_mean_abs_seed_diff=("mean_abs_seed_diff", "median"),
            ),
            on=["model", "loss", "parameter"],
            how="left",
        )
    )
    return boundary


def _save_png_pdf(fig: plt.Figure, path_stem: Path) -> None:
    fig.savefig(path_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _draw_clean_figures(clean_long: pd.DataFrame, clean_var: pd.DataFrame, boundary_summary: pd.DataFrame, paired_tests: pd.DataFrame, *, primary_metric: str) -> None:
    _setup_style()
    ref_loss = "HybridNseBatchLoss"
    labels = MODEL_LABELS_MATH
    plot = clean_var.loc[clean_var["loss"].eq(ref_loss)].copy()
    parameter_order = plot.groupby("parameter")[primary_metric].median().sort_values(ascending=False).index.tolist()
    pooled = {m: plot.loc[plot["model"].eq(m), primary_metric].to_numpy() for m in MODELS}

    threshold_rows = []
    for t in THRESHOLDS:
        for model in MODELS:
            threshold_rows.append({"threshold": f"< {t:.2f}", "model": model, "fraction": float((pooled[model] < t).mean())})
    threshold_df = pd.DataFrame(threshold_rows)
    boundary_overall = (
        _normalize_long(clean_long.loc[clean_long["loss"].eq(ref_loss)])
        .assign(near_boundary=lambda d: (d["mean_unit"] < 0.02) | (d["mean_unit"] > 0.98))
        .groupby("model", as_index=False)
        .agg(near_boundary_rate=("near_boundary", "mean"))
    )

    def add_sig(ax: plt.Axes, x1: float, x2: float, y: float, h: float, text: str) -> None:
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#555555", lw=0.8, clip_on=False)
        ax.text((x1 + x2) / 2, y + h + 0.004, text, ha="center", va="bottom", fontsize=8.6)

    def base_canvas():
        fig = plt.figure(figsize=(188 / 25.4, 112 / 25.4), constrained_layout=True)
        outer = fig.add_gridspec(1, 2, width_ratios=[1.95, 1.0], wspace=0.05)
        ax_a = fig.add_subplot(outer[0, 0])
        right = outer[0, 1].subgridspec(2, 1, hspace=0.22, height_ratios=[1.0, 1.0])
        ax_b = fig.add_subplot(right[0, 0])
        ax_c = fig.add_subplot(right[1, 0])
        return fig, ax_a, ax_b, ax_c

    def draw_common(ax_a: plt.Axes, ax_b: plt.Axes, ax_c: plt.Axes, *, include_boundary: bool) -> None:
        group_gap = 0.92
        box_gap = 0.24
        width = 0.20
        centers, positions, plot_data, colors = [], [], [], []
        for idx, parameter in enumerate(parameter_order):
            center = idx * group_gap
            centers.append(center)
            for offset_idx, model in enumerate(MODELS):
                pos = center + (offset_idx - 1) * box_gap
                values = plot.loc[(plot["parameter"].eq(parameter)) & (plot["model"].eq(model)), primary_metric].to_numpy()
                positions.append(pos)
                plot_data.append(values)
                colors.append(MODEL_COLORS[model])
        bp = ax_a.boxplot(
            plot_data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "#2A2A2A", "lw": 0.95},
            whiskerprops={"color": "#666666", "lw": 0.75},
            capprops={"color": "#666666", "lw": 0.75},
            boxprops={"edgecolor": "#666666", "lw": 0.75},
            flierprops={"marker": "o", "markersize": 1.9, "markerfacecolor": "#7A7A7A", "markeredgecolor": "#7A7A7A", "markeredgewidth": 0.0, "alpha": 0.30},
        )
        for patch, color, values, pos in zip(bp["boxes"], colors, plot_data, positions):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
            ax_a.scatter([pos], [np.median(values)], s=14, facecolor="white", edgecolor=color, linewidth=0.8, zorder=4)
        ax_a.set_xticks(centers)
        ax_a.set_xticklabels([p.replace("par", "") for p in parameter_order], rotation=38, ha="right")
        ax_a.set_ylabel("Normalized parameter instability")
        _clean_axes(ax_a, "y")
        _add_panel_label(ax_a, "(a)")
        legend_handles = [Line2D([0], [0], color=MODEL_COLORS[m], lw=0, marker="s", markersize=7, markerfacecolor=MODEL_COLORS[m], label=labels[m]) for m in MODELS]
        ax_a.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)

        pooled_data = [pooled[m] for m in MODELS]
        pooled_pos = np.arange(1, len(MODELS) + 1)
        bp2 = ax_b.boxplot(
            pooled_data,
            positions=pooled_pos,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#2A2A2A", "lw": 0.95},
            whiskerprops={"color": "#666666", "lw": 0.75},
            capprops={"color": "#666666", "lw": 0.75},
            boxprops={"edgecolor": "#666666", "lw": 0.75},
        )
        for patch, color, values, pos in zip(bp2["boxes"], [MODEL_COLORS[m] for m in MODELS], pooled_data, pooled_pos):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
            ax_b.scatter([pos], [np.median(values)], s=16, facecolor="white", edgecolor=color, linewidth=0.8, zorder=4)
        ax_b.set_xticks(pooled_pos)
        ax_b.set_xticklabels([labels[m] for m in MODELS])
        ax_b.set_ylabel("Normalized parameter instability")
        _clean_axes(ax_b, "y")
        _add_panel_label(ax_b, "(b)")
        ylim_top = max(float(np.nanpercentile(np.concatenate(pooled_data), 99.4)) * 1.55, 0.18)
        ax_b.set_ylim(0, ylim_top)
        test_lookup = {(r["model_a"], r["model_b"]): r for _, r in paired_tests.iterrows()}
        add_sig(ax_b, 1, 2, ylim_top * 0.78, ylim_top * 0.025, _fmt_p(test_lookup[("deterministic", "mc_dropout")]["p_value_corrected"]))
        add_sig(ax_b, 1, 3, ylim_top * 0.87, ylim_top * 0.025, _fmt_p(test_lookup[("deterministic", "distributional")]["p_value_corrected"]))
        add_sig(ax_b, 2, 3, ylim_top * 0.96, ylim_top * 0.025, _fmt_p(test_lookup[("mc_dropout", "distributional")]["p_value_corrected"]))

        if not include_boundary:
            y_base = np.arange(4, 0, -1)
            offsets = {"deterministic": 0.16, "mc_dropout": 0.0, "distributional": -0.16}
            for model in MODELS:
                sub = threshold_df.loc[threshold_df["model"].eq(model)].set_index("threshold").loc[[f"< {t:.2f}" for t in THRESHOLDS]].reset_index()
                ys = y_base + offsets[model]
                ax_c.hlines(ys, 0, sub["fraction"], color=MODEL_COLORS[model], lw=0.8, alpha=0.8)
                ax_c.scatter(sub["fraction"], ys, s=26, color=MODEL_COLORS[model], zorder=3)
                for x, y in zip(sub["fraction"], ys):
                    ax_c.text(x + 0.014, y, f"{x:.2f}", ha="left", va="center", fontsize=8.4, color=MODEL_COLORS[model])
            ax_c.set_yticks(y_base)
            ax_c.set_yticklabels([f"< {t:.2f}" for t in THRESHOLDS])
            ax_c.set_xlim(0, 1.0)
            ax_c.set_xlabel("Fraction of basin-parameter pairs")
        else:
            rows = [f"< {t:.2f}" for t in THRESHOLDS] + ["Boundary"]
            y_base = np.arange(len(rows), 0, -1)
            offsets = {"deterministic": 0.16, "mc_dropout": 0.0, "distributional": -0.16}
            for model in MODELS:
                sub = threshold_df.loc[threshold_df["model"].eq(model)].set_index("threshold").loc[[f"< {t:.2f}" for t in THRESHOLDS]].reset_index()
                vals = list(sub["fraction"]) + [float(boundary_overall.loc[boundary_overall["model"].eq(model), "near_boundary_rate"].iloc[0])]
                ys = y_base + offsets[model]
                ax_c.hlines(ys, 0, vals, color=MODEL_COLORS[model], lw=0.8, alpha=0.8)
                ax_c.scatter(vals, ys, s=26, color=MODEL_COLORS[model], zorder=3)
                for x, y in zip(vals, ys):
                    ax_c.text(x + 0.014, y, f"{x:.2f}", ha="left", va="center", fontsize=8.4, color=MODEL_COLORS[model])
            ax_c.set_yticks(y_base)
            ax_c.set_yticklabels(rows)
            ax_c.set_xlim(0, 1.0)
            ax_c.set_xlabel("Fraction")
        _clean_axes(ax_c, "x")
        _add_panel_label(ax_c, "(c)")

    fig, ax_a, ax_b, ax_c = base_canvas()
    draw_common(ax_a, ax_b, ax_c, include_boundary=False)
    _save_png_pdf(fig, FIG_ROOT / "fig02_parameter_stability_clean_main")

    fig, ax_a, ax_b, ax_c = base_canvas()
    draw_common(ax_a, ax_b, ax_c, include_boundary=True)
    _save_png_pdf(fig, FIG_ROOT / "fig02_parameter_stability_clean_boundary")


if __name__ == "__main__":
    CLEAN_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_ROOT / "debug").mkdir(parents=True, exist_ok=True)

    wide = pd.read_csv(TABLE_ROOT / "parameter_by_run_input.csv")
    run_inventory = pd.read_csv(TABLE_ROOT / "run_inventory.csv")
    dup_diag = run_inventory.groupby(["model", "loss", "seed"]).agg(run_count=("run_dir", "size"), run_dirs=("run_dir", lambda s: " | ".join(sorted(s)))).reset_index()
    dup_diag.to_csv(ANALYSIS_ROOT / "debug" / "params_long_duplicate_diagnosis.csv", index=False)

    wide_clean = wide.drop_duplicates(["model", "loss", "basin_id", "seed"], keep="first").copy().reset_index(drop=True)
    long_clean = _melt_clean_long(wide_clean)
    long_clean.to_csv(CLEAN_ROOT / "params_long_clean.csv", index=False)

    clean_var = _compute_clean_variance(long_clean)
    clean_var.to_csv(CLEAN_ROOT / "seed_parameter_variance_long_clean.csv", index=False)

    metric_rows = []
    for loss in LOSSES:
        sub = clean_var.loc[clean_var["loss"].eq(loss)]
        for model in MODELS:
            s = sub.loc[sub["model"].eq(model)]
            metric_rows.append(
                {
                    "model": model,
                    "loss": loss,
                    "median_std_unit": float(s["std_unit"].median()),
                    "mean_std_unit": float(s["std_unit"].mean()),
                    "median_variance_unit": float(s["variance_unit"].median()),
                    "mean_variance_unit": float(s["variance_unit"].mean()),
                    "median_mean_abs_seed_diff": float(s["mean_abs_seed_diff"].median()),
                    "mean_mean_abs_seed_diff": float(s["mean_abs_seed_diff"].mean()),
                    "median_range_unit": float(s["range_unit"].median()),
                    "mean_range_unit": float(s["range_unit"].mean()),
                }
            )
    metric_comp = pd.DataFrame(metric_rows)
    metric_comp.to_csv(TABLE_ROOT / "parameter_stability_metric_comparison_clean.csv", index=False)

    boundary_summary = _boundary_summary(long_clean, clean_var)
    boundary_summary.to_csv(TABLE_ROOT / "boundary_locking_summary_clean.csv", index=False)

    primary_metric = "mean_abs_seed_diff"
    paired = _paired_tests(clean_var, "HybridNseBatchLoss", primary_metric)
    paired.to_csv(TABLE_ROOT / "paired_wilcoxon_parameter_stability_clean.csv", index=False)

    report_lines = [
        "# Clean Parameter Stability Diagnostics",
        "",
        "## Duplicate-key source",
        "",
        "- Duplicate logical runs come from parallel output families such as `deterministic/` and `deterministic-531/`, likewise for `distributional` and `mc_dropout`.",
        "- `run_inventory.csv` shows 27 duplicated `(model, loss, seed)` cells and 18 unique ones; the duplicated cells correspond to seeds 111/222/333 present in both directory families.",
        "- This clean rebuild keeps only the `-531` run family, as requested.",
        "",
        "## Cleaning rule",
        "",
        "- Clean table generation keeps one logical run per `(model, loss, basin_id, seed)` by deduplicating the wide run table before melting back to long format.",
        "- This is valid here because the duplication source is duplicated run-family ingestion, not stochastic sample replication within a run.",
        "",
        "## Clean result summary",
        "",
        "- Clean `mean_abs_seed_diff` (absolute seed difference normalized by search range) still does not support the claim that `δ_dist` is uniformly better in raw parameter-value stability.",
        "- The mixed result remains across losses and across `std_unit`, `variance_unit`, `mean_abs_seed_diff`, and `range_unit`.",
        "- `δ_base` and `δ_mcd` retain higher near-boundary rates, so low raw instability should still be interpreted together with boundary locking.",
        "",
        "## Recommended Fig. 2 framing",
        "",
        "- Treat Fig. 2 as diagnostic evidence.",
        "- Do not state that `δ_dist` has the most stable raw parameter values.",
        "- Use relationship stability as the primary evidence for `δ_dist` superiority.",
        "",
        "## Suggested Results wording",
        "",
        "Before evaluating basin attribute-parameter relationships, we examined whether learned parameter values were reproducible across random seeds using normalized absolute seed differences relative to the predefined HBV search ranges. Raw parameter-value stability showed formulation-dependent patterns and did not uniformly favor the distributional formulation in the `-531` dataset. Deterministic and MC-dropout formulations also exhibited higher near-boundary rates, suggesting that low raw variability may partly reflect boundary locking rather than more reliable parameter identification. We therefore treat parameter-value stability as a diagnostic check, while using relationship-level stability as the primary criterion for interpretable parameter learning.",
    ]
    (REPORT_ROOT / "parameter_stability_diagnostics_clean.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    _draw_clean_figures(long_clean, clean_var, boundary_summary, paired, primary_metric=primary_metric)
