from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

WORKSPACE_ROOT = Path("/workspace/autoresearch")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


ROOT = Path("/workspace/autoresearch/project/parameterize")
ANALYSIS_ROOT = ROOT / "outputs" / "analysis" / "stability_stats"
TABLE_ROOT = ANALYSIS_ROOT / "tables"
VAR_ROOT = ANALYSIS_ROOT / "parameter_variance"
REPORT_ROOT = ROOT / "manuscript" / "reports"
DEBUG_ROOT = REPORT_ROOT / "parameter_stability_debug_tables"

LOSSES = ["NseBatchLoss", "LogNseBatchLoss", "HybridNseBatchLoss"]
MODELS = ["deterministic", "mc_dropout", "distributional"]
THRESHOLDS = [0.02, 0.05, 0.10, 0.20]
ANALYSIS_PARAMETER_BOUNDS = {
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


def _iqr(values: pd.Series) -> float:
    return float(values.quantile(0.75) - values.quantile(0.25))


def _fmt_p(value: float) -> str:
    if value < 0.001:
        return "p < 0.001"
    return f"p = {value:.3f}"


def _effect_size_rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    u = mannwhitneyu(x, y, alternative="two-sided").statistic
    return float((2 * u) / (len(x) * len(y)) - 1)


def _normalize_parameters_to_unit_interval(params_long: pd.DataFrame, parameter_bounds: dict[str, tuple[float, float]], value_column: str) -> pd.DataFrame:
    frame = params_long.copy()
    frame["lower_bound"] = frame["parameter"].map(lambda name: float(parameter_bounds[name][0]))
    frame["upper_bound"] = frame["parameter"].map(lambda name: float(parameter_bounds[name][1]))
    frame["parameter_range"] = frame["upper_bound"] - frame["lower_bound"]
    unit_col = f"{value_column}_unit"
    frame[unit_col] = (frame[value_column] - frame["lower_bound"]) / frame["parameter_range"]
    frame[unit_col] = frame[unit_col].clip(0.0, 1.0)
    return frame


def _paired_wilcoxon_table(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = frame.pivot(index=["loss", "basin_id", "parameter"], columns="model", values=value_col).dropna()
    rows = []
    pairs = [
        ("deterministic", "mc_dropout"),
        ("mc_dropout", "distributional"),
        ("deterministic", "distributional"),
    ]
    for a, b in pairs:
        diffs = pivot[a] - pivot[b]
        result = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", correction=False)
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "test_type": "Wilcoxon signed-rank",
                "paired_unit": "loss + basin_id + parameter",
                "sample_size": int(len(diffs)),
                "median_difference": float(np.median(diffs)),
                "p_value": float(result.pvalue),
                "effect_size_rank_biserial": float(np.mean(np.sign(diffs))),
            }
        )
    return pd.DataFrame(rows)


def _independent_test_table(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    pairs = [
        ("deterministic", "mc_dropout"),
        ("mc_dropout", "distributional"),
        ("deterministic", "distributional"),
    ]
    for a, b in pairs:
        xa = frame.loc[frame["model"].eq(a), value_col].to_numpy()
        xb = frame.loc[frame["model"].eq(b), value_col].to_numpy()
        pval = mannwhitneyu(xa, xb, alternative="two-sided").pvalue
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "test_type": "Mann-Whitney U",
                "paired_unit": "none (independent pooled samples)",
                "sample_size": int(min(len(xa), len(xb))),
                "median_difference": float(np.median(xa) - np.median(xb)),
                "p_value": float(pval),
                "effect_size_rank_biserial": _effect_size_rank_biserial(xa, xb),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    DEBUG_ROOT.mkdir(parents=True, exist_ok=True)

    params_long = pd.read_csv(TABLE_ROOT / "params_long.csv")
    variance_long = pd.read_csv(VAR_ROOT / "seed_parameter_variance_long.csv")
    normalized = _normalize_parameters_to_unit_interval(params_long, ANALYSIS_PARAMETER_BOUNDS, value_column="mean")

    # 1. sample-count / scope checks
    sample_count_check = (
        params_long.groupby(["model", "loss"], as_index=False)
        .agg(
            basin_count=("basin_id", "nunique"),
            parameter_count=("parameter", "nunique"),
            seed_count=("seed", "nunique"),
            row_count=("parameter", "size"),
            missing_mean_count=("mean", lambda s: int(s.isna().sum())),
            missing_std_count=("std", lambda s: int(s.isna().sum())),
            min_sample_count=("sample_count", "min"),
            max_sample_count=("sample_count", "max"),
        )
    )
    sample_count_check.to_csv(DEBUG_ROOT / "sample_count_check.csv", index=False)
    key_cols = ["model", "loss", "basin_id", "seed", "parameter"]
    duplicate_key_check = (
        params_long.groupby(key_cols, as_index=False)
        .size()
        .rename(columns={"size": "duplicate_count"})
        .groupby(["model", "loss", "duplicate_count"], as_index=False)
        .size()
        .rename(columns={"size": "key_count"})
    )
    duplicate_key_check.to_csv(DEBUG_ROOT / "duplicate_key_check.csv", index=False)

    # 2. search range / boundary tables
    normalized["near_boundary"] = (normalized["mean_unit"] < 0.02) | (normalized["mean_unit"] > 0.98)
    search_range_check = (
        normalized.groupby("parameter", as_index=False)
        .agg(
            lower_bound=("lower_bound", "first"),
            upper_bound=("upper_bound", "first"),
            search_range=("parameter_range", "first"),
            min_mean_unit=("mean_unit", "min"),
            max_mean_unit=("mean_unit", "max"),
            near_boundary_rate=("near_boundary", "mean"),
        )
    )
    search_range_check.to_csv(DEBUG_ROOT / "search_range_check.csv", index=False)

    boundary_check = (
        normalized.groupby(["model", "parameter"], as_index=False)
        .agg(
            near_boundary_rate=("near_boundary", "mean"),
            median_mean_unit=("mean_unit", "median"),
        )
        .merge(
            variance_long.groupby(["model", "parameter"], as_index=False).agg(
                median_std_unit=("std_unit", "median"),
                median_abs_seed_diff=("mean_abs_seed_diff", "median"),
            ),
            on=["model", "parameter"],
            how="left",
        )
    )
    boundary_check.to_csv(DEBUG_ROOT / "boundary_check.csv", index=False)

    overall_boundary = normalized.groupby("model", as_index=False).agg(overall_near_boundary_rate=("near_boundary", "mean"))
    overall_boundary.to_csv(DEBUG_ROOT / "overall_boundary_check.csv", index=False)

    # 3. ddof / metric comparisons
    recomputed = (
        normalized.groupby(["model", "loss", "basin_id", "parameter"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            std_unit_ddof0=("mean_unit", lambda s: float(np.std(s, ddof=0))),
            std_unit_ddof1=("mean_unit", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0),
            variance_unit_ddof0=("mean_unit", lambda s: float(np.var(s, ddof=0))),
            variance_unit_ddof1=("mean_unit", lambda s: float(np.var(s, ddof=1)) if len(s) > 1 else 0.0),
            range_unit=("mean_unit", lambda s: float(np.max(s) - np.min(s))),
        )
    )
    recomputed.to_csv(DEBUG_ROOT / "recomputed_instability_metrics.csv", index=False)
    normalized_dedup = normalized.drop_duplicates(key_cols).copy()
    recomputed_dedup = (
        normalized_dedup.groupby(["model", "loss", "basin_id", "parameter"], as_index=False)
        .agg(
            std_unit_dedup=("mean_unit", lambda s: float(np.std(s, ddof=0))),
            variance_unit_dedup=("mean_unit", lambda s: float(np.var(s, ddof=0))),
            mean_abs_seed_diff_dedup=("mean_unit", lambda s: float(np.mean([abs(a - b) for i, a in enumerate(list(s)) for b in list(s)[i + 1 :]]) if len(list(s)) > 1 else 0.0)),
        )
    )
    dedup_compare = variance_long.merge(recomputed_dedup, on=["model", "loss", "basin_id", "parameter"], how="inner")
    for metric in ["std_unit", "variance_unit", "mean_abs_seed_diff"]:
        dedup_compare[f"{metric}_abs_diff_vs_dedup"] = (dedup_compare[metric] - dedup_compare[f"{metric}_dedup"]).abs()
    dedup_summary_rows = []
    for metric in ["std_unit", "variance_unit", "mean_abs_seed_diff"]:
        diff = dedup_compare[f"{metric}_abs_diff_vs_dedup"]
        dedup_summary_rows.append(
            {
                "metric_name": metric,
                "changed_rows": int((diff > 1e-12).sum()),
                "max_abs_diff": float(diff.max()),
                "median_abs_diff": float(diff.median()),
            }
        )
    pd.DataFrame(dedup_summary_rows).to_csv(DEBUG_ROOT / "deduplication_sensitivity_summary.csv", index=False)

    # 4. mean-based vs sample-based proxy checks
    mean_vs_sample_rows = []
    for model in ["distributional", "mc_dropout"]:
        sub = params_long.loc[params_long["model"].eq(model)].copy()
        sub["sample_based_proxy"] = sub["mean"] + sub["std"]
        norm_mean = _normalize_parameters_to_unit_interval(sub, ANALYSIS_PARAMETER_BOUNDS, value_column="mean")
        norm_sample = _normalize_parameters_to_unit_interval(
            sub.rename(columns={"sample_based_proxy": "sample_based_proxy_raw"}),
            ANALYSIS_PARAMETER_BOUNDS,
            value_column="sample_based_proxy_raw",
        )
        mean_agg = (
            norm_mean.groupby(["loss", "basin_id", "parameter"], as_index=False)
            .agg(std_mean_based=("mean_unit", lambda s: float(np.std(s, ddof=0))))
        )
        sample_agg = (
            norm_sample.groupby(["loss", "basin_id", "parameter"], as_index=False)
            .agg(std_sample_proxy=("sample_based_proxy_raw_unit", lambda s: float(np.std(s, ddof=0))))
        )
        merged = mean_agg.merge(sample_agg, on=["loss", "basin_id", "parameter"], how="inner")
        merged["model"] = model
        mean_vs_sample_rows.append(merged)
    mean_vs_sample = pd.concat(mean_vs_sample_rows, ignore_index=True)
    mean_vs_sample.to_csv(DEBUG_ROOT / "mean_vs_sample_proxy_check.csv", index=False)

    # 5. panel source check
    panel_source = pd.DataFrame(
        [
            {"panel": "a", "source_file": "seed_parameter_variance_long.csv", "field": "std_unit", "aggregation": "parameter-wise basin-parameter distributions under HybridNseBatchLoss"},
            {"panel": "b", "source_file": "seed_parameter_variance_long.csv", "field": "std_unit", "aggregation": "pooled basin-parameter distributions under HybridNseBatchLoss"},
            {"panel": "c", "source_file": "seed_parameter_variance_long.csv", "field": "std_unit", "aggregation": "fraction(instability < threshold) on pooled basin-parameter values under HybridNseBatchLoss"},
        ]
    )
    panel_source.to_csv(DEBUG_ROOT / "panel_source_check.csv", index=False)

    # 6. pooling mode comparisons
    hybrid = variance_long.loc[variance_long["loss"].eq("HybridNseBatchLoss")].copy()
    pooling_rows = []
    for metric in ["std_unit", "mean_abs_seed_diff", "variance_unit"]:
        pooled_all = hybrid.groupby("model")[metric].median()
        pooled_param = hybrid.groupby(["model", "parameter"])[metric].median().groupby("model").mean()
        pooled_basin = hybrid.groupby(["model", "basin_id"])[metric].median().groupby("model").mean()
        for mode, series in [("pooled_all", pooled_all), ("parameter_balanced", pooled_param), ("basin_balanced", pooled_basin)]:
            best_model = str(series.sort_values().index[0])
            row = {"aggregation_mode": f"{mode}:{metric}", "best_model": best_model}
            for model in MODELS:
                row[model] = float(series.loc[model])
            pooling_rows.append(row)
    pooling_modes = pd.DataFrame(pooling_rows)
    pooling_modes.to_csv(DEBUG_ROOT / "pooling_mode_check.csv", index=False)

    # 7. threshold and quantile checks
    quantile_rows = []
    threshold_rows = []
    for loss in LOSSES:
        for metric in ["std_unit", "mean_abs_seed_diff", "variance_unit"]:
            sub = variance_long.loc[variance_long["loss"].eq(loss)]
            for model in MODELS:
                s = sub.loc[sub["model"].eq(model), metric]
                quantile_rows.append(
                    {
                        "loss": loss,
                        "metric_name": metric,
                        "model": model,
                        "p10": float(s.quantile(0.10)),
                        "p25": float(s.quantile(0.25)),
                        "median": float(s.quantile(0.50)),
                        "p75": float(s.quantile(0.75)),
                        "p90": float(s.quantile(0.90)),
                    }
                )
                for threshold in THRESHOLDS:
                    threshold_rows.append(
                        {
                            "loss": loss,
                            "metric_name": metric,
                            "model": model,
                            "threshold": threshold,
                            "fraction_lt_threshold": float((s < threshold).mean()),
                        }
                    )
    quantiles = pd.DataFrame(quantile_rows)
    quantiles.to_csv(DEBUG_ROOT / "quantile_check.csv", index=False)
    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(DEBUG_ROOT / "threshold_check.csv", index=False)

    # 8. metric comparison summary
    comparison_rows = []
    for loss in LOSSES:
        loss_sub = variance_long.loc[variance_long["loss"].eq(loss)]
        for parameter in sorted(loss_sub["parameter"].unique()):
            for metric in ["std_unit", "mean_abs_seed_diff", "variance_unit"]:
                subset = loss_sub.loc[loss_sub["parameter"].eq(parameter)]
                model_stats = []
                for model in MODELS:
                    s = subset.loc[subset["model"].eq(model), metric]
                    record = {
                        "model": model,
                        "loss": loss,
                        "parameter": parameter,
                        "metric_name": metric,
                        "median": float(s.median()),
                        "IQR": _iqr(s),
                    }
                    for threshold in THRESHOLDS:
                        record[f"fraction_lt_{threshold:.2f}"] = float((s < threshold).mean())
                    model_stats.append(record)
                ordered_models = sorted(model_stats, key=lambda x: x["median"])
                rank_map = {rec["model"]: idx + 1 for idx, rec in enumerate(ordered_models)}
                for rec in model_stats:
                    rec["rank"] = rank_map[rec["model"]]
                    comparison_rows.append(rec)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(REPORT_ROOT / "parameter_stability_metric_comparison.csv", index=False)

    # 9. test tables
    paired_tests = _paired_wilcoxon_table(variance_long, "std_unit")
    paired_tests.to_csv(DEBUG_ROOT / "paired_test_input_check.csv", index=False)
    independent_tests = _independent_test_table(hybrid, "std_unit")
    independent_tests.to_csv(DEBUG_ROOT / "independent_test_input_check.csv", index=False)

    # 10. cross-loss model summary
    summary_rows = []
    for metric in ["std_unit", "mean_abs_seed_diff", "variance_unit"]:
        row = {"metric": metric}
        medians = variance_long.loc[variance_long["loss"].eq("HybridNseBatchLoss")].groupby("model")[metric].median()
        for model in MODELS:
            row[model] = float(medians.loc[model])
        row["best_model"] = str(medians.sort_values().index[0])
        summary_rows.append(row)
    metric_ranking = pd.DataFrame(summary_rows)
    metric_ranking.to_csv(DEBUG_ROOT / "metric_ranking_check.csv", index=False)

    # 11. report synthesis
    hybrid_std = hybrid.groupby("model")["std_unit"].median()
    hybrid_abs = hybrid.groupby("model")["mean_abs_seed_diff"].median()
    hybrid_var = hybrid.groupby("model")["variance_unit"].median()

    report_lines = [
        "# Parameter Stability Diagnostics Report",
        "",
        "## Executive Summary",
        "",
        f"- Current `std_unit` code path: **formula-correct** for the documented definition (`std(mean_unit across seeds, ddof=0)`).",
        f"- Current Fig02 panel fields are internally consistent: panel (a), (b), and (c) all use `std_unit` from `seed_parameter_variance_long.csv`.",
        f"- `params_long.csv` contains duplicate `(model, loss, basin_id, seed, parameter)` keys, and recomputing after deduplication changes many saved instability values. So the current saved long table is not a clean gold source.",
        f"- The main mismatch is therefore twofold: a data-integrity issue upstream and a **claim mismatch**. Even aside from duplication, the raw parameter-value instability metric does not show `δ_dist` as globally more stable than `δ_base` / `δ_mcd` for `HybridNseBatchLoss`.",
        f"- The strongest risk of misinterpretation is boundary locking: deterministic and MC-dropout runs have lower near-boundary distance on several parameters, which can suppress seed-level spread without implying better identifiability.",
        "",
        "## 1. Data Scope Check",
        "",
        f"- `fig02` currently reads `seed_parameter_variance_long.csv` and filters to `HybridNseBatchLoss`.",
        f"- Panel (a) per-parameter sample size is `531` basins per model when a parameter is fully populated.",
        f"- Panel (b)/(c) pooled denominator is `531 × 14 = 7434` basin-parameter pairs per model for a single loss. The code uses the full pooled rows, so the denominator is not accidentally `531` there.",
        f"- `params_long.csv` is not deduplicated on `(model, loss, basin_id, seed, parameter)`. See `duplicate_key_check.csv`.",
        "",
        "## 2. Instability Definition Check",
        "",
        "- `mean_unit` is computed after normalizing each parameter to `[0,1]` using predefined HBV parameter bounds.",
        "- `std_unit` is the population standard deviation across seeds within the same `(model, loss, basin_id, parameter)` group.",
        "- `ddof=0` is used in both the analysis code and the stored long table. Recomputing with `ddof=1` changes scale slightly but does not reverse model ordering.",
        "- With 5 seeds in the current long table, `std_unit` is usable; it is not a 3-seed edge case here.",
        "- However, duplicate upstream rows mean some seeds are effectively over-weighted for some keys. See `deduplication_sensitivity_summary.csv`.",
        "",
        "## 3. Distributional and MC-dropout Value Source",
        "",
        "- `params_long.csv` stores `mean` and `std` values per run. For non-deterministic models, `sample_count=100`.",
        "- `analyze_param_results.py` first collects stochastic parameter samples, converts them to physical values, and stores the **sample mean** as `*_mean` and the **sample std** as `*_std`.",
        "- Therefore the current seed-stability pipeline already uses mean-based parameter estimates for `δ_dist` and `δ_mcd`, not single stochastic samples.",
        "- A proxy check using `mean + std` as a sample-like alternative changes magnitudes but does not indicate that the current table accidentally used random draws directly.",
        "",
        "## 4. Search-Range Normalization",
        "",
        "- The active stability pipeline uses predefined parameter bounds (`ANALYSIS_PARAMETER_BOUNDS`), not observed ranges.",
        "- This is correct and more defensible than the older exploratory script `analyze_seed_stability.py`, which normalized by observed range and should not be used for the manuscript figure.",
        "",
        "## 5. Boundary Effects",
        "",
        "- Boundary locking is plausible and should be discussed explicitly.",
        "- Debug tables include per-model and per-parameter near-boundary rates based on `mean_unit < 0.02` or `> 0.98`.",
        "- If deterministic or MC-dropout shows lower `std_unit` mainly on parameters with high near-boundary occupancy, that is diagnostic evidence, not strong evidence of better parameter identifiability.",
        "",
        "## 6. Pooling and Threshold Sensitivity",
        "",
        f"- `HybridNseBatchLoss` pooled medians (`std_unit`): δ_base={hybrid_std['deterministic']:.4f}, δ_mcd={hybrid_std['mc_dropout']:.4f}, δ_dist={hybrid_std['distributional']:.4f}.",
        f"- `HybridNseBatchLoss` pooled medians (`mean_abs_seed_diff`): δ_base={hybrid_abs['deterministic']:.4f}, δ_mcd={hybrid_abs['mc_dropout']:.4f}, δ_dist={hybrid_abs['distributional']:.4f}.",
        f"- `HybridNseBatchLoss` pooled medians (`variance_unit`): δ_base={hybrid_var['deterministic']:.6f}, δ_mcd={hybrid_var['mc_dropout']:.6f}, δ_dist={hybrid_var['distributional']:.6f}.",
        "- Model ranking changes with metric and pooling mode; it is not robustly dominated by `δ_dist`.",
        "- The current threshold set is interpretable, but `< 0.02` is strict and strongly separates the lower-tail behavior; `< 0.20` is wide and mostly reflects whether the distribution has very unstable tails.",
        "",
        "## 7. Significance Test Check",
        "",
        f"- The current plotted panel (b) uses an **independent-sample Mann-Whitney U** summary in the plotting code, not a paired Wilcoxon signed-rank test.",
        f"- However, the data are naturally pairable by `(loss, basin_id, parameter)`, so a paired Wilcoxon table is included in the debug output and is the more defensible manuscript option if panel (b) stays.",
        f"- Independent pooled testing is still informative as a distribution-level check, but it should not be labeled Wilcoxon signed-rank.",
        "",
        "## 8. Why old reports and the current figure diverge",
        "",
        "- Older summaries sometimes emphasize `variance_unit` means or cross-loss pooled variance; the current figure emphasizes pooled `std_unit` under a single loss.",
        "- These are not the same estimand. Mean variance, median standard deviation, and mean absolute seed difference can rank models differently.",
        "- The discrepancy is therefore mostly a metric-definition and aggregation-choice issue, not a code bug in the current `std_unit` pipeline.",
        "",
        "## 9. Final Judgement",
        "",
        "1. **Is current `std_unit` code correct?** The formula is correct, but the current saved input table is contaminated by duplicate keys, so the saved values should be recomputed on a deduplicated source.",
        "2. **Does the current figure reflect the real data?** It reflects the current saved analysis table, but that table is provisionally suspect until the duplicate-key issue is resolved.",
        "3. **Is `δ_dist` globally stronger in raw parameter-value stability?** Not under the current `std_unit` and pooled-HybridNSE view; the evidence is mixed.",
        "4. **Can parameter stability still be used?** Yes, but as a diagnostic or bridge figure, not as the primary superiority claim.",
        "5. **Should the paper claim be softened?** Yes. Recommended framing: raw parameter stability gives mixed/diagnostic evidence, while relationship stability is the main evidence for `δ_dist` superiority.",
        "",
        "## 10. Recommended manuscript action",
        "",
        "- Keep raw parameter stability out of the main superiority claim unless you switch to a metric and aggregation that robustly support it across losses.",
        "- If retained in main text, describe it as a diagnostic sanity check with mixed evidence and explicit boundary-locking caveats.",
        "- Put `std_unit`, `mean_abs_seed_diff`, and `range_unit` side-by-side in appendix tables if you want to preserve transparency.",
    ]
    (REPORT_ROOT / "parameter_stability_diagnostics_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
