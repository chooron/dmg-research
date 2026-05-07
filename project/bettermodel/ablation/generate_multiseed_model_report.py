from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

THIS_FILE = Path(__file__).resolve()
PROJECT_BETTERMODEL_DIR = THIS_FILE.parent.parent
REPO_ROOT = PROJECT_BETTERMODEL_DIR.parent.parent
for path in (REPO_ROOT, PROJECT_BETTERMODEL_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from project.bettermodel.multiseed.summarize_multiseed_metrics import (
    DEFAULT_SEEDS,
    PROJECT_DIR,
    TARGET_METRICS,
    clean_values,
    compute_stats,
    load_model_runs,
    pick_metrics_dir,
    read_json,
    resolve_config_paths,
    write_csv,
)

REPORT_DIR = Path(__file__).resolve().parent / "report"
CONFIG_ORDER = (
    "config_dhbv_lstm.yaml",
    "config_dhbv_tcn.yaml",
    "config_dhbv_transformer.yaml",
    "config_dhbv_tsmixer.yaml",
    "config_dhbv_hopev1.yaml",
    "config_dhbv_hopev2.yaml",
    "config_dhbv_hopev3.yaml",
)
DISPLAY_NAMES = {
    "dhbv_lstm": "dhbv_lstm",
    "dhbv_tcn": "dhbv_tcn",
    "dhbv_transformer": "dhbv_transformer",
    "dhbv_tsmixer": "dhbv_tsmixer",
    "dhbv_hopev1": "s4d",
    "dhbv_hopev2": "s5dv1",
    "dhbv_hopev3": "s5dv2",
}
DISPLAY_ORDER = [DISPLAY_NAMES[name.removeprefix("config_").removesuffix(".yaml")] for name in CONFIG_ORDER]
CONFIG_PATHS = [str(PROJECT_DIR / "conf" / name) for name in CONFIG_ORDER]


@dataclass(frozen=True)
class ReportOutputs:
    combined_summary: Path
    per_seed_summary: Path
    seed_stability_summary: Path
    comprehensive_ranking: Path
    seed_basin_metrics: Path
    friedman: Path
    pairwise_wilcoxon: Path
    camels531_combined_summary: Path
    camels531_per_seed_summary: Path
    camels531_seed_basin_metrics: Path
    camels531_pairwise_wilcoxon: Path
    report_md: Path


def apply_display_order(frame: pd.DataFrame, column: str = "model", order: list[str] | None = None) -> pd.DataFrame:
    categories = DISPLAY_ORDER if order is None else order
    ordered = frame.copy()
    ordered[column] = pd.Categorical(ordered[column], categories=categories, ordered=True)
    ordered = ordered.sort_values([column] + [col for col in ordered.columns if col != column]).reset_index(drop=True)
    ordered[column] = ordered[column].astype(str)
    return ordered


def _collect_seed_basin_rows(
    model_specs: list[dict[str, Any]],
    seeds: tuple[int, ...],
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in model_specs:
        model_name = str(spec["model"])
        display_name = str(spec["display"])
        output_root = Path(spec["output_root"])
        split_epoch = int(spec["split_epoch"])
        for seed in seeds:
            seed_dir = output_root / f"seed_{seed}"
            metrics_dir, issue = pick_metrics_dir(seed_dir, split=split, split_epoch=split_epoch)
            if metrics_dir is None:
                missing.append(f"{model_name} seed={seed}: {issue}")
                continue
            if issue is not None:
                missing.append(f"{model_name} seed={seed}: warning={issue}")

            metrics = read_json(metrics_dir / "metrics.json")
            metric_arrays = {metric: clean_values(metrics[metric]) for metric in TARGET_METRICS}
            lengths = {metric: int(values.size) for metric, values in metric_arrays.items()}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"Inconsistent basin counts for {model_name} seed {seed}: {lengths}")

            basin_count = lengths[TARGET_METRICS[0]]
            for basin_index in range(basin_count):
                row = {
                    "model": model_name,
                    "model_display": display_name,
                    "seed": seed,
                    "basin_index": basin_index,
                }
                for metric in TARGET_METRICS:
                    row[metric] = float(metric_arrays[metric][basin_index])
                rows.append(row)

    if missing:
        hard_missing = [item for item in missing if "warning=" not in item]
        if hard_missing:
            raise FileNotFoundError("Missing required runs:\n" + "\n".join(hard_missing))

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No seed/basin metrics were loaded.")
    expected_rows = len(model_specs) * len(seeds) * frame["basin_index"].nunique()
    if len(frame) != expected_rows:
        raise RuntimeError(
            f"Unexpected row count: {len(frame)} != {expected_rows}. "
            "At least one model/seed/basin combination is missing."
        )
    return frame.sort_values(["model_display", "seed", "basin_index"]).reset_index(drop=True)


def load_seed_basin_metrics(seeds: tuple[int, ...] = DEFAULT_SEEDS, split: str = "test") -> pd.DataFrame:
    config_paths = resolve_config_paths(CONFIG_PATHS)
    model_runs = load_model_runs(config_paths, split=split)
    model_specs = [
        {
            "model": model_run.model,
            "display": DISPLAY_NAMES[model_run.model],
            "output_root": model_run.output_root,
            "split_epoch": model_run.split_epoch or 100,
        }
        for model_run in model_runs
    ]
    return _collect_seed_basin_rows(model_specs=model_specs, seeds=seeds, split=split)


def load_camels531_lstm_s4d_metrics(seeds: tuple[int, ...] = DEFAULT_SEEDS, split: str = "test") -> pd.DataFrame:
    model_specs = [
        {
            "model": "dhbv_lstm",
            "display": "dhbv_lstm",
            "output_root": PROJECT_DIR / "outputs" / "dhbv_lstm" / "camels_531",
            "split_epoch": 100,
        },
        {
            "model": "dhbv_hopev1",
            "display": "s4d",
            "output_root": PROJECT_DIR / "outputs" / "dhbv_hopev1" / "camels_531",
            "split_epoch": 100,
        },
    ]
    return _collect_seed_basin_rows(model_specs=model_specs, seeds=seeds, split=split)


def summarize_per_seed(long_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (model_display, seed), group in long_df.groupby(["model_display", "seed"], sort=False):
        row: dict[str, Any] = {
            "model": model_display,
            "seed": int(seed),
            "basin_count": int(len(group)),
        }
        for metric in TARGET_METRICS:
            stats = compute_stats(group[metric].to_numpy(dtype=float))
            for key, value in stats.items():
                row[f"{metric}_{key}"] = value
        rows.append(row)
    return rows


def summarize_combined(long_df: pd.DataFrame, seed_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_display, group in long_df.groupby("model_display", sort=False):
        row: dict[str, Any] = {
            "model": model_display,
            "seed_count": seed_count,
            "total_records": int(len(group)),
            "basin_count_per_seed": int(group["basin_index"].nunique()),
        }
        for metric in TARGET_METRICS:
            stats = compute_stats(group[metric].to_numpy(dtype=float))
            for key, value in stats.items():
                row[f"{metric}_{key}"] = value
        rows.append(row)
    return rows


def summarize_seed_stability(per_seed_df: pd.DataFrame) -> list[dict[str, Any]]:
    tracked_columns = [
        "nse_mean",
        "nse_median",
        "kge_mean",
        "kge_median",
    ]
    rows: list[dict[str, Any]] = []
    for model_display, group in per_seed_df.groupby("model", sort=False):
        row: dict[str, Any] = {
            "model": model_display,
            "seed_count": int(len(group)),
        }
        for column in tracked_columns:
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_seed_mean"] = float(np.mean(values))
            row[f"{column}_seed_std"] = float(np.std(values, ddof=0))
            row[f"{column}_seed_min"] = float(np.min(values))
            row[f"{column}_seed_max"] = float(np.max(values))
        rows.append(row)
    return rows


def rank_desc(values: pd.Series) -> pd.Series:
    return values.rank(method="min", ascending=False)


def rank_asc(values: pd.Series) -> pd.Series:
    return values.rank(method="min", ascending=True)


def build_comprehensive_ranking(
    combined_df: pd.DataFrame,
    stability_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = combined_df.merge(
        stability_df.drop(columns=["seed_count"]),
        on="model",
        how="inner",
    )

    performance_columns = ["nse_mean", "nse_median", "kge_mean", "kge_median"]
    stability_columns = [
        "nse_mean_seed_std",
        "nse_median_seed_std",
        "kge_mean_seed_std",
        "kge_median_seed_std",
    ]
    for column in performance_columns:
        merged[f"{column}_rank"] = rank_desc(merged[column])
    for column in stability_columns:
        merged[f"{column}_rank"] = rank_asc(merged[column])

    merged["performance_rank_mean"] = merged[[f"{c}_rank" for c in performance_columns]].mean(axis=1)
    merged["stability_rank_mean"] = merged[[f"{c}_rank" for c in stability_columns]].mean(axis=1)
    merged["comprehensive_rank_score"] = merged[
        [f"{c}_rank" for c in performance_columns + stability_columns]
    ].mean(axis=1)
    merged["comprehensive_rank"] = merged["comprehensive_rank_score"].rank(method="min", ascending=True)
    merged["comprehensive_rank"] = merged["comprehensive_rank"].astype(int)
    merged = merged.sort_values(["comprehensive_rank_score", "model"]).reset_index(drop=True)
    return merged


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [math.nan] * m
    running_max = 0.0
    for rank, (original_index, p_value) in enumerate(ordered, start=1):
        candidate = min(1.0, (m - rank + 1) * p_value)
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def build_significance_outputs(
    long_df: pd.DataFrame,
    display_order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_models = DISPLAY_ORDER if display_order is None else display_order
    basin_mean_df = (
        long_df.groupby(["model_display", "basin_index"], as_index=False)[list(TARGET_METRICS)]
        .mean()
        .sort_values(["model_display", "basin_index"])
        .reset_index(drop=True)
    )

    friedman_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []

    for metric in TARGET_METRICS:
        pivot = (
            basin_mean_df.pivot(index="basin_index", columns="model_display", values=metric)
            .reindex(columns=ordered_models)
            .dropna(axis=0, how="any")
        )
        arrays = [pivot[column].to_numpy(dtype=float) for column in ordered_models]
        if len(arrays) >= 3:
            statistic, p_value = friedmanchisquare(*arrays)
            friedman_rows.append(
                {
                    "metric": metric,
                    "n_basins": int(len(pivot)),
                    "friedman_statistic": float(statistic),
                    "friedman_p_value": float(p_value),
                    "significant_at_0_05": bool(p_value < 0.05),
                }
            )
        else:
            friedman_rows.append(
                {
                    "metric": metric,
                    "n_basins": int(len(pivot)),
                    "friedman_statistic": float("nan"),
                    "friedman_p_value": float("nan"),
                    "significant_at_0_05": False,
                }
            )

        metric_rows: list[dict[str, Any]] = []
        p_values: list[float] = []
        for idx, model_a in enumerate(ordered_models):
            for model_b in ordered_models[idx + 1 :]:
                diff = pivot[model_a].to_numpy(dtype=float) - pivot[model_b].to_numpy(dtype=float)
                result = wilcoxon(diff, alternative="two-sided", zero_method="wilcox", correction=False)
                wins_a = int(np.sum(diff > 0))
                wins_b = int(np.sum(diff < 0))
                ties = int(np.sum(diff == 0))
                row = {
                    "metric": metric,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_basins": int(len(diff)),
                    "mean_diff_a_minus_b": float(np.mean(diff)),
                    "median_diff_a_minus_b": float(np.median(diff)),
                    "wins_a": wins_a,
                    "wins_b": wins_b,
                    "ties": ties,
                    "win_rate_a": float(wins_a / len(diff)),
                    "wilcoxon_statistic": float(result.statistic),
                    "p_value": float(result.pvalue),
                    "better_model_by_mean_diff": model_a if np.mean(diff) > 0 else model_b if np.mean(diff) < 0 else "tie",
                }
                metric_rows.append(row)
                p_values.append(float(result.pvalue))

        adjusted = holm_adjust(p_values)
        for row, adjusted_p in zip(metric_rows, adjusted):
            row["p_value_holm"] = float(adjusted_p)
            row["significant_at_0_05_holm"] = bool(adjusted_p < 0.05)
            pairwise_rows.append(row)

    return pd.DataFrame(friedman_rows), pd.DataFrame(pairwise_rows)


def format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def top_lines_for_metric(ranking_df: pd.DataFrame, metric: str) -> list[str]:
    ordered = ranking_df.sort_values(metric, ascending=False)[["model", metric]].head(3).values.tolist()
    return [f"- `{model}`: {format_float(value)}" for model, value in ordered]


def build_report_markdown(
    combined_df: pd.DataFrame,
    per_seed_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    friedman_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    camels531_combined_df: pd.DataFrame,
    camels531_per_seed_df: pd.DataFrame,
    camels531_pairwise_df: pd.DataFrame,
    outputs: ReportOutputs,
) -> str:
    def table_from_df(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
        lines = []
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        lines.extend([header, separator])
        for _, row in frame.iterrows():
            cells = []
            for column in columns:
                value = row[column]
                if isinstance(value, float):
                    cells.append(format_float(value, digits))
                else:
                    cells.append(str(value))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    combined_display = ranking_df[
        [
            "comprehensive_rank",
            "model",
            "comprehensive_rank_score",
            "nse_mean",
            "nse_median",
            "kge_mean",
            "kge_median",
            "nse_mean_seed_std",
            "kge_mean_seed_std",
        ]
    ].copy()
    per_seed_display = per_seed_df[
        ["model", "seed", "nse_mean", "nse_median", "kge_mean", "kge_median"]
    ].copy()
    significance_display = pairwise_df[
        pairwise_df["significant_at_0_05_holm"]
    ][
        ["metric", "model_a", "model_b", "mean_diff_a_minus_b", "p_value_holm", "better_model_by_mean_diff"]
    ].copy()
    camels531_pairwise_display = camels531_pairwise_df[
        ["metric", "model_a", "model_b", "mean_diff_a_minus_b", "p_value_holm", "better_model_by_mean_diff"]
    ].copy()

    nse_friedman = friedman_df[friedman_df["metric"] == "nse"].iloc[0]
    kge_friedman = friedman_df[friedman_df["metric"] == "kge"].iloc[0]
    camels531_summary = camels531_combined_df[["model", "nse_mean", "nse_median", "kge_mean", "kge_median"]].copy()
    camels531_wide = {
        row["model"]: row
        for _, row in camels531_summary.iterrows()
    }

    lines = [
        "# 7-Model Multi-Seed Comparison Report",
        "",
        "## Scope",
        "",
        "本报告基于 `project/bettermodel/conf` 中以下 7 个配置，对 `camels_671` 测试集进行 5-seed 汇总比较：",
        "",
        "- `config_dhbv_lstm.yaml` -> `dhbv_lstm`",
        "- `config_dhbv_tcn.yaml` -> `dhbv_tcn`",
        "- `config_dhbv_transformer.yaml` -> `dhbv_transformer`",
        "- `config_dhbv_tsmixer.yaml` -> `dhbv_tsmixer`",
        "- `config_dhbv_hopev1.yaml` -> `s4d`",
        "- `config_dhbv_hopev2.yaml` -> `s5dv1`",
        "- `config_dhbv_hopev3.yaml` -> `s5dv2`",
        "",
        "固定使用 seeds: `111, 222, 333, 444, 555`。",
        "",
        "统计重点：",
        "",
        "- `NSE`",
        "- `KGE`",
        "- 所有 `seed × basin` 合并后的综合统计",
        "- 5 个 seed 各自的结果",
        "- 基于 basin 级 seed 平均分数的显著性分析",
        "",
        "## Output Files",
        "",
        f"- [combined summary]({outputs.combined_summary})",
        f"- [per-seed summary]({outputs.per_seed_summary})",
        f"- [seed stability summary]({outputs.seed_stability_summary})",
        f"- [comprehensive ranking]({outputs.comprehensive_ranking})",
        f"- [seed-basin long table]({outputs.seed_basin_metrics})",
        f"- [Friedman significance]({outputs.friedman})",
        f"- [pairwise Wilcoxon significance]({outputs.pairwise_wilcoxon})",
        f"- [CAMELS 531 combined summary]({outputs.camels531_combined_summary})",
        f"- [CAMELS 531 per-seed summary]({outputs.camels531_per_seed_summary})",
        f"- [CAMELS 531 seed-basin long table]({outputs.camels531_seed_basin_metrics})",
        f"- [CAMELS 531 pairwise Wilcoxon]({outputs.camels531_pairwise_wilcoxon})",
        "",
        "## Method",
        "",
        "1. 从每个模型的 `test1995-2010_Ep100/metrics.json` 读取 basin 级 `NSE` 和 `KGE`。",
        "2. 将 5 个 seed 的所有 basin 指标合并，得到综合统计结果。",
        "3. 分别统计每个 seed 自身的 `mean / median / std / quartiles`。",
        "4. 用 basin 级 seed 平均指标进行显著性分析，避免把同一 basin 的多 seed 重复当作独立样本。",
        "5. 显著性检验包含：",
        "   - 7 模型整体比较：Friedman test",
        "   - 两两比较：Wilcoxon signed-rank test + Holm 校正",
        "",
        "## Comprehensive Ranking",
        "",
        "综合排名的构成：",
        "",
        "- 性能项：`nse_mean`, `nse_median`, `kge_mean`, `kge_median`，越高越好",
        "- 稳定性项：`nse_mean_seed_std`, `nse_median_seed_std`, `kge_mean_seed_std`, `kge_median_seed_std`，越低越好",
        "- `comprehensive_rank_score` 为上述 8 项 rank 的平均值，越低越好",
        "",
        table_from_df(
            combined_display,
            [
                "comprehensive_rank",
                "model",
                "comprehensive_rank_score",
                "nse_mean",
                "nse_median",
                "kge_mean",
                "kge_median",
                "nse_mean_seed_std",
                "kge_mean_seed_std",
            ],
        ),
        "",
        "主要观察：",
        "",
    ]

    top3 = ranking_df.head(3)
    for _, row in top3.iterrows():
        lines.append(
            f"- Top {int(row['comprehensive_rank'])}: `{row['model']}` "
            f"(score={format_float(row['comprehensive_rank_score'])}, "
            f"NSE mean={format_float(row['nse_mean'])}, "
            f"KGE mean={format_float(row['kge_mean'])})"
        )

    lines.extend(
        [
            "",
            "按综合 pooled 指标看，前 3 名如下：",
            "",
            "### NSE Mean Top 3",
            *top_lines_for_metric(ranking_df, "nse_mean"),
            "",
            "### KGE Mean Top 3",
            *top_lines_for_metric(ranking_df, "kge_mean"),
            "",
            "## Per-Seed Results",
            "",
            table_from_df(
                per_seed_display,
                ["model", "seed", "nse_mean", "nse_median", "kge_mean", "kge_median"],
            ),
            "",
            "## Seed Stability Summary",
            "",
            table_from_df(
                stability_df[
                    [
                        "model",
                        "nse_mean_seed_mean",
                        "nse_mean_seed_std",
                        "nse_median_seed_mean",
                        "nse_median_seed_std",
                        "kge_mean_seed_mean",
                        "kge_mean_seed_std",
                        "kge_median_seed_mean",
                        "kge_median_seed_std",
                    ]
                ],
                [
                    "model",
                    "nse_mean_seed_mean",
                    "nse_mean_seed_std",
                    "nse_median_seed_mean",
                    "nse_median_seed_std",
                    "kge_mean_seed_mean",
                    "kge_mean_seed_std",
                    "kge_median_seed_mean",
                    "kge_median_seed_std",
                ],
            ),
            "",
            "## Significance Analysis",
            "",
            f"- `NSE` Friedman: statistic={format_float(float(nse_friedman['friedman_statistic']))}, "
            f"p={format_float(float(nse_friedman['friedman_p_value']), 6)}",
            f"- `KGE` Friedman: statistic={format_float(float(kge_friedman['friedman_statistic']))}, "
            f"p={format_float(float(kge_friedman['friedman_p_value']), 6)}",
            "",
            "若 `p < 0.05`，说明 7 个模型在该指标上整体存在显著差异。",
            "",
        ]
    )

    if significance_display.empty:
        lines.extend(
            [
                "Holm 校正后没有模型对在 `0.05` 水平下显著。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Holm 校正后显著的两两比较：",
                "",
                table_from_df(
                    significance_display,
                    [
                        "metric",
                        "model_a",
                        "model_b",
                        "mean_diff_a_minus_b",
                        "p_value_holm",
                        "better_model_by_mean_diff",
                    ],
                    digits=6,
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- `combined summary` 是把 5 个 seed 的全部 basin 指标直接拼接后再统计，因此最接近“总体表现”。",
            "- `per-seed summary` 适合看单次训练波动。",
            "- `seed stability summary` 反映模型对随机种子的敏感性。",
            "- 显著性分析基于 basin 级 seed 平均指标，更适合做结构之间的稳健比较。",
            "- 若需要论文写法，优先引用 `combined summary` + `pairwise Wilcoxon significance`。",
            "",
            "## CAMELS 531 Supplement: `dhbv_lstm` vs `s4d`",
            "",
            "补充说明：这一节只比较 `dhbv_lstm` 和 `s4d` 在 `camels_531` 上的 5-seed 测试结果，用于和上面的 `camels_671` 主结论互相参照。",
            "",
            table_from_df(
                camels531_summary,
                ["model", "nse_mean", "nse_median", "kge_mean", "kge_median"],
            ),
            "",
            "逐 seed 结果：",
            "",
            table_from_df(
                camels531_per_seed_df[["model", "seed", "nse_mean", "nse_median", "kge_mean", "kge_median"]],
                ["model", "seed", "nse_mean", "nse_median", "kge_mean", "kge_median"],
            ),
            "",
            "补充结论：",
            "",
            f"- 在 `camels_531` 上，`s4d` 的 pooled `NSE mean` 为 {format_float(float(camels531_wide['s4d']['nse_mean']))}，高于 `dhbv_lstm` 的 {format_float(float(camels531_wide['dhbv_lstm']['nse_mean']))}。",
            f"- 在 `camels_531` 上，`s4d` 的 pooled `KGE mean` 为 {format_float(float(camels531_wide['s4d']['kge_mean']))}，高于 `dhbv_lstm` 的 {format_float(float(camels531_wide['dhbv_lstm']['kge_mean']))}。",
            "- 这说明主报告里 `s4d` 相对 `dhbv_lstm` 的优势，不只出现在 `camels_671`，在 `camels_531` 上也保持一致。",
            "",
            "基于 basin 级 seed 平均分数的两模型 Wilcoxon 检验：",
            "",
            table_from_df(
                camels531_pairwise_display,
                ["metric", "model_a", "model_b", "mean_diff_a_minus_b", "p_value_holm", "better_model_by_mean_diff"],
                digits=6,
            ),
            "",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = ReportOutputs(
        combined_summary=REPORT_DIR / "multiseed_7model_combined_summary.csv",
        per_seed_summary=REPORT_DIR / "multiseed_7model_per_seed_summary.csv",
        seed_stability_summary=REPORT_DIR / "multiseed_7model_seed_stability_summary.csv",
        comprehensive_ranking=REPORT_DIR / "multiseed_7model_comprehensive_ranking.csv",
        seed_basin_metrics=REPORT_DIR / "multiseed_7model_seed_basin_metrics.csv",
        friedman=REPORT_DIR / "multiseed_7model_friedman_significance.csv",
        pairwise_wilcoxon=REPORT_DIR / "multiseed_7model_pairwise_wilcoxon.csv",
        camels531_combined_summary=REPORT_DIR / "camels531_lstm_s4d_combined_summary.csv",
        camels531_per_seed_summary=REPORT_DIR / "camels531_lstm_s4d_per_seed_summary.csv",
        camels531_seed_basin_metrics=REPORT_DIR / "camels531_lstm_s4d_seed_basin_metrics.csv",
        camels531_pairwise_wilcoxon=REPORT_DIR / "camels531_lstm_s4d_pairwise_wilcoxon.csv",
        report_md=REPORT_DIR / "multiseed_7model_comparison_report.md",
    )

    long_df = load_seed_basin_metrics()
    per_seed_rows = summarize_per_seed(long_df)
    per_seed_df = apply_display_order(pd.DataFrame(per_seed_rows), column="model").sort_values(
        ["model", "seed"]
    ).reset_index(drop=True)
    combined_df = apply_display_order(
        pd.DataFrame(summarize_combined(long_df, seed_count=len(DEFAULT_SEEDS))),
        column="model",
    )
    stability_df = apply_display_order(pd.DataFrame(summarize_seed_stability(per_seed_df)), column="model")
    ranking_df = build_comprehensive_ranking(combined_df, stability_df)
    friedman_df, pairwise_df = build_significance_outputs(long_df)
    pairwise_df = pairwise_df.sort_values(["metric", "p_value_holm", "model_a", "model_b"]).reset_index(drop=True)

    camels531_long_df = load_camels531_lstm_s4d_metrics()
    camels531_per_seed_df = apply_display_order(
        pd.DataFrame(summarize_per_seed(camels531_long_df)),
        column="model",
        order=["dhbv_lstm", "s4d"],
    ).sort_values(["model", "seed"]).reset_index(drop=True)
    camels531_combined_df = apply_display_order(
        pd.DataFrame(summarize_combined(camels531_long_df, seed_count=len(DEFAULT_SEEDS))),
        column="model",
        order=["dhbv_lstm", "s4d"],
    )
    _, camels531_pairwise_df = build_significance_outputs(
        camels531_long_df,
        display_order=["dhbv_lstm", "s4d"],
    )
    camels531_pairwise_df = camels531_pairwise_df.sort_values(
        ["metric", "p_value_holm", "model_a", "model_b"]
    ).reset_index(drop=True)

    long_df.to_csv(outputs.seed_basin_metrics, index=False)
    write_csv(outputs.per_seed_summary, per_seed_df.to_dict(orient="records"))
    write_csv(outputs.combined_summary, combined_df.to_dict(orient="records"))
    write_csv(outputs.seed_stability_summary, stability_df.to_dict(orient="records"))
    write_csv(outputs.comprehensive_ranking, ranking_df.to_dict(orient="records"))
    write_csv(outputs.friedman, friedman_df.to_dict(orient="records"))
    write_csv(outputs.pairwise_wilcoxon, pairwise_df.to_dict(orient="records"))
    camels531_long_df.to_csv(outputs.camels531_seed_basin_metrics, index=False)
    write_csv(outputs.camels531_per_seed_summary, camels531_per_seed_df.to_dict(orient="records"))
    write_csv(outputs.camels531_combined_summary, camels531_combined_df.to_dict(orient="records"))
    write_csv(outputs.camels531_pairwise_wilcoxon, camels531_pairwise_df.to_dict(orient="records"))

    report_text = build_report_markdown(
        combined_df=combined_df,
        per_seed_df=per_seed_df,
        stability_df=stability_df,
        ranking_df=ranking_df,
        friedman_df=friedman_df,
        pairwise_df=pairwise_df,
        camels531_combined_df=camels531_combined_df,
        camels531_per_seed_df=camels531_per_seed_df,
        camels531_pairwise_df=camels531_pairwise_df,
        outputs=outputs,
    )
    outputs.report_md.write_text(report_text, encoding="utf-8")

    print(outputs.report_md)


if __name__ == "__main__":
    main()
