from __future__ import annotations

import numpy as np
import pandas as pd

from .common import (
    FOCUSED_PAIRS,
    MODEL_LABELS,
    PipelineLog,
    pairwise_mean_abs_diff,
    report_common_sections,
    save_csv,
    sign_consistency,
    write_md,
)


def run(dirs: dict[str, dict[str, object]], context: dict[str, object], log: PipelineLog) -> None:
    block = "02_seed_loss_sensitivity"
    data_dir = dirs[block]["data"]
    reports_dir = dirs[block]["reports"]
    methods_dir = dirs[block]["methods"]
    logs_dir = dirs[block]["logs"]

    corr: pd.DataFrame = context["corr_long"].copy()
    corr["top3_flag"] = False
    corr["top5_flag"] = False
    corr["dominant_flag"] = False
    ranks = corr.groupby(["model_raw", "loss", "seed", "parameter"])["abs_spearman_rho"].rank(method="first", ascending=False)
    corr["top3_flag"] = ranks <= 3
    corr["top5_flag"] = ranks <= 5
    corr["dominant_flag"] = ranks == 1

    seed_stats = (
        corr.groupby(["model_raw", "model_label", "loss", "parameter", "attribute"], as_index=False)
        .agg(
            seed_mean_rho=("spearman_rho", "mean"),
            seed_sd_rho=("spearman_rho", "std"),
            seed_min_rho=("spearman_rho", "min"),
            seed_max_rho=("spearman_rho", "max"),
            mean_pairwise_seed_abs_diff=("spearman_rho", pairwise_mean_abs_diff),
            sign_consistency_across_seeds=("spearman_rho", sign_consistency),
            topk_seed_rate=("top5_flag", "mean"),
            dominant_seed_rate=("dominant_flag", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    seed_stats["seed_range_rho"] = seed_stats["seed_max_rho"] - seed_stats["seed_min_rho"]
    save_csv(seed_stats, data_dir / "cross_seed_relationship_sensitivity.csv", log)

    seed_first = (
        corr.groupby(["model_raw", "model_label", "loss", "parameter", "attribute"], as_index=False)
        .agg(
            loss_cell_rho=("spearman_rho", "mean"),
            loss_cell_topk_rate=("top5_flag", "mean"),
            loss_cell_dominant_rate=("dominant_flag", "mean"),
        )
    )
    loss_stats = (
        seed_first.groupby(["model_raw", "model_label", "parameter", "attribute"], as_index=False)
        .agg(
            loss_mean_rho=("loss_cell_rho", "mean"),
            cross_loss_sd_rho=("loss_cell_rho", "std"),
            cross_loss_min_rho=("loss_cell_rho", "min"),
            cross_loss_max_rho=("loss_cell_rho", "max"),
            mean_pairwise_loss_abs_diff=("loss_cell_rho", pairwise_mean_abs_diff),
            sign_consistency_across_losses=("loss_cell_rho", sign_consistency),
            topk_loss_rate=("loss_cell_topk_rate", "mean"),
            dominant_loss_rate=("loss_cell_dominant_rate", "mean"),
            n_losses=("loss", "nunique"),
        )
    )
    loss_stats["cross_loss_range_rho"] = loss_stats["cross_loss_max_rho"] - loss_stats["cross_loss_min_rho"]
    save_csv(loss_stats, data_dir / "cross_loss_relationship_sensitivity.csv", log)

    seed_summary = (
        seed_stats.groupby(["model_raw", "model_label"], as_index=False)
        .agg(
            mean_seed_sd_rho=("seed_sd_rho", "mean"),
            mean_seed_range_rho=("seed_range_rho", "mean"),
            mean_topk_seed_rate=("topk_seed_rate", "mean"),
            mean_dominant_seed_rate=("dominant_seed_rate", "mean"),
        )
    )
    loss_summary = (
        loss_stats.groupby(["model_raw", "model_label"], as_index=False)
        .agg(
            mean_cross_loss_sd_rho=("cross_loss_sd_rho", "mean"),
            mean_cross_loss_range_rho=("cross_loss_range_rho", "mean"),
            mean_topk_loss_rate=("topk_loss_rate", "mean"),
            mean_dominant_loss_rate=("dominant_loss_rate", "mean"),
        )
    )
    ratio = seed_summary.merge(loss_summary, on=["model_raw", "model_label"], how="outer")
    ratio["seed_sd_to_loss_sd_ratio"] = ratio["mean_seed_sd_rho"] / ratio["mean_cross_loss_sd_rho"]
    ratio["seed_range_to_loss_range_ratio"] = ratio["mean_seed_range_rho"] / ratio["mean_cross_loss_range_rho"]
    ratio["topk_seed_minus_loss_stability"] = ratio["mean_topk_seed_rate"] - ratio["mean_topk_loss_rate"]
    save_csv(ratio, data_dir / "seed_vs_loss_sensitivity_summary.csv", log)

    focused_seed = seed_stats.merge(pd.DataFrame(FOCUSED_PAIRS, columns=["parameter", "attribute"]), on=["parameter", "attribute"], how="inner")
    focused_loss = loss_stats.merge(pd.DataFrame(FOCUSED_PAIRS, columns=["parameter", "attribute"]), on=["parameter", "attribute"], how="inner")
    focused = focused_seed.merge(
        focused_loss,
        on=["model_raw", "model_label", "parameter", "attribute"],
        how="outer",
        suffixes=("_seed", "_loss"),
    )
    focused["focused_pair"] = focused["parameter"] + " - " + focused["attribute"]
    focused["is_topk_in_any_seed_loss"] = focused["topk_seed_rate"].fillna(0) > 0
    save_csv(focused, data_dir / "focused_pair_seed_loss_stability.csv", log)

    stat_rows = []
    models = sorted(ratio["model_raw"].dropna().unique())
    for metric in ["mean_seed_sd_rho", "mean_cross_loss_sd_rho", "mean_seed_range_rho", "mean_cross_loss_range_rho"]:
        for left in models:
            for right in models:
                if left >= right:
                    continue
                left_vals = seed_stats.loc[seed_stats["model_raw"].eq(left), "seed_sd_rho"] if "seed" in metric else loss_stats.loc[loss_stats["model_raw"].eq(left), "cross_loss_sd_rho"]
                right_vals = seed_stats.loc[seed_stats["model_raw"].eq(right), "seed_sd_rho"] if "seed" in metric else loss_stats.loc[loss_stats["model_raw"].eq(right), "cross_loss_sd_rho"]
                stat_rows.append(
                    {
                        "comparison": f"{MODEL_LABELS[left]} vs {MODEL_LABELS[right]}",
                        "metric": metric,
                        "left_median": float(np.nanmedian(left_vals)),
                        "right_median": float(np.nanmedian(right_vals)),
                    }
                )
    save_csv(pd.DataFrame(stat_rows), data_dir / "statistical_tests_seed_loss_sensitivity.csv", log)

    lowest_seed = ratio.sort_values("mean_seed_sd_rho").head(1).to_dict(orient="records")
    lowest_loss = ratio.sort_values("mean_cross_loss_sd_rho").head(1).to_dict(orient="records")
    write_md(
        reports_dir / "seed_loss_sensitivity_report.md",
        "Seed and Loss Sensitivity Report",
        report_common_sections(
            objective="Separate relationship-structure sensitivity to random seed from sensitivity to loss function.",
            input_files=["01_model_consistency/data/seed_loss_correlation_matrix_long.csv"],
            data_filters=["Seed variability is computed within each model/loss.", "Loss variability uses seed-first summaries."],
            metric_definitions=[
                "seed_sd_rho and seed_range_rho summarize cross-seed variability.",
                "cross_loss_sd_rho and cross_loss_range_rho summarize loss sensitivity after seed aggregation.",
                "topk and dominant rates track relationship retention.",
            ],
            main_results=[
                f"Lowest cross-seed variability: {lowest_seed}.",
                f"Lowest cross-loss variability: {lowest_loss}.",
                f"Seed/loss summary: {ratio.to_dict(orient='records')}.",
            ],
            tables=[
                "data/cross_seed_relationship_sensitivity.csv",
                "data/cross_loss_relationship_sensitivity.csv",
                "data/seed_vs_loss_sensitivity_summary.csv",
                "data/focused_pair_seed_loss_stability.csv",
                "data/statistical_tests_seed_loss_sensitivity.csv",
            ],
            caveats=["Top-k retention is descriptive and should not be treated as an independent hypothesis test."],
            wording=[
                "Use `more reproducibly recovers` for lower sensitivity.",
                "Describe sensitive pairs as relationship-level diagnostics rather than failures.",
            ],
            figure_usage=["Use these tables later for sensitivity panels or supplement checks."],
        ),
        log,
    )
    write_md(methods_dir / "method_definitions.md", "Seed Loss Method Definitions", {"Definitions": ["Seed and loss sensitivity are computed separately to avoid pseudo-replication."]}, log)
    write_md(logs_dir / "seed_loss_sensitivity_log.md", "Seed Loss Sensitivity Log", {"Summary": [f"Seed rows: {len(seed_stats)}", f"Loss rows: {len(loss_stats)}"]}, log)

