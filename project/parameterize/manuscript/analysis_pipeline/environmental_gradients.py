from __future__ import annotations

import numpy as np
import pandas as pd

from .common import CORE_GRADIENTS, KEY_GRADIENT_PAIRS, PipelineLog, make_tercile_assignments, mann_whitney_test, report_common_sections, save_csv, write_md


def run(dirs: dict[str, dict[str, object]], context: dict[str, object], log: PipelineLog) -> None:
    block = "05_environmental_gradient_groups"
    data_dir = dirs[block]["data"]
    reports_dir = dirs[block]["reports"]
    methods_dir = dirs[block]["methods"]
    logs_dir = dirs[block]["logs"]
    mean_maps: pd.DataFrame = context["distributional_mean_maps"]
    attrs: pd.DataFrame = context["attributes"]

    assignments = []
    for gradient in CORE_GRADIENTS:
        assignments.append(make_tercile_assignments(attrs, gradient))
    assign = pd.concat(assignments, ignore_index=True)
    save_csv(assign, data_dir / "gradient_group_assignments.csv", log)

    merged = mean_maps.merge(assign, on="basin_id", how="inner")
    stats = (
        merged.groupby(["gradient_attribute", "gradient_group", "parameter"], as_index=False, observed=True)
        .agg(
            n_basins=("basin_id", "nunique"),
            median_parameter_mean=("parameter_mean_unit", "median"),
            q25=("parameter_mean_unit", lambda x: x.quantile(0.25)),
            q75=("parameter_mean_unit", lambda x: x.quantile(0.75)),
        )
    )
    stats["iqr"] = stats["q75"] - stats["q25"]
    wide = stats.pivot_table(index=["gradient_attribute", "parameter"], columns="gradient_group", values="median_parameter_mean", observed=True)
    stats = stats.merge((wide.get("high") - wide.get("low")).rename("high_minus_low_median_difference").reset_index(), on=["gradient_attribute", "parameter"], how="left")
    save_csv(stats, data_dir / "gradient_parameter_group_stats.csv", log)

    test_rows = []
    for (gradient, parameter), sub in merged.groupby(["gradient_attribute", "parameter"]):
        low = sub.loc[sub["gradient_group"].eq("low"), "parameter_mean_unit"]
        high = sub.loc[sub["gradient_group"].eq("high"), "parameter_mean_unit"]
        test = mann_whitney_test(low, high)
        med_diff = high.median() - low.median()
        medians = sub.groupby("gradient_group", observed=True)["parameter_mean_unit"].median()
        trend_score = np.nan
        if {"low", "middle", "high"}.issubset(set(medians.index)):
            trend_score = float(np.sign(medians["middle"] - medians["low"]) + np.sign(medians["high"] - medians["middle"]))
        test_rows.append({"gradient_attribute": gradient, "parameter": parameter, "high_minus_low_median_difference": med_diff, "monotonic_trend_score_optional": trend_score, **test})
    tests = pd.DataFrame(test_rows)
    save_csv(tests, data_dir / "gradient_high_low_tests.csv", log)
    save_csv(tests, data_dir / "statistical_tests_gradient_high_low.csv", log)

    key = tests.merge(pd.DataFrame(KEY_GRADIENT_PAIRS, columns=["gradient_attribute", "parameter"]), on=["gradient_attribute", "parameter"], how="inner")
    strongest = tests.assign(abs_diff=lambda d: d["high_minus_low_median_difference"].abs()).sort_values("abs_diff", ascending=False).head(12)
    write_md(
        reports_dir / "environmental_gradient_report.md",
        "Environmental Gradient Report",
        report_common_sections(
            objective="Translate parameter mean-attribute correlations into low/middle/high environmental gradient comparisons.",
            input_files=["Distributional mean maps", "Basin attributes"],
            data_filters=["Lower/middle/upper terciles per gradient attribute.", "Distributional primary-loss seed-averaged means."],
            metric_definitions=["Group summaries use median and IQR.", "High-low tests use Mann-Whitney U.", "Trend score is descriptive monotonic direction count."],
            main_results=[f"Strongest high-low differences: {strongest[['gradient_attribute','parameter','high_minus_low_median_difference','p_value']].to_dict(orient='records')}.", f"Key gradient results: {key.to_dict(orient='records')}."],
            tables=["data/gradient_group_assignments.csv", "data/gradient_parameter_group_stats.csv", "data/gradient_high_low_tests.csv"],
            caveats=["Tercile gradients simplify continuous environmental variation.", "Collinearity between gradients can complicate single-gradient interpretation."],
            wording=["Use `structured gradients` and `consistent with` the heatmap associations."],
            figure_usage=["Use later for gradient contrast panels."],
        ),
        log,
    )
    write_md(methods_dir / "method_definitions.md", "Environmental Gradient Definitions", {"Definitions": ["Terciles are rank-based to keep sample sizes nearly balanced."]}, log)
    write_md(logs_dir / "environmental_gradient_log.md", "Environmental Gradient Log", {"Summary": [f"Assignment rows: {len(assign)}", f"Test rows: {len(tests)}"]}, log)

