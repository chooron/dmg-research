from __future__ import annotations

import pandas as pd

from .common import REPRESENTATIVE_GRADIENTS, PipelineLog, make_tercile_assignments, mann_whitney_test, report_common_sections, save_csv, write_md


def run(dirs: dict[str, dict[str, object]], context: dict[str, object], log: PipelineLog) -> None:
    block = "08_representative_basin_groups"
    data_dir = dirs[block]["data"]
    reports_dir = dirs[block]["reports"]
    methods_dir = dirs[block]["methods"]
    logs_dir = dirs[block]["logs"]

    mean_maps: pd.DataFrame = context["distributional_mean_maps"]
    unc: pd.DataFrame = context["distributional_uncertainty_maps"]
    attrs: pd.DataFrame = context["attributes"]
    coords: pd.DataFrame = context["coordinates"]
    values = mean_maps[["basin_id", "parameter", "parameter_mean_unit"]].merge(
        unc[["basin_id", "parameter", "parameter_std_unit"]], on=["basin_id", "parameter"], how="left"
    )

    assigns = []
    for gradient in REPRESENTATIVE_GRADIENTS:
        a = make_tercile_assignments(attrs, gradient)
        a = a.loc[a["gradient_group"].isin(["low", "high"])].copy()
        a["group_label"] = a["gradient_attribute"] + "_" + a["gradient_group"].astype(str)
        assigns.append(a)
    assign = pd.concat(assigns, ignore_index=True)
    save_csv(assign, data_dir / "representative_group_assignments.csv", log)

    merged = values.merge(assign, on="basin_id", how="inner")
    global_medians = values.groupby("parameter", as_index=False).agg(global_mean_median=("parameter_mean_unit", "median"), global_std_median=("parameter_std_unit", "median"))
    profiles = (
        merged.groupby(["gradient_attribute", "group_label", "gradient_group", "parameter"], as_index=False, observed=True)
        .agg(
            sample_count=("basin_id", "nunique"),
            mean_median_unit=("parameter_mean_unit", "median"),
            mean_iqr_unit=("parameter_mean_unit", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            std_median_unit=("parameter_std_unit", "median"),
            std_iqr_unit=("parameter_std_unit", lambda x: x.quantile(0.75) - x.quantile(0.25)),
        )
        .merge(global_medians, on="parameter", how="left")
    )
    profiles["mean_difference_vs_global_median"] = profiles["mean_median_unit"] - profiles["global_mean_median"]
    profiles["std_difference_vs_global_median"] = profiles["std_median_unit"] - profiles["global_std_median"]
    wide = profiles.pivot_table(index=["gradient_attribute", "parameter"], columns="gradient_group", values="mean_median_unit", observed=True)
    profiles = profiles.merge((wide.get("high") - wide.get("low")).rename("high_minus_low_difference").reset_index(), on=["gradient_attribute", "parameter"], how="left")
    save_csv(profiles, data_dir / "representative_group_parameter_profiles.csv", log)

    tests = []
    for (gradient, parameter), sub in merged.groupby(["gradient_attribute", "parameter"]):
        low = sub.loc[sub["gradient_group"].eq("low"), "parameter_mean_unit"]
        high = sub.loc[sub["gradient_group"].eq("high"), "parameter_mean_unit"]
        test = mann_whitney_test(low, high)
        tests.append({"gradient_attribute": gradient, "parameter": parameter, "high_minus_low_difference": high.median() - low.median(), **test})
    tests_df = pd.DataFrame(tests)
    save_csv(tests_df, data_dir / "group_high_low_tests.csv", log)
    save_csv(tests_df, data_dir / "statistical_tests_representative_groups.csv", log)

    candidates = []
    attr_coord = attrs.merge(coords, on="basin_id", how="left")
    for gradient in REPRESENTATIVE_GRADIENTS:
        low = attr_coord.nsmallest(10, gradient).copy()
        high = attr_coord.nlargest(10, gradient).copy()
        for label, sub in [("low_extreme", low), ("high_extreme", high)]:
            for _, row in sub.iterrows():
                candidates.append({"gradient_attribute": gradient, "candidate_type": label, "basin_id": row["basin_id"], "gradient_value": row[gradient], "longitude": row.get("longitude"), "latitude": row.get("latitude")})
    cand = pd.DataFrame(candidates)
    save_csv(cand, data_dir / "representative_basin_candidates.csv", log)

    best_groups = tests_df.assign(abs_diff=lambda d: d["high_minus_low_difference"].abs()).sort_values("abs_diff", ascending=False).head(12)
    write_md(
        reports_dir / "representative_basin_group_report.md",
        "Representative Basin Group Report",
        report_common_sections(
            objective="Construct interpretable low/high basin archetypes for major environmental gradients.",
            input_files=["Distributional mean/std summaries", "Basin attributes", "Coordinates"],
            data_filters=["Lower and upper terciles only.", "Distributional primary-loss summaries."],
            metric_definitions=["Profiles use median and IQR for parameter means and std values.", "High-low tests use Mann-Whitney U."],
            main_results=[f"Largest group contrasts: {best_groups[['gradient_attribute','parameter','high_minus_low_difference','p_value']].to_dict(orient='records')}.", f"Candidate basins saved: {len(cand)}."],
            tables=["data/representative_group_assignments.csv", "data/representative_group_parameter_profiles.csv", "data/representative_basin_candidates.csv", "data/group_high_low_tests.csv"],
            caveats=["Extreme basin candidates are illustrative examples, not independent evidence.", "Groups overlap across gradients."],
            wording=["Use group profiles to support gradient interpretation, not to claim universal basin archetypes."],
            figure_usage=["Use later for basin contrast panels or case-example annotations."],
        ),
        log,
    )
    write_md(methods_dir / "method_definitions.md", "Representative Basin Group Definitions", {"Definitions": ["Low/high groups are rank-balanced terciles by each gradient attribute."]}, log)
    write_md(logs_dir / "representative_basin_group_log.md", "Representative Basin Group Log", {"Summary": [f"Assignments: {len(assign)}", f"Candidates: {len(cand)}"]}, log)

