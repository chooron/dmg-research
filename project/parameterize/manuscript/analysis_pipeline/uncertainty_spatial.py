from __future__ import annotations

import numpy as np
import pandas as pd

from .common import PRIMARY_LOSS, PipelineLog, correlation_value, report_common_sections, save_csv, write_md


def run(dirs: dict[str, dict[str, object]], context: dict[str, object], log: PipelineLog) -> None:
    block = "06_uncertainty_spatial_data"
    data_dir = dirs[block]["data"]
    reports_dir = dirs[block]["reports"]
    methods_dir = dirs[block]["methods"]
    logs_dir = dirs[block]["logs"]

    params: pd.DataFrame = context["params"]
    coords: pd.DataFrame = context["coordinates"]
    dist = params.loc[(params["model_raw"].eq("distributional")) & (params["loss"].eq(PRIMARY_LOSS))].copy()
    q05 = (dist["parameter_mean_unit"] - 1.645 * dist["parameter_std_unit"]).clip(0, 1)
    q95 = (dist["parameter_mean_unit"] + 1.645 * dist["parameter_std_unit"]).clip(0, 1)
    dist["q05_optional"] = q05
    dist["q50_optional"] = dist["parameter_mean_unit"]
    dist["q95_optional"] = q95
    dist["interval_width_90_optional"] = q95 - q05

    by_basin = (
        dist.groupby(["basin_id", "parameter", "parameter_family"], as_index=False)
        .agg(
            parameter_mean_unit=("parameter_mean_unit", "mean"),
            parameter_std_unit=("parameter_std_unit", "mean"),
            q05_optional=("q05_optional", "mean"),
            q50_optional=("q50_optional", "mean"),
            q95_optional=("q95_optional", "mean"),
            interval_width_90_optional=("interval_width_90_optional", "mean"),
            distance_to_boundary=("distance_to_boundary", "mean"),
            near_boundary_flag=("near_boundary_flag", "max"),
            seed_sd_std_unit=("parameter_std_unit", "std"),
            n_seeds=("seed", "nunique"),
        )
        .merge(coords, on="basin_id", how="left")
    )
    by_basin["mean_std_coupling"] = np.nan
    save_csv(by_basin, data_dir / "distributional_parameter_uncertainty_maps_long.csv", log)
    context["distributional_uncertainty_maps"] = by_basin

    diag_rows = []
    boundary_rows = []
    for parameter, sub in by_basin.groupby("parameter"):
        mean_std, p_mean_std, n = correlation_value(sub["parameter_mean_unit"], sub["parameter_std_unit"], "spearman")
        boundary_std, p_boundary, _ = correlation_value(sub["distance_to_boundary"], sub["parameter_std_unit"], "spearman")
        diag_rows.append({"parameter": parameter, "mean_std_spearman": mean_std, "mean_std_p_value": p_mean_std, "n_basins": n})
        boundary_rows.append({"parameter": parameter, "boundary_distance_std_spearman": boundary_std, "boundary_distance_std_p_value": p_boundary, "near_boundary_share": float(sub["near_boundary_flag"].mean()), "n_basins": n})
    diag = pd.DataFrame(diag_rows)
    boundary = pd.DataFrame(boundary_rows)
    save_csv(diag, data_dir / "mean_std_coupling_diagnostics.csv", log)
    save_csv(boundary, data_dir / "boundary_uncertainty_diagnostics.csv", log)

    summary = (
        by_basin.groupby(["parameter", "parameter_family"], as_index=False)
        .agg(
            n_basins=("basin_id", "nunique"),
            median_std_unit=("parameter_std_unit", "median"),
            iqr_std_unit=("parameter_std_unit", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            range_std_unit=("parameter_std_unit", lambda x: x.max() - x.min()),
            median_interval_width_90=("interval_width_90_optional", "median"),
        )
        .merge(diag, on="parameter", how="left")
        .merge(boundary.drop(columns=["n_basins"], errors="ignore"), on="parameter", how="left")
        .sort_values("range_std_unit", ascending=False)
    )
    save_csv(summary, data_dir / "uncertainty_summary_by_parameter.csv", log)

    write_md(
        reports_dir / "uncertainty_spatial_data_report.md",
        "Uncertainty Spatial Data Report",
        report_common_sections(
            objective="Prepare distributional parameter uncertainty map-ready tables and diagnostics without drawing maps.",
            input_files=["Deduplicated distributional parameter std fields", "Coordinates"],
            data_filters=[f"Distributional model, `{PRIMARY_LOSS}` only, seed-averaged by basin/parameter."],
            metric_definitions=["parameter_std_unit is std normalized by HBV search range.", "Boundary distance is min(value, 1-value) on normalized mean.", "q05/q95 use normal approximation and are clipped to [0,1]."],
            main_results=[f"Uncertainty ranges: {summary[['parameter','range_std_unit','median_std_unit','mean_std_spearman','boundary_distance_std_spearman']].to_dict(orient='records')}."],
            tables=["data/distributional_parameter_uncertainty_maps_long.csv", "data/uncertainty_summary_by_parameter.csv", "data/mean_std_coupling_diagnostics.csv", "data/boundary_uncertainty_diagnostics.csv"],
            caveats=["Approximate intervals are descriptive and not calibrated predictive intervals.", "High mean-std coupling or boundary sensitivity reduces interpretability."],
            wording=["Use `uncertainty shows structured gradients` only when diagnostics are not strongly boundary-sensitive."],
            figure_usage=["Use later for uncertainty spatial maps and diagnostic supplements."],
        ),
        log,
    )
    write_md(methods_dir / "method_definitions.md", "Uncertainty Spatial Definitions", {"Definitions": ["Uncertainty is represented by normalized distributional std fields."]}, log)
    write_md(logs_dir / "uncertainty_spatial_log.md", "Uncertainty Spatial Log", {"Summary": [f"Rows: {len(by_basin)}"]}, log)
