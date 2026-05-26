from __future__ import annotations

import pandas as pd

from .common import PRIMARY_LOSS, PipelineLog, report_common_sections, save_csv, write_md


def run(dirs: dict[str, dict[str, object]], context: dict[str, object], log: PipelineLog) -> None:
    block = "03_distributional_parameter_spatial_data"
    data_dir = dirs[block]["data"]
    reports_dir = dirs[block]["reports"]
    methods_dir = dirs[block]["methods"]
    logs_dir = dirs[block]["logs"]

    params: pd.DataFrame = context["params"]
    coords: pd.DataFrame = context["coordinates"]
    dist = params.loc[params["model_raw"].eq("distributional")].copy()

    seed_summary = (
        dist.groupby(["loss", "basin_id", "parameter", "parameter_family"], as_index=False)
        .agg(
            parameter_mean_unit=("parameter_mean_unit", "mean"),
            parameter_mean_physical=("mean", "mean"),
            seed_mean=("parameter_mean_unit", "mean"),
            seed_sd=("parameter_mean_unit", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    all_loss = (
        seed_summary.groupby(["basin_id", "parameter", "parameter_family"], as_index=False)
        .agg(loss_mean_optional=("parameter_mean_unit", "mean"), loss_sd_optional=("parameter_mean_unit", "std"), n_losses=("loss", "nunique"))
    )
    primary = seed_summary.loc[seed_summary["loss"].eq(PRIMARY_LOSS)].merge(all_loss, on=["basin_id", "parameter", "parameter_family"], how="left")
    primary = primary.merge(coords, on="basin_id", how="left")
    primary = primary[
        [
            "basin_id",
            "longitude",
            "latitude",
            "parameter",
            "parameter_family",
            "parameter_mean_unit",
            "seed_mean",
            "seed_sd",
            "loss_mean_optional",
            "loss_sd_optional",
            "n_seeds",
            "n_losses",
        ]
    ]
    save_csv(primary, data_dir / "distributional_parameter_mean_maps_long.csv", log)
    context["distributional_mean_maps"] = primary

    summary = (
        primary.groupby(["parameter", "parameter_family"], as_index=False)
        .agg(
            n_basins=("basin_id", "nunique"),
            min_unit=("parameter_mean_unit", "min"),
            q25_unit=("parameter_mean_unit", lambda x: x.quantile(0.25)),
            median_unit=("parameter_mean_unit", "median"),
            q75_unit=("parameter_mean_unit", lambda x: x.quantile(0.75)),
            max_unit=("parameter_mean_unit", "max"),
            range_unit=("parameter_mean_unit", lambda x: x.max() - x.min()),
            sd_unit=("parameter_mean_unit", "std"),
            missing_coordinates=("longitude", lambda x: int(x.isna().sum())),
        )
        .sort_values("range_unit", ascending=False)
    )
    save_csv(summary, data_dir / "parameter_map_summary_by_parameter.csv", log)

    write_md(
        reports_dir / "parameter_spatial_data_report.md",
        "Distributional Parameter Spatial Data Report",
        report_common_sections(
            objective="Prepare map-ready distributional parameter mean data for all 14 HBV parameters without drawing maps.",
            input_files=["Deduplicated parameter table", "CAMELS coordinate shapefile"],
            data_filters=[f"Distributional model only.", f"Primary table uses `{PRIMARY_LOSS}` seed-averaged means."],
            metric_definitions=["parameter_mean_unit is normalized to [0,1] search-space scale.", "seed_sd is cross-seed SD under the primary loss."],
            main_results=[
                f"Rows saved: {len(primary)}.",
                f"Parameter ranges by normalized value: {summary[['parameter','range_unit','n_basins']].to_dict(orient='records')}.",
                "Unified [0,1] color scales are valid for normalized values; independent min-max scales emphasize within-parameter contrast but limit cross-parameter magnitude comparison.",
            ],
            tables=["data/distributional_parameter_mean_maps_long.csv", "data/parameter_map_summary_by_parameter.csv"],
            caveats=["Spatial structure is summarized numerically here; no spatial autocorrelation model is fit in this stage."],
            wording=["Distributional parameter means show structured spatial gradients when ranges differ coherently across basins."],
            figure_usage=["Use the long table later for 14-parameter map panels."],
        ),
        log,
    )
    write_md(methods_dir / "method_definitions.md", "Spatial Parameter Mean Definitions", {"Definitions": ["Seed-averaged normalized distributional parameter means under the primary loss."]}, log)
    write_md(logs_dir / "parameter_spatial_data_log.md", "Parameter Spatial Data Log", {"Summary": [f"Rows: {len(primary)}"]}, log)

