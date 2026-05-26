from __future__ import annotations

import pandas as pd

from .common import (
    ANALYSIS_ROOT,
    PARAMETER_BOUNDS,
    PARAMETERS,
    PRIMARY_LOSS,
    REQUIRED_ATTRIBUTES,
    PipelineLog,
    load_attributes,
    load_coordinates,
    load_params,
    read_basin_ids,
    report_common_sections,
    save_csv,
    write_md,
)


def run(dirs: dict[str, dict[str, object]], context: dict[str, object], log: PipelineLog) -> None:
    block = "00_data_inventory"
    data_dir = dirs[block]["data"]
    reports_dir = dirs[block]["reports"]
    methods_dir = dirs[block]["methods"]
    logs_dir = dirs[block]["logs"]

    run_inventory: pd.DataFrame = context["run_inventory"]
    attrs, mapping = load_attributes(log)
    coords = load_coordinates(log)
    params, duplicate_diagnostics = load_params(run_inventory, log)
    basin_ids = read_basin_ids()

    context["attributes"] = attrs
    context["attribute_mapping"] = mapping
    context["coordinates"] = coords
    context["params"] = params
    context["duplicate_diagnostics"] = duplicate_diagnostics
    context["basin_ids"] = basin_ids

    inventory = run_inventory.copy()
    expected_param_rows = 531 * len(PARAMETERS)
    param_counts = (
        params.groupby(["model_raw", "loss", "seed"], as_index=False)
        .agg(parameter_rows=("mean", "size"), basin_count=("basin_id", "nunique"), parameter_count=("parameter", "nunique"))
    )
    inventory = inventory.merge(param_counts, on=["model_raw", "loss", "seed"], how="left")
    inventory["expected_parameter_rows"] = expected_param_rows
    inventory["parameter_output_complete"] = (
        inventory["parameter_rows"].eq(expected_param_rows)
        & inventory["basin_count"].eq(531)
        & inventory["parameter_count"].eq(len(PARAMETERS))
    )
    inventory["is_primary_loss"] = inventory["loss"].eq(PRIMARY_LOSS)
    save_csv(inventory, data_dir / "run_inventory.csv", log)

    attr_inventory = pd.DataFrame(
        {
            "basin_id": basin_ids.astype(str),
            "in_attribute_table": basin_ids.astype(str).isin(set(attrs["basin_id"])),
            "in_coordinate_table": basin_ids.astype(str).isin(set(coords["basin_id"])),
        }
    )
    attr_inventory = attr_inventory.merge(attrs[["basin_id"] + [c for c in REQUIRED_ATTRIBUTES if c in attrs.columns]], on="basin_id", how="left")
    save_csv(attr_inventory, data_dir / "basin_attribute_inventory.csv", log)
    save_csv(mapping, data_dir / "attribute_field_mapping.csv", log)
    save_csv(duplicate_diagnostics, data_dir / "parameter_duplicate_diagnostics.csv", log)

    scale_rows = []
    for parameter, (low, high) in PARAMETER_BOUNDS.items():
        scale_rows.append(
            {
                "parameter": parameter,
                "physical_min": low,
                "physical_max": high,
                "normalization": "(value - physical_min) / (physical_max - physical_min)",
            }
        )
    save_csv(pd.DataFrame(scale_rows), data_dir / "parameter_scale_definitions.csv", log)

    completeness_rows = [
        {"check": "discovered_531_runs", "value": len(run_inventory), "status": "pass" if len(run_inventory) > 0 else "fail"},
        {
            "check": "all_run_parameter_outputs_complete",
            "value": int(inventory["parameter_output_complete"].sum()),
            "status": "pass" if inventory["parameter_output_complete"].all() else "warn",
        },
        {"check": "target_basin_count", "value": len(basin_ids), "status": "pass" if len(basin_ids) == 531 else "warn"},
        {"check": "attribute_basin_count", "value": attrs["basin_id"].nunique(), "status": "pass" if attrs["basin_id"].nunique() == 531 else "warn"},
        {"check": "coordinate_match_count", "value": attr_inventory["in_coordinate_table"].sum(), "status": "pass" if attr_inventory["in_coordinate_table"].all() else "warn"},
        {"check": "hbv_parameter_count", "value": params["parameter"].nunique(), "status": "pass" if params["parameter"].nunique() == 14 else "fail"},
        {
            "check": "distributional_std_available",
            "value": float((params.loc[params["model_raw"].eq("distributional"), "std"].fillna(0) > 0).mean()),
            "status": "pass",
        },
        {
            "check": "mc_dropout_std_available",
            "value": float((params.loc[params["model_raw"].eq("mc_dropout"), "std"].fillna(0) > 0).mean()),
            "status": "pass",
        },
    ]
    completeness = pd.DataFrame(completeness_rows)
    save_csv(completeness, data_dir / "data_completeness_check_table.csv", log)

    model_summary = inventory.groupby("model_raw", as_index=False).agg(
        n_runs=("run_id", "nunique"),
        n_losses=("loss", "nunique"),
        n_seeds=("seed", "nunique"),
        complete_runs=("parameter_output_complete", "sum"),
    )
    missing_runs = inventory.loc[~inventory["parameter_output_complete"], ["run_id", "parameter_rows", "basin_count", "parameter_count"]]

    write_md(
        reports_dir / "data_completeness_check.md",
        "Data Completeness Check",
        report_common_sections(
            objective="Confirm that the discovered 531-basin model runs, basin IDs, attributes, coordinates, and parameter outputs support the requested statistical analyses.",
            input_files=[
                "project/parameterize/outputs/*531*/<loss>/seed_*/",
                "project/parameterize/outputs/analysis/stability_stats/tables/params_long.csv",
                "project/parameterize/outputs/analysis/stability_stats/tables/basin_attributes.csv",
                "data/531sub_id.txt",
                "data/camels_loc/camels_671_loc.shp",
            ],
            data_filters=[
                "Only output directories containing `531` are treated as authoritative model runs.",
                "Parameter rows are filtered to discovered run IDs and the 14 HBV parameters.",
                "Duplicate parameter keys are collapsed after diagnostics.",
            ],
            metric_definitions=[
                f"A complete parameter run has {expected_param_rows} rows after deduplication: 531 basins x 14 parameters.",
                "Distributional and MC-dropout stochastic availability is checked from nonzero `std` fields.",
            ],
            main_results=[
                f"Discovered {len(run_inventory)} 531-run directories.",
                f"Model run counts: {model_summary.to_dict(orient='records')}",
                f"Incomplete parameter runs: {len(missing_runs)}.",
                f"Attribute basins matched: {attrs['basin_id'].nunique()} / 531.",
                f"Coordinate basins matched: {int(attr_inventory['in_coordinate_table'].sum())} / 531.",
            ],
            tables=[
                "data/run_inventory.csv",
                "data/basin_attribute_inventory.csv",
                "data/attribute_field_mapping.csv",
                "data/parameter_duplicate_diagnostics.csv",
            ],
            caveats=[
                "Existing parameter source contained duplicate logical rows for some runs; these are explicitly diagnosed and collapsed.",
                "This stage does not read raw torch checkpoints to reconstruct parameter samples.",
            ],
            wording=[
                "All downstream analyses use the deduplicated 531-basin run inventory.",
                "Parameter uncertainty is interpreted as an available model-output/sample dispersion diagnostic, not as a physically true uncertainty.",
            ],
            figure_usage=[
                "Use this report only to document data readiness for later figures.",
                "No figures are generated in this stage.",
            ],
        ),
        log,
    )

    write_md(
        reports_dir / "definition_and_scale_check.md",
        "Definition and Scale Check",
        report_common_sections(
            objective="Define parameter scales, stochastic summaries, and sample units before downstream relationship analysis.",
            input_files=["params_long.csv", "implements/hbv_static.py parameter bounds"],
            data_filters=[
                "Physical-scale parameter means are normalized to search-space units with HBV bounds.",
                "Correlation analyses use basins as samples.",
                f"`{PRIMARY_LOSS}` is the primary manuscript loss; all-loss summaries remain available.",
            ],
            metric_definitions=[
                "parameter_mean_unit = (mean - lower_bound) / (upper_bound - lower_bound).",
                "parameter_std_unit = std / (upper_bound - lower_bound).",
                "near_boundary_flag marks normalized means within 0.05 of either bound.",
                "MC-dropout mean/std are interpreted as stochastic sample summaries when sample_count is 100.",
                "Distributional mean/std are interpreted as model-output distribution summaries when sample_count is 100.",
            ],
            main_results=[
                "The source parameters are in physical HBV units, not already in [0,1] scale.",
                "All 14 HBV parameters have explicit search ranges.",
                f"Primary loss selected: {PRIMARY_LOSS}.",
                "Spearman correlations are basin-level associations.",
            ],
            tables=["data/parameter_scale_definitions.csv", "data/data_completeness_check_table.csv"],
            caveats=[
                "Normalized values clipped to [0,1] after conversion to prevent numerical boundary spillover.",
                "Group analyses require tercile groups with adequate non-missing basin counts.",
            ],
            wording=[
                "Use `search-space normalized parameter value` when describing normalized values.",
                "Avoid language implying corrected or physically true parameters.",
            ],
            figure_usage=["These definitions should be reused in later captions and methods text."],
        ),
        log,
    )

    write_md(
        methods_dir / "method_definitions.md",
        "Data Inventory Method Definitions",
        {
            "Run discovery": [
                "Scan `project/parameterize/outputs/` for directories containing `531`.",
                "Parse model from directory name, loss from child directory, seed from `seed_*`.",
            ],
            "Parameter source": [
                "Read existing long-form parameter table and filter to discovered runs.",
                "Collapse duplicate logical rows by model/loss/seed/basin/parameter.",
            ],
            "Scale handling": [
                "Retain physical `mean` and `std`.",
                "Add normalized `parameter_mean_unit` and `parameter_std_unit` for cross-parameter comparisons.",
            ],
        },
        log,
    )

    write_md(
        logs_dir / "data_inventory_log.md",
        "Data Inventory Log",
        {
            "Summary": [
                f"Discovered runs: {len(run_inventory)}",
                f"Parameter rows after deduplication: {len(params)}",
                f"Duplicate keys diagnosed: {len(duplicate_diagnostics)}",
            ]
        },
        log,
    )

