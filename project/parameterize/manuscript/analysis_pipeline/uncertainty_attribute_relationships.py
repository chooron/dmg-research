from __future__ import annotations

import numpy as np
import pandas as pd

from .common import KEY_STD_PAIRS, PipelineLog, correlation_value, fdr_bh, report_common_sections, save_csv, sign_label, write_md


def _interpretation_flag(abs_mean_std: float, abs_boundary: float, near_share: float) -> str:
    flags = []
    if pd.notna(abs_mean_std) and abs_mean_std >= 0.5:
        flags.append("mean-coupled")
    if (pd.notna(abs_boundary) and abs_boundary >= 0.4) or (pd.notna(near_share) and near_share >= 0.25):
        flags.append("boundary-sensitive")
    if not flags:
        return "clean"
    if len(flags) == 2:
        return "interpret with caution"
    return flags[0]


def run(dirs: dict[str, dict[str, object]], context: dict[str, object], log: PipelineLog) -> None:
    block = "07_uncertainty_attribute_relationships"
    data_dir = dirs[block]["data"]
    reports_dir = dirs[block]["reports"]
    methods_dir = dirs[block]["methods"]
    logs_dir = dirs[block]["logs"]

    unc: pd.DataFrame = context["distributional_uncertainty_maps"]
    attrs: pd.DataFrame = context["attributes"]
    attr_cols = [c for c in context["attribute_mapping"]["mapped_attribute"].dropna().unique() if c in attrs.columns]
    merged = unc.merge(attrs[["basin_id"] + attr_cols], on="basin_id", how="inner")
    diag = pd.read_csv(dirs["06_uncertainty_spatial_data"]["data"] / "mean_std_coupling_diagnostics.csv")
    boundary = pd.read_csv(dirs["06_uncertainty_spatial_data"]["data"] / "boundary_uncertainty_diagnostics.csv")
    diag_all = diag.merge(boundary, on=["parameter", "n_basins"], how="outer")

    rows = []
    for parameter, sub in merged.groupby("parameter"):
        for attr in attr_cols:
            rho, p, n = correlation_value(sub["parameter_std_unit"], sub[attr], "spearman")
            rows.append({"parameter": parameter, "attribute": attr, "spearman_rho": rho, "p_value": p, "abs_rho": abs(rho) if pd.notna(rho) else np.nan, "sign": sign_label(rho), "n_basins": n})
    corr = pd.DataFrame(rows)
    corr["q_value"] = fdr_bh(corr["p_value"])
    corr["rank_abs_rho"] = corr.groupby("parameter")["abs_rho"].rank(method="first", ascending=False)
    corr["relationship_role"] = np.where(corr["rank_abs_rho"].eq(1), "dominant", np.where((corr["abs_rho"] >= 0.3) | (corr["rank_abs_rho"] <= 5), "supportive", "weak"))
    corr = corr.merge(diag_all, on=["parameter", "n_basins"], how="left")
    corr["interpretation_flag"] = corr.apply(lambda r: _interpretation_flag(abs(r.get("mean_std_spearman", np.nan)), abs(r.get("boundary_distance_std_spearman", np.nan)), r.get("near_boundary_share", np.nan)), axis=1)
    save_csv(corr, data_dir / "distributional_std_attribute_correlations.csv", log)

    dom = corr.loc[corr["relationship_role"].eq("dominant")].copy()
    save_csv(dom, data_dir / "std_dominant_relationships.csv", log)
    flags = corr.loc[(corr["relationship_role"].isin(["dominant", "supportive"])) & (corr["abs_rho"] >= 0.3)].copy()
    flags["headline_candidate"] = flags["interpretation_flag"].eq("clean") & (flags["q_value"] <= 0.1)
    save_csv(flags, data_dir / "std_headline_gradient_flags.csv", log)
    save_csv(corr[["parameter", "attribute", "spearman_rho", "p_value", "q_value", "relationship_role", "interpretation_flag"]], data_dir / "statistical_tests_std_attribute_correlations.csv", log)

    key = corr.merge(pd.DataFrame(KEY_STD_PAIRS, columns=["parameter", "attribute"]), on=["parameter", "attribute"], how="inner")
    write_md(
        reports_dir / "uncertainty_attribute_relationship_report.md",
        "Uncertainty Attribute Relationship Report",
        report_common_sections(
            objective="Assess whether distributional parameter uncertainty follows basin attribute gradients.",
            input_files=["06_uncertainty_spatial_data uncertainty map table", "Basin attributes"],
            data_filters=["Distributional primary-loss seed-averaged std values.", "Basin-level complete cases."],
            metric_definitions=["Spearman rho relates parameter_std_unit to each attribute.", "Interpretation flags combine mean-std coupling and boundary diagnostics."],
            main_results=[
                f"Dominant std relationships: {dom[['parameter','attribute','spearman_rho','q_value','interpretation_flag']].to_dict(orient='records')}.",
                f"Key std relationships: {key[['parameter','attribute','spearman_rho','q_value','interpretation_flag']].to_dict(orient='records')}.",
            ],
            tables=["data/distributional_std_attribute_correlations.csv", "data/std_dominant_relationships.csv", "data/std_headline_gradient_flags.csv", "data/statistical_tests_std_attribute_correlations.csv"],
            caveats=["Mean-coupled or boundary-sensitive gradients should be framed cautiously.", "Routing parameter std patterns are diagnostic unless strong and clean."],
            wording=["Use `uncertainty gradients are structured` for clean/supportive rows; use `interpret with caution` for flagged rows."],
            figure_usage=["Use later for uncertainty heatmap and flagged supplement table."],
        ),
        log,
    )
    write_md(methods_dir / "method_definitions.md", "Uncertainty Attribute Definitions", {"Definitions": ["Relationship roles mirror mean-attribute analysis but use parameter std."]}, log)
    write_md(logs_dir / "uncertainty_attribute_log.md", "Uncertainty Attribute Log", {"Summary": [f"Rows: {len(corr)}", f"Headline candidates: {int(flags['headline_candidate'].sum()) if not flags.empty else 0}"]}, log)

