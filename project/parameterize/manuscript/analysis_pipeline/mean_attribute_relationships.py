from __future__ import annotations

import numpy as np
import pandas as pd

from .common import KEY_MEAN_PAIRS, PipelineLog, correlation_value, fdr_bh, report_common_sections, save_csv, sign_label, write_md


def run(dirs: dict[str, dict[str, object]], context: dict[str, object], log: PipelineLog) -> None:
    block = "04_mean_attribute_relationships"
    data_dir = dirs[block]["data"]
    reports_dir = dirs[block]["reports"]
    methods_dir = dirs[block]["methods"]
    logs_dir = dirs[block]["logs"]

    mean_maps: pd.DataFrame = context["distributional_mean_maps"]
    attrs: pd.DataFrame = context["attributes"]
    attr_cols = [c for c in context["attribute_mapping"]["mapped_attribute"].dropna().unique() if c in attrs.columns]
    merged = mean_maps.merge(attrs[["basin_id"] + attr_cols], on="basin_id", how="inner")
    rows = []
    for parameter, sub in merged.groupby("parameter"):
        for attr in attr_cols:
            rho, p, n = correlation_value(sub["parameter_mean_unit"], sub[attr], "spearman")
            pear, pear_p, _ = correlation_value(sub["parameter_mean_unit"], sub[attr], "pearson")
            tau, tau_p, _ = correlation_value(sub["parameter_mean_unit"], sub[attr], "kendall")
            rows.append({"parameter": parameter, "attribute": attr, "spearman_rho": rho, "pearson_r_optional": pear, "kendall_tau_optional": tau, "p_value": p, "abs_rho": abs(rho) if pd.notna(rho) else np.nan, "sign": sign_label(rho), "n_basins": n})
    corr = pd.DataFrame(rows)
    corr["q_value"] = fdr_bh(corr["p_value"])
    corr["rank_abs_rho"] = corr.groupby("parameter")["abs_rho"].rank(method="first", ascending=False)
    corr["relationship_role"] = np.where(corr["rank_abs_rho"].eq(1), "dominant", np.where((corr["abs_rho"] >= 0.3) | (corr["rank_abs_rho"] <= 5), "supportive", "weak"))
    save_csv(corr, data_dir / "distributional_mean_attribute_correlations.csv", log)
    context["distributional_mean_attribute_correlations"] = corr

    dominant = corr.loc[corr["relationship_role"].eq("dominant")].copy()
    save_csv(dominant, data_dir / "distributional_dominant_mean_relationships.csv", log)
    selected = corr.merge(pd.DataFrame(KEY_MEAN_PAIRS, columns=["parameter", "attribute"]), on=["parameter", "attribute"], how="inner")
    save_csv(selected, data_dir / "selected_key_relationships.csv", log)
    save_csv(corr[["parameter", "attribute", "spearman_rho", "p_value", "q_value", "relationship_role"]], data_dir / "statistical_tests_mean_attribute_correlations.csv", log)

    consistency = context.get("model_dominant_consistency", pd.DataFrame())
    shared = dominant.merge(consistency[["parameter", "relationship_class"]] if not consistency.empty else pd.DataFrame(columns=["parameter", "relationship_class"]), on="parameter", how="left")
    write_md(
        reports_dir / "mean_attribute_relationship_report.md",
        "Distributional Mean Attribute Relationship Report",
        report_common_sections(
            objective="Quantify distributional parameter mean associations with basin attributes.",
            input_files=["03_distributional_parameter_spatial_data/data/distributional_parameter_mean_maps_long.csv", "Basin attributes"],
            data_filters=["Distributional primary-loss seed-averaged parameter means.", "Basin-level complete cases per attribute pair."],
            metric_definitions=["Dominant is max absolute Spearman rho per parameter.", "Supportive is abs(rho) >= 0.3 or top-5 within parameter.", "FDR q-values use Benjamini-Hochberg."],
            main_results=[
                f"Strongest relationships: {corr.sort_values('abs_rho', ascending=False).head(10)[['parameter','attribute','spearman_rho','q_value','relationship_role']].to_dict(orient='records')}.",
                f"Dominant relationships with shared-structure context: {shared[['parameter','attribute','spearman_rho','relationship_class']].to_dict(orient='records')}.",
            ],
            tables=["data/distributional_mean_attribute_correlations.csv", "data/distributional_dominant_mean_relationships.csv", "data/selected_key_relationships.csv", "data/statistical_tests_mean_attribute_correlations.csv"],
            caveats=["Attribute collinearity can make multiple environmental controls statistically interchangeable.", "These are hydrologically interpretable associations, not causal proof."],
            wording=["Use `candidate dominant relationships` and `consistent with major environmental gradients`."],
            figure_usage=["Use later for mean parameter-attribute heatmap and key gradient panels."],
        ),
        log,
    )
    write_md(methods_dir / "method_definitions.md", "Mean Attribute Relationship Definitions", {"Definitions": ["Spearman correlations use 531 basins as samples where available."]}, log)
    write_md(logs_dir / "mean_attribute_relationship_log.md", "Mean Attribute Relationship Log", {"Summary": [f"Rows: {len(corr)}"]}, log)

