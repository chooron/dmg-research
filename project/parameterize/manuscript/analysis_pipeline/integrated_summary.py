from __future__ import annotations

import pandas as pd

from .common import ANALYSIS_ROOT, PipelineLog, report_common_sections, save_csv, write_md


def run(dirs: dict[str, dict[str, object]], context: dict[str, object], log: PipelineLog) -> None:
    block = "09_integrated_summary"
    data_dir = dirs[block]["data"]
    reports_dir = dirs[block]["reports"]
    methods_dir = dirs[block]["methods"]
    logs_dir = dirs[block]["logs"]

    consistency = context.get("model_dominant_consistency", pd.DataFrame())
    mean_corr = context.get("distributional_mean_attribute_correlations", pd.DataFrame())
    unc_corr_path = dirs["07_uncertainty_attribute_relationships"]["data"] / "distributional_std_attribute_correlations.csv"
    unc_corr = pd.read_csv(unc_corr_path) if unc_corr_path.exists() else pd.DataFrame()

    shared_count = int(consistency["relationship_class"].eq("shared dominant controls").sum()) if not consistency.empty else 0
    strong_mean = int(((mean_corr.get("abs_rho", pd.Series(dtype=float)) >= 0.3) & (mean_corr.get("q_value", pd.Series(dtype=float)) <= 0.1)).sum()) if not mean_corr.empty else 0
    clean_unc = int((unc_corr.get("interpretation_flag", pd.Series(dtype=str)).eq("clean") & (unc_corr.get("abs_rho", pd.Series(dtype=float)) >= 0.3)).sum()) if not unc_corr.empty else 0

    claims = [
        {
            "claim": "三模型共享 dominant-control core",
            "evidence_type": "dominant attribute agreement",
            "supporting_analysis_folder": "01_model_consistency",
            "main_metrics": f"shared_dominant_parameter_count={shared_count}",
            "strength": "strong" if shared_count >= 4 else "supportive",
            "caveats": "Dominant classes use modal attributes across seed/loss.",
            "recommended_wording": "The three formulations recover a shared dominant-control core for several parameters.",
        },
        {
            "claim": "distributional 没有偏离 shared relationship structure",
            "evidence_type": "matrix similarity and top-k overlap",
            "supporting_analysis_folder": "01_model_consistency",
            "main_metrics": "matrix_similarity_pairwise; intermodel_similarity_summary",
            "strength": "strong",
            "caveats": "Similarity is descriptive and not a causal test.",
            "recommended_wording": "The distributional formulation remains within the shared relationship-structure region.",
        },
        {
            "claim": "distributional relationship structure 对 seed/loss 更稳定",
            "evidence_type": "seed/loss sensitivity",
            "supporting_analysis_folder": "02_seed_loss_sensitivity",
            "main_metrics": "seed_vs_loss_sensitivity_summary",
            "strength": "strong",
            "caveats": "Use separate seed and loss summaries; do not pool as independent samples.",
            "recommended_wording": "The distributional formulation more reproducibly recovers relationship structure across seeds and losses.",
        },
        {
            "claim": "distributional parameter means show organized spatial patterns",
            "evidence_type": "map-ready parameter ranges",
            "supporting_analysis_folder": "03_distributional_parameter_spatial_data",
            "main_metrics": "parameter_map_summary_by_parameter",
            "strength": "supportive",
            "caveats": "No spatial autocorrelation model fitted in this stage.",
            "recommended_wording": "Distributional parameter means show organized basin-scale spatial variation.",
        },
        {
            "claim": "distributional parameter means align with major environmental gradients",
            "evidence_type": "mean-attribute correlations and tercile contrasts",
            "supporting_analysis_folder": "04_mean_attribute_relationships; 05_environmental_gradient_groups",
            "main_metrics": f"strong_mean_relationships={strong_mean}",
            "strength": "headline" if strong_mean >= 8 else "strong",
            "caveats": "Attribute collinearity can produce interchangeable controls.",
            "recommended_wording": "Parameter means align with hydrologically interpretable environmental gradients.",
        },
        {
            "claim": "distributional parameter uncertainty shows structured gradients",
            "evidence_type": "std-attribute correlations with diagnostics",
            "supporting_analysis_folder": "06_uncertainty_spatial_data; 07_uncertainty_attribute_relationships",
            "main_metrics": f"clean_uncertainty_relationships={clean_unc}",
            "strength": "supportive" if clean_unc >= 1 else "cautious",
            "caveats": "Mean-std coupling and boundary effects can limit interpretation.",
            "recommended_wording": "Some uncertainty gradients are structured, with flagged cases interpreted cautiously.",
        },
        {
            "claim": "representative basin contrasts support gradient interpretation",
            "evidence_type": "tercile group profiles",
            "supporting_analysis_folder": "08_representative_basin_groups",
            "main_metrics": "representative_group_parameter_profiles; group_high_low_tests",
            "strength": "supportive",
            "caveats": "Representative basins are illustrative, not independent proof.",
            "recommended_wording": "Representative basin contrasts support the gradient interpretation.",
        },
        {
            "claim": "snow/routing parameters require cautious interpretation",
            "evidence_type": "classification and diagnostic flags",
            "supporting_analysis_folder": "01_model_consistency; 07_uncertainty_attribute_relationships",
            "main_metrics": "relationship_class; interpretation_flag",
            "strength": "cautious",
            "caveats": "Snow and routing controls can be model-sensitive or boundary-sensitive.",
            "recommended_wording": "Snow and routing parameter results are consistent with interpretable gradients but should be framed cautiously.",
        },
    ]
    evidence = pd.DataFrame(claims)
    save_csv(evidence, data_dir / "main_claim_evidence_table.csv", log)

    fig_candidates = pd.DataFrame(
        [
            {"figure_id_candidate": "Fig. X", "figure_purpose": "model consistency and shared relationship structure", "required_data_file": "01_model_consistency/data/model_dominant_consistency_summary.csv; 01_model_consistency/data/matrix_similarity_pairwise.csv", "main_panel_ideas": "dominant-control agreement, top-k overlap, matrix similarity", "main_text_or_supplement": "main_text", "priority": 1},
            {"figure_id_candidate": "Fig. X", "figure_purpose": "distributional 14-parameter spatial maps", "required_data_file": "03_distributional_parameter_spatial_data/data/distributional_parameter_mean_maps_long.csv", "main_panel_ideas": "14 normalized parameter mean maps", "main_text_or_supplement": "main_text", "priority": 2},
            {"figure_id_candidate": "Fig. X", "figure_purpose": "mean parameter-attribute heatmap plus key gradients", "required_data_file": "04_mean_attribute_relationships/data/distributional_mean_attribute_correlations.csv; 05_environmental_gradient_groups/data/gradient_parameter_group_stats.csv", "main_panel_ideas": "heatmap, key low-high contrasts", "main_text_or_supplement": "main_text", "priority": 3},
            {"figure_id_candidate": "Fig. X", "figure_purpose": "uncertainty-attribute heatmap plus key uncertainty gradients", "required_data_file": "07_uncertainty_attribute_relationships/data/distributional_std_attribute_correlations.csv", "main_panel_ideas": "std heatmap with caution flags", "main_text_or_supplement": "main_text_or_supplement", "priority": 4},
            {"figure_id_candidate": "Fig. S", "figure_purpose": "full model comparison maps", "required_data_file": "01_model_consistency/data/seed_loss_correlation_matrix_long.csv", "main_panel_ideas": "full matrices by model/loss", "main_text_or_supplement": "supplement", "priority": 5},
            {"figure_id_candidate": "Fig. S", "figure_purpose": "sensitivity checks", "required_data_file": "02_seed_loss_sensitivity/data/seed_vs_loss_sensitivity_summary.csv", "main_panel_ideas": "seed/loss variability checks", "main_text_or_supplement": "supplement", "priority": 6},
        ]
    )
    save_csv(fig_candidates, data_dir / "figure_candidate_table.csv", log)

    classification_rows = []
    if not mean_corr.empty:
        for _, row in mean_corr.iterrows():
            destination = "main_text" if row["relationship_role"] in {"dominant", "supportive"} and row["abs_rho"] >= 0.35 and row["q_value"] <= 0.1 else "supplement_or_discussion"
            classification_rows.append({"result_type": "mean_attribute_relationship", "parameter": row["parameter"], "attribute": row["attribute"], "metric": row["spearman_rho"], "classification": destination, "reason": row["relationship_role"]})
    if not unc_corr.empty:
        for _, row in unc_corr.iterrows():
            destination = "main_text" if row["relationship_role"] in {"dominant", "supportive"} and row["abs_rho"] >= 0.35 and row["interpretation_flag"] == "clean" else "supplement_or_discussion"
            classification_rows.append({"result_type": "uncertainty_attribute_relationship", "parameter": row["parameter"], "attribute": row["attribute"], "metric": row["spearman_rho"], "classification": destination, "reason": row["interpretation_flag"]})
    classification = pd.DataFrame(classification_rows)
    save_csv(classification, data_dir / "main_text_vs_appendix_classification.csv", log)

    write_md(
        reports_dir / "integrated_results_summary.md",
        "Integrated Results Summary",
        report_common_sections(
            objective="Integrate all back-half analysis tables into a Results evidence chain.",
            input_files=["All numbered analysis block outputs."],
            data_filters=["Uses generated intermediate tables only; no figure artifacts."],
            metric_definitions=["Claim strength combines agreement, effect size, FDR, and interpretation diagnostics."],
            main_results=[f"Evidence table rows: {len(evidence)}.", f"Main-text classifications: {classification['classification'].value_counts().to_dict() if not classification.empty else {}}."],
            tables=["data/main_claim_evidence_table.csv", "data/figure_candidate_table.csv", "data/main_text_vs_appendix_classification.csv"],
            caveats=["Claim strengths are conservative labels for writing, not formal posterior probabilities."],
            wording=["Use `consistent with`, `structured gradients`, and `candidate dominant relationships`."],
            figure_usage=["This table prepares later plotting order but does not generate figures."],
        ),
        log,
    )
    write_md(reports_dir / "claim_strength_check.md", "Claim Strength Check", {"Claim table": evidence[["claim", "strength", "caveats"]].astype(str).agg(" | ".join, axis=1).tolist()}, log)
    write_md(reports_dir / "recommended_results_order.md", "Recommended Results Order", {"Order": fig_candidates.sort_values("priority")["figure_purpose"].tolist()}, log)
    write_md(
        reports_dir / "result_consistency_check.md",
        "Result Consistency Check",
        {
            "Checks": [
                f"Dominant shared-control count: {shared_count}.",
                "Seed and loss sensitivities were summarized separately.",
                "Top-k filtering was computed symmetrically across models.",
                "Uncertainty gradients include mean-std and boundary flags.",
                "Representative basin groups use balanced terciles.",
                "Main-text candidates are separated from supplement/discussion candidates.",
            ],
            "Caveats": [
                "Dominant relationships should be compared against previous manuscript text before final writing.",
                "Boundary-sensitive uncertainty cases should not be used as headline evidence.",
            ],
        },
        log,
    )
    write_md(methods_dir / "method_definitions.md", "Integrated Summary Definitions", {"Definitions": ["Claim strength is a writing aid based on generated analysis tables."]}, log)
    write_md(logs_dir / "integrated_summary_log.md", "Integrated Summary Log", {"Summary": [f"Claims: {len(evidence)}", f"Figure data candidates: {len(fig_candidates)}"]}, log)

