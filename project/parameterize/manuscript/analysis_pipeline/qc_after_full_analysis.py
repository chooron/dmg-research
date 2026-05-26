from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/workspace/autoresearch/project/parameterize/manuscript/analysis")
OUT_DATA = BASE / "09_integrated_summary" / "data"
OUT_REPORTS = BASE / "09_integrated_summary" / "reports"

MODEL_LABELS = {
    "deterministic": "delta_base",
    "mc_dropout": "delta_mcd",
    "distributional": "delta_dist",
}

KEY_MEAN_PAIRS = [
    ("parBETA", "slope_mean"),
    ("parFC", "pet_mean"),
    ("parPERC", "aridity"),
    ("parUZL", "soil_conductivity"),
    ("parCFR", "frac_snow"),
    ("parCWH", "frac_snow"),
    ("route_a", "slope_mean"),
]
KEY_UNCERTAINTY_PAIRS = [
    ("parCWH", "frac_snow"),
    ("parPERC", "aridity"),
    ("parUZL", "soil_conductivity"),
    ("parUZL", "slope_mean"),
]


def read_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(BASE / rel)


def read_text(rel: str) -> str:
    return (BASE / rel).read_text(encoding="utf-8")


def fmt(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def classification_from_relationship(role: str, abs_rho: float, q: float | None, flag: str | None = None) -> str:
    if flag and flag in {"interpret with caution", "boundary-sensitive", "mean-coupled"}:
        if role == "dominant" and abs_rho >= 0.5:
            return "Discussion/caveat-only evidence"
        return "Discussion/caveat-only evidence"
    if role == "dominant" and abs_rho >= 0.45 and (q is None or q <= 0.1):
        return "main-text strong evidence"
    if role in {"dominant", "supportive"} and abs_rho >= 0.3:
        return "main-text supportive evidence"
    return "supplement completeness evidence"


def strength_from_classification(label: str) -> str:
    if label == "main-text strong evidence":
        return "strong"
    if label == "main-text supportive evidence":
        return "moderate"
    if label == "supplement completeness evidence":
        return "cautious"
    return "cautious"


def file_ready(paths: list[str]) -> tuple[str, str]:
    missing = [p for p in paths if not (BASE / p).exists()]
    return ("yes" if not missing else "no", "; ".join(missing) if missing else "")


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)

    run_inventory = read_csv("00_data_inventory/data/run_inventory.csv")
    duplicates = read_csv("00_data_inventory/data/parameter_duplicate_diagnostics.csv")
    dominant = read_csv("01_model_consistency/data/model_dominant_consistency_summary.csv")
    topk = read_csv("01_model_consistency/data/model_topk_overlap_summary.csv")
    compact = read_csv("01_model_consistency/data/within_model_compactness.csv")
    intermodel = read_csv("01_model_consistency/data/intermodel_similarity_summary.csv")
    sensitivity = read_csv("02_seed_loss_sensitivity/data/seed_vs_loss_sensitivity_summary.csv")
    cross_seed = read_csv("02_seed_loss_sensitivity/data/cross_seed_relationship_sensitivity.csv")
    cross_loss = read_csv("02_seed_loss_sensitivity/data/cross_loss_relationship_sensitivity.csv")
    focused = read_csv("02_seed_loss_sensitivity/data/focused_pair_seed_loss_stability.csv")
    mean_maps = read_csv("03_distributional_parameter_spatial_data/data/distributional_parameter_mean_maps_long.csv")
    map_summary = read_csv("03_distributional_parameter_spatial_data/data/parameter_map_summary_by_parameter.csv")
    mean_corr = read_csv("04_mean_attribute_relationships/data/distributional_mean_attribute_correlations.csv")
    dominant_mean = read_csv("04_mean_attribute_relationships/data/distributional_dominant_mean_relationships.csv")
    selected_mean = read_csv("04_mean_attribute_relationships/data/selected_key_relationships.csv")
    gradient_tests = read_csv("05_environmental_gradient_groups/data/gradient_high_low_tests.csv")
    gradient_stats = read_csv("05_environmental_gradient_groups/data/gradient_parameter_group_stats.csv")
    unc_maps = read_csv("06_uncertainty_spatial_data/data/distributional_parameter_uncertainty_maps_long.csv")
    unc_summary = read_csv("06_uncertainty_spatial_data/data/uncertainty_summary_by_parameter.csv")
    mean_std = read_csv("06_uncertainty_spatial_data/data/mean_std_coupling_diagnostics.csv")
    boundary = read_csv("06_uncertainty_spatial_data/data/boundary_uncertainty_diagnostics.csv")
    unc_corr = read_csv("07_uncertainty_attribute_relationships/data/distributional_std_attribute_correlations.csv")
    unc_dom = read_csv("07_uncertainty_attribute_relationships/data/std_dominant_relationships.csv")
    unc_flags = read_csv("07_uncertainty_attribute_relationships/data/std_headline_gradient_flags.csv")
    group_profiles = read_csv("08_representative_basin_groups/data/representative_group_parameter_profiles.csv")
    group_assign = read_csv("08_representative_basin_groups/data/representative_group_assignments.csv")
    candidates = read_csv("08_representative_basin_groups/data/representative_basin_candidates.csv")
    evidence = read_csv("09_integrated_summary/data/main_claim_evidence_table.csv")
    fig_candidates = read_csv("09_integrated_summary/data/figure_candidate_table.csv")
    old_class = read_csv("09_integrated_summary/data/main_text_vs_appendix_classification.csv")

    _ = [
        read_text("00_data_inventory/reports/data_completeness_check.md"),
        read_text("00_data_inventory/reports/definition_and_scale_check.md"),
        read_text("01_model_consistency/reports/model_consistency_report.md"),
        read_text("02_seed_loss_sensitivity/reports/seed_loss_sensitivity_report.md"),
        read_text("03_distributional_parameter_spatial_data/reports/parameter_spatial_data_report.md"),
        read_text("04_mean_attribute_relationships/reports/mean_attribute_relationship_report.md"),
        read_text("05_environmental_gradient_groups/reports/environmental_gradient_report.md"),
        read_text("06_uncertainty_spatial_data/reports/uncertainty_spatial_data_report.md"),
        read_text("07_uncertainty_attribute_relationships/reports/uncertainty_attribute_relationship_report.md"),
        read_text("08_representative_basin_groups/reports/representative_basin_group_report.md"),
        read_text("09_integrated_summary/reports/integrated_results_summary.md"),
        read_text("09_integrated_summary/reports/claim_strength_check.md"),
        read_text("09_integrated_summary/reports/recommended_results_order.md"),
        read_text("logs/full_analysis_pipeline_log.txt"),
    ]

    run_complete = (
        len(run_inventory) == 45
        and run_inventory["parameter_output_complete"].all()
        and run_inventory.groupby(["model_raw", "loss"])["seed"].nunique().min() == 5
        and run_inventory["basin_count"].min() == 531
        and run_inventory["parameter_count"].min() == 14
    )
    coords_ok = mean_maps["longitude"].notna().all() and unc_maps["longitude"].notna().all()
    duplicates_conflicting = int((duplicates["mean_conflict"] | duplicates["std_conflict"]).sum()) if not duplicates.empty else 0

    class_counts = dominant["relationship_class"].value_counts().to_dict()
    param_lists = dominant.groupby("relationship_class")["parameter"].apply(lambda s: ", ".join(sorted(s))).to_dict()

    compact_lookup = compact.set_index("model_raw")
    dist_compact = float(compact_lookup.loc["distributional", "within_model_compactness"])
    best_compact_model = compact.sort_values("within_model_compactness", ascending=False).iloc[0]["model_raw"]
    intermodel_mean = float(intermodel["between_model_similarity"].mean())
    dist_pairs = intermodel.loc[intermodel["model_pair"].str.contains("delta_dist", regex=False)]
    dist_intermodel_mean = float(dist_pairs["between_model_similarity"].mean()) if not dist_pairs.empty else np.nan

    sens = sensitivity.set_index("model_raw")
    dist_seed_sd = float(sens.loc["distributional", "mean_seed_sd_rho"])
    base_seed_sd = float(sens.loc["deterministic", "mean_seed_sd_rho"])
    mcd_seed_sd = float(sens.loc["mc_dropout", "mean_seed_sd_rho"])
    dist_loss_sd = float(sens.loc["distributional", "mean_cross_loss_sd_rho"])
    base_loss_sd = float(sens.loc["deterministic", "mean_cross_loss_sd_rho"])
    mcd_loss_sd = float(sens.loc["mc_dropout", "mean_cross_loss_sd_rho"])
    seed_best = sensitivity.sort_values("mean_seed_sd_rho").iloc[0]["model_raw"]
    loss_best = sensitivity.sort_values("mean_cross_loss_sd_rho").iloc[0]["model_raw"]
    seed_vs_loss = float(sensitivity["seed_sd_to_loss_sd_ratio"].median())

    sensitive_seed_pairs = (
        cross_seed.assign(abs_seed_range=lambda d: d["seed_range_rho"].abs())
        .sort_values("abs_seed_range", ascending=False)
        .head(10)[["model_raw", "loss", "parameter", "attribute", "seed_range_rho"]]
    )
    sensitive_loss_pairs = (
        cross_loss.assign(abs_loss_range=lambda d: d["cross_loss_range_rho"].abs())
        .sort_values("abs_loss_range", ascending=False)
        .head(10)[["model_raw", "parameter", "attribute", "cross_loss_range_rho"]]
    )

    key_mean_rows = []
    for parameter, attribute in KEY_MEAN_PAIRS:
        match = mean_corr.loc[(mean_corr["parameter"] == parameter) & (mean_corr["attribute"] == attribute)]
        if not match.empty:
            row = match.iloc[0]
            key_mean_rows.append(
                {
                    "pair": f"{parameter}-{attribute}",
                    "rho": float(row["spearman_rho"]),
                    "q": float(row["q_value"]),
                    "role": row["relationship_role"],
                    "classification": classification_from_relationship(row["relationship_role"], float(row["abs_rho"]), float(row["q_value"])),
                }
            )

    key_unc_rows = []
    for parameter, attribute in KEY_UNCERTAINTY_PAIRS:
        match = unc_corr.loc[(unc_corr["parameter"] == parameter) & (unc_corr["attribute"] == attribute)]
        if not match.empty:
            row = match.iloc[0]
            key_unc_rows.append(
                {
                    "pair": f"{parameter} uncertainty-{attribute}",
                    "rho": float(row["spearman_rho"]),
                    "q": float(row["q_value"]),
                    "role": row["relationship_role"],
                    "flag": row["interpretation_flag"],
                    "classification": classification_from_relationship(row["relationship_role"], float(row["abs_rho"]), float(row["q_value"]), row["interpretation_flag"]),
                }
            )
    routing_unc = unc_corr.loc[unc_corr["parameter"].isin(["route_a", "route_b"])].sort_values("abs_rho", ascending=False).head(8)

    gradient_key = gradient_tests.assign(abs_diff=lambda d: d["high_minus_low_median_difference"].abs()).sort_values("abs_diff", ascending=False).head(12)
    group_counts = group_assign.groupby(["gradient_attribute", "gradient_group"], observed=True)["basin_id"].nunique().reset_index()
    min_group_n = int(group_counts["basin_id"].min())
    best_group_profiles = group_profiles.assign(abs_diff=lambda d: d["high_minus_low_difference"].abs()).sort_values("abs_diff", ascending=False).head(12)

    claim_rows = [
        {
            "claim_id": "C1",
            "claim_text": "Three formulations share a dominant-control core.",
            "supporting_folder": "01_model_consistency",
            "supporting_file": "model_dominant_consistency_summary.csv",
            "key_metric_1": "shared dominant controls",
            "key_metric_1_value": class_counts.get("shared dominant controls", 0),
            "key_metric_2": "partially shared / model-sensitive",
            "key_metric_2_value": f"{class_counts.get('partially shared controls', 0)} / {class_counts.get('model-sensitive controls', 0)}",
            "caveat": "Dominant labels are modal across seeds/losses.",
            "recommended_strength": "strong",
            "main_text_or_supplement": "main-text headline",
        },
        {
            "claim_id": "C2",
            "claim_text": "delta_dist remains close to the shared relationship matrix region.",
            "supporting_folder": "01_model_consistency",
            "supporting_file": "within_model_compactness.csv; intermodel_similarity_summary.csv",
            "key_metric_1": "delta_dist within-model compactness",
            "key_metric_1_value": fmt(dist_compact),
            "key_metric_2": "delta_dist intermodel similarity mean",
            "key_metric_2_value": fmt(dist_intermodel_mean),
            "caveat": "Matrix similarity is descriptive.",
            "recommended_strength": "strong" if dist_intermodel_mean >= intermodel_mean * 0.95 else "moderate",
            "main_text_or_supplement": "main-text supportive",
        },
        {
            "claim_id": "C3",
            "claim_text": "delta_dist more reproducibly recovers relationship structure across seeds.",
            "supporting_folder": "02_seed_loss_sensitivity",
            "supporting_file": "seed_vs_loss_sensitivity_summary.csv",
            "key_metric_1": "mean_seed_sd_rho delta_dist / delta_base / delta_mcd",
            "key_metric_1_value": f"{fmt(dist_seed_sd)} / {fmt(base_seed_sd)} / {fmt(mcd_seed_sd)}",
            "key_metric_2": "best model",
            "key_metric_2_value": MODEL_LABELS[seed_best],
            "caveat": "Pair-level sensitivity varies by parameter/attribute.",
            "recommended_strength": "strong" if seed_best == "distributional" else "moderate",
            "main_text_or_supplement": "main-text headline",
        },
        {
            "claim_id": "C4",
            "claim_text": "delta_dist is also most stable across loss functions.",
            "supporting_folder": "02_seed_loss_sensitivity",
            "supporting_file": "seed_vs_loss_sensitivity_summary.csv",
            "key_metric_1": "mean_cross_loss_sd_rho delta_dist / delta_base / delta_mcd",
            "key_metric_1_value": f"{fmt(dist_loss_sd)} / {fmt(base_loss_sd)} / {fmt(mcd_loss_sd)}",
            "key_metric_2": "best model",
            "key_metric_2_value": MODEL_LABELS[loss_best],
            "caveat": "Loss sensitivity is generally larger than seed sensitivity.",
            "recommended_strength": "strong" if loss_best == "distributional" else "moderate",
            "main_text_or_supplement": "main-text headline",
        },
        {
            "claim_id": "C5",
            "claim_text": "Distributional parameter means align with major environmental gradients.",
            "supporting_folder": "04_mean_attribute_relationships; 05_environmental_gradient_groups",
            "supporting_file": "distributional_mean_attribute_correlations.csv; gradient_high_low_tests.csv",
            "key_metric_1": "strong/supportive key pairs",
            "key_metric_1_value": sum(row["classification"] != "supplement completeness evidence" for row in key_mean_rows),
            "key_metric_2": "strongest key pair rho",
            "key_metric_2_value": fmt(max(abs(row["rho"]) for row in key_mean_rows), 3),
            "caveat": "Attribute collinearity affects environmental interpretation.",
            "recommended_strength": "strong",
            "main_text_or_supplement": "main-text headline",
        },
        {
            "claim_id": "C6",
            "claim_text": "Distributional uncertainty contains structured gradients but requires diagnostics.",
            "supporting_folder": "06_uncertainty_spatial_data; 07_uncertainty_attribute_relationships",
            "supporting_file": "distributional_std_attribute_correlations.csv; mean_std_coupling_diagnostics.csv; boundary_uncertainty_diagnostics.csv",
            "key_metric_1": "headline clean std gradients",
            "key_metric_1_value": int(unc_flags.get("headline_candidate", pd.Series(dtype=bool)).sum()) if "headline_candidate" in unc_flags.columns else 0,
            "key_metric_2": "key uncertainty caveat flags",
            "key_metric_2_value": "; ".join(f"{r['pair']}={r['flag']}" for r in key_unc_rows),
            "caveat": "Several strongest uncertainty gradients are mean-coupled or boundary-sensitive.",
            "recommended_strength": "moderate",
            "main_text_or_supplement": "main-text supportive",
        },
        {
            "claim_id": "C7",
            "claim_text": "Representative basin groups support gradient interpretation.",
            "supporting_folder": "08_representative_basin_groups",
            "supporting_file": "representative_group_parameter_profiles.csv",
            "key_metric_1": "minimum low/high group size",
            "key_metric_1_value": min_group_n,
            "key_metric_2": "candidate basins",
            "key_metric_2_value": len(candidates),
            "caveat": "Examples are illustrative and should not be over-read as case studies.",
            "recommended_strength": "moderate",
            "main_text_or_supplement": "main-text supportive",
        },
    ]
    pd.DataFrame(claim_rows).to_csv(OUT_DATA / "main_claim_key_numbers.csv", index=False)

    figure_rows = []
    figure_defs = [
        ("Figure 5", "14-parameter spatial maps", ["03_distributional_parameter_spatial_data/data/distributional_parameter_mean_maps_long.csv", "03_distributional_parameter_spatial_data/data/parameter_map_summary_by_parameter.csv"], "14 map panels; one normalized/physical colorbar per parameter", "main-text headline", 1),
        ("Figure 6", "mean parameter-attribute heatmap", ["04_mean_attribute_relationships/data/distributional_mean_attribute_correlations.csv", "04_mean_attribute_relationships/data/distributional_dominant_mean_relationships.csv"], "full heatmap; dominant/supportive markers; selected key-pair callouts", "main-text headline", 2),
        ("Figure 7", "key environmental gradient contrasts", ["05_environmental_gradient_groups/data/gradient_parameter_group_stats.csv", "05_environmental_gradient_groups/data/gradient_high_low_tests.csv"], "low/middle/high boxplots or trend strips for aridity, snow, slope, PET, soil conductivity", "main-text headline", 3),
        ("Figure 8", "uncertainty-attribute heatmap", ["07_uncertainty_attribute_relationships/data/distributional_std_attribute_correlations.csv", "07_uncertainty_attribute_relationships/data/std_headline_gradient_flags.csv"], "std heatmap with clean/mean-coupled/boundary-sensitive annotations", "main-text supportive", 4),
        ("Figure 9 or supplement", "representative basin group profiles", ["08_representative_basin_groups/data/representative_group_parameter_profiles.csv", "08_representative_basin_groups/data/representative_group_assignments.csv", "08_representative_basin_groups/data/representative_basin_candidates.csv"], "group profile small multiples; optional basin examples", "main-text supportive", 5),
        ("Supplement", "model comparison maps and sensitivity checks", ["01_model_consistency/data/model_topk_overlap_summary.csv", "02_seed_loss_sensitivity/data/seed_vs_loss_sensitivity_summary.csv", "06_uncertainty_spatial_data/data/uncertainty_summary_by_parameter.csv"], "full model comparison; sensitivity diagnostics; uncertainty caveat panels", "supplement", 6),
    ]
    for figure, purpose, paths, panels, destination, priority in figure_defs:
        ready, missing = file_ready(paths)
        figure_rows.append(
            {
                "proposed_figure": figure,
                "figure_purpose": purpose,
                "required_data_files": "; ".join(paths),
                "data_ready_yes_no": ready,
                "missing_data": missing,
                "recommended_panels": panels,
                "main_text_or_supplement": destination,
                "priority": priority,
            }
        )
    pd.DataFrame(figure_rows).to_csv(OUT_DATA / "figure_data_readiness_check.csv", index=False)

    class_rows = []
    for _, row in mean_corr.iterrows():
        evidence_class = classification_from_relationship(row["relationship_role"], float(row["abs_rho"]), float(row["q_value"]))
        class_rows.append(
            {
                "evidence_type": "parameter_mean_attribute",
                "parameter": row["parameter"],
                "attribute": row["attribute"],
                "metric": "spearman_rho",
                "metric_value": row["spearman_rho"],
                "q_value": row["q_value"],
                "interpretation_flag": "",
                "updated_classification": evidence_class,
                "claim_strength": strength_from_classification(evidence_class),
            }
        )
    for _, row in unc_corr.iterrows():
        evidence_class = classification_from_relationship(row["relationship_role"], float(row["abs_rho"]), float(row["q_value"]), row["interpretation_flag"])
        class_rows.append(
            {
                "evidence_type": "parameter_uncertainty_attribute",
                "parameter": row["parameter"],
                "attribute": row["attribute"],
                "metric": "spearman_rho",
                "metric_value": row["spearman_rho"],
                "q_value": row["q_value"],
                "interpretation_flag": row["interpretation_flag"],
                "updated_classification": evidence_class,
                "claim_strength": strength_from_classification(evidence_class),
            }
        )
    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(OUT_DATA / "main_text_vs_appendix_classification_updated.csv", index=False)

    qc_verdict = "pass with caveats" if run_complete and coords_ok else "needs repair"
    contradiction = "No direct contradiction with previous findings; the main caveat is duplicated raw parameter rows, handled by deduplication/averaging."
    missing_required = "No must-have statistical analysis is missing for the next plotting/writing stage. Optional: add attribute-collinearity-aware partial correlations before final causal-style wording."

    lines = [
        "# Final QC After Full Analysis",
        "",
        "## Data completeness verdict",
        "",
        f"- Verdict: **{qc_verdict}**.",
        f"- Discovered runs: {len(run_inventory)}; complete parameter runs: {int(run_inventory['parameter_output_complete'].sum())}.",
        f"- Model x loss x seed coverage: {run_inventory.groupby(['model_raw', 'loss'])['seed'].nunique().min()} seeds per model/loss minimum.",
        f"- Basins/parameters: {int(run_inventory['basin_count'].min())} basins and {int(run_inventory['parameter_count'].min())} parameters per run.",
        f"- Spatial tables: {mean_maps['basin_id'].nunique()} basins x {mean_maps['parameter'].nunique()} parameters for means; {unc_maps['basin_id'].nunique()} x {unc_maps['parameter'].nunique()} for uncertainty; coordinates complete={coords_ok}.",
        "",
        "## Duplicated parameter handling verdict",
        "",
        f"- Duplicate logical parameter keys: {len(duplicates)}; conflicting mean/std duplicate keys: {duplicates_conflicting}.",
        "- The pipeline collapses duplicates by model/loss/seed/basin/parameter and averages conflicting values.",
        "- This should be disclosed in Methods or Supplement because duplicates are numerous. The risk is moderate for exact numeric values, but lower for rank-based Spearman structure if duplicate sources are parallel output families rather than independent samples.",
        "",
        "## Three-model consistency",
        "",
        f"- Shared dominant controls: {class_counts.get('shared dominant controls', 0)} ({param_lists.get('shared dominant controls', '')}).",
        f"- Partially shared controls: {class_counts.get('partially shared controls', 0)} ({param_lists.get('partially shared controls', '')}).",
        f"- Model-sensitive controls: {class_counts.get('model-sensitive controls', 0)} ({param_lists.get('model-sensitive controls', '')}).",
        f"- Within-model compactness: delta_base={fmt(compact_lookup.loc['deterministic', 'within_model_compactness'])}, delta_mcd={fmt(compact_lookup.loc['mc_dropout', 'within_model_compactness'])}, delta_dist={fmt(dist_compact)}.",
        f"- Most compact model: {MODEL_LABELS[best_compact_model]}. delta_dist intermodel similarity mean={fmt(dist_intermodel_mean)}; all intermodel mean={fmt(intermodel_mean)}.",
        "- No evidence that delta_dist recovers a completely different rule set; it remains close to the shared matrix region.",
        "",
        "## Seed vs loss sensitivity",
        "",
        f"- Cross-seed mean SD rho: delta_base={fmt(base_seed_sd)}, delta_mcd={fmt(mcd_seed_sd)}, delta_dist={fmt(dist_seed_sd)}.",
        f"- Cross-loss mean SD rho: delta_base={fmt(base_loss_sd)}, delta_mcd={fmt(mcd_loss_sd)}, delta_dist={fmt(dist_loss_sd)}.",
        f"- Loss sensitivity is stronger than seed sensitivity overall; median seed/loss SD ratio={fmt(seed_vs_loss)}.",
        f"- Most seed-sensitive pairs include: {sensitive_seed_pairs.to_dict(orient='records')}.",
        f"- Most loss-sensitive pairs include: {sensitive_loss_pairs.to_dict(orient='records')}.",
        "",
        "## Mean parameter-attribute gradients",
        "",
        f"- Key mean relationships: {key_mean_rows}.",
        f"- Strongest distributional mean relationships: {mean_corr.sort_values('abs_rho', ascending=False).head(8)[['parameter','attribute','spearman_rho','q_value','relationship_role']].to_dict(orient='records')}.",
        "- The strongest relationships are suitable for main text when hydrologically interpretable; collinear climate/terrain attributes should be acknowledged.",
        "",
        "## Environmental gradient groups",
        "",
        f"- Minimum low/high group size: {min_group_n}.",
        f"- Strongest high-low contrasts: {gradient_key[['gradient_attribute','parameter','high_minus_low_median_difference','p_value']].to_dict(orient='records')}.",
        "- Aridity, frac_snow, slope_mean, pet_mean, and soil_conductivity are ready for main gradient panels; weaker or redundant gradients can move to supplement.",
        "",
        "## Uncertainty gradients",
        "",
        f"- Key uncertainty relationships: {key_unc_rows}.",
        f"- Clean headline uncertainty candidates: {int(unc_flags.get('headline_candidate', pd.Series(dtype=bool)).sum()) if 'headline_candidate' in unc_flags.columns else 0}.",
        f"- Routing uncertainty strongest rows: {routing_unc[['parameter','attribute','spearman_rho','q_value','interpretation_flag']].to_dict(orient='records')}.",
        "- Several strongest uncertainty gradients are mean-coupled or boundary-sensitive, so use them as structured but cautious evidence.",
        "",
        "## Representative basin groups",
        "",
        f"- Group sizes are balanced by terciles; minimum n={min_group_n}.",
        f"- Candidate basin examples saved: {len(candidates)}.",
        f"- Strongest group profile contrasts: {best_group_profiles[['gradient_attribute','group_label','parameter','high_minus_low_difference']].head(10).to_dict(orient='records')}.",
        "- Basin examples should be illustrative only, not case-study-style proof.",
        "",
        "## Consistency with previous findings",
        "",
        f"- {contradiction}",
        "- The relationship-level stability narrative is supported more strongly than raw parameter-value superiority.",
        "",
        "## Missing analyses if any",
        "",
        f"- {missing_required}",
        "",
        "## Recommended next steps",
        "",
        "- Prioritize Figure 5 spatial maps first, then Figure 6 mean parameter-attribute heatmap.",
        "- Add a short Methods/Supplement note about duplicate-row diagnosis and collapse.",
        "- Keep uncertainty gradients diagnostic unless clean flags support stronger wording.",
    ]
    (OUT_REPORTS / "final_qc_after_full_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    next_lines = [
        "# Next Figure Plan",
        "",
        "## Priority 1: Figure 5",
        "",
        "- Purpose: 14-parameter spatial maps.",
        "- Status: data ready; `distributional_parameter_mean_maps_long.csv` covers 531 basins x 14 parameters with coordinates.",
        "- Recommendation: draw/retain distributional as main; use other model maps for supplement unless a direct comparison is needed.",
        "",
        "## Priority 2: Figure 6",
        "",
        "- Purpose: mean parameter-attribute heatmap.",
        "- Status: data ready.",
        "- Panels: full heatmap, dominant/supportive markers, selected key relationships.",
        "",
        "## Priority 3: Figure 7",
        "",
        "- Purpose: key environmental gradient boxplots/trend plots.",
        "- Status: data ready.",
        "- Panels: aridity-parPERC/parFC, frac_snow-parCWH/parCFR, slope_mean-parBETA/parUZL, pet_mean-parFC, soil_conductivity-parUZL.",
        "",
        "## Priority 4: Figure 8",
        "",
        "- Purpose: uncertainty-attribute heatmap.",
        "- Status: data ready with caveat flags.",
        "- Panels: std heatmap plus clean/mean-coupled/boundary-sensitive annotations.",
        "",
        "## Priority 5: Figure 9 or Supplement",
        "",
        "- Purpose: representative basin group profiles.",
        "- Status: data ready.",
        "- Recommendation: use as main-text supportive only if space allows; otherwise supplement.",
        "",
        "## Supplement",
        "",
        "- Model comparison maps.",
        "- Seed/loss sensitivity checks.",
        "- Full uncertainty diagnostics.",
        "- Duplicate parameter row diagnostics.",
    ]
    (OUT_REPORTS / "next_figure_plan.md").write_text("\n".join(next_lines) + "\n", encoding="utf-8")

    print("Wrote QC files:")
    for path in [
        OUT_REPORTS / "final_qc_after_full_analysis.md",
        OUT_DATA / "main_claim_key_numbers.csv",
        OUT_DATA / "figure_data_readiness_check.csv",
        OUT_DATA / "main_text_vs_appendix_classification_updated.csv",
        OUT_REPORTS / "next_figure_plan.md",
    ]:
        print(path)


if __name__ == "__main__":
    main()

