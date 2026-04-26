"""Final completeness check for Results 3.3-3.4."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

from project.parameterize.analysis.common import (
    build_parser,
    frame_to_markdown,
    load_analysis_data,
    normalize_parameters_to_unit_interval,
    save_frame,
    save_json,
    write_markdown,
)
from project.parameterize.figures.common import COLORS, apply_wrr_style, pretty_model_name, pretty_parameter_name, save_figure


KEY_ATTRIBUTES = [
    "aridity",
    "pet_mean",
    "frac_snow",
    "slope_mean",
    "soil_conductivity",
    "elev_mean",
    "soil_depth_pelletier",
]
KEY_PARAMETERS = ["parBETA", "parFC", "parPERC", "parUZL", "parCFR"]
HEADLINE_STD_PAIRS = [
    ("parCWH", "frac_snow"),
    ("parPERC", "aridity"),
    ("parUZL", "soil_conductivity"),
    ("parCWH", "slope_mean"),
    ("parUZL", "slope_mean"),
]


def _partial_corr(y: np.ndarray, x: np.ndarray, controls: np.ndarray) -> float:
    model_y = LinearRegression().fit(controls, y)
    model_x = LinearRegression().fit(controls, x)
    y_resid = y - model_y.predict(controls)
    x_resid = x - model_x.predict(controls)
    return float(spearmanr(y_resid, x_resid).statistic)


def build_attribute_collinearity(
    attributes: pd.DataFrame,
    params_long: pd.DataFrame,
    output_dir: Path,
    figures_dir: Path,
    reports_dir: Path,
) -> dict[str, Path]:
    attr_frame = attributes[KEY_ATTRIBUTES].copy()
    spearman = attr_frame.corr(method="spearman")
    pearson = attr_frame.corr(method="pearson")
    rows = []
    for left in KEY_ATTRIBUTES:
        for right in KEY_ATTRIBUTES:
            rows.append(
                {
                    "attribute_a": left,
                    "attribute_b": right,
                    "spearman_rho": float(spearman.loc[left, right]),
                    "pearson_r": float(pearson.loc[left, right]),
                    "high_correlation_flag": bool(left != right and abs(float(spearman.loc[left, right])) >= 0.6),
                }
            )
    collinearity = pd.DataFrame(rows)
    csv_path = save_frame(collinearity, output_dir / "finalcheck_attribute_collinearity.csv")

    apply_wrr_style()
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    sns.heatmap(spearman, annot=True, fmt=".2f", cmap="vlag", center=0.0, ax=ax, cbar_kws={"label": "Spearman rho"})
    heatmap_paths = save_figure(fig, "finalcheck_attribute_collinearity_heatmap", figures_dir, formats=("png",))

    dist_means = (
        params_long.loc[(params_long["model"] == "distributional") & (params_long["parameter"].isin(["parBETA", "parFC", "parPERC", "parUZL"]))]
        .groupby(["basin_id", "parameter"], as_index=False)["mean"]
        .mean()
        .merge(attributes[["basin_id"] + KEY_ATTRIBUTES], on="basin_id", how="inner")
    )
    partial_rows = []
    partial_specs = [
        ("parBETA", "slope_mean", "elev_mean"),
        ("parFC", "pet_mean", "aridity"),
        ("parPERC", "aridity", "pet_mean"),
        ("parUZL", "soil_conductivity", "slope_mean"),
    ]
    for parameter, target_attr, control_attr in partial_specs:
        subset = dist_means.loc[dist_means["parameter"] == parameter].dropna(subset=[target_attr, control_attr, "mean"])
        partial_rows.append(
            {
                "parameter": parameter,
                "target_attribute": target_attr,
                "control_attribute": control_attr,
                "raw_spearman": float(spearmanr(subset["mean"], subset[target_attr]).statistic),
                "partial_spearman": _partial_corr(
                    subset["mean"].to_numpy(),
                    subset[target_attr].to_numpy(),
                    subset[[control_attr]].to_numpy(),
                ),
            }
        )
    partial_df = pd.DataFrame(partial_rows)
    high_corr_pairs = collinearity.loc[collinearity["high_correlation_flag"]].drop_duplicates(subset=["attribute_a", "attribute_b"])
    report = write_markdown(
        reports_dir / "finalcheck_attribute_collinearity.md",
        title="Final Check: Attribute Collinearity",
        sections=[
            ("High-Correlation Attribute Pairs", frame_to_markdown(high_corr_pairs)),
            ("Partial Correlation Checks", frame_to_markdown(partial_df)),
            (
                "Short Notes",
                "\n".join(
                    [
                        "High-correlation attribute pairs are flagged at |Spearman rho| >= 0.6.",
                        "Partial checks are included only for a small set of headline relationships.",
                        f"Figure: `{heatmap_paths['png']}`",
                    ]
                ),
            ),
        ],
    )
    return {
        "finalcheck_attribute_collinearity": csv_path,
        "finalcheck_attribute_collinearity_heatmap": heatmap_paths["png"],
        "finalcheck_attribute_collinearity_report": report,
    }


def build_results332_confirmation(
    matrix_similarity: pd.DataFrame,
    matrix_embedding: pd.DataFrame,
    output_dir: Path,
    figures_dir: Path,
    reports_dir: Path,
) -> dict[str, Path]:
    within = (
        matrix_similarity.loc[(matrix_similarity["same_model"]) & (matrix_similarity["run_id_a"] != matrix_similarity["run_id_b"])]
        .groupby("model_a", as_index=False)
        .agg(
            mean_frobenius_distance=("frobenius_distance", "mean"),
            mean_matrix_corr=("matrix_corr_pearson", "mean"),
            mean_cosine_similarity=("cosine_similarity", "mean"),
        )
        .rename(columns={"model_a": "model"})
    )
    centroid_rows = []
    for left, right in [("deterministic", "distributional"), ("mc_dropout", "distributional"), ("deterministic", "mc_dropout")]:
        pairs = matrix_similarity.loc[
            (matrix_similarity["model_a"] == left) & (matrix_similarity["model_b"] == right)
        ]
        centroid_rows.append(
            {
                "model_a": left,
                "model_b": right,
                "mean_intermodel_corr": float(pairs["matrix_corr_pearson"].mean()),
                "mean_intermodel_distance": float(pairs["frobenius_distance"].mean()),
            }
        )
    centroid_df = pd.DataFrame(centroid_rows)

    apply_wrr_style()
    corr_df = pd.read_parquet(output_dir / "results332_spearman_matrices.parquet")
    corr_df["run_id"] = corr_df["model"] + "__" + corr_df["loss"] + "__seed_" + corr_df["seed"].astype(str)
    flat = corr_df.pivot_table(index="run_id", columns=["attribute", "parameter"], values="corr").sort_index(axis=1)
    run_order = matrix_embedding.sort_values(["model", "loss", "seed"])["run_id"].tolist()
    similarity_matrix = flat.loc[run_order].T.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    sns.heatmap(similarity_matrix, cmap="vlag", center=0.0, xticklabels=False, yticklabels=False, ax=ax, cbar_kws={"label": "Matrix corr"})
    ax.set_title("Final Check: Matrix Similarity Heatmap")
    heatmap_paths = save_figure(fig, "finalcheck_results332_similarity_heatmap", figures_dir, formats=("png",))

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    for model in ["deterministic", "mc_dropout", "distributional"]:
        subset = matrix_embedding.loc[matrix_embedding["model"] == model]
        ax.scatter(subset["mds_x"], subset["mds_y"], color=COLORS[model], s=48, label=pretty_model_name(model), alpha=0.85)
    ax.legend(frameon=False)
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.set_title("Final Check: Matrix Embedding")
    embedding_paths = save_figure(fig, "finalcheck_results332_embedding", figures_dir, formats=("png",))

    report = write_markdown(
        reports_dir / "finalcheck_results332_confirmation.md",
        title="Final Check: Results 3.3.2",
        sections=[
            ("Within-Model Compactness", frame_to_markdown(within)),
            ("Intermodel Similarity", frame_to_markdown(centroid_df)),
            (
                "Claim Options",
                "\n".join(
                    [
                        "Strong claim: Distributional occupies the same broad correlation-structure region as the other models while showing the tightest within-model clustering.",
                        "Careful claim: Distributional appears more coherent at matrix level and remains close to the shared correlation-structure region rather than separating into a qualitatively different cluster.",
                    ]
                ),
            ),
        ],
    )
    return {
        "finalcheck_results332_confirmation_report": report,
        "finalcheck_results332_similarity_heatmap": heatmap_paths["png"],
        "finalcheck_results332_embedding": embedding_paths["png"],
    }


def build_results333_confirmation(
    importance_alignment: pd.DataFrame,
    importance_overlap: pd.DataFrame,
    output_dir: Path,
    reports_dir: Path,
) -> dict[str, Path]:
    alignment = importance_alignment.copy()
    grades = []
    for _, row in alignment.iterrows():
        if row["dominant_attribute_top1_rate"] >= 0.5 or row["dominant_attribute_top3_rate"] >= 0.8:
            grade = "main-text supportive"
        elif row["dominant_attribute_top3_rate"] >= 0.4:
            grade = "appendix-only supportive"
        else:
            grade = "too weak for interpretation"
        grades.append(grade)
    alignment["evidence_grade"] = grades
    grading_path = save_frame(alignment, output_dir / "finalcheck_explainability_grading.csv")

    per_param = (
        alignment.groupby("parameter", as_index=False)
        .agg(
            mean_top1_rate=("dominant_attribute_top1_rate", "mean"),
            mean_top3_rate=("dominant_attribute_top3_rate", "mean"),
            share_main_text=("evidence_grade", lambda values: float(np.mean(np.asarray(values) == "main-text supportive"))),
        )
        .sort_values(["share_main_text", "mean_top3_rate"], ascending=[False, False])
    )
    report = write_markdown(
        reports_dir / "finalcheck_results333_confirmation.md",
        title="Final Check: Results 3.3.3",
        sections=[
            ("Per-Parameter Explainability Grade", frame_to_markdown(alignment)),
            ("Parameter-Level Summary", frame_to_markdown(per_param)),
            ("Model Overlap", frame_to_markdown(importance_overlap)),
        ],
    )
    return {
        "finalcheck_explainability_grading": grading_path,
        "finalcheck_results333_confirmation_report": report,
    }


def build_results342_confirmation(
    params_long: pd.DataFrame,
    parameter_bounds: dict[str, tuple[float, float]],
    std_relationships: pd.DataFrame,
    output_dir: Path,
    reports_dir: Path,
) -> dict[str, Path]:
    dist = (
        params_long.loc[(params_long["model"] == "distributional")]
        .groupby(["basin_id", "parameter"], as_index=False)
        .agg(mean=("mean", "mean"), std=("std", "mean"))
    )
    dist = normalize_parameters_to_unit_interval(dist, parameter_bounds, value_column="mean")
    dist["std_unit"] = dist.apply(
        lambda row: float(row["std"] / (parameter_bounds[row["parameter"]][1] - parameter_bounds[row["parameter"]][0])),
        axis=1,
    )
    dist["distance_to_boundary"] = dist[["mean_unit"]].apply(lambda row: float(min(row["mean_unit"], 1.0 - row["mean_unit"])), axis=1)
    coupling_rows = []
    for parameter, subset in dist.groupby("parameter"):
        coupling_rows.append(
            {
                "parameter": parameter,
                "mean_std_spearman": float(spearmanr(subset["mean"], subset["std"]).statistic),
                "meanunit_stdunit_spearman": float(spearmanr(subset["mean_unit"], subset["std_unit"]).statistic),
                "boundary_distance_std_spearman": float(spearmanr(subset["distance_to_boundary"], subset["std_unit"]).statistic),
                "near_boundary_share": float(np.mean((subset["mean_unit"] <= 0.1) | (subset["mean_unit"] >= 0.9))),
            }
        )
    coupling = pd.DataFrame(coupling_rows)
    coupling_path = save_frame(coupling, output_dir / "finalcheck_mean_std_coupling.csv")

    flags = std_relationships.loc[
        std_relationships.apply(lambda row: (row["parameter"], row["attribute"]) in HEADLINE_STD_PAIRS, axis=1)
    ].copy()
    flags = flags.merge(coupling, on="parameter", how="left")
    labels = []
    for _, row in flags.iterrows():
        if row["near_boundary_share"] >= 0.4 and row["boundary_distance_std_spearman"] >= 0.75:
            labels.append("boundary-sensitive / interpret with caution")
        elif abs(row["mean_std_spearman"]) >= 0.6:
            labels.append("possibly mean-coupled")
        else:
            labels.append("likely genuine uncertainty structure")
    flags["interpretation_flag"] = labels
    flags_path = save_frame(flags, output_dir / "finalcheck_uncertainty_interpretation_flags.csv")

    report = write_markdown(
        reports_dir / "finalcheck_results342_confirmation.md",
        title="Final Check: Results 3.4.2",
        sections=[
            ("Mean-Std Coupling", frame_to_markdown(coupling)),
            ("Headline Std-Gradient Flags", frame_to_markdown(flags)),
        ],
    )
    return {
        "finalcheck_mean_std_coupling": coupling_path,
        "finalcheck_uncertainty_interpretation_flags": flags_path,
        "finalcheck_results342_confirmation_report": report,
    }


def build_results343_confirmation(
    representative_basins: pd.DataFrame,
    basin_group_summary: pd.DataFrame,
    attributes: pd.DataFrame,
    output_dir: Path,
    reports_dir: Path,
) -> dict[str, Path]:
    attr_lookup = attributes.set_index("basin_id")
    grading_rows = []
    for _, row in representative_basins.iterrows():
        gradient = row["gradient_attribute"]
        value = row[gradient]
        series = attributes[gradient]
        percentile = float((series <= value).mean())
        group_summary = basin_group_summary.loc[basin_group_summary["group_label"] == row["group_label"]]
        mean_dev = float(group_summary["mean_median_vs_global"].abs().mean())
        std_dev = float(group_summary["std_median_vs_global"].abs().mean())
        if ("high" in row["group_label"] and percentile >= 0.8) or ("low" in row["group_label"] and percentile <= 0.2):
            extremeness = "extreme"
        elif ("high" in row["group_label"] and percentile >= 0.67) or ("low" in row["group_label"] and percentile <= 0.33):
            extremeness = "typical-endmember"
        else:
            extremeness = "not-strong-endmember"
        if extremeness == "extreme" and (mean_dev >= 0.05 or std_dev >= 0.01):
            grade = "main-text case"
        elif extremeness != "not-strong-endmember":
            grade = "appendix case"
        else:
            grade = "not representative enough"
        grading_rows.append(
            {
                "basin_id": int(row["basin_id"]),
                "group_label": row["group_label"],
                "gradient_attribute": gradient,
                "gradient_percentile": percentile,
                "extremeness_label": extremeness,
                "group_mean_deviation": mean_dev,
                "group_std_deviation": std_dev,
                "representativeness_grade": grade,
            }
        )
    grading = pd.DataFrame(grading_rows)
    grading_path = save_frame(grading, output_dir / "finalcheck_representative_basin_grading.csv")

    report = write_markdown(
        reports_dir / "finalcheck_results343_confirmation.md",
        title="Final Check: Results 3.4.3",
        sections=[
            ("Representative Basin Grading", frame_to_markdown(grading)),
            ("Basin Group Summary", frame_to_markdown(basin_group_summary)),
        ],
    )
    return {
        "finalcheck_representative_basin_grading": grading_path,
        "finalcheck_results343_confirmation_report": report,
    }


def build_claim_strength(
    results332_similarity: pd.DataFrame,
    explainability_grading: pd.DataFrame,
    uncertainty_flags: pd.DataFrame,
    collinearity: pd.DataFrame,
    output_dir: Path,
    reports_dir: Path,
) -> dict[str, Path]:
    high_corr_attrs = sorted(set(collinearity.loc[collinearity["high_correlation_flag"], "attribute_a"]))
    within = (
        results332_similarity.loc[(results332_similarity["same_model"]) & (results332_similarity["run_id_a"] != results332_similarity["run_id_b"])]
        .groupby("model_a", as_index=False)["frobenius_distance"]
        .mean()
        .rename(columns={"model_a": "model", "frobenius_distance": "mean_frobenius_distance"})
    )
    claim_rows = [
        {
            "claim": "Distributional reinforces shared correlation structure without departing from it",
            "claim_strength": "headline claim",
            "supporting_sections": "3.3.1, 3.3.2",
            "main_evidence": "distributional has the tightest within-model matrix compactness and remains near the shared matrix region",
            "caveats": "matrix conclusion is strong, but should still be phrased as coherence rather than identity",
            "recommended_wording_strength": "strong",
        },
        {
            "claim": "Post-hoc explainability supports the dominant-control results best for parFC and partially for parUZL/parCFR",
            "claim_strength": "supportive claim",
            "supporting_sections": "3.3.3",
            "main_evidence": "top1/top3 alignment and overlap tables",
            "caveats": "parBETA and parPERC have weaker alignment",
            "recommended_wording_strength": "careful",
        },
        {
            "claim": "Distributional parameter uncertainty shows structured gradients across basin environments",
            "claim_strength": "headline claim",
            "supporting_sections": "3.4.2",
            "main_evidence": "strong std-attribute correlations and low/high group differences",
            "caveats": f"attribute collinearity among {', '.join(high_corr_attrs)} should be acknowledged",
            "recommended_wording_strength": "strong",
        },
        {
            "claim": "Representative basin groups translate the gradient statistics into intuitive hydrologic contrasts",
            "claim_strength": "supportive claim",
            "supporting_sections": "3.4.3",
            "main_evidence": "group profiles and representative basin examples",
            "caveats": "case examples remain illustrative rather than inferential",
            "recommended_wording_strength": "careful",
        },
        {
            "claim": "Individual snow/routing disagreements imply universal process interpretation should be avoided",
            "claim_strength": "cautious / discussion-only claim",
            "supporting_sections": "3.3.1, 3.4.2",
            "main_evidence": "model-sensitive parameters and uncertainty flags",
            "caveats": "use as limitation/discussion language only",
            "recommended_wording_strength": "cautious",
        },
    ]
    claim_table = pd.DataFrame(claim_rows)
    claim_path = save_frame(claim_table, output_dir / "finalcheck_claim_strength_table.csv")
    report = write_markdown(
        reports_dir / "finalcheck_claim_strength.md",
        title="Final Check: Claim Strength",
        sections=[
            ("Claim Strength Table", frame_to_markdown(claim_table)),
        ],
    )
    return {
        "finalcheck_claim_strength_table": claim_path,
        "finalcheck_claim_strength_report": report,
    }


def build_final_report(
    reports_dir: Path,
    claim_table: pd.DataFrame,
    explainability_grading: pd.DataFrame,
    uncertainty_flags: pd.DataFrame,
    representative_grading: pd.DataFrame,
    collinearity: pd.DataFrame,
) -> Path:
    explain_rank = {"main-text supportive": 3, "appendix-only supportive": 2, "too weak for interpretation": 1}
    explain_param = (
        explainability_grading.assign(rank=explainability_grading["evidence_grade"].map(explain_rank))
        .groupby("parameter", as_index=False)
        .agg(max_rank=("rank", "max"), min_rank=("rank", "min"))
    )
    main_text_explain = explain_param.loc[explain_param["max_rank"] == 3, "parameter"].tolist()
    appendix_explain = explain_param.loc[(explain_param["max_rank"] == 2), "parameter"].tolist()
    weak_explain = explain_param.loc[(explain_param["max_rank"] == 1), "parameter"].tolist()
    mixed_explain = explain_param.loc[(explain_param["max_rank"] == 3) & (explain_param["min_rank"] == 1), "parameter"].tolist()
    main_cases = representative_grading.loc[
        representative_grading["representativeness_grade"] == "main-text case", "basin_id"
    ].astype(str).tolist()
    appendix_cases = representative_grading.loc[
        representative_grading["representativeness_grade"] == "appendix case", "basin_id"
    ].astype(str).tolist()
    high_corr_pairs = collinearity.loc[collinearity["high_correlation_flag"]][["attribute_a", "attribute_b", "spearman_rho"]].drop_duplicates()
    recommendation = "draft-ready but needs wording caution"
    report = write_markdown(
        reports_dir / "final_completeness_check.md",
        title="Final Completeness Check",
        sections=[
            (
                "What is already strong enough",
                "\n".join(
                    [
                        "- Matrix-level compactness clearly favors distributional.",
                        "- The dominant-control core from 3.3.1 remains intact under the matrix-level check.",
                        "- Uncertainty gradients show multiple strong, environment-aligned structures.",
                    ]
                ),
            ),
            (
                "What still needs cautious wording",
                "\n".join(
                    [
                        f"- Explainability is strongest for {', '.join(sorted(main_text_explain)) if main_text_explain else 'none'}, appendix-only for {', '.join(sorted(appendix_explain)) if appendix_explain else 'none'}, and weak for {', '.join(sorted(weak_explain)) if weak_explain else 'none'}.",
                        f"- Mixed-strength explainability parameters: {', '.join(sorted(mixed_explain)) if mixed_explain else 'none'}.",
                        f"- High-correlation attribute pairs need brief caveat language: {', '.join((high_corr_pairs['attribute_a'] + '–' + high_corr_pairs['attribute_b']).tolist())}.",
                        f"- Uncertainty flags requiring caution: {', '.join(uncertainty_flags.loc[uncertainty_flags['interpretation_flag'] != 'likely genuine uncertainty structure', 'parameter'] + '–' + uncertainty_flags.loc[uncertainty_flags['interpretation_flag'] != 'likely genuine uncertainty structure', 'attribute'])}.",
                    ]
                ),
            ),
            (
                "Which figures are main-paper-ready",
                "\n".join(
                    [
                        "- results332_matrix_similarity_heatmap.png",
                        "- results332_matrix_embedding.png",
                        "- results342_distributional_std_heatmap.png",
                        "- results342_std_gradient_boxplots.png",
                        "- results343_group_parameter_profiles.png",
                    ]
                ),
            ),
            (
                "Which figures/tables should move to appendix",
                "\n".join(
                    [
                        "- results333_importance_rankings.png",
                        "- results333_importance_overlap.png",
                        "- appendix_results342_std_vs_mean_compare.png",
                        "- results343_representative_basin_examples.png",
                    ]
                ),
            ),
            (
                "Minimal limitations paragraph to include in the paper",
                "Several headline basin controls were stable across methods and scales, but some attribute gradients remain correlated with one another, and post-hoc explainability was not equally informative for every parameter. We therefore interpret the dominant shared controls and the strongest uncertainty gradients as robust, while treating weaker explainability alignment and the most model-sensitive snow/routing cases as supportive rather than definitive evidence.",
            ),
            (
                "Final recommendation",
                f"- {recommendation}",
            ),
            (
                "Answers to the Required Questions",
                "\n".join(
                    [
                        "1. The current 3.3-3.4 results are complete enough to support the main paper line, with matrix-level coherence, dominant-control commonality, mean gradients, and std gradients all in place.",
                        "2. The weakest remaining link worth confirming before drafting is how strongly explainability should be used in the main text, because its support is parameter-dependent.",
                        f"3. Headline claims that can be written confidently are the matrix-level coherence of distributional, the existence of a shared dominant-control core, and the presence of structured uncertainty gradients; use {', '.join(sorted(main_text_explain)) if main_text_explain else 'no parameter'} as the strongest explainability-supported parameters.",
                        "4. Cautious wording is still required around attribute collinearity, weaker explainability alignment, and model-sensitive snow/routing cases.",
                    ]
                ),
            ),
        ],
    )
    return report


def run_final_completeness_check(
    params_long: pd.DataFrame,
    attributes: pd.DataFrame,
    parameter_bounds: dict[str, tuple[float, float]],
    analysis_root: Path,
) -> dict[str, Path]:
    output_dir = analysis_root / "correlation_summaries"
    figures_dir = analysis_root / "figures"
    reports_dir = analysis_root / "reports"

    results332_similarity = pd.read_csv(output_dir / "results332_matrix_similarity.csv")
    results332_embedding = pd.read_csv(output_dir / "results332_matrix_embedding.csv")
    results333_alignment = pd.read_csv(output_dir / "results333_importance_alignment.csv")
    results333_overlap = pd.read_csv(output_dir / "results333_importance_overlap.csv")
    results342_std = pd.read_csv(output_dir / "results342_distributional_std_relationships.csv")
    results343_groups = pd.read_csv(output_dir / "results343_basin_group_summary.csv")
    representative_basins = pd.read_csv(output_dir / "results343_representative_basins.csv")
    mean_relationships = pd.read_csv(output_dir / "results341_distributional_mean_relationships.csv")

    collinearity_paths = build_attribute_collinearity(attributes, params_long, output_dir, figures_dir, reports_dir)
    results332_paths = build_results332_confirmation(results332_similarity, results332_embedding, output_dir, figures_dir, reports_dir)
    results333_paths = build_results333_confirmation(results333_alignment, results333_overlap, output_dir, reports_dir)
    results342_paths = build_results342_confirmation(params_long, parameter_bounds, results342_std, output_dir, reports_dir)
    results343_paths = build_results343_confirmation(representative_basins, results343_groups, attributes, output_dir, reports_dir)

    claim_paths = build_claim_strength(
        results332_similarity,
        pd.read_csv(results333_paths["finalcheck_explainability_grading"]),
        pd.read_csv(results342_paths["finalcheck_uncertainty_interpretation_flags"]),
        pd.read_csv(collinearity_paths["finalcheck_attribute_collinearity"]),
        output_dir,
        reports_dir,
    )

    final_report = build_final_report(
        reports_dir=reports_dir,
        claim_table=pd.read_csv(claim_paths["finalcheck_claim_strength_table"]),
        explainability_grading=pd.read_csv(results333_paths["finalcheck_explainability_grading"]),
        uncertainty_flags=pd.read_csv(results342_paths["finalcheck_uncertainty_interpretation_flags"]),
        representative_grading=pd.read_csv(results343_paths["finalcheck_representative_basin_grading"]),
        collinearity=pd.read_csv(collinearity_paths["finalcheck_attribute_collinearity"]),
    )

    return {
        **collinearity_paths,
        **results332_paths,
        **results333_paths,
        **results342_paths,
        **results343_paths,
        **claim_paths,
        "final_completeness_check": final_report,
    }


def main() -> None:
    parser = build_parser("Run final completeness check for Results 3.3-3.4.")
    args = parser.parse_args()
    data = load_analysis_data(
        config_path=args.config,
        outputs_root=args.outputs_root,
        analysis_root=args.analysis_root,
        device=args.device,
        parameter_csv=args.parameter_csv,
        attribute_csv=args.attribute_csv,
    )
    if "params_long" not in data or "attributes" not in data:
        raise ValueError("--parameter-csv and --attribute-csv are required.")
    paths = run_final_completeness_check(
        params_long=data["params_long"],
        attributes=data["attributes"],
        parameter_bounds=data["parameter_bounds"],
        analysis_root=data["stability_analysis_root"],
    )
    print(paths["final_completeness_check"])


if __name__ == "__main__":
    main()
