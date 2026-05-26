from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .common import (
    MODEL_LABELS,
    PipelineLog,
    correlation_value,
    report_common_sections,
    save_csv,
    sign_label,
    write_md,
)


def _pair_class(attrs: list[str]) -> str:
    n = len(set([a for a in attrs if pd.notna(a)]))
    if n == 1:
        return "all_same"
    if n == 2:
        return "two_of_three"
    return "all_different"


def _relationship_class(consistency: str) -> str:
    return {
        "all_same": "shared dominant controls",
        "two_of_three": "partially shared controls",
        "all_different": "model-sensitive controls",
    }.get(consistency, "model-sensitive controls")


def run(dirs: dict[str, dict[str, object]], context: dict[str, object], log: PipelineLog) -> None:
    block = "01_model_consistency"
    data_dir = dirs[block]["data"]
    reports_dir = dirs[block]["reports"]
    methods_dir = dirs[block]["methods"]
    logs_dir = dirs[block]["logs"]

    params: pd.DataFrame = context["params"]
    attrs: pd.DataFrame = context["attributes"]
    attr_cols = [c for c in context["attribute_mapping"]["mapped_attribute"].dropna().unique() if c in attrs.columns]

    merged = params.merge(attrs[["basin_id"] + attr_cols], on="basin_id", how="inner")
    rows = []
    for (model, loss, seed, parameter), sub in merged.groupby(["model_raw", "loss", "seed", "parameter"], sort=False):
        for attr in attr_cols:
            rho, p_value, n = correlation_value(sub["parameter_mean_unit"], sub[attr], method="spearman")
            rows.append(
                {
                    "model_raw": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "loss": loss,
                    "seed": int(seed),
                    "parameter": parameter,
                    "attribute": attr,
                    "spearman_rho": rho,
                    "abs_spearman_rho": abs(rho) if pd.notna(rho) else np.nan,
                    "p_value_optional": p_value,
                    "n_basins": n,
                }
            )
    corr = pd.DataFrame(rows)
    save_csv(corr, data_dir / "seed_loss_correlation_matrix_long.csv", log)
    context["corr_long"] = corr

    dominant = (
        corr.sort_values(["model_raw", "loss", "seed", "parameter", "abs_spearman_rho"], ascending=[True, True, True, True, False])
        .groupby(["model_raw", "model_label", "loss", "seed", "parameter"], as_index=False)
        .head(1)
        .copy()
    )
    dominant["dominant_sign"] = dominant["spearman_rho"].map(sign_label)
    dominant = dominant.rename(columns={"attribute": "dominant_attribute", "spearman_rho": "dominant_spearman_rho"})
    save_csv(dominant, data_dir / "dominant_attribute_by_run.csv", log)
    context["dominant_by_run"] = dominant

    modal_rows = []
    for (model, parameter), sub in dominant.groupby(["model_raw", "parameter"]):
        attr_mode = sub["dominant_attribute"].mode()
        attr = attr_mode.iloc[0] if len(attr_mode) else np.nan
        sign_mode = sub["dominant_sign"].mode()
        sign = sign_mode.iloc[0] if len(sign_mode) else np.nan
        modal_rows.append(
            {
                "model_raw": model,
                "parameter": parameter,
                "modal_dominant_attribute": attr,
                "modal_direction": sign,
                "dominant_modal_rate": float((sub["dominant_attribute"] == attr).mean()) if pd.notna(attr) else np.nan,
                "mean_abs_dominant_rho": sub["abs_spearman_rho"].mean(),
            }
        )
    modal = pd.DataFrame(modal_rows)
    consistency_rows = []
    for parameter, sub in modal.groupby("parameter"):
        by_model = sub.set_index("model_raw")
        attrs3 = [by_model.loc[m, "modal_dominant_attribute"] if m in by_model.index else np.nan for m in ["deterministic", "mc_dropout", "distributional"]]
        signs3 = [by_model.loc[m, "modal_direction"] if m in by_model.index else np.nan for m in ["deterministic", "mc_dropout", "distributional"]]
        consistency = _pair_class(attrs3)
        direction_consistency = "all_same" if len(set([s for s in signs3 if pd.notna(s)])) <= 1 else "sign_flip_present"
        consistency_rows.append(
            {
                "parameter": parameter,
                "deterministic_dominant_attribute": attrs3[0],
                "mc_dropout_dominant_attribute": attrs3[1],
                "distributional_dominant_attribute": attrs3[2],
                "dominant_attribute_consistency": consistency,
                "direction_consistency": direction_consistency,
                "relationship_class": _relationship_class(consistency),
                "deterministic_modal_rate": by_model.loc["deterministic", "dominant_modal_rate"] if "deterministic" in by_model.index else np.nan,
                "mc_dropout_modal_rate": by_model.loc["mc_dropout", "dominant_modal_rate"] if "mc_dropout" in by_model.index else np.nan,
                "distributional_modal_rate": by_model.loc["distributional", "dominant_modal_rate"] if "distributional" in by_model.index else np.nan,
            }
        )
    consistency_df = pd.DataFrame(consistency_rows)
    save_csv(consistency_df, data_dir / "model_dominant_consistency_summary.csv", log)
    context["model_dominant_consistency"] = consistency_df

    top_rows = []
    model_pairs = [("deterministic", "mc_dropout"), ("deterministic", "distributional"), ("mc_dropout", "distributional")]
    for (loss, seed, parameter), sub in corr.groupby(["loss", "seed", "parameter"]):
        for k in (3, 5):
            top_by_model = {
                model: set(
                    sub.loc[sub["model_raw"].eq(model)]
                    .sort_values("abs_spearman_rho", ascending=False)
                    .head(k)["attribute"]
                )
                for model in MODEL_LABELS
            }
            for left, right in model_pairs:
                union = top_by_model[left] | top_by_model[right]
                inter = top_by_model[left] & top_by_model[right]
                top_rows.append(
                    {
                        "loss": loss,
                        "seed": int(seed),
                        "parameter": parameter,
                        "top_k": k,
                        "model_pair": f"{MODEL_LABELS[left]} vs {MODEL_LABELS[right]}",
                        "left_model_raw": left,
                        "right_model_raw": right,
                        "jaccard_overlap": len(inter) / len(union) if union else np.nan,
                        "intersection_count": len(inter),
                        "union_count": len(union),
                    }
                )
    topk = pd.DataFrame(top_rows)
    save_csv(topk, data_dir / "model_topk_overlap_summary.csv", log)

    matrix_rows = []
    vectors = {}
    for (model, loss, seed), sub in corr.groupby(["model_raw", "loss", "seed"]):
        matrix = sub.pivot_table(index="attribute", columns="parameter", values="spearman_rho").sort_index(axis=0).sort_index(axis=1)
        vectors[(model, loss, int(seed))] = matrix.to_numpy().ravel()
    for left, right in itertools.combinations(vectors, 2):
        a, b = vectors[left], vectors[right]
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 2:
            matrix_corr = cosine = frob = np.nan
        else:
            av, bv = a[valid], b[valid]
            matrix_corr = float(np.corrcoef(av, bv)[0, 1])
            frob = float(np.linalg.norm(av - bv))
            denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
            cosine = float(np.dot(av, bv) / denom) if denom else np.nan
        matrix_rows.append(
            {
                "left_model_raw": left[0],
                "left_loss": left[1],
                "left_seed": left[2],
                "right_model_raw": right[0],
                "right_loss": right[1],
                "right_seed": right[2],
                "pair_type": "within_model" if left[0] == right[0] else "between_model",
                "same_loss": left[1] == right[1],
                "same_seed": left[2] == right[2],
                "pairwise_matrix_correlation": matrix_corr,
                "frobenius_distance": frob,
                "cosine_similarity": cosine,
            }
        )
    matrix_pairwise = pd.DataFrame(matrix_rows)
    save_csv(matrix_pairwise, data_dir / "matrix_similarity_pairwise.csv", log)

    compact = (
        matrix_pairwise.loc[matrix_pairwise["pair_type"].eq("within_model")]
        .groupby("left_model_raw", as_index=False)
        .agg(
            within_model_compactness=("pairwise_matrix_correlation", "mean"),
            mean_frobenius_distance=("frobenius_distance", "mean"),
            mean_cosine_similarity=("cosine_similarity", "mean"),
            n_pairs=("pairwise_matrix_correlation", "size"),
        )
        .rename(columns={"left_model_raw": "model_raw"})
    )
    compact["model_label"] = compact["model_raw"].map(MODEL_LABELS)
    save_csv(compact, data_dir / "within_model_compactness.csv", log)

    inter = (
        matrix_pairwise.loc[matrix_pairwise["pair_type"].eq("between_model")]
        .assign(model_pair=lambda d: d.apply(lambda r: " vs ".join(sorted([MODEL_LABELS[r["left_model_raw"]], MODEL_LABELS[r["right_model_raw"]]])), axis=1))
        .groupby("model_pair", as_index=False)
        .agg(
            between_model_similarity=("pairwise_matrix_correlation", "mean"),
            mean_frobenius_distance=("frobenius_distance", "mean"),
            mean_cosine_similarity=("cosine_similarity", "mean"),
            n_pairs=("pairwise_matrix_correlation", "size"),
        )
    )
    save_csv(inter, data_dir / "intermodel_similarity_summary.csv", log)

    class_counts = consistency_df["relationship_class"].value_counts().to_dict()
    dist_compact = compact.loc[compact["model_raw"].eq("distributional")]
    write_md(
        reports_dir / "model_consistency_report.md",
        "Model Consistency Report",
        report_common_sections(
            objective="Assess whether the three model formulations share dominant basin attribute-parameter relationship structure.",
            input_files=["Deduplicated parameter table", "Basin attribute table"],
            data_filters=["All discovered 531-basin runs.", "Spearman correlations use basin-level normalized parameter means."],
            metric_definitions=[
                "Dominant attribute is the largest absolute Spearman rho for each model/loss/seed/parameter.",
                "Top-k overlap is Jaccard overlap of top-3/top-5 attributes.",
                "Matrix similarity uses full vectorized attribute-parameter correlation matrices.",
            ],
            main_results=[
                f"Relationship class counts: {class_counts}.",
                f"Distributional compactness row: {dist_compact.to_dict(orient='records')}.",
                f"Mean intermodel similarities: {inter.to_dict(orient='records')}.",
            ],
            tables=[
                "data/seed_loss_correlation_matrix_long.csv",
                "data/dominant_attribute_by_run.csv",
                "data/model_dominant_consistency_summary.csv",
                "data/model_topk_overlap_summary.csv",
                "data/matrix_similarity_pairwise.csv",
                "data/within_model_compactness.csv",
                "data/intermodel_similarity_summary.csv",
            ],
            caveats=[
                "Dominant-control classes use modal dominant attributes across losses and seeds.",
                "Similarity summaries describe association structure, not causal mechanisms.",
            ],
            wording=[
                "The models more reproducibly recover a shared dominant-control core where all three modal dominant attributes match.",
                "Model-sensitive controls should be treated as formulation-dependent associations.",
            ],
            figure_usage=[
                "Use these tables later for a model-consistency heatmap or matrix-similarity panel.",
                "No figure files were generated.",
            ],
        ),
        log,
    )
    write_md(
        methods_dir / "method_definitions.md",
        "Model Consistency Method Definitions",
        {
            "Correlation matrix": ["Spearman rho for parameter mean unit versus each basin attribute, per model/loss/seed."],
            "Dominant controls": ["Attribute with maximum absolute rho per parameter and run."],
            "Matrix region": ["Compared by vectorized full correlation matrices using correlation, Frobenius distance, and cosine similarity."],
        },
        log,
    )
    write_md(logs_dir / "model_consistency_log.md", "Model Consistency Log", {"Summary": [f"Correlation rows: {len(corr)}", f"Matrix pair rows: {len(matrix_pairwise)}"]}, log)

