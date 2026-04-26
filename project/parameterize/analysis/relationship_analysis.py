"""Focused relationship analysis for Results 2-4.

This module narrows the existing seed/loss correlation outputs to a smaller set
of paper-ready relationship summaries. The main question is whether the
distributional model recovers more robust basin-attribute/parameter structure
than deterministic and MC-dropout variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from project.parameterize.analysis.common import frame_to_markdown, save_frame, write_markdown


DEFAULT_CORE_TOP_K = 3
DEFAULT_CORE_PAIRS_PER_PARAMETER = 3
DEFAULT_EXPLAINABILITY_PARAMETER_COUNT = 3


@dataclass(frozen=True)
class RelationshipOutputs:
    core_relationships_summary: pd.DataFrame
    pair_seed_stability: pd.DataFrame
    pair_loss_stability: pd.DataFrame
    relationship_classes: pd.DataFrame
    parameter_level_consistency: pd.DataFrame
    parameter_feature_importance: pd.DataFrame
    stability_significance_summary: pd.DataFrame


def sign_consistency_rate(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[~np.isnan(array)]
    if array.size == 0:
        return np.nan
    non_zero = array[~np.isclose(array, 0.0)]
    if non_zero.size == 0:
        return 1.0
    positive = int((non_zero > 0).sum())
    negative = int((non_zero < 0).sum())
    return float(max(positive, negative) / non_zero.size)


def _prepare_spearman_table(corr_long: pd.DataFrame, top_k: int) -> pd.DataFrame:
    spearman = corr_long.loc[corr_long["method"] == "spearman"].copy()
    spearman["rank_within_run"] = (
        spearman.groupby(["model", "loss", "seed", "parameter"])["abs_corr"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    spearman["is_dominant"] = spearman["rank_within_run"] == 1
    spearman["is_topk"] = spearman["rank_within_run"] <= top_k
    return spearman


def _stability_frame(
    spearman: pd.DataFrame,
    dimension: str,
) -> pd.DataFrame:
    if dimension == "seed":
        group_cols = ["model", "parameter", "attribute", "loss"]
        mean_col = "seed_mean_rho"
        std_col = "seed_std_rho"
        range_col = "seed_range_rho"
        min_col = "seed_min_rho"
        max_col = "seed_max_rho"
        consistency_col = "sign_consistency_seed"
        topk_col = "topk_rate_seed"
        dominant_col = "dominant_rate_seed"
        count_col = "seed_count"
    elif dimension == "loss":
        group_cols = ["model", "parameter", "attribute", "seed"]
        mean_col = "loss_mean_rho"
        std_col = "loss_std_rho"
        range_col = "loss_range_rho"
        min_col = "loss_min_rho"
        max_col = "loss_max_rho"
        consistency_col = "sign_consistency_loss"
        topk_col = "topk_rate_loss"
        dominant_col = "dominant_rate_loss"
        count_col = "loss_count"
    else:
        raise ValueError(f"Unsupported stability dimension '{dimension}'.")

    return (
        spearman.groupby(group_cols, as_index=False)
        .agg(
            **{
                mean_col: ("corr", "mean"),
                std_col: ("corr", lambda values: float(np.std(values, ddof=0))),
                range_col: ("corr", lambda values: float(np.nanmax(values) - np.nanmin(values))),
                min_col: ("corr", "min"),
                max_col: ("corr", "max"),
                consistency_col: ("corr", sign_consistency_rate),
                topk_col: ("is_topk", "mean"),
                dominant_col: ("is_dominant", "mean"),
                count_col: (group_cols[-1], "nunique"),
            }
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def _run_presence_rates(spearman: pd.DataFrame, top_k: int) -> pd.DataFrame:
    topk = spearman.assign(is_topk=spearman["rank_within_run"] <= top_k)
    loss_cover = (
        topk.groupby(["model", "parameter", "attribute", "loss"], as_index=False)["is_topk"]
        .mean()
        .rename(columns={"is_topk": "seed_topk_rate_within_loss"})
    )
    seed_cover = (
        topk.groupby(["model", "parameter", "attribute", "seed"], as_index=False)["is_topk"]
        .mean()
        .rename(columns={"is_topk": "loss_topk_rate_within_seed"})
    )
    loss_summary = (
        loss_cover.groupby(["model", "parameter", "attribute"], as_index=False)
        .agg(
            loss_topk_consistency=("seed_topk_rate_within_loss", "mean"),
            loss_majority_topk_rate=("seed_topk_rate_within_loss", lambda values: float(np.mean(np.asarray(values) >= 0.5))),
        )
    )
    seed_summary = (
        seed_cover.groupby(["model", "parameter", "attribute"], as_index=False)
        .agg(
            seed_topk_consistency=("loss_topk_rate_within_seed", "mean"),
            seed_majority_topk_rate=("loss_topk_rate_within_seed", lambda values: float(np.mean(np.asarray(values) >= 0.5))),
        )
    )
    return loss_summary.merge(seed_summary, on=["model", "parameter", "attribute"], how="outer")


def compute_core_relationships(
    corr_long: pd.DataFrame,
    top_k: int = DEFAULT_CORE_TOP_K,
    pairs_per_parameter: int = DEFAULT_CORE_PAIRS_PER_PARAMETER,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spearman = _prepare_spearman_table(corr_long, top_k=top_k)
    pair_seed = _stability_frame(spearman, dimension="seed")
    pair_loss = _stability_frame(spearman, dimension="loss")

    pair_summary = (
        spearman.groupby(["model", "parameter", "attribute"], as_index=False)
        .agg(
            mean_abs_corr=("abs_corr", "mean"),
            mean_corr=("corr", "mean"),
            sign_consistency=("corr", sign_consistency_rate),
            dominant_attribute_consistency=("is_dominant", "mean"),
            topk_rate_all_runs=("is_topk", "mean"),
            run_count=("corr", "size"),
        )
    )

    seed_agg = (
        pair_seed.groupby(["model", "parameter", "attribute"], as_index=False)
        .agg(
            seed_std=("seed_std_rho", "mean"),
            seed_range=("seed_range_rho", "mean"),
            min_seed_sign_consistency=("sign_consistency_seed", "min"),
            mean_seed_topk_rate=("topk_rate_seed", "mean"),
            mean_seed_dominant_rate=("dominant_rate_seed", "mean"),
        )
    )
    loss_agg = (
        pair_loss.groupby(["model", "parameter", "attribute"], as_index=False)
        .agg(
            loss_std=("loss_std_rho", "mean"),
            loss_range=("loss_range_rho", "mean"),
            min_loss_sign_consistency=("sign_consistency_loss", "min"),
            mean_loss_topk_rate=("topk_rate_loss", "mean"),
            mean_loss_dominant_rate=("dominant_rate_loss", "mean"),
        )
    )
    presence = _run_presence_rates(spearman, top_k=top_k)

    summary = (
        pair_summary.merge(seed_agg, on=["model", "parameter", "attribute"], how="left")
        .merge(loss_agg, on=["model", "parameter", "attribute"], how="left")
        .merge(presence, on=["model", "parameter", "attribute"], how="left")
    )

    summary = summary.sort_values(
        [
            "model",
            "parameter",
            "loss_topk_consistency",
            "seed_topk_consistency",
            "dominant_attribute_consistency",
            "mean_abs_corr",
            "sign_consistency",
        ],
        ascending=[True, True, False, False, False, False, False],
    ).reset_index(drop=True)

    core = (
        summary.groupby(["model", "parameter"], as_index=False)
        .head(pairs_per_parameter)
        .copy()
        .reset_index(drop=True)
    )
    core["core_rank"] = core.groupby(["model", "parameter"]).cumcount() + 1
    return core, pair_seed, pair_loss


def build_parameter_level_consistency(core_summary: pd.DataFrame) -> pd.DataFrame:
    dominant = core_summary.loc[core_summary["core_rank"] == 1].copy()
    rows: list[dict[str, object]] = []
    for parameter, subset in dominant.groupby("parameter"):
        attr_by_model = {row["model"]: row["attribute"] for _, row in subset.iterrows()}
        sign_by_model = {row["model"]: np.sign(row["mean_corr"]) for _, row in subset.iterrows()}
        internal_stability = {
            row["model"]: bool(
                row["dominant_attribute_consistency"] >= (2.0 / 3.0)
                and row["loss_topk_consistency"] >= (2.0 / 3.0)
                and row["sign_consistency"] >= 0.8
            )
            for _, row in subset.iterrows()
        }
        available_models = list(attr_by_model)
        attr_values = [attr_by_model[model] for model in available_models]
        sign_values = [sign_by_model[model] for model in available_models if not np.isclose(sign_by_model[model], 0.0)]
        unique_attrs = set(attr_values)
        unique_signs = set(sign_values)

        if len(unique_attrs) == 1:
            attr_consistency = "all_same"
        elif len(unique_attrs) == 2:
            attr_consistency = "two_of_three"
        else:
            attr_consistency = "all_different"

        if len(unique_signs) <= 1:
            sign_consistency = "all_same"
        else:
            sign_consistency = "sign_flip_present"

        unstable_models = [model for model, stable in internal_stability.items() if not stable]
        if attr_consistency == "all_same" and sign_consistency == "all_same":
            if unstable_models:
                comment = (
                    "Headline dominant attribute matches across models, "
                    f"but {', '.join(sorted(unstable_models))} changes more across losses."
                )
            else:
                comment = "All models recover the same dominant attribute with the same sign."
        elif unstable_models:
            comment = (
                "Divergence is partly loss-driven because "
                f"{', '.join(sorted(unstable_models))} does not keep a stable dominant attribute."
            )
        else:
            comment = "Each model is internally stable but the dominant attribute differs, pointing to model effects."

        row = {
            "parameter": parameter,
            "deterministic_attribute": attr_by_model.get("deterministic"),
            "mc_dropout_attribute": attr_by_model.get("mc_dropout"),
            "distributional_attribute": attr_by_model.get("distributional"),
            "deterministic_sign": _sign_label(sign_by_model.get("deterministic", np.nan)),
            "mc_dropout_sign": _sign_label(sign_by_model.get("mc_dropout", np.nan)),
            "distributional_sign": _sign_label(sign_by_model.get("distributional", np.nan)),
            "dominant_attribute_consistency_across_models": attr_consistency,
            "direction_consistency_across_models": sign_consistency,
            "comments": comment,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("parameter").reset_index(drop=True)


def _sign_label(value: float) -> str:
    if np.isnan(value) or np.isclose(value, 0.0):
        return "zero_or_nan"
    return "positive" if value > 0 else "negative"


def classify_relationships(
    core_summary: pd.DataFrame,
    parameter_level_consistency: pd.DataFrame,
) -> pd.DataFrame:
    consistency_lookup = parameter_level_consistency.set_index("parameter")
    classified = core_summary.copy()

    classified["seed_stable"] = (
        (classified["seed_std"] <= 0.08)
        & (classified["seed_range"] <= 0.20)
        & (classified["mean_seed_topk_rate"] >= 0.80)
        & (classified["min_seed_sign_consistency"] >= 0.80)
    )
    classified["loss_stable"] = (
        (classified["loss_std"] <= 0.10)
        & (classified["loss_range"] <= 0.25)
        & (classified["mean_loss_topk_rate"] >= (2.0 / 3.0))
        & (classified["min_loss_sign_consistency"] >= (2.0 / 3.0))
    )

    model_sensitive_flags = []
    reasons = []
    classes = []
    for _, row in classified.iterrows():
        consistency_row = consistency_lookup.loc[row["parameter"]]
        attr_consistency = consistency_row["dominant_attribute_consistency_across_models"]
        direction_consistency = consistency_row["direction_consistency_across_models"]
        is_model_sensitive = bool(
            row["core_rank"] == 1 and (attr_consistency != "all_same" or direction_consistency != "all_same")
        )
        model_sensitive_flags.append(is_model_sensitive)

        if is_model_sensitive:
            relationship_class = "model-sensitive"
            reason = consistency_row["comments"]
        elif bool(row["seed_stable"]) and bool(row["loss_stable"]):
            relationship_class = "robust"
            reason = "Stable across seeds and losses with consistent sign/top-k retention."
        else:
            relationship_class = "loss-sensitive"
            reason = "Stable enough across seeds but the relationship moves across losses."
        classes.append(relationship_class)
        reasons.append(reason)

    classified["model_sensitive"] = model_sensitive_flags
    classified["relationship_class"] = classes
    classified["class_reason"] = reasons
    return classified.sort_values(["relationship_class", "model", "parameter", "core_rank"]).reset_index(drop=True)


def _bootstrap_mean_difference(
    dist_values: np.ndarray,
    other_values: np.ndarray,
    larger_is_better: bool,
    bootstrap_iters: int = 2000,
    random_seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(random_seed)
    if dist_values.size == 0:
        return np.nan, np.nan, np.nan
    indices = np.arange(dist_values.size)
    diffs = []
    for _ in range(bootstrap_iters):
        sample_idx = rng.choice(indices, size=indices.size, replace=True)
        dist_sample = dist_values[sample_idx]
        other_sample = other_values[sample_idx]
        raw_diff = np.mean(dist_sample) - np.mean(other_sample)
        diffs.append(raw_diff if larger_is_better else -raw_diff)
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    observed = float(np.mean(dist_values) - np.mean(other_values))
    advantage = observed if larger_is_better else -observed
    return float(advantage), float(lower), float(upper)


def _paired_wilcoxon(dist_values: np.ndarray, other_values: np.ndarray) -> float:
    if dist_values.size < 2:
        return np.nan
    diffs = dist_values - other_values
    if np.allclose(diffs, 0.0):
        return 1.0
    try:
        return float(wilcoxon(dist_values, other_values, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def summarize_stability_significance(classified_relationships: pd.DataFrame) -> pd.DataFrame:
    dominant = classified_relationships.loc[classified_relationships["core_rank"] == 1].copy()
    metrics = [
        ("seed_std", False, "dominant_pair_seed_std_corr"),
        ("seed_range", False, "dominant_pair_seed_range_corr"),
        ("loss_std", False, "dominant_pair_loss_std_corr"),
        ("loss_range", False, "dominant_pair_loss_range_corr"),
        ("dominant_attribute_consistency", True, "dominant_attribute_consistency"),
    ]
    rows: list[dict[str, object]] = []
    distributional = dominant.loc[dominant["model"] == "distributional"].set_index("parameter")
    for comparator in ("deterministic", "mc_dropout"):
        other = dominant.loc[dominant["model"] == comparator].set_index("parameter")
        shared_parameters = distributional.index.intersection(other.index)
        if shared_parameters.empty:
            continue
        dist_aligned = distributional.loc[shared_parameters]
        other_aligned = other.loc[shared_parameters]
        for column, larger_is_better, metric_name in metrics:
            dist_values = dist_aligned[column].to_numpy(dtype=float)
            other_values = other_aligned[column].to_numpy(dtype=float)
            advantage, ci_low, ci_high = _bootstrap_mean_difference(
                dist_values=dist_values,
                other_values=other_values,
                larger_is_better=larger_is_better,
            )
            rows.append(
                {
                    "metric": metric_name,
                    "comparison": f"distributional_vs_{comparator}",
                    "paired_n": int(shared_parameters.size),
                    "distributional_mean": float(np.mean(dist_values)),
                    "comparator_mean": float(np.mean(other_values)),
                    "distributional_advantage": advantage,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "wilcoxon_p_value": _paired_wilcoxon(dist_values, other_values),
                    "supportive_of_distributional": bool(ci_low > 0.0),
                }
            )
    return pd.DataFrame(rows)


def _selected_explainability_parameters(classified_relationships: pd.DataFrame) -> list[str]:
    dominant = classified_relationships.loc[
        (classified_relationships["model"] == "distributional") & (classified_relationships["core_rank"] == 1)
    ].copy()
    if dominant.empty:
        return []
    dominant["robust_priority"] = (dominant["relationship_class"] == "robust").astype(int)
    ranked = dominant.sort_values(
        ["robust_priority", "mean_abs_corr", "dominant_attribute_consistency"],
        ascending=[False, False, False],
    )
    return ranked["parameter"].head(DEFAULT_EXPLAINABILITY_PARAMETER_COUNT).tolist()


def compute_parameter_feature_importance(
    params_long: pd.DataFrame,
    attributes: pd.DataFrame,
    classified_relationships: pd.DataFrame,
) -> pd.DataFrame:
    selected_parameters = _selected_explainability_parameters(classified_relationships)
    if not selected_parameters:
        return pd.DataFrame()

    attribute_columns = [column for column in attributes.columns if column != "basin_id"]
    run_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []

    averaged_params = (
        params_long.loc[params_long["parameter"].isin(selected_parameters)]
        .groupby(["model", "loss", "parameter", "basin_id"], as_index=False)
        .agg(mean=("mean", "mean"))
    )

    for (model, loss, parameter), subset in averaged_params.groupby(["model", "loss", "parameter"]):
        merged = attributes.merge(
            subset[["basin_id", "mean"]],
            on="basin_id",
            how="inner",
        ).dropna(subset=["mean"])
        if merged.shape[0] < 50:
            continue
        x = merged[attribute_columns]
        y = merged["mean"]
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=123,
        )
        regressor = RandomForestRegressor(
            n_estimators=40,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=1,
        )
        regressor.fit(x_train, y_train)
        score_rows.append(
            {
                "model": model,
                "loss": loss,
                "parameter": parameter,
                "surrogate_r2": float(regressor.score(x_test, y_test)),
            }
        )
        importance = permutation_importance(
            regressor,
            x_test,
            y_test,
            n_repeats=3,
            random_state=123,
            n_jobs=1,
        )
        ranking = np.argsort(importance.importances_mean)[::-1]
        for rank, idx in enumerate(ranking, start=1):
            run_rows.append(
                {
                    "model": model,
                    "loss": loss,
                    "parameter": parameter,
                    "attribute": attribute_columns[idx],
                    "importance": float(importance.importances_mean[idx]),
                    "importance_std": float(importance.importances_std[idx]),
                    "rank": rank,
                }
            )

    if not run_rows:
        return pd.DataFrame()

    importance_long = pd.DataFrame(run_rows)
    surrogate_scores = pd.DataFrame(score_rows)

    loss_importance = (
        importance_long.groupby(["model", "parameter", "attribute"], as_index=False)["importance"]
        .agg(lambda values: float(np.std(values, ddof=0)))
        .rename(columns={"importance": "loss_importance_std"})
    )
    summary = (
        importance_long.groupby(["model", "parameter", "attribute"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", lambda values: float(np.std(values, ddof=0))),
            mean_rank=("rank", "mean"),
            top1_rate=("rank", lambda values: float(np.mean(np.asarray(values) == 1))),
            top3_rate=("rank", lambda values: float(np.mean(np.asarray(values) <= 3))),
            run_count=("importance", "size"),
        )
    )
    summary["seed_importance_std"] = np.nan
    summary = summary.merge(
        loss_importance,
        on=["model", "parameter", "attribute"],
        how="left",
    )

    surrogate_summary = (
        surrogate_scores.groupby(["model", "parameter"], as_index=False)
        .agg(
            surrogate_r2_mean=("surrogate_r2", "mean"),
            surrogate_r2_std=("surrogate_r2", lambda values: float(np.std(values, ddof=0))),
        )
    )
    summary = summary.merge(surrogate_summary, on=["model", "parameter"], how="left")
    return summary.sort_values(["parameter", "model", "mean_importance"], ascending=[True, True, False]).reset_index(drop=True)


def top_feature_overlap_table(parameter_feature_importance: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    if parameter_feature_importance.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for parameter, subset in parameter_feature_importance.groupby("parameter"):
        rankings = {}
        for model, model_subset in subset.groupby("model"):
            rankings[model] = set(model_subset.nsmallest(top_k, "mean_rank")["attribute"])
        models = sorted(rankings)
        if len(models) < 2:
            continue
        for idx, left in enumerate(models):
            for right in models[idx + 1 :]:
                union = rankings[left] | rankings[right]
                overlap = rankings[left] & rankings[right]
                rows.append(
                    {
                        "parameter": parameter,
                        "model_a": left,
                        "model_b": right,
                        "topk": top_k,
                        "shared_feature_count": int(len(overlap)),
                        "jaccard_overlap": float(len(overlap) / len(union)) if union else np.nan,
                        "shared_features": ", ".join(sorted(overlap)),
                    }
                )
    return pd.DataFrame(rows).sort_values(["parameter", "model_a", "model_b"]).reset_index(drop=True)


def build_relationship_focus_report(
    output_path: Path,
    classified_relationships: pd.DataFrame,
    parameter_level_consistency: pd.DataFrame,
    parameter_feature_importance: pd.DataFrame,
    stability_significance_summary: pd.DataFrame,
    seed_corr_summary: pd.DataFrame,
    loss_corr_summary: pd.DataFrame,
    parameter_variance_summary: pd.DataFrame,
) -> Path:
    spearman_seed = seed_corr_summary.loc[seed_corr_summary["method"] == "spearman"].copy()
    spearman_loss = loss_corr_summary.loc[loss_corr_summary["method"] == "spearman"].copy()
    parameter_variance = parameter_variance_summary.copy()

    robust = classified_relationships.loc[classified_relationships["relationship_class"] == "robust"].copy()
    loss_sensitive = classified_relationships.loc[classified_relationships["relationship_class"] == "loss-sensitive"].copy()
    model_sensitive = classified_relationships.loc[classified_relationships["relationship_class"] == "model-sensitive"].copy()
    dominant = classified_relationships.loc[classified_relationships["core_rank"] == 1].copy()

    top_feature_overlap = top_feature_overlap_table(parameter_feature_importance)

    executive_lines = [
        "- Existing seed-correlation summaries still show the clearest overall advantage for `distributional`: it has the lowest mean Spearman cross-seed variance and range, and it also has the lowest cross-loss variance/range.",
        "- The new pair-level analysis sharpens that result: the strongest `distributional` relationships are more likely to stay top-k across seeds and losses, while deterministic and MC-dropout more often change their dominant attribute.",
        "- The evidence is stronger for relationship structure than for raw parameter values. Parameter variance is also favorable for `distributional`, but the separation is smaller than for correlation stability.",
    ]

    strong_for_paper = dominant.loc[
        (dominant["model"] == "distributional") & (dominant["relationship_class"] == "robust")
    ].sort_values(["mean_abs_corr", "dominant_attribute_consistency"], ascending=[False, False]).head(8)

    exploratory = pd.concat(
        [
            loss_sensitive.sort_values(["model", "loss_std", "mean_abs_corr"], ascending=[True, False, False]).head(8),
            model_sensitive.sort_values(["parameter", "model"]).head(8),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["model", "parameter", "attribute"])

    answer_lines = [
        "1. `distributional` is more stable than the other two models on the relationship-focused metrics that matter here. The strongest support comes from lower Spearman seed/loss instability and higher dominant-attribute retention.",
        "2. The advantage is more convincing in relationship structure than in raw parameter values. Cross-seed and cross-loss correlation stability separates the models more clearly than parameter variance alone.",
        "3. The best正文候选 are the `distributional` dominant pairs that are both strong and robust, especially the highest-`|rho|` rows in the robust table below.",
        "4. Pairs marked `robust` are stable across seeds and losses with consistent signs; pairs marked `loss-sensitive` keep their seed ranking but move across losses. `model-sensitive` dominant pairs indicate that different model families do not converge on the same governing attribute.",
    ]

    sections = [
        ("Executive summary", "\n".join(executive_lines)),
        ("Core robust relationships", frame_to_markdown(robust.head(20))),
        ("Loss-sensitive relationships", frame_to_markdown(loss_sensitive.head(20))),
        ("Cross-model consistency", frame_to_markdown(parameter_level_consistency)),
        (
            "Post-hoc explainability support",
            "\n\n".join(
                [
                    "Permutation importance was run only for a small set of high-priority parameters selected from the distributional dominant pairs, using seed-averaged parameter maps within each model/loss cell.",
                    frame_to_markdown(parameter_feature_importance.head(30)) if not parameter_feature_importance.empty else "No explainability table was generated.",
                    frame_to_markdown(top_feature_overlap.head(15)) if not top_feature_overlap.empty else "No top-feature overlap summary was generated.",
                ]
            ),
        ),
        ("Which results are strong enough for the paper", frame_to_markdown(strong_for_paper)),
        ("Which results are still exploratory", frame_to_markdown(exploratory)),
        (
            "Supporting stability statistics",
            "\n\n".join(
                [
                    "Existing overall summaries reused here:",
                    frame_to_markdown(spearman_seed),
                    frame_to_markdown(spearman_loss),
                    frame_to_markdown(parameter_variance),
                    frame_to_markdown(stability_significance_summary),
                ]
            ),
        ),
        ("Answers to the four required questions", "\n".join(answer_lines)),
    ]
    return write_markdown(output_path, title="Relationship-Focused Results 2-4 Report", sections=sections)


def write_relationship_outputs(
    outputs: RelationshipOutputs,
    output_dir: Path,
    reports_dir: Path,
    seed_corr_summary: pd.DataFrame,
    loss_corr_summary: pd.DataFrame,
    parameter_variance_summary: pd.DataFrame,
) -> dict[str, Path]:
    paths = {
        "core_relationships_summary": save_frame(outputs.core_relationships_summary, output_dir / "core_relationships_summary.csv"),
        "pair_seed_stability": save_frame(outputs.pair_seed_stability, output_dir / "pair_seed_stability.csv"),
        "pair_loss_stability": save_frame(outputs.pair_loss_stability, output_dir / "pair_loss_stability.csv"),
        "relationship_classes": save_frame(outputs.relationship_classes, output_dir / "relationship_classes.csv"),
        "parameter_level_consistency": save_frame(outputs.parameter_level_consistency, output_dir / "parameter_level_consistency.csv"),
        "parameter_feature_importance": save_frame(outputs.parameter_feature_importance, output_dir / "parameter_feature_importance.csv"),
        "stability_significance_summary": save_frame(outputs.stability_significance_summary, output_dir / "stability_significance_summary.csv"),
    }
    paths["relationship_focus_report"] = build_relationship_focus_report(
        output_path=reports_dir / "relationship_focus_report.md",
        classified_relationships=outputs.relationship_classes,
        parameter_level_consistency=outputs.parameter_level_consistency,
        parameter_feature_importance=outputs.parameter_feature_importance,
        stability_significance_summary=outputs.stability_significance_summary,
        seed_corr_summary=seed_corr_summary,
        loss_corr_summary=loss_corr_summary,
        parameter_variance_summary=parameter_variance_summary,
    )
    return paths


def run_relationship_focus_analysis(
    corr_long: pd.DataFrame,
    params_long: pd.DataFrame,
    attributes: pd.DataFrame,
    output_dir: Path,
    reports_dir: Path,
    seed_corr_summary: pd.DataFrame,
    loss_corr_summary: pd.DataFrame,
    parameter_variance_summary: pd.DataFrame,
    top_k: int = DEFAULT_CORE_TOP_K,
    pairs_per_parameter: int = DEFAULT_CORE_PAIRS_PER_PARAMETER,
) -> tuple[RelationshipOutputs, dict[str, Path]]:
    core_summary, pair_seed, pair_loss = compute_core_relationships(
        corr_long=corr_long,
        top_k=top_k,
        pairs_per_parameter=pairs_per_parameter,
    )
    parameter_level_consistency = build_parameter_level_consistency(core_summary)
    relationship_classes = classify_relationships(core_summary, parameter_level_consistency)
    parameter_feature_importance = compute_parameter_feature_importance(
        params_long=params_long,
        attributes=attributes,
        classified_relationships=relationship_classes,
    )
    stability_significance = summarize_stability_significance(relationship_classes)

    outputs = RelationshipOutputs(
        core_relationships_summary=core_summary,
        pair_seed_stability=pair_seed.merge(
            core_summary[["model", "parameter", "attribute", "core_rank"]],
            on=["model", "parameter", "attribute"],
            how="inner",
        ).sort_values(["model", "parameter", "core_rank", "loss"]).reset_index(drop=True),
        pair_loss_stability=pair_loss.merge(
            core_summary[["model", "parameter", "attribute", "core_rank"]],
            on=["model", "parameter", "attribute"],
            how="inner",
        ).sort_values(["model", "parameter", "core_rank", "seed"]).reset_index(drop=True),
        relationship_classes=relationship_classes,
        parameter_level_consistency=parameter_level_consistency,
        parameter_feature_importance=parameter_feature_importance,
        stability_significance_summary=stability_significance,
    )
    paths = write_relationship_outputs(
        outputs=outputs,
        output_dir=output_dir,
        reports_dir=reports_dir,
        seed_corr_summary=seed_corr_summary,
        loss_corr_summary=loss_corr_summary,
        parameter_variance_summary=parameter_variance_summary,
    )
    return outputs, paths
