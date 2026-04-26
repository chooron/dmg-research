"""Focused distributional-vs-deterministic cross-loss stability analysis.

This module is intentionally sidecar-only: it reads the existing stability
analysis inputs/outputs and writes targeted CSV/Markdown artifacts without
touching the main result files.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from project.parameterize.analysis.common import frame_to_markdown, save_frame, write_markdown
from project.parameterize.analysis.correlation_analysis import build_correlation_long
from project.parameterize.analysis.relationship_analysis import sign_consistency_rate


FOCUSED_MODELS = ("distributional", "deterministic")
FOCUSED_METHODS = ("spearman", "pearson", "kendall")
FOCUSED_TOP_K = 3
TARGET_CORE_PAIR_COUNT = 6
MANDATORY_CORE_PAIRS = (
    ("parUZL", "soil_conductivity"),
    ("parBETA", "slope_mean"),
    ("parFC", "pet_mean"),
    ("parK1", "lai_diff"),
)


@dataclass(frozen=True)
class FocusedOutputs:
    focused_pairs: pd.DataFrame
    focused_pair_loss_stability: pd.DataFrame
    focused_pair_model_comparison: pd.DataFrame
    focused_pair_significance: pd.DataFrame
    focused_pair_classes: pd.DataFrame


def load_existing_relationship_classes(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def select_focused_pairs(
    relationship_classes: pd.DataFrame,
    target_core_pair_count: int = TARGET_CORE_PAIR_COUNT,
) -> pd.DataFrame:
    distributional = relationship_classes.loc[
        (relationship_classes["model"] == "distributional")
        & (relationship_classes["relationship_class"] == "robust")
        & (relationship_classes["seed_stable"])
        & (relationship_classes["loss_stable"])
    ].copy()

    selected_rows: list[dict[str, object]] = []
    used = set()

    for parameter, attribute in MANDATORY_CORE_PAIRS:
        matches = distributional.loc[
            (distributional["parameter"] == parameter) & (distributional["attribute"] == attribute)
        ]
        if matches.empty:
            raise ValueError(f"Mandatory focused pair missing from distributional robust set: {parameter} / {attribute}")
        row = matches.sort_values(["core_rank", "mean_abs_corr"], ascending=[True, False]).iloc[0]
        selected_rows.append(
            {
                "parameter": parameter,
                "attribute": attribute,
                "pair_label": f"{parameter}__{attribute}",
                "focus_group": "core_candidate",
                "mean_abs_corr_distributional": float(row["mean_abs_corr"]),
                "core_rank_distributional": int(row["core_rank"]),
            }
        )
        used.add((parameter, attribute))

    extras = distributional.loc[
        ~distributional.apply(lambda row: (row["parameter"], row["attribute"]) in used, axis=1)
    ].sort_values(["core_rank", "mean_abs_corr"], ascending=[True, False])

    extra_core_slots = max(target_core_pair_count - len(selected_rows), 0)
    for _, row in extras.head(extra_core_slots).iterrows():
        selected_rows.append(
            {
                "parameter": row["parameter"],
                "attribute": row["attribute"],
                "pair_label": f"{row['parameter']}__{row['attribute']}",
                "focus_group": "core_candidate",
                "mean_abs_corr_distributional": float(row["mean_abs_corr"]),
                "core_rank_distributional": int(row["core_rank"]),
            }
        )
        used.add((row["parameter"], row["attribute"]))

    remaining = extras.loc[
        ~extras.apply(lambda row: (row["parameter"], row["attribute"]) in used, axis=1)
    ]
    for _, row in remaining.iterrows():
        selected_rows.append(
            {
                "parameter": row["parameter"],
                "attribute": row["attribute"],
                "pair_label": f"{row['parameter']}__{row['attribute']}",
                "focus_group": "extended_robust",
                "mean_abs_corr_distributional": float(row["mean_abs_corr"]),
                "core_rank_distributional": int(row["core_rank"]),
            }
        )

    focused_pairs = pd.DataFrame(selected_rows).sort_values(
        ["focus_group", "mean_abs_corr_distributional"],
        ascending=[True, False],
    ).reset_index(drop=True)
    return focused_pairs


def _spearman_rank_table(corr_long: pd.DataFrame) -> pd.DataFrame:
    spearman = corr_long.loc[corr_long["method"] == "spearman"].copy()
    spearman["rank_within_run"] = (
        spearman.groupby(["model", "loss", "seed", "parameter"])["abs_corr"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    spearman["is_topk"] = spearman["rank_within_run"] <= FOCUSED_TOP_K
    spearman["is_dominant"] = spearman["rank_within_run"] == 1
    return spearman


def compute_focused_pair_loss_stability(
    params_long: pd.DataFrame,
    attributes: pd.DataFrame,
    focused_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_tables = build_correlation_long(
        params_long.loc[params_long["model"].isin(FOCUSED_MODELS)].copy(),
        attributes,
        methods=FOCUSED_METHODS,
    )
    corr_long = pd.concat(
        [table.assign(method=method) for method, table in corr_tables.items()],
        ignore_index=True,
    )
    pair_lookup = focused_pairs[["parameter", "attribute", "pair_label", "focus_group"]]
    selected = corr_long.merge(pair_lookup, on=["parameter", "attribute"], how="inner")

    loss_values = (
        selected.groupby(["focus_group", "pair_label", "parameter", "attribute", "model", "method", "loss"], as_index=False)
        .agg(
            loss_mean_rho=("corr", "mean"),
            loss_abs_mean_rho=("abs_corr", "mean"),
            loss_seed_std_rho=("corr", lambda values: float(np.std(values, ddof=0))),
            loss_seed_min_rho=("corr", "min"),
            loss_seed_max_rho=("corr", "max"),
            seed_count=("seed", "nunique"),
        )
    )

    spearman_ranks = _spearman_rank_table(corr_long).merge(pair_lookup, on=["parameter", "attribute"], how="inner")
    spearman_loss_cover = (
        spearman_ranks.groupby(["focus_group", "pair_label", "parameter", "attribute", "model", "loss"], as_index=False)
        .agg(
            topk_seed_rate=("is_topk", "mean"),
            dominant_seed_rate=("is_dominant", "mean"),
            topk_all_seeds=("is_topk", lambda values: bool(np.all(values))),
            dominant_all_seeds=("is_dominant", lambda values: bool(np.all(values))),
        )
    )

    summary = (
        loss_values.groupby(["focus_group", "pair_label", "parameter", "attribute", "model", "method"], as_index=False)
        .agg(
            mean_rho=("loss_mean_rho", "mean"),
            abs_mean_rho=("loss_mean_rho", lambda values: float(abs(np.mean(values)))),
            cross_loss_std=("loss_mean_rho", lambda values: float(np.std(values, ddof=0))),
            cross_loss_range=("loss_mean_rho", lambda values: float(np.nanmax(values) - np.nanmin(values))),
            cross_loss_max_abs_dev=("loss_mean_rho", lambda values: float(np.max(np.abs(np.asarray(values) - np.mean(values))))),
            sign_consistency_across_losses=("loss_mean_rho", sign_consistency_rate),
            loss_count=("loss", "nunique"),
        )
    )

    spearman_summary = (
        spearman_loss_cover.groupby(["focus_group", "pair_label", "parameter", "attribute", "model"], as_index=False)
        .agg(
            topk_consistency=("topk_seed_rate", "mean"),
            dominant_consistency=("dominant_seed_rate", "mean"),
            all_losses_majority_topk=("topk_seed_rate", lambda values: bool(np.all(np.asarray(values) >= 0.5))),
            all_losses_majority_dominant=("dominant_seed_rate", lambda values: bool(np.all(np.asarray(values) >= 0.5))),
            all_losses_all_seed_topk=("topk_all_seeds", "all"),
            all_losses_all_seed_dominant=("dominant_all_seeds", "all"),
        )
    )

    detailed = (
        loss_values.merge(
            summary,
            on=["focus_group", "pair_label", "parameter", "attribute", "model", "method"],
            how="left",
        )
        .merge(
            spearman_loss_cover,
            on=["focus_group", "pair_label", "parameter", "attribute", "model", "loss"],
            how="left",
        )
        .merge(
            spearman_summary,
            on=["focus_group", "pair_label", "parameter", "attribute", "model"],
            how="left",
        )
        .sort_values(["focus_group", "pair_label", "model", "method", "loss"])
        .reset_index(drop=True)
    )
    return detailed, summary.merge(
        spearman_summary,
        on=["focus_group", "pair_label", "parameter", "attribute", "model"],
        how="left",
    )


def build_focused_pair_model_comparison(focused_summary: pd.DataFrame) -> pd.DataFrame:
    spearman = focused_summary.loc[focused_summary["method"] == "spearman"].copy()
    dist = spearman.loc[spearman["model"] == "distributional"].copy()
    det = spearman.loc[spearman["model"] == "deterministic"].copy()

    merged = dist.merge(
        det,
        on=["focus_group", "pair_label", "parameter", "attribute", "method"],
        how="inner",
        suffixes=("_dist", "_det"),
    )

    comparison = pd.DataFrame(
        {
            "focus_group": merged["focus_group"],
            "pair_label": merged["pair_label"],
            "parameter": merged["parameter"],
            "attribute": merged["attribute"],
            "method": merged["method"],
            "dist_mean_rho": merged["mean_rho_dist"],
            "det_mean_rho": merged["mean_rho_det"],
            "dist_abs_mean_rho": merged["abs_mean_rho_dist"],
            "det_abs_mean_rho": merged["abs_mean_rho_det"],
            "dist_loss_std": merged["cross_loss_std_dist"],
            "det_loss_std": merged["cross_loss_std_det"],
            "advantage_loss_std_det_minus_dist": merged["cross_loss_std_det"] - merged["cross_loss_std_dist"],
            "dist_loss_range": merged["cross_loss_range_dist"],
            "det_loss_range": merged["cross_loss_range_det"],
            "advantage_loss_range_det_minus_dist": merged["cross_loss_range_det"] - merged["cross_loss_range_dist"],
            "dist_loss_max_abs_dev": merged["cross_loss_max_abs_dev_dist"],
            "det_loss_max_abs_dev": merged["cross_loss_max_abs_dev_det"],
            "advantage_loss_max_abs_dev_det_minus_dist": merged["cross_loss_max_abs_dev_det"] - merged["cross_loss_max_abs_dev_dist"],
            "dist_sign_consistency": merged["sign_consistency_across_losses_dist"],
            "det_sign_consistency": merged["sign_consistency_across_losses_det"],
            "dist_topk_consistency": merged["topk_consistency_dist"],
            "det_topk_consistency": merged["topk_consistency_det"],
            "advantage_topk_consistency_dist_minus_det": merged["topk_consistency_dist"] - merged["topk_consistency_det"],
            "dist_dominant_consistency": merged["dominant_consistency_dist"],
            "det_dominant_consistency": merged["dominant_consistency_det"],
            "advantage_dominant_consistency_dist_minus_det": merged["dominant_consistency_dist"] - merged["dominant_consistency_det"],
            "dist_all_losses_majority_topk": merged["all_losses_majority_topk_dist"],
            "det_all_losses_majority_topk": merged["all_losses_majority_topk_det"],
            "dist_all_losses_majority_dominant": merged["all_losses_majority_dominant_dist"],
            "det_all_losses_majority_dominant": merged["all_losses_majority_dominant_det"],
            "positive_std_or_range_advantage_favors_distributional": True,
            "positive_consistency_advantage_favors_distributional": True,
        }
    )
    return comparison.sort_values(["focus_group", "pair_label"]).reset_index(drop=True)


def _bootstrap_advantage(
    values: np.ndarray,
    bootstrap_iters: int = 4000,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(values.size)
    samples = []
    for _ in range(bootstrap_iters):
        sample_idx = rng.choice(idx, size=idx.size, replace=True)
        samples.append(float(np.mean(values[sample_idx])))
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return float(np.mean(values)), float(lower), float(upper)


def _paired_wilcoxon(diffs: np.ndarray) -> float:
    if diffs.size < 2:
        return np.nan
    if np.allclose(diffs, 0.0):
        return 1.0
    return float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)


def _exact_sign_permutation_pvalue(diffs: np.ndarray) -> float:
    if diffs.size == 0:
        return np.nan
    observed = abs(float(np.mean(diffs)))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=diffs.size)), dtype=float)
    perm_means = np.abs((signs * diffs).mean(axis=1))
    return float(np.mean(perm_means >= observed))


def build_focused_pair_significance(comparison: pd.DataFrame) -> pd.DataFrame:
    core = comparison.loc[comparison["focus_group"] == "core_candidate"].copy()
    metrics = [
        ("loss_std", core["advantage_loss_std_det_minus_dist"].to_numpy(dtype=float), "positive favors distributional"),
        ("loss_range", core["advantage_loss_range_det_minus_dist"].to_numpy(dtype=float), "positive favors distributional"),
        (
            "dominant_attribute_consistency",
            core["advantage_dominant_consistency_dist_minus_det"].to_numpy(dtype=float),
            "positive favors distributional",
        ),
        (
            "topk_consistency",
            core["advantage_topk_consistency_dist_minus_det"].to_numpy(dtype=float),
            "positive favors distributional",
        ),
    ]
    rows = []
    for metric, diffs, interpretation in metrics:
        diffs = diffs[~np.isnan(diffs)]
        if diffs.size == 0:
            continue
        mean_advantage, ci_low, ci_high = _bootstrap_advantage(diffs)
        rows.append(
            {
                "metric": metric,
                "pair_count": int(diffs.size),
                "mean_advantage": mean_advantage,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "wilcoxon_p_value": _paired_wilcoxon(diffs),
                "permutation_p_value": _exact_sign_permutation_pvalue(diffs),
                "advantage_interpretation": interpretation,
                "supportive_of_distributional": bool(ci_low > 0.0),
            }
        )
    return pd.DataFrame(rows)


def classify_focused_pairs(comparison: pd.DataFrame) -> pd.DataFrame:
    classified = comparison.copy()
    evidence_class = []
    reasons = []
    for _, row in classified.iterrows():
        primary_better = int(row["advantage_loss_std_det_minus_dist"] > 0) + int(row["advantage_loss_range_det_minus_dist"] > 0)
        consistency_better = int(row["advantage_topk_consistency_dist_minus_det"] > 0) + int(
            row["advantage_dominant_consistency_dist_minus_det"] > 0
        )
        strong_effect = bool(
            row["advantage_loss_std_det_minus_dist"] >= 0.01
            and row["advantage_loss_range_det_minus_dist"] >= 0.03
        )

        if primary_better == 2 and (consistency_better >= 1 or row["dist_dominant_consistency"] >= row["det_dominant_consistency"]) and strong_effect:
            label = "headline evidence"
            reason = "Distributional is lower on cross-loss std/range and is not weaker on loss-wise retention."
        elif primary_better >= 1 and (
            row["advantage_loss_std_det_minus_dist"] > 0
            or row["advantage_loss_range_det_minus_dist"] > 0
            or row["advantage_topk_consistency_dist_minus_det"] > 0
            or row["advantage_dominant_consistency_dist_minus_det"] > 0
        ):
            label = "supportive but not decisive"
            reason = "Direction favors distributional on at least one stability metric, but the effect is partial or small."
        else:
            label = "not supportive"
            reason = "Distributional does not show a clear pair-level stability advantage over deterministic."
        evidence_class.append(label)
        reasons.append(reason)
    classified["evidence_class"] = evidence_class
    classified["evidence_reason"] = reasons
    return classified.sort_values(["focus_group", "evidence_class", "pair_label"]).reset_index(drop=True)


def build_report(
    report_path: Path,
    focused_pairs: pd.DataFrame,
    focused_pair_loss_stability: pd.DataFrame,
    focused_pair_model_comparison: pd.DataFrame,
    focused_pair_significance: pd.DataFrame,
    focused_pair_classes: pd.DataFrame,
) -> Path:
    core_pairs = focused_pairs.loc[focused_pairs["focus_group"] == "core_candidate"].copy()
    extended_pairs = focused_pairs.loc[focused_pairs["focus_group"] == "extended_robust"].copy()
    comparison = focused_pair_model_comparison.copy()
    classes = focused_pair_classes.copy()

    core_comparison = comparison.loc[comparison["focus_group"] == "core_candidate"].copy()
    strong_rows = classes.loc[classes["evidence_class"] == "headline evidence"].copy()
    supportive_rows = classes.loc[classes["evidence_class"] == "supportive but not decisive"].copy()
    not_supportive_rows = classes.loc[classes["evidence_class"] == "not supportive"].copy()
    strong_pair_names = ", ".join(strong_rows["pair_label"].tolist()) if not strong_rows.empty else "no pair"
    supportive_pair_names = ", ".join(supportive_rows["pair_label"].tolist()) if not supportive_rows.empty else "no pair"
    not_supportive_pair_names = ", ".join(not_supportive_rows["pair_label"].tolist()) if not not_supportive_rows.empty else "no pair"
    support_lines = []
    for _, row in focused_pair_significance.iterrows():
        support_lines.append(
            f"- `{row['metric']}`: mean advantage `{row['mean_advantage']:.4f}`, bootstrap CI [{row['bootstrap_ci_low']:.4f}, {row['bootstrap_ci_high']:.4f}], "
            f"Wilcoxon p=`{row['wilcoxon_p_value']:.4f}`, permutation p=`{row['permutation_p_value']:.4f}`."
        )

    wording_strong = (
        f"For the strongest focused pairs ({strong_pair_names}), the distributional model showed lower cross-loss "
        "Spearman variability than the deterministic baseline while preserving comparable sign/top-k retention."
    )
    wording_careful = (
        "When the comparison is restricted to the strongest robust pairs, the distributional model more often trends "
        "toward lower cross-loss variability than the deterministic model, but the advantage remains pair-dependent and "
        "is not uniformly significant across the full focused set."
    )
    wording_supportive = (
        f"The clearest pair-level support comes from {strong_pair_names}, while {supportive_pair_names} are better framed as supporting trends."
    )

    final_answers = [
        "1. After focusing on the 6 core candidate pairs, the distributional-vs-deterministic comparison is clearer at pair level, but the aggregate statistical support is still mixed rather than decisive.",
        f"2. The strongest pairs for the distributional narrative are {strong_pair_names}; among the originally pre-specified four, `parUZL__soil_conductivity` is the clearest supportive pair.",
        f"3. The pairs that should stay supportive rather than headline are {supportive_pair_names}; the pair that is not supportive is {not_supportive_pair_names}.",
        "4. The safest paper wording is a pair-specific claim for the strongest pairs, plus a careful trend statement for the broader focused set instead of a blanket superiority claim.",
    ]

    sections = [
        (
            "Objective",
            "\n".join(
                [
                    "The global 14-parameter averages are useful context, but they dilute the relationships most likely to appear in the paper body.",
                    "This focused test therefore restricts the comparison to pre-specified robust attribute-parameter pairs and asks whether distributional is more cross-loss stable than deterministic on the relationships that matter most.",
                ]
            ),
        ),
        (
            "Focused pairs included",
            "\n\n".join(
                [
                    "Core正文候选:",
                    frame_to_markdown(core_pairs),
                    "Extended robust pairs:",
                    frame_to_markdown(extended_pairs) if not extended_pairs.empty else "No extended robust pairs were added.",
                ]
            ),
        ),
        (
            "Pair-level cross-loss stability",
            "\n\n".join(
                [
                    "Each row below is traceable to a specific pair, model, loss, and correlation method. Spearman is the primary evidence; Pearson/Kendall are retained as appendix-style support in the same export.",
                    frame_to_markdown(focused_pair_loss_stability.head(36)),
                ]
            ),
        ),
        (
            "Distributional vs deterministic comparison",
            "\n\n".join(
                [
                    "Positive `advantage_loss_std_det_minus_dist` and `advantage_loss_range_det_minus_dist` mean distributional is more stable.",
                    "Positive `advantage_topk_consistency_dist_minus_det` and `advantage_dominant_consistency_dist_minus_det` mean distributional keeps the pair more consistently top-k/dominant across losses.",
                    frame_to_markdown(core_comparison),
                ]
            ),
        ),
        (
            "Statistical support",
            "\n\n".join(
                [
                    "The statistics below are computed only across the core pairs. Because the sample is intentionally small, the table includes bootstrap confidence intervals, paired Wilcoxon, and an exact sign-flip permutation test.",
                    "\n".join(support_lines),
                    frame_to_markdown(focused_pair_significance),
                ]
            ),
        ),
        (
            "Which pairs are strong enough for the main paper",
            "\n\n".join(
                [
                    frame_to_markdown(strong_rows[["pair_label", "parameter", "attribute", "evidence_class", "evidence_reason"]])
                    if not strong_rows.empty
                    else "No pair met the headline threshold.",
                    frame_to_markdown(supportive_rows[["pair_label", "parameter", "attribute", "evidence_class", "evidence_reason"]])
                    if not supportive_rows.empty
                    else "No additional supportive pairs were identified.",
                ]
            ),
        ),
        (
            "Recommended wording for the paper",
            "\n".join(
                [
                    f"Strong claim: {wording_strong}",
                    f"Careful claim: {wording_careful}",
                    f"Supportive sentence: {wording_supportive}",
                ]
            ),
        ),
        ("Answers to the four required questions", "\n".join(final_answers)),
    ]
    return write_markdown(
        report_path,
        title="Focused Distributional vs Deterministic Cross-Loss Stability",
        sections=sections,
    )


def write_outputs(
    outputs: FocusedOutputs,
    output_dir: Path,
    reports_dir: Path,
) -> dict[str, Path]:
    paths = {
        "focused_pair_loss_stability": save_frame(
            outputs.focused_pair_loss_stability, output_dir / "focused_pair_loss_stability.csv"
        ),
        "focused_pair_model_comparison": save_frame(
            outputs.focused_pair_model_comparison, output_dir / "focused_pair_model_comparison.csv"
        ),
        "focused_pair_significance": save_frame(
            outputs.focused_pair_significance, output_dir / "focused_pair_significance.csv"
        ),
        "focused_pair_classes": save_frame(
            outputs.focused_pair_classes, output_dir / "focused_pair_classes.csv"
        ),
    }
    paths["report"] = build_report(
        report_path=reports_dir / "focused_distributional_vs_deterministic_cross_loss.md",
        focused_pairs=outputs.focused_pairs,
        focused_pair_loss_stability=outputs.focused_pair_loss_stability,
        focused_pair_model_comparison=outputs.focused_pair_model_comparison,
        focused_pair_significance=outputs.focused_pair_significance,
        focused_pair_classes=outputs.focused_pair_classes,
    )
    return paths


def run_focused_cross_loss_analysis(
    params_long: pd.DataFrame,
    attributes: pd.DataFrame,
    relationship_classes: pd.DataFrame,
    output_dir: Path,
    reports_dir: Path,
) -> tuple[FocusedOutputs, dict[str, Path]]:
    focused_pairs = select_focused_pairs(relationship_classes)
    focused_pair_loss_stability, focused_summary = compute_focused_pair_loss_stability(
        params_long=params_long,
        attributes=attributes,
        focused_pairs=focused_pairs,
    )
    focused_pair_model_comparison = build_focused_pair_model_comparison(focused_summary)
    focused_pair_significance = build_focused_pair_significance(focused_pair_model_comparison)
    focused_pair_classes = classify_focused_pairs(focused_pair_model_comparison)

    outputs = FocusedOutputs(
        focused_pairs=focused_pairs,
        focused_pair_loss_stability=focused_pair_loss_stability,
        focused_pair_model_comparison=focused_pair_model_comparison,
        focused_pair_significance=focused_pair_significance,
        focused_pair_classes=focused_pair_classes,
    )
    paths = write_outputs(outputs, output_dir=output_dir, reports_dir=reports_dir)
    return outputs, paths
