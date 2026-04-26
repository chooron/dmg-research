"""Standalone analysis for Results 3.3.2, 3.3.3, 3.4.2, and 3.4.3."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from project.parameterize.analysis.common import (
    build_parser,
    correlation_value,
    frame_to_markdown,
    load_analysis_data,
    normalize_parameters_to_unit_interval,
    save_frame,
    save_json,
    write_markdown,
)
from project.parameterize.analysis.results331_results341_analysis import (
    GRADIENT_ATTRIBUTES,
    RESULTS341_ATTRIBUTES,
    RESULTS341_PARAMETERS,
)
from project.parameterize.figures.common import (
    COLORS,
    MODEL_ORDER,
    apply_wrr_style,
    pretty_model_name,
    pretty_parameter_name,
    save_figure,
)


RESULTS333_PARAMETERS = ["parBETA", "parFC", "parPERC", "parUZL", "parCFR"]
RESULTS342_PARAMETERS = ["parBETA", "parFC", "parPERC", "parUZL", "parCFR", "parCWH"]
RESULTS342_ATTRIBUTES = RESULTS341_ATTRIBUTES
RESULTS343_PARAMETERS = RESULTS342_PARAMETERS
RESULTS343_GROUPS = ["aridity", "frac_snow", "slope_mean"]
RESULTS332_METHODS = ("spearman", "pearson", "kendall")


def _build_correlation_long_value(
    params_long: pd.DataFrame,
    attributes: pd.DataFrame,
    value_column: str,
    methods: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    merged = params_long.merge(attributes, on="basin_id", how="inner")
    attribute_columns = [column for column in attributes.columns if column != "basin_id"]
    rows_by_method = {method: [] for method in methods}

    for (model, loss, seed, parameter), subset in merged.groupby(["model", "loss", "seed", "parameter"]):
        values = subset[value_column]
        for attribute in attribute_columns:
            for method in methods:
                corr_value, p_value = correlation_value(values, subset[attribute], method=method)
                rows_by_method[method].append(
                    {
                        "model": model,
                        "loss": loss,
                        "seed": int(seed),
                        "parameter": parameter,
                        "attribute": attribute,
                        "corr": corr_value,
                        "p_value": p_value,
                        "abs_corr": float(abs(corr_value)) if not np.isnan(corr_value) else np.nan,
                    }
                )
    return {method: pd.DataFrame(rows) for method, rows in rows_by_method.items()}


def _run_id(frame: pd.DataFrame) -> pd.Series:
    return frame["model"] + "__" + frame["loss"] + "__seed_" + frame["seed"].astype(str)


def build_results332_outputs(
    params_long: pd.DataFrame,
    attributes: pd.DataFrame,
    output_dir: Path,
    figures_dir: Path,
    reports_dir: Path,
) -> dict[str, Path]:
    corr_tables = _build_correlation_long_value(params_long, attributes, value_column="mean", methods=RESULTS332_METHODS)
    spearman = corr_tables["spearman"].copy()
    spearman["run_id"] = _run_id(spearman)
    parquet_path = output_dir / "results332_spearman_matrices.parquet"
    spearman.to_parquet(parquet_path, index=False)

    attribute_order = sorted(attributes.columns.drop("basin_id"))
    parameter_order = sorted(params_long["parameter"].unique())
    matrix_table = (
        spearman.assign(
            attribute=pd.Categorical(spearman["attribute"], categories=attribute_order, ordered=True),
            parameter=pd.Categorical(spearman["parameter"], categories=parameter_order, ordered=True),
        )
        .pivot_table(
            index=["run_id", "model", "loss", "seed"],
            columns=["attribute", "parameter"],
            values="corr",
        )
        .sort_index(axis=1)
    )
    run_meta = matrix_table.index.to_frame(index=False)
    vectors = matrix_table.to_numpy(dtype=float)
    run_ids = run_meta["run_id"].tolist()

    similarity_rows = []
    distance_matrix = np.zeros((len(run_ids), len(run_ids)), dtype=float)
    corr_matrix = np.zeros((len(run_ids), len(run_ids)), dtype=float)

    for i, j in itertools.combinations_with_replacement(range(len(run_ids)), 2):
        left = vectors[i]
        right = vectors[j]
        pearson_corr = float(pearsonr(left, right).statistic) if i != j else 1.0
        spearman_corr = float(spearmanr(left, right).statistic) if i != j else 1.0
        frobenius = float(np.linalg.norm(left - right))
        cosine = float(cosine_similarity(left.reshape(1, -1), right.reshape(1, -1))[0, 0]) if i != j else 1.0
        distance_matrix[i, j] = distance_matrix[j, i] = frobenius
        corr_matrix[i, j] = corr_matrix[j, i] = pearson_corr
        row = {
            "run_id_a": run_ids[i],
            "run_id_b": run_ids[j],
            "model_a": run_meta.loc[i, "model"],
            "model_b": run_meta.loc[j, "model"],
            "loss_a": run_meta.loc[i, "loss"],
            "loss_b": run_meta.loc[j, "loss"],
            "seed_a": int(run_meta.loc[i, "seed"]),
            "seed_b": int(run_meta.loc[j, "seed"]),
            "matrix_corr_pearson": pearson_corr,
            "matrix_corr_spearman": spearman_corr,
            "frobenius_distance": frobenius,
            "cosine_similarity": cosine,
            "same_model": bool(run_meta.loc[i, "model"] == run_meta.loc[j, "model"]),
            "same_loss": bool(run_meta.loc[i, "loss"] == run_meta.loc[j, "loss"]),
            "same_seed": bool(run_meta.loc[i, "seed"] == run_meta.loc[j, "seed"]),
        }
        similarity_rows.append(row)
        if i != j:
            similarity_rows.append({**row, "run_id_a": row["run_id_b"], "run_id_b": row["run_id_a"], "model_a": row["model_b"], "model_b": row["model_a"], "loss_a": row["loss_b"], "loss_b": row["loss_a"], "seed_a": row["seed_b"], "seed_b": row["seed_a"]})

    similarity = pd.DataFrame(similarity_rows)
    similarity_path = save_frame(similarity, output_dir / "results332_matrix_similarity.csv")

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
    coords = mds.fit_transform(distance_matrix)
    pca = PCA(n_components=2, random_state=42).fit_transform(vectors)
    embedding = run_meta.copy()
    embedding["mds_x"] = coords[:, 0]
    embedding["mds_y"] = coords[:, 1]
    embedding["pca_x"] = pca[:, 0]
    embedding["pca_y"] = pca[:, 1]
    embedding_path = save_frame(embedding, output_dir / "results332_matrix_embedding.csv")

    apply_wrr_style()
    corr_df = pd.DataFrame(corr_matrix, index=run_ids, columns=run_ids)
    row_colors = run_meta["model"].map(COLORS).tolist()
    cluster = sns.clustermap(
        corr_df,
        cmap="vlag",
        center=0.0,
        figsize=(12, 12),
        row_colors=row_colors,
        col_colors=row_colors,
        xticklabels=False,
        yticklabels=False,
    )
    heatmap_paths = save_figure(cluster.fig, "results332_matrix_similarity_heatmap", figures_dir, formats=("png",))

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    for model in MODEL_ORDER:
        subset = embedding.loc[embedding["model"] == model]
        ax.scatter(subset["mds_x"], subset["mds_y"], s=45, color=COLORS[model], label=pretty_model_name(model), alpha=0.85)
        for _, row in subset.iterrows():
            ax.text(row["mds_x"] + 0.01, row["mds_y"] + 0.01, f"{row['loss'].replace('BatchLoss','')}:{row['seed']}", fontsize=6)
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.legend(frameon=False)
    ax.set_title("Matrix-Level Correlation Structure Embedding")
    embedding_paths = save_figure(fig, "results332_matrix_embedding", figures_dir, formats=("png",))

    within_summary = (
        similarity.loc[(similarity["same_model"]) & (similarity["run_id_a"] != similarity["run_id_b"])]
        .groupby("model_a", as_index=False)
        .agg(
            mean_frobenius_distance=("frobenius_distance", "mean"),
            mean_matrix_corr=("matrix_corr_pearson", "mean"),
            mean_cosine_similarity=("cosine_similarity", "mean"),
        )
        .rename(columns={"model_a": "model"})
    )
    centroid_rows = []
    centroids = {model: vectors[run_meta["model"] == model].mean(axis=0) for model in MODEL_ORDER}
    for left, right in itertools.combinations(MODEL_ORDER, 2):
        centroid_rows.append(
            {
                "model_a": left,
                "model_b": right,
                "centroid_corr_pearson": float(pearsonr(centroids[left], centroids[right]).statistic),
                "centroid_corr_spearman": float(spearmanr(centroids[left], centroids[right]).statistic),
                "centroid_frobenius_distance": float(np.linalg.norm(centroids[left] - centroids[right])),
            }
        )
    centroid_df = pd.DataFrame(centroid_rows)
    closest_pairs = similarity.loc[similarity["run_id_a"] != similarity["run_id_b"]].sort_values("frobenius_distance").head(10)
    farthest_pairs = similarity.loc[similarity["run_id_a"] != similarity["run_id_b"]].sort_values("frobenius_distance", ascending=False).head(10)

    report = write_markdown(
        reports_dir / "results332_report.md",
        title="Results 3.3.2 Matrix-Level Similarity",
        sections=[
            ("Within-Model Compactness", frame_to_markdown(within_summary)),
            ("Model Centroid Similarity", frame_to_markdown(centroid_df)),
            ("Closest Run Pairs", frame_to_markdown(closest_pairs)),
            ("Farthest Run Pairs", frame_to_markdown(farthest_pairs)),
            (
                "Direct Answers",
                "\n".join(
                    [
                        "1. Shared structure can be assessed from the model-centroid correlations and the clustered heatmap.",
                        "2. Distributional coherence can be assessed from its within-model compactness relative to deterministic and mc_dropout.",
                        "3. The most similar and most distant run pairs are listed above.",
                        "4. Figure files: "
                        f"`{heatmap_paths['png']}`, `{embedding_paths['png']}`",
                    ]
                ),
            ),
        ],
    )
    return {
        "results332_spearman_matrices": parquet_path,
        "results332_matrix_similarity": similarity_path,
        "results332_matrix_embedding": embedding_path,
        "results332_heatmap": heatmap_paths["png"],
        "results332_embedding": embedding_paths["png"],
        "results332_report": report,
    }


def _selected_attribute_columns(attributes: pd.DataFrame) -> list[str]:
    return [column for column in attributes.columns if column != "basin_id"]


def _importance_runs(
    params_long: pd.DataFrame,
    attributes: pd.DataFrame,
    selected_parameters: Iterable[str],
) -> pd.DataFrame:
    attr_cols = _selected_attribute_columns(attributes)
    rows = []
    for (model, loss, seed, parameter), subset in params_long.loc[
        params_long["parameter"].isin(selected_parameters)
    ].groupby(["model", "loss", "seed", "parameter"]):
        merged = attributes.merge(subset[["basin_id", "mean"]], on="basin_id", how="inner").dropna(subset=["mean"])
        if merged.shape[0] < 50:
            continue
        x = merged[attr_cols]
        y = merged["mean"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=seed)
        regressor = RandomForestRegressor(
            n_estimators=40,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=1,
        )
        regressor.fit(x_train, y_train)
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
            rows.append(
                {
                    "model": model,
                    "loss": loss,
                    "seed": int(seed),
                    "parameter": parameter,
                    "attribute": attr_cols[idx],
                    "importance": float(importance.importances_mean[idx]),
                    "importance_std": float(importance.importances_std[idx]),
                    "rank": rank,
                    "surrogate_r2": float(regressor.score(x_test, y_test)),
                }
            )
    return pd.DataFrame(rows)


def build_results333_outputs(
    params_long: pd.DataFrame,
    attributes: pd.DataFrame,
    dominant_summary: pd.DataFrame,
    output_dir: Path,
    figures_dir: Path,
    reports_dir: Path,
) -> dict[str, Path]:
    importance_runs = _importance_runs(params_long, attributes, RESULTS333_PARAMETERS)
    loss_level = (
        importance_runs.groupby(["model", "loss", "parameter", "attribute"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            seed_level_std=("importance", lambda values: float(np.std(values, ddof=0))),
            mean_rank=("rank", "mean"),
            top1_rate=("rank", lambda values: float(np.mean(np.asarray(values) == 1))),
            top3_rate=("rank", lambda values: float(np.mean(np.asarray(values) <= 3))),
            surrogate_r2_mean=("surrogate_r2", "mean"),
        )
    )
    loss_std = (
        loss_level.groupby(["model", "parameter", "attribute"], as_index=False)["mean_importance"]
        .agg(lambda values: float(np.std(values, ddof=0)))
        .rename(columns={"mean_importance": "loss_level_std"})
    )
    importance_summary = loss_level.merge(loss_std, on=["model", "parameter", "attribute"], how="left")
    importance_path = save_frame(importance_summary, output_dir / "results333_parameter_feature_importance.csv")

    dominant_long = dominant_summary.melt(
        id_vars=["parameter"],
        value_vars=["deterministic_attribute", "mc_dropout_attribute", "distributional_attribute"],
        var_name="model_field",
        value_name="dominant_attribute",
    )
    dominant_long = dominant_long.loc[dominant_long["parameter"].isin(RESULTS333_PARAMETERS)].copy()
    dominant_long["model"] = dominant_long["model_field"].str.replace("_attribute", "", regex=False)
    alignment_rows = []
    overall_importance = (
        importance_runs.groupby(["model", "parameter", "attribute"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            mean_rank=("rank", "mean"),
            top1_rate=("rank", lambda values: float(np.mean(np.asarray(values) == 1))),
            top3_rate=("rank", lambda values: float(np.mean(np.asarray(values) <= 3))),
        )
    )
    for _, row in dominant_long.iterrows():
        matches = overall_importance.loc[
            (overall_importance["model"] == row["model"])
            & (overall_importance["parameter"] == row["parameter"])
            & (overall_importance["attribute"] == row["dominant_attribute"])
        ]
        if matches.empty:
            alignment_rows.append(
                {
                    "model": row["model"],
                    "parameter": row["parameter"],
                    "dominant_attribute": row["dominant_attribute"],
                    "dominant_attribute_mean_rank": np.nan,
                    "dominant_attribute_top1_rate": 0.0,
                    "dominant_attribute_top3_rate": 0.0,
                    "dominant_attribute_in_top1": False,
                    "dominant_attribute_in_top3": False,
                }
            )
        else:
            match = matches.iloc[0]
            alignment_rows.append(
                {
                    "model": row["model"],
                    "parameter": row["parameter"],
                    "dominant_attribute": row["dominant_attribute"],
                    "dominant_attribute_mean_rank": match["mean_rank"],
                    "dominant_attribute_top1_rate": match["top1_rate"],
                    "dominant_attribute_top3_rate": match["top3_rate"],
                    "dominant_attribute_in_top1": bool(match["top1_rate"] > 0),
                    "dominant_attribute_in_top3": bool(match["top3_rate"] > 0),
                }
            )
    alignment = pd.DataFrame(alignment_rows)
    alignment_path = save_frame(alignment, output_dir / "results333_importance_alignment.csv")

    overlap_rows = []
    overall_importance = overall_importance.sort_values(["model", "parameter", "mean_rank"])
    for parameter in RESULTS333_PARAMETERS:
        param_table = overall_importance.loc[overall_importance["parameter"] == parameter]
        rankings = {model: set(param_table.loc[param_table["model"] == model].nsmallest(3, "mean_rank")["attribute"]) for model in MODEL_ORDER}
        for left, right in itertools.combinations(MODEL_ORDER, 2):
            union = rankings[left] | rankings[right]
            inter = rankings[left] & rankings[right]
            overlap_rows.append(
                {
                    "parameter": parameter,
                    "model_a": left,
                    "model_b": right,
                    "topk": 3,
                    "jaccard_overlap": float(len(inter) / len(union)) if union else np.nan,
                    "shared_feature_count": int(len(inter)),
                    "shared_features": ", ".join(sorted(inter)),
                }
            )
    overlap = pd.DataFrame(overlap_rows)
    overlap_path = save_frame(overlap, output_dir / "results333_importance_overlap.csv")

    apply_wrr_style()
    fig, axes = plt.subplots(len(RESULTS333_PARAMETERS), len(MODEL_ORDER), figsize=(12, 2.4 * len(RESULTS333_PARAMETERS)), sharex=False)
    for row_idx, parameter in enumerate(RESULTS333_PARAMETERS):
        param_table = overall_importance.loc[overall_importance["parameter"] == parameter]
        for col_idx, model in enumerate(MODEL_ORDER):
            ax = axes[row_idx, col_idx] if len(RESULTS333_PARAMETERS) > 1 else axes[col_idx]
            model_table = param_table.loc[param_table["model"] == model].nsmallest(5, "mean_rank").sort_values("mean_importance")
            ax.barh(model_table["attribute"], model_table["mean_rank"].max() - model_table["mean_rank"] + 1, color=COLORS[model])
            if row_idx == 0:
                ax.set_title(pretty_model_name(model))
            if col_idx == 0:
                ax.set_ylabel(pretty_parameter_name(parameter))
            else:
                ax.set_ylabel("")
            ax.set_xlabel("relative top rank")
    rankings_paths = save_figure(fig, "results333_importance_rankings", figures_dir, formats=("png",))

    pivot_overlap = overlap.pivot(index="parameter", columns=["model_a", "model_b"], values="jaccard_overlap")
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    sns.heatmap(pivot_overlap, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "Top-3 Jaccard overlap"}, ax=ax)
    ax.set_yticklabels([pretty_parameter_name(param) for param in pivot_overlap.index], rotation=0)
    overlap_paths = save_figure(fig, "results333_importance_overlap", figures_dir, formats=("png",))

    best_alignment = alignment.sort_values(["dominant_attribute_top3_rate", "dominant_attribute_top1_rate"], ascending=False)
    report = write_markdown(
        reports_dir / "results333_report.md",
        title="Results 3.3.3 Post-hoc Explainability",
        sections=[
            ("Dominant-Attribute Alignment", frame_to_markdown(best_alignment)),
            ("Top-3 Overlap", frame_to_markdown(overlap)),
            (
                "Notes",
                "\n".join(
                    [
                        "Use `results333_parameter_feature_importance.csv` for model x loss detail.",
                        f"Figure files: `{rankings_paths['png']}`, `{overlap_paths['png']}`",
                    ]
                ),
            ),
        ],
    )
    return {
        "results333_parameter_feature_importance": importance_path,
        "results333_importance_alignment": alignment_path,
        "results333_importance_overlap": overlap_path,
        "results333_importance_rankings": rankings_paths["png"],
        "results333_importance_overlap_fig": overlap_paths["png"],
        "results333_report": report,
    }


def build_results342_outputs(
    params_long: pd.DataFrame,
    attributes: pd.DataFrame,
    mean_relationships: pd.DataFrame,
    parameter_bounds: dict[str, tuple[float, float]],
    output_dir: Path,
    figures_dir: Path,
    reports_dir: Path,
) -> dict[str, Path]:
    selected = params_long.loc[
        (params_long["parameter"].isin(RESULTS342_PARAMETERS))
        & (params_long["model"].isin(["distributional", "mc_dropout"]))
    ].copy()
    std_corr_tables = _build_correlation_long_value(selected, attributes, value_column="std", methods=("spearman",))
    std_corr = std_corr_tables["spearman"]
    std_corr = std_corr.loc[std_corr["attribute"].isin(RESULTS342_ATTRIBUTES)]

    dist_std = (
        std_corr.loc[std_corr["model"] == "distributional"]
        .groupby(["parameter", "attribute"], as_index=False)
        .agg(
            mean_spearman_corr=("corr", "mean"),
            abs_spearman_corr=("abs_corr", "mean"),
        )
    )
    mean_lookup = mean_relationships[["parameter", "attribute", "mean_spearman_corr"]].rename(columns={"mean_spearman_corr": "mean_relationship_corr"})
    dist_std = dist_std.merge(mean_lookup, on=["parameter", "attribute"], how="left")
    dist_std["sign"] = dist_std["mean_spearman_corr"].map(lambda value: "positive" if value > 0 else "negative")
    dist_std["direction_vs_mean_relationship"] = dist_std.apply(
        lambda row: "same"
        if np.sign(row["mean_spearman_corr"]) == np.sign(row["mean_relationship_corr"])
        else "opposite",
        axis=1,
    )
    dist_std["headline_support_level"] = dist_std["abs_spearman_corr"].map(
        lambda value: "headline" if value >= 0.40 else ("supportive" if value >= 0.25 else "secondary")
    )
    dist_std_path = save_frame(dist_std, output_dir / "results342_distributional_std_relationships.csv")

    averaged = (
        selected.loc[selected["model"] == "distributional"]
        .groupby(["basin_id", "parameter"], as_index=False)
        .agg(std=("std", "mean"), mean=("mean", "mean"))
    )
    averaged = normalize_parameters_to_unit_interval(averaged, parameter_bounds, value_column="mean")
    averaged["std_unit"] = averaged.apply(
        lambda row: float(row["std"] / (parameter_bounds[row["parameter"]][1] - parameter_bounds[row["parameter"]][0])),
        axis=1,
    )
    averaged = averaged.merge(attributes, on="basin_id", how="inner")
    gradient_rows = []
    std_plot_rows = []
    for gradient in GRADIENT_ATTRIBUTES:
        averaged[f"{gradient}_group"] = pd.qcut(averaged[gradient], q=3, labels=["low", "mid", "high"], duplicates="drop")
        for parameter in RESULTS342_PARAMETERS:
            subset = averaged.loc[averaged["parameter"] == parameter].dropna(subset=[f"{gradient}_group"])
            low_values = subset.loc[subset[f"{gradient}_group"] == "low", "std_unit"]
            high_values = subset.loc[subset[f"{gradient}_group"] == "high", "std_unit"]
            p_value = float(mannwhitneyu(high_values, low_values, alternative="two-sided").pvalue) if len(low_values) and len(high_values) else np.nan
            effect = float(high_values.median() - low_values.median()) if len(low_values) and len(high_values) else np.nan
            for group_name, group_df in subset.groupby(f"{gradient}_group", observed=False):
                gradient_rows.append(
                    {
                        "gradient_attribute": gradient,
                        "gradient_group": str(group_name),
                        "parameter": parameter,
                        "sample_count": int(group_df.shape[0]),
                        "median_parameter_std": float(group_df["std"].median()),
                        "median_parameter_std_unit": float(group_df["std_unit"].median()),
                        "high_minus_low": effect,
                        "high_vs_low_mannwhitney_p": p_value,
                    }
                )
            std_plot_rows.append(
                subset[["basin_id", "parameter", "std", "std_unit", gradient, f"{gradient}_group"]].rename(
                    columns={f"{gradient}_group": "gradient_group"}
                ).assign(gradient_attribute=gradient)
            )
    gradient_std = pd.DataFrame(gradient_rows).sort_values(["gradient_attribute", "parameter", "gradient_group"]).reset_index(drop=True)
    gradient_std_path = save_frame(gradient_std, output_dir / "results342_gradient_std_group_stats.csv")
    std_plot_data = pd.concat(std_plot_rows, ignore_index=True)

    compare = (
        std_corr.groupby(["model", "parameter", "attribute"], as_index=False)["corr"]
        .mean()
        .pivot(index=["parameter", "attribute"], columns="model", values="corr")
        .reset_index()
    )
    compare["distributional_minus_mc_dropout"] = compare["distributional"] - compare["mc_dropout"]
    compare_path = save_frame(compare, output_dir / "results342_distributional_vs_dropout_std_compare.csv")

    apply_wrr_style()
    heatmap_df = dist_std.pivot(index="attribute", columns="parameter", values="mean_spearman_corr").reindex(index=RESULTS342_ATTRIBUTES, columns=RESULTS342_PARAMETERS)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="coolwarm", center=0.0, cbar_kws={"label": "Std-attribute Spearman corr"}, ax=ax)
    ax.set_xticklabels([pretty_parameter_name(param) for param in heatmap_df.columns], rotation=20, ha="right")
    ax.set_yticklabels(heatmap_df.index, rotation=0)
    std_heatmap_paths = save_figure(fig, "results342_distributional_std_heatmap", figures_dir, formats=("png",))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
    palette = {"low": "#4C78A8", "mid": "#BEBEBE", "high": "#F58518"}
    for ax, gradient in zip(axes, GRADIENT_ATTRIBUTES):
        subset = std_plot_data.loc[std_plot_data["gradient_attribute"] == gradient]
        sns.boxplot(
            data=subset,
            x="parameter",
            y="std_unit",
            hue="gradient_group",
            order=RESULTS342_PARAMETERS,
            hue_order=["low", "mid", "high"],
            palette=palette,
            showfliers=False,
            ax=ax,
        )
        ax.set_title(gradient)
        ax.set_xlabel("")
        ax.set_ylabel("normalized parameter std" if gradient == GRADIENT_ATTRIBUTES[0] else "")
        ax.set_xticks(range(len(RESULTS342_PARAMETERS)))
        ax.set_xticklabels([pretty_parameter_name(param) for param in RESULTS342_PARAMETERS], rotation=20, ha="right")
        if ax is not axes[0]:
            ax.get_legend().remove()
    axes[0].legend(frameon=False, title="Gradient group")
    std_boxplot_paths = save_figure(fig, "results342_std_gradient_boxplots", figures_dir, formats=("png",))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    mean_heat = mean_relationships.pivot(index="attribute", columns="parameter", values="mean_spearman_corr").reindex(index=RESULTS342_ATTRIBUTES, columns=RESULTS342_PARAMETERS)
    sns.heatmap(mean_heat, cmap="coolwarm", center=0.0, ax=axes[0], cbar=False)
    axes[0].set_title("Parameter mean gradients")
    sns.heatmap(heatmap_df, cmap="coolwarm", center=0.0, ax=axes[1], cbar_kws={"label": "corr"})
    axes[1].set_title("Parameter std gradients")
    appendix_paths = save_figure(fig, "appendix_results342_std_vs_mean_compare", figures_dir, formats=("png",))

    report = write_markdown(
        reports_dir / "results342_report.md",
        title="Results 3.4.2 Parameter Uncertainty Gradients",
        sections=[
            ("Distributional Std Relationships", frame_to_markdown(dist_std)),
            ("Gradient Group Statistics", frame_to_markdown(gradient_std)),
            ("Distributional vs MC-Dropout", frame_to_markdown(compare)),
            (
                "Figure Files",
                "\n".join(
                    [
                        f"`{std_heatmap_paths['png']}`",
                        f"`{std_boxplot_paths['png']}`",
                        f"`{appendix_paths['png']}`",
                    ]
                ),
            ),
        ],
    )
    return {
        "results342_distributional_std_relationships": dist_std_path,
        "results342_gradient_std_group_stats": gradient_std_path,
        "results342_distributional_vs_dropout_std_compare": compare_path,
        "results342_std_heatmap": std_heatmap_paths["png"],
        "results342_std_boxplots": std_boxplot_paths["png"],
        "results342_std_vs_mean_compare": appendix_paths["png"],
        "results342_report": report,
        "std_plot_data": save_frame(std_plot_data, output_dir / "results342_std_plot_data.csv"),
    }


def build_results343_outputs(
    params_long: pd.DataFrame,
    attributes: pd.DataFrame,
    parameter_bounds: dict[str, tuple[float, float]],
    output_dir: Path,
    figures_dir: Path,
    reports_dir: Path,
) -> dict[str, Path]:
    distributional = (
        params_long.loc[(params_long["model"] == "distributional") & (params_long["parameter"].isin(RESULTS343_PARAMETERS))]
        .groupby(["basin_id", "parameter"], as_index=False)
        .agg(mean=("mean", "mean"), std=("std", "mean"))
    )
    distributional = normalize_parameters_to_unit_interval(distributional, parameter_bounds, value_column="mean")
    distributional["std_unit"] = distributional.apply(
        lambda row: float(row["std"] / (parameter_bounds[row["parameter"]][1] - parameter_bounds[row["parameter"]][0])),
        axis=1,
    )
    distributional = distributional.merge(attributes, on="basin_id", how="inner")

    group_frames = []
    representative_rows = []
    representative_plot_rows = []
    for gradient in RESULTS343_GROUPS:
        low_threshold = distributional[gradient].quantile(1.0 / 3.0)
        high_threshold = distributional[gradient].quantile(2.0 / 3.0)
        for label, selector in [
            (f"{gradient}_low", distributional[gradient] <= low_threshold),
            (f"{gradient}_high", distributional[gradient] >= high_threshold),
        ]:
            group_df = distributional.loc[selector].copy()
            for parameter in RESULTS343_PARAMETERS:
                subset = group_df.loc[group_df["parameter"] == parameter]
                full_subset = distributional.loc[distributional["parameter"] == parameter]
                group_frames.append(
                    {
                        "group_label": label,
                        "gradient_attribute": gradient,
                        "parameter": parameter,
                        "sample_count": int(subset.shape[0]),
                        "mean_median_unit": float(subset["mean_unit"].median()),
                        "mean_iqr_unit": float(subset["mean_unit"].quantile(0.75) - subset["mean_unit"].quantile(0.25)),
                        "std_median_unit": float(subset["std_unit"].median()),
                        "std_iqr_unit": float(subset["std_unit"].quantile(0.75) - subset["std_unit"].quantile(0.25)),
                        "mean_median_vs_global": float(subset["mean_unit"].median() - full_subset["mean_unit"].median()),
                        "std_median_vs_global": float(subset["std_unit"].median() - full_subset["std_unit"].median()),
                    }
                )
            basin_scores = (
                group_df[["basin_id", gradient]]
                .drop_duplicates()
                .assign(extremeness=lambda frame: np.abs(frame[gradient] - (low_threshold if "low" in label else high_threshold)))
                .sort_values(["extremeness", "basin_id"], ascending=[False, True])
            )
            selected_basins = basin_scores.head(2)["basin_id"].tolist()
            for basin_id in selected_basins:
                basin_subset = group_df.loc[group_df["basin_id"] == basin_id]
                row = {
                    "basin_id": int(basin_id),
                    "group_label": label,
                    "gradient_attribute": gradient,
                    "aridity": float(basin_subset["aridity"].iloc[0]),
                    "frac_snow": float(basin_subset["frac_snow"].iloc[0]),
                    "slope_mean": float(basin_subset["slope_mean"].iloc[0]),
                }
                for parameter in RESULTS343_PARAMETERS:
                    psubset = basin_subset.loc[basin_subset["parameter"] == parameter]
                    row[f"{parameter}_mean"] = float(psubset["mean"].iloc[0])
                    row[f"{parameter}_mean_unit"] = float(psubset["mean_unit"].iloc[0])
                    row[f"{parameter}_std"] = float(psubset["std"].iloc[0])
                    row[f"{parameter}_std_unit"] = float(psubset["std_unit"].iloc[0])
                representative_rows.append(row)
            first_basin = basin_scores.head(1)["basin_id"].tolist()
            if first_basin:
                representative_plot_rows.append(group_df.loc[group_df["basin_id"] == first_basin[0]].assign(group_label=label))

    basin_group_summary = pd.DataFrame(group_frames).sort_values(["gradient_attribute", "group_label", "parameter"]).reset_index(drop=True)
    representative_basins = pd.DataFrame(representative_rows).drop_duplicates(subset=["basin_id", "group_label"]).reset_index(drop=True)
    summary_path = save_frame(basin_group_summary, output_dir / "results343_basin_group_summary.csv")
    basins_path = save_frame(representative_basins, output_dir / "results343_representative_basins.csv")

    apply_wrr_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharex=True)
    profile = basin_group_summary.pivot(index="group_label", columns="parameter", values="mean_median_unit").reindex(
        [f"{grad}_low" for grad in RESULTS343_GROUPS] + [f"{grad}_high" for grad in RESULTS343_GROUPS]
    )
    std_profile = basin_group_summary.pivot(index="group_label", columns="parameter", values="std_median_unit").reindex(profile.index)
    for group_label in profile.index:
        axes[0].plot(RESULTS343_PARAMETERS, profile.loc[group_label, RESULTS343_PARAMETERS], marker="o", linewidth=1.6, label=group_label)
        axes[1].plot(RESULTS343_PARAMETERS, std_profile.loc[group_label, RESULTS343_PARAMETERS], marker="o", linewidth=1.6, label=group_label)
    axes[0].set_title("Parameter mean profiles")
    axes[1].set_title("Parameter std profiles")
    for ax in axes:
        ax.set_xticks(range(len(RESULTS343_PARAMETERS)))
        ax.set_xticklabels([pretty_parameter_name(param) for param in RESULTS343_PARAMETERS], rotation=20, ha="right")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    group_profile_paths = save_figure(fig, "results343_group_parameter_profiles", figures_dir, formats=("png",))

    plot_df = pd.concat(representative_plot_rows, ignore_index=True)
    groups_in_fig = plot_df["group_label"].drop_duplicates().tolist()
    fig, axes = plt.subplots(2, len(groups_in_fig), figsize=(2.8 * len(groups_in_fig), 7.0), sharey="row")
    if len(groups_in_fig) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for col_idx, group_label in enumerate(groups_in_fig):
        subset = plot_df.loc[plot_df["group_label"] == group_label]
        basin_id = int(subset["basin_id"].iloc[0])
        axes[0, col_idx].bar(np.arange(len(RESULTS343_PARAMETERS)), subset.set_index("parameter").loc[RESULTS343_PARAMETERS, "mean_unit"], color="#4C78A8")
        axes[1, col_idx].bar(np.arange(len(RESULTS343_PARAMETERS)), subset.set_index("parameter").loc[RESULTS343_PARAMETERS, "std_unit"], color="#F58518")
        axes[0, col_idx].set_title(f"{group_label}\n{basin_id}")
        axes[1, col_idx].set_xticks(range(len(RESULTS343_PARAMETERS)))
        axes[1, col_idx].set_xticklabels([pretty_parameter_name(param) for param in RESULTS343_PARAMETERS], rotation=20, ha="right")
    axes[0, 0].set_ylabel("mean unit")
    axes[1, 0].set_ylabel("std unit")
    example_paths = save_figure(fig, "results343_representative_basin_examples", figures_dir, formats=("png",))

    report = write_markdown(
        reports_dir / "results343_report.md",
        title="Results 3.4.3 Representative Basin Groups",
        sections=[
            ("Basin Group Summary", frame_to_markdown(basin_group_summary)),
            ("Representative Basins", frame_to_markdown(representative_basins)),
            (
                "Figure Files",
                "\n".join(
                    [
                        f"`{group_profile_paths['png']}`",
                        f"`{example_paths['png']}`",
                    ]
                ),
            ),
        ],
    )
    return {
        "results343_basin_group_summary": summary_path,
        "results343_representative_basins": basins_path,
        "results343_group_parameter_profiles": group_profile_paths["png"],
        "results343_representative_basin_examples": example_paths["png"],
        "results343_report": report,
    }


def build_total_report(
    reports_dir: Path,
    results332_report: Path,
    results333_report: Path,
    results342_report: Path,
    results343_report: Path,
    results332_similarity: pd.DataFrame,
    results333_alignment: pd.DataFrame,
    results342_std_relationships: pd.DataFrame,
    results343_groups: pd.DataFrame,
) -> Path:
    within_model = (
        results332_similarity.loc[(results332_similarity["same_model"]) & (results332_similarity["run_id_a"] != results332_similarity["run_id_b"])]
        .groupby("model_a", as_index=False)
        .agg(mean_frobenius_distance=("frobenius_distance", "mean"), mean_matrix_corr=("matrix_corr_pearson", "mean"))
        .rename(columns={"model_a": "model"})
    )
    best_alignment = results333_alignment.sort_values(["dominant_attribute_top3_rate", "dominant_attribute_top1_rate"], ascending=False)
    strongest_std = results342_std_relationships.sort_values("abs_spearman_corr", ascending=False).head(12)
    biggest_group = results343_groups.sort_values(["std_median_vs_global", "mean_median_vs_global"], ascending=False).head(12)
    return write_markdown(
        reports_dir / "results332_333_342_343_report.md",
        title="Results 3.3.2, 3.3.3, 3.4.2, and 3.4.3",
        sections=[
            (
                "Objective",
                "\n".join(
                    [
                        "These analyses extend the completed Results 3.3.1 and 3.4.1 outputs.",
                        "They focus on matrix-level structure, post-hoc feature usage, uncertainty gradients, and representative basin groups.",
                    ]
                ),
            ),
            ("Results 3.3.2 summary", frame_to_markdown(within_model)),
            ("Results 3.3.3 summary", frame_to_markdown(best_alignment)),
            ("Results 3.4.2 summary", frame_to_markdown(strongest_std)),
            ("Results 3.4.3 summary", frame_to_markdown(biggest_group)),
            (
                "Main-paper-ready findings",
                "\n".join(
                    [
                        "1. Use the matrix-level compactness and embedding to judge whether distributional reinforces shared structure without deviating from it.",
                        "2. Use dominant-attribute top-1/top-3 alignment and top-k overlap as explainability support.",
                        "3. Use the strongest distributional std gradients and the low/high basin-group profiles as the main uncertainty results.",
                    ]
                ),
            ),
            (
                "Supporting / appendix findings",
                "\n".join(
                    [
                        f"`{results332_report}`",
                        f"`{results333_report}`",
                        f"`{results342_report}`",
                        f"`{results343_report}`",
                    ]
                ),
            ),
            (
                "Suggested wording for Results paragraphs",
                "\n".join(
                    [
                        "Results 3.3.2: Matrix-level similarity should be described using centroid similarity and within-model compactness rather than a binary same/different claim.",
                        "Results 3.3.3: Explainability should be framed as supporting evidence that can either align with or qualify the dominant-control story.",
                        "Results 3.4.2: Uncertainty gradients should be described as large-scale structure in parameter identifiability rather than generic uncertainty magnitude.",
                        "Results 3.4.3: Basin-group and representative-case figures should be used to make the statistical contrasts visually concrete.",
                    ]
                ),
            ),
            (
                "Suggested wording for Discussion transition",
                "\n".join(
                    [
                        "The remaining disagreement is concentrated in a smaller subset of parameters and appears mainly as weaker matrix-level dispersion, partial explainability mismatch, or uncertainty-gradient heterogeneity.",
                        "These are better interpreted as discussion-level evidence about identifiability and process compensation than as failures of the dominant shared structure.",
                    ]
                ),
            ),
            (
                "Answers to the Required Questions",
                "\n".join(
                    [
                        "1. Use the matrix-level similarity outputs to assess whether distributional is better seen as stable reinforcement of shared structure.",
                        "2. Use the explainability alignment and overlap tables to assess whether post-hoc feature usage supports the dominant-control results.",
                        "3. Use the std-relationship and std-gradient outputs to assess whether parameter uncertainty follows clear basin gradients.",
                        "4. Use the representative basin group summary and example figure to assess whether the statistical results translate into intuitive hydrologic contrasts.",
                    ]
                ),
            ),
        ],
    )


def run_results332_333_342_343(
    params_long: pd.DataFrame,
    attributes: pd.DataFrame,
    parameter_bounds: dict[str, tuple[float, float]],
    analysis_root: Path,
    dominant_summary: pd.DataFrame,
    mean_relationships: pd.DataFrame,
) -> dict[str, Path]:
    output_dir = analysis_root / "correlation_summaries"
    figures_dir = analysis_root / "figures"
    reports_dir = analysis_root / "reports"

    results332_paths = build_results332_outputs(params_long, attributes, output_dir, figures_dir, reports_dir)
    results333_paths = build_results333_outputs(params_long, attributes, dominant_summary, output_dir, figures_dir, reports_dir)
    results342_paths = build_results342_outputs(params_long, attributes, mean_relationships, parameter_bounds, output_dir, figures_dir, reports_dir)
    results343_paths = build_results343_outputs(params_long, attributes, parameter_bounds, output_dir, figures_dir, reports_dir)

    total_report = build_total_report(
        reports_dir=reports_dir,
        results332_report=results332_paths["results332_report"],
        results333_report=results333_paths["results333_report"],
        results342_report=results342_paths["results342_report"],
        results343_report=results343_paths["results343_report"],
        results332_similarity=pd.read_csv(results332_paths["results332_matrix_similarity"]),
        results333_alignment=pd.read_csv(results333_paths["results333_importance_alignment"]),
        results342_std_relationships=pd.read_csv(results342_paths["results342_distributional_std_relationships"]),
        results343_groups=pd.read_csv(results343_paths["results343_basin_group_summary"]),
    )

    return {
        **results332_paths,
        **results333_paths,
        **results342_paths,
        **results343_paths,
        "results332_333_342_343_report": total_report,
    }


def main() -> None:
    parser = build_parser("Run Results 3.3.2/3.3.3/3.4.2/3.4.3 analyses.")
    parser.add_argument(
        "--results331-dominant-summary-csv",
        default="project/parameterize/outputs/analysis/stability_stats/correlation_summaries/results331_dominant_attribute_summary.csv",
    )
    parser.add_argument(
        "--results341-mean-relationships-csv",
        default="project/parameterize/outputs/analysis/stability_stats/correlation_summaries/results341_distributional_mean_relationships.csv",
    )
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

    paths = run_results332_333_342_343(
        params_long=data["params_long"],
        attributes=data["attributes"],
        parameter_bounds=data["parameter_bounds"],
        analysis_root=data["stability_analysis_root"],
        dominant_summary=pd.read_csv(args.results331_dominant_summary_csv),
        mean_relationships=pd.read_csv(args.results341_mean_relationships_csv),
    )
    print(paths["results332_333_342_343_report"])


if __name__ == "__main__":
    main()
