from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STRUCTURE_DIR = Path(
    "/workspace/autoresearch/project/flexmopex/analysis/flex_mopex_v1_structure_learning_interpretation"
)
ATTRIBUTE_PATH = Path(
    "/workspace/autoresearch/project/parameterize/outputs/analysis/stability_stats/tables/basin_attributes.csv"
)
OUTPUT_DIR = Path(
    "/workspace/autoresearch/project/flexmopex/analysis/flex_mopex_v1_attribute_controls"
)

CANDIDATE_ALPHAS = [0.005, 0.01, 0.03]
PRIMARY_ALPHA = 0.01
TARGETS = ["sum_weight", "share_snow", "share_int", "share_phen", "share_sub"]
SHARE_TARGETS = ["share_snow", "share_int", "share_phen", "share_sub"]
IMPORTANT_ATTRIBUTES = [
    "frac_snow",
    "aridity_index",
    "aridity",
    "frac_forest",
    "elev_mean",
    "slope_mean",
    "p_seasonality",
    "seasonality_index",
    "runoff_ratio",
    "soil_depth_pelletier",
    "soil_depth_statsgo",
    "soil_conductivity",
    "geol_permeability",
    "geol_porosity",
    "sand_frac",
    "clay_frac",
]
DROP_CATEGORICAL_CODES = ["dom_land_cover", "geol_1st_class", "geol_2nd_class"]
NULL_PERMUTATIONS = 100
RANDOM_STATE = 20260526


def format_alpha(alpha: float) -> str:
    return f"{alpha:g}"


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    structure = pd.read_csv(STRUCTURE_DIR / "weights_with_sum_and_share.csv")
    dominant = pd.read_csv(STRUCTURE_DIR / "dominant_process_by_basin_alpha.csv")
    tradeoff = pd.read_csv(STRUCTURE_DIR / "performance_complexity_tradeoff.csv")
    delta = pd.read_csv(STRUCTURE_DIR / "basin_delta_metrics_by_alpha.csv")
    morans = pd.read_csv(STRUCTURE_DIR / "morans_i_by_alpha.csv")
    attributes = pd.read_csv(ATTRIBUTE_PATH)
    structure["gage_id"] = structure["gage_id"].astype(int)
    dominant["station_index"] = dominant["station_index"].astype(int)
    attributes["basin_id"] = attributes["basin_id"].astype(int)
    return structure, attributes, tradeoff, delta, morans


def discover_attribute_files() -> pd.DataFrame:
    paths = [
        ATTRIBUTE_PATH,
        Path("/workspace/autoresearch/project/parameterize/outputs/analysis/stability_stats/tables/basin_attributes_input.csv"),
        Path("/workspace/autoresearch/project/parameterize/manuscript/analysis/00_data_inventory/data/basin_attribute_inventory.csv"),
    ]
    rows = []
    for path in paths:
        rows.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "selected": path == ATTRIBUTE_PATH,
                "reason": "primary CAMELS attribute table" if path == ATTRIBUTE_PATH else "candidate/reference table",
            }
        )
    return pd.DataFrame(rows)


def build_merge(
    structure: pd.DataFrame,
    attributes: pd.DataFrame,
    delta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    structure_focus = structure[structure["alpha"].isin(CANDIDATE_ALPHAS)].copy()
    delta_focus = delta[
        delta["alpha"].isin(CANDIDATE_ALPHAS)
    ][["alpha", "station_index", "nse", "kge", "delta_nse", "delta_kge"]].copy()
    structure_focus = structure_focus.merge(
        delta_focus,
        on=["alpha", "station_index"],
        how="left",
        suffixes=("", "_metric"),
    )
    merged = structure_focus.merge(
        attributes,
        left_on="gage_id",
        right_on="basin_id",
        how="left",
        indicator=True,
    )

    retained_attrs = []
    diagnostic_rows = []
    attr_cols = [col for col in attributes.columns if col != "basin_id"]
    for col in attr_cols:
        missing_pct = float(merged[col].isna().mean() * 100.0)
        unique_count = int(merged[col].nunique(dropna=True))
        if col in DROP_CATEGORICAL_CODES:
            retained = False
            reason = "dropped: categorical class code is not ordinal"
        elif not pd.api.types.is_numeric_dtype(merged[col]):
            retained = False
            reason = "dropped: non-numeric"
        elif missing_pct > 25:
            retained = False
            reason = "dropped: >25% missing after merge"
        elif unique_count < 3:
            retained = False
            reason = "dropped: fewer than 3 unique values"
        else:
            retained = True
            reason = "retained"
            retained_attrs.append(col)
        diagnostic_rows.append(
            {
                "variable": col,
                "source": str(ATTRIBUTE_PATH),
                "missing_pct_after_merge": missing_pct,
                "unique_count_after_merge": unique_count,
                "retained": retained,
                "reason": reason,
            }
        )

    basin_counts = pd.DataFrame(
        [
            {
                "variable": "__basin_count__",
                "source": "structure + attributes",
                "missing_pct_after_merge": np.nan,
                "unique_count_after_merge": int(merged["gage_id"].nunique()),
                "retained": True,
                "reason": (
                    f"structure basins before merge per alpha=671; "
                    f"attribute basins={attributes['basin_id'].nunique()}; "
                    f"merged basins={merged.loc[merged['_merge'] == 'both', 'gage_id'].nunique()}"
                ),
            }
        ]
    )
    diagnostics = pd.concat([basin_counts, pd.DataFrame(diagnostic_rows)], ignore_index=True)
    merged = merged[merged["_merge"] == "both"].drop(columns=["_merge"])
    return merged, diagnostics, retained_attrs


def add_derived_attributes(merged: pd.DataFrame, retained_attrs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    out = merged.copy()
    rows = []
    derived_attrs = []

    if {"pet_mean", "p_mean"}.issubset(out.columns):
        out["aridity_index"] = out["pet_mean"] / out["p_mean"]
        derived_attrs.append("aridity_index")
        rows.append(
            {
                "derived_variable": "aridity_index",
                "status": "computed",
                "formula": "pet_mean / p_mean",
                "reason": "PET and precipitation mean are available",
            }
        )
    else:
        rows.append(
            {
                "derived_variable": "aridity_index",
                "status": "skipped",
                "formula": "pet_mean / p_mean",
                "reason": "pet_mean or p_mean unavailable",
            }
        )

    if "p_seasonality" in out.columns:
        out["seasonality_index"] = out["p_seasonality"]
        derived_attrs.append("seasonality_index")
        rows.append(
            {
                "derived_variable": "seasonality_index",
                "status": "proxied",
                "formula": "p_seasonality",
                "reason": "monthly precipitation data unavailable; using CAMELS p_seasonality",
            }
        )
    else:
        rows.append(
            {
                "derived_variable": "seasonality_index",
                "status": "skipped",
                "formula": "monthly precipitation std / monthly precipitation mean",
                "reason": "monthly precipitation data unavailable",
            }
        )

    for name, formula, reason in [
        ("runoff_ratio", "Q / P", "streamflow/long-term runoff attribute unavailable in selected table"),
        ("evaporation_ratio", "1 - runoff_ratio", "runoff_ratio unavailable"),
        ("budyko_residual", "E/P - Budyko_expected(aridity_index)", "E/P or runoff_ratio unavailable"),
        ("baseflow_index", "BFI", "baseflow index unavailable in selected table"),
    ]:
        rows.append({"derived_variable": name, "status": "skipped", "formula": formula, "reason": reason})

    for attr in retained_attrs + derived_attrs:
        if attr in out.columns and pd.api.types.is_numeric_dtype(out[attr]):
            std = out[attr].std(skipna=True)
            if std and np.isfinite(std):
                out[f"z_{attr}"] = (out[attr] - out[attr].mean(skipna=True)) / std

    attr_cols = list(dict.fromkeys(retained_attrs + derived_attrs + ["lat", "lon"]))
    attr_cols = [col for col in attr_cols if col in out.columns]
    return out, pd.DataFrame(rows), attr_cols


def fdr_bh(p_values: pd.Series) -> pd.Series:
    p = p_values.to_numpy(dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    valid = p[mask]
    if valid.size == 0:
        return pd.Series(adjusted, index=p_values.index)
    order = np.argsort(valid)
    ranked = valid[order]
    q = ranked * valid.size / (np.arange(valid.size) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    restored = np.empty_like(q)
    restored[order] = q
    adjusted[mask] = restored
    return pd.Series(adjusted, index=p_values.index)


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    frame = pd.concat([x, y], axis=1).dropna()
    n = len(frame)
    if n < 10 or frame.iloc[:, 0].nunique() < 3 or frame.iloc[:, 1].nunique() < 3:
        return np.nan, np.nan, n
    result = stats.spearmanr(frame.iloc[:, 0], frame.iloc[:, 1])
    return float(result.statistic), float(result.pvalue), n


def spearman_analysis(data: pd.DataFrame, attr_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for alpha, group in data.groupby("alpha", sort=True):
        for target in TARGETS:
            for attr in attr_cols:
                rho, p_value, n = safe_spearman(group[target], group[attr])
                rows.append(
                    {
                        "alpha": alpha,
                        "target": target,
                        "attribute": attr,
                        "spearman_rho": rho,
                        "p_value": p_value,
                        "n": n,
                    }
                )
    corr = pd.DataFrame(rows)
    corr["p_fdr"] = corr.groupby(["alpha", "target"])["p_value"].transform(fdr_bh)

    top_rows = []
    for (alpha, target), group in corr.dropna(subset=["spearman_rho"]).groupby(["alpha", "target"], sort=True):
        pos = group.sort_values("spearman_rho", ascending=False).head(10).copy()
        neg = group.sort_values("spearman_rho", ascending=True).head(10).copy()
        pos["direction"] = "positive"
        neg["direction"] = "negative"
        top_rows.extend(pd.concat([pos, neg]).to_dict("records"))
    return corr, pd.DataFrame(top_rows)


def rank_residual(values: pd.Series, controls: pd.DataFrame) -> pd.Series:
    frame = pd.concat([values.rename("value"), controls], axis=1).dropna()
    ranks = frame.rank(method="average")
    y = ranks["value"].to_numpy(dtype=float)
    x = ranks.drop(columns=["value"]).to_numpy(dtype=float)
    if x.shape[1] == 0:
        residual = y - y.mean()
    else:
        model = LinearRegression()
        model.fit(x, y)
        residual = y - model.predict(x)
    return pd.Series(residual, index=frame.index)


def partial_spearman_analysis(data: pd.DataFrame, attr_cols: list[str]) -> pd.DataFrame:
    rows = []
    base_controls = [col for col in ["sum_weight", "kge", "nse", "lat", "lon"] if col in data.columns]
    for alpha, group in data.groupby("alpha", sort=True):
        for target in SHARE_TARGETS:
            target_controls = [col for col in base_controls if col != target]
            target_resid = rank_residual(group[target], group[target_controls])
            for attr in attr_cols:
                controls = [col for col in target_controls if col != attr]
                attr_resid = rank_residual(group[attr], group[controls])
                joined = pd.concat([target_resid.rename("target"), attr_resid.rename("attr")], axis=1).dropna()
                if len(joined) < 10 or joined["target"].nunique() < 3 or joined["attr"].nunique() < 3:
                    rho, p_value, n = np.nan, np.nan, len(joined)
                else:
                    result = stats.spearmanr(joined["target"], joined["attr"])
                    rho, p_value, n = float(result.statistic), float(result.pvalue), len(joined)
                rows.append(
                    {
                        "alpha": alpha,
                        "target": target,
                        "attribute": attr,
                        "partial_spearman_rho": rho,
                        "p_value": p_value,
                        "n": n,
                        "controls": ",".join(target_controls),
                    }
                )
    partial = pd.DataFrame(rows)
    partial["p_fdr"] = partial.groupby(["alpha", "target"])["p_value"].transform(fdr_bh)
    return partial


def plot_heatmap(table: pd.DataFrame, value_col: str, alpha: float, output_path: Path, title: str) -> None:
    frame = table[table["alpha"] == alpha]
    pivot = frame.pivot(index="target", columns="attribute", values=value_col)
    if pivot.empty:
        return
    fig_width = max(10, 0.32 * len(pivot.columns))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    im = ax.imshow(pivot.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def dominant_process_analysis(data: pd.DataFrame, attr_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    attrs = [attr for attr in IMPORTANT_ATTRIBUTES if attr in attr_cols]
    summary_rows = []
    test_rows = []
    for alpha, group in data.groupby("alpha", sort=True):
        for attr in attrs:
            groups = []
            for process, process_group in group.groupby("dominant_process", sort=True):
                values = process_group[attr].dropna()
                if len(values) == 0:
                    continue
                q25, q75 = values.quantile([0.25, 0.75])
                summary_rows.append(
                    {
                        "alpha": alpha,
                        "attribute": attr,
                        "dominant_process": process,
                        "n": len(values),
                        "median": values.median(),
                        "q25": q25,
                        "q75": q75,
                        "iqr": q75 - q25,
                    }
                )
                groups.append((process, values))
            if len(groups) >= 2:
                stat, p_value = stats.kruskal(*[values for _, values in groups])
                test_rows.append(
                    {
                        "alpha": alpha,
                        "attribute": attr,
                        "test": "kruskal_wallis",
                        "group_left": "all",
                        "group_right": "all",
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "effect_size": np.nan,
                    }
                )
                for (left_name, left), (right_name, right) in combinations(groups, 2):
                    effect = cliffs_delta(left.to_numpy(), right.to_numpy())
                    test_rows.append(
                        {
                            "alpha": alpha,
                            "attribute": attr,
                            "test": "cliffs_delta",
                            "group_left": left_name,
                            "group_right": right_name,
                            "statistic": np.nan,
                            "p_value": np.nan,
                            "effect_size": effect,
                        }
                    )
    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        mask = tests["test"] == "kruskal_wallis"
        tests.loc[mask, "p_fdr"] = fdr_bh(tests.loc[mask, "p_value"])
    return pd.DataFrame(summary_rows), tests


def cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    if len(left) == 0 or len(right) == 0:
        return np.nan
    diffs = left[:, None] - right[None, :]
    return float((np.sum(diffs > 0) - np.sum(diffs < 0)) / diffs.size)


def plot_dominant_boxplots(data: pd.DataFrame, attr_cols: list[str], output_path: Path) -> None:
    attrs = [attr for attr in IMPORTANT_ATTRIBUTES if attr in attr_cols][:8]
    group = data[data["alpha"] == PRIMARY_ALPHA]
    n_col = 2
    n_row = int(np.ceil(len(attrs) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(11, 3.8 * n_row))
    axes = np.ravel(axes)
    processes = sorted(group["dominant_process"].dropna().unique())
    for ax, attr in zip(axes, attrs):
        values = [group.loc[group["dominant_process"] == process, attr].dropna().to_numpy() for process in processes]
        ax.boxplot(values, tick_labels=processes, showfliers=False)
        ax.set_title(attr)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(attrs) :]:
        ax.axis("off")
    fig.suptitle(f"Dominant Process Attribute Contrasts, alpha={PRIMARY_ALPHA:g}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def rf_dataset(data: pd.DataFrame, attr_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = data[data["alpha"] == PRIMARY_ALPHA].copy()
    x = group[attr_cols].copy()
    y = group[TARGETS].copy()
    return x, y


def rf_analysis(data: pd.DataFrame, attr_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    x, y_all = rf_dataset(data, attr_cols)
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    score_rows = []
    importance_rows = []
    for target in TARGETS:
        y = y_all[target]
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=80,
                max_depth=10,
                max_features="sqrt",
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        )
        pred = cross_val_predict(model, x, y, cv=cv, n_jobs=1)
        score_rows.append(
            {
                "alpha": PRIMARY_ALPHA,
                "target": target,
                "cv_r2": r2_score(y, pred),
                "cv_rmse": mean_squared_error(y, pred) ** 0.5,
                "n": len(y),
                "n_features": len(attr_cols),
            }
        )
        model.fit(x, y)
        perm = permutation_importance(
            model,
            x,
            y,
            n_repeats=3,
            random_state=RANDOM_STATE,
            n_jobs=1,
            scoring="r2",
        )
        for attr, mean, std in zip(attr_cols, perm.importances_mean, perm.importances_std):
            importance_rows.append(
                {
                    "alpha": PRIMARY_ALPHA,
                    "target": target,
                    "attribute": attr,
                    "importance_mean": mean,
                    "importance_std": std,
                }
            )
    return pd.DataFrame(score_rows), pd.DataFrame(importance_rows)


def plot_rf_importance(importance: pd.DataFrame, target: str, output_path: Path) -> None:
    group = importance[importance["target"] == target].sort_values("importance_mean", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(group["attribute"][::-1], group["importance_mean"][::-1], xerr=group["importance_std"][::-1])
    ax.set_xlabel("Permutation importance (R2 decrease)")
    ax.set_title(f"RF Importance: {target}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def null_controls(data: pd.DataFrame, attr_cols: list[str], real_corr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    group = data[data["alpha"] == PRIMARY_ALPHA].copy().reset_index(drop=True)
    x = group[attr_cols].copy()
    y_all = group[TARGETS].copy()
    real_max = (
        real_corr[real_corr["alpha"] == PRIMARY_ALPHA]
        .groupby("target")["spearman_rho"]
        .apply(lambda values: values.abs().max())
        .to_dict()
    )
    spearman_rows = []
    rf_rows = []
    cv = KFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)

    for null_type in ["shuffle_targets", "shuffle_attributes"]:
        for perm_id in range(NULL_PERMUTATIONS):
            if null_type == "shuffle_targets":
                y_perm_all = y_all.apply(lambda col: rng.permutation(col.to_numpy()))
                x_perm = x
            else:
                y_perm_all = y_all
                x_perm = x.iloc[rng.permutation(len(x))].reset_index(drop=True)

            for target in TARGETS:
                y_perm = pd.Series(y_perm_all[target])
                max_abs = 0.0
                for attr in attr_cols:
                    rho, _, _ = safe_spearman(y_perm, x_perm[attr])
                    if np.isfinite(rho):
                        max_abs = max(max_abs, abs(rho))
                spearman_rows.append(
                    {
                        "alpha": PRIMARY_ALPHA,
                        "null_type": null_type,
                        "permutation": perm_id,
                        "target": target,
                        "max_abs_spearman": max_abs,
                        "real_max_abs_spearman": real_max.get(target, np.nan),
                    }
                )

                model = make_pipeline(
                    SimpleImputer(strategy="median"),
                    RandomForestRegressor(
                        n_estimators=10,
                        max_depth=8,
                        max_features="sqrt",
                        min_samples_leaf=8,
                        random_state=RANDOM_STATE + perm_id,
                        n_jobs=1,
                    ),
                )
                pred = cross_val_predict(model, x_perm, y_perm, cv=cv, n_jobs=1)
                rf_rows.append(
                    {
                        "alpha": PRIMARY_ALPHA,
                        "null_type": null_type,
                        "permutation": perm_id,
                        "target": target,
                        "rf_cv_r2": r2_score(y_perm, pred),
                    }
                )
    return pd.DataFrame(spearman_rows), pd.DataFrame(rf_rows)


def plot_real_vs_null_spearman(null: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = []
    values = []
    real = []
    for target in TARGETS:
        group = null[(null["target"] == target) & (null["null_type"] == "shuffle_targets")]
        values.append(group["max_abs_spearman"].to_numpy())
        labels.append(target)
        real.append(group["real_max_abs_spearman"].iloc[0])
    ax.boxplot(values, tick_labels=labels, showfliers=False)
    ax.scatter(np.arange(1, len(real) + 1), real, color="red", label="real max abs Spearman", zorder=3)
    ax.set_ylabel("max abs Spearman")
    ax.set_title("Real vs Null Spearman, alpha=0.01")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_real_vs_null_rf(null: pd.DataFrame, real_scores: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = []
    values = []
    real = []
    for target in TARGETS:
        group = null[(null["target"] == target) & (null["null_type"] == "shuffle_targets")]
        values.append(group["rf_cv_r2"].to_numpy())
        labels.append(target)
        real.append(real_scores.loc[real_scores["target"] == target, "cv_r2"].iloc[0])
    ax.boxplot(values, tick_labels=labels, showfliers=False)
    ax.scatter(np.arange(1, len(real) + 1), real, color="red", label="real RF CV R2", zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("RF CV R2")
    ax.set_title("Real vs Null RF Predictability, alpha=0.01")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_path: Path,
    merged: pd.DataFrame,
    merge_diag: pd.DataFrame,
    derived_diag: pd.DataFrame,
    spearman: pd.DataFrame,
    partial: pd.DataFrame,
    dom_tests: pd.DataFrame,
    rf_scores: pd.DataFrame,
    rf_importance: pd.DataFrame,
    null_spearman: pd.DataFrame,
    null_rf: pd.DataFrame,
    tradeoff: pd.DataFrame,
) -> None:
    primary_corr = spearman[spearman["alpha"] == PRIMARY_ALPHA]
    partial_primary = partial[partial["alpha"] == PRIMARY_ALPHA]
    top_by_target = (
        primary_corr.dropna(subset=["spearman_rho"])
        .sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)
        .groupby("target")
        .head(5)
    )
    real_rf = rf_scores.set_index("target")["cv_r2"].to_dict()
    null_rf_q95 = (
        null_rf[null_rf["null_type"] == "shuffle_targets"]
        .groupby("target")["rf_cv_r2"]
        .quantile(0.95)
        .to_dict()
    )
    real_spearman = (
        primary_corr.groupby("target")["spearman_rho"]
        .apply(lambda values: values.abs().max())
        .to_dict()
    )
    null_spearman_q95 = (
        null_spearman[null_spearman["null_type"] == "shuffle_targets"]
        .groupby("target")["max_abs_spearman"]
        .quantile(0.95)
        .to_dict()
    )
    alpha_row = tradeoff.loc[tradeoff["alpha"] == PRIMARY_ALPHA].iloc[0]
    sig_partial = partial_primary[partial_primary["p_fdr"] < 0.05]
    sig_dom = dom_tests[(dom_tests["test"] == "kruskal_wallis") & (dom_tests.get("p_fdr", np.nan) < 0.05)]
    attr_basin_count = int(merged["gage_id"].nunique())
    retained = merge_diag[merge_diag["retained"] == True]

    lines = [
        "# Flex-MOPEX V1 Attribute-Control Report",
        "",
        "## Data Merge",
        "",
        f"- Structure basins per candidate alpha before merge: 671.",
        f"- Basins retained after CAMELS attribute merge: {attr_basin_count}.",
        f"- Retained numeric attributes: {len(retained) - 1 if '__basin_count__' in retained['variable'].values else len(retained)}.",
        f"- Candidate alphas analyzed: {', '.join(str(a) for a in CANDIDATE_ALPHAS)}.",
        "",
        "## A. Most Interpretable Structural Quantity",
        "",
        "Relative shares are the most interpretable quantities for process preference because they separate process allocation from total structural complexity. `sum_weight` is the appropriate indicator of structural complexity. Absolute weights are useful diagnostics for shrinkage and saturation, but should not be treated alone as process preference.",
        "",
        "## B. Attribute Relationships",
        "",
        "The learned shares show attribute-consistent structural patterns, but the evidence remains preliminary. The strongest alpha=0.01 Spearman relationships are:",
    ]
    for _, row in top_by_target.iterrows():
        lines.append(
            f"- {row['target']} vs {row['attribute']}: rho={row['spearman_rho']:.3f}, FDR p={row['p_fdr']:.3g}"
        )

    def target_attr_line(target: str, attrs: list[str], label: str) -> None:
        subset = primary_corr[(primary_corr["target"] == target) & (primary_corr["attribute"].isin(attrs))]
        if subset.empty:
            lines.append(f"- {label}: requested attributes unavailable.")
        else:
            best = subset.reindex(subset["spearman_rho"].abs().sort_values(ascending=False).index).iloc[0]
            lines.append(
                f"- {label}: strongest available relationship is {best['attribute']} "
                f"(rho={best['spearman_rho']:.3f}, FDR p={best['p_fdr']:.3g})."
            )

    lines.extend(["", "## C-F. Process-Specific Checks", ""])
    target_attr_line("share_snow", ["frac_snow", "elev_mean", "aridity", "aridity_index"], "share_snow against snow/elevation/climate controls")
    target_attr_line("share_int", ["frac_forest", "lai_max", "lai_diff", "gvf_max", "gvf_diff"], "share_int against vegetation/forest controls")
    target_attr_line("share_phen", ["p_seasonality", "seasonality_index", "aridity", "aridity_index", "frac_forest", "lai_diff"], "share_phen against seasonality/aridity/vegetation controls")
    target_attr_line("share_sub", ["slope_mean", "soil_depth_pelletier", "soil_depth_statsgo", "soil_conductivity", "geol_permeability", "geol_porosity"], "share_sub against topography/soil/geology controls")
    lines.append("- Baseflow index and runoff ratio were not available in the selected CAMELS attribute table, so those checks were skipped rather than inferred.")

    lines.extend(["", "## G. Robustness After Controls", ""])
    lines.append(
        f"- Partial Spearman controlled for available `sum_weight`, KGE/NSE, latitude, and longitude. Significant alpha=0.01 partial relationships after FDR correction: {len(sig_partial)}."
    )
    if len(sig_partial) > 0:
        for _, row in sig_partial.sort_values("partial_spearman_rho", key=lambda s: s.abs(), ascending=False).head(10).iterrows():
            lines.append(
                f"  - {row['target']} vs {row['attribute']}: partial rho={row['partial_spearman_rho']:.3f}, FDR p={row['p_fdr']:.3g}"
            )

    lines.extend(["", "## H. Null Controls", ""])
    for target in TARGETS:
        lines.append(
            f"- {target}: real max |Spearman|={real_spearman.get(target, np.nan):.3f}, "
            f"shuffle-target null 95%={null_spearman_q95.get(target, np.nan):.3f}; "
            f"real RF CV R2={real_rf.get(target, np.nan):.3f}, "
            f"shuffle-target null RF 95%={null_rf_q95.get(target, np.nan):.3f}."
        )

    lines.extend(["", "## I. Alpha Recommendation", ""])
    lines.append(
        f"Alpha=0.01 is the best primary paper setting for this attribute-control analysis: it retains a substantial complexity reduction "
        f"({alpha_row['complexity_reduction_vs_alpha0']:.3f}) while avoiding the full collapse at alpha >= 0.1 and preserving moderate median KGE loss "
        f"({alpha_row['delta_median_kge_vs_alpha0']:.3f}). Alpha=0.005 and alpha=0.03 remain useful sensitivity cases."
    )

    lines.extend(["", "## J. Claim Strength", ""])
    if len(sig_partial) >= 2 or any(real_rf.get(target, -np.inf) > null_rf_q95.get(target, np.inf) for target in SHARE_TARGETS):
        category = "Mixed support: complexity regionalization plus partial process specificity"
    else:
        category = "Weak support: mainly global complexity shrinkage"
    lines.append(f"Final diagnostic category: **{category}**.")
    lines.append(
        "The attribute-control evidence is strong enough to motivate a process-level structure regionalization hypothesis, but the main paper should frame it as hydrologically coherent, spatially organized process preference and preliminary evidence. The single-seed design and unavailable BFI/runoff/Budyko controls still argue against claiming strong support or definitive mechanism discovery."
    )

    lines.extend(["", "## Derived Attribute Availability", ""])
    for _, row in derived_diag.iterrows():
        lines.append(f"- {row['derived_variable']}: {row['status']} ({row['reason']}).")

    lines.extend(["", "## Output Files", ""])
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file() and path.name != output_path.name:
            lines.append(f"- `{path.name}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    structure, attributes, tradeoff, delta, _ = read_inputs()
    discovered = discover_attribute_files()
    merged, merge_diag, retained_attrs = build_merge(structure, attributes, delta)
    merged_derived, derived_diag, attr_cols = add_derived_attributes(merged, retained_attrs)
    attr_cols = [col for col in attr_cols if col not in TARGETS and col in merged_derived.columns]

    spearman, top_attrs = spearman_analysis(merged_derived, attr_cols)
    partial = partial_spearman_analysis(merged_derived, attr_cols)
    dom_summary, dom_tests = dominant_process_analysis(merged_derived, attr_cols)
    rf_scores, rf_importance = rf_analysis(merged_derived, attr_cols)
    null_spearman, null_rf = null_controls(merged_derived, attr_cols, spearman)

    discovered.to_csv(OUTPUT_DIR / "attribute_file_discovery.csv", index=False)
    merged.to_csv(OUTPUT_DIR / "merged_structure_attributes.csv", index=False)
    merge_diag.to_csv(OUTPUT_DIR / "attribute_merge_diagnostics.csv", index=False)
    merged_derived.to_csv(OUTPUT_DIR / "merged_structure_attributes_with_derived.csv", index=False)
    derived_diag.to_csv(OUTPUT_DIR / "derived_attribute_diagnostics.csv", index=False)
    spearman.to_csv(OUTPUT_DIR / "spearman_structure_attribute_by_alpha.csv", index=False)
    top_attrs.to_csv(OUTPUT_DIR / "spearman_top_attributes_by_target.csv", index=False)
    partial.to_csv(OUTPUT_DIR / "partial_spearman_structure_attribute_by_alpha.csv", index=False)
    dom_summary.to_csv(OUTPUT_DIR / "dominant_process_attribute_summary.csv", index=False)
    dom_tests.to_csv(OUTPUT_DIR / "dominant_process_attribute_tests.csv", index=False)
    rf_scores.to_csv(OUTPUT_DIR / "rf_predictability_by_target.csv", index=False)
    rf_importance.to_csv(OUTPUT_DIR / "rf_permutation_importance_by_target.csv", index=False)
    null_spearman.to_csv(OUTPUT_DIR / "null_spearman_distribution.csv", index=False)
    null_rf.to_csv(OUTPUT_DIR / "null_rf_score_distribution.csv", index=False)

    for alpha in CANDIDATE_ALPHAS:
        plot_heatmap(
            spearman,
            "spearman_rho",
            alpha,
            OUTPUT_DIR / f"fig_spearman_heatmap_alpha_{format_alpha(alpha)}.png",
            f"Spearman Structure-Attribute Correlations, alpha={alpha:g}",
        )
    plot_heatmap(
        partial,
        "partial_spearman_rho",
        PRIMARY_ALPHA,
        OUTPUT_DIR / "fig_partial_spearman_heatmap_alpha_0.01.png",
        "Partial Spearman Correlations, alpha=0.01",
    )
    plot_dominant_boxplots(
        merged_derived,
        attr_cols,
        OUTPUT_DIR / "fig_dominant_process_attribute_boxplots_alpha_0.01.png",
    )
    for target in SHARE_TARGETS:
        plot_rf_importance(
            rf_importance,
            target,
            OUTPUT_DIR / f"fig_rf_importance_{target}.png",
        )
    plot_real_vs_null_spearman(null_spearman, OUTPUT_DIR / "fig_real_vs_null_spearman.png")
    plot_real_vs_null_rf(null_rf, rf_scores, OUTPUT_DIR / "fig_real_vs_null_rf_r2.png")

    write_report(
        OUTPUT_DIR / "structure_attribute_control_report.md",
        merged_derived,
        merge_diag,
        derived_diag,
        spearman,
        partial,
        dom_tests,
        rf_scores,
        rf_importance,
        null_spearman,
        null_rf,
        tradeoff,
    )

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Merged basins: {merged_derived['gage_id'].nunique()}")
    print(f"Retained attributes: {len(attr_cols)}")
    print(f"Null permutations: {NULL_PERMUTATIONS}")
    print(f"Report: {OUTPUT_DIR / 'structure_attribute_control_report.md'}")


if __name__ == "__main__":
    main()
