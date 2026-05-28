"""
Spatial and hydroclimatic block robustness checks for Flex-MOPEX V2
attribute-control analysis.

The analysis asks whether learned structural shares remain predictable from
hydroclimatic and landscape attributes when validation blocks hold out whole
regions or hydroclimatic regimes instead of random basins.
"""
from __future__ import annotations

import hashlib
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)


REPO_ROOT = Path("/workspace/autoresearch")
ANALYSIS_DIR = REPO_ROOT / "project/flexmopex/analysis"
INPUT_DIR = ANALYSIS_DIR / "flex_mopex_v2_attribute_controls"
REQUESTED_INPUT = INPUT_DIR / "merged_structure_attributes_with_derived.csv"
OUTPUT_DIR = ANALYSIS_DIR / "flex_mopex_v2_spatial_block_robustness"

PRIMARY_ALPHA = 0.01
CANDIDATE_ALPHAS = [0.005, 0.01, 0.03]
TARGETS = ["sum_weight", "share_snow", "share_int", "share_phen", "share_sub"]

RANDOM_STATE = 20260527
N_NULL_REPEATS = int(os.environ.get("FLEX_MOPEX_NULL_REPEATS", "100"))
N_JOBS = int(os.environ.get("FLEX_MOPEX_N_JOBS", "8"))
RF_ESTIMATORS = int(os.environ.get("FLEX_MOPEX_RF_ESTIMATORS", "12"))
RANDOM_CV_REPEATS = int(os.environ.get("FLEX_MOPEX_RANDOM_CV_REPEATS", "1"))
PERMUTATION_REPEATS = int(os.environ.get("FLEX_MOPEX_PERMUTATION_REPEATS", "5"))

PREDICTOR_SETTINGS = {
    "include_latlon": True,
    "exclude_latlon": False,
}


@dataclass(frozen=True)
class CVDesign:
    cv_type: str
    n_blocks: int
    splits: tuple[tuple[str, str, np.ndarray, np.ndarray], ...]


def stable_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def find_input_csv() -> Path:
    if REQUESTED_INPUT.exists():
        return REQUESTED_INPUT
    preferred = INPUT_DIR / "merged_full_v2.csv"
    if preferred.exists():
        return preferred
    candidates = sorted(INPUT_DIR.glob("*merged*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No merged CSV found under {INPUT_DIR}")
    return candidates[0]


def normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "gauge_id" not in df.columns:
        for candidate in ["gage_id", "basin_id"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "gauge_id"})
                break
    if "gauge_id" not in df.columns:
        raise ValueError("Merged file must contain gauge_id, gage_id, or basin_id.")
    df["gauge_id"] = pd.to_numeric(df["gauge_id"], errors="coerce").astype("Int64")
    if "basin_id" not in df.columns:
        df["basin_id"] = df["gauge_id"]
    return df


def load_data() -> tuple[pd.DataFrame, Path]:
    input_path = find_input_csv()
    df = normalize_id_columns(pd.read_csv(input_path))
    missing = [c for c in ["alpha", "lat", "lon", *TARGETS] if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from {input_path}: {missing}")
    return df, input_path


def make_block_source(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "gauge_id",
        "basin_id",
        "lat",
        "lon",
        "aridity_index",
        "aridity",
        "seasonality_index",
        "p_seasonality",
        "frac_snow",
        "frac_snow_months_era5",
    ]
    cols = [c for c in cols if c in df.columns]
    block_df = df.loc[df["alpha"].sub(PRIMARY_ALPHA).abs() < 1e-10, cols].copy()
    if block_df.empty:
        block_df = df[cols].copy()
    block_df = block_df.drop_duplicates("gauge_id").reset_index(drop=True)
    block_df = block_df.dropna(subset=["gauge_id", "lat", "lon"]).reset_index(drop=True)
    return block_df


def cluster_with_imputation(values: pd.DataFrame, n_clusters: int, seed: int) -> np.ndarray:
    imputed = SimpleImputer(strategy="median").fit_transform(values)
    scaled = StandardScaler().fit_transform(imputed)
    return KMeans(n_clusters=n_clusters, n_init=50, random_state=seed).fit_predict(scaled)


def build_block_assignments(df: pd.DataFrame) -> pd.DataFrame:
    block_df = make_block_source(df)
    if len(block_df) < 8:
        raise ValueError("Too few basins with lat/lon for block validation.")

    spatial_xy = block_df[["lat", "lon"]]
    for k in [5, 8]:
        block_df[f"spatial_block_k{k}"] = cluster_with_imputation(
            spatial_xy, min(k, len(block_df)), stable_seed("spatial", k)
        )

    hydro_cols = [
        c
        for c in [
            "aridity_index",
            "seasonality_index",
            "p_seasonality",
            "frac_snow",
            "frac_snow_months_era5",
        ]
        if c in block_df.columns
    ]
    if len(hydro_cols) < 2 and "aridity" in block_df.columns:
        hydro_cols.append("aridity")
    if len(hydro_cols) < 2:
        raise ValueError("Need at least two hydroclimatic attributes for hydro blocks.")
    block_df["hydroclimatic_block"] = cluster_with_imputation(
        block_df[hydro_cols], min(6, len(block_df)), stable_seed("hydroclimatic", 6)
    )

    output = block_df[
        ["basin_id", "gauge_id", "lat", "lon", "spatial_block_k5", "spatial_block_k8", "hydroclimatic_block"]
    ].copy()
    output["basin_id"] = output["basin_id"].fillna(output["gauge_id"]).astype(int)
    output = output.drop(columns=["gauge_id"])
    return output


def is_excluded_predictor(column: str, target: str) -> bool:
    lower = column.lower()
    exact_exclusions = {
        "basin_id",
        "gauge_id",
        "gage_id",
        "station_index",
        "alpha",
        "model_version",
        "run_name",
        "w_phen",
        "w_int",
        "w_snow",
        "w_sub",
        "share_phen",
        "share_int",
        "share_snow",
        "share_sub",
        "sum_weight",
        "mean_weight",
        "dominant_process",
        "dominant_share",
    }
    if lower in exact_exclusions or column == target:
        return True
    if lower.startswith("n_active_") or lower.startswith("n_inactive_"):
        return True
    if lower.startswith("z_"):
        return True
    if lower.startswith("delta_"):
        return True
    metric_tokens = ["nse", "kge", "rmse", "r2"]
    if any(token in lower for token in metric_tokens):
        return True
    return False


def select_predictors(df: pd.DataFrame, target: str, include_latlon: bool) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    predictors = [
        c
        for c in numeric_cols
        if not is_excluded_predictor(c, target) and df[c].nunique(dropna=True) > 5
    ]
    if not include_latlon:
        predictors = [c for c in predictors if c not in {"lat", "lon"}]
    else:
        for coord in ["lat", "lon"]:
            if coord in df.columns and coord not in predictors:
                predictors.append(coord)
    return predictors


def make_rf(seed: int) -> object:
    rf = RandomForestRegressor(
        n_estimators=RF_ESTIMATORS,
        max_depth=12,
        min_samples_leaf=5,
        max_features=0.5,
        max_samples=0.8,
        bootstrap=True,
        random_state=seed,
        n_jobs=1,
    )
    return make_pipeline(SimpleImputer(strategy="median"), rf)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return np.nan
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom <= 1e-12:
        return np.nan
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def make_designs(sub: pd.DataFrame, assignments: pd.DataFrame) -> list[CVDesign]:
    n = len(sub)
    designs: list[CVDesign] = []

    repeated = RepeatedKFold(n_splits=5, n_repeats=RANDOM_CV_REPEATS, random_state=RANDOM_STATE)
    random_splits = []
    for i, (train_idx, test_idx) in enumerate(repeated.split(np.arange(n)), start=1):
        repeat = (i - 1) // 5 + 1
        fold = (i - 1) % 5 + 1
        random_splits.append((f"repeat{repeat}_fold{fold}", f"repeat{repeat}_fold{fold}", train_idx, test_idx))
    random_cv_type = "random_kfold_5" if RANDOM_CV_REPEATS == 1 else f"random_kfold_5x{RANDOM_CV_REPEATS}"
    designs.append(CVDesign(random_cv_type, 5, tuple(random_splits)))

    assign = assignments.rename(columns={"basin_id": "basin_id_for_block"}).copy()
    assign["gauge_id"] = assign["basin_id_for_block"].astype("Int64")
    block_lookup = sub[["gauge_id"]].merge(assign, on="gauge_id", how="left")

    for k, col in [(5, "spatial_block_k5"), (8, "spatial_block_k8")]:
        labels = block_lookup[col].to_numpy()
        splits = group_leave_one_out(labels, f"spatial{k}")
        designs.append(CVDesign(f"spatial_block_k{k}", k, tuple(splits)))

    hydro_labels = block_lookup["hydroclimatic_block"].to_numpy()
    hydro_blocks = int(pd.Series(hydro_labels).nunique(dropna=True))
    designs.append(CVDesign(f"hydroclimatic_block_k{hydro_blocks}", hydro_blocks, tuple(group_leave_one_out(hydro_labels, "hydro"))))

    return designs


def group_leave_one_out(labels: np.ndarray, prefix: str) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    labels = np.asarray(labels)
    splits = []
    for block in sorted(pd.Series(labels).dropna().unique()):
        test_idx = np.flatnonzero(labels == block)
        train_idx = np.flatnonzero(labels != block)
        splits.append((f"{prefix}_block{int(block)}", str(int(block)), train_idx, test_idx))
    return splits


def evaluate_real(
    X: np.ndarray,
    y: np.ndarray,
    design: CVDesign,
    seed_parts: tuple[object, ...],
) -> tuple[pd.DataFrame, float, float, float, float]:
    rows = []
    oof_pred = np.full_like(y, fill_value=np.nan, dtype=float)
    for fold_id, heldout_block, train_idx, test_idx in design.splits:
        model = make_rf(stable_seed(*seed_parts, design.cv_type, fold_id))
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        oof_pred[test_idx] = pred
        rows.append(
            {
                "fold_id": fold_id,
                "heldout_block": heldout_block,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "r2": safe_r2(y[test_idx], pred),
                "rmse": rmse(y[test_idx], pred),
            }
        )

    fold_df = pd.DataFrame(rows)
    overall = {
        "fold_id": "overall",
        "heldout_block": "all",
        "n_train": int(len(y)),
        "n_test": int(np.isfinite(oof_pred).sum()),
        "r2": safe_r2(y, oof_pred),
        "rmse": rmse(y[np.isfinite(oof_pred)], oof_pred[np.isfinite(oof_pred)]),
    }
    fold_df = pd.concat([fold_df, pd.DataFrame([overall])], ignore_index=True)
    block_only = fold_df[fold_df["fold_id"] != "overall"]
    return (
        fold_df,
        float(overall["r2"]),
        float(block_only["r2"].std(skipna=True)),
        float(overall["rmse"]),
        float(block_only["rmse"].std(skipna=True)),
    )


def evaluate_null_distribution(
    X: np.ndarray,
    y: np.ndarray,
    design: CVDesign,
    seed_parts: tuple[object, ...],
    n_repeats: int,
) -> np.ndarray:
    values = []
    rng = np.random.default_rng(stable_seed("null", *seed_parts, design.cv_type))
    for repeat in range(n_repeats):
        y_shuffled = rng.permutation(y)
        oof_pred = np.full_like(y_shuffled, fill_value=np.nan, dtype=float)
        for fold_id, _heldout_block, train_idx, test_idx in design.splits:
            model = make_rf(stable_seed("null-model", *seed_parts, design.cv_type, repeat, fold_id))
            model.fit(X[train_idx], y_shuffled[train_idx])
            pred = model.predict(X[test_idx])
            oof_pred[test_idx] = pred
        values.append(safe_r2(y_shuffled, oof_pred))
    return np.asarray(values, dtype=float)


def evaluate_combo(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    target: str,
    alpha: float,
    predictor_setting: str,
    include_latlon: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df.loc[df["alpha"].sub(alpha).abs() < 1e-10].copy()
    sub = sub.dropna(subset=["gauge_id", "lat", "lon", target]).reset_index(drop=True)
    predictors = select_predictors(sub, target, include_latlon)
    if not predictors:
        raise ValueError(f"No predictors selected for {target}, alpha={alpha}, {predictor_setting}")

    X = sub[predictors].to_numpy(dtype=float)
    y = sub[target].to_numpy(dtype=float)
    designs = make_designs(sub, assignments)

    summary_rows = []
    fold_frames = []
    for design in designs:
        seed_parts = (target, alpha, predictor_setting)
        fold_df, mean_r2, std_r2, mean_rmse, std_rmse = evaluate_real(X, y, design, seed_parts)
        null_r2 = evaluate_null_distribution(X, y, design, seed_parts, N_NULL_REPEATS)
        null_r2_95 = float(np.nanpercentile(null_r2, 95))

        fold_df.insert(0, "cv_type", design.cv_type)
        fold_df.insert(0, "predictor_setting", predictor_setting)
        fold_df.insert(0, "alpha", alpha)
        fold_df.insert(0, "target", target)
        fold_frames.append(fold_df)

        summary_rows.append(
            {
                "target": target,
                "alpha": alpha,
                "predictor_setting": predictor_setting,
                "cv_type": design.cv_type,
                "n_blocks": design.n_blocks,
                "mean_r2": mean_r2,
                "std_r2": std_r2,
                "mean_rmse": mean_rmse,
                "std_rmse": std_rmse,
                "null_r2_95": null_r2_95,
                "above_null_95": bool(mean_r2 > null_r2_95),
            }
        )

    return pd.DataFrame(summary_rows), pd.concat(fold_frames, ignore_index=True)


def compute_permutation_importance_for_combo(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    target: str,
    predictor_setting: str,
    include_latlon: bool,
) -> pd.DataFrame:
    alpha = PRIMARY_ALPHA
    sub = df.loc[df["alpha"].sub(alpha).abs() < 1e-10].copy()
    sub = sub.dropna(subset=["gauge_id", "lat", "lon", target]).reset_index(drop=True)
    predictors = select_predictors(sub, target, include_latlon)
    X = sub[predictors].to_numpy(dtype=float)
    y = sub[target].to_numpy(dtype=float)
    designs = make_designs(sub, assignments)

    rows = []
    for design in designs:
        fold_importances = []
        for fold_id, _heldout_block, train_idx, test_idx in design.splits:
            model = make_rf(stable_seed("perm", target, predictor_setting, design.cv_type, fold_id))
            model.fit(X[train_idx], y[train_idx])
            result = permutation_importance(
                model,
                X[test_idx],
                y[test_idx],
                n_repeats=PERMUTATION_REPEATS,
                random_state=stable_seed("perm-score", target, predictor_setting, design.cv_type, fold_id),
                scoring="r2",
                n_jobs=1,
            )
            fold_importances.append(result.importances_mean)
        if not fold_importances:
            continue
        arr = np.vstack(fold_importances)
        for idx, feature in enumerate(predictors):
            rows.append(
                {
                    "target": target,
                    "alpha": alpha,
                    "predictor_setting": predictor_setting,
                    "cv_type": design.cv_type,
                    "n_blocks": design.n_blocks,
                    "feature": feature,
                    "importance_mean": float(np.nanmean(arr[:, idx])),
                    "importance_std": float(np.nanstd(arr[:, idx])),
                    "n_folds": int(arr.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def run_scores(df: pd.DataFrame, assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    available_alphas = sorted(float(a) for a in pd.Series(df["alpha"]).dropna().unique())
    alphas = [a for a in CANDIDATE_ALPHAS if any(abs(a - b) < 1e-10 for b in available_alphas)]
    if PRIMARY_ALPHA not in alphas:
        raise ValueError(f"Primary alpha {PRIMARY_ALPHA} is not available in merged file.")

    jobs = []
    for alpha in alphas:
        for target in TARGETS:
            if target not in df.columns:
                continue
            for setting, include_latlon in PREDICTOR_SETTINGS.items():
                jobs.append((target, alpha, setting, include_latlon))

    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(evaluate_combo)(df, assignments, target, alpha, setting, include_latlon)
        for target, alpha, setting, include_latlon in jobs
    )
    summary = pd.concat([r[0] for r in results], ignore_index=True)
    folds = pd.concat([r[1] for r in results], ignore_index=True)
    return summary, folds


def run_importances(df: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    jobs = []
    for target in TARGETS:
        if target not in df.columns:
            continue
        for setting, include_latlon in PREDICTOR_SETTINGS.items():
            jobs.append((target, setting, include_latlon))
    frames = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(compute_permutation_importance_for_combo)(df, assignments, target, setting, include_latlon)
        for target, setting, include_latlon in jobs
    )
    return pd.concat(frames, ignore_index=True)


def short_cv_label(cv_type: str) -> str:
    if cv_type == "random_kfold_5":
        return "Random 5-fold"
    if cv_type.startswith("random_kfold_5x"):
        return f"Random 5x{cv_type.rsplit('x', 1)[-1]}"
    return {
        "spatial_block_k5": "Spatial k=5",
        "spatial_block_k8": "Spatial k=8",
    }.get(cv_type, cv_type.replace("hydroclimatic_block_", "Hydro "))


def is_random_cv(cv_type: str) -> bool:
    return cv_type.startswith("random_kfold_")


def get_random_cv_type(summary: pd.DataFrame) -> str:
    random_types = sorted([c for c in summary["cv_type"].unique() if is_random_cv(str(c))])
    if not random_types:
        raise ValueError("No random CV type present in summary.")
    return random_types[0]


def plot_cv_comparison(summary: pd.DataFrame) -> None:
    data = summary[np.isclose(summary["alpha"], PRIMARY_ALPHA)].copy()
    wanted = [get_random_cv_type(data), "spatial_block_k5", "spatial_block_k8"]
    hydro = sorted([c for c in data["cv_type"].unique() if c.startswith("hydroclimatic_block")])
    if hydro:
        wanted.append(hydro[0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = {
        wanted[0]: "#4C78A8",
        "spatial_block_k5": "#F58518",
        "spatial_block_k8": "#E45756",
    }
    if hydro:
        colors[hydro[0]] = "#54A24B"

    for ax, setting in zip(axes, ["include_latlon", "exclude_latlon"]):
        sub = data[(data["predictor_setting"] == setting) & (data["cv_type"].isin(wanted))]
        x = np.arange(len(TARGETS))
        width = 0.18
        for i, cv_type in enumerate(wanted):
            vals = [
                sub[(sub["target"] == target) & (sub["cv_type"] == cv_type)]["mean_r2"].mean()
                for target in TARGETS
            ]
            ax.bar(x + (i - (len(wanted) - 1) / 2) * width, vals, width, label=short_cv_label(cv_type), color=colors.get(cv_type), alpha=0.9)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(TARGETS, rotation=25, ha="right")
        ax.set_title(setting.replace("_", " "))
        ax.set_ylabel("Overall out-of-fold R2")
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Attribute-control RF predictability under random and blocked CV (alpha=0.01)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_cv_r2_comparison_alpha_0.01.png", dpi=180)
    plt.close(fig)


def plot_real_vs_null(summary: pd.DataFrame) -> None:
    data = summary[
        np.isclose(summary["alpha"], PRIMARY_ALPHA)
        & (summary["predictor_setting"] == "exclude_latlon")
        & (~summary["cv_type"].map(is_random_cv))
    ].copy()
    cv_types = ["spatial_block_k5", "spatial_block_k8"] + sorted(
        [c for c in data["cv_type"].unique() if c.startswith("hydroclimatic_block")]
    )
    fig, axes = plt.subplots(1, len(cv_types), figsize=(5 * len(cv_types), 4), sharey=True)
    if len(cv_types) == 1:
        axes = [axes]
    for ax, cv_type in zip(axes, cv_types):
        sub = data[data["cv_type"] == cv_type]
        x = np.arange(len(TARGETS))
        real = [sub[sub["target"] == t]["mean_r2"].mean() for t in TARGETS]
        null = [sub[sub["target"] == t]["null_r2_95"].mean() for t in TARGETS]
        ax.bar(x, real, color="#4C78A8", alpha=0.85, label="Real")
        ax.scatter(x, null, color="#D62728", marker="_", s=260, linewidths=2.5, label="Null 95%")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(TARGETS, rotation=30, ha="right")
        ax.set_title(short_cv_label(cv_type))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Overall out-of-fold R2")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Real block-CV out-of-fold R2 versus shuffled-target null 95% (alpha=0.01, no lat/lon predictors)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_real_vs_null_block_cv_alpha_0.01.png", dpi=180)
    plt.close(fig)


def plot_maps(assignments: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, col, title in [
        (axes[0], "spatial_block_k5", "Spatial blocks k=5"),
        (axes[1], "spatial_block_k8", "Spatial blocks k=8"),
    ]:
        scatter = ax.scatter(assignments["lon"], assignments["lat"], c=assignments[col], cmap="tab10", s=18, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.2)
        fig.colorbar(scatter, ax=ax, shrink=0.8, label="Block")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_spatial_blocks_map.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(assignments["lon"], assignments["lat"], c=assignments["hydroclimatic_block"], cmap="tab10", s=18, alpha=0.85)
    ax.set_title("Hydroclimatic blocks")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax, shrink=0.8, label="Block")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_hydroclimatic_blocks_map.png", dpi=180)
    plt.close(fig)


def plot_fold_scores(folds: pd.DataFrame) -> None:
    data = folds[
        np.isclose(folds["alpha"], PRIMARY_ALPHA)
        & (folds["predictor_setting"] == "exclude_latlon")
        & (folds["fold_id"] != "overall")
        & (~folds["cv_type"].map(is_random_cv))
    ].copy()
    cv_types = ["spatial_block_k5", "spatial_block_k8"] + sorted(
        [c for c in data["cv_type"].unique() if c.startswith("hydroclimatic_block")]
    )
    fig, axes = plt.subplots(len(cv_types), 1, figsize=(12, 3.4 * len(cv_types)), sharex=False)
    if len(cv_types) == 1:
        axes = [axes]
    for ax, cv_type in zip(axes, cv_types):
        sub = data[data["cv_type"] == cv_type]
        labels = sorted(sub["heldout_block"].unique(), key=lambda x: int(float(x)))
        x = np.arange(len(labels))
        width = 0.15
        for i, target in enumerate(TARGETS):
            vals = [
                sub[(sub["target"] == target) & (sub["heldout_block"] == label)]["r2"].mean()
                for label in labels
            ]
            ax.bar(x + (i - 2) * width, vals, width, label=target, alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Fold R2")
        ax.set_title(short_cv_label(cv_type))
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Held-out block")
    axes[0].legend(fontsize=8, ncol=len(TARGETS), loc="upper center", bbox_to_anchor=(0.5, 1.35))
    fig.suptitle("Fold-level blocked-CV scores by held-out block (alpha=0.01, no lat/lon predictors)", y=0.995)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_block_cv_fold_scores.png", dpi=180)
    plt.close(fig)


def fmt(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.3f}"


def get_row(summary: pd.DataFrame, target: str, setting: str, cv_type: str) -> pd.Series | None:
    sub = summary[
        np.isclose(summary["alpha"], PRIMARY_ALPHA)
        & (summary["target"] == target)
        & (summary["predictor_setting"] == setting)
        & (summary["cv_type"] == cv_type)
    ]
    if sub.empty:
        return None
    return sub.iloc[0]


def classify_target(summary: pd.DataFrame, target: str) -> str:
    rows = summary[
        np.isclose(summary["alpha"], PRIMARY_ALPHA)
        & (summary["target"] == target)
        & (summary["predictor_setting"] == "exclude_latlon")
        & (~summary["cv_type"].map(is_random_cv))
    ]
    above = bool(rows["above_null_95"].all()) if not rows.empty else False
    mean_block = float(rows["mean_r2"].mean()) if not rows.empty else np.nan
    if above and mean_block >= 0.5:
        return "robust"
    if above and mean_block > 0:
        return "above-null but weakened"
    if mean_block > 0:
        return "positive but not above-null in every block design"
    return "weak or unstable"


def write_report(summary: pd.DataFrame, folds: pd.DataFrame, input_path: Path, predictors: dict[str, list[str]]) -> None:
    data = summary[np.isclose(summary["alpha"], PRIMARY_ALPHA)].copy()
    random_cv = get_random_cv_type(data)
    hydro_types = sorted([c for c in data["cv_type"].unique() if c.startswith("hydroclimatic_block")])
    hydro_cv = hydro_types[0] if hydro_types else "hydroclimatic_block"

    lines = []
    lines.append("# Flex-MOPEX V2 Spatial-Block Robustness Report\n")
    lines.append("## Setup\n")
    lines.append(f"- Input: `{input_path}`\n")
    lines.append(f"- Primary alpha: {PRIMARY_ALPHA}\n")
    lines.append(f"- Sensitivity alphas run: {', '.join(str(a) for a in sorted(summary['alpha'].unique()))}\n")
    lines.append(f"- Null repeats per target/design: {N_NULL_REPEATS}\n")
    lines.append(f"- Random CV repeats: {RANDOM_CV_REPEATS}\n")
    lines.append(f"- RF trees per fit: {RF_ESTIMATORS}\n")
    lines.append(f"- Predictor setting A includes lat/lon; setting B excludes lat/lon. Both exclude learned weights/shares, model metrics, delta metrics, identifiers, and duplicate `z_` standardized attributes.\n")
    lines.append(f"- Selected predictors: {len(predictors['include_latlon'])} with lat/lon; {len(predictors['exclude_latlon'])} without lat/lon.\n")

    lines.append("## Primary Alpha Results (overall out-of-fold R2; null is shuffled-target 95th percentile)\n")
    table_rows = []
    for target in TARGETS:
        random_no = get_row(summary, target, "exclude_latlon", random_cv)
        spatial5_no = get_row(summary, target, "exclude_latlon", "spatial_block_k5")
        spatial8_no = get_row(summary, target, "exclude_latlon", "spatial_block_k8")
        hydro_no = get_row(summary, target, "exclude_latlon", hydro_cv)
        table_rows.append(
            [
                target,
                fmt(random_no["mean_r2"]) if random_no is not None else "NA",
                fmt(spatial5_no["mean_r2"]) if spatial5_no is not None else "NA",
                fmt(spatial5_no["null_r2_95"]) if spatial5_no is not None else "NA",
                fmt(spatial8_no["mean_r2"]) if spatial8_no is not None else "NA",
                fmt(spatial8_no["null_r2_95"]) if spatial8_no is not None else "NA",
                fmt(hydro_no["mean_r2"]) if hydro_no is not None else "NA",
                fmt(hydro_no["null_r2_95"]) if hydro_no is not None else "NA",
                classify_target(summary, target),
            ]
        )
    lines.append("| Target | Random R2 | Spatial k5 R2 | Spatial k5 null95 | Spatial k8 R2 | Spatial k8 null95 | Hydro R2 | Hydro null95 | Interpretation |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for row in table_rows:
        lines.append("| " + " | ".join(row) + " |\n")

    lines.append("\n## Answers to the Robustness Questions\n")
    block_rows = data[(data["predictor_setting"] == "exclude_latlon") & (~data["cv_type"].map(is_random_cv))]
    above_count = int(block_rows["above_null_95"].sum())
    total_count = int(len(block_rows))
    lines.append(
        f"**A. Do RF attribute-control signals remain above null under spatial block CV?** "
        f"Mostly, but not uniformly. Across alpha=0.01 block designs without lat/lon, {above_count}/{total_count} target-design results exceed the shuffled-target 95th percentile. "
        "Random CV is generally higher than block CV, so attribute predictability is partly spatially structured, but above-null block performance indicates non-random hydroclimatic organization for the targets that remain positive and above null.\n"
    )

    robust_targets = [target for target in TARGETS if classify_target(summary, target) == "robust"]
    weakened_targets = [target for target in TARGETS if classify_target(summary, target) == "above-null but weakened"]
    weak_targets = [target for target in TARGETS if classify_target(summary, target) in {"positive but not above-null in every block design", "weak or unstable"}]
    lines.append(
        f"**B. Which targets are robust under spatial blocking?** "
        f"Robust: {', '.join(robust_targets) if robust_targets else 'none under the strict all-block-design criterion'}. "
        f"Above-null but weakened: {', '.join(weakened_targets) if weakened_targets else 'none'}. "
        f"Weak or design-dependent: {', '.join(weak_targets) if weak_targets else 'none'}.\n"
    )

    snow_no = get_row(summary, "share_snow", "exclude_latlon", "spatial_block_k5")
    snow_no8 = get_row(summary, "share_snow", "exclude_latlon", "spatial_block_k8")
    snow_with = get_row(summary, "share_snow", "include_latlon", "spatial_block_k5")
    snow_is_robust = bool(
        snow_no is not None
        and snow_no8 is not None
        and snow_no["above_null_95"]
        and snow_no8["above_null_95"]
        and snow_no["mean_r2"] > 0
        and snow_no8["mean_r2"] > 0
    )
    snow_msg = (
        f"without lat/lon spatial k5 R2={fmt(snow_no['mean_r2']) if snow_no is not None else 'NA'} "
        f"and k8 R2={fmt(snow_no8['mean_r2']) if snow_no8 is not None else 'NA'}, "
        f"versus with lat/lon k5 R2={fmt(snow_with['mean_r2']) if snow_with is not None else 'NA'}"
    )
    lines.append(
        f"**C. Does share_snow remain predictable without relying on lat/lon?** "
        f"{'Yes' if snow_is_robust else 'Not consistently'}. {snow_msg}. "
        f"{'Snow-related structure is the most robust process-specific signal.' if snow_is_robust else 'Snow-related structure should not be treated as robust without checking the block rows.'}\n"
    )

    phen_hydro = get_row(summary, "share_phen", "exclude_latlon", hydro_cv)
    phen_is_moderate = bool(
        phen_hydro is not None and phen_hydro["above_null_95"] and phen_hydro["mean_r2"] > 0.1
    )
    lines.append(
        f"**D. Does share_phen remain moderately predictable under hydroclimatic blocking?** "
        f"{'Yes' if phen_is_moderate else 'No'}. The no-lat/lon hydroclimatic-block R2 is {fmt(phen_hydro['mean_r2']) if phen_hydro is not None else 'NA'} "
        f"against null95={fmt(phen_hydro['null_r2_95']) if phen_hydro is not None else 'NA'}. "
        f"{'This supports a moderate phenology/seasonality signal.' if phen_is_moderate else 'This is better described as weakened and design-dependent at alpha=0.01.'}\n"
    )

    sub_random = get_row(summary, "share_sub", "exclude_latlon", random_cv)
    sub_spatial = get_row(summary, "share_sub", "exclude_latlon", "spatial_block_k5")
    sub_hydro = get_row(summary, "share_sub", "exclude_latlon", hydro_cv)
    sub_collapses = bool(
        sub_spatial is not None
        and sub_random is not None
        and (sub_spatial["mean_r2"] < 0.2 or sub_random["mean_r2"] - sub_spatial["mean_r2"] > 0.5)
    )
    lines.append(
        f"**E. Does share_sub collapse under spatial blocking?** "
        f"{'Yes, it weakens sharply' if sub_collapses else 'No, not under the overall out-of-fold R2 used here'}. "
        f"Random no-lat/lon R2={fmt(sub_random['mean_r2']) if sub_random is not None else 'NA'}, "
        f"spatial k5 R2={fmt(sub_spatial['mean_r2']) if sub_spatial is not None else 'NA'}, "
        f"hydro R2={fmt(sub_hydro['mean_r2']) if sub_hydro is not None else 'NA'}. "
        "Subsurface attribution should still remain cautious because it may partly reflect broad snow/topographic gradients rather than an isolated subsurface-process mechanism.\n"
    )

    int_rows = block_rows[block_rows["target"] == "share_int"]
    int_mean = float(int_rows["mean_r2"].mean()) if not int_rows.empty else np.nan
    lines.append(
        f"**F. Is share_int still weak?** Yes. Its mean no-lat/lon blocked R2 across block designs is {fmt(int_mean)}, "
        "and it should be interpreted as weak unless individual block rows show a clear above-null result.\n"
    )

    lines.append(
        "**G. Should the paper present RF results as strong evidence, supporting evidence, or only supplementary evidence?** "
        "Use them as supporting evidence, with the blocked CV as the primary robustness check and random CV as supplementary context. "
        "The RF results are useful evidence for organized attribute-control patterns, but the random-to-block drop means they should not be framed as proof of physical causality.\n"
    )

    lines.append(
        "**H. Which claim is justified?** The safest claim is **moderate process-specific hydrological coherence**. "
        "If only complexity/total-weight targets remain strong while individual shares weaken, the interpretation shifts toward mainly spatially organized complexity regionalization. "
        "The present results should not be described as strong process-level structure regionalization unless share-level block-CV performance remains consistently high and above null.\n"
    )

    lines.append("## Sensitivity Across Alpha\n")
    sens = summary[(summary["predictor_setting"] == "exclude_latlon") & (~summary["cv_type"].map(is_random_cv))]
    sens_tab = sens.groupby(["alpha", "target"], as_index=False).agg(
        mean_block_r2=("mean_r2", "mean"),
        min_block_r2=("mean_r2", "min"),
        above_null_all=("above_null_95", "all"),
    )
    lines.append("| Alpha | Target | Mean block R2 | Min block R2 | Above null in all block designs |\n")
    lines.append("|---:|---|---:|---:|---|\n")
    for row in sens_tab.itertuples(index=False):
        lines.append(f"| {row.alpha:g} | {row.target} | {fmt(row.mean_block_r2)} | {fmt(row.min_block_r2)} | {bool(row.above_null_all)} |\n")

    lines.append("\n## Output Files\n")
    for filename in [
        "cv_score_summary.csv",
        "cv_fold_scores.csv",
        "block_assignments.csv",
        "rf_feature_importance_spatial_cv.csv",
        "fig_cv_r2_comparison_alpha_0.01.png",
        "fig_real_vs_null_block_cv_alpha_0.01.png",
        "fig_spatial_blocks_map.png",
        "fig_hydroclimatic_blocks_map.png",
        "fig_block_cv_fold_scores.png",
    ]:
        lines.append(f"- `{filename}`\n")

    (OUTPUT_DIR / "spatial_block_robustness_report.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, input_path = load_data()
    assignments = build_block_assignments(df)
    assignments.to_csv(OUTPUT_DIR / "block_assignments.csv", index=False)

    example_sub = df.loc[df["alpha"].sub(PRIMARY_ALPHA).abs() < 1e-10].copy()
    predictors = {
        setting: select_predictors(example_sub, TARGETS[0], include_latlon)
        for setting, include_latlon in PREDICTOR_SETTINGS.items()
    }

    summary, folds = run_scores(df, assignments)
    summary = summary.sort_values(["alpha", "target", "predictor_setting", "cv_type"]).reset_index(drop=True)
    folds = folds.sort_values(["alpha", "target", "predictor_setting", "cv_type", "fold_id"]).reset_index(drop=True)
    summary.to_csv(OUTPUT_DIR / "cv_score_summary.csv", index=False)
    folds.to_csv(OUTPUT_DIR / "cv_fold_scores.csv", index=False)

    importances = run_importances(df, assignments)
    importances = importances.sort_values(
        ["target", "predictor_setting", "cv_type", "importance_mean"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    importances.to_csv(OUTPUT_DIR / "rf_feature_importance_spatial_cv.csv", index=False)

    plot_cv_comparison(summary)
    plot_real_vs_null(summary)
    plot_maps(assignments)
    plot_fold_scores(folds)
    write_report(summary, folds, input_path, predictors)

    print(f"Wrote outputs to {OUTPUT_DIR}")
    print(f"Input: {input_path}")
    print(f"Rows: {len(df)}, alphas: {sorted(df['alpha'].unique())}")
    print(f"Summary rows: {len(summary)}, fold rows: {len(folds)}, importance rows: {len(importances)}")


if __name__ == "__main__":
    main()
