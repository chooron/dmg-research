from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DIAG_DIR = Path("/workspace/autoresearch/project/flexmopex/analysis/flex_mopex_v1_npz_diagnostics")
MODEL_ROOT = Path("/workspace/autoresearch/project/flexmopex/outputs/flex_mopex_v1")
LOCATION_SHP = Path("/workspace/autoresearch/data/camels_loc/camels_671_loc.shp")
GAGE_ID_PATH = Path("/workspace/autoresearch/data/gage_id.npy")
OUTPUT_DIR = Path(
    "/workspace/autoresearch/project/flexmopex/analysis/flex_mopex_v1_structure_learning_interpretation"
)

WEIGHT_NAMES = ["w_phen", "w_int", "w_snow", "w_sub"]
SHARE_NAMES = ["share_phen", "share_int", "share_snow", "share_sub"]
PROCESS_LABELS = {
    "w_phen": "phen",
    "w_int": "int",
    "w_snow": "snow",
    "w_sub": "sub",
    "share_phen": "phen",
    "share_int": "int",
    "share_snow": "snow",
    "share_sub": "sub",
}
PROCESS_COLORS = {
    "phen": "#4C78A8",
    "int": "#F58518",
    "snow": "#54A24B",
    "sub": "#B279A2",
    "undefined": "#9E9E9E",
}
FOCUS_ALPHAS = [0.003, 0.005, 0.007, 0.01, 0.03, 0.05]
PERMUTATIONS = 999
K_NEIGHBORS = 8


def format_alpha(alpha: float) -> str:
    return f"{alpha:g}"


def read_basin_locations() -> pd.DataFrame:
    gage_ids = np.load(GAGE_ID_PATH, allow_pickle=True).astype(int)
    order = pd.DataFrame({"station_index": np.arange(len(gage_ids)), "gage_id": gage_ids})
    gdf = gpd.read_file(LOCATION_SHP)
    loc = pd.DataFrame(
        {
            "gage_id": gdf["gage_id"].astype(int),
            "lat": gdf["lat"].astype(float),
            "lon": gdf["lon"].astype(float),
        }
    )
    merged = order.merge(loc, on="gage_id", how="left")
    missing = merged[["lat", "lon"]].isna().any(axis=1).sum()
    if missing:
        raise ValueError(f"Missing coordinates for {missing} station rows")
    return merged


def load_metrics_by_basin() -> pd.DataFrame:
    rows = []
    for path in sorted(MODEL_ROOT.rglob("metrics.json")):
        alpha = parse_alpha_from_parts(path)
        raw = json.load(path.open("r", encoding="utf-8"))
        data = json.loads(raw) if isinstance(raw, str) else raw
        n_basin = len(data["nse"])
        for station_index in range(n_basin):
            rows.append(
                {
                    "alpha": alpha,
                    "station_index": station_index,
                    "nse": float(data["nse"][station_index]),
                    "kge": float(data["kge"][station_index]),
                    "r2": float(data["r2"][station_index]),
                    "rmse": float(data["rmse"][station_index]),
                }
            )
    return pd.DataFrame(rows)


def parse_alpha_from_parts(path: Path) -> float:
    for part in path.parts:
        if part.startswith("alpha_"):
            value = part.replace("alpha_", "").replace("_", ".")
            return float(value)
    raise ValueError(f"Cannot parse alpha from path: {path}")


def load_weights() -> pd.DataFrame:
    basin = pd.read_csv(DIAG_DIR / "basin_complexity.csv")
    basin = basin.sort_values(["alpha", "station_index"]).reset_index(drop=True)
    return basin


def add_shares(basin: pd.DataFrame) -> pd.DataFrame:
    out = basin.copy()
    out["sum_weight"] = out[WEIGHT_NAMES].sum(axis=1)
    valid = out["sum_weight"] > 1e-6
    for weight_name, share_name in zip(WEIGHT_NAMES, SHARE_NAMES):
        out[share_name] = np.nan
        out.loc[valid, share_name] = out.loc[valid, weight_name] / out.loc[valid, "sum_weight"]
    out["dominant_process"] = "undefined"
    out["dominant_share"] = np.nan
    share_values = out.loc[valid, SHARE_NAMES].to_numpy(dtype=float)
    dominant_index = np.argmax(share_values, axis=1)
    out.loc[valid, "dominant_process"] = [
        PROCESS_LABELS[SHARE_NAMES[index]] for index in dominant_index
    ]
    out.loc[valid, "dominant_share"] = np.max(share_values, axis=1)
    return out


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan
    if method == "pearson":
        result = stats.pearsonr(x, y)
    elif method == "spearman":
        result = stats.spearmanr(x, y)
    else:
        raise ValueError(f"Unsupported correlation method: {method}")
    return float(result.statistic), float(result.pvalue)


def ensure_locations(frame: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    if "lon" in frame.columns and "lat" in frame.columns:
        return frame
    return frame.merge(locations, on="station_index", how="left")


def pairwise_correlations(weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alpha, group in weights.groupby("alpha", sort=True):
        for left_index, left in enumerate(WEIGHT_NAMES):
            for right in WEIGHT_NAMES[left_index + 1 :]:
                x = group[left].to_numpy(dtype=float)
                y = group[right].to_numpy(dtype=float)
                pearson, pearson_p = safe_corr(x, y, "pearson")
                spearman, spearman_p = safe_corr(x, y, "spearman")
                rows.append(
                    {
                        "alpha": alpha,
                        "weight_left": left,
                        "weight_right": right,
                        "pearson": pearson,
                        "pearson_p": pearson_p,
                        "spearman": spearman,
                        "spearman_p": spearman_p,
                    }
                )
    return pd.DataFrame(rows)


def weight_sum_correlations(weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alpha, group in weights.groupby("alpha", sort=True):
        sum_weight = group["sum_weight"].to_numpy(dtype=float)
        for weight_name in WEIGHT_NAMES:
            values = group[weight_name].to_numpy(dtype=float)
            pearson, pearson_p = safe_corr(values, sum_weight, "pearson")
            spearman, spearman_p = safe_corr(values, sum_weight, "spearman")
            rows.append(
                {
                    "alpha": alpha,
                    "weight_name": weight_name,
                    "pearson_with_sum_weight": pearson,
                    "pearson_p": pearson_p,
                    "spearman_with_sum_weight": spearman,
                    "spearman_p": spearman_p,
                }
            )
    return pd.DataFrame(rows)


def jaccard_similarity(weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alpha, group in weights.groupby("alpha", sort=True):
        for threshold in [0.5, 0.8]:
            active = {name: set(group.loc[group[name] > threshold, "station_index"]) for name in WEIGHT_NAMES}
            for left_index, left in enumerate(WEIGHT_NAMES):
                for right in WEIGHT_NAMES[left_index + 1 :]:
                    union = active[left] | active[right]
                    intersection = active[left] & active[right]
                    score = 1.0 if len(union) == 0 else len(intersection) / len(union)
                    rows.append(
                        {
                            "alpha": alpha,
                            "threshold": threshold,
                            "weight_left": left,
                            "weight_right": right,
                            "n_left_active": len(active[left]),
                            "n_right_active": len(active[right]),
                            "n_intersection": len(intersection),
                            "n_union": len(union),
                            "jaccard": score,
                        }
                    )
    return pd.DataFrame(rows)


def share_summary(weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = weights[weights["sum_weight"] > 1e-6]
    for (alpha, share_name), group in valid.melt(
        id_vars=["alpha", "station_index"],
        value_vars=SHARE_NAMES,
        var_name="share_name",
        value_name="share_value",
    ).groupby(["alpha", "share_name"], sort=True):
        values = group["share_value"]
        rows.append(
            {
                "alpha": alpha,
                "share_name": share_name,
                "n_station": int(values.count()),
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),
                "q05": values.quantile(0.05),
                "q25": values.quantile(0.25),
                "q75": values.quantile(0.75),
                "q95": values.quantile(0.95),
                "min": values.min(),
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def dominant_process_table(weights: pd.DataFrame) -> pd.DataFrame:
    return weights[
        [
            "alpha",
            "run_name",
            "station_index",
            "sum_weight",
            *SHARE_NAMES,
            "dominant_process",
            "dominant_share",
        ]
    ].copy()


def saturation_summary(weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (alpha, weight_name), group in weights.melt(
        id_vars=["alpha", "station_index"],
        value_vars=WEIGHT_NAMES,
        var_name="weight_name",
        value_name="weight_value",
    ).groupby(["alpha", "weight_name"], sort=True):
        values = group["weight_value"]
        rows.append(
            {
                "alpha": alpha,
                "weight_name": weight_name,
                "n_station": len(values),
                "frac_lt_0p01": (values < 0.01).mean(),
                "frac_lt_0p10": (values < 0.1).mean(),
                "frac_mid_0p10_0p90": ((values >= 0.1) & (values <= 0.9)).mean(),
                "frac_gt_0p90": (values > 0.9).mean(),
                "q05": values.quantile(0.05),
                "q25": values.quantile(0.25),
                "q50": values.quantile(0.50),
                "q75": values.quantile(0.75),
                "q95": values.quantile(0.95),
            }
        )
    return pd.DataFrame(rows)


def performance_complexity_tradeoff(complexity: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    metric_wide = metrics.pivot_table(index="alpha", columns="metric_name", values=["median", "mean"])
    metric_wide.columns = [f"{metric}_{stat}" for stat, metric in metric_wide.columns]
    metric_wide = metric_wide.reset_index()
    out = complexity.merge(metric_wide, on="alpha", how="left")
    ref = out.loc[out["alpha"] == 0].iloc[0]
    out["delta_median_kge_vs_alpha0"] = out["kge_median"] - ref["kge_median"]
    out["delta_median_nse_vs_alpha0"] = out["nse_median"] - ref["nse_median"]
    out["delta_mean_kge_vs_alpha0"] = out["kge_mean"] - ref["kge_mean"]
    out["delta_mean_nse_vs_alpha0"] = out["nse_mean"] - ref["nse_mean"]
    out["complexity_reduction_vs_alpha0"] = 1.0 - out["mean_sum_weight"] / ref["mean_sum_weight"]
    out["candidate_score"] = (
        out["complexity_reduction_vs_alpha0"]
        - 8.0 * np.maximum(0.0, -out["delta_median_kge_vs_alpha0"])
        - 3.0 * np.maximum(0.0, -out["delta_median_nse_vs_alpha0"])
    )
    return out.sort_values("alpha")


def select_candidate_alphas(tradeoff: pd.DataFrame) -> list[float]:
    eligible = tradeoff[(tradeoff["alpha"] > 0) & (tradeoff["alpha"] < 0.1)].copy()
    targets = [
        {"target_reduction": 0.50, "max_kge_loss": 0.02},
        {"target_reduction": 0.65, "max_kge_loss": 0.03},
        {"target_reduction": 0.78, "max_kge_loss": 0.03},
    ]
    selected = []
    for target in targets:
        subset = eligible[eligible["delta_median_kge_vs_alpha0"] >= -target["max_kge_loss"]].copy()
        if subset.empty:
            subset = eligible.copy()
        subset["target_distance"] = (
            subset["complexity_reduction_vs_alpha0"] - target["target_reduction"]
        ).abs()
        alpha = float(subset.sort_values(["target_distance", "candidate_score"], ascending=[True, False]).iloc[0]["alpha"])
        if alpha not in selected:
            selected.append(alpha)
    if len(selected) < 2:
        fallback = eligible.sort_values("candidate_score", ascending=False)["alpha"].tolist()
        for alpha in fallback:
            if float(alpha) not in selected:
                selected.append(float(alpha))
            if len(selected) == 3:
                break
    return selected[:3]


def load_complexity_and_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    complexity = pd.read_csv(DIAG_DIR / "complexity_alpha_summary.csv")
    metrics = pd.read_csv(DIAG_DIR / "metrics_alpha_summary.csv")
    return complexity, metrics


def basin_delta_metrics(weights: pd.DataFrame, metrics_by_basin: pd.DataFrame) -> pd.DataFrame:
    merged = weights[
        ["alpha", "station_index", "sum_weight", *WEIGHT_NAMES, *SHARE_NAMES, "dominant_process"]
    ].merge(metrics_by_basin, on=["alpha", "station_index"], how="left")
    ref = merged[merged["alpha"] == 0][["station_index", "sum_weight", "nse", "kge"]].rename(
        columns={
            "sum_weight": "sum_weight_alpha0",
            "nse": "nse_alpha0",
            "kge": "kge_alpha0",
        }
    )
    out = merged.merge(ref, on="station_index", how="left")
    out["delta_nse"] = out["nse"] - out["nse_alpha0"]
    out["delta_kge"] = out["kge"] - out["kge_alpha0"]
    out["delta_sum_weight"] = out["sum_weight"] - out["sum_weight_alpha0"]
    return out


def build_knn_indices(coords: np.ndarray, k: int) -> np.ndarray:
    radians = np.radians(coords[:, [1, 0]])
    model = NearestNeighbors(n_neighbors=k + 1, metric="haversine")
    model.fit(radians)
    return model.kneighbors(radians, return_distance=False)[:, 1:]


def morans_i(values: np.ndarray, neighbor_idx: np.ndarray) -> float:
    z = values - values.mean()
    denominator = float(np.dot(z, z))
    if denominator == 0:
        return np.nan
    lag = z[neighbor_idx].mean(axis=1)
    return float(np.dot(z, lag) / denominator)


def morans_i_permutation(values: np.ndarray, neighbor_idx: np.ndarray, permutations: int, rng: np.random.Generator) -> tuple[float, float]:
    observed = morans_i(values, neighbor_idx)
    if np.isnan(observed):
        return np.nan, np.nan
    count = 0
    for _ in range(permutations):
        permuted = rng.permutation(values)
        permuted_i = morans_i(permuted, neighbor_idx)
        if abs(permuted_i) >= abs(observed):
            count += 1
    p_value = (count + 1) / (permutations + 1)
    return observed, p_value


def compute_morans(weights: pd.DataFrame, locations: pd.DataFrame, candidate_alphas: list[float]) -> pd.DataFrame:
    coords = locations.sort_values("station_index")[["lon", "lat"]].to_numpy(dtype=float)
    neighbor_idx = build_knn_indices(coords, K_NEIGHBORS)
    rng = np.random.default_rng(20260526)
    rows = []
    variable_groups = {
        "absolute_weight": WEIGHT_NAMES,
        "relative_share": SHARE_NAMES,
        "complexity": ["sum_weight"],
    }
    for alpha in candidate_alphas:
        group = weights[weights["alpha"] == alpha].sort_values("station_index")
        for variable_group, variables in variable_groups.items():
            for variable in variables:
                values = group[variable].to_numpy(dtype=float)
                mask = np.isfinite(values)
                if mask.sum() != len(values):
                    local_neighbors = build_knn_indices(coords[mask], K_NEIGHBORS)
                    observed, p_value = morans_i_permutation(
                        values[mask], local_neighbors, PERMUTATIONS, rng
                    )
                    n_station = int(mask.sum())
                else:
                    observed, p_value = morans_i_permutation(values, neighbor_idx, PERMUTATIONS, rng)
                    n_station = int(len(values))
                rows.append(
                    {
                        "alpha": alpha,
                        "variable_group": variable_group,
                        "variable": variable,
                        "morans_i": observed,
                        "p_value": p_value,
                        "n_station": n_station,
                        "k_neighbors": K_NEIGHBORS,
                        "permutations": PERMUTATIONS,
                    }
                )
    return pd.DataFrame(rows)


def plot_corr_heatmaps(pairwise: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for ax, alpha in zip(axes.ravel(), FOCUS_ALPHAS):
        matrix = pd.DataFrame(np.eye(len(WEIGHT_NAMES)), index=WEIGHT_NAMES, columns=WEIGHT_NAMES)
        alpha_rows = pairwise[pairwise["alpha"] == alpha]
        for _, row in alpha_rows.iterrows():
            matrix.loc[row["weight_left"], row["weight_right"]] = row["pearson"]
            matrix.loc[row["weight_right"], row["weight_left"]] = row["pearson"]
        im = ax.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(f"alpha={format_alpha(alpha)}")
        ax.set_xticks(range(len(WEIGHT_NAMES)), WEIGHT_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(len(WEIGHT_NAMES)), WEIGHT_NAMES)
        for i in range(len(WEIGHT_NAMES)):
            for j in range(len(WEIGHT_NAMES)):
                ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="Pearson r")
    fig.suptitle("Weight Pairwise Correlation Heatmaps")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_share_distribution(weights: pd.DataFrame, output_path: Path) -> None:
    valid = weights[weights["sum_weight"] > 1e-6]
    alphas = FOCUS_ALPHAS
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey=True)
    for ax, share_name in zip(axes.ravel(), SHARE_NAMES):
        values = [valid.loc[valid["alpha"] == alpha, share_name].dropna().to_numpy() for alpha in alphas]
        ax.boxplot(values, tick_labels=[format_alpha(alpha) for alpha in alphas], showfliers=False)
        ax.set_title(share_name)
        ax.set_xlabel("alpha")
        ax.set_ylabel("relative share")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Relative Process Share Distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def scatter_map(
    ax: plt.Axes,
    frame: pd.DataFrame,
    value: str,
    title: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    sc = ax.scatter(
        frame["lon"],
        frame["lat"],
        c=frame[value],
        s=16,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
    )
    ax.set_title(title)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)


def plot_dominant_process_maps(weights: pd.DataFrame, locations: pd.DataFrame, candidate_alphas: list[float], output_path: Path) -> None:
    n_col = len(candidate_alphas)
    fig, axes = plt.subplots(1, n_col, figsize=(4.4 * n_col, 4.4), sharex=True, sharey=True)
    if n_col == 1:
        axes = [axes]
    color_map = {name: idx for idx, name in enumerate(PROCESS_COLORS)}
    colors = [PROCESS_COLORS[name] for name in color_map]
    cmap = matplotlib.colors.ListedColormap(colors)
    for ax, alpha in zip(axes, candidate_alphas):
        frame = ensure_locations(weights[weights["alpha"] == alpha], locations).copy()
        frame["dominant_code"] = frame["dominant_process"].map(color_map)
        ax.scatter(frame["lon"], frame["lat"], c=frame["dominant_code"], s=16, cmap=cmap, vmin=0, vmax=len(colors) - 1)
        ax.set_title(f"alpha={format_alpha(alpha)}")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_aspect("equal", adjustable="box")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=name, markerfacecolor=color, markersize=7)
        for name, color in PROCESS_COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles))
    fig.suptitle("Dominant Relative Process by Basin")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dominant_fraction(dominant: pd.DataFrame, output_path: Path) -> None:
    counts = (
        dominant.groupby(["alpha", "dominant_process"], sort=True)
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby("alpha")["count"].transform("sum")
    counts["fraction"] = counts["count"] / totals
    pivot = counts.pivot(index="alpha", columns="dominant_process", values="fraction").fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for process, color in PROCESS_COLORS.items():
        values = pivot[process].to_numpy() if process in pivot else np.zeros(len(pivot))
        ax.bar(x, values, bottom=bottom, label=process, color=color)
        bottom += values
    ax.set_xticks(x, [format_alpha(alpha) for alpha in pivot.index], rotation=45)
    ax.set_xlabel("alpha")
    ax.set_ylabel("fraction of basins")
    ax.set_title("Dominant Process Fraction by Alpha")
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(tradeoff: pd.DataFrame, y_col: str, output_path: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.plot(tradeoff["mean_sum_weight"], tradeoff[y_col], marker="o", linewidth=1.6)
    for _, row in tradeoff.iterrows():
        ax.text(row["mean_sum_weight"], row[y_col], format_alpha(row["alpha"]), fontsize=8)
    ax.set_xlabel("mean_sum_weight")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs Structural Complexity")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_delta_tradeoff(tradeoff: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.plot(
        tradeoff["complexity_reduction_vs_alpha0"],
        tradeoff["delta_median_kge_vs_alpha0"],
        marker="o",
        linewidth=1.6,
    )
    for _, row in tradeoff.iterrows():
        ax.text(
            row["complexity_reduction_vs_alpha0"],
            row["delta_median_kge_vs_alpha0"],
            format_alpha(row["alpha"]),
            fontsize=8,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("complexity reduction vs alpha=0")
    ax.set_ylabel("delta median KGE vs alpha=0")
    ax.set_title("Performance Loss vs Complexity Reduction")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_morans(morans: pd.DataFrame, variable_group: str, output_path: Path) -> None:
    group = morans[morans["variable_group"] == variable_group]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for variable, rows in group.groupby("variable", sort=True):
        rows = rows.sort_values("alpha")
        ax.plot(rows["alpha"], rows["morans_i"], marker="o", label=variable)
        significant = rows[rows["p_value"] < 0.05]
        ax.scatter(significant["alpha"], significant["morans_i"], s=80, facecolors="none", edgecolors="black")
    ax.set_xlabel("candidate alpha")
    ax.set_ylabel("Moran's I")
    ax.set_title(f"Spatial Autocorrelation: {variable_group}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_variable_maps(weights: pd.DataFrame, locations: pd.DataFrame, candidate_alphas: list[float], variables: list[str], output_path: Path, title: str, vmin: float, vmax: float) -> None:
    fig, axes = plt.subplots(len(variables), len(candidate_alphas), figsize=(4.2 * len(candidate_alphas), 3.5 * len(variables)), sharex=True, sharey=True)
    if len(variables) == 1:
        axes = np.array([axes])
    for row_idx, variable in enumerate(variables):
        for col_idx, alpha in enumerate(candidate_alphas):
            ax = axes[row_idx, col_idx]
            frame = ensure_locations(weights[weights["alpha"] == alpha], locations)
            scatter_map(ax, frame, variable, f"{variable}, alpha={format_alpha(alpha)}", vmin=vmin, vmax=vmax)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_delta_kge_distribution(delta: pd.DataFrame, output_path: Path) -> None:
    alphas = [alpha for alpha in sorted(delta["alpha"].unique()) if alpha > 0]
    values = [delta.loc[delta["alpha"] == alpha, "delta_kge"].dropna().to_numpy() for alpha in alphas]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.boxplot(values, tick_labels=[format_alpha(alpha) for alpha in alphas], showfliers=False)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("alpha")
    ax.set_ylabel("delta KGE vs alpha=0")
    ax.set_title("Basin-Level KGE Loss Distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_delta_kge_vs_complexity(delta: pd.DataFrame, candidate_alphas: list[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for alpha in candidate_alphas:
        group = delta[delta["alpha"] == alpha]
        ax.scatter(group["delta_sum_weight"], group["delta_kge"], s=12, alpha=0.45, label=f"alpha={format_alpha(alpha)}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("delta sum_weight vs alpha=0")
    ax.set_ylabel("delta KGE vs alpha=0")
    ax.set_title("Basin Performance Loss vs Complexity Compression")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_delta_kge_maps(delta: pd.DataFrame, locations: pd.DataFrame, candidate_alphas: list[float], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(candidate_alphas), figsize=(4.4 * len(candidate_alphas), 4.3), sharex=True, sharey=True)
    if len(candidate_alphas) == 1:
        axes = [axes]
    for ax, alpha in zip(axes, candidate_alphas):
        frame = ensure_locations(delta[delta["alpha"] == alpha], locations)
        scatter_map(ax, frame, "delta_kge", f"alpha={format_alpha(alpha)}", cmap="coolwarm", vmin=-0.5, vmax=0.2)
    fig.suptitle("Basin-Level Delta KGE Maps")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_quantile_path(saturation: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, sharey=True)
    quantiles = ["q05", "q25", "q50", "q75", "q95"]
    for ax, weight_name in zip(axes.ravel(), WEIGHT_NAMES):
        group = saturation[saturation["weight_name"] == weight_name].sort_values("alpha")
        for quantile in quantiles:
            ax.plot(group["alpha"], group[quantile], marker="o", linewidth=1.2, label=quantile)
        positive = group[group["alpha"] > 0]["alpha"]
        if len(positive) > 0:
            ax.set_xscale("log")
        ax.set_title(weight_name)
        ax.set_xlabel("alpha")
        ax.set_ylabel("weight")
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("Weight Quantile Paths")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_midrange_fraction(saturation: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for weight_name in WEIGHT_NAMES:
        group = saturation[saturation["weight_name"] == weight_name].sort_values("alpha")
        ax.plot(group["alpha"], group["frac_mid_0p10_0p90"], marker="o", label=weight_name)
    positive = saturation[saturation["alpha"] > 0]["alpha"]
    if len(positive) > 0:
        ax.set_xscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("fraction 0.1 <= w <= 0.9")
    ax.set_title("Midrange Weight Fraction by Alpha")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_specificity(pairwise: pd.DataFrame, sum_corr: pd.DataFrame, jaccard: pd.DataFrame) -> dict[str, float]:
    focus_pairwise = pairwise[pairwise["alpha"].isin(FOCUS_ALPHAS)]
    focus_sum = sum_corr[sum_corr["alpha"].isin(FOCUS_ALPHAS)]
    focus_jaccard = jaccard[(jaccard["alpha"].isin(FOCUS_ALPHAS)) & (jaccard["threshold"] == 0.5)]
    return {
        "mean_abs_pairwise_pearson": float(focus_pairwise["pearson"].abs().mean()),
        "max_abs_pairwise_pearson": float(focus_pairwise["pearson"].abs().max()),
        "mean_abs_sum_pearson": float(focus_sum["pearson_with_sum_weight"].abs().mean()),
        "mean_jaccard_0p5": float(focus_jaccard["jaccard"].mean()),
    }


def write_report(
    output_path: Path,
    pairwise: pd.DataFrame,
    sum_corr: pd.DataFrame,
    jaccard: pd.DataFrame,
    share_summary_df: pd.DataFrame,
    dominant: pd.DataFrame,
    tradeoff: pd.DataFrame,
    candidates: list[float],
    morans: pd.DataFrame,
    delta: pd.DataFrame,
    saturation: pd.DataFrame,
) -> None:
    specificity = summarize_specificity(pairwise, sum_corr, jaccard)
    candidate_text = ", ".join(format_alpha(alpha) for alpha in candidates)
    candidate_rows = tradeoff[tradeoff["alpha"].isin(candidates)].sort_values("alpha")
    significant_shares = morans[
        (morans["variable_group"] == "relative_share") & (morans["p_value"] < 0.05)
    ]
    significant_abs = morans[
        (morans["variable_group"] == "absolute_weight") & (morans["p_value"] < 0.05)
    ]
    high_midrange = saturation[
        (saturation["alpha"].isin(FOCUS_ALPHAS)) & (saturation["frac_mid_0p10_0p90"] > 0.25)
    ]

    lines = [
        "# Flex-MOPEX V1 Structure Learning Interpretation",
        "",
        "## Executive Diagnosis",
        "",
        f"- Candidate alphas selected for follow-up: {candidate_text}.",
        f"- Mean absolute pairwise Pearson correlation among process weights in focus alphas: {specificity['mean_abs_pairwise_pearson']:.3f}.",
        f"- Mean absolute correlation between each process weight and `sum_weight`: {specificity['mean_abs_sum_pearson']:.3f}.",
        f"- Mean active-set Jaccard similarity at threshold 0.5 in focus alphas: {specificity['mean_jaccard_0p5']:.3f}.",
        "",
        "## A. Structure-Complexity Regularization",
        "",
        "The V1 run clearly expresses structure-complexity regularization. `mean_sum_weight` decreases monotonically from alpha=0 to alpha=1, and alpha >= 0.1 collapses all four weights to near zero. This is a strong global complexity shrinkage signal.",
        "",
        "## B. Process-Specific Selection",
        "",
        "The evidence is mixed but not purely global. The process weights are not uniformly correlated across basins in the focus range. `w_int` often has weak or negative correlation with `sum_weight`, while `w_snow` and `w_sub` explain more of the remaining complexity at moderate alpha. `w_phen` and `w_int` become tightly coupled at alpha=0.03-0.05, which suggests one process-specific pattern may be a joint phenology/interception shutdown rather than four independent gates.",
        "",
        "## C. Recommended Alpha Values",
        "",
    ]
    for _, row in candidate_rows.iterrows():
        lines.append(
            f"- alpha={format_alpha(row['alpha'])}: median KGE delta={row['delta_median_kge_vs_alpha0']:.4f}, "
            f"median NSE delta={row['delta_median_nse_vs_alpha0']:.4f}, "
            f"complexity reduction={row['complexity_reduction_vs_alpha0']:.3f}, "
            f"mean_sum_weight={row['mean_sum_weight']:.3f}."
        )

    lines.extend(
        [
            "",
            "These alpha values preserve a usable performance-complexity tradeoff and avoid the alpha >= 0.1 collapse. They span conservative, balanced, and aggressive-but-not-collapsed compression.",
            "",
            "## D. Absolute Weight vs Relative Share",
            "",
            "Absolute weights are best for diagnosing complexity shrinkage. Relative shares are more useful for hydrologic interpretation because they separate total regularization strength from process preference. However, relative shares become unreliable when `sum_weight` is near zero; the report only computes shares for basins with `sum_weight > 1e-6`.",
            "",
            "## E. Spatial Structure",
            "",
        ]
    )
    if significant_abs.empty and significant_shares.empty:
        lines.append("- Moran's I did not find significant spatial autocorrelation for candidate-alpha absolute weights or shares at p < 0.05.")
    else:
        lines.append(f"- Significant absolute-weight Moran rows at p < 0.05: {len(significant_abs)}.")
        lines.append(f"- Significant relative-share Moran rows at p < 0.05: {len(significant_shares)}.")
        for _, row in pd.concat([significant_abs, significant_shares]).sort_values(["alpha", "variable_group", "variable"]).head(12).iterrows():
            lines.append(
                f"  - alpha={format_alpha(row['alpha'])} {row['variable']}: I={row['morans_i']:.3f}, p={row['p_value']:.3f}"
            )

    lines.extend(
        [
            "",
            "## F. Current Weakest Link",
            "",
            "The largest current gap is seed stability. This analysis uses one V1 training realization, so process-specific claims should be treated as provisional until repeated seeds show the same alpha path, dominant-process maps, and Moran's I patterns. Attribute control is the next priority: direct climate, snow, aridity, vegetation, and seasonality attributes are needed to test whether share_snow/share_int/share_phen patterns align with physical basin regimes. The model implementation appears diagnostically coherent because NPZ keys, shapes, value ranges, and time-constant static weights are internally consistent.",
            "",
            "## G. Next Experiment Priority",
            "",
            "1. Rerun V1 at candidate alpha values with at least 3 seeds and compare weight/share stability.",
            "2. Join CAMELS attributes and test whether relative shares are predictable from snow fraction, aridity, forest fraction, elevation, and seasonality after controlling for performance.",
            "3. Repeat this interpretation for dynamic V2/V3 weights using the same absolute-vs-share and Moran framework.",
            "4. Add explicit null controls, such as shuffled basin attributes or permuted station labels, to calibrate whether process maps are stronger than chance.",
            "",
            "## Supporting Findings",
            "",
            f"- Saturation rows with >25% midrange weights in focus alphas: {len(high_midrange)}.",
            f"- Median basin delta KGE for selected candidates: "
            + ", ".join(
                f"alpha={format_alpha(alpha)} {delta.loc[delta['alpha'] == alpha, 'delta_kge'].median():.4f}"
                for alpha in candidates
            )
            + ".",
            "",
            "## Output Files",
            "",
        ]
    )
    for file_name in sorted(path.name for path in OUTPUT_DIR.iterdir() if path.is_file() and path.name != output_path.name):
        lines.append(f"- `{file_name}`")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    locations = read_basin_locations()
    base_weights = load_weights()
    weights = add_shares(base_weights).merge(locations, on="station_index", how="left")
    complexity, metrics = load_complexity_and_metrics()
    metrics_by_basin = load_metrics_by_basin()

    pairwise = pairwise_correlations(weights)
    sum_corr = weight_sum_correlations(weights)
    jaccard = jaccard_similarity(weights)
    share_summary_df = share_summary(weights)
    dominant = dominant_process_table(weights)
    saturation = saturation_summary(weights)
    tradeoff = performance_complexity_tradeoff(complexity, metrics)
    candidates = select_candidate_alphas(tradeoff)
    morans = compute_morans(weights, locations, candidates)
    delta = basin_delta_metrics(weights, metrics_by_basin).merge(locations, on="station_index", how="left")

    pairwise.to_csv(OUTPUT_DIR / "weight_pairwise_corr_by_alpha.csv", index=False)
    sum_corr.to_csv(OUTPUT_DIR / "weight_sum_corr_by_alpha.csv", index=False)
    jaccard.to_csv(OUTPUT_DIR / "active_set_jaccard_by_alpha.csv", index=False)
    weights.to_csv(OUTPUT_DIR / "weights_with_sum_and_share.csv", index=False)
    share_summary_df.to_csv(OUTPUT_DIR / "share_alpha_summary.csv", index=False)
    dominant.to_csv(OUTPUT_DIR / "dominant_process_by_basin_alpha.csv", index=False)
    tradeoff.to_csv(OUTPUT_DIR / "performance_complexity_tradeoff.csv", index=False)
    morans.to_csv(OUTPUT_DIR / "morans_i_by_alpha.csv", index=False)
    delta.to_csv(OUTPUT_DIR / "basin_delta_metrics_by_alpha.csv", index=False)
    saturation.to_csv(OUTPUT_DIR / "weight_saturation_summary.csv", index=False)

    plot_corr_heatmaps(pairwise, OUTPUT_DIR / "fig_weight_corr_heatmaps_candidate_alpha.png")
    plot_share_distribution(weights, OUTPUT_DIR / "fig_share_distribution_by_alpha.png")
    plot_dominant_process_maps(weights, locations, candidates, OUTPUT_DIR / "fig_dominant_process_map_candidate_alpha.png")
    plot_dominant_fraction(dominant, OUTPUT_DIR / "fig_dominant_process_fraction_by_alpha.png")
    plot_tradeoff(tradeoff, "kge_median", OUTPUT_DIR / "fig_kge_vs_complexity.png", "median KGE")
    plot_tradeoff(tradeoff, "nse_median", OUTPUT_DIR / "fig_nse_vs_complexity.png", "median NSE")
    plot_delta_tradeoff(tradeoff, OUTPUT_DIR / "fig_delta_kge_vs_complexity_reduction.png")
    plot_morans(morans, "absolute_weight", OUTPUT_DIR / "fig_morans_i_absolute_weights.png")
    plot_morans(morans, "relative_share", OUTPUT_DIR / "fig_morans_i_relative_shares.png")
    plot_variable_maps(
        weights,
        locations,
        candidates,
        WEIGHT_NAMES,
        OUTPUT_DIR / "fig_weight_maps_candidate_alpha.png",
        "Absolute Weight Maps",
        0.0,
        1.0,
    )
    plot_variable_maps(
        weights,
        locations,
        candidates,
        SHARE_NAMES,
        OUTPUT_DIR / "fig_share_maps_candidate_alpha.png",
        "Relative Share Maps",
        0.0,
        1.0,
    )
    plot_delta_kge_distribution(delta, OUTPUT_DIR / "fig_delta_kge_distribution_by_alpha.png")
    plot_delta_kge_vs_complexity(delta, candidates, OUTPUT_DIR / "fig_delta_kge_vs_delta_complexity.png")
    plot_delta_kge_maps(delta, locations, candidates, OUTPUT_DIR / "fig_basin_map_delta_kge_candidate_alpha.png")
    plot_quantile_path(saturation, OUTPUT_DIR / "fig_weight_quantile_path.png")
    plot_midrange_fraction(saturation, OUTPUT_DIR / "fig_weight_midrange_fraction_by_alpha.png")

    write_report(
        OUTPUT_DIR / "structure_learning_v1_interpretation.md",
        pairwise,
        sum_corr,
        jaccard,
        share_summary_df,
        dominant,
        tradeoff,
        candidates,
        morans,
        delta,
        saturation,
    )

    print(f"Output directory: {OUTPUT_DIR}")
    print("Candidate alphas: " + ", ".join(format_alpha(alpha) for alpha in candidates))
    print(f"Pairwise correlation rows: {len(pairwise)}")
    print(f"Moran rows: {len(morans)}")
    print(f"Delta metric rows: {len(delta)}")
    print(f"Report: {OUTPUT_DIR / 'structure_learning_v1_interpretation.md'}")


if __name__ == "__main__":
    main()
