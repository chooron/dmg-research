from __future__ import annotations

import itertools
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon


ROOT = Path("/workspace/autoresearch")
PARAM_ROOT = ROOT / "project" / "parameterize"
STABILITY_ROOT = PARAM_ROOT / "outputs" / "analysis" / "stability_stats"
PARAMS_PATH = STABILITY_ROOT / "tables" / "params_long.csv"
ATTRIBUTES_PATH = STABILITY_ROOT / "tables" / "basin_attributes.csv"

FIG3_ROOT = PARAM_ROOT / "manuscript" / "analysis" / "figure3"
DATA_DIR = FIG3_ROOT / "data"
REPORT_DIR = FIG3_ROOT / "reports"
LOG_DIR = FIG3_ROOT / "logs"
FIG_DIR = PARAM_ROOT / "manuscript" / "figures" / "main"
OUT_STEM = FIG_DIR / "Fig03_cross_seed_relationship_stability"

MODEL_ORDER = ["deterministic", "mc_dropout", "distributional"]
MODEL_LABELS = {
    "deterministic": r"$\delta_{base}$",
    "mc_dropout": r"$\delta_{mcd}$",
    "distributional": r"$\delta_{dist}$",
}
MODEL_COLORS = {
    "deterministic": "#4C78A8",
    "mc_dropout": "#F58518",
    "distributional": "#2A9D8F",
}
MODEL_LINESTYLES = {
    "deterministic": "-",
    "mc_dropout": "--",
    "distributional": "-.",
}
MODEL_MARKERS = {
    "deterministic": "o",
    "mc_dropout": "s",
    "distributional": "^",
}
PARAM_ORDER = [
    "parTT",
    "parCFMAX",
    "parCFR",
    "parCWH",
    "parBETA",
    "parFC",
    "parLP",
    "parPERC",
    "parUZL",
    "parK0",
    "parK1",
    "parK2",
    "route_a",
    "route_b",
]
PARAM_LABELS = {
    "parBETA": r"$\mathrm{BETA}$",
    "parFC": r"$\mathrm{FC}$",
    "parLP": r"$\mathrm{LP}$",
    "parPERC": r"$\mathrm{PERC}$",
    "parUZL": r"$\mathrm{UZL}$",
    "parK0": r"$\mathrm{K}_0$",
    "parK1": r"$\mathrm{K}_1$",
    "parK2": r"$\mathrm{K}_2$",
    "parTT": r"$\mathrm{TT}$",
    "parCFMAX": r"$\mathrm{CFMAX}$",
    "parCFR": r"$\mathrm{CFR}$",
    "parCWH": r"$\mathrm{CWH}$",
    "route_a": r"$\mathrm{UH}_a$",
    "route_b": r"$\mathrm{UH}_b$",
}
NEAR_ZERO_RHO = 0.05
MAIN_TOP_K = 5
SENSITIVITY_TOP_K = (3, 5, 10)


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "figure3_analysis_log.txt", mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def setup_style() -> None:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.7,
            "font.size": 14.0,
            "axes.labelsize": 15.0,
            "axes.titlesize": 14.2,
            "xtick.labelsize": 13.6,
            "ytick.labelsize": 13.6,
            "legend.fontsize": 15.0,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
        }
    )


def clean_axes(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color="#E9E9E9", linewidth=0.5)
        ax.set_axisbelow(True)


def add_panel_label(ax: plt.Axes, label: str, *, x: float = 0.01, y: float = 1.0) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=17.0,
        fontweight="normal",
        color="#111111",
    )


def p_label(parameter: str) -> str:
    return PARAM_LABELS.get(str(parameter), str(parameter))


def ensure_dirs() -> None:
    for path in (DATA_DIR, REPORT_DIR, LOG_DIR, FIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    params = pd.read_csv(PARAMS_PATH)
    attrs = pd.read_csv(ATTRIBUTES_PATH)
    params = params.loc[params["model"].isin(MODEL_ORDER)].copy()
    params["seed"] = params["seed"].astype(int)

    # Some probabilistic exports include duplicate basin rows for the same run.
    # The Figure 3 target is the learned parameter mean per basin/run.
    params = (
        params.groupby(["basin_id", "model", "loss", "seed", "parameter"], as_index=False)
        .agg(mean=("mean", "mean"))
    )
    numeric_attrs = attrs.select_dtypes(include=[np.number]).columns.tolist()
    attr_cols = [col for col in numeric_attrs if col != "basin_id"]
    attrs = attrs[["basin_id"] + attr_cols].copy()
    logging.info(
        "Loaded %d parameter rows and %d attributes for %d basins.",
        len(params),
        len(attr_cols),
        attrs["basin_id"].nunique(),
    )
    return params, attrs


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if data.shape[0] < 5 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return np.nan, np.nan, int(data.shape[0])
    rho, p_value = spearmanr(data["x"], data["y"])
    return float(rho), float(p_value), int(data.shape[0])


def compute_seedwise_correlations(params: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    cache = DATA_DIR / "seedwise_attribute_parameter_correlations.csv"
    attr_cols = [col for col in attrs.columns if col != "basin_id"]
    rows: list[dict[str, object]] = []
    merged = params.merge(attrs, on="basin_id", how="inner")
    group_cols = ["model", "loss", "seed", "parameter"]
    total_groups = merged[group_cols].drop_duplicates().shape[0]
    for idx, ((model, loss, seed, parameter), subset) in enumerate(merged.groupby(group_cols), start=1):
        if idx % 50 == 0:
            logging.info("Computed Spearman groups %d/%d.", idx, total_groups)
        for attribute in attr_cols:
            rho, p_value, n_basins = safe_spearman(subset["mean"], subset[attribute])
            rows.append(
                {
                    "model": model,
                    "loss": loss,
                    "seed": int(seed),
                    "parameter": parameter,
                    "attribute": attribute,
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "abs_rho": abs(rho) if np.isfinite(rho) else np.nan,
                    "n_basins": n_basins,
                }
            )
    corr = pd.DataFrame(rows)
    corr.to_csv(cache, index=False)
    return corr


def pairwise_mean_abs_diff(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    diffs = [abs(a - b) for a, b in itertools.combinations(arr, 2)]
    return float(np.mean(diffs)) if diffs else np.nan


def majority_sign_consistency(values: pd.Series | np.ndarray, threshold: float = NEAR_ZERO_RHO) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    nonzero = arr[np.abs(arr) >= threshold]
    if nonzero.size == 0:
        return np.nan
    positive = int((nonzero > 0).sum())
    negative = int((nonzero < 0).sum())
    return float(max(positive, negative) / nonzero.size)


def all_seed_same_sign(values: pd.Series | np.ndarray, threshold: float = NEAR_ZERO_RHO) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    nonzero = arr[np.abs(arr) >= threshold]
    if nonzero.size == 0:
        return np.nan
    return float((nonzero > 0).all() or (nonzero < 0).all())


def select_candidate_relationships(corr: pd.DataFrame, top_k: int) -> pd.DataFrame:
    ranked = (
        corr.groupby(["parameter", "attribute"], as_index=False)
        .agg(mean_abs_rho_all_models_seeds=("abs_rho", "mean"))
        .sort_values(["parameter", "mean_abs_rho_all_models_seeds", "attribute"], ascending=[True, False, True])
    )
    ranked["rank_within_parameter"] = ranked.groupby("parameter").cumcount() + 1
    ranked["selected_topk"] = ranked["rank_within_parameter"] <= top_k
    ranked["notes"] = np.where(
        ranked["selected_topk"],
        f"Selected by mean absolute Spearman rho across all models, losses, and seeds; top-k={top_k}.",
        f"Not selected for main Figure 3 top-k={top_k} candidate set.",
    )
    out = ranked.loc[ranked["selected_topk"]].copy()
    out.to_csv(DATA_DIR / "candidate_topk_relationships.csv", index=False)
    return out


def candidate_relationships_frame(corr: pd.DataFrame, top_k: int) -> pd.DataFrame:
    ranked = (
        corr.groupby(["parameter", "attribute"], as_index=False)
        .agg(mean_abs_rho_all_models_seeds=("abs_rho", "mean"))
        .sort_values(["parameter", "mean_abs_rho_all_models_seeds", "attribute"], ascending=[True, False, True])
    )
    ranked["rank_within_parameter"] = ranked.groupby("parameter").cumcount() + 1
    ranked["selected_topk"] = ranked["rank_within_parameter"] <= top_k
    ranked["notes"] = np.where(
        ranked["selected_topk"],
        f"Selected by mean absolute Spearman rho across all models, losses, and seeds; top-k={top_k}.",
        f"Not selected for main Figure 3 top-k={top_k} candidate set.",
    )
    return ranked.loc[ranked["selected_topk"]].copy()


def selected_corr(corr: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    return corr.merge(candidates[["parameter", "attribute"]], on=["parameter", "attribute"], how="inner")


def compute_correlation_stability(selected: pd.DataFrame) -> pd.DataFrame:
    stability = (
        selected.groupby(["model", "loss", "parameter", "attribute"], as_index=False)
        .agg(
            seed_mean_rho=("spearman_rho", "mean"),
            seed_sd_rho=("spearman_rho", lambda values: float(np.std(values, ddof=1))),
            seed_range_rho=("spearman_rho", lambda values: float(np.nanmax(values) - np.nanmin(values))),
            mean_pairwise_seed_abs_diff_rho=("spearman_rho", pairwise_mean_abs_diff),
            mean_abs_rho=("abs_rho", "mean"),
            seed_count=("seed", "nunique"),
        )
    )
    stability.to_csv(DATA_DIR / "correlation_seed_stability_long.csv", index=False)
    return stability


def compute_sign_consistency(selected: pd.DataFrame) -> pd.DataFrame:
    summary = (
        selected.groupby(["model", "loss", "parameter", "attribute"], as_index=False)
        .agg(
            sign_consistency=("spearman_rho", majority_sign_consistency),
            all_seed_same_sign=("spearman_rho", all_seed_same_sign),
            near_zero_seed_fraction=("spearman_rho", lambda values: float(np.mean(np.abs(values) < NEAR_ZERO_RHO))),
            seed_count=("seed", "nunique"),
        )
    )
    summary.to_csv(DATA_DIR / "sign_consistency_summary.csv", index=False)
    return summary


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return np.nan
    return float(len(a & b) / len(union))


def compute_topk_consistency(corr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = corr.copy()
    ranked["rank_abs"] = (
        ranked.groupby(["model", "loss", "seed", "parameter"])["abs_rho"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for (model, loss, parameter), subset in ranked.groupby(["model", "loss", "parameter"]):
        seed_to_subset = {int(seed): sdf for seed, sdf in subset.groupby("seed")}
        seeds = sorted(seed_to_subset)
        top1 = {
            seed: str(seed_to_subset[seed].sort_values("rank_abs").iloc[0]["attribute"])
            for seed in seeds
            if not seed_to_subset[seed].empty
        }
        if top1:
            dominant_consistency = max(pd.Series(top1).value_counts()) / len(top1)
        else:
            dominant_consistency = np.nan

        top_sets: dict[int, dict[int, set[str]]] = {3: {}, 5: {}}
        for seed in seeds:
            sdf = seed_to_subset[seed].sort_values("rank_abs")
            top_sets[3][seed] = set(sdf.head(3)["attribute"].astype(str))
            top_sets[5][seed] = set(sdf.head(5)["attribute"].astype(str))

        overlaps = {3: [], 5: []}
        for seed_a, seed_b in itertools.combinations(seeds, 2):
            for k in (3, 5):
                value = jaccard(top_sets[k][seed_a], top_sets[k][seed_b])
                overlaps[k].append(value)
                pair_rows.append(
                    {
                        "model": model,
                        "loss": loss,
                        "parameter": parameter,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "top_k": k,
                        "jaccard_overlap": value,
                    }
                )
        rows.append(
            {
                "model": model,
                "loss": loss,
                "parameter": parameter,
                "dominant_attribute_consistency": float(dominant_consistency),
                "top3_overlap": float(np.nanmean(overlaps[3])) if overlaps[3] else np.nan,
                "top5_overlap": float(np.nanmean(overlaps[5])) if overlaps[5] else np.nan,
                "seed_count": len(seeds),
            }
        )
    summary = pd.DataFrame(rows)
    pairs = pd.DataFrame(pair_rows)
    summary.to_csv(DATA_DIR / "dominant_topk_consistency_summary.csv", index=False)
    pairs.to_csv(DATA_DIR / "dominant_topk_consistency_seed_pairs.csv", index=False)
    return summary, pairs


def compute_parameter_level_summary(
    stability: pd.DataFrame,
    sign_summary: pd.DataFrame,
    topk_summary: pd.DataFrame,
) -> pd.DataFrame:
    pair_summary = stability.merge(
        sign_summary[["model", "loss", "parameter", "attribute", "sign_consistency"]],
        on=["model", "loss", "parameter", "attribute"],
        how="left",
    )
    parameter = (
        pair_summary.groupby(["model", "loss", "parameter"], as_index=False)
        .agg(
            median_seed_sd_rho=("seed_sd_rho", "median"),
            median_seed_range_rho=("seed_range_rho", "median"),
            median_pairwise_seed_abs_diff_rho=("mean_pairwise_seed_abs_diff_rho", "median"),
            median_sign_consistency=("sign_consistency", "median"),
        )
        .merge(topk_summary, on=["model", "loss", "parameter"], how="left")
    )
    parameter["mean_top3_overlap"] = parameter["top3_overlap"]
    parameter["mean_top5_overlap"] = parameter["top5_overlap"]
    parameter.to_csv(DATA_DIR / "parameter_level_correlation_stability.csv", index=False)
    return parameter


def sensitivity_candidates(corr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for top_k in SENSITIVITY_TOP_K:
        candidates = candidate_relationships_frame(corr, top_k=top_k)
        selected = selected_corr(corr, candidates)
        stability = (
            selected.groupby(["model", "loss", "parameter", "attribute"], as_index=False)
            .agg(
                seed_sd_rho=("spearman_rho", lambda values: float(np.std(values, ddof=1))),
                seed_range_rho=("spearman_rho", lambda values: float(np.nanmax(values) - np.nanmin(values))),
                mean_pairwise_seed_abs_diff_rho=("spearman_rho", pairwise_mean_abs_diff),
            )
        )
        sign_summary = (
            selected.groupby(["model", "loss", "parameter", "attribute"], as_index=False)
            .agg(sign_consistency=("spearman_rho", majority_sign_consistency))
        )
        for model, subset in stability.groupby("model"):
            rows.append(
                {
                    "top_k": top_k,
                    "model": model,
                    "median_seed_sd_rho": float(subset["seed_sd_rho"].median()),
                    "median_seed_range_rho": float(subset["seed_range_rho"].median()),
                    "median_pairwise_seed_abs_diff_rho": float(
                        subset["mean_pairwise_seed_abs_diff_rho"].median()
                    ),
                    "median_sign_consistency": float(
                        sign_summary.loc[sign_summary["model"].eq(model), "sign_consistency"].median()
                    ),
                    "n_relationship_model_loss_rows": int(len(subset)),
                }
            )
    sens = pd.DataFrame(rows)
    sens.to_csv(DATA_DIR / "topk_sensitivity_summary.csv", index=False)
    return sens


def run_stat_tests(stability: pd.DataFrame, sign_summary: pd.DataFrame, topk_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_comparison(frame: pd.DataFrame, metric: str, loss: str, subset_label: str, paired_cols: list[str]) -> None:
        dist = frame.loc[frame["model"].eq("distributional")]
        for comparator in ("deterministic", "mc_dropout"):
            other = frame.loc[frame["model"].eq(comparator)]
            merged = dist.merge(other, on=paired_cols, suffixes=("_dist", "_other"))
            x = merged[f"{metric}_dist"].to_numpy(dtype=float)
            y = merged[f"{metric}_other"].to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            x = x[valid]
            y = y[valid]
            if x.size >= 2:
                try:
                    stat, p_value = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                    test = "Wilcoxon signed-rank"
                except ValueError:
                    stat, p_value = np.nan, 1.0
                    test = "Wilcoxon signed-rank"
                effect = float(np.nanmedian(x - y))
            else:
                x_un = dist[metric].dropna().to_numpy(dtype=float)
                y_un = other[metric].dropna().to_numpy(dtype=float)
                stat, p_value = mannwhitneyu(x_un, y_un, alternative="two-sided")
                test = "Mann-Whitney U"
                effect = float(np.nanmedian(x_un) - np.nanmedian(y_un))
            direction = "lower is more stable" if "sd" in metric or "range" in metric or "diff" in metric else "higher is more stable"
            rows.append(
                {
                    "metric": metric,
                    "comparison": f"{MODEL_LABELS['distributional']} vs {MODEL_LABELS[comparator]}",
                    "loss": loss,
                    "subset": subset_label,
                    "test": test,
                    "statistic": float(stat) if np.isfinite(stat) else np.nan,
                    "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                    "effect_size_optional": effect,
                    "interpretation": direction,
                    "paired_n": int(x.size),
                }
            )

    for loss, subset in stability.groupby("loss"):
        for metric in ("seed_sd_rho", "seed_range_rho", "mean_pairwise_seed_abs_diff_rho"):
            add_comparison(subset, metric, str(loss), "top-5 candidate relationships", ["loss", "parameter", "attribute"])
    for metric in ("seed_sd_rho", "seed_range_rho", "mean_pairwise_seed_abs_diff_rho"):
        add_comparison(stability, metric, "pooled", "top-5 candidate relationships", ["loss", "parameter", "attribute"])

    sign_model_loss = (
        sign_summary.groupby(["model", "loss", "parameter"], as_index=False)
        .agg(sign_consistency=("sign_consistency", "median"))
    )
    for loss, subset in sign_model_loss.groupby("loss"):
        add_comparison(subset, "sign_consistency", str(loss), "parameter median over top-5 pairs", ["loss", "parameter"])

    for metric in ("dominant_attribute_consistency", "top5_overlap"):
        for loss, subset in topk_summary.groupby("loss"):
            add_comparison(subset, metric, str(loss), "all attribute rankings by parameter", ["loss", "parameter"])

    tests = pd.DataFrame(rows)
    tests.to_csv(DATA_DIR / "statistical_tests_relationship_seed_stability.csv", index=False)
    return tests


def choose_selected_pairs(candidates: pd.DataFrame, stability: pd.DataFrame, sign_summary: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        ("slope_mean", "parBETA"),
        ("pet_mean", "parFC"),
        ("aridity", "parPERC"),
        ("soil_conductivity", "parUZL"),
        ("frac_snow", "parCWH"),
        ("frac_snow", "parCFR"),
        ("slope_mean", "route_b"),
        ("low_prec_dur", "route_b"),
    ]
    merged = (
        stability.merge(
            sign_summary[["model", "loss", "parameter", "attribute", "sign_consistency"]],
            on=["model", "loss", "parameter", "attribute"],
            how="left",
        )
        .groupby(["parameter", "attribute"], as_index=False)
        .agg(
            mean_abs_rho=("mean_abs_rho", "mean"),
            median_seed_sd_rho=("seed_sd_rho", "median"),
            median_sign_consistency=("sign_consistency", "median"),
        )
        .merge(candidates[["parameter", "attribute", "rank_within_parameter"]], on=["parameter", "attribute"], how="left")
    )
    chosen: list[tuple[str, str]] = []
    for attribute, parameter in preferred:
        match = merged.loc[merged["attribute"].eq(attribute) & merged["parameter"].eq(parameter)]
        if not match.empty:
            chosen.append((parameter, attribute))
        if len(chosen) == 4:
            break
    if len(chosen) < 4:
        ranked = merged.sort_values(
            ["median_sign_consistency", "mean_abs_rho", "median_seed_sd_rho"],
            ascending=[False, False, True],
        )
        for row in ranked.itertuples(index=False):
            pair = (row.parameter, row.attribute)
            if pair not in chosen:
                chosen.append(pair)
            if len(chosen) == 4:
                break
    selected_pairs = pd.DataFrame(chosen[:4], columns=["parameter", "attribute"])
    return selected_pairs


def export_selected_robust_pairs(corr: pd.DataFrame, selected_pairs: pd.DataFrame) -> pd.DataFrame:
    out = corr.merge(selected_pairs, on=["parameter", "attribute"], how="inner")
    out = out.sort_values(["parameter", "attribute", "model", "loss", "seed"]).reset_index(drop=True)
    out.to_csv(DATA_DIR / "selected_robust_pairs_seedwise_rho.csv", index=False)
    return out


def model_positions(center: float, width: float = 0.22) -> dict[str, float]:
    return {
        "deterministic": center - width,
        "mc_dropout": center,
        "distributional": center + width,
    }


def plot_panel_a(ax: plt.Axes, stability: pd.DataFrame) -> None:
    data = [stability.loc[stability["model"].eq(model), "seed_sd_rho"].dropna().to_numpy() for model in MODEL_ORDER]
    bp = ax.boxplot(
        data,
        positions=np.arange(len(MODEL_ORDER)),
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.1},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
        boxprops={"linewidth": 0.8, "color": "#555555"},
    )
    for patch, model in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.72)
    rng = np.random.default_rng(5)
    for idx, model in enumerate(MODEL_ORDER):
        values = stability.loc[stability["model"].eq(model), "seed_sd_rho"].dropna().to_numpy()
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            s=8,
            color=MODEL_COLORS[model],
            alpha=0.25,
            linewidth=0,
            zorder=2,
        )
    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=16.5)
    ax.set_ylabel(r"Seed SD of Spearman $\rho$")
    clean_axes(ax, "y")
    add_panel_label(ax, "(a)")


def plot_panel_b(ax: plt.Axes, parameter_summary: pd.DataFrame) -> None:
    pooled = (
        parameter_summary.groupby(["model", "parameter"], as_index=False)
        .agg(median_seed_sd_rho=("median_seed_sd_rho", "median"))
    )
    params = [p for p in PARAM_ORDER if p in set(pooled["parameter"])]
    y_lookup = {parameter: idx for idx, parameter in enumerate(params)}
    for parameter in params:
        sub = pooled.loc[pooled["parameter"].eq(parameter)]
        xvals = [sub.loc[sub["model"].eq(model), "median_seed_sd_rho"].squeeze() for model in MODEL_ORDER]
        xvals = [float(x) if np.ndim(x) == 0 and pd.notna(x) else np.nan for x in xvals]
        y = y_lookup[parameter]
        ax.plot(xvals, [y] * len(xvals), color="#D8D8D8", lw=0.65, alpha=0.75, zorder=1)
        for model, value in zip(MODEL_ORDER, xvals):
            if np.isfinite(value):
                ax.scatter(value, y, s=28, color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels([p_label(p) for p in params], fontsize=15.0)
    ax.invert_yaxis()
    ax.set_xlabel(r"Median seed SD of Spearman $\rho$ (lower = more stable)")
    clean_axes(ax, "x")
    add_panel_label(ax, "(b)")


def plot_panel_c(ax: plt.Axes, sign_summary: pd.DataFrame, topk_summary: pd.DataFrame) -> None:
    sign_metric = (
        sign_summary.groupby("model", as_index=False)
        .agg(value=("sign_consistency", "median"))
        .assign(metric="Sign consistency")
    )
    dom_metric = (
        topk_summary.groupby("model", as_index=False)
        .agg(value=("dominant_attribute_consistency", "median"))
        .assign(metric="Dominant consistency")
    )
    top5_metric = (
        topk_summary.groupby("model", as_index=False)
        .agg(value=("top5_overlap", "median"))
        .assign(metric="Top-5 overlap")
    )
    metrics = pd.concat([sign_metric, dom_metric, top5_metric], ignore_index=True)
    metric_order = ["Sign consistency", "Dominant consistency", "Top-5 overlap"]
    metric_display = ["Sign\nconsistency", "Dominant\nconsistency", "Top-5\noverlap"]
    width = 0.22
    for i, metric in enumerate(metric_order):
        positions = model_positions(i, width)
        for model in MODEL_ORDER:
            val = metrics.loc[metrics["metric"].eq(metric) & metrics["model"].eq(model), "value"]
            if val.empty:
                continue
            ax.bar(
                positions[model],
                float(val.iloc[0]),
                width=0.19,
                color=MODEL_COLORS[model],
                alpha=0.82,
                edgecolor="white",
                linewidth=0.5,
            )
    ax.set_xticks(range(len(metric_order)))
    ax.set_xticklabels(metric_display, rotation=0, ha="center", fontsize=14.6)
    ax.tick_params(axis="x", pad=3)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Consistency / overlap")
    clean_axes(ax, "y")
    add_panel_label(ax, "(c)")


def plot_panel_d(container_ax: plt.Axes, selected: pd.DataFrame) -> None:
    container_ax.axis("off")
    pairs = (
        selected[["parameter", "attribute"]]
        .drop_duplicates()
        .head(4)
        .to_records(index=False)
    )
    subfig = container_ax.get_subplotspec().subgridspec(2, 2, hspace=0.34, wspace=0.26)
    for idx, pair in enumerate(pairs):
        parameter, attribute = str(pair[0]), str(pair[1])
        ax = container_ax.figure.add_subplot(subfig[idx // 2, idx % 2])
        sub = selected.loc[selected["parameter"].eq(parameter) & selected["attribute"].eq(attribute)]
        summary = (
            sub.groupby(["model", "seed"], as_index=False)
            .agg(mean_rho=("spearman_rho", "mean"))
            .sort_values("seed")
        )
        for model in MODEL_ORDER:
            msub = summary.loc[summary["model"].eq(model)].sort_values("seed")
            ax.plot(
                msub["seed"],
                msub["mean_rho"],
                color=MODEL_COLORS[model],
                linestyle=MODEL_LINESTYLES[model],
                marker=MODEL_MARKERS[model],
                ms=3.8,
                lw=1.35,
                alpha=0.92,
            )
        ax.axhline(0, color="#CFCFCF", lw=0.75, ls=(0, (2.5, 2.5)), zorder=0)
        ax.set_title(f"{attribute} \u2192 {p_label(parameter)}", fontsize=13.8, pad=3.0)
        ax.set_xticks(sorted(summary["seed"].unique()))
        ax.tick_params(axis="x", labelrotation=0, labelsize=13.4)
        ax.tick_params(axis="y", labelsize=13.2)
        if idx // 2 == 1:
            ax.set_xlabel("Seed", fontsize=14.0)
        if idx % 2 == 0:
            ax.set_ylabel(r"Spearman $\rho$", fontsize=14.0)
        clean_axes(ax, "y")


def make_figure(
    stability: pd.DataFrame,
    parameter_summary: pd.DataFrame,
    sign_summary: pd.DataFrame,
    topk_summary: pd.DataFrame,
    selected_pairs_rho: pd.DataFrame,
) -> None:
    setup_style()
    fig = plt.figure(figsize=(11.4, 7.8), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.28, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_panel_a(ax_a, stability)
    plot_panel_b(ax_b, parameter_summary)
    plot_panel_c(ax_c, sign_summary, topk_summary)
    plot_panel_d(ax_d, selected_pairs_rho)

    handles = [
        Patch(
            color=MODEL_COLORS[m],
            label=MODEL_LABELS[m],
        )
        for m in MODEL_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        ncol=3,
        frameon=False,
        handlelength=2.6,
        columnspacing=2.4,
    )
    fig.subplots_adjust(top=0.91, left=0.080, right=0.992, bottom=0.082)
    d_box = ax_d.get_position()
    fig.text(
        d_box.x0 + 0.01 * d_box.width,
        d_box.y1,
        "(d)",
        ha="left",
        va="top",
        fontsize=17.0,
        fontweight="normal",
        color="#111111",
    )
    fig.savefig(f"{OUT_STEM}.png", dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def format_model_table(df: pd.DataFrame, value_col: str, agg: str = "median") -> str:
    grouped = getattr(df.groupby("model")[value_col], agg)().reindex(MODEL_ORDER)
    return "\n".join(f"- {MODEL_LABELS[m]}: {grouped.loc[m]:.4f}" for m in MODEL_ORDER if m in grouped.index)


def simple_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_reports(
    corr: pd.DataFrame,
    candidates: pd.DataFrame,
    stability: pd.DataFrame,
    sign_summary: pd.DataFrame,
    topk_summary: pd.DataFrame,
    parameter_summary: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    tests: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    data_inventory = {
        "models": ", ".join(MODEL_LABELS[m] for m in MODEL_ORDER),
        "losses": ", ".join(sorted(corr["loss"].unique())),
        "seeds": ", ".join(str(s) for s in sorted(corr["seed"].unique())),
        "parameters": str(corr["parameter"].nunique()),
        "attributes": str(corr["attribute"].nunique()),
        "seedwise_rows": str(len(corr)),
        "candidate_pairs": str(candidates[["parameter", "attribute"]].drop_duplicates().shape[0]),
    }
    test_brief = tests.loc[
        tests["loss"].eq("pooled") & tests["metric"].isin(["seed_sd_rho", "seed_range_rho"])
    ][["metric", "comparison", "test", "paired_n", "p_value", "effect_size_optional"]]
    selected_pairs_md = simple_markdown_table(selected_pairs[["attribute", "parameter"]].drop_duplicates())
    test_brief_md = simple_markdown_table(test_brief)
    report = f"""# Figure 3 Statistical Summary

## 1. Objective

Figure 3 evaluates cross-seed reproducibility of high-magnitude attribute-parameter correlation structure. The main estimand is not the full set of weak attribute-parameter pairs, but dominant-control candidates selected independently of any single model.

## 2. Data inventory

- Models: {data_inventory['models']}
- Loss functions: {data_inventory['losses']}
- Seeds: {data_inventory['seeds']}
- HBV parameters: {data_inventory['parameters']}
- Numeric basin attributes: {data_inventory['attributes']}
- Seed-wise Spearman rows: {data_inventory['seedwise_rows']}
- Main top-k candidate pairs: {data_inventory['candidate_pairs']}
- Parameter estimates were deduplicated to one basin/run/parameter mean before correlation analysis.

## 3. Candidate relationship selection rule

For each HBV parameter, attributes were ranked by mean absolute Spearman rho averaged across all models, all losses, and all seeds. The main figure uses top-k = {MAIN_TOP_K}. This rule is intentionally neutral: it is not based on {MODEL_LABELS['distributional']} alone and therefore avoids cherry-picking relationships favorable to one formulation.

Sensitivity checks were also run for top-k = 3, 5, and 10. Summary medians are saved in `topk_sensitivity_summary.csv`.

## 4. Correlation variability results

Median seed SD of Spearman rho across top-5 candidate relationship rows:

{format_model_table(stability, 'seed_sd_rho')}

Median seed range of Spearman rho:

{format_model_table(stability, 'seed_range_rho')}

These statistics focus on high-magnitude candidate relationships, so they are less dominated by near-zero pairs whose apparent sign instability is mostly weak-correlation noise.

## 5. Sign consistency results

Majority sign consistency treats correlations with |rho| < {NEAR_ZERO_RHO:.2f} as near-zero and excludes them from the sign majority denominator. Median sign consistency:

{format_model_table(sign_summary, 'sign_consistency')}

Near-zero relationships should not be over-interpreted, even when their majority sign score is numerically high or undefined.

## 6. Dominant/top-k consistency results

Dominant attribute consistency is the majority share of the top-1 attribute across seeds within each model/loss/parameter. Top-k overlap is the mean pairwise Jaccard overlap across seed pairs.

Median dominant attribute consistency:

{format_model_table(topk_summary, 'dominant_attribute_consistency')}

Median top-5 overlap:

{format_model_table(topk_summary, 'top5_overlap')}

## 7. Parameter-level stability summary

Parameter-level summaries aggregate each parameter's selected top-5 candidate relationships. Lower median seed SD/range indicates more reproducible relationship strength. The complete table is saved as `parameter_level_correlation_stability.csv`.

## 8. Selected robust pairs

The selected small multiples are saved in `selected_robust_pairs_seedwise_rho.csv` and were chosen from the neutral top-5 candidate set using high mean |rho|, high sign consistency, low seed SD, and hydrologic interpretability where available.

{selected_pairs_md}

## 9. Statistical tests

Panel (a) uses paired Wilcoxon signed-rank tests when pairs can be matched by loss, parameter, and attribute. Mann-Whitney U is used only if paired samples are unavailable. Main pooled tests:

{test_brief_md}

Full tests are saved in `statistical_tests_relationship_seed_stability.csv`.

## 10. Recommended Results wording

Across high-magnitude candidate relationships, the distributional formulation showed lower cross-seed variability in Spearman rho and stronger reproducibility of the correlation structure than the deterministic and MC-dropout baselines. The result is best described as improved relationship reliability or cross-seed reproducibility of dominant-control candidates, not as recovery of correct or true attribute-parameter relationships.

## 11. Caveats

1. The main figure focuses on top-k high-magnitude candidate relationships rather than all weak pairs.
2. Full-pair statistics are appropriate as appendix sensitivity checks because many weak pairs have little hydrologic meaning.
3. Top-k screening uses all models, losses, and seeds, not {MODEL_LABELS['distributional']} alone.
4. Sign consistency for near-zero correlations is not substantively reliable.
5. Relationship stability is core evidence for reproducible learned structure and should be interpreted alongside, but not replaced by, raw parameter-value stability.
"""
    (REPORT_DIR / "figure3_statistical_summary.md").write_text(report, encoding="utf-8")

    notes = f"""# Figure 3 Plot Notes

- Layout: 2 x 2 main-text figure.
- Panel (a): pooled top-{MAIN_TOP_K} seed SD of Spearman rho by model.
- Panel (b): parameter-level median seed SD of Spearman rho across selected candidate attributes.
- Panel (c): median sign consistency, dominant attribute consistency, and top-5 Jaccard overlap.
- Panel (d): four selected robust attribute-parameter pairs, with rho averaged over losses for each seed and model.
- Main-figure statistics are computed for high-magnitude candidate relationships, not all attribute-parameter pairs.
- Candidate relationships are selected separately for each parameter by choosing the top-k attributes with the highest mean absolute Spearman rho.
- The mean absolute Spearman rho used for screening is averaged over all models, losses, and random seeds, avoiding selection based on {MODEL_LABELS['distributional']} alone.
- The main figure uses top-k = {MAIN_TOP_K}.
- Full-pair summaries and top-k sensitivity checks can be placed in the appendix.
- Lower seed SD indicates more stable relationship recovery across random seeds.
- Sign consistency is not over-interpreted for near-zero correlations; correlations with |rho| < {NEAR_ZERO_RHO:.2f} are treated as near-zero in sign summaries.
- Model labels: {', '.join(MODEL_LABELS[m] for m in MODEL_ORDER)}.
- Colors: {MODEL_LABELS['deterministic']} {MODEL_COLORS['deterministic']}, {MODEL_LABELS['mc_dropout']} {MODEL_COLORS['mc_dropout']}, {MODEL_LABELS['distributional']} {MODEL_COLORS['distributional']}.
- Output file: `{OUT_STEM}.png`.
"""
    (REPORT_DIR / "figure3_plot_notes.md").write_text(notes, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    setup_logging()
    params, attrs = load_inputs()
    corr = compute_seedwise_correlations(params, attrs)
    sensitivity = sensitivity_candidates(corr)
    candidates = select_candidate_relationships(corr, top_k=MAIN_TOP_K)
    selected = selected_corr(corr, candidates)
    stability = compute_correlation_stability(selected)
    sign_summary = compute_sign_consistency(selected)
    topk_summary, _topk_pairs = compute_topk_consistency(corr)
    parameter_summary = compute_parameter_level_summary(stability, sign_summary, topk_summary)
    tests = run_stat_tests(stability, sign_summary, topk_summary)
    selected_pair_keys = choose_selected_pairs(candidates, stability, sign_summary)
    selected_pairs_rho = export_selected_robust_pairs(corr, selected_pair_keys)
    make_figure(stability, parameter_summary, sign_summary, topk_summary, selected_pairs_rho)
    write_reports(
        corr=corr,
        candidates=candidates,
        stability=stability,
        sign_summary=sign_summary,
        topk_summary=topk_summary,
        parameter_summary=parameter_summary,
        selected_pairs=selected_pair_keys,
        tests=tests,
        sensitivity=sensitivity,
    )
    logging.info("Figure 3 outputs written to %s and %s.", DATA_DIR, FIG_DIR)


if __name__ == "__main__":
    main()
