from __future__ import annotations

import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from dmg.core.data.loaders import HydroLoader
from dmg.core.utils.utils import initialize_config
from omegaconf import OmegaConf
from scipy.stats import mannwhitneyu, wilcoxon
from torch import nn

ROOT = Path("/workspace/autoresearch")
PARAM_ROOT = ROOT / "project" / "parameterize"
OUTPUT_ROOT = PARAM_ROOT / "outputs"
FIGURE_ROOT = PARAM_ROOT / "manuscript" / "analysis" / "figure2"
DATA_DIR = FIGURE_ROOT / "data"
REPORT_DIR = FIGURE_ROOT / "reports"
LOG_DIR = FIGURE_ROOT / "logs"
CONFIG_PATH = PARAM_ROOT / "conf" / "config_param_paper.yaml"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PARAM_ROOT))
sys.path.insert(0, str(ROOT / "project" / "Invariant"))

from project.parameterize.implements.hbv_static import HbvStatic  # noqa: E402
from project.parameterize.paper_variants import build_paper_dpl, normalize_paper_config  # noqa: E402
from project.parameterize.train_dmotpy import (  # noqa: E402
    _build_loader_config,
    _normalize_runtime_paths,
    _resolve_path,
)
from project.parameterize.implements.basin_utils import (  # noqa: E402
    basin_subset_indices,
    load_basin_ids,
)


MODEL_ORDER = ["deterministic", "mc_dropout", "distributional"]
MODEL_LABEL = {
    "deterministic": "delta_base",
    "mc_dropout": "delta_mcd",
    "distributional": "delta_dist",
}
MODEL_LATEX = {
    "deterministic": r"$\delta_{base}$",
    "mc_dropout": r"$\delta_{mcd}$",
    "distributional": r"$\delta_{dist}$",
}
LOSS_ORDER = ["HybridNseBatchLoss", "LogNseBatchLoss", "NseBatchLoss"]
PARAMETER_SPECS = (
    list(HbvStatic.parameter_bounds.items())
    + list(HbvStatic.routing_parameter_bounds.items())
)
PARAMETER_NAMES = [name for name, _ in PARAMETER_SPECS]
PARAMETER_GROUP = {
    "parTT": "snow",
    "parCFMAX": "snow",
    "parCFR": "snow",
    "parCWH": "snow",
    "route_a": "routing",
    "route_b": "routing",
}
N_STOCHASTIC_SAMPLES = 100
EPS = 1e-12


@dataclass(frozen=True)
class RunSpec:
    model_raw: str
    loss: str
    seed: int
    run_dir: Path
    checkpoint_path: Path | None
    run_meta_path: Path | None
    source_family: str
    status: str


def iqr(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.quantile(0.75) - values.quantile(0.25))


def q25(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").quantile(0.25))


def q75(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").quantile(0.75))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_runtime_config(variant: str, seed: int, loss: str, device: str = "cpu") -> dict[str, Any]:
    raw_config = OmegaConf.load(_resolve_path(str(CONFIG_PATH)))
    raw_config["mode"] = "test"
    raw_config["seed"] = int(seed)
    raw_config["device"] = device
    raw_config["gpu_id"] = 0
    raw_config.setdefault("paper", {})
    raw_config["paper"]["variant"] = variant
    raw_config.setdefault("train", {}).setdefault("loss_function", {})
    raw_config["train"]["loss_function"]["name"] = loss
    _normalize_runtime_paths(raw_config)
    normalize_paper_config(raw_config)
    config = initialize_config(raw_config)
    config["device"] = device
    return config


def load_basin_inputs() -> tuple[np.ndarray, torch.Tensor]:
    config = load_runtime_config("mc_dropout", 111, "HybridNseBatchLoss", device="cpu")
    loader = HydroLoader(_build_loader_config(config), test_split=True, overwrite=False)
    reference_ids = load_basin_ids(config["data"]["basin_ids_reference_path"])
    subset_ids = load_basin_ids(config["data"]["basin_ids_path"])
    subset_idx = basin_subset_indices(reference_ids, subset_ids)
    normalized_static = loader.eval_dataset["xc_nn_norm"][0, subset_idx, :].detach().cpu()
    return subset_ids.astype(np.int64), normalized_static


def resolve_checkpoint(run_dir: Path, test_epoch: int = 100) -> Path | None:
    expected = run_dir / "model" / f"model_epoch{test_epoch}.pt"
    if expected.exists():
        return expected
    candidates = sorted(
        (run_dir / "model").glob("model_epoch*.pt"),
        key=lambda path: int(path.stem.replace("model_epoch", "")),
    )
    return candidates[-1] if candidates else None


def discover_runs() -> tuple[list[RunSpec], pd.DataFrame]:
    runs: list[RunSpec] = []
    inventory_rows: list[dict[str, Any]] = []
    for model_raw in MODEL_ORDER:
        family_root = OUTPUT_ROOT / f"{model_raw}-531"
        for loss_dir in sorted(family_root.glob("*")):
            if not loss_dir.is_dir():
                continue
            loss = loss_dir.name
            for seed_dir in sorted(loss_dir.glob("seed_*")):
                if not seed_dir.is_dir():
                    continue
                try:
                    seed = int(seed_dir.name.split("_", 1)[1])
                except (IndexError, ValueError):
                    continue
                meta_path = seed_dir / "run_meta.json"
                checkpoint_path = resolve_checkpoint(seed_dir)
                status = "ok" if checkpoint_path is not None and meta_path.exists() else "missing_artifact"
                run = RunSpec(
                    model_raw=model_raw,
                    loss=loss,
                    seed=seed,
                    run_dir=seed_dir,
                    checkpoint_path=checkpoint_path,
                    run_meta_path=meta_path if meta_path.exists() else None,
                    source_family=f"{model_raw}-531",
                    status=status,
                )
                runs.append(run)
                meta = read_json(meta_path) if meta_path.exists() else {}
                inventory_rows.append(
                    {
                        "model_raw": model_raw,
                        "model_label": MODEL_LABEL[model_raw],
                        "loss": loss,
                        "seed": seed,
                        "run_dir": str(seed_dir),
                        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
                        "run_meta_path": str(meta_path) if meta_path.exists() else "",
                        "status": status,
                        "nn_name": meta.get("nn_name", ""),
                        "mc_samples_config": meta.get("mc_samples", np.nan),
                        "output_activation": meta.get("output_activation", ""),
                        "static_pool": meta.get("static_pool", ""),
                        "data_basin_ids_path": meta.get("data_basin_ids_path", ""),
                    }
                )
    return runs, pd.DataFrame(inventory_rows)


def normalized_to_physical(samples: np.ndarray) -> np.ndarray:
    physical = np.empty_like(samples, dtype=np.float64)
    for idx, (_, bounds) in enumerate(PARAMETER_SPECS):
        low, high = bounds
        physical[..., idx] = samples[..., idx] * (high - low) + low
    return physical


def extract_normalized_samples(run: RunSpec, inputs: torch.Tensor) -> np.ndarray:
    if run.checkpoint_path is None:
        raise FileNotFoundError(f"Missing checkpoint for {run.run_dir}")
    config = load_runtime_config(run.model_raw, run.seed, run.loss, device="cpu")
    model = build_paper_dpl(config).to("cpu")
    state_dict = torch.load(run.checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    nn_model = model.nn_model
    nn_model.eval()

    if run.model_raw == "deterministic":
        with torch.inference_mode():
            output = nn_model(inputs)
        if output.ndim == 3:
            output = output[-1]
        return output.detach().cpu().numpy()[np.newaxis, ...].astype(np.float64)

    dropout_modules = [module for module in nn_model.modules() if isinstance(module, nn.Dropout)]
    dropout_states = [module.training for module in dropout_modules]
    if run.model_raw == "mc_dropout":
        for module in dropout_modules:
            module.train(True)

    samples: list[np.ndarray] = []
    rng_state = torch.get_rng_state()
    try:
        with torch.inference_mode():
            for sample_idx in range(N_STOCHASTIC_SAMPLES):
                torch.manual_seed(int(run.seed) * 1000 + sample_idx)
                if run.model_raw == "mc_dropout":
                    output = nn_model(inputs)
                elif run.model_raw == "distributional":
                    output = nn_model.sample_parameters(inputs)
                else:
                    raise ValueError(run.model_raw)
                if output.ndim == 3:
                    output = output[-1]
                samples.append(output.detach().cpu().numpy().astype(np.float64))
    finally:
        torch.set_rng_state(rng_state)
        for module, was_training in zip(dropout_modules, dropout_states):
            module.train(was_training)
    return np.stack(samples, axis=0)


def sample_rows_for_run(run: RunSpec, basin_ids: np.ndarray, samples_norm: np.ndarray) -> pd.DataFrame:
    samples_phys = normalized_to_physical(samples_norm)
    rows: list[pd.DataFrame] = []
    for param_idx, (parameter, bounds) in enumerate(PARAMETER_SPECS):
        low, high = map(float, bounds)
        search_range = high - low
        values_norm = samples_norm[:, :, param_idx]
        values_phys = samples_phys[:, :, param_idx]
        q_norm = np.quantile(values_norm, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
        q_phys = np.quantile(values_phys, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
        mean_norm = values_norm.mean(axis=0)
        std_norm = values_norm.std(axis=0, ddof=0)
        mean_phys = values_phys.mean(axis=0)
        std_phys = values_phys.std(axis=0, ddof=0)
        rows.append(
            pd.DataFrame(
                {
                    "model_raw": run.model_raw,
                    "model_label": MODEL_LABEL[run.model_raw],
                    "loss": run.loss,
                    "seed": run.seed,
                    "basin_id": basin_ids,
                    "parameter": parameter,
                    "parameter_label": parameter.replace("par", ""),
                    "parameter_group": PARAMETER_GROUP.get(parameter, "hbv"),
                    "n_parameter_samples": samples_norm.shape[0],
                    "parameter_lower_bound": low,
                    "parameter_upper_bound": high,
                    "parameter_range": search_range,
                    "estimate_norm": mean_norm,
                    "estimate_physical": mean_phys,
                    "sample_std_norm": std_norm,
                    "sample_std_physical": std_phys,
                    "q05_norm": q_norm[0],
                    "q25_norm": q_norm[1],
                    "q50_norm": q_norm[2],
                    "q75_norm": q_norm[3],
                    "q95_norm": q_norm[4],
                    "q05_physical": q_phys[0],
                    "q25_physical": q_phys[1],
                    "q50_physical": q_phys[2],
                    "q75_physical": q_phys[3],
                    "q95_physical": q_phys[4],
                    "source_checkpoint": str(run.checkpoint_path),
                    "source_run_dir": str(run.run_dir),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def compute_parameter_stability(estimates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["model_raw", "model_label", "loss", "basin_id", "parameter"]
    stability = (
        estimates.groupby(group_cols, as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            seed_mean=("estimate_physical", "mean"),
            seed_sd=("estimate_physical", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else np.nan),
            normalized_seed_sd=("estimate_norm", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else np.nan),
            seed_range=("estimate_physical", lambda s: float(np.max(s) - np.min(s))),
            normalized_seed_range=("estimate_norm", lambda s: float(np.max(s) - np.min(s))),
            parameter_range=("parameter_range", "first"),
            parameter_lower_bound=("parameter_lower_bound", "first"),
            parameter_upper_bound=("parameter_upper_bound", "first"),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )

    pair_rows: list[dict[str, Any]] = []
    for keys, sub in estimates.groupby(group_cols, sort=True):
        ordered = sub.sort_values("seed")
        for row_i, row_j in itertools.combinations(ordered.itertuples(index=False), 2):
            pair_rows.append(
                {
                    "model_raw": keys[0],
                    "model_label": keys[1],
                    "loss": keys[2],
                    "basin_id": keys[3],
                    "parameter": keys[4],
                    "seed_i": int(row_i.seed),
                    "seed_j": int(row_j.seed),
                    "abs_diff": abs(float(row_i.estimate_physical) - float(row_j.estimate_physical)),
                    "normalized_abs_diff": abs(float(row_i.estimate_norm) - float(row_j.estimate_norm)),
                    "parameter_range": float(row_i.parameter_range),
                }
            )
    pairwise = pd.DataFrame(pair_rows)
    return stability, pairwise


def compute_boundary(estimates: pd.DataFrame, stability: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    boundary_input = estimates.copy()
    boundary_input["near_boundary_02"] = (boundary_input["estimate_norm"] <= 0.02) | (
        boundary_input["estimate_norm"] >= 0.98
    )
    boundary_input["near_boundary_05"] = (boundary_input["estimate_norm"] <= 0.05) | (
        boundary_input["estimate_norm"] >= 0.95
    )
    boundary_input["distance_to_boundary"] = np.minimum(
        boundary_input["estimate_norm"],
        1.0 - boundary_input["estimate_norm"],
    )
    saturation = (
        boundary_input.groupby(["model_raw", "model_label", "loss", "seed", "parameter"], as_index=False)
        .agg(
            n_basins=("basin_id", "nunique"),
            saturation_rate_02=("near_boundary_02", "mean"),
            saturation_rate_05=("near_boundary_05", "mean"),
        )
        .sort_values(["loss", "parameter", "model_raw", "seed"])
        .reset_index(drop=True)
    )
    distance = (
        boundary_input.groupby(["model_raw", "model_label", "loss", "seed", "parameter"], as_index=False)
        .agg(
            n_basins=("basin_id", "nunique"),
            mean_distance_to_boundary=("distance_to_boundary", "mean"),
            median_distance_to_boundary=("distance_to_boundary", "median"),
            p10_distance_to_boundary=("distance_to_boundary", lambda s: float(np.quantile(s, 0.10))),
        )
        .sort_values(["loss", "parameter", "model_raw", "seed"])
        .reset_index(drop=True)
    )

    sat_summary = (
        saturation.groupby(["model_raw", "model_label", "loss", "parameter"], as_index=False)
        .agg(
            mean_saturation_rate_02=("saturation_rate_02", "mean"),
            mean_saturation_rate_05=("saturation_rate_05", "mean"),
        )
    )
    stability_summary = (
        stability.groupby(["model_raw", "model_label", "loss", "parameter"], as_index=False)
        .agg(median_normalized_seed_sd=("normalized_seed_sd", "median"))
    )
    combined = sat_summary.merge(stability_summary, on=["model_raw", "model_label", "loss", "parameter"], how="left")
    rows: list[dict[str, Any]] = []
    for (loss, parameter), sub in combined.groupby(["loss", "parameter"], sort=True):
        base_mcd = sub[sub["model_raw"].isin(["deterministic", "mc_dropout"])]
        dist = sub[sub["model_raw"].eq("distributional")]
        max_base_sat02 = float(base_mcd["mean_saturation_rate_02"].max()) if not base_mcd.empty else np.nan
        max_base_sat05 = float(base_mcd["mean_saturation_rate_05"].max()) if not base_mcd.empty else np.nan
        dist_sat02 = float(dist["mean_saturation_rate_02"].iloc[0]) if not dist.empty else np.nan
        dist_sat05 = float(dist["mean_saturation_rate_05"].iloc[0]) if not dist.empty else np.nan
        sat_gap02 = max_base_sat02 - dist_sat02 if np.isfinite(max_base_sat02) and np.isfinite(dist_sat02) else np.nan
        sat_gap05 = max_base_sat05 - dist_sat05 if np.isfinite(max_base_sat05) and np.isfinite(dist_sat05) else np.nan
        high_base = bool((max_base_sat02 > 0.30) or (max_base_sat05 > 0.50))
        higher_than_dist = bool((sat_gap02 > 0.10) or (sat_gap05 > 0.15))
        apparent_lock = False
        if not base_mcd.empty and not dist.empty:
            best_raw = sub.sort_values("median_normalized_seed_sd").iloc[0]
            apparent_lock = bool(best_raw["model_raw"] in {"deterministic", "mc_dropout"} and best_raw["mean_saturation_rate_02"] > dist_sat02 + 0.10)
        reasons = []
        if high_base:
            reasons.append("baseline saturation exceeds threshold")
        if higher_than_dist:
            reasons.append("baseline saturation higher than delta_dist")
        if apparent_lock:
            reasons.append("raw stability ranking may reflect boundary locking")
        rows.append(
            {
                "loss": loss,
                "parameter": parameter,
                "is_boundary_sensitive": bool(reasons),
                "reason": "; ".join(reasons),
                "max_baseline_saturation_rate_02": max_base_sat02,
                "delta_dist_saturation_rate_02": dist_sat02,
                "baseline_minus_delta_dist_saturation_rate_02": sat_gap02,
                "max_baseline_saturation_rate_05": max_base_sat05,
                "delta_dist_saturation_rate_05": dist_sat05,
                "baseline_minus_delta_dist_saturation_rate_05": sat_gap05,
            }
        )
    sensitive = pd.DataFrame(rows)
    return saturation, distance, sensitive


def summarize_stability(
    stability: pd.DataFrame,
    pairwise: pd.DataFrame,
    sensitive: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parameter_summary = (
        stability.groupby(["model_raw", "model_label", "loss", "parameter"], as_index=False)
        .agg(
            n_basin_parameters=("normalized_seed_sd", "size"),
            median_normalized_seed_sd=("normalized_seed_sd", "median"),
            iqr_normalized_seed_sd=("normalized_seed_sd", iqr),
            mean_normalized_seed_sd=("normalized_seed_sd", "mean"),
            median_normalized_seed_range=("normalized_seed_range", "median"),
        )
        .merge(
            pairwise.groupby(["model_raw", "model_label", "loss", "parameter"], as_index=False).agg(
                median_normalized_abs_diff=("normalized_abs_diff", "median"),
                iqr_normalized_abs_diff=("normalized_abs_diff", iqr),
            ),
            on=["model_raw", "model_label", "loss", "parameter"],
            how="left",
        )
    )
    pooled = (
        stability.groupby(["model_raw", "model_label", "loss"], as_index=False)
        .agg(
            n_basin_parameters=("normalized_seed_sd", "size"),
            median_normalized_seed_sd=("normalized_seed_sd", "median"),
            iqr_normalized_seed_sd=("normalized_seed_sd", iqr),
            mean_normalized_seed_sd=("normalized_seed_sd", "mean"),
            fraction_normalized_seed_sd_lt_0_02=("normalized_seed_sd", lambda s: float((s < 0.02).mean())),
            fraction_normalized_seed_sd_lt_0_05=("normalized_seed_sd", lambda s: float((s < 0.05).mean())),
            fraction_normalized_seed_sd_lt_0_10=("normalized_seed_sd", lambda s: float((s < 0.10).mean())),
            fraction_normalized_seed_sd_lt_0_20=("normalized_seed_sd", lambda s: float((s < 0.20).mean())),
        )
        .merge(
            pairwise.groupby(["model_raw", "model_label", "loss"], as_index=False).agg(
                median_normalized_abs_diff=("normalized_abs_diff", "median"),
                iqr_normalized_abs_diff=("normalized_abs_diff", iqr),
            ),
            on=["model_raw", "model_label", "loss"],
            how="left",
        )
    )
    sensitive_set = set(sensitive.loc[sensitive["is_boundary_sensitive"], "parameter"])
    stability_ex = stability[~stability["parameter"].isin(sensitive_set)].copy()
    pairwise_ex = pairwise[~pairwise["parameter"].isin(sensitive_set)].copy()
    excluding = (
        stability_ex.groupby(["model_raw", "model_label", "loss"], as_index=False)
        .agg(
            excluded_parameters=("parameter", lambda s: ",".join(sorted(sensitive_set))),
            retained_parameter_count=("parameter", "nunique"),
            n_basin_parameters=("normalized_seed_sd", "size"),
            median_normalized_seed_sd=("normalized_seed_sd", "median"),
            iqr_normalized_seed_sd=("normalized_seed_sd", iqr),
            mean_normalized_seed_sd=("normalized_seed_sd", "mean"),
            fraction_normalized_seed_sd_lt_0_02=("normalized_seed_sd", lambda s: float((s < 0.02).mean())),
            fraction_normalized_seed_sd_lt_0_05=("normalized_seed_sd", lambda s: float((s < 0.05).mean())),
            fraction_normalized_seed_sd_lt_0_10=("normalized_seed_sd", lambda s: float((s < 0.10).mean())),
            fraction_normalized_seed_sd_lt_0_20=("normalized_seed_sd", lambda s: float((s < 0.20).mean())),
        )
        .merge(
            pairwise_ex.groupby(["model_raw", "model_label", "loss"], as_index=False).agg(
                median_normalized_abs_diff=("normalized_abs_diff", "median"),
                iqr_normalized_abs_diff=("normalized_abs_diff", iqr),
            ),
            on=["model_raw", "model_label", "loss"],
            how="left",
        )
    )
    return parameter_summary, pooled, excluding


def compute_probabilistic_intervals(estimates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prob = estimates[estimates["model_raw"].isin(["mc_dropout", "distributional"])].copy()
    quantiles = prob[
        [
            "model_raw",
            "model_label",
            "loss",
            "seed",
            "basin_id",
            "parameter",
            "n_parameter_samples",
            "q05_norm",
            "q25_norm",
            "q50_norm",
            "q75_norm",
            "q95_norm",
            "q05_physical",
            "q25_physical",
            "q50_physical",
            "q75_physical",
            "q95_physical",
        ]
    ].copy()
    quantiles["interval_width_90"] = quantiles["q95_norm"] - quantiles["q05_norm"]
    quantiles["interval_width_50"] = quantiles["q75_norm"] - quantiles["q25_norm"]

    overlap_rows: list[dict[str, Any]] = []
    group_cols = ["model_raw", "model_label", "loss", "basin_id", "parameter"]
    for keys, sub in quantiles.groupby(group_cols, sort=True):
        ordered = sub.sort_values("seed")
        for row_i, row_j in itertools.combinations(ordered.itertuples(index=False), 2):
            lower_i, upper_i = float(row_i.q05_norm), float(row_i.q95_norm)
            lower_j, upper_j = float(row_j.q05_norm), float(row_j.q95_norm)
            intersection = max(0.0, min(upper_i, upper_j) - max(lower_i, lower_j))
            union = max(upper_i, upper_j) - min(lower_i, lower_j)
            if abs(union) <= EPS:
                overlap = 1.0 if abs(lower_i - lower_j) <= EPS and abs(upper_i - upper_j) <= EPS else np.nan
            else:
                overlap = intersection / union
            overlap_rows.append(
                {
                    "model_raw": keys[0],
                    "model_label": keys[1],
                    "loss": keys[2],
                    "basin_id": keys[3],
                    "parameter": keys[4],
                    "seed_i": int(row_i.seed),
                    "seed_j": int(row_j.seed),
                    "lower_i": lower_i,
                    "upper_i": upper_i,
                    "lower_j": lower_j,
                    "upper_j": upper_j,
                    "intersection_width": intersection,
                    "union_width": union,
                    "overlap_ratio": overlap,
                }
            )
    overlap = pd.DataFrame(overlap_rows)
    width_summary = (
        quantiles.groupby(["model_raw", "model_label", "loss", "parameter"], as_index=False)
        .agg(
            n_intervals=("interval_width_90", "size"),
            median_interval_width_90=("interval_width_90", "median"),
            iqr_interval_width_90=("interval_width_90", iqr),
            median_interval_width_50=("interval_width_50", "median"),
            iqr_interval_width_50=("interval_width_50", iqr),
        )
    )
    overlap_width = (
        overlap.groupby(["model_raw", "model_label", "loss", "parameter"], as_index=False)
        .agg(
            n_seed_pairs=("overlap_ratio", "size"),
            median_overlap=("overlap_ratio", "median"),
            iqr_overlap=("overlap_ratio", iqr),
        )
        .merge(width_summary, on=["model_raw", "model_label", "loss", "parameter"], how="left")
    )
    return quantiles, overlap, width_summary, overlap_width


def paired_or_mwu(
    left: pd.DataFrame,
    right: pd.DataFrame,
    metric: str,
    paired_cols: list[str],
) -> tuple[str, float, float, float, int]:
    merged = left[paired_cols + [metric]].merge(
        right[paired_cols + [metric]],
        on=paired_cols,
        suffixes=("_left", "_right"),
    ).dropna()
    if len(merged) >= 2:
        diffs = merged[f"{metric}_left"] - merged[f"{metric}_right"]
        if float(np.sum(np.abs(diffs))) <= EPS:
            return "Wilcoxon signed-rank", 0.0, 1.0, 0.0, int(len(merged))
        res = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        effect = float(np.median(diffs))
        return "Wilcoxon signed-rank", float(res.statistic), float(res.pvalue), effect, int(len(merged))
    left_values = left[metric].dropna()
    right_values = right[metric].dropna()
    if len(left_values) == 0 or len(right_values) == 0:
        return "not enough data", np.nan, np.nan, np.nan, 0
    res = mannwhitneyu(left_values, right_values, alternative="two-sided")
    effect = float(left_values.median() - right_values.median())
    return "Mann-Whitney U", float(res.statistic), float(res.pvalue), effect, int(min(len(left_values), len(right_values)))


def compute_tests(
    stability: pd.DataFrame,
    pairwise: pd.DataFrame,
    saturation: pd.DataFrame,
    distance: pd.DataFrame,
    quantiles: pd.DataFrame,
    overlap: pd.DataFrame,
    sensitive: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sensitive_set = set(sensitive.loc[sensitive["is_boundary_sensitive"], "parameter"])
    pairs = [
        ("deterministic", "mc_dropout"),
        ("deterministic", "distributional"),
        ("mc_dropout", "distributional"),
    ]
    for subset_name, stab_sub, pair_sub in [
        ("all_parameters", stability, pairwise),
        (
            "excluding_boundary_sensitive",
            stability[~stability["parameter"].isin(sensitive_set)],
            pairwise[~pairwise["parameter"].isin(sensitive_set)],
        ),
    ]:
        for loss in sorted(stab_sub["loss"].unique()):
            loss_stab = stab_sub[stab_sub["loss"].eq(loss)]
            for a, b in pairs:
                left = loss_stab[loss_stab["model_raw"].eq(a)]
                right = loss_stab[loss_stab["model_raw"].eq(b)]
                test, stat, pval, eff, n = paired_or_mwu(
                    left,
                    right,
                    "normalized_seed_sd",
                    ["loss", "basin_id", "parameter"],
                )
                rows.append(
                    {
                        "comparison": f"{MODEL_LABEL[a]} vs {MODEL_LABEL[b]}",
                        "metric": "normalized_seed_sd",
                        "subset": subset_name,
                        "loss": loss,
                        "test": test,
                        "n_pairs_or_min_n": n,
                        "statistic": stat,
                        "p_value": pval,
                        "effect_size_optional": eff,
                        "interpretation": "paired by basin-parameter within loss where possible",
                    }
                )
            loss_pair = pair_sub[pair_sub["loss"].eq(loss)]
            for a, b in pairs:
                left = loss_pair[loss_pair["model_raw"].eq(a)]
                right = loss_pair[loss_pair["model_raw"].eq(b)]
                test, stat, pval, eff, n = paired_or_mwu(
                    left,
                    right,
                    "normalized_abs_diff",
                    ["loss", "basin_id", "parameter", "seed_i", "seed_j"],
                )
                rows.append(
                    {
                        "comparison": f"{MODEL_LABEL[a]} vs {MODEL_LABEL[b]}",
                        "metric": "normalized_abs_diff",
                        "subset": subset_name,
                        "loss": loss,
                        "test": test,
                        "n_pairs_or_min_n": n,
                        "statistic": stat,
                        "p_value": pval,
                        "effect_size_optional": eff,
                        "interpretation": "paired by basin-parameter-seed-pair within loss where possible",
                    }
                )

    for loss in sorted(saturation["loss"].unique()):
        for baseline in ["deterministic", "mc_dropout"]:
            for metric in ["saturation_rate_02", "saturation_rate_05"]:
                left = saturation[(saturation["loss"].eq(loss)) & (saturation["model_raw"].eq("distributional"))]
                right = saturation[(saturation["loss"].eq(loss)) & (saturation["model_raw"].eq(baseline))]
                test, stat, pval, eff, n = paired_or_mwu(
                    left,
                    right,
                    metric,
                    ["loss", "seed", "parameter"],
                )
                rows.append(
                    {
                        "comparison": f"delta_dist vs {MODEL_LABEL[baseline]}",
                        "metric": metric,
                        "subset": "boundary",
                        "loss": loss,
                        "test": test,
                        "n_pairs_or_min_n": n,
                        "statistic": stat,
                        "p_value": pval,
                        "effect_size_optional": eff,
                        "interpretation": "positive effect means delta_dist metric is higher than baseline",
                    }
                )
            left = distance[(distance["loss"].eq(loss)) & (distance["model_raw"].eq("distributional"))]
            right = distance[(distance["loss"].eq(loss)) & (distance["model_raw"].eq(baseline))]
            test, stat, pval, eff, n = paired_or_mwu(
                left,
                right,
                "median_distance_to_boundary",
                ["loss", "seed", "parameter"],
            )
            rows.append(
                {
                    "comparison": f"delta_dist vs {MODEL_LABEL[baseline]}",
                    "metric": "median_distance_to_boundary",
                    "subset": "boundary",
                    "loss": loss,
                    "test": test,
                    "n_pairs_or_min_n": n,
                    "statistic": stat,
                    "p_value": pval,
                    "effect_size_optional": eff,
                    "interpretation": "positive effect means delta_dist farther from boundary",
                }
            )

    for loss in sorted(quantiles["loss"].unique()):
        left = quantiles[(quantiles["loss"].eq(loss)) & (quantiles["model_raw"].eq("distributional"))]
        right = quantiles[(quantiles["loss"].eq(loss)) & (quantiles["model_raw"].eq("mc_dropout"))]
        test, stat, pval, eff, n = paired_or_mwu(
            left,
            right,
            "interval_width_90",
            ["loss", "seed", "basin_id", "parameter"],
        )
        rows.append(
            {
                "comparison": "delta_dist vs delta_mcd",
                "metric": "interval_width_90",
                "subset": "probabilistic_intervals",
                "loss": loss,
                "test": test,
                "n_pairs_or_min_n": n,
                "statistic": stat,
                "p_value": pval,
                "effect_size_optional": eff,
                "interpretation": "positive effect means delta_dist intervals are wider",
            }
        )
        left = overlap[(overlap["loss"].eq(loss)) & (overlap["model_raw"].eq("distributional"))]
        right = overlap[(overlap["loss"].eq(loss)) & (overlap["model_raw"].eq("mc_dropout"))]
        test, stat, pval, eff, n = paired_or_mwu(
            left,
            right,
            "overlap_ratio",
            ["loss", "basin_id", "parameter", "seed_i", "seed_j"],
        )
        rows.append(
            {
                "comparison": "delta_dist vs delta_mcd",
                "metric": "overlap_ratio",
                "subset": "probabilistic_intervals",
                "loss": loss,
                "test": test,
                "n_pairs_or_min_n": n,
                "statistic": stat,
                "p_value": pval,
                "effect_size_optional": eff,
                "interpretation": "positive effect means delta_dist overlap is higher",
            }
        )
    return pd.DataFrame(rows)


def fmt(value: float, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    if df.empty:
        return "(no rows)"
    small = df.loc[:, columns].head(max_rows).copy()
    rendered = small.copy()
    for column in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[column]):
            rendered[column] = rendered[column].map(lambda value: fmt(float(value), 4))
        else:
            rendered[column] = rendered[column].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(rendered.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(rendered.columns)) + " |"
    body = [
        "| " + " | ".join(str(row[column]) for column in rendered.columns) + " |"
        for _, row in rendered.iterrows()
    ]
    return "\n".join([header, separator, *body])


def write_reports(
    inventory: pd.DataFrame,
    pooled: pd.DataFrame,
    excluding: pd.DataFrame,
    parameter_summary: pd.DataFrame,
    saturation: pd.DataFrame,
    distance: pd.DataFrame,
    sensitive: pd.DataFrame,
    quantiles: pd.DataFrame,
    overlap_width: pd.DataFrame,
    tests: pd.DataFrame,
) -> None:
    ok_inventory = inventory[inventory["status"].eq("ok")]
    sensitive_params = sorted(sensitive.loc[sensitive["is_boundary_sensitive"], "parameter"].unique())
    sat_model_param = (
        saturation.groupby(["model_raw", "model_label", "parameter"], as_index=False)
        .agg(saturation_rate_02=("saturation_rate_02", "mean"), saturation_rate_05=("saturation_rate_05", "mean"))
        .merge(
            distance.groupby(["model_raw", "model_label", "parameter"], as_index=False).agg(
                median_distance_to_boundary=("median_distance_to_boundary", "median"),
                p10_distance_to_boundary=("p10_distance_to_boundary", "median"),
            ),
            on=["model_raw", "model_label", "parameter"],
            how="left",
        )
    )
    prob_summary = (
        overlap_width.groupby(["model_raw", "model_label", "parameter"], as_index=False)
        .agg(
            median_overlap=("median_overlap", "median"),
            iqr_overlap=("iqr_overlap", "median"),
            median_interval_width_90=("median_interval_width_90", "median"),
            iqr_interval_width_90=("iqr_interval_width_90", "median"),
        )
    )
    pooled_rank = pooled.sort_values(["loss", "median_normalized_seed_sd"])
    excluding_rank = excluding.sort_values(["loss", "median_normalized_seed_sd"])

    methods = [
        "# Methods and Definitions",
        "",
        "Model implementation was checked before computing statistics.",
        "",
        "- `deterministic` (`delta_base`) uses `DeterministicParamModel.forward`, a static MLP followed by `sigmoid`; it returns one bounded parameter vector per basin.",
        "- `mc_dropout` (`delta_mcd`) uses `McMlpModel.forward`, also followed by `sigmoid`; dropout layers are re-enabled at extraction time and 100 stochastic forward passes are saved as parameter samples in this analysis.",
        "- `distributional` (`delta_dist`) uses `DistributionalParamModel.sample_parameters`; it samples a latent normal value, then applies the same output activation. Intervals here are Monte Carlo quantiles from 100 sampled bounded parameters, not a normal approximation imposed on bounded scale.",
        "- All neural parameter outputs are normalized search-space values in `[0, 1]` because `paper_variants.normalize_paper_config` forces `output_activation='sigmoid'`. `HbvStatic._unpack` maps them to physical/search ranges with `change_param_range` during hydrologic simulation.",
        "- The deterministic model has no intrinsic probabilistic parameter interval in this implementation.",
        "",
        "Instability is computed within each `model x loss x basin x parameter` group across random seeds. Physical parameter SD is reported, and `normalized_seed_sd` is the SD on `[0, 1]` search-space scale, equivalent to physical SD divided by the HBV search range.",
        "",
        "Near-boundary is evaluated on normalized parameter estimates: `theta <= 0.02 or theta >= 0.98`, with a secondary `0.05/0.95` threshold.",
    ]
    (REPORT_DIR / "methods_and_definitions.md").write_text("\n".join(methods) + "\n", encoding="utf-8")

    inventory_report = [
        "# Data Inventory",
        "",
        f"Discovered {len(inventory)} run directories; {len(ok_inventory)} had both `run_meta.json` and a checkpoint.",
        "",
        markdown_table(
            ok_inventory.groupby(["model_raw", "model_label", "loss"], as_index=False).agg(
                n_seeds=("seed", "nunique"),
                seeds=("seed", lambda s: ",".join(map(str, sorted(set(s))))),
                n_runs=("run_dir", "size"),
                nn_name=("nn_name", "first"),
                output_activation=("output_activation", "first"),
            ),
            ["model_raw", "model_label", "loss", "n_seeds", "seeds", "n_runs", "nn_name", "output_activation"],
            max_rows=60,
        ),
        "",
        "Primary source family used: `project/parameterize/outputs/{model}-531/{loss}/seed_{seed}`.",
    ]
    missing = inventory[~inventory["status"].eq("ok")]
    if not missing.empty:
        inventory_report += ["", "## Missing artifacts", "", markdown_table(missing, ["model_raw", "loss", "seed", "run_dir", "status"], 100)]
    (REPORT_DIR / "data_inventory.md").write_text("\n".join(inventory_report) + "\n", encoding="utf-8")

    boundary_report = [
        "# Boundary Saturation Summary",
        "",
        "Boundary-sensitive parameters were identified automatically using baseline saturation thresholds and the baseline-minus-distributional saturation gap.",
        "",
        f"Boundary-sensitive parameters across losses: {', '.join(sensitive_params) if sensitive_params else 'none'}.",
        "",
        "## Parameter-Level Saturation",
        "",
        markdown_table(
            sat_model_param.sort_values(["parameter", "model_raw"]),
            ["model_label", "parameter", "saturation_rate_02", "saturation_rate_05", "median_distance_to_boundary", "p10_distance_to_boundary"],
            max_rows=80,
        ),
        "",
        "## Automatically Flagged Parameters",
        "",
        markdown_table(
            sensitive[sensitive["is_boundary_sensitive"]].sort_values(["parameter", "loss"]),
            [
                "loss",
                "parameter",
                "reason",
                "max_baseline_saturation_rate_02",
                "delta_dist_saturation_rate_02",
                "baseline_minus_delta_dist_saturation_rate_02",
            ],
            max_rows=80,
        ),
    ]
    (REPORT_DIR / "boundary_saturation_summary.md").write_text("\n".join(boundary_report) + "\n", encoding="utf-8")

    prob_report = [
        "# Probabilistic Interval Summary",
        "",
        "Only `delta_mcd` and `delta_dist` are included. Intervals are q05-q95 and q25-q75 on normalized search-space parameters from 100 stochastic parameter samples per run.",
        "",
        markdown_table(
            prob_summary.sort_values(["parameter", "model_raw"]),
            ["model_label", "parameter", "median_overlap", "iqr_overlap", "median_interval_width_90", "iqr_interval_width_90"],
            max_rows=80,
        ),
        "",
        "The overlap ratio is intersection width divided by union width for seed-pair q05-q95 intervals. Zero-width identical intervals are assigned overlap 1; otherwise zero union is reported as missing.",
    ]
    (REPORT_DIR / "probabilistic_interval_summary.md").write_text("\n".join(prob_report) + "\n", encoding="utf-8")
    (REPORT_DIR / "probabilistic_overlap_width_summary.md").write_text("\n".join(prob_report) + "\n", encoding="utf-8")

    def best_by_loss(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.sort_values(["loss", "median_normalized_seed_sd"]).groupby("loss", as_index=False).first()

    best_full = best_by_loss(pooled)
    best_ex = best_by_loss(excluding)
    dist_vs_mcd_overlap = prob_summary.pivot(index="parameter", columns="model_raw", values="median_overlap")
    dist_higher_overlap = int((dist_vs_mcd_overlap.get("distributional") > dist_vs_mcd_overlap.get("mc_dropout")).sum()) if {"distributional", "mc_dropout"}.issubset(dist_vs_mcd_overlap.columns) else 0
    width_pivot = prob_summary.pivot(index="parameter", columns="model_raw", values="median_interval_width_90")
    dist_wider = int((width_pivot.get("distributional") > width_pivot.get("mc_dropout")).sum()) if {"distributional", "mc_dropout"}.issubset(width_pivot.columns) else 0

    main = [
        "# Figure 2 Statistical Summary",
        "",
        "## 1. Objective",
        "",
        "The Figure 2 statistics evaluate whether learned HBV parameter estimates are stable across random seeds, whether apparent raw stability is confounded by boundary saturation, and whether probabilistic formulations yield cross-seed parameter intervals that overlap without becoming excessively wide.",
        "",
        "## 2. Data inventory",
        "",
        f"The analysis used {len(ok_inventory)} completed runs from the `*-531` output family, covering models {', '.join(MODEL_ORDER)}, losses {', '.join(sorted(ok_inventory['loss'].unique()))}, seeds {', '.join(map(str, sorted(ok_inventory['seed'].unique())))}, {quantiles['basin_id'].nunique()} basins, and {len(PARAMETER_NAMES)} parameters.",
        "",
        "## 3. Definition of parameter instability",
        "",
        "For each model, loss, basin, and parameter, the seed-level estimate is the mean of stochastic parameter samples for probabilistic models and the single parameter output for the deterministic model. `normalized_seed_sd = sd(theta_seed) / search_range`, computed directly on `[0, 1]` search-space values. Pairwise instability is `abs(theta_i - theta_j) / search_range`.",
        "",
        "## 4. Raw parameter-value stability",
        "",
        markdown_table(
            pooled_rank,
            [
                "loss",
                "model_label",
                "median_normalized_seed_sd",
                "iqr_normalized_seed_sd",
                "mean_normalized_seed_sd",
                "fraction_normalized_seed_sd_lt_0_05",
                "median_normalized_abs_diff",
            ],
            max_rows=30,
        ),
        "",
        "Best apparent raw-stability model by loss:",
        "",
        markdown_table(best_full, ["loss", "model_label", "median_normalized_seed_sd", "median_normalized_abs_diff"], 20),
        "",
        "## 5. Boundary saturation and boundary-sensitive parameters",
        "",
        f"Automatically flagged boundary-sensitive parameters: {', '.join(sensitive_params) if sensitive_params else 'none'}. This includes CWH: {'yes' if 'parCWH' in sensitive_params else 'no'}; CFR: {'yes' if 'parCFR' in sensitive_params else 'no'}; snow-related parameters: {', '.join([p for p in ['parTT', 'parCFMAX', 'parCFR', 'parCWH'] if p in sensitive_params]) or 'none'}; routing parameters: {', '.join([p for p in ['route_a', 'route_b'] if p in sensitive_params]) or 'none'}.",
        "",
        markdown_table(
            sat_model_param.sort_values(["saturation_rate_02"], ascending=False),
            ["model_label", "parameter", "saturation_rate_02", "saturation_rate_05", "median_distance_to_boundary", "p10_distance_to_boundary"],
            max_rows=30,
        ),
        "",
        "## 6. Stability after excluding boundary-sensitive parameters",
        "",
        markdown_table(
            excluding_rank,
            [
                "loss",
                "model_label",
                "retained_parameter_count",
                "median_normalized_seed_sd",
                "iqr_normalized_seed_sd",
                "mean_normalized_seed_sd",
                "median_normalized_abs_diff",
            ],
            max_rows=30,
        ),
        "",
        "Best model after excluding boundary-sensitive parameters:",
        "",
        markdown_table(best_ex, ["loss", "model_label", "median_normalized_seed_sd", "median_normalized_abs_diff"], 20),
        "",
        "## 7. Probabilistic interval consistency for $\\delta_{mcd}$ and $\\delta_{dist}$",
        "",
        markdown_table(
            prob_summary.sort_values(["parameter", "model_raw"]),
            ["model_label", "parameter", "median_overlap", "iqr_overlap", "median_interval_width_90", "iqr_interval_width_90"],
            max_rows=80,
        ),
        "",
        f"Across parameters, $\\delta_{{dist}}$ had higher median overlap than $\\delta_{{mcd}}$ for {dist_higher_overlap} of {len(dist_vs_mcd_overlap)} parameters. It had wider median q05-q95 intervals for {dist_wider} of {len(width_pivot)} parameters.",
        "",
        "## 8. Recommended Figure 2 panels",
        "",
        "- Panel (a): raw parameter instability by model, shown as all parameters beside the version excluding boundary-sensitive parameters.",
        "- Panel (b): boundary saturation rate and/or distance-to-boundary by parameter, emphasizing CWH/CFR/snow/routing parameters if flagged by the data.",
        "- Panel (c): q05-q95 interval overlap and interval width for $\\delta_{mcd}$ versus $\\delta_{dist}$, plotted jointly so high overlap is not interpreted without sharpness.",
        "",
        "## 9. Recommended cautious wording",
        "",
        "- Across random seeds, raw parameter-value stability was formulation- and parameter-dependent rather than uniformly favoring the distributional formulation.",
        "- Several low-variability parameters were also close to the search-range boundary, indicating that apparent stability can partly reflect boundary locking.",
        "- After excluding boundary-sensitive parameters, the separation among formulations became smaller for raw parameter estimates.",
        "- For probabilistic formulations, cross-seed interval overlap should be interpreted jointly with q05-q95 width; higher overlap alone does not imply sharper or more reliable uncertainty.",
        "- The distributional formulation reduced boundary saturation for several flagged parameters while retaining an explicit parameter-sampling mechanism.",
        "",
        "## 10. Caveats",
        "",
        "- Raw stability may be confounded by boundary saturation.",
        "- The deterministic model does not provide probabilistic parameter intervals.",
        "- MC dropout and distributional intervals are not identical uncertainty concepts.",
        "- Interval overlap must be interpreted together with interval width.",
        "- Loss functions are retained as grouping variables and are not pooled as independent replicates.",
        "",
        "## Final Answers",
        "",
        f"1. Raw parameter-value stability supports $\\delta_{{dist}}$ as more stable only where it ranks best by loss/parameter; the pooled all-parameter ranking is: {best_full[['loss','model_label']].to_dict('records')}.",
        f"2. Boundary saturation affects this interpretation because flagged parameters ({', '.join(sensitive_params) if sensitive_params else 'none'}) can make boundary-locked estimates appear stable.",
        f"3. Boundary-sensitive parameters are: {', '.join(sensitive_params) if sensitive_params else 'none'}.",
        f"4. After excluding boundary-sensitive parameters, best-by-loss rankings are: {best_ex[['loss','model_label']].to_dict('records')}; compare `parameter_stability_excluding_boundary_sensitive.csv` for full values.",
        "5. See `boundary_saturation_by_parameter.csv`; negative delta_dist-minus-baseline test effects in `statistical_tests_summary.csv` indicate lower saturation for $\\delta_{dist}$.",
        f"6. $\\delta_{{dist}}$ has higher seed-to-seed median interval overlap than $\\delta_{{mcd}}$ for {dist_higher_overlap} parameters.",
        f"7. Higher overlap is accompanied by wider q05-q95 intervals for {dist_wider} parameters, so overlap must be judged jointly with width.",
        "8. Figure 2 should combine raw instability, boundary saturation, and interval overlap-width panels, avoiding any claim that raw parameter stability alone proves superior parameter learning.",
    ]
    (REPORT_DIR / "figure2_statistical_summary.md").write_text("\n".join(main) + "\n", encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_lines = ["Figure 2 statistics preparation log"]
    runs, inventory = discover_runs()
    inventory.to_csv(DATA_DIR / "data_inventory_runs.csv", index=False)
    log_lines.append(f"Discovered {len(runs)} runs.")

    basin_ids, normalized_static = load_basin_inputs()
    log_lines.append(f"Loaded {len(basin_ids)} basin ids and static input tensor {tuple(normalized_static.shape)}.")

    frames: list[pd.DataFrame] = []
    for idx, run in enumerate(runs, start=1):
        if run.status != "ok":
            log_lines.append(f"SKIP missing artifacts: {run.run_dir}")
            continue
        log_lines.append(f"[{idx}/{len(runs)}] extracting {run.model_raw} {run.loss} seed {run.seed}")
        samples_norm = extract_normalized_samples(run, normalized_static)
        frames.append(sample_rows_for_run(run, basin_ids, samples_norm))

    estimates = pd.concat(frames, ignore_index=True)
    estimates.to_csv(DATA_DIR / "parameter_estimates_by_run_long.csv", index=False)

    stability, pairwise = compute_parameter_stability(estimates)
    saturation, distance, sensitive = compute_boundary(estimates, stability)
    parameter_summary, pooled, excluding = summarize_stability(stability, pairwise, sensitive)
    quantiles, overlap, width_summary, overlap_width = compute_probabilistic_intervals(estimates)
    tests = compute_tests(stability, pairwise, saturation, distance, quantiles, overlap, sensitive)

    stability.to_csv(DATA_DIR / "parameter_seed_stability_long.csv", index=False)
    pairwise.to_csv(DATA_DIR / "parameter_pairwise_seed_diff_long.csv", index=False)
    parameter_summary.to_csv(DATA_DIR / "parameter_stability_summary_by_parameter.csv", index=False)
    pooled.to_csv(DATA_DIR / "parameter_stability_summary_pooled.csv", index=False)
    excluding.to_csv(DATA_DIR / "parameter_stability_excluding_boundary_sensitive.csv", index=False)
    saturation.to_csv(DATA_DIR / "boundary_saturation_by_parameter.csv", index=False)
    distance.to_csv(DATA_DIR / "distance_to_boundary_by_parameter.csv", index=False)
    sensitive.to_csv(DATA_DIR / "boundary_sensitive_parameters.csv", index=False)
    quantiles.to_csv(DATA_DIR / "probabilistic_interval_quantiles.csv", index=False)
    overlap.to_csv(DATA_DIR / "probabilistic_seed_interval_overlap.csv", index=False)
    width_summary.to_csv(DATA_DIR / "probabilistic_interval_width_summary.csv", index=False)
    overlap_width.to_csv(DATA_DIR / "probabilistic_overlap_width_summary.csv", index=False)
    tests.to_csv(DATA_DIR / "statistical_tests_summary.csv", index=False)

    write_reports(
        inventory=inventory,
        pooled=pooled,
        excluding=excluding,
        parameter_summary=parameter_summary,
        saturation=saturation,
        distance=distance,
        sensitive=sensitive,
        quantiles=quantiles,
        overlap_width=overlap_width,
        tests=tests,
    )

    log_lines.append(f"Wrote data tables to {DATA_DIR}")
    log_lines.append(f"Wrote reports to {REPORT_DIR}")
    (LOG_DIR / "figure2_analysis_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
