"""Data discovery and long-table construction for WRR figures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from dmg.core.data.loaders import HydroLoader
from dmg.core.utils.utils import initialize_config

from project.parameterize.figures.common import (
    LOSS_LABELS,
    LOSS_ORDER,
    MODEL_ORDER,
    PARAM_LABELS,
    choose_available_attributes,
    normalize_parameter_name,
    resolve_analysis_output_root,
)
from project.parameterize.implements.basin_utils import basin_subset_indices, load_basin_ids
from project.parameterize.implements.hbv_static import HbvStatic
from project.parameterize.paper_variants import build_paper_dpl, normalize_paper_config
from project.parameterize.train_dmotpy import (
    _build_loader_config,
    _normalize_runtime_paths,
    _resolve_path,
)


PARAMETER_SPECS = (
    list(HbvStatic.parameter_bounds.items())
    + list(HbvStatic.routing_parameter_bounds.items())
)
PARAMETER_NAMES = [name for name, _ in PARAMETER_SPECS]
ANALYSIS_PARAMETER_BOUNDS = {
    "parBETA": (1.0, 6.0),
    "parFC": (50.0, 1000.0),
    "parK0": (0.05, 0.9),
    "parK1": (0.01, 0.5),
    "parK2": (0.001, 0.2),
    "parLP": (0.2, 1.0),
    "parPERC": (0.0, 10.0),
    "parUZL": (0.0, 100.0),
    "parTT": (-2.5, 2.5),
    "parCFMAX": (0.5, 10.0),
    "parCFR": (0.0, 0.1),
    "parCWH": (0.0, 0.2),
    "route_a": (0.0, 2.9),
    "route_b": (0.0, 6.5),
}
HBV_PRIORS = {
    "FC": (50.0, 500.0),
    "LP": (0.3, 1.0),
    "BETA": (1.0, 6.0),
    "K0": (0.05, 0.5),
    "K1": (0.01, 0.3),
    "K2": (0.001, 0.1),
    "UZL": (0.0, 70.0),
    "PERC": (0.0, 6.0),
    "TT": (-2.0, 2.0),
}


@dataclass(frozen=True)
class RunSpec:
    model: str
    loss: str
    seed: int
    run_dir: Path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_runtime_config(
    config_path: str,
    model: str,
    loss: str,
    seed: int,
    device: str,
) -> dict:
    raw_config = OmegaConf.load(_resolve_path(config_path))
    raw_config["mode"] = "test"
    raw_config["seed"] = int(seed)
    raw_config["device"] = device
    raw_config["gpu_id"] = 0
    raw_config.setdefault("paper", {})
    raw_config["paper"]["variant"] = model
    raw_config["train"]["loss_function"]["name"] = loss
    _normalize_runtime_paths(raw_config)
    normalize_paper_config(raw_config)
    config = initialize_config(raw_config)
    config["device"] = device
    return config


def discover_runs(outputs_root: Path) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for model_dir in sorted(outputs_root.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "analysis":
            continue
        for loss_dir in sorted(model_dir.iterdir()):
            if not loss_dir.is_dir():
                continue
            for seed_dir in sorted(loss_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                try:
                    seed = int(seed_dir.name.split("_", maxsplit=1)[1])
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"Invalid seed directory name '{seed_dir.name}'.") from exc

                meta_path = seed_dir / "run_meta.json"
                if meta_path.exists():
                    metadata = read_json(meta_path)
                    meta_model = str(metadata.get("paper_variant", model_dir.name))
                    meta_loss = str(metadata.get("loss_name", loss_dir.name))
                    meta_seed = int(metadata.get("seed", seed))
                else:
                    meta_model = model_dir.name
                    meta_loss = loss_dir.name
                    meta_seed = seed

                if meta_model != model_dir.name:
                    raise ValueError(
                        f"Run metadata model '{meta_model}' does not match directory '{model_dir.name}'."
                    )
                if meta_loss != loss_dir.name:
                    raise ValueError(
                        f"Run metadata loss '{meta_loss}' does not match directory '{loss_dir.name}'."
                    )
                if meta_seed != seed:
                    raise ValueError(
                        f"Run metadata seed '{meta_seed}' does not match directory '{seed_dir.name}'."
                    )

                runs.append(RunSpec(model=model_dir.name, loss=loss_dir.name, seed=seed, run_dir=seed_dir))

    if not runs:
        raise FileNotFoundError(f"No runs found under {outputs_root}.")
    return runs


def has_checkpoint(run: RunSpec) -> bool:
    model_dir = run.run_dir / "model"
    return model_dir.exists() and any(model_dir.glob("model_epoch*.pt"))


def has_metrics(run: RunSpec) -> bool:
    results_path = run.run_dir / f"results_seed{run.seed}.csv"
    metrics_path = run.run_dir / "metrics_avg.json"
    fallback_metrics = run.run_dir / "metrics.json"
    return results_path.exists() and (metrics_path.exists() or fallback_metrics.exists())


def load_basin_attribute_data(config_path: str) -> tuple[pd.DataFrame, torch.Tensor]:
    config = load_runtime_config(
        config_path=config_path,
        model="mc_dropout",
        loss="HybridNseBatchLoss",
        seed=111,
        device="cpu",
    )
    loader = HydroLoader(_build_loader_config(config), test_split=True, overwrite=False)
    eval_dataset = loader.eval_dataset

    reference_ids = load_basin_ids(config["data"]["basin_ids_reference_path"])
    subset_ids = load_basin_ids(config["data"]["basin_ids_path"])
    subset_idx = basin_subset_indices(reference_ids, subset_ids)

    attr_names = list(config["model"]["nn"]["attributes"])
    attr_frame = pd.DataFrame(
        eval_dataset["c_nn"][subset_idx].cpu().numpy(),
        columns=attr_names,
    )
    attr_frame.insert(0, "basin_id", subset_ids.astype(np.int64))
    normalized_static = eval_dataset["xc_nn_norm"][0, subset_idx, :].detach().cpu()
    return attr_frame, normalized_static


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: str) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)


def resolve_checkpoint(run_dir: Path, config: dict) -> Path:
    model_dir = run_dir / "model"
    test_epoch = int(config["test"].get("test_epoch", 100))
    expected = model_dir / f"model_epoch{test_epoch}.pt"
    if expected.exists():
        return expected

    checkpoints = sorted(
        model_dir.glob("model_epoch*.pt"),
        key=lambda path: int(path.stem.replace("model_epoch", "")),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No model_epoch*.pt files found in {model_dir}.")
    return checkpoints[-1]


def normalized_to_physical(sample_stack: np.ndarray) -> np.ndarray:
    physical = np.empty_like(sample_stack, dtype=np.float64)
    for idx, (_, bounds) in enumerate(PARAMETER_SPECS):
        low, high = bounds
        physical[..., idx] = sample_stack[..., idx] * (high - low) + low
    return physical


def collect_sample_stack(
    model: torch.nn.Module,
    model_name: str,
    inputs: torch.Tensor,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    nn_model = model.nn_model
    nn_model.eval()

    dropout_modules = [module for module in nn_model.modules() if isinstance(module, torch.nn.Dropout)]
    module_training_state = [module.training for module in dropout_modules]
    if model_name == "mc_dropout":
        for module in dropout_modules:
            module.train(True)

    samples: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for sample_idx in range(n_samples):
                torch.manual_seed(seed * 1000 + sample_idx)
                if model_name == "deterministic":
                    output = nn_model(inputs)
                elif model_name == "mc_dropout":
                    output = nn_model(inputs)
                elif model_name == "distributional":
                    output = nn_model.sample_parameters(inputs)
                else:
                    raise ValueError(f"Unsupported model '{model_name}'.")
                if output.ndim == 3:
                    output = output[-1]
                samples.append(output.detach().cpu().numpy())
    finally:
        for module, was_training in zip(dropout_modules, module_training_state):
            module.train(was_training)
    return np.stack(samples, axis=0)


def extract_parameters_for_run(
    run: RunSpec,
    config_path: str,
    normalized_static: torch.Tensor,
    stochastic_samples: int,
    device: str,
    basin_ids: np.ndarray,
) -> pd.DataFrame:
    config = load_runtime_config(config_path, run.model, run.loss, run.seed, device)
    model = build_paper_dpl(config).to(device)
    checkpoint_path = resolve_checkpoint(run.run_dir, config)
    load_checkpoint(model, checkpoint_path, device)

    sample_count = 1 if run.model == "deterministic" else stochastic_samples
    sample_stack = collect_sample_stack(
        model=model,
        model_name=run.model,
        inputs=normalized_static.to(device),
        n_samples=sample_count,
        seed=run.seed,
    )
    physical_stack = normalized_to_physical(sample_stack)
    mean_params = physical_stack.mean(axis=0)
    std_params = physical_stack.std(axis=0, ddof=0)

    frame = pd.DataFrame({"basin_id": basin_ids.astype(np.int64)})
    frame["model"] = run.model
    frame["loss"] = run.loss
    frame["seed"] = run.seed
    frame["sample_count"] = sample_count
    for idx, param_name in enumerate(PARAMETER_NAMES):
        frame[f"{param_name}_mean"] = mean_params[:, idx]
        frame[f"{param_name}_std"] = std_params[:, idx]
    return frame


def load_metrics_for_run(run: RunSpec) -> pd.DataFrame:
    results_path = run.run_dir / f"results_seed{run.seed}.csv"
    metrics_path = run.run_dir / "metrics_avg.json"
    if not metrics_path.exists():
        metrics_path = run.run_dir / "metrics.json"
    if not results_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics artifacts in {run.run_dir}.")

    basin_frame = pd.read_csv(results_path)
    raw_metrics = read_json(metrics_path)
    expected_length = len(basin_frame)
    for metric in ("nse", "kge", "bias", "pbias_abs"):
        values = raw_metrics.get(metric)
        if values is not None and len(values) == expected_length:
            basin_frame[metric] = values

    basin_frame["bias_abs"] = basin_frame.get("bias", pd.Series(np.nan, index=basin_frame.index)).abs()
    basin_frame["model"] = run.model
    basin_frame["loss"] = run.loss
    basin_frame["seed"] = run.seed
    return basin_frame


def to_parameter_long(parameter_by_run: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in parameter_by_run.iterrows():
        for parameter in PARAMETER_NAMES:
            records.append(
                {
                    "basin_id": int(row["basin_id"]),
                    "model": row["model"],
                    "loss": row["loss"],
                    "seed": int(row["seed"]),
                    "sample_count": int(row["sample_count"]),
                    "parameter": parameter,
                    "parameter_label": PARAM_LABELS.get(parameter, parameter),
                    "mean": float(row[f"{parameter}_mean"]),
                    "std": float(row[f"{parameter}_std"]),
                }
            )
    return pd.DataFrame.from_records(records)


def build_corr_long(parameter_long: pd.DataFrame, attributes: pd.DataFrame) -> pd.DataFrame:
    merged = parameter_long.merge(attributes, on="basin_id", how="inner")
    attribute_columns = [column for column in attributes.columns if column != "basin_id"]
    rows: list[dict[str, Any]] = []
    for (model, loss, seed, parameter), subset in merged.groupby(["model", "loss", "seed", "parameter"]):
        for attribute in attribute_columns:
            rho, p_value = spearmanr(subset["mean"], subset[attribute], nan_policy="omit")
            rows.append(
                {
                    "model": model,
                    "loss": loss,
                    "seed": int(seed),
                    "parameter": parameter,
                    "parameter_label": PARAM_LABELS.get(parameter, parameter),
                    "attribute": attribute,
                    "spearman_rho": float(rho),
                    "spearman_p": float(p_value),
                    "spearman_r2": float(rho**2) if not np.isnan(rho) else np.nan,
                    "abs_rho": float(abs(rho)) if not np.isnan(rho) else np.nan,
                }
            )
    return pd.DataFrame.from_records(rows)


def build_data_dict(
    config_path: str,
    outputs_root: str | None = None,
    stochastic_samples: int = 100,
    device: str = "cpu",
) -> dict[str, Any]:
    analysis_root = resolve_analysis_output_root(config_path)
    outputs_root_path = Path(outputs_root).resolve() if outputs_root else analysis_root.parent.parent
    runs = discover_runs(outputs_root_path)
    parameter_runs = [run for run in runs if has_checkpoint(run)]
    metric_runs = [run for run in runs if has_metrics(run)]
    skipped_metric_runs = [run for run in runs if run not in metric_runs]
    if not parameter_runs:
        raise FileNotFoundError(f"No checkpoint-backed runs found under {outputs_root_path}.")
    if not metric_runs:
        raise FileNotFoundError(f"No metric-backed runs found under {outputs_root_path}.")

    attr_frame, normalized_static = load_basin_attribute_data(config_path)
    basin_ids = attr_frame["basin_id"].to_numpy(dtype=np.int64)
    parameter_frames = [
        extract_parameters_for_run(
            run=run,
            config_path=config_path,
            normalized_static=normalized_static,
            stochastic_samples=stochastic_samples,
            device=device,
            basin_ids=basin_ids,
        )
        for run in parameter_runs
    ]
    parameter_by_run = pd.concat(parameter_frames, ignore_index=True)
    parameter_long = to_parameter_long(parameter_by_run)
    metrics_long = pd.concat([load_metrics_for_run(run) for run in metric_runs], ignore_index=True)
    corr_long = build_corr_long(parameter_long, attr_frame)

    raw_config = OmegaConf.load(_resolve_path(config_path))
    configured_reference_loss = str(raw_config["train"]["loss_function"]["name"])
    loss_order = [loss for loss in LOSS_ORDER if loss in {run.loss for run in parameter_runs}]
    loss_order.extend(loss for loss in sorted({run.loss for run in parameter_runs}) if loss not in loss_order)

    metric_coverage_rows = []
    for loss, subset in pd.DataFrame(
        [{"model": run.model, "loss": run.loss, "seed": run.seed} for run in metric_runs]
    ).groupby("loss"):
        metric_coverage_rows.append(
            {
                "loss": loss,
                "model_count": int(subset["model"].nunique()),
                "run_count": int(len(subset)),
                "seed_count": int(subset["seed"].nunique()),
                "configured": int(loss == configured_reference_loss),
            }
        )
    metric_coverage = pd.DataFrame(metric_coverage_rows)
    if metric_coverage.empty:
        reference_loss = configured_reference_loss
    else:
        metric_coverage = metric_coverage.sort_values(
            ["model_count", "run_count", "seed_count", "configured", "loss"],
            ascending=[False, False, False, False, True],
        )
        reference_loss = str(metric_coverage.iloc[0]["loss"])

    selected_attrs = choose_available_attributes([column for column in attr_frame.columns if column != "basin_id"])
    figure14_params = [normalize_parameter_name(name) for name in ("FC", "BETA", "K1", "TT")]
    focus_params = [param for param in figure14_params if param in PARAMETER_NAMES]
    climate_params = [normalize_parameter_name(name) for name in ("FC", "BETA", "K1") if normalize_parameter_name(name) in PARAMETER_NAMES]

    return {
        "runs": runs,
        "parameter_runs": parameter_runs,
        "metric_runs": metric_runs,
        "skipped_metric_runs": skipped_metric_runs,
        "outputs_root": outputs_root_path,
        "analysis_root": analysis_root,
        "reference_loss": reference_loss,
        "configured_reference_loss": configured_reference_loss,
        "model_order": [model for model in MODEL_ORDER if model in {run.model for run in runs}],
        "loss_order": loss_order,
        "seed_order": sorted({run.seed for run in runs}),
        "attributes": attr_frame,
        "attribute_names": [column for column in attr_frame.columns if column != "basin_id"],
        "key_attributes": selected_attrs,
        "focus_parameters": focus_params,
        "climate_parameters": climate_params,
        "metrics_long": metrics_long,
        "parameter_by_run": parameter_by_run,
        "params_long": parameter_long,
        "corr_long": corr_long,
        "param_names": PARAMETER_NAMES,
        "param_labels": PARAM_LABELS,
        "parameter_bounds": ANALYSIS_PARAMETER_BOUNDS,
        "hbv_priors": HBV_PRIORS,
        "loss_labels": {loss: LOSS_LABELS.get(loss, loss) for loss in loss_order},
    }
