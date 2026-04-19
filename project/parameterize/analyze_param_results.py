"""Analyze paper-stack parameterization outputs across methods and seeds.

This script extracts basin-level HBV parameters from trained checkpoints for
the three paper variants, computes basin-attribute/parameter correlations, and
summarizes model accuracy from the existing evaluation artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dmg.core.data.loaders import HydroLoader
from dmg.core.utils.utils import initialize_config
from omegaconf import OmegaConf
from torch import nn

from project.parameterize.implements.basin_utils import basin_subset_indices, load_basin_ids
from project.parameterize.implements.hbv_static import HbvStatic
from project.parameterize.paper_variants import build_paper_dpl, normalize_paper_config
from project.parameterize.train_dmotpy import (
    _build_loader_config,
    _normalize_runtime_paths,
    _resolve_path,
)


VARIANTS = ("deterministic", "mc_dropout", "distributional")
PARAMETER_SPECS = (
    list(HbvStatic.parameter_bounds.items())
    + list(HbvStatic.routing_parameter_bounds.items())
)
PARAMETER_NAMES = [name for name, _ in PARAMETER_SPECS]


@dataclass(frozen=True)
class RunSpec:
    variant: str
    seed: int
    run_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract basin parameters and analyze paper-stack outputs."
    )
    parser.add_argument(
        "--config",
        default="project/parameterize/conf/config_param_paper.yaml",
        help="Base config used for loading the static attribute dataset.",
    )
    parser.add_argument(
        "--outputs-dir",
        default="project/parameterize/outputs",
        help="Directory containing variant_seedXXX output folders.",
    )
    parser.add_argument(
        "--analysis-dir",
        default="project/parameterize/analysis",
        help="Directory where CSV summaries and plots will be written.",
    )
    parser.add_argument(
        "--stochastic-samples",
        type=int,
        default=100,
        help="Monte Carlo sample count used for stochastic parameter extraction.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "mps"),
        help="Device used for checkpoint loading and parameter extraction.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI.",
    )
    args = parser.parse_args()
    if args.stochastic_samples < 100:
        raise ValueError(
            "--stochastic-samples must be at least 100 to analyze MC/dropout uncertainty robustly."
        )
    return args


def load_runtime_config(
    config_path: str,
    variant: str,
    seed: int,
    device: str,
) -> dict:
    raw_config = OmegaConf.load(_resolve_path(config_path))
    raw_config["mode"] = "test"
    raw_config["seed"] = int(seed)
    raw_config["device"] = device
    raw_config["gpu_id"] = 0
    raw_config.setdefault("paper", {})
    raw_config["paper"]["variant"] = variant
    _normalize_runtime_paths(raw_config)
    normalize_paper_config(raw_config)
    config = initialize_config(raw_config)
    config["device"] = device
    return config


def discover_runs(outputs_dir: Path) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for variant in VARIANTS:
        for run_dir in sorted(outputs_dir.glob(f"{variant}_seed*")):
            seed_text = run_dir.name.rsplit("seed", maxsplit=1)[-1]
            if not seed_text.isdigit():
                continue
            runs.append(RunSpec(variant=variant, seed=int(seed_text), run_dir=run_dir))
    if not runs:
        raise FileNotFoundError(f"No run directories found under {outputs_dir}.")
    return runs


def load_basin_attribute_data(config_path: str) -> tuple[pd.DataFrame, torch.Tensor]:
    config = load_runtime_config(config_path, variant="mc_dropout", seed=111, device="cpu")
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
    variant: str,
    inputs: torch.Tensor,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    nn_model = model.nn_model
    nn_model.eval()

    if variant == "deterministic":
        with torch.inference_mode():
            output = nn_model(inputs)
        if output.ndim == 3:
            output = output[-1]
        return output.detach().cpu().numpy()[np.newaxis, ...]

    dropout_modules = [module for module in nn_model.modules() if isinstance(module, nn.Dropout)]
    module_training_state = [module.training for module in dropout_modules]
    if variant == "mc_dropout":
        for module in dropout_modules:
            module.train(True)

    samples: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for sample_idx in range(n_samples):
                torch.manual_seed(seed * 1000 + sample_idx)
                if variant == "mc_dropout":
                    output = nn_model(inputs)
                elif variant == "distributional":
                    output = nn_model.sample_parameters(inputs)
                else:
                    raise ValueError(f"Unsupported variant '{variant}'.")
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
) -> pd.DataFrame:
    config = load_runtime_config(config_path, run.variant, run.seed, device)
    model = build_paper_dpl(config).to(device)
    checkpoint_path = resolve_checkpoint(run.run_dir, config)
    load_checkpoint(model, checkpoint_path, device)

    inputs = normalized_static.to(device)
    sample_count = 1 if run.variant == "deterministic" else stochastic_samples
    sample_stack = collect_sample_stack(
        model=model,
        variant=run.variant,
        inputs=inputs,
        n_samples=sample_count,
        seed=run.seed,
    )
    physical_stack = normalized_to_physical(sample_stack)
    mean_params = physical_stack.mean(axis=0)
    std_params = physical_stack.std(axis=0, ddof=0)

    frame = pd.DataFrame({"basin_id": load_basin_ids(config["data"]["basin_ids_path"])})
    frame["variant"] = run.variant
    frame["seed"] = run.seed
    frame["sample_count"] = sample_count

    for idx, param_name in enumerate(PARAMETER_NAMES):
        frame[f"{param_name}_mean"] = mean_params[:, idx]
        frame[f"{param_name}_std"] = std_params[:, idx]
    return frame


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_accuracy(runs: Iterable[RunSpec]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    basin_frames: list[pd.DataFrame] = []
    per_run_rows: list[dict[str, float | int | str]] = []

    for run in runs:
        results_path = run.run_dir / f"results_seed{run.seed}.csv"
        metrics_agg_path = run.run_dir / "metrics_agg.json"
        if not results_path.exists() or not metrics_agg_path.exists():
            continue

        basin_frame = pd.read_csv(results_path)
        basin_frame["variant"] = run.variant
        basin_frame["seed"] = run.seed
        basin_frames.append(basin_frame)

        metric_summary = read_json(metrics_agg_path)
        row: dict[str, float | int | str] = {
            "variant": run.variant,
            "seed": run.seed,
            "basin_count": int(len(basin_frame)),
            "kge_mean_empirical": float(basin_frame["kge"].mean()),
            "kge_median_empirical": float(basin_frame["kge"].median()),
            "kge_std_empirical": float(basin_frame["kge"].std(ddof=0)),
            "share_kge_gt_0": float((basin_frame["kge"] > 0).mean()),
            "share_kge_gt_05": float((basin_frame["kge"] > 0.5).mean()),
        }
        for metric_name in ("kge", "nse", "rmse", "corr", "mae", "pbias_abs"):
            if metric_name in metric_summary:
                row[f"{metric_name}_mean"] = float(metric_summary[metric_name]["mean"])
                row[f"{metric_name}_median"] = float(metric_summary[metric_name]["median"])
                row[f"{metric_name}_std"] = float(metric_summary[metric_name]["std"])
        per_run_rows.append(row)

    basin_scores = pd.concat(basin_frames, ignore_index=True)
    per_run = pd.DataFrame(per_run_rows).sort_values(["variant", "seed"]).reset_index(drop=True)

    metric_cols = [
        column
        for column in per_run.columns
        if column not in {"variant", "seed"}
    ]
    per_method = per_run.groupby("variant")[metric_cols].agg(["mean", "std"])
    per_method.columns = [f"{metric}_{stat}" for metric, stat in per_method.columns]
    per_method = per_method.reset_index()
    return basin_scores, per_run, per_method


def aggregate_parameter_frames(parameter_by_seed: pd.DataFrame) -> pd.DataFrame:
    mean_columns = [f"{name}_mean" for name in PARAMETER_NAMES]
    std_columns = [f"{name}_std" for name in PARAMETER_NAMES]
    agg_map = {column: "mean" for column in mean_columns + std_columns}
    aggregated = (
        parameter_by_seed.groupby(["basin_id", "variant"], as_index=False)
        .agg(agg_map)
        .sort_values(["variant", "basin_id"])
        .reset_index(drop=True)
    )
    return aggregated


def correlation_matrix(
    attr_frame: pd.DataFrame,
    parameter_frame: pd.DataFrame,
    suffix: str,
) -> pd.DataFrame:
    attributes = attr_frame.set_index("basin_id")
    param_cols = [f"{name}_{suffix}" for name in PARAMETER_NAMES]
    params = parameter_frame.set_index("basin_id")[param_cols]
    params.columns = PARAMETER_NAMES
    combined = pd.concat([attributes, params], axis=1)
    corr = combined.corr(method="spearman").loc[attributes.columns, PARAMETER_NAMES]
    return corr


def flatten_correlation(corr: pd.DataFrame, variant: str, target: str) -> pd.DataFrame:
    rows = []
    for attribute, row in corr.iterrows():
        for parameter, value in row.items():
            rows.append(
                {
                    "variant": variant,
                    "target": target,
                    "attribute": attribute,
                    "parameter": parameter,
                    "spearman_rho": float(value),
                    "abs_rho": float(abs(value)),
                }
            )
    return pd.DataFrame(rows).sort_values("abs_rho", ascending=False).reset_index(drop=True)


def plot_heatmap(
    corr: pd.DataFrame,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    fig_width = max(10.0, corr.shape[1] * 0.55)
    fig_height = max(10.0, corr.shape[0] * 0.30)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(corr.shape[1]))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(corr.shape[0]))
    ax.set_yticklabels(corr.index)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_boxplot(basin_scores: pd.DataFrame, output_path: Path, dpi: int) -> None:
    ordered_variants = [variant for variant in VARIANTS if variant in basin_scores["variant"].unique()]
    data = [
        basin_scores.loc[basin_scores["variant"] == variant, "kge"].to_numpy()
        for variant in ordered_variants
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=ordered_variants, showfliers=False)
    ax.set_ylabel("KGE")
    ax.set_title("Basin-level KGE by method")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_seed_summary(per_run: pd.DataFrame, output_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in VARIANTS:
        subset = per_run.loc[per_run["variant"] == variant].sort_values("seed")
        if subset.empty:
            continue
        ax.plot(
            subset["seed"],
            subset["kge_mean_empirical"],
            marker="o",
            linewidth=1.8,
            label=variant,
        )
    ax.set_xlabel("Seed")
    ax.set_ylabel("Mean basin KGE")
    ax.set_title("Mean KGE across seeds")
    ax.grid(linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_report(
    report_path: Path,
    accuracy_by_method: pd.DataFrame,
    top_correlations: pd.DataFrame,
) -> None:
    accuracy_view = accuracy_by_method.copy()
    top_view = (
        top_correlations.groupby(["variant", "target"], as_index=False)
        .head(5)
        .reset_index(drop=True)
    )

    lines = [
        "# Parameterization Analysis Report",
        "",
        "## Accuracy summary by method",
        "",
        accuracy_view.to_string(index=False),
        "",
        "## Top attribute-parameter correlations",
        "",
        top_view.to_string(index=False),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir).resolve()
    analysis_dir = Path(args.analysis_dir).resolve()
    figures_dir = analysis_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(outputs_dir)
    attr_frame, normalized_static = load_basin_attribute_data(args.config)
    attr_frame.to_csv(analysis_dir / "basin_attributes.csv", index=False)

    parameter_frames = [
        extract_parameters_for_run(
            run=run,
            config_path=args.config,
            normalized_static=normalized_static,
            stochastic_samples=args.stochastic_samples,
            device=args.device,
        )
        for run in runs
    ]
    parameter_by_seed = pd.concat(parameter_frames, ignore_index=True)
    parameter_by_seed.to_csv(analysis_dir / "predicted_parameters_by_seed.csv", index=False)

    parameter_ensemble = aggregate_parameter_frames(parameter_by_seed)
    parameter_ensemble.to_csv(
        analysis_dir / "predicted_parameters_seed_ensemble.csv",
        index=False,
    )

    top_corr_frames: list[pd.DataFrame] = []
    for variant in VARIANTS:
        variant_params = parameter_ensemble.loc[parameter_ensemble["variant"] == variant]
        if variant_params.empty:
            continue

        mean_corr = correlation_matrix(attr_frame, variant_params, suffix="mean")
        mean_corr.to_csv(analysis_dir / f"correlation_attr_param_mean_{variant}.csv")
        plot_heatmap(
            mean_corr,
            title=f"{variant}: attribute vs parameter mean",
            output_path=figures_dir / f"correlation_attr_param_mean_{variant}.png",
            dpi=args.dpi,
        )
        top_corr_frames.append(flatten_correlation(mean_corr, variant, "param_mean"))

        if variant != "deterministic":
            std_corr = correlation_matrix(attr_frame, variant_params, suffix="std")
            std_corr.to_csv(analysis_dir / f"correlation_attr_param_std_{variant}.csv")
            plot_heatmap(
                std_corr,
                title=f"{variant}: attribute vs parameter std",
                output_path=figures_dir / f"correlation_attr_param_std_{variant}.png",
                dpi=args.dpi,
            )
            top_corr_frames.append(flatten_correlation(std_corr, variant, "param_std"))

    top_correlations = pd.concat(top_corr_frames, ignore_index=True)
    top_correlations.to_csv(analysis_dir / "top_attribute_parameter_correlations.csv", index=False)

    basin_scores, accuracy_by_run, accuracy_by_method = summarize_accuracy(runs)
    basin_scores.to_csv(analysis_dir / "accuracy_basin_kge.csv", index=False)
    accuracy_by_run.to_csv(analysis_dir / "accuracy_by_run.csv", index=False)
    accuracy_by_method.to_csv(analysis_dir / "accuracy_by_method.csv", index=False)

    plot_accuracy_boxplot(
        basin_scores=basin_scores,
        output_path=figures_dir / "accuracy_kge_boxplot.png",
        dpi=args.dpi,
    )
    plot_accuracy_seed_summary(
        per_run=accuracy_by_run,
        output_path=figures_dir / "accuracy_kge_seed_summary.png",
        dpi=args.dpi,
    )

    render_report(
        report_path=analysis_dir / "analysis_report.md",
        accuracy_by_method=accuracy_by_method,
        top_correlations=top_correlations,
    )

    metadata = {
        "outputs_dir": str(outputs_dir),
        "analysis_dir": str(analysis_dir),
        "stochastic_samples": int(args.stochastic_samples),
        "device": args.device,
        "variants": list(VARIANTS),
        "run_count": len(runs),
        "parameter_names": PARAMETER_NAMES,
    }
    (analysis_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
