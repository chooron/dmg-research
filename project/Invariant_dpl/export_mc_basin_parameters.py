#!/usr/bin/env python3
"""Export MC-dropout basin-parameter samples and summary tables.

This script loads a trained Causal-dPL checkpoint, runs repeated MC-dropout
forward passes through ``McMlpModel``, and saves:

- per-basin / per-pass parameter samples,
- per-basin parameter mean and variance,
- a merged attribute + parameter summary table for downstream correlation
  analysis.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from dmg.core.data.loaders import HydroLoader
from dmg.core.utils import set_randomseed
from dmg.core.utils.utils import initialize_config

from implements import GnannEnvironmentSplitter, build_causal_dpl


_MODEL_CHECKPOINT_RE = re.compile(r"model_epoch(\d+)\.pt$")


def _resolve_path(path_str: str) -> str:
    path = Path(path_str)
    if path.exists():
        return str(path)

    repo_path = REPO_ROOT / path_str
    if repo_path.exists():
        return str(repo_path)

    project_path = PROJECT_DIR / path_str
    if project_path.exists():
        return str(project_path)

    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="conf/config_vrex_dhbv.yaml",
        help="Path to config yaml.",
    )
    parser.add_argument(
        "--holdout",
        type=str,
        default=None,
        help="Held-out effective cluster label (A-G).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Experiment seed used to resolve output/model directories.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Checkpoint epoch to load. Defaults to the latest checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        choices=("holdout", "train", "eval"),
        default="holdout",
        help="Which basin split to export.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=10,
        help="Number of MC-dropout passes to export.",
    )
    parser.add_argument(
        "--mc-seed-base",
        type=int,
        default=None,
        help="Base seed for MC passes. Defaults to config seed.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional export directory. Defaults to <output_dir>/analysis.",
    )
    return parser.parse_args()


def _load_basin_ids(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path, allow_pickle=True).astype(int)
    return np.loadtxt(path, dtype=int)


def _build_loader_config(config: dict[str, Any]) -> dict[str, Any]:
    loader_config = copy.deepcopy(config)
    loader_config["device"] = "cpu"
    return loader_config


def _find_checkpoint(model_dir: str, epoch: int | None) -> tuple[int, str]:
    model_dir_path = Path(model_dir)
    if not model_dir_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if epoch is not None:
        checkpoint = model_dir_path / f"model_epoch{epoch}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return epoch, str(checkpoint)

    latest_epoch = -1
    latest_path: Path | None = None
    for path in model_dir_path.iterdir():
        match = _MODEL_CHECKPOINT_RE.fullmatch(path.name)
        if match is None:
            continue
        candidate_epoch = int(match.group(1))
        if candidate_epoch > latest_epoch:
            latest_epoch = candidate_epoch
            latest_path = path

    if latest_path is None:
        raise FileNotFoundError(f"No checkpoint matched 'model_epoch*.pt' in {model_dir}")
    return latest_epoch, str(latest_path)


def _select_dataset(
    splitter: GnannEnvironmentSplitter,
    train_dataset: dict[str, torch.Tensor],
    eval_dataset: dict[str, torch.Tensor],
    dataset_name: str,
) -> tuple[dict[str, torch.Tensor], np.ndarray, str]:
    if dataset_name == "holdout":
        if len(splitter.holdout_indices) == 0:
            raise ValueError("Holdout split is empty; specify a valid --holdout cluster.")
        return (
            splitter.holdout_dataset(eval_dataset),
            splitter.holdout_indices.copy(),
            "holdout",
        )

    if dataset_name == "train":
        if splitter.train_env_indices:
            basin_indices = np.sort(np.concatenate(list(splitter.train_env_indices.values())))
            return (
                GnannEnvironmentSplitter._index_dataset(train_dataset, basin_indices),
                basin_indices,
                "train",
            )
        basin_indices = np.arange(train_dataset["c_nn"].shape[0])
        return train_dataset, basin_indices, "train"

    basin_indices = np.arange(eval_dataset["c_nn"].shape[0])
    return eval_dataset, basin_indices, "eval"


def _enable_mc_dropout(model: torch.nn.Module, enabled: bool) -> None:
    model.eval()
    if not enabled:
        return
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train(True)


def _mc_seeds(seed_base: int, mc_samples: int) -> list[int]:
    return [seed_base + i for i in range(mc_samples)]


def _seed_mc_pass(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parameter_names(phy_model: Any) -> list[str]:
    names: list[str] = []
    for name in phy_model.parameter_bounds:
        if int(phy_model.nmul) == 1:
            names.append(name)
        else:
            names.extend(f"{name}_{i}" for i in range(int(phy_model.nmul)))
    names.extend(list(phy_model.routing_parameter_bounds))
    return names


def _flatten_physical_parameters(
    phy_params: dict[str, torch.Tensor],
    route_params: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {}
    for name, tensor in phy_params.items():
        values = tensor.detach().cpu().numpy()
        if values.ndim == 1:
            flat[name] = values
        elif values.ndim == 2 and values.shape[1] == 1:
            flat[name] = values[:, 0]
        else:
            for idx in range(values.shape[1]):
                flat[f"{name}_{idx}"] = values[:, idx]

    for name, tensor in route_params.items():
        flat[name] = tensor.detach().cpu().numpy()
    return flat


def _build_attribute_frame(
    basin_meta: dict[str, np.ndarray],
    c_nn: torch.Tensor,
    attribute_names: list[str],
    split_name: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "basin_id": basin_meta["basin_id"],
            "effective_cluster": basin_meta["effective_cluster"],
            "gauge_cluster": basin_meta["gauge_cluster"],
            "dataset_split": split_name,
        }
    )
    c_nn_np = c_nn.detach().cpu().numpy()
    for idx, name in enumerate(attribute_names):
        frame[name] = c_nn_np[:, idx]
    return frame


def _build_summary_frame(
    samples_df: pd.DataFrame,
    id_columns: list[str],
    value_columns: list[str],
) -> pd.DataFrame:
    mean_df = (
        samples_df.groupby(id_columns, as_index=False)[value_columns]
        .mean()
        .rename(columns={col: f"{col}_mean" for col in value_columns})
    )
    var_df = (
        samples_df.groupby(id_columns, as_index=False)[value_columns]
        .var(ddof=0)
        .rename(columns={col: f"{col}_var" for col in value_columns})
    )
    return mean_df.merge(var_df, on=id_columns, how="inner")


def main() -> None:
    args = parse_args()

    raw_config = OmegaConf.load(_resolve_path(args.config))
    if args.holdout is not None:
        raw_config.setdefault("causal", {})
        raw_config["causal"]["holdout_cluster"] = args.holdout
    if args.seed is not None:
        raw_config["seed"] = args.seed
    if args.device is not None:
        raw_config["device"] = args.device

    config = initialize_config(raw_config)
    set_randomseed(int(config["seed"]))

    basin_ids = _load_basin_ids(config["causal"]["basin_ids_path"])
    splitter = GnannEnvironmentSplitter(
        cluster_csv=config["causal"]["cluster_csv"],
        basin_ids=basin_ids,
        holdout_cluster=config["causal"].get("holdout_cluster"),
    )

    data_loader = HydroLoader(
        _build_loader_config(config),
        test_split=True,
        overwrite=False,
    )
    export_dataset, basin_indices, split_name = _select_dataset(
        splitter,
        data_loader.train_dataset,
        data_loader.eval_dataset,
        args.dataset,
    )

    basin_meta = splitter.basin_metadata(basin_indices)
    attribute_names = list(config["model"]["nn"].get("attributes", []))
    if export_dataset["c_nn"].shape[1] != len(attribute_names):
        raise ValueError(
            "Attribute count mismatch between dataset and config: "
            f"{export_dataset['c_nn'].shape[1]} vs {len(attribute_names)}."
        )

    model = build_causal_dpl(config).to(config["device"])
    checkpoint_epoch, checkpoint_path = _find_checkpoint(config["model_dir"], args.epoch)
    state_dict = torch.load(checkpoint_path, map_location=config["device"])
    model.load_state_dict(state_dict)

    mc_seed_base = int(args.mc_seed_base if args.mc_seed_base is not None else config["seed"])
    seeds = _mc_seeds(mc_seed_base, int(args.mc_samples))
    output_dir = args.output_dir or os.path.join(config["output_dir"], "analysis")
    os.makedirs(output_dir, exist_ok=True)

    attr_df = _build_attribute_frame(
        basin_meta=basin_meta,
        c_nn=export_dataset["c_nn"],
        attribute_names=attribute_names,
        split_name=split_name,
    )
    base_columns = ["basin_id", "effective_cluster", "gauge_cluster", "dataset_split"]
    param_names = _parameter_names(model.phy_model)

    samples: list[pd.DataFrame] = []
    xc_nn_norm = export_dataset["xc_nn_norm"].to(config["device"])

    was_training = model.training
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    _enable_mc_dropout(model, enabled=len(seeds) > 1)

    try:
        with torch.no_grad():
            for pass_idx, seed in enumerate(seeds):
                _seed_mc_pass(seed)
                parameters = model.nn_model(xc_nn_norm)
                if parameters.ndim != 3:
                    raise ValueError(
                        f"Expected McMlpModel output [T, B, ny], got {tuple(parameters.shape)}."
                    )

                last_parameters = parameters[-1].detach().cpu().numpy()
                phy_params, route_params = model.phy_model._unpack(parameters)
                physical = _flatten_physical_parameters(phy_params, route_params)

                frame = attr_df[base_columns].copy()
                frame["pass_index"] = pass_idx
                frame["mc_seed"] = seed
                for idx, name in enumerate(param_names):
                    frame[f"norm_{name}"] = last_parameters[:, idx]
                for name, values in physical.items():
                    frame[name] = values
                samples.append(frame)
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        model.train(was_training)

    samples_df = pd.concat(samples, ignore_index=True)
    value_columns = [
        col for col in samples_df.columns
        if col not in base_columns + ["pass_index", "mc_seed"]
    ]
    summary_df = _build_summary_frame(samples_df, base_columns, value_columns)
    analysis_df = attr_df.merge(summary_df, on=base_columns, how="inner")

    suffix = f"{split_name}_mc{len(seeds)}"
    samples_path = os.path.join(output_dir, f"mc_parameter_samples_{suffix}.csv")
    summary_path = os.path.join(output_dir, f"mc_parameter_summary_{suffix}.csv")
    analysis_path = os.path.join(output_dir, f"mc_attribute_parameter_table_{suffix}.csv")
    meta_path = os.path.join(output_dir, f"mc_parameter_export_meta_{suffix}.json")

    samples_df.to_csv(samples_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    analysis_df.to_csv(analysis_path, index=False)

    metadata = {
        "config": args.config,
        "dataset": split_name,
        "holdout_cluster": config["causal"].get("holdout_cluster"),
        "seed": int(config["seed"]),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_path": checkpoint_path,
        "mc_samples": len(seeds),
        "mc_seed_base": mc_seed_base,
        "device": str(config["device"]),
        "output_dir": output_dir,
        "attribute_names": attribute_names,
        "parameter_names": param_names,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Dataset split: {split_name} | basins: {len(attr_df)} | MC passes: {len(seeds)}")
    print(f"Saved samples to {samples_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved analysis table to {analysis_path}")


if __name__ == "__main__":
    main()
