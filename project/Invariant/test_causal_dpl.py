#!/usr/bin/env python3
"""Evaluate a trained causal-dPL checkpoint on test basins with MC dropout."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from dmg.core.data.loaders import HydroLoader
from dmg.core.utils import set_randomseed
from dmg.core.utils.utils import initialize_config

from implements import CausalTrainer, build_causal_dpl
from train_causal_dpl import (
    _build_loader_config,
    _log_effective_cluster_fold_sizes,
    _normalize_runtime_paths,
    _resolve_path,
    _run_model_preflight,
    _validate_mc_mlp_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_causal_dpl")

_MODEL_CHECKPOINT_RE = re.compile(r"model_epoch(\d+)\.pt$")


def build_parser(
    description: str,
    default_config: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to dmg config yaml.",
    )
    parser.add_argument(
        "--holdout",
        type=str,
        default=None,
        help="Held-out effective cluster label (A-G). Overrides config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override experiment seed used to resolve output/model directories.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Checkpoint epoch to load. Defaults to test.test_epoch in config.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=10,
        help="Number of MC-dropout forward passes to evaluate.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--dataset",
        choices=("holdout", "eval"),
        default="holdout",
        help="Evaluate only held-out basins or the full eval split.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional metrics/result directory. Defaults to <output_dir>/test_<dataset>_epochX_mcY.",
    )
    parser.add_argument(
        "--sim-dir",
        default=None,
        help="Optional simulation output directory. Defaults to <sim_dir>/test_<dataset>_epochX_mcY.",
    )
    return parser


def _find_checkpoint(model_dir: str, epoch: int) -> str:
    model_dir_path = Path(model_dir)
    if not model_dir_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    checkpoint = model_dir_path / f"model_epoch{epoch}.pt"
    if checkpoint.exists():
        return str(checkpoint)

    available_epochs = sorted(
        int(match.group(1))
        for path in model_dir_path.iterdir()
        if (match := _MODEL_CHECKPOINT_RE.fullmatch(path.name)) is not None
    )
    if available_epochs:
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. "
            f"Available epochs: {available_epochs}"
        )
    raise FileNotFoundError(f"No checkpoint matched 'model_epoch*.pt' in {model_dir}")


def _default_eval_subdir(dataset: str, epoch: int, mc_samples: int) -> str:
    return f"test_{dataset}_epoch{epoch}_mc{mc_samples}"


def _configure_eval_output_dirs(
    config: dict[str, Any],
    dataset: str,
    epoch: int,
    mc_samples: int,
    output_dir: str | None,
    sim_dir: str | None,
) -> tuple[str, str]:
    subdir = _default_eval_subdir(dataset, epoch, mc_samples)
    base_output_dir = Path(config["output_dir"])
    base_sim_dir = Path(config["sim_dir"])

    final_output_dir = Path(output_dir) if output_dir is not None else base_output_dir / subdir
    final_sim_dir = Path(sim_dir) if sim_dir is not None else base_sim_dir / subdir

    config["output_dir"] = str(final_output_dir)
    config["sim_dir"] = str(final_sim_dir)

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["sim_dir"], exist_ok=True)
    return config["output_dir"], config["sim_dir"]


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)


def _select_eval_data(
    trainer: CausalTrainer,
    dataset_name: str,
 ) -> tuple[dict[str, torch.Tensor], bool]:
    if dataset_name == "holdout":
        holdout_cluster = trainer.splitter.holdout_cluster
        if holdout_cluster is None:
            raise ValueError("Holdout evaluation requires causal.holdout_cluster to be set.")

        eval_data = trainer.splitter.holdout_dataset(trainer.eval_dataset)
        return eval_data, True

    return trainer.eval_dataset, False


def _remove_stale_selection_artifacts(
    config: dict[str, Any],
    holdout: bool,
) -> None:
    stale_paths = [
        Path(config["output_dir"]) / "metrics_avg.json",
        Path(config["output_dir"]) / "metrics_avg_agg.json",
        Path(config["output_dir"]) / "mc_selection.json",
        Path(config["sim_dir"]) / "streamflow_mc_selected.npy",
    ]
    if holdout:
        holdout_cluster = config["causal"].get("holdout_cluster")
        if holdout_cluster is not None:
            stale_paths.append(
                Path(config["output_dir"])
                / f"results_avg_held_out_{holdout_cluster}_seed{config['seed']}.csv"
            )

    for path in stale_paths:
        if path.exists():
            path.unlink()


def _evaluate_mc_mean_prediction(
    trainer: CausalTrainer,
    eval_data: dict[str, torch.Tensor],
    holdout: bool,
) -> dict[str, Any]:
    n_samples = eval_data["xc_nn_norm"].shape[1]
    if n_samples == 0:
        raise ValueError("Evaluation dataset is empty.")

    os.makedirs(trainer.config["output_dir"], exist_ok=True)
    os.makedirs(trainer.config["sim_dir"], exist_ok=True)

    seeds = trainer._mc_seeds()
    target_name = trainer.config["train"]["target"][0]
    obs_array = eval_data["target"][:, :, 0].cpu().numpy()
    mc_path = os.path.join(trainer.config["sim_dir"], f"{target_name}_mc.npy")

    mean_accumulator = None
    mc_memmap = None
    mc_raw_metrics: list[dict[str, Any]] = []
    mc_agg_metrics: list[dict[str, Any]] = []

    for pass_idx, batch_predictions in enumerate(trainer._forward_mc_predictions(eval_data, seeds)):
        prediction = trainer._batch_data(batch_predictions, target_name).astype(
            np.float32,
            copy=False,
        )

        if mc_memmap is None:
            mc_shape = (len(seeds),) + prediction.shape
            mc_memmap = np.lib.format.open_memmap(
                mc_path,
                mode="w+",
                dtype=np.float32,
                shape=mc_shape,
            )
            mean_accumulator = np.zeros_like(prediction, dtype=np.float64)

        mc_memmap[pass_idx] = prediction
        mean_accumulator += prediction

        pass_metrics = trainer._build_metrics_from_prediction_array(
            prediction,
            eval_data["target"],
        )
        mc_raw_metrics.append(
            {
                "pass_index": pass_idx,
                "metrics": trainer._serialize_metrics(pass_metrics),
            }
        )
        mc_agg_metrics.append(
            {
                "pass_index": pass_idx,
                "agg_stats": trainer._jsonify(pass_metrics.calc_stats()),
            }
        )

    if mc_memmap is None or mean_accumulator is None:
        raise RuntimeError("No MC predictions were produced during evaluation.")

    mc_memmap.flush()
    mean_prediction = (mean_accumulator / len(seeds)).astype(np.float32)
    np.save(os.path.join(trainer.config["sim_dir"], f"{target_name}.npy"), mean_prediction)
    np.save(os.path.join(trainer.config["sim_dir"], f"{target_name}_obs.npy"), obs_array)

    metrics = trainer._build_metrics_from_prediction_array(
        mean_prediction,
        eval_data["target"],
    )
    trainer._write_metrics_files(metrics, prefix="metrics")
    trainer._write_mc_metrics(mc_raw_metrics, mc_agg_metrics)
    if holdout:
        trainer._save_holdout_results(metrics)

    return {
        "metrics": metrics,
        "seeds": seeds,
        "mc_path": mc_path,
    }


def _save_eval_metadata(
    output_dir: str,
    payload: dict[str, Any],
) -> None:
    with open(os.path.join(output_dir, "test_meta.json"), "w") as f:
        json.dump(payload, f, indent=4)


def run_evaluation(
    args: argparse.Namespace,
    trainer_cls: type[CausalTrainer],
    logger_name: str,
) -> None:
    logger = logging.getLogger(logger_name)

    raw_config = OmegaConf.load(_resolve_path(args.config))
    raw_config["mode"] = "test"
    if args.holdout is not None:
        raw_config.setdefault("causal", {})
        raw_config["causal"]["holdout_cluster"] = args.holdout
    if args.seed is not None:
        raw_config["seed"] = args.seed
    if args.device is not None:
        raw_config["device"] = args.device
    raw_config.setdefault("test", {})
    raw_config["test"]["mc_dropout"] = True
    raw_config["test"]["mc_samples"] = int(args.mc_samples)

    _normalize_runtime_paths(raw_config)

    config = initialize_config(raw_config)
    _validate_mc_mlp_config(config)

    causal_cfg = config.get("causal", {})
    for required in ("cluster_csv", "basin_ids_path"):
        if required not in causal_cfg:
            raise ValueError(
                f"Missing 'causal.{required}' in config. This evaluation entry point "
                "expects the same split metadata used during training."
            )

    _log_effective_cluster_fold_sizes(causal_cfg)
    set_randomseed(int(config["seed"]))

    checkpoint_epoch = int(args.epoch if args.epoch is not None else config["test"]["test_epoch"])
    checkpoint_path = _find_checkpoint(config["model_dir"], checkpoint_epoch)
    output_dir, sim_dir = _configure_eval_output_dirs(
        config,
        dataset=args.dataset,
        epoch=checkpoint_epoch,
        mc_samples=int(config["test"]["mc_samples"]),
        output_dir=args.output_dir,
        sim_dir=args.sim_dir,
    )

    logger.info(
        "Evaluating checkpoint epoch %d on %s dataset | seed=%s | mc_samples=%d",
        checkpoint_epoch,
        args.dataset,
        config["seed"],
        config["test"]["mc_samples"],
    )
    logger.info("Model dir: %s", config["model_dir"])
    logger.info("Evaluation output dir: %s", output_dir)
    logger.info("Evaluation sim dir: %s", sim_dir)

    data_loader = HydroLoader(
        _build_loader_config(config),
        test_split=True,
        overwrite=False,
    )

    model = build_causal_dpl(config).to(config["device"])
    _run_model_preflight(model, data_loader.train_dataset, config)
    _load_checkpoint(model, checkpoint_path, config["device"])
    logger.info("Loaded checkpoint: %s", checkpoint_path)

    trainer = trainer_cls(
        config,
        model=model,
        train_dataset=data_loader.train_dataset,
        eval_dataset=data_loader.eval_dataset,
        write_out=True,
        verbose=False,
    )

    eval_data, holdout = _select_eval_data(
        trainer,
        dataset_name=args.dataset,
    )
    _remove_stale_selection_artifacts(config, holdout=holdout)
    evaluation_result = _evaluate_mc_mean_prediction(
        trainer=trainer,
        eval_data=eval_data,
        holdout=holdout,
    )
    metrics = evaluation_result["metrics"]

    agg_stats = trainer._jsonify(metrics.calc_stats())
    _save_eval_metadata(
        output_dir,
        {
            "config": args.config,
            "dataset": args.dataset,
            "holdout_cluster": config["causal"].get("holdout_cluster"),
            "seed": int(config["seed"]),
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_path": checkpoint_path,
            "mc_samples": len(evaluation_result["seeds"]),
            "device": str(config["device"]),
            "output_dir": output_dir,
            "sim_dir": sim_dir,
            "mc_prediction_path": evaluation_result["mc_path"],
            "metrics_agg": agg_stats,
        },
    )

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Evaluated dataset: {args.dataset}")
    print(f"Saved metrics to: {output_dir}")
    print(f"Saved simulations to: {sim_dir}")
    print(json.dumps(agg_stats, indent=2))


def parse_args() -> argparse.Namespace:
    return build_parser(
        description="Test a trained causal-dPL / VREx / IRM checkpoint with MC dropout.",
        default_config="conf/config_vrex_dhbv.yaml",
    ).parse_args()


def main() -> None:
    run_evaluation(
        args=parse_args(),
        trainer_cls=CausalTrainer,
        logger_name="test_causal_dpl",
    )


if __name__ == "__main__":
    main()
