"""Paper-facing main-split HBV parameter-learning entrypoint."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent.parent
INVARIANT_DIR = REPO_ROOT / "project" / "Invariant"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(INVARIANT_DIR))

from dmg.core.data.loaders import HydroLoader
from dmg.core.utils import print_config, set_randomseed
from dmg.core.utils.utils import initialize_config
from omegaconf import OmegaConf

from project.parameterize.implements import build_paper_trainer
from paper_variants import (
    build_paper_dpl,
    normalize_paper_config,
    validate_paper_config,
    write_run_metadata,
)
from runtime_overrides import apply_runtime_overrides
from train_dmotpy import (
    _build_loader_config,
    _build_optimizer_and_scheduler,
    _normalize_runtime_paths,
    _resolve_path,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_param_paper")


def _run_model_preflight(model, train_dataset: dict, config: dict) -> None:
    """Run a minimal forward pass to catch shape/device mismatches early."""
    if train_dataset is None:
        raise ValueError("train_dataset is required for paper-stack preflight.")

    required_keys = ("xc_nn_norm", "x_phy")
    missing = [key for key in required_keys if key not in train_dataset]
    if missing:
        raise ValueError(f"train_dataset missing required keys: {missing}")

    sample = {}
    for key in required_keys:
        value = train_dataset[key]
        if not torch.is_tensor(value):
            raise TypeError(f"train_dataset['{key}'] must be a torch.Tensor.")
        if value.ndim < 2:
            raise ValueError(
                f"train_dataset['{key}'] must be at least 2D, got {tuple(value.shape)}."
            )

        if value.ndim >= 3:
            sample[key] = value[:, :1].to(config["device"])
        else:
            sample[key] = value[:1].to(config["device"])

    with torch.no_grad():
        predictions = model(sample, eval=True)

    if not isinstance(predictions, dict) or "streamflow" not in predictions:
        raise RuntimeError("Paper-stack preflight expected a prediction dict with 'streamflow'.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/test paper-facing HBV parameter-learning variants"
    )
    parser.add_argument(
        "--config",
        default="conf/config_param_paper.yaml",
        help="Path to paper-facing config yaml.",
    )
    parser.add_argument(
        "--variant",
        choices=["deterministic", "mc_dropout", "distributional"],
        default=None,
        help="Override paper.variant in config.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Override paper.split in config. Slice 1 supports only 'main'.",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="Override config mode: train | test | train_test",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Override config device.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Override config gpu_id when device=cuda.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override experiment seed.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Override paper.seeds list.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train.epochs in config.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=None,
        help="Override test.mc_samples in config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_config = OmegaConf.load(_resolve_path(args.config))
    apply_runtime_overrides(raw_config, args)
    normalize_paper_config(raw_config)
    _normalize_runtime_paths(raw_config)

    config = initialize_config(raw_config)
    validate_paper_config(config)

    lr_sched = config["train"].get("lr_scheduler")
    if isinstance(lr_sched, dict):
        config["train"]["lr_scheduler"] = lr_sched.get("name", str(lr_sched))
    print_config(config)
    if isinstance(lr_sched, dict):
        config["train"]["lr_scheduler"] = lr_sched

    set_randomseed(config["seed"])
    log.info(
        "Paper stack | variant=%s | seed=%s | output_dir=%s",
        config["paper"]["variant"],
        config["seed"],
        config["output_dir"],
    )

    data_loader = HydroLoader(
        _build_loader_config(config),
        test_split=True,
        overwrite=False,
    )

    model = build_paper_dpl(config).to(config["device"])
    _run_model_preflight(model, data_loader.train_dataset, config)
    optimizer, scheduler = _build_optimizer_and_scheduler(config, model)

    trainer = build_paper_trainer(
        config=config,
        model=model,
        train_dataset=data_loader.train_dataset,
        eval_dataset=data_loader.eval_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        write_out=True,
        verbose=False,
    )

    mode = config["mode"]
    if "train" in mode:
        log.info("Starting paper-stack training...")
        trainer.train()
        log.info("Training stage finished. Model dir: %s", config["model_dir"])

    if "test" in mode or mode == "train_test":
        log.info("Evaluating paper stack...")
        metrics = trainer.evaluate()
        log.info("Evaluation metrics: %s", metrics)

    run_meta_path = write_run_metadata(config)
    log.info("Saved run metadata to %s", run_meta_path)


if __name__ == "__main__":
    main()
