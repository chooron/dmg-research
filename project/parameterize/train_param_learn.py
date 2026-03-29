"""Train/evaluate static parameter learning on a fixed 531-basin subset."""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import torch
from dmg.core.data.loaders import HydroLoader
from dmg.core.utils import print_config, set_randomseed
from dmg.core.utils.utils import initialize_config
from omegaconf import OmegaConf

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from implements import ParamLearnTrainer, build_causal_dpl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_param_learn")


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


def _preserve_trailing_separator(original: str, resolved: Path) -> str:
    resolved_str = str(resolved)
    if original.endswith(("/", "\\")):
        return resolved_str.rstrip("/\\") + "/"
    return resolved_str


def _resolve_input_path(path_str: str, base_dir: Path = REPO_ROOT) -> str:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return _preserve_trailing_separator(path_str, path)

    candidates = [
        Path.cwd() / path,
        base_dir / path,
        PROJECT_DIR / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return _preserve_trailing_separator(path_str, candidate.resolve())

    return _preserve_trailing_separator(path_str, (base_dir / path).resolve())


def _resolve_output_path(path_str: str, base_dir: Path = PROJECT_DIR) -> str:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return _preserve_trailing_separator(path_str, path)
    return _preserve_trailing_separator(path_str, (base_dir / path).resolve())


def _normalize_runtime_paths(raw_config) -> None:
    observations_cfg = raw_config.get("observations")
    if observations_cfg and observations_cfg.get("data_path"):
        observations_cfg["data_path"] = _resolve_input_path(observations_cfg["data_path"])

    data_cfg = raw_config.get("data")
    if data_cfg:
        for key in ("basin_ids_path", "basin_ids_reference_path"):
            if data_cfg.get(key):
                data_cfg[key] = _resolve_input_path(data_cfg[key])

    if raw_config.get("output_dir"):
        raw_config["output_dir"] = _resolve_output_path(raw_config["output_dir"])


def _build_loader_config(config: dict) -> dict:
    loader_config = copy.deepcopy(config)
    loader_config["device"] = "cpu"
    return loader_config


def _build_optimizer_and_scheduler(config: dict, model: torch.nn.Module):
    optimizer_name = config["train"].get("optimizer", {}).get("name", "Adam")
    if optimizer_name != "Adam":
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Only Adam is supported.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
    )

    lr_cfg = config["train"].get("lr_scheduler", {})
    scheduler_name = lr_cfg.get("name", "CosineAnnealingLR") if isinstance(lr_cfg, dict) else lr_cfg
    if scheduler_name != "CosineAnnealingLR":
        log.info("Skipping custom scheduler wiring for unsupported scheduler '%s'.", scheduler_name)
        return optimizer, None

    t_max = (
        lr_cfg.get("T_max", config["train"]["epochs"])
        if isinstance(lr_cfg, dict)
        else config["train"]["epochs"]
    )
    eta_min = lr_cfg.get("eta_min", 1e-5) if isinstance(lr_cfg, dict) else 1e-5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_max,
        eta_min=eta_min,
    )
    return optimizer, scheduler


def _validate_static_nn_config(config: dict) -> None:
    phy_cfg = config["model"].get("phy", {})
    nn_cfg = config["model"].get("nn", {})
    nn_name = str(nn_cfg.get("name", "")).lower()
    if nn_name not in {"mcmlpmodel", "mcmlp", "mc_mlp", "fastkan", "fast_kan"}:
        raise ValueError(f"Unsupported static NN model '{nn_cfg.get('name')}'.")

    if nn_cfg.get("forcings"):
        raise ValueError("Static parameter learning expects model.nn.forcings to be [].")

    output_activation = str(nn_cfg.get("output_activation", "sigmoid")).lower()
    if output_activation != "sigmoid":
        raise ValueError(
            "HbvStatic expects normalized parameters in [0, 1]; "
            "use model.nn.output_activation='sigmoid'."
        )

    if int(phy_cfg.get("nmul", 1) or 1) != 1:
        raise ValueError("This setup expects model.phy.nmul == 1.")

    if not nn_cfg.get("attributes"):
        raise ValueError("Static parameter learning requires model.nn.attributes.")


def _run_model_preflight(
    model: torch.nn.Module,
    dataset: dict[str, torch.Tensor],
    config: dict,
) -> None:
    if dataset is None:
        raise ValueError("Training dataset is required for model preflight.")

    if "xc_nn_norm" not in dataset or "x_phy" not in dataset:
        raise KeyError("Preflight expects dataset keys 'xc_nn_norm' and 'x_phy'.")

    xc_nn_norm = dataset["xc_nn_norm"]
    x_phy = dataset["x_phy"]
    if xc_nn_norm.ndim != 3:
        raise ValueError(
            f"xc_nn_norm must have shape [T, B, nx], got {tuple(xc_nn_norm.shape)}."
        )
    if x_phy.ndim != 3:
        raise ValueError(
            f"x_phy must have shape [T, B, n_forcings], got {tuple(x_phy.shape)}."
        )

    expected_nx = (
        len(config["model"]["nn"].get("forcings", []))
        + len(config["model"]["nn"].get("attributes", []))
    )
    if xc_nn_norm.shape[-1] != expected_nx:
        raise ValueError(
            "Dataset/config feature mismatch for xc_nn_norm: "
            f"{xc_nn_norm.shape[-1]} vs expected {expected_nx}."
        )

    expected_forcings = len(config["model"]["phy"].get("forcings", []))
    if x_phy.shape[-1] != expected_forcings:
        raise ValueError(
            "Dataset/config forcing mismatch for x_phy: "
            f"{x_phy.shape[-1]} vs expected {expected_forcings}."
        )

    sample_basin_count = min(2, xc_nn_norm.shape[1])
    sample = {
        "xc_nn_norm": xc_nn_norm[:, :sample_basin_count].to(config["device"]),
        "x_phy": x_phy[:, :sample_basin_count].to(config["device"]),
    }

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            parameters = model.nn_model(sample["xc_nn_norm"])
            expected_ny = model.phy_model.learnable_param_count

            if parameters.ndim != 3:
                raise ValueError(
                    "NN output must have shape [T, B, ny] for the static dPL/HbvStatic "
                    f"interface, got {tuple(parameters.shape)}."
                )
            if parameters.shape[:2] != sample["xc_nn_norm"].shape[:2]:
                raise ValueError(
                    "NN output time/basin dimensions do not match xc_nn_norm: "
                    f"{tuple(parameters.shape[:2])} vs {tuple(sample['xc_nn_norm'].shape[:2])}."
                )
            if parameters.shape[-1] != expected_ny:
                raise ValueError(
                    "NN output size does not match phy_model.learnable_param_count: "
                    f"{parameters.shape[-1]} vs {expected_ny}."
                )
            if not parameters.is_contiguous():
                raise ValueError("Static NN output must be contiguous.")
            if not torch.allclose(parameters[0], parameters[-1]):
                raise ValueError(
                    "Static NN output is expected to be basin-static and repeated across time."
                )

            predictions = model(sample, eval=True)
            if "streamflow" not in predictions:
                raise KeyError("Model prediction dictionary must contain 'streamflow'.")

            streamflow = predictions["streamflow"]
            if streamflow.ndim != 3:
                raise ValueError(
                    f"streamflow must have shape [T_pred, B, 1], got {tuple(streamflow.shape)}."
                )
            if streamflow.shape[1] != sample_basin_count or streamflow.shape[2] != 1:
                raise ValueError(
                    "Unexpected streamflow output shape: "
                    f"{tuple(streamflow.shape)} for basin count {sample_basin_count}."
                )

        log.info(
            "Model preflight passed: xc_nn_norm %s -> parameters %s -> streamflow %s",
            tuple(sample["xc_nn_norm"].shape),
            tuple(parameters.shape),
            tuple(streamflow.shape),
        )
    finally:
        model.train(was_training)


def parse_args():
    parser = argparse.ArgumentParser(description="Static parameter learning training/evaluation")
    parser.add_argument(
        "--config",
        default="conf/config_param_learn.yaml",
        help="Path to dmg config yaml",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="Override config mode: train | test | train_test",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train.epochs in config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override experiment seed",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=None,
        help="Override test.mc_samples in config",
    )
    parser.add_argument(
        "--mc-selection-metric",
        default=None,
        choices=("mse", "rmse", "loss", "kge"),
        help="Override test.mc_selection_metric in config",
    )
    parser.add_argument(
        "--nn-model",
        default=None,
        choices=("mc_mlp", "fast_kan"),
        help="Override config NN model.",
    )
    return parser.parse_args()


def _canonical_nn_name(nn_model: str) -> str:
    return {
        "mc_mlp": "McMlpModel",
        "fast_kan": "FastKAN",
    }[nn_model]


def main():
    args = parse_args()

    raw_config = OmegaConf.load(_resolve_path(args.config))
    original_epochs = raw_config["train"]["epochs"]

    if args.mode:
        raw_config["mode"] = args.mode
    if args.epochs is not None:
        raw_config["train"]["epochs"] = args.epochs
        lr_cfg = raw_config["train"].get("lr_scheduler")
        if (
            isinstance(lr_cfg, dict)
            and lr_cfg.get("name") == "CosineAnnealingLR"
            and ("T_max" not in lr_cfg or lr_cfg.get("T_max") == original_epochs)
        ):
            lr_cfg["T_max"] = args.epochs
    if args.seed is not None:
        raw_config["seed"] = args.seed
    if args.mc_samples is not None:
        raw_config["test"]["mc_samples"] = args.mc_samples
    if args.mc_selection_metric is not None:
        raw_config["test"]["mc_selection_metric"] = args.mc_selection_metric
    if args.nn_model is not None:
        raw_config["model"]["nn"]["name"] = _canonical_nn_name(args.nn_model)

    _normalize_runtime_paths(raw_config)

    config = initialize_config(raw_config)
    _validate_static_nn_config(config)

    lr_sched = config["train"].get("lr_scheduler")
    if isinstance(lr_sched, dict):
        config["train"]["lr_scheduler"] = lr_sched.get("name", str(lr_sched))
    print_config(config)
    if isinstance(lr_sched, dict):
        config["train"]["lr_scheduler"] = lr_sched

    set_randomseed(config["seed"])

    log.info("Loading datasets...")
    data_loader = HydroLoader(
        _build_loader_config(config),
        test_split=True,
        overwrite=False,
    )

    log.info("Building static dPL model...")
    model = build_causal_dpl(config)
    model = model.to(config["device"])
    log.info("Model device: %s", config["device"])
    _run_model_preflight(model, data_loader.train_dataset, config)
    optimizer, scheduler = _build_optimizer_and_scheduler(config, model)

    trainer = ParamLearnTrainer(
        config,
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
        log.info("Starting parameter-learning training...")
        trainer.train()
        log.info("Training stage finished. Model dir: %s", config["model_dir"])

    if "test" in mode or mode == "train_test":
        log.info("Evaluating on the fixed basin subset test split...")
        trainer.evaluate()
        log.info("Metrics saved to %s", config["output_dir"])


if __name__ == "__main__":
    main()
