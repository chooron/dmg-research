"""Train/test dmotpy hydrology models with FasterTrainer."""

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

# dmotpy imports
from dmotpy.models import HydrologyModel
from dmotpy.neural_networks.calibrate import Calibrate
from dmotpy.neural_networks.parameterize import Parameterize
from dmotpy.trainers import FasterTrainer

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(REPO_ROOT))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_dmotpy")


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
        observations_cfg["data_path"] = _resolve_input_path(
            observations_cfg["data_path"]
        )

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
        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}'. Only Adam is supported."
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
    )

    lr_cfg = config["train"].get("lr_scheduler", {})
    scheduler_name = (
        lr_cfg.get("name", "CosineAnnealingLR") if isinstance(lr_cfg, dict) else lr_cfg
    )
    if scheduler_name != "CosineAnnealingLR":
        log.info(
            "Skipping custom scheduler wiring for unsupported scheduler '%s'.",
            scheduler_name,
        )
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


def _build_physical_model(config: dict) -> HydrologyModel:
    """Build dmotpy hydrology model."""
    model_config = config["model"]["phy"]
    # Get model name from phy.name (list) or phy.model_name
    if "name" in model_config:
        name = model_config["name"]
        if isinstance(name, (list, tuple)):
            model_name = name[0]
        else:
            model_name = name
    else:
        model_name = model_config["model_name"]

    phy_model = HydrologyModel(
        config={
            "model_name": model_name,
            "warm_up": model_config.get("warm_up", config["model"].get("warm_up", 365)),
        },
        device=config["device"],
    )
    log.info(
        "Built physical model: %s (%d parameters, %d states)",
        model_name,
        len(phy_model.parameter_bounds),
        phy_model.n_states,
    )
    return phy_model


def _build_neural_network(config: dict, phy_model: HydrologyModel):
    """Build neural network (Calibrate or Parameterize)."""
    nn_config = config["model"]["nn"]
    nn_name = nn_config["name"]

    # Get input dimensions from config
    nx = len(nn_config.get("attributes", [])) + len(nn_config.get("forcings", []))
    ny = len(config["model"]["phy"].get("forcings", []))

    if nn_name == "Calibrate":
        nn_model = Calibrate(
            nx=nx,
            ny=ny,
            num_basins=nn_config.get("num_basins", 531),
            num_start=nn_config.get("num_start", 10),
            init_strategy=nn_config.get("init_strategy", "lhs_logit"),
            device=config["device"],
        )
    elif nn_name == "Parameterize":
        nn_model = Parameterize(
            nx=nx,
            ny=ny,
            hidden_size=nn_config.get("hidden_size", 128),
            num_layers=nn_config.get("hidden_layers", 2),
            dropout_rate=nn_config.get("dropout", 0.4),
            device=config["device"],
        )
    else:
        raise ValueError(f"Unknown neural network: {nn_name}")

    log.info("Built neural network: %s (nx=%d, ny=%d)", nn_name, nx, ny)
    return nn_model


class DifferentiableModel(torch.nn.Module):
    """Combine neural network and physical model."""

    def __init__(self, nn_model, phy_model):
        super().__init__()
        self.nn_model = nn_model
        self.phy_model = phy_model

    def forward(self, x_dict, eval=False):
        # Neural network predicts parameters
        _, raw_params = self.nn_model(x_dict)
        # Physical model simulation
        output = self.phy_model(x_dict, (None, raw_params))
        return output


def _adapt_dataset_for_dmotpy(dmg_dataset: dict) -> dict:
    """Adapt dmg dataset format to dmotpy expected format."""
    if dmg_dataset is None:
        return {}
    # dmg dataset keys: xc_nn_norm, x_phy, target, etc.
    # dmotpy expects: x_phy, c_nn_norm, target, batch_sample
    adapted = {}
    if "x_phy" in dmg_dataset:
        adapted["x_phy"] = dmg_dataset["x_phy"]
    if "xc_nn_norm" in dmg_dataset:
        adapted["c_nn_norm"] = dmg_dataset["xc_nn_norm"]
    if "target" in dmg_dataset:
        adapted["target"] = dmg_dataset["target"]
    if "batch_sample" in dmg_dataset:
        adapted["batch_sample"] = dmg_dataset["batch_sample"]
    # Add any other keys that might be needed
    for key in dmg_dataset:
        if key not in adapted:
            adapted[key] = dmg_dataset[key]
    return adapted


def parse_args():
    parser = argparse.ArgumentParser(description="Train/test dmotpy hydrology models")
    parser.add_argument(
        "--config",
        default="conf/config_dmotpy_test.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="Override config mode: train | test | train_test",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override phy.model_name in config",
    )
    parser.add_argument(
        "--nn-model",
        default=None,
        choices=["Calibrate", "Parameterize"],
        help="Override nn.name in config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override experiment seed",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train.epochs in config",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    raw_config = OmegaConf.load(_resolve_path(args.config))
    original_epochs = raw_config["train"]["epochs"]

    if args.mode:
        raw_config["mode"] = args.mode
    if args.model_name:
        raw_config["model"]["phy"]["model_name"] = args.model_name
    if args.nn_model:
        raw_config["model"]["nn"]["name"] = args.nn_model
    if args.seed is not None:
        raw_config["seed"] = args.seed
    if args.epochs is not None:
        raw_config["train"]["epochs"] = args.epochs
        lr_cfg = raw_config["train"].get("lr_scheduler")
        if (
            isinstance(lr_cfg, dict)
            and lr_cfg.get("name") == "CosineAnnealingLR"
            and ("T_max" not in lr_cfg or lr_cfg.get("T_max") == original_epochs)
        ):
            lr_cfg["T_max"] = args.epochs
    _normalize_runtime_paths(raw_config)
    config = initialize_config(raw_config)

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

    log.info("Building physical model...")
    phy_model = _build_physical_model(config)

    log.info("Building neural network...")
    nn_model = _build_neural_network(config, phy_model)

    log.info("Building differentiable model...")
    model = DifferentiableModel(nn_model, phy_model)
    model = model.to(config["device"])

    optimizer, scheduler = _build_optimizer_and_scheduler(config, model)

    # Adapt dataset format for dmotpy
    train_dataset = _adapt_dataset_for_dmotpy(data_loader.train_dataset)
    eval_dataset = _adapt_dataset_for_dmotpy(data_loader.eval_dataset)

    trainer = FasterTrainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        verbose=True,
    )

    mode = config["mode"]
    if "train" in mode:
        log.info("Starting training...")
        trainer.train()
        log.info("Training finished. Model dir: %s", config["model_dir"])

    if "test" in mode or mode == "train_test":
        log.info("Evaluating...")
        metrics = trainer.evaluate()
        log.info("Evaluation metrics: %s", metrics)
        log.info("Metrics saved to %s", config["output_dir"])


if __name__ == "__main__":
    main()
