"""Train/test dmotpy hydrology models with FasterTrainer."""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from dmg.core.data.loaders import HydroLoader
from dmg.core.utils import print_config, set_randomseed
from dmg.core.utils.utils import initialize_config
from omegaconf import OmegaConf

# dmotpy imports
from dmotpy.models import HydrologyModel
from dmotpy.data_contract import (
    add_calendar_forcing,
    attach_training_mask,
    dataset_manifest,
    write_manifest,
)
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


def _build_neural_network(config: dict, phy_model: HydrologyModel, num_basins: int | None = None):
    """Build neural network (Calibrate or Parameterize)."""
    nn_config = config["model"]["nn"]
    nn_name = nn_config["name"]

    # Get input dimensions from config
    nx = len(nn_config.get("attributes", [])) + len(nn_config.get("forcings", []))
    # The NN output dimension is the physical parameter count, not the
    # number of forcing channels.  The latter happened to mask this bug for
    # small models but makes MOPEX4/5 index past the raw parameter tensor.
    ny = len(phy_model.parameter_bounds)

    if nn_name == "Calibrate":
        configured_basins = nn_config.get("num_basins")
        effective_basins = num_basins or configured_basins or 531
        if configured_basins is not None and num_basins is not None and int(configured_basins) != int(num_basins):
            log.warning("Overriding stale num_basins=%s with loader basin count=%s", configured_basins, num_basins)
        nn_model = Calibrate(
            nx=nx,
            ny=ny,
            num_basins=effective_basins,
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
        self.model_name = phy_model.model_name
        self.config = getattr(phy_model, "config", {})
        self.output_dict = {}
        self.model_dict = {self.model_name: self}
        self.loss_func = None
        self.loss_dict = {self.model_name: 0.0}

    def forward(self, x_dict, eval=False):
        # Neural network predicts parameters
        _, raw_params = self.nn_model(x_dict)
        # Physical model simulation
        output = self.phy_model(x_dict, (None, raw_params))
        self.output_dict = {self.model_name: output}
        return self.output_dict

    def get_parameters(self):
        return list(self.parameters())

    def calc_loss(self, dataset_sample, loss_func=None):
        criterion = loss_func or self.loss_func
        if criterion is None:
            raise ValueError("No loss function defined")
        output = self.output_dict[self.model_name]["streamflow"]
        target = dataset_sample["target"]
        mask = dataset_sample.get("mask", torch.isfinite(target))
        if output.ndim > 2 and output.shape[-1] == 1:
            output = output.squeeze(-1)
        if target.ndim > 2 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        if mask.ndim > 2 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        n = min(output.shape[0], target.shape[0])
        output, target, mask = output[-n:], target[-n:], mask[-n:]
        return criterion(
            output,
            target,
            mask=mask,
            sample_ids=dataset_sample.get("batch_sample"),
            basin_ids=dataset_sample.get("batch_sample"),
            time_index=dataset_sample.get("time_index"),
        )

    def save_model(self, epoch: int) -> None:
        model_dir = Path(self.config.get("model_dir", "."))
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), model_dir / f"model_epoch{int(epoch)}.pt")

    def load_model(self, epoch: int = 0) -> None:
        model_dir = Path(self.config.get("model_dir", "."))
        checkpoint = model_dir / f"model_epoch{int(epoch)}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        self.load_state_dict(torch.load(checkpoint, map_location=self.phy_model.device, weights_only=True), strict=True)

    def export_hydrological_state(self):
        return {}

    def export_uh_state(self):
        return {name: value.detach().cpu() for name, value in self.named_buffers()}

    def export_warmup_state(self):
        return {"warm_up": int(getattr(self.phy_model, "warm_up", 0))}


def _adapt_dataset_for_dmotpy(
    dmg_dataset: dict,
    config: dict,
    scope: str,
) -> dict:
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

    target = adapted.get("target")
    x_phy = adapted.get("x_phy")
    if isinstance(target, torch.Tensor) and isinstance(x_phy, torch.Tensor):
        n_basins = target.shape[1]
        adapted["batch_sample"] = torch.arange(n_basins, device=x_phy.device, dtype=torch.long)
        adapted = attach_training_mask(adapted)
        model_cfg = config["model"]["phy"]
        model_name = model_cfg.get("model_name") or model_cfg.get("name")
        if isinstance(model_name, (list, tuple)):
            model_name = model_name[0]
        start, end = config[f"{scope}_time"]
        dates = pd.date_range(start, end, freq="D")
        x_phy, doy = add_calendar_forcing(x_phy, dates, model_name=str(model_name))
        adapted["x_phy"] = x_phy
        if doy is not None:
            adapted["doy"] = doy.expand(-1, n_basins, -1)
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
    parser.add_argument("--batch-size", type=int, default=None, help="Override smoke batch size")
    parser.add_argument("--rho", type=int, default=None, help="Override smoke sequence length")
    parser.add_argument("--warm-up", type=int, default=None, help="Override smoke warm-up length")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit smoke batches per epoch")
    parser.add_argument("--save-epoch", type=int, default=None, help="Override checkpoint interval")
    return parser.parse_args()


def main():
    args = parse_args()

    raw_config = OmegaConf.load(_resolve_path(args.config))
    original_epochs = raw_config["train"]["epochs"]

    if args.mode:
        raw_config["mode"] = args.mode
    if args.model_name:
        raw_config["model"]["phy"]["model_name"] = args.model_name
        raw_config["model"]["phy"]["name"] = [args.model_name]
        raw_config["model"]["phy"]["dynamic_params"] = {args.model_name: []}
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
    if args.batch_size is not None:
        raw_config["train"]["batch_size"] = args.batch_size
    if args.rho is not None:
        raw_config["model"]["rho"] = args.rho
    if args.warm_up is not None:
        raw_config["model"]["warm_up"] = args.warm_up
        raw_config["model"]["warmup"] = args.warm_up
        raw_config["model"]["phy"]["warm_up"] = args.warm_up
    if args.max_batches is not None:
        raw_config["train"]["max_batches"] = args.max_batches
    if args.save_epoch is not None:
        raw_config["train"]["save_epoch"] = args.save_epoch
    _normalize_runtime_paths(raw_config)
    config = initialize_config(raw_config)

    lr_sched = config["train"].get("lr_scheduler")
    print_config(config)

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
    n_data_basins = int(data_loader.train_dataset["x_phy"].shape[1])
    nn_model = _build_neural_network(config, phy_model, num_basins=n_data_basins)

    log.info("Building differentiable model...")
    model = DifferentiableModel(nn_model, phy_model)
    model.config = config
    model = model.to(config["device"])

    optimizer, scheduler = _build_optimizer_and_scheduler(config, model)

    # Adapt dataset format for dmotpy
    train_dataset = _adapt_dataset_for_dmotpy(data_loader.train_dataset, config, "train")
    eval_dataset = _adapt_dataset_for_dmotpy(data_loader.eval_dataset, config, "test")

    manifest = dataset_manifest(
        dataset_name=config["observations"]["name"],
        source_path=config["observations"]["data_path"],
        train_period=tuple(config["train_time"]),
        validation_period=tuple(config["test_time"]),
        test_period=tuple(config["test_time"]),
    )
    config["dataset_manifest_hash"] = write_manifest(
        Path(config["output_dir"]) / "dataset_manifest.json", manifest
    )

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
