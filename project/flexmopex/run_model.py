from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dmg.core.data.samplers.hydro_sampler import HydroSampler
from dmg.core.utils import import_data_loader, set_randomseed

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent.parent

for path in (REPO_ROOT, PROJECT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from project.bettermodel.implements.my_trainer import MyTrainer  # noqa: E402
from project.flexmopex import get_env_path, load_config  # noqa: E402
from project.flexmopex.models.pub_trainer import PubTrainer  # noqa: E402  (local copy, avoids cartopy dep)
from project.flexmopex.models.pub_sampler import PubSampler  # noqa: E402
from project.flexmopex.local_model_handler import FlexMopexModelHandler  # noqa: E402
from project.flexmopex.models.nse_aic_batch_loss import NseAicBatchLoss  # noqa: E402
from project.flexmopex.models.nse_dyn_aic_batch_loss import NseDynAicBatchLoss  # noqa: E402

BASIN_GROUPS_DIR = get_env_path("BASIN_GROUPS_DIR")
TOTAL_BASINS = 671


RUNTIME_PATH_KEYS = (
    "output_dir",
    "model_dir",
    "plot_dir",
    "sim_dir",
    "save_path",
    "model_path",
    "out_path",
)

LOSS_REGISTRY = {
    "NseAicBatchLoss": NseAicBatchLoss,
    "NseDynAicBatchLoss": NseDynAicBatchLoss,
}


class FlexMopexSampler(HydroSampler):
    """HydroSampler variant that passes day-of-year inputs to MOPEX models."""

    def get_training_sample(
        self,
        dataset: dict[str, torch.Tensor],
        ngrid_train: int,
        nt: int,
    ) -> dict[str, torch.Tensor]:
        batch_size = self.config["train"]["batch_size"]
        from dmg.core.data.data import random_index

        i_sample, i_t = random_index(
            ngrid_train,
            nt,
            (batch_size, self.rho),
            warmup=self.warmup,
        )
        # Re-select all tensors with the same indices so doy stays aligned.
        sample = {
            "x_phy": self.select_subset(dataset["x_phy"], i_sample, i_t),
            "c_phy": dataset["c_phy"][i_sample],
            "c_nn": dataset["c_nn"][i_sample],
            "xc_nn_norm": self.select_subset(
                dataset["xc_nn_norm"],
                i_sample,
                i_t,
                has_grad=False,
            ),
            "target": self.select_subset(dataset["target"], i_sample, i_t, warmup=0),
            "batch_sample": i_sample,
        }
        if "doy" in dataset:
            sample["doy"] = self.select_subset(dataset["doy"], i_sample, i_t)
        return sample


class FlexMopexPubSampler(FlexMopexSampler):
    """FlexMopexSampler variant for PubTrainer (LORO): samples only from train basins."""

    def __init__(self, config: dict[str, Any], val_indices: list[int]) -> None:
        super().__init__(config)
        self.val_indices = val_indices
        # Build train indices = all basins except val_indices
        val_set = set(val_indices)
        self._train_basin_indices: list[int] = config.get(
            "train_basin_indices",
            [i for i in range(TOTAL_BASINS) if i not in val_set],
        )

    def get_training_sample(
        self,
        dataset: dict[str, torch.Tensor],
        ngrid_train: int,
        nt: int,
    ) -> dict[str, torch.Tensor]:
        # Override ngrid_train with number of actual train basins
        n_train = len(self._train_basin_indices)
        batch_size = self.config["train"]["batch_size"]
        from dmg.core.data.data import random_index

        i_sample_local, i_t = random_index(
            n_train,
            nt,
            (batch_size, self.rho),
            warmup=self.warmup,
        )
        # Map local train indices back to global basin indices
        train_idx_tensor = torch.tensor(self._train_basin_indices, dtype=torch.long)
        i_sample = train_idx_tensor[i_sample_local]

        sample = {
            "x_phy": self.select_subset(dataset["x_phy"], i_sample, i_t),
            "c_phy": dataset["c_phy"][i_sample],
            "c_nn": dataset["c_nn"][i_sample],
            "xc_nn_norm": self.select_subset(
                dataset["xc_nn_norm"],
                i_sample,
                i_t,
                has_grad=False,
            ),
            "target": self.select_subset(dataset["target"], i_sample, i_t, warmup=0),
            "batch_sample": i_sample,
        }
        if "doy" in dataset:
            sample["doy"] = self.select_subset(dataset["doy"], i_sample, i_t)
        return sample


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run flexmopex training/evaluation from a single entrypoint.",
    )
    parser.add_argument(
        "alpha_pos",
        nargs="?",
        type=float,
        help="Backward-compatible positional AIC alpha, e.g. python run_model.py 0.1.",
    )
    parser.add_argument(
        "--config",
        default="conf/config_dmopex_v1.yaml",
        help="Path to the YAML config file, relative to project/flexmopex by default.",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Override loss_function.aic_alpha.")
    parser.add_argument(
        "--mode",
        choices=("train", "test", "train_test"),
        default=None,
        help="Override config mode.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random_seed.")
    parser.add_argument("--gpu-id", type=int, default=None, help="Override gpu_id.")
    parser.add_argument("--test-epoch", type=int, default=None, help="Override test.test_epoch.")
    parser.add_argument("--start-epoch", type=int, default=None, help="Override train.start_epoch.")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train/test batch_size.")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_DIR / "outputs"),
        help="Root directory for new flexmopex runs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run folder name under output-root.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Build data/model/loss and run one forward+loss check without training.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable verbose ModelHandler/trainer output.",
    )
    parser.add_argument(
        "--model-type",
        choices=("base", "full", "flex"),
        default="flex",
        help=(
            "Model type: 'base' = FixedWeightMopex(all weights=0), "
            "'full' = FixedWeightMopex(all weights=1), "
            "'flex' = LearnedWeightMopex (default)."
        ),
    )
    parser.add_argument(
        "--loro-holdout-region",
        type=int,
        default=None,
        dest="loro_holdout_region",
        help=(
            "Leave-one-region-out holdout region index (0-6). "
            "Loads holdout basin indices from basin_groups/group_{11+region}.npy. "
            "Uses PubTrainer for LORO training."
        ),
    )
    return parser.parse_args(argv)


def _resolve_config(config_path: str) -> str:
    path = Path(config_path)
    if not path.is_absolute():
        project_relative = PROJECT_DIR / path
        path = project_relative if project_relative.exists() else Path.cwd() / path
    return str(path.resolve())


def _loss_name(config: dict[str, Any]) -> str | None:
    loss_config = config.get("loss_function") or config.get("train", {}).get("loss_function") or {}
    if isinstance(loss_config, str):
        return loss_config
    return loss_config.get("name") or loss_config.get("model")


def _set_loss_name(config: dict[str, Any], loss_name: str) -> None:
    loss_config = config.setdefault("loss_function", {})
    loss_config["model"] = loss_name
    loss_config["name"] = loss_name
    config.setdefault("train", {})["loss_function"] = {"name": loss_name, "model": loss_name}


def _alpha_label(alpha: float | None) -> str:
    if alpha is None:
        return "alpha_config"
    return f"alpha_{alpha:g}".replace(".", "_")


def _refresh_runtime_paths(config: dict[str, Any], output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    path_values = {
        "output_dir": output_dir,
        "model_dir": output_dir / "model",
        "plot_dir": output_dir / "plot",
        "sim_dir": output_dir / "sim",
        "save_path": output_dir,
        "model_path": output_dir / "model",
        "out_path": output_dir / "sim",
        "trained_model": output_dir / "model",
    }
    for key, value in path_values.items():
        config[key] = str(value)
    for key in RUNTIME_PATH_KEYS:
        Path(config[key]).mkdir(parents=True, exist_ok=True)


def _load_loro_basin_split(region_id: int) -> int:
    """Return the group_id for a LORO region (11 + region_id).

    PubSampler uses config['test']['test_group_id'] to load the npy file and
    builds train/val index splits internally via gage_id.npy.
    """
    return 11 + region_id


def _align_loro_eval_time(config: dict[str, Any]) -> None:
    """Use the training period for LORO evaluation so the split is purely spatial."""
    train_cfg = config["train"]
    test_cfg = config.setdefault("test", {})
    test_cfg["start_time"] = train_cfg["start_time"]
    test_cfg["end_time"] = train_cfg["end_time"]
    config["test_time"] = [train_cfg["start_time"], train_cfg["end_time"]]


def apply_runtime_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    config_path: str,
) -> None:
    alpha = args.alpha if args.alpha is not None else args.alpha_pos
    config["trainer"] = "MyTrainer"

    if args.mode is not None:
        config["mode"] = args.mode
    if args.seed is not None:
        config["seed"] = args.seed
        config["random_seed"] = args.seed
    if args.gpu_id is not None:
        config["gpu_id"] = args.gpu_id
        if str(config.get("device", "")).startswith("cuda"):
            config["device"] = f"cuda:{args.gpu_id}"
            if torch.cuda.is_available():
                torch.cuda.set_device(config["device"])
    if args.test_epoch is not None:
        config.setdefault("test", {})["test_epoch"] = args.test_epoch
    if args.start_epoch is not None:
        config.setdefault("train", {})["start_epoch"] = args.start_epoch
    if args.epochs is not None:
        config.setdefault("train", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("train", {})["batch_size"] = args.batch_size
        config.setdefault("test", {})["batch_size"] = args.batch_size

    loss_name = _loss_name(config)
    if loss_name:
        _set_loss_name(config, loss_name)
    if alpha is not None:
        config.setdefault("loss_function", {})["aic_alpha"] = alpha

    # Apply --model-type overrides
    model_type = args.model_type
    if model_type == "base":
        config.setdefault("delta_model", {}).setdefault("phy_model", {})["model"] = ["FixedWeightMopex"]
        config["delta_model"]["phy_model"]["fixed_weights"] = {
            "w_phen": 0.0,
            "w_int": 0.0,
            "w_snow": 0.0,
            "w_sub": 0.0,
        }
    elif model_type == "full":
        config.setdefault("delta_model", {}).setdefault("phy_model", {})["model"] = ["FixedWeightMopex"]
        config["delta_model"]["phy_model"]["fixed_weights"] = {
            "w_phen": 1.0,
            "w_int": 1.0,
            "w_snow": 1.0,
            "w_sub": 1.0,
        }
    else:  # flex
        config.setdefault("delta_model", {}).setdefault("phy_model", {})["model"] = ["LearnedWeightMopex"]

    # Apply --loro-holdout-region overrides
    region_id = args.loro_holdout_region
    if region_id is not None:
        group_id = _load_loro_basin_split(region_id)
        config["loro_holdout_region"] = region_id
        config.setdefault("test", {})["test_group_id"] = group_id
        _align_loro_eval_time(config)
        os.environ.setdefault("DATA_PATH", str(BASIN_GROUPS_DIR.parent))

    # Build run_name
    config_stem = Path(config_path).stem
    if args.run_name:
        run_name = args.run_name
    elif region_id is not None:
        run_name = f"{config_stem}/{model_type}_region{region_id}/seed_{config['seed']}"
    else:
        run_name = f"{config_stem}/{model_type}_{_alpha_label(alpha)}/seed_{config['seed']}"
    _refresh_runtime_paths(config, Path(args.output_root) / run_name)


def _build_data_loader(config: dict[str, Any]):
    loader_config = copy.deepcopy(config)
    loader_config["device"] = "cpu"
    data_loader_cls = import_data_loader(config["data_loader"])
    data_loader = data_loader_cls(loader_config, test_split=True, overwrite=False)
    _attach_doy(data_loader.train_dataset, config["train"])
    _attach_doy(data_loader.eval_dataset, config["test"])
    return data_loader


def _daily_doy_tensor(
    start_time: str,
    end_time: str,
    *,
    n_basins: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    start = datetime.strptime(start_time, "%Y/%m/%d")
    end = datetime.strptime(end_time, "%Y/%m/%d")
    n_days = (end - start).days + 1
    values = [float((start + timedelta(days=i)).timetuple().tm_yday) for i in range(n_days)]
    doy = torch.tensor(values, dtype=torch.float32, device=device).view(n_days, 1, 1)
    return doy.repeat(1, n_basins, 1)


def _attach_doy(dataset: dict[str, torch.Tensor], time_config: dict[str, Any]) -> None:
    if "doy" in dataset:
        return
    reference = dataset["x_phy"]
    doy = _daily_doy_tensor(
        time_config["start_time"],
        time_config["end_time"],
        n_basins=reference.shape[1],
        device=reference.device,
    )
    if doy.shape[0] != reference.shape[0]:
        doy = doy[: reference.shape[0]]
    dataset["doy"] = doy


def _build_loss(config: dict[str, Any], train_dataset: dict[str, torch.Tensor]) -> torch.nn.Module:
    loss_name = _loss_name(config)
    if loss_name not in LOSS_REGISTRY:
        available = ", ".join(sorted(LOSS_REGISTRY))
        raise ValueError(f"Unsupported flexmopex loss '{loss_name}'. Available: {available}")
    loss_config = copy.deepcopy(config["loss_function"])
    loss_config.setdefault("name", loss_name)
    loss_config.setdefault("model", loss_name)
    return LOSS_REGISTRY[loss_name](
        loss_config,
        config["device"],
        y_obs=train_dataset["target"],
        aic_alpha=loss_config.get("aic_alpha", 0.0),
    )


def _run_model_preflight(
    model: FlexMopexModelHandler,
    loss_func: torch.nn.Module,
    train_dataset: dict[str, Any],
    config: dict[str, Any],
) -> None:
    sampler = FlexMopexSampler(config)
    n_basins = train_dataset["xc_nn_norm"].shape[1]
    n_timesteps = train_dataset["xc_nn_norm"].shape[0]
    sample = sampler.get_training_sample(train_dataset, n_basins, n_timesteps)

    _ = model(sample)
    loss = model.calc_loss(sample, loss_func=loss_func)
    if not torch.isfinite(loss):
        raise RuntimeError(f"FlexMOPEX preflight produced non-finite loss: {loss.item()}")
    loss.backward()
    for parameter in model.get_parameters():
        parameter.grad = None
    print(f"Preflight passed. loss={loss.item():.6f}")


def run_train(config: dict[str, Any], verbose: bool, *, preflight_only: bool = False) -> None:
    config["mode"] = "train"
    set_randomseed(config["random_seed"])
    data_loader = _build_data_loader(config)
    model = FlexMopexModelHandler(config, verbose=verbose)
    loss_func = _build_loss(config, data_loader.train_dataset)
    _run_model_preflight(model, loss_func, data_loader.train_dataset, config)
    if preflight_only:
        return
    trainer = MyTrainer(
        config,
        model,
        train_dataset=data_loader.train_dataset,
        loss_func=loss_func,
        verbose=verbose,
    )
    trainer.sampler = FlexMopexSampler(config)
    print("Training model...")
    trainer.train()
    print(f"Training complete. Model saved to \n{config['model_path']}")


def run_loro_train(config: dict[str, Any], verbose: bool, *, preflight_only: bool = False) -> None:
    """LORO training using PubTrainer + PubSampler with LORO basin splits."""
    config["mode"] = "train"
    set_randomseed(config["random_seed"])

    data_loader = _build_data_loader(config)
    model = FlexMopexModelHandler(config, verbose=verbose)
    loss_func = _build_loss(config, data_loader.train_dataset)

    # Preflight check (uses all basins — just verifies forward/loss)
    _run_model_preflight(model, loss_func, data_loader.train_dataset, config)
    if preflight_only:
        return

    # PubSampler reads config['test']['test_group_id'] and DATA_PATH env var
    # to build train/val index splits from basin IDs in group_XX.npy + gage_id.npy
    sampler = PubSampler(config)
    n_train = len(sampler.train_indices)
    n_val = len(sampler.val_indices)
    print(
        "LORO temporal split: "
        f"train={config['train']['start_time']}..{config['train']['end_time']}, "
        f"eval={config['test']['start_time']}..{config['test']['end_time']}"
    )

    trainer = PubTrainer(
        config,
        model,
        train_dataset=data_loader.train_dataset,
        loss_func=loss_func,
        verbose=verbose,
    )
    trainer.sampler = sampler
    print(f"LORO training: {n_train} train basins, {n_val} holdout basins...")
    trainer.train()
    print(f"Training complete. Model saved to \n{config['model_path']}")

    # Evaluate on holdout (val) basins
    print("Evaluating on holdout basins...")
    eval_trainer = PubTrainer(
        config,
        model,
        train_dataset=data_loader.train_dataset,
        eval_dataset=data_loader.eval_dataset,
        loss_func=loss_func,
        verbose=verbose,
    )
    eval_trainer.sampler = sampler
    eval_trainer.evaluate()
    print(f"Metrics and predictions saved to \n{config['out_path']}")


def run_test(config: dict[str, Any], verbose: bool) -> None:
    config["mode"] = "test"
    set_randomseed(config["random_seed"])
    data_loader = _build_data_loader(config)
    model = FlexMopexModelHandler(config, verbose=verbose)
    loss_func = _build_loss(config, data_loader.train_dataset)
    trainer = MyTrainer(
        config,
        model,
        train_dataset=data_loader.train_dataset,
        eval_dataset=data_loader.eval_dataset,
        loss_func=loss_func,
        verbose=verbose,
    )
    trainer.sampler = FlexMopexSampler(config)
    print("Evaluating model...")
    trainer.evaluate()
    print(f"Metrics and predictions saved to \n{config['out_path']}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = _resolve_config(args.config)
    config = load_config(config_path)
    apply_runtime_overrides(config, args, config_path=config_path)

    # LORO path: use PubTrainer
    if args.loro_holdout_region is not None:
        mode = config.get("mode", "train")
        if mode in ("train", "train_test"):
            run_loro_train(config, args.verbose, preflight_only=args.preflight_only)
        elif mode == "test":
            run_test(config, args.verbose)
        else:
            raise ValueError(f"Unsupported mode for LORO: {mode!r}")
        return

    # Standard path
    mode = config.get("mode", "train")
    if mode == "train_test":
        run_train(config, args.verbose, preflight_only=args.preflight_only)
        if not args.preflight_only:
            run_test(config, args.verbose)
    elif mode == "train":
        run_train(config, args.verbose, preflight_only=args.preflight_only)
    elif mode == "test":
        run_test(config, args.verbose)
    else:
        raise ValueError(f"Unsupported mode: {mode!r}")


if __name__ == "__main__":
    main()
