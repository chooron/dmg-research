from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import torch
from dmg.core.utils import import_data_loader, set_randomseed

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent.parent

for path in (REPO_ROOT, PROJECT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from project.bettermodel import load_config  # noqa: E402
from project.bettermodel.implements.my_trainer import MyTrainer  # noqa: E402
from project.bettermodel.local_model_handler import LocalModelHandler  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bettermodel training/evaluation from a single entrypoint.",
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--mode",
        choices=("train", "test", "train_test"),
        default=None,
        help="Override config mode.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random_seed.")
    parser.add_argument(
        "--test-epoch",
        type=int,
        default=None,
        help="Override test.test_epoch.",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=None,
        help="Override train.start_epoch.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train.epochs.",
    )
    parser.add_argument(
        "--loss",
        default=None,
        help="Override loss_function.model or loss_function.name.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable verbose ModelHandler/trainer output.",
    )
    return parser.parse_args(argv)


def _resolve_config(config_path: str) -> str:
    path = Path(config_path)
    if not path.is_absolute():
        project_relative = PROJECT_DIR / path
        path = project_relative if project_relative.exists() else Path.cwd() / path
    return str(path.resolve())


def _override_loss(config: dict[str, Any], loss_name: str) -> None:
    loss_config = config.setdefault("loss_function", {})
    loss_config["model"] = loss_name
    loss_config["name"] = loss_name
    config.setdefault("train", {})
    config["train"]["loss_function"] = {"name": loss_name, "model": loss_name}


def apply_runtime_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    config["trainer"] = "MyTrainer"
    if args.mode is not None:
        config["mode"] = args.mode
    if args.seed is not None:
        config["seed"] = args.seed
        config["random_seed"] = args.seed
    if args.test_epoch is not None:
        config.setdefault("test", {})["test_epoch"] = args.test_epoch
    if args.start_epoch is not None:
        config.setdefault("train", {})["start_epoch"] = args.start_epoch
    if args.epochs is not None:
        config.setdefault("train", {})["epochs"] = args.epochs
    if args.loss is not None:
        _override_loss(config, args.loss)


def _build_data_loader(config: dict[str, Any]):
    loader_config = copy.deepcopy(config)
    loader_config["device"] = "cpu"
    data_loader_cls = import_data_loader(config["data_loader"])
    return data_loader_cls(loader_config, test_split=True, overwrite=False)


def _run_model_preflight(
    model: LocalModelHandler,
    train_dataset: dict[str, Any],
    config: dict[str, Any],
) -> None:
    sample_steps = min(
        int(config["model"].get("warm_up", 0)) + int(config["model"]["rho"]),
        train_dataset["xc_nn_norm"].shape[0],
    )
    sample = {}
    for key in ("xc_nn_norm", "x_phy", "c_nn_norm", "c_nn"):
        value = train_dataset.get(key)
        if value is None:
            continue
        if value.ndim >= 3:
            sample[key] = value[:sample_steps, :1].to(config["device"])
        else:
            sample[key] = value[:1].to(config["device"])
    if "target" in train_dataset:
        sample["target"] = train_dataset["target"][:sample_steps, :1].to(config["device"])
    if "batch_sample" in train_dataset:
        sample["batch_sample"] = train_dataset["batch_sample"][:1].to(config["device"])

    with torch.no_grad():
        output = model(sample, eval=True)
    model_name = config["model"]["phy"]["name"][0]
    if model_name not in output or "streamflow" not in output[model_name]:
        raise RuntimeError("Bettermodel preflight expected a prediction dict with 'streamflow'.")


def run_train(config: dict[str, Any], verbose: bool) -> None:
    config["mode"] = "train"
    set_randomseed(config["random_seed"])
    data_loader = _build_data_loader(config)
    model = LocalModelHandler(config, verbose=verbose)
    _run_model_preflight(model, data_loader.train_dataset, config)
    trainer = MyTrainer(
        config,
        model,
        train_dataset=data_loader.train_dataset,
        verbose=verbose,
    )
    print("Training model...")
    trainer.train()
    print(f"Training complete. Model saved to \n{config['model_path']}")


def run_test(config: dict[str, Any], verbose: bool) -> None:
    config["mode"] = "test"
    set_randomseed(config["random_seed"])
    data_loader = _build_data_loader(config)
    model = LocalModelHandler(config, verbose=verbose)
    trainer = MyTrainer(
        config,
        model,
        train_dataset=data_loader.train_dataset,
        eval_dataset=data_loader.eval_dataset,
        verbose=verbose,
    )
    print("Evaluating model...")
    trainer.evaluate()
    print(f"Metrics and predictions saved to \n{config['out_path']}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(_resolve_config(args.config))
    apply_runtime_overrides(config, args)

    mode = config.get("mode", "train")
    if mode == "train_test":
        run_train(config, args.verbose)
        run_test(config, args.verbose)
    elif mode == "train":
        run_train(config, args.verbose)
    elif mode == "test":
        run_test(config, args.verbose)
    else:
        raise ValueError(f"Unsupported mode: {mode!r}")


if __name__ == "__main__":
    main()
