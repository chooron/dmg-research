from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dmg.core.utils import import_data_loader, set_randomseed

PROJECT_DIR = Path(__file__).resolve().parents[3]
REPO_ROOT = PROJECT_DIR.parent.parent
DEFAULT_CONFIGS = [
    "conf/config_dhbv_ablation_s4d_baseline.yaml",
    "conf/config_dhbv_ablation_s4d_ln.yaml",
    "conf/config_dhbv_ablation_s4d_softsign.yaml",
    "conf/config_dhbv_ablation_s4d_ln_softsign.yaml",
    "conf/config_dhbv_ablation_s5d_conv_only.yaml",
    "conf/config_dhbv_ablation_s5d_full.yaml",
]

for path in (REPO_ROOT, PROJECT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from project.bettermodel import load_config  # noqa: E402
from project.bettermodel.implements.neural_networks.ablation.diagnostics import (  # noqa: E402
    save_dynamic_parameter_diagnostics,
)
from project.bettermodel.local_model_handler import LocalModelHandler  # noqa: E402


def _resolve_config(config_path: str) -> str:
    path = Path(config_path)
    if not path.is_absolute():
        project_relative = PROJECT_DIR / path
        path = project_relative if project_relative.exists() else Path.cwd() / path
    return str(path.resolve())


def _build_eval_dataset(config: dict[str, Any]) -> dict[str, torch.Tensor]:
    loader_config = copy.deepcopy(config)
    loader_config["device"] = "cpu"
    data_loader_cls = import_data_loader(config["data_loader"])
    return data_loader_cls(loader_config, test_split=True, overwrite=False).eval_dataset


def _dynamic_param_names(config: dict[str, Any]) -> list[str]:
    phy_name = config["model"]["phy"]["name"][0]
    dynamic_params = config["model"]["phy"].get("dynamic_params", {})
    if isinstance(dynamic_params, dict):
        return list(dynamic_params.get(phy_name, []))
    return list(dynamic_params)


def _predict_dynamic_parameters(
    nn_model: torch.nn.Module,
    xc_nn_norm: torch.Tensor,
    *,
    device: str,
    basin_batch_size: int,
    max_days: int | None,
    max_basins: int | None,
) -> np.ndarray:
    if max_days is not None:
        xc_nn_norm = xc_nn_norm[:max_days]
    if max_basins is not None:
        xc_nn_norm = xc_nn_norm[:, :max_basins]

    chunks = []
    nn_model.eval()
    with torch.no_grad():
        for start in range(0, xc_nn_norm.shape[1], basin_batch_size):
            end = min(start + basin_batch_size, xc_nn_norm.shape[1])
            z1 = xc_nn_norm[:, start:end].to(device)
            params = nn_model.predict_timevar_parameters(z1)
            chunks.append(params.detach().cpu())
    return torch.cat(chunks, dim=1).numpy()


def export_config_diagnostics(
    config_path: str,
    *,
    test_epoch: int | None = None,
    basin_batch_size: int = 25,
    max_days: int | None = None,
    max_basins: int | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    config = load_config(_resolve_config(config_path))
    config["mode"] = "test"
    if test_epoch is not None:
        config["test"]["test_epoch"] = test_epoch

    set_randomseed(config["random_seed"])
    eval_dataset = _build_eval_dataset(config)
    handler = LocalModelHandler(config, verbose=False)
    phy_name = config["model"]["phy"]["name"][0]
    nn_model = handler.model_dict[phy_name].nn_model

    parameters = _predict_dynamic_parameters(
        nn_model,
        eval_dataset["xc_nn_norm"],
        device=config["device"],
        basin_batch_size=basin_batch_size,
        max_days=max_days,
        max_basins=max_basins,
    )

    variant = config["paper"]["variant"]
    diagnostics_dir = output_dir or str(Path(config["out_path"]).parent / "ablation_diagnostics")
    return save_dynamic_parameter_diagnostics(
        diagnostics_dir,
        parameters,
        variant=variant,
        parameter_names=_dynamic_param_names(config),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export S4D/S5D ablation dynamic-parameter diagnostics.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Variant config path. Repeat to export multiple variants.",
    )
    parser.add_argument("--test-epoch", type=int, default=None)
    parser.add_argument("--basin-batch-size", type=int, default=25)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--max-basins", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_paths = args.config or DEFAULT_CONFIGS
    for config_path in config_paths:
        stats = export_config_diagnostics(
            config_path,
            test_epoch=args.test_epoch,
            basin_batch_size=args.basin_batch_size,
            max_days=args.max_days,
            max_basins=args.max_basins,
            output_dir=args.output_dir,
        )
        print(f"{stats['variant']}: wrote {stats['trajectory_file']}")


if __name__ == "__main__":
    main()
