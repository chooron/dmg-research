from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dmg.core.utils import import_data_loader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.bettermodel import load_config
from project.bettermodel.local_model_handler import LocalModelHandler


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "parameters" / "results"
GAGE_ID_PATH = PROJECT_DIR.parent.parent / "data" / "gage_id.npy"
DEFAULT_WINDOW_LENGTH = 730
DEFAULT_TEST_EPOCH = 100
DEFAULT_BATCH_SIZE = 32
SEED_PRIORITY = (111, 42, 222, 333, 444, 555)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    config_path: str
    group: str


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("lstm", "LSTM", "conf/config_dhbv_lstm.yaml", "core"),
    ModelSpec("s4d", "S4D", "conf/config_dhbv_hopev1.yaml", "core"),
    ModelSpec("s5dv1", "S5Dv1", "conf/config_dhbv_hopev2.yaml", "core"),
    ModelSpec("s5dv2", "S5Dv2", "conf/config_dhbv_hopev3.yaml", "core"),
    ModelSpec("transformer", "Transformer", "conf/config_dhbv_transformer.yaml", "other"),
    ModelSpec("timemixer", "TimeMixer", "conf/config_dhbv_tsmixer.yaml", "other"),
    ModelSpec("tcn", "TCN", "conf/config_dhbv_tcn.yaml", "other"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract normalized dynamic parameter trajectories for all 671 CAMELS basins.",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=[spec.key for spec in MODEL_SPECS],
        default=None,
        help="Model key to extract. Repeat to select multiple models. Default: all.",
    )
    parser.add_argument(
        "--seed",
        default="auto",
        help="Seed to load. Use an integer or 'auto' to choose the first common available seed.",
    )
    parser.add_argument("--test-epoch", type=int, default=DEFAULT_TEST_EPOCH)
    parser.add_argument("--window-length", type=int, default=DEFAULT_WINDOW_LENGTH)
    parser.add_argument(
        "--window-anchor",
        choices=("tail", "head"),
        default="tail",
        help="Whether to take the first or last N evaluation days.",
    )
    parser.add_argument("--basin-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    return parser.parse_args()


def _selected_specs(model_keys: list[str] | None) -> list[ModelSpec]:
    if not model_keys:
        return list(MODEL_SPECS)
    key_set = set(model_keys)
    return [spec for spec in MODEL_SPECS if spec.key in key_set]


def _set_seed_paths(config: dict[str, Any], seed: int) -> None:
    original_seed = str(config.get("seed", config.get("random_seed", seed)))
    replacements = {
        "seed": seed,
        "random_seed": seed,
    }
    config.update(replacements)
    for key in ("output_dir", "model_dir", "plot_dir", "sim_dir", "save_path", "model_path", "out_path"):
        value = config.get(key)
        if isinstance(value, str):
            config[key] = value.replace(f"seed_{original_seed}", f"seed_{seed}")


def _checkpoint_exists(config: dict[str, Any], test_epoch: int) -> bool:
    phy_name = config["model"]["phy"]["name"][0].lower()
    checkpoint = Path(config["model_dir"]) / f"{phy_name}_ep{test_epoch}.pt"
    return checkpoint.exists()


def _available_seeds(spec: ModelSpec, test_epoch: int) -> list[int]:
    config = load_config(str(PROJECT_DIR / spec.config_path))
    root = Path(config["output_dir"]).parent
    seeds: list[int] = []
    if not root.exists():
        return seeds
    for seed_dir in sorted(root.glob("seed_*")):
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        probe = copy.deepcopy(config)
        _set_seed_paths(probe, seed)
        if _checkpoint_exists(probe, test_epoch):
            seeds.append(seed)
    return seeds


def _resolve_seed(specs: list[ModelSpec], requested_seed: str, test_epoch: int) -> int:
    if requested_seed != "auto":
        return int(requested_seed)

    available_by_model = {spec.key: set(_available_seeds(spec, test_epoch)) for spec in specs}
    common = set.intersection(*(seeds for seeds in available_by_model.values())) if available_by_model else set()
    for seed in SEED_PRIORITY:
        if seed in common:
            return seed
    if common:
        return min(common)

    for seed in SEED_PRIORITY:
        if all(seed in seeds for seeds in available_by_model.values()):
            return seed
    raise FileNotFoundError(
        "Could not find a common seed with available checkpoints for the requested models."
    )


def _resolve_device(config: dict[str, Any]) -> str:
    requested = str(config.get("device", "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _load_eval_dataset(config: dict[str, Any]) -> dict[str, torch.Tensor]:
    loader_config = copy.deepcopy(config)
    loader_config["device"] = "cpu"
    data_loader_cls = import_data_loader(config["data_loader"])
    loader = data_loader_cls(loader_config, test_split=True, overwrite=False)
    loader.load_dataset()
    return loader.eval_dataset


def _window_slice(time_steps: int, window_length: int, anchor: str) -> slice:
    if window_length <= 0 or window_length >= time_steps:
        return slice(0, time_steps)
    if anchor == "head":
        return slice(0, window_length)
    return slice(time_steps - window_length, time_steps)


def _extract_dynamic_tensor(
    nn_model: torch.nn.Module,
    xc_nn_norm: torch.Tensor,
    c_nn: torch.Tensor,
    *,
    device: str,
    basin_batch_size: int,
    n_parameters: int,
) -> np.ndarray:
    if xc_nn_norm.ndim != 3:
        raise ValueError(f"Expected xc_nn_norm with 3 dims, got {tuple(xc_nn_norm.shape)}")

    chunks: list[torch.Tensor] = []
    nn_model.eval()
    with torch.no_grad():
        for start in range(0, xc_nn_norm.shape[1], basin_batch_size):
            end = min(start + basin_batch_size, xc_nn_norm.shape[1])
            dynamic_out, _ = nn_model(
                {
                    "xc_nn_norm": xc_nn_norm[:, start:end].to(device),
                    "c_nn_norm": c_nn[start:end].to(device),
                }
            )
            if dynamic_out.ndim != 3:
                raise ValueError(
                    f"Expected dynamic output with 3 dims, got {tuple(dynamic_out.shape)}"
                )
            chunks.append(dynamic_out.detach().cpu())

    dynamic_tensor = torch.cat(chunks, dim=1)
    if dynamic_tensor.shape[-1] % n_parameters != 0:
        raise ValueError(
            "Dynamic output channel count is not divisible by the number of dynamic parameters: "
            f"{dynamic_tensor.shape[-1]} vs {n_parameters}"
        )
    n_components = dynamic_tensor.shape[-1] // n_parameters
    reshaped = dynamic_tensor.reshape(
        dynamic_tensor.shape[0],
        dynamic_tensor.shape[1],
        n_parameters,
        n_components,
    )
    return reshaped.numpy()


def _save_npz(
    output_dir: Path,
    spec: ModelSpec,
    seed: int,
    config: dict[str, Any],
    normalized_parameters: np.ndarray,
    basin_ids: np.ndarray,
    time_slice: slice,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{spec.key}_seed{seed}_normalized_dynamic_parameters.npz"
    metadata = {
        "model_key": spec.key,
        "model_label": spec.label,
        "config_path": spec.config_path,
        "seed": seed,
        "test_epoch": int(config["test"]["test_epoch"]),
        "window_start": int(time_slice.start or 0),
        "window_stop": int(time_slice.stop or normalized_parameters.shape[0]),
        "window_length": int(normalized_parameters.shape[0]),
        "parameter_names": list(config["model"]["phy"]["dynamic_params"][config["model"]["phy"]["name"][0]]),
        "dtype": "float16",
        "shape": list(normalized_parameters.shape),
    }
    np.savez_compressed(
        output_path,
        normalized_parameters=normalized_parameters.astype(np.float16),
        basin_ids=np.asarray(basin_ids, dtype=np.int64),
        parameter_names=np.asarray(metadata["parameter_names"], dtype=object),
        model_key=np.asarray(spec.key),
        model_label=np.asarray(spec.label),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return output_path


def extract_one_model(
    spec: ModelSpec,
    *,
    seed: int,
    test_epoch: int,
    window_length: int,
    window_anchor: str,
    basin_batch_size: int,
    output_dir: Path,
    basin_ids: np.ndarray,
) -> Path:
    config = load_config(str(PROJECT_DIR / spec.config_path))
    _set_seed_paths(config, seed)
    config["mode"] = "test"
    config["test"]["test_epoch"] = test_epoch
    config["device"] = _resolve_device(config)

    if not _checkpoint_exists(config, test_epoch):
        raise FileNotFoundError(
            f"Missing checkpoint for {spec.label}: {Path(config['model_dir'])}"
        )

    eval_dataset = _load_eval_dataset(config)
    time_slice = _window_slice(eval_dataset["xc_nn_norm"].shape[0], window_length, window_anchor)
    parameter_names = list(config["model"]["phy"]["dynamic_params"][config["model"]["phy"]["name"][0]])
    n_parameters = len(parameter_names)

    handler = LocalModelHandler(config, verbose=False)
    phy_name = config["model"]["phy"]["name"][0]
    nn_model = handler.model_dict[phy_name].nn_model

    dynamic_tensor = _extract_dynamic_tensor(
        nn_model,
        eval_dataset["xc_nn_norm"][time_slice],
        eval_dataset["c_nn"],
        device=config["device"],
        basin_batch_size=basin_batch_size,
        n_parameters=n_parameters,
    )

    output_path = _save_npz(
        output_dir=output_dir,
        spec=spec,
        seed=seed,
        config=config,
        normalized_parameters=dynamic_tensor,
        basin_ids=basin_ids,
        time_slice=time_slice,
    )

    print(
        f"{spec.label}: seed={seed}, shape={tuple(dynamic_tensor.shape)}, "
        f"parameters={parameter_names}, wrote={output_path}"
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_path


def main() -> None:
    args = _parse_args()
    specs = _selected_specs(args.model)
    output_dir = Path(args.output_dir).resolve()
    basin_ids = np.load(GAGE_ID_PATH, allow_pickle=True)
    seed = _resolve_seed(specs, str(args.seed), args.test_epoch)
    print(f"Using seed {seed} for models: {[spec.label for spec in specs]}")

    for spec in specs:
        extract_one_model(
            spec,
            seed=seed,
            test_epoch=args.test_epoch,
            window_length=args.window_length,
            window_anchor=args.window_anchor,
            basin_batch_size=args.basin_batch_size,
            output_dir=output_dir,
            basin_ids=basin_ids,
        )


if __name__ == "__main__":
    main()
