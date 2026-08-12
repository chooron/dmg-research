"""Versioned, deterministic training checkpoint support for dMoT."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = 1
REQUIRED_KEYS = {
    "schema_version",
    "git_commit",
    "model_name",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "epoch",
    "global_step",
    "raw_parameters",
    "physical_parameters",
    "hydrological_states",
    "uh_states",
    "warmup_state",
    "cpu_rng_state",
    "cuda_rng_state",
    "numpy_rng_state",
    "python_rng_state",
    "sampler_state",
    "dataloader_position",
    "training_config",
    "dataset_manifest_hash",
    "model_config_hash",
}


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def _model_name(model: torch.nn.Module) -> str:
    config = getattr(model, "config", {})
    if isinstance(config, dict):
        phy = config.get("model", {}).get("phy", {})
        if isinstance(phy, dict):
            value = phy.get("model_name") or phy.get("name")
            if isinstance(value, (list, tuple)):
                value = value[0]
            if value:
                return str(value)
        if config.get("model_name"):
            return str(config["model_name"])
    return model.__class__.__name__


def _optional_model_state(model: torch.nn.Module, method: str) -> Any:
    callback = getattr(model, method, None)
    return callback() if callable(callback) else {}


def _sampler_state(sampler: Any) -> Any:
    if sampler is None:
        return {}
    callback = getattr(sampler, "state_dict", None)
    return callback() if callable(callback) else {}


def _raw_and_physical_parameters(model: torch.nn.Module) -> tuple[dict[str, Any], dict[str, Any]]:
    raw: dict[str, Any] = {}
    physical: dict[str, Any] = {}
    if hasattr(model, "raw_parameters") and isinstance(model.raw_parameters, torch.Tensor):
        raw["raw_parameters"] = model.raw_parameters.detach().cpu()
    if hasattr(model, "physical_parameters"):
        value = model.physical_parameters
        physical["physical_parameters"] = value.detach().cpu() if isinstance(value, torch.Tensor) else value
    return raw, physical


def save_training_checkpoint(
    directory: str | os.PathLike[str],
    *,
    model: torch.nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    config: dict[str, Any] | None = None,
    sampler: Any = None,
    global_step: int | None = None,
    hydrological_states: Any = None,
    uh_states: Any = None,
    warmup_state: Any = None,
    clear_prior: bool = False,
) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    if clear_prior:
        for prior in path.glob("trainer_state_ep*.pt"):
            prior.unlink()

    config = dict(config or {})
    mname = _model_name(model).lower()
    if mname == "mopex4":
        config.setdefault("parameter_names", list(getattr(model, "phy_param_names", ())))
        config.setdefault("interception_schema", "original")
    raw, physical = _raw_and_physical_parameters(model)
    dataset_manifest_hash = str(config.get("dataset_manifest_hash", "UNSPECIFIED"))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": _git_commit(),
        "model_name": _model_name(model),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else {},
        "epoch": int(epoch),
        "global_step": int(global_step if global_step is not None else epoch),
        "raw_parameters": raw,
        "physical_parameters": physical,
        "hydrological_states": hydrological_states if hydrological_states is not None else _optional_model_state(model, "export_hydrological_state"),
        "uh_states": uh_states if uh_states is not None else _optional_model_state(model, "export_uh_state"),
        "warmup_state": warmup_state if warmup_state is not None else _optional_model_state(model, "export_warmup_state"),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "sampler_state": _sampler_state(sampler),
        "dataloader_position": {"epoch": int(epoch), "global_step": int(global_step if global_step is not None else epoch)},
        "training_config": config,
        "dataset_manifest_hash": dataset_manifest_hash,
        "model_config_hash": _stable_hash(config.get("model", config)),
    }
    output = path / f"trainer_state_ep{int(epoch)}.pt"
    torch.save(payload, output)
    return output


def validate_checkpoint(payload: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_KEYS.difference(payload))
    if missing:
        raise ValueError(f"checkpoint schema v{SCHEMA_VERSION} missing keys: {missing}")
    if int(payload["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema {payload['schema_version']}; expected {SCHEMA_VERSION}"
        )


def load_training_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    sampler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    validate_checkpoint(payload)
    mopex_model = str(payload.get("model_name", "")).lower()
    if mopex_model == "mopex4":
        training_config = payload.get("training_config") or {}
        names = training_config.get("parameter_names")
        if not names:
            raise ValueError(
                "MOPEX4 checkpoint has no parameter schema; refusing to interpret "
                "parameter values without an explicit schema."
            )
        from dmotpy.models.core.mopex4 import validate_mopex4_parameter_schema
        # restored mopex4 uses the original alpha/is_time schema
        validate_mopex4_parameter_schema(names, legacy_f0=True)
        current_names = tuple(getattr(model, "phy_param_names", ()))
        if current_names and tuple(names) != current_names:
            raise ValueError(f"MOPEX4 checkpoint parameter schema mismatch: {names} != {current_names}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload["scheduler_state_dict"]:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if sampler is not None and payload["sampler_state"]:
        callback = getattr(sampler, "load_state_dict", None)
        if callable(callback):
            callback(payload["sampler_state"])
    torch.set_rng_state(payload["cpu_rng_state"].cpu())
    if torch.cuda.is_available() and payload["cuda_rng_state"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_state"]])
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    restore = getattr(model, "import_training_state", None)
    if callable(restore):
        restore(
            hydrological_states=payload["hydrological_states"],
            uh_states=payload["uh_states"],
            warmup_state=payload["warmup_state"],
        )
    return payload
