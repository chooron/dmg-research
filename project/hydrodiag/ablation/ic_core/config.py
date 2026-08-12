from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_TOP_LEVEL = {
    "experiment_name", "project_root", "dataset_path", "gage_ids_path", "dates_path",
    "basin_list_path", "output_root", "model_keys", "periods", "window",
    "forcing_dtype", "model_dtype", "metric_dtype", "target_raw_unit",
    "target_model_unit", "area", "device", "seed", "boundary_handling",
    "save_test_metrics", "batching", "objective",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "UNVERIFIED"


def validate_config(config: dict[str, Any]) -> None:
    unknown = sorted(set(config) - REQUIRED_TOP_LEVEL - {
        "source_config_path", "config_hash", "model_variant", "tgd_structure_version",
    })
    missing = sorted(REQUIRED_TOP_LEVEL - set(config))
    if unknown:
        raise ValueError(f"Unknown config fields: {unknown}")
    if missing:
        raise ValueError(f"Missing config fields: {missing}")
    if len(config["model_keys"]) == 0:
        raise ValueError("model_keys must not be empty")
    if "559" in str(config["basin_list_path"]):
        raise ValueError("531 foundation config must not use a 559 basin list")
    if str(config["target_raw_unit"]).lower() not in {"ft3/s", "ft^3/s"}:
        raise ValueError("target_raw_unit must be ft3/s")
    if str(config["target_model_unit"]).lower() not in {"mm/day", "mm/d"}:
        raise ValueError("target_model_unit must be mm/day")
    if int(config["window"]["warmup_days"]) != 365:
        raise ValueError("the dPL CAMELS-531 protocol requires 365 warmup days")
    if config["area"]["unit"] != "km2" or int(config["area"]["attribute_index"]) != 11:
        raise ValueError("area must use the verified area_gages2 km2 attribute at index 11")
    if config["boundary_handling"] != "clip_0_1":
        raise ValueError("foundation boundary handling must be clip_0_1")


def load_resolved_config(path: str | Path, *, device_override: str | None = None) -> dict[str, Any]:
    source = Path(path).resolve()
    with source.open() as handle:
        raw = json.load(handle)
    config = dict(raw)
    config["project_root"] = str(PROJECT_ROOT)
    for field in ("dataset_path", "gage_ids_path", "dates_path", "basin_list_path", "output_root"):
        config[field] = str(Path(config[field]).expanduser().resolve())
    if device_override is not None:
        config["device"] = device_override
    config["source_config_path"] = str(source)
    validate_config(config)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config["config_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    return config


def environment_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "project_root": str(PROJECT_ROOT),
        "config_hash": config.get("config_hash"),
    }
    try:
        import torch
        snapshot["torch"] = torch.__version__
        snapshot["torch_cuda"] = torch.version.cuda
        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        snapshot["device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        snapshot["torch_error"] = repr(exc)
    try:
        import evotorch
        snapshot["evotorch"] = evotorch.__version__
    except Exception as exc:
        snapshot["evotorch_error"] = repr(exc)
    for name in ("dataset_path", "gage_ids_path", "dates_path", "basin_list_path"):
        path = Path(config[name])
        snapshot.setdefault("fingerprints", {})[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    return snapshot
