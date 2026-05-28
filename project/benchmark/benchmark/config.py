"""Configuration loading and path normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCHMARK_ROOT.parents[1]


def _to_container(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True)


def resolve_path(path_value: str | Path, base: Path = REPO_ROOT) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_benchmark_config(path: str | Path) -> dict[str, Any]:
    cfg = _to_container(OmegaConf.load(path))
    cfg.setdefault("paths", {})
    cfg.setdefault("camels", {})
    cfg.setdefault("calibration", {})
    cfg.setdefault("tasks", {})

    for key in ("data_path", "basin_ids_path", "basin_ids_reference_path"):
        if cfg["paths"].get(key):
            cfg["paths"][key] = str(resolve_path(cfg["paths"][key]))

    for key in ("output_dir", "log_dir"):
        if cfg["paths"].get(key):
            cfg["paths"][key] = str(resolve_path(cfg["paths"][key]))

    return cfg
