"""Resolved, immutable configuration helpers for production CMA-ES runs."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT = Path(__file__).resolve().parents[1]


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_resolved_config(path: str | Path) -> dict[str, Any]:
    """Load YAML inheritance recursively and retain the source path."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = EXPERIMENT / "configs" / config_path
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    parent = raw.get("extends")
    resolved = _merge(load_resolved_config(config_path.parent / parent), raw) if parent else _merge({}, raw)
    resolved["_resolved_from"] = str(config_path.resolve())
    return resolved


def validate_full_run_config(config: dict[str, Any]) -> None:
    optimization = config["optimization"]
    warmup = config["warmup"]
    data = config["data"]
    if optimization["starts"] != 10:
        raise ValueError("production configuration must use exactly 10 independent starts")
    if optimization["generations"] != 300:
        raise ValueError("production configuration must use exactly 300 generations")
    if warmup["mode"] != "repeat_forcing":
        raise ValueError("production warm-up must use repeat_forcing")
    if int(warmup["repetitions"]) != 5:
        raise ValueError("production warm-up must repeat the source period five times")
    if data["train"]["start_time"] != "1989-01-01" or data["train"]["end_time"] != "1998-12-31":
        raise ValueError("production training split must remain 1989-01-01..1998-12-31")
    if data["test"]["start_time"] != "1999-01-01" or data["test"]["end_time"] != "2009-12-31":
        raise ValueError("production test split is frozen and must not be used for selection")
