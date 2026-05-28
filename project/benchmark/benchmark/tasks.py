"""Task-table generation for benchmark runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .basins import load_basin_ids
from .models import available_model_ids


def resolve_models(config: dict[str, Any]) -> list[str]:
    models = config.get("tasks", {}).get("models", "all")
    if isinstance(models, str) and models.lower() == "all":
        return available_model_ids()
    if isinstance(models, str):
        return [models]
    return [str(model) for model in models]


def resolve_objectives(config: dict[str, Any]) -> list[str]:
    objectives = config.get("tasks", {}).get("objectives", ["nse", "log_nse"])
    if isinstance(objectives, str):
        return [objectives]
    return [str(objective) for objective in objectives]


def generate_independent_calibration_tasks(
    config: dict[str, Any],
    output_path: str | Path,
    limit_basins: int | None = None,
) -> Path:
    basin_ids = load_basin_ids(config["paths"]["basin_ids_path"])
    if limit_basins is not None:
        basin_ids = basin_ids[:limit_basins]

    rows = [
        {"basin_id": int(basin_id), "model_id": model_id, "objective": objective}
        for basin_id in basin_ids
        for model_id in resolve_models(config)
        for objective in resolve_objectives(config)
    ]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["basin_id", "model_id", "objective"])
        writer.writeheader()
        writer.writerows(rows)
    return path
