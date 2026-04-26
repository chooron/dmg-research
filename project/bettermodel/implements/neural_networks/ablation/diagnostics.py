from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def parameter_temporal_variability(
    normalized_parameters: np.ndarray,
) -> dict[str, Any]:
    """Summarize day-to-day changes in normalized dynamic parameters."""
    if normalized_parameters.shape[0] < 2:
        return {
            "median_abs_day_to_day_change": None,
            "mean_abs_day_to_day_change": None,
            "max_abs_day_to_day_change": None,
        }

    abs_change = np.abs(np.diff(normalized_parameters, axis=0))
    return {
        "median_abs_day_to_day_change": float(np.nanmedian(abs_change)),
        "mean_abs_day_to_day_change": float(np.nanmean(abs_change)),
        "max_abs_day_to_day_change": float(np.nanmax(abs_change)),
    }


def boundary_saturation_ratio(
    normalized_parameters: np.ndarray,
    *,
    eps: float = 0.01,
) -> dict[str, float]:
    """Return the fraction of normalized outputs close to 0 or 1."""
    lower = normalized_parameters <= eps
    upper = normalized_parameters >= 1.0 - eps
    return {
        "lower_boundary_ratio": float(np.nanmean(lower)),
        "upper_boundary_ratio": float(np.nanmean(upper)),
        "either_boundary_ratio": float(np.nanmean(lower | upper)),
    }


def save_dynamic_parameter_diagnostics(
    output_dir: str | Path,
    normalized_parameters: np.ndarray,
    *,
    variant: str,
    parameter_names: list[str] | None = None,
    basin_ids: list[str] | None = None,
    saturation_eps: float = 0.01,
) -> dict[str, Any]:
    """Save trajectories and ablation diagnostics for one trained variant.

    ``normalized_parameters`` is expected to be normalized to [0, 1] and shaped
    as time-first, for example ``(T, B, n_dynamic, nmul)``.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    parameters = np.asarray(normalized_parameters, dtype=np.float32)
    trajectory_path = output_path / f"{variant}_dynamic_parameter_trajectories.npz"
    np.savez_compressed(
        trajectory_path,
        normalized_parameters=parameters,
        parameter_names=np.array(parameter_names or [], dtype=object),
        basin_ids=np.array(basin_ids or [], dtype=object),
    )

    stats = {
        "variant": variant,
        "shape": list(parameters.shape),
        "trajectory_file": trajectory_path.name,
        "saturation_eps": saturation_eps,
        **parameter_temporal_variability(parameters),
        **boundary_saturation_ratio(parameters, eps=saturation_eps),
    }
    stats_path = output_path / f"{variant}_dynamic_parameter_stats.json"
    with stats_path.open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)
        fp.write("\n")

    return stats
