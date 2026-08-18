from __future__ import annotations

from typing import Any

import numpy as np

FT3_TO_M3 = 0.028316846592
SECONDS_PER_DAY = 86400.0
MM_PER_M = 1000.0
M2_PER_KM2 = 1_000_000.0
FT3S_TO_MMDAY_FACTOR = FT3_TO_M3 * SECONDS_PER_DAY * MM_PER_M / M2_PER_KM2


def convert_ft3s_to_mm_day(
    discharge_ft3s: np.ndarray,
    area_km2: np.ndarray,
    *,
    return_valid_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Convert raw CAMELS discharge to basin-average runoff depth.

    Invalid raw values (non-finite or negative) remain ``NaN`` and are not
    clipped. Valid zero flow remains zero. Areas are aligned to the first
    dimension of ``discharge_ft3s`` and broadcast over all trailing dimensions.
    """
    values = np.asarray(discharge_ft3s)
    areas = np.asarray(area_km2)
    if values.ndim < 1:
        raise ValueError("discharge_ft3s must have at least one dimension")
    if areas.ndim != 1 or areas.shape[0] != values.shape[0]:
        raise ValueError(
            f"area_km2 must have shape [{values.shape[0]}], got {areas.shape}"
        )
    if not np.isfinite(areas).all() or (areas <= 0).any():
        raise ValueError("area_km2 must contain finite positive values")

    raw = values.astype(np.float64, copy=False)
    output = np.full(raw.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(raw) & (raw >= 0.0)
    factor = FT3S_TO_MMDAY_FACTOR / areas.astype(np.float64)
    output[valid] = (
        raw[valid]
        * np.broadcast_to(factor.reshape((-1,) + (1,) * (raw.ndim - 1)), raw.shape)[
            valid
        ]
    )
    if return_valid_mask:
        return output, valid
    return output


def conversion_metadata() -> dict[str, Any]:
    return {
        "raw_unit": "ft3/s",
        "converted_unit": "mm/day",
        "formula": "Q_ft3s * 0.028316846592 * 86400 * 1000 / (area_km2 * 1e6)",
        "constant_factor_per_km2": FT3S_TO_MMDAY_FACTOR,
        "missing_policy": "nonfinite and negative raw discharge remain NaN; valid zero remains zero",
    }
