"""Shared data/forcing contract for dMoT training adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch


CALENDAR_MODELS = frozenset({"mopex4", "mopex5", "vic"})


def calendar_features(
    dates: Iterable[Any],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("calendar dates must be unique and monotonically increasing")
    doy = torch.as_tensor(index.dayofyear.to_numpy(), dtype=dtype, device=device)
    return doy.view(-1, 1, 1)


def add_calendar_forcing(
    x_phy: torch.Tensor,
    dates: Iterable[Any],
    *,
    model_name: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Append day-of-year once at the public forcing adapter boundary.

    Ordinary models retain their original three forcing channels.  Calendar
    models receive a fourth channel; their model implementation already
    consumes channel four and does not read files or infer dates.
    """
    if model_name.lower() not in CALENDAR_MODELS:
        return x_phy, None
    if x_phy.ndim != 3 or x_phy.shape[-1] != 3:
        raise ValueError(f"expected [time, basin, 3] forcing, got {tuple(x_phy.shape)}")
    doy = calendar_features(dates, dtype=x_phy.dtype, device=x_phy.device)
    if doy.shape[0] != x_phy.shape[0]:
        raise ValueError("calendar feature length does not match forcing length")
    doy = doy.expand(-1, x_phy.shape[1], -1)
    return torch.cat((x_phy, doy), dim=-1), doy


def attach_training_mask(dataset: dict[str, Any]) -> dict[str, Any]:
    """Attach an explicit observation-validity mask without touching forcing."""
    result = dict(dataset)
    target = result.get("target")
    if isinstance(target, torch.Tensor):
        result["mask"] = torch.isfinite(target)
    return result


def dataset_manifest(
    *,
    dataset_name: str,
    source_path: str,
    train_period: tuple[str, str],
    validation_period: tuple[str, str],
    test_period: tuple[str, str],
    normalization_scope: str = "train_only",
) -> dict[str, Any]:
    manifest = {
        "dataset_name": dataset_name,
        "source_path": source_path,
        "variables": [
            {"variable_name": "prcp", "physical_meaning": "daily precipitation", "raw_unit": "mm/d", "target_unit": "mm/d", "conversion_formula": "identity"},
            {"variable_name": "pet", "physical_meaning": "daily potential evapotranspiration", "raw_unit": "mm/d", "target_unit": "mm/d", "conversion_formula": "identity"},
            {"variable_name": "tmean", "physical_meaning": "daily mean temperature", "raw_unit": "degC", "target_unit": "degC", "conversion_formula": "identity"},
            {"variable_name": "streamflow", "physical_meaning": "daily discharge", "raw_unit": "ft3/s", "target_unit": "mm/d", "conversion_formula": "q * 0.0283168 * 86400 * 1000 / (area_km2 * 1e6)"},
        ],
        "missing_value": "NaN; excluded by explicit observation mask",
        "qc_flag": "not present in source pickle; no QC filtering inferred",
        "valid_range": {"prcp": ">=0", "pet": ">=0", "streamflow": ">=0 after conversion", "tmean": "source-defined"},
        "time_zone": "UTC-naive daily index as supplied by source metadata",
        "time_resolution": "1 day",
        "calendar": "proleptic Gregorian / pandas dayofyear",
        "basin_area_source": "area_gages2 attribute",
        "basin_area_unit": "km2",
        "date_alignment": "forcing and target share the metadata daily index; no overlap between train and test intervals",
        "train_period": list(train_period),
        "validation_period": list(validation_period),
        "test_period": list(test_period),
        "normalization_scope": normalization_scope,
    }
    return manifest


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, sort_keys=True, indent=2, default=str).encode("utf-8")
    output.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    digest_path = output.with_suffix(output.suffix + ".sha256")
    digest_path.write_text(digest + "\n", encoding="utf-8")
    return digest
