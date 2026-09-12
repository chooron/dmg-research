"""CAMELS interface for the paper protocol.

This adapter reuses the repository's existing ``data/camels_dataset`` bundle
format without importing benchmark hydrological equations.  The bundle is
expected to contain ``(forcing, target, attributes)`` with basin-major time
series; forcing columns 0:3 are ``prcp, tmean, pet`` and target column 0 is
USGS flow in ft^3/s.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .protocol import ExperimentProtocol



DEFAULT_MANIFEST = Path(__file__).with_name("manifests") / "camels_544.json"


def load_basin_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, object]:
    """Load and verify the explicit HydroShare-derived 544-basin manifest."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text())
    rows = payload.get("basins", [])
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    actual_hash = hashlib.sha256(canonical).hexdigest()
    if actual_hash != payload.get("manifest_sha256"):
        raise ValueError(f"manifest hash mismatch for {manifest_path}")
    if payload.get("count") != len(rows) or payload.get("count") != 544:
        raise ValueError(f"manifest must contain exactly 544 rows: {manifest_path}")
    if len({row["basin_id"] for row in rows}) != len(rows):
        raise ValueError(f"manifest contains duplicate basin IDs: {manifest_path}")
    return payload


def manifest_hru_ids(path: str | Path = DEFAULT_MANIFEST) -> np.ndarray:
    """Return numeric CAMELS hru IDs in the manifest's stable row order."""
    payload = load_basin_manifest(path)
    return np.asarray([row["hru_id"] for row in payload["basins"]], dtype=np.int64)
@dataclass(frozen=True)
class CamelsBundle:
    forcing: np.ndarray
    target: np.ndarray
    attributes: np.ndarray
    basin_ids: np.ndarray
    dates: tuple[date, ...]

    @classmethod
    def load(
        cls,
        data_path: str | Path,
        basin_id_path: str | Path,
        *,
        start: date = date(1980, 10, 1),
        end: date = date(2014, 9, 30),
    ) -> "CamelsBundle":
        with Path(data_path).open("rb") as handle:
            forcing, target, attributes = pickle.load(handle)
        ids = np.load(basin_id_path) if Path(basin_id_path).suffix == ".npy" else np.asarray(ast.literal_eval(Path(basin_id_path).read_text()))
        forcing = np.asarray(forcing)
        target = np.asarray(target)
        attributes = np.asarray(attributes)
        ids = np.asarray(ids).reshape(-1).astype(np.int64)
        if forcing.ndim != 3 or forcing.shape[0] != len(ids):
            raise ValueError("CAMELS forcing must have shape [basin, time, feature] aligned to basin IDs")
        if target.shape[:2] != forcing.shape[:2] or attributes.shape[0] != len(ids):
            raise ValueError("CAMELS target/attributes are not aligned with forcing and basin IDs")
        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)
        if len(dates) != forcing.shape[1]:
            raise ValueError(f"CAMELS date axis has {len(dates)} days but forcing has {forcing.shape[1]}")
        return cls(forcing, target, attributes, ids, tuple(dates))

    def indices(self, period: tuple[date, date]) -> np.ndarray:
        left, right = period
        return np.asarray([i for i, current in enumerate(self.dates) if left <= current <= right], dtype=np.int64)

    def select(self, basin_ids: Sequence[int], period: tuple[date, date]) -> tuple[torch.Tensor, torch.Tensor]:
        requested = np.asarray(basin_ids, dtype=np.int64)
        lookup = {int(value): i for i, value in enumerate(self.basin_ids)}
        try:
            basin_index = np.asarray([lookup[int(value)] for value in requested], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"CAMELS basin ID is not present in bundle: {exc.args[0]}") from exc
        time_index = self.indices(period)
        x = self.forcing[basin_index[:, None], time_index[None, :], :3]
        y = self.target[basin_index[:, None], time_index[None, :], 0].astype(np.float64)
        area_km2 = self.attributes[basin_index, 11].astype(np.float64)
        # Same conversion used by the existing project benchmark loader.
        y *= (0.0283168 * 86400.0 * 1.0e3 / (area_km2 * 1.0e6))[:, None]
        x = np.transpose(x, (1, 0, 2))
        y = y.T
        return torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)


def paper_period(protocol: ExperimentProtocol, name: str) -> tuple[date, date]:
    values = {
        "warmup": (protocol.forcing_start, protocol.simulation_start - timedelta(days=1)),
        "simulation": (protocol.simulation_start, protocol.simulation_end),
        "calibration": (protocol.calibration_start, protocol.calibration_end),
        "evaluation": (protocol.evaluation_start, protocol.evaluation_end),
    }
    try:
        return values[name]
    except KeyError as exc:
        raise KeyError(f"unknown paper period: {name}") from exc
