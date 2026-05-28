"""CAMELS data access for benchmark runners.

This module reads the same pickled CAMELS tensor bundle used by the existing
`dmg`/`dmotpy` stack. It keeps only benchmark-specific slicing and unit
conversion here; hydrologic simulation remains delegated to `dmotpy`.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .basins import basin_index, load_basin_ids


@dataclass(frozen=True)
class BasinPeriod:
    x_phy: torch.Tensor
    target: torch.Tensor


class CamelsStore:
    def __init__(self, config: dict) -> None:
        self.config = config
        paths = config["paths"]
        camels = config["camels"]

        self.reference_ids = load_basin_ids(paths["basin_ids_reference_path"])
        self.selected_ids = load_basin_ids(paths["basin_ids_path"])
        self.all_time = pd.date_range(
            camels["all_start_time"],
            camels["all_end_time"],
            freq="D",
        )
        self.forcing_names = list(camels["forcings"])
        self.attribute_names = list(camels["attributes"])
        self.area_name = camels.get("area_name", "area_gages2")
        self.input_unit = camels.get("input_unit", "ft3/s")
        self.output_unit = camels.get("output_unit", "mm/d")

        with Path(paths["data_path"]).open("rb") as handle:
            self.forcings, self.target, self.attributes = pickle.load(handle)

    def period(self, basin_id: int | str, split_name: str, device: str) -> BasinPeriod:
        idx = basin_index(self.reference_ids, basin_id)
        split = self.config["splits"][split_name]
        start = self.all_time.get_loc(pd.Timestamp(split["start_time"]))
        end = self.all_time.get_loc(pd.Timestamp(split["end_time"])) + 1

        forcing_idx = [self.forcing_names.index(name) for name in self.config["model"]["forcings"]]
        x_np = self.forcings[idx : idx + 1, start:end, :][:, :, forcing_idx]
        y_np = self.target[idx : idx + 1, start:end, :].copy()
        y_np = self._convert_streamflow(idx, y_np)

        x = torch.as_tensor(np.transpose(x_np, (1, 0, 2)), dtype=torch.float32, device=device)
        y = torch.as_tensor(np.transpose(y_np, (1, 0, 2)), dtype=torch.float32, device=device)
        return BasinPeriod(x_phy=x, target=y)

    def _convert_streamflow(self, basin_index_: int, target: np.ndarray) -> np.ndarray:
        if self.input_unit == self.output_unit:
            return target
        if self.output_unit != "mm/d":
            raise ValueError(f"Unsupported benchmark output_unit: {self.output_unit}")

        area = float(self.attributes[basin_index_, self.attribute_names.index(self.area_name)])
        if self.input_unit == "ft3/s":
            target[:, :, 0] = target[:, :, 0] * 0.0283168 * 3600 * 24 * 1e3 / (area * 1e6)
        elif self.input_unit == "m3/s":
            target[:, :, 0] = target[:, :, 0] * 3600 * 24 * 1e3 / (area * 1e6)
        else:
            raise ValueError(f"Unsupported CAMELS input_unit: {self.input_unit}")
        return target
