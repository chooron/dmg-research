from __future__ import annotations

from collections.abc import Callable

import torch

from models.core.tcm import baseflow_6 as tcm_baseflow_6
from models.flux.baseflow import baseflow_6, baseflow_9
from models.flux.evap import evap_14, evap_16
from models.flux.interflow import interflow_11, interflow_12
from models.flux.mopex import mopex_interception_4 as mopex4_interception_4
from models.flux.melt import melt_3
from models.flux.phenology import phenology_1
from models.flux.rainfall import rainfall_1
from models.flux.saturation import saturation_1, saturation_9, saturation_11
from models.flux.smooth import (
    soft_gate_storage_above,
    soft_gate_temperature_below,
)
from models.flux.snowfall import snowfall_1


TensorFn = Callable[[torch.Tensor], torch.Tensor]


def constant_like(x: torch.Tensor, value: float) -> torch.Tensor:
    return torch.full_like(x, float(value))


def smooth_storage_gate_above(x: torch.Tensor) -> torch.Tensor:
    return soft_gate_storage_above(x, constant_like(x, 50.0))


def smooth_temperature_snow(x: torch.Tensor) -> torch.Tensor:
    return soft_gate_temperature_below(x, constant_like(x, 0.0))


def snowfall_1_wrapper(x: torch.Tensor) -> torch.Tensor:
    return snowfall_1(constant_like(x, 10.0), x, constant_like(x, 0.0))


def rainfall_1_wrapper(x: torch.Tensor) -> torch.Tensor:
    return rainfall_1(constant_like(x, 10.0), x, constant_like(x, 0.0))


def melt_3_wrapper(x: torch.Tensor) -> torch.Tensor:
    return melt_3(
        constant_like(x, 3.0),
        constant_like(x, 0.0),
        constant_like(x, 3.0),
        constant_like(x, 20.0),
        x,
        constant_like(x, 0.01),
    )


def saturation_1_wrapper(x: torch.Tensor) -> torch.Tensor:
    return saturation_1(constant_like(x, 10.0), x, constant_like(x, 50.0))


def saturation_9_wrapper(x: torch.Tensor) -> torch.Tensor:
    return saturation_9(constant_like(x, 10.0), x, constant_like(x, 0.01))


def saturation_11_wrapper(x: torch.Tensor) -> torch.Tensor:
    return saturation_11(
        constant_like(x, 1.5),
        constant_like(x, 2.0),
        x,
        constant_like(x, 10.0),
        constant_like(x, 100.0),
        constant_like(x, 10.0),
    )


def evap_14_wrapper(x: torch.Tensor) -> torch.Tensor:
    return evap_14(
        constant_like(x, 0.7),
        constant_like(x, 2.0),
        constant_like(x, 8.0),
        constant_like(x, 5.0),
        x,
        constant_like(x, 0.1),
    )


def evap_16_wrapper(x: torch.Tensor) -> torch.Tensor:
    return evap_16(
        constant_like(x, 0.7),
        constant_like(x, 1.0e6),
        x,
        constant_like(x, 0.1),
        constant_like(x, 8.0),
    )


def interflow_11_wrapper(x: torch.Tensor) -> torch.Tensor:
    return interflow_11(constant_like(x, 5.0), constant_like(x, 50.0), x)


def interflow_12_wrapper(x: torch.Tensor) -> torch.Tensor:
    return interflow_12(
        constant_like(x, 0.3),
        constant_like(x, 0.4),
        constant_like(x, 1.5),
        x,
        constant_like(x, 100.0),
    )


def baseflow_6_wrapper(x: torch.Tensor) -> torch.Tensor:
    return baseflow_6(constant_like(x, 0.01), constant_like(x, 10.0), x)


def baseflow_9_wrapper(x: torch.Tensor) -> torch.Tensor:
    return baseflow_9(constant_like(x, 0.2), constant_like(x, 50.0), x)


def phenology_1_wrapper(x: torch.Tensor) -> torch.Tensor:
    return phenology_1(x, constant_like(x, -5.0), constant_like(x, 5.0), constant_like(x, 8.0))


def mopex4_interception_4_wrapper(x: torch.Tensor) -> torch.Tensor:
    return mopex4_interception_4(
        constant_like(x, 10.0),
        x,
        constant_like(x, 0.4),
        constant_like(x, 183.0),
    )


def tcm_baseflow_6_scaled_wrapper(x: torch.Tensor) -> torch.Tensor:
    return tcm_baseflow_6(constant_like(x, 0.01), constant_like(x, 0.0), x)


WRAPPERS: dict[str, TensorFn] = {
    "F001": smooth_storage_gate_above,
    "F002": smooth_temperature_snow,
    "F003": snowfall_1_wrapper,
    "F004": rainfall_1_wrapper,
    "F005": melt_3_wrapper,
    "F006": saturation_1_wrapper,
    "F007": saturation_9_wrapper,
    "F008": saturation_11_wrapper,
    "F009": evap_14_wrapper,
    "F010": evap_16_wrapper,
    "F011": interflow_11_wrapper,
    "F012": interflow_12_wrapper,
    "F013": baseflow_6_wrapper,
    "F014": baseflow_9_wrapper,
    "F015": phenology_1_wrapper,
    "F016": mopex4_interception_4_wrapper,
    "F017": tcm_baseflow_6_scaled_wrapper,
}
