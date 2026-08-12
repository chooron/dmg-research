from __future__ import annotations

from typing import Any

import numpy as np
import torch

from models.parameter_specs import (
    GR4J_CN_PARAM_SPECS, GR4J_PD_PARAM_SPECS, GR4J_PARAM_SPECS, GR4J_TGD2_PARAM_SPECS,
    HBV_PARAM_SPECS, SIMHYD_CN_PARAM_SPECS, SIMHYD_PD_PARAM_SPECS, SIMHYD_PARAM_SPECS, SIMHYD_TGD2_PARAM_SPECS,
    XAJ_CN_PARAM_SPECS, XAJ_PD_PARAM_SPECS, XAJ_PARAM_SPECS, XAJ_TGD2_PARAM_SPECS,
    XAJ_2S_PARAM_SPECS, XAJ_RWPE_PARAM_SPECS,
    XAJ_CONTROLLED_N_PARAM_SPECS, XAJ_DE_PARAM_SPECS, XAJ_GE_PARAM_SPECS,
    XAJ_DR_PARAM_SPECS, XAJ_GR_PARAM_SPECS, CEMANEIGE_PARAM_SPECS,
)


PARAMETER_SPECS = {
    "XAJ": XAJ_PARAM_SPECS,
    "N": {**CEMANEIGE_PARAM_SPECS, **XAJ_CONTROLLED_N_PARAM_SPECS},
    "D_E": {**CEMANEIGE_PARAM_SPECS, **XAJ_DE_PARAM_SPECS},
    "G_E": {**CEMANEIGE_PARAM_SPECS, **XAJ_GE_PARAM_SPECS},
    "D_R": {**CEMANEIGE_PARAM_SPECS, **XAJ_DR_PARAM_SPECS},
    "G_R": {**CEMANEIGE_PARAM_SPECS, **XAJ_GR_PARAM_SPECS},
    "XAJ_CN": XAJ_CN_PARAM_SPECS,
    "XAJ_2S": XAJ_2S_PARAM_SPECS,
    "XAJ_RWPE": XAJ_RWPE_PARAM_SPECS,
    "XAJ_PD": XAJ_PD_PARAM_SPECS,
    "XAJ_TGD2": XAJ_TGD2_PARAM_SPECS,
    "GR4J": GR4J_PARAM_SPECS,
    "GR4J_CN": GR4J_CN_PARAM_SPECS,
    "GR4J_PD": GR4J_PD_PARAM_SPECS,
    "GR4J_TGD2": GR4J_TGD2_PARAM_SPECS,
    "SIMHYD": SIMHYD_PARAM_SPECS,
    "SIMHYD_CN": SIMHYD_CN_PARAM_SPECS,
    "SIMHYD_PD": SIMHYD_PD_PARAM_SPECS,
    "SIMHYD_TGD2": SIMHYD_TGD2_PARAM_SPECS,
    "HBV": HBV_PARAM_SPECS,
}
LOG_SCALED_PARAMETERS = frozenset({"tgd_tau_warm", "tgd_delta_tau_cold"})


def get_parameter_spec(model_key: str) -> dict[str, dict[str, Any]]:
    try:
        return PARAMETER_SPECS[model_key]
    except KeyError as exc:
        raise KeyError(f"Unknown IC model key: {model_key}") from exc


def _bounds(model_key: str, *, dtype: torch.dtype = torch.float64, device: torch.device | None = None):
    specs = get_parameter_spec(model_key)
    names = tuple(specs)
    lower = torch.tensor([specs[name]["lower"] for name in names], dtype=dtype, device=device)
    upper = torch.tensor([specs[name]["upper"] for name in names], dtype=dtype, device=device)
    return names, lower, upper


def validate_parameter_shape(model_key: str, theta: Any) -> None:
    expected = len(get_parameter_spec(model_key))
    shape = tuple(theta.shape)
    if not shape or shape[-1] != expected:
        raise ValueError(f"{model_key} expects trailing parameter dimension {expected}, got {shape}")


def normalized_to_physical(model_key: str, theta_01: Any, *, clip: bool = True):
    validate_parameter_shape(model_key, theta_01)
    if isinstance(theta_01, torch.Tensor):
        names, lower, upper = _bounds(model_key, dtype=theta_01.dtype, device=theta_01.device)
        value = theta_01.clamp(0.0, 1.0) if clip else theta_01
        physical = lower + value * (upper - lower)
        for index, name in enumerate(names):
            if name in LOG_SCALED_PARAMETERS:
                physical[..., index] = torch.exp(
                    torch.log(lower[index]) + value[..., index] * (torch.log(upper[index]) - torch.log(lower[index]))
                )
        return physical
    value = np.asarray(theta_01)
    if not np.issubdtype(value.dtype, np.floating):
        value = value.astype(np.float64)
    if clip:
        value = np.clip(value, 0.0, 1.0)
    names = tuple(get_parameter_spec(model_key))
    lower = np.asarray([get_parameter_spec(model_key)[name]["lower"] for name in names], dtype=np.float64)
    upper = np.asarray([get_parameter_spec(model_key)[name]["upper"] for name in names], dtype=np.float64)
    physical = lower + value * (upper - lower)
    for index, name in enumerate(names):
        if name in LOG_SCALED_PARAMETERS:
            physical[..., index] = np.exp(np.log(lower[index]) + value[..., index] * (np.log(upper[index]) - np.log(lower[index])))
    return physical


def physical_to_normalized(model_key: str, physical: Any, *, clip: bool = False):
    validate_parameter_shape(model_key, physical)
    if isinstance(physical, torch.Tensor):
        names, lower, upper = _bounds(model_key, dtype=physical.dtype, device=physical.device)
        value = physical.clone()
        for index, name in enumerate(names):
            if name in LOG_SCALED_PARAMETERS:
                value[..., index] = (torch.log(value[..., index]) - torch.log(lower[index])) / (torch.log(upper[index]) - torch.log(lower[index]))
            else:
                value[..., index] = (value[..., index] - lower[index]) / (upper[index] - lower[index])
        return value.clamp(0.0, 1.0) if clip else value
    value = np.asarray(physical)
    if not np.issubdtype(value.dtype, np.floating):
        value = value.astype(np.float64)
    names = tuple(get_parameter_spec(model_key))
    specs = get_parameter_spec(model_key)
    lower = np.asarray([specs[name]["lower"] for name in names], dtype=np.float64)
    upper = np.asarray([specs[name]["upper"] for name in names], dtype=np.float64)
    normalized = np.empty_like(value, dtype=np.float64)
    for index, name in enumerate(names):
        if name in LOG_SCALED_PARAMETERS:
            normalized[..., index] = (np.log(value[..., index]) - np.log(lower[index])) / (np.log(upper[index]) - np.log(lower[index]))
        else:
            normalized[..., index] = (value[..., index] - lower[index]) / (upper[index] - lower[index])
    return np.clip(normalized, 0.0, 1.0) if clip else normalized


def parameter_summary(model_key: str) -> dict[str, Any]:
    specs = get_parameter_spec(model_key)
    return {
        "model_key": model_key,
        "parameter_count": len(specs),
        "parameter_names": list(specs),
        "log_scaled_parameters": sorted(name for name in specs if name in LOG_SCALED_PARAMETERS),
        "bounds": {name: {"lower": spec["lower"], "upper": spec["upper"]} for name, spec in specs.items()},
    }
