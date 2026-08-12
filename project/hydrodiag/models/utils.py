from __future__ import annotations

from typing import Any

import torch


def extract_params(
    params: dict[str, torch.Tensor],
    param_specs: dict[str, dict[str, Any]],
    prefix: str = "",
) -> torch.Tensor:
    """Extract parameters into a tensor [batch, n_params] in spec order.

    Args:
        params: Dict of parameter name -> tensor [batch].
        param_specs: Parameter specifications dict.
        prefix: Optional prefix to strip from param names (e.g. 'gr4j_').

    Returns:
        Tensor of shape [batch, n_params].
    """
    result = []
    for name in param_specs:
        key = name if not prefix else name[len(prefix):]
        result.append(params[name].reshape(-1, 1))
    return torch.cat(result, dim=1)


def validate_forcings(forcings: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate and extract forcing tensors.

    Args:
        forcings: Dict with keys 'precip', 'pet', 'temp'.

    Returns:
        Tuple of (precip, pet, temp, device).
    """
    required = ["precip", "pet", "temp"]
    for key in required:
        if key not in forcings:
            raise KeyError(f"Missing forcing key: {key}")
        if not isinstance(forcings[key], torch.Tensor):
            raise TypeError(f"Forcing '{key}' must be a torch.Tensor")

    precip = forcings["precip"]
    pet = forcings["pet"]
    temp = forcings["temp"]

    if precip.ndim != 2 or pet.ndim != 2 or temp.ndim != 2:
        raise ValueError("All forcings must have shape [batch, time]")

    assert precip.shape == pet.shape == temp.shape, (
        f"Shape mismatch: precip {precip.shape}, pet {pet.shape}, temp {temp.shape}"
    )

    device = precip.device
    dtype = precip.dtype
    assert pet.device == device and temp.device == device, "Device mismatch in forcings"
    assert pet.dtype == dtype and temp.dtype == dtype, "Dtype mismatch in forcings"

    return precip, pet, temp, device


def validate_params(
    params: dict[str, torch.Tensor],
    param_specs: dict[str, dict[str, Any]],
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Validate that params dict has correct keys and shapes.

    Args:
        params: Parameter dict to validate.
        param_specs: Expected parameter specifications.
        batch: Expected batch size.
        device: Expected device.
        dtype: Expected dtype.
    """
    for name in param_specs:
        if name not in params:
            raise KeyError(f"Missing parameter: {name}")
        p = params[name]
        if p.ndim != 1:
            raise ValueError(
                f"Parameter '{name}' must have shape [batch], got {p.shape}"
            )
        if p.shape[0] != batch:
            raise ValueError(
                f"Parameter '{name}' batch size mismatch: expected {batch}, got {p.shape[0]}"
            )
        if p.device != device:
            raise ValueError(f"Parameter '{name}' device mismatch")
        if p.dtype != dtype:
            raise ValueError(f"Parameter '{name}' dtype mismatch")


def expand_for_timestep(
    tensor: torch.Tensor,
    timestep_idx: int,
) -> torch.Tensor:
    """Index into a [batch, time] tensor for a given timestep.

    Args:
        tensor: [batch, time] tensor.
        timestep_idx: Integer index.

    Returns:
        [batch] tensor at the given timestep.
    """
    return tensor[:, timestep_idx]
