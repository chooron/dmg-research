"""Small compile-safe numerical helpers for structure-diagnosis kernels."""

from __future__ import annotations

import torch

POWER_FLOOR = 1e-6


def stable_positive_power(
    value: torch.Tensor,
    exponent: torch.Tensor,
    *,
    floor: float = POWER_FLOOR,
) -> torch.Tensor:
    """Evaluate ``value ** exponent`` without a fractional-power zero branch.

    The logarithm is evaluated on a positive floor, while the original zero
    mask is applied afterwards.  Thus an exact zero has exactly zero output,
    but nonzero values retain the differentiable log-exp form.  A small
    first-order correction makes the exponent-one forward value exact below
    ``floor`` without suppressing the exponent derivative.
    """
    safe_value = torch.clamp(value, min=floor)
    log_safe = torch.log(safe_value)
    powered = torch.exp(exponent * log_safe)

    # For values below the numerical floor, retain the exact linear forward
    # value at exponent one without removing the exponent derivative.  The
    # first-order correction is inactive above the floor and gives the
    # requested x**gamma*log(x) derivative at gamma=1 for small positive x.
    linearized = (
        powered
        + (value - safe_value)
        + (exponent - 1.0) * (value * log_safe - safe_value * log_safe)
    )
    powered = torch.where(value < floor, linearized, powered)
    return torch.where(
        value > 0.0,
        powered,
        torch.zeros_like(powered),
    )


def log_map_normalized(
    normalized: torch.Tensor,
    lower: float,
    upper: float,
) -> torch.Tensor:
    """Map a normalized coordinate to a positive physical range in log space."""
    lower_t = normalized.new_tensor(lower)
    upper_t = normalized.new_tensor(upper)
    return torch.exp(
        torch.log(lower_t) + normalized * (torch.log(upper_t) - torch.log(lower_t))
    )
