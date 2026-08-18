"""Evaporation vertical-organization ladder for XAJ.

This module is deliberately a process kernel, not a second XAJ host.  The
host supplies the XAJ lower/deep states and capacities immediately after EU
has been removed.  It returns separate EL and ED fluxes so the two vertical
withdrawals remain observable to later diagnostics.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .parameter_specs import EVAPORATION_GAMMA_PARAM_SPECS
from .structure_utils import log_map_normalized, stable_positive_power

EVAPORATION_POWER_FLOOR = 1e-6


def normalized_to_gamma(normalized: torch.Tensor) -> torch.Tensor:
    """Map normalized ``[0, 1]`` coordinates to gamma in log space.

    ``gamma=0.2`` and ``gamma=5`` are the frozen ladder bounds; consequently
    normalized midpoint maps to exactly one.
    """
    spec = EVAPORATION_GAMMA_PARAM_SPECS["gamma"]
    return log_map_normalized(normalized, spec["lower"], spec["upper"])


def _parallel_evaporation_step(
    remaining_pet: torch.Tensor,
    wl: torch.Tensor,
    wd: torch.Tensor,
    lm: torch.Tensor,
    dm: torch.Tensor,
    gamma: torch.Tensor,
    nearzero: float,
) -> tuple[torch.Tensor, ...]:
    """Shared full/lite kernel for D_E and G_E.

    All ratios and both raw withdrawals are formed from the same pre-extraction
    WL/WD values.  The returned tuple is intentionally tensor-only and has a
    fixed layout so it is suitable for ``torch.compile(fullgraph=True)``.
    """
    er = torch.clamp(remaining_pet, min=0.0)

    lm_safe = torch.where(lm > nearzero, lm, torch.ones_like(lm))
    dm_safe = torch.where(dm > nearzero, dm, torch.ones_like(dm))
    root_capacity = lm + dm
    root_capacity_safe = torch.where(
        root_capacity > nearzero,
        root_capacity,
        torch.ones_like(root_capacity),
    )

    x_l = torch.clamp(wl / lm_safe, min=0.0, max=1.0)
    x_d = torch.clamp(wd / dm_safe, min=0.0, max=1.0)
    r_l = lm / root_capacity_safe
    r_d = dm / root_capacity_safe

    stress_l = stable_positive_power(x_l, gamma, floor=EVAPORATION_POWER_FLOOR)
    stress_d = stable_positive_power(x_d, gamma, floor=EVAPORATION_POWER_FLOOR)
    el_raw = r_l * er * stress_l
    ed_raw = r_d * er * stress_d

    # XAJ states are bounded below by zero.  The explicit available tensors
    # make the hard caps robust for direct kernel callers as well.
    wl_available = torch.clamp(wl, min=0.0)
    wd_available = torch.clamp(wd, min=0.0)
    el = torch.minimum(el_raw, wl_available)
    ed = torch.minimum(ed_raw, wd_available)
    wl_new = wl - el
    wd_new = wd - ed

    return (
        el,
        ed,
        er,
        wl_new,
        wd_new,
        el_raw,
        ed_raw,
        x_l,
        x_d,
        r_l,
        r_d,
    )


def _de_evaporation_step(
    remaining_pet: torch.Tensor,
    wl: torch.Tensor,
    wd: torch.Tensor,
    lm: torch.Tensor,
    dm: torch.Tensor,
    nearzero: float,
) -> tuple[torch.Tensor, ...]:
    """Full D_E kernel; the unity exponent is an exact linear reduction."""
    return _parallel_evaporation_step(
        remaining_pet,
        wl,
        wd,
        lm,
        dm,
        torch.ones_like(wl),
        nearzero,
    )


def _ge_evaporation_step(
    remaining_pet: torch.Tensor,
    wl: torch.Tensor,
    wd: torch.Tensor,
    lm: torch.Tensor,
    dm: torch.Tensor,
    gamma: torch.Tensor,
    nearzero: float,
) -> tuple[torch.Tensor, ...]:
    """Full G_E kernel using one basin-specific exponent."""
    return _parallel_evaporation_step(
        remaining_pet,
        wl,
        wd,
        lm,
        dm,
        gamma,
        nearzero,
    )


def _de_evaporation_step_lite(*args) -> tuple[torch.Tensor, ...]:
    """Compact D_E output: EL, ED and the two updated XAJ stores."""
    out = _de_evaporation_step(*args)
    return out[0], out[1], out[3], out[4]


def _ge_evaporation_step_lite(*args) -> tuple[torch.Tensor, ...]:
    """Compact G_E output: EL, ED and the two updated XAJ stores."""
    out = _ge_evaporation_step(*args)
    return out[0], out[1], out[3], out[4]


class _EvaporationModule(nn.Module):
    """Common compile wrapper matching the project's full/lite kernel style."""

    _full_step = staticmethod(_parallel_evaporation_step)
    _lite_step = staticmethod(_parallel_evaporation_step)
    uses_gamma = False

    def __init__(
        self, nearzero: float = 1e-8, *, lite: bool = False, compile_step: bool = True
    ):
        super().__init__()
        self.nearzero = nearzero
        self.lite = lite
        step = self._lite_step if lite else self._full_step
        self._step = torch.compile(step, fullgraph=True) if compile_step else step

    def _gamma(self, wl: torch.Tensor, gamma: torch.Tensor | None) -> torch.Tensor:
        if self.uses_gamma:
            if gamma is None:
                raise ValueError("G_E requires a basin-specific gamma tensor")
            return gamma
        return torch.ones_like(wl)

    def forward(
        self,
        remaining_pet: torch.Tensor,
        wl: torch.Tensor,
        wd: torch.Tensor,
        lm: torch.Tensor,
        dm: torch.Tensor,
        gamma: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        out = self._step(
            remaining_pet,
            wl,
            wd,
            lm,
            dm,
            self._gamma(wl, gamma),
            self.nearzero,
        )
        if self.lite:
            return out[0], out[1], out[3], out[4]
        return out


class EvaporationDE(_EvaporationModule):
    """Full D_E process module."""


class EvaporationDELite(EvaporationDE):
    """Lite D_E process module with the same scientific kernel."""

    def __init__(self, nearzero: float = 1e-8, *, compile_step: bool = True):
        super().__init__(nearzero, lite=True, compile_step=compile_step)


class EvaporationGE(_EvaporationModule):
    """Full G_E process module."""

    uses_gamma = True


class EvaporationGELite(EvaporationGE):
    """Lite G_E process module with the same scientific kernel."""

    def __init__(self, nearzero: float = 1e-8, *, compile_step: bool = True):
        super().__init__(nearzero, lite=True, compile_step=compile_step)


# Short names make the ladder usable in small host adapters without creating
# another naming convention.  The descriptive names above remain canonical.
DE = EvaporationDE
DELite = EvaporationDELite
GE = EvaporationGE
GELite = EvaporationGELite

__all__ = [
    "DE",
    "DELite",
    "GE",
    "GELite",
    "EvaporationDE",
    "EvaporationDELite",
    "EvaporationGE",
    "EvaporationGELite",
    "EVAPORATION_POWER_FLOOR",
    "normalized_to_gamma",
    "_de_evaporation_step",
    "_de_evaporation_step_lite",
    "_ge_evaporation_step",
    "_ge_evaporation_step_lite",
]
