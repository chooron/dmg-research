"""Candidate formula pool adapter for Static Formula-MoE.

Provides a uniform lookup/call interface on top of FORMULA_REGISTRY
without exposing the internal registry structure.  Does NOT perform
full HBV state advancement — callers are responsible for state updates.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from model.flux import snow, recharge, aet, response
from model.flux.formula_registry import (
    FORMULA_REGISTRY,
    get_node_formulas,
    get_routing_policy,
)

# ---------------------------------------------------------------------------
# Direct function lookup + positional-arg order per formula ID
# ---------------------------------------------------------------------------

def _seasonal_wrapper(*args: torch.Tensor) -> torch.Tensor:
    """S4: seasonal degree-day melt = cfmax_seasonal + linear melt."""
    T, TT, CFMAX_0, a_s, phi_s, doy, SWE = args
    CFMAX_t = snow.cfmax_seasonal(CFMAX_0, a_s, phi_s, doy)
    return snow.snowmelt_linear_degreeday(T, TT, CFMAX_t, SWE)


_FORMULA_META = {
    "S0": {"func": snow.snowmelt_linear_degreeday,
           "args": ["T", "TT", "CFMAX", "SWE"]},
    "S4": {"func": _seasonal_wrapper,
           "args": ["T", "TT", "CFMAX_0", "a_s", "phi_s", "doy", "SWE"]},
    "S5": {"func": snow.snowmelt_exponential,
           "args": ["T", "TT", "CFMAX", "c_m", "SWE"]},
    "R0": {"func": recharge.beta_recharge,
           "args": ["I", "SM", "FC", "beta"]},
    "R4": {"func": recharge.saturation_threshold_recharge,
           "args": ["I", "SM", "FC", "a_r", "c_r"]},
    "R5": {"func": recharge.variable_contributing_area_recharge,
           "args": ["I", "SM", "FC", "b_v"]},
    "E0": {"func": aet.aet_hbv_default,
           "args": ["PET", "SM", "LP", "FC"]},
    "E3": {"func": aet.aet_power_law,
           "args": ["PET", "SM", "FC", "gamma_E"]},
    "E4": {"func": aet.feddes_threshold_aet,
           "args": ["PET", "SM", "FC", "s_w", "s_o"]},
    "Q0": {"func": response.response_two_reservoir,
           "args": ["SUZ", "SLZ", "K_0", "K_1", "K_2", "UZL"]},
    "Q2": {"func": response.response_nonlinear,
           "args": ["SUZ", "SLZ", "K_1", "K_2", "alpha_Q"]},
    "Q5": {"func": response.response_delayed_step,
           "args": ["R_in", "S_1", "S_2", "PART", "K_1", "K_2"]},
}


class CandidateFormulaPool:
    """Read-only view over the formula registry with call dispatch.

    Usage::

        pool = CandidateFormulaPool()
        for fid in pool.formulas("snow"):
            result = pool.call_formula("snow", fid, T=..., TT=..., ...)
    """

    def __init__(self, registry: dict | None = None) -> None:
        self._registry = registry if registry is not None else FORMULA_REGISTRY

    # -- query ------------------------------------------------------------

    def nodes(self) -> list[str]:
        """Return sorted list of registered process nodes."""
        return sorted(self._registry.keys())

    def formulas(self, node: str, status: str = "main") -> list[str]:
        """Return formula-ids for *node* filtered by *status*."""
        return [e["id"] for e in get_node_formulas(node, status)]

    def routing_policy(self, node: str) -> str:
        """Return the MoE routing policy for *node*."""
        return get_routing_policy(node)

    def formula_info(self, node: str, formula_id: str) -> dict:
        """Return the full registry entry dict for *formula_id*."""
        for status in ("main", "ablation_only", "extension_only", "pet_correction"):
            for e in get_node_formulas(node, status):
                if e["id"] == formula_id:
                    return dict(e)
        raise ValueError(
            f"Formula '{formula_id}' not found in node '{node}'."
        )

    def get_formula(self, node: str, formula_id: str) -> Callable:
        """Return the callable for *formula_id*.

        Raises ValueError if the formula is not registered in _FORMULA_META.
        """
        if formula_id not in _FORMULA_META:
            raise ValueError(
                f"Formula '{formula_id}' is not in the callable dispatch table."
            )
        return _FORMULA_META[formula_id]["func"]

    def call_formula(self, node: str, formula_id: str, **kwargs: Any) -> torch.Tensor | tuple:
        """Call *formula_id* with keyword arguments, dispatched positionally.

        Returns the formula's raw output (tensor or tuple of tensors).
        """
        meta = _FORMULA_META[formula_id]
        pos_args = [kwargs[name] for name in meta["args"]]
        return meta["func"](*pos_args)
