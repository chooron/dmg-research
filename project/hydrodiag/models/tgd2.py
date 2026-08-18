"""Two-parameter temperature-dependent generic precipitation memory.

TGD2 is a temperature-dependent generic precipitation-memory module, not a
snow accumulation and melt model. Every precipitation input enters one linear
reservoir and continues to leak under cold conditions.
"""
from __future__ import annotations

from typing import Any, Optional
import torch

from .base import BaseHydrologicalModel
from .parameter_specs import TGD2_EPS_DAYS, TGD2_PARAM_SPECS, TGD2_STRUCTURE_VERSION, TGD2_T_REF_C, TGD2_T_SCALE_C
from .utils import validate_forcings, validate_params


def tgd2_step(precip_t: torch.Tensor, temp_t: torch.Tensor, storage: torch.Tensor,
              tau_warm: torch.Tensor, delta_tau_cold: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One conservative TGD2 update, vectorized over the leading batch axis."""
    cold_gate = torch.sigmoid((TGD2_T_REF_C - temp_t) / TGD2_T_SCALE_C)
    tau = tau_warm + delta_tau_cold * cold_gate
    retention = torch.exp(-1.0 / tau.clamp_min(TGD2_EPS_DAYS))
    # Physical precipitation and storage are non-negative.  Explicit clamps
    # keep long dry sequences from propagating tiny negative round-off values
    # into the continuous state export.
    precip_t = precip_t.clamp_min(0.0)
    storage = storage.clamp_min(0.0)
    available = (storage + precip_t).clamp_min(0.0)
    effective = ((1.0 - retention) * available).clamp_min(0.0)
    storage_new = (retention * available).clamp_min(0.0)
    return effective, storage_new, tau, retention


class TemperatureDependentGenericDelay2(BaseHydrologicalModel):
    """Conservative two-parameter precipitation memory with fixed temperature gate."""
    structure_version = TGD2_STRUCTURE_VERSION

    def __init__(self, compact_output: bool = False):
        super().__init__()
        self.compact_output = compact_output
        self._step = torch.compile(tgd2_step, fullgraph=True)

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return TGD2_PARAM_SPECS

    def forward(self, forcings: dict[str, torch.Tensor], params: dict[str, torch.Tensor],
                initial_states: Optional[dict[str, torch.Tensor]] = None, return_states: bool = False):
        precip, _pet, temp, device = validate_forcings(forcings)
        batch, nsteps = precip.shape
        validate_params(params, self.parameter_specs, batch, device, precip.dtype)
        if initial_states is None:
            storage = torch.zeros(batch, device=device, dtype=precip.dtype)
        else:
            storage = initial_states.get("storage")
            if storage is None or storage.shape != (batch,) or storage.device != device or storage.dtype != precip.dtype:
                raise ValueError("Initial TGD2 state 'storage' must be [batch] on the forcing device/dtype")
        effective = torch.empty_like(precip)
        diagnostics = not self.compact_output or return_states
        if diagnostics:
            storage_trace, tau_trace, retention_trace = (torch.empty_like(precip) for _ in range(3))
        for t in range(nsteps):
            effective_t, storage, tau_t, retention_t = self._step(
                precip[:, t], temp[:, t], storage, params["tgd_tau_warm"], params["tgd_delta_tau_cold"]
            )
            effective[:, t] = effective_t
            if diagnostics:
                storage_trace[:, t], tau_trace[:, t], retention_trace[:, t] = storage, tau_t, retention_t
        if not diagnostics:
            return effective, {}
        aux: dict[str, Any] = {
            "tgd2_structure_version": self.structure_version,
            "effective_precipitation": effective,
            "tgd2_storage": storage_trace,
            "tgd2_tau": tau_trace,
            "tgd2_retention": retention_trace,
        }
        if return_states:
            aux["final_states"] = {"storage": storage}
        return effective, aux
