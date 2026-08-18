"""Temperature-agnostic, parameter-matched precipitation-delay control."""

from __future__ import annotations

from typing import Any, Optional

import torch

from .base import BaseHydrologicalModel
from .parameter_specs import PRECIP_DELAY_PARAM_SPECS
from .utils import validate_forcings, validate_params

PRECIP_DELAY_MIN_TAU = 1e-6


def _precip_delay_step(
    precip_t: torch.Tensor,
    storage: torch.Tensor,
    alpha: torch.Tensor,
    tau: torch.Tensor,
    nearzero: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the conservative temporary precipitation reservoir.

    The update is:

        S_pre = S[t-1] + alpha * P[t]
        R[t] = (1 - exp(-1 / tau)) * S_pre
        S[t] = S_pre - R[t]
        P_star[t] = (1 - alpha) * P[t] + R[t]

    ``nearzero`` is retained in the signature for consistency with the other
    compiled hydrological kernels.  ``expm1`` avoids loss of precision for
    large time scales.
    """
    alpha_safe = torch.clamp(alpha, 0.0, 1.0)
    tau_safe = torch.clamp(tau, min=PRECIP_DELAY_MIN_TAU)
    storage_pre = storage + alpha_safe * precip_t
    release_fraction = -torch.expm1(-1.0 / tau_safe)
    release = release_fraction * storage_pre
    storage = storage_pre - release
    effective_precip = (1.0 - alpha_safe) * precip_t + release
    return effective_precip, storage, release


class PrecipitationDelay(BaseHydrologicalModel):
    """Two-parameter temperature-independent precipitation-delay control.

    ``pd_alpha`` is the fraction of precipitation entering temporary storage;
    ``pd_tau`` is its release time scale in days.  The module is conservative
    and can be inserted between raw precipitation and a runoff model.
    """

    def __init__(self, nearzero: float = 1e-8):
        super().__init__()
        self.nearzero = nearzero
        self._step = torch.compile(_precip_delay_step, fullgraph=True)

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return PRECIP_DELAY_PARAM_SPECS

    def forward(
        self,
        forcings: dict[str, torch.Tensor],
        params: dict[str, torch.Tensor],
        initial_states: Optional[dict[str, torch.Tensor]] = None,
        return_states: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        precip, _pet, _temp, device = validate_forcings(forcings)
        batch, nsteps = precip.shape
        dtype = precip.dtype
        validate_params(params, self.parameter_specs, batch, device, dtype)

        storage = (
            initial_states.get("S", torch.zeros(batch, device=device, dtype=dtype))
            if initial_states is not None
            else torch.zeros(batch, device=device, dtype=dtype)
        )
        alpha = params["pd_alpha"]
        tau = params["pd_tau"]
        effective = torch.zeros_like(precip)
        releases = torch.zeros_like(precip)
        storage_trace = torch.zeros_like(precip)

        for t in range(nsteps):
            effective[:, t], storage, releases[:, t] = self._step(
                precip[:, t], storage, alpha, tau, self.nearzero
            )
            storage_trace[:, t] = storage

        aux: dict[str, Any] = {
            "effective_precip": effective,
            "released_precip": releases,
            "delay_storage": storage,
            "storage_trace": storage_trace,
            "release_fraction": -torch.expm1(
                -1.0 / torch.clamp(tau, min=PRECIP_DELAY_MIN_TAU)
            ),
        }
        if return_states:
            aux["final_states"] = {"S": storage}
        return effective, aux
