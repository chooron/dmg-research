"""Differentiable SIMHYD rainfall-runoff model with gamma-UH routing.

The runoff-generation equations follow ``hydrogo/LHMP``'s
``simhyd_cemaneige.py``.  Snow processing is deliberately kept outside this
module so that exactly two configurations are exposed: :class:`SIMHYD`
(SIMHYD + gamma UH) and ``SIMHYDWithCemaNeige`` (CemaNeige + SIMHYD + gamma
UH).  Both use the same differentiable, state-carrying gamma unit hydrograph
as the local XAJ implementation.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .base import BaseHydrologicalModel
from .parameter_specs import SIMHYD_PARAM_SPECS
from .utils import validate_forcings, validate_params
from .xaj import _apply_uh_routing, _gamma_uh_ordinates


SIMHYD_UH_MAX_LEN = 15
SIMHYD_MIN_INSC = 1e-6
SIMHYD_MIN_COEFF = 1e-6
SIMHYD_MIN_SMSC = 1e-6


def _simhyd_step_impl(
    precip_t: torch.Tensor,
    pet_t: torch.Tensor,
    soil: torch.Tensor,
    groundwater: torch.Tensor,
    insc: torch.Tensor,
    coeff: torch.Tensor,
    sq: torch.Tensor,
    smsc: torch.Tensor,
    sub: torch.Tensor,
    crak: torch.Tensor,
    k: torch.Tensor,
    etmul: torch.Tensor,
    nearzero: float,
    return_diagnostics: bool,
) -> tuple[torch.Tensor, ...]:
    """Advance the SIMHYD stores by one daily time step.

    Unlike the linked NumPy code, soil overflow is computed before clipping
    the soil store and is transferred to groundwater.  ET and baseflow are
    capped by available storage.  These two details make the discrete update
    exactly mass conserving over the full advertised parameter range.
    """
    precip = torch.clamp(precip_t, min=0.0)
    pet = torch.clamp(pet_t * etmul, min=0.0)

    # The reference implementation treats intercepted rainfall as same-day
    # evaporation, limited by interception capacity, rainfall, and PET.
    # A positive floor also avoids a repeated min-branch tie between a zero
    # interception capacity and the exact zero liquid precipitation emitted
    # by CemaNeige during cold periods.
    insc_safe = torch.clamp(insc, min=SIMHYD_MIN_INSC)
    interception = torch.minimum(torch.minimum(insc_safe, pet), precip)
    rainfall_excess = precip - interception
    pet_remaining = pet - interception

    # ``smsc=0`` is outside the calibrated range but can be supplied by a
    # direct caller or reached at a degenerate optimizer boundary.  Using the
    # raw value makes the soil-capacity branch repeatedly tie at zero during
    # a dry rollout, which gives KGE backward an enormous ``smsc`` gradient.
    smsc_safe = torch.clamp(smsc, min=SIMHYD_MIN_SMSC)
    soil_ratio = torch.clamp(soil / (smsc_safe + nearzero), 0.0, 1.0)
    # An exact zero coefficient lies on a repeated ``minimum`` branch tie
    # when the soil is dry.  Across a long rollout that tie can amplify the
    # coefficient gradient to ~1e33 even though the forward values stay
    # finite.  Keep both calibrated and directly supplied values away from
    # that singular boundary.
    coeff_safe = torch.clamp(coeff, min=SIMHYD_MIN_COEFF)
    infiltration_capacity = coeff_safe * torch.exp(-sq * soil_ratio)
    infiltration = torch.minimum(infiltration_capacity, rainfall_excess)
    direct_runoff = rainfall_excess - infiltration

    interflow = sub * soil_ratio * infiltration
    recharge = crak * soil_ratio * (infiltration - interflow)
    soil_fill = infiltration - interflow - recharge

    soil_available = soil + soil_fill
    soil_evap = torch.minimum(10.0 * soil_ratio, pet_remaining)
    soil_evap = torch.minimum(soil_evap, soil_available)

    soil_after_evap = soil_available - soil_evap
    soil_overflow = torch.clamp(soil_after_evap - smsc_safe, min=0.0)
    soil_new = soil_after_evap - soil_overflow

    recharge_total = recharge + soil_overflow
    # The valid parameter range is 0 <= k <= 1, so this product is exactly
    # equivalent to ``minimum(k * groundwater, groundwater)`` for physical
    # non-negative groundwater.  Avoiding the equality branch at k=1 keeps
    # the dry-store backward pass single-valued and finite.
    k_safe = torch.clamp(k, min=0.0, max=1.0)
    baseflow = k_safe * groundwater
    groundwater_new = groundwater + recharge_total - baseflow

    runoff = direct_runoff + interflow + baseflow
    evap = interception + soil_evap

    if not return_diagnostics:
        return runoff, soil_new, groundwater_new

    return (
        runoff,
        evap,
        soil_new,
        groundwater_new,
        interception,
        direct_runoff,
        interflow,
        recharge_total,
        baseflow,
    )


def _simhyd_step(
    precip_t: torch.Tensor,
    pet_t: torch.Tensor,
    soil: torch.Tensor,
    groundwater: torch.Tensor,
    insc: torch.Tensor,
    coeff: torch.Tensor,
    sq: torch.Tensor,
    smsc: torch.Tensor,
    sub: torch.Tensor,
    crak: torch.Tensor,
    k: torch.Tensor,
    etmul: torch.Tensor,
    nearzero: float,
) -> tuple[torch.Tensor, ...]:
    """Historical SIMHYD step including diagnostic outputs."""
    return _simhyd_step_impl(
        precip_t, pet_t, soil, groundwater, insc, coeff, sq, smsc,
        sub, crak, k, etmul, nearzero, True,
    )


def _simhyd_step_compact(
    precip_t: torch.Tensor,
    pet_t: torch.Tensor,
    soil: torch.Tensor,
    groundwater: torch.Tensor,
    insc: torch.Tensor,
    coeff: torch.Tensor,
    sq: torch.Tensor,
    smsc: torch.Tensor,
    sub: torch.Tensor,
    crak: torch.Tensor,
    k: torch.Tensor,
    etmul: torch.Tensor,
    nearzero: float,
) -> tuple[torch.Tensor, ...]:
    """Lean SIMHYD step returning only runoff and recursive states."""
    return _simhyd_step_impl(
        precip_t, pet_t, soil, groundwater, insc, coeff, sq, smsc,
        sub, crak, k, etmul, nearzero, False,
    )


def _routing_pending(
    runoff_buffer: torch.Tensor,
    uh_ordinates: torch.Tensor,
) -> torch.Tensor:
    """Return water still queued in a causal finite gamma UH."""
    # Buffer order is oldest -> newest.  Its pending fractions are therefore
    # [w[-1], w[-2:].sum(), ..., w[1:].sum()].
    pending_fractions = torch.cumsum(
        torch.flip(uh_ordinates[:, 1:], dims=[-1]), dim=-1
    )
    return (runoff_buffer * pending_fractions).sum(dim=-1)


def _route_simhyd_runoff(
    runoff_instant: torch.Tensor,
    runoff_uh_buffer: torch.Tensor,
    a: torch.Tensor,
    theta: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route a completed SIMHYD runoff-generation sequence.

    Keeping this sequence-level operation separate is intentional: unlike the
    CN/TGD preprocessing and SIMHYD storage update, the finite UH needs the
    complete instantaneous-runoff sequence (plus continuation history).
    """
    uh_ordinates = _gamma_uh_ordinates(a, theta, SIMHYD_UH_MAX_LEN, device, dtype)
    uh_ordinates = uh_ordinates / uh_ordinates.sum(dim=-1, keepdim=True)
    runoff_with_history = torch.cat((runoff_uh_buffer, runoff_instant), dim=1)
    routed_all = _apply_uh_routing(runoff_with_history, uh_ordinates)
    start = SIMHYD_UH_MAX_LEN - 1
    qsim = routed_all[:, start:start + runoff_instant.shape[1]]
    next_buffer = runoff_with_history[:, -(SIMHYD_UH_MAX_LEN - 1):]
    routing_storage = _routing_pending(next_buffer, uh_ordinates)
    return qsim, next_buffer, uh_ordinates, routing_storage


class SIMHYD(BaseHydrologicalModel):
    """SIMHYD runoff generation with differentiable gamma-UH routing.

    Inputs, parameter tensors, state passing, and outputs follow the same
    ``BaseHydrologicalModel`` interface as :class:`models.xaj.XAJ`.
    """

    state_names = ["soil", "groundwater", "runoff_uh_buffer"]
    routing_method = "gamma"

    def __init__(self, nearzero: float = 1e-8, compact_output: bool = False):
        super().__init__()
        self.nearzero = nearzero
        self.compact_output = compact_output
        self._step = torch.compile(_simhyd_step, fullgraph=True)
        self._compact_step = torch.compile(_simhyd_step_compact, fullgraph=True)

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return SIMHYD_PARAM_SPECS

    def forward(
        self,
        forcings: dict[str, torch.Tensor],
        params: dict[str, torch.Tensor],
        initial_states: Optional[dict[str, torch.Tensor]] = None,
        return_states: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        precip, pet, _temp, device = validate_forcings(forcings)
        batch, nsteps = precip.shape
        dtype = precip.dtype
        validate_params(params, self.parameter_specs, batch, device, dtype)

        soil, groundwater, runoff_uh_buffer = self._init_states(
            batch, device, dtype, params["simhyd_smsc"], initial_states
        )

        if self.compact_output and not return_states:
            return self._forward_lite(
                precip, pet, params, soil, groundwater,
                runoff_uh_buffer, device, dtype,
            )
        else:
            runoff_instant = torch.zeros(batch, nsteps, device=device, dtype=dtype)
            evap = torch.zeros_like(runoff_instant)
            interception = torch.zeros_like(runoff_instant)
            direct_runoff = torch.zeros_like(runoff_instant)
            interflow = torch.zeros_like(runoff_instant)
            recharge = torch.zeros_like(runoff_instant)
            baseflow = torch.zeros_like(runoff_instant)

        for t in range(nsteps):
            # ``[:, t]`` is a view with a time-varying storage offset.  The
            # fullgraph daily kernel must receive contiguous vectors so a
            # long sequence does not compile one specialization per day.
            precip_t = precip[:, t].contiguous()
            pet_t = pet[:, t].contiguous()
            (
                runoff_instant[:, t], evap[:, t], soil, groundwater,
                interception[:, t], direct_runoff[:, t], interflow[:, t],
                recharge[:, t], baseflow[:, t],
            ) = self._step(
                precip_t, pet_t, soil, groundwater,
                params["simhyd_insc"], params["simhyd_coeff"],
                params["simhyd_sq"], params["simhyd_smsc"],
                params["simhyd_sub"], params["simhyd_crak"],
                params["simhyd_k"], params["simhyd_etmul"],
                self.nearzero,
            )

        qsim, runoff_uh_buffer, uh_ordinates, routing_storage = _route_simhyd_runoff(
            runoff_instant,
            runoff_uh_buffer,
            params["simhyd_a"],
            params["simhyd_theta"],
            device,
            dtype,
        )

        aux: dict[str, Any] = {
            "routing_method": self.routing_method,
            "gamma_uh_ordinates": uh_ordinates,
            "evap": evap,
            "runoff_instant": runoff_instant,
            "runoff_routed": qsim,
            "interception": interception,
            "direct_runoff": direct_runoff,
            "interflow": interflow,
            "recharge": recharge,
            "baseflow": baseflow,
            "soil": soil,
            "groundwater": groundwater,
            "routing_storage": routing_storage,
        }
        states = (soil, groundwater, runoff_uh_buffer)
        if return_states:
            aux["final_states"] = {
                name: value for name, value in zip(self.state_names, states)
            }
        return qsim, aux

    def _forward_lite(
        self,
        precip: torch.Tensor,
        pet: torch.Tensor,
        params: dict[str, torch.Tensor],
        soil: torch.Tensor,
        groundwater: torch.Tensor,
        runoff_uh_buffer: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Branch-free streamflow-only SIMHYD path."""
        runoff_values = []
        for t in range(precip.shape[1]):
            # Keep the compiled kernel independent of the time-slice offset.
            precip_t = precip[:, t].contiguous()
            pet_t = pet[:, t].contiguous()
            runoff_t, soil, groundwater = self._compact_step(
                precip_t, pet_t, soil, groundwater,
                params["simhyd_insc"], params["simhyd_coeff"],
                params["simhyd_sq"], params["simhyd_smsc"],
                params["simhyd_sub"], params["simhyd_crak"],
                params["simhyd_k"], params["simhyd_etmul"],
                self.nearzero,
            )
            runoff_values.append(runoff_t)
        runoff_instant = torch.stack(runoff_values, dim=1)
        qsim, _runoff_uh_buffer, _uh_ordinates, _routing_storage = _route_simhyd_runoff(
            runoff_instant, runoff_uh_buffer,
            params["simhyd_a"], params["simhyd_theta"], device, dtype,
        )
        return qsim, {}

    def _init_states(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        smsc: torch.Tensor,
        initial_states: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if initial_states is None:
            initial_states = {}
        # Initialization happens before the daily step kernel, so protect the
        # default soil store here as well as the step's downstream smsc uses.
        smsc_safe = torch.clamp(smsc, min=SIMHYD_MIN_SMSC)
        return (
            initial_states.get("soil", 0.5 * smsc_safe),
            initial_states.get(
                "groundwater", torch.zeros(batch, device=device, dtype=dtype)
            ),
            initial_states.get(
                "runoff_uh_buffer",
                torch.zeros(
                    batch, SIMHYD_UH_MAX_LEN - 1, device=device, dtype=dtype
                ),
            ),
        )


class SIMHYDLite(SIMHYD):
    """SIMHYD training path that returns only routed streamflow."""

    def __init__(self, nearzero: float = 1e-8):
        super().__init__(nearzero=nearzero, compact_output=True)
