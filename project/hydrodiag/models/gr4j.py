from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .base import BaseHydrologicalModel
from .parameter_specs import GR4J_PARAM_SPECS
from .unit_hydro import compute_gr4j_uh_ordinates
from .utils import validate_forcings, validate_params

GR4J_UH1_MAX = 15
GR4J_UH2_MAX = 30


def _gr4j_step(
    precip_t: torch.Tensor,
    pet_t: torch.Tensor,
    s_prod: torch.Tensor,
    s_route: torch.Tensor,
    uh1_buf: torch.Tensor,
    uh2_buf: torch.Tensor,
    uh1_ord: torch.Tensor,
    uh2_ord: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    nearzero: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B = precip_t.shape[0]

    mask_pn = precip_t >= pet_t
    p_n = torch.where(mask_pn, precip_t - pet_t, torch.zeros_like(precip_t))
    pe_n = torch.where(mask_pn, torch.zeros_like(precip_t), pet_t - precip_t)

    # Production store computations
    ratio = torch.clamp(s_prod / (x1 + nearzero), min=0.0, max=1.0)
    # When precip >= pet: p_s from eq. 3
    tanh_pn_x1 = torch.tanh(p_n / (x1 + nearzero))
    p_s_calc = (x1 * (1.0 - ratio * ratio) * tanh_pn_x1) / (
        1.0 + ratio * tanh_pn_x1 + nearzero
    )
    # When precip < pet: e_s from eq. 4
    tanh_pen_x1 = torch.tanh(pe_n / (x1 + nearzero))
    e_s_calc = (s_prod * (2.0 - ratio) * tanh_pen_x1) / (
        1.0 + (1.0 - ratio) * tanh_pen_x1 + nearzero
    )
    p_s = torch.where(mask_pn, p_s_calc, torch.zeros_like(precip_t))
    e_s = torch.where(mask_pn, torch.zeros_like(precip_t), e_s_calc)

    s_prod = s_prod - e_s + p_s

    # Percolation (eq. 5)
    n4 = 4.0 / 9.0 * s_prod / (x1 + nearzero)
    perc = s_prod * (1.0 - (1.0 + n4**4.0) ** (-0.25))
    s_prod = s_prod - perc

    # Total water reaching routing
    p_r = perc + (p_n - p_s)
    p_r = torch.clamp(p_r, min=0.0)

    # Split for unit hydrographs
    p_r_uh1 = 0.9 * p_r
    p_r_uh2 = 0.1 * p_r

    # Update UH1 buffer (shift left, add new contributions)
    uh1_shifted = torch.cat(
        [
            uh1_buf[:, 1:],
            torch.zeros(B, 1, device=precip_t.device, dtype=precip_t.dtype),
        ],
        dim=1,
    )
    uh1_buf = uh1_shifted + uh1_ord * p_r_uh1[:, None]
    q_uh1_out = uh1_buf[:, 0]

    # Update UH2 buffer
    uh2_shifted = torch.cat(
        [
            uh2_buf[:, 1:],
            torch.zeros(B, 1, device=precip_t.device, dtype=precip_t.dtype),
        ],
        dim=1,
    )
    uh2_buf = uh2_shifted + uh2_ord * p_r_uh2[:, None]
    q_uh2_out = uh2_buf[:, 0]

    # Groundwater exchange (eq. 18)
    ratio_r = torch.clamp(s_route / (x3 + nearzero), min=0.0)
    gw_exchange = x2 * ratio_r**3.5

    # Update routing store
    s_route_in = s_route + q_uh1_out + gw_exchange
    s_route = torch.clamp(s_route_in, min=0.0)

    # Outflow of routing store
    ratio_r2 = torch.clamp(s_route / (x3 + nearzero), min=0.0)
    q_r = s_route * (1.0 - (1.0 + ratio_r2**4.0) ** (-0.25))
    s_route = s_route - q_r

    # Direct flow component
    q_d = torch.clamp(q_uh2_out + gw_exchange, min=0.0)

    q_t = q_r + q_d

    return q_t, s_prod, s_route, uh1_buf, uh2_buf


class GR4J(BaseHydrologicalModel):
    """GR4J model — 4-parameter lumped hydrological model.

    Based on:
    Perrin, C., Michel, C., & Andreassian, V. (2003). Journal of Hydrology, 279(1).

    Unit hydrograph ordinates are computed via a differentiable S-curve difference
    method (inspired by dmotpy DplUHBase pattern), ensuring x4 remains learnable
    through gradient descent.

    Routing uses sequential UH buffer updates within the compiled step kernel,
    with UH ordinates pre-computed outside the step loop.
    """

    UH1_MAX = GR4J_UH1_MAX
    UH2_MAX = GR4J_UH2_MAX

    def __init__(self, nearzero: float = 1e-8, compact_output: bool = False):
        super().__init__()
        self.nearzero = nearzero
        self.compact_output = compact_output
        self._step = torch.compile(_gr4j_step, fullgraph=True)

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return GR4J_PARAM_SPECS

    def forward(
        self,
        forcings: dict[str, torch.Tensor],
        params: dict[str, torch.Tensor],
        initial_states: Optional[dict[str, torch.Tensor]] = None,
        return_states: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        precip, pet, temp, device = validate_forcings(forcings)
        batch, nsteps = precip.shape
        dtype = precip.dtype
        validate_params(params, self.parameter_specs, batch, device, dtype)

        x1 = params["x1"]
        x2 = params["x2"]
        x3 = params["x3"]
        x4 = params["x4"]

        uh1_ord, uh2_ord = compute_gr4j_uh_ordinates(x4, self.UH1_MAX)
        uh2_ord = compute_gr4j_uh_ordinates(x4, self.UH2_MAX)[1]

        s_prod, s_route, uh1_buf, uh2_buf = self._init_states(
            batch, device, dtype, initial_states, x1, x3
        )

        qsim, (s_prod, s_route, uh1_buf, uh2_buf) = self._step_loop(
            precip,
            pet,
            nsteps,
            batch,
            device,
            dtype,
            s_prod,
            s_route,
            uh1_buf,
            uh2_buf,
            uh1_ord,
            uh2_ord,
            x1,
            x2,
            x3,
        )

        if self.compact_output and not return_states:
            return qsim, {}

        aux = {
            "s_prod": s_prod,
            "s_route": s_route,
        }
        if return_states:
            aux["final_states"] = {
                "s_prod": s_prod,
                "s_route": s_route,
                # The UH buffers hold water that is scheduled to leave in
                # subsequent timesteps.  They are therefore part of the
                # model state and must be retained for an exact continuation.
                "uh1_buf": uh1_buf,
                "uh2_buf": uh2_buf,
            }

        return qsim, aux

    def _init_states(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        initial_states: Optional[dict[str, torch.Tensor]] = None,
        x1: Optional[torch.Tensor] = None,
        x3: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        if initial_states is not None:
            return (
                initial_states.get(
                    "s_prod",
                    x1 * 0.5
                    if x1 is not None
                    else torch.full((batch,), 100.0, device=device, dtype=dtype),
                ),
                initial_states.get(
                    "s_route",
                    x3 * 0.5
                    if x3 is not None
                    else torch.full((batch,), 100.0, device=device, dtype=dtype),
                ),
                initial_states.get(
                    "uh1_buf",
                    torch.zeros(batch, self.UH1_MAX, device=device, dtype=dtype),
                ),
                initial_states.get(
                    "uh2_buf",
                    torch.zeros(batch, self.UH2_MAX, device=device, dtype=dtype),
                ),
            )
        return (
            x1 * 0.5
            if x1 is not None
            else torch.full((batch,), 100.0, device=device, dtype=dtype),
            x3 * 0.5
            if x3 is not None
            else torch.full((batch,), 100.0, device=device, dtype=dtype),
            torch.zeros(batch, self.UH1_MAX, device=device, dtype=dtype),
            torch.zeros(batch, self.UH2_MAX, device=device, dtype=dtype),
        )

    def _step_loop(
        self,
        precip: torch.Tensor,
        pet: torch.Tensor,
        nsteps: int,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        s_prod: torch.Tensor,
        s_route: torch.Tensor,
        uh1_buf: torch.Tensor,
        uh2_buf: torch.Tensor,
        uh1_ord: torch.Tensor,
        uh2_ord: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        qsim = torch.zeros(batch, nsteps, device=device, dtype=dtype)
        nz = self.nearzero

        for t in range(nsteps):
            qsim[:, t], s_prod, s_route, uh1_buf, uh2_buf = self._step(
                precip[:, t],
                pet[:, t],
                s_prod,
                s_route,
                uh1_buf,
                uh2_buf,
                uh1_ord,
                uh2_ord,
                x1,
                x2,
                x3,
                nz,
            )

        return qsim, (s_prod, s_route, uh1_buf, uh2_buf)


class GR4JLite(GR4J):
    """GR4J training path returning only streamflow."""

    def __init__(self, nearzero: float = 1e-8):
        super().__init__(nearzero=nearzero, compact_output=True)
