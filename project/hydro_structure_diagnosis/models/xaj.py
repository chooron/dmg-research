"""XinAnJiang (XAJ) model.

Core XAJ runoff-generation processes:
- Three-layer evaporation model
- Tension water capacity curve (runoff generation)
- Free water reservoir separation (surface / interflow / groundwater)
- Linear reservoir routing for interflow and groundwater
- Gamma-distribution unit hydrograph routing for surface runoff

Reference:
https://github.com/OuyangWenyu/hydromodel/blob/master/hydromodel/models/xaj.py
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydrodl2.core.calc import uh_conv, uh_gamma

from .base import BaseHydrologicalModel
from .parameter_specs import XAJ_LITE_PARAM_SPECS, XAJ_PARAM_SPECS
from .utils import validate_forcings, validate_params
from .structure_evaporation import _parallel_evaporation_step
from .structure_response import _analytic_subsurface_response_step

# All Gamma-UH routes use the bettermodel/HBV convention: hydrodl2's
# differentiable Gamma UH with a 15-day kernel.
XAJ_UH_MAX_LEN = 15
XAJ_LITE_UH_MAX_LEN = XAJ_UH_MAX_LEN


def _rootzone_moisture_stress_evaporation(
    remaining_pet: torch.Tensor,
    wl: torch.Tensor,
    wd: torch.Tensor,
    lm: torch.Tensor,
    dm: torch.Tensor,
    tau_e: torch.Tensor,
    nearzero: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Conservative aggregated-root-zone evaporation and WL/WD bookkeeping."""
    root_storage = wl + wd
    root_capacity = lm + dm
    z_root = torch.clamp(root_storage / (root_capacity + nearzero), min=0.0, max=1.0)
    stress = torch.clamp(z_root / (tau_e + nearzero), min=0.0, max=1.0)
    er = torch.minimum(remaining_pet * stress, root_storage)
    safe_root = root_storage > nearzero
    el = torch.where(safe_root, er * wl / (root_storage + nearzero), torch.zeros_like(er))
    el = torch.minimum(el, wl)
    ed = er - el
    # Reassign only a possible floating-point deep-layer excess to WL.  This
    # preserves EL + ED == ER instead of silently discarding evaporation.
    excess_ed = torch.clamp(ed - wd, min=0.0)
    ed = ed - excess_ed
    el = el + excess_ed
    return el, ed, er, z_root, stress


def _xaj_step_impl(
    precip_t: torch.Tensor,
    pet_t: torch.Tensor,
    wu: torch.Tensor,
    wl: torch.Tensor,
    wd: torch.Tensor,
    s: torch.Tensor,
    fr: torch.Tensor,
    qi: torch.Tensor,
    qg: torch.Tensor,
    k: torch.Tensor,
    b: torch.Tensor,
    im: torch.Tensor,
    um: torch.Tensor,
    lm: torch.Tensor,
    dm: torch.Tensor,
    c: torch.Tensor,
    sm: torch.Tensor,
    ex: torch.Tensor,
    ki: torch.Tensor,
    kg: torch.Tensor,
    ci: torch.Tensor,
    cg: torch.Tensor,
    nearzero: float,
    wm: torch.Tensor,
    wmm: torch.Tensor,
    ms: torch.Tensor,
    one_minus_im: torch.Tensor,
    one_minus_ki_kg: torch.Tensor,
    return_diagnostics: bool,
    evaporation_scheme: int = 0,
    root_tau_e: Optional[torch.Tensor] = None,
    response_scheme: int = 0,
    structure_gamma: Optional[torch.Tensor] = None,
    structure_tau_0: Optional[torch.Tensor] = None,
    structure_beta: Optional[torch.Tensor] = None,
    structure_z0: Optional[torch.Tensor] = None,
) -> tuple:
    B = precip_t.shape[0]

    # --- Runoff generation: three-layer evaporation model ---
    prcp = torch.clamp(precip_t, min=0.0)
    pet_adj = pet_t * k
    pet_adj = torch.clamp(pet_adj, min=0.0)

    eu = torch.minimum(wu + prcp, pet_adj)
    remaining_pet = torch.clamp(pet_adj - eu, min=0.0)
    if evaporation_scheme == 0:
        # Historical sequential lower/deep-layer evaporation.
        ed = torch.where(
            (wl < c * lm) & (wl < c * remaining_pet),
            c * remaining_pet - wl,
            torch.zeros_like(wl),
        )
        # Deep-layer evaporation cannot withdraw more water than is stored in
        # the deep layer.  Without this cap a dry catchment can report ED > WD
        # and the later state clamp silently destroys water.
        ed = torch.minimum(ed, wd)
        el = torch.where(
            wu + prcp >= pet_adj,
            torch.zeros_like(wl),
            torch.where(
                wl >= c * lm,
                remaining_pet * wl / (lm + nearzero),
                torch.where(wl >= c * remaining_pet, c * remaining_pet, wl),
            ),
        )
    elif evaporation_scheme == 1:
        # Aggregated root-zone moisture-stress evaporation.  WL and WD remain
        # separate XAJ stores; their sum is used only to diagnose root-zone
        # stress, then the total withdrawal is allocated proportionally for
        # conservative state bookkeeping.
        if root_tau_e is None:
            raise ValueError("root_tau_e is required for aggregated root-zone evaporation")
        el, ed, _er, _z_root, _stress = _rootzone_moisture_stress_evaporation(
            remaining_pet, wl, wd, lm, dm, root_tau_e, nearzero,
        )
    else:
        # Structure-diagnosis D_E/G_E: both layers use the pre-extraction
        # states in one parallel kernel.  Scheme 2 is D_E (gamma=1), scheme 3
        # is G_E (the caller supplies the only generic exponent).
        if evaporation_scheme == 2:
            evaporation_gamma = torch.ones_like(wl)
        else:
            if structure_gamma is None:
                raise ValueError("structure_gamma is required for G_E")
            evaporation_gamma = structure_gamma
        el, ed, _er, _wl_new, _wd_new, _el_raw, _ed_raw, _xl, _xd, _rl, _rd = _parallel_evaporation_step(
            remaining_pet, wl, wd, lm, dm, evaporation_gamma, nearzero,
        )

    evap_total = eu + el + ed

    # --- Runoff generation: tension water capacity curve ---
    w0 = wu + wl + wd
    w0 = torch.clamp(w0, max=wm - nearzero)

    prcp_diff = prcp - evap_total
    pe = torch.clamp(prcp_diff, min=0.0)

    # Keep the fractional-power base away from zero in float32.  At a full
    # tension-water store, roundoff can make this base exactly zero; because
    # the exponent is below one, PowBackward is then singular and returns NaN.
    base = torch.clamp(1.0 - w0 / (wm + nearzero), min=1e-6)
    a = wmm * (1.0 - base ** (1.0 / (1.0 + b)))

    r_cal = torch.where(
        pe > 0.0,
        torch.where(
            pe + a < wmm,
            pe - (wm - w0) + wm * (1.0 - torch.clamp((a + pe) / (wmm + nearzero), max=1.0)) ** (1.0 + b),
            pe - (wm - w0),
        ),
        torch.zeros_like(pe),
    )
    r = torch.clamp(r_cal, min=0.0)
    r_im = torch.clamp(pe * im, min=0.0)

    # --- Update tension water storages ---
    wu_new = torch.where(
        pe > 0.0,
        torch.where(wu + pe - r < um, wu + pe - r, um),
        torch.where(wu + prcp_diff > 0.0, wu + prcp_diff, torch.zeros_like(wu)),
    )
    wd_new = torch.where(
        pe > 0.0,
        torch.where(wu + wl + pe - r > um + lm, wu + wl + wd + pe - r - um - lm, wd),
        wd - ed,
    )
    wl_new = torch.where(
        pe > 0.0,
        wu + wl + wd + pe - r - wu_new - wd_new,
        wl - el,
    )
    wu = torch.clamp(wu_new, torch.zeros_like(wu_new), um)
    wl = torch.clamp(wl_new, torch.zeros_like(wl_new), lm)
    wd = torch.clamp(wd_new, torch.zeros_like(wd_new), dm)

    # --- Runoff separation: free water reservoir ---
    fr_mask = r > 0.0
    fr_old = fr  # save old fr for ss computation below
    fr_new = torch.where(fr_mask, r / (pe + nearzero), fr_old)
    fr = torch.clamp(fr_new, 0.0, 1.0)

    ss = torch.where(fr_mask, fr_old * s / (fr_new + nearzero), s)
    ss = torch.clamp(ss, max=sm - nearzero)

    free_base = torch.clamp(1.0 - ss / (sm + nearzero), min=1e-6)
    au = ms * (1.0 - free_base ** (1.0 / (1.0 + ex)))

    rs = torch.zeros_like(r)
    rs_fr = torch.where(
        pe + au < ms,
        fr * (pe - sm + ss + sm * (1.0 - torch.clamp((pe + au) / (ms + nearzero), max=1.0)) ** (1.0 + ex)),
        fr * (pe + ss - sm),
    )
    rs = torch.where(fr_mask, torch.min(rs_fr, r), rs)
    rs = torch.clamp(rs, min=0.0)

    s_new = ss + torch.where(fr_mask, (r - rs) / (fr + nearzero), torch.zeros_like(s))
    # A negative free-water store has no physical destination and can make
    # subsequent RI/RG generation negative at extreme parameter corners.
    s = torch.minimum(torch.maximum(s_new, torch.zeros_like(s_new)), sm - nearzero)

    if response_scheme == 0:
        ri = ki * s * fr
        rg = kg * s * fr
        s_next = s * one_minus_ki_kg

        # --- Linear reservoirs for interflow and groundwater ---
        qi = ci * qi + (1.0 - ci) * (ri * one_minus_im)
        qg = cg * qg + (1.0 - cg) * (rg * one_minus_im)
        response_storage = torch.zeros_like(qi)
        response_input = (ri + rg) * one_minus_im
    else:
        # D_R/G_R receive only the effective total generated subsurface input.
        # ``ki`` is KSS and ``kg`` is a zero placeholder in the controlled
        # variant call; no RI/RG identities are created here.
        response_input = ki * s * fr * one_minus_im
        s_next = s * (1.0 - ki)
        if structure_tau_0 is None or structure_z0 is None:
            raise ValueError("structure_tau_0 and structure_z0 are required for D_R/G_R")
        if response_scheme == 1:
            response_beta = torch.ones_like(s)
        else:
            if structure_beta is None:
                raise ValueError("structure_beta is required for G_R")
            response_beta = structure_beta
        response_q, response_storage, _response_available, _response_q_raw, _response_ratio = _analytic_subsurface_response_step(
            response_input, qi, structure_tau_0, response_beta,
            structure_z0, nearzero,
        )
        qi = response_q
        qg = torch.zeros_like(response_q)
        ri = response_input
        rg = torch.zeros_like(response_input)

    # --- Total instantaneous runoff (surface component for UH routing) ---
    rs_adj = rs * one_minus_im + r_im
    if response_scheme != 0:
        if return_diagnostics:
            return (rs_adj + qi, rs_adj, qi, evap_total,
                    wu, wl, wd, s_next, fr,
                    rs, response_input, rg, eu, el, ed,
                    response_storage, response_input)
        return rs_adj, qi, response_storage, wu, wl, wd, s_next, fr
    if not return_diagnostics:
        # The compact training path only needs the three runoff components and
        # the recursive states.  In particular, avoid materialising q_out and
        # the diagnostic-only intermediate values in the compiled output tuple.
        return rs_adj, qi, qg, wu, wl, wd, s_next, fr

    q_out = rs_adj + qi + qg
    return (q_out, rs_adj, qi, qg, evap_total,
            wu, wl, wd, s_next, fr,
            rs, ri, rg, eu, el, ed)


def _xaj_step(
    precip_t: torch.Tensor,
    pet_t: torch.Tensor,
    wu: torch.Tensor,
    wl: torch.Tensor,
    wd: torch.Tensor,
    s: torch.Tensor,
    fr: torch.Tensor,
    qi: torch.Tensor,
    qg: torch.Tensor,
    k: torch.Tensor,
    b: torch.Tensor,
    im: torch.Tensor,
    um: torch.Tensor,
    lm: torch.Tensor,
    dm: torch.Tensor,
    c: torch.Tensor,
    sm: torch.Tensor,
    ex: torch.Tensor,
    ki: torch.Tensor,
    kg: torch.Tensor,
    ci: torch.Tensor,
    cg: torch.Tensor,
    nearzero: float,
) -> tuple:
    """Historical XAJ step including all diagnostic outputs."""
    wm = um + lm + dm
    return _xaj_step_impl(
        precip_t, pet_t, wu, wl, wd, s, fr, qi, qg,
        k, b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg,
        nearzero, wm, wm * (1.0 + b), sm * (1.0 + ex),
        1.0 - im, 1.0 - ki - kg, True,
    )


def _xaj_step_compact(
    precip_t: torch.Tensor,
    pet_t: torch.Tensor,
    wu: torch.Tensor,
    wl: torch.Tensor,
    wd: torch.Tensor,
    s: torch.Tensor,
    fr: torch.Tensor,
    qi: torch.Tensor,
    qg: torch.Tensor,
    k: torch.Tensor,
    b: torch.Tensor,
    im: torch.Tensor,
    um: torch.Tensor,
    lm: torch.Tensor,
    dm: torch.Tensor,
    c: torch.Tensor,
    sm: torch.Tensor,
    ex: torch.Tensor,
    ki: torch.Tensor,
    kg: torch.Tensor,
    ci: torch.Tensor,
    cg: torch.Tensor,
    nearzero: float,
    wm: torch.Tensor,
    wmm: torch.Tensor,
    ms: torch.Tensor,
    one_minus_im: torch.Tensor,
    one_minus_ki_kg: torch.Tensor,
) -> tuple:
    """Lean XAJ step used when only streamflow is requested."""
    return _xaj_step_impl(
        precip_t, pet_t, wu, wl, wd, s, fr, qi, qg,
        k, b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg,
        nearzero, wm, wmm, ms, one_minus_im, one_minus_ki_kg, False,
    )


def _xaj_rwpe_step(
    precip_t: torch.Tensor, pet_t: torch.Tensor,
    wu: torch.Tensor, wl: torch.Tensor, wd: torch.Tensor, s: torch.Tensor,
    fr: torch.Tensor, qi: torch.Tensor, qg: torch.Tensor,
    k: torch.Tensor, b: torch.Tensor, im: torch.Tensor, um: torch.Tensor,
    lm: torch.Tensor, dm: torch.Tensor, tau_e: torch.Tensor, sm: torch.Tensor,
    ex: torch.Tensor, ki: torch.Tensor, kg: torch.Tensor, ci: torch.Tensor,
    cg: torch.Tensor, nearzero: float,
) -> tuple:
    """XAJ step with aggregated root-zone moisture-stress evaporation."""
    wm = um + lm + dm
    return _xaj_step_impl(
        precip_t, pet_t, wu, wl, wd, s, fr, qi, qg,
        k, b, im, um, lm, dm, tau_e, sm, ex, ki, kg, ci, cg,
        nearzero, wm, wm * (1.0 + b), sm * (1.0 + ex),
        1.0 - im, 1.0 - ki - kg, True, 1, tau_e,
    )


def _xaj_rwpe_step_compact(
    precip_t: torch.Tensor, pet_t: torch.Tensor,
    wu: torch.Tensor, wl: torch.Tensor, wd: torch.Tensor, s: torch.Tensor,
    fr: torch.Tensor, qi: torch.Tensor, qg: torch.Tensor,
    k: torch.Tensor, b: torch.Tensor, im: torch.Tensor, um: torch.Tensor,
    lm: torch.Tensor, dm: torch.Tensor, tau_e: torch.Tensor, sm: torch.Tensor,
    ex: torch.Tensor, ki: torch.Tensor, kg: torch.Tensor, ci: torch.Tensor,
    cg: torch.Tensor, nearzero: float, wm: torch.Tensor, wmm: torch.Tensor,
    ms: torch.Tensor, one_minus_im: torch.Tensor,
    one_minus_ki_kg: torch.Tensor,
) -> tuple:
    """Compact streamflow-only RWPE step."""
    return _xaj_step_impl(
        precip_t, pet_t, wu, wl, wd, s, fr, qi, qg,
        k, b, im, um, lm, dm, tau_e, sm, ex, ki, kg, ci, cg,
        nearzero, wm, wmm, ms, one_minus_im, one_minus_ki_kg, False, 1, tau_e,
    )


def _prepare_xaj_parameters(params: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Extract and safely normalize XAJ parameters for a forward pass.

    Keeping this small preparation step outside ``forward`` lets composed
    models run their upstream snow/delay step and XAJ's runoff step in one
    time loop without duplicating the boundary handling in two places.
    """
    k = params["xaj_k"]
    b = params["xaj_b"]
    im = params["xaj_im"]
    um = params["xaj_um"]
    lm = params["xaj_lm"]
    dm = params["xaj_dm"]
    c = params["xaj_c"]
    sm = params["xaj_sm"]
    ex = params["xaj_ex"]
    ki = params["xaj_ki"]
    kg = params["xaj_kg"]
    ci = params["xaj_ci"]
    cg = params["xaj_cg"]
    a_uh = params["xaj_a"]
    theta_uh = params["xaj_theta"]

    # Preserve the original boundary behavior while avoiding an inactive
    # zero-denominator branch when ki=kg=0.
    ki_kg_sum = ki + kg
    safe_ki_kg_sum = torch.clamp(ki_kg_sum, min=1e-6)
    scale = torch.where(
        ki_kg_sum < 1.0,
        torch.ones_like(ki),
        (1.0 - 1e-5) / safe_ki_kg_sum,
    )
    ki = ki * scale
    kg = kg * scale
    return (
        k, b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg, a_uh, theta_uh,
    )


def _route_xaj_surface_runoff(
    rs_store: torch.Tensor,
    rs_uh_buffer: torch.Tensor,
    a_uh: torch.Tensor,
    theta_uh: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply XAJ's gamma unit hydrograph, including continuation history."""
    uh_ords = _gamma_uh_ordinates(a_uh, theta_uh, XAJ_UH_MAX_LEN, device, dtype)
    rs_with_history = torch.cat((rs_uh_buffer, rs_store), dim=1)
    rs_routed_all = _apply_uh_routing(rs_with_history, uh_ords)
    # ``uh_conv`` keeps the full history-plus-input sequence.  Return only
    # the samples corresponding to the current chunk; the leading history is
    # consumed by the causal convolution and must not leak into the output.
    start = XAJ_UH_MAX_LEN - 1
    rs_routed = rs_routed_all[:, start:start + rs_store.shape[1]]
    next_buffer = rs_with_history[:, -(XAJ_UH_MAX_LEN - 1):]
    return rs_routed, next_buffer


def _route_xaj_surface_runoff_hydrodl2(
    rs_store: torch.Tensor,
    rs_uh_buffer: torch.Tensor,
    a_uh: torch.Tensor,
    theta_uh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route XAJ surface runoff with hydrodl2's verified HBV implementation.

    ``hydrodl2`` expects ``[batch, variable, time]`` fluxes and UH weights;
    its convolution is causal and returns the same sequence length as the
    input.  The preceding buffer makes chunked execution equivalent to a
    single long sequence, just as in the historical XAJ route.
    """
    kernel_len = XAJ_UH_MAX_LEN
    batch = rs_store.shape[0]
    # uh_gamma expects [time, batch, variable] parameters repeated along time.
    a_rep = a_uh.reshape(1, batch, 1).expand(kernel_len, -1, -1)
    theta_rep = theta_uh.reshape(1, batch, 1).expand(kernel_len, -1, -1)
    uh = uh_gamma(a_rep, theta_rep, lenF=kernel_len).permute(1, 2, 0)

    rs_with_history = torch.cat((rs_uh_buffer, rs_store), dim=1)
    routed_all = uh_conv(rs_with_history.unsqueeze(1), uh).squeeze(1)
    rs_routed = routed_all[:, kernel_len - 1:kernel_len - 1 + rs_store.shape[1]]
    next_buffer = rs_with_history[:, -(kernel_len - 1):]
    return rs_routed, next_buffer


class XAJ(BaseHydrologicalModel):
    """XinAnJiang (XAJ) model.

    Core XAJ runoff-generation + gamma UH surface routing + linear reservoir
    routing for interflow/groundwater.
    """

    # ``rs_uh_buffer`` carries the preceding ``XAJ_UH_MAX_LEN - 1``
    # surface-runoff ordinates so
    # a continuation run has the same gamma-UH output as a single long run.
    state_names = ["wu", "wl", "wd", "s", "fr", "qi", "qg", "rs_uh_buffer"]
    routing_method = "gamma"
    uh_max_len = XAJ_UH_MAX_LEN
    use_hydrodl2_uh = False

    def __init__(self, nearzero: float = 1e-8, compact_output: bool = False):
        super().__init__()
        self.nearzero = nearzero
        # Keep the historical diagnostic output path as the default.  XAJLite
        # opts into the compact state/output path for training workloads.
        self.compact_output = compact_output
        self._step = torch.compile(_xaj_step, fullgraph=True)
        self._compact_step = torch.compile(_xaj_step_compact, fullgraph=True)

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return XAJ_PARAM_SPECS

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

        (
            k, b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg,
            a_uh, theta_uh,
        ) = _prepare_xaj_parameters(params)

        (wu, wl, wd, s, fr, qi, qg, rs_uh_buffer) = self._init_states(
            batch, device, dtype, initial_states, um, lm, dm, sm,
        )

        loop_args = (
            precip, pet, nsteps, batch, device, dtype,
            wu, wl, wd, s, fr, qi, qg, rs_uh_buffer,
            k, b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg,
            a_uh, theta_uh,
        )
        if self.compact_output and not return_states:
            qsim, aux, states = self._step_loop_lite(*loop_args)
        else:
            qsim, aux, states = self._step_loop(*loop_args, compact=False)

        if return_states:
            aux["final_states"] = {k: v for k, v in zip(self.state_names, states)}

        return qsim, aux

    def _step_loop_lite(
        self,
        precip: torch.Tensor,
        pet: torch.Tensor,
        nsteps: int,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        wu: torch.Tensor,
        wl: torch.Tensor,
        wd: torch.Tensor,
        s: torch.Tensor,
        fr: torch.Tensor,
        qi: torch.Tensor,
        qg: torch.Tensor,
        rs_uh_buffer: torch.Tensor,
        k: torch.Tensor,
        b: torch.Tensor,
        im: torch.Tensor,
        um: torch.Tensor,
        lm: torch.Tensor,
        dm: torch.Tensor,
        c: torch.Tensor,
        sm: torch.Tensor,
        ex: torch.Tensor,
        ki: torch.Tensor,
        kg: torch.Tensor,
        ci: torch.Tensor,
        cg: torch.Tensor,
        a_uh: torch.Tensor,
        theta_uh: torch.Tensor,
    ) -> tuple:
        """Branch-free streamflow-only XAJ loop."""
        wm = um + lm + dm
        wmm = wm * (1.0 + b)
        ms = sm * (1.0 + ex)
        one_minus_im = 1.0 - im
        one_minus_ki_kg = 1.0 - ki - kg
        rs_values = []
        baseflow_values = []
        for t in range(nsteps):
            # ``[:, t]`` is a view with a time-varying storage offset. Feed
            # contiguous daily vectors to the compiled kernel so a long
            # sequence does not create one Dynamo specialization per day.
            precip_t = precip[:, t].contiguous()
            pet_t = pet[:, t].contiguous()
            rs_adj_t, qi_t, qg_t, wu, wl, wd, s, fr = self._compact_step(
                precip_t, pet_t, wu, wl, wd, s, fr, qi, qg,
                k, b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg,
                self.nearzero, wm, wmm, ms, one_minus_im, one_minus_ki_kg,
            )
            rs_values.append(rs_adj_t)
            baseflow_values.append(qi_t + qg_t)
            qi, qg = qi_t, qg_t
        rs_store = torch.stack(rs_values, dim=1)
        baseflow = torch.stack(baseflow_values, dim=1)
        rs_routed, rs_uh_buffer = _route_xaj_surface_runoff_hydrodl2(
            rs_store, rs_uh_buffer, a_uh, theta_uh,
        )
        qsim = rs_routed + baseflow
        return qsim, {}, (wu, wl, wd, s, fr, qi, qg, rs_uh_buffer)

    def _init_states(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        initial_states: Optional[dict[str, torch.Tensor]] = None,
        um: Optional[torch.Tensor] = None,
        lm: Optional[torch.Tensor] = None,
        dm: Optional[torch.Tensor] = None,
        sm: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        if initial_states is not None:
            return (
                initial_states.get("wu", torch.full((batch,), 0.6, device=device, dtype=dtype) * (um if um is not None else 20.0)),
                initial_states.get("wl", torch.full((batch,), 0.6, device=device, dtype=dtype) * (lm if lm is not None else 80.0)),
                initial_states.get("wd", torch.full((batch,), 0.6, device=device, dtype=dtype) * (dm if dm is not None else 40.0)),
                initial_states.get("s", torch.full((batch,), 0.5, device=device, dtype=dtype) * (sm if sm is not None else 15.0)),
                initial_states.get("fr", torch.full((batch,), 0.1, device=device, dtype=dtype)),
                initial_states.get("qi", torch.full((batch,), 0.1, device=device, dtype=dtype)),
                initial_states.get("qg", torch.full((batch,), 0.1, device=device, dtype=dtype)),
                initial_states.get(
                    "rs_uh_buffer",
                    torch.zeros(batch, self.uh_max_len - 1, device=device, dtype=dtype),
                ),
            )
        return (
            torch.full((batch,), 0.6, device=device, dtype=dtype) * um if um is not None else torch.full((batch,), 20.0 * 0.6, device=device, dtype=dtype),
            torch.full((batch,), 0.6, device=device, dtype=dtype) * lm if lm is not None else torch.full((batch,), 80.0 * 0.6, device=device, dtype=dtype),
            torch.full((batch,), 0.6, device=device, dtype=dtype) * dm if dm is not None else torch.full((batch,), 40.0 * 0.6, device=device, dtype=dtype),
            torch.full((batch,), 0.5, device=device, dtype=dtype) * sm if sm is not None else torch.full((batch,), 15.0, device=device, dtype=dtype),
            torch.full((batch,), 0.1, device=device, dtype=dtype),
            torch.full((batch,), 0.1, device=device, dtype=dtype),
            torch.full((batch,), 0.1, device=device, dtype=dtype),
            torch.zeros(batch, self.uh_max_len - 1, device=device, dtype=dtype),
        )

    def _step_loop(
        self,
        precip: torch.Tensor,
        pet: torch.Tensor,
        nsteps: int,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        wu: torch.Tensor,
        wl: torch.Tensor,
        wd: torch.Tensor,
        s: torch.Tensor,
        fr: torch.Tensor,
        qi: torch.Tensor,
        qg: torch.Tensor,
        rs_uh_buffer: torch.Tensor,
        k: torch.Tensor,
        b: torch.Tensor,
        im: torch.Tensor,
        um: torch.Tensor,
        lm: torch.Tensor,
        dm: torch.Tensor,
        c: torch.Tensor,
        sm: torch.Tensor,
        ex: torch.Tensor,
        ki: torch.Tensor,
        kg: torch.Tensor,
        ci: torch.Tensor,
        cg: torch.Tensor,
        a_uh: torch.Tensor,
        theta_uh: torch.Tensor,
        compact: bool = False,
    ) -> tuple:
        if compact:
            rs_values = []
            baseflow_values = []
        else:
            qsim = torch.zeros(batch, nsteps, device=device, dtype=dtype)
            rs_store = torch.zeros(batch, nsteps, device=device, dtype=dtype)
            qi_store = torch.zeros(batch, nsteps, device=device, dtype=dtype)
            qg_store = torch.zeros(batch, nsteps, device=device, dtype=dtype)
            evap_store = torch.zeros(batch, nsteps, device=device, dtype=dtype)
        nz = self.nearzero
        if compact:
            wm = um + lm + dm
            wmm = wm * (1.0 + b)
            ms = sm * (1.0 + ex)
            one_minus_im = 1.0 - im
            one_minus_ki_kg = 1.0 - ki - kg

        for t in range(nsteps):
            if compact:
                (rs_adj_t, qi_t, qg_t,
                 wu, wl, wd, s, fr) = self._compact_step(
                    precip[:, t], pet[:, t],
                    wu, wl, wd, s, fr, qi, qg,
                    k, b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg,
                    nz, wm, wmm, ms, one_minus_im, one_minus_ki_kg,
                )
            else:
                (_, rs_adj_t, qi_t, qg_t, evap_t,
                 wu, wl, wd, s, fr,
                 rs, ri, rg, eu, el, ed) = self._step(
                    precip[:, t],
                    pet[:, t],
                    wu, wl, wd, s, fr, qi, qg,
                    k, b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg,
                    nz,
                )
            if compact:
                rs_values.append(rs_adj_t)
                baseflow_values.append(qi_t + qg_t)
            else:
                rs_store[:, t] = rs_adj_t
                qi_store[:, t] = qi_t
                qg_store[:, t] = qg_t
                evap_store[:, t] = evap_t
            # Carry the updated linear-reservoir states into the next day.
            # Without these assignments every step reused the initial qi/qg
            # values, creating an artificial fixed baseflow even when ki=kg=0.
            qi = qi_t
            qg = qg_t

        # Route the current surface runoff together with the history retained
        # from a preceding chunk.  With an all-zero buffer this is identical
        # to the original full-series routing path.
        if compact:
            rs_store = torch.stack(rs_values, dim=1)
            baseflow_store = torch.stack(baseflow_values, dim=1)
        if self.use_hydrodl2_uh:
            rs_routed, rs_uh_buffer = _route_xaj_surface_runoff_hydrodl2(
                rs_store, rs_uh_buffer, a_uh, theta_uh,
            )
        else:
            rs_routed, rs_uh_buffer = _route_xaj_surface_runoff(
                rs_store, rs_uh_buffer, a_uh, theta_uh, device, dtype,
            )

        if compact:
            qsim = rs_routed + baseflow_store
        else:
            qsim = rs_routed + qi_store + qg_store

        if compact:
            return qsim, {}, (
                wu, wl, wd, s, fr,
                qi, qg, rs_uh_buffer,
            )

        aux = {
            "evap": evap_store,
            "rs_instant": rs_store,
            "rs_routed": rs_routed,
            "qi": qi_store,
            "qg": qg_store,
            "wu": wu, "wl": wl, "wd": wd,
            "s": s, "fr": fr,
        }

        return qsim, aux, (wu, wl, wd, s, fr, qi_store[:, -1], qg_store[:, -1], rs_uh_buffer)


class XAJLite(XAJ):
    """XAJ training variant that returns only simulated streamflow.

    ``XAJ`` keeps its historical diagnostic outputs by default.  This class
    uses the compact step/output path unless ``return_states=True`` is
    requested, in which case the full historical diagnostics are retained for
    audits and chunked-state tests.
    """

    def __init__(self, nearzero: float = 1e-8):
        super().__init__(nearzero=nearzero, compact_output=True)
        self.uh_max_len = XAJ_LITE_UH_MAX_LEN
        self.use_hydrodl2_uh = True

    @property
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        return XAJ_LITE_PARAM_SPECS


def _gamma_uh_ordinates(
    a: torch.Tensor,      # [batch] shape parameter (n)
    theta: torch.Tensor,   # [batch] scale parameter (k)
    max_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute Gamma-UH ordinates using hydrodl2's HBV implementation.

    w(t) ∝ t^(a-1) * exp(-t/theta)  for t = 1, 2, ..., max_len
    Ordinates normalized to sum to 1.

    Args:
        a: shape parameter [batch], clamped to >= 0.5
        theta: scale parameter [batch], clamped to >= 0.5
        max_len: number of UH ordinates
    Returns:
        uh: [batch, max_len] normalized UH ordinates
    """
    batch = a.shape[0]
    a_rep = a.reshape(1, batch, 1).expand(max_len, -1, -1)
    theta_rep = theta.reshape(1, batch, 1).expand(max_len, -1, -1)
    return uh_gamma(a_rep, theta_rep, lenF=max_len).permute(1, 2, 0).squeeze(1)


def _apply_uh_routing(
    flux: torch.Tensor,      # [batch, time]
    uh_ords: torch.Tensor,   # [batch, kernel_len]
) -> torch.Tensor:
    """Apply hydrodl2/HBV grouped Gamma-UH convolution."""
    return uh_conv(flux.unsqueeze(1), uh_ords.unsqueeze(1)).squeeze(1)
