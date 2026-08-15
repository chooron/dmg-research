"""Recorded forward passes for the three R3 fitted structures.

The production models export daily discharge and some fluxes, but only the
*final* soil-moisture/snow states.  R3 needs per-day comparable states for
truth generation (``X*``) and for truth-relative state errors of fitted
runs.  These harnesses replay the exact production time loops (same compiled
step kernels, same state update order, same routing calls) while additionally
recording per-day states.  They are *not* second model implementations.

Numerical identity with the production forward is enforced by
``validate_recorded_forward`` (discharge and final states must match the
production model call), and every R3 artifact is produced through these
validated paths.

Lite-path semantics (matching the IC/dPL fit pipelines):

- CN (``XAJWithCemaNeigeLite``): fused ``_cemaneige_xaj_fused_step`` per day
  + conv gamma-UH routing ``_route_xaj_surface_runoff``;
- Base (``XAJLite``): ``_xaj_step_compact`` per day + hydrodl2 gamma-UH
  routing ``_route_xaj_surface_runoff_hydrodl2``;
- TGD2 (``XAJWithTGD2Lite``): ``tgd2_step`` over the full sequence + the
  XAJLite path on effective precipitation.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .common import COMMON_XAJ

COMMON_STATES = ("wu", "wl", "wd", "s", "fr", "qi", "qg")


def _prepare_loop_quantities(xaj_params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    from models.xaj import _prepare_xaj_parameters

    names = ("k", "b", "im", "um", "lm", "dm", "c", "sm", "ex", "ki", "kg",
             "ci", "cg", "a_uh", "theta_uh")
    values = _prepare_xaj_parameters(xaj_params)
    return dict(zip(names, values))


def recorded_cn_forward(model, forcings: dict[str, torch.Tensor],
                        params: dict[str, torch.Tensor],
                        device: torch.device, dtype: torch.dtype):
    """Mirror ``XAJWithCemaNeigeLite.forward`` and record per-day states.

    Returns (qsim, states) where ``states`` holds per-day tensors
    [batch, time] for wu/wl/wd/s/fr/qi/qg/rs_instant/evap plus CN snow
    diagnostics G/eTG/sca/rain/melt and effective_precipitation.
    """
    from models.cemaneige import _estimate_psol_annual, _init_basic_states
    from models.xaj import _route_xaj_surface_runoff

    precip = forcings["precip"]
    temp = forcings["temp"]
    pet = forcings["pet"]
    batch, nsteps = precip.shape

    ctg = params["cn_ctg"]
    kf = params["cn_kf"]
    g_thresh = 0.9 * _estimate_psol_annual(precip, temp)
    G, eTG = _init_basic_states(batch, device, dtype, None)
    xaj_params = {k: v for k, v in params.items() if k.startswith("xaj_")}
    q = _prepare_loop_quantities(xaj_params)
    xaj_step_params = tuple(q[n] for n in (
        "k", "b", "im", "um", "lm", "dm", "c", "sm", "ex", "ki", "kg", "ci", "cg"))
    a_uh, theta_uh = q["a_uh"], q["theta_uh"]
    (wu, wl, wd, s, fr, qi, qg, rs_uh_buffer) = model._xaj._init_states(
        batch, device, dtype, None, q["um"], q["lm"], q["dm"], q["sm"]
    )

    stores = {name: torch.zeros(batch, nsteps, device=device, dtype=dtype)
              for name in ("wu", "wl", "wd", "s", "fr", "qi", "qg",
                           "rs_instant", "evap", "G", "eTG", "sca", "rain",
                           "melt", "effective_precip")}
    for t in range(nsteps):
        (effective_t, G, eTG, sca_t, rain_t, melt_t,
         _q_out, rs_adj_t, qi_t, qg_t, evap_t,
         wu, wl, wd, s, fr, _rs, _ri, _rg, _eu, _el, _ed) = model._fused_step(
            precip[:, t], temp[:, t], pet[:, t],
            (G, eTG),
            (wu, wl, wd, s, fr, qi, qg),
            (ctg, kf, g_thresh),
            xaj_step_params,
            model.nearzero,
        )
        stores["wu"][:, t], stores["wl"][:, t] = wu, wl
        stores["wd"][:, t], stores["s"][:, t] = wd, s
        stores["fr"][:, t], stores["qi"][:, t] = fr, qi_t
        stores["qg"][:, t], stores["rs_instant"][:, t] = qg_t, rs_adj_t
        stores["evap"][:, t] = evap_t
        stores["G"][:, t], stores["eTG"][:, t] = G, eTG
        stores["sca"][:, t], stores["rain"][:, t] = sca_t, rain_t
        stores["melt"][:, t], stores["effective_precip"][:, t] = melt_t, effective_t
        qi, qg = qi_t, qg_t

    rs_routed, rs_uh_buffer = _route_xaj_surface_runoff(
        stores["rs_instant"], rs_uh_buffer, a_uh, theta_uh, device, dtype
    )
    qsim = rs_routed + stores["qi"] + stores["qg"]
    final_states = {
        "wu": wu, "wl": wl, "wd": wd, "s": s, "fr": fr,
        "qi": stores["qi"][:, -1], "qg": stores["qg"][:, -1],
        "G": G, "eTG": eTG,
    }
    return qsim, stores, final_states


def recorded_base_forward(model, forcings: dict[str, torch.Tensor],
                          params: dict[str, torch.Tensor],
                          device: torch.device, dtype: torch.dtype):
    """Mirror ``XAJLite._step_loop_lite`` and record per-day states."""
    from models.xaj import _route_xaj_surface_runoff_hydrodl2

    precip = forcings["precip"]
    pet = forcings["pet"]
    batch, nsteps = precip.shape

    q = _prepare_loop_quantities(params)
    wm = q["um"] + q["lm"] + q["dm"]
    wmm = wm * (1.0 + q["b"])
    ms = q["sm"] * (1.0 + q["ex"])
    one_minus_im = 1.0 - q["im"]
    one_minus_ki_kg = 1.0 - q["ki"] - q["kg"]
    (wu, wl, wd, s, fr, qi, qg, rs_uh_buffer) = model._init_states(
        batch, device, dtype, None, q["um"], q["lm"], q["dm"], q["sm"]
    )
    step_args = (q["k"], q["b"], q["im"], q["um"], q["lm"], q["dm"], q["c"],
                 q["sm"], q["ex"], q["ki"], q["kg"], q["ci"], q["cg"])

    stores = {name: torch.zeros(batch, nsteps, device=device, dtype=dtype)
              for name in ("wu", "wl", "wd", "s", "fr", "qi", "qg", "rs_instant")}
    for t in range(nsteps):
        rs_adj_t, qi_t, qg_t, wu, wl, wd, s, fr = model._compact_step(
            precip[:, t].contiguous(), pet[:, t].contiguous(),
            wu, wl, wd, s, fr, qi, qg,
            *step_args, model.nearzero, wm, wmm, ms, one_minus_im, one_minus_ki_kg,
        )
        stores["wu"][:, t], stores["wl"][:, t] = wu, wl
        stores["wd"][:, t], stores["s"][:, t] = wd, s
        stores["fr"][:, t], stores["qi"][:, t] = fr, qi_t
        stores["qg"][:, t], stores["rs_instant"][:, t] = qg_t, rs_adj_t
        qi, qg = qi_t, qg_t

    rs_routed, rs_uh_buffer = _route_xaj_surface_runoff_hydrodl2(
        stores["rs_instant"], rs_uh_buffer, q["a_uh"], q["theta_uh"]
    )
    qsim = rs_routed + stores["qi"] + stores["qg"]
    final_states = {
        "wu": wu, "wl": wl, "wd": wd, "s": s, "fr": fr,
        "qi": stores["qi"][:, -1], "qg": stores["qg"][:, -1],
    }
    return qsim, stores, final_states


def recorded_tgd2_forward(model, forcings: dict[str, torch.Tensor],
                          params: dict[str, torch.Tensor],
                          device: torch.device, dtype: torch.dtype):
    """Mirror ``XAJWithTGD2Lite.forward`` and record per-day states.

    The TGD2 delay runs over the full sequence first (as in the production
    composition); its storage trace is recorded for diagnostics.  The XAJ
    part uses the same compact step and hydrodl2 routing as Base.
    """
    from models.tgd2 import tgd2_step

    precip = forcings["precip"]
    temp = forcings["temp"]
    pet = forcings["pet"]
    batch, nsteps = precip.shape

    tau_warm = params["tgd_tau_warm"]
    delta_tau_cold = params["tgd_delta_tau_cold"]
    storage = torch.zeros(batch, device=device, dtype=dtype)
    effective = torch.empty_like(precip)
    storage_trace = torch.empty_like(precip)
    tau_trace = torch.empty_like(precip)
    retention_trace = torch.empty_like(precip)
    for t in range(nsteps):
        effective_t, storage, tau_t, retention_t = tgd2_step(
            precip[:, t], temp[:, t], storage, tau_warm, delta_tau_cold
        )
        effective[:, t] = effective_t
        storage_trace[:, t], tau_trace[:, t], retention_trace[:, t] = (
            storage, tau_t, retention_t
        )

    xaj_params = {k: v for k, v in params.items() if k.startswith("xaj_")}
    qsim, stores, final_states = recorded_base_forward(
        model._runoff, {"precip": effective, "pet": pet, "temp": temp},
        xaj_params, device, dtype,
    )
    stores["tgd_storage"] = storage_trace
    stores["tgd_tau"] = tau_trace
    stores["tgd_retention"] = retention_trace
    final_states["tgd_storage"] = storage
    return qsim, stores, final_states


def validate_recorded_forward(model, recorded: tuple, forcings: dict[str, torch.Tensor],
                              params: dict[str, torch.Tensor], *,
                              atol: float = 1e-5, rtol: float = 1e-5) -> dict[str, float]:
    """Compare a recorded forward against the production model forward.

    Returns a dict with max absolute discharge diff, max relative discharge
    diff, and per-final-state max abs diff.
    """
    with torch.no_grad():
        q_prod, aux = model(forcings=forcings, params=params, return_states=True)
    qsim, _stores, final_states = recorded
    q_prod = q_prod.detach()
    qsim = qsim.detach()
    abs_diff = (q_prod - qsim).abs().max().item()
    denom = q_prod.abs().clamp_min(1e-6)
    rel_diff = ((q_prod - qsim).abs() / denom).max().item()
    state_diffs: dict[str, float] = {}
    prod_final = aux.get("final_states", {})
    for key, value in final_states.items():
        if key in prod_final:
            state_diffs[key] = float((value.detach() - prod_final[key]).abs().max().item())
    max_state = max(state_diffs.values(), default=0.0)
    if abs_diff > atol or max_state > atol:
        raise RuntimeError(
            f"recorded forward diverged from production forward: "
            f"q_abs={abs_diff:.3e} q_rel={rel_diff:.3e} state_max={max_state:.3e}"
        )
    return {"q_abs_max": abs_diff, "q_rel_max": rel_diff, "state_abs_max": max_state}


def recorded_forward_for_structure(
    structure: str, model, forcings: dict[str, torch.Tensor],
    params: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype,
):
    """Dispatch to the recorded forward matching the fitted structure key."""
    if structure == "XAJ_CN":
        return recorded_cn_forward(model, forcings, params, device, dtype)
    if structure == "XAJ":
        return recorded_base_forward(model, forcings, params, device, dtype)
    if structure == "XAJ_TGD2":
        return recorded_tgd2_forward(model, forcings, params, device, dtype)
    raise KeyError(f"unknown structure: {structure}")


def structure_shared_params(structure: str, full_params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract the 15 shared XAJ parameters from a full structure's params."""
    return {name: full_params[name] for name in COMMON_XAJ}


def build_forcing_dict(forcing_np: Any, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Convert a [batch, time, 3] P,T,PET array into the model forcing dict."""
    forcing = torch.as_tensor(np_float32(forcing_np), device=device, dtype=dtype)
    return {
        "precip": forcing[:, :, 0],
        "temp": forcing[:, :, 1],
        "pet": forcing[:, :, 2],
    }


def np_float32(values: Any) -> Any:
    import numpy as np

    return np.asarray(values, dtype=np.float32)
