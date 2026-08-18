"""R4 post-hoc continuous forward and per-day state export.

This module is a port of the R3 recorded-forward harness
(``manuscript/r3/recorded_forward.py``, branch ``hydrodiag/R3-exp``) — the same
per-day-state recording wrappers around the *production* compiled step
kernels, with the same numerical-identity guarantee against the production
model forward.  It is reused here rather than re-implemented; see
``validate_recorded_forward``.

Conventions (R4, matching R1/R2 and R3):

- continuous forward over the full 12418-day axis from zero initial states;
- the CN snow module computes ``psol_annual`` / ``g_thresh = 0.9 * psol_annual``
  from the forcing window passed to the forward (window-based semantics —
  the R1/R2 historical semantics; the R3 ``canonical_cn_psol_annual`` path is
  deliberately NOT used here);
- target periods are sliced from the continuous simulation:
  train = 1981-10-01..1995-09-30, test = 1995-10-01..2010-09-30;
- Base/TGD2 never construct pseudo-SWE columns: their snow-related exports
  are empty and the manifest states ``has_snow_module=false``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .common import PERIOD_INDEX

COMMON_STATES = ("wu", "wl", "wd", "s", "fr", "qi", "qg")

# State keys exported per day by each structure.
CN_STATE_KEYS = (
    "G",
    "eTG",
    "sca",
    "rain",
    "melt",
    "effective_precip",
    "wu",
    "wl",
    "wd",
    "s",
    "fr",
    "qi",
    "qg",
    "rs_instant",
    "evap",
)
BASE_STATE_KEYS = ("wu", "wl", "wd", "s", "fr", "qi", "qg", "rs_instant")
TGD2_STATE_KEYS = BASE_STATE_KEYS + ("tgd_storage", "tgd_tau", "tgd_retention")


def _prepare_loop_quantities(
    xaj_params: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    from models.xaj import _prepare_xaj_parameters

    names = (
        "k",
        "b",
        "im",
        "um",
        "lm",
        "dm",
        "c",
        "sm",
        "ex",
        "ki",
        "kg",
        "ci",
        "cg",
        "a_uh",
        "theta_uh",
    )
    values = _prepare_xaj_parameters(xaj_params)
    return dict(zip(names, values))


def recorded_cn_forward(
    model,
    forcings: dict[str, torch.Tensor],
    params: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
):
    """Mirror ``XAJWithCemaNeigeLite.forward`` and record per-day states.

    Returns (qsim, states, final_states); ``states`` holds per-day tensors
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
    xaj_step_params = tuple(
        q[n]
        for n in (
            "k",
            "b",
            "im",
            "um",
            "lm",
            "dm",
            "c",
            "sm",
            "ex",
            "ki",
            "kg",
            "ci",
            "cg",
        )
    )
    a_uh, theta_uh = q["a_uh"], q["theta_uh"]
    (wu, wl, wd, s, fr, qi, qg, rs_uh_buffer) = model._xaj._init_states(
        batch, device, dtype, None, q["um"], q["lm"], q["dm"], q["sm"]
    )

    stores = {
        name: torch.zeros(batch, nsteps, device=device, dtype=dtype)
        for name in (
            "wu",
            "wl",
            "wd",
            "s",
            "fr",
            "qi",
            "qg",
            "rs_instant",
            "evap",
            "G",
            "eTG",
            "sca",
            "rain",
            "melt",
            "effective_precip",
        )
    }
    for t in range(nsteps):
        (
            effective_t,
            G,
            eTG,
            sca_t,
            rain_t,
            melt_t,
            _q_out,
            rs_adj_t,
            qi_t,
            qg_t,
            evap_t,
            wu,
            wl,
            wd,
            s,
            fr,
            _rs,
            _ri,
            _rg,
            _eu,
            _el,
            _ed,
        ) = model._fused_step(
            precip[:, t],
            temp[:, t],
            pet[:, t],
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
        "wu": wu,
        "wl": wl,
        "wd": wd,
        "s": s,
        "fr": fr,
        "qi": stores["qi"][:, -1],
        "qg": stores["qg"][:, -1],
        "G": G,
        "eTG": eTG,
    }
    return qsim, stores, final_states


def recorded_base_forward(
    model,
    forcings: dict[str, torch.Tensor],
    params: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
):
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
    step_args = (
        q["k"],
        q["b"],
        q["im"],
        q["um"],
        q["lm"],
        q["dm"],
        q["c"],
        q["sm"],
        q["ex"],
        q["ki"],
        q["kg"],
        q["ci"],
        q["cg"],
    )

    stores = {
        name: torch.zeros(batch, nsteps, device=device, dtype=dtype)
        for name in ("wu", "wl", "wd", "s", "fr", "qi", "qg", "rs_instant")
    }
    for t in range(nsteps):
        rs_adj_t, qi_t, qg_t, wu, wl, wd, s, fr = model._compact_step(
            precip[:, t].contiguous(),
            pet[:, t].contiguous(),
            wu,
            wl,
            wd,
            s,
            fr,
            qi,
            qg,
            *step_args,
            model.nearzero,
            wm,
            wmm,
            ms,
            one_minus_im,
            one_minus_ki_kg,
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
        "wu": wu,
        "wl": wl,
        "wd": wd,
        "s": s,
        "fr": fr,
        "qi": stores["qi"][:, -1],
        "qg": stores["qg"][:, -1],
    }
    return qsim, stores, final_states


def recorded_tgd2_forward(
    model,
    forcings: dict[str, torch.Tensor],
    params: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
):
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
            storage,
            tau_t,
            retention_t,
        )

    xaj_params = {k: v for k, v in params.items() if k.startswith("xaj_")}
    qsim, stores, final_states = recorded_base_forward(
        model._runoff,
        {"precip": effective, "pet": pet, "temp": temp},
        xaj_params,
        device,
        dtype,
    )
    stores["tgd_storage"] = storage_trace
    stores["tgd_tau"] = tau_trace
    stores["tgd_retention"] = retention_trace
    final_states["tgd_storage"] = storage
    return qsim, stores, final_states


def validate_recorded_forward(
    model,
    recorded: tuple,
    forcings: dict[str, torch.Tensor],
    params: dict[str, torch.Tensor],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, float]:
    """Compare a recorded forward against the production model forward.

    Returns a dict with max absolute discharge diff, max relative discharge
    diff, and per-final-state max abs diff.  Raises RuntimeError on
    divergence beyond tolerance.
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
            state_diffs[key] = float(
                (value.detach() - prod_final[key]).abs().max().item()
            )
    max_state = max(state_diffs.values(), default=0.0)
    if abs_diff > atol or max_state > atol:
        raise RuntimeError(
            f"recorded forward diverged from production forward: "
            f"q_abs={abs_diff:.3e} q_rel={rel_diff:.3e} state_max={max_state:.3e}"
        )
    return {"q_abs_max": abs_diff, "q_rel_max": rel_diff, "state_abs_max": max_state}


def recorded_forward_for_structure(
    structure: str,
    model,
    forcings: dict[str, torch.Tensor],
    params: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
):
    """Dispatch to the recorded forward matching the fitted structure key."""
    if structure == "XAJ_CN":
        return recorded_cn_forward(model, forcings, params, device, dtype)
    if structure == "XAJ":
        return recorded_base_forward(model, forcings, params, device, dtype)
    if structure == "XAJ_TGD2":
        return recorded_tgd2_forward(model, forcings, params, device, dtype)
    raise KeyError(f"unknown structure: {structure}")


def build_forcing_dict(
    forcing_np: Any, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Convert a [batch, time, 3] P,T,PET array into the model forcing dict."""
    import numpy as np

    forcing = torch.as_tensor(
        np.asarray(forcing_np, dtype=np.float32), device=device, dtype=dtype
    )
    return {
        "precip": forcing[:, :, 0],
        "temp": forcing[:, :, 1],
        "pet": forcing[:, :, 2],
    }


def model_instances(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    """Canonical lite model instances for the three R1/R2 structures."""
    from models import XAJLite, XAJWithCemaNeigeLite, XAJWithTGD2Lite

    return {
        "XAJ": XAJLite().to(device).eval(),
        "XAJ_CN": XAJWithCemaNeigeLite().to(device).eval(),
        "XAJ_TGD2": XAJWithTGD2Lite().to(device).eval(),
    }


def continuous_forward(
    structure: str,
    model: Any,
    theta_hat: Any,  # np.ndarray [n, D] physical parameters
    forcing_full: Any,  # np.ndarray [n, time, 3] P,T,PET (float32)
    device: torch.device,
    dtype: torch.dtype,
    batch: int = 64,
    validate_subset: Optional[int] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Continuous full-axis recorded forward in batches.

    Returns (q, states) with q = np.ndarray [n, time] and states = dict of
    np.ndarray [n, time].  ``validate_subset`` (int or None) additionally
    runs the production forward on the first N basins and asserts identity
    via ``validate_recorded_forward`` (raises on divergence).
    """
    import numpy as np
    from ablation.ic_core.parameter_adapter import get_parameter_spec

    names = tuple(get_parameter_spec(structure))
    n = theta_hat.shape[0]
    time = forcing_full.shape[1]

    q_full = np.empty((n, time), dtype=np.float64)
    state_keys = None
    states_full: dict[str, Any] = {}
    if validate_subset is not None and validate_subset > 0:
        # identity check on the first `validate_subset` basins
        sub = min(validate_subset, n)
        fc = build_forcing_dict(forcing_full[:sub], device, dtype)
        params = {
            name: torch.from_numpy(theta_hat[:sub, i]).to(device, dtype=dtype)
            for i, name in enumerate(names)
        }
        recorded = recorded_forward_for_structure(
            structure, model, fc, params, device, dtype
        )
        validate_recorded_forward(model, recorded, fc, params)

    for left in range(0, n, batch):
        right = min(n, left + batch)
        fc = build_forcing_dict(forcing_full[left:right], device, dtype)
        params = {
            name: torch.from_numpy(theta_hat[left:right, i]).to(device, dtype=dtype)
            for i, name in enumerate(names)
        }
        qsim, stores, _final = recorded_forward_for_structure(
            structure, model, fc, params, device, dtype
        )
        q_full[left:right] = qsim.detach().cpu().numpy().astype(np.float64)
        if state_keys is None:
            state_keys = sorted(stores.keys())
            for key in state_keys:
                states_full[key] = np.empty((n, time), dtype=np.float64)
        for key in state_keys:
            states_full[key][left:right] = (
                stores[key].detach().cpu().numpy().astype(np.float64)
            )
    return q_full, states_full


def cn_psol_gthresh(
    precip_window: Any,
    temp_window: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, Any]:
    """psol_annual / g_thresh for a forcing window (R1/R2 window-based semantics).

    Exactly mirrors what the CN forward computes internally:
    ``g_thresh = 0.9 * _estimate_psol_annual(precip, temp)``.
    Returns numpy arrays [n].
    """
    import numpy as np
    from models.cemaneige import _estimate_psol_annual

    p = torch.as_tensor(
        np.asarray(precip_window, dtype=np.float32), device=device, dtype=dtype
    )
    t = torch.as_tensor(
        np.asarray(temp_window, dtype=np.float32), device=device, dtype=dtype
    )
    psol = _estimate_psol_annual(p, t)
    return psol.detach().cpu().numpy(), (0.9 * psol).detach().cpu().numpy()


def period_slices_full(time: int) -> dict[str, slice]:
    """R1/R2 period slices on the full axis (alias of common.period_slices)."""
    p = PERIOD_INDEX
    return {
        "full": slice(0, time),
        "train": slice(p["train_start"], p["train_end"] + 1),
        "test": slice(p["test_start"], p["test_end"] + 1),
        "train_forcing": slice(p["train_forcing_start"], p["train_forcing_end"]),
        "test_forcing": slice(p["test_forcing_start"], p["test_forcing_end"]),
    }
