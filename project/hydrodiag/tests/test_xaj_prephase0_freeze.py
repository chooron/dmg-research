"""Small pre-Phase-0 freeze checks for native XAJ response semantics."""

from __future__ import annotations

import math
import pytest
import torch

from models import (
    XAJ, XAJLite, XAJControlledN, XAJControlledNLite,
    XAJDE, XAJGE, XAJDR, XAJGR,
    normalized_to_controlled_ci, normalized_to_controlled_cg, normalized_to_tau0,
)
from models.xaj import _prepare_xaj_parameters
from models.parameter_specs import (
    XAJ_PARAM_SPECS, XAJ_DE_PARAM_SPECS, XAJ_GE_PARAM_SPECS,
    XAJ_DR_PARAM_SPECS, XAJ_GR_PARAM_SPECS,
)
from scripts.freeze_native_subsurface_scale import compute_global_z0
from models.structure_response import (
    native_linear_storage, native_linear_step_from_storage, native_linear_tau,
    response_conditioning_tensors, summarize_response_conditioning,
)
from tests.test_xaj_structure_variants import _active_params, _forcing, _initial


CASES = (
    ("legacy native", XAJ, XAJ_PARAM_SPECS),
    ("controlled N", XAJControlledN, XAJControlledN().parameter_specs),
    ("XAJDE", XAJDE, XAJ_DE_PARAM_SPECS),
    ("XAJGE", XAJGE, XAJ_GE_PARAM_SPECS),
    ("XAJDR", XAJDR, XAJ_DR_PARAM_SPECS),
    ("XAJGR", XAJGR, XAJ_GR_PARAM_SPECS),
)


def _native_step_diagnostics(forcing, params, initial):
    model = XAJ()
    batch = forcing["precip"].shape[0]
    k, b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg, _a, _theta = _prepare_xaj_parameters(params)
    wu, wl, wd, s, fr, qi, qg, _buffer = model._init_states(
        batch, forcing["precip"].device, forcing["precip"].dtype, initial, um, lm, dm, sm,
    )
    names = ("evap_total", "rs_instant", "qi", "qg", "wu", "wl", "wd", "s_next", "fr")
    outputs = {name: [] for name in names}
    for t in range(forcing["precip"].shape[1]):
        out = model._step(
            forcing["precip"][:, t], forcing["pet"][:, t],
            wu, wl, wd, s, fr, qi, qg,
            k, b, im, um, lm, dm, c, sm, ex, ki, kg, ci, cg, 1e-8,
        )
        (_q, rs_adj, qi, qg, evap, wu, wl, wd, s, fr,
         _rs, _ri, _rg, _eu, _el, _ed) = out
        for name, value in zip(names, (evap, rs_adj, qi, qg, wu, wl, wd, s, fr)):
            outputs[name].append(value)
    return {name: torch.stack(values, dim=1) for name, values in outputs.items()}


def _balance(model_cls, specs, dtype):
    forcing = _forcing(96, dtype)
    if model_cls in (XAJ, XAJControlledN):
        params = {
            name: torch.full((1,), float(spec["default"]), dtype=dtype)
            for name, spec in specs.items()
        }
        params["xaj_im"] = torch.zeros(1, dtype=dtype)
        p_for_initial = params
        initial = {
            "wu": 0.55 * p_for_initial["xaj_um"],
            "wl": 0.45 * p_for_initial["xaj_lm"],
            "wd": 0.35 * p_for_initial["xaj_dm"],
            "s": 0.30 * p_for_initial["xaj_sm"],
            "fr": torch.full((1,), 0.25, dtype=dtype),
            "qi": torch.full((1,), 0.2, dtype=dtype),
            "qg": torch.full((1,), 0.1, dtype=dtype),
            "rs_uh_buffer": torch.zeros(1, 14, dtype=dtype),
        }
    else:
        params = _active_params(specs, dtype=dtype)
        initial = _initial(model_cls, specs, dtype)
    aux = _native_step_diagnostics(forcing, params, initial) if model_cls in (XAJ, XAJControlledN) else model_cls()(forcing, params, initial_states=initial)[1]
    soil = aux["wu"] + aux["wl"] + aux["wd"]
    free = aux["fr"] * aux["s_next"]
    if model_cls in (XAJDR, XAJGR):
        storage = soil + free + aux["z"]
        out = aux["rs_instant"] + aux["q_ss"]
        initial_storage = initial["wu"] + initial["wl"] + initial["wd"] + initial["fr"] * initial["s"] + initial["z"]
    else:
        if model_cls in (XAJ, XAJControlledN):
            ci, cg = params["xaj_ci"], params["xaj_cg"]
            qi, qg = aux["qi"], aux["qg"]
            initial_qi, initial_qg = initial["qi"], initial["qg"]
        else:
            ci, cg = params["xaj_ci"], params["xaj_cg"]
            qi, qg = aux["qi"], aux["qg"]
            initial_qi, initial_qg = initial["qi"], initial["qg"]
        storage = soil + free + ci.unsqueeze(1) * qi / (1.0 - ci).unsqueeze(1) + cg.unsqueeze(1) * qg / (1.0 - cg).unsqueeze(1)
        out = aux["rs_instant"] + qi + qg
        initial_storage = (
            initial["wu"] + initial["wl"] + initial["wd"] + initial["fr"] * initial["s"]
            + ci * initial_qi / (1.0 - ci) + cg * initial_qg / (1.0 - cg)
        )
    delta = torch.cat((storage[:, :1] - initial_storage[:, None], storage[:, 1:] - storage[:, :-1]), dim=1)
    residual = forcing["precip"] - aux["evap_total"] - out - delta
    scale = torch.maximum(torch.ones_like(residual), forcing["precip"].cumsum(1) + out.cumsum(1) + storage.abs())
    return residual, residual.abs().max(), (residual.abs() / scale).max()


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_native_and_controlled_water_balance_same_host_floor(dtype):
    values = []
    for _name, model_cls, specs in CASES:
        residual, max_abs, max_norm = _balance(model_cls, specs, dtype)
        assert torch.isfinite(residual).all()
        values.append((float(max_abs), float(max_norm)))
        if dtype == torch.float64:
            assert float(max_abs) <= 5e-7
        else:
            assert float(max_abs) <= 2e-3
    # The controlled arms must remain in the same order of numerical closure
    # as native XAJ; this is a baseline comparison, not a new equation.
    native_abs = values[0][0]
    assert max(abs(value[0] - native_abs) for value in values[1:]) < 1e-6


def test_controlled_response_domain_mapping_and_native_identity():
    dtype = torch.float64
    n = torch.tensor([0.0, 0.5, 1.0], dtype=dtype)
    assert torch.equal(normalized_to_controlled_ci(n), torch.tensor([0.1, 0.5, 0.9], dtype=dtype))
    assert torch.allclose(normalized_to_controlled_cg(n), torch.tensor([0.9, 0.949, 0.998], dtype=dtype), atol=1e-15, rtol=1e-15)
    tau = normalized_to_tau0(n)
    assert torch.allclose(
        tau,
        torch.tensor([-1.0 / math.log(0.1), math.sqrt((-1.0 / math.log(0.1)) * (-1.0 / math.log(0.998))), -1.0 / math.log(0.998)], dtype=dtype),
        rtol=1e-12, atol=1e-12,
    )
    assert torch.all(tau[1:] > tau[:-1])

    forcing = _forcing(24, dtype)
    params = {name: torch.full((1,), float(spec["default"]), dtype=dtype) for name, spec in XAJ_PARAM_SPECS.items()}
    params["xaj_ci"] = torch.tensor([0.55], dtype=dtype)
    params["xaj_cg"] = torch.tensor([0.96], dtype=dtype)
    initial = {
        "wu": torch.tensor([8.0], dtype=dtype), "wl": torch.tensor([25.0], dtype=dtype),
        "wd": torch.tensor([15.0], dtype=dtype), "s": torch.tensor([4.0], dtype=dtype),
        "fr": torch.tensor([0.25], dtype=dtype), "qi": torch.tensor([0.1], dtype=dtype),
        "qg": torch.tensor([0.1], dtype=dtype), "rs_uh_buffer": torch.zeros(1, 14, dtype=dtype),
    }
    legacy_q, legacy_aux = XAJ()(forcing, params, initial_states=initial, return_states=True)
    controlled_q, controlled_aux = XAJControlledN()(forcing, params, initial_states=initial, return_states=True)
    controlled_lite_q, controlled_lite_aux = XAJControlledNLite()(forcing, params, initial_states=initial)
    assert torch.equal(legacy_q, controlled_q)
    assert torch.allclose(controlled_q, controlled_lite_q, atol=2e-6, rtol=2e-6)
    assert controlled_lite_aux == {}
    for key in ("wu", "wl", "wd", "s", "fr", "qi", "qg"):
        assert torch.equal(legacy_aux[key], controlled_aux[key])
    assert XAJ().parameter_specs["xaj_ci"]["upper"] == 1.0
    assert XAJ().parameter_specs["xaj_cg"]["upper"] == 1.0
    assert XAJControlledN().parameter_specs["xaj_ci"]["upper"] == 0.9
    assert XAJControlledN().parameter_specs["xaj_cg"]["upper"] == 0.998
    assert all(float(spec["upper"]) < 1.0 for name, spec in XAJControlledN().parameter_specs.items() if name in ("xaj_ci", "xaj_cg"))
    assert XAJControlledNLite().parameter_specs is XAJControlledN().parameter_specs


def test_c_equal_one_is_forward_defined_after_latent_initialization():
    dtype = torch.float64
    inputs = torch.tensor([0.0, 2.0, 0.0, 5.0, 1.0], dtype=dtype)
    c = torch.ones(1, dtype=dtype)
    z = torch.zeros(1, dtype=dtype)
    q_values = []
    z_values = []
    for value in inputs:
        q, z = native_linear_step_from_storage(z + value.reshape(1), c)
        q_values.append(q)
        z_values.append(z)
    assert torch.equal(torch.cat(q_values), torch.zeros(len(inputs), dtype=dtype))
    assert torch.equal(torch.cat(z_values), inputs.cumsum(0))

    # A finite latent initialization is also valid at C=1, but is not
    # recoverable from a nonzero native Q0 alone.
    z = torch.tensor([3.0], dtype=dtype)
    for value in inputs:
        q, z = native_linear_step_from_storage(z + value.reshape(1), c)
        assert q == 0.0 and torch.isfinite(z).all()
    assert z == 3.0 + inputs.sum()

    c_near = torch.tensor([0.999999], dtype=dtype)
    z = torch.tensor([3.0], dtype=dtype)
    near_q = []
    for value in inputs:
        q, z = native_linear_step_from_storage(z + value.reshape(1), c_near)
        near_q.append(q)
    assert torch.isfinite(torch.cat(near_q)).all()
    assert float(torch.cat(near_q).max()) < 1e-4


def test_global_z0_center_is_deterministic_and_equal_basin_weighted():
    latent = torch.tensor([[1.0, 2.0, 4.0] * 34, [2.0, 4.0, 8.0] * 34], dtype=torch.float64).numpy()
    ids = ["a", "b"]
    finite = torch.tensor([True, True]).numpy()
    counts = torch.tensor([102, 102]).numpy()
    z0_a, summary_a = compute_global_z0(latent, ids, finite, counts)
    z0_b, summary_b = compute_global_z0(latent, ids, finite, counts)
    assert z0_a == z0_b
    assert summary_a == summary_b
    assert abs(z0_a - 2.8284271247461903) < 1e-12


def test_native_interflow_and_groundwater_recursions_equal_latent_storage():
    dtype = torch.float64
    rainfall = torch.tensor([0.0, 2.0, 0.0, 5.0, 1.0, 0.0, 3.0] * 8, dtype=dtype)
    for _channel in ("interflow", "groundwater"):
        for c_value in (0.1, 0.5, 0.9, 0.99, 0.999):
            c = torch.tensor([c_value], dtype=dtype)
            tau = native_linear_tau(c)
            assert torch.allclose(torch.exp(-1.0 / tau), c, atol=1e-15, rtol=1e-15)
            for q0_value in (0.0, 0.1, 2.0, 10.0):
                q_previous = torch.tensor([q0_value], dtype=dtype)
                z_previous = native_linear_storage(q_previous, c)
                q_recursion = []
                q_latent = []
                for input_value in rainfall:
                    r = input_value.reshape(1)
                    q_next = c * q_previous + (1.0 - c) * r
                    z_available = z_previous + r
                    q_from_z, z_next = native_linear_step_from_storage(z_available, c)
                    q_recursion.append(q_next)
                    q_latent.append(q_from_z)
                    q_previous, z_previous = q_next, z_next
                assert torch.allclose(torch.cat(q_recursion), torch.cat(q_latent), atol=2e-15, rtol=2e-15)
                expected_z = native_linear_storage(q_previous, c)
                assert torch.allclose(z_previous, expected_z, atol=2e-15, rtol=2e-15)


def test_gr_conditioning_diagnostics_and_extinction():
    dtype = torch.float64
    z_available = torch.tensor([0.0, 1e-3, 0.01, 1.0, 10.0], dtype=dtype)
    z_new = torch.tensor([0.0, 0.0, 0.002, 0.5, 9.0], dtype=dtype)
    z0 = torch.tensor([1.0], dtype=dtype)
    extinct, positive, log_ratio = response_conditioning_tensors(z_available, z_new, z0)
    summary = summarize_response_conditioning(z_available, z_new, z0)
    assert torch.equal(extinct, torch.tensor([False, True, False, False, False]))
    assert torch.equal(positive, torch.tensor([False, True, True, True, True]))
    assert log_ratio[0] == 0
    assert summary["extinction_count"] == 1
    assert summary["positive_available_count"] == 4
    assert torch.allclose(summary["f_extinct"], torch.tensor(0.25, dtype=dtype))
    assert torch.isfinite(torch.stack(tuple(summary.values()))).all()


def test_xajgr_exposes_conditioning_summary_and_default_fixed_z0():
    dtype = torch.float64
    forcing = _forcing(24, dtype)
    specs = XAJ_GR_PARAM_SPECS
    params = _active_params(specs, dtype=dtype)
    model = XAJGR(compile_step=False)
    _q, aux = model(forcing, params)
    for key in (
        "z_available", "z", "extinction_mask", "log_z_ratio",
        "g_r_extinction_count", "g_r_positive_available_count",
        "g_r_f_extinct", "g_r_log_z_ratio_std",
    ):
        assert key in aux
    assert torch.isfinite(aux["log_z_ratio"]).all()
    assert torch.isfinite(aux["g_r_f_extinct"])
    assert abs(float(model.z0) - 3.1553493591016335) < 1e-12
    explicit = XAJGR(compile_step=False, z0=3.1553493591016335)
    q_explicit, _ = explicit(forcing, params)
    assert torch.equal(_q, q_explicit)
