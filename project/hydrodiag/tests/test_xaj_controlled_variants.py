"""Physics, gradient and integration checks for the controlled XAJ+CN variants."""

from __future__ import annotations

import io

import pytest
import torch

from ablation.ic_core.model_adapter import get_parameter_spec
from models import (
    XAJ, XAJWithCemaNeige,
    XAJ2SWithCemaNeige, XAJ2SWithCemaNeigeLite,
    XAJRWPEWithCemaNeige, XAJRWPEWithCemaNeigeLite,
)
from models.parameter_specs import XAJ_2S_PARAM_SPECS, XAJ_RWPE_PARAM_SPECS, XAJ_CN_PARAM_SPECS, XAJ_PARAM_SPECS
from models.xaj import _rootzone_moisture_stress_evaporation
from training.dpl.run_dpl_model import LITE_MODEL_REGISTRY, MODEL_REGISTRY


def _params(specs, batch=1, dtype=torch.float64, requires_grad=False):
    return {
        name: torch.tensor([spec["default"]] * batch, dtype=dtype, requires_grad=requires_grad)
        for name, spec in specs.items()
    }


def _forcings(kind="mixed", n=32, dtype=torch.float64):
    t = torch.arange(n, dtype=dtype)
    if kind == "zero":
        precip, pet, temp = torch.zeros(n, dtype=dtype), torch.zeros(n, dtype=dtype), torch.zeros(n, dtype=dtype)
    elif kind == "pulse":
        precip, pet, temp = torch.zeros(n, dtype=dtype), torch.full((n,), 1.0, dtype=dtype), torch.full((n,), 4.0, dtype=dtype)
        precip[2] = 35.0
    elif kind == "snow":
        precip = torch.where(t < n // 2, torch.full_like(t, 4.0), torch.zeros_like(t))
        pet = torch.full_like(t, 1.0)
        temp = torch.where(t < n // 2, torch.full_like(t, -3.0), torch.full_like(t, 4.0))
    else:
        precip = torch.where((t % 5) < 2, 5.0 + (t % 3), 0.0)
        pet = 1.0 + (t % 4) * 0.4
        temp = torch.where((t % 7) < 3, -1.5, 4.0)
    return {
        "precip": precip.to(dtype=dtype).unsqueeze(0),
        "pet": pet.to(dtype=dtype).unsqueeze(0),
        "temp": temp.to(dtype=dtype).unsqueeze(0),
    }


def _initial(dtype=torch.float64):
    # Explicit stores make the balance accounting below independent of model
    # defaults and exercise recession, dry-down and snow carry state.
    return {
        "cn_G": torch.tensor([0.0], dtype=dtype), "cn_eTG": torch.tensor([-1.0], dtype=dtype),
        "xaj_wu": torch.tensor([8.0], dtype=dtype), "xaj_wl": torch.tensor([25.0], dtype=dtype),
        "xaj_wd": torch.tensor([15.0], dtype=dtype), "xaj_s": torch.tensor([4.0], dtype=dtype),
        "xaj_fr": torch.tensor([0.25], dtype=dtype), "xaj_qi": torch.tensor([0.0], dtype=dtype),
        "xaj_qg": torch.tensor([0.0], dtype=dtype), "xaj_qb": torch.tensor([0.0], dtype=dtype),
    }


def test_existing_xaj_and_xaj_cn_regression():
    """The shared-kernel extension must leave both established paths intact."""
    forcing = {
        "precip": torch.tensor([[0., 3., 9., 0., 2.]], dtype=torch.float64),
        "pet": torch.tensor([[1., 2., 1., 3., 2.]], dtype=torch.float64),
        "temp": torch.tensor([[-2., 1., 4., 3., -1.]], dtype=torch.float64),
    }
    expected = {
        XAJ: [0.37669, 0.3797010711767224, 0.5381540771125795, 0.4920415187612006, 0.39790624785320394],
        XAJWithCemaNeige: [0.37669, 0.3525812, 0.5097357156675512, 0.4682594474737844, 0.37971273166156605],
    }
    for model_cls, specs in ((XAJ, XAJ_PARAM_SPECS), (XAJWithCemaNeige, XAJ_CN_PARAM_SPECS)):
        params = _params(specs)
        qsim, _ = model_cls().to(dtype=torch.float64)(forcing, params)
        assert torch.allclose(qsim[0], torch.tensor(expected[model_cls], dtype=torch.float64), atol=1e-12, rtol=1e-12)


def _mass_residuals(aux, params, initial, two_source):
    """Return CN and raw-XAJ daily residuals, including slow reservoir storage.

    Surface routing is intentionally excluded here: ``rs_adj`` is the flux
    entering the UH, and its finite convolution storage is a separate route
    state.  This isolates exact daily snow+XAJ water bookkeeping.
    """
    effective = aux["effective_precip"]
    snow = aux["snow_pack"]
    snow_prev = torch.cat((initial["cn_G"].unsqueeze(1), snow[:, :-1]), dim=1)
    cn_resid = effective - (aux["rain"] + aux["melt"])
    cn_tol = 1e-12 if effective.dtype == torch.float64 else 2e-6
    assert torch.max(torch.abs(cn_resid)).item() < cn_tol
    # CemaNeige does not expose snowfall directly.  Recover it from its exact
    # state update G_next = G_prev + snowfall - melt, then check the full CN
    # identity P_total + G_prev = effective_liquid + G_next.
    snowfall = snow - snow_prev + aux["melt"]
    raw_precip = aux["rain"] + snowfall
    assert torch.max(torch.abs(raw_precip + snow_prev - effective - snow)).item() < cn_tol

    soil = aux["wu"] + aux["wl"] + aux["wd"]
    free = aux["fr"] * aux["s_next"]
    if two_source:
        cb = params["xaj_cb"].unsqueeze(1)
        route = cb * aux["qb"] / (1.0 - cb)
        route0 = params["xaj_cb"] * initial["xaj_qb"] / (1.0 - params["xaj_cb"])
        out = aux["rs_adj"] + aux["qb"]
    else:
        ci = params["xaj_ci"].unsqueeze(1); cg = params["xaj_cg"].unsqueeze(1)
        route = ci * aux["qi"] / (1.0 - ci) + cg * aux["qg"] / (1.0 - cg)
        route0 = params["xaj_ci"] * initial["xaj_qi"] / (1.0 - params["xaj_ci"]) + params["xaj_cg"] * initial["xaj_qg"] / (1.0 - params["xaj_cg"])
        out = aux["rs_adj"] + aux["qi"] + aux["qg"]
    state = soil + free + route
    state0 = (initial["xaj_wu"] + initial["xaj_wl"] + initial["xaj_wd"] + initial["xaj_fr"] * initial["xaj_s"] + route0)
    delta = torch.cat((state[:, :1] - state0.unsqueeze(1), state[:, 1:] - state[:, :-1]), dim=1)
    return effective - aux["evap_total"] - out - delta


@pytest.mark.parametrize("model_cls,specs,two_source", [
    (XAJ2SWithCemaNeige, XAJ_2S_PARAM_SPECS, True),
    (XAJRWPEWithCemaNeige, XAJ_RWPE_PARAM_SPECS, False),
])
def test_xaj_variants_forward_diagnostics_and_cn(model_cls, specs, two_source):
    model = model_cls().to(dtype=torch.float64)
    qsim, aux = model(_forcings("snow", 18), _params(specs), initial_states=_initial(), return_states=True)
    assert qsim.shape == (1, 18)
    assert torch.isfinite(qsim).all() and torch.all(qsim >= 0.0)
    assert aux["model_name"] in {"XAJ_2S", "XAJ_RWPE"}
    assert aux["effective_precip"].shape == qsim.shape
    assert {"cn_G", "cn_eTG", "xaj_rs_uh_buffer"} <= set(aux["final_states"])
    if two_source:
        assert {"qb", "rb", "rs_adj", "evap_total", "s_next", "fr"} <= set(aux)
        assert "qi" not in aux and "qg" not in aux
        assert "xaj_qb" in aux["final_states"]
    else:
        assert aux["evaporation_scheme"] == "aggregated_rootzone_moisture_stress"
        assert {"qi", "qg", "ri", "rg", "eu", "el", "ed", "er", "z_root", "root_stress", "tau_e"} <= set(aux)


def test_xaj_rwpe_evaporation_invariants():
    params = _params(XAJ_RWPE_PARAM_SPECS)
    forcing = _forcings("mixed", 25)
    _, aux = XAJRWPEWithCemaNeige().to(dtype=torch.float64)(forcing, params, initial_states=_initial())
    pet_adj = forcing["pet"] * params["xaj_k"].unsqueeze(1)
    assert torch.all(aux["eu"] >= 0) and torch.all(aux["el"] >= 0) and torch.all(aux["ed"] >= 0)
    assert torch.all(aux["evap_total"] <= pet_adj + 1e-12)
    # State after each step cannot have supplied more lower/deep water than
    # was available before it; the explicit initial condition covers t=0.
    wl_prev = torch.cat((_initial()["xaj_wl"].view(1, 1), aux["wl"][:, :-1]), dim=1)
    wd_prev = torch.cat((_initial()["xaj_wd"].view(1, 1), aux["wd"][:, :-1]), dim=1)
    assert torch.all(aux["el"] <= wl_prev + 1e-12)
    assert torch.all(aux["ed"] <= wd_prev + 1e-12)
    assert torch.allclose(aux["el"] + aux["ed"], aux["er"], atol=1e-12, rtol=1e-12)
    assert torch.all(aux["er"] <= aux["z_root"] * 0 + (wl_prev + wd_prev) + 1e-12)


def test_xaj_rwpe_rootzone_stress_formula_invariants():
    dtype = torch.float64
    demand = torch.tensor([4.0, 4.0, 4.0, 4.0], dtype=dtype)
    wl = torch.tensor([0.0, 20.0, 20.0, 50.0], dtype=dtype)
    wd = torch.tensor([0.0, 0.0, 40.0, 50.0], dtype=dtype)
    lm = torch.full((4,), 50.0, dtype=dtype); dm = torch.full((4,), 50.0, dtype=dtype)
    tau = torch.tensor([0.05, 1.0, 0.6, 0.5], dtype=dtype)
    el, ed, er, z_root, stress = _rootzone_moisture_stress_evaporation(demand, wl, wd, lm, dm, tau, 1e-8)
    assert er[0] == 0 and el[0] == 0 and ed[0] == 0
    assert stress[0] == 0 and stress[-1] == 1
    assert torch.all(er <= demand) and torch.all(er <= wl + wd)
    assert torch.allclose(el + ed, er, atol=1e-12, rtol=1e-12)
    assert torch.all(el <= wl) and torch.all(ed <= wd)
    # Increasing root moisture cannot decrease ER at fixed capacity/tau.
    _el0, _ed0, er_dry, _z0, _s0 = _rootzone_moisture_stress_evaporation(
        torch.tensor([4.0], dtype=dtype), torch.tensor([10.0], dtype=dtype), torch.tensor([10.0], dtype=dtype),
        torch.tensor([50.0], dtype=dtype), torch.tensor([50.0], dtype=dtype), torch.tensor([0.5], dtype=dtype), 1e-8,
    )
    _el1, _ed1, er_wet, _z1, _s1 = _rootzone_moisture_stress_evaporation(
        torch.tensor([4.0], dtype=dtype), torch.tensor([30.0], dtype=dtype), torch.tensor([30.0], dtype=dtype),
        torch.tensor([50.0], dtype=dtype), torch.tensor([50.0], dtype=dtype), torch.tensor([0.5], dtype=dtype), 1e-8,
    )
    assert er_wet >= er_dry


@pytest.mark.parametrize("tau_e", [0.05, 0.050001, 0.5, 0.999999, 1.0])
def test_xaj_rwpe_tau_e_boundary_formula_is_finite_and_conservative(tau_e):
    """The mapped tau_e interval is stable at both physical endpoints."""
    dtype = torch.float64
    el, ed, er, z_root, stress = _rootzone_moisture_stress_evaporation(
        torch.tensor([12.0, 12.0, 12.0], dtype=dtype),
        torch.tensor([0.0, 20.0, 50.0], dtype=dtype),
        torch.tensor([0.0, 20.0, 50.0], dtype=dtype),
        torch.full((3,), 50.0, dtype=dtype), torch.full((3,), 50.0, dtype=dtype),
        torch.full((3,), tau_e, dtype=dtype), 1e-8,
    )
    assert torch.isfinite(torch.stack((el, ed, er, z_root, stress))).all()
    assert torch.all((stress >= 0.0) & (stress <= 1.0))
    assert torch.allclose(el + ed, er, atol=1e-12, rtol=1e-12)
    assert torch.all(el <= torch.tensor([0.0, 20.0, 50.0], dtype=dtype))
    assert torch.all(ed <= torch.tensor([0.0, 20.0, 50.0], dtype=dtype))


@pytest.mark.parametrize("model_cls,specs", [
    (XAJ2SWithCemaNeige, XAJ_2S_PARAM_SPECS),
    (XAJRWPEWithCemaNeige, XAJ_RWPE_PARAM_SPECS),
])
def test_xaj_variants_eager_and_compiled_forward_backward_agree(model_cls, specs):
    forcing = _forcings("mixed", 32)
    forcing["pet"] = torch.full_like(forcing["pet"], 5.0)
    eager_params = _params(specs, requires_grad=True)
    compiled_params = _params(specs, requires_grad=True)
    eager_q, _ = model_cls(compile_step=False).to(dtype=torch.float64)(forcing, eager_params, initial_states=_initial())
    compiled_q, _ = model_cls().to(dtype=torch.float64)(forcing, compiled_params, initial_states=_initial())
    assert torch.allclose(eager_q, compiled_q, atol=1e-12, rtol=1e-12)
    eager_q.square().mean().backward(); compiled_q.square().mean().backward()
    for name in specs:
        eager_grad, compiled_grad = eager_params[name].grad, compiled_params[name].grad
        if eager_grad is not None: assert torch.isfinite(eager_grad).all()
        if compiled_grad is not None: assert torch.isfinite(compiled_grad).all()
    core = {"cn_kf", "xaj_k"} | ({"xaj_kb", "xaj_cb"} if "xaj_kb" in specs else {"xaj_tau_e"})
    for name in core:
        assert eager_params[name].grad is not None and compiled_params[name].grad is not None


@pytest.mark.parametrize("model_cls,specs,two_source", [
    (XAJ2SWithCemaNeige, XAJ_2S_PARAM_SPECS, True),
    (XAJRWPEWithCemaNeige, XAJ_RWPE_PARAM_SPECS, False),
])
@pytest.mark.parametrize("kind", ["zero", "pulse", "mixed", "snow"])
def test_xaj_variants_one_step_and_long_sequence_mass_balance(model_cls, specs, two_source, kind):
    # No project tolerance existed for this newly exposed diagnostic.  These
    # thresholds distinguish float64 arithmetic from float32 production noise.
    maxima = []
    for dtype, tolerance in ((torch.float64, 5e-7), (torch.float32, 2e-3)):
        model = model_cls().to(dtype=dtype)
        forcing = {key: value.to(dtype=dtype) for key, value in _forcings(kind, 80).items()}
        initial = {key: value.to(dtype=dtype) for key, value in _initial().items()}
        params = _params(specs, dtype=dtype)
        # IM is set to zero to keep the native XAJ tension/free-water balance
        # definition explicit; its direct impervious branch is separately
        # exercised by forward/boundary tests.
        params["xaj_im"] = torch.zeros(1, dtype=dtype)
        _, aux = model(forcing, params, initial_states=initial)
        resid = _mass_residuals(aux, params, initial, two_source)
        maxima.append(float(resid.abs().max()))
        assert torch.isfinite(resid).all()
        assert resid.abs().max().item() < tolerance
        assert resid.abs().mean().item() < tolerance
    assert maxima[0] <= maxima[1] + 1e-8


@pytest.mark.parametrize("model_cls,specs", [
    (XAJ2SWithCemaNeige, XAJ_2S_PARAM_SPECS),
    (XAJRWPEWithCemaNeige, XAJ_RWPE_PARAM_SPECS),
])
def test_xaj_variants_parameter_boundaries_and_gradients(model_cls, specs):
    params = {}
    for name, spec in specs.items():
        params[name] = torch.tensor([spec["lower"], (spec["lower"] + spec["upper"]) / 2, spec["upper"]], dtype=torch.float64, requires_grad=True)
    forcing = {key: value.expand(3, -1) for key, value in _forcings("mixed", 20).items()}
    qsim, aux = model_cls().to(dtype=torch.float64)(forcing, params, initial_states=None, return_states=True)
    assert torch.isfinite(qsim).all()
    for key in ("wu", "wl", "wd", "s_next", "fr", "snow_pack"):
        assert torch.isfinite(aux[key]).all() and torch.all(aux[key] >= -1e-10)
    for key, capacity in (("wu", "xaj_um"), ("wl", "xaj_lm"), ("wd", "xaj_dm"), ("s_next", "xaj_sm")):
        assert torch.all(aux[key] <= params[capacity].unsqueeze(1) + 1e-8)
    assert torch.all(aux["fr"] <= 1.0 + 1e-10)
    if model_cls is XAJRWPEWithCemaNeige:
        assert "xaj_c" not in specs and "xaj_tau_e" in specs
        assert torch.allclose(aux["el"] + aux["ed"], aux["er"], atol=1e-10, rtol=1e-10)
    (qsim.square().mean()).backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in params.values())


def _finite_difference(model_cls, specs, name, epsilon):
    forcing = _forcings("mixed", 32)
    # Activate both CN melt sensitivity and the lower/deep evaporation path;
    # finite differences at an inactive piecewise branch are not informative.
    forcing["pet"] = torch.full_like(forcing["pet"], 5.0)
    initial = _initial()
    analytic = _params(specs, requires_grad=True)
    loss = model_cls().to(dtype=torch.float64)(forcing, analytic, initial_states=initial)[0].sum()
    loss.backward()
    ad = analytic[name].grad.item()
    assert abs(ad) > 1e-10, f"{name} did not receive an activated gradient"
    plus = _params(specs); minus = _params(specs)
    plus[name] = plus[name] + epsilon; minus[name] = minus[name] - epsilon
    fp = model_cls().to(dtype=torch.float64)(forcing, plus, initial_states=initial)[0].sum().item()
    fm = model_cls().to(dtype=torch.float64)(forcing, minus, initial_states=initial)[0].sum().item()
    fd = (fp - fm) / (2.0 * epsilon)
    assert abs(ad - fd) <= 2e-3 + 2e-2 * max(abs(ad), abs(fd))


@pytest.mark.parametrize("name,eps", [("xaj_kb", 1e-5), ("xaj_cb", 1e-5), ("xaj_b", 1e-5), ("cn_kf", 1e-5)])
def test_xaj_2s_finite_difference(name, eps):
    _finite_difference(XAJ2SWithCemaNeige, XAJ_2S_PARAM_SPECS, name, eps)


@pytest.mark.parametrize("name,eps", [("xaj_k", 1e-5), ("xaj_tau_e", 1e-5), ("xaj_lm", 1e-4), ("xaj_dm", 1e-4), ("cn_kf", 1e-5)])
def test_xaj_rwpe_finite_difference(name, eps):
    _finite_difference(XAJRWPEWithCemaNeige, XAJ_RWPE_PARAM_SPECS, name, eps)


@pytest.mark.parametrize("full_cls,lite_cls,specs,key", [
    (XAJ2SWithCemaNeige, XAJ2SWithCemaNeigeLite, XAJ_2S_PARAM_SPECS, "XAJ_2S"),
    (XAJRWPEWithCemaNeige, XAJRWPEWithCemaNeigeLite, XAJ_RWPE_PARAM_SPECS, "XAJ_RWPE"),
])
def test_xaj_variants_registry_lite_and_checkpoint_roundtrip(full_cls, lite_cls, specs, key):
    assert MODEL_REGISTRY[key][0] is full_cls and LITE_MODEL_REGISTRY[key][0] is lite_cls
    assert get_parameter_spec(key) is specs
    forcing, params = _forcings("snow", 16), _params(specs)
    model = full_cls().to(dtype=torch.float64)
    q0, _ = model(forcing, params)
    payload = {"schema": model.checkpoint_schema, "state_dict": model.state_dict(), "model_key": key}
    buffer = io.BytesIO(); torch.save(payload, buffer); buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)
    clone = full_cls().to(dtype=torch.float64); clone.load_state_dict(restored["state_dict"])
    q1, _ = clone(forcing, params)
    assert restored["schema"] == model.checkpoint_schema and restored["model_key"] == key
    assert torch.allclose(q0, q1, atol=1e-12, rtol=1e-12)
    q_lite, aux_lite = lite_cls().to(dtype=torch.float64)(forcing, params)
    assert aux_lite == {} and torch.allclose(q0, q_lite, atol=1e-10, rtol=1e-10)
    if key == "XAJ_RWPE":
        with pytest.raises(RuntimeError, match="parallel layer weights"):
            full_cls.validate_checkpoint_schema("xaj_rwpe_cn_v1")
