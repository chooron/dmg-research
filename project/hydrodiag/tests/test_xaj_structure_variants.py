"""Integration, derivative and long-horizon checks for the four XAJ arms."""

from __future__ import annotations

import math

import pytest
import torch
from models import (
    DE,
    DR,
    GE,
    GR,
    XAJDE,
    XAJDR,
    XAJGE,
    XAJGR,
    XAJDELite,
    XAJDRLite,
    XAJGELite,
    XAJGRLite,
    native_effective_kss,
    normalized_to_beta,
    normalized_to_gamma,
    normalized_to_kss,
    normalized_to_tau0,
)
from models.parameter_specs import (
    XAJ_DE_PARAM_SPECS,
    XAJ_DR_PARAM_SPECS,
    XAJ_GE_PARAM_SPECS,
    XAJ_GR_PARAM_SPECS,
)
from models.structure_response import _analytic_subsurface_response_step

E_CASES = (
    (XAJDE, XAJDELite, XAJ_DE_PARAM_SPECS),
    (XAJGE, XAJGELite, XAJ_GE_PARAM_SPECS),
)
R_CASES = (
    (XAJDR, XAJDRLite, XAJ_DR_PARAM_SPECS),
    (XAJGR, XAJGRLite, XAJ_GR_PARAM_SPECS),
)
ALL_CASES = E_CASES + R_CASES


def _forcing(steps: int = 64, dtype: torch.dtype = torch.float64):
    t = torch.arange(steps, dtype=dtype)
    precip = torch.where((t % 11) < 3, 5.0 + (t % 5), torch.zeros_like(t))
    precip = precip + torch.where((t % 37) == 0, 35.0, torch.zeros_like(t))
    pet = 1.0 + 2.0 * ((t % 9) / 8.0)
    temp = torch.where((t % 13) < 5, torch.full_like(t, -2.0), torch.full_like(t, 5.0))
    return {
        "precip": precip.unsqueeze(0),
        "pet": pet.unsqueeze(0),
        "temp": temp.unsqueeze(0),
    }


def _params(specs, dtype=torch.float64, batch=1, requires_grad=False):
    return {
        name: torch.full(
            (batch,), float(spec["default"]), dtype=dtype, requires_grad=requires_grad
        )
        for name, spec in specs.items()
    }


def _active_params(specs, dtype=torch.float64, requires_grad=False):
    p = _params(specs, dtype=dtype, requires_grad=requires_grad)
    if "xaj_gamma" in p:
        p["xaj_gamma"] = torch.full((1,), 2.0, dtype=dtype, requires_grad=requires_grad)
    if "xaj_beta" in p:
        p["xaj_beta"] = torch.full((1,), 1.3, dtype=dtype, requires_grad=requires_grad)
    if "xaj_tau0" in p:
        p["xaj_tau0"] = torch.full((1,), 12.0, dtype=dtype, requires_grad=requires_grad)
    if "xaj_kss" in p:
        p["xaj_kss"] = torch.full((1,), 0.55, dtype=dtype, requires_grad=requires_grad)
    # Isolate the XAJ water balance from the impervious direct-runoff branch.
    if "xaj_im" in p:
        p["xaj_im"] = torch.zeros((1,), dtype=dtype, requires_grad=requires_grad)
    return p


def _initial(model_cls, specs, dtype=torch.float64):
    p = _params(specs, dtype=dtype)
    state = {
        "wu": torch.full((1,), 0.55, dtype=dtype) * p["xaj_um"],
        "wl": torch.full((1,), 0.45, dtype=dtype) * p["xaj_lm"],
        "wd": torch.full((1,), 0.35, dtype=dtype) * p["xaj_dm"],
        "s": torch.full((1,), 0.30, dtype=dtype) * p["xaj_sm"],
        "fr": torch.full((1,), 0.25, dtype=dtype),
        "rs_uh_buffer": torch.zeros(1, 14, dtype=dtype),
    }
    if "R" in model_cls.__name__:
        state["z"] = torch.full((1,), 2.0, dtype=dtype)
    else:
        state["qi"] = torch.full((1,), 0.2, dtype=dtype)
        state["qg"] = torch.full((1,), 0.1, dtype=dtype)
    return state


@pytest.mark.parametrize("full_cls,lite_cls,specs", ALL_CASES)
def test_controlled_variant_eager_compiled_full_lite_and_gradients(
    full_cls, lite_cls, specs
):
    forcing = _forcing(48)
    params = _active_params(specs, requires_grad=True)
    initial = _initial(full_cls, specs)
    eager, aux = full_cls(compile_step=False)(
        forcing, params, initial_states=initial, return_states=True
    )
    compiled, _ = full_cls()(
        forcing, _active_params(specs), initial_states=initial, return_states=True
    )
    lite, lite_aux = lite_cls()(forcing, _active_params(specs), initial_states=initial)

    assert torch.isfinite(eager).all() and torch.isfinite(compiled).all()
    assert torch.allclose(eager, compiled, atol=2e-6, rtol=2e-6)
    assert torch.allclose(eager, lite, atol=2e-5, rtol=2e-5)
    assert lite_aux == {}
    assert aux["model_name"]

    eager.square().mean().backward()
    assert all(
        v.grad is not None and torch.isfinite(v.grad).all() for v in params.values()
    )
    active_name = next(
        (n for n in ("xaj_gamma", "xaj_beta", "xaj_tau0", "xaj_kss") if n in params),
        None,
    )
    if active_name is not None:
        assert abs(float(params[active_name].grad.item())) > 1e-10


def _balance_residual(model_cls, specs, steps=96):
    dtype = torch.float64
    forcing = _forcing(steps, dtype)
    params = _active_params(specs, dtype=dtype)
    initial = _initial(model_cls, specs, dtype)
    _, aux = model_cls(compile_step=False)(forcing, params, initial_states=initial)
    soil = aux["wu"] + aux["wl"] + aux["wd"]
    free = aux["fr"] * aux["s_next"]
    if "R" in model_cls.__name__:
        storage = soil + free + aux["z"]
        out = aux["rs_instant"] + aux["q_ss"]
    else:
        ci = params["xaj_ci"].unsqueeze(1)
        cg = params["xaj_cg"].unsqueeze(1)
        route = ci * aux["qi"] / (1.0 - ci) + cg * aux["qg"] / (1.0 - cg)
        storage = soil + free + route
        out = aux["rs_instant"] + aux["qi"] + aux["qg"]
    initial_soil = initial["wu"] + initial["wl"] + initial["wd"]
    initial_free = initial["fr"] * initial["s"]
    if "R" in model_cls.__name__:
        initial_storage = initial_soil + initial_free + initial["z"]
    else:
        initial_storage = (
            initial_soil
            + initial_free
            + params["xaj_ci"] * initial["qi"] / (1.0 - params["xaj_ci"])
            + params["xaj_cg"] * initial["qg"] / (1.0 - params["xaj_cg"])
        )
    delta = torch.cat(
        (
            storage[:, :1] - initial_storage.unsqueeze(1),
            storage[:, 1:] - storage[:, :-1],
        ),
        dim=1,
    )
    residual = forcing["precip"] - aux["evap_total"] - out - delta
    scale = torch.maximum(
        torch.ones_like(residual),
        forcing["precip"].cumsum(1) + out.cumsum(1) + storage.abs(),
    )
    return residual, residual.abs().max(), (residual.abs() / scale).max(), aux


@pytest.mark.parametrize("full_cls,lite_cls,specs", ALL_CASES)
def test_controlled_variant_water_balance_and_long_sequence(full_cls, lite_cls, specs):
    residual, max_abs, max_norm, aux = _balance_residual(full_cls, specs)
    assert torch.isfinite(residual).all()
    # Native XAJ's established balance test uses a 5e-7 float64 absolute
    # tolerance because its nearzero denominators are part of the host
    # convention.  The controlled arms must stay within that same inherited
    # accounting tolerance.
    assert float(max_norm) <= 3e-9
    assert float(max_abs) <= 5e-7
    for key in ("wu", "wl", "wd", "s_next", "fr"):
        assert torch.isfinite(aux[key]).all() and torch.all(aux[key] >= -1e-10)
    p_check = _params(specs, dtype=torch.float64)
    for key, cap in (
        ("wu", "xaj_um"),
        ("wl", "xaj_lm"),
        ("wd", "xaj_dm"),
        ("s_next", "xaj_sm"),
    ):
        assert torch.all(aux[key] <= p_check[cap].unsqueeze(1) + 1e-8)
    # A longer deterministic sequence exercises repeated depletion/recharge.
    long_forcing = _forcing(512)
    p = _active_params(specs)
    q, long_aux = full_cls(compile_step=False)(
        long_forcing, p, initial_states=_initial(full_cls, specs), return_states=True
    )
    assert torch.isfinite(q).all()
    assert all(torch.isfinite(v).all() for v in long_aux["final_states"].values())
    assert (
        min(float(long_aux[k].min()) for k in ("wu", "wl", "wd", "s_next", "fr"))
        >= -1e-10
    )
    if "R" in full_cls.__name__:
        assert float(long_aux["z"].min()) >= -1e-10


def test_controlled_float32_float64_forward_and_gradient_consistency():
    forcing32 = _forcing(96, torch.float32)
    forcing64 = _forcing(96, torch.float64)
    for full_cls, _lite_cls, specs in ALL_CASES:

        def run(dtype, forcing):
            params = _active_params(specs, dtype=dtype, requires_grad=True)
            initial = _initial(full_cls, specs, dtype)
            q, _ = full_cls(compile_step=False)(forcing, params, initial_states=initial)
            q.sum().backward()
            grad = torch.cat(
                [params[name].grad.reshape(-1).to(torch.float64) for name in specs]
            )
            return q.detach().to(torch.float64), grad

        q32, g32 = run(torch.float32, forcing32)
        q64, g64 = run(torch.float64, forcing64)
        cosine = torch.nn.functional.cosine_similarity(g32, g64, dim=0)
        median_relative = ((g32 - g64).abs() / g64.abs().clamp_min(1e-8)).median()
        assert torch.isfinite(q32).all() and torch.isfinite(q64).all()
        assert float((q32 - q64).abs().max()) <= 5e-5
        assert float(cosine) > 0.999
        assert float(median_relative) < 1e-3


def _central(fn, x: float, h: float = 1e-5):
    return (fn(x + h) - fn(x - h)) / (2.0 * h)


def test_module_autograd_matches_normalized_finite_difference():
    dtype = torch.float64
    # Interior points avoid both storage caps and the zero/floor branch.
    er = torch.tensor([3.0], dtype=dtype)
    wl = torch.tensor([32.0], dtype=dtype)
    wd = torch.tensor([18.0], dtype=dtype)
    lm = torch.tensor([80.0], dtype=dtype)
    dm = torch.tensor([40.0], dtype=dtype)
    n_gamma = torch.tensor([0.62], dtype=dtype, requires_grad=True)
    loss = GE(compile_step=False)(er, wl, wd, lm, dm, normalized_to_gamma(n_gamma))[
        0
    ].sum()
    loss.backward()
    fd_gamma = _central(
        lambda x: float(
            GE(compile_step=False)(
                er, wl, wd, lm, dm, normalized_to_gamma(torch.tensor([x], dtype=dtype))
            )[0]
        ),
        0.62,
    )
    assert abs(float(n_gamma.grad) - fd_gamma) / max(1.0, abs(fd_gamma)) < 1e-4

    r = torch.tensor([0.3], dtype=dtype)
    z = torch.tensor([2.0], dtype=dtype)
    n_beta = torch.tensor([0.57], dtype=dtype, requires_grad=True)
    tau = normalized_to_tau0(torch.tensor([0.55], dtype=dtype))
    loss = GR(compile_step=False)(r, z, tau, normalized_to_beta(n_beta))[0].sum()
    loss.backward()
    fd_beta = _central(
        lambda x: float(
            GR(compile_step=False)(
                r, z, tau, normalized_to_beta(torch.tensor([x], dtype=dtype))
            )[0]
        ),
        0.57,
    )
    assert abs(float(n_beta.grad) - fd_beta) / max(1.0, abs(fd_beta)) < 1e-4

    n_tau = torch.tensor([0.55], dtype=dtype, requires_grad=True)
    loss = DR(compile_step=False)(r, z, normalized_to_tau0(n_tau))[0].sum()
    loss.backward()
    fd_tau = _central(
        lambda x: float(
            DR(compile_step=False)(
                r, z, normalized_to_tau0(torch.tensor([x], dtype=dtype))
            )[0]
        ),
        0.55,
    )
    assert abs(float(n_tau.grad) - fd_tau) / max(1.0, abs(fd_tau)) < 1e-4


@pytest.mark.parametrize("gamma", [0.2, 0.5, 0.999, 1.0, 1.001, 2.0, 5.0])
def test_gamma_grid_is_finite_and_conservative(gamma):
    dtype = torch.float64
    x = torch.tensor(
        [0.0, 1e-8, 1e-6, 1e-4, 0.01, 0.5, 0.999, 1.0], dtype=dtype, requires_grad=True
    )
    wl = 80.0 * x
    wd = 40.0 * x
    g = torch.full_like(x, gamma, requires_grad=True)
    out = GE(compile_step=False)(
        torch.full_like(x, 3.0),
        wl,
        wd,
        torch.full_like(x, 80.0),
        torch.full_like(x, 40.0),
        g,
    )
    (out[0].sum() + out[1].sum()).backward()
    assert torch.isfinite(torch.stack(out[:5])).all()
    assert torch.isfinite(x.grad).all() and torch.isfinite(g.grad).all()
    assert torch.all(out[0] >= 0) and torch.all(out[1] >= 0)
    assert torch.all(out[0] <= wl) and torch.all(out[1] <= wd)
    assert out[0][0] == 0 and out[1][0] == 0


@pytest.mark.parametrize("beta", [0.5, 0.9, 0.999, 1.0, 1.001, 1.1, 2.0])
def test_beta_grid_and_z0_conditioning(beta):
    dtype = torch.float64
    for z0_value in (1e-3, 1.0, 1e3):
        z0 = torch.tensor([z0_value], dtype=dtype)
        z_raw = torch.tensor(
            [0.0, 1e-8, 1e-6, 1e-3, 0.1, 1.0, 10.0, 100.0],
            dtype=dtype,
            requires_grad=True,
        )
        z = z_raw * z0_value
        tau = torch.tensor(
            [0.5, 1.0, 10.0, 10.0, 10.0, 22.0, 100.0, 1000.0],
            dtype=dtype,
            requires_grad=True,
        )
        b = torch.full_like(z, beta, requires_grad=True)
        out = GR(nearzero=1e-12, z0=z0, compile_step=False)(
            torch.zeros_like(z), z, tau, b
        )
        (out[0].sum() + out[1].sum()).backward()
        assert torch.isfinite(torch.stack(out)).all()
        assert (
            torch.isfinite(z_raw.grad).all()
            and torch.isfinite(tau.grad).all()
            and torch.isfinite(b.grad).all()
        )
        assert out[0][0] == 0 and out[1][0] == 0
        assert torch.all(out[0] >= 0) and torch.all(out[0] <= z)
        assert torch.all(out[1] >= 0)
        assert torch.isfinite(
            torch.log(torch.clamp(z.detach() / z0_value, min=1e-30))
        ).all()


def test_native_kss_mapping_and_response_input():
    ki = torch.tensor([0.0, 0.3, 0.7, 0.7], dtype=torch.float64)
    kg = torch.tensor([0.0, 0.2, 0.7, 0.7], dtype=torch.float64)
    effective = native_effective_kss(ki, kg)
    assert torch.allclose(
        effective, torch.tensor([0.0, 0.5, 0.99999, 0.99999], dtype=torch.float64)
    )
    assert torch.allclose(
        normalized_to_kss(torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)),
        torch.tensor([0.0, 0.499995, 0.99999], dtype=torch.float64),
    )


@pytest.mark.parametrize("full_cls,lite_cls,specs", ALL_CASES)
def test_controlled_variant_parameter_and_state_boundary_sweep(
    full_cls, lite_cls, specs
):
    dtype = torch.float64
    forcing = _forcing(12, dtype)
    boundary_name = next(
        (n for n in ("xaj_gamma", "xaj_beta", "xaj_tau0", "xaj_kss") if n in specs),
        None,
    )
    boundary_values = (0.0, 1e-6, 0.5, 1.0 - 1e-6, 1.0)
    for normalized in boundary_values:
        p = _params(specs, dtype=dtype)
        p["xaj_im"] = torch.zeros(1, dtype=dtype)
        if boundary_name == "xaj_gamma":
            p[boundary_name] = normalized_to_gamma(
                torch.tensor([normalized], dtype=dtype)
            )
        elif boundary_name == "xaj_beta":
            p[boundary_name] = normalized_to_beta(
                torch.tensor([normalized], dtype=dtype)
            )
        elif boundary_name == "xaj_tau0":
            p[boundary_name] = normalized_to_tau0(
                torch.tensor([normalized], dtype=dtype)
            )
        elif boundary_name == "xaj_kss":
            p[boundary_name] = normalized_to_kss(
                torch.tensor([normalized], dtype=dtype)
            )
        for state_mode in ("zero", "full"):
            initial = _initial(full_cls, specs, dtype)
            if state_mode == "zero":
                for key in ("wu", "wl", "wd", "s", "fr", "qi", "qg", "z"):
                    if key in initial:
                        initial[key] = torch.zeros_like(initial[key])
            else:
                initial["wu"] = p["xaj_um"].clone()
                initial["wl"] = p["xaj_lm"].clone()
                initial["wd"] = p["xaj_dm"].clone()
                initial["s"] = p["xaj_sm"].clone()
            _q, aux = full_cls(compile_step=False)(forcing, p, initial_states=initial)
            assert torch.isfinite(_q).all()
            for key in ("wu", "wl", "wd", "s_next", "fr"):
                assert torch.isfinite(aux[key]).all() and torch.all(aux[key] >= -1e-10)
            for key, cap in (
                ("wu", "xaj_um"),
                ("wl", "xaj_lm"),
                ("wd", "xaj_dm"),
                ("s_next", "xaj_sm"),
            ):
                assert torch.all(aux[key] <= p[cap].unsqueeze(1) + 1e-8)
            if "R" in full_cls.__name__:
                assert torch.all(aux["z"] >= -1e-10)
                assert torch.all(aux["q_ss"] >= -1e-10)
            else:
                er = torch.clamp(
                    forcing["pet"] * p["xaj_k"].unsqueeze(1) - aux["eu"], min=0.0
                )
                assert torch.all(aux["el"] + aux["ed"] <= er + 2e-10)


@pytest.mark.parametrize(
    "full_cls,specs", ((XAJDR, XAJ_DR_PARAM_SPECS), (XAJGR, XAJ_GR_PARAM_SPECS))
)
def test_controlled_kss_normalized_finite_difference(full_cls, specs):
    dtype = torch.float64
    forcing = _forcing(32, dtype)
    base = _active_params(specs, dtype)
    n_kss = torch.tensor([0.57], dtype=dtype, requires_grad=True)
    params = {name: value.clone() for name, value in base.items()}
    params["xaj_kss"] = normalized_to_kss(n_kss)
    loss = full_cls(compile_step=False)(
        forcing, params, initial_states=_initial(full_cls, specs, dtype)
    )[0].sum()
    loss.backward()

    def evaluate(normalized):
        candidate = {name: value.clone() for name, value in base.items()}
        candidate["xaj_kss"] = normalized_to_kss(
            torch.tensor([normalized], dtype=dtype)
        )
        return float(
            full_cls(compile_step=False)(
                forcing, candidate, initial_states=_initial(full_cls, specs, dtype)
            )[0].sum()
        )

    fd = (evaluate(0.57 + 1e-5) - evaluate(0.57 - 1e-5)) / (2e-5)
    assert abs(float(n_kss.grad) - fd) / max(1.0, abs(fd)) < 1e-4


def test_beta_one_is_d_r_and_extinction_is_conservative():
    dtype = torch.float64
    r = torch.zeros(3, dtype=dtype)
    z = torch.tensor([0.0, 1e-8, 0.01], dtype=dtype)
    tau = torch.full((3,), 10.0, dtype=dtype)
    one = torch.ones(3, dtype=dtype)
    d = DR(compile_step=False)(r, z, tau)
    g = GR(compile_step=False)(r, z, tau, one)
    assert torch.equal(d[0], g[0]) and torch.equal(d[1], g[1])
    # beta<1 and y=0 is finite-time exhaustion, not a negative storage.
    q, zn, za, *_ = _analytic_subsurface_response_step(
        torch.zeros(1, dtype=dtype),
        torch.zeros(1, dtype=dtype),
        torch.full((1,), 0.5, dtype=dtype),
        torch.full((1,), 0.5, dtype=dtype),
        torch.ones(1, dtype=dtype),
        1e-12,
    )
    assert q.item() == 0.0 and zn.item() == 0.0 and za.item() == 0.0
