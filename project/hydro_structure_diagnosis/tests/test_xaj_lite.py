"""Regression tests for the streamflow-only XAJ execution path."""

import torch

from models import XAJLite
from models.parameter_specs import XAJ_LITE_PARAM_SPECS


def _forcing(batch=2, steps=24, dtype=torch.float32):
    torch.manual_seed(123)
    return {
        "precip": torch.rand(batch, steps, dtype=dtype) * 8.0,
        "pet": torch.rand(batch, steps, dtype=dtype) * 4.0,
        "temp": torch.rand(batch, steps, dtype=dtype) * 20.0 - 5.0,
    }


def _params(batch=2, dtype=torch.float32, requires_grad=False):
    values = {}
    for name, spec in XAJ_LITE_PARAM_SPECS.items():
        values[name] = torch.full(
            (batch,), float(spec["default"]), dtype=dtype,
            requires_grad=requires_grad,
        )
    return values


def test_xaj_lite_compact_and_diagnostic_paths_match():
    forcing = _forcing()
    params = _params()
    with torch.no_grad():
        model = XAJLite()
        q_diagnostic, aux_diagnostic = model(
            forcing, params, return_states=True,
        )
        q_lite, aux_lite = model(forcing, params)

    assert torch.allclose(q_lite, q_diagnostic, atol=2e-6, rtol=2e-6)
    assert {"evap", "rs_instant", "qi", "qg"} <= set(aux_diagnostic)
    assert "final_states" in aux_diagnostic
    assert aux_lite == {}


def test_xaj_lite_gradients_are_finite_and_stateful_diagnostics_remain_available():
    forcing = _forcing()
    params = _params(requires_grad=True)
    model = XAJLite()
    diagnostic_model = XAJLite()
    qsim, _ = model(forcing, params)
    loss = qsim.square().mean()
    reference_params = _params(requires_grad=True)
    q_reference, _ = diagnostic_model(
        forcing, reference_params, return_states=True,
    )
    reference_loss = q_reference.square().mean()
    loss.backward()
    reference_loss.backward()

    assert torch.isfinite(qsim).all()
    assert torch.allclose(qsim, q_reference, atol=2e-6, rtol=2e-6)
    assert all(value.grad is not None for value in params.values())
    assert all(torch.isfinite(value.grad).all() for value in params.values())
    for name in params:
        assert torch.allclose(
            params[name].grad, reference_params[name].grad,
            atol=3e-5, rtol=3e-5,
        )

    with torch.no_grad():
        _, diagnostic_aux = model(forcing, _params(), return_states=True)
    assert "evap" in diagnostic_aux
    assert "final_states" in diagnostic_aux


def test_xaj_lite_chunked_run_matches_full_run_with_hydrodl2_uh_buffer():
    torch.manual_seed(5)
    model = XAJLite()
    forcing = _forcing(batch=1, steps=40)
    params = _params(batch=1)
    with torch.no_grad():
        q_full, _ = model(forcing, params)
        _, first_aux = model(
            {key: value[:, :20] for key, value in forcing.items()},
            params,
            return_states=True,
        )
        q_second, _ = model(
            {key: value[:, 20:] for key, value in forcing.items()},
            params,
            initial_states=first_aux["final_states"],
        )
    assert torch.allclose(q_full[:, 20:], q_second, atol=2e-6, rtol=2e-6)


def test_xaj_lite_hbv_routing_bounds_have_finite_gradients():
    forcing = _forcing(batch=2, steps=18)
    params = {
        name: torch.tensor(
            [float(spec["lower"]), float(spec["upper"])],
            dtype=torch.float32,
            requires_grad=True,
        )
        for name, spec in XAJ_LITE_PARAM_SPECS.items()
    }
    qsim, _ = XAJLite()(forcing, params)
    qsim.square().mean().backward()
    assert torch.isfinite(qsim).all()
    assert all(value.grad is not None for value in params.values())
    assert all(torch.isfinite(value.grad).all() for value in params.values())
