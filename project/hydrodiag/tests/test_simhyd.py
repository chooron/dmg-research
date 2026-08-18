"""Numerical-integrity tests for SIMHYD compositions."""

from __future__ import annotations

import pytest
import torch

# This module deliberately exercises CPU/CUDA, float32/float64, grad/no-grad
# variants of the compiled daily kernel in one process.
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.recompile_limit = 64

from models import SIMHYD, SIMHYDWithCemaNeige, SIMHYDWithPrecipitationDelay
from models.parameter_specs import (
    SIMHYD_CN_PARAM_SPECS,
    SIMHYD_PARAM_SPECS,
    SIMHYD_PD_PARAM_SPECS,
)
from models.simhyd import SIMHYD_UH_MAX_LEN, _simhyd_step


def _devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _params(specs: dict, batch: int, device: str, dtype: torch.dtype) -> dict:
    return {
        name: torch.full((batch,), float(spec["default"]), device=device, dtype=dtype)
        for name, spec in specs.items()
    }


def _forcing(batch: int, time: int, device: str, dtype: torch.dtype) -> dict:
    generator = torch.Generator(device=device).manual_seed(20260720)
    precip = (
        torch.rand(batch, time, generator=generator, device=device, dtype=dtype) * 12.0
    )
    pet = torch.rand(batch, time, generator=generator, device=device, dtype=dtype) * 5.0
    # Alternating cold/warm blocks exercise both CemaNeige accumulation and melt.
    phase = torch.arange(time, device=device, dtype=dtype)
    temp = (8.0 * torch.sin(phase * 0.11) - 1.0).expand(batch, -1).clone()
    return {"precip": precip, "pet": pet, "temp": temp}


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_simhyd_forward_is_finite_and_nonnegative(device: str, dtype: torch.dtype):
    batch, time = 3, 64
    model = SIMHYD().to(device=device, dtype=dtype)
    qsim, aux = model(
        forcings=_forcing(batch, time, device, dtype),
        params=_params(SIMHYD_PARAM_SPECS, batch, device, dtype),
        return_states=True,
    )
    assert qsim.shape == (batch, time)
    assert torch.isfinite(qsim).all()
    assert (qsim >= 0.0).all()
    assert torch.isfinite(aux["evap"]).all()
    assert aux["routing_method"] == "gamma"
    assert aux["gamma_uh_ordinates"].shape == (batch, SIMHYD_UH_MAX_LEN)
    assert torch.allclose(
        aux["gamma_uh_ordinates"].sum(dim=-1),
        torch.ones(batch, device=device, dtype=dtype),
        atol=1e-6 if dtype == torch.float32 else 1e-12,
        rtol=0.0,
    )
    assert set(aux["final_states"]) == {"soil", "groundwater", "runoff_uh_buffer"}


def test_simhyd_exposes_gamma_parameters_only():
    """There is no alternative routing parameter set or routing branch."""
    names = set(SIMHYD_PARAM_SPECS)
    assert {"simhyd_a", "simhyd_theta"} <= names
    assert not {"simhyd_delay", "simhyd_x_m", "DELAY", "X_m"} & names
    assert SIMHYD.routing_method == "gamma"
    assert SIMHYDWithCemaNeige.routing_method == "gamma"


@pytest.mark.parametrize(
    "model_cls,specs",
    [
        (SIMHYD, SIMHYD_PARAM_SPECS),
        (SIMHYDWithCemaNeige, SIMHYD_CN_PARAM_SPECS),
        (SIMHYDWithPrecipitationDelay, SIMHYD_PD_PARAM_SPECS),
    ],
)
@pytest.mark.parametrize("device", _devices())
def test_simhyd_boundary_values_are_finite_and_gradient_safe(
    model_cls, specs, device: str
):
    """Audit exact bounds plus the formerly unstable zero-coeff corner."""
    dtype = torch.float32
    names = list(specs)
    defaults = [float(spec["default"]) for spec in specs.values()]
    rows = [
        [float(spec["lower"]) for spec in specs.values()],
        [float(spec["upper"]) for spec in specs.values()],
    ]
    for index, spec in enumerate(specs.values()):
        for edge in ("lower", "upper"):
            row = defaults.copy()
            row[index] = float(spec[edge])
            rows.append(row)

    # This exact combination previously produced a ~7e33 coeff gradient.
    pathological = defaults.copy()
    edge_values = {
        "simhyd_insc": 0.0,
        "simhyd_coeff": 0.0,  # also tests runtime protection outside specs
        "simhyd_sq": 10.0,
        "simhyd_smsc": 1.0,
        "simhyd_sub": 0.0,
        "simhyd_crak": 1.0,
        "simhyd_k": 1.0,
        "simhyd_etmul": 3.0,
        "simhyd_a": 10.0,
        "simhyd_theta": 0.5,
        "cn_thacc": 0.0,  # runtime protection for direct callers
    }
    for name, value in edge_values.items():
        if name in names:
            pathological[names.index(name)] = value
    rows.append(pathological)

    values = torch.tensor(rows, device=device, dtype=dtype)
    batch, time = values.shape[0], 64
    phase = torch.arange(time, device=device, dtype=dtype)
    precip = torch.clamp(4.0 + 4.0 * torch.sin(phase * 0.41), min=0.0)
    precip = precip.expand(batch, -1).clone()
    precip[:, ::11] = 25.0
    pet = torch.clamp(2.5 + 2.0 * torch.cos(phase * 0.17), min=0.0)
    pet = pet.expand(batch, -1).clone()
    temp = (9.0 * torch.sin(phase * 0.13) - 1.0).expand(batch, -1).clone()
    params = {
        name: torch.nn.Parameter(values[:, index].clone())
        for index, name in enumerate(names)
    }

    qsim, aux = model_cls().to(device=device, dtype=dtype)(
        {"precip": precip, "pet": pet, "temp": temp}, params, return_states=True
    )
    tensor_outputs = [qsim] + [
        value for value in aux.values() if isinstance(value, torch.Tensor)
    ]
    assert all(torch.isfinite(value).all() for value in tensor_outputs)

    loss = (
        qsim.square().mean()
        + 0.01 * aux["evap"].mean()
        + 1e-5 * aux["routing_storage"].mean()
    )
    loss.backward()
    for name, parameter in params.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().max() < 1e6, name


@pytest.mark.parametrize(
    "model_cls,specs",
    [
        (SIMHYD, SIMHYD_PARAM_SPECS),
        (SIMHYDWithCemaNeige, SIMHYD_CN_PARAM_SPECS),
        (SIMHYDWithPrecipitationDelay, SIMHYD_PD_PARAM_SPECS),
    ],
)
@pytest.mark.parametrize("device", _devices())
def test_simhyd_full_system_water_balance(model_cls, specs, device: str):
    """P - ET - Q equals the change in all physical and routing stores."""
    dtype = torch.float64
    batch, time = 2, 180
    forcings = _forcing(batch, time, device, dtype)
    params = _params(specs, batch, device, dtype)
    model = model_cls().to(device=device, dtype=dtype)

    qsim, aux = model(forcings=forcings, params=params, return_states=True)
    initial_storage = 0.5 * params["simhyd_smsc"]
    final_storage = aux["soil"] + aux["groundwater"] + aux["routing_storage"]
    if model_cls is SIMHYDWithCemaNeige:
        final_storage = final_storage + aux["final_states"]["cn_G"]
    elif model_cls is SIMHYDWithPrecipitationDelay:
        final_storage = final_storage + aux["final_states"]["pd_S"]

    residual = (
        forcings["precip"].sum(dim=1)
        - aux["evap"].sum(dim=1)
        - qsim.sum(dim=1)
        - (final_storage - initial_storage)
    )
    assert torch.allclose(residual, torch.zeros_like(residual), atol=2e-8, rtol=0.0), (
        residual
    )


@pytest.mark.parametrize(
    "model_cls,specs",
    [
        (SIMHYD, SIMHYD_PARAM_SPECS),
        (SIMHYDWithCemaNeige, SIMHYD_CN_PARAM_SPECS),
        (SIMHYDWithPrecipitationDelay, SIMHYD_PD_PARAM_SPECS),
    ],
)
@pytest.mark.parametrize("device", _devices())
def test_simhyd_parameter_gradients_are_finite(model_cls, specs, device: str):
    dtype = torch.float32
    batch, time = 2, 96
    forcings = _forcing(batch, time, device, dtype)
    values = _params(specs, batch, device, dtype)
    # Put infiltration capacity inside the rainfall range so coeff and sq are
    # active rather than identically masked by min(infiltration, rainfall).
    values["simhyd_coeff"][:] = 8.0
    values["simhyd_smsc"][:] = 100.0
    params = {name: torch.nn.Parameter(value.clone()) for name, value in values.items()}
    model = model_cls().to(device=device, dtype=dtype)

    qsim, aux = model(forcings=forcings, params=params)
    loss = qsim.square().mean() + 0.01 * aux["evap"].mean()
    loss.backward()

    for name, parameter in params.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("device", _devices())
def test_simhyd_chunked_run_matches_full_run(device: str):
    dtype = torch.float32
    forcings = _forcing(2, 140, device, dtype)
    params = _params(SIMHYD_PARAM_SPECS, 2, device, dtype)
    model = SIMHYD().to(device=device, dtype=dtype)

    with torch.no_grad():
        q_full, _ = model(forcings=forcings, params=params)
        _, first_aux = model(
            forcings={key: value[:, :70] for key, value in forcings.items()},
            params=params,
            return_states=True,
        )
        q_second, _ = model(
            forcings={key: value[:, 70:] for key, value in forcings.items()},
            params=params,
            initial_states=first_aux["final_states"],
        )
    assert torch.allclose(q_full[:, 70:], q_second, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("device", _devices())
def test_simhyd_step_compiles_fullgraph(device: str):
    dtype = torch.float32
    batch = 3
    params = _params(SIMHYD_PARAM_SPECS, batch, device, dtype)
    inputs = (
        torch.full((batch,), 9.0, device=device, dtype=dtype),
        torch.full((batch,), 3.0, device=device, dtype=dtype),
        torch.full((batch,), 80.0, device=device, dtype=dtype),
        torch.full((batch,), 20.0, device=device, dtype=dtype),
        params["simhyd_insc"],
        params["simhyd_coeff"],
        params["simhyd_sq"],
        params["simhyd_smsc"],
        params["simhyd_sub"],
        params["simhyd_crak"],
        params["simhyd_k"],
        params["simhyd_etmul"],
        1e-8,
    )
    compiled = torch.compile(_simhyd_step, fullgraph=True)
    with torch.no_grad():
        eager_out = _simhyd_step(*inputs)
        compiled_out = compiled(*inputs)
    for eager, actual in zip(eager_out, compiled_out):
        assert torch.allclose(eager, actual, atol=1e-5, rtol=1e-5)
