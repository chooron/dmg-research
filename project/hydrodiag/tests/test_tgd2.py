"""Invariants and optimizer-interface tests for the active two-parameter TGD2."""

from __future__ import annotations

import pytest
import torch
from ablation.ic_core.model_adapter import ModelAdapter
from ablation.ic_core.parameter_adapter import (
    normalized_to_physical,
    physical_to_normalized,
)
from models import (
    XAJ,
    TemperatureDependentGenericDelay2,
    XAJWithTGD2,
)
from models.parameter_specs import (
    TGD2_PARAM_SPECS,
    XAJ_PARAM_SPECS,
    XAJ_TGD2_PARAM_SPECS,
)


def _forcings(batch: int = 2, steps: int = 40, dtype: torch.dtype = torch.float64):
    torch.manual_seed(730)
    return {
        "precip": torch.rand(batch, steps, dtype=dtype) * 8.0,
        "pet": torch.rand(batch, steps, dtype=dtype) * 3.0,
        "temp": torch.linspace(-12.0, 12.0, steps, dtype=dtype).repeat(batch, 1),
    }


def _params(specs, batch: int = 2, dtype: torch.dtype = torch.float64):
    return {
        name: torch.full((batch,), spec["default"], dtype=dtype)
        for name, spec in specs.items()
    }


def test_tgd2_mass_conservation_long_series_and_finite_bounds():
    forcing = _forcings(3, 1000)
    params = _params(TGD2_PARAM_SPECS, 3)
    model = TemperatureDependentGenericDelay2()
    effective, aux = model(forcing, params, return_states=True)
    previous = torch.cat(
        (torch.zeros_like(aux["tgd2_storage"][:, :1]), aux["tgd2_storage"][:, :-1]),
        dim=1,
    )
    assert torch.allclose(
        previous + forcing["precip"],
        effective + aux["tgd2_storage"],
        atol=2e-12,
        rtol=2e-12,
    )
    assert torch.allclose(
        forcing["precip"].sum(1),
        effective.sum(1) + aux["final_states"]["storage"],
        atol=2e-11,
        rtol=2e-12,
    )
    assert torch.isfinite(effective).all() and torch.isfinite(aux["tgd2_storage"]).all()
    assert (aux["tgd2_storage"] >= 0).all()
    for name, spec in TGD2_PARAM_SPECS.items():
        edge = {
            key: torch.full((3,), value["lower"], dtype=torch.float64)
            for key, value in TGD2_PARAM_SPECS.items()
        }
        edge[name].fill_(spec["upper"])
        edge_out, edge_aux = model(forcing, edge, return_states=True)
        assert (
            torch.isfinite(edge_out).all()
            and torch.isfinite(edge_aux["tgd2_tau"]).all()
        )


def test_tgd2_temperature_and_residence_time_behavior():
    model = TemperatureDependentGenericDelay2()
    impulse = torch.zeros(1, 8, dtype=torch.float64)
    impulse[:, 0] = 10.0
    pet = torch.zeros_like(impulse)
    params = {
        "tgd_tau_warm": torch.tensor([0.2], dtype=torch.float64),
        "tgd_delta_tau_cold": torch.tensor([30.0], dtype=torch.float64),
    }
    warm, warm_aux = model(
        {"precip": impulse, "pet": pet, "temp": torch.full_like(impulse, 12.0)},
        params,
        return_states=True,
    )
    cold, cold_aux = model(
        {"precip": impulse, "pet": pet, "temp": torch.full_like(impulse, -12.0)},
        params,
        return_states=True,
    )
    assert warm[0, 0] > cold[0, 0]
    assert cold[0, 0] > 0.0  # a cold day leaks; this is not an explicit snow store.
    assert cold_aux["tgd2_tau"][0, 0] > warm_aux["tgd2_tau"][0, 0]
    larger_delta = {
        **params,
        "tgd_delta_tau_cold": torch.tensor([90.0], dtype=torch.float64),
    }
    cold_large, _ = model(
        {"precip": impulse, "pet": pet, "temp": torch.full_like(impulse, -12.0)},
        larger_delta,
        return_states=True,
    )
    assert cold_large[0, 0] < cold[0, 0]
    warmed = torch.cat(
        (
            torch.full((1, 3), -12.0, dtype=torch.float64),
            torch.full((1, 5), 12.0, dtype=torch.float64),
        ),
        dim=1,
    )
    release, aux = model(
        {"precip": impulse, "pet": pet, "temp": warmed}, params, return_states=True
    )
    assert aux["tgd2_retention"][0, 3] < aux["tgd2_retention"][0, 2]
    assert release[0, 3] > release[0, 2]


def test_tgd2_zero_precip_batch_equivalence_gradients_and_base_limit():
    model = TemperatureDependentGenericDelay2()
    zero = _forcings(2, 20)
    zero["precip"].zero_()
    output, aux = model(zero, _params(TGD2_PARAM_SPECS), return_states=True)
    assert torch.equal(output, torch.zeros_like(output)) and torch.equal(
        aux["tgd2_storage"], torch.zeros_like(output)
    )
    forcing = _forcings(2, 20)
    batched, _ = model(forcing, _params(TGD2_PARAM_SPECS), return_states=True)
    for index in range(2):
        single_forcing = {
            key: value[index : index + 1] for key, value in forcing.items()
        }
        single_params = {
            key: value[index : index + 1]
            for key, value in _params(TGD2_PARAM_SPECS).items()
        }
        single, _ = model(single_forcing, single_params, return_states=True)
        assert torch.allclose(batched[index : index + 1], single)
    grad_params = {
        name: torch.tensor([spec["default"]], dtype=torch.float64, requires_grad=True)
        for name, spec in TGD2_PARAM_SPECS.items()
    }
    grad_out, _ = model({key: value[:1] for key, value in forcing.items()}, grad_params)
    grad_out[:, 1:].square().mean().backward()
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in grad_params.values()
    )
    near_base = {
        name: torch.tensor([spec["lower"]], dtype=torch.float64)
        for name, spec in TGD2_PARAM_SPECS.items()
    }
    effective, _ = model({key: value[:1] for key, value in forcing.items()}, near_base)
    assert torch.allclose(effective, forcing["precip"][:1], atol=5e-4, rtol=5e-4)


def test_xaj_tgd2_registry_and_log_parameter_mapping():
    model_key, model_cls, specs = "XAJ_TGD2", XAJWithTGD2, XAJ_TGD2_PARAM_SPECS
    assert len(specs) == len(XAJ_PARAM_SPECS) + 2
    assert tuple(specs)[:2] == ("tgd_tau_warm", "tgd_delta_tau_cold")
    theta = torch.linspace(0, 1, len(specs), dtype=torch.float64).unsqueeze(0)
    physical = normalized_to_physical(model_key, theta)
    assert torch.allclose(physical_to_normalized(model_key, physical), theta)
    forcing = _forcings(1, 16)
    params = {name: physical[:, i] for i, name in enumerate(specs)}
    qsim, _ = model_cls()(forcing, params)
    assert qsim.shape == (1, 16) and torch.isfinite(qsim).all()
    adapter = ModelAdapter(model_key, dtype=torch.float32, variant="lite")
    adapter_q, _ = adapter.run_model(
        torch.stack(
            (
                forcing["precip"][0].float(),
                forcing["temp"][0].float(),
                forcing["pet"][0].float(),
            ),
            dim=-1,
        ),
        physical.float(),
    )
    assert torch.isfinite(adapter_q).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tgd2_cpu_cuda_consistency():
    forcing = _forcings(2, 30, torch.float32)
    params = _params(TGD2_PARAM_SPECS, 2, torch.float32)
    cpu, _ = TemperatureDependentGenericDelay2()(forcing, params)
    gpu_forcing = {key: value.cuda() for key, value in forcing.items()}
    gpu_params = {key: value.cuda() for key, value in params.items()}
    gpu, _ = TemperatureDependentGenericDelay2().cuda()(gpu_forcing, gpu_params)
    assert torch.allclose(cpu, gpu.cpu(), atol=2e-6, rtol=2e-6)
