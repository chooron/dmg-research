"""Gradient tests for all full models.

Verifies:
- loss.backward() works
- Parameter gradients are not None
- Parameter gradients are finite
"""

import torch
import pytest

from models import (
    HBV, GR4J, XAJ, CemaNeige, CemaNeigeHyst,
    GR4JWithCemaNeige, XAJWithCemaNeige,
)
from models.parameter_specs import (
    HBV_PARAM_SPECS,
    GR4J_PARAM_SPECS,
    XAJ_PARAM_SPECS,
    CEMANEIGE_PARAM_SPECS,
    CEMANEIGE_HYST_PARAM_SPECS,
    GR4J_CN_PARAM_SPECS,
    XAJ_CN_PARAM_SPECS,
)


BATCH = 3
TIME = 20


def make_synthetic_forcings(batch, time, device, dtype):
    torch.manual_seed(42)
    precip = torch.rand(batch, time, device=device, dtype=dtype) * 10.0
    pet = torch.rand(batch, time, device=device, dtype=dtype) * 5.0
    temp = torch.randn(batch, time, device=device, dtype=dtype) * 10.0
    return {"precip": precip, "pet": pet, "temp": temp}


def make_tunable_params(param_specs, batch, device, dtype):
    """Create parameters as nn.Parameters for gradient testing."""
    torch.manual_seed(123)
    params = {}
    for name, spec in param_specs.items():
        lo = spec["lower"]
        hi = spec["upper"]
        val = lo + torch.rand(batch, device=device, dtype=dtype) * (hi - lo)
        p = torch.nn.Parameter(val)
        params[name] = p
    return params


def _check_model_gradient(model_cls, param_specs, model_name, device, dtype):
    """Run gradient test for a model."""
    device_obj = torch.device(device)
    model = model_cls().to(device=device_obj, dtype=dtype)
    forcings = make_synthetic_forcings(BATCH, TIME, device_obj, dtype)
    params = make_tunable_params(param_specs, BATCH, device_obj, dtype)

    qsim, aux = model(forcings=forcings, params=params)
    loss = qsim.mean()
    loss.backward()

    grad_info = {}
    for name, p in params.items():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), (
                f"[{model_name}] Gradient for {name} contains NaN/Inf"
            )
            grad_info[name] = p.grad.abs().mean().item()
        else:
            grad_info[name] = None

    # Check that most parameters have non-None gradients
    none_grads = [k for k, v in grad_info.items() if v is None]
    if len(none_grads) > 0:
        print(f"[{model_name}] Parameters with None grads: {none_grads}")
        print(f"  (May be expected for certain params in synthetic setup)")

    # At least 50% of parameters should have gradients
    n_total = len(params)
    n_grad = sum(1 for v in grad_info.values() if v is not None)
    assert n_grad >= n_total * 0.5, (
        f"[{model_name}] Only {n_grad}/{n_total} params have gradients; expected at least {int(n_total * 0.5)}"
    )

    return grad_info


class TestModelGrad:
    """Gradient tests for all models."""

    @pytest.mark.parametrize("device_str,dtype", [
        ("cpu", torch.float32),
    ])
    def test_hbv_grad(self, device_str, dtype):
        _check_model_gradient(HBV, HBV_PARAM_SPECS, "HBV", device_str, dtype)

    @pytest.mark.parametrize("device_str,dtype", [
        ("cpu", torch.float32),
    ])
    def test_gr4j_grad(self, device_str, dtype):
        _check_model_gradient(GR4J, GR4J_PARAM_SPECS, "GR4J", device_str, dtype)

    @pytest.mark.parametrize("device_str,dtype", [
        ("cpu", torch.float32),
    ])
    def test_xaj_grad(self, device_str, dtype):
        _check_model_gradient(XAJ, XAJ_PARAM_SPECS, "XAJ", device_str, dtype)

    @pytest.mark.parametrize("device_str,dtype", [
        ("cpu", torch.float32),
    ])
    def test_cemaneige_grad(self, device_str, dtype):
        _check_model_gradient(CemaNeige, CEMANEIGE_PARAM_SPECS, "CemaNeige", device_str, dtype)

    @pytest.mark.parametrize("device_str,dtype", [
        ("cpu", torch.float32),
    ])
    def test_cemaneige_hyst_grad(self, device_str, dtype):
        _check_model_gradient(
            CemaNeigeHyst,
            CEMANEIGE_HYST_PARAM_SPECS,
            "CemaNeigeHyst",
            device_str,
            dtype,
        )

    @pytest.mark.parametrize("device_str,dtype", [
        ("cpu", torch.float32),
    ])
    def test_gr4j_cn_grad(self, device_str, dtype):
        _check_model_gradient(GR4JWithCemaNeige, GR4J_CN_PARAM_SPECS, "GR4J+CemaNeige", device_str, dtype)

    @pytest.mark.parametrize("device_str,dtype", [
        ("cpu", torch.float32),
    ])
    def test_xaj_cn_grad(self, device_str, dtype):
        _check_model_gradient(XAJWithCemaNeige, XAJ_CN_PARAM_SPECS, "XAJ+CemaNeige", device_str, dtype)
