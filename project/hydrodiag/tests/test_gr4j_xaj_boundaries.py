"""Boundary-value numerical safety for GR4J/XAJ and snow compositions."""

from __future__ import annotations

import pytest
import torch
from models import (
    GR4J,
    XAJ,
    GR4JWithCemaNeige,
    GR4JWithPrecipitationDelay,
    XAJWithCemaNeige,
    XAJWithPrecipitationDelay,
)
from models.parameter_specs import (
    GR4J_CN_PARAM_SPECS,
    GR4J_PARAM_SPECS,
    GR4J_PD_PARAM_SPECS,
    XAJ_CN_PARAM_SPECS,
    XAJ_PARAM_SPECS,
    XAJ_PD_PARAM_SPECS,
)

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.recompile_limit = 64


MODEL_CASES = [
    (GR4J, GR4J_PARAM_SPECS),
    (GR4JWithCemaNeige, GR4J_CN_PARAM_SPECS),
    (GR4JWithPrecipitationDelay, GR4J_PD_PARAM_SPECS),
    (XAJ, XAJ_PARAM_SPECS),
    (XAJWithCemaNeige, XAJ_CN_PARAM_SPECS),
    (XAJWithPrecipitationDelay, XAJ_PD_PARAM_SPECS),
]

DEVICE_DTYPES = [("cpu", torch.float64)]
if torch.cuda.is_available():
    DEVICE_DTYPES.append(("cuda", torch.float32))


@pytest.fixture(scope="module", autouse=True)
def _isolate_boundary_compile_cache():
    """Do not leak stress-test compile variants into the regular test suite."""
    yield
    torch._dynamo.reset()


def _flatten_tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        tensors = []
        for child in value.values():
            tensors.extend(_flatten_tensors(child))
        return tensors
    return []


def _boundary_rows(specs: dict) -> tuple[list[str], list[list[float]]]:
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

    # Explicit regression for the former XAJ inactive-branch division by zero.
    if "xaj_ki" in names:
        row = defaults.copy()
        row[names.index("xaj_ki")] = 0.0
        row[names.index("xaj_kg")] = 0.0
        rows.append(row)
    return names, rows


@pytest.mark.parametrize("model_cls,specs", MODEL_CASES)
@pytest.mark.parametrize("device,dtype", DEVICE_DTYPES)
def test_gr4j_xaj_parameter_boundaries_are_finite_and_gradient_safe(
    model_cls, specs, device: str, dtype: torch.dtype
):
    names, rows = _boundary_rows(specs)
    values = torch.tensor(rows, device=device, dtype=dtype)
    batch, time = values.shape[0], 72
    phase = torch.arange(time, device=device, dtype=dtype)
    precip = torch.clamp(4.0 + 4.0 * torch.sin(phase * 0.41), min=0.0)
    precip = precip.expand(batch, -1).clone()
    precip[:, ::11] = 25.0
    precip[:, 5::13] = 0.0
    pet = torch.clamp(2.5 + 2.0 * torch.cos(phase * 0.17), min=0.0)
    pet = pet.expand(batch, -1).clone()
    pet[:, 7::17] = 0.0
    temp = (9.0 * torch.sin(phase * 0.13) - 1.0).expand(batch, -1).clone()
    params = {
        name: torch.nn.Parameter(values[:, index].clone())
        for index, name in enumerate(names)
    }

    qsim, aux = model_cls().to(device=device, dtype=dtype)(
        {"precip": precip, "pet": pet, "temp": temp},
        params,
        return_states=True,
    )
    outputs = [qsim] + _flatten_tensors(aux)
    assert all(torch.isfinite(output).all() for output in outputs)

    loss = qsim.square().mean()
    if "evap" in aux:
        loss = loss + 0.01 * aux["evap"].mean()
    final_tensors = _flatten_tensors(aux.get("final_states", {}))
    if final_tensors:
        loss = loss + 1e-7 * sum(value.mean() for value in final_tensors)
    loss.backward()

    for name, parameter in params.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().max() < 1e6, name
