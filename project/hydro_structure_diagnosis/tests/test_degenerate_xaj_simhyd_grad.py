"""Temporary eager A/B gradient audit for the dry, low-runoff boundary.

This deliberately bypasses ``torch.compile``.  It uses the same KGE loss as
the dPL training path and checks both physical parameters and forcing inputs.
"""

from __future__ import annotations

import pytest
import torch

from models import SIMHYD, XAJ
from models.parameter_specs import SIMHYD_PARAM_SPECS, XAJ_PARAM_SPECS
from models.xaj import _xaj_step
from models.simhyd import _simhyd_step
from training.dpl.run_dpl_model import kge_per_basin


DEGENERATE_TIME = 96
DEGENERATE_BATCH = 2


def _dry_forcing() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    precip = torch.zeros(
        DEGENERATE_BATCH, DEGENERATE_TIME, dtype=torch.float32, requires_grad=True
    )
    pet = torch.full(
        (DEGENERATE_BATCH, DEGENERATE_TIME), 1e-8,
        dtype=torch.float32,
        requires_grad=True,
    )
    temp = torch.zeros(DEGENERATE_BATCH, DEGENERATE_TIME, dtype=torch.float32)
    qobs = torch.linspace(0.1, 4.0, DEGENERATE_TIME).expand(DEGENERATE_BATCH, -1)
    return {"precip": precip, "pet": pet, "temp": temp}, qobs


def _zero_states(model_cls: type[torch.nn.Module]) -> dict[str, torch.Tensor]:
    if model_cls is XAJ:
        names = ("wu", "wl", "wd", "s", "fr", "qi", "qg")
        states = {name: torch.zeros(DEGENERATE_BATCH) for name in names}
        states["rs_uh_buffer"] = torch.zeros(DEGENERATE_BATCH, 89)
        return states
    states = {
        "soil": torch.zeros(DEGENERATE_BATCH),
        "groundwater": torch.zeros(DEGENERATE_BATCH),
        "runoff_uh_buffer": torch.zeros(DEGENERATE_BATCH, 89),
    }
    return states


def _params(
    specs: dict[str, dict[str, float]],
    overrides: dict[str, float],
) -> dict[str, torch.nn.Parameter]:
    params = {
        name: torch.nn.Parameter(
            torch.full((DEGENERATE_BATCH,), float(spec["default"]))
        )
        for name, spec in specs.items()
    }
    for name, value in overrides.items():
        params[name].data.fill_(value)
    return params


@pytest.mark.parametrize(
    ("model_cls", "step", "specs", "overrides"),
    [
        pytest.param(
            XAJ,
            _xaj_step,
            XAJ_PARAM_SPECS,
            {"xaj_im": 0.0, "xaj_ki": 0.0, "xaj_kg": 0.0,
             "xaj_ci": 1.0, "xaj_cg": 1.0},
            id="XAJ",
        ),
        pytest.param(
            SIMHYD,
            _simhyd_step,
            SIMHYD_PARAM_SPECS,
            {"simhyd_insc": 0.0, "simhyd_coeff": 0.0, "simhyd_sq": 10.0,
             "simhyd_smsc": 0.0, "simhyd_sub": 0.0, "simhyd_crak": 1.0,
             "simhyd_k": 1.0, "simhyd_etmul": 3.0,
             "simhyd_a": 10.0, "simhyd_theta": 0.5},
            id="SIMHYD",
        ),
    ],
)
def test_degenerate_kge_backward_has_finite_parameter_and_forcing_grads(
    monkeypatch: pytest.MonkeyPatch,
    model_cls: type[torch.nn.Module],
    step,
    specs: dict[str, dict[str, float]],
    overrides: dict[str, float],
) -> None:
    """A/B result: both models must survive the same eager KGE backward path."""
    # Disable compilation before model construction; this test is specifically
    # intended to establish whether the failure is mathematical or Inductor.
    monkeypatch.setattr(torch, "compile", lambda function, *args, **kwargs: function)
    model = model_cls()
    assert model._step is step

    forcings, qobs = _dry_forcing()
    params = _params(specs, overrides)
    qsim, _ = model(
        forcings=forcings,
        params=params,
        initial_states=_zero_states(model_cls),
    )
    loss = (1.0 - kge_per_basin(qsim, qobs, eps=1e-6)).mean()
    assert torch.isfinite(qsim).all()
    assert torch.isfinite(loss).all()
    loss.backward()

    for name, parameter in params.items():
        assert parameter.grad is not None, f"{model_cls.__name__}.{name} has no grad"
        assert torch.isfinite(parameter.grad).all(), (
            f"{model_cls.__name__}.{name} parameter grad is non-finite"
        )
        assert parameter.grad.abs().max() < 1e6, (
            f"{model_cls.__name__}.{name} parameter grad is unstable"
        )
    for name in ("precip", "pet"):
        gradient = forcings[name].grad
        assert gradient is not None, f"{model_cls.__name__}.{name} has no grad"
        assert torch.isfinite(gradient).all(), (
            f"{model_cls.__name__}.{name} forcing grad is non-finite"
        )
        assert gradient.abs().max() < 1e6, (
            f"{model_cls.__name__}.{name} forcing grad is unstable"
        )
