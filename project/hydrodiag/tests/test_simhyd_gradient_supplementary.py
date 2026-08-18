"""Supplementary regression tests for the SIMHYD gradient safeguards.

These tests deliberately cover the production compiled step, the default
state initializer, and a normal-basin no-floor reference.  They are float32
tests because that is the production dPL model dtype.
"""

from __future__ import annotations

import pytest
import torch
from models import SIMHYD
from models.parameter_specs import SIMHYD_PARAM_SPECS
from models.simhyd import SIMHYD_MIN_SMSC, SIMHYD_UH_MAX_LEN, _simhyd_step
from training.dpl.run_dpl_model import kge_per_basin

BATCH = 2
TIME = 96
ATOL_GRAD = 1e-5
RTOL_GRAD = 1e-4
MAX_GRAD = 1e6


def _degenerate_forcing() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    precip = torch.zeros(BATCH, TIME, dtype=torch.float32, requires_grad=True)
    pet = torch.full((BATCH, TIME), 1e-8, dtype=torch.float32, requires_grad=True)
    temp = torch.zeros(BATCH, TIME, dtype=torch.float32)
    qobs = torch.linspace(0.1, 4.0, TIME, dtype=torch.float32).expand(BATCH, -1)
    return {"precip": precip, "pet": pet, "temp": temp}, qobs


def _zero_initial_states() -> dict[str, torch.Tensor]:
    return {
        "soil": torch.zeros(BATCH),
        "groundwater": torch.zeros(BATCH),
        "runoff_uh_buffer": torch.zeros(BATCH, SIMHYD_UH_MAX_LEN - 1),
    }


def _degenerate_params(smsc: float) -> dict[str, torch.nn.Parameter]:
    overrides = {
        "simhyd_insc": 0.0,
        "simhyd_coeff": 0.0,
        "simhyd_sq": 10.0,
        "simhyd_smsc": smsc,
        "simhyd_sub": 0.0,
        "simhyd_crak": 1.0,
        "simhyd_k": 1.0,
        "simhyd_etmul": 3.0,
        "simhyd_a": 10.0,
        "simhyd_theta": 0.5,
    }
    return {
        name: torch.nn.Parameter(
            torch.full((BATCH,), overrides.get(name, float(spec["default"])))
        )
        for name, spec in SIMHYD_PARAM_SPECS.items()
    }


def _run_kge(
    model: SIMHYD,
    smsc: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    forcings, qobs = _degenerate_forcing()
    params = _degenerate_params(smsc)
    qsim, _ = model(
        forcings=forcings,
        params=params,
        initial_states=_zero_initial_states(),
    )
    loss = (1.0 - kge_per_basin(qsim, qobs, eps=1e-6)).mean()
    assert torch.isfinite(qsim).all()
    assert torch.isfinite(loss).all()
    loss.backward()
    grads = {
        **{name: parameter.grad.detach().clone() for name, parameter in params.items()},
        "forcing_precip": forcings["precip"].grad.detach().clone(),
        "forcing_pet": forcings["pet"].grad.detach().clone(),
    }
    return loss.detach(), grads, params


def _assert_finite_and_bounded(grads: dict[str, torch.Tensor]) -> None:
    for name, grad in grads.items():
        assert torch.isfinite(grad).all(), f"{name} gradient is non-finite"
        assert grad.abs().max() < MAX_GRAD, f"{name} gradient is unstable"


@pytest.mark.parametrize(
    "smsc",
    [SIMHYD_MIN_SMSC, SIMHYD_MIN_SMSC * 1.01],
    ids=["smsc_at_floor", "smsc_just_above_floor"],
)
def test_simhyd_compiled_degenerate_kge_backward_matches_eager(
    smsc: float,
) -> None:
    """Production compiled backward is finite and agrees with eager backward."""
    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        pytest.skip("torch.compile is unavailable in this PyTorch environment")

    torch._dynamo.reset()
    compiled_model = SIMHYD()  # real production constructor: compile enabled
    compiled_loss, compiled_grads, _ = _run_kge(compiled_model, smsc)

    eager_model = SIMHYD()
    eager_model._step = _simhyd_step
    eager_loss, eager_grads, _ = _run_kge(eager_model, smsc)

    _assert_finite_and_bounded(compiled_grads)
    _assert_finite_and_bounded(eager_grads)
    torch.testing.assert_close(
        compiled_loss, eager_loss, rtol=RTOL_GRAD, atol=ATOL_GRAD
    )
    for name in eager_grads:
        torch.testing.assert_close(
            compiled_grads[name],
            eager_grads[name],
            rtol=RTOL_GRAD,
            atol=ATOL_GRAD,
        )
    print(
        f"[compiled A/B] smsc={smsc:.3e}: finite=True, "
        f"grad_match=True, max_grad="
        f"{max(float(g.abs().max()) for g in compiled_grads.values()):.3e}"
    )


@pytest.mark.parametrize("smsc", [0.0, 1e-9], ids=["smsc_zero", "smsc_below_floor"])
def test_simhyd_min_smsc_covers_default_initializer_and_backward(
    monkeypatch: pytest.MonkeyPatch,
    smsc: float,
) -> None:
    """The default ``0.5 * smsc`` initializer must receive the same floor."""
    monkeypatch.setattr(torch, "compile", lambda function, *args, **kwargs: function)
    model = SIMHYD()
    params = _degenerate_params(smsc)

    soil, groundwater, routing_buffer = model._init_states(
        BATCH,
        torch.device("cpu"),
        torch.float32,
        params["simhyd_smsc"],
        initial_states=None,
    )
    expected_soil = torch.full((BATCH,), 0.5 * SIMHYD_MIN_SMSC)
    assert torch.equal(soil, expected_soil), (
        "default soil initializer bypasses SIMHYD_MIN_SMSC"
    )
    assert torch.equal(groundwater, torch.zeros(BATCH))
    assert torch.equal(routing_buffer, torch.zeros(BATCH, SIMHYD_UH_MAX_LEN - 1))

    forcings, qobs = _degenerate_forcing()
    qsim, _ = model(forcings, params, initial_states=None)
    loss = (1.0 - kge_per_basin(qsim, qobs, eps=1e-6)).mean()
    assert torch.isfinite(loss).all()
    loss.backward()
    grads = {
        **{name: parameter.grad for name, parameter in params.items()},
        "forcing_precip": forcings["precip"].grad,
        "forcing_pet": forcings["pet"].grad,
    }
    assert all(grad is not None for grad in grads.values())
    _assert_finite_and_bounded(grads)  # type: ignore[arg-type]


def _simhyd_step_without_new_floors(
    precip_t: torch.Tensor,
    pet_t: torch.Tensor,
    soil: torch.Tensor,
    groundwater: torch.Tensor,
    insc: torch.Tensor,
    coeff: torch.Tensor,
    sq: torch.Tensor,
    smsc: torch.Tensor,
    sub: torch.Tensor,
    crak: torch.Tensor,
    k: torch.Tensor,
    etmul: torch.Tensor,
    nearzero: float,
) -> tuple[torch.Tensor, ...]:
    """Pre-fix reference: only bypass the new smsc and k safeguards."""
    precip = torch.clamp(precip_t, min=0.0)
    pet = torch.clamp(pet_t * etmul, min=0.0)
    insc_safe = torch.clamp(insc, min=1e-6)
    interception = torch.minimum(torch.minimum(insc_safe, pet), precip)
    rainfall_excess = precip - interception
    pet_remaining = pet - interception
    soil_ratio = torch.clamp(soil / (smsc + nearzero), 0.0, 1.0)
    coeff_safe = torch.clamp(coeff, min=1e-6)
    infiltration_capacity = coeff_safe * torch.exp(-sq * soil_ratio)
    infiltration = torch.minimum(infiltration_capacity, rainfall_excess)
    direct_runoff = rainfall_excess - infiltration
    interflow = sub * soil_ratio * infiltration
    recharge = crak * soil_ratio * (infiltration - interflow)
    soil_fill = infiltration - interflow - recharge
    soil_available = soil + soil_fill
    soil_evap = torch.minimum(10.0 * soil_ratio, pet_remaining)
    soil_evap = torch.minimum(soil_evap, soil_available)
    soil_after_evap = soil_available - soil_evap
    soil_overflow = torch.clamp(soil_after_evap - smsc, min=0.0)
    soil_new = soil_after_evap - soil_overflow
    recharge_total = recharge + soil_overflow
    baseflow = torch.minimum(k * groundwater, groundwater)
    groundwater_new = groundwater + recharge_total - baseflow
    runoff = direct_runoff + interflow + baseflow
    evap = interception + soil_evap
    return (
        runoff,
        evap,
        soil_new,
        groundwater_new,
        interception,
        direct_runoff,
        interflow,
        recharge_total,
        baseflow,
    )


def _normal_forcing() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    torch.manual_seed(20260721)
    precip = torch.rand(BATCH, TIME) * 15.0
    pet = torch.rand(BATCH, TIME) * 4.0 + 0.5
    temp = torch.randn(BATCH, TIME) * 5.0 + 8.0
    qobs = torch.rand(BATCH, TIME) * 8.0 + 0.1
    return {"precip": precip, "pet": pet, "temp": temp}, qobs


def _normal_params() -> dict[str, torch.nn.Parameter]:
    values = {
        "simhyd_insc": 3.0,
        "simhyd_coeff": 20.0,
        "simhyd_sq": 2.0,
        "simhyd_smsc": 250.0,
        "simhyd_sub": 0.35,
        "simhyd_crak": 0.2,
        "simhyd_k": 0.65,
        "simhyd_etmul": 1.1,
        "simhyd_a": 2.5,
        "simhyd_theta": 2.0,
    }
    return {
        name: torch.nn.Parameter(torch.full((BATCH,), values[name]))
        for name in SIMHYD_PARAM_SPECS
    }


def _run_normal(step) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model = SIMHYD()
    model._step = step
    forcings, qobs = _normal_forcing()
    params = _normal_params()
    qsim, _ = model(forcings, params)
    loss = (1.0 - kge_per_basin(qsim, qobs, eps=1e-6)).mean()
    loss.backward()
    grads = {
        name: parameter.grad.detach().clone() for name, parameter in params.items()
    }
    return qsim.detach(), grads


def test_simhyd_normal_basin_matches_reference_without_new_floors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal-basin forward and KGE parameter gradients remain unchanged."""
    monkeypatch.setattr(torch, "compile", lambda function, *args, **kwargs: function)
    qsim, grads = _run_normal(_simhyd_step)
    reference_qsim, reference_grads = _run_normal(_simhyd_step_without_new_floors)

    assert torch.max(torch.abs(qsim - reference_qsim)) < 1e-6
    for name in grads:
        assert torch.max(torch.abs(grads[name] - reference_grads[name])) < 1e-5, name
    print(
        "[normal reference] qsim_match=True, grads_match=True, "
        f"max_qsim_diff={float(torch.abs(qsim - reference_qsim).max()):.3e}, "
        f"max_grad_diff={max(float((grads[n] - reference_grads[n]).abs().max()) for n in grads):.3e}"
    )
