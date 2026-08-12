"""Regression tests for the SIMHYD reference and Wetland Euler formulations."""
from __future__ import annotations

import pytest
import torch

from models.core.simhyd import create_initial_state as simhyd_initial_state
from models.core.simhyd import simhyd_step
from models.core.wetland import wetland_step
from models.flux.baseflow import baseflow_1
from models.flux.evap import evap_1
from models.flux.excess import excess_1
from models.flux.interception import interception_2
from models.flux.saturation import saturation_2


def _x(value: float, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor([[value]], dtype=dtype)


def _wetland_tolerance(dtype: torch.dtype) -> float:
    return 2.0e-12 if dtype == torch.float64 else 3.0e-5


def _assert_wetland_step_invariants(
    P: torch.Tensor,
    PET: torch.Tensor,
    S1: torch.Tensor,
    dw: torch.Tensor,
    betaw: torch.Tensor,
    swmax: torch.Tensor,
    kw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qsim, ea, s1_new = wetland_step(
        P, torch.zeros_like(P), PET, dw, betaw, swmax, kw, S1
    )
    tolerance = _wetland_tolerance(P.dtype)
    for value in (qsim, ea, s1_new):
        assert bool(torch.isfinite(value).all())
    assert bool((qsim >= -tolerance).all())
    assert bool((ea >= -tolerance).all())
    assert bool((s1_new >= -tolerance).all())
    assert bool((s1_new <= swmax + tolerance).all())
    torch.testing.assert_close(P + S1, qsim + ea + s1_new, rtol=0.0, atol=tolerance)
    return qsim, ea, s1_new


def test_simhyd_no_uh_reference_step_is_mass_conserving_and_has_two_states() -> None:
    P, PET = _x(12.0), _x(5.0)
    insc, coeff, sq, smsc, sub, crak, k = (_x(v) for v in (3.0, 80.0, 2.0, 220.0, 0.35, 0.2, 0.15))
    soil, groundwater = _x(70.0), _x(40.0)
    q, ea, soil_new, groundwater_new = simhyd_step(
        P, _x(0.0), PET, insc, coeff, sq, smsc, sub, crak, k, soil, groundwater,
    )

    # The requested reference variant has no interception state and no UH;
    # all rainfall leaves as ET, immediate runoff, or a change in two stores.
    torch.testing.assert_close(P + soil + groundwater, q + ea + soil_new + groundwater_new)
    assert bool((q >= 0).all() and (ea >= 0).all() and (soil_new >= 0).all() and (groundwater_new >= 0).all())
    assert len(simhyd_initial_state(2, 3, torch.device("cpu"))) == 2


def test_wetland_matches_sequential_euler_fluxes_and_closes_water_balance() -> None:
    """Regression for pe -> saturation excess -> ET -> baseflow splitting."""
    P, PET, S1 = _x(6.0), _x(2.0), _x(80.0)
    dw, betaw, swmax, kw = (_x(v) for v in (1.0, 1.5, 200.0, 0.05))
    q, ea, s1 = wetland_step(P, _x(0.0), PET, dw, betaw, swmax, kw, S1)

    pe = interception_2(P, dw)
    ei = P - pe
    qwsof = torch.clamp(
        saturation_2(S1, swmax, betaw, pe),
        min=torch.zeros_like(P),
        max=pe,
    )
    storage_after_fast = S1 + pe - qwsof
    qwsof = qwsof + excess_1(storage_after_fast, swmax)
    storage_after_fast = S1 + pe - qwsof
    ew = evap_1(storage_after_fast, PET)
    storage_after_et = storage_after_fast - ew
    qwgw = baseflow_1(kw, storage_after_et)
    torch.testing.assert_close(q, qwsof + qwgw)
    torch.testing.assert_close(ea, ei + ew)
    torch.testing.assert_close(s1, storage_after_et - qwgw)
    torch.testing.assert_close(P + S1, q + ea + s1, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize(
    ("P", "PET", "S1", "dw", "betaw", "swmax", "kw"),
    [
        (0.0, 0.0, 0.0, 1.0, 1.0, 100.0, 0.2),       # P=PET=S1=0
        (0.0, 100.0, 1.0, 1.0, 1.0, 100.0, 1.0),     # high PET, kw=1
        (3.0, 0.0, 10.0, 0.0, 1.0, 100.0, 0.0),      # dw=0, kw=0
        (2.0, 1.0, 10.0, 5.0, 1.0, 100.0, 0.3),      # dw>P
        (15.0, 2.0, 99.9, 0.0, 0.0, 100.0, 0.4),     # betaw=0
        (500.0, 0.0, 99.9, 0.0, 9.999, 100.0, 0.4),  # capacity overflow
    ],
)
def test_wetland_step_boundary_cases_are_finite_conservative_and_bounded(
    dtype: torch.dtype,
    P: float,
    PET: float,
    S1: float,
    dw: float,
    betaw: float,
    swmax: float,
    kw: float,
) -> None:
    _assert_wetland_step_invariants(
        _x(P, dtype), _x(PET, dtype), _x(S1, dtype), _x(dw, dtype),
        _x(betaw, dtype), _x(swmax, dtype), _x(kw, dtype),
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_wetland_long_dry_high_pet_sequence_has_no_water_balance_drift(
    dtype: torch.dtype,
) -> None:
    """No final nearzero clamp may inject water during a long dry sequence."""
    P = _x(0.0, dtype)
    PET = _x(100.0, dtype)
    S1 = _x(30.0, dtype)
    dw, betaw, swmax, kw = (_x(v, dtype) for v in (1.0, 1.5, 100.0, 0.7))
    initial_storage = S1.clone()
    total_q = torch.zeros_like(S1)
    total_ea = torch.zeros_like(S1)
    for _ in range(730):
        qsim, ea, S1 = _assert_wetland_step_invariants(
            P, PET, S1, dw, betaw, swmax, kw
        )
        total_q = total_q + qsim
        total_ea = total_ea + ea
    torch.testing.assert_close(
        initial_storage, total_q + total_ea + S1,
        rtol=0.0, atol=_wetland_tolerance(dtype),
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_wetland_parameter_gradients_are_finite(dtype: torch.dtype) -> None:
    """Interior parameter gradients must remain finite through a short sequence."""
    dw = _x(1.0, dtype).requires_grad_()
    betaw = _x(1.7, dtype).requires_grad_()
    swmax = _x(100.0, dtype).requires_grad_()
    kw = _x(0.35, dtype).requires_grad_()
    S1 = _x(65.0, dtype)
    loss = torch.zeros((), dtype=dtype)
    for day in range(12):
        P = _x(8.0 + float(day % 3), dtype)
        PET = _x(2.0 + 0.5 * float(day % 2), dtype)
        qsim, ea, S1 = wetland_step(
            P, _x(0.0, dtype), PET, dw, betaw, swmax, kw, S1
        )
        loss = loss + (day + 1.0) * qsim.sum() + 0.1 * ea.sum() + 0.01 * S1.sum()
    loss.backward()
    for parameter in (dw, betaw, swmax, kw):
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_wetland_batched_step_matches_individual_steps(dtype: torch.dtype) -> None:
    P = torch.tensor([[0.0, 3.0, 100.0], [8.0, 2.0, 500.0]], dtype=dtype)
    PET = torch.tensor([[5.0, 0.0, 2.0], [1.0, 100.0, 0.0]], dtype=dtype)
    S1 = torch.tensor([[0.0, 10.0, 99.0], [50.0, 1.0, 99.9]], dtype=dtype)
    dw = torch.tensor([[1.0, 0.0, 5.0], [2.0, 5.0, 0.0]], dtype=dtype)
    betaw = torch.tensor([[1.0, 0.0, 9.99], [1.5, 2.0, 8.0]], dtype=dtype)
    swmax = torch.full_like(P, 100.0)
    kw = torch.tensor([[0.0, 0.2, 1.0], [0.3, 1.0, 0.4]], dtype=dtype)

    batched = wetland_step(P, torch.zeros_like(P), PET, dw, betaw, swmax, kw, S1)
    for row in range(P.shape[0]):
        for column in range(P.shape[1]):
            individual = wetland_step(
                P[row : row + 1, column : column + 1],
                torch.zeros((1, 1), dtype=dtype),
                PET[row : row + 1, column : column + 1],
                dw[row : row + 1, column : column + 1],
                betaw[row : row + 1, column : column + 1],
                swmax[row : row + 1, column : column + 1],
                kw[row : row + 1, column : column + 1],
                S1[row : row + 1, column : column + 1],
            )
            for batched_value, individual_value in zip(batched, individual):
                torch.testing.assert_close(
                    batched_value[row : row + 1, column : column + 1],
                    individual_value,
                    rtol=0.0,
                    atol=_wetland_tolerance(dtype),
                )
