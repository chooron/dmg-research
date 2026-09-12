from datetime import date
import torch
import pytest

from dfuse import SEQUENTIAL_ORDERS, compile_diagnostics, enumerate_structures, get_structure, reset_compile_diagnostics, simulate, simulate_explicit, simulate_sequential
from dfuse.spec import default_parameters
from dfuse.kernel import _day_of_year


FORCING = torch.tensor(
    [
        [5.0, 2.0, 10.0],
        [0.0, 2.0, -2.0],
        [4.0, 2.0, 10.0],
        [3.0, 1.5, 8.0],
    ],
    dtype=torch.float64,
)


def test_all_78_structures_have_finite_minimal_forward():
    for spec in enumerate_structures():
        result = simulate(spec.model_id, FORCING, implicit_iterations=16)
        assert result.states.shape == (len(FORCING) + 1, len(spec.state_names))
        assert result.q.shape == (len(FORCING),)
        assert torch.isfinite(result.states).all()
        assert torch.isfinite(result.q).all()
        assert torch.isfinite(result.water_balance_residual).all()


def test_default_initialization_matches_frozen_fortran_fraction():
    result = simulate(84, FORCING[:1], implicit_iterations=1)
    torch.testing.assert_close(result.states[0, 0], torch.tensor(25.0, dtype=torch.float64))
def test_leap_year_day_of_year_is_preserved_for_snow_timing():
    days, leap = _day_of_year(
        3, [date(2000, 2, 28), date(2000, 2, 29), date(2000, 3, 1)],
        dtype=torch.float64, device=torch.device("cpu")
    )
    torch.testing.assert_close(days, torch.tensor([59.0, 60.0, 61.0], dtype=torch.float64))
    assert leap.tolist() == [True, True, True]


def test_water_balance_and_snow_balance_for_representative_structure():
    cold_forcing = torch.tensor([[4.0, 1.0, -5.0], [0.0, 1.0, -4.0], [2.0, 1.0, 8.0]], dtype=torch.float64)
    result = simulate(84, cold_forcing, implicit_iterations=24)
    assert float(result.max_abs_water_balance_error.detach()) < 1.0e-5
    assert float(result.snow_balance_residual.abs().amax().detach()) < 1.0e-10
    assert float(result.snow.max()) > 0.0

def test_snow_balance_checker_omits_precipitation_at_exact_pxtemp():
    pxtemp = default_parameters()["PXTEMP"]
    forcing = torch.tensor([[15.45, 0.5, pxtemp]], dtype=torch.float64)
    result = simulate(84, forcing, compile_inner=False)
    torch.testing.assert_close(result.snow_balance_residual, torch.zeros_like(result.snow_balance_residual), rtol=0.0, atol=0.0)
    assert (result.q >= 0.0).all()


def test_topmodel_quadrature_gradients_and_inactive_parameter_mask():
    raw = {
        name: torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for name, value in default_parameters().items()
    }
    result = simulate(210, FORCING, raw, implicit_iterations=12)
    result.q.sum().backward()
    for name in ("LOGLAMB", "TISHAPE", "QB_POWR"):
        assert raw[name].grad is not None
        assert torch.isfinite(raw[name].grad)
    assert raw["AXV_BEXP"].grad is None or torch.equal(raw["AXV_BEXP"].grad, torch.zeros_like(raw["AXV_BEXP"].grad))
def test_bound_stress_is_finite_and_conservative():
    forcing = torch.tensor([[5000.0, 0.0, 10.0]] * 3, dtype=torch.float64)
    for model_id in (162, 208):
        result = simulate(model_id, forcing, implicit_iterations=16)
        assert torch.isfinite(result.q).all()
        assert torch.isfinite(result.states).all()
        assert all(torch.isfinite(value).all() for value in result.fluxes.values())
        assert float(result.max_abs_water_balance_error.detach()) < 1.0e-8
def test_explicit_substeps_are_finite_for_mother_models():
    for model_id in (2, 108, 178, 210):
        for n_substeps in (1, 4, 48):
            result = simulate_explicit(model_id, FORCING, n_substeps=n_substeps)
            assert result.states.shape == (len(FORCING) + 1, len(get_structure(model_id).state_names))
            assert torch.isfinite(result.q).all()
            assert torch.isfinite(result.states).all()
            assert all(torch.isfinite(value).all() for value in result.fluxes.values())
            assert float(result.max_abs_water_balance_error.detach()) < 1.0e-5

def test_explicit_substeps_are_finite_for_all_78_structures():
    for spec in enumerate_structures():
        result = simulate_explicit(spec.model_id, FORCING, n_substeps=4)
        assert torch.isfinite(result.q).all()
        assert torch.isfinite(result.states).all()
        assert float(result.max_abs_water_balance_error.detach()) < 1.0e-5



def test_explicit_substeps_reject_nonpositive_values():
    try:
        simulate(2, FORCING[:1], solver="explicit", n_substeps=0)
    except ValueError as exc:
        assert "positive integer" in str(exc)
    else:
        raise AssertionError("nonpositive explicit substeps must be rejected")

def test_explicit_topmodel_gradients_remain_finite_and_masked():
    raw = {
        name: torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for name, value in default_parameters().items()
    }
    result = simulate(210, FORCING, raw, solver="explicit", n_substeps=24)
    result.q.sum().backward()
    for name in ("LOGLAMB", "TISHAPE", "QB_POWR"):
        assert raw[name].grad is not None
        assert torch.isfinite(raw[name].grad)
    assert raw["AXV_BEXP"].grad is None or torch.equal(raw["AXV_BEXP"].grad, torch.zeros_like(raw["AXV_BEXP"].grad))





def test_forward_is_autograd_friendly():
    params = {
        "MAXWATR_1": torch.tensor(100.0, dtype=torch.float64, requires_grad=True),
        "AXV_BEXP": torch.tensor(0.3, dtype=torch.float64, requires_grad=True),
        "TIMEDELAY": torch.tensor(0.9, dtype=torch.float64, requires_grad=True),
    }
    result = simulate(84, FORCING[:3], params, implicit_iterations=16)
    result.q.sum().backward()
    assert torch.isfinite(params["MAXWATR_1"].grad)
    assert torch.isfinite(params["AXV_BEXP"].grad)
    assert torch.isfinite(params["TIMEDELAY"].grad)


def test_sequential_orders_are_fixed_and_topology_aware():
    assert tuple(SEQUENTIAL_ORDERS) == ("S1", "S2", "S3", "S4")
    assert SEQUENTIAL_ORDERS["S1"][0] == "recharge"
    assert SEQUENTIAL_ORDERS["S2"][1] == "surface_runoff"
    assert SEQUENTIAL_ORDERS["S3"][1:3] == ("et", "percolation")
    assert SEQUENTIAL_ORDERS["S4"][1:3] == ("percolation", "baseflow")
    for order in SEQUENTIAL_ORDERS:
        result = simulate_sequential(210, FORCING, order=order, compile_step=False, n_substeps=2)
        assert torch.isfinite(result.states).all()
        assert torch.isfinite(result.q).all()
        assert float(result.water_balance_residual.abs().amax()) < 1.0e-8


def test_sequential_compiled_step_parity_and_structure_isolation():
    if not torch.cuda.is_available():
        pytest.skip("compiled sequential validation is GPU-first")
    forcing = FORCING[:2].cuda()
    reset_compile_diagnostics()
    eager = simulate_sequential(2, forcing, order="S1", compile_step=False)
    compiled = simulate_sequential(2, forcing, order="S1", compile_step=True)
    other_structure = simulate_sequential(210, forcing, order="S1", compile_step=True)
    torch.testing.assert_close(compiled.states, eager.states, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(compiled.q, eager.q, rtol=1.0e-12, atol=1.0e-12)
    assert torch.isfinite(other_structure.q).all()
    variants = compile_diagnostics()["sequential"]["variants"]
    assert len(variants) == 2
    for variant in variants.values():
        assert variant["compile_attempts"] == 1
        assert variant["compile_successes"] == 1
        assert variant["fallbacks"] == 0
        assert variant["graph_breaks"] == 0
        assert variant["recompilations"] == 0
        assert variant["unique_graphs"] == 1


def test_sequential_compiled_gradients_are_finite_and_masked():
    if not torch.cuda.is_available():
        pytest.skip("compiled sequential gradient validation is GPU-first")
    forcing = FORCING[:2].cuda()
    raw = {
        name: torch.tensor(value, dtype=torch.float64, device="cuda", requires_grad=True)
        for name, value in default_parameters().items()
    }
    result = simulate_sequential(210, forcing, raw, order="S1", compile_step=True)
    result.q.sum().backward()
    for name in ("LOGLAMB", "TISHAPE", "QB_POWR", "TIMEDELAY"):
        assert raw[name].grad is not None
        assert torch.isfinite(raw[name].grad)
    assert raw["AXV_BEXP"].grad is None or torch.equal(raw["AXV_BEXP"].grad, torch.zeros_like(raw["AXV_BEXP"].grad))


def test_simulate_dispatches_sequential_solver():
    direct = simulate_sequential(2, FORCING[:1], order="S3", compile_step=False)
    dispatched = simulate(2, FORCING[:1], solver="sequential", sequential_order="S3", compile_step=False)
    torch.testing.assert_close(direct.q, dispatched.q)
    torch.testing.assert_close(direct.states, dispatched.states)
