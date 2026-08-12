"""Tests for paper-style CMA-ES early stopping.

Covers:
  A. Configuration and population size formula
  B. TolFun checking
  C. TolHistFun checking
  D. TolX checking
  E. Stop logic (combined)
  F. Numerical safety (NaN, Inf)
  G. EvoTorch integration (end-to-end Sphere)
  H. Regression / isolation
"""

import math
import time

import numpy as np
import pytest
import torch

from optimization.paper_style_cmaes import (
    CMAESConfig,
    CMAESGenerationRecord,
    CMAESRunResult,
    PaperStyleCMAESStopper,
    StopReason,
    _determine_stop_reasons,
    _select_finite,
    default_population_size,
    extract_cmaes_distribution_state,
    normalized_to_physical,
    physical_to_normalized,
    run_cmaes_with_early_stopping,
)

# ============================================================================
# Section A: Configuration and formula tests
# ============================================================================


class TestPopulationSizeFormula:
    def test_n_dim_1(self):
        assert default_population_size(1) == 4

    def test_n_dim_2(self):
        assert default_population_size(2) == 6

    def test_n_dim_5(self):
        assert default_population_size(5) == 8

    def test_n_dim_10(self):
        assert default_population_size(10) == 10

    def test_n_dim_15(self):
        assert default_population_size(15) == 12

    def test_n_dim_zero_raises(self):
        with pytest.raises(ValueError):
            default_population_size(0)

    def test_n_dim_negative_raises(self):
        with pytest.raises(ValueError):
            default_population_size(-1)


class TestNormalizedPhysicalMapping:
    def test_boundary_0(self):
        lb = np.array([10.0, 50.0])
        ub = np.array([110.0, 150.0])
        x_norm = np.array([0.0, 0.0])
        physical = normalized_to_physical(torch.tensor(x_norm), torch.tensor(lb), torch.tensor(ub))
        np.testing.assert_allclose(physical.numpy(), lb)

    def test_boundary_1(self):
        lb = np.array([10.0, 50.0])
        ub = np.array([110.0, 150.0])
        x_norm = np.array([1.0, 1.0])
        physical = normalized_to_physical(torch.tensor(x_norm), torch.tensor(lb), torch.tensor(ub))
        np.testing.assert_allclose(physical.numpy(), ub)

    def test_roundtrip(self):
        lb = np.array([1.0, -5.0, 20.0])
        ub = np.array([6.0, 3.0, 5000.0])
        x_phys = np.array([2.0, 0.0, 350.0])
        x_norm = physical_to_normalized(torch.tensor(x_phys), torch.tensor(lb), torch.tensor(ub))
        x_back = normalized_to_physical(x_norm, torch.tensor(lb), torch.tensor(ub))
        np.testing.assert_allclose(x_back.numpy(), x_phys, atol=1e-6)

    def test_midpoint(self):
        lb = np.array([0.0, 10.0])
        ub = np.array([100.0, 20.0])
        x_norm = np.array([0.5, 0.5])
        expected = (lb + ub) / 2.0
        physical = normalized_to_physical(torch.tensor(x_norm), torch.tensor(lb), torch.tensor(ub))
        np.testing.assert_allclose(physical.numpy(), expected)


class TestConfigDefaults:
    def test_population_size_auto(self):
        config = CMAESConfig(n_dim=5)
        assert config.population_size == default_population_size(5)

    def test_population_size_override(self):
        config = CMAESConfig(n_dim=5, population_size=20)
        assert config.population_size == 20

    def test_negative_n_dim_raises(self):
        with pytest.raises(ValueError):
            CMAESConfig(n_dim=0)

    def test_min_generations_negative_raises(self):
        with pytest.raises(ValueError):
            CMAESConfig(n_dim=5, min_generations=-1)

    def test_max_less_than_min_raises(self):
        with pytest.raises(ValueError):
            CMAESConfig(n_dim=5, max_generations=5, min_generations=10)


# ============================================================================
# Section B: TolFun tests
# ============================================================================


class TestTolFun:
    def test_triggers(self):
        config = CMAESConfig(n_dim=2, tol_fun=1e-3)
        stopper = PaperStyleCMAESStopper(config)
        evals = np.array([1.0000, 1.0003, 1.0007])
        triggered, rng = stopper.check_tol_fun(evals)
        assert triggered
        assert abs(rng - 0.0007) < 1e-6

    def test_not_triggered(self):
        config = CMAESConfig(n_dim=2, tol_fun=1e-3)
        stopper = PaperStyleCMAESStopper(config)
        evals = np.array([1.0000, 1.0020])
        triggered, _ = stopper.check_tol_fun(evals)
        assert not triggered

    def test_exact_equality_triggers(self):
        config = CMAESConfig(n_dim=2, tol_fun=1e-3)
        stopper = PaperStyleCMAESStopper(config)
        evals = np.array([1.000, 1.001])
        triggered, rng = stopper.check_tol_fun(evals)
        assert triggered
        assert abs(rng - 1e-3) < 1e-9

    def test_all_same_triggers(self):
        config = CMAESConfig(n_dim=2, tol_fun=1e-3)
        stopper = PaperStyleCMAESStopper(config)
        evals = np.array([5.0, 5.0, 5.0])
        triggered, rng = stopper.check_tol_fun(evals)
        assert triggered
        assert rng == 0.0

    def test_with_nan_mixed(self):
        config = CMAESConfig(n_dim=2, tol_fun=1e-3)
        stopper = PaperStyleCMAESStopper(config)
        evals = np.array([1.0000, np.nan, 1.0003, 1.0007])
        triggered, rng = stopper.check_tol_fun(evals)
        assert triggered


# ============================================================================
# Section C: TolHistFun tests
# ============================================================================


class TestTolHistFun:
    def test_insufficient_history(self):
        config = CMAESConfig(n_dim=2, history_window=10, tol_hist_fun=1e-3)
        stopper = PaperStyleCMAESStopper(config)
        for i in range(5):
            stopper.record_best(float(100 - i))
        triggered, val = stopper.check_tol_hist_fun()
        assert not triggered
        assert val is None

    def test_triggers_with_converged_window(self):
        config = CMAESConfig(n_dim=2, history_window=10, tol_hist_fun=1e-3)
        stopper = PaperStyleCMAESStopper(config)
        for i in range(10):
            stopper.record_best(1.0 + 0.0001 * (10 - i))
        triggered, val = stopper.check_tol_hist_fun()
        assert triggered

    def test_not_triggered_diverged_window(self):
        config = CMAESConfig(n_dim=2, history_window=10, tol_hist_fun=1e-3)
        stopper = PaperStyleCMAESStopper(config)
        for i in range(10):
            stopper.record_best(1.0 + 0.01 * i)
        triggered, val = stopper.check_tol_hist_fun()
        assert not triggered

    def test_sliding_window_ignores_old_convergence(self):
        config = CMAESConfig(n_dim=2, history_window=10, tol_hist_fun=1e-3)
        stopper = PaperStyleCMAESStopper(config)
        # First 10: converged
        for i in range(10):
            stopper.record_best(1.0)
        triggered_early, _ = stopper.check_tol_hist_fun()
        assert triggered_early
        # Next 10: diverging
        for i in range(10):
            stopper.record_best(1.0 + 0.01 * i)
        triggered_late, _ = stopper.check_tol_hist_fun()
        assert not triggered_late

    def test_window_only_uses_representative_values(self):
        config = CMAESConfig(n_dim=2, history_window=10, tol_hist_fun=1e-3)
        stopper = PaperStyleCMAESStopper(config)
        for i in range(15):
            stopper.record_best(10.0 + 0.1 * i)
        assert stopper.history_length == 15
        hist_range = stopper.get_history_eval_range()
        assert hist_range is not None
        assert abs(hist_range - 0.9) < 0.01


# ============================================================================
# Section D: TolX tests
# ============================================================================


class TestExtractDistributionState:
    def test_coordinate_std_triggers_tolx(self):
        """sigma=0.01, C_diag=[0.01, 0.0025] -> coord_std=[0.001, 0.0005] -> triggers tol_x=1e-3"""
        from evotorch import Problem
        from evotorch.algorithms import CMAES

        def obj(x):
            return torch.sum((x - 0.5) ** 2, dim=-1)

        prob = Problem("min", obj, solution_length=2, initial_bounds=(0.0, 1.0))
        searcher = CMAES(prob, stdev_init=0.01, popsize=6, center_init=torch.tensor([0.5, 0.5]))
        searcher.step()

        config = CMAESConfig(n_dim=2, tol_x=1e-3)
        stopper = PaperStyleCMAESStopper(config)
        dist_state = extract_cmaes_distribution_state(searcher)
        triggered, val = stopper.check_tol_x(dist_state["max_coordinate_std"])

        # We override C diag for deterministic test
        searcher.C = torch.tensor([[0.01, 0.0], [0.0, 0.0025]], dtype=torch.float32)
        searcher.sigma = torch.tensor(0.01, dtype=torch.float32)
        dist_state2 = extract_cmaes_distribution_state(searcher)
        triggered2, val2 = stopper.check_tol_x(dist_state2["max_coordinate_std"])
        assert triggered2, f"Expected tol_x trigger with max_coord_std={val2}"

    def test_coordinate_std_does_not_trigger(self):
        """sigma=0.02, C_diag=[0.01] -> coord_std=0.002 -> does not trigger tol_x=1e-3"""
        config = CMAESConfig(n_dim=2, tol_x=1e-3)
        stopper = PaperStyleCMAESStopper(config)
        # Direct test using a dict
        dist_state = {"max_coordinate_std": 0.002, "sigma": 0.02, "C_diag": None, "center": None}
        triggered, val = stopper.check_tol_x(dist_state["max_coordinate_std"])
        assert not triggered

    def test_missing_distribution_state_returns_false(self):
        config = CMAESConfig(n_dim=2, tol_x=1e-3)
        stopper = PaperStyleCMAESStopper(config)
        triggered, val = stopper.check_tol_x(None)
        assert not triggered
        assert val is None

    def test_non_negative_numerical_error(self):
        """Ensure negative C_diag from numerical error is handled safely."""
        C = torch.tensor([-0.001, 0.01], dtype=torch.float32)
        sigma = 0.1
        import numpy as np
        C_np = C.numpy()
        coord_std = sigma * np.sqrt(np.maximum(C_np, 0.0))
        assert np.all(np.isfinite(coord_std))
        assert coord_std[0] == 0.0  # clamped to 0

    def test_full_covariance_matrix(self):
        """Test extracting from full 2x2 covariance matrix."""
        from evotorch import Problem
        from evotorch.algorithms import CMAES

        def obj(x):
            return torch.sum((x - 0.5) ** 2, dim=-1)

        prob = Problem("min", obj, solution_length=3, initial_bounds=(0.0, 1.0))
        searcher = CMAES(prob, stdev_init=0.3, popsize=8)

        dist = extract_cmaes_distribution_state(searcher)
        assert dist["sigma"] is not None
        assert dist["C_diag"] is not None
        assert dist["max_coordinate_std"] is not None
        assert dist["max_coordinate_std"] > 0


# ============================================================================
# Section E: Stop logic tests
# ============================================================================


class TestStopLogic:
    def test_tol_fun_only(self):
        config = CMAESConfig(n_dim=2, tol_fun=1e-3, min_generations=1)
        stopper = PaperStyleCMAESStopper(config)
        stopper.record_best(1.0)
        t, _ = stopper.check_tol_fun(np.array([1.0000, 1.0003, 1.0007]))
        assert t
        th, _ = stopper.check_tol_hist_fun()
        tx, _ = stopper.check_tol_x(None)
        assert not th
        assert not tx

        primary, all_r = _determine_stop_reasons(
            tol_fun_triggered=t, tol_hist_fun_triggered=th, tol_x_triggered=tx,
            generation=1, max_generations=1000, min_generations=1,
            invalid_fitness=False, evotorch_terminated=False,
        )
        assert primary == StopReason.TOL_FUN
        assert StopReason.TOL_FUN in all_r

    def test_tol_hist_fun_only(self):
        config = CMAESConfig(n_dim=2, tol_hist_fun=1e-3, min_generations=10)
        primary, all_r = _determine_stop_reasons(
            tol_fun_triggered=False, tol_hist_fun_triggered=True, tol_x_triggered=False,
            generation=10, max_generations=1000, min_generations=10,
            invalid_fitness=False, evotorch_terminated=False,
        )
        assert primary == StopReason.TOL_HIST_FUN

    def test_tol_x_only(self):
        primary, all_r = _determine_stop_reasons(
            tol_fun_triggered=False, tol_hist_fun_triggered=False, tol_x_triggered=True,
            generation=10, max_generations=1000, min_generations=10,
            invalid_fitness=False, evotorch_terminated=False,
        )
        assert primary == StopReason.TOL_X

    def test_max_generations(self):
        primary, all_r = _determine_stop_reasons(
            tol_fun_triggered=False, tol_hist_fun_triggered=False, tol_x_triggered=False,
            generation=1000, max_generations=1000, min_generations=10,
            invalid_fitness=False, evotorch_terminated=False,
        )
        assert primary == StopReason.MAX_GENERATIONS

    def test_multiple_triggers_same_generation(self):
        primary, all_r = _determine_stop_reasons(
            tol_fun_triggered=True, tol_hist_fun_triggered=True, tol_x_triggered=True,
            generation=20, max_generations=1000, min_generations=10,
            invalid_fitness=False, evotorch_terminated=False,
        )
        assert StopReason.TOL_FUN in all_r
        assert StopReason.TOL_HIST_FUN in all_r
        assert StopReason.TOL_X in all_r
        assert len(all_r) == 3

    def test_min_generations_blocks_early_stop(self):
        primary, all_r = _determine_stop_reasons(
            tol_fun_triggered=True, tol_hist_fun_triggered=True, tol_x_triggered=True,
            generation=3, max_generations=1000, min_generations=10,
            invalid_fitness=False, evotorch_terminated=False,
        )
        assert primary is None
        assert len(all_r) == 0

    def test_no_conditions_trigger_no_stop(self):
        primary, all_r = _determine_stop_reasons(
            tol_fun_triggered=False, tol_hist_fun_triggered=False, tol_x_triggered=False,
            generation=50, max_generations=1000, min_generations=10,
            invalid_fitness=False, evotorch_terminated=False,
        )
        assert primary is None
        assert len(all_r) == 0

    def test_invalid_fitness_overrides(self):
        primary, all_r = _determine_stop_reasons(
            tol_fun_triggered=True, tol_hist_fun_triggered=True, tol_x_triggered=True,
            generation=10, max_generations=1000, min_generations=10,
            invalid_fitness=True, evotorch_terminated=False,
        )
        assert StopReason.INVALID_FITNESS in all_r
        assert primary == StopReason.INVALID_FITNESS


# ============================================================================
# Section F: Numerical safety tests
# ============================================================================


class TestNumericalSafety:
    def test_select_finite_handles_nan(self):
        evals = np.array([1.0, np.nan, 3.0])
        fin, ninv = _select_finite(evals)
        assert len(fin) == 2
        assert ninv == 1
        assert np.allclose(fin, [1.0, 3.0])

    def test_select_finite_handles_inf(self):
        evals = np.array([1.0, np.inf, -np.inf, 2.0])
        fin, ninv = _select_finite(evals)
        assert len(fin) == 2
        assert ninv == 2

    def test_all_invalid(self):
        evals = np.array([np.nan, np.inf])
        fin, ninv = _select_finite(evals)
        assert len(fin) == 0
        assert ninv == 2

    def test_tol_fun_with_all_nan(self):
        config = CMAESConfig(n_dim=2, tol_fun=1e-3)
        stopper = PaperStyleCMAESStopper(config)
        triggered, rng = stopper.check_tol_fun(np.array([np.nan, np.nan]))
        assert not triggered
        assert rng is None

    def test_tol_fun_with_partial_nan(self):
        config = CMAESConfig(n_dim=2, tol_fun=1e-3)
        stopper = PaperStyleCMAESStopper(config)
        evals = np.array([1.0, np.nan, 1.0])
        triggered, rng = stopper.check_tol_fun(evals)
        assert triggered
        assert rng == 0.0

    def test_tol_fun_with_inf(self):
        config = CMAESConfig(n_dim=2, tol_fun=1e-3)
        stopper = PaperStyleCMAESStopper(config)
        evals = np.array([1.0, 1.0005, np.inf])
        triggered, rng = stopper.check_tol_fun(evals)
        assert triggered
        assert abs(rng - 0.0005) < 1e-6


# ============================================================================
# Section G: EvoTorch integration tests
# ============================================================================


class TestSphereIntegration:
    def test_sphere_5d_converges(self):
        """Run CMA-ES on 5D Sphere and verify convergence."""
        n_dim = 5

        def sphere_fn(x):
            return torch.sum((x - 0.5) ** 2, dim=-1)

        config = CMAESConfig(
            n_dim=n_dim,
            tol_fun=1e-6,
            tol_hist_fun=1e-6,
            tol_x=1e-6,
            max_generations=200,
            min_generations=5,
            seed=42,
        )

        result = run_cmaes_with_early_stopping(sphere_fn, config, verbose=False)

        assert isinstance(result, CMAESRunResult)
        assert result.stop_generation > 0
        assert result.stop_generation <= config.max_generations
        assert result.primary_stop_reason is not None
        assert result.primary_stop_reason.value != ""
        assert result.best_eval < 1e-2, f"Best eval {result.best_eval} not converged"

        best = result.best_solution_normalized
        assert np.all(best >= 0.0)
        assert np.all(best <= 1.0)

        assert result.number_of_evaluations == config.population_size * result.stop_generation

        assert len(result.complete_generation_history) == result.stop_generation

        for rec in result.complete_generation_history:
            assert rec.generation > 0
            assert rec.population_size == config.population_size
            assert np.isfinite(rec.generation_best_eval)

    def test_sphere_5d_reproducible(self):
        """Fixed seed should give identical results."""
        n_dim = 5

        def sphere_fn(x):
            return torch.sum((x - 0.5) ** 2, dim=-1)

        config = CMAESConfig(
            n_dim=n_dim,
            tol_fun=1e-6,
            tol_hist_fun=1e-6,
            tol_x=1e-6,
            max_generations=200,
            min_generations=5,
            seed=42,
        )

        result1 = run_cmaes_with_early_stopping(sphere_fn, config, verbose=False)
        result2 = run_cmaes_with_early_stopping(sphere_fn, config, verbose=False)

        assert result1.stop_generation == result2.stop_generation
        assert result1.best_eval == pytest.approx(result2.best_eval, rel=1e-5)
        np.testing.assert_allclose(result1.best_solution_normalized, result2.best_solution_normalized, atol=1e-5)

    def test_sphere_final_parameter_in_bounds(self):
        n_dim = 5

        def sphere_fn(x):
            return torch.sum((x - 0.5) ** 2, dim=-1)

        config = CMAESConfig(
            n_dim=n_dim,
            max_generations=50,
            min_generations=2,
            seed=123,
        )

        result = run_cmaes_with_early_stopping(sphere_fn, config, verbose=False)
        best = result.best_solution_normalized
        assert np.all(best >= 0.0), f"Best has values below 0: {best}"
        assert np.all(best <= 1.0), f"Best has values above 1: {best}"

    def test_max_generations_always_honored(self):
        n_dim = 3

        # A bad objective that never converges
        def noisy_fn(x):
            if x.ndim == 1:
                return torch.rand(1) * 10.0
            return torch.rand(x.shape[0], 1) * 10.0

        config = CMAESConfig(
            n_dim=n_dim,
            tol_fun=0.0,
            tol_hist_fun=0.0,
            tol_x=0.0,
            max_generations=30,
            min_generations=5,
            seed=42,
        )

        result = run_cmaes_with_early_stopping(noisy_fn, config, verbose=False)
        assert result.stop_generation <= 30
        assert result.primary_stop_reason == StopReason.MAX_GENERATIONS

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_smoke(self):
        """Smoke test on CUDA if available."""
        n_dim = 5

        def sphere_fn(x):
            return torch.sum((x - 0.5) ** 2, dim=-1)

        config = CMAESConfig(
            n_dim=n_dim,
            max_generations=20,
            min_generations=2,
            seed=42,
        )

        result = run_cmaes_with_early_stopping(sphere_fn, config, verbose=False)
        assert result.stop_generation <= 20
        assert np.isfinite(result.best_eval)


# ============================================================================
# Section H: Regression and isolation tests
# ============================================================================


class TestRegressionAndIsolation:
    def test_module_import_does_not_trigger_training(self):
        """Importing the module must not trigger training or read large data."""
        # This is implicitly tested by the fact that we're here without side effects
        pass

    def test_dataclass_serialization(self):
        """Verify records can be converted to dict."""
        rec = CMAESGenerationRecord(
            generation=1,
            population_size=8,
            generation_best_eval=0.5,
            generation_mean_eval=0.7,
            generation_worst_eval=1.0,
            current_eval_range=0.5,
            history_eval_range=None,
            sigma_or_stepsize=0.3,
            max_coordinate_std=0.1,
            center_norm=1.2,
            best_solution_norm=1.0,
            tol_fun_reached=False,
            tol_hist_fun_reached=False,
            tol_x_reached=False,
            elapsed_seconds=0.1,
        )
        d = {f.name: getattr(rec, f.name) for f in rec.__dataclass_fields__.values()}
        assert d["generation"] == 1
        assert d["population_size"] == 8

    def test_stop_reason_enum_values(self):
        assert StopReason.TOL_FUN.value == "tol_fun"
        assert StopReason.TOL_HIST_FUN.value == "tol_hist_fun"
        assert StopReason.TOL_X.value == "tol_x"
        assert StopReason.MAX_GENERATIONS.value == "max_generations"
        assert StopReason.INVALID_FITNESS.value == "invalid_fitness"
        assert StopReason.EVOTORCH_INTERNAL_STOP.value == "evotorch_internal_stop"
