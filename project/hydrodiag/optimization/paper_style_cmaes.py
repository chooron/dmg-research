"""CMA-ES training controller with paper-style early stopping rules.

Implements a minimal verifiable outer training loop for EvoTorch CMAES
that checks objective function convergence, history convergence, and
search distribution contraction after each generation.

Reference:
  "A Brief Analysis of Conceptual Model Structure Uncertainty Using
   36 Models and 559 Catchments"
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from evotorch import Problem
from evotorch.algorithms import CMAES


class StopReason(str, Enum):
    TOL_FUN = "tol_fun"
    TOL_HIST_FUN = "tol_hist_fun"
    TOL_X = "tol_x"
    MAX_GENERATIONS = "max_generations"
    INVALID_FITNESS = "invalid_fitness"
    EVOTORCH_INTERNAL_STOP = "evotorch_internal_stop"


_STOP_REASON_PRIORITY: Dict[StopReason, int] = {
    StopReason.INVALID_FITNESS: 0,
    StopReason.EVOTORCH_INTERNAL_STOP: 1,
    StopReason.TOL_X: 2,
    StopReason.TOL_HIST_FUN: 3,
    StopReason.TOL_FUN: 4,
    StopReason.MAX_GENERATIONS: 5,
}


def default_population_size(n_dim: int) -> int:
    """Compute population size = 4 + floor(3 * log(n_dim)).
    Raises ValueError if n_dim <= 0.
    """
    if n_dim <= 0:
        raise ValueError(f"n_dim must be positive, got {n_dim}")
    return 4 + int(math.floor(3.0 * math.log(n_dim)))


def normalized_to_physical(
    x_normalized: torch.Tensor,
    lower_bounds: torch.Tensor,
    upper_bounds: torch.Tensor,
) -> torch.Tensor:
    """Map normalized [0, 1] parameters to physical scale.
    theta_i = lower_i + x_i * (upper_i - lower_i)
    """
    return lower_bounds + x_normalized * (upper_bounds - lower_bounds)


def physical_to_normalized(
    x_physical: torch.Tensor,
    lower_bounds: torch.Tensor,
    upper_bounds: torch.Tensor,
) -> torch.Tensor:
    """Map physical parameters to normalized [0, 1] scale.
    x_i = (theta_i - lower_i) / (upper_i - lower_i)
    """
    return (x_physical - lower_bounds) / (upper_bounds - lower_bounds)


def _extract_cov_diag(C: torch.Tensor) -> torch.Tensor:
    """Extract diagonal of covariance/shape matrix C.
    Handles both dense d×d matrices and separable diagonal vectors.
    """
    if C.ndim == 1:
        return C.detach().clone()
    elif C.ndim == 2:
        return C.diag().detach().clone()
    else:
        raise ValueError(f"Unexpected C shape: {C.shape}, expected 1-D or 2-D tensor")


def extract_cmaes_distribution_state(searcher: CMAES) -> Dict[str, Any]:
    """Extract distribution state from a CMAES searcher.
    Returns a dict with:
        sigma: float, global step size
        C_diag: 1-D tensor, diagonal of covariance shape matrix
        center: 1-D tensor, current distribution mean
        max_coordinate_std: float, max(sigma * sqrt(C_ii))
    """
    try:
        sigma = float(searcher.sigma.cpu().item())
    except (AttributeError, RuntimeError):
        sigma = None

    try:
        C_diag = _extract_cov_diag(searcher.C)
    except (AttributeError, RuntimeError):
        C_diag = None

    try:
        center = searcher.m.detach().cpu().clone()
    except (AttributeError, RuntimeError):
        center = None

    max_coordinate_std = None
    if sigma is not None and C_diag is not None:
        coord_std = sigma * np.sqrt(
            np.maximum(C_diag.cpu().numpy().astype(np.float64), 0.0)
        )
        max_coordinate_std = float(np.max(coord_std))

    return {
        "sigma": sigma,
        "C_diag": C_diag,
        "center": center,
        "max_coordinate_std": max_coordinate_std,
    }


def _select_finite(evals: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return finite values and count of non-finite entries."""
    finite_mask = np.isfinite(evals)
    return evals[finite_mask], int((~finite_mask).sum())


@dataclass
class CMAESConfig:
    n_dim: int
    population_size: Optional[int] = None
    initial_center: float = 0.5
    initial_stdev: float = 0.3
    bounds: Tuple[float, float] = (0.0, 1.0)
    max_generations: int = 1000
    min_generations: int = 10
    history_window: int = 10
    tol_fun: float = 1e-3
    tol_hist_fun: float = 1e-3
    tol_x: float = 1e-3
    seed: Optional[int] = None

    def __post_init__(self):
        if self.population_size is None:
            self.population_size = default_population_size(self.n_dim)
        if self.n_dim <= 0:
            raise ValueError(f"n_dim must be positive, got {self.n_dim}")
        if self.min_generations < 0:
            raise ValueError(
                f"min_generations must be >= 0, got {self.min_generations}"
            )
        if self.max_generations < self.min_generations:
            raise ValueError(
                f"max_generations ({self.max_generations}) < min_generations ({self.min_generations})"
            )


@dataclass
class CMAESGenerationRecord:
    generation: int
    population_size: int
    generation_best_eval: float
    generation_mean_eval: float
    generation_worst_eval: float
    current_eval_range: Optional[float]
    history_eval_range: Optional[float]
    sigma_or_stepsize: Optional[float]
    max_coordinate_std: Optional[float]
    center_norm: Optional[float]
    best_solution_norm: Optional[float]
    tol_fun_reached: bool
    tol_hist_fun_reached: bool
    tol_x_reached: bool
    elapsed_seconds: float


@dataclass
class CMAESRunResult:
    best_solution_normalized: np.ndarray
    best_solution_physical: Optional[np.ndarray]
    best_eval: float
    stop_generation: int
    number_of_evaluations: int
    primary_stop_reason: StopReason
    all_triggered_reasons: List[StopReason]
    final_current_eval_range: Optional[float]
    final_history_eval_range: Optional[float]
    final_max_coordinate_std: Optional[float]
    complete_generation_history: List[CMAESGenerationRecord]


class PaperStyleCMAESStopper:
    """Early-stopping controller that checks tol_fun, tol_hist_fun, and tol_x."""

    def __init__(self, config: CMAESConfig):
        self.config = config
        self._history_best: List[float] = []

    def record_best(self, best_eval: float):
        """Record the best evaluation of the current generation."""
        self._history_best.append(float(best_eval))

    def check_tol_fun(self, evals: np.ndarray) -> Tuple[bool, Optional[float]]:
        """Check if current generation eval range is within tol_fun."""
        finite_evals, _ninvalid = _select_finite(evals)
        if len(finite_evals) == 0:
            return False, None
        current_range = float(np.max(finite_evals) - np.min(finite_evals))
        return current_range <= self.config.tol_fun, current_range

    def check_tol_hist_fun(self) -> Tuple[bool, Optional[float]]:
        """Check if recent history of best evals is within tol_hist_fun.
        Only checks the last history_window best values.
        Returns False if history is shorter than history_window.
        """
        window = self.config.history_window
        if len(self._history_best) < window:
            return False, None
        recent = self._history_best[-window:]
        hist_range = float(max(recent) - min(recent))
        return hist_range <= self.config.tol_hist_fun, hist_range

    def check_tol_x(
        self, max_coordinate_std: Optional[float]
    ) -> Tuple[bool, Optional[float]]:
        """Check if the maximum coordinate-wise standard deviation
        is within tol_x.
        """
        if max_coordinate_std is None:
            return False, None
        return max_coordinate_std <= self.config.tol_x, max_coordinate_std

    def get_history_eval_range(self) -> Optional[float]:
        window = self.config.history_window
        if len(self._history_best) < window:
            return None
        recent = self._history_best[-window:]
        return float(max(recent) - min(recent))

    @property
    def history_length(self) -> int:
        return len(self._history_best)

    def history_best_values(self) -> List[float]:
        return list(self._history_best)


def _determine_stop_reasons(
    tol_fun_triggered: bool,
    tol_hist_fun_triggered: bool,
    tol_x_triggered: bool,
    generation: int,
    max_generations: int,
    min_generations: int,
    invalid_fitness: bool,
    evotorch_terminated: bool,
) -> Tuple[Optional[StopReason], List[StopReason]]:
    """Determine primary and all stop reasons.
    All conditions are checked; any one triggers a stop after min_generations.
    Priority only determines primary_stop_reason for logging.
    """
    all_reasons: List[StopReason] = []

    if invalid_fitness:
        all_reasons.append(StopReason.INVALID_FITNESS)

    if evotorch_terminated:
        all_reasons.append(StopReason.EVOTORCH_INTERNAL_STOP)

    if generation >= max_generations:
        all_reasons.append(StopReason.MAX_GENERATIONS)

    if generation >= min_generations:
        if tol_fun_triggered:
            all_reasons.append(StopReason.TOL_FUN)
        if tol_hist_fun_triggered:
            all_reasons.append(StopReason.TOL_HIST_FUN)
        if tol_x_triggered:
            all_reasons.append(StopReason.TOL_X)

    if not all_reasons:
        return None, []

    primary = min(all_reasons, key=lambda r: _STOP_REASON_PRIORITY.get(r, 99))
    return primary, all_reasons


def run_cmaes_with_early_stopping(
    objective_fn: Callable[[torch.Tensor], torch.Tensor],
    config: CMAESConfig,
    lower_bounds: Optional[np.ndarray] = None,
    upper_bounds: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> CMAESRunResult:
    """Run CMA-ES with paper-style early stopping.

    Args:
        objective_fn: Function mapping (batch, n_dim) -> (batch,) torch tensor
                      of objective values to minimize. Operates on
                      normalized parameters in [0, 1].
        config: CMAESConfig with all hyperparameters.
        lower_bounds: Optional physical-scale lower bounds for output mapping.
        upper_bounds: Optional physical-scale upper bounds for output mapping.
        verbose: Print per-generation diagnostics.

    Returns:
        CMAESRunResult with complete trace.
    """
    lo = config.bounds[0]
    hi = config.bounds[1]

    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    problem = Problem(
        "min",
        objective_fn,
        solution_length=config.n_dim,
        initial_bounds=(lo, hi),
    )

    init_center = torch.full((config.n_dim,), config.initial_center)
    searcher = CMAES(
        problem,
        stdev_init=config.initial_stdev,
        popsize=config.population_size,
        center_init=init_center,
    )

    stopper = PaperStyleCMAESStopper(config)
    generation_history: List[CMAESGenerationRecord] = []
    stop_reason: Optional[StopReason] = None
    all_reasons: List[StopReason] = []
    best_overall_eval = float("inf")
    best_overall_solution_norm = None

    t_start = time.monotonic()

    for gen in range(1, config.max_generations + 1):
        t_gen_start = time.monotonic()

        searcher.step()

        pop_values = searcher.population.values.detach().cpu().numpy()
        pop_evals_np = searcher.population.evals.detach().cpu().numpy().flatten()

        finite_evals, n_invalid = _select_finite(pop_evals_np)

        # Handle all-invalid generation
        if len(finite_evals) == 0:
            stop_reason = StopReason.INVALID_FITNESS
            all_reasons = [StopReason.INVALID_FITNESS]
            if verbose:
                print(f"  Gen {gen}: ALL evals invalid (NaN/Inf) - stopping")
            break

        gen_best = float(np.min(finite_evals))
        gen_mean = float(np.mean(finite_evals))
        gen_worst = float(np.max(finite_evals))

        stopper.record_best(gen_best)

        dist_state = extract_cmaes_distribution_state(searcher)

        tol_fun_triggered, cur_eval_range = stopper.check_tol_fun(finite_evals)
        tol_hist_fun_triggered, hist_eval_range = stopper.check_tol_hist_fun()
        tol_x_triggered, _ = stopper.check_tol_x(dist_state["max_coordinate_std"])

        # Track best solution
        if gen_best < best_overall_eval:
            best_overall_eval = gen_best
            best_idx = int(np.argmin(finite_evals))
            finite_idx = np.where(np.isfinite(pop_evals_np))[0][best_idx]
            best_overall_solution_norm = pop_values[finite_idx].copy()

        # Norms for diagnostics
        center_norm = None
        if dist_state["center"] is not None:
            center_norm = float(np.linalg.norm(dist_state["center"].cpu().numpy()))

        best_sol_norm = None
        if best_overall_solution_norm is not None:
            best_sol_norm = float(np.linalg.norm(best_overall_solution_norm))

        record = CMAESGenerationRecord(
            generation=gen,
            population_size=config.population_size,
            generation_best_eval=gen_best,
            generation_mean_eval=gen_mean,
            generation_worst_eval=gen_worst,
            current_eval_range=cur_eval_range,
            history_eval_range=hist_eval_range,
            sigma_or_stepsize=dist_state["sigma"],
            max_coordinate_std=dist_state["max_coordinate_std"],
            center_norm=center_norm,
            best_solution_norm=best_sol_norm,
            tol_fun_reached=tol_fun_triggered,
            tol_hist_fun_reached=tol_hist_fun_triggered,
            tol_x_reached=tol_x_triggered,
            elapsed_seconds=time.monotonic() - t_start,
        )
        generation_history.append(record)

        if verbose:
            print(
                f"  Gen {gen:4d} | best={gen_best:.6e} | sigma={dist_state['sigma']:.4e} "
                f"| max_std={dist_state['max_coordinate_std']:.4e} "
                f"| cur_range={cur_eval_range:.4e} | hist_range={hist_eval_range or 'N/A'}"
            )

        # Determine stop
        invalid_fitness = n_invalid == len(pop_evals_np)
        evotorch_terminated = bool(searcher.is_terminated)

        primary, all_trig = _determine_stop_reasons(
            tol_fun_triggered=tol_fun_triggered,
            tol_hist_fun_triggered=tol_hist_fun_triggered,
            tol_x_triggered=tol_x_triggered,
            generation=gen,
            max_generations=config.max_generations,
            min_generations=config.min_generations,
            invalid_fitness=invalid_fitness,
            evotorch_terminated=evotorch_terminated,
        )

        if primary is not None:
            stop_reason = primary
            all_reasons = all_trig
            if verbose:
                print(
                    f"  STOPPING at gen {gen}: primary={stop_reason.value}, "
                    f"all={[r.value for r in all_reasons]}"
                )
            break

    # If loop exhausted without explicit stop, set max_generations
    if stop_reason is None:
        stop_reason = StopReason.MAX_GENERATIONS
        all_reasons = [StopReason.MAX_GENERATIONS]

    # Fill in best solution if none found (edge case)
    if best_overall_solution_norm is None:
        best_overall_solution_norm = (
            searcher.population.values.detach().cpu().numpy()[0].copy()
        )
        best_overall_eval = float(
            searcher.population.evals.detach().cpu().numpy().flatten()[0]
        )

    # Physical-scale mapping
    best_solution_physical = None
    if lower_bounds is not None and upper_bounds is not None:
        best_solution_physical = lower_bounds + best_overall_solution_norm * (
            upper_bounds - lower_bounds
        )

    # Compute final diagnostics
    final_cur_range: Optional[float] = None
    final_hist_range: Optional[float] = None
    final_max_std: Optional[float] = None

    if generation_history:
        last_rec = generation_history[-1]
        final_cur_range = last_rec.current_eval_range
        final_hist_range = last_rec.history_eval_range
        final_max_std = last_rec.max_coordinate_std

    total_evals = (
        config.population_size * len(generation_history) if generation_history else 0
    )

    return CMAESRunResult(
        best_solution_normalized=best_overall_solution_norm,
        best_solution_physical=best_solution_physical,
        best_eval=best_overall_eval,
        stop_generation=generation_history[-1].generation if generation_history else 0,
        number_of_evaluations=total_evals,
        primary_stop_reason=stop_reason,
        all_triggered_reasons=all_reasons,
        final_current_eval_range=final_cur_range,
        final_history_eval_range=final_hist_range,
        final_max_coordinate_std=final_max_std,
        complete_generation_history=generation_history,
    )
