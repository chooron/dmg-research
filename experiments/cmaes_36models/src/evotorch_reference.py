"""Single-problem EvoTorch reference path used only for regression testing.

EvoTorch 0.6.1 deliberately rejects bounded Problems; callers must pass the same
latent objective used by `BatchedCMAES`, not physical parameters.
"""
from __future__ import annotations
import torch
from evotorch import Problem
from evotorch.algorithms import CMAES


def run_reference(objective, dimension: int, popsize: int, generations: int, center: torch.Tensor, *, seed: int, device: str = "cpu") -> torch.Tensor:
    def evaluate(values: torch.Tensor) -> torch.Tensor:
        return objective(values)
    # `initial_bounds` only seeds EvoTorch's initial population; CMAES still sees
    # the unbounded latent Problem required by `ensure_unbounded()`.
    problem = Problem("max", evaluate, solution_length=dimension, dtype=torch.float64, eval_dtype=torch.float64,
                      device=device, vectorized=True, seed=seed, initial_bounds=(-1.0, 1.0))
    searcher = CMAES(problem, stdev_init=0.10, popsize=popsize, center_init=center, active=True, separable=False)
    for _ in range(generations): searcher.step()
    return searcher.status["best"].values.clone()
