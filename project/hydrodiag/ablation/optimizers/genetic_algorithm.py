from typing import Any

import numpy as np
import torch
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.operators import PolynomialMutation, SimulatedBinaryCrossOver

from .base import OptimizerAdapter
from .registry import register


@register("GeneticAlgorithm")
class GAAdapter(OptimizerAdapter):
    def __init__(self):
        self.ga = None
        self.best_candidate = None
        self.best_fitness = -float("inf")
        self.generation = 0
        self.dimension = None

    def initialize(
        self,
        dimension: int,
        population: int,
        center_init: np.ndarray,
        stdev_init: float,
        seed: int,
        device: str,
        dtype: str,
        config: dict,
    ) -> None:
        self.dimension = dimension
        self.population = population
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.np_dtype = np.float64 if str(self.dtype) == "torch.float64" else np.float32

        torch.manual_seed(seed)

        def dummy_fn(x):
            return torch.zeros(x.shape[0], dtype=torch.float64)

        self.problem = Problem(
            "max",
            dummy_fn,
            solution_length=dimension,
            initial_bounds=(0.0, 1.0),
            bounds=(0.0, 1.0),
            dtype=self.dtype,
            eval_dtype=torch.float64,
            vectorized=True,
            device=device,
        )

        operators = [
            SimulatedBinaryCrossOver(
                self.problem, tournament_size=4, cross_over_rate=1.0, eta=8
            ),
            PolynomialMutation(self.problem, eta=20),
        ]

        self.ga = GeneticAlgorithm(
            self.problem, popsize=population, operators=operators, elitist=True
        )

        # Override initial population center if requested (GA initializes uniformly)
        # We can just inject center_init as one of the solutions, or shift it.
        c_init = torch.tensor(center_init, dtype=self.dtype, device=device)
        self.ga.population.access_values()[0].copy_(c_init)

        self.best_candidate = None
        self.best_fitness = -float("inf")
        self.generation = 0
        self.latest_batch = None
        self.is_first = True

    def ask(self) -> np.ndarray:
        if self.is_first:
            self.latest_batch = self.ga.population
        else:
            from evotorch.algorithms.ga import _use_operators

            self.latest_batch = _use_operators(self.ga.population, self.ga._operators)

        return self.latest_batch.values.cpu().numpy().astype(self.np_dtype)

    def tell(self, fitness: np.ndarray) -> None:
        fit_t = torch.tensor(fitness, dtype=torch.float64, device=self.device)
        self.latest_batch.set_evals(fit_t)

        max_idx = torch.argmax(fit_t)
        max_fit = fit_t[max_idx].item()
        if max_fit > self.best_fitness:
            self.best_fitness = max_fit
            self.best_candidate = (
                self.latest_batch.values[max_idx]
                .clone()
                .cpu()
                .numpy()
                .astype(self.np_dtype)
            )

        if self.is_first:
            self.ga._population = self.latest_batch
            self.is_first = False
        else:
            from evotorch.core import SolutionBatch

            extended = SolutionBatch.cat([self.ga.population, self.latest_batch])
            self.ga._population = extended.take_best(self.ga._popsize)

        self.generation += 1

    def get_center(self) -> np.ndarray:
        # GA doesn't have a center, we just return the mean of the population
        if self.ga.population is None:
            return np.zeros(self.dimension)
        return self.ga.population.values.mean(dim=0).cpu().numpy().astype(self.np_dtype)

    def get_best(self) -> tuple[np.ndarray, float]:
        if self.best_candidate is None:
            return np.zeros(self.dimension), -float("inf")
        return self.best_candidate, self.best_fitness

    def state_dict(self) -> dict[str, Any]:
        state = {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "best_candidate": self.best_candidate.tolist()
            if self.best_candidate is not None
            else None,
            "population": self.ga.population.values.cpu().numpy().tolist()
            if self.ga.population is not None
            else None,
            "evals": self.ga.population.evals.cpu().numpy().tolist()
            if (self.ga.population is not None and self.ga.population.evals is not None)
            else None,
            "is_first": self.is_first,
        }
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.generation = state["generation"]
        self.best_fitness = state["best_fitness"]
        if state["best_candidate"] is not None:
            self.best_candidate = np.array(state["best_candidate"], dtype=self.np_dtype)
        self.is_first = state["is_first"]

        if state["population"] is not None:
            self.ga.population.set_values(
                torch.tensor(state["population"], dtype=self.dtype, device=self.device)
            )
        if state["evals"] is not None:
            self.ga.population.set_evals(
                torch.tensor(state["evals"], dtype=torch.float64, device=self.device)
            )

    def get_diagnostics(self) -> dict:
        return {}

    @property
    def name(self) -> str:
        return "GeneticAlgorithm"

    @property
    def supports_exact_resume(self) -> bool:
        return False
