from typing import Any

import numpy as np
import torch
from evotorch import Problem
from evotorch.algorithms import CMAES

from .base import OptimizerAdapter
from .registry import register


@register("CMAES")
class CMAESAdapter(OptimizerAdapter):
    def __init__(self):
        self.cmaes = None
        self.best_candidate = None
        self.best_fitness = -float("inf")
        self.generation = 0
        self.dimension = None
        self.latest_zs = None
        self.latest_ys = None
        self.latest_xs = None

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
            dtype=self.dtype,
            eval_dtype=torch.float64,
            vectorized=True,
            device=device,
        )

        # We need center_init to be a tensor
        c_init = torch.tensor(center_init, dtype=self.dtype, device=device)
        self.cmaes = CMAES(
            self.problem, popsize=population, stdev_init=stdev_init, center_init=c_init
        )

        self.best_candidate = None
        self.best_fitness = -float("inf")
        self.generation = 0

    def ask(self) -> np.ndarray:
        self.latest_zs, self.latest_ys, self.latest_xs = (
            self.cmaes.sample_distribution()
        )
        return self.latest_xs.cpu().numpy().astype(self.np_dtype)

    def tell(self, fitness: np.ndarray) -> None:
        fit_t = torch.tensor(fitness, dtype=torch.float64, device=self.device)

        max_idx = torch.argmax(fit_t)
        max_fit = fit_t[max_idx].item()
        if max_fit > self.best_fitness:
            self.best_fitness = max_fit
            self.best_candidate = (
                self.latest_xs[max_idx].clone().cpu().numpy().astype(self.np_dtype)
            )

        # CMAES weight assignment (replaces problem.evaluate + get_population_weights)
        indices = torch.argsort(fit_t, descending=True)
        ranks = torch.zeros_like(indices)
        ranks[indices] = torch.arange(
            self.cmaes.popsize, dtype=indices.dtype, device=indices.device
        )
        assigned_weights = self.cmaes.weights[ranks]

        zs, ys = self.latest_zs, self.latest_ys
        local_m_displacement, shaped_m_displacement = self.cmaes.update_m(
            zs, ys, assigned_weights
        )
        self.cmaes.update_p_sigma(local_m_displacement)
        self.cmaes.update_sigma()

        from evotorch.algorithms.cmaes import _h_sig, _limit_stdev

        h_sig = _h_sig(self.cmaes.p_sigma, self.cmaes.c_sigma, self.cmaes._steps_count)
        self.cmaes.update_p_c(shaped_m_displacement, h_sig)
        self.cmaes.update_C(zs, ys, assigned_weights, h_sig)

        if self.cmaes.stdev_min is not None or self.cmaes.stdev_max is not None:
            self.cmaes.C = _limit_stdev(
                self.cmaes.sigma,
                self.cmaes.C,
                self.cmaes.stdev_min,
                self.cmaes.stdev_max,
            )

        if (self.cmaes._steps_count + 1) % self.cmaes.decompose_C_freq == 0:
            try:
                self.cmaes.decompose_C()
            except Exception:
                # Add diagonal jitter if C matrix becomes ill-conditioned / non-positive-definite
                eps = 1e-8 * torch.eye(
                    self.cmaes.solution_length,
                    dtype=self.cmaes.C.dtype,
                    device=self.cmaes.C.device,
                )
                self.cmaes.C = self.cmaes.C + eps
                try:
                    self.cmaes.decompose_C()
                except Exception:
                    # Reset C to identity if jitter fails
                    self.cmaes.C = torch.eye(
                        self.cmaes.solution_length,
                        dtype=self.cmaes.C.dtype,
                        device=self.cmaes.C.device,
                    )
                    self.cmaes.decompose_C()

        self.cmaes._steps_count += 1
        self.generation += 1

    def get_center(self) -> np.ndarray:
        return self.cmaes.m.cpu().numpy().astype(self.np_dtype)

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
            "m": self.cmaes.m.cpu().numpy().tolist(),
            "C": self.cmaes.C.cpu().numpy().tolist(),
            "sigma": self.cmaes.sigma.cpu().numpy().tolist(),
            "steps_count": self.cmaes._steps_count,
            "p_c": self.cmaes.p_c.cpu().numpy().tolist(),
            "p_sigma": self.cmaes.p_sigma.cpu().numpy().tolist(),
        }
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.generation = state["generation"]
        self.best_fitness = state["best_fitness"]
        if state["best_candidate"] is not None:
            self.best_candidate = np.array(state["best_candidate"], dtype=self.np_dtype)

        self.cmaes.m.copy_(
            torch.tensor(state["m"], dtype=self.dtype, device=self.device)
        )
        self.cmaes.C.copy_(
            torch.tensor(state["C"], dtype=self.dtype, device=self.device)
        )
        self.cmaes.sigma.copy_(
            torch.tensor(state["sigma"], dtype=self.dtype, device=self.device)
        )
        self.cmaes._steps_count = state["steps_count"]
        # EvoTorch initializes these paths as Python 0.0 until the first
        # update in some releases.  Assignment, rather than ``copy_``, makes
        # an atomic checkpoint from a later generation resumable in both
        # initialization states.
        self.cmaes.p_c = torch.tensor(
            state["p_c"], dtype=self.dtype, device=self.device
        )
        self.cmaes.p_sigma = torch.tensor(
            state["p_sigma"], dtype=self.dtype, device=self.device
        )
        self.cmaes.decompose_C()

    def get_diagnostics(self) -> dict:
        return {}

    @property
    def name(self) -> str:
        return "CMAES"

    @property
    def supports_exact_resume(self) -> bool:
        return False
