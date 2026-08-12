import numpy as np
import torch
from typing import Any
from evotorch.algorithms import PGPE
from evotorch import Problem
from .base import OptimizerAdapter
from .registry import register

@register("PGPE")
class PGPEAdapter(OptimizerAdapter):
    def __init__(self):
        self.pgpe = None
        self.best_candidate = None
        self.best_fitness = -float('inf')
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
        self.np_dtype = np.float64 if str(self.dtype) == 'torch.float64' else np.float32
        
        torch.manual_seed(seed)
        
        def dummy_fn(x):
            return torch.zeros(x.shape[0], dtype=torch.float64)
            
        self.problem = Problem(
            'max', 
            dummy_fn, 
            solution_length=dimension, 
            initial_bounds=(0.0, 1.0),
            dtype=self.dtype,
            eval_dtype=torch.float64, 
            vectorized=True, 
            device=device
        )
        
        c_init = torch.tensor(center_init, dtype=self.dtype, device=device)
        clr = config.get('center_learning_rate', 0.15)
        slr = config.get('stdev_learning_rate', 0.1)
        self.pgpe = PGPE(
            self.problem, 
            popsize=population, 
            center_learning_rate=clr,
            stdev_learning_rate=slr,
            stdev_init=stdev_init,
            center_init=c_init
        )
        
        self.best_candidate = None
        self.best_fitness = -float('inf')
        self.generation = 0
        self.latest_samples = None

    def ask(self) -> np.ndarray:
        self.latest_samples = self.pgpe._distribution.sample(
            num_solutions=self.population, 
            generator=self.problem
        )
        return self.latest_samples.cpu().numpy().astype(self.np_dtype)

    def tell(self, fitness: np.ndarray) -> None:
        fit_t = torch.tensor(fitness, dtype=torch.float64, device=self.device)
        
        max_idx = torch.argmax(fit_t)
        max_fit = fit_t[max_idx].item()
        if max_fit > self.best_fitness:
            self.best_fitness = max_fit
            self.best_candidate = self.latest_samples[max_idx].clone().cpu().numpy().astype(self.np_dtype)
            
        grads = self.pgpe._distribution.compute_gradients(
            self.latest_samples, 
            fit_t, 
            objective_sense='max',
            ranking_method=self.pgpe._ranking_method
        )
        self.pgpe._update_distribution(grads)
        self.generation += 1

    def get_center(self) -> np.ndarray:
        if hasattr(self.pgpe._distribution, 'mu'):
            return self.pgpe._distribution.mu.cpu().numpy().astype(self.np_dtype)
        elif hasattr(self.pgpe._distribution, 'center'):
            return self.pgpe._distribution.center.cpu().numpy().astype(self.np_dtype)
        return np.zeros(self.dimension)

    def get_best(self) -> tuple[np.ndarray, float]:
        if self.best_candidate is None:
            return np.zeros(self.dimension), -float('inf')
        return self.best_candidate, self.best_fitness

    def state_dict(self) -> dict[str, Any]:
        state = {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'best_candidate': self.best_candidate.tolist() if self.best_candidate is not None else None,
        }
        if hasattr(self.pgpe._distribution, 'mu'):
            state['mu'] = self.pgpe._distribution.mu.cpu().numpy().tolist()
        if hasattr(self.pgpe._distribution, 'sigma'):
            state['sigma'] = self.pgpe._distribution.sigma.cpu().numpy().tolist()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.generation = state['generation']
        self.best_fitness = state['best_fitness']
        if state['best_candidate'] is not None:
            self.best_candidate = np.array(state['best_candidate'], dtype=self.np_dtype)
            
        if 'mu' in state and hasattr(self.pgpe._distribution, 'mu'):
            self.pgpe._distribution.mu.copy_(torch.tensor(state['mu'], dtype=self.dtype, device=self.device))
        if 'sigma' in state and hasattr(self.pgpe._distribution, 'sigma'):
            self.pgpe._distribution.sigma.copy_(torch.tensor(state['sigma'], dtype=self.dtype, device=self.device))

    def get_diagnostics(self) -> dict:
        return {}

    @property
    def name(self) -> str:
        return "PGPE"

    @property
    def supports_exact_resume(self) -> bool:
        return False
