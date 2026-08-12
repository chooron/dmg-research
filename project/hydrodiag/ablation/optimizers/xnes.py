from __future__ import annotations
import numpy as np
import torch
from typing import Any
from evotorch.algorithms import XNES
from evotorch import Problem
from .base import OptimizerAdapter
from .registry import register

@register("XNES")
class XNESAdapter(OptimizerAdapter):
    def __init__(self):
        self.xnes = None
        self.best_candidate = None
        self.best_fitness = -float('inf')
        self.generation = 0
        self.ranking_method = 'nes'
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
        
        self.ranking_method = config.get('ranking_method', 'nes')
        
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
        
        self.xnes = XNES(
            self.problem, 
            popsize=population, 
            stdev_init=stdev_init,
            center_init=torch.tensor(center_init, dtype=self.dtype, device=device)
        )
        
        self.best_candidate = None
        self.best_fitness = -float('inf')
        self.generation = 0
        self.latest_samples = None

    def ask(self) -> np.ndarray:
        self.latest_samples = self.xnes._distribution.sample(
            num_solutions=self.population, 
            generator=self.problem
        )
        return self.latest_samples.cpu().numpy().astype(self.np_dtype)

    def tell(self, fitness: np.ndarray) -> None:
        fit_t = torch.tensor(fitness, dtype=torch.float64, device=self.device)
        
        # Update best so far
        max_idx = torch.argmax(fit_t)
        max_fit = fit_t[max_idx].item()
        if max_fit > self.best_fitness:
            self.best_fitness = max_fit
            self.best_candidate = self.latest_samples[max_idx].clone().cpu().numpy().astype(self.np_dtype)
            
        grads = self.xnes._distribution.compute_gradients(
            self.latest_samples, 
            fit_t, 
            objective_sense='max',
            ranking_method=self.ranking_method
        )
        self.xnes._update_distribution(grads)
        self.generation += 1

    def get_center(self) -> np.ndarray:
        if hasattr(self.xnes._distribution, 'mu'):
            return self.xnes._distribution.mu.cpu().numpy().astype(self.np_dtype)
        elif hasattr(self.xnes._distribution, 'center'):
            return self.xnes._distribution.center.cpu().numpy().astype(self.np_dtype)
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
        
        # Save dist params
        if hasattr(self.xnes._distribution, 'mu'):
            state['mu'] = self.xnes._distribution.mu.cpu().numpy().tolist()
        if hasattr(self.xnes._distribution, 'sigma'):
            state['sigma'] = self.xnes._distribution.sigma.cpu().numpy().tolist()
        if hasattr(self.xnes._distribution, 'A'):
            state['A'] = self.xnes._distribution.A.cpu().numpy().tolist()
            
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.generation = state['generation']
        self.best_fitness = state['best_fitness']
        if state['best_candidate'] is not None:
            self.best_candidate = np.array(state['best_candidate'], dtype=self.np_dtype)
            
        if 'mu' in state and hasattr(self.xnes._distribution, 'mu'):
            self.xnes._distribution.mu.copy_(torch.tensor(state['mu'], dtype=self.dtype, device=self.device))
        if 'sigma' in state and hasattr(self.xnes._distribution, 'sigma'):
            self.xnes._distribution.sigma.copy_(torch.tensor(state['sigma'], dtype=self.dtype, device=self.device))
        if 'A' in state and hasattr(self.xnes._distribution, 'A'):
            self.xnes._distribution.A.copy_(torch.tensor(state['A'], dtype=self.dtype, device=self.device))

    @property
    def name(self) -> str:
        return "XNES"

    @property
    def supports_exact_resume(self) -> bool:
        # XNES has internal state for optimizer (e.g. Adam momentum if used), which we aren't saving.
        # But we only use ExpGaussian default which might just be A and mu.
        # However, random generator state is not saved, so sampling will diverge after resume.
        return False

    def get_diagnostics(self) -> dict:
        return {}
