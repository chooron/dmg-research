from __future__ import annotations
import abc
import numpy as np
from typing import Any

class OptimizerAdapter(abc.ABC):
    """Optimizer-neutral adapter interface for IC ablation parameter search.
    
    The runtime owns data loading and fitness evaluation.
    The adapter owns search state, ask/tell protocol, and checkpoint.
    """
    
    @abc.abstractmethod
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
        """Initialize optimizer state. Must be called before ask/tell."""
        ...
    
    @abc.abstractmethod
    def ask(self) -> np.ndarray:
        """Return candidate parameters, shape [population, dimension], in [0,1]."""
        ...
    
    @abc.abstractmethod
    def tell(self, fitness: np.ndarray) -> None:
        """Update optimizer state with fitness values [population], maximize direction."""
        ...
    
    @abc.abstractmethod
    def get_center(self) -> np.ndarray:
        """Return current distribution center, shape [dimension], in [0,1]."""
        ...
    
    @abc.abstractmethod
    def get_best(self) -> tuple[np.ndarray, float]:
        """Return (best_candidate [dimension], best_fitness) seen so far."""
        ...
    
    @abc.abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return serializable state dict for checkpointing."""
        ...
    
    @abc.abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint dict."""
        ...
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Optimizer name string."""
        ...
    
    @property
    @abc.abstractmethod  
    def supports_exact_resume(self) -> bool:
        """True only if state serialization allows bit-exact resume."""
        ...

    @abc.abstractmethod
    def get_diagnostics(self) -> dict:
        """Return a dict of internal optimizer metrics."""
        ...
