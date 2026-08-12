from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class OptimizerAdapter(Protocol):
    """Minimal optimizer-neutral contract for future IC optimizers."""

    name: str

    def initialize(self, *, dimension: int, seed: int, population: int) -> None:
        ...

    def ask(self) -> np.ndarray:
        """Return normalized candidates with shape [population, dimension]."""
        ...

    def tell(self, candidates_01: np.ndarray, fitness: np.ndarray) -> None:
        """Update state using maximize-oriented fitness."""
        ...

    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        ...

    def reset(self, *, seed: int | None = None) -> None:
        ...
