from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn


class BaseHydrologicalModel(nn.Module, ABC):
    """Abstract base class for all hydrological models.

    All models must implement the unified interface:

        qsim, aux = model(
            forcings=forcings,
            params=params,
            initial_states=None,
            return_states=False,
        )
    """

    @property
    @abstractmethod
    def parameter_specs(self) -> dict[str, dict[str, Any]]:
        """Return parameter specifications for the model."""

    @abstractmethod
    def forward(
        self,
        forcings: dict[str, torch.Tensor],
        params: dict[str, torch.Tensor],
        initial_states: Optional[dict[str, torch.Tensor]] = None,
        return_states: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run model forward pass.

        Args:
            forcings: Dict with keys 'precip', 'pet', 'temp'. Each tensor [batch, time].
            params: Dict mapping parameter names to physical-scale tensors [batch].
            initial_states: Optional dict mapping state names to initial values.
            return_states: If True, return final states in aux.

        Returns:
            qsim: [batch, time] simulated streamflow.
            aux: Dict with intermediate variables and optionally final states.
        """
