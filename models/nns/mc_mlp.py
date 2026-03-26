"""Monte Carlo MLP for static basin attributes -> hydrologic parameters.

Interface contract
------------------
- Input:
  - ``[B, nx]`` for direct parameter prediction, or
  - ``[T, B, nx]`` when called from ``CausalDplModel`` with ``xc_nn_norm``.
- Output:
  - ``[B, ny]`` for 2D inputs, or
  - ``[T, B, ny]`` for 3D inputs.

For the ``HbvStatic`` pathway, the 3D output shape is important because
``CausalDplModel`` forwards the tensor directly into ``HbvStatic``, whose
parameter unpacking logic expects ``[T, B, N_PHY * nmul + N_ROUTE]`` and reads
the last timestep as the basin-static parameter vector.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class McMlpModel(nn.Module):
    """MC-dropout MLP that predicts a static parameter vector per basin."""

    def __init__(self, config: dict[str, Any], nx: int, ny: int) -> None:
        super().__init__()
        self.nx = int(nx)
        self.ny = int(ny)
        hidden_size = int(config.get('hidden_size', 128))
        dropout = float(config.get('dropout', 0.1))

        self.name = 'McMlpModel'
        self.output_activation = str(config.get('output_activation', 'sigmoid')).lower()
        self.static_pool = str(config.get('static_pool', 'last')).lower()

        self.layers = nn.Sequential(
            nn.Linear(nx, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, ny),
        )

    def _pool_static_input(self, x: torch.Tensor) -> tuple[torch.Tensor, int | None]:
        if x.ndim == 2:
            if x.shape[-1] != self.nx:
                raise ValueError(
                    f"McMlpModel expected {self.nx} input features, got {x.shape[-1]}."
                )
            return x, None
        if x.ndim != 3:
            raise ValueError(
                f"McMlpModel expects a 2D or 3D tensor, got shape {tuple(x.shape)}."
            )
        if x.shape[-1] != self.nx:
            raise ValueError(
                f"McMlpModel expected {self.nx} input features, got {x.shape[-1]}."
            )

        nt = x.shape[0]
        if self.static_pool == 'mean':
            pooled = x.mean(dim=0)
        else:
            pooled = x[-1]
        return pooled, nt

    def _apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_activation == 'sigmoid':
            return torch.sigmoid(x)
        if self.output_activation == 'softplus':
            return F.softplus(x)
        if self.output_activation in {'identity', 'none'}:
            return x
        raise ValueError(
            f"Unsupported output activation '{self.output_activation}'. "
            "Expected one of: sigmoid, softplus, identity."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, nt = self._pool_static_input(x)
        out = self.layers(pooled)
        out = self._apply_output_activation(out)
        if out.shape[-1] != self.ny:
            raise RuntimeError(
                f"McMlpModel produced {out.shape[-1]} outputs, expected {self.ny}."
            )

        if nt is None:
            return out
        return out.unsqueeze(0).repeat(nt, 1, 1).contiguous()
