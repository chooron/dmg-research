"""Unconnected, read-only helper for CMA-ES output-bias initialization."""

from __future__ import annotations

import torch
import torch.nn as nn


def initialize_parameterizer_output_bias(
    parameterizer: nn.Module,
    cma_theta_median: torch.Tensor,
    *,
    clamp_eps: float = 1e-5,
) -> torch.Tensor:
    """Return and optionally install output logits for normalized CMA theta.

    ``CatchmentParameterizer`` stores its final ``nn.Linear`` as ``net[-1]``
    (`dpl/nn_parameterizer.py:35-37`) and applies sigmoid in `forward` at
    `:48-49`.  This helper only sets that existing bias; it does not connect to
    a trainer or optimizer.
    """
    return parameterizer.initialize_output_bias_from_theta(
        cma_theta_median,
        clamp_eps=clamp_eps,
    )
