"""
Routing (translation / convolution) formulas.

Reference: HBV-light manual (UZH, 2005).
"""

import torch

from ._utils import causal_conv1d, gamma_weights, triangular_weights


def maxbas_routing(Q, MAXBAS):
    """T0: MAXBAS triangular routing.

    Q_out[t] = sum_{i=1}^{MAXBAS} w_i * Q[t - i + 1]

    where w forms a symmetric triangular kernel of length MAXBAS.

    NOTE: MAXBAS must be an integer (Python int or scalar 0-d tensor).
    This is NOT a continuous differentiable MAXBAS parameter.  For
    learnable routing length, route through multiple fixed-MAXBAS experts
    in the MoE framework.

    Args:
        Q: inflow sequence, shape (..., time).
        MAXBAS: routing length parameter (int, >= 1).

    Returns:
        Q_out: routed outflow, same shape as Q.

    Raises:
        ValueError: if MAXBAS is a tensor with numel() != 1.
    """
    if torch.is_tensor(MAXBAS):
        if MAXBAS.numel() != 1:
            raise ValueError("MAXBAS must be a scalar integer for maxbas_routing.")
        MAXBAS = int(MAXBAS.detach().cpu().item())
    MAXBAS = max(int(round(MAXBAS)), 1)

    w = triangular_weights(MAXBAS).to(Q.device)
    return causal_conv1d(Q, w)


def gamma_routing(Q, a, b, length):
    """T1: Gamma routing.

    w_i = i^(a-1) * exp(-i/b) / sum_j(j^(a-1) * exp(-j/b))
    Q_out[t] = sum_i w_i * Q[t - i + 1]

    Args:
        Q: inflow sequence, shape (..., time).
        a: shape parameter (>0), scalar or 0-d tensor.
        b: scale parameter (>0), scalar or 0-d tensor.
        length: number of weights (int, >= 1).

    Returns:
        Q_out: routed outflow, same shape as Q.
    """
    w = gamma_weights(a, b, length).to(Q.device)
    return causal_conv1d(Q, w)
