import torch
import torch.nn.functional as F


def exchange_1(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    S: torch.Tensor,
    fmax: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Two-way channel exchange: linear and exponential.
    Formula: out = (p1 * |S| + p2 * (1 - exp(-p3 * |S|))) * sign(S)
    Constraint: out >= -fmax
    Note: dt is assumed to be 1.0.
    """
    s_abs = torch.abs(S)
    linear_part = p1 * S
    arg = torch.clamp(-p3 * s_abs, min=-30.0, max=0.0)
    exp_term = 1.0 - torch.exp(arg)
    smooth_sign = S / (s_abs + nearzero)
    flow = linear_part + p2 * exp_term * smooth_sign
    return torch.maximum(flow, -torch.abs(fmax))


def exchange_2(
    p1: torch.Tensor,
    S1: torch.Tensor,
    S1max: torch.Tensor,
    S2: torch.Tensor,
    S2max: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Water exchange based on relative storages.
    Formula: out = p1 * (S1/S1max - S2/S2max)
    """
    ratio1 = S1 / (S1max + nearzero)
    ratio2 = S2 / (S2max + nearzero)
    return p1 * (ratio1 - ratio2)


def exchange_3(
    p1: torch.Tensor, S: torch.Tensor, p2: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Water exchange with infinite size store based on threshold.
    Formula: out = p1 * (S - p2)
    """
    return p1 * (S - p2)


def exchange_gr4j(
    x2: torch.Tensor,
    S: torch.Tensor,
    x3: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """GR4J groundwater-exchange term from routing-store saturation."""
    return x2 * (S / (x3 + nearzero)).pow(3.5)
