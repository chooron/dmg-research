import torch
import torch.nn.functional as F
from .smooth import smooth_threshold_storage_logistic


def baseflow_1(
    p1: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Baseflow 1: Outflow from a linear reservoir
    Formula: out = p1 * S
    """
    return p1 * S


def baseflow_2(
    S: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Baseflow 2: Non-linear outflow from a reservoir
    Constraint: f <= S
    Formula: out = (S / p1)^(1 / p2)
    """
    term_flow = (S / (p1 + nearzero)).pow(1.0 / (p2 + nearzero))
    return torch.minimum(term_flow, S)


def baseflow_3(
    S: torch.Tensor, Smax: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Baseflow 3: Empirical non-linear outflow
    Formula: out = Smax^(-4) / 4 * S^5
    """
    return ((Smax + nearzero).pow(-4.0) / 4.0) * S.pow(5)


def baseflow_4(
    p1: torch.Tensor, p2: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Baseflow 4: Exponential outflow from deficit store
    Formula: out = p1 * exp(-p2 * S)
    """
    return p1 * torch.exp(-p2 * S)


def baseflow_5(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Baseflow 5: Non-linear scaled outflow
    Constraint: f <= S
    ratio: (S / Smax)^p2
    """
    ratio = S / (Smax + nearzero)
    # Same FP32 power-overflow guard as percolation_5: ratio >> 1 selects S
    # in the final min either way, so clamping the ratio before the power
    # only removes the inf intermediate (0*inf backward NaN), never the
    # finite-domain result.
    safe_ratio = torch.clamp(ratio + nearzero, max=1e3)
    term_flow = p1 * safe_ratio.pow(p2)
    return torch.minimum(S, term_flow)


def baseflow_6(
    p1: torch.Tensor, p2: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Baseflow 6: Quadratic outflow if storage threshold is exceeded
    """
    # Preserve the original expression exactly for finite, ordinary storage,
    # but keep the square inside the dtype's finite domain.  This avoids the
    # ``0 * inf`` NaN at valid p1=0 without changing normal-case rounding.
    finite_storage = torch.where(torch.isfinite(S), S, torch.zeros_like(S))
    max_square_base = torch.sqrt(
        torch.full_like(finite_storage, torch.finfo(S.dtype).max)
    )
    safe_storage = torch.clamp(finite_storage, min=-max_square_base, max=max_square_base)
    q_quadratic = torch.minimum(safe_storage, p1 * safe_storage.pow(2))

    # sf returns ~1 when S > p2
    sf = smooth_threshold_storage_logistic(safe_storage, p2, nearzero=nearzero)
    return q_quadratic * sf


def baseflow_tcm(
    p1: torch.Tensor, p2: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """TCM quadratic slow-flow reservoir with the model's 1/1000 scale."""
    k2_scaled = p1 / 1000.0
    q_unconstrained = k2_scaled * S.pow(2)
    gate_open = smooth_threshold_storage_logistic(S, p2, nearzero=nearzero)
    return q_unconstrained * gate_open


def baseflow_7(
    p1: torch.Tensor, p2: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Baseflow 7: Non-linear outflow
    Formula: out = min(S, p1 * S^p2)
    """
    term_flow = p1 * (S + nearzero).pow(p2)
    return torch.minimum(S, term_flow)


def baseflow_8(
    p1: torch.Tensor,
    p2: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Baseflow 8: Exponential scaled outflow from deficit store
    Formula: out = p1 * (exp(p2 * min(1, max(S,0)/Smax)) - 1)
    """
    ratio = S / (Smax + nearzero)
    ratio_clamped = torch.clamp(ratio, max=1.0)
    return p1 * (torch.exp(p2 * ratio_clamped) - 1.0)


def baseflow_9(
    p1: torch.Tensor, p2: torch.Tensor, S: torch.Tensor, nearzero: float = 1e-6
) -> torch.Tensor:
    """
    Baseflow 9: Linear flow above a threshold
    Formula: out = p1 * max(0, S - p2)
    """
    # Using Softplus for smooth transition
    excess_storage = F.softplus(S - p2, beta=50.0)
    return p1 * excess_storage
