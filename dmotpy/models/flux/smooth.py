import torch


def soft_gate_storage_above(
    S: torch.Tensor,
    threshold: torch.Tensor,
    k: float = 10.0,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """dMoT differentiable storage gate active above a threshold."""
    thresh_abs = torch.abs(threshold) + nearzero
    scale = torch.clamp(k / thresh_abs, max=50.0)
    return torch.sigmoid(scale * (S - threshold))


def soft_gate_storage_below(
    S: torch.Tensor,
    threshold: torch.Tensor,
    k: float = 10.0,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """dMoT differentiable storage gate active below a threshold."""
    return 1.0 - soft_gate_storage_above(S, threshold, k=k, nearzero=nearzero)


def soft_gate_temperature_below(
    T: torch.Tensor,
    threshold: torch.Tensor,
    k: float = 5.0,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """dMoT differentiable temperature gate active below a threshold."""
    return torch.sigmoid(k * (threshold - T))


def soft_gate_temperature_above(
    T: torch.Tensor,
    threshold: torch.Tensor,
    k: float = 5.0,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """dMoT differentiable temperature gate active above a threshold."""
    return 1.0 - soft_gate_temperature_below(T, threshold, k=k, nearzero=nearzero)


# Legacy aliases for backward compatibility
smooth_threshold_storage_logistic = soft_gate_storage_above
smooth_threshold_temperature_logistic = soft_gate_temperature_below


def smooth_relu(x: torch.Tensor, tau: float = 1e-3) -> torch.Tensor:
    """Smooth approximation of ReLU used by the differentiable GSFB step."""
    return tau * torch.nn.functional.softplus(x / tau)


def smooth_min(
    a: torch.Tensor, b: torch.Tensor, tau: float = 1e-3
) -> torch.Tensor:
    """Smooth approximation of ``minimum(a, b)``."""
    return -tau * torch.logsumexp(
        torch.stack((-a / tau, -b / tau), dim=0), dim=0
    )


def smooth_cap_flux(
    q_pot: torch.Tensor, available: torch.Tensor, tau: float = 1e-3
) -> torch.Tensor:
    """Smoothly constrain a candidate flux to nonnegative available water."""
    return smooth_min(
        smooth_relu(q_pot, tau=tau),
        smooth_relu(available, tau=tau),
        tau=tau,
    )
