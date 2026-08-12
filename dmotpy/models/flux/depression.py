import torch
import torch.nn.functional as F


def depression_1(
    ads: torch.Tensor,
    md: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """
    Exponential inflow to surface depression store.
    Formula: out = min(p1 * exp(-p2 * S / max(Smax - S, 0)) * flux, max(Smax - S, 0))
    """
    capacity = F.relu(Smax - S)
    exponent = torch.clamp(
        -md * S / (capacity + nearzero), min=-30.0, max=0.0
    )
    potential_inflow = ads * torch.exp(exponent) * incoming_flux
    return torch.minimum(torch.minimum(potential_inflow, capacity), incoming_flux)
