"""Differentiable GR4J unit hydrograph ordinates via S-curve difference method.

Based on the dmotpy DplUHBase pattern:
- S-curves are computed using pure tensor operations (differentiable in params)
- Ordinates = S(t) - S(t-1)
- Output shape [batch, max_len] for direct use in routing

Reference: /home/jingxin/code/dmg-research/dmotpy/models/unithydro/
This file is an independent re-implementation; dmotpy is not imported.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_gr4j_uh_ordinates(
    x4: torch.Tensor,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GR4J UH1 and UH2 ordinates, fully differentiable in x4.

    UH1 (half bell): S1(t) = 0 if t<=0; (t/x4)^2.5 if t<x4; 1 otherwise
    UH2 (full bell): S2(t) = 0 if t<=0; 0.5*(t/x4)^2.5 if t<=x4;
                             1-0.5*(2-t/x4)^2.5 if t<2*x4; 1 otherwise

    Ordinates are computed as S(t) - S(t-1) for t = 1..max_len.

    Args:
        x4: [batch] GR4J time base parameter.
        max_len: Maximum number of UH ordinates (compile-time constant).

    Returns:
        uh1_ord: [batch, max_len] normalized UH1 ordinates.
        uh2_ord: [batch, max_len] normalized UH2 ordinates.
    """
    d_base = torch.clamp(x4, min=1e-3)  # [batch]

    t = torch.arange(
        1, max_len + 1, device=x4.device, dtype=x4.dtype
    ).view(1, -1)  # [1, max_len]

    ratio = t / d_base.unsqueeze(-1)  # [batch, max_len]

    # UH1: Half bell S-curve
    s1 = torch.clamp(ratio, max=1.0).pow(2.5)  # [batch, max_len]
    s1_padded = F.pad(s1, (1, 0), value=0.0)  # S(0)=0, [batch, max_len+1]
    uh1_raw = s1_padded[:, 1:] - s1_padded[:, :-1]  # [batch, max_len]
    uh1 = _normalize_ordinates(uh1_raw)

    # UH2: Full bell S-curve
    s2_part1 = 0.5 * ratio.pow(2.5)
    term_b = torch.clamp(2.0 - ratio, min=0.0)
    s2_part2 = 1.0 - 0.5 * term_b.pow(2.5)
    s2 = torch.where(ratio <= 1.0, s2_part1, s2_part2)
    s2 = torch.clamp(s2, max=1.0)
    s2_padded = F.pad(s2, (1, 0), value=0.0)
    uh2_raw = s2_padded[:, 1:] - s2_padded[:, :-1]
    uh2 = _normalize_ordinates(uh2_raw)

    return uh1, uh2


def _normalize_ordinates(
    raw: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize UH ordinates to sum to 1 along last dim."""
    s = raw.sum(dim=-1, keepdim=True)
    return raw / (s + eps)


def apply_unit_hydrograph_routing(
    flux_in: torch.Tensor,
    uh_ordinates: torch.Tensor,
) -> torch.Tensor:
    """Apply unit hydrograph routing via grouped 1D convolution.

    Uses the same pattern as dmotpy DplUHBase: input reshaped to
    [1, batch, time] with groups=batch_size for per-basin convolution.

    Args:
        flux_in: [batch, time] instantaneous inflow series.
        uh_ordinates: [batch, kernel_len] unit hydrograph weights.

    Returns:
        routed: [batch, time] routed outflow series (same length as input).
    """
    batch_size, time_steps = flux_in.shape
    kernel_len = uh_ordinates.shape[-1]

    # Normalize
    s = uh_ordinates.sum(dim=-1, keepdim=True)
    w = uh_ordinates / (s + 1e-8)

    # Flip for convolution (conv1d does correlation, we want convolution)
    w_flipped = torch.flip(w, dims=[-1]).unsqueeze(1)  # [batch, 1, kernel_len]

    # Reshape input: [1, batch, time] for grouped conv1d
    x = flux_in.view(1, batch_size, time_steps)

    pad_size = kernel_len - 1
    out = F.conv1d(
        input=x,
        weight=w_flipped,
        groups=batch_size,
        padding=pad_size,
    )  # [1, batch, time + pad_size]

    if pad_size > 0:
        out = out[:, :, :time_steps]

    return out.view(batch_size, time_steps)  # [batch, time]
