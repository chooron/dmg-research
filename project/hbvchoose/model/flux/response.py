"""
Response / routing formulas: HBV reservoir response functions.

References:
  - HBV-light manual (UZH, 2005).
  - HESS (2020) doi:10.5194/hess-24-4441-2020
  - Hydrology Research 46(4):607, doi:10.2166/nh.2014.098
"""

import torch

from ._utils import softplus_t


# ---------------------------------------------------------------------------
# Q0 / Q1 — HBV linear two-reservoir
# ---------------------------------------------------------------------------

def response_two_reservoir(SUZ, SLZ, K_0, K_1, K_2, UZL):
    """Q0: HBV default two-reservoir linear response.

    Q_0 = K_0 * max(SUZ - UZL, 0),  capped by SUZ.
    Q_1 = K_1 * (SUZ - Q_0),          capped by remaining SUZ.
    Q_2 = K_2 * SLZ,                  capped by SLZ.

    Constraint: 0 < K_2 < K_1 < K_0 < 1.

    Args:
        SUZ: storage in upper zone.
        SLZ: storage in lower zone.
        K_0: fast recession coefficient.
        K_1: intermediate recession coefficient.
        K_2: slow recession coefficient.
        UZL: fast-flow threshold.

    Returns:
        Q_0, Q_1, Q_2, Q_total
    """
    SUZ = torch.clamp(SUZ, min=0.0)
    SLZ = torch.clamp(SLZ, min=0.0)

    Q_0 = K_0 * torch.clamp(SUZ - UZL, min=0.0)
    Q_0 = torch.minimum(Q_0, SUZ)

    remaining_suz = SUZ - Q_0
    Q_1 = K_1 * remaining_suz
    Q_1 = torch.minimum(Q_1, remaining_suz)

    Q_2 = K_2 * SLZ
    Q_2 = torch.minimum(Q_2, SLZ)
    Q = Q_0 + Q_1 + Q_2
    return Q_0, Q_1, Q_2, Q


def response_smooth_threshold(SUZ, SLZ, K_0, K_1, K_2, UZL, tau_Q):
    """Q1: Smooth HBV two-reservoir threshold response.

    Q_0 = K_0 * softplus(SUZ - UZL, tau_Q),  capped by SUZ.
    Q_1 = K_1 * (SUZ - Q_0),                  capped by remaining SUZ.
    Q_2 = K_2 * SLZ,                          capped by SLZ.

    Differentiable version of Q0.

    Args:
        SUZ, SLZ, K_0, K_1, K_2, UZL: as in Q0.
        tau_Q: smoothing temperature (>0).

    Returns:
        Q_0, Q_1, Q_2, Q_total
    """
    SUZ = torch.clamp(SUZ, min=0.0)
    SLZ = torch.clamp(SLZ, min=0.0)

    Q_0 = K_0 * softplus_t(SUZ - UZL, tau_Q)
    Q_0 = torch.minimum(Q_0, SUZ)

    remaining_suz = SUZ - Q_0
    Q_1 = K_1 * remaining_suz
    Q_1 = torch.minimum(Q_1, remaining_suz)

    Q_2 = K_2 * SLZ
    Q_2 = torch.minimum(Q_2, SLZ)
    Q = Q_0 + Q_1 + Q_2
    return Q_0, Q_1, Q_2, Q


# ---------------------------------------------------------------------------
# Q2 — Nonlinear reservoir
# ---------------------------------------------------------------------------

def response_nonlinear(SUZ, SLZ, K_1, K_2, alpha_Q):
    """Q2: HBV nonlinear reservoir response.

    Q_uz = K_1 * SUZ ** alpha_Q,  capped by SUZ.
    Q_lz = K_2 * SLZ,             capped by SLZ.
    Q = Q_uz + Q_lz

    Args:
        SUZ: storage in upper zone.
        SLZ: storage in lower zone.
        K_1: upper recession coefficient.
        K_2: lower recession coefficient.
        alpha_Q: nonlinearity exponent (>0; alpha_Q=1 recovers Q0).

    Returns:
        Q_uz, Q_lz, Q_total
    """
    SUZ = torch.clamp(SUZ, min=0.0)
    SLZ = torch.clamp(SLZ, min=0.0)
    alpha_Q = torch.clamp(alpha_Q, min=1.0)
    Q_uz = K_1 * SUZ ** alpha_Q
    Q_uz = torch.minimum(Q_uz, SUZ)
    Q_lz = K_2 * SLZ
    Q_lz = torch.minimum(Q_lz, SLZ)
    Q = Q_uz + Q_lz
    return Q_uz, Q_lz, Q


# ---------------------------------------------------------------------------
# Q3 — Single reservoir
# ---------------------------------------------------------------------------

def response_single_reservoir(S, K):
    """Q3: Single reservoir linear response.

    Q = K * S,  capped by S.

    Simplest parsimonious response candidate.

    Args:
        S: total storage.
        K: recession coefficient.

    Returns:
        Q: outflow.
    """
    S = torch.clamp(S, min=0.0)
    K = torch.clamp(K, min=0.0, max=1.0)
    Q = torch.minimum(K * S, S)
    return Q


# ---------------------------------------------------------------------------
# Q4 — Two reservoirs in parallel
# ---------------------------------------------------------------------------

def response_two_parallel(R, S_f, S_s, K_f, K_s, p):
    """Q4: Two-reservoir parallel linear response.

    R_f = p * R    (fast)
    R_s = (1-p) * R  (slow)
    Q_f = K_f * S_f,  capped by S_f.
    Q_s = K_s * S_s,  capped by S_s.
    Q = Q_f + Q_s

    Args:
        R: total recharge.
        S_f: fast storage.
        S_s: slow storage.
        K_f: fast recession coefficient.
        K_s: slow recession coefficient.
        p: partition coefficient (0 <= p <= 1).

    Returns:
        R_f, R_s, Q_f, Q_s, Q_total
    """
    p = torch.clamp(p, 0.0, 1.0)
    R = torch.clamp(R, min=0.0)
    S_f = torch.clamp(S_f, min=0.0)
    S_s = torch.clamp(S_s, min=0.0)

    R_f = p * R
    R_s = (1.0 - p) * R
    Q_f = K_f * S_f
    Q_f = torch.minimum(Q_f, S_f)
    Q_s = K_s * S_s
    Q_s = torch.minimum(Q_s, S_s)
    Q = Q_f + Q_s
    return R_f, R_s, Q_f, Q_s, Q


# ---------------------------------------------------------------------------
# Q5 — Delayed response (HBV-light)
# ---------------------------------------------------------------------------

def response_delayed_step(R, S_1, S_2, PART, K_1, K_2):
    """Q5: HBV-light delayed response — single step.

    R_imm = PART * R
    R_del = (1 - PART) * R
    Q_1 = K_1 * S_1,  capped by S_1.
    Q_2 = K_2 * S_2,  capped by S_2.
    Q = Q_1 + Q_2

    Note: The delay itself (moving-average over DELAY steps) is handled
    separately by delay_buffer().

    Args:
        R: total recharge at this step.
        S_1: storage in upper (immediate) reservoir.
        S_2: storage in lower (delayed) reservoir.
        PART: fraction routed immediately (0 <= PART <= 1).
        K_1: upper recession coefficient.
        K_2: lower recession coefficient.

    Returns:
        R_imm, R_del, Q_1, Q_2, Q_total
    """
    PART = torch.clamp(PART, 0.0, 1.0)
    R = torch.clamp(R, min=0.0)
    S_1 = torch.clamp(S_1, min=0.0)
    S_2 = torch.clamp(S_2, min=0.0)

    R_imm = PART * R
    R_del = (1.0 - PART) * R
    Q_1 = K_1 * S_1
    Q_1 = torch.minimum(Q_1, S_1)
    Q_2 = K_2 * S_2
    Q_2 = torch.minimum(Q_2, S_2)
    Q = Q_1 + Q_2
    return R_imm, R_del, Q_1, Q_2, Q


def delay_buffer(R_del_seq, DELAY):
    """Smooth delayed recharge via moving average over DELAY steps.

    R_del_eff[t] = (1 / DELAY) * sum_{j=0}^{DELAY-1} R_del[t - j]

    NOTE: DELAY must be an integer (Python int or scalar 0-d tensor).
    This is a discrete delay, NOT a continuous differentiable parameter.
    For learnable delay, use a soft delay kernel or multiple fixed-delay
    expert routing in the MoE framework.

    Args:
        R_del_seq: sequence of delayed recharge, shape (..., time).
        DELAY: number of averaging steps (int, >= 1).

    Returns:
        R_del_eff: smoothed delayed recharge, same shape as input.

    Raises:
        ValueError: if DELAY is a tensor with numel() != 1.
    """
    if torch.is_tensor(DELAY):
        if DELAY.numel() != 1:
            raise ValueError("DELAY must be a scalar integer for delay_buffer.")
        DELAY = int(DELAY.detach().cpu().item())
    DELAY = max(int(round(DELAY)), 1)

    weight = torch.ones(DELAY, device=R_del_seq.device, dtype=R_del_seq.dtype) / DELAY

    original_shape = R_del_seq.shape
    if R_del_seq.dim() == 1:
        x = R_del_seq.unsqueeze(0).unsqueeze(0)
    elif R_del_seq.dim() == 2:
        x = R_del_seq.unsqueeze(1)
    else:
        x = R_del_seq

    w = weight.flip(0).unsqueeze(0).unsqueeze(0)
    x_padded = torch.nn.functional.pad(x, (DELAY - 1, 0))
    y = torch.nn.functional.conv1d(x_padded, w)

    if len(original_shape) == 1:
        y = y.squeeze(0).squeeze(0)
    elif len(original_shape) == 2:
        y = y.squeeze(1)
    return y
