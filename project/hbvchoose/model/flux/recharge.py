"""
Soil / recharge formulas: HBV beta-recharge family and threshold variants.

Reference: HBV.IANIGLA R package, Soil_HBV documentation.
"""

import torch


def beta_recharge(I, SM, FC, beta):
    """R0: HBV default beta recharge.

    r = (SM / FC) ** beta
    R = I * r

    Args:
        I: total input (rainfall + snowmelt).
        SM: soil moisture.
        FC: field capacity.
        beta: shape parameter (>0).

    Returns:
        R: groundwater recharge, 0 <= R <= I.
    """
    FC = torch.clamp(FC, min=1e-6)
    I = torch.clamp(I, min=0.0)
    sat = torch.clamp(SM / FC, 0.0, 1.0)
    if torch.is_tensor(beta):
        beta = torch.clamp(beta, min=1e-6)
    else:
        beta = max(float(beta), 1e-6)
    R = I * sat ** beta
    R = torch.clamp(R, min=0.0)
    R = torch.minimum(R, I)
    return R


def linear_recharge(I, SM, FC):
    """R1: HBV linear recharge (beta = 1).

    R = I * SM / FC

    Equivalent to beta_recharge with beta=1.
    """
    return beta_recharge(I, SM, FC, beta=1.0)


def strong_nonlinear_recharge(I, SM, FC, beta_h):
    """R2: Strong nonlinear beta recharge (beta_h > 1).

    R = I * (SM / FC) ** beta_h

    Typical range: beta_h in [2, 6].  Runoff only becomes significant
    when soil is wet.
    """
    return beta_recharge(I, SM, FC, beta_h)


def weak_nonlinear_recharge(I, SM, FC, beta_l):
    """R3: Weak nonlinear beta recharge (0 < beta_l < 1).

    R = I * (SM / FC) ** beta_l

    Typical range: beta_l in (0, 1).  Recharge can occur even under
    relatively dry conditions.
    """
    return beta_recharge(I, SM, FC, beta_l)


def saturation_threshold_recharge(I, SM, FC, a_r, c_r):
    """R4: Normalized logistic saturation-threshold recharge.

    lo = sigmoid(-a_r * c_r)      # value at sat = 0
    hi = sigmoid(a_r * (1 - c_r)) # value at sat = 1
    frac = (sigmoid(a_r*(sat-c_r)) - lo) / clamp(hi - lo, eps)
    R = I * clamp(frac, 0, 1)

    Guarantees: R -> 0 when SM -> 0,  R -> I when SM -> FC.

    Args:
        I: total input.
        SM: soil moisture.
        FC: field capacity.
        a_r: steepness (>0).
        c_r: threshold (0 < c_r < 1).

    Returns:
        R: groundwater recharge, 0 <= R <= I.
    """
    FC = torch.clamp(FC, min=1e-6)
    I = torch.clamp(I, min=0.0)
    sat = torch.clamp(SM / FC, 0.0, 1.0)
    a_r = torch.clamp(a_r, min=1e-6)
    c_r = torch.clamp(c_r, min=0.0, max=1.0)

    lo = torch.sigmoid(-a_r * c_r)
    hi = torch.sigmoid(a_r * (1.0 - c_r))
    raw = torch.sigmoid(a_r * (sat - c_r))
    denom = torch.clamp(hi - lo, min=1e-6)
    frac = torch.clamp((raw - lo) / denom, 0.0, 1.0)
    R = I * frac
    R = torch.minimum(R, I)
    return R


def variable_contributing_area_recharge(I, SM, FC, b_v):
    """R5: Variable contributing area recharge (XAJ/VIC-style).

    A_s = 1 - (1 - s) ** b_v   where s = clamp(SM / FC, 0, 1)
    R = I * A_s

    Args:
        I: total input (rainfall + snowmelt).
        SM: soil moisture.
        FC: field capacity.
        b_v: shape parameter ([0.3, 1.5] recommended).

    Returns:
        R: groundwater recharge, 0 <= R <= I.
    """
    FC = torch.clamp(FC, min=1e-6)
    I = torch.clamp(I, min=0.0)
    s = torch.clamp(SM / FC, 0.0, 1.0)
    if torch.is_tensor(b_v):
        b_v = torch.clamp(b_v, min=1e-6)
    else:
        b_v = max(float(b_v), 1e-6)
    A_s = 1.0 - (1.0 - s) ** b_v
    R = I * A_s
    R = torch.clamp(R, min=0.0)
    R = torch.minimum(R, I)
    return R
