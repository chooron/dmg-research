"""
AET (actual evapotranspiration) constraint formulas.

Reference: HBV-light manual (UZH, 2005).
"""

import torch

from ._utils import smoothmin_t


def aet_hbv_default(PET, SM, LP, FC):
    """E0: HBV default AET.

    ET = PET * min(SM / (LP * FC), 1)

    Constrained so that 0 <= ET <= PET and ET <= SM.

    Args:
        PET: potential evapotranspiration.
        SM: soil moisture.
        LP: limit potential parameter (fraction of FC).
        FC: field capacity.

    Returns:
        ET: actual evapotranspiration.
    """
    PET = torch.clamp(PET, min=0.0)
    SM = torch.clamp(SM, min=0.0)
    threshold = torch.clamp(LP * FC, min=1e-6)
    frac = torch.clamp(SM / threshold, 0.0, 1.0)
    ET = PET * frac
    ET = torch.minimum(ET, SM)
    return ET


def aet_smooth_hbv(PET, SM, LP, FC, tau_E):
    """E1: Smooth HBV AET.

    ET = PET * smoothmin(SM / (LP*FC), 1, tau_E)

    Uses smooth approximation of min() for full differentiability.

    Args:
        PET: potential evapotranspiration.
        SM: soil moisture.
        LP: limit potential parameter.
        FC: field capacity.
        tau_E: smoothing temperature (>0).

    Returns:
        ET: actual evapotranspiration.
    """
    PET = torch.clamp(PET, min=0.0)
    SM = torch.clamp(SM, min=0.0)
    x = SM / torch.clamp(LP * FC, min=1e-6)
    frac = smoothmin_t(x, threshold=1.0, tau=tau_E)
    frac = torch.clamp(frac, 0.0, 1.0)
    ET = PET * frac
    ET = torch.minimum(ET, SM)
    return ET


def temperature_corrected_aet(PET_m, T_t, T_m, CET, SM, LP, FC):
    """E2: Temperature-corrected PET + HBV AET.

    PET_t = PET_m * (1 + CET * (T_t - T_m))
    ET = PET_t * min(SM / (LP * FC), 1)

    PET_t is clamped to [0, 2 * PET_m] to prevent unrealistic extremes.

    Args:
        PET_m: long-term mean PET.
        T_t: current temperature.
        T_m: long-term mean temperature.
        CET: temperature correction coefficient.
        SM: soil moisture.
        LP: limit potential parameter.
        FC: field capacity.

    Returns:
        PET_t: temperature-corrected PET.
        ET: actual evapotranspiration.
    """
    PET_t = PET_m * (1.0 + CET * (T_t - T_m))
    PET_t = torch.clamp(PET_t, min=0.0, max=2.0 * PET_m)
    ET = aet_hbv_default(PET_t, SM, LP, FC)
    return PET_t, ET


def aet_power_law(PET, SM, FC, gamma_E):
    """E3: Power-law soil-moisture stress AET.

    ET = PET * (SM / FC) ** gamma_E

    Internally clamps SM/FC to [eps, 1] to avoid the gradient singularity
    of x^gamma at x=0 when gamma < 1.

    Args:
        PET: potential evapotranspiration.
        SM: soil moisture.
        FC: field capacity.
        gamma_E: exponent (>0).

    Returns:
        ET: actual evapotranspiration.
    """
    EPS = 1e-6
    PET = torch.clamp(PET, min=0.0)
    SM = torch.clamp(SM, min=0.0)
    FC = torch.clamp(FC, min=EPS)
    if torch.is_tensor(gamma_E):
        gamma_E = torch.clamp(gamma_E, min=EPS)
    else:
        gamma_E = max(float(gamma_E), EPS)
    frac = torch.clamp(SM / FC, min=EPS, max=1.0)
    ET_raw = PET * torch.pow(frac, gamma_E)
    ET = torch.minimum(ET_raw, SM)
    ET = torch.clamp(ET, min=0.0)
    return ET


def feddes_threshold_aet(PET, SM, FC, s_w, s_o):
    """E4: Feddes-style threshold AET.

    f = clamp((s - s_w) / (s_o - s_w), 0, 1)  where s = clamp(SM / FC, 0, 1)
    ET = PET * f

    Args:
        PET: potential evapotranspiration.
        SM: soil moisture.
        FC: field capacity.
        s_w: stress-onset threshold ([0.05, 0.25] recommended).
        s_o: no-stress threshold ([0.45, 0.85] recommended).

    Returns:
        ET: actual evapotranspiration, 0 <= ET <= PET, ET <= SM.
    """
    PET = torch.clamp(PET, min=0.0)
    SM = torch.clamp(SM, min=0.0)
    FC = torch.clamp(FC, min=1e-6)
    s = torch.clamp(SM / FC, 0.0, 1.0)
    s_w = torch.clamp(s_w, min=0.0, max=1.0) if torch.is_tensor(s_w) else max(min(float(s_w), 1.0), 0.0)
    s_o = torch.clamp(s_o, min=0.01, max=1.0) if torch.is_tensor(s_o) else max(min(float(s_o), 1.0), 0.01)
    denom = torch.clamp(s_o - s_w, min=1e-6)
    f = torch.clamp((s - s_w) / denom, 0.0, 1.0)
    ET = PET * f
    ET = torch.clamp(ET, max=PET)
    ET = torch.minimum(ET, SM)
    return ET
