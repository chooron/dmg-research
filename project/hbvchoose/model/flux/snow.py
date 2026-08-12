"""
Snow routine formulas: rain/snow partition, snowmelt, and refreezing.

Reference: HESS (2020) doi:10.5194/hess-24-4441-2020
"""

import torch

from ._utils import softplus_t


def rain_snow_partition_hard(P, T, TT):
    """S0: HBV default rain/snow partition with hard threshold.

    P_s = P  if T < TT else 0
    P_r = P - P_s

    Args:
        P: total precipitation (tensor).
        T: temperature (tensor).
        TT: threshold temperature (scalar).

    Returns:
        P_s: snowfall.
        P_r: rainfall.
    """
    P_s = torch.where(T < TT, P, torch.zeros_like(P))
    P_r = P - P_s
    return P_s, P_r


def rain_snow_partition_smooth(P, T, TT, tau_T):
    """S1: Smooth rain/snow partition via sigmoid.

    f_s = 1 - sigmoid((T - TT) / tau_T)
    P_s = P * f_s,  P_r = P * (1 - f_s)

    Args:
        P: total precipitation.
        T: temperature.
        TT: threshold temperature.
        tau_T: smoothing temperature (>0).

    Returns:
        P_s: snowfall.
        P_r: rainfall.
    """
    tau_T = torch.clamp(tau_T, min=1e-6) if torch.is_tensor(tau_T) else max(float(tau_T), 1e-6)
    f_s = 1.0 - torch.sigmoid((T - TT) / tau_T)
    P_s = P * f_s
    P_r = P * (1.0 - f_s)
    return P_s, P_r


def snowmelt_linear_degreeday(T, TT, CFMAX, SWE):
    """S2: HBV default linear degree-day snowmelt.

    M = CFMAX * max(T - TT, 0),  capped by SWE.

    Args:
        T: temperature.
        TT: threshold temperature.
        CFMAX: degree-day factor (mm/deg/day).
        SWE: snow water equivalent.

    Returns:
        M: snowmelt.
    """
    M = CFMAX * torch.clamp(T - TT, min=0.0)
    M = torch.min(M, SWE)
    return M


def snowmelt_smooth_degreeday(T, TT, CFMAX, tau_M, SWE):
    """S3: Smooth linear degree-day snowmelt via softplus.

    M = CFMAX * tau_M * log(1 + exp((T - TT) / tau_M)),  capped by SWE.

    Args:
        T: temperature.
        TT: threshold temperature.
        CFMAX: degree-day factor.
        tau_M: smoothing temperature (>0).
        SWE: snow water equivalent.

    Returns:
        M: snowmelt.
    """
    M = CFMAX * softplus_t(T - TT, tau_M)
    M = torch.min(M, SWE)
    return M


def cfmax_seasonal(CFMAX_0, a_s, phi_s, doy):
    """S4: Seasonal degree-day factor.

    CFMAX(t) = CFMAX_0 * (1 + a_s * sin(2*pi*(doy - phi_s) / 365))

    Args:
        CFMAX_0: base degree-day factor.
        a_s: seasonal amplitude (0 <= a_s <= 1 recommended).
        phi_s: phase shift (day of year).
        doy: day of year tensor (1-365).

    Returns:
        CFMAX_t: time-varying degree-day factor, clamped to >= 0.
    """
    CFMAX_t = CFMAX_0 * (1.0 + a_s * torch.sin(2.0 * torch.pi * (doy - phi_s) / 365.0))
    CFMAX_t = torch.clamp(CFMAX_t, min=1e-6)
    return CFMAX_t


def snowmelt_exponential(T, TT, CFMAX, c_m, SWE):
    """S5: Exponential snowmelt function (threshold-consistent).

    M = CFMAX * [exp(c_m * (T - TT)) - 1]_+,  capped by SWE.

    Uses torch.clamp on raw (exp-1) instead of softplus to ensure
    M -> 0 when T <= TT.  Exponent is bounded to prevent overflow.

    Args:
        T: temperature.
        TT: threshold temperature.
        CFMAX: degree-day factor.
        c_m: exponential coefficient (>0).
        SWE: snow water equivalent.

    Returns:
        M: snowmelt.
    """
    x = torch.clamp(c_m * (T - TT), min=-20.0, max=20.0)
    raw = torch.exp(x) - 1.0
    M = CFMAX * torch.clamp(raw, min=0.0)
    M = torch.minimum(M, SWE.clamp(min=0.0))
    return M


def refreezing(T, TT, CFR, CFMAX, LW):
    """S6: Refreezing formula.

    R_f = CFR * CFMAX * max(TT - T, 0),  capped by LW.

    Args:
        T: temperature.
        TT: threshold temperature.
        CFR: refreezing coefficient.
        CFMAX: degree-day factor.
        LW: liquid water content in snowpack.

    Returns:
        R_f: refreezing amount.
    """
    R_f = CFR * CFMAX * torch.clamp(TT - T, min=0.0)
    R_f = torch.minimum(R_f, LW.clamp(min=0.0))
    return R_f
