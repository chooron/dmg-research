"""Shared MOPEX-family process formulas.

The MOPEX core modules differ in store layout, but reuse these evaporation,
snow partition, routing, interception, and phenology equations.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import torch
import torch.nn.functional as F


PET_BUDGET_MODES = ("legacy", "interception_first", "soil_et_first")
PET_LIMITER_MODES = ("hard", "smooth")


_MOPEX_TRAINING_CONTEXT: ContextVar[tuple[float, float, float, str, str]] = ContextVar(
    "mopex_training_context", default=(1.0, 1.0, 50.0, "legacy", "hard")
)


@contextmanager
def mopex_training_context(
    *,
    lambda_i: float = 1.0,
    lambda_p: float = 1.0,
    beta: float = 50.0,
    pet_budget: str = "legacy",
    pet_limiter: str = "hard",
):
    """LEGACY / DIAGNOSTIC / REFERENCE ONLY — not the frozen MOPEX4 production forward.

    The frozen production ``mopex4_step`` uses a single fixed interception-first
    water path with the exact hard limiter and ignores this context entirely.
    This context exists only to reproduce legacy continuation behavior for
    MOPEX5 and for historical MOPEX4 diagnostics.

    The physical flux is evaluated first in all cases.  The context scales
    the resulting interception flux and mixes the real phenology GSI with the
    identity PET path.  ``pet_budget`` selects how interception and soil ET
    share the daily PET demand:

    - ``legacy``: current production behavior; ET1/ET2 each see full PET and
      interception is not PET-limited.
    - ``interception_first``: interception is limited to PET first, then ET1
      and ET2 consume the remaining PET budget in the existing state order.
    - ``soil_et_first``: ET1 keeps the current full-PET priority, the residual
      budget is offered to interception, and ET2 consumes the remainder.

    Defaults exactly match the production equations.
    """
    if pet_budget not in PET_BUDGET_MODES:
        raise ValueError(f"invalid MOPEX pet_budget mode: {pet_budget}")
    if pet_limiter not in PET_LIMITER_MODES:
        raise ValueError(f"invalid MOPEX pet_limiter mode: {pet_limiter}")
    values = (float(lambda_i), float(lambda_p), float(beta), pet_budget, pet_limiter)
    if not all(torch.isfinite(torch.tensor(value)) for value in values[:3]):
        raise ValueError("MOPEX continuation values must be finite")
    if not 0.0 <= values[0] <= 1.0 or not 0.0 <= values[1] <= 1.0 or values[2] <= 0.0:
        raise ValueError("invalid MOPEX continuation values")
    token = _MOPEX_TRAINING_CONTEXT.set(values)
    try:
        yield
    finally:
        _MOPEX_TRAINING_CONTEXT.reset(token)


def _training_values() -> tuple[float, float, float]:
    return _MOPEX_TRAINING_CONTEXT.get()[:3]


def _pet_budget_mode() -> str:
    return _MOPEX_TRAINING_CONTEXT.get()[3]


def _pet_limiter() -> str:
    return _MOPEX_TRAINING_CONTEXT.get()[4]


def mopex_pet_budget_limit(
    i_pot: torch.Tensor,
    pet_available: torch.Tensor,
    *,
    smooth: bool = False,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """Bound interception potential by the currently available PET demand.

    This is a diagnostic budget limiter, not a new learnable parameter.
    ``smooth=False`` uses the exact hard budget ``min(i_pot, pet_available)``;
    ``smooth=True`` uses a smooth non-trainable saturation ``i_pot*pet/(i_pot+pet)``
    that is a lower bound of the hard budget.  Both keep ``0 <= I <= i_pot`` and
    ``0 <= I <= pet_available``.
    """
    pet_available = torch.clamp(pet_available, min=0.0)
    if smooth:
        denom = i_pot + pet_available + nearzero
        return i_pot * pet_available / denom
    return torch.minimum(i_pot, pet_available)


def mopex_evap_7(
    S: torch.Tensor,
    Smax: torch.Tensor,
    Ep: torch.Tensor,
    dt: float = 1.0,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    ratio = torch.clamp(S / (Smax + nearzero), max=1.0)
    return torch.minimum(Ep * ratio * dt, S)


def mopex_saturation_1(
    P: torch.Tensor,
    S: torch.Tensor,
    Smax: torch.Tensor,
    r: float = 0.01,
    e: float = 5.0,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    threshold = Smax * (1.0 - r)
    scale = Smax * r * e + nearzero
    return P * torch.sigmoid((S - threshold) / scale)


def mopex_baseflow_1(k: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    return torch.minimum(k * S, S)


def mopex_recharge_3(k: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    return torch.minimum(k * S, S)


def mopex_snowfall_1(
    P: torch.Tensor, T: torch.Tensor, tcrit: torch.Tensor, r: float = 0.01
) -> torch.Tensor:
    scale = torch.abs(tcrit) * r + r
    return P * torch.sigmoid((tcrit - T) / (scale + 1e-6))


def mopex_rainfall_1(
    P: torch.Tensor, T: torch.Tensor, tcrit: torch.Tensor, r: float = 0.01
) -> torch.Tensor:
    scale = torch.abs(tcrit) * r + r
    return P * torch.sigmoid((T - tcrit) / (scale + 1e-6))


def mopex_melt_1(
    ddf: torch.Tensor,
    tcrit: torch.Tensor,
    T: torch.Tensor,
    Sn: torch.Tensor,
    dt: float = 1.0,
) -> torch.Tensor:
    melt_drive = torch.sigmoid(T - tcrit) * F.softplus(T - tcrit)
    return torch.minimum(ddf * melt_drive * dt, Sn)


# ---------------------------------------------------------------------------
# Deprecated F0 legacy interception implementation
# ---------------------------------------------------------------------------
# DEPRECATED / LEGACY REPRODUCTION ONLY.
#
# This helper is retained so previously reported MOPEX4 experiments and
# MOPEX5's unchanged implementation remain reproducible.  The old form couples
# seasonal level and amplitude through alpha, can exchange interception with
# soil/storage parameters, and its softplus plus precipitation cap can reduce
# gradients in parts of parameter space.  Earlier MOPEX4 diagnostics also found
# material optimization sensitivity to the learned seasonal phase.  These are
# diagnostic observations, not universal causal claims; this helper is not the
# recommended MOPEX4 interception structure.
def _mopex_interception_4_legacy(
    flux_pr: torch.Tensor,
    doy: torch.Tensor,
    alpha: torch.Tensor,
    is_time: torch.Tensor,
    tmax: float = 365.25,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    del nearzero
    lambda_i, _lambda_p, beta = _training_values()
    radians = 2.0 * torch.pi * (doy - is_time) / tmax
    season_raw = alpha + (1.0 - alpha) * torch.cos(radians)
    fraction = F.softplus(beta * season_raw) / beta
    legacy_raw = torch.minimum(fraction * flux_pr, flux_pr)
    return lambda_i * legacy_raw


# Compatibility name used by existing MOPEX5 and legacy diagnostics.
def mopex_interception_4(
    flux_pr: torch.Tensor,
    doy: torch.Tensor,
    alpha: torch.Tensor,
    is_time: torch.Tensor,
    tmax: float = 365.25,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    return _mopex_interception_4_legacy(
        flux_pr, doy, alpha, is_time, tmax=tmax, nearzero=nearzero
    )


# ---------------------------------------------------------------------------
# MOPEX4 Liu-type interception candidate
# ---------------------------------------------------------------------------
# This is an adapted/simplified Liu-type smooth saturation kernel, not the
# complete Liu model.  Rutter (1971) and Gash (1979) support canopy storage,
# saturation, and wet-canopy evaporation as core interception concepts.  Liu
# (1997, 2001) supports smooth interception formulations with canopy
# storage/closure meaning, while de Groen & Savenije (2006) supports an
# effective threshold conceptualization at daily aggregation.  The present
# kernel extracts only the smooth wetting/saturation term; it does not claim
# equivalence to the complete Liu model, which also includes wet-canopy
# evaporation and rainfall-intensity terms.
#
# References:
# Liu, S. (1997), Ecological Modelling 99, 151-159,
# DOI: 10.1016/S0304-3800(97)01948-0.
# Liu, S. (2001), Hydrological Processes, DOI: 10.1002/hyp.264.
# Gash, J.H.C. (1979), Q.J.R. Meteorol. Soc. 105, 43-55,
# DOI: 10.1002/qj.49710544304.
# Rutter, A.J. et al. (1971), Agricultural Meteorology 9, 367-384,
# DOI: 10.1016/0002-1571(71)90034-3.
# de Groen, M.M. & Savenije, H.H.G. (2006), Water Resources Research,
# DOI: 10.1029/2006WR005013.
def mopex_interception_4_liu(
    flux_pr: torch.Tensor,
    S_eff: torch.Tensor,
    c: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """MOPEX4 adapted Liu-type daily interception kernel.

    ``S_eff`` is an effective daily interception threshold, not a literal
    single-event canopy storage capacity.  ``c`` is effective canopy closure /
    wetting efficiency.  Both inputs are physical-space parameters and are
    bounded by the MOPEX4 parameter transform before this function is called.
    """
    lambda_i, _lambda_p, _beta = _training_values()
    safe_S_eff = torch.clamp(S_eff, min=nearzero)
    x = c * torch.clamp(flux_pr, min=0.0) / safe_S_eff
    physical = safe_S_eff * (-torch.expm1(-x))
    return lambda_i * physical


# Explicit alias for diagnostics that want to state the candidate name.
mopex_interception_4_liu_type = mopex_interception_4_liu


# Frozen final MOPEX4 interception: T1a single-threshold kernel (c = 1 fixed).
# Kept as a reference/diagnostic helper; not used by the production hot path.
def mopex_interception_4_t1a(
    flux_pr: torch.Tensor,
    S_eff: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """MOPEX4 T1a interception potential with c fixed at 1 (reference/diagnostic).

    ``I_pot = S_eff * (-expm1(-Pr / S_eff))`` on post-snow liquid rainfall.
    """
    safe_S_eff = torch.clamp(S_eff, min=nearzero)
    x = torch.clamp(flux_pr, min=0.0) / safe_S_eff
    return safe_S_eff * (-torch.expm1(-x))


# ---------------------------------------------------------------------------
# Frozen final MOPEX4 interception: two-parameter Liu kernel (S_eff and c)
# ---------------------------------------------------------------------------
def mopex_interception_4_liu2(
    flux_pr: torch.Tensor,
    S_eff: torch.Tensor,
    c: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """Final MOPEX4 two-parameter interception potential.

    ``I_pot = S_eff * (-expm1(-c * Pr / S_eff))`` on post-snow liquid rainfall.
    This is the fixed compute graph used by the frozen production
    ``mopex4_step``: no continuation context, no seasonal phase, and both
    ``S_eff`` and ``c`` carry active gradient paths.  ``S_eff`` is an effective
    daily interception threshold [mm]; ``c`` is effective canopy closure /
    wetting efficiency [-].
    """
    safe_S_eff = torch.clamp(S_eff, min=nearzero)
    x = c * torch.clamp(flux_pr, min=0.0) / safe_S_eff
    return safe_S_eff * (-torch.expm1(-x))

def mopex_interception_4_circular(
    flux_pr: torch.Tensor,
    doy: torch.Tensor,
    alpha: torch.Tensor,
    phase_cos: torch.Tensor,
    phase_sin: torch.Tensor,
    tmax: float = 365.25,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    """Internal circular-phase equivalent of :func:`mopex_interception_4`.

    ``phase_cos`` and ``phase_sin`` describe the seasonal phase directly.
    This helper is intentionally separate from the scalar public function so
    ordinary MOPEX4/5 calls retain their exact parameter interface.
    """
    radius = torch.sqrt(phase_cos.square() + phase_sin.square() + nearzero)
    cos_phi = phase_cos / radius
    sin_phi = phase_sin / radius
    theta = 2.0 * torch.pi * doy / tmax
    seasonal_cosine = torch.cos(theta) * cos_phi + torch.sin(theta) * sin_phi
    fraction = alpha + (1.0 - alpha) * seasonal_cosine
    _lambda_i, _lambda_p, beta = _training_values()
    positive_fraction = F.softplus(fraction * beta) / beta
    return torch.minimum(positive_fraction * flux_pr, flux_pr) * _lambda_i


def mopex_phenology_1(
    T: torch.Tensor,
    tmin: torch.Tensor,
    trange: torch.Tensor,
    PET: torch.Tensor,
    nearzero: float = 1e-6,
) -> torch.Tensor:
    _lambda_i, lambda_p, _beta = _training_values()
    # trange is bounded to [1, 20] for MOPEX5.  Clamp only invalid external
    # inputs instead of adding epsilon, which biases the saturated GSI below 1.
    safe_trange = torch.clamp(trange, min=nearzero)
    gsi = torch.clamp((T - tmin) / safe_trange, 0.0, 1.0)
    return ((1.0 - lambda_p) + lambda_p * gsi) * PET
