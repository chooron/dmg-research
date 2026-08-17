"""Experimental V1 / amplitude-decoupled interception steps (interception 2x2 study).

This module is strictly experimental.  ``models/mopex_core.py`` (production V0)
is NOT modified.  The 2x2 factorial axes are:

  PET semantics:
    V0 (production): interception consumes the shared PET budget before soil ET
        (``pet_for_soil = relu(PET_effective - flux_i)``).
    V1 (experimental): interception loss is independent of the soil PET budget
        (``pet_for_soil = PET_effective``); interception itself is still computed
        exactly as in production, including its ``min(P, PET_effective)`` cap.

  Interception parameterization:
    original (production): ``flux_potential = alpha * P * season_factor``
        (``alpha`` controls both amplitude and seasonal shape).
    decoupled (experimental): ``flux_potential = P * C_REF * g_shape(t)`` where
        ``g_shape`` has unit annual mean so ``alpha`` controls only seasonal
        shape/contrast and ``is_time`` only seasonal timing; the overall
        amplitude is the fixed, non-learned constant ``C_REF``.  ``w_int``
        remains the single amplitude coordinate.

Decoupled formulation (per experiment protocol):

    g_raw(t; alpha, phi) = smoothpositive[alpha + (1-alpha) * cos(2*pi*(t-phi)/365.25)]
        where smoothpositive is the production seasonal approximation
        0.5*(cos+1) applied to cos (the existing differentiable positive
        approximation), i.e. g_raw = alpha + (1-alpha) * season_factor in [0,1].

    g_shape(t) = g_raw(t) / mean_over_fixed_annual_grid(g_raw)   (annual mean == 1)

    I_pot,t = P_t * C_REF * g_shape(t),   C_REF = mean(g_raw(alpha_ref=0.5))
    I_t     = w_int * I_pot,t

C_REF is fixed (not learned, not basin-specific, excluded from AIC, not chosen
from learned outcomes).  The production interception caps are preserved:
``flux_i_pot = min(flux_potential, min(P, PET_effective))``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple

# ---------------------------------------------------------------------------
# Decoupled-interception constants and helpers
# ---------------------------------------------------------------------------

# Fixed full-annual phase grid (one full cycle sampled symmetrically at 366
# phases, 0 <= phase < 365.25).  Symmetric sampling makes the annual mean of
# g_raw exactly invariant to the phase shift is_time (permutation of the
# sample points), i.e. is_time changes timing but never normalization
# amplitude.  Used only for the annual normalization; independent of the
# forcing record.
PHASE_GRID = 365.25 * torch.arange(366) / 366

# Deterministic midpoint of the verified alpha range [0.0, 1.0]
# (see MOPEX_PARAMS_BOUNDS in models/base_mopex.py / mopex_core.py).
ALPHA_REF = 0.5

_EPS = 1e-6


def _season_factor(rad: torch.Tensor) -> torch.Tensor:
    """Production seasonal factor: differentiable positive approximation of cos."""
    return 0.5 * (torch.cos(rad) + 1.0)


def _g_raw(phase: torch.Tensor, alpha: torch.Tensor, is_time: torch.Tensor) -> torch.Tensor:
    """Seasonal raw factor g_raw = alpha + (1-alpha)*smoothpositive(cos) in [alpha, 1]."""
    is_time_safe = torch.clamp(is_time, 0.0, 365.0)
    rad = 2.0 * torch.pi * (phase - is_time_safe) / 365.25
    return alpha + (1.0 - alpha) * _season_factor(rad)


def decoupled_norm_mean(
    alpha: torch.Tensor,
    is_time: torch.Tensor,
    eps: float = _EPS,
) -> torch.Tensor:
    """Annual mean of g_raw over the fixed full-annual phase grid.

    Vectorized over the grid; differentiable w.r.t. ``alpha`` (no detach).
    Independent of ``is_time`` by construction (the grid covers a full year).
    Shape: same as ``alpha``.
    """
    grid = PHASE_GRID.to(device=alpha.device, dtype=alpha.dtype).view(-1, *([1] * alpha.dim()))
    raw = _g_raw(grid, alpha, is_time)  # (366, *alpha.shape)
    return raw.mean(dim=0)              # (*alpha.shape)


def decoupled_shape(
    doy: torch.Tensor,
    alpha: torch.Tensor,
    is_time: torch.Tensor,
    norm_mean: torch.Tensor,
    eps: float = _EPS,
) -> torch.Tensor:
    """Normalized seasonal shape g_shape(t) = g_raw(t) / mean(g_raw); annual mean ~ 1."""
    raw = _g_raw(doy, alpha, is_time)
    return raw / (norm_mean + eps)


def interception_c_ref(alpha_ref: float = ALPHA_REF) -> float:
    """Fixed reference amplitude scale: annual-mean g_raw at alpha_ref (== 0.75)."""
    alpha_t = torch.as_tensor(alpha_ref, dtype=PHASE_GRID.dtype)
    is_time_t = torch.as_tensor(0.0, dtype=PHASE_GRID.dtype)
    raw = _g_raw(PHASE_GRID.unsqueeze(-1), alpha_t, is_time_t)
    return float(raw.mean())


# Fixed, non-learned, non-basin-specific reference amplitude (excluded from AIC).
C_REF = interception_c_ref()


# ---------------------------------------------------------------------------
# 2x2 step functions
# ---------------------------------------------------------------------------

def _mopex_step_impl(
    # --- Inputs ---
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    doy: torch.Tensor,
    # --- Structural Weights ---
    w_phen: torch.Tensor,
    w_int: torch.Tensor,
    w_snow: torch.Tensor,
    w_sub: torch.Tensor,
    # --- Parameters ---
    Sb1: torch.Tensor,
    tw: torch.Tensor,
    tu: torch.Tensor,
    Se: torch.Tensor,
    tc: torch.Tensor,
    ddf: torch.Tensor,
    tcrit: torch.Tensor,
    Sb2: torch.Tensor,
    alpha: torch.Tensor,
    is_time: torch.Tensor,
    tmin: torch.Tensor,
    tmax: torch.Tensor,
    # --- States ---
    S1: torch.Tensor,
    S2: torch.Tensor,
    Sc1: torch.Tensor,
    Sc2: torch.Tensor,
    Sn: torch.Tensor,
    nearzero: float = 1e-6,
    *,
    pet_independent: bool = False,
    decoupled: bool = False,
    season_shape: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Body shared by all experimental step variants.

    ``decoupled=True`` requires ``season_shape`` (per-timestep normalized
    seasonal shape, precomputed once per forward in the model loop).
    """
    # ============================================================
    # 0. Guards
    # ============================================================
    S1 = F.relu(S1)
    S2 = F.relu(S2)
    Sc1 = F.relu(Sc1)
    Sc2 = F.relu(Sc2)
    Sn = F.relu(Sn)

    # ============================================================
    # 1. Phenology Module (Soft Switch) - GSI with trange
    # ============================================================
    trange = torch.clamp(tmax - tmin, min=0.1)
    flux_gsi = torch.clamp((T - tmin) / trange, 0.0, 1.0)

    PET_bio = PET * flux_gsi
    PET_effective = w_phen * PET_bio + (1.0 - w_phen) * PET

    # ============================================================
    # 2. Interception Module (Flux Gating)
    # ============================================================
    if decoupled:
        # Amplitude-decoupled: overall amplitude is the fixed C_REF; alpha
        # only shapes seasonality (unit annual mean), is_time only timing.
        flux_potential = P * C_REF * season_shape
    else:
        # Production parameterization: alpha scales amplitude and seasonality.
        is_time_safe = torch.clamp(is_time, 0.0, 365.0)
        rad = 2.0 * torch.pi * (doy - is_time_safe) / 365.0
        season_factor = 0.5 * (torch.cos(rad) + 1.0)
        flux_potential = alpha * P * season_factor

    # Production safety cap preserved in all arms.
    flux_i_pot = torch.minimum(flux_potential, torch.minimum(P, PET_effective))
    flux_i = flux_i_pot * w_int
    P_through = P - flux_i

    if pet_independent:
        # V1: interception loss is independent of the soil PET budget.
        pet_for_soil = PET_effective
    else:
        # V0 (production): interception consumes PET before soil ET.
        pet_for_soil = F.relu(PET_effective - flux_i)

    # ============================================================
    # 3. Snow Module (Soft Switch)
    # ============================================================
    is_rain = torch.sigmoid(T - tcrit)

    P_bypass = P_through * is_rain + P_through * (1.0 - is_rain) * (1.0 - w_snow)
    P_to_snow = P_through * (1.0 - is_rain) * w_snow

    melt_drive = torch.sigmoid(T - tcrit) * F.softplus(T - tcrit)
    flux_qn = torch.minimum(melt_drive * ddf, Sn)

    Sn_new = Sn + P_to_snow - flux_qn
    P_eff = P_bypass + flux_qn

    # ============================================================
    # 4. Surface Soil Module (S1)
    # ============================================================
    flux_q1f = F.relu((S1 + P_eff) - Sb1)
    S1 = S1 + P_eff - flux_q1f

    ratio_s1 = torch.clamp(S1 / (Sb1 + nearzero), max=1.0)
    flux_et1 = torch.minimum(pet_for_soil * ratio_s1, S1)
    S1 = S1 - flux_et1

    flux_qw = S1 * (1.0 - torch.exp(-1.0 / (tw + nearzero)))
    S1_new = S1 - flux_qw

    # ============================================================
    # 5. Subsurface Module (S2)
    # ============================================================
    S2 = S2 + flux_qw

    flux_q2f_pot = F.relu(S2 - Sb2)
    flux_q2f = flux_q2f_pot * w_sub
    S2 = S2 - flux_q2f

    flux_q2u = S2 * (1.0 - torch.exp(-1.0 / (tu + nearzero)))
    S2 = S2 - flux_q2u

    remaining_pet = F.relu(pet_for_soil - flux_et1)
    ratio_s2 = torch.clamp(S2 / (Se + nearzero), max=1.0)
    flux_et2 = torch.minimum(remaining_pet * ratio_s2, S2)
    S2_new = S2 - flux_et2

    # ============================================================
    # 6. Routing
    # ============================================================
    Sc1 = Sc1 + flux_q1f + flux_q2f
    flux_qf = Sc1 * (1.0 - torch.exp(-1.0 / (tc + nearzero)))
    Sc1_new = Sc1 - flux_qf

    Sc2 = Sc2 + flux_q2u
    flux_qs = Sc2 * (1.0 - torch.exp(-1.0 / (tc + nearzero)))
    Sc2_new = Sc2 - flux_qs

    # ============================================================
    # Summary
    # ============================================================
    ET_total = flux_et1 + flux_et2 + flux_i
    Q_total = flux_qf + flux_qs

    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new


def mopex_step_v1(
    # --- Inputs ---
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    doy: torch.Tensor,
    # --- Structural Weights ---
    w_phen: torch.Tensor,
    w_int: torch.Tensor,
    w_snow: torch.Tensor,
    w_sub: torch.Tensor,
    # --- Parameters ---
    Sb1: torch.Tensor,
    tw: torch.Tensor,
    tu: torch.Tensor,
    Se: torch.Tensor,
    tc: torch.Tensor,
    ddf: torch.Tensor,
    tcrit: torch.Tensor,
    Sb2: torch.Tensor,
    alpha: torch.Tensor,
    is_time: torch.Tensor,
    tmin: torch.Tensor,
    tmax: torch.Tensor,
    # --- States ---
    S1: torch.Tensor,
    S2: torch.Tensor,
    Sc1: torch.Tensor,
    Sc2: torch.Tensor,
    Sn: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """V1: original parameterization, independent interception-loss PET semantics."""
    return _mopex_step_impl(
        P, T, PET, doy, w_phen, w_int, w_snow, w_sub,
        Sb1, tw, tu, Se, tc, ddf, tcrit, Sb2, alpha, is_time, tmin, tmax,
        S1, S2, Sc1, Sc2, Sn, nearzero,
        pet_independent=True, decoupled=False,
    )


def mopex_step_decoupled(
    # --- Inputs ---
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    doy: torch.Tensor,
    # --- Structural Weights ---
    w_phen: torch.Tensor,
    w_int: torch.Tensor,
    w_snow: torch.Tensor,
    w_sub: torch.Tensor,
    # --- Parameters ---
    Sb1: torch.Tensor,
    tw: torch.Tensor,
    tu: torch.Tensor,
    Se: torch.Tensor,
    tc: torch.Tensor,
    ddf: torch.Tensor,
    tcrit: torch.Tensor,
    Sb2: torch.Tensor,
    alpha: torch.Tensor,
    is_time: torch.Tensor,
    tmin: torch.Tensor,
    tmax: torch.Tensor,
    # --- States ---
    S1: torch.Tensor,
    S2: torch.Tensor,
    Sc1: torch.Tensor,
    Sc2: torch.Tensor,
    Sn: torch.Tensor,
    season_shape: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """V0 PET semantics + amplitude-decoupled interception."""
    return _mopex_step_impl(
        P, T, PET, doy, w_phen, w_int, w_snow, w_sub,
        Sb1, tw, tu, Se, tc, ddf, tcrit, Sb2, alpha, is_time, tmin, tmax,
        S1, S2, Sc1, Sc2, Sn, nearzero,
        pet_independent=False, decoupled=True, season_shape=season_shape,
    )


def mopex_step_v1_decoupled(
    # --- Inputs ---
    P: torch.Tensor,
    T: torch.Tensor,
    PET: torch.Tensor,
    doy: torch.Tensor,
    # --- Structural Weights ---
    w_phen: torch.Tensor,
    w_int: torch.Tensor,
    w_snow: torch.Tensor,
    w_sub: torch.Tensor,
    # --- Parameters ---
    Sb1: torch.Tensor,
    tw: torch.Tensor,
    tu: torch.Tensor,
    Se: torch.Tensor,
    tc: torch.Tensor,
    ddf: torch.Tensor,
    tcrit: torch.Tensor,
    Sb2: torch.Tensor,
    alpha: torch.Tensor,
    is_time: torch.Tensor,
    tmin: torch.Tensor,
    tmax: torch.Tensor,
    # --- States ---
    S1: torch.Tensor,
    S2: torch.Tensor,
    Sc1: torch.Tensor,
    Sc2: torch.Tensor,
    Sn: torch.Tensor,
    season_shape: torch.Tensor,
    nearzero: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """V1 PET semantics + amplitude-decoupled interception."""
    return _mopex_step_impl(
        P, T, PET, doy, w_phen, w_int, w_snow, w_sub,
        Sb1, tw, tu, Se, tc, ddf, tcrit, Sb2, alpha, is_time, tmin, tmax,
        S1, S2, Sc1, Sc2, Sn, nearzero,
        pet_independent=True, decoupled=True, season_shape=season_shape,
    )
