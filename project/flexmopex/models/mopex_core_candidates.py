"""Candidate interception formulas E (bounded linear cosine) and F (bounded
logistic cosine) with explicit PET-cap semantics S0/S1/S2.

Experimental only.  Production ``models/mopex_core.py`` and the 2x2 study
``models/mopex_core_v1.py`` are NOT modified.

Candidate semantics (shared interception-block parameterization):

  s_E(t; kappa, phi) = 0.5 * (1 + kappa * cos(2*pi*(d_t - phi)/365.25))   kappa in [0,1]
  s_F(t; kappa, phi) = sigmoid(kappa * cos(2*pi*(d_t - phi)/365.25))     kappa in [0, KAPPA_MAX]
  I_t = w_int * P_t * s_t

  * ``w_int`` is the only explicit multiplicative amplitude control;
  * ``kappa`` controls seasonal contrast/sharpness only (calendar mean of s
    is ~0.5 for every kappa: E by construction, F via sigma(x)+sigma(-x)=1);
  * ``phi`` controls seasonal timing only;
  * 0 <= s_t <= 1 so I_t <= P_t automatically (precipitation cap redundant,
    kept as an explicit safety minimum for numerical parity).

Semantics (identical downstream sequence in all cases):

  S0: production-style PET cap ``min(flux_potential, min(P, PET_effective))``
      and shared PET budget ``pet_for_soil = relu(PET_effective - flux_i)``.
  S1: V1 independent interception-loss semantics: same PET cap, but
      ``pet_for_soil = PET_effective`` (interception does not consume the
      soil PET budget).
  S2: experimental independent-loss semantics WITHOUT the interception PET
      cap: ``flux_i_pot = min(flux_potential, P)`` (I <= P preserved) and
      ``pet_for_soil = PET_effective``.

The two internal parameters reuse the existing two interception slots
(alpha-slot index 8 -> kappa, is_time-slot index 9 -> phi), so the network
output dimension, parameter ordering and AIC bookkeeping are unchanged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple

# Upper bound of the kappa range for candidate F (bounded logistic cosine).
# Justified by the deterministic shape/gradient sweep in
# scripts/screen_interception_candidates.py (weak-to-strong seasonality with
# retained d(s)/dkappa and d(s)/dphi gradients; near-total saturation avoided).
KAPPA_MAX = 5.0


# ---------------------------------------------------------------------------
# seasonal gates
# ---------------------------------------------------------------------------
def _phase_rad(doy: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    phi_safe = torch.clamp(phi, 0.0, 365.0)
    return 2.0 * torch.pi * (doy - phi_safe) / 365.25


def season_linear(doy: torch.Tensor, kappa: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Candidate E: bounded linear cosine, s in [0, 1] for kappa in [0, 1]."""
    return 0.5 * (1.0 + kappa * torch.cos(_phase_rad(doy, phi)))


def season_logistic(doy: torch.Tensor, kappa: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Candidate F: bounded logistic cosine, s in (0, 1) for any kappa >= 0."""
    return torch.sigmoid(kappa * torch.cos(_phase_rad(doy, phi)))


# ---------------------------------------------------------------------------
# step function
# ---------------------------------------------------------------------------
def _mopex_step_candidate(
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
    kappa: torch.Tensor,
    phi: torch.Tensor,
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
    season_mode: str = "linear",
    pet_cap: bool = True,
    pet_independent: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Body shared by candidates E/F under S0/S1/S2 semantics.

    ``season_mode``: "linear" (candidate E) or "logistic" (candidate F).
    ``pet_cap``: apply the production interception PET cap (S0/S1) or not (S2).
    ``pet_independent``: V1-style independent interception loss (S1/S2) or
    production shared PET budget (S0).
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
    # 2. Interception Module (bounded seasonal gate)
    # ============================================================
    if season_mode == "linear":
        season = season_linear(doy, kappa, phi)
    elif season_mode == "logistic":
        season = season_logistic(doy, kappa, phi)
    else:
        raise ValueError(f"Unknown season_mode {season_mode!r}")

    flux_potential = P * season
    if pet_cap:
        # Production safety cap (S0/S1): interception never exceeds P or PET.
        flux_i_pot = torch.minimum(flux_potential, torch.minimum(P, PET_effective))
    else:
        # S2: no PET cap; precipitation bound preserved (redundant for s<=1).
        flux_i_pot = torch.minimum(flux_potential, P)

    # w_int is the only multiplicative amplitude control, applied after caps
    # (identical placement to production V0).
    flux_i = flux_i_pot * w_int
    P_through = P - flux_i

    if pet_independent:
        # S1/S2: interception loss is independent of the soil PET budget.
        pet_for_soil = PET_effective
    else:
        # S0: interception consumes PET before soil ET.
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


def _make_step(season_mode: str, pet_cap: bool, pet_independent: bool):
    def step(
        # --- Inputs ---
        P, T, PET, doy,
        # --- Structural Weights ---
        w_phen, w_int, w_snow, w_sub,
        # --- Parameters ---
        Sb1, tw, tu, Se, tc, ddf, tcrit, Sb2, kappa, phi, tmin, tmax,
        # --- States ---
        S1, S2, Sc1, Sc2, Sn,
        nearzero: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _mopex_step_candidate(
            P, T, PET, doy, w_phen, w_int, w_snow, w_sub,
            Sb1, tw, tu, Se, tc, ddf, tcrit, Sb2, kappa, phi, tmin, tmax,
            S1, S2, Sc1, Sc2, Sn, nearzero,
            season_mode=season_mode, pet_cap=pet_cap, pet_independent=pet_independent,
        )
    step.__name__ = f"mopex_step_{'E' if season_mode == 'linear' else 'F'}_{'S0' if pet_cap and not pet_independent else 'S1' if pet_cap else 'S2'}"
    return step


# Candidate E (bounded linear cosine): S0 / S1 / S2
mopex_step_E_S0 = _make_step("linear", pet_cap=True, pet_independent=False)
mopex_step_E_S1 = _make_step("linear", pet_cap=True, pet_independent=True)
mopex_step_E_S2 = _make_step("linear", pet_cap=False, pet_independent=True)

# Candidate F (bounded logistic cosine): S0 / S1 / S2
mopex_step_F_S0 = _make_step("logistic", pet_cap=True, pet_independent=False)
mopex_step_F_S1 = _make_step("logistic", pet_cap=True, pet_independent=True)
mopex_step_F_S2 = _make_step("logistic", pet_cap=False, pet_independent=True)


# ---------------------------------------------------------------------------
# cap-free interception diagnostics (no state loop needed: interception only
# depends on P, PET_effective and the seasonal gate)
# ---------------------------------------------------------------------------
def interception_series(
    P: torch.Tensor,
    PET: torch.Tensor,
    doy: torch.Tensor,
    season_mode: str,
    kappa: torch.Tensor,
    phi: torch.Tensor,
    w_int: torch.Tensor,
    pet_cap: bool,
    w_phen: torch.Tensor,
    T: torch.Tensor,
    tmin: torch.Tensor,
    tmax: torch.Tensor,
) -> torch.Tensor:
    """Per-timestep interception flux I_t (after all caps), no state loop.

    Shapes broadcast: P/PET/doy (T, B) or (T, B, S) with kappa/phi/w_int
    (B, S).  Returns I_t (T, B, S).
    """
    trange = torch.clamp(tmax - tmin, min=0.1)
    flux_gsi = torch.clamp((T - tmin) / trange, 0.0, 1.0)
    PET_effective = w_phen * (PET * flux_gsi) + (1.0 - w_phen) * PET

    if season_mode == "linear":
        season = season_linear(doy, kappa, phi)
    elif season_mode == "logistic":
        season = season_logistic(doy, kappa, phi)
    elif season_mode == "original":
        season = kappa * 0.5 * (torch.cos(_phase_rad(doy, phi)) + 1.0)
    elif season_mode == "normalized":
        # B/D: g_shape with unit annual mean, C_REF amplitude (alpha=kappa slot)
        from project.flexmopex.models import mopex_core_v1 as v1
        norm_mean = v1.decoupled_norm_mean(kappa, phi)
        season = v1.decoupled_shape(doy, kappa, phi, norm_mean) * v1.C_REF
    else:
        raise ValueError(season_mode)

    flux_potential = P * season
    if pet_cap:
        flux_i_pot = torch.minimum(flux_potential, torch.minimum(P, PET_effective))
    else:
        flux_i_pot = torch.minimum(flux_potential, P)
    return flux_i_pot * w_int
