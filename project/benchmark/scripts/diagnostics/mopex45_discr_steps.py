"""Benchmark-only diagnostic step variants for the MOPEX4/5 sequential
discretization audit.

These are NOT production code.  They exist only to (a) expose intermediate
fluxes of the production sequential discretization, and (b) implement two
diagnostic same-state variants of the MOPEX soil bucket.  Every variant
copies the production ``mopex4_step`` / ``mopex5_step`` runtime equations
exactly (importing the same flux helpers from ``dmotpy.models.flux.mopex``),
so the only differences are:

  S0  sequential (identical to production, plus flux recording)
  S1  same-state: q1f/qw are driven by the reference soil state
      S1_ref = S1 + pr + qn - et1 (the pre-interception state), instead of
      the post-interception state; the interception cap boundary
      (min(i_raw, S1_ref)) is unchanged.
  S2  same-state + deficit-proportional smooth cap (no hard min kink on the
      interception path): when i_raw + q1f_raw + qw_raw exceeds S1_ref, all
      three are scaled by S1_ref / total so no storage goes negative and the
      cap region remains differentiable.

Assumptions (recorded, not asserted as "correct MARRMoT"):
- MARRMoT formulates dS2/dt = pr + qn - et1 - i - q1f - qw as a continuous
  ODE; Euler-explicit discretization evaluates every flux from a common
  reference state.  S1 variants use S1_ref = S1 + pr + qn - et1.
- S2 is a purely mechanical smooth-cap test; it is not claimed to match any
  reference implementation.

Public APIs, parameter names/order/bounds, the IC path, the continuation
context defaults and all other models are untouched.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from dmotpy.models.flux.mopex import (
    mopex_baseflow_1 as baseflow_1,
    mopex_evap_7 as evap_7,
    mopex_interception_4 as interception_4,
    mopex_melt_1 as melt_1,
    mopex_phenology_1 as phenology_1,
    mopex_rainfall_1 as rainfall_1,
    mopex_recharge_3 as recharge_3,
    mopex_saturation_1 as saturation_1,
    mopex_snowfall_1 as snowfall_1,
)

# ---------------------------------------------------------------------------
# S0: production-identical sequential step with flux recording (MOPEX4)
# ---------------------------------------------------------------------------

def mopex4_step_diag(
    P: torch.Tensor, T: torch.Tensor, PET: torch.Tensor,
    tcrit: torch.Tensor, ddf: torch.Tensor, Sb1: torch.Tensor, tw: torch.Tensor,
    alpha: torch.Tensor, is_time: torch.Tensor, tu: torch.Tensor, Se: torch.Tensor,
    Sb2: torch.Tensor, tc: torch.Tensor,
    S1: torch.Tensor, S2: torch.Tensor, Sc1: torch.Tensor, Sc2: torch.Tensor, Sn: torch.Tensor,
    delta_t: float = 1.0, nearzero: float = 1e-6, *, doy: torch.Tensor = None,
) -> Tuple[torch.Tensor, ...]:
    """Line-by-line copy of dmotpy/models/core/mopex4.py::mopex4_step that
    additionally returns the intermediate soil fluxes in the last tuple slot.
    Returns (Q, ET, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new, fluxes)."""
    Sn = F.relu(Sn); S1 = F.relu(S1); S2 = F.relu(S2); Sc1 = F.relu(Sc1); Sc2 = F.relu(Sc2)

    flux_ps = snowfall_1(P, T, tcrit)
    flux_pr = rainfall_1(P, T, tcrit)
    flux_qn = melt_1(ddf, tcrit, T, Sn, delta_t)
    Sn = Sn + flux_ps
    Sn_new = Sn - flux_qn

    s1_pre_input = S1.detach().clone()
    S1 = S1 + flux_pr + flux_qn
    s1_post_input = S1.detach().clone()

    flux_et1 = evap_7(S1, Sb1, PET, delta_t, nearzero)
    flux_et1 = torch.minimum(flux_et1, S1)
    S1 = S1 - flux_et1
    s1_before_i = S1.detach().clone()

    flux_i_raw = interception_4(flux_pr, doy, alpha, is_time, nearzero=nearzero)
    flux_i = torch.minimum(flux_i_raw, S1)
    S1 = S1 - flux_i
    s1_after_i = S1.detach().clone()

    flux_q1f_raw = saturation_1(flux_pr + flux_qn, S1, Sb1, nearzero=nearzero)
    flux_q1f = torch.minimum(flux_q1f_raw, S1)
    S1 = S1 - flux_q1f
    flux_qw_raw = recharge_3(tw, S1)
    flux_qw = torch.minimum(flux_qw_raw, S1)
    S1_new = S1 - flux_qw

    S2 = S2 + flux_qw
    flux_q2f = saturation_1(flux_qw, S2, Sb2, nearzero=nearzero)
    flux_q2f = torch.minimum(flux_q2f, S2)
    S2 = S2 - flux_q2f
    flux_q2u = baseflow_1(tu, S2)
    S2 = S2 - flux_q2u
    se_abs = Se * Sb2
    flux_et2 = evap_7(S2, se_abs, PET, delta_t, nearzero)
    flux_et2 = torch.minimum(flux_et2, S2)
    S2_new = S2 - flux_et2

    Sc1 = Sc1 + flux_q1f + flux_q2f
    flux_qf = baseflow_1(tc, Sc1)
    Sc1_new = Sc1 - flux_qf
    Sc2 = Sc2 + flux_q2u
    flux_qs = baseflow_1(tc, Sc2)
    Sc2_new = Sc2 - flux_qs

    Q_total = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2 + flux_i

    fluxes = {
        "et1": flux_et1, "i_raw": flux_i_raw, "i": flux_i,
        "q1f_raw": flux_q1f_raw, "q1f": flux_q1f,
        "qw_raw": flux_qw_raw, "qw": flux_qw,
        "q2f": flux_q2f, "q2u": flux_q2u, "et2": flux_et2,
        "qf": flux_qf, "qs": flux_qs,
        "s1_pre_input": s1_pre_input, "s1_post_input": s1_post_input,
        "s1_before_i": s1_before_i, "s1_after_i": s1_after_i,
        "S1_new": S1_new, "S2_new": S2_new,
    }
    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new, fluxes


# ---------------------------------------------------------------------------
# S1: same-state variant (q1f/qw driven by pre-interception reference state)
# ---------------------------------------------------------------------------

def mopex4_step_samestate(
    P: torch.Tensor, T: torch.Tensor, PET: torch.Tensor,
    tcrit: torch.Tensor, ddf: torch.Tensor, Sb1: torch.Tensor, tw: torch.Tensor,
    alpha: torch.Tensor, is_time: torch.Tensor, tu: torch.Tensor, Se: torch.Tensor,
    Sb2: torch.Tensor, tc: torch.Tensor,
    S1: torch.Tensor, S2: torch.Tensor, Sc1: torch.Tensor, Sc2: torch.Tensor, Sn: torch.Tensor,
    delta_t: float = 1.0, nearzero: float = 1e-6, *, doy: torch.Tensor = None,
) -> Tuple[torch.Tensor, ...]:
    """Same-state soil bucket: et1, interception cap, q1f, qw all use
    S1_ref = S1 + pr + qn - et1.  Caps stay sequential min() so storage never
    goes negative (S1 variant)."""
    Sn = F.relu(Sn); S1 = F.relu(S1); S2 = F.relu(S2); Sc1 = F.relu(Sc1); Sc2 = F.relu(Sc2)

    flux_ps = snowfall_1(P, T, tcrit)
    flux_pr = rainfall_1(P, T, tcrit)
    flux_qn = melt_1(ddf, tcrit, T, Sn, delta_t)
    Sn = Sn + flux_ps
    Sn_new = Sn - flux_qn

    s1_pre_input = S1.detach().clone()
    S1_post = S1 + flux_pr + flux_qn
    s1_post_input = S1_post.detach().clone()

    flux_et1 = evap_7(S1_post, Sb1, PET, delta_t, nearzero)
    flux_et1 = torch.minimum(flux_et1, S1_post)
    S1_ref = S1_post - flux_et1                       # reference state
    s1_before_i = S1_ref.detach().clone()

    flux_i_raw = interception_4(flux_pr, doy, alpha, is_time, nearzero=nearzero)
    flux_i = torch.minimum(flux_i_raw, S1_ref)        # same cap boundary as production
    s1_after_i = (S1_ref - flux_i).detach().clone()

    flux_q1f_raw = saturation_1(flux_pr + flux_qn, S1_ref, Sb1, nearzero=nearzero)  # driven by S1_ref
    flux_q1f = torch.minimum(flux_q1f_raw, S1_ref - flux_i)
    flux_qw_raw = recharge_3(tw, S1_ref)                                             # driven by S1_ref
    flux_qw = torch.minimum(flux_qw_raw, S1_ref - flux_i - flux_q1f)
    S1_new = S1_ref - flux_i - flux_q1f - flux_qw

    S2 = S2 + flux_qw
    flux_q2f = saturation_1(flux_qw, S2, Sb2, nearzero=nearzero)
    flux_q2f = torch.minimum(flux_q2f, S2)
    S2 = S2 - flux_q2f
    flux_q2u = baseflow_1(tu, S2)
    S2 = S2 - flux_q2u
    se_abs = Se * Sb2
    flux_et2 = evap_7(S2, se_abs, PET, delta_t, nearzero)
    flux_et2 = torch.minimum(flux_et2, S2)
    S2_new = S2 - flux_et2

    Sc1 = Sc1 + flux_q1f + flux_q2f
    flux_qf = baseflow_1(tc, Sc1)
    Sc1_new = Sc1 - flux_qf
    Sc2 = Sc2 + flux_q2u
    flux_qs = baseflow_1(tc, Sc2)
    Sc2_new = Sc2 - flux_qs

    Q_total = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2 + flux_i

    fluxes = {
        "et1": flux_et1, "i_raw": flux_i_raw, "i": flux_i,
        "q1f_raw": flux_q1f_raw, "q1f": flux_q1f,
        "qw_raw": flux_qw_raw, "qw": flux_qw,
        "q2f": flux_q2f, "q2u": flux_q2u, "et2": flux_et2,
        "qf": flux_qf, "qs": flux_qs,
        "s1_pre_input": s1_pre_input, "s1_post_input": s1_post_input,
        "s1_before_i": s1_before_i, "s1_after_i": s1_after_i,
        "S1_new": S1_new, "S2_new": S2_new,
    }
    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new, fluxes


# ---------------------------------------------------------------------------
# S2: same-state + deficit-proportional smooth cap
# ---------------------------------------------------------------------------

def mopex4_step_samestate_smoothcap(
    P: torch.Tensor, T: torch.Tensor, PET: torch.Tensor,
    tcrit: torch.Tensor, ddf: torch.Tensor, Sb1: torch.Tensor, tw: torch.Tensor,
    alpha: torch.Tensor, is_time: torch.Tensor, tu: torch.Tensor, Se: torch.Tensor,
    Sb2: torch.Tensor, tc: torch.Tensor,
    S1: torch.Tensor, S2: torch.Tensor, Sc1: torch.Tensor, Sc2: torch.Tensor, Sn: torch.Tensor,
    delta_t: float = 1.0, nearzero: float = 1e-6, *, doy: torch.Tensor = None,
) -> Tuple[torch.Tensor, ...]:
    """Same-state soil bucket with deficit-proportional scaling instead of the
    hard min() cap: all soil outflows are computed from S1_ref and, if their
    sum exceeds S1_ref, scaled by S1_ref / total (smooth, differentiable)."""
    Sn = F.relu(Sn); S1 = F.relu(S1); S2 = F.relu(S2); Sc1 = F.relu(Sc1); Sc2 = F.relu(Sc2)

    flux_ps = snowfall_1(P, T, tcrit)
    flux_pr = rainfall_1(P, T, tcrit)
    flux_qn = melt_1(ddf, tcrit, T, Sn, delta_t)
    Sn = Sn + flux_ps
    Sn_new = Sn - flux_qn

    s1_pre_input = S1.detach().clone()
    S1_post = S1 + flux_pr + flux_qn
    s1_post_input = S1_post.detach().clone()

    flux_et1 = evap_7(S1_post, Sb1, PET, delta_t, nearzero)
    flux_et1 = torch.minimum(flux_et1, S1_post)
    S1_ref = S1_post - flux_et1
    s1_before_i = S1_ref.detach().clone()

    flux_i_raw = interception_4(flux_pr, doy, alpha, is_time, nearzero=nearzero)
    flux_q1f_raw = saturation_1(flux_pr + flux_qn, S1_ref, Sb1, nearzero=nearzero)
    flux_qw_raw = recharge_3(tw, S1_ref)
    total_raw = flux_i_raw + flux_q1f_raw + flux_qw_raw
    scale = torch.minimum(S1_ref / (total_raw + nearzero), torch.ones_like(S1_ref))
    flux_i = flux_i_raw * scale
    flux_q1f = flux_q1f_raw * scale
    flux_qw = flux_qw_raw * scale
    S1_new = S1_ref - flux_i - flux_q1f - flux_qw
    s1_after_i = (S1_ref - flux_i).detach().clone()

    S2 = S2 + flux_qw
    flux_q2f = saturation_1(flux_qw, S2, Sb2, nearzero=nearzero)
    flux_q2f = torch.minimum(flux_q2f, S2)
    S2 = S2 - flux_q2f
    flux_q2u = baseflow_1(tu, S2)
    S2 = S2 - flux_q2u
    se_abs = Se * Sb2
    flux_et2 = evap_7(S2, se_abs, PET, delta_t, nearzero)
    flux_et2 = torch.minimum(flux_et2, S2)
    S2_new = S2 - flux_et2

    Sc1 = Sc1 + flux_q1f + flux_q2f
    flux_qf = baseflow_1(tc, Sc1)
    Sc1_new = Sc1 - flux_qf
    Sc2 = Sc2 + flux_q2u
    flux_qs = baseflow_1(tc, Sc2)
    Sc2_new = Sc2 - flux_qs

    Q_total = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2 + flux_i

    fluxes = {
        "et1": flux_et1, "i_raw": flux_i_raw, "i": flux_i,
        "q1f_raw": flux_q1f_raw, "q1f": flux_q1f,
        "qw_raw": flux_qw_raw, "qw": flux_qw,
        "q2f": flux_q2f, "q2u": flux_q2u, "et2": flux_et2,
        "qf": flux_qf, "qs": flux_qs,
        "s1_pre_input": s1_pre_input, "s1_post_input": s1_post_input,
        "s1_before_i": s1_before_i, "s1_after_i": s1_after_i,
        "S1_new": S1_new, "S2_new": S2_new,
        "cap_scale": scale,
    }
    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new, fluxes


# ---------------------------------------------------------------------------
# MOPEX5: S0 (production-identical diag) and S1 (same-state)
# phenology_1 is left untouched.
# ---------------------------------------------------------------------------

def _mopex5_core(
    P: torch.Tensor, T: torch.Tensor, PET: torch.Tensor,
    tcrit: torch.Tensor, ddf: torch.Tensor, Sb1: torch.Tensor, tw: torch.Tensor,
    alpha: torch.Tensor, is_time: torch.Tensor, tmin: torch.Tensor, trange: torch.Tensor,
    tu: torch.Tensor, Se: torch.Tensor, Sb2: torch.Tensor, tc: torch.Tensor,
    S1: torch.Tensor, S2: torch.Tensor, Sc1: torch.Tensor, Sc2: torch.Tensor, Sn: torch.Tensor,
    delta_t: float = 1.0, nearzero: float = 1e-6, *, doy: torch.Tensor = None,
    soil_mode: str = "sequential",
):
    """Shared MOPEX5 body. soil_mode: 'sequential' (production), 'samestate'."""
    Sn = F.relu(Sn); S1 = F.relu(S1); S2 = F.relu(S2); Sc1 = F.relu(Sc1); Sc2 = F.relu(Sc2)

    PET_epc = phenology_1(T, tmin, trange, PET, nearzero)

    flux_ps = snowfall_1(P, T, tcrit)
    flux_pr = rainfall_1(P, T, tcrit)
    flux_qn = melt_1(ddf, tcrit, T, Sn, delta_t)
    Sn = Sn + flux_ps
    Sn_new = Sn - flux_qn

    s1_pre_input = S1.detach().clone()
    S1_post = S1 + flux_pr + flux_qn
    s1_post_input = S1_post.detach().clone()

    if soil_mode == "sequential":
        S1 = S1_post
        flux_et1 = evap_7(S1, Sb1, PET_epc, delta_t, nearzero)
        flux_et1 = torch.minimum(flux_et1, S1)
        S1 = S1 - flux_et1
        s1_before_i = S1.detach().clone()
        flux_i_raw = interception_4(flux_pr, doy, alpha, is_time, nearzero=nearzero)
        flux_i = torch.minimum(flux_i_raw, S1)
        S1 = S1 - flux_i
        s1_after_i = S1.detach().clone()
        flux_q1f_raw = saturation_1(flux_pr + flux_qn, S1, Sb1, nearzero=nearzero)
        flux_q1f = torch.minimum(flux_q1f_raw, S1)
        S1 = S1 - flux_q1f
        flux_qw_raw = recharge_3(tw, S1)
        flux_qw = torch.minimum(flux_qw_raw, S1)
        S1_new = S1 - flux_qw
    else:  # samestate
        flux_et1 = evap_7(S1_post, Sb1, PET_epc, delta_t, nearzero)
        flux_et1 = torch.minimum(flux_et1, S1_post)
        S1_ref = S1_post - flux_et1
        s1_before_i = S1_ref.detach().clone()
        flux_i_raw = interception_4(flux_pr, doy, alpha, is_time, nearzero=nearzero)
        flux_i = torch.minimum(flux_i_raw, S1_ref)
        s1_after_i = (S1_ref - flux_i).detach().clone()
        flux_q1f_raw = saturation_1(flux_pr + flux_qn, S1_ref, Sb1, nearzero=nearzero)
        flux_q1f = torch.minimum(flux_q1f_raw, S1_ref - flux_i)
        flux_qw_raw = recharge_3(tw, S1_ref)
        flux_qw = torch.minimum(flux_qw_raw, S1_ref - flux_i - flux_q1f)
        S1_new = S1_ref - flux_i - flux_q1f - flux_qw

    S2 = S2 + flux_qw
    flux_q2f = saturation_1(flux_qw, S2, Sb2, nearzero=nearzero)
    flux_q2f = torch.minimum(flux_q2f, S2)
    S2 = S2 - flux_q2f
    flux_q2u = baseflow_1(tu, S2)
    S2 = S2 - flux_q2u
    se_abs = Se * Sb2
    flux_et2 = evap_7(S2, se_abs, PET_epc, delta_t, nearzero)
    flux_et2 = torch.minimum(flux_et2, S2)
    S2_new = S2 - flux_et2

    Sc1 = Sc1 + flux_q1f + flux_q2f
    flux_qf = baseflow_1(tc, Sc1)
    Sc1_new = Sc1 - flux_qf
    Sc2 = Sc2 + flux_q2u
    flux_qs = baseflow_1(tc, Sc2)
    Sc2_new = Sc2 - flux_qs

    Q_total = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2 + flux_i

    fluxes = {
        "et1": flux_et1, "i_raw": flux_i_raw, "i": flux_i,
        "q1f_raw": flux_q1f_raw, "q1f": flux_q1f,
        "qw_raw": flux_qw_raw, "qw": flux_qw,
        "q2f": flux_q2f, "q2u": flux_q2u, "et2": flux_et2,
        "qf": flux_qf, "qs": flux_qs,
        "s1_pre_input": s1_pre_input, "s1_post_input": s1_post_input,
        "s1_before_i": s1_before_i, "s1_after_i": s1_after_i,
        "S1_new": S1_new, "S2_new": S2_new, "pet_epc": PET_epc,
    }
    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new, fluxes


def mopex5_step_diag(
    P: torch.Tensor, T: torch.Tensor, PET: torch.Tensor,
    tcrit: torch.Tensor, ddf: torch.Tensor, Sb1: torch.Tensor, tw: torch.Tensor,
    alpha: torch.Tensor, is_time: torch.Tensor, tmin: torch.Tensor, trange: torch.Tensor,
    tu: torch.Tensor, Se: torch.Tensor, Sb2: torch.Tensor, tc: torch.Tensor,
    S1: torch.Tensor, S2: torch.Tensor, Sc1: torch.Tensor, Sc2: torch.Tensor, Sn: torch.Tensor,
    delta_t: float = 1.0, nearzero: float = 1e-6, *, doy: torch.Tensor = None,
) -> Tuple[torch.Tensor, ...]:
    return _mopex5_core(P, T, PET, tcrit, ddf, Sb1, tw, alpha, is_time, tmin, trange,
                        tu, Se, Sb2, tc, S1, S2, Sc1, Sc2, Sn, delta_t, nearzero,
                        doy=doy, soil_mode="sequential")


def mopex5_step_samestate(
    P: torch.Tensor, T: torch.Tensor, PET: torch.Tensor,
    tcrit: torch.Tensor, ddf: torch.Tensor, Sb1: torch.Tensor, tw: torch.Tensor,
    alpha: torch.Tensor, is_time: torch.Tensor, tmin: torch.Tensor, trange: torch.Tensor,
    tu: torch.Tensor, Se: torch.Tensor, Sb2: torch.Tensor, tc: torch.Tensor,
    S1: torch.Tensor, S2: torch.Tensor, Sc1: torch.Tensor, Sc2: torch.Tensor, Sn: torch.Tensor,
    delta_t: float = 1.0, nearzero: float = 1e-6, *, doy: torch.Tensor = None,
) -> Tuple[torch.Tensor, ...]:
    return _mopex5_core(P, T, PET, tcrit, ddf, Sb1, tw, alpha, is_time, tmin, trange,
                        tu, Se, Sb2, tc, S1, S2, Sc1, Sc2, Sn, delta_t, nearzero,
                        doy=doy, soil_mode="samestate")
