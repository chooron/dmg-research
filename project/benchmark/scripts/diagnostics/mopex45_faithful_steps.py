"""Benchmark-only MARRMoT-faithful flux variants for the MOPEX4/5 audit.

These replace the mismatching flux formulas in dmotpy/models/flux/mopex.py
with the exact MARRMoT MATLAB formulas (MARRMoT/Models/Flux files), keeping
the same sequential within-step ordering as the production Python model so
that the *formula* effect is isolated from the *discretization* effect.

Changed helpers (only these):
  saturation_1 : P*(1 - 1/(1+exp((S - Smax + r*e*Smax)/(r*Smax))))
  evap_7       : min(S/Smax*Ep, S)                        (no Smax clamp)
  snowfall_1   : P/(1+exp((T-tcrit)/0.01))                (fixed width)
  rainfall_1   : P*(1 - 1/(1+exp((T-tcrit)/0.01)))
  melt_1       : clamp(min(ddf*(T-tcrit), Sn), min=0)     (linear drive)
  interception_4: max(0, alpha+(1-alpha)*cos(2pi(doy-is_time)/365.25))*Pr
                  (hard kink; lambda_i continuation multiplier applied as in
                   production so the endpoint lambda_i=1 matches)

Unchanged: recharge_3, baseflow_1, phenology_1 (already identical), the
sequential state-update ordering, state layout, parameter order/bounds.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def faithful_saturation_1(P, S, Smax, r=0.01, e=5.0, nearzero=1e-6):
    Smax = torch.clamp(Smax, min=0.0)
    denom = r * Smax
    safe = torch.where(denom == 0, torch.full_like(denom, nearzero), denom)
    out = 1.0 / (1.0 + torch.exp((S - Smax + r * e * Smax) / safe))
    return P * (1.0 - out)


def faithful_evap_7(S, Smax, Ep, dt=1.0, nearzero=1e-6):
    return torch.minimum(S / (Smax + nearzero) * Ep, S / dt)


def faithful_snowfall_1(P, T, tcrit):
    return P / (1.0 + torch.exp((T - tcrit) / 0.01))


def faithful_rainfall_1(P, T, tcrit):
    return P * (1.0 - 1.0 / (1.0 + torch.exp((T - tcrit) / 0.01)))


def faithful_melt_1(ddf, tcrit, T, Sn, dt=1.0):
    return torch.clamp(torch.minimum(ddf * (T - tcrit), Sn / dt), min=0.0)


def faithful_interception_4(flux_pr, doy, alpha, is_time, tmax=365.25, nearzero=1e-6,
                            lambda_i=1.0):
    fraction = alpha + (1.0 - alpha) * torch.cos(2.0 * torch.pi * (doy - is_time) / tmax)
    return torch.clamp(fraction, min=0.0) * flux_pr * lambda_i


# ---------------------------------------------------------------------------
# MOPEX4 faithful-flux step (sequential ordering identical to production)
# ---------------------------------------------------------------------------

def mopex4_step_faithful(
    P: torch.Tensor, T: torch.Tensor, PET: torch.Tensor,
    tcrit: torch.Tensor, ddf: torch.Tensor, Sb1: torch.Tensor, tw: torch.Tensor,
    alpha: torch.Tensor, is_time: torch.Tensor, tu: torch.Tensor, Se: torch.Tensor,
    Sb2: torch.Tensor, tc: torch.Tensor,
    S1: torch.Tensor, S2: torch.Tensor, Sc1: torch.Tensor, Sc2: torch.Tensor, Sn: torch.Tensor,
    delta_t: float = 1.0, nearzero: float = 1e-6, *, doy: torch.Tensor = None,
) -> torch.Tensor:
    from dmotpy.models.flux.mopex import mopex_training_context, _training_values
    lambda_i, _lp, _beta = _training_values()
    Sn = F.relu(Sn); S1 = F.relu(S1); S2 = F.relu(S2); Sc1 = F.relu(Sc1); Sc2 = F.relu(Sc2)

    flux_ps = faithful_snowfall_1(P, T, tcrit)
    flux_pr = faithful_rainfall_1(P, T, tcrit)
    flux_qn = faithful_melt_1(ddf, tcrit, T, Sn, delta_t)
    Sn = Sn + flux_ps
    Sn_new = Sn - flux_qn

    S1 = S1 + flux_pr + flux_qn
    flux_et1 = faithful_evap_7(S1, Sb1, PET, delta_t)
    flux_et1 = torch.minimum(flux_et1, S1)
    S1 = S1 - flux_et1
    flux_i = faithful_interception_4(flux_pr, doy, alpha, is_time, nearzero=nearzero, lambda_i=lambda_i)
    flux_i = torch.minimum(flux_i, S1)
    S1 = S1 - flux_i
    flux_q1f = faithful_saturation_1(flux_pr + flux_qn, S1, Sb1, nearzero=nearzero)
    flux_q1f = torch.minimum(flux_q1f, S1)
    S1 = S1 - flux_q1f
    flux_qw = torch.minimum(tw * S1, S1)
    S1_new = S1 - flux_qw

    S2 = S2 + flux_qw
    flux_q2f = faithful_saturation_1(flux_qw, S2, Sb2, nearzero=nearzero)
    flux_q2f = torch.minimum(flux_q2f, S2)
    S2 = S2 - flux_q2f
    flux_q2u = torch.minimum(tu * S2, S2)
    S2 = S2 - flux_q2u
    se_abs = Se * Sb2
    flux_et2 = faithful_evap_7(S2, se_abs, PET, delta_t)
    flux_et2 = torch.minimum(flux_et2, S2)
    S2_new = S2 - flux_et2

    Sc1 = Sc1 + flux_q1f + flux_q2f
    flux_qf = torch.minimum(tc * Sc1, Sc1)
    Sc1_new = Sc1 - flux_qf
    Sc2 = Sc2 + flux_q2u
    flux_qs = torch.minimum(tc * Sc2, Sc2)
    Sc2_new = Sc2 - flux_qs

    Q_total = flux_qf + flux_qs
    ET_total = flux_et1 + flux_et2 + flux_i
    return Q_total, ET_total, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new
