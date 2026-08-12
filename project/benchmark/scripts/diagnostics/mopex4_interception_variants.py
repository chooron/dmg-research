"""Benchmark-only MOPEX4 interception parameterization variants.

F0 is the existing production diagnostic step imported from
mopex45_discr_steps. F1/F2 below copy the remaining MOPEX4 sequential step
and change only the interception fraction:

F1: alpha * softplus(beta*cos(theta))/beta, normalized by its value at cos=1
    (half-wave amplitude/phase decoupled; alpha=0 is exact zero)
F2: alpha * 0.5*(1+cos(theta)) (shifted-cosine amplitude/phase decoupled)

All other fluxes, state caps, ordering, and continuation context are retained.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from dmotpy.models.flux.mopex import (
    mopex_baseflow_1 as baseflow_1, mopex_evap_7 as evap_7,
    mopex_melt_1 as melt_1, mopex_rainfall_1 as rainfall_1,
    mopex_recharge_3 as recharge_3, mopex_saturation_1 as saturation_1,
    mopex_snowfall_1 as snowfall_1, mopex_phenology_1 as phenology_1,
    _training_values,
)


def _interception_fraction(doy, alpha, is_time, mode, beta=50.0, tmax=365.25):
    theta = 2.0 * torch.pi * (doy - is_time) / tmax
    c = torch.cos(theta)
    if mode == "F1":
        smooth = F.softplus(beta * c) / beta
        peak = F.softplus(torch.as_tensor(beta, dtype=c.dtype, device=c.device)) / beta
        return alpha * smooth / peak
    if mode == "F2":
        return alpha * 0.5 * (1.0 + c)
    raise ValueError(mode)


def mopex4_step_variant(
    P, T, PET, tcrit, ddf, Sb1, tw, alpha, is_time, tu, Se, Sb2, tc,
    S1, S2, Sc1, Sc2, Sn, delta_t=1.0, nearzero=1e-6, *, doy=None, mode="F1",
):
    Sn = F.relu(Sn); S1 = F.relu(S1); S2 = F.relu(S2); Sc1 = F.relu(Sc1); Sc2 = F.relu(Sc2)
    lambda_i, _lp, beta = _training_values()
    flux_ps = snowfall_1(P, T, tcrit); flux_pr = rainfall_1(P, T, tcrit)
    flux_qn = melt_1(ddf, tcrit, T, Sn, delta_t)
    Sn = Sn + flux_ps; Sn_new = Sn - flux_qn
    S1 = S1 + flux_pr + flux_qn
    flux_et1 = torch.minimum(evap_7(S1, Sb1, PET, delta_t, nearzero), S1); S1 = S1 - flux_et1
    frac = _interception_fraction(doy, alpha, is_time, mode, beta=beta)
    flux_i_raw = torch.minimum(frac * flux_pr, flux_pr) * lambda_i
    flux_i = torch.minimum(flux_i_raw, S1); S1 = S1 - flux_i
    flux_q1f_raw = saturation_1(flux_pr + flux_qn, S1, Sb1, nearzero=nearzero)
    flux_q1f = torch.minimum(flux_q1f_raw, S1); S1 = S1 - flux_q1f
    flux_qw_raw = recharge_3(tw, S1); flux_qw = torch.minimum(flux_qw_raw, S1)
    S1_new = S1 - flux_qw
    S2 = S2 + flux_qw
    flux_q2f = torch.minimum(saturation_1(flux_qw, S2, Sb2, nearzero=nearzero), S2); S2 = S2 - flux_q2f
    flux_q2u = baseflow_1(tu, S2); S2 = S2 - flux_q2u
    flux_et2 = torch.minimum(evap_7(S2, Se * Sb2, PET, delta_t, nearzero), S2); S2_new = S2 - flux_et2
    Sc1 = Sc1 + flux_q1f + flux_q2f; flux_qf = baseflow_1(tc, Sc1); Sc1_new = Sc1 - flux_qf
    Sc2 = Sc2 + flux_q2u; flux_qs = baseflow_1(tc, Sc2); Sc2_new = Sc2 - flux_qs
    Q = flux_qf + flux_qs; ET = flux_et1 + flux_et2 + flux_i
    fluxes = {"i": flux_i, "i_raw": flux_i_raw, "et1": flux_et1, "et2": flux_et2,
              "q1f": flux_q1f, "qw": flux_qw, "S1_new": S1_new, "S2_new": S2_new}
    return Q, ET, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new, fluxes


def make_variant(mode):
    return lambda *args, **kwargs: mopex4_step_variant(*args, mode=mode, **kwargs)
