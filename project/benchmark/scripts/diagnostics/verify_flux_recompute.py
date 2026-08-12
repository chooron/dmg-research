#!/usr/bin/env python3
"""Verify make_fluxes_step recomputation is consistent with mopex4_step."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
from src.model_registry import build_model
from dmotpy.models.core.mopex4 import mopex4_step
from dmotpy.models.flux.mopex import (
    mopex_baseflow_1 as baseflow_1, mopex_evap_7 as evap_7,
    mopex_interception_4_liu2 as interception_4, mopex_melt_1 as melt_1,
    mopex_rainfall_1 as rainfall_1, mopex_recharge_3 as recharge_3,
    mopex_saturation_1 as saturation_1, mopex_snowfall_1 as snowfall_1,
)

hydro = build_model("mopex4", torch.device("cuda"), warm_up=365, backend="eager",
                    parameter_mapping="auto", warmup_grad_mode="detach")
# physical midpoint params
pv = []
for name in hydro.phy_param_names:
    lo, hi = hydro.parameter_bounds[name]
    pv.append(torch.full((4,), (lo + hi) / 2, device="cuda"))
pv = [p.unsqueeze(-1) for p in pv]

torch.manual_seed(0)
P = torch.clamp(torch.randn(4, device="cuda") * 4 + 4, min=0.0)
T = 5.0 + 15.0 * torch.sin(torch.arange(4, device="cuda"))
PET = torch.clamp(2.0 + 4.0 * torch.sin(torch.arange(4, device="cuda")), min=0.5)
S1 = torch.rand(4, device="cuda") * 50
S2 = torch.rand(4, device="cuda") * 100
Sc1 = torch.rand(4, device="cuda") * 10
Sc2 = torch.rand(4, device="cuda") * 10
Sn = torch.rand(4, device="cuda") * 20

out = mopex4_step(P, T, PET, *pv, S1, S2, Sc1, Sc2, Sn, delta_t=1.0, nearzero=hydro.nearzero)
Q_step, ET_step, S1n, S2n, Sc1n, Sc2n, Snn = out

tcrit, ddf, Sb1, tw, S_eff, c, tu, Se, Sb2, tc = pv
nearzero = hydro.nearzero
flux_ps = snowfall_1(P, T, tcrit)
flux_pr = rainfall_1(P, T, tcrit)
flux_qn = melt_1(ddf, tcrit, T, Sn, 1.0)
i_pot = interception_4(flux_pr, S_eff, c, nearzero=nearzero)
flux_i = torch.minimum(i_pot, PET)
pet_after_i = PET - flux_i
soil_input = (flux_pr - flux_i) + flux_qn
S1_w = S1 + soil_input
flux_et1 = torch.minimum(evap_7(S1_w, Sb1, pet_after_i, 1.0, nearzero), S1_w)
pet_after_et1 = pet_after_i - flux_et1
S1_after_et1 = S1_w - flux_et1
flux_q1f = torch.minimum(saturation_1(soil_input, S1_after_et1, Sb1, nearzero=nearzero), S1_after_et1)
S1_after_q1f = S1_after_et1 - flux_q1f
flux_qw = recharge_3(tw, S1_after_q1f)
S2_w = S2 + flux_qw
flux_q2f = torch.minimum(saturation_1(flux_qw, S2_w, Sb2, nearzero=nearzero), S2_w)
S2_after_q2f = S2_w - flux_q2f
flux_q2u = baseflow_1(tu, S2_after_q2f)
S2_after_q2u = S2_after_q2f - flux_q2u
se_abs = Se * Sb2
flux_et2 = torch.minimum(evap_7(S2_after_q2u, se_abs, pet_after_et1, 1.0, nearzero), S2_after_q2u)
Sc1_w = Sc1 + flux_q1f + flux_q2f
flux_qf = baseflow_1(tc, Sc1_w)
Sc2_w = Sc2 + flux_q2u
flux_qs = baseflow_1(tc, Sc2_w)
Q_manual = flux_qf + flux_qs

print("Q_step    :", Q_step.squeeze(-1).tolist())
print("Q_manual  :", Q_manual.squeeze(-1).tolist())
print("max|Q diff|:", (Q_step.squeeze(-1) - Q_manual.squeeze(-1)).abs().max().item())
print("ET_step   :", ET_step.squeeze(-1).tolist())
print("ET_manual :", (flux_i + flux_et1 + flux_et2).squeeze(-1).tolist())
# also check saturation_1/evap_7/recharge_3/baseflow_1 signatures used match
import inspect
for f in (evap_7, saturation_1, recharge_3, baseflow_1, melt_1, snowfall_1, rainfall_1):
    print(f.__name__, inspect.signature(f))
