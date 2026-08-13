#!/usr/bin/env python3
"""Pre-training validation: MOPEX5 original formulas + corrected process order.

MOPEX5 = restored-MOPEX4 core hydrology (original alpha/is_time interception,
corrected process order, shared daily PET budget) + the original GSI
phenology PET extension (tmin/trange, ``PET_epc = GSI(T) * PET``).

Checks:
  A. forward / source-water semantics (rain-only, snow-only, rain+snowmelt,
     zero P, wet event, near-zero; two GSI-distinct temperature conditions)
  B. PET-budget consistency (I + ET1 + ET2 <= PET_epc <= PET daily;
     GSI in [0,1]; exceedance fraction / max exceedance over a year)
  C. water balance (single step + 365-day sequence; max daily abs residual,
     cumulative residual)
  D. gradient health through the canonical differentiable dPL path +
     finite-difference spot-checks for the clamp/threshold parameters
     (tmin, trange) and the interception parameters (alpha, is_time)
  E. state mapping (create_initial_state -> wrapper reorder), eager vs
     compile, fullgraph, canonical registry/wrapper/mapping integration
  F. regression: restored mopex4 unchanged, 36-model registry, prod/mirror
     mopex5 core identical

No formal training is started.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
# The authoritative production `dmotpy` lives at the worktree root.  The shared
# venv's site-packages still contains a stale pre-correction copy of MOPEX4/5
# (and lacks mopex_doy_model), so prepend the repo root to resolve determinis-
# tically to this worktree's package instead of silently validating the old code.
REPO = ROOT.parent.parent
sys.path[:0] = [str(REPO), str(ROOT), str(ROOT / "src")]

from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from dmotpy.models.core.mopex4 import interception_4 as mopex4_interception
from dmotpy.models.core.mopex5 import MOPEX5_PARAMS_BOUNDS, mopex5_step, phenology_effective_pet
from dmotpy.models.flux.mopex import (
    mopex_baseflow_1 as baseflow_1,
    mopex_evap_7 as evap_7,
    mopex_melt_1 as melt_1,
    mopex_rainfall_1 as rainfall_1,
    mopex_recharge_3 as recharge_3,
    mopex_saturation_1 as saturation_1,
    mopex_snowfall_1 as snowfall_1,
)
from dmotpy.models.registry import NPARAM_INFO, PARAM_INFO
from src.data_selection import load_ids
from src.model_registry import NPARAM_INFO_36, build_model

DT = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = ROOT / "results/mopex5_process_semantics_20260812"
OUT.mkdir(parents=True, exist_ok=True)

PARAMS = [0.0, 4.0, 200.0, 0.1, 0.5, 180.0, -2.0, 12.0, 0.1, 0.5, 300.0, 0.2]  # tcrit..tc


def log(msg: str) -> None:
    print(msg, flush=True)


def step(P, T, PET, pv, doy, states):
    return mopex5_step(P, T, PET, *pv, states[1], states[2], states[3], states[4], states[0],
                       delta_t=1.0, nearzero=1e-6, doy=doy)


def recompute_fluxes(P, T, PET, pv, doy, states):
    """Local flux-by-flux mirror of the corrected mopex5_step (for WB + checks)."""
    p = [torch.tensor(v, dtype=DT) for v in pv]
    Sn_e, S1_e, S2_e, Sc1_e, Sc2_e = [torch.relu(torch.tensor(v, dtype=DT)) for v in states]
    Sn, S1, S2, Sc1, Sc2 = Sn_e, S1_e, S2_e, Sc1_e, Sc2_e
    P_t, T_t, PET_t = torch.tensor(P, dtype=DT), torch.tensor(T, dtype=DT), torch.tensor(PET, dtype=DT)
    doy_t = torch.tensor(doy, dtype=DT)
    pet_epc = phenology_effective_pet(T_t, p[6], p[7], PET_t, 1e-6)
    ps = snowfall_1(P_t, T_t, p[0])
    pr = rainfall_1(P_t, T_t, p[0])
    qn = melt_1(p[1], p[0], T_t, Sn, 1.0)
    Sn_new = Sn + ps - qn
    i_pot = mopex4_interception(pr, doy_t, p[4], p[5], nearzero=1e-6)
    flux_i = torch.minimum(i_pot, pet_epc)
    pr_net = pr - flux_i
    pet_after_i = pet_epc - flux_i
    soil_input = pr_net + qn
    S1 = S1 + soil_input
    et1 = torch.minimum(evap_7(S1, p[2], pet_after_i, 1.0, 1e-6), S1)
    S1 = S1 - et1
    pet_remaining = pet_after_i - et1
    q1f = torch.minimum(saturation_1(soil_input, S1, p[2], nearzero=1e-6), S1)
    S1 = S1 - q1f
    qw = recharge_3(p[3], S1)
    S1_new = S1 - qw
    S2 = S2 + qw
    q2f = torch.minimum(saturation_1(qw, S2, p[10], nearzero=1e-6), S2)
    S2 = S2 - q2f
    q2u = baseflow_1(p[8], S2)
    S2 = S2 - q2u
    et2 = torch.minimum(evap_7(S2, p[9] * p[10], pet_remaining, 1.0, 1e-6), S2)
    S2_new = S2 - et2
    Sc1 = Sc1 + q1f + q2f
    qf = baseflow_1(p[11], Sc1)
    Sc1_new = Sc1 - qf
    Sc2 = Sc2 + q2u
    qs = baseflow_1(p[11], Sc2)
    Sc2_new = Sc2 - qs
    Q = qf + qs
    # dS must be entry-vs-final (the working S1/S2/Sc1/Sc2 locals are mutated
    # intermediates, so differencing them would silently drop the increments).
    dS = (Sn_new - Sn_e) + (S1_new - S1_e) + (S2_new - S2_e) + (Sc1_new - Sc1_e) + (Sc2_new - Sc2_e)
    residual = (ps + pr) - flux_i - et1 - et2 - Q - dS
    return {"Q": Q, "I": flux_i, "ET1": et1, "ET2": et2, "pr": pr, "qn": qn,
            "soil_input": soil_input, "pet_epc": pet_epc, "residual": residual,
            "S1n": S1_new, "S2n": S2_new, "Sc1n": Sc1_new, "Sc2n": Sc2_new, "Sn": Sn_new,
            "pet_after_et1": pet_remaining}


# ---------------------------------------------------------------- A
def check_forward_source_water() -> dict:
    log("\n[A] forward / source-water semantics")
    cases = [
        ("rain_only", 20.0, 15.0, 3.0, [0.0] * 5, 180.0),
        ("snow_only", 20.0, -10.0, 2.0, [0.0] * 5, 180.0),
        ("rain_plus_snowmelt", 15.0, 8.0, 2.0, [30.0, 0.0, 0.0, 0.0, 0.0], 181.0),
        ("zero_precip", 0.0, 12.0, 3.0, [0.0] * 5, 180.0),
        ("wet_event", 80.0, 18.0, 4.0, [10.0, 50.0, 100.0, 5.0, 5.0], 200.0),
        ("near_zero", 1e-6, 12.0, 2.0, [0.0] * 5, 180.0),
        ("cold_gsi", 10.0, 0.0, 3.0, [0.0] * 5, 180.0),     # T=tmin+2 -> partial GSI
        ("warm_gsi", 10.0, 15.0, 3.0, [0.0] * 5, 180.0),    # T>tmin+trange -> GSI=1
    ]
    results = {}
    for name, P, T, PET, st, doy in cases:
        pv = [torch.tensor(v, dtype=DT) for v in PARAMS]
        stv = [torch.tensor(v, dtype=DT) for v in st]
        out = step(P, T, PET, pv, doy, stv)
        Q, ET, S1n, S2n, Sc1n, Sc2n, Snn = out
        comp = recompute_fluxes(P, T, PET, PARAMS, doy, st)
        finite = all(torch.isfinite(x).all().item() for x in out) and torch.isfinite(comp["I"]).item()
        pr_c, I, qn = comp["pr"], comp["I"], comp["qn"]
        checks = {
            "finite": bool(finite),
            "I>=0": bool(I >= -1e-12),
            "I<=Pr_supply": bool(I <= pr_c + 1e-9),
            "I<=PET_epc": bool(I <= comp["pet_epc"] + 1e-9),
            "soil_input=Pr_net+qn": bool(torch.isclose(comp["soil_input"], (pr_c - I) + qn, atol=1e-9)),
            "Q_step==Q_recompute": bool(torch.isclose(Q, comp["Q"], atol=1e-9)),
            "states_nonneg": all(bool(x >= -1e-12) for x in out[2:]),
        }
        results[name] = checks
        log(f"  [{name:18s}] finite={checks['finite']} I>={checks['I>=0']} I<=Pr={checks['I<=Pr_supply']} "
            f"I<=PETepc={checks['I<=PET_epc']} soil=Prnet+qn={checks['soil_input=Pr_net+qn']} "
            f"Qmatch={checks['Q_step==Q_recompute']} nonneg={checks['states_nonneg']} "
            f"I={float(I):.4f} Pr={float(pr_c):.4f} PETepc={float(comp['pet_epc']):.4f}")
    # phenology path: warm day must have larger ET than cold day at equal P/PET.
    # phenology_effective_pet returns PET_epc = GSI*PET (mm/d), not the GSI
    # itself, so derive GSI = PET_epc/PET for the [0,1] range check.
    pet_epc_cold = float(phenology_effective_pet(torch.tensor(0.0, dtype=DT), torch.tensor(-2.0, dtype=DT),
                                                 torch.tensor(12.0, dtype=DT), torch.tensor(3.0, dtype=DT)))
    pet_epc_warm = float(phenology_effective_pet(torch.tensor(15.0, dtype=DT), torch.tensor(-2.0, dtype=DT),
                                                 torch.tensor(12.0, dtype=DT), torch.tensor(3.0, dtype=DT)))
    gsi_cold = pet_epc_cold / 3.0
    gsi_warm = pet_epc_warm / 3.0
    results["_gsi"] = {"pet_epc_cold": pet_epc_cold, "pet_epc_warm": pet_epc_warm,
                       "cold<warm": pet_epc_cold < pet_epc_warm,
                       "gsi_cold": gsi_cold, "gsi_warm": gsi_warm,
                       "gsi_in_01": 0.0 <= gsi_cold <= 1.0 and 0.0 <= gsi_warm <= 1.0}
    et_cold = float(recompute_fluxes(10.0, 0.0, 3.0, PARAMS, 180.0, [0.0] * 5)["ET1"] +
                    recompute_fluxes(10.0, 0.0, 3.0, PARAMS, 180.0, [0.0] * 5)["ET2"] +
                    recompute_fluxes(10.0, 0.0, 3.0, PARAMS, 180.0, [0.0] * 5)["I"])
    et_warm = float(recompute_fluxes(10.0, 15.0, 3.0, PARAMS, 180.0, [0.0] * 5)["ET1"] +
                    recompute_fluxes(10.0, 15.0, 3.0, PARAMS, 180.0, [0.0] * 5)["ET2"] +
                    recompute_fluxes(10.0, 15.0, 3.0, PARAMS, 180.0, [0.0] * 5)["I"])
    results["_phenology_et_path"] = {"cold_ET": et_cold, "warm_ET": et_warm,
                                     "warm>cold": et_warm > et_cold}
    log(f"  GSI: cold(T=0)={gsi_cold:.4f} warm(T=15)={gsi_warm:.4f} cold<warm={results['_gsi']['cold<warm']} (PET_epc: {pet_epc_cold:.4f}/{pet_epc_warm:.4f} mm/d)")
    log(f"  ET path: cold_ET={et_cold:.4f} warm_ET={et_warm:.4f} warm>cold={et_warm > et_cold} (phenology enters demand)")
    return results


# ---------------------------------------------------------------- B
def check_pet_budget() -> dict:
    log("\n[B] PET-budget consistency (I + ET1 + ET2 <= PET_epc <= PET)")
    torch.manual_seed(3)
    P = torch.clamp(torch.randn(365, dtype=DT) * 6 + 6, min=0.0)
    T = (5.0 + 18.0 * torch.sin(torch.arange(365, dtype=DT) / 365.0 * 2 * math.pi))
    PET = torch.clamp(2.0 + 3.0 * torch.randn(365, dtype=DT), min=0.1)
    pv = [torch.tensor(v, dtype=DT) for v in PARAMS]
    st = [torch.tensor(1e-6, dtype=DT) for _ in range(5)]
    max_exceed_epc = 0.0; exceed_days_epc = 0
    max_exceed_pet = 0.0; exceed_days_pet = 0
    gsi_min, gsi_max = 1.0, 0.0
    for t in range(365):
        out = step(float(P[t]), float(T[t]), float(PET[t]), pv, 180.0 + t, st)
        ET = out[1]
        pet_epc = float(phenology_effective_pet(T[t], pv[6], pv[7], PET[t], 1e-6))
        gsi = float(torch.clamp((T[t] - pv[6]) / torch.clamp(pv[7], min=1e-6), 0.0, 1.0))
        gsi_min, gsi_max = min(gsi_min, gsi), max(gsi_max, gsi)
        exc_epc = float(ET) - pet_epc
        exc_pet = float(ET) - float(PET[t])
        if exc_epc > 1e-6: exceed_days_epc += 1
        if exc_pet > 1e-6: exceed_days_pet += 1
        max_exceed_epc = max(max_exceed_epc, exc_epc)
        max_exceed_pet = max(max_exceed_pet, exc_pet)
        # mopex5_step returns (Q, ET, S1, S2, Sc1, Sc2, Sn) in STEP order; the
        # harness tracks states in create order (Sn, S1, S2, Sc1, Sc2), so
        # convert before feeding the next day (no silent rotation).
        st = [out[6], out[2], out[3], out[4], out[5]]
    res = {"exceedance_days_vs_PET_epc": exceed_days_epc, "max_exceedance_vs_PET_epc": max_exceed_epc,
           "exceedance_days_vs_raw_PET": exceed_days_pet, "max_exceedance_vs_raw_PET": max_exceed_pet,
           "gsi_range": [gsi_min, gsi_max], "closure_pass_epc": exceed_days_epc == 0,
           "closure_pass_pet": exceed_days_pet == 0}
    log(f"  365d: exceed_days(vs PET_epc)={exceed_days_epc} max_exceed={max_exceed_epc:.3e}")
    log(f"        exceed_days(vs raw PET)={exceed_days_pet} max_exceed={max_exceed_pet:.3e}")
    log(f"        GSI range=[{gsi_min:.3f},{gsi_max:.3f}] (guarantees PET_epc <= PET)")
    return res


# ---------------------------------------------------------------- C
def check_water_balance() -> dict:
    log("\n[C] water balance")
    res = {}
    # single step
    comp = recompute_fluxes(15.0, 10.0, 3.0, PARAMS, 200.0, [5.0, 20.0, 10.0, 2.0, 3.0])
    res["single_step_max_daily_abs_residual"] = float(abs(comp["residual"]))
    log(f"  single step: max_daily_abs_residual={abs(comp['residual']):.3e}")
    # 365-day sequence (with snow buildup + melt + wet season)
    torch.manual_seed(5)
    P = torch.clamp(torch.randn(365, dtype=DT) * 6 + 6, min=0.0)
    T = (2.0 + 14.0 * torch.sin(torch.arange(365, dtype=DT) / 365.0 * 2 * math.pi))
    PET = torch.clamp(2.0 + 3.0 * torch.randn(365, dtype=DT), min=0.1)
    pv = [torch.tensor(v, dtype=DT) for v in PARAMS]
    st = [torch.tensor(1e-6, dtype=DT) for _ in range(5)]
    p_sum, et_sum, q_sum = 0.0, 0.0, 0.0
    max_daily = 0.0
    for t in range(365):
        out = step(float(P[t]), float(T[t]), float(PET[t]), pv, 180.0 + t, st)
        comp = recompute_fluxes(float(P[t]), float(T[t]), float(PET[t]), PARAMS, 180.0 + t, st)
        max_daily = max(max_daily, float(abs(comp["residual"])))
        p_sum += float(P[t]); et_sum += float(out[1]); q_sum += float(out[0])
        # step-order (S1,S2,Sc1,Sc2,Sn) output -> create-order (Sn,S1,S2,Sc1,Sc2)
        # state for the next day (matches the wrapper's state contract).
        st = [out[6], out[2], out[3], out[4], out[5]]
    final = sum(float(s) for s in st)
    cumulative = p_sum - et_sum - q_sum - (final - 5e-6)
    res["seq365_max_daily_abs_residual"] = max_daily
    res["seq365_cumulative_residual"] = cumulative
    log(f"  365d seq: max_daily_abs_residual={max_daily:.3e} cumulative_residual={cumulative:.3e}")
    return res


# ---------------------------------------------------------------- D
_DATA = None

def _synthetic_data() -> tuple:
    """Lazy 20-basin, 365-day synthetic forcing used by D and E."""
    global _DATA
    if _DATA is None:
        ids = [int(x) for x in load_ids("data/531sub_id.txt")]
        torch.manual_seed(11)
        b = torch.randperm(len(ids), device=DEVICE)[:20]
        attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device=DEVICE, method="zscore").to(DT)
        P = torch.clamp(torch.randn(365, 20, device=DEVICE, dtype=DT) * 4 + 4, min=0.0)
        T = (5.0 + 15.0 * torch.sin(torch.arange(365, dtype=DT, device=DEVICE) / 365.0 * 2 * math.pi)).view(365, 1).expand(365, 20)
        PET = torch.clamp(2.0 + 3.0 * torch.randn(365, 20, device=DEVICE, dtype=DT), min=0.1)
        doy = ((torch.arange(365, device=DEVICE) % 365) + 1).to(DT).view(365, 1, 1).expand(365, 20, 1)
        _DATA = (torch.cat([P.unsqueeze(-1), T.unsqueeze(-1), PET.unsqueeze(-1), doy], dim=-1), attrs)
    return _DATA


def check_gradient_health() -> dict:
    log("\n[D] gradient health (canonical dPL path + FD spot-checks)")
    res = {}
    x, attrs = _synthetic_data()
    hydro = build_model("mopex5", DEVICE, warm_up=60, backend="eager",
                        parameter_mapping="auto", warmup_grad_mode="detach")
    n = len(hydro.phy_param_names)
    theta = torch.full((20, n, 1), 0.5, dtype=DT, device=DEVICE)
    theta[:, 4] = 0.55; theta[:, 5] = 0.45; theta[:, 6] = 0.6; theta[:, 7] = 0.5
    theta.requires_grad_(True)
    q = hydro({"x_phy": x}, (None, theta))["streamflow"].squeeze(-1).squeeze(-1)
    loss = q.square().mean()
    loss.backward()
    assert theta.grad is not None and torch.isfinite(theta.grad).all()
    g = theta.grad.detach()
    names = list(hydro.phy_param_names)
    per_param = {}
    for j, name in enumerate(names):
        vals = g[:, j]
        per_param[name] = {"mean_abs_grad": float(vals.abs().mean()),
                           "nonzero_fraction": float((vals != 0).float().mean())}
    res["params"] = per_param
    for name in ("alpha", "is_time", "tmin", "trange"):
        p = per_param[name]
        log(f"  {name:8s} mean|grad|={p['mean_abs_grad']:.3e} nonzero={p['nonzero_fraction']*100:.1f}%")
    all_params_ok = all(p["mean_abs_grad"] > 0 and p["nonzero_fraction"] > 0.5 for p in per_param.values())
    res["all_params_nonzero_majority"] = all_params_ok
    log(f"  all 12 params majority-nonzero: {all_params_ok}")
    res["loss_finite"] = bool(torch.isfinite(loss))
    res["grad_finite"] = bool(torch.isfinite(g).all())

    # FD spot-checks: tmin, trange (interior, no clamp), alpha, is_time (uncapped)
    def loss_at(raw_slots, basin=None):
        th = torch.full((20, n, 1), 0.5, dtype=DT, device=DEVICE)
        th[:, 4] = 0.55; th[:, 5] = 0.45; th[:, 6] = 0.6; th[:, 7] = 0.5
        for slot, v in raw_slots:
            if basin is None:
                th[:, slot] = v
            else:
                th[basin, slot] = v
        qq = hydro({"x_phy": x}, (None, th))["streamflow"].squeeze(-1).squeeze(-1)
        return qq.square().mean()

    fd_report = {}
    # Single-basin perturbation: perturbing every basin's slot simultaneously and
    # comparing with the basin-mean autograd is off by a factor of n_basins (the
    # loss is a mean over basins while FD perturbs all of them).  Perturb one
    # basin and compare with that basin's own autograd (a true directional
    # derivative, dL/dtheta_{j,b}).
    FD_BASIN = 0
    for slot, name, delta in [(6, "tmin", 1e-3), (7, "trange", 1e-3), (4, "alpha", 1e-3), (5, "is_time", 1e-3)]:
        eps = delta
        with torch.no_grad():
            lp = loss_at([(slot, 0.5 + eps)], basin=FD_BASIN).item()
            lm = loss_at([(slot, 0.5 - eps)], basin=FD_BASIN).item()
        fd = (lp - lm) / (2 * eps)
        th = torch.full((20, n, 1), 0.5, dtype=DT, device=DEVICE)
        th[:, 4] = 0.55; th[:, 5] = 0.45; th[:, 6] = 0.6; th[:, 7] = 0.5
        th[FD_BASIN, slot] = 0.5
        th.requires_grad_(True)
        ll = hydro({"x_phy": x}, (None, th))["streamflow"].squeeze(-1).squeeze(-1).square().mean()
        ll.backward()
        ag = float(th.grad[FD_BASIN, slot])
        rel = abs(ag - fd) / (abs(fd) + 1e-12)
        fd_report[name] = {"autograd": ag, "fd": fd, "rel_err": rel,
                           "perturbation": f"single-basin[{FD_BASIN}]"}
        log(f"  FD {name:8s}: autograd={ag:+.3e} fd={fd:+.3e} rel_err={rel:.2e} (single-basin[{FD_BASIN}])")
    res["fd_spot_checks"] = fd_report
    return res


# ---------------------------------------------------------------- E
def check_state_mapping_compile() -> dict:
    log("\n[E] state mapping / compile / canonical integration")
    res = {}
    x, _ = _synthetic_data()
    n = NPARAM_INFO_36["mopex5"]
    from dmotpy.models.hydrology_model import HydrologyModel
    config = {"model_name": "mopex5", "warm_up": 0, "warm_up_states": True,
              "variables": ["prcp", "tmean", "pet"], "nearzero": 1e-6,
              "parameter_mapping": "linear", "backend": "eager"}
    model = HydrologyModel(config, device=torch.device("cpu"), backend="eager")
    assert model.model_name == "mopex5"
    forcing = torch.tensor([[[10.0, -5.0, 2.0, 20.0]], [[20.0, 8.0, 2.0, 21.0]], [[0.0, 8.0, 2.0, 22.0]]], dtype=DT)
    raw = torch.full((1, 12, 1), 0.5, dtype=DT)
    params_dict = model._descale_params(model.unpack_parameters((None, raw)))
    param_values = [params_dict[name] for name in model.phy_param_names]
    sn0, s1_0, s2_0, sc1_0, sc2_0 = 1.0, 2.0, 3.0, 4.0, 5.0
    init_states = tuple(torch.tensor(v, dtype=DT) for v in (sn0, s1_0, s2_0, sc1_0, sc2_0))
    wrapped = model._run_model({"x_phy": forcing}, init_states, params_dict, 1)
    wrapped_q = wrapped["streamflow"].squeeze()
    step_states = (init_states[1], init_states[2], init_states[3], init_states[4], init_states[0])
    ref, curr = [], step_states
    for t in range(forcing.shape[0]):
        outputs = model.raw_step_fn(forcing[t, :, 0:1], forcing[t, :, 1:2], forcing[t, :, 2:3],
                                    *param_values, *curr, delta_t=1.0, nearzero=model.nearzero,
                                    doy=forcing[t, :, 3:4])
        ref.append(outputs[0]); curr = tuple(outputs[2:])
    ref_q = torch.stack(ref).squeeze()
    torch.testing.assert_close(wrapped_q, ref_q, rtol=1e-9, atol=1e-9)
    wrong, curr = [], init_states
    for t in range(forcing.shape[0]):
        outputs = model.raw_step_fn(forcing[t, :, 0:1], forcing[t, :, 1:2], forcing[t, :, 2:3],
                                    *param_values, *curr, delta_t=1.0, nearzero=model.nearzero,
                                    doy=forcing[t, :, 3:4])
        wrong.append(outputs[0]); curr = tuple(outputs[2:])
    wrong_q = torch.stack(wrong).squeeze()
    res["state_reorder_correct"] = bool(not torch.allclose(wrapped_q, wrong_q, rtol=1e-9, atol=1e-9))
    log(f"  state reorder (Sn,S1,S2,Sc1,Sc2)->(S1,S2,Sc1,Sc2,Sn): correct={res['state_reorder_correct']}")

    # eager vs compile + fullgraph + canonical instantiation
    from dmotpy.models.core.mopex5 import mopex5_step as raw5
    pv = [torch.tensor(v, dtype=DT) for v in PARAMS]
    st = [torch.tensor(1e-6, dtype=DT) for _ in range(5)]
    qs_eager = []
    for t in range(20):
        out = step(5.0, 10.0, 3.0, pv, 180.0 + t, st)
        qs_eager.append(out[0]); st = [out[6], out[2], out[3], out[4], out[5]]
    comp_step = torch.compile(raw5, backend="inductor", mode="default", fullgraph=True)
    pv2 = [torch.tensor(v, dtype=DT) for v in PARAMS]
    st2 = [torch.tensor(1e-6, dtype=DT) for _ in range(5)]
    qs_comp = []
    for t in range(20):
        out = comp_step(torch.tensor(5.0, dtype=DT), torch.tensor(10.0, dtype=DT), torch.tensor(3.0, dtype=DT),
                        *pv2, st2[1], st2[2], st2[3], st2[4], st2[0], delta_t=1.0, nearzero=1e-6,
                        doy=torch.tensor(180.0 + t, dtype=DT))
        qs_comp.append(out[0]); st2 = [out[6], out[2], out[3], out[4], out[5]]
    res["fullgraph_compile"] = True
    res["eager_vs_compile_q_max_diff"] = float(max(abs(float(a) - float(b)) for a, b in zip(qs_eager, qs_comp)))
    log(f"  fullgraph compile ok=True; eager-vs-compile max|dQ|={res['eager_vs_compile_q_max_diff']:.2e}")

    hydro_comp = build_model("mopex5", DEVICE, warm_up=60, backend="compile",
                             parameter_mapping="auto", warmup_grad_mode="detach")
    qc = hydro_comp({"x_phy": x[:120]}, (None, torch.full((20, n, 1), 0.5, dtype=DT, device=DEVICE)))["streamflow"]
    hydro = build_model("mopex5", DEVICE, warm_up=60, backend="eager",
                        parameter_mapping="auto", warmup_grad_mode="detach")
    qe = hydro({"x_phy": x[:120]}, (None, torch.full((20, n, 1), 0.5, dtype=DT, device=DEVICE)))["streamflow"]
    res["canonical_compile_fwd"] = bool(torch.isfinite(qc).all())
    res["canonical_eager_vs_compile_max_diff"] = float((qc - qe).abs().max())
    log(f"  canonical build_model(compile, auto, detach) fwd finite={bool(torch.isfinite(qc).all())} "
        f"eager-vs-compile max diff={res['canonical_eager_vs_compile_max_diff']:.2e}")
    res["n_params"] = n
    res["nparam36"] = NPARAM_INFO_36["mopex5"]
    res["registry_nparams_match"] = (n == NPARAM_INFO_36["mopex5"] == NPARAM_INFO["mopex5"])
    res["registry_36_models"] = len(NPARAM_INFO_36) == 36 and "mopex4_1" not in NPARAM_INFO_36
    log(f"  registry: n_params={n} nparam36={NPARAM_INFO_36['mopex5']} 36-model={res['registry_36_models']}")
    return res


def main() -> None:
    log("=== MOPEX5 pre-training validation: original formulas + corrected process order ===")
    log(f"interception: original alpha/is_time (shared with restored MOPEX4); "
        f"phenology: PET_epc = GSI(T) * PET (tmin={PARAMS[6]}, trange={PARAMS[7]})")
    t0 = time.time()
    report = {"mopex5_param_bounds": {k: v for k, v in MOPEX5_PARAMS_BOUNDS.items()}}
    report["forward"] = check_forward_source_water()
    report["pet_budget"] = check_pet_budget()
    report["water_balance"] = check_water_balance()
    report["gradient"] = check_gradient_health()
    report["state_compile"] = check_state_mapping_compile()
    report["elapsed_seconds"] = time.time() - t0
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    log(f"\nreport written: {OUT / 'validation_report.json'}")
    verdict_ok = all(
        all(v for k, v in c.items() if not k.startswith("_")) for c in report["forward"].values()
    ) if isinstance(report["forward"], dict) else False
    log(f"forward all-ok: {verdict_ok} | PET closure: {report['pet_budget']['closure_pass_epc']} | "
        f"WB max daily: {report['water_balance']['seq365_max_daily_abs_residual']:.2e}")


if __name__ == "__main__":
    main()
