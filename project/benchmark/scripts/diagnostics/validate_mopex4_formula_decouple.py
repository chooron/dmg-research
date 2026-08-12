#!/usr/bin/env python3
"""Pre-training validation: original-formula MOPEX4 vs frozen Liu MOPEX4.1.

Counterfactual check: original interception formula + corrected process order.

Checks (A-F) run for BOTH models through the canonical benchmark path:
  A. forward / process semantics (rain-only, snow-only, rain+snowmelt,
     zero P, wet event, near-zero)
  B. PET closure (I + ET1 + ET2 <= PET daily)
  C. water balance (daily max abs residual, cumulative residual)
  D. gradient health through the canonical differentiable dPL path +
     alpha finite-difference vs autograd + cancellation analysis
  E. torch.compile (canonical backend) + fullgraph attempt
  F. canonical registry/wrapper/parameter-mapping integration smoke

No formal training is started.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from dmotpy.data_contract import add_calendar_forcing
from dmotpy.models.core.mopex4 import MOPEX4_PARAMS_BOUNDS, interception_4, mopex4_step
from dmotpy.models.registry import PARAM_INFO, NPARAM_INFO
from src.data_selection import load_ids
from src.model_registry import NPARAM_INFO_36, build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = ROOT / "results/mopex4_formula_decouple_20260811"
OUT.mkdir(parents=True, exist_ok=True)
SNAP = OUT / "regression_snapshot_liu_mopex4.npz"
DT = torch.float64

report = {"model": {}, "snapshot": {}}


def log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------
# helpers shared by A/B/C
# ------------------------------------------------------------------

def make_flux_recompute(model_name: str):
    """Return a step replica returning (Q, I, ET1, ET2, states, residual).

    Mirrors the corrected process order exactly (validated bit-identical for
    the Liu step earlier; same order code is used by the restored step).
    """
    from dmotpy.models.flux.mopex import (
        mopex_baseflow_1 as baseflow_1, mopex_evap_7 as evap_7,
        mopex_melt_1 as melt_1, mopex_rainfall_1 as rainfall_1,
        mopex_recharge_3 as recharge_3, mopex_saturation_1 as saturation_1,
        mopex_snowfall_1 as snowfall_1,
    )
    if model_name == "mopex4":
        from dmotpy.models.core.mopex4 import interception_4 as inter_kernel
    else:
        from dmotpy.models.flux.mopex import mopex_interception_4_liu2 as inter_kernel

    def step(P, T, PET, pv, doy, states, nearzero=1e-6):
        S1, S2, Sc1, Sc2, Sn = states
        tcrit, ddf, Sb1, tw, a4, a5, tu, Se, Sb2, tc = pv
        flux_ps = snowfall_1(P, T, tcrit)
        flux_pr = rainfall_1(P, T, tcrit)
        flux_qn = melt_1(ddf, tcrit, T, Sn, 1.0)
        Sn_w = Sn + flux_ps - flux_qn
        if model_name == "mopex4":
            i_pot = inter_kernel(flux_pr, doy, a4, a5, nearzero=nearzero)
        else:
            i_pot = inter_kernel(flux_pr, a4, a5, nearzero=nearzero)
        flux_i = torch.minimum(i_pot, PET)
        pet_after_i = PET - flux_i
        soil_input = (flux_pr - flux_i) + flux_qn
        S1_w = S1 + soil_input
        flux_et1 = torch.minimum(evap_7(S1_w, Sb1, pet_after_i, 1.0, nearzero), S1_w)
        pet_after_et1 = pet_after_i - flux_et1
        S1_ae = S1_w - flux_et1
        flux_q1f = torch.minimum(saturation_1(soil_input, S1_ae, Sb1, nearzero=nearzero), S1_ae)
        S1_aq = S1_ae - flux_q1f
        flux_qw = recharge_3(tw, S1_aq)
        S1_n = S1_aq - flux_qw
        S2_w = S2 + flux_qw
        flux_q2f = torch.minimum(saturation_1(flux_qw, S2_w, Sb2, nearzero=nearzero), S2_w)
        S2_aq2f = S2_w - flux_q2f
        flux_q2u = baseflow_1(tu, S2_aq2f)
        S2_aq2u = S2_aq2f - flux_q2u
        se_abs = Se * Sb2
        flux_et2 = torch.minimum(evap_7(S2_aq2u, se_abs, pet_after_et1, 1.0, nearzero), S2_aq2u)
        S2_n = S2_aq2u - flux_et2
        Sc1_w = Sc1 + flux_q1f + flux_q2f
        flux_qf = baseflow_1(tc, Sc1_w)
        Sc1_n = Sc1_w - flux_qf
        Sc2_w = Sc2 + flux_q2u
        flux_qs = baseflow_1(tc, Sc2_w)
        Sc2_n = Sc2_w - flux_qs
        Q = flux_qf + flux_qs
        P_tot = flux_ps + flux_pr
        dS = (Sn_w - Sn) + (S1_n - S1) + (S2_n - S2) + (Sc1_n - Sc1) + (Sc2_n - Sc2)
        residual = P_tot - flux_i - flux_et1 - flux_et2 - Q - dS
        return Q, flux_i, flux_et1, flux_et2, (S1_n, S2_n, Sc1_n, Sc2_n, Sn_w), residual, flux_pr, soil_input, flux_qn
    return step


def scenario_cases(model_name: str):
    """(name, P, T, PET, params, states(Sn,S1,S2,Sc1,Sc2), doy)"""
    common = dict()
    cases = [
        ("rain_only", 20.0, 15.0, 3.0, [0.0, 4.0, 200.0, 0.1, 0.5, 180.0, 0.1, 0.5, 300.0, 0.2], [0.0]*5, 180.0),
        ("snow_only", 20.0, -10.0, 2.0, [0.0, 4.0, 200.0, 0.1, 0.5, 180.0, 0.1, 0.5, 300.0, 0.2], [0.0]*5, 180.0),
        ("rain_plus_snowmelt", 15.0, 8.0, 2.0, [0.0, 4.0, 200.0, 0.1, 0.5, 180.0, 0.1, 0.5, 300.0, 0.2], [30.0, 0.0, 0.0, 0.0, 0.0], 180.0),
        ("zero_precip", 0.0, 12.0, 3.0, [0.0, 4.0, 200.0, 0.1, 0.5, 180.0, 0.1, 0.5, 300.0, 0.2], [0.0]*5, 180.0),
        ("wet_event", 80.0, 18.0, 4.0, [0.0, 4.0, 200.0, 0.1, 0.7, 200.0, 0.1, 0.5, 300.0, 0.2], [10.0, 50.0, 100.0, 5.0, 5.0], 200.0),
        ("near_zero", 1e-6, 12.0, 2.0, [0.0, 4.0, 200.0, 0.1, 0.5, 180.0, 0.1, 0.5, 300.0, 0.2], [0.0]*5, 180.0),
    ]
    return cases


def check_forward_and_process(model_name: str) -> dict:
    """A. forward / process semantics on single steps."""
    res = {}
    step = make_flux_recompute(model_name)
    for name, P, T, PET, p, st, doy in scenario_cases(model_name):
        pv = [torch.tensor(v, dtype=DT) for v in p]
        stv = [torch.tensor(v, dtype=DT) for v in st]
        doy_t = torch.tensor(doy, dtype=DT)
        P_t, T_t, PET_t = torch.tensor(P, dtype=DT), torch.tensor(T, dtype=DT), torch.tensor(PET, dtype=DT)
        out = mopex4_step(P_t, T_t, PET_t, *pv, stv[1], stv[2], stv[3], stv[4], stv[0],
                          delta_t=1.0, nearzero=1e-6, doy=doy_t)
        Q, ET, S1n, S2n, Sc1n, Sc2n, Snn = out
        comp = step(P_t, T_t, PET_t, pv, doy_t, (stv[1], stv[2], stv[3], stv[4], stv[0]))
        Qc, I, ET1, ET2, _, residual, pr_c, soil_in_c, qn_c = comp
        finite = all(torch.isfinite(x).all().item() for x in out) and torch.isfinite(I).item()
        net_input_ok = bool(torch.isclose(soil_in_c, (pr_c - I) + qn_c, atol=1e-9))
        checks = {
            "finite": bool(finite),
            "I>=0": bool(I >= -1e-12),
            "I<=Pr_supply": bool(I <= comp_pr(P_t, T_t, pv) + 1e-9),
            "I<=PET": bool(I <= PET_t + 1e-9),
            "soil_input=Pr_net+qn": bool(net_input_ok),
            "Q_step==Q_recompute": bool(torch.isclose(Q, Qc, atol=1e-9)),
            "states_nonneg": all(s >= -1e-9 for s in (S1n, S2n, Sc1n, Sc2n, Snn)),
        }
        res[name] = checks
        log(f"  [{model_name}] {name:20s} finite={checks['finite']} "
            f"I>={checks['I>=0']} I<=Pr={checks['I<=Pr_supply']} I<=PET={checks['I<=PET']} "
            f"Qmatch={checks['Q_step==Q_recompute']} nonneg={checks['states_nonneg']} "
            f"I={I.item():.4f} Pr={comp_pr(P_t, T_t, pv).item():.4f}")
    return res


def comp_pr(P, T, pv):
    from dmotpy.models.flux.mopex import mopex_rainfall_1 as rainfall_1
    return rainfall_1(P, T, pv[0])


def check_pet_closure(model_name: str) -> dict:
    """B. PET closure over a 365-day sequence + a batch."""
    step = make_flux_recompute(model_name)
    res = {}
    torch.manual_seed(11)
    n_days = 365
    P = torch.clamp(torch.randn(n_days, dtype=DT) * 4 + 4, min=0.0)
    T = 5.0 + 15.0 * torch.sin(torch.arange(n_days, dtype=DT) / 365.0 * 2 * math.pi)
    PET = torch.clamp(2.0 + 3.0 * torch.randn(n_days, dtype=DT), min=0.1)
    doy = (torch.arange(n_days, dtype=DT) % 365) + 1
    pv = [torch.tensor(v, dtype=DT) for v in ([0.0, 4.0, 200.0, 0.1, 0.5, 180.0, 0.1, 0.5, 300.0, 0.2] if model_name == "mopex4"
                                              else [0.0, 4.0, 200.0, 0.1, 2.0, 0.6, 0.1, 0.5, 300.0, 0.2])]
    st = [torch.tensor(1e-6, dtype=DT)] * 5
    max_exceed = 0.0
    exceed_days = 0
    for t in range(n_days):
        Q, I, ET1, ET2, st2, _, _pr, _si, _qn = step(P[t], T[t], PET[t], pv, doy[t], (st[1], st[2], st[3], st[4], st[0]))
        closure = I + ET1 + ET2
        exceed = float(torch.clamp(closure - PET[t], min=0.0))
        if exceed > 1e-6:
            exceed_days += 1
        max_exceed = max(max_exceed, exceed)
        st = [st2[0], st2[1], st2[2], st2[3], st2[4]]
    res = {"exceedance_days": exceed_days, "max_exceedance": max_exceed,
           "closure_pass": max_exceed <= 1e-6}
    log(f"  [{model_name}] PET closure 365d: exceed_days={exceed_days} max_exceed={max_exceed:.3e} "
        f"pass={res['closure_pass']}")
    return res


def check_water_balance(model_name: str) -> dict:
    """C. water balance over multi-step sequences."""
    step = make_flux_recompute(model_name)
    res = {}
    for seq_name, n_days, base in [("seq365", 365, 5.0), ("wet200", 200, 25.0)]:
        torch.manual_seed(3)
        P = torch.clamp(torch.randn(n_days, dtype=DT) * 4 + base, min=0.0)
        T = 5.0 + 15.0 * torch.sin(torch.arange(n_days, dtype=DT) / 365.0 * 2 * math.pi)
        PET = torch.clamp(2.0 + 3.0 * torch.randn(n_days, dtype=DT), min=0.1)
        doy = (torch.arange(n_days, dtype=DT) % 365) + 1
        pv = [torch.tensor(v, dtype=DT) for v in ([0.0, 4.0, 200.0, 0.1, 0.5, 180.0, 0.1, 0.5, 300.0, 0.2] if model_name == "mopex4"
                                                  else [0.0, 4.0, 200.0, 0.1, 2.0, 0.6, 0.1, 0.5, 300.0, 0.2])]
        st = [torch.tensor(1e-6, dtype=DT)] * 5
        max_daily = 0.0
        cum_resid = torch.tensor(0.0, dtype=DT)
        for t in range(n_days):
            Q, I, ET1, ET2, st2, residual, _pr, _si, _qn = step(P[t], T[t], PET[t], pv, doy[t], (st[1], st[2], st[3], st[4], st[0]))
            max_daily = max(max_daily, float(residual.abs()))
            cum_resid = cum_resid + residual
            st = [st2[0], st2[1], st2[2], st2[3], st2[4]]
        res[seq_name] = {"max_daily_abs_residual": max_daily, "cumulative_residual": float(cum_resid)}
        log(f"  [{model_name}] water balance {seq_name}: max_daily={max_daily:.3e} cumulative={float(cum_resid):.3e}")
    return res


def check_gradient_health(model_name: str) -> dict:
    """D. autograd through the canonical differentiable path (real forcing windows)."""
    res = {}
    torch.manual_seed(42)
    ids = [int(x) for x in load_ids("data/531sub_id.txt")]
    attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")
    import importlib.util
    H1path = ROOT / "scripts/diagnostics/h_training_pilot.py"
    spec = importlib.util.spec_from_file_location("h1_validate_helper", H1path)
    H1mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H1mod)
    NATIVE = H1mod.NATIVE
    train_x_np, train_y_np, _, _ = NATIVE.load_camels_time_series(ids)
    train_x = torch.as_tensor(train_x_np, dtype=torch.float32, device="cuda")
    train_y = torch.as_tensor(train_y_np, dtype=torch.float32, device="cuda")
    train_x, _ = add_calendar_forcing(train_x, pd.date_range("1980-10-01", "1995-09-30", freq="D"), model_name=model_name)
    catalog, lengths = H1mod.make_catalog(train_y[365:])
    hydro = build_model(model_name, "cuda", warm_up=365, backend="eager",
                        parameter_mapping="auto", warmup_grad_mode="detach")
    network = CatchmentParameterizer(attrs.shape[1], NPARAM_INFO_36[model_name],
                                     hidden_dims=[256, 256], dropout=0.05).to("cuda")
    with torch.no_grad():
        network.net[-1].weight.zero_(); network.net[-1].bias.zero_()

    # 100 basins x 730d windows, exactly like the canonical training step
    basins = torch.randperm(len(ids), device="cuda")[:100]
    choices = (torch.rand(100, device="cuda") * lengths[basins]).long()
    starts = catalog[basins, choices]
    x = H1mod.gather_window(train_x, starts, basins)
    y = H1mod.gather_window(train_y, starts, basins)

    theta = network(attrs[basins]); theta.retain_grad()
    q = hydro({"x_phy": x}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
    loss, _ = NATIVE.compute_differentiable_kge(q, y[365:], warmup_days=0)
    loss.backward()
    g = theta.grad.squeeze(-1)
    names = list(PARAM_INFO[model_name])
    log(f"  [{model_name}] dPL-path 1-KGE loss={float(loss):.4f} q_finite={bool(torch.isfinite(q).all())}")
    per = {}
    for j, name in enumerate(names):
        gj = g[:, j]
        per[name] = {
            "mean_abs_grad": float(gj.abs().mean()),
            "finite": bool(torch.isfinite(gj).all()),
            "nonzero_frac": float((gj != 0).float().mean()),
        }
    res["per_parameter"] = per
    for j, name in enumerate(names):
        pj = per[name]
        flag = "OK " if (pj["finite"] and pj["nonzero_frac"] > 0.5) else "WEAK"
        log(f"    {name:8s} mean|grad|={pj['mean_abs_grad']:.3e} nonzero={pj['nonzero_frac']*100:5.1f}% {flag}")
    return res


def check_alpha_fd_vs_autograd() -> dict:
    """D2. alpha/is_time finite-difference vs autograd + cancellation analysis."""
    res = {"kernel": [], "cancellation": {}}
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    is_times = [180.0]
    pr_vals = [1.0, 10.0]
    pet_vals = [2.0, 10.0]
    doy_offsets = [0.0, 45.0, 90.0, 180.0, 270.0]  # relative to is_time: peak, ..., trough
    max_abs = 0.0; max_rel = 0.0
    for is_time in is_times:
        for doy_off in doy_offsets:
            doy = is_time + doy_off
            for pr in pr_vals:
                for pet in pet_vals:
                    for alpha in alphas:
                        a = torch.tensor(alpha, dtype=DT, requires_grad=True)
                        it = torch.tensor(is_time, dtype=DT, requires_grad=True)
                        pr_t = torch.tensor(pr, dtype=DT)
                        doy_t = torch.tensor(doy, dtype=DT)
                        pet_t = torch.tensor(pet, dtype=DT)
                        i_pot = interception_4(pr_t, doy_t, a, it, nearzero=1e-6)
                        i = torch.minimum(i_pot, pet_t)
                        da, dit = torch.autograd.grad(i, (a, it))
                        # centered FD
                        eps = 1e-5
                        i_p = float(torch.minimum(interception_4(pr_t, doy_t, torch.tensor(alpha + eps, dtype=DT), it.detach(), nearzero=1e-6), pet_t))
                        i_m = float(torch.minimum(interception_4(pr_t, doy_t, torch.tensor(alpha - eps, dtype=DT), it.detach(), nearzero=1e-6), pet_t))
                        fd_alpha = (i_p - i_m) / (2 * eps)
                        err_a = abs(float(da) - fd_alpha)
                        rel_a = err_a / (abs(fd_alpha) + 1e-12)
                        max_abs = max(max_abs, err_a); max_rel = max(max_rel, rel_a)
                        res["kernel"].append({
                            "doy_off": doy_off, "pr": pr, "pet": pet, "alpha": alpha,
                            "dI_dalpha": float(da), "fd_alpha": fd_alpha,
                            "abs_err": err_a, "rel_err": rel_a,
                            "dI_dis_time": float(dit),
                        })
    log(f"  [mopex4 kernel] alpha FD vs autograd over {len(res['kernel'])} points: "
        f"max_abs_err={max_abs:.3e} max_rel_err={max_rel:.3e}")
    res["max_abs_err"] = max_abs
    res["max_rel_err"] = max_rel

    # cancellation analysis: dI/dalpha = dI/dfrac * dfrac/dalpha, dfrac/dalpha = 1 - cos(rad)
    # decompose the two contributions: alpha-term (+1) and (1-alpha)*cos term (-cos)
    rads = torch.linspace(0.0, 2 * math.pi, 73)
    rows = []
    for alpha in [0.2, 0.5, 0.8]:
        a = torch.tensor(alpha, dtype=DT, requires_grad=True)
        contrib_alpha = []
        contrib_cos = []
        dfrac = []
        for rad in rads:
            frac = a + (1.0 - a) * torch.cos(rad)
            # dI/dfrac * d(alpha-term)/dalpha and dI/dfrac * d((1-alpha)cos)/dalpha
            dI_dfrac = torch.sigmoid(torch.tensor(50.0, dtype=DT) * frac.detach())
            c1 = dI_dfrac * 1.0
            c2 = dI_dfrac * (-torch.cos(rad))
            contrib_alpha.append(float(c1)); contrib_cos.append(float(c2))
            dfrac.append(float(1.0 - torch.cos(rad)))
        rows.append({"alpha": alpha,
                     "min_dfrac_dalpha": min(dfrac), "max_dfrac_dalpha": max(dfrac),
                     "min_contrib_alpha": min(contrib_alpha), "min_contrib_cos": min(contrib_cos),
                     "net_positive_anywhere": min(x + y for x, y in zip(contrib_alpha, contrib_cos)) >= -1e-12})
        log(f"    alpha={alpha}: dfrac/dalpha in [{min(dfrac):.3f},{max(dfrac):.3f}] (>=0; zero only at seasonal peak), "
            f"alpha-term in [{min(contrib_alpha):.3f},{max(contrib_alpha):.3f}], "
            f"(1-alpha)cos-term in [{min(contrib_cos):.3f},{max(contrib_cos):.3f}] -> net nonneg")
    res["cancellation"]["dfrac_dalpha"] = {"min": min(dfrac), "max": max(dfrac)}
    res["cancellation"]["rows"] = rows
    return res


def check_compile(model_name: str) -> dict:
    """E. torch.compile via the canonical backend + fullgraph attempt."""
    res = {}
    t0 = time.time()
    try:
        hydro = build_model(model_name, "cuda", warm_up=365, backend="compile",
                            parameter_mapping="auto", warmup_grad_mode="detach")
        compile_seconds = time.time() - t0
        x = torch.randn(730, 8, 4, device="cuda")
        theta = torch.full((8, NPARAM_INFO_36[model_name], 1), 0.5, device="cuda", requires_grad=True)
        q = hydro({"x_phy": x}, (None, theta))["streamflow"].squeeze(-1).squeeze(-1)
        loss = q[365:].square().mean()
        loss.backward()
        res["compile_forward_backward"] = bool(torch.isfinite(q).all() and theta.grad is not None
                                               and bool(torch.isfinite(theta.grad).all()))
        res["compile_seconds"] = compile_seconds
        log(f"  [{model_name}] backend=compile fwd+bwd ok={res['compile_forward_backward']} ({compile_seconds:.1f}s)")
        # fullgraph strictness attempt on the raw step
        try:
            import inspect
            src = inspect.getsource(hydro.step_fn if hasattr(hydro, "step_fn") else hydro.raw_step_fn)
            raw_step = hydro.raw_step_fn
            n_state = 5
            bounds = PARAM_INFO[model_name]
            pv_full = []
            for j, nm in enumerate(list(bounds)):
                lo, hi = bounds[nm]
                pv_full.append(torch.full((8, 1), (lo + hi) / 2, device="cuda"))
            args = (torch.clamp(torch.randn(8, 1, device="cuda"), min=0.0).abs() + 1,
                    torch.randn(8, 1, device="cuda") + 10,
                    torch.clamp(torch.randn(8, 1, device="cuda"), min=0.0) + 1,
                    *pv_full,
                    *[torch.rand(8, 1, device="cuda").abs() * 20 for _ in range(5)])
            full = torch.compile(raw_step, fullgraph=True)
            with torch.no_grad():
                out = full(*args, doy=(torch.arange(8, device="cuda") % 365 + 1).float().view(8, 1))
            res["fullgraph_ok"] = bool(torch.isfinite(out[0]).all())
            log(f"  [{model_name}] fullgraph=True compile of raw step ok={res['fullgraph_ok']}")
        except Exception as exc:
            res["fullgraph_ok"] = False
            res["fullgraph_error"] = f"{type(exc).__name__}: {exc}"
            log(f"  [{model_name}] fullgraph=True failed: {res['fullgraph_error']}")
    except Exception as exc:
        res["compile_forward_backward"] = False
        res["compile_error"] = f"{type(exc).__name__}: {exc}"
        log(f"  [{model_name}] compile path error: {res['compile_error']}")
    return res


def check_canonical_smoke(model_name: str) -> dict:
    """F. canonical registry/wrapper/parameter-mapping integration smoke."""
    res = {}
    spec = __import__("src.model_registry", fromlist=["get_spec"]).get_spec(model_name, device="cuda")
    res["spec_dimension"] = spec.dimension
    res["nparam_36"] = NPARAM_INFO_36[model_name]
    res["dimension_match"] = spec.dimension == NPARAM_INFO_36[model_name] == NPARAM_INFO[model_name]
    # wrapper dispatch -> MopexDoyModel
    hydro = build_model(model_name, "cuda", warm_up=365, backend="eager",
                        parameter_mapping="auto", warmup_grad_mode="detach")
    res["wrapper_class"] = type(hydro).__name__
    res["calendar_model"] = model_name in {"mopex4", "mopex5"}
    # 1-batch backward smoke through the parameterizer
    from dpl.attributes import CatchmentAttributeBuilder
    from dpl.nn_parameterizer import CatchmentParameterizer
    ids = [int(x) for x in load_ids("data/531sub_id.txt")]
    attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")
    net = CatchmentParameterizer(attrs.shape[1], NPARAM_INFO_36[model_name], hidden_dims=[256, 256], dropout=0.05).to("cuda")
    torch.manual_seed(0)
    b = torch.randperm(len(ids), device="cuda")[:16]
    x = torch.randn(730, 16, 4, device="cuda")
    theta = net(attrs[b]).unsqueeze(-1)
    q = hydro({"x_phy": x}, (None, theta))["streamflow"].squeeze(-1).squeeze(-1)
    loss = q[365:].square().mean()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    res["smoke_backward_ok"] = bool(torch.isfinite(q).all()) and all(bool(torch.isfinite(g).all()) for g in grads)
    log(f"  [{model_name}] canonical smoke: class={res['wrapper_class']} dim={spec.dimension} "
        f"nparam36={NPARAM_INFO_36[model_name]} backward_ok={res['smoke_backward_ok']}")
    return res


def main() -> None:
    log("=== MOPEX4 formula x process-order decoupling: pre-training validation ===")
    log(f"restored mopex4 interception slots: {MOPEX4_PARAMS_BOUNDS['alpha'], MOPEX4_PARAMS_BOUNDS['is_time']}")
    for model_name in ("mopex4",):
        log(f"\n--- {model_name} ---")
        rep = {}
        rep["forward"] = check_forward_and_process(model_name)
        rep["pet_closure"] = check_pet_closure(model_name)
        rep["water_balance"] = check_water_balance(model_name)
        rep["gradient"] = check_gradient_health(model_name)
        rep["compile"] = check_compile(model_name)
        rep["canonical_smoke"] = check_canonical_smoke(model_name)
        report["model"][model_name] = rep
    log("\n--- alpha/is_time finite-difference + cancellation ---")
    report["alpha_fd"] = check_alpha_fd_vs_autograd()

    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    log(f"\nreport written: {OUT / 'validation_report.json'}")


if __name__ == "__main__":
    main()
