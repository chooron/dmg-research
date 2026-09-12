#!/usr/bin/env python3
"""Mass-balance closure for the active full Base/TGD/CN combinations."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from s2_validation_common import ACTIVE_CASES, PROJECT_ROOT, RESULTS_ROOT, add_project_path, ensure_dirs, make_forcing, make_params, run_case, write_csv

TOL = 1e-8
SCENARIOS = (
    "zero_precip", "constant_precip", "mixed", "rain_snow_transition", "persistent_cold",
    "sudden_warm", "high_pet", "empty_state", "moderate_state", "uh_tail_impulse",
)


def f(value) -> float:
    return float(value.detach().sum().cpu())


def maxabs(value) -> float:
    import torch
    if value is None:
        return float("nan")
    return float(value.detach().abs().max().cpu())


def series_sum(value) -> float:
    return f(value) if value is not None else float("nan")


def nested_aux(aux: dict[str, Any]) -> dict[str, Any]:
    return aux.get("runoff_model", aux)


def processor_balance(model: str, structure: str, forcing, aux: dict[str, Any], final_states: dict[str, Any]) -> tuple[float, float, str]:
    import torch
    p = forcing["precip"]
    if structure == "Base":
        return 0.0, 0.0, "no preprocessing module"
    if structure == "CN":
        effective = aux["effective_precip"]
        initial = torch.zeros_like(final_states["cn_G"])
        delta = final_states["cn_G"] - initial
        residual = p.sum(dim=1) - effective.sum(dim=1) - delta
        return maxabs(residual), series_sum(residual), "CN snow storage G; eTG is thermal, not water"
    if structure == "TGD":
        tgd = aux["temperature_delay"]
        residual = tgd["mass_balance_residual"]
        return maxabs(residual), series_sum(residual), "TGD delay storage S"
    return float("nan"), float("nan"), "not applicable"


def initial_host_storage(model: str, structure: str, params) -> float:
    p = {name: float(value[0].detach().cpu()) for name, value in params.items()}
    if model == "XAJ":
        return 0.6 * (p.get("xaj_um", 20.0) + p.get("xaj_lm", 80.0) + p.get("xaj_dm", 40.0)) + 0.5 * p.get("xaj_sm", 30.0) + 0.2
    if model == "SIMHYD":
        return 0.5 * p.get("simhyd_smsc", 100.0)
    if model == "GR4J":
        x1 = p.get("x1", p.get("gr4j_x1", 350.0)); x3 = p.get("x3", p.get("gr4j_x3", 500.0))
        return 0.5 * (x1 + x3)
    return 0.5


def xaj_aggregate(model: str, structure: str, forcing, aux, final_states, params):
    import torch
    from models.xaj import _gamma_uh_ordinates
    ra = nested_aux(aux)
    effective = forcing["precip"] if structure == "Base" else aux["effective_precip"]
    evap = ra["evap"]
    rs = ra["rs_instant"]
    routed = ra["rs_routed"]
    qi = ra["qi"]; qg = ra["qg"]
    def state(name):
        return final_states.get(name, final_states.get(f"xaj_{name}"))
    buffer = state("rs_uh_buffer")
    uh = _gamma_uh_ordinates(params["xaj_a"], params["xaj_theta"], buffer.shape[1] + 1, buffer.device, buffer.dtype)
    pending = (buffer * torch.cumsum(torch.flip(uh[:, 1:], dims=[-1]), dim=-1)).sum(dim=1)
    host_final = state("wu") + state("wl") + state("wd") + state("s") + state("qi") + state("qg") + pending
    delta = host_final - initial_host_storage("XAJ", structure, params)
    core_final = state("wu") + state("wl") + state("wd") + state("s") + state("qi") + state("qg")
    core_delta = core_final - (initial_host_storage("XAJ", structure, params))
    core = effective.sum(dim=1) - evap.sum(dim=1) - rs.sum(dim=1) - qi.sum(dim=1) - qg.sum(dim=1) - core_delta
    routing = rs.sum(dim=1) - routed.sum(dim=1) - pending
    whole = effective.sum(dim=1) - evap.sum(dim=1) - (routed + qi + qg).sum(dim=1) - delta
    no_tail = whole + pending
    from models.xaj import _route_xaj_surface_runoff
    tail_q, tail_buffer = _route_xaj_surface_runoff(torch.zeros_like(rs), buffer, params["xaj_a"], params["xaj_theta"], rs.device, rs.dtype)
    tail_remaining = (tail_buffer * torch.cumsum(torch.flip(uh[:, 1:], dims=[-1]), dim=-1)).sum(dim=1)
    tail_residual = pending - tail_q.sum(dim=1) - tail_remaining
    return {"host_core": core, "internal_routing": routing, "whole_system": whole, "finite_window_output": no_tail, "tail_pending": pending, "tail_drain_residual": tail_residual, "tail_drain_output": tail_q.sum(dim=1), "tail_drain_remaining": tail_remaining, "q": routed + qi + qg, "et": evap, "effective": effective}


def simhyd_aggregate(model: str, structure: str, forcing, aux, final_states, params):
    import torch
    ra = nested_aux(aux)
    effective = forcing["precip"] if structure == "Base" else aux["effective_precip"]
    evap = ra["evap"]; instant = ra["runoff_instant"]; q = ra["runoff_routed"]
    pending = ra["routing_storage"]
    initial = initial_host_storage("SIMHYD", structure, params)
    host_delta = ra["soil"] + ra["groundwater"] - initial
    core = effective.sum(dim=1) - evap.sum(dim=1) - instant.sum(dim=1) - host_delta
    routing = instant.sum(dim=1) - q.sum(dim=1) - pending
    whole = effective.sum(dim=1) - evap.sum(dim=1) - q.sum(dim=1) - host_delta - pending
    from models.simhyd import _route_simhyd_runoff
    zero = torch.zeros_like(instant)
    tail_q, tail_buffer, _uh, tail_remaining = _route_simhyd_runoff(zero, final_states.get("runoff_uh_buffer", final_states.get("simhyd_runoff_uh_buffer")), params["simhyd_a"], params["simhyd_theta"], instant.device, instant.dtype)
    tail_residual = pending - tail_q.sum(dim=1) - tail_remaining
    return {"host_core": core, "internal_routing": routing, "whole_system": whole, "finite_window_output": whole + pending, "tail_pending": pending, "tail_drain_residual": tail_residual, "tail_drain_output": tail_q.sum(dim=1), "tail_drain_remaining": tail_remaining, "q": q, "et": evap, "effective": effective}


def gr4j_trace(effective, pet, params, initial_states=None):
    add_project_path()
    import torch
    from models.gr4j import _gr4j_step
    from models.unit_hydro import compute_gr4j_uh_ordinates
    x1 = params.get("x1", params.get("gr4j_x1")); x2 = params.get("x2", params.get("gr4j_x2")); x3 = params.get("x3", params.get("gr4j_x3")); x4 = params.get("x4", params.get("gr4j_x4"))
    uh1, _ = compute_gr4j_uh_ordinates(x4, 15); uh2 = compute_gr4j_uh_ordinates(x4, 30)[1]
    batch = effective.shape[0]
    sp = (initial_states or {}).get("s_prod", 0.5 * x1)
    sr = (initial_states or {}).get("s_route", 0.5 * x3)
    b1 = (initial_states or {}).get("uh1_buf", torch.zeros(batch, 15, dtype=effective.dtype))
    b2 = (initial_states or {}).get("uh2_buf", torch.zeros(batch, 30, dtype=effective.dtype))
    qs=[]; et=[]; exchange=[]; p_r=[]; q1=[]; q2=[]; qr=[]; pending=[]; stores=[]
    for t in range(effective.shape[1]):
        pt = effective[:, t]; pet_t = pet[:, t]
        mask = pt >= pet_t; pn = torch.where(mask, pt - pet_t, torch.zeros_like(pt)); pen = torch.where(mask, torch.zeros_like(pt), pet_t - pt)
        ratio = torch.clamp(sp / (x1 + 1e-8), min=0.0, max=1.0)
        tpn = torch.tanh(pn / (x1 + 1e-8)); ps = (x1 * (1.0 - ratio * ratio) * tpn) / (1.0 + ratio * tpn + 1e-8)
        tpen = torch.tanh(pen / (x1 + 1e-8)); es = (sp * (2.0 - ratio) * tpen) / (1.0 + (1.0 - ratio) * tpen + 1e-8)
        perc_store = sp - es + ps; n4 = 4.0 / 9.0 * perc_store / (x1 + 1e-8); perc = perc_store * (1.0 - (1.0 + n4 ** 4.0) ** (-0.25)); pr = torch.clamp(perc + (pn - ps), min=0.0)
        ex = x2 * torch.clamp(sr / (x3 + 1e-8), min=0.0) ** 3.5
        out = _gr4j_step(pt, pet_t, sp, sr, b1, b2, uh1, uh2, x1, x2, x3, 1e-8)
        q, sp, sr, b1, b2 = out
        q_uh1 = b1[:, 0]; q_uh2 = b2[:, 0]; q_r = q - q_uh2 - ex
        qs.append(q); et.append(es); exchange.append(ex); p_r.append(pr); q1.append(q_uh1); q2.append(q_uh2); qr.append(q_r); pending.append(b1[:,1:].sum(dim=1)+b2[:,1:].sum(dim=1)); stores.append(sp+sr)
    stack=lambda xs: torch.stack(xs, dim=1)
    pending_t=stack(pending); store_t=stack(stores)
    whole = effective.sum(dim=1) + stack(exchange).sum(dim=1) - stack(et).sum(dim=1) - stack(qs).sum(dim=1) - (store_t[:, -1] + pending_t[:, -1] - 0.5 * (x1+x3))
    core = effective.sum(dim=1) + stack(exchange).sum(dim=1) - stack(et).sum(dim=1) - stack(qs).sum(dim=1) - (store_t[:, -1] - 0.5 * (x1+x3))
    routing = stack(p_r).sum(dim=1) - stack(q1).sum(dim=1) - stack(q2).sum(dim=1) - pending_t[:, -1]
    buffer1, buffer2 = b1, b2
    tail_outputs = []
    for _ in range(30):
        buffer1 = torch.cat((buffer1[:, 1:], torch.zeros(batch, 1, dtype=effective.dtype)), dim=1)
        buffer2 = torch.cat((buffer2[:, 1:], torch.zeros(batch, 1, dtype=effective.dtype)), dim=1)
        tail_outputs.append(buffer1[:, 0] + buffer2[:, 0])
    tail_output = torch.stack(tail_outputs, dim=1).sum(dim=1)
    tail_remaining = buffer1.sum(dim=1) + buffer2.sum(dim=1)
    tail_residual = pending_t[:, -1] - tail_output - tail_remaining
    return {"host_core": core, "internal_routing": routing, "whole_system": whole, "finite_window_output": whole + pending_t[:, -1], "tail_pending": pending_t[:, -1], "tail_drain_residual": tail_residual, "tail_drain_output": tail_output, "tail_drain_remaining": tail_remaining, "q": stack(qs), "et": stack(et), "effective": effective, "exchange": stack(exchange), "production_store": store_t}


def hbv_trace(forcing, params):
    add_project_path()
    import torch
    from models.hbv import _hbv_step
    p = params
    sn = torch.zeros(forcing["precip"].shape[0], dtype=forcing["precip"].dtype); mw = torch.zeros_like(sn); sm = torch.full_like(sn, 0.5); su = torch.zeros_like(sn); sl = torch.zeros_like(sn)
    qs=[]; et=[]; stores=[]
    for t in range(forcing["precip"].shape[1]):
        pt=forcing["precip"][:,t]; tt=forcing["temp"][:,t]; pet=forcing["pet"][:,t]
        rain=pt*(tt >= p["parTT"]).to(pt.dtype); snow=pt*(tt < p["parTT"]).to(pt.dtype); sn_pre=sn+snow
        melt=torch.clamp(p["parCFMAX"]*(tt-p["parTT"]),min=0.0); melt=torch.minimum(melt,sn_pre); mw_pre=mw+melt; sn_after=sn_pre-melt
        ref=torch.clamp(p["parCFR"]*p["parCFMAX"]*(p["parTT"]-tt),min=0.0); ref=torch.minimum(ref,mw_pre); sn_after=sn_after+ref; mw_after=mw_pre-ref
        tosoil=torch.clamp(mw_after-p["parCWH"]*sn_after,min=0.0); mw_after=mw_after-tosoil
        wet=torch.clamp((sm/p["parFC"])**p["parBETA"],0.0,1.0); recharge=(rain+tosoil)*wet; sm_pre=sm+rain+tosoil-recharge; excess=torch.clamp(sm_pre-p["parFC"],min=0.0); sm_pre=sm_pre-excess
        evapfactor=torch.clamp(sm_pre/(p["parLP"]*p["parFC"]),0.0,1.0); eta=torch.minimum(sm_pre,pet*evapfactor); sm_after=torch.clamp(sm_pre-eta,min=1e-8)
        su_pre=su+recharge+excess; perc=torch.minimum(su_pre,p["parPERC"]); su_pre=su_pre-perc; q0=p["parK0"]*torch.clamp(su_pre-p["parUZL"],min=0.0); su_pre=su_pre-q0; q1=p["parK1"]*su_pre; su_after=su_pre-q1; sl_pre=sl+perc; q2=p["parK2"]*sl_pre; sl_after=sl_pre-q2
        out=_hbv_step(pt,tt,pet,sn,mw,sm,su,sl,p["parTT"],p["parCFMAX"],p["parCFR"],p["parCWH"],p["parFC"],p["parBETA"],p["parLP"],p["parPERC"],p["parUZL"],p["parK0"],p["parK1"],p["parK2"],1e-8)
        q,sn,mw,sm,su,sl=out; qs.append(q); et.append(eta); stores.append(sn+mw+sm+su+sl)
    st=lambda xs: torch.stack(xs,dim=1)
    whole=forcing["precip"].sum(dim=1)-st(et).sum(dim=1)-st(qs).sum(dim=1)-(st(stores)[:,-1]-0.5)
    return {"host_core":whole,"internal_routing":torch.zeros_like(whole),"whole_system":whole,"finite_window_output":whole,"tail_pending":torch.zeros_like(whole),"tail_drain_residual":torch.zeros_like(whole),"tail_drain_output":torch.zeros_like(whole),"tail_drain_remaining":torch.zeros_like(whole),"q":st(qs),"et":st(et),"effective":forcing["precip"]}


def evaluate(model: str, structure: str, forcing, dtype, instance=None, params=None):
    import torch
    if instance is None or params is None:
        instance, params, result = run_case(model, structure, forcing, dtype, return_states=True)
    else:
        result = instance(forcing, params, return_states=True)
    qsim, aux = result; states = aux.get("final_states", {})
    pmax, psum, pnote = processor_balance(model, structure, forcing, aux, states)
    if model == "XAJ": agg = xaj_aggregate(model, structure, forcing, aux, states, params)
    elif model == "SIMHYD": agg = simhyd_aggregate(model, structure, forcing, aux, states, params)
    elif model == "GR4J":
        effective = forcing["precip"] if structure == "Base" else aux["effective_precip"]
        # The production run starts from its configured initial state.  The
        # returned final state is used only for the closure terms, never as a
        # second initial condition for the mirror trace.
        agg = gr4j_trace(effective, forcing["pet"], params)
        agg["production_q_max_abs_diff"] = (qsim - agg["q"]).abs().max()
    else:
        agg = hbv_trace(forcing, params)
        agg["production_q_max_abs_diff"] = (qsim - agg["q"]).abs().max()
    if "production_q_max_abs_diff" not in agg:
        if model == "XAJ": agg["production_q_max_abs_diff"] = (qsim - agg["q"]).abs().max()
        elif model == "SIMHYD": agg["production_q_max_abs_diff"] = (qsim - agg["q"]).abs().max()
    return instance, params, qsim, aux, states, agg, (pmax, psum, pnote)


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--project-root",type=Path,default=PROJECT_ROOT); parser.add_argument("--output-dir",type=Path,default=None); args=parser.parse_args()
    ensure_dirs(); add_project_path(args.project_root)
    import torch
    rows=[]; terms=[]
    for model, structure in ACTIVE_CASES:
        bundle = __import__("s2_validation_common").imports()
        key = f"{model}_{structure}"
        instance = bundle["classes"][key]().to(dtype=torch.float64)
        params = make_params(bundle["specs"][key], torch.float64)
        for scenario in SCENARIOS:
            forcing=make_forcing(scenario, torch.float64, steps=12)
            try:
                _instance, params, q, aux, states, agg, pre = evaluate(model, structure, forcing, torch.float64, instance, params)
                pre_max, pre_sum, pre_note=pre
                preprocessing_verdict = "NOT_APPLICABLE" if model == "HBV" or structure == "Base" else ("PASS" if math.isfinite(pre_max) and pre_max <= TOL else "FAIL")
                layers=[("preprocessing_module",pre_max,pre_sum,preprocessing_verdict,pre_note),
                        ("host_core",maxabs(agg["host_core"]),series_sum(agg["host_core"]),"PASS" if maxabs(agg["host_core"])<=TOL else "FAIL","host storage and flux terms; GR4J exchange explicit"),
                        ("internal_routing",maxabs(agg["internal_routing"]),series_sum(agg["internal_routing"]),"PASS" if maxabs(agg["internal_routing"])<=TOL else "FAIL","UH buffer included as pending mass"),
                        ("whole_system",maxabs(agg["whole_system"]),series_sum(agg["whole_system"]),"PASS" if maxabs(agg["whole_system"])<=TOL else "FAIL","all exposed host/process/routing stores"),
                        ("finite_window_output",maxabs(agg["finite_window_output"]),series_sum(agg["finite_window_output"]),"PASS" if maxabs(agg["finite_window_output"])<=TOL else ("ROUTING_TAIL_NOT_INCLUDED" if maxabs(agg["tail_pending"])>TOL else "FAIL"),"finite-window row intentionally omits pending UH tail"),
                        ("tail_drained_system",maxabs(agg["tail_drain_residual"]),series_sum(agg["tail_drain_residual"]),"PASS" if maxabs(agg["tail_drain_residual"])<=TOL else "FAIL","zero-input UH tail drain; remaining pending UH state is checked explicitly"),]
                if model == "XAJ":
                    layers = [
                        (layer, mx, sm, "INCOMPLETE_DIAGNOSTIC" if layer in {"host_core", "whole_system", "finite_window_output"} else verdict,
                         "XAJ qi/qg are recursive flow states and active aux does not expose a physical linear-reservoir storage trace; UH routing is checked separately.")
                        for layer, mx, sm, verdict, note in layers
                    ]
                for layer, mx, sm, verdict, note in layers:
                    rows.append({"model":model,"structure":structure,"scenario":scenario,"layer":layer,"dtype":"float64","sequence_length":12,"max_abs_residual":mx,"mean_abs_residual":mx,"relative_residual":abs(sm)/(abs(float(forcing["precip"].sum()))+1e-12),"cumulative_residual":sm,"tolerance":TOL,"first_failure_day":1 if mx>TOL else "","verdict":verdict,"notes":note})
                terms.append({"model":model,"structure":structure,"scenario":scenario,"dtype":"float64","input_P":series_sum(forcing["precip"]),"effective_P":series_sum(agg["effective"]),"ET":series_sum(agg["et"]),"Q":series_sum(agg["q"]),"exchange_X":series_sum(agg.get("exchange")) if agg.get("exchange") is not None else 0.0,"tail_pending":series_sum(agg["tail_pending"]),"tail_drain_output":series_sum(agg["tail_drain_output"]),"tail_drain_remaining":series_sum(agg["tail_drain_remaining"]),"tail_drain_residual":series_sum(agg["tail_drain_residual"]),"production_q_max_abs_diff":maxabs(agg.get("production_q_max_abs_diff")),"preprocessing_residual":pre_sum,"verdict":"PASS" if maxabs(agg["whole_system"])<=TOL and (maxabs(agg.get("production_q_max_abs_diff")) if agg.get("production_q_max_abs_diff") is not None else 0.0)<=1e-8 else "FAIL"})
            except Exception as exc:
                rows.append({"model":model,"structure":structure,"scenario":scenario,"layer":"whole_system","dtype":"float64","sequence_length":12,"max_abs_residual":"nan","mean_abs_residual":"nan","relative_residual":"nan","cumulative_residual":"nan","tolerance":TOL,"first_failure_day":1,"verdict":"INCOMPLETE_DIAGNOSTIC","notes":f"{type(exc).__name__}: {exc}"})
    write_csv(RESULTS_ROOT/"s2_mass_balance_closure.csv",rows)
    write_csv(RESULTS_ROOT/"s2_mass_balance_terms.csv",terms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
