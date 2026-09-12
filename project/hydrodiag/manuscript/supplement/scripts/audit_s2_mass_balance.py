#!/usr/bin/env python3
"""Run deterministic small forward and mass-balance probes."""
from __future__ import annotations
import argparse, math, sys, traceback
from pathlib import Path
import numpy as np
from s2_audit_utils import ensure_dirs, json_safe, project_root_from_args, supplement_dir, write_csv, write_json

HOSTS = ("XAJ","GR4J","SIMHYD")
STRUCTURES = ("Base","CN","TGD")

def _params(key, torch):
    from models.parameter_specs import (
        HBV_PARAM_SPECS, GR4J_PARAM_SPECS, XAJ_PARAM_SPECS,
        XAJ_CN_PARAM_SPECS, XAJ_TGD_PARAM_SPECS,
        GR4J_CN_PARAM_SPECS, GR4J_TGD_PARAM_SPECS,
        SIMHYD_PARAM_SPECS, SIMHYD_CN_PARAM_SPECS, SIMHYD_TGD_PARAM_SPECS,
    )
    specs = {"XAJ":XAJ_PARAM_SPECS,"XAJ_CN":XAJ_CN_PARAM_SPECS,"XAJ_TGD":XAJ_TGD_PARAM_SPECS,"GR4J":GR4J_PARAM_SPECS,"GR4J_CN":GR4J_CN_PARAM_SPECS,"GR4J_TGD":GR4J_TGD_PARAM_SPECS,"SIMHYD":SIMHYD_PARAM_SPECS,"SIMHYD_CN":SIMHYD_CN_PARAM_SPECS,"SIMHYD_TGD":SIMHYD_TGD_PARAM_SPECS,"HBV":HBV_PARAM_SPECS}[key]
    vals=[]
    for n,s in specs.items():
        if n == "tgd_tau": vals.append(math.sqrt(s["lower"]*s["upper"]))
        else: vals.append((s["lower"]+s["upper"])/2)
    return {n: torch.tensor([v], dtype=torch.float32) for n,v in zip(specs,vals)}, specs

def probe(root: Path, out: Path) -> dict:
    sys.path.insert(0,str(root))
    import torch
    torch.manual_seed(1729); np.random.seed(1729)
    from models import XAJ, XAJWithCemaNeige, XAJWithTemperatureConditionedDelay, GR4J, GR4JWithCemaNeige, GR4JWithTemperatureConditionedDelay, SIMHYD, SIMHYDWithCemaNeige, SIMHYDWithTemperatureConditionedDelay, HBV
    classes={"XAJ":XAJ,"XAJ_CN":XAJWithCemaNeige,"XAJ_TGD":XAJWithTemperatureConditionedDelay,"GR4J":GR4J,"GR4J_CN":GR4JWithCemaNeige,"GR4J_TGD":GR4JWithTemperatureConditionedDelay,"SIMHYD":SIMHYD,"SIMHYD_CN":SIMHYDWithCemaNeige,"SIMHYD_TGD":SIMHYDWithTemperatureConditionedDelay,"HBV":HBV}
    P=torch.tensor([[1.0,0.0,8.0,0.0,4.0,12.0,0.0,3.0]],dtype=torch.float32)
    T=torch.tensor([[-4.0,-2.0,0.0,2.0,5.0,1.0,-1.0,7.0]],dtype=torch.float32)
    PET=torch.full_like(P,2.0)
    rows=[]; one=[]; grads=[]; details=[]
    for key, cls in classes.items():
        try:
            params,specs=_params(key,torch)
            forcing={"precip":P,"temp":T,"pet":PET}
            if "TGD" in key: forcing.update({"temp_mean_train":torch.tensor([1.0]),"temp_std_train":torch.tensor([4.0])})
            model=cls().eval()
            q,aux=model(forcings=forcing,params=params,return_states=True)
            finite=bool(torch.isfinite(q).all())
            one.append({"model_key":key,"status":"PASS" if finite else "FAIL","q_shape":str(tuple(q.shape)),"finite":finite,"max_abs_q":float(q.abs().max()),"evidence":"runtime forward, float32 CPU"})
            if "TGD" in key:
                res=aux.get("temperature_delay",{}).get("mass_balance_residual",aux.get("mass_balance_residual"))
                residual=float(res.abs().max()) if res is not None else float("nan")
                input_used=aux["effective_precip"]
                storage_delta=float(aux["final_states"]["tgd_S"].item()) if "final_states" in aux else float("nan")
                module_status="PASS" if residual < 1e-5 else "FAIL"
            elif "CN" in key:
                residual=float((P.sum()-aux["effective_precip"].sum()-aux["final_states"]["cn_G"].item()).abs())
                input_used=aux["effective_precip"]
                storage_delta=float(aux["final_states"]["cn_G"].item())
                module_status="PASS" if residual < 1e-4 else "FAIL"
            elif key=="HBV":
                fs=aux["final_states"]; initial=0.5
                delta=sum(float(fs[n].item()) for n in ("SNOWPACK","MELTWATER","SM","SUZ","SLZ"))-initial
                # HBV q and all water-state diagnostics are available, so this is a full finite-window diagnostic.
                evap=float((P*0+0).sum()) # ET is not returned by the active HBV wrapper.
                residual=float(P.sum()-q.sum()-delta)
                input_used=P
                storage_delta=delta
                module_status="UNRESOLVED_ET_DIAGNOSTIC" # ETact is computed in _hbv_step but not exposed.
            elif key=="SIMHYD":
                fs=aux["final_states"]; delta=float(fs["soil"].item()+fs["groundwater"].item()-(0.5*params["simhyd_smsc"].item()))
                residual=float(P.sum()-aux["evap"].sum()-aux["runoff_instant"].sum()-delta)
                input_used=P; storage_delta=delta; module_status="PASS" if abs(residual)<1e-4 else "FAIL"
            else:
                residual=float("nan"); input_used=P; storage_delta=float("nan"); module_status="UNRESOLVED_DIAGNOSTIC_STATES"
            rows.append({"model_key":key,"control_volume":"preprocessing + host core where diagnostics expose all terms","status":module_status,"max_abs_residual":residual,"mean_abs_residual":residual,"relative_residual":abs(residual)/(float(P.sum())+1e-8) if math.isfinite(residual) else float("nan"),"dtype":"float32","sequence_length":8,"routing_tail_included":False,"note":"Finite UH tail is excluded; XAJ/GR4J/HBV diagnostics do not expose all daily ET/storage terms for an exact whole-system balance."})
            details.append({"model_key":key,"q":json_safe(q),"aux_keys":sorted(aux.keys())})
        except Exception as exc:
            one.append({"model_key":key,"status":"FAIL","q_shape":"","finite":False,"max_abs_q":"","evidence":f"{type(exc).__name__}: {exc}"})
            rows.append({"model_key":key,"control_volume":"runtime forward","status":"FAIL","max_abs_residual":"","mean_abs_residual":"","relative_residual":"","dtype":"float32","sequence_length":8,"routing_tail_included":False,"note":f"{type(exc).__name__}: {exc}"})
    write_csv(out/"results"/"s2_mass_balance_results.csv",rows)
    write_csv(out/"results"/"s2_one_step_results.csv",one)
    (out/"results"/"s2_runtime_probe_details.json").write_text(__import__("json").dumps(details,indent=2)+"\n")
    return {"one_step":one,"mass_balance":rows}

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--project-root"); p.add_argument("--output-dir"); a=p.parse_args(); root=project_root_from_args(a.project_root); out=supplement_dir(root,a.output_dir); ensure_dirs(out); probe(root,out)
