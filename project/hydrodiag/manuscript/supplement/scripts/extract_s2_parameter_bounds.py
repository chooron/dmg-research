#!/usr/bin/env python3
"""Extract active 531 parameter bounds and initialization facts from code."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from s2_audit_utils import ensure_dirs, project_root_from_args, supplement_dir, write_csv, write_json

ACTIVE = {
    "XAJ-Base": "XAJ", "XAJ-CN": "XAJ_CN", "XAJ-TGD": "XAJ_TGD",
    "GR4J-Base": "GR4J", "GR4J-CN": "GR4J_CN", "GR4J-TGD": "GR4J_TGD",
    "SIMHYD-Base": "SIMHYD", "SIMHYD-CN": "SIMHYD_CN", "SIMHYD-TGD": "SIMHYD_TGD",
    "HBV-reference": "HBV",
}

SYMBOLS = {
    "xaj_k": "k", "xaj_b": "B", "xaj_im": "IM", "xaj_um": "UM", "xaj_lm": "LM", "xaj_dm": "DM",
    "xaj_c": "C", "xaj_sm": "SM", "xaj_ex": "EX", "xaj_ki": "KI", "xaj_kg": "KG", "xaj_ci": "CI",
    "xaj_cg": "CG", "xaj_a": "a_UH", "xaj_theta": "theta_UH", "cn_ctg": "CTG", "cn_kf": "Kf",
    "tgd_alpha": "alpha", "tgd_tau": "tau", "tgd_beta": "beta", "x1": "X1", "x2": "X2", "x3": "X3", "x4": "X4",
    "gr4j_x1": "X1", "gr4j_x2": "X2", "gr4j_x3": "X3", "gr4j_x4": "X4",
    "simhyd_insc": "INSC", "simhyd_coeff": "COEFF", "simhyd_sq": "SQ", "simhyd_smsc": "SMSC",
    "simhyd_sub": "SUB", "simhyd_crak": "CRAK", "simhyd_k": "K", "simhyd_etmul": "ETMUL",
    "simhyd_a": "a_UH", "simhyd_theta": "theta_UH", "parBETA": "BETA", "parFC": "FC", "parK0": "K0",
    "parK1": "K1", "parK2": "K2", "parLP": "LP", "parPERC": "PERC", "parUZL": "UZL", "parTT": "TT",
    "parCFMAX": "CFMAX", "parCFR": "CFR", "parCWH": "CWH",
}

def source_line(name: str, structure: str) -> str:
    if name.startswith("cn_"): return "models/parameter_specs.py:321-340"
    if name.startswith("tgd_"): return "models/parameter_specs.py:391-419"
    if name.startswith("xaj_"): return "models/parameter_specs.py:163-299"
    if name.startswith("gr4j_"): return "models/parameter_specs.py:421-424 or 495-498"
    if name.startswith("simhyd_"): return "models/parameter_specs.py:432-473"
    if name.startswith("par"): return "models/parameter_specs.py:13-122"
    if structure.startswith("GR4J"): return "models/parameter_specs.py:124-161"
    return "models/parameter_specs.py"

def runtime_details(name: str) -> tuple[str, str, str]:
    if name == "tgd_alpha": return "clamp(alpha,0,1)", "none", "preprocessing"
    if name == "tgd_tau": return "clamp(tau,1e-6,3650); dynamic tau clamped again", "none", "preprocessing"
    if name == "tgd_beta": return "none on beta itself; enters exp(-beta*tanh(...))", "none", "preprocessing"
    if name in {"xaj_a", "simhyd_a"}: return "hydrodl2 uh_gamma: relu(a)+0.1", "none", "routing"
    if name in {"xaj_theta", "simhyd_theta"}: return "hydrodl2 uh_gamma: relu(theta)+0.5", "none", "routing"
    if name == "xaj_ki" or name == "xaj_kg": return "no individual clamp; joint rescaling when KI+KG>=1", "KI+KG<1 unchanged; otherwise multiply both by (1-1e-5)/max(KI+KG,1e-6)", "host runoff/routing"
    if name == "x4" or name == "gr4j_x4": return "compute_gr4j_uh_ordinates: max(X4,1e-3)", "none", "routing"
    if name == "simhyd_insc": return "max(INSC,1e-6)", "none", "host runoff"
    if name == "simhyd_coeff": return "max(COEFF,1e-6)", "none", "host runoff"
    if name == "simhyd_smsc": return "max(SMSC,1e-6)", "none", "host runoff"
    if name == "simhyd_k": return "clamp(K,0,1)", "none", "host runoff"
    return "none on parameter; derived expressions have state/flux min/max", "none", "host runoff/routing"

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--project-root"); p.add_argument("--output-dir"); a=p.parse_args()
    root=project_root_from_args(a.project_root); out=supplement_dir(root,a.output_dir); ensure_dirs(out); sys.path.insert(0,str(root))
    from models.parameter_specs import (HBV_PARAM_SPECS, XAJ_PARAM_SPECS, XAJ_CN_PARAM_SPECS, XAJ_TGD_PARAM_SPECS, GR4J_PARAM_SPECS, GR4J_CN_PARAM_SPECS, GR4J_TGD_PARAM_SPECS, SIMHYD_PARAM_SPECS, SIMHYD_CN_PARAM_SPECS, SIMHYD_TGD_PARAM_SPECS)
    specs={"XAJ":XAJ_PARAM_SPECS,"XAJ_CN":XAJ_CN_PARAM_SPECS,"XAJ_TGD":XAJ_TGD_PARAM_SPECS,"GR4J":GR4J_PARAM_SPECS,"GR4J_CN":GR4J_CN_PARAM_SPECS,"GR4J_TGD":GR4J_TGD_PARAM_SPECS,"SIMHYD":SIMHYD_PARAM_SPECS,"SIMHYD_CN":SIMHYD_CN_PARAM_SPECS,"SIMHYD_TGD":SIMHYD_TGD_PARAM_SPECS,"HBV":HBV_PARAM_SPECS}
    rows=[]
    for panel,key in ACTIVE.items():
        for order,(name,item) in enumerate(specs[key].items(),1):
            clamp,joint,scope=runtime_details(name)
            transform="log interpolation" if name=="tgd_tau" else "linear normalized-to-physical"
            rows.append({"model_or_module":panel,"active_model_key":key,"active_order":order,"code_name":name,"symbol":SYMBOLS.get(name,name),"source_file":"models/parameter_specs.py","source_line":source_line(name,panel),"lower_bound":item["lower"],"upper_bound":item["upper"],"default_value":item.get("default","UNRESOLVED"),"transform":transform,"runtime_clamp":clamp,"joint_constraint":joint,"used_by":"Base" if panel.endswith("Base") else "CN" if panel.endswith("CN") else "TGD" if panel.endswith("TGD") else "HBV reference","model_scope":scope,"basin_scope":"basin-specific physical value at model call; dPL network weights shared","evidence_status":"CODE_VERIFIED"})
    write_csv(out/"results/s2_parameter_bounds_from_code.csv",rows)
    init=[
      {"model_or_module":"XAJ","state":"WU,WL,WD,S,FR,QI,QG,rs_uh_buffer","initial_value":"0.6*UM,0.6*LM,0.6*DM,0.5*SM,0.1,0.1,0.1,zeros(14)","parameter_default_rule":"parameter defaults are in parameter_specs; dPL head bias is initialized to normalized defaults","evidence":"models/xaj.py:467-501; training/dpl/run_dpl_model.py:148-164"},
      {"model_or_module":"GR4J","state":"s_prod,s_route,uh1_buf,uh2_buf","initial_value":"0.5*X1,0.5*X3,zeros(15),zeros(30)","parameter_default_rule":"parameter defaults are in parameter_specs","evidence":"models/gr4j.py:169-190"},
      {"model_or_module":"SIMHYD","state":"soil,groundwater,runoff_uh_buffer","initial_value":"0.5*max(SMSC,1e-6),0,zeros(14)","parameter_default_rule":"parameter defaults are in parameter_specs","evidence":"models/simhyd.py:329-353"},
      {"model_or_module":"HBV","state":"SNOWPACK,MELTWATER,SM,SUZ,SLZ","initial_value":"0,0,0.5,0,0","parameter_default_rule":"parameter defaults are in parameter_specs","evidence":"models/hbv.py:145-170"},
      {"model_or_module":"CN","state":"G,eTG","initial_value":"0,0","parameter_default_rule":"CTG=0.5,Kf=3.0 defaults in parameter_specs","evidence":"models/cemaneige.py:154-169; models/parameter_specs.py:321-340"},
      {"model_or_module":"TGD","state":"S","initial_value":"0","parameter_default_rule":"alpha=.5,tau=5,beta=0 defaults in parameter_specs; tgd_tau normalized default uses logarithmic interpolation","evidence":"models/temperature_delay.py:127-136; models/parameter_specs.py:391-419; training/dpl/run_dpl_model.py:148-164"},
      {"model_or_module":"Gamma UH","state":"UH weights","initial_value":"computed from parameter tensors; hydrodl2 uses relu(a)+0.1, relu(theta)+0.5, t=0.5,... and normalizes sum to 1","parameter_default_rule":"host parameter vector supplies routing parameters","evidence":"models/xaj.py:644-666; /home/jingxin/code/dmg-research/.venv/lib/python3.10/site-packages/hydrodl2/core/calc/uh_routing.py:5-22"},
    ]
    write_csv(out/"results/s2_parameter_initialization.csv",init)
    summary={"active_parameter_counts":{"XAJ":len(XAJ_PARAM_SPECS),"GR4J":len(GR4J_PARAM_SPECS),"SIMHYD":len(SIMHYD_PARAM_SPECS),"HBV":len(HBV_PARAM_SPECS),"CN":2,"TGD":3},"active_model_keys":list(ACTIVE.values()),"full_default_evidence":["ablation/ic_core/model_adapter.py:31-45","training/dpl/run_dpl_model.py:627-629"],"excluded_from_this_audit":["PD","GD","legacy 559"],"code_bound_rows":len(rows),"status":"CODE_BOUNDS_EXTRACTED","notes":["XAJ has 15 parameters including xaj_a and xaj_theta.","SIMHYD has 10 active parameters including ETMUL and Gamma UH a/theta; a 9-parameter expectation conflicts with the active spec.","Gamma UH shape/scale effective values differ from calibration bounds because hydrodl2 applies positive offsets.","Runtime clamps are not calibration bounds."],"evidence_status_counts":{"CODE_VERIFIED":len(rows)}}
    write_json(out/"results/s2_parameter_audit.json",summary)

if __name__ == "__main__": main()
