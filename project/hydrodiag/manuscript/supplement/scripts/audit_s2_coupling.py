#!/usr/bin/env python3
"""Export the daily coupling order for all active host/structure combinations."""
from __future__ import annotations
import argparse
from pathlib import Path
from s2_audit_utils import ensure_dirs, project_root_from_args, supplement_dir, write_csv

def generate(root: Path,out:Path)->None:
    rows=[]
    for host in ("XAJ","GR4J","SIMHYD"):
        for st in ("Base","PD","TGD","CN"):
            key=host if st=="Base" else f"{host}_{st}"
            pre="P_t is passed directly" if st=="Base" else ("PrecipitationDelay updates a storage using P_t" if st=="PD" else ("TGD updates S and emits effective_precip using P_t,T_t and frozen training T statistics" if st=="TGD" else "CemaNeige partitions P_t using T_t, updates G/eTG and emits rain+melt"))
            rows.append({"host":host,"structure":st,"active_model_key":key,"daily_order":"P,T,PET -> preprocessing -> effective_precip -> host runoff generation -> host routing -> qsim","preprocessing":pre,"host_input":"effective_precip, PET; temperature is not passed to Base host step but is required by wrapper validation","pet_modified":"no","routing":"host-specific finite UH / GR4J UH+route store","training_and_evaluation_path":"same model class and forward implementation; full default, lite explicit","evidence":"models/composed.py:46-53; models/composed_temperature_delay.py:32-110; models/composed_delay.py:66-110; models/*.py forward","status":"VERIFIED_CODE"})
    rows.append({"host":"HBV","structure":"reference","active_model_key":"HBV","daily_order":"P,T,PET -> rain/snow -> snowpack/meltwater -> soil/recharge -> upper/lower zones -> qsim","preprocessing":"HBV internal explicit snow routine; no CN/TGD wrapper","host_input":"P,T,PET","pet_modified":"no","routing":"three linear response components, no external UH","training_and_evaluation_path":"standalone registry class","evidence":"models/hbv.py:13-71, 97-170","status":"VERIFIED_CODE"})
    write_csv(out/"results"/"s2_module_coupling_matrix.csv",rows)
    init=[{"model":"XAJ","initialization":"WU=.5UM, WL=.5LM, WD=.5DM, S=.5SM, FR=QI=QG=0, surface UH buffer=zeros(14)","routing":"finite gamma UH length 15; qi/qg linear reservoirs","evidence":"models/xaj.py:467-493"},{"model":"GR4J","initialization":"Sprod=.5X1, Sroute=.5X3, UH buffers zero","routing":"UH1 length 15, UH2 length 30; X3 routing store","evidence":"models/gr4j.py:144-190"},{"model":"SIMHYD","initialization":"soil=.5*SMSC_safe, groundwater=0, UH buffer=zeros(14)","routing":"finite gamma UH length 15","evidence":"models/simhyd.py:329-353"},{"model":"HBV","initialization":"SNOWPACK=0, MELTWATER=0, SM=.5, SUZ=SLZ=0","routing":"Q0/Q1/Q2 linear response","evidence":"models/hbv.py:145-170"},{"model":"CN/TGD","initialization":"CN G=eTG=0; TGD S=0; wrapper states prefixed with cn_/tgd_","routing":"preprocessing state is carried between daily steps","evidence":"models/cemaneige.py:154-169; models/temperature_delay.py:127-136; models/composed_temperature_delay.py:503-523"}]
    write_csv(out/"results"/"s2_initialization_and_routing.csv",init)
    (out/"results"/"s2_data_flow.mmd").write_text("flowchart LR\n  P[P_t] --> M{Base / PD / TGD / CN}\n  T[T_t] --> M\n  PET[PET_t] --> H[Host runoff generation]\n  M --> E[effective_precip]\n  E --> H\n  H --> R[host routing]\n  R --> Q[qsim]\n  T --> H\n")

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--project-root"); p.add_argument("--output-dir"); a=p.parse_args(); root=project_root_from_args(a.project_root); out=supplement_dir(root,a.output_dir); ensure_dirs(out); generate(root,out)

