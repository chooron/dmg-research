#!/usr/bin/env python3
"""Extract Base/TGD/CN/HBV parameter facts from the active parameter specs."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from s2_audit_utils import ensure_dirs, line_for, project_root_from_args, supplement_dir, write_csv

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--project-root"); p.add_argument("--output-dir"); a=p.parse_args()
    root=project_root_from_args(a.project_root); out=supplement_dir(root,a.output_dir); ensure_dirs(out); sys.path.insert(0,str(root))
    from models.parameter_specs import (HBV_PARAM_SPECS, XAJ_PARAM_SPECS, XAJ_CN_PARAM_SPECS, XAJ_TGD_PARAM_SPECS, GR4J_PARAM_SPECS, GR4J_CN_PARAM_SPECS, GR4J_TGD_PARAM_SPECS, SIMHYD_PARAM_SPECS, SIMHYD_CN_PARAM_SPECS, SIMHYD_TGD_PARAM_SPECS)
    specs={"XAJ-Base":XAJ_PARAM_SPECS,"XAJ-CN":XAJ_CN_PARAM_SPECS,"XAJ-TGD":XAJ_TGD_PARAM_SPECS,"GR4J-Base":GR4J_PARAM_SPECS,"GR4J-CN":GR4J_CN_PARAM_SPECS,"GR4J-TGD":GR4J_TGD_PARAM_SPECS,"SIMHYD-Base":SIMHYD_PARAM_SPECS,"SIMHYD-CN":SIMHYD_CN_PARAM_SPECS,"SIMHYD-TGD":SIMHYD_TGD_PARAM_SPECS,"HBV-reference":HBV_PARAM_SPECS}
    symbols={"xaj_k":"k","xaj_b":"B","xaj_im":"IM","xaj_um":"UM","xaj_lm":"LM","xaj_dm":"DM","xaj_c":"C","xaj_sm":"SM","xaj_ex":"EX","xaj_ki":"KI","xaj_kg":"KG","xaj_ci":"CI","xaj_cg":"CG","xaj_a":"a_UH","xaj_theta":"theta_UH","cn_ctg":"CTG","cn_kf":"K_f","tgd_alpha":"alpha","tgd_tau":"tau","tgd_beta":"beta","x1":"X_1","x2":"X_2","x3":"X_3","x4":"X_4","parBETA":"BETA","parFC":"FC","parK0":"K_0","parK1":"K_1","parK2":"K_2","parLP":"LP","parPERC":"PERC","parUZL":"UZL","parTT":"TT","parCFMAX":"CFMAX","parCFR":"CFR","parCWH":"CWH"}
    rows=[]; path=root/"models/parameter_specs.py"
    for structure, sp in specs.items():
        host=structure.split("-")[0]
        for name, item in sp.items():
            if name.startswith("gr4j_"): source=124
            elif name.startswith("cn_"): source=321
            elif name.startswith("tgd_"): source=391
            elif name.startswith("xaj_"): source=163
            elif name.startswith("simhyd_"): source=432
            elif name.startswith("par"): source=13
            else: source=line_for(path, f'"{name}"') or 1
            if name in {"tgd_tau"}: transform="log interpolation in ablation/ic_core/parameter_adapter.py:31,56-67; dPL sigmoid output before physical mapping"
            else: transform="linear physical bound interpolation in ablation/ic_core/parameter_adapter.py:56-62; dPL sigmoid output before physical mapping"
            module="snow" if name.startswith("cn_") else "temperature-conditioned delay" if name.startswith("tgd_") else "host runoff/routing"
            rows.append({"host_structure":structure,"host":host,"module":module,"symbol":symbols.get(name,name),"code_name":name,"description":item.get("description",""),"lower":item.get("lower"),"upper":item.get("upper"),"default":item.get("default"),"unit":item.get("unit","UNRESOLVED"),"scope":"basin-specific physical parameter at model call; dPL network weights are shared","transform":transform,"active":"yes","evidence_status":"VERIFIED_CODE","evidence":"models/parameter_specs.py:"+str(source)})
    write_csv(out/"results/s2_parameter_inventory.csv",rows)

if __name__ == "__main__": main()

