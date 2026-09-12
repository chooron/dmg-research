#!/usr/bin/env python3
"""Run representative autograd versus finite-difference checks on the active full path."""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path
from s2_audit_utils import ensure_dirs, project_root_from_args, supplement_dir, write_csv

def run(root:Path,out:Path)->None:
    # Calling the unwrapped step/fused-step functions is intentional.  These
    # are the exact functions passed to torch.compile by the active classes,
    # so this checks the dPL differentiable kernel without compiling every
    # finite-difference replicate.
    sys.path.insert(0,str(root)); import torch
    from models.xaj import _prepare_xaj_parameters, _xaj_step
    from models.gr4j import _gr4j_step
    from models.simhyd import _simhyd_step
    from models.unit_hydro import compute_gr4j_uh_ordinates
    from models.cemaneige import _cemaneige_step, _estimate_psol_annual
    from models.temperature_delay import _temperature_conditioned_delay_step
    from models.composed import _cemaneige_xaj_fused_step, _cemaneige_gr4j_fused_step, _cemaneige_simhyd_fused_step
    from models.composed_temperature_delay import _tgd_xaj_fused_step, _tgd_gr4j_fused_step, _tgd_simhyd_fused_step
    from models.parameter_specs import (XAJ_PARAM_SPECS,XAJ_CN_PARAM_SPECS,XAJ_TGD_PARAM_SPECS,GR4J_PARAM_SPECS,GR4J_CN_PARAM_SPECS,GR4J_TGD_PARAM_SPECS,SIMHYD_PARAM_SPECS,SIMHYD_CN_PARAM_SPECS,SIMHYD_TGD_PARAM_SPECS)
    specs_map={"XAJ":XAJ_PARAM_SPECS,"XAJ_CN":XAJ_CN_PARAM_SPECS,"XAJ_TGD":XAJ_TGD_PARAM_SPECS,"GR4J":GR4J_PARAM_SPECS,"GR4J_CN":GR4J_CN_PARAM_SPECS,"GR4J_TGD":GR4J_TGD_PARAM_SPECS,"SIMHYD":SIMHYD_PARAM_SPECS,"SIMHYD_CN":SIMHYD_CN_PARAM_SPECS,"SIMHYD_TGD":SIMHYD_TGD_PARAM_SPECS}
    P=torch.tensor([20.0],dtype=torch.float32); T=torch.tensor([1.2],dtype=torch.float32); PET=torch.tensor([1.7],dtype=torch.float32)
    def value(params,key,host):
        return params[key if key in params else f"{host.lower()}_{key}"]
    def evaluate(key, params):
        host=key.split("_")[0]
        hparams={k:v for k,v in params.items() if k.startswith(host.lower()+"_") or (host=="XAJ" and k.startswith("xaj_")) or (host=="SIMHYD" and k.startswith("simhyd_")) or (host=="GR4J" and k in {"x1","x2","x3","x4"})}
        if host=="XAJ":
            # XAJ parameter keys are xaj_* for all wrappers.
            xs={k:v for k,v in params.items() if k.startswith("xaj_")}
            pp=_prepare_xaj_parameters(xs); state=(0.5*pp[3],0.5*pp[4],0.5*pp[5],0.5*pp[7],torch.tensor([0.1]),torch.tensor([0.2]),torch.tensor([0.3]))
            if key.endswith("CN"):
                gth=0.9*_estimate_psol_annual(P.unsqueeze(0),T.unsqueeze(0)); out=_cemaneige_xaj_fused_step(P,T,PET,(torch.zeros(1),torch.zeros(1)),state,(params["cn_ctg"],params["cn_kf"],gth),pp[:-2],1e-8); return out[6]
            if key.endswith("TGD"):
                out=_tgd_xaj_fused_step(P,T,PET,torch.zeros(1),state,(params["tgd_alpha"],params["tgd_tau"],params["tgd_beta"]),pp[:-2],(torch.tensor([1.0]),torch.tensor([4.0])),1e-8); return out[5]
            return _xaj_step(P,PET,*state,*pp[:-2],1e-8)[0]
        if host=="GR4J":
            xs={k[5:] if k.startswith("gr4j_") else k:v for k,v in params.items() if k.startswith("gr4j_") or k in {"x1","x2","x3","x4"}}
            uh1,uh2=compute_gr4j_uh_ordinates(xs["x4"],15); uh2=compute_gr4j_uh_ordinates(xs["x4"],30)[1]; state=(0.5*xs["x1"],0.5*xs["x3"],torch.zeros(1,15),torch.zeros(1,30)); gp=(uh1,uh2,xs["x1"],xs["x2"],xs["x3"])
            if key.endswith("CN"):
                gth=0.9*_estimate_psol_annual(P.unsqueeze(0),T.unsqueeze(0)); out=_cemaneige_gr4j_fused_step(P,T,PET,(torch.zeros(1),torch.zeros(1)),state,(params["cn_ctg"],params["cn_kf"],gth),gp,1e-8); return out[6]
            if key.endswith("TGD"):
                out=_tgd_gr4j_fused_step(P,T,PET,torch.zeros(1),state,(params["tgd_alpha"],params["tgd_tau"],params["tgd_beta"]),gp,(torch.tensor([1.0]),torch.tensor([4.0])),1e-8); return out[5]
            return _gr4j_step(P,PET,*state,*gp,1e-8)[0]
        xs={k:v for k,v in params.items() if k.startswith("simhyd_")}; state=(0.5*xs["simhyd_smsc"],torch.tensor([0.3]))
        sp=(xs["simhyd_insc"],xs["simhyd_coeff"],xs["simhyd_sq"],xs["simhyd_smsc"],xs["simhyd_sub"],xs["simhyd_crak"],xs["simhyd_k"],xs["simhyd_etmul"])
        if key.endswith("CN"):
            gth=0.9*_estimate_psol_annual(P.unsqueeze(0),T.unsqueeze(0)); out=_cemaneige_simhyd_fused_step(P,T,PET,(torch.zeros(1),torch.zeros(1)),state,(params["cn_ctg"],params["cn_kf"],gth),sp,1e-8); return out[6]
        if key.endswith("TGD"):
            out=_tgd_simhyd_fused_step(P,T,PET,torch.zeros(1),state,(params["tgd_alpha"],params["tgd_tau"],params["tgd_beta"]),sp,(torch.tensor([1.0]),torch.tensor([4.0])),1e-8); return out[5]
        return _simhyd_step(P,PET,*state,*sp,1e-8)[0]
    rows=[]
    for key,specs in specs_map.items():
        names=list(specs); vals=[]
        for n,s in specs.items(): vals.append(math.sqrt(s["lower"]*s["upper"]) if n=="tgd_tau" else (s["lower"]+s["upper"])/2)
        params={n:torch.tensor([v],dtype=torch.float32,requires_grad=True) for n,v in zip(names,vals)}
        if key.startswith("GR4J"):
            # A one-day UH probe at the midpoint x4 has an almost zero first
            # ordinate in float32.  Use an interior, non-boundary x4 to make
            # the one-step derivative observable without changing the model.
            x4_name = "gr4j_x4" if "gr4j_x4" in params else "x4"
            params[x4_name] = torch.tensor([1.5], dtype=torch.float32, requires_grad=True)
        forc={"precip":P,"temp":T,"pet":PET}
        if "TGD" in key: forc.update({"temp_mean_train":torch.tensor([1.0]),"temp_std_train":torch.tensor([4.0])})
        try:
            q=evaluate(key,params)
            selected=[n for n in names if n in {"xaj_b","x1","gr4j_x1","simhyd_smsc","cn_kf","tgd_alpha","tgd_tau"}][:2]
            for n in selected:
                g=torch.autograd.grad(q.sum(),params[n],retain_graph=True,allow_unused=True)[0]
                base=params[n].detach().clone(); h=torch.tensor(1e-3,dtype=torch.float32)
                plus={k:v.detach().clone() for k,v in params.items()}; minus={k:v.detach().clone() for k,v in params.items()}; plus[n]=base+h; minus[n]=base-h
                qp=evaluate(key,plus); qm=evaluate(key,minus)
                fd=(qp.sum()-qm.sum())/(2*h); av=float(g.item()) if g is not None else float("nan"); fv=float(fd.item()); err=abs(av-fv); rel=err/(abs(fv)+1e-6)
                rows.append({"model_key":key,"parameter":n,"autograd":av,"finite_difference":fv,"absolute_error":err,"relative_error":rel,"status":"PASS" if (rel<5e-2 or err<1e-5) else "FAIL","dtype":"float32","note":"Representative interior point; finite UH is included in q and threshold branch points avoided. Absolute tolerance is used when float32 finite differences are near zero."})
        except Exception as exc:
            rows.append({"model_key":key,"parameter":"","autograd":"","finite_difference":"","absolute_error":"","relative_error":"","status":"FAIL","dtype":"float32","note":f"{type(exc).__name__}: {exc}"})
    write_csv(out/"results"/"s2_gradient_check_results.csv",rows)

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--project-root"); p.add_argument("--output-dir"); a=p.parse_args(); root=project_root_from_args(a.project_root); out=supplement_dir(root,a.output_dir); ensure_dirs(out); run(root,out)
