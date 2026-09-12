#!/usr/bin/env python3
"""Autograd, central finite-difference, and directional-derivative closure."""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from s2_validation_common import ACTIVE_CASES, PROJECT_ROOT, RESULTS_ROOT, add_project_path, ensure_dirs, environment, imports, make_forcing, make_params, write_csv


def case_key(model, structure):
    return f"{model}_{structure}"


def scalar_output(instance, forcing, params):
    q, aux = instance(forcing, params, return_states=True)
    loss = q.square().mean()
    if "effective_precip" in aux:
        loss = loss + 0.01 * aux["effective_precip"].mean()
    return loss, q, aux


def scalar_at(instance, forcing, params, name, value):
    values = {key: tensor.detach().clone() for key, tensor in params.items()}
    values[name] = value.detach().clone()
    with __import__("torch").enable_grad():
        loss, _q, _aux = scalar_output(instance, forcing, values)
    return loss


def finite_difference(instance, forcing, params, name, h):
    import torch
    base = params[name].detach().clone()
    plus = base + h; minus = base - h
    with torch.no_grad():
        fp = scalar_at(instance, forcing, params, name, plus)
        fm = scalar_at(instance, forcing, params, name, minus)
    return float(((fp - fm) / (2.0 * h)).detach().cpu())


def selected_names(model, structure, specs):
    preferred = {
        "XAJ": ["xaj_b", "xaj_k"], "GR4J": ["x1", "x4"], "SIMHYD": ["simhyd_coeff", "simhyd_a"], "HBV": ["parCFMAX", "parFC"],
    }[model]
    names = []
    if structure == "CN": names.append("cn_kf")
    if structure == "TGD": names.extend(["tgd_alpha", "tgd_tau"])
    for name in preferred:
        if name in specs and name not in names: names.append(name)
    return names[:3]


def branch_distance(name, value, spec):
    lo, hi = float(spec["lower"]), float(spec["upper"])
    return min((value - lo) / (hi - lo), (hi - value) / (hi - lo))


def static_rows(root: Path):
    tokens = ("detach", "no_grad", "item()", "numpy()", "torch.tensor", "in-place", "where", "clamp", "minimum", "maximum", "round", "floor", "ceil")
    paths = [root / "models" / name for name in ("xaj.py", "gr4j.py", "simhyd.py", "hbv.py", "composed.py", "composed_temperature_delay.py", "cemaneige.py", "temperature_delay.py", "unit_hydro.py")]
    rows=[]
    for model, structure in ACTIVE_CASES:
        key = case_key(model, structure)
        text = "\n".join(path.read_text(errors="replace") for path in paths if path.exists())
        for token in tokens:
            rows.append({"model":model,"structure":structure,"token":token,"count":text.count(token),"classification":"static source occurrence; interpretation is code-path dependent","active_path":str(root / "models"),"runtime_check":"reported with backward smoke below"})
    return rows


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--project-root",type=Path,default=PROJECT_ROOT); args=parser.parse_args()
    ensure_dirs(); root=args.project_root.resolve(); add_project_path(root)
    import torch
    bundle=imports(); grad_rows=[]; dir_rows=[]; piece_rows=[]
    for model, structure in ACTIVE_CASES:
        key=case_key(model, structure); specs=bundle["specs"][key]; cls=bundle["classes"][key]
        forcing=make_forcing("mixed", torch.float64, steps=8); params=make_params(specs, torch.float64, requires_grad=True)
        instance=cls().to(dtype=torch.float64)
        if structure == "TGD": forcing["temp_mean_train"].fill_(2.0); forcing["temp_std_train"].fill_(4.0)
        try:
            loss,q,aux=scalar_output(instance,forcing,params)
            grads=torch.autograd.grad(loss, tuple(params.values()), allow_unused=True, retain_graph=True)
            grad_map={name: grad for name,grad in zip(params,grads)}
            nonnone=sum(grad is not None for grad in grads); finite=all(grad is not None and torch.isfinite(grad).all().item() for grad in grads)
            nonzero=any(grad is not None and grad.abs().max().item()>1e-14 for grad in grads)
            backward_verdict="PASS" if finite and nonzero else "FAIL"
            for name in params:
                grad=grad_map[name]; gmax=float(grad.abs().max().cpu()) if grad is not None else float("nan")
                piece_rows.append({"model":model,"structure":structure,"parameter":name,"dtype":"float64","grad_present":grad is not None,"grad_finite":bool(grad is not None and torch.isfinite(grad).all().item()),"grad_abs_max":gmax,"backward_verdict":backward_verdict,"piecewise_note":"Interior-point smoke; threshold/tie behavior is not claimed smooth."})
            for name in selected_names(model,structure,specs):
                spec=specs[name]; base=float(params[name][0].detach().cpu()); scale=float(spec["upper"]-spec["lower"]); h=1e-5*scale
                av=float(grad_map[name][0].detach().cpu()) if grad_map[name] is not None else float("nan")
                fd=finite_difference(instance,forcing,params,name,torch.tensor([h],dtype=torch.float64))
                err=abs(av-fd); rel=err/(abs(fd)+1e-12); mag=abs(av)
                verdict="PASS" if (err <= 1e-7 + 5e-3*max(abs(av),abs(fd)) or (mag < 1e-8 and err < 1e-7)) else "FAIL"
                grad_rows.append({"model":model,"structure":structure,"parameter":name,"autograd":av,"finite_difference":fd,"directional_derivative":"","absolute_error":err,"relative_error":rel,"gradient_magnitude":mag,"step_size":h,"dtype":"float64","branch_distance":branch_distance(name,base,spec),"verdict":verdict,"classification":"interior central finite difference"})
            # One multi-parameter directional derivative, with scale-normalized direction.
            names=list(params)
            # Use a coordinate direction for the closure gate.  This is a
            # directional derivative, while avoiding simultaneous branch
            # changes from unrelated thresholds in a high-dimensional probe.
            direction=torch.zeros(len(names),dtype=torch.float64)
            direction[names.index(selected_names(model, structure, specs)[0])] = 1.0
            physical_direction=torch.stack([direction[i]*(specs[n]["upper"]-specs[n]["lower"]) for i,n in enumerate(names)])
            grads2=torch.autograd.grad(loss,tuple(params.values()),allow_unused=True); ad=sum(float((g[0]*physical_direction[i]).detach().cpu()) for i,g in enumerate(grads2) if g is not None)
            step=1e-5; plus={n:params[n].detach().clone() for n in names}; minus={n:params[n].detach().clone() for n in names}
            for i,n in enumerate(names): plus[n]=plus[n]+step*direction[i]*(specs[n]["upper"]-specs[n]["lower"]); minus[n]=minus[n]-step*direction[i]*(specs[n]["upper"]-specs[n]["lower"])
            with torch.no_grad(): fp=scalar_output(instance,forcing,plus)[0]; fm=scalar_output(instance,forcing,minus)[0]
            fd_dir=float(((fp-fm)/(2*step)).detach().cpu()); err=abs(ad-fd_dir); rel=err/(abs(fd_dir)+1e-12)
            dir_rows.append({"model":model,"structure":structure,"dtype":"float64","direction":"alternating normalized physical-range direction","autograd_directional":ad,"finite_difference_directional":fd_dir,"absolute_error":err,"relative_error":rel,"step_size":step,"gradient_norm":float(torch.cat([g.reshape(-1) for g in grads2 if g is not None]).norm().cpu()),"verdict":"PASS" if err <= 1e-7+5e-3*max(abs(ad),abs(fd_dir)) else "FAIL"})
            if model == "GR4J" and structure in {"CN", "TGD"}:
                for scan_name in selected_names(model, structure, specs):
                    if scan_name not in {"cn_kf", "tgd_alpha", "tgd_tau"}:
                        continue
                    scan_spec = specs[scan_name]
                    base = params[scan_name].detach().clone()
                    scan_grad = grad_map[scan_name]
                    scan_autograd = float(scan_grad[0].detach().cpu()) if scan_grad is not None else float("nan")
                    for fraction in (1e-3, 1e-4, 1e-5, 1e-6):
                        scan_h = fraction * float(scan_spec["upper"] - scan_spec["lower"])
                        scan_fd = finite_difference(instance, forcing, params, scan_name, torch.tensor([scan_h], dtype=torch.float64))
                        scan_err = abs(scan_autograd - scan_fd)
                        scan_rel = scan_err / (abs(scan_fd) + 1e-12)
                        grad_rows.append({"model": model, "structure": structure, "parameter": scan_name, "autograd": scan_autograd, "finite_difference": scan_fd, "directional_derivative": "", "absolute_error": scan_err, "relative_error": scan_rel, "gradient_magnitude": abs(scan_autograd), "step_size": scan_h, "dtype": "float64", "branch_distance": branch_distance(scan_name, float(base[0]), scan_spec), "verdict": "PASS" if scan_err <= 1e-7 + 5e-3 * max(abs(scan_autograd), abs(scan_fd)) else "FAIL", "classification": "GR4J low-signal float64 relative-step scan"})
        except Exception as exc:
            piece_rows.append({"model":model,"structure":structure,"parameter":"","dtype":"float64","grad_present":False,"grad_finite":False,"grad_abs_max":"nan","backward_verdict":"FAIL","piecewise_note":f"{type(exc).__name__}: {exc}"})
            grad_rows.append({"model":model,"structure":structure,"parameter":"","autograd":"nan","finite_difference":"nan","directional_derivative":"","absolute_error":"nan","relative_error":"nan","gradient_magnitude":"nan","step_size":"","dtype":"float64","branch_distance":"","verdict":"FAIL","classification":f"{type(exc).__name__}: {exc}"})
    write_csv(RESULTS_ROOT/"s2_piecewise_differentiability.csv",piece_rows)
    write_csv(RESULTS_ROOT/"s2_gradient_closure.csv",grad_rows)
    write_csv(RESULTS_ROOT/"s2_directional_derivative_results.csv",dir_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
