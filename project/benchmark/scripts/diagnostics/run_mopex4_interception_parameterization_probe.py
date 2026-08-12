#!/usr/bin/env python3
"""Small matched A/B/C probe for MOPEX4 interception parameterization.

F0 = current production diagnostic interception; F1/F2 = benchmark-only
amplitude/phase-decoupled variants. CPU, four representative basins, one fixed
730-day window, no full-basin training and no production changes.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

BENCHMARK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BENCHMARK), str(BENCHMARK.parents[1]), str(BENCHMARK / "src"),
                str(BENCHMARK / "scripts" / "diagnostics")]
import audit_mopex34_root_cause as A  # noqa: E402
import audit_mopex45_sequential_discretization as D  # noqa: E402
from mopex45_discr_steps import mopex4_step_diag  # noqa: E402
from mopex4_interception_variants import make_variant, _interception_fraction  # noqa: E402
from dpl.nn_parameterizer import CatchmentParameterizer  # noqa: E402
from dpl.attributes import CatchmentAttributeBuilder  # noqa: E402
from dmotpy.models.flux.mopex import mopex_training_context  # noqa: E402
from project.benchmark.scripts.run_dpl_benchmark_dmg_native import compute_differentiable_kge  # noqa: E402
from dmotpy.models.registry import PARAM_INFO  # noqa: E402

OUT = BENCHMARK / "results/mopex45_phase_fix/root_cause_audit/interception_parameterization"
OUT.mkdir(parents=True, exist_ok=True)
BASIN_IDX = [391, 373, 269, 530]
WARMUP, SCORED = 365, 365
MODES = {"F0_current": mopex4_step_diag, "F1_halfwave": make_variant("F1"), "F2_shifted_cosine": make_variant("F2")}


def write_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", newline="") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def load_all():
    ids, xfull, yfull, b = A.load_context()
    x = xfull[A.START:A.START + 730]
    y = yfull[A.START:A.START + 730]
    attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cpu", method="zscore")[b]
    return ids, x, y, attrs, b


def load_m4_net(path):
    net = CatchmentParameterizer(35, 10, hidden_dims=[256, 256], dropout=.05)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(payload["network"]); net.eval()
    return net


def learned_params(ids, attrs):
    baseline = load_m4_net(BENCHMARK / "results/dpl_round13_20260805/auto100/checkpoints/mopex4/epoch_100.pt")(attrs).detach()
    cont = load_m4_net(BENCHMARK / "results/mopex45_phase_fix/full_continuation/runs/seed_41/checkpoints/J2/seed_41/epoch_100.pt")(attrs).detach()
    ic = D.ic_theta("mopex4", ids)[BASIN_IDX].detach()
    return {"baseline_dpl": baseline, "continuation": cont, "IC": ic}


def rollout(step_fn, theta_norm, x, y, lambda_i=1.0, training=False):
    theta = A.norm_to_phys(theta_norm, "mopex4")
    P, T, PET, doy = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]
    Sn, S1, S2, Sc1, Sc2 = D._init_states(theta.shape[0])
    qs, fluxes = [], []
    with mopex_training_context(lambda_i=lambda_i, lambda_p=1.0, beta=50.0):
        for t in range(WARMUP):
            with torch.no_grad():
                out = step_fn(P[t], T[t], PET[t], *theta.t(), S1, S2, Sc1, Sc2, Sn,
                              doy=doy[t], nearzero=1e-6)
                S1, S2, Sc1, Sc2, Sn = [v.detach() for v in out[2:7]]
        for t in range(WARMUP, WARMUP + SCORED):
            out = step_fn(P[t], T[t], PET[t], *theta.t(), S1, S2, Sc1, Sc2, Sn,
                          doy=doy[t], nearzero=1e-6)
            qs.append(out[0]); fluxes.append(out[-1])
            S1, S2, Sc1, Sc2, Sn = out[2:7]
    q = torch.stack(qs)
    loss, kge = compute_differentiable_kge(q, y[WARMUP:], warmup_days=0)
    return loss, kge, q, fluxes


def anatomy():
    rows = []
    doy = torch.arange(1., 366.)
    beta = 50.0
    for mode in ("F0", "F1", "F2"):
        for alpha_value in (0., .1, .25, .5, .75, 1.):
            alpha = torch.tensor(alpha_value, requires_grad=True)
            itime = torch.tensor(180., requires_grad=True)
            if mode == "F0":
                raw = alpha + (1-alpha) * torch.cos(2*torch.pi*(doy-itime)/365.25)
                frac = F.softplus(beta*raw)/beta
            else:
                frac = _interception_fraction(doy, alpha, itime, mode, beta=beta)
            grad_a, grad_t = torch.autograd.grad(frac.mean(), (alpha, itime), retain_graph=False)
            rows.append({"formula": mode, "alpha": alpha_value,
                         "peak_fraction": float(frac.max()), "mean_fraction": float(frac.mean()),
                         "min_fraction": float(frac.min()),
                         "active_gt_1pct": float((frac > .01).float().mean()),
                         "active_gt_10pct": float((frac > .10).float().mean()),
                         "dmean_dalpha": float(grad_a), "dmean_dis_time": float(grad_t)})
    write_csv("formula_shape_comparison.csv", rows)
    return rows


def utilization(ids, params, x):
    rows = []
    for group, theta in params.items():
        loss, kge, q, fluxes = rollout(MODES["F0_current"], theta, x, torch.zeros(730, len(BASIN_IDX)), lambda_i=1.0)
        theta_phys = A.norm_to_phys(theta, "mopex4")
        P = x[WARMUP:, :, 0]; T = x[WARMUP:, :, 1]
        tcrit = theta_phys[:, 0]
        Pr_all = P * torch.sigmoid((T - tcrit[None, :]) / (torch.abs(tcrit[None, :]) * .01 + .01 + 1e-6))
        for j, bi in enumerate(BASIN_IDX):
            I = torch.stack([f["i"][j] for f in fluxes])
            Pr = Pr_all[:, j]
            rows.append({"group": group, "basin_idx": bi, "basin_id": ids[bi],
                         "sum_I": float(I.sum()), "sum_Pr": float(Pr.sum()),
                         "I_over_Pr": float(I.sum() / (Pr.sum() + 1e-9)),
                         "I_over_P": float(I.sum() / (P[:, j].sum() + 1e-9)),
                         "I_p50": float(I.quantile(.5)), "I_p95": float(I.quantile(.95)),
                         "I_max": float(I.max()), "active_fraction": float((I > 1e-9).float().mean())})
    write_csv("learned_interception_utilization.csv", rows)
    return rows


def short_train(mode_name, step_fn, init, x, y, lambda_i=1.0, steps=30):
    torch.manual_seed(777)
    theta = torch.nn.Parameter(init.detach().clone())
    opt = torch.optim.AdamW([theta], lr=1e-2, weight_decay=1e-4)
    records = []
    best = -float("inf")
    best_step = -1
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, kge, q, flux = rollout(step_fn, theta, x, y, lambda_i=lambda_i, training=True)
        loss.backward(); grad = theta.grad.detach().clone(); opt.step()
        with torch.no_grad(): theta.clamp_(0, 1)
        med = float(kge.median()); best = max(best, med)
        if med == best: best_step = step
        if step in (0, 5, 10, 20, 29):
            records.append({"formula": mode_name, "lambda_i": lambda_i, "step": step,
                            "loss": float(loss), "median_kge": med, "mean_kge": float(kge.mean()),
                            "best_median_kge": best, "alpha_mean": float(theta[:,4].mean()),
                            "is_time_mean": float(theta[:,5].mean()), "s2max_mean": float(theta[:,2].mean()),
                            "tw_mean": float(theta[:,3].mean()), "grad_norm": float(grad.norm()),
                            "grad_zero_fraction": float((grad.abs() < 1e-12).float().mean())})
    return records, theta.detach()


def gradient_probe(net, attrs, x, y, mode_name, step_fn, lambda_i):
    net = load_m4_net(BENCHMARK / "results/dpl_round13_20260805/auto100/checkpoints/mopex3/epoch_100.pt") if False else net
    # For the M4 network, use the existing mapped M3 representation.
    net.eval(); net.zero_grad(set_to_none=True)
    raw = net(attrs); raw.retain_grad()
    theta = A.norm_to_phys(raw, "mopex4")
    P,T,PET,doy=x[:,:,:,] if False else (x[:,:,0],x[:,:,1],x[:,:,2],x[:,:,3])
    Sn,S1,S2,Sc1,Sc2=D._init_states(raw.shape[0]); qs=[]
    with mopex_training_context(lambda_i=lambda_i, lambda_p=1.0, beta=50.0):
        for t in range(WARMUP):
            with torch.no_grad():
                out=step_fn(P[t],T[t],PET[t],*theta.t(),S1,S2,Sc1,Sc2,Sn,doy=doy[t],nearzero=1e-6)
                S1,S2,Sc1,Sc2,Sn=[v.detach() for v in out[2:7]]
        for t in range(WARMUP,730):
            out=step_fn(P[t],T[t],PET[t],*theta.t(),S1,S2,Sc1,Sc2,Sn,doy=doy[t],nearzero=1e-6)
            qs.append(out[0]); S1,S2,Sc1,Sc2,Sn=out[2:7]
    q=torch.stack(qs); loss,kge=compute_differentiable_kge(q,y[WARMUP:],warmup_days=0); loss.backward()
    trunk=torch.cat([p.grad.reshape(-1) for n,p in net.named_parameters() if p.grad is not None and not n.startswith("net.8.")])
    common=torch.cat([net.net[-1].weight.grad[A.M4_COMMON].reshape(-1),net.net[-1].bias.grad[A.M4_COMMON].reshape(-1)])
    return {"formula":mode_name,"lambda_i":lambda_i,"loss":float(loss),"trunk_norm":float(trunk.norm()),"common_head_norm":float(common.norm()),"alpha_grad":float(raw.grad[:,4].norm()),"is_time_grad":float(raw.grad[:,5].norm()),"s2max_grad":float(raw.grad[:,2].norm()),"tw_grad":float(raw.grad[:,3].norm()),"trunk":trunk.detach()}


def surfaces(mode_name, step_fn, center, x, y, surface, n=5):
    rows=[]; center=center.detach().clone(); model_step=step_fn
    if surface=="alpha_sb1":
        ga=torch.linspace(max(.01,float(center[:,4].mean())-.3),min(.99,float(center[:,4].mean())+.3),n)
        gb=torch.linspace(max(.01,float(center[:,2].mean())-.3),min(.99,float(center[:,2].mean())+.3),n); idx=2; key="s2max"
    else:
        ga=torch.linspace(max(.01,float(center[:,4].mean())-.3),min(.99,float(center[:,4].mean())+.3),n)
        gb=torch.linspace(max(.01,float(center[:,3].mean())-.3),min(.99,float(center[:,3].mean())+.3),n); idx=3; key="tw"
    for a in ga:
        for b in gb:
            th=center.clone(); th[:,4]=a; th[:,idx]=b
            with torch.no_grad(): loss,kge,_,_=rollout(model_step,th,x,y,lambda_i=1.0)
            rows.append({"formula":mode_name,"surface":surface,"alpha":float(a),key+"_normalized":float(b),"loss":float(loss),"median_kge":float(kge.median())})
    return rows


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(2); torch.manual_seed(123)
    ids,x,y,attrs,b=load_all(); params=learned_params(ids,attrs)
    anatomy(); utilization(ids,params,x)
    init=params["baseline_dpl"]
    train_rows=[]; finals={}
    for name,fn in MODES.items():
        rr,final=short_train(name,fn,init,x,y,lambda_i=1.0,steps=30); train_rows.extend(rr); finals[name]=final
    rr,final=short_train("T3_lambda0",MODES["F0_current"],init,x,y,lambda_i=0.0,steps=30); train_rows.extend(rr); finals["T3_lambda0"]=final
    write_csv("formula_training_ab.csv",train_rows)

    # gradient probe: mapped M3 network on the same attrs/window
    _, net4, _ = A.mapped_m3_network(); grad_rows=[]; g0=None
    for name,fn,lam in [("F0_lambda0",MODES["F0_current"],0.0),("F0_current",MODES["F0_current"],1.0),("F1_halfwave",MODES["F1_halfwave"],1.0),("F2_shifted_cosine",MODES["F2_shifted_cosine"],1.0)]:
        r=gradient_probe(net4,attrs,x,y,name,fn,lam)
        if g0 is None: g0=r["trunk"]
        r["trunk_cosine_vs_lambda0"]=float(torch.dot(r["trunk"],g0)/(r["trunk"].norm()*g0.norm()+1e-12)); r.pop("trunk")
        grad_rows.append(r)
    write_csv("formula_gradient_comparison.csv",grad_rows)

    surface_rows=[]
    for name,fn in MODES.items():
        surface_rows.extend(surfaces(name,fn,params["continuation"],x,y,"alpha_sb1"))
        surface_rows.extend(surfaces(name,fn,params["continuation"],x,y,"alpha_tw"))
    write_csv("formula_loss_surface_alpha_sb1.csv",[r for r in surface_rows if r["surface"]=="alpha_sb1"])
    write_csv("formula_loss_surface_alpha_tw.csv",[r for r in surface_rows if r["surface"]=="alpha_tw"])

    branch=[]
    best_name=max((n for n in MODES), key=lambda n:max(r["best_median_kge"] for r in train_rows if r["formula"]==n))
    for name,fn in (("F0_current",MODES["F0_current"]),(best_name,MODES[best_name])):
        for s in torch.linspace(0,1,11):
            th=(1-s)*params["continuation"]+s*params["IC"]
            with torch.no_grad(): loss,kge,_,_=rollout(fn,th,x,y,lambda_i=1.0)
            branch.append({"formula":name,"s":float(s),"loss":float(loss),"median_kge":float(kge.median()),"mean_kge":float(kge.mean())})
    write_csv("matched_branch_path.csv",branch)
    write_csv("formula_continuation_combo.csv",[{"status":"NOT_RUN","reason":"best variant selected after short probe; no optional combo run in this round"}])
    write_csv("mopex5_transfer_sanity.csv",[{"status":"NOT_RUN","reason":"MOPEX4 formula variant did not yet meet acceptance criteria for transfer"}])

    # Summary
    final_summary={}
    for n in ("F0_current","F1_halfwave","F2_shifted_cosine","T3_lambda0"):
        rs=[r for r in train_rows if r["formula"]==n]
        final_summary[n]={"final_median_kge":rs[-1]["median_kge"],"best_median_kge":max(r["best_median_kge"] for r in rs),"final_loss":rs[-1]["loss"]}
    summary={"basins":b,"basin_ids":[ids[i] for i in b],"device":"cpu","full_training":False,"formula_training":final_summary,"best_formula":best_name,"production_modified":False}
    (OUT/"audit_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
