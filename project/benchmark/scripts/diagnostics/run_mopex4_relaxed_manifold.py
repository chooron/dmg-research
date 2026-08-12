#!/usr/bin/env python3
"""Training-only relaxed-amplitude manifold probe for MOPEX4.

Uses CPU and four representative basins. Production F0 is never changed.
The extra gamma parameter exists only in this diagnostic script:

a_eff = (1-rho)*(1-alpha) + rho*gamma
season_raw = alpha + a_eff*cos(theta)

rho=0 is exact F0. rho=1 with gamma=1-alpha is also exact F0 at
initialization. R2 explicitly projects rho back to zero before its final
reported score.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

BENCHMARK=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(BENCHMARK),str(BENCHMARK.parents[1]),str(BENCHMARK/'src'),str(BENCHMARK/'scripts/diagnostics')]
import audit_mopex34_root_cause as A
import audit_mopex45_sequential_discretization as D
from dpl.attributes import CatchmentAttributeBuilder
from dmotpy.models.flux.mopex import mopex_training_context
from project.benchmark.scripts.run_dpl_benchmark_dmg_native import compute_differentiable_kge

OUT=BENCHMARK/'results/mopex45_phase_fix/root_cause_audit/relaxed_manifold'
OUT.mkdir(parents=True,exist_ok=True)
BASIN_IDX=[391,373,269,530]; WARMUP=SCORED=365


def write_csv(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r)); w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader();w.writerows(rows)


def relaxed_step(P,T,PET,tcrit,ddf,Sb1,tw,alpha,is_time,gamma,tu,Se,Sb2,tc,S1,S2,Sc1,Sc2,Sn,doy,rho,lambda_i=1.0):
    from dmotpy.models.flux.mopex import (mopex_snowfall_1 as snowfall, mopex_rainfall_1 as rainfall,
        mopex_melt_1 as melt, mopex_evap_7 as evap, mopex_saturation_1 as sat,
        mopex_recharge_3 as recharge, mopex_baseflow_1 as baseflow)
    Sn=F.relu(Sn);S1=F.relu(S1);S2=F.relu(S2);Sc1=F.relu(Sc1);Sc2=F.relu(Sc2)
    ps=snowfall(P,T,tcrit);pr=rainfall(P,T,tcrit);qn=melt(ddf,tcrit,T,Sn,1.0);Sn_new=Sn+ps-qn
    S1=S1+pr+qn;et1=torch.minimum(evap(S1,Sb1,PET,1.0,1e-6),S1);S1=S1-et1
    beta=50.0;theta=2*torch.pi*(doy-is_time)/365.25;c=torch.cos(theta)
    aeff=(1-rho)*(1-alpha)+rho*gamma
    frac=F.softplus(beta*(alpha+aeff*c))/beta
    iraw=torch.minimum(frac*pr,pr)*lambda_i;i=torch.minimum(iraw,S1);S1=S1-i
    q1=torch.minimum(sat(pr+qn,S1,Sb1,nearzero=1e-6),S1);S1=S1-q1
    qw=torch.minimum(recharge(tw,S1),S1);S1n=S1-qw
    S2=S2+qw;q2f=torch.minimum(sat(qw,S2,Sb2,nearzero=1e-6),S2);S2=S2-q2f
    q2u=baseflow(tu,S2);S2=S2-q2u;et2=torch.minimum(evap(S2,Se*Sb2,PET,1.0,1e-6),S2);S2n=S2-et2
    Sc1=Sc1+q1+q2f;qf=baseflow(tc,Sc1);Sc1n=Sc1-qf
    Sc2=Sc2+q2u;qs=baseflow(tc,Sc2);Sc2n=Sc2-qs
    return qf+qs,et1+et2+i,S1n,S2n,Sc1n,Sc2n,Sn_new,{"i":i,"i_raw":iraw,"ET_total":et1+et2+i,"S1_new":S1n,"S2_new":S2n}


def load_all():
    ids,xfull,yfull,b=A.load_context();x=xfull[A.START:A.START+730];y=yfull[A.START:A.START+730]
    attrs=CatchmentAttributeBuilder().build_normalized_attributes(ids,device='cpu',method='zscore')[b]
    return ids,x,y,attrs,b


def learned_params(ids,attrs):
    def net(path):
        from dpl.nn_parameterizer import CatchmentParameterizer
        n=CatchmentParameterizer(35,10,hidden_dims=[256,256],dropout=.05);n.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['network']);n.eval();return n(attrs).detach()
    baseline=net(BENCHMARK/'results/dpl_round13_20260805/auto100/checkpoints/mopex4/epoch_100.pt')
    cont=net(BENCHMARK/'results/mopex45_phase_fix/full_continuation/runs/seed_41/checkpoints/J2/seed_41/epoch_100.pt')
    ic=D.ic_theta('mopex4',ids)[BASIN_IDX]
    return {'baseline':baseline,'continuation':cont,'IC':ic}


def rollout(theta,x,y,rho=0.,gamma=None,lam=1.,training=False):
    phys=A.norm_to_phys(theta,'mopex4'); gamma=theta[:,4].detach() if gamma is None else gamma
    P,T,PET,doy=x[:,:,0],x[:,:,1],x[:,:,2],x[:,:,3];Sn,S1,S2,Sc1,Sc2=D._init_states(theta.shape[0]);qs=[];fxs=[]
    with mopex_training_context(lambda_i=lam,lambda_p=1.,beta=50.):
        for t in range(365):
            with torch.no_grad():
                out=relaxed_step(P[t],T[t],PET[t],*phys[:,:4].t(),phys[:,4],phys[:,5],gamma,phys[:,6],phys[:,7],phys[:,8],phys[:,9],S1,S2,Sc1,Sc2,Sn,doy[t],rho,lam)
                S1,S2,Sc1,Sc2,Sn=[v.detach() for v in out[2:7]]
        for t in range(365,730):
            out=relaxed_step(P[t],T[t],PET[t],*phys[:,:4].t(),phys[:,4],phys[:,5],gamma,phys[:,6],phys[:,7],phys[:,8],phys[:,9],S1,S2,Sc1,Sc2,Sn,doy[t],rho,lam)
            qs.append(out[0]);fxs.append(out[-1]);S1,S2,Sc1,Sc2,Sn=out[2:7]
    q=torch.stack(qs);loss,kge=compute_differentiable_kge(q,y[365:],warmup_days=0);return loss,kge,q,fxs


def f0_rollout(theta,x,y,lam=1.):
    # current F0 via prior diagnostic step
    return D.rollout_m4(D.mopex4_step_diag,x,None,theta,list(range(theta.shape[0])),0,warmup=365,scored=365,lambda_i=lam,use_compiled=False)


def anatomy():
    rows=[];doy=torch.arange(1.,366.);beta=50.
    for a0 in (0.,.1,.25,.5,.75,1.):
        a=torch.tensor(a0,requires_grad=True);it=torch.tensor(180.,requires_grad=True)
        raw=a+(1-a)*torch.cos(2*torch.pi*(doy-it)/365.25);f=F.softplus(beta*raw)/beta
        ga,gt=torch.autograd.grad(f.mean(),(a,it))
        rows.append({'formula':'F0','alpha':a0,'peak':float(f.max()),'mean':float(f.mean()),'active_gt1pct':float((f>.01).float().mean()),'active_gt10pct':float((f>.1).float().mean()),'dmean_dalpha':float(ga),'dmean_dis_time':float(gt)})
    write_csv('formula_shape_comparison.csv',rows)


def endpoint_equivalence(theta,x,y):
    rows=[]
    f0=f0_rollout(theta,x,y,lam=1.);q0=torch.stack([z['Q'] for z in f0]); e0=torch.stack([z['ET'] for z in f0])
    # D.rollout_m4 records dict with Q at top and flux dict; extract state/flux
    for rho,gam,label in [(0.,1-theta[:,4],'rho0'),(1.,1-theta[:,4],'rho1_initial')]:
        l,k,q,fx=rollout(theta,x,y,rho=rho,gamma=gam,lam=1.)
        etdiff=float((torch.stack([z['ET'] for z in f0])-torch.stack([z['ET_total'] for z in fx])).abs().max())
        idiff=float((torch.stack([z['flux']['i'] for z in f0])-torch.stack([z['i'] for z in fx])).abs().max())
        sdiff=max(float((torch.stack([z['flux']['S1_new'] for z in f0])-torch.stack([z['S1_new'] for z in fx])).abs().max()),float((torch.stack([z['flux']['S2_new'] for z in f0])-torch.stack([z['S2_new'] for z in fx])).abs().max()))
        rows.append({'comparison':label,'q_max_abs_diff':float((q-q0).abs().max()),'et_max_abs_diff':etdiff,'interception_max_abs_diff':idiff,'S1_S2_state_max_abs_diff':sdiff,'loss':float(l),'median_kge':float(k.median())})
    write_csv('relaxed_endpoint_equivalence.csv',rows)


def train_variant(name,init,x,y,kind,steps=60):
    th=torch.nn.Parameter(init.detach().clone());gam=torch.nn.Parameter((1-init[:,4]).detach().clone()) if kind in ('r1','r2') else None
    opt=torch.optim.AdamW([th]+([] if gam is None else [gam]),lr=1e-2,weight_decay=1e-4); rows=[];best=-1e9;rho=0.
    for s in range(steps):
        if kind=='r0': rho=0.;lam=1.;g=None
        elif kind=='r1': rho=1.;lam=1.;g=gam
        elif kind=='r2': rho=1. if s<20 else max(0.,1.-(s-20)/20.)
        elif kind=='r3': rho=0.;lam=min(1.,(s+1)/steps*5/4);g=None
        if kind=='r2': lam=1.;g=gam
        opt.zero_grad(set_to_none=True)
        loss,k,_,_=rollout(th,x,y,rho=rho,gamma=g,lam=lam);loss.backward()
        gg=th.grad.detach().clone(); alpha_g=float(gg[:,4].norm()); it_g=float(gg[:,5].norm()); gamma_g=float(gam.grad.norm()) if gam is not None and gam.grad is not None else 0.
        opt.step();
        with torch.no_grad(): th.clamp_(0,1); gam.clamp_(0,1) if gam is not None else None
        med=float(k.median());best=max(best,med)
        if s in (0,5,10,19,20,29,39,49,59):
            rows.append({'variant':name,'step':s,'rho':rho,'lambda_i':lam,'loss':float(loss),'median_kge':med,'mean_kge':float(k.mean()),'best_kge':best,'alpha_mean':float(th[:,4].mean()),'gamma_mean':float(gam.mean()) if gam is not None else None,'a_eff_mean':float(((1-rho)*(1-th[:,4])+(rho*gam if gam is not None else 0)).mean()) if rho else float((1-th[:,4]).mean()),'is_time_mean':float(th[:,5].mean()),'s2max_mean':float(th[:,2].mean()),'tw_mean':float(th[:,3].mean()),'alpha_grad':alpha_g,'is_time_grad':it_g,'gamma_grad':gamma_g})
    return rows,th.detach(),gam.detach() if gam is not None else None


def gradient_geometry(theta_init,attrs,x,y):
    """At matched F0/R1 initialization compare F0 lambda=1 and relaxed
    lambda=1 gradients; also report both against the lambda=0 reference."""
    rows=[]; g0=None; g1=None; th=theta_init.detach().clone()
    specs=[('F0_lambda0',0.,0.,None),('F0_lambda1',0.,1.,None),('R1_matched',1.,1.,1-th[:,4])]
    for label,rho,lam,gam in specs:
        t=torch.nn.Parameter(th.clone()); g=None if gam is None else torch.nn.Parameter(gam.clone())
        loss,k,_,_=rollout(t,x,y,rho=rho,gamma=g,lam=lam); loss.backward(); gr=t.grad.detach(); common=gr[:,A.M4_COMMON].reshape(-1)
        if label=='F0_lambda0': g0=common.clone()
        if label=='F0_lambda1': g1=common.clone()
        rows.append({'variant':label,'forward_matched_initial':label in ('F0_lambda1','R1_matched'),
                     'loss':float(loss),'median_kge':float(k.median()),'common_grad_norm':float(common.norm()),
                     'alpha_grad_norm':float(gr[:,4].norm()),'gamma_grad_norm':float(g.grad.norm()) if g is not None and g.grad is not None else 0.,
                     'cosine_vs_F0_lambda0':float(torch.dot(common,g0)/(common.norm()*g0.norm()+1e-12)),
                     'cosine_vs_F0_lambda1':float(torch.dot(common,g1)/(common.norm()*g1.norm()+1e-12)) if g1 is not None else None})
    return rows


def surfaces(theta,x,y):
    rows=[]
    for formula,rho,fn in [('F0',0.,None),('R1',1.,None)]:
        for surface,idx in [('alpha_sb1',2),('alpha_tw',3),('alpha_gamma',-1)]:
            ga=torch.linspace(max(.01,float(theta[:,4].mean())-.3),min(.99,float(theta[:,4].mean())+.3),5)
            if surface=='alpha_gamma': gb=torch.linspace(.01,.99,5)
            else: gb=torch.linspace(max(.01,float(theta[:,idx].mean())-.3),min(.99,float(theta[:,idx].mean())+.3),5)
            for a in ga:
                for b in gb:
                    th=theta.clone();th[:,4]=a
                    gam=(1-th[:,4]) if rho==0 else th[:,4].clone()
                    if surface=='alpha_gamma': gam[:]=b
                    else: th[:,idx]=b
                    with torch.no_grad(): loss,k,_,_=rollout(th,x,y,rho=rho,gamma=gam,lam=1.)
                    rows.append({'formula':formula,'surface':surface,'alpha':float(a),'second_normalized':float(b),'loss':float(loss),'median_kge':float(k.median())})
    return rows


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(2026)
    ids,x,y,attrs,b=load_all();params=learned_params(ids,attrs);init=params['baseline']
    anatomy();endpoint_equivalence(init,x,y)
    train=[];finals={}
    for name,kind in [('R0_F0_joint','r0'),('R1_relaxed_joint','r1'),('R2_relaxed_project','r2'),('R3_F0_continuation','r3')]:
        rr,th,g=train_variant(name,init,x,y,kind);train.extend(rr);finals[name]=(th,g)
    write_csv('relaxed_training_ab.csv',train)
    write_csv('relaxed_stage_transitions.csv',[r for r in train if r['step'] in (19,20,39,59)])
    write_csv('relaxed_gradient_geometry.csv',gradient_geometry(init,attrs,x,y))
    ss=surfaces(params['continuation'],x,y);write_csv('relaxed_loss_surface_alpha_sb1.csv',[r for r in ss if r['surface']=='alpha_sb1']);write_csv('relaxed_loss_surface_alpha_tw.csv',[r for r in ss if r['surface']=='alpha_tw']);write_csv('relaxed_loss_surface_alpha_gamma.csv',[r for r in ss if r['surface']=='alpha_gamma'])
    # physical plausibility from learned groups and final R1/R2/R0, current F0
    phys=[]
    for label,th in [('baseline',params['baseline']),('continuation',params['continuation']),('IC',params['IC']),('R0_final',finals['R0_F0_joint'][0]),('R1_relaxed',finals['R1_relaxed_joint'][0]),('R2_final_F0',finals['R2_relaxed_project'][0])]:
        l,k,q,fx=rollout(th,x,y,rho=0.,gamma=1-th[:,4],lam=1.)
        for j,bi in enumerate(b):
            ii=torch.stack([f['i'][j] for f in fx]); phys.append({'group':label,'basin_idx':bi,'I_over_Pr_proxy':float(ii.sum()/(ii.sum()+1e-9)),'I_p95':float(ii.quantile(.95)),'I_max':float(ii.max()),'active_fraction':float((ii>1e-9).float().mean())})
    write_csv('relaxed_physical_plausibility.csv',phys)
    write_csv('formula_continuation_combo.csv',[{'status':'NOT_RUN','reason':'R1/R2 acceptance evaluated first; no optional combo yet'}])
    write_csv('mopex5_transfer_sanity.csv',[{'status':'NOT_RUN','reason':'MOPEX4 relaxed manifold did not yet satisfy projection/KGE acceptance'}])
    summary={'basins':b,'basin_ids':[ids[i] for i in b],'device':'cpu','full_training':False,'finals':{k:{'final_loss':v[-1]['loss'],'final_median_kge':v[-1]['median_kge'],'best_kge':max(r['best_kge'] for r in v)} for k,v in [(n,[r for r in train if r['variant']==n]) for n in finals]},'production_modified':False}
    (OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
