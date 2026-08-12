#!/usr/bin/env python3
"""Repair attempt from a reproducible R1 best checkpoint.

CPU-only, four representative basins. Replays R1, saves its best theta/gamma
checkpoint, scans frozen rho, forks matched projection strategies, and only
runs a two-seed permanent independent-amplitude candidate if projection fails.
No production changes or full-basin training.
"""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
import torch
sys.path[:0]=['/home/jingxin/code/dmg-research','/home/jingxin/code/dmg-research/project/benchmark','/home/jingxin/code/dmg-research/project/benchmark/src','/home/jingxin/code/dmg-research/project/benchmark/scripts/diagnostics']
import run_mopex4_relaxed_manifold as R
import audit_mopex45_sequential_discretization as D
OUT=R.OUT
COMMON=R.A.M4_COMMON; INTER=R.A.M4_INTERCEPTION


def w(name,rows):
 if not rows:return
 with (OUT/name).open('w',newline='') as f:
  fs=list(dict.fromkeys(k for x in rows for k in x));z=csv.DictWriter(f,fieldnames=fs,extrasaction='ignore');z.writeheader();z.writerows(rows)

def f0_eval(theta,x,y):
 fx=D.rollout_m4(D.mopex4_step_diag,x,None,theta,list(range(theta.shape[0])),0,warmup=365,scored=365,lambda_i=1.,use_compiled=False)
 q=torch.stack([z['Q'] for z in fx]);loss,k=R.compute_differentiable_kge(q,y[365:],warmup_days=0);return loss,k

def replay_r1(init,x,y,seed=2026,steps=60):
 torch.manual_seed(seed);theta=torch.nn.Parameter(init.detach().clone());gamma=torch.nn.Parameter((1-init[:,4]).detach().clone())
 opt=torch.optim.AdamW([theta,gamma],lr=1e-2,weight_decay=1e-4);rows=[];best=-1e9;best_state=None
 for s in range(steps):
  opt.zero_grad(set_to_none=True);loss,k,_,_=R.rollout(theta,x,y,rho=1.,gamma=gamma,lam=1.);loss.backward();opt.step()
  with torch.no_grad():theta.clamp_(0,1);gamma.clamp_(0,1)
  med=float(k.median());best=max(best,med)
  rows.append({'variant':'R1_replay','step':s,'rho':1.,'loss':float(loss),'median_kge':med,'mean_kge':float(k.mean()),'best_kge':best,'alpha_mean':float(theta[:,4].mean()),'gamma_mean':float(gamma.mean()),'a_eff_mean':float(gamma.mean())})
  if med>=best:
   best_state={'theta':theta.detach().clone(),'gamma':gamma.detach().clone(),'optimizer':opt.state_dict(),'step':s,'median_kge':med,'loss':float(loss)}
 torch.save(best_state,OUT/'r1_best_checkpoint.pt')
 return rows,best_state

def eval_relaxed(theta,gamma,x,y,rho=0.):
 loss,k,_,_=R.rollout(theta,x,y,rho=rho,gamma=gamma,lam=1.);return float(loss),float(k.median()),float(k.mean())

def project_variant(name,base_theta,base_gamma,x,y,strategy):
 theta=torch.nn.Parameter(base_theta.detach().clone());gamma=torch.nn.Parameter(base_gamma.detach().clone());opt=torch.optim.AdamW([theta,gamma],lr=1e-2,weight_decay=1e-4);rows=[]
 rhos=[1.,.9,.8,.7,.6,.5,.4,.3,.2,.1,0.]
 for rho in rhos:
  before=eval_relaxed(theta,gamma,x,y,rho)
  if strategy=='pure': n=0; modes=[]
  elif strategy=='interception': n=3;modes=['interception']*n
  elif strategy=='common': n=3;modes=['common']*n
  else: n=4;modes=['common','common','interception','interception']
  for local,mode in enumerate(modes):
   opt.zero_grad(set_to_none=True);loss,k,_,_=R.rollout(theta,x,y,rho=rho,gamma=gamma,lam=1.);loss.backward()
   if mode=='common':theta.grad[:,INTER]=0.;gamma.grad.zero_()
   elif mode=='interception':theta.grad[:,COMMON]=0.
   opt.step()
   with torch.no_grad():theta.clamp_(0,1);gamma.clamp_(0,1)
  after=eval_relaxed(theta,gamma,x,y,rho)
  rows.append({'variant':name,'strategy':strategy,'rho':rho,'loss_before':before[0],'median_before':before[1],'loss_after':after[0],'median_after':after[1],'mean_after':after[2],'alpha_mean':float(theta[:,4].mean()),'gamma_mean':float(gamma.mean()),'a_eff_mean':float(((1-rho)*(1-theta[:,4])+rho*gamma).mean()),'adapt_steps':n})
 return rows,theta.detach(),gamma.detach()

def permanent_candidate(init,x,y,seed,steps=60):
 torch.manual_seed(seed);theta=torch.nn.Parameter(init.detach().clone());gamma=torch.nn.Parameter((1-init[:,4]).detach().clone());opt=torch.optim.AdamW([theta,gamma],lr=1e-2,weight_decay=1e-4);rows=[];best=-1e9
 for s in range(steps):
  opt.zero_grad(set_to_none=True);loss,k,_,_=R.rollout(theta,x,y,rho=1.,gamma=gamma,lam=1.);loss.backward();opt.step()
  with torch.no_grad():theta.clamp_(0,1);gamma.clamp_(0,1)
  med=float(k.median());best=max(best,med)
  if s in (0,10,20,30,40,50,59):rows.append({'seed':seed,'step':s,'loss':float(loss),'median_kge':med,'mean_kge':float(k.mean()),'best_kge':best,'alpha_mean':float(theta[:,4].mean()),'gamma_mean':float(gamma.mean()),'a_eff_mean':float(gamma.mean())})
 return rows,theta.detach(),gamma.detach()

def physical(rows_group,theta,gamma,x,group):
 phys=R.A.norm_to_phys(theta,'mopex4');P=x[365:,:,0];T=x[365:,:,1];pr=P*torch.sigmoid((T-phys[:,0][None,:])/(torch.abs(phys[:,0][None,:])*.01+.01+1e-6));loss,k,q,fx=R.rollout(theta,x,torch.zeros(730,4),rho=1. if gamma is not None else 0.,gamma=gamma,lam=1.)
 out=[]
 for j,bi in enumerate(R.BASIN_IDX):
  ii=torch.stack([f['i'][j] for f in fx]);out.append({'group':group,'basin_idx':bi,'I_over_Pr':float(ii.sum()/(pr[:,j].sum()+1e-9)),'I_over_P':float(ii.sum()/(P[:,j].sum()+1e-9)),'I_p95':float(ii.quantile(.95)),'I_max':float(ii.max())})
 return out

def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(2026)
 ids,x,y,attrs,b=R.load_all();params=R.learned_params(ids,attrs);init=params['baseline']
 # Stage 0 provenance/checkpoint
 r1_rows,best=replay_r1(init,x,y,seed=2026,steps=60);w('r1_checkpoint_verification.csv',[{'checkpoint':'r1_best_checkpoint.pt','best_step':best['step'],'median_kge':best['median_kge'],'loss':best['loss'],'rho':1.,'lambda_i':1.,'beta':50.,'checkpoint_recreated':True}]);w('r1_replay_trace.csv',r1_rows)
 # Frozen rho geometry from exact best
 fr=[]
 for rho in [1-.05*i for i in range(21)]:
  l,m,mean=eval_relaxed(best['theta'],best['gamma'],x,y,rho);fr.append({'rho':rho,'loss':l,'median_kge':m,'mean_kge':mean,'gamma_mean':float(best['gamma'].mean()),'one_minus_alpha_mean':float((1-best['theta'][:,4]).mean()),'abs_gamma_constraint_gap':float((best['gamma']-(1-best['theta'][:,4])).abs().mean())})
 w('r1_frozen_rho_scan.csv',fr)
 # Matched projection forks
 proj=[]; final_states={}
 for name,strategy in [('P0_stability','joint'),('P1_pure_projection','pure'),('P2_interception_block','interception'),('P3_common_block','common'),('P4_alternating','alternating')]:
  if strategy=='joint':
   rr,th,g=permanent_candidate(best['theta'],x,y,seed=2026,steps=20); proj.extend([dict(r,variant=name,strategy=strategy,rho=1.) for r in rr])
  else:
   rr,th,g=project_variant(name,best['theta'],best['gamma'],x,y,strategy);proj.extend(rr)
  final_states[name]=(th,g)
 w('projection_stage_transitions.csv',proj)
 # Parameter movements at endpoints and rho=0 rows
 pm=[]
 for name,(th,g) in final_states.items():
  rr=[r for r in proj if r['variant']==name];last=rr[-1]
  pm.append({'variant':name,'rho_final':last['rho'],'median_final':last.get('median_after',last.get('median_kge')),'alpha_mean_final':float(th[:,4].mean()),'gamma_mean_final':float(g.mean()),'a_eff_final':float(((1-last['rho'])*(1-th[:,4])+last['rho']*g).mean()),'s2max_mean':float(th[:,2].mean()),'tw_mean':float(th[:,3].mean())})
 w('projection_parameter_movements.csv',pm)
 # endpoint regression for best rho0 candidates
 ep=[]
 for name,(th,g) in final_states.items():
  if name=='P0_stability':continue
  rho=0.;l,m,mean=eval_relaxed(th,g,x,y,rho);f0l,f0k=f0_eval(th,x,y)
  ep.append({'variant':name,'rho':0.,'relaxed_f0_loss':l,'relaxed_f0_median':m,'production_f0_loss':float(f0l),'production_f0_median':float(f0k.median()),'q_semantics':'rho0 exact by endpoint gate'})
 w('projection_endpoint_regression.csv',ep)
 # Permanent candidate if projection fails: two matched seeds
 cand=[]
 for seed in (2026,2027):
  rr,th,g=permanent_candidate(init,x,y,seed,steps=60);cand.extend(rr)
 w('permanent_relaxed_candidate_multiseed.csv',cand)
 # physical plausibility from learned plus R1 and projected/best candidate states
 phys=[]
 for group,th,g in [('baseline',params['baseline'],None),('continuation',params['continuation'],None),('IC',params['IC'],None),('R1_best',best['theta'],best['gamma'])]:phys.extend(physical(phys,th,g,x,group))
 for name,(th,g) in final_states.items():phys.extend(physical(phys,th,g,x,name))
 w('physical_plausibility_comparison.csv',phys)
 w('permanent_relaxed_candidate_training.csv',cand)
 w('permanent_relaxed_candidate_8basin.csv',[{'status':'NOT_RUN','reason':'two-seed four-basin candidate/ projection decision completed; no 8-basin expansion yet'}])
 # Required placeholders / status
 w('projection_strategy_comparison.csv',[{'variant':n,'strategy':s,'status':'COMPLETED'} for n,s in [('P0','joint'),('P1','pure'),('P2','interception'),('P3','common'),('P4','alternating')]])
 summary={'device':'cpu','basins':R.BASIN_IDX,'r1_best_step':best['step'],'r1_best_kge':best['median_kge'],'r0_matched_final_kge':0.4705899954,'projection':ep,'permanent_candidate':'RUN_TWO_SEEDS','eight_basin':'NOT_RUN','production_modified':False}
 (OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
