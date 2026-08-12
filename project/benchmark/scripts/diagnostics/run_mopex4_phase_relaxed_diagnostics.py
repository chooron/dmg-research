#!/usr/bin/env python3
"""Cheap phase/rectification diagnostics for the relaxed-manifold probe."""
from __future__ import annotations
import csv,sys
from pathlib import Path
import torch
sys.path[:0]=['/home/jingxin/code/dmg-research','/home/jingxin/code/dmg-research/project/benchmark','/home/jingxin/code/dmg-research/project/benchmark/src','/home/jingxin/code/dmg-research/project/benchmark/scripts/diagnostics']
import run_mopex4_relaxed_manifold as R
import audit_mopex45_sequential_discretization as D
from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
OUT=R.OUT

def w(name,rows):
 with (OUT/name).open('w',newline='') as f:
  fs=list(dict.fromkeys(k for x in rows for k in x)); z=csv.DictWriter(f,fieldnames=fs);z.writeheader();z.writerows(rows)

def main():
 ids,x,y,attrs,b=R.load_all(); params=R.learned_params(ids,attrs)
 rows=[]; grid=range(1,366,20)
 for group in ('baseline','continuation'):
  th=params[group]
  for day in grid:
   tn=th.clone();tn[:,5]=(float(day)-1)/364.
   loss,k,_,_=R.rollout(tn,x,y,rho=0.,gamma=1-tn[:,4],lam=1.)
   eps=1./364.; ap=tn.clone();am=tn.clone();ap[:,5]+=eps;am[:,5]-=eps
   lp,_,_,_=R.rollout(ap,x,y,rho=0.,gamma=1-ap[:,4],lam=1.);lm,_,_,_=R.rollout(am,x,y,rho=0.,gamma=1-am[:,4],lam=1.)
   rows.append({'group':group,'is_time_day':day,'loss':float(loss),'median_kge':float(k.median()),'dL_dis_time_physical':float((lp-lm)/(2.))})
 w('phase_landscape_scan.csv',rows)
 # raw transform diagnostics for baseline/continuation networks; IC raw unavailable.
 tr=[]
 for group,path in [('baseline',R.BENCHMARK/'results/dpl_round13_20260805/auto100/checkpoints/mopex4/epoch_100.pt'),('continuation',R.BENCHMARK/'results/mopex45_phase_fix/full_continuation/runs/seed_41/checkpoints/J2/seed_41/epoch_100.pt')]:
  n=CatchmentParameterizer(35,10,hidden_dims=[256,256],dropout=.05);n.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['network']);n.eval()
  raw=n.net(attrs);norm=torch.sigmoid(raw[:,5]); phys=1+norm*364.; der=norm*(1-norm)*364.
  for j,bi in enumerate(b):tr.append({'group':group,'basin_idx':bi,'basin_id':ids[bi],'raw_is_time':float(raw[j,5]),'normalized_is_time':float(norm[j]),'physical_is_time':float(phys[j]),'physical_transform_derivative':float(der[j]),'boundary_distance':float(min(norm[j],1-norm[j]))})
 w('phase_transform_diagnostics.csv',tr)
 # current F0 rectification/dead-zone on cycle, rainfall-weighted with representative forcing.
 dz=[]
 doy=x[365:,:,3];P=x[365:,:,0];T=x[365:,:,1]
 for group in ('baseline','continuation','IC'):
  th=params[group]; phys=R.A.norm_to_phys(th,'mopex4');alpha=phys[:,4];it=phys[:,5]
  raw=alpha[None,:]+(1-alpha)[None,:]*torch.cos(2*torch.pi*(doy-it[None,:])/365.25)
  for j,bi in enumerate(b):
   rainy=P[:,j]>0.1; wgt=P[:,j]/(P[:,j].sum()+1e-9)
   dz.append({'group':group,'basin_idx':bi,'basin_id':ids[bi],'season_raw_negative_fraction':float((raw[:,j]<0).float().mean()),'rainfall_weighted_raw_negative_fraction':float((wgt[:,j] if wgt.dim()>1 else wgt)*(raw[:,j]<0).float().sum()) if False else float((wgt*(raw[:,j]<0).float()).sum()),'season_softplus_nearzero_fraction':float((raw[:,j]<-0.1).float().mean()),'rainy_fraction_raw_negative':float(((raw[:,j]<0)&rainy).float().sum()/rainy.float().sum().clamp_min(1))})
 w('rectification_deadzone_stats.csv',dz)
if __name__=='__main__':main()
