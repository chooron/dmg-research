#!/usr/bin/env python3
"""MOPEX4 R1 relaxed-branch -> exact F0 forward reachability test.

CPU-only, four representative basins. No training and no production changes.
The F0 alpha/is_time match is selected by an analytic coarse scan of the exact
seasonal fraction, then refined/selected using the actual runtime interception
trajectory and cap through mopex4_step_diag with lambda_i=1.
"""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path[:0]=['/home/jingxin/code/dmg-research','/home/jingxin/code/dmg-research/project/benchmark','/home/jingxin/code/dmg-research/project/benchmark/src','/home/jingxin/code/dmg-research/project/benchmark/scripts/diagnostics']
import run_mopex4_relaxed_manifold as R
import audit_mopex45_sequential_discretization as D
from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from dmotpy.models.flux.mopex import mopex_training_context

OUT=R.BENCHMARK/'results/mopex45_phase_fix/root_cause_audit/f0_reachability'
OUT.mkdir(parents=True,exist_ok=True)
BASINS=[391,373,269,530]


def w(name,rows):
 if not rows:return
 with (OUT/name).open('w',newline='') as f:
  fs=list(dict.fromkeys(k for x in rows for k in x));z=csv.DictWriter(f,fieldnames=fs,extrasaction='ignore');z.writeheader();z.writerows(rows)

def load():
 ids,x,y,attrs,b=R.load_all();return ids,x,y

def f0_roll(theta,x,y):
 fx=D.rollout_m4(D.mopex4_step_diag,x,None,theta,list(range(theta.shape[0])),0,warmup=365,scored=365,lambda_i=1.,use_compiled=False)
 q=torch.stack([z['Q'] for z in fx]);et=torch.stack([z['ET'] for z in fx]);ii=torch.stack([z['flux']['i'] for z in fx]);loss,k=R.compute_differentiable_kge(q,y[365:],warmup_days=0);return fx,q,et,ii,loss,k

def r1_roll(theta,gamma,x,y):
 return R.rollout(theta,x,y,rho=1.,gamma=gamma,lam=1.)

def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(77)
 ids,x,y=load(); ck=torch.load(R.OUT/'r1_best_checkpoint.pt',map_location='cpu',weights_only=False)
 theta=ck['theta'];gamma=ck['gamma'];rho=float(ck.get('rho',1.));
 loss1,k1,q1,fx1=r1_roll(theta,gamma,x,y)
 w('r1_checkpoint_reverification.csv',[{'checkpoint':'r1_best_checkpoint.pt','checkpoint_step':ck['step'],'rho':rho,'lambda_i':1.,'beta':50.,'loss':float(loss1),'median_kge':float(k1.median()),'mean_kge':float(k1.mean()),'finite':bool(torch.isfinite(q1).all())}])
 # R1 anatomy and target trajectory/fraction.
 phys=R.A.norm_to_phys(theta,'mopex4');P=x[365:,:,0];T=x[365:,:,1];doy=x[365:,:,3]
 pr=P*torch.sigmoid((T-phys[:,0][None,:])/(torch.abs(phys[:,0][None,:])*.01+.01+1e-6))
 theta_angle=2*torch.pi*(doy-phys[:,5][None,:])/365.25
 f_r1=F.softplus(50*(phys[:,4][None,:]+gamma[None,:]*torch.cos(theta_angle)))/50
 anatomy=[]
 for j,bi in enumerate(BASINS):
  I=torch.stack([z['i'][j] for z in fx1]);Iraw=torch.stack([z['i_raw'][j] for z in fx1]);
  anatomy.append({'basin_idx':bi,'basin_id':ids[bi],'alpha':float(phys[j,4]),'gamma':float(gamma[j]),'one_minus_alpha':float(1-theta[j,4]),'gamma_minus_one_minus_alpha':float(gamma[j]-(1-theta[j,4])),'is_time':float(phys[j,5]),'lambda_i':1.,'beta':50.,'mean_season_raw':float((phys[j,4]+gamma[j]*torch.cos(theta_angle[:,j])).mean()),'min_season_raw':float((phys[j,4]+gamma[j]*torch.cos(theta_angle[:,j])).min()),'max_season_raw':float((phys[j,4]+gamma[j]*torch.cos(theta_angle[:,j])).max()),'mean_fraction':float(f_r1[:,j].mean()),'min_fraction':float(f_r1[:,j].min()),'max_fraction':float(f_r1[:,j].max()),'fraction_active_ratio':float((f_r1[:,j]>.01).float().mean()),'sum_I_over_Pr':float(I.sum()/(pr[:,j].sum()+1e-9)),'sum_I_over_P':float(I.sum()/(P[:,j].sum()+1e-9)),'kge':float(k1[j]),'loss':float(1-k1[j])})
 w('r1_per_basin_parameter_anatomy.csv',anatomy)
 with (OUT/'r1_parameter_anatomy_summary.md').open('w') as f:
  f.write('# R1 parameter anatomy\n\n')
  f.write('The previous alpha≈0.315,gamma≈0.01 point was a relaxed alpha-gamma grid minimum, not the R1 learned point. The actual R1 checkpoint values are in `r1_per_basin_parameter_anatomy.csv`.\n\n')
  f.write('|basin|alpha|gamma|1-alpha|is_time|I/Pr|KGE|\n|---:|---:|---:|---:|---:|---:|---:|\n')
  for r in anatomy:f.write(f"|{r['basin_id']}|{r['alpha']:.4f}|{r['gamma']:.4f}|{r['one_minus_alpha']:.4f}|{r['is_time']:.1f}|{r['sum_I_over_Pr']:.4f}|{r['kge']:.4f}|\n")
 # F0 analytic coarse matching, then exact I trajectory selection among top candidates.
 match_params=[];metrics=[]
 for j,bi in enumerate(BASINS):
  target_frac=f_r1[:,j]; candidates=[]
  # deterministic coarse grid: alpha .025; phase every 5 days.
  for ai in range(41):
   a=ai/40.;
   for it in range(1,366,5):
    c=torch.cos(2*torch.pi*(doy[:,j]-it)/365.25); frac=F.softplus(50*(a+(1-a)*c))/50
    candidates.append((float((frac-target_frac).square().mean()),a,float(it),float((frac-target_frac).abs().mean())))
  candidates=sorted(candidates)[:10]
  refined=[]
  for _,a,it,analytic_mae in candidates:
   th=theta[j:j+1].clone();th[:,4]=a;th[:,5]=(it-1)/364.
   _,q,et,ii,loss,k=f0_roll(th,x[:,j:j+1],y[:,j:j+1])
   # actual trajectory target/current
   I1=torch.stack([z['i'][j] for z in fx1]); fr0=torch.stack([z['flux']['i_raw'][0] for z in _]) if False else None
   # fraction F0 analytic on this exact period
   c=torch.cos(2*torch.pi*(doy[:,j]-it)/365.25); frac0=F.softplus(50*(a+(1-a)*c))/50
   refined.append({'basin_idx':bi,'basin_id':ids[bi],'alpha_star':a,'is_time_star':it,'analytic_fraction_rmse':float((frac0-target_frac).square().mean().sqrt()),'analytic_fraction_mae':float((frac0-target_frac).abs().mean()),'I_rmse':float((ii[:,0]-I1).square().mean().sqrt()),'I_mae':float((ii[:,0]-I1).abs().mean()),'KGE_F0match':float(k[0]),'loss_F0match':float(1-k[0]),'Q_placeholder':q[:,0].detach()})
  best=min(refined,key=lambda z:z['I_rmse'])
  # rerun best for Q/ET metrics and record fraction rmse
  th=theta[j:j+1].clone();th[:,4]=best['alpha_star'];th[:,5]=(best['is_time_star']-1)/364.
  _,q0,et0,i0,l0,k0=f0_roll(th,x[:,j:j+1],y[:,j:j+1]);I1=torch.stack([z['i'][j] for z in fx1]);
  et1_target = torch.stack([z['ET_total'] for z in fx1])[:, j]
  best['Q_rmse']=float((q0[:,0]-q1[:,j]).square().mean().sqrt());best['Q_corr']=float(torch.corrcoef(torch.stack([q0[:,0],q1[:,j]]))[0,1]);best['ET_rmse']=float((et0[:,0]-et1_target).square().mean().sqrt());best.pop('Q_placeholder',None)
  match_params.append({'basin_idx':bi,'basin_id':ids[bi],'alpha_star':best['alpha_star'],'is_time_star':best['is_time_star'],'fraction_rmse':best['analytic_fraction_rmse'],'I_rmse':best['I_rmse']})
  metrics.append({'basin_idx':bi,'basin_id':ids[bi],'alpha_star':best['alpha_star'],'is_time_star':best['is_time_star'],'I_R1_over_Pr':anatomy[j]['sum_I_over_Pr'],'I_F0match_over_Pr':float(i0[:,0].sum()/(pr[:,j].sum()+1e-9)),'KGE_R1':float(k1[j]),'KGE_F0match':float(k0[0]),'delta_KGE':float(k0[0]-k1[j]),'loss_R1':float(1-k1[j]),'loss_F0match':float(1-k0[0]),'I_RMSE':best['I_rmse'],'fraction_RMSE':best['analytic_fraction_rmse'],'Q_RMSE':best['Q_rmse'],'Q_corr':best['Q_corr'],'ET_RMSE':best['ET_rmse']})
 w('f0_forward_matching_parameters.csv',match_params);w('f0_forward_matching_trajectory_metrics.csv',metrics);w('f0_forward_matching_kge.csv',metrics)
 # alpha raw diagnostics from existing R0/continuation network checkpoints
 ar=[]
 attrs=CatchmentAttributeBuilder().build_normalized_attributes(ids,device='cpu',method='zscore')[BASINS]
 for group,path in [('R0_baseline',R.BENCHMARK/'results/dpl_round13_20260805/auto100/checkpoints/mopex4/epoch_100.pt'),('continuation',R.BENCHMARK/'results/mopex45_phase_fix/full_continuation/runs/seed_41/checkpoints/J2/seed_41/epoch_100.pt')]:
  n=CatchmentParameterizer(35,10,hidden_dims=[256,256],dropout=.05);n.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['network']);n.eval();raw=n.net(attrs);norm=torch.sigmoid(raw[:,4]);
  for j,bi in enumerate(BASINS):ar.append({'group':group,'basin_idx':bi,'basin_id':ids[bi],'alpha_raw':float(raw[j,4]),'alpha_physical':float(norm[j]),'d_alpha_d_raw':float(norm[j]*(1-norm[j])),'boundary_distance':float(min(norm[j],1-norm[j])),'history_available':False})
 w('alpha_boundary_diagnostic.csv',ar)
 summary={'r1_best_median_kge':float(k1.median()),'f0match_median_kge':float(torch.tensor([r['KGE_F0match'] for r in metrics]).median()),'median_delta_kge':float(torch.tensor([r['delta_KGE'] for r in metrics]).median()),'basins_retaining_most_r1_gain':int(sum(r['KGE_F0match']>=r['KGE_R1']-.05 for r in metrics)),'verdict':'H1_optimization_accessibility_supported' if float(torch.tensor([r['KGE_F0match'] for r in metrics]).median())>float(torch.tensor([r['KGE_R1'] for r in metrics]).median())-.05 else 'H2_forward_expressivity_restriction_supported_or_mixed','lambda_i':1.,'beta':50.,'production_modified':False}
 (OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
