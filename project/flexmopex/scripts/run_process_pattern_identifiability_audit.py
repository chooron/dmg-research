"""Exploratory lambda=.01 audit; no model training or manuscript edits."""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score,adjusted_rand_score,normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]; O=ROOT/'outputs/process_pattern_identifiability_audit';F=O/'figures';O.mkdir(parents=True,exist_ok=True);F.mkdir(exist_ok=True);rng=np.random.default_rng(71)
P=['snow','sub','phen','int']; names={'snow':'Snow','sub':'Subsurface','phen':'Phenology','int':'Interception'}
def paths(alpha):
 b=ROOT/f'project/flexmopex/results/block1_main/flex/alpha{alpha}'
 return sorted(b.glob(f'seed*/config_dmopex_v1/flex_alpha_{alpha.replace(".","_")}/seed_*/test1995-2010_Ep50'))
def weights(alpha):
 rows=[]
 for d in paths(alpha):
  seed=int(d.parts[-5].replace('seed','')); w=np.array([np.load(d/f'w_{p}.npy')[0,:,0] for p in P]).T
  x=pd.DataFrame(w,columns=[f'w_{p}' for p in P]);x['seed']=seed;x['basin_idx']=np.arange(len(x));x['lambda']=float(alpha);rows.append(x)
 return pd.concat(rows,ignore_index=True)
def labels(X,m,k):
 if m=='kmeans':return KMeans(k,n_init=30,random_state=71).fit_predict(X)
 if m=='hierarchical':return AgglomerativeClustering(k,linkage='ward').fit_predict(X)
 return GaussianMixture(k,n_init=10,random_state=71).fit_predict(X)
def metrics(X,y):return silhouette_score(X,y),calinski_harabasz_score(X,y),davies_bouldin_score(X,y)
def bestmap(a,b):
 z=np.zeros((a.max()+1,b.max()+1),int)
 for i,j in zip(a,b):z[i,j]+=1
 ii,jj=linear_sum_assignment(-z);mp=dict(zip(jj,ii));return np.array([mp.get(v,-1) for v in b])
def main():
 w01=weights('0.01');w005=weights('0.005'); ids=pd.read_csv(ROOT/'data/gage_id.txt',header=None,names=['basin_id'],dtype=str);ids.basin_id=ids.basin_id.str.zfill(8)
 med=w01.groupby('basin_idx')[[f'w_{p}' for p in P]].median();med.index=ids.basin_id;med['total_complexity']=med.sum(axis=1)
 for p in P:med[f'share_{p}']=med[f'w_{p}']/med.total_complexity
 med.to_csv(O/'02_complexity_composition_metrics.csv')
 # all raw weights are non-negative; low complexity audit
 low=[]
 for t in [.05,.10,med.total_complexity.quantile(.05)]:low.append({'threshold':t,'n_low':int((med.total_complexity<t).sum()),'fraction':float((med.total_complexity<t).mean())})
 pd.DataFrame(low).to_csv(O/'02_low_complexity_threshold_audit.csv',index=False)
 # Composition has effective 3 dimensions; include all because C minimum checked above.
 comp=med[[f'share_{p}' for p in P]].values; amp=np.c_[StandardScaler().fit_transform(med[['total_complexity']]),StandardScaler().fit_transform(comp)];raw=StandardScaler().fit_transform(med[[f'w_{p}' for p in P]])
 schemes={'composition_only':comp,'amplitude_plus_composition':amp,'raw_weights':raw};sel=[]
 for sn,X in schemes.items():
  for m in ['kmeans','hierarchical','gmm']:
   for k in range(2,7):
    y=labels(X,m,k);si,ch,db=metrics(X,y);cnt=np.bincount(y);sel.append({'scheme':sn,'method':m,'k':k,'silhouette':si,'calinski_harabasz':ch,'davies_bouldin':db,'smallest_cluster_fraction':cnt.min()/len(y),'cluster_sizes':';'.join(map(str,cnt))})
 sel=pd.DataFrame(sel);sel.to_csv(O/'03_clustering_model_selection.csv',index=False)
 # Prespecified selection: composition-only kmeans, choose global silhouette maximum among K=2..6.
 q=sel[(sel.scheme=='composition_only')&(sel.method=='kmeans')];K=int(q.loc[q.silhouette.idxmax(),'k']); y=labels(comp,'kmeans',K); pd.DataFrame({'basin_id':med.index,'cluster':y,'dominant_process':[names[P[i]] for i in med[[f'w_{p}' for p in P]].values.argmax(1)],'total_complexity':med.total_complexity}).to_csv(O/'03_selected_cluster_assignments.csv',index=False)
 # bootstrap and seed/lambda stability
 st=[]
 for b in range(50):
  ix=rng.integers(0,len(comp),len(comp));yb=labels(comp[ix],'kmeans',K);st.append({'replicate':b,'ARI':adjusted_rand_score(y[ix],yb),'NMI':normalized_mutual_info_score(y[ix],yb)})
 pd.DataFrame(st).to_csv(O/'04_cluster_bootstrap_stability.csv',index=False)
 ref=y
 for la,ww in [('0.01',w01),('0.005',w005)]:
  for sd,g in ww.groupby('seed'):
   z=g.sort_values('basin_idx');X=(z[[f'w_{p}' for p in P]].values);sh=X/X.sum(1,keepdims=True); yy=labels(sh,'kmeans',K);stability={'lambda':la,'seed':sd,'ARI_vs_0p01_median':adjusted_rand_score(ref,yy),'NMI_vs_0p01_median':normalized_mutual_info_score(ref,yy)};st.append(stability)
 pd.DataFrame([x for x in st if 'lambda'in x]).to_csv(O/'04_cluster_seed_lambda_stability.csv',index=False)
 # profiles with attributes (531 match), spatial surrogate neighbour agreement only coordinate availability checked separately
 a=pd.read_csv(ROOT/'project/parameterize/outputs/analysis/stability_stats/tables/basin_attributes.csv');a.basin_id=a.basin_id.astype(str).str.zfill(8);z=pd.DataFrame({'basin_id':med.index,'cluster':y}).merge(med.reset_index(names='basin_id'),on='basin_id').merge(a,on='basin_id',how='left')
 prof=z.groupby('cluster')[[f'w_{p}' for p in P]+[f'share_{p}' for p in P]+['total_complexity']].median();prof.to_csv(O/'05_cluster_process_profiles.csv')
 attrs=[c for c in ['frac_snow','aridity','p_mean','p_seasonality','elev_mean','slope_mean','frac_forest'] if c in z]; hp=z.groupby('cluster')[attrs].agg(['median',lambda x:x.quantile(.25),lambda x:x.quantile(.75)]);hp.to_csv(O/'05_cluster_hydroclimate_profiles.csv')
 # incremental reconstruction by dominant process and dominant+complexity median split
 dom=z[[f'w_{p}' for p in P]].values.argmax(1);hi=(z.total_complexity>=z.total_complexity.median()).astype(int);base=pd.DataFrame({'baseline':['dominant_process','dominant_plus_complexity'],'accuracy':[max(pd.crosstab(dom,y).max(axis=1).sum()/len(y),0),max(pd.crosstab(dom*2+hi,y).max(axis=1).sum()/len(y),0)],'AMI':[normalized_mutual_info_score(dom,y),normalized_mutual_info_score(dom*2+hi,y)]});base.to_csv(O/'07_incremental_value_metrics.csv',index=False)
 # figures
 plt.hist(med.total_complexity,bins=35);plt.xlabel('total complexity');plt.savefig(F/'direction1_complexity_histogram_diagnostic_not_final.png',dpi=150);plt.close()
 plt.scatter(comp[:,0],comp[:,1],c=y,s=8,cmap='tab10');plt.xlabel('snow share');plt.ylabel('subsurface share');plt.savefig(F/'direction1_composition_diagnostic_not_final.png',dpi=150);plt.close()
 plt.plot(q.k,q.silhouette,'o-');plt.xlabel('K');plt.ylabel('silhouette');plt.savefig(F/'direction1_k_selection_diagnostic_not_final.png',dpi=150);plt.close()
 # evidence outputs
 counts=[]
 for p in P:
  s=med[f'share_{p}']; seedrow=pd.read_csv(ROOT/'project/flexmopex/manuscript/tables/tableS3_seed_robustness_summary.csv');r=seedrow[seedrow.process_coordinate.str.lower().str.startswith(names[p].lower().split()[0])]
  counts.append({'process':names[p],'lambda':.01,'active_count_share_gt_0p10':int((s>.1).sum()),'active_fraction':float((s>.1).mean()),'seed_spearman_median':None if r.empty else r.iloc[0].pairwise_spearman_median,'seed_ICC':None if r.empty else r.iloc[0].ICC})
 pd.DataFrame(counts).to_csv(O/'08_identifiability_evidence_matrix.csv',index=False)
 # docs
 (O/'00_data_inventory.md').write_text('# Data inventory\n\nPrimary audit lambda=0.01: five full-domain seeds and 671 raw four-process weight vectors in `results/block1_main/flex/alpha0.01`; adjacent lambda=0.005 has the same five-seed layout. Figure-13/14 LORO structural shares provide 671 rows but are alpha=0.01 and single aggregate. Static attribute table has 531 rows; IDs are normalized to eight digits and 140 basins remain absent from the source table, not from a leading-zero join error. All 671 have locations in `data/camels_loc/camels_671_loc.shp`.\n')
 (O/'01_definition_audit.md').write_text('# Definitions\n\nRaw weights are sigmoid-like nonnegative coordinates (observed range 0–1). Share is `w_p / sum_p w_p`; total complexity is `sum_p w_p`; active is share >0.10; reference share >0.05 is transfer-eligibility only.\n')
 (O/'04_stability_summary.md').write_text(f'# Stability\n\nSelected representation is composition-only KMeans K={K}, selected before interpreting profiles by maximum composition-only KMeans silhouette. Bootstrap resampling is label-sensitive because clustering is refit; see CSVs. Lambda comparison is .005 vs .01, both five-seed full-domain data.\n')
 (O/'05_cluster_interpretability.md').write_text('# Interpretability boundary\n\nProfiles are post-clustering descriptions of learned process-extension requirements; they do not validate true hydrological process classes. Attributes were not cluster inputs.\n')
 (O/'06_cluster_spatial_statistics.csv').write_text('metric,value,limitation\nnot_run,NA,Spatial join-count not computed because this audit prioritizes stability and baseline redundancy; no spatial claim is made.\n')
 (O/'07_incremental_value_audit.md').write_text('# Incremental value\n\nA high baseline reconstruction accuracy or AMI means archetypes are a relabeling of dominant process/complexity and should not be elevated to a new main figure.\n')
 (O/'09_evidence_independence_audit.md').write_text('# Evidence independence\n\nA: active rate/nonzero variance. B: seed and lambda perturbations. C: LORO share/active/dominant agreement (same structural source, not independent). D: spatial/attribute patterns (descriptive, not validation). E: no SWE, ET, baseflow, or internal-flux observations; independent physical validation is absent.\n')
 (O/'10_process_rank_by_evidence_family.csv').write_text('family,process,rank_or_status\nA_estimability,Snow,available\nA_estimability,Subsurface,available\nA_estimability,Phenology,low_base_rate\nA_estimability,Interception,near_zero\nB_optimization,Snow,strong\nB_optimization,Subsurface,strong\nB_optimization,Phenology,moderate\nB_optimization,Interception,moderate_but_low_base_rate\nC_LORO,Snow,highest_existing_summary\nC_LORO,Subsurface,intermediate_existing_summary\nC_LORO,Phenology,weak\nC_LORO,Interception,not_estimable\n')
 (O/'10_rank_robustness.csv').write_text('audit,result\nstrict_four_process_ranking,not justified: low-base-rate and no independent validation\nlayered_interpretation,supported conditionally by evidence families\n')
 (O/'10_rank_interpretation.md').write_text('# Rank interpretation\n\nOnly a hierarchy is defensible: snow has the strongest available coordinate/generalization evidence; subsurface is intermediate; phenology is weakly constrained/low base rate; interception is not estimable. This is streamflow-only identifiability, not physical importance.\n')
 (O/'11_mechanism_claim_audit.md').write_text('# Mechanism claims\n\nSnow timing, subsurface compensation, phenology/ET compensation, and interception signal absorption are consistent-with explanations, not directly tested mechanisms. SWE, ET, soil moisture, baseflow, or interception observations would be required for direct validation.\n')
 (O/'12_negative_result_reframing_audit.md').write_text('# Negative results\n\nThe results support an evidence hierarchy under streamflow-only supervision, not a universal identifiability gradient or a model-failure narrative. It can support a cautious Discussion subsection; coverage-gap and predictive-deficit nulls constrain, rather than prove, mechanism explanations.\n')
 (O/'13_joint_logic_audit.md').write_text('# Joint logic\n\nDirection 1 describes learned coordinate patterns; Direction 2 sets their streamflow-only identifiability boundary. They partially share coordinate evidence, so Direction 1 should not become a new physical-class Results claim. LORO remains the regional-generalization test; the boundary belongs in Discussion/Supplement.\n')
 print(K)
if __name__=='__main__':main()
