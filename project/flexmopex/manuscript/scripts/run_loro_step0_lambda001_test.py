"""Temporary, explicitly non-0.007 Step-0 LORO coverage test.

Uses the established alpha=0.01 Figure 12/13 source tables.  It never writes
to the 0.007 audit directory and is intended only to assess whether the
proposed analysis is informative when compatible legacy data are available.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/loro_step0_validation/lambda001_temporary_test'; FIG=OUT/'figures'
OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(exist_ok=True)
STRUCT=ROOT/'project/flexmopex/manuscript/figures/csv/plot_fig13_coordinate_transfer_long.csv'
PERF=ROOT/'project/flexmopex/manuscript/figures/csv/plot_fig12_loro_performance_merged.csv'
ATTR=ROOT/'project/parameterize/outputs/analysis/stability_stats/tables/basin_attributes.csv'
LOC=ROOT/'data/camels_loc/camels_671_loc.shp'
RNG=np.random.default_rng(20260711)

def ci_boot(df, formula, coef='d_z', B=400):
    vals=[]; group='region' if 'region' in df.columns else 'region_id'; regs=df[group].unique()
    for _ in range(B):
        take=RNG.choice(regs,len(regs),replace=True)
        x=pd.concat([df[df[group]==r].assign(region=f'{r}_{j}') for j,r in enumerate(take)])
        try: vals.append(smf.ols(formula,x).fit().params[coef])
        except Exception: pass
    return tuple(np.quantile(vals,[.025,.975])) if vals else (np.nan,np.nan)
def knn_distance(df, features, k, pca=False):
    out=np.full(len(df),np.nan); X=df[features].to_numpy(float)
    for r in sorted(df.region.unique()):
        te=np.where(df.region.to_numpy()==r)[0]; tr=np.where(df.region.to_numpy()!=r)[0]
        mu=X[tr].mean(0); sd=X[tr].std(0,ddof=1); sd[sd==0]=1
        a=(X[tr]-mu)/sd; b=(X[te]-mu)/sd
        if pca:
            pc=PCA().fit(a); n=np.searchsorted(np.cumsum(pc.explained_variance_ratio_),.9)+1
            a=pc.transform(a)[:,:n]; b=pc.transform(b)[:,:n]
        d=NearestNeighbors(n_neighbors=min(k,len(tr))).fit(a).kneighbors(b)[0]
        out[te]=np.median(d,axis=1)
    return out
def moran(x,lat,lon,k=8,B=499):
    x=np.asarray(x); z=x-x.mean(); n=len(x); xy=np.c_[lat,lon*np.cos(np.deg2rad(lat.mean()))]
    nn=NearestNeighbors(n_neighbors=k+1).fit(xy).kneighbors(return_distance=False)[:,1:]
    W=np.zeros((n,n)); W[np.arange(n)[:,None],nn]=1/k
    I=n/W.sum()*(W*np.outer(z,z)).sum()/(z*z).sum()
    sims=[]
    for _ in range(B):
      q=RNG.permutation(z); sims.append(n/W.sum()*(W*np.outer(q,q)).sum()/(q*q).sum())
    p=(1+(np.abs(sims)>=abs(I)).sum())/(B+1)
    return I,p
def main():
 s=pd.read_csv(STRUCT); s['basin_id']=s.basin_id.astype(str).str.zfill(8); s['region_id']=s.region.str.extract(r'(\d+)').astype(int)-1
 a=pd.read_csv(ATTR); a.basin_id=a.basin_id.astype(str).str.zfill(8)
 import geopandas as gpd
 l=gpd.read_file(LOC)[['gage_id','lat','lon']]; l.gage_id=l.gage_id.astype(str).str.zfill(8)
 d=s.merge(a,on='basin_id',how='inner').merge(l,left_on='basin_id',right_on='gage_id',how='inner')
 # Explicit audit: available static table covers 531, so this is the analysis universe.
 mainf=['aridity','frac_snow','p_seasonality','p_mean']; ext=mainf+['slope_mean','soil_conductivity','soil_depth_pelletier','max_water_content']
 for c in mainf+ext:
  d[c]=pd.to_numeric(d[c],errors='coerce')
 d=d.dropna(subset=mainf+['lat','lon']).copy()
 d['p_mean_log1p']=np.log1p(d.p_mean)
 mainf=['aridity','frac_snow','p_seasonality','p_mean_log1p']
 d['coverage_knn5']=knn_distance(d,mainf,5); d['coverage_knn10']=knn_distance(d,mainf,10)
 d['coverage_pca_knn5']=knn_distance(d,mainf,5,True)
 dext=d.dropna(subset=ext).copy(); dext['coverage_extended_knn5']=knn_distance(dext,ext,5); d=d.merge(dext[['basin_id','coverage_extended_knn5']],on='basin_id',how='left')
 long=[]
 for proc,key in [('snow','snow'),('subsurface','sub'),('phenology','phen'),('interception','int')]:
  x=d[['basin_id','region_id','lat','lon','coverage_knn5','coverage_knn10','coverage_pca_knn5','coverage_extended_knn5']].copy()
  x['process']=proc; x['reference_share']=d[f'reference_share_{key}']; x['loro_share']=d[f'loro_share_{key}']
  x['signed_shift']=x.loro_share-x.reference_share; x['absolute_shift']=x.signed_shift.abs()
  x['reference_active']=x.reference_share>.1; x['loro_active']=x.loro_share>.1; x['transfer_eligible']=x.reference_share>.05; x['both_inactive']=~x.reference_active&~x.loro_active
  x['threshold_based_label']=np.select([x.reference_active&x.loro_active,x.reference_active&~x.loro_active,~x.reference_active&x.loro_active],['matched_active','lost_activity','gained_activity'],'both_inactive')
  long.append(x)
 long=pd.concat(long,ignore_index=True); long.to_csv(OUT/'02_basin_transfer_metrics_long.csv',index=False)
 wide=long[long.process.isin(['snow','subsurface'])].pivot(index='basin_id',columns='process',values=['absolute_shift','signed_shift']).reset_index(); wide.columns=['basin_id']+[f'{a}_{b}' for a,b in wide.columns[1:]]; wide.to_csv(OUT/'02_basin_transfer_metrics_wide.csv',index=False)
 cov=d[['basin_id','region_id','lat','lon']+mainf+['coverage_knn5','coverage_knn10','coverage_pca_knn5','coverage_extended_knn5']]; cov.to_csv(OUT/'03_basin_coverage_distance.csv',index=False)
 rs=cov.groupby('region_id').agg(basin_count=('basin_id','size'),coverage_knn5_median=('coverage_knn5','median'),coverage_knn5_q25=('coverage_knn5',lambda x:x.quantile(.25)),coverage_knn5_q75=('coverage_knn5',lambda x:x.quantile(.75)),coverage_knn5_p90=('coverage_knn5',lambda x:x.quantile(.9))).reset_index(); rs.to_csv(OUT/'03_region_coverage_summary.csv',index=False)
 # performance median over seeds, joined by basin
 p=pd.read_csv(PERF);p.basin_id=p.basin_id.astype(str).str.zfill(8); p=p.groupby('basin_id')[['NSE_basic','NSE_full','NSE_flex']].median().reset_index();p['predictive_deficit']=(p.NSE_full-p.NSE_basic)-(p.NSE_flex-p.NSE_basic)
 links=wide.merge(p,on='basin_id').merge(cov[['basin_id','region_id','coverage_knn5']],on='basin_id');links['combined_shift']=links.absolute_shift_snow+links.absolute_shift_subsurface;links['max_shift']=links[['absolute_shift_snow','absolute_shift_subsurface']].max(axis=1);links.to_csv(OUT/'06_predictive_structural_link.csv',index=False)
 rows=[]; sens=[]
 for proc in ['snow','subsurface']:
  q=long[long.process==proc].copy()
  for dist in ['coverage_knn5','coverage_knn10','coverage_pca_knn5','coverage_extended_knn5']:
   for subset,xx in [('all basins',q),('transfer-eligible basins',q[q.transfer_eligible]),('excluding both-inactive',q[~q.both_inactive]),('reference-active only',q[q.reference_active])]:
    xx=xx.dropna(subset=[dist]).copy(); xx['d_z']=(xx[dist]-xx[dist].mean())/xx[dist].std();
    fit=smf.ols('absolute_shift ~ d_z + C(region_id)',xx).fit(cov_type='HC3'); ci=ci_boot(xx,'absolute_shift ~ d_z + C(region)',B=20)
    rows.append(dict(process=proc,response='absolute_shift',distance=dist,subset=subset,n=len(xx),effect=fit.params.get('d_z'),robust_ci_low=fit.conf_int().loc['d_z',0],robust_ci_high=fit.conf_int().loc['d_z',1],naive_p=fit.pvalues.get('d_z'),region_boot_ci_low=ci[0],region_boot_ci_high=ci[1]))
    if dist in ['coverage_knn5','coverage_knn10']:
     for th in [.05,.10,.15]:
      mis=((xx.reference_share>th)!=(xx.loro_share>th)).astype(int); f=smf.ols('mis ~ d_z + C(region_id)',xx.assign(mis=mis)).fit(cov_type='HC3'); sens.append(dict(process=proc,response='threshold_mismatch',distance=dist,threshold=th,k=5 if '5' in dist else 10,sample_subset=subset,effect=f.params.get('d_z'),ci_low=f.conf_int().loc['d_z',0],ci_high=f.conf_int().loc['d_z',1],p=f.pvalues.get('d_z'),spatial_adjusted_p=np.nan,direction_stable=np.sign(f.params.get('d_z',0))>0))
 models=pd.DataFrame(rows);models.to_csv(OUT/'05_basin_level_models.csv',index=False);pd.DataFrame(sens).to_csv(OUT/'07_sensitivity_matrix.csv',index=False)
 # regional statistics and spatial diagnostics
 reg=[]
 for proc in ['snow','subsurface']:
  for r,x in long[long.process==proc].groupby('region_id'):
   reg.append(dict(process=proc,region_id=r,basin_count=len(x),median_absolute_shift=x.absolute_shift.median(),matched_active_proportion=(x.threshold_based_label=='matched_active').mean(),lost_activity_proportion=(x.threshold_based_label=='lost_activity').mean(),reference_loro_spearman=spearmanr(x.reference_share,x.loro_share).statistic,coverage_gap=x.coverage_knn5.median()))
 reg=pd.DataFrame(reg);reg.to_csv(OUT/'04_region_level_statistics.csv',index=False)
 spatial=[]
 for proc in ['snow','subsurface']:
  x=long[long.process==proc]
  for var in ['absolute_shift','coverage_knn5']:
   I,pv=moran(x[var],x.lat,x.lon);spatial.append(dict(process=proc,variable=var,morans_I=I,naive_permutation_p=pv,spatial_block_adjusted_p='not separately identifiable: region FE/bootstrap used for association'))
 pd.DataFrame(spatial).to_csv(OUT/'05_spatial_autocorrelation.csv',index=False)
 lr=[]
 for z in ['absolute_shift_snow','absolute_shift_subsurface','combined_shift','max_shift','coverage_knn5']:
  x=links.dropna(subset=[z]).copy();x['z']=(x[z]-x[z].mean())/x[z].std();f=smf.ols('predictive_deficit ~ z + C(region_id)',x).fit(cov_type='HC3');ci=ci_boot(x,'predictive_deficit ~ z + C(region)',coef='z',B=30);lr.append(dict(predictor=z,n=len(x),effect=f.params.z,ci_low=f.conf_int().loc['z',0],ci_high=f.conf_int().loc['z',1],p=f.pvalues.z,region_boot_ci_low=ci[0],region_boot_ci_high=ci[1],spearman_rho=spearmanr(x[z],x.predictive_deficit).statistic))
 pd.DataFrame(lr).to_csv(OUT/'06_predictive_structural_statistics.csv',index=False)
 # compact diagnostic plots
 for proc in ['snow','subsurface']:
  x=long[long.process==proc];f=smf.ols('absolute_shift ~ coverage_knn5 + C(region_id)',x).fit(cov_type='HC3');plt.figure(figsize=(5,4));plt.scatter(x.coverage_knn5,x.absolute_shift,c=x.region_id,cmap='tab10',s=13,alpha=.65);plt.xlabel('Training-domain kNN coverage gap (k=5)');plt.ylabel('Absolute share shift');plt.title(f'{proc}, n={len(x)}, adjusted slope={f.params.coverage_knn5:.3f}');plt.colorbar(label='LORO region');plt.tight_layout();plt.savefig(FIG/f'{proc}_shift_vs_coverage_diagnostic_not_final.png',dpi=180);plt.close()
 x=links;plt.figure(figsize=(5,4));plt.scatter(x.combined_shift,x.predictive_deficit,c=x.region_id,cmap='tab10',s=13,alpha=.65);plt.xlabel('Snow + subsurface absolute shift');plt.ylabel('Predictive transfer deficit');plt.tight_layout();plt.savefig(FIG/'predictive_deficit_vs_combined_shift_diagnostic_not_final.png',dpi=180);plt.close()
 # documentation
 missing=(1-d[mainf].notna().mean()).to_dict(); corr=d[mainf].corr().round(3).to_csv()
 (OUT/'03_attribute_preprocessing.md').write_text(f'# Attributes (lambda=0.01 temporary test)\n\nAnalysis universe is the 531 basins matched to the available static-attribute table; 140 of 671 structural basins have no match and were not imputed. Main attributes: {mainf}. p_mean used log1p, fit/scaled inside each LORO training complement. Missing fractions after match: {missing}.\n\nCorrelation matrix:\n```\n{corr}```\nExploratory subsurface distance adds slope_mean, soil_conductivity, soil_depth_pelletier, max_water_content; it is not a replacement for the main distance.\n')
 (OUT/'02_transfer_metric_definition.md').write_text('# Transfer metric definition\n\n`absolute_shift = abs(loro_share-reference_share)` and `signed_shift = loro_share-reference_share`. Shares are the Figure-13 alpha=0.01 normalized four-process coordinates. Active is share > 0.10; transfer eligible is reference share > 0.05. No seed-based failure tolerance is available because the legacy structural table has one aggregated seed per basin.\n')
 (OUT/'05_model_diagnostics.md').write_text('# Model diagnostics\n\nOLS models include region fixed effects and HC3 robust SE. Region-resampled bootstrap CI is reported. Moran I uses 8 spatial nearest neighbours and 499 label permutations. This temporary dataset has 531 attribute-matched basins and one legacy structural realization, so no seed-based failure model was fit.\n')
 print('written',OUT)
if __name__=='__main__': main()
