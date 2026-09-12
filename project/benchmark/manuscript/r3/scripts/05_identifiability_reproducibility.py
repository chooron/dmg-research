#!/usr/bin/env python3
"""Relate archived IC restart uncertainty to cross-estimator persistence.

Restart uncertainty is computed on normalized IC coordinates from the existing
10-start archive.  No restart or optimization is rerun.  The analysis reports
pooled, within-model, model-level, and leave-one-model-out associations without
claiming that identifiability causes disagreement.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from r3_common import MODEL_ORDER, R2, R3, R3_TABLES, RESTART_ROOT, STRICT_MODELS, write_audit, write_csv, write_json


def corr(x, y) -> tuple[float, float, int]:
    a = np.asarray(x, float); b = np.asarray(y, float); ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.unique(a[ok]).size < 2 or np.unique(b[ok]).size < 2: return np.nan, np.nan, int(ok.sum())
    value = spearmanr(a[ok], b[ok]); return float(value.statistic), float(value.pvalue), int(ok.sum())


def bootstrap_corr(x, y, seed, n=1000) -> tuple[float, float]:
    x=np.asarray(x,float); y=np.asarray(y,float); rng=np.random.default_rng(seed); values=[]
    for _ in range(n):
        idx=rng.integers(0,len(x),len(x)); values.append(corr(x[idx],y[idx])[0])
    values=np.asarray(values,float); return float(np.nanquantile(values,.025)),float(np.nanquantile(values,.975))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--n-bootstrap",type=int,default=1000); args=parser.parse_args()
    if args.n_bootstrap<1000: raise ValueError("formal identifiability CI requires >=1000 resamples")
    started=time.time()
    restart_path=RESTART_ROOT/"C05_IC_RESTART_PARAMETER_UNCERTAINTY.csv"
    gate_path=RESTART_ROOT/"C05_RESTART_DATA_AVAILABILITY_GATE.csv"
    restart=pd.read_csv(restart_path,dtype={"model":str,"basin_id":str})
    gate=pd.read_csv(gate_path)
    if not gate.empty and not (gate.astype(str).apply(lambda c: c.str.contains("complete canonical 531|PASS",case=False).any()).any()): raise RuntimeError("restart availability gate does not report complete coverage")
    required={"model","basin_id","parameter_index","parameter","restart_count","restart_u_sd","restart_u_iqr","restart_u_range","fitness_best_second_gap"}
    if required-set(restart.columns): raise RuntimeError(f"restart table missing {required-set(restart.columns)}")
    if len(restart)!=271*531 or restart.duplicated(["model","basin_id","parameter_index"]).any(): raise RuntimeError("restart parameter coverage/duplicates failed")
    restart["basin_id"]=restart.basin_id.astype(str).str.replace(r"\.0$","",regex=True).str.zfill(8)
    agg=restart.groupby(["model","parameter_index","parameter"],as_index=False).agg(restart_u_sd_median=("restart_u_sd","median"),restart_u_iqr_median=("restart_u_iqr","median"),restart_u_range_median=("restart_u_range","median"),fitness_best_second_gap_median=("fitness_best_second_gap","median"),restart_count_min=("restart_count","min"),restart_basin_count=("basin_id","nunique"))
    profile=pd.read_csv(R3_TABLES/"R3_PARAMETER_PROFILE_REPRODUCIBILITY.csv"); profile=profile[(profile.space=="information_cluster")][["model","parameter_index","profile_reproducibility","profile_bootstrap_ci_low","profile_bootstrap_ci_high"]]
    boundary=pd.read_csv(R2/"tables/R2_PARAMETER_BOUNDARY_TIE_AUDIT.csv"); boundary=boundary[boundary.estimator=="IC"][["model","parameter_index","total_boundary_fraction","tie_fraction","effective_rank_support"]]
    atlas=pd.read_csv(R2/"tables/R2_IC_RAW_RELATIONSHIP_ATLAS.csv"); atlas=atlas.groupby(["model","parameter_index"],as_index=False).agg(ic_relationship_sign_stability_median=("bootstrap_sign_probability","median"),ic_relationship_ci_excludes_fraction=("bootstrap_ci_excludes_zero","mean"),ic_relationship_top1_stability_max=("bootstrap_top1_membership","max"))
    merged=agg.merge(profile,on=["model","parameter_index"],validate="one_to_one").merge(boundary,on=["model","parameter_index"],validate="one_to_one").merge(atlas,on=["model","parameter_index"],validate="one_to_one")
    skill=pd.read_csv(R2.parent.parent/"results/ic_dpl_seenbasin_formal_20260901/03_MODEL_PERFORMANCE_SUMMARY.csv") if False else pd.read_csv(R2/"../../results/ic_dpl_seenbasin_formal_20260901/03_MODEL_PERFORMANCE_SUMMARY.csv")
    if "model" in skill:
        skill=skill[[c for c in skill.columns if c in {"model","KGE_IC_median","KGE_IC_mean","test_KGE_IC_median","KGE_IC"}]].drop_duplicates("model")
        merged=merged.merge(skill,on="model",how="left")
    merged["population"]="all36"; merged["identifiability_coordinate"]="normalized_u"; merged["restart_source"] = str(restart_path)
    write_csv(R3_TABLES/"R3_IDENTIFIABILITY_REPRODUCIBILITY.csv",merged)
    assoc=[]
    for metric in ["restart_u_sd_median","restart_u_iqr_median","restart_u_range_median","fitness_best_second_gap_median","total_boundary_fraction","tie_fraction"]:
        value=merged[metric].to_numpy(float); outcome=merged.profile_reproducibility.to_numpy(float); rho,p,n=corr(value,outcome); low,high=bootstrap_corr(value,outcome,20260902+len(assoc),args.n_bootstrap)
        assoc.append({"record_type":"pooled_association","metric":metric,"outcome":"information_cluster_profile_reproducibility","population":"all36","rho":rho,"p_value":p,"n":n,"bootstrap_ci_low":low,"bootstrap_ci_high":high,"leave_one_model_out_range":np.nan,"interpretation":"association, not causation"})
        for model in MODEL_ORDER:
            sub=merged[merged.model==model]
            if len(sub)>=3:
                r,pv,nn=corr(sub[metric],sub.profile_reproducibility); assoc.append({"record_type":"within_model_association","metric":metric,"outcome":"information_cluster_profile_reproducibility","population":model,"rho":r,"p_value":pv,"n":nn,"bootstrap_ci_low":np.nan,"bootstrap_ci_high":np.nan,"leave_one_model_out_range":np.nan,"interpretation":"within-model rank association"})
        loo=[]
        for removed in MODEL_ORDER:
            sub=merged[merged.model!=removed]; r,_,_=corr(sub[metric],sub.profile_reproducibility); loo.append(r)
        assoc.append({"record_type":"leave_one_model_out_association","metric":metric,"outcome":"information_cluster_profile_reproducibility","population":"all36","rho":rho,"p_value":p,"n":n,"bootstrap_ci_low":low,"bootstrap_ci_high":high,"leave_one_model_out_range":f"{np.nanmin(loo):.6f}..{np.nanmax(loo):.6f}","interpretation":"association, not causation"})
    association=pd.DataFrame(assoc); write_csv(R3_TABLES/"R3_IDENTIFIABILITY_ASSOCIATIONS.csv",association)
    model_rows=[]
    model_repro=pd.read_csv(R3_TABLES/"R3_MODEL_REPRODUCIBILITY.csv"); model_repro=model_repro[model_repro.space=="information_cluster"]
    for model in MODEL_ORDER:
        sub=merged[merged.model==model]; mr=model_repro[model_repro.model==model].iloc[0]
        model_rows.append({"model":model,"parameter_count":len(sub),"restart_u_sd_median":float(sub.restart_u_sd_median.median()),"restart_u_iqr_median":float(sub.restart_u_iqr_median.median()),"restart_u_range_median":float(sub.restart_u_range_median.median()),"profile_reproducibility":float(mr.model_reproducibility),"boundary_fraction_median":float(sub.total_boundary_fraction.median()),"ic_relationship_sign_stability_median":float(sub.ic_relationship_sign_stability_median.median()),"population":"all36","restart_coverage":"531 basins x 10 starts archived"})
    model_frame=pd.DataFrame(model_rows); write_csv(R3_TABLES/"R3_MODEL_LEVEL_IDENTIFIABILITY.csv",model_frame)
    pooled=association[(association.record_type=="pooled_association") & (association.metric=="restart_u_sd_median")].iloc[0]
    result=f"Parameter-level pooled association between median restart-u SD and profile reproducibility was rho={pooled.rho:.6f} (p={pooled.p_value:.6g}, n={pooled.n}); bootstrap CI [{pooled.bootstrap_ci_low:.6f},{pooled.bootstrap_ci_high:.6f}]. Model-level and within-model rank associations plus LOO ranges are saved."
    write_json(R3 / "cache/R3_IDENTIFIABILITY_MANIFEST.json",{"analysis":"R3-F identifiability/reproducibility","restart_source":str(restart_path),"restart_gate":str(gate_path),"coverage":"36 models x 531 basins x 10 starts x model parameters","coordinate":"normalized_u","n_bootstrap":args.n_bootstrap,"seed":20260902,"runtime_seconds":time.time()-started})
    write_audit(R3/"R3_IDENTIFIABILITY_AUDIT.md","R3-F Identifiability and Reproducibility Audit","Is poorer IC parameter identifiability associated with weaker cross-estimator relationship persistence?",f"Existing archived restart uncertainty `{restart_path}` and availability gate `{gate_path}`; primary R3 parameter profiles and R2 boundary/tie/IC bootstrap tables. No restart, optimizer, training, or checkpoint modification was performed.","Outcome is each model×parameter information-cluster profile reproducibility. Predictors are median restart_u SD/IQR/range and fitness separation on normalized parameter coordinates; boundary fraction, tie fraction, IC relationship stability, model skill, and parameter count are covariates/descriptors.","Parameter denominator is 271 common model-parameter cells; basin-level restart aggregation requires 531 basins and archived 10 starts per cell. Pooled, within-model, model-level, and leave-one-model-out association levels are distinct.","Spearman associations and 1000-row bootstrap CIs are reported without filtering on outcome. LOO ranges assess influential models. The result is not a mixed-effects causal decomposition.",result,"Boundary saturation and low restart spread can coexist with poor cross-estimator persistence, and restart spread may reflect equifinality rather than statistical noise. Sensitivity does not identify a unique mechanism.","A reviewer may say identifiability explains all disagreement. This audit can support only an association; residual model/parameter heterogeneity must be assessed from the profile, dominant, and construction-artifact tables. Use language 'associated with weaker persistence' only; never say a percentage of disagreement is caused by identifiability.","R3_F_READY",time.time()-started)


if __name__=="__main__": main()
