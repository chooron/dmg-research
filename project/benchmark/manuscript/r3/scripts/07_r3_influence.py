#!/usr/bin/env python3
"""Audit pooled, model, boundary, and identifiability sensitivities for R3.

All headline summaries are repeated in raw and cluster spaces, all36 and
exclude_simhyd, model-equal and cell-equal forms, leave-one-model-out, boundary
filtered, and high/low restart-identifiability subsets.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from r3_common import MODEL_ORDER, R2, R3, R3_TABLES, STRICT_MODELS, write_audit, write_csv, write_json

SPACES=("raw_attribute","information_cluster")


def add(rows, analysis, space, population, sensitivity, metric, value, denominator, note=""):
    rows.append({"analysis":analysis,"space":space,"population":population,"sensitivity":sensitivity,"metric":metric,"value":value,"denominator":denominator,"note":note})


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--boundary-cutoff",type=float,default=.20); args=parser.parse_args()
    started=time.time(); profile=pd.read_csv(R3_TABLES/"R3_PARAMETER_PROFILE_REPRODUCIBILITY.csv"); model=pd.read_csv(R3_TABLES/"R3_MODEL_REPRODUCIBILITY.csv"); sign=pd.read_csv(R3_TABLES/"R3_SIGN_AGREEMENT_AUDIT.csv"); dom_raw=pd.read_csv(R3_TABLES/"R3_DOMINANT_ATTRIBUTE_AGREEMENT.csv"); dom_cluster=pd.read_csv(R3_TABLES/"R3_DOMINANT_CLUSTER_AGREEMENT.csv"); ident=pd.read_csv(R3_TABLES/"R3_IDENTIFIABILITY_REPRODUCIBILITY.csv")
    rows=[]
    for space in SPACES:
        for population, models in (("all36",MODEL_ORDER),("exclude_simhyd",STRICT_MODELS)):
            pm=model[(model.space==space)&model.model.isin(models)]
            add(rows,"pooled",space,population,"model_equal_primary","profile_reproducibility_median",float(pm.model_reproducibility.median()),len(models),"median of model medians")
            if pm.ic_anchored_reproducibility.notna().any():
                add(rows,"pooled",space,population,"ic_anchored_model_equal","profile_reproducibility_median",float(pm.ic_anchored_reproducibility.median()),int(pm.ic_anchored_reproducibility.notna().sum()),"IC-stable feature-anchored sensitivity")
            add(rows,"pooled",space,population,"model_equal_mean","profile_reproducibility_mean",float(pm.model_reproducibility.mean()),len(models),"mean of model medians")
            add(rows,"pooled",space,population,"cell_equal","profile_reproducibility_median",float(profile[(profile.space==space)&profile.model.isin(models)].profile_reproducibility.median()),int(len(profile[(profile.space==space)&profile.model.isin(models)])),"median of model-parameter profile cells")
            s=sign[(sign.space==space)&(sign.population==population)]
            for definition in s.definition.unique(): add(rows,"pooled",space,population,"sign_"+definition,"sign_agreement_rate",float(s[s.definition==definition].agreement_rate.iloc[0]),int(s[s.definition==definition].denominator.iloc[0]))
            d=dom_raw if space=="raw_attribute" else dom_cluster
            d=d[d.model.isin(models)]
            add(rows,"pooled",space,population,"dominant_top1","top1_exact_agreement",float(d.top1_exact_agreement.mean()),len(d))
            add(rows,"pooled",space,population,"dominant_top3","top3_overlap_count",float(d.top3_overlap_count.mean()),len(d))
            add(rows,"pooled",space,population,"dominant_top5","top5_overlap_count",float(d.top5_overlap_count.mean()),len(d))
            for removed in models:
                kept=[m for m in models if m!=removed]
                if not kept: continue
                keptm=pm[pm.model.isin(kept)]
                add(rows,"leave_one_model_out",space,population,removed,"profile_reproducibility_model_equal",float(keptm.model_reproducibility.median()),len(kept))
                if keptm.ic_anchored_reproducibility.notna().any():
                    add(rows,"leave_one_model_out",space,population,removed,"profile_reproducibility_ic_anchored",float(keptm.ic_anchored_reproducibility.median()),int(keptm.ic_anchored_reproducibility.notna().sum()),"IC-stable feature-anchored sensitivity")
                keptd=d[d.model.isin(kept)]
                add(rows,"leave_one_model_out",space,population,removed,"dominant_top1",float(keptd.top1_exact_agreement.mean()),len(keptd))
                for definition in s.definition.unique():
                    # Recalculate sign after removing the model from paired long cells.
                    l=pd.read_csv(R3/"tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv"); l=l[(l.space==space)&l.model.isin(kept)]; ic=l[l.method=="IC"].set_index(["model","parameter_index","feature_index"])[["rho","bootstrap_sign_probability"]].rename(columns={"rho":"ic","bootstrap_sign_probability":"ic_stability"}); dp=l[l.method=="dPL"].set_index(["model","parameter_index","feature_index"])["rho"].rename("dpl"); pair=ic.join(dp);
                    if definition=="A_sign_all": mask=np.ones(len(pair),bool)
                    elif definition=="A_sign_nontrivial": mask=(pair.ic.abs()>=.10)|(pair.dpl.abs()>=.10)
                    elif definition=="A_sign_both_nontrivial": mask=(pair.ic.abs()>=.10)&(pair.dpl.abs()>=.10)
                    else: mask=(pair.ic.abs()>=.20)&(pair.ic_stability>=.95)
                    add(rows,"leave_one_model_out",space,population,removed,"sign_"+definition,float((np.sign(pair.ic[mask])==np.sign(pair.dpl[mask])).mean()),int(mask.sum()))
    median_boundary=float(ident.total_boundary_fraction.median());
    for cutoff in (.10,.20,.30):
        for space in SPACES:
            sub=profile[(profile.space==space)].merge(ident[["model","parameter_index","total_boundary_fraction"]],on=["model","parameter_index"],validate="many_to_one")
            kept=sub[sub.total_boundary_fraction<=cutoff]
            for population,models in (("all36",MODEL_ORDER),("exclude_simhyd",STRICT_MODELS)):
                vals=kept[kept.model.isin(models)].profile_reproducibility
                add(rows,"boundary_sensitivity",space,population,f"exclude_boundary_gt_{cutoff}","cell_equal_profile_median",float(vals.median()),len(vals),"parameter cells above cutoff excluded")
                anchored_vals=kept[kept.model.isin(models)].ic_anchored_profile_reproducibility
                add(rows,"boundary_sensitivity",space,population,f"exclude_boundary_gt_{cutoff}","ic_anchored_cell_median",float(anchored_vals.median()),int(anchored_vals.notna().sum()),"IC-stable feature-anchored parameter profiles")
    for space in SPACES:
        for subset, mask in (("high_identifiability",ident.restart_u_sd_median<=ident.restart_u_sd_median.median()),("low_identifiability",ident.restart_u_sd_median>ident.restart_u_sd_median.median())):
            keys=set(zip(ident.loc[mask,"model"],ident.loc[mask,"parameter_index"])); sub=profile[(profile.space==space)&profile.apply(lambda r:(r.model,r.parameter_index) in keys,axis=1)]
            add(rows,"identifiability_subset",space,"all36",subset,"cell_equal_profile_median",float(sub.profile_reproducibility.median()),len(sub),"split at pooled median restart_u_sd")
            anchored_sub=sub.ic_anchored_profile_reproducibility
            add(rows,"identifiability_subset",space,"all36",subset,"ic_anchored_cell_median",float(anchored_sub.median()),int(anchored_sub.notna().sum()),"IC-stable feature-anchored sensitivity")
    output=pd.DataFrame(rows); write_csv(R3_TABLES/"R3_POOLED_INFLUENCE.csv",output); write_json(R3/"cache/R3_INFLUENCE_MANIFEST.json",{"analysis":"R3-H pooled influence","spaces":list(SPACES),"populations":{"all36":list(MODEL_ORDER),"exclude_simhyd":list(STRICT_MODELS)},"model_equal":True,"cell_equal":True,"leave_one_model_out":True,"boundary_cutoffs":[.10,.20,.30],"identifiability_split":"pooled median restart_u_sd","runtime_seconds":time.time()-started})
    primary=output[(output.analysis=="pooled")&(output.space=="information_cluster")&(output.population=="all36")&(output.sensitivity=="model_equal_primary")].value.iloc[0]; loo=output[(output.analysis=="leave_one_model_out")&(output.space=="information_cluster")&(output.population=="all36")&(output.metric=="profile_reproducibility_model_equal")]
    result=f"Primary cluster-space model-equal reproducibility was {primary:.6f}; leave-one-model-out range was {loo.value.min():.6f}--{loo.value.max():.6f}. Influence table includes raw/cluster, model/cell weighting, all36/exclude_simhyd, LOO, boundary cutoffs, sign, dominant, and identifiability subsets."
    write_audit(R3/"R3_POOLED_INFLUENCE_AUDIT.md","R3-H Pooled Influence Audit","Are R3 reproducibility, sign, and dominant-information conclusions driven by weighting, SIMHYD, one model, boundary cells, or identifiability strata?",f"Primary profiles `{R3_TABLES/'R3_PARAMETER_PROFILE_REPRODUCIBILITY.csv'}`, sign `{R3_TABLES/'R3_SIGN_AGREEMENT_AUDIT.csv'}`, dominant tables, boundary and identifiability tables. All inputs are frozen IC/dPL post-processing.","Repeat profile reproducibility, sign agreement, and dominant top-k summaries in raw and cluster spaces under model-equal, cell-equal, all36, exclude_simhyd, leave-one-model-out, boundary-filtered, and high/low-identifiability definitions.","Model denominator is 36/35; profile-cell denominator is the actual common parameter count by model; sign/dominant denominators are explicit in source rows. No representative model is selected.","Every model is removed in turn. Boundary cutoffs .10/.20/.30 and identifiability split at pooled median restart-u SD are fixed before interpretation.",result,"Aggregation sensitivity can alter a headline without changing the underlying cell table. LOO only tests influence, not exchangeability. High/low identifiability subsets are descriptive and can be confounded with model family or parameter role.","A robust R3 claim requires the primary cluster profile correspondence and IC-stable sign signal to remain directionally similar across these sensitivity families and to exceed the cross-estimator null. dPL-emergent cells cannot rescue a failed primary result.","R3_H_READY",time.time()-started)


if __name__=="__main__": main()
