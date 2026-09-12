#!/usr/bin/env python3
"""Audit whether dPL-strong relationships are independently supported by IC.

Because dPL is an attribute-to-parameter network, a strong dPL association is not
an independent hydrological validation. This script classifies every raw and
cluster relationship by the IC anchor and labels IC-weak/absent dPL-strong cells
as dPL-emergent.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from r3_common import MODEL_ORDER, R3, R3_TABLES, STRICT_MODELS, write_audit, write_csv, write_json

SPACES = ("raw_attribute", "information_cluster")


def classify(ic: float, dpl: float, threshold: float) -> str:
    if abs(dpl) < threshold: return "dPL-not-strong"
    if abs(ic) >= threshold and np.sign(ic) == np.sign(dpl): return "IC-supported"
    if abs(ic) >= threshold and np.sign(ic) != np.sign(dpl): return "IC-opposite-sign"
    if abs(ic) >= .10: return "IC-weak"
    return "IC-absent"


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--effect-threshold",type=float,default=.20); args=parser.parse_args()
    started=time.time(); long=pd.read_csv(R3/"tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv"); rows=[]
    for space in SPACES:
        ic=long[(long.space==space)&(long.method=="IC")].set_index(["model","parameter_index","feature_index"])
        dpl=long[(long.space==space)&(long.method=="dPL")].set_index(["model","parameter_index","feature_index"])
        paired=ic[["feature","rho","bootstrap_sign_probability"]].rename(columns={"feature":"feature_ic","rho":"rho_ic","bootstrap_sign_probability":"ic_sign_stability"}).join(dpl[["feature","rho","bootstrap_sign_probability"]].rename(columns={"feature":"feature_dpl","rho":"rho_dpl","bootstrap_sign_probability":"dpl_sign_stability"}))
        for key,item in paired.iterrows():
            ic_r=float(item.rho_ic); dpl_r=float(item.rho_dpl); category=classify(ic_r,dpl_r,args.effect_threshold)
            rows.append({"space":space,"model":key[0],"parameter_index":int(key[1]),"feature_index":int(key[2]),"feature_ic":item.feature_ic,"feature_dpl":item.feature_dpl,"rho_ic":ic_r,"rho_dpl":dpl_r,"ic_bootstrap_sign_stability":item.ic_sign_stability,"dpl_bootstrap_sign_stability":item.dpl_sign_stability,"dpl_strong":abs(dpl_r)>=args.effect_threshold,"classification":category,"dpl_emergent":category in {"IC-weak","IC-absent","IC-opposite-sign"},"eligible_for_cross_estimator_claim":category=="IC-supported"})
    output=pd.DataFrame(rows); write_csv(R3_TABLES/"R3_DPL_CONSTRUCTION_ARTIFACT_AUDIT.csv",output)
    summary=output[output.dpl_strong].groupby(["space","classification"],as_index=False).agg(cells=("dpl_strong","size"),models=("model","nunique")); summary["fraction_of_dpl_strong_space"]=summary.cells/summary.groupby("space").cells.transform("sum"); write_csv(R3_TABLES/"R3_DPL_CONSTRUCTION_ARTIFACT_SUMMARY.csv",summary)
    write_json(R3/"cache/R3_DPL_ARTIFACT_MANIFEST.json",{"analysis":"R3-G dPL construction artifact audit","classification":"dPL strong >=.20; IC-supported >=.20 same sign; IC-opposite-sign >=.20 opposite; IC-weak .10-.20; IC-absent <.10","spaces":list(SPACES),"runtime_seconds":time.time()-started})
    strong=output[output.dpl_strong]; supported=float((strong.classification=="IC-supported").mean()) if len(strong) else np.nan; emergent=float((strong.classification!="IC-supported").mean()) if len(strong) else np.nan
    result=f"Among {len(strong)} dPL-strong cells across both spaces, {supported:.6f} were IC-supported and {emergent:.6f} were not independently supported (IC-weak/absent/opposite-sign). Only IC-supported cells are eligible for cross-estimator persistence claims."
    write_audit(R3/"R3_DPL_CONSTRUCTION_ARTIFACT_AUDIT.md","R3-G dPL Construction-Artifact Audit","Are strong dPL attribute--parameter relationships independently supported by IC, or are they inevitable consequences of the dPL construction?",f"Paired relationship matrices `{R3 / 'tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv'}` for raw and frozen information-cluster spaces; IC is independently fitted and dPL is an X-to-theta network.","Classify dPL-strong cells at |rho_dPL|>=0.20 as IC-supported (IC |rho|>=0.20, same sign), IC-opposite-sign, IC-weak (0.10≤|rho_IC|<0.20), or IC-absent (<0.10). dPL-emergent includes all non-supported strong cells.","Denominator is every paired model×parameter×feature cell for each space; strong-cell summaries use only dPL-strong cells. all36 and exclude_simhyd can be filtered by model; primary classification does not select dPL-emergent evidence.","The same thresholds and paired cells are used in both spaces. Bootstrap sign stability is retained as context but does not make dPL independent.",result,"The dPL mapping can produce strong relationships by construction. A dPL-strong/IC-weak or absent cell cannot support shared hydrological information; an opposite-sign cell is disagreement, not evidence for dPL. Even IC-supported cells remain association-level evidence.","A reviewer can argue that IC and dPL share data/forcing and are not independent in a broad statistical sense. The correct claim is persistence of an IC relationship under a different parameter-estimation procedure, not independent replication of a causal law. Only IC-supported cells enter the R3 persistence interpretation; dPL-emergent rows remain an explicit artifact audit and are not promoted to validation evidence.","R3_G_READY",time.time()-started)


if __name__=="__main__": main()
