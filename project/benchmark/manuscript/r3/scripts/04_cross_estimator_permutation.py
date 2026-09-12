#!/usr/bin/env python3
"""Run the R3 cross-estimator basin-correspondence permutation null.

For each model, dPL normalized parameter rows are jointly permuted across the 531
basin labels, preserving dPL parameter marginals while breaking IC/dPL basin
correspondence. The unfiltered and IC-anchored information-cluster profile
estimands use identical permutations and aggregation; the anchored estimand
restricts features using IC effect and IC bootstrap stability only.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from r3_common import MODEL_ORDER, N_BASINS, R2, R3, R3_CACHE, R3_TABLES, SEED, STRICT_MODELS, get_spec, load_attributes, load_canonical_table, load_ids, relationship_matrix, write_audit, write_csv, write_json

CLUSTER_THRESHOLD = 0.70
CHUNK = 25


def profile(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.unique(a[ok]).size < 2 or np.unique(b[ok]).size < 2:
        return np.nan
    return float(spearmanr(a[ok], b[ok]).statistic)


def prepare() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, list[str]]:
    ids = load_ids(); canonical = load_canonical_table(); _, _, _ = load_attributes(ids)
    score = pd.read_parquet(R2 / "cache/information_cluster_scores.parquet")
    score = score[score.threshold == CLUSTER_THRESHOLD]
    cluster_ids = sorted(score.cluster_id.unique())
    score_wide = score.pivot(index="basin_id", columns="cluster_id", values="score_pc1").reindex(ids)
    if score_wide.shape != (N_BASINS, len(cluster_ids)) or score_wide.isna().any().any():
        raise RuntimeError("cluster score matrix is not complete")
    feature_rank = np.column_stack([rankdata(score_wide[c].to_numpy(float), method="average") for c in cluster_ids])
    theta_rank = {}; theta_values = {}
    for model in MODEL_ORDER:
        spec = get_spec(model, device="cpu")
        sub = canonical[canonical.model == model]
        values = sub.pivot(index="basin_id", columns="parameter_index", values="normalized_parameter_dpl").reindex([str(int(x)).zfill(8) for x in ids]).to_numpy(float)
        if values.shape != (N_BASINS, spec.dimension) or not np.isfinite(values).all():
            raise RuntimeError(f"invalid dPL matrix {model}")
        theta_values[model] = values
        theta_rank[model] = np.column_stack([rankdata(values[:, p], method="average") for p in range(spec.dimension)])
    return theta_values, theta_rank, feature_rank, ids, cluster_ids


def permuted_model_profiles(indices: np.ndarray, theta_rank: np.ndarray, feature_rank: np.ndarray, ic_profile: np.ndarray, anchor_mask: np.ndarray) -> tuple[float, float]:
    y = theta_rank[indices]; yc = y - y.mean(axis=0, keepdims=True)
    x = feature_rank - feature_rank.mean(axis=0, keepdims=True)
    numerator = x.T @ yc
    denom = np.sqrt(np.einsum("nk,nk->k", x, x)[:, None] * np.einsum("np,np->p", yc, yc)[None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        dpl_rho = (numerator / denom).T
    full_values = [profile(ic_profile[p], dpl_rho[p]) for p in range(ic_profile.shape[0])]
    anchored_values = [profile(ic_profile[p, anchor_mask[p]], dpl_rho[p, anchor_mask[p]]) if int(anchor_mask[p].sum()) >= 3 else np.nan for p in range(ic_profile.shape[0])]
    return float(np.nanmedian(full_values)), float(np.nanmedian(anchored_values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--n-perm", type=int, default=1000); parser.add_argument("--seed", type=int, default=SEED); args = parser.parse_args()
    if args.n_perm < 20 or args.n_perm > 1000: raise ValueError("formal bounded permutation count is 20..1000")
    started = time.time(); _, dpl_rank, feature_rank, _, cluster_ids = prepare()
    long = pd.read_csv(R3 / "tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv")
    ic = relationship_matrix(long, "IC", "information_cluster")
    ic_long = long[(long.method == "IC") & (long.space == "information_cluster")]
    anchor_mask = {}
    for model in MODEL_ORDER:
        sub = ic_long[ic_long.model == model].sort_values(["parameter_index", "feature_index"])
        matrix = sub.pivot(index="parameter_index", columns="feature_index", values="rho").reindex(index=range(ic[model].shape[0]), columns=range(len(cluster_ids)))
        stability = sub.pivot(index="parameter_index", columns="feature_index", values="bootstrap_sign_probability").reindex(index=range(ic[model].shape[0]), columns=range(len(cluster_ids)))
        anchor_mask[model] = (matrix.abs().to_numpy(float) >= .20) & (stability.to_numpy(float) >= .95)
    model_repro = pd.read_csv(R3_TABLES / "R3_MODEL_REPRODUCIBILITY.csv"); overall = pd.read_csv(R3_TABLES / "R3_OVERALL_REPRODUCIBILITY.csv")
    observed_full = {m: float(model_repro.query("space == 'information_cluster' and model == @m").model_reproducibility.iloc[0]) for m in MODEL_ORDER}
    observed_anchor = {m: float(model_repro.query("space == 'information_cluster' and model == @m").ic_anchored_reproducibility.iloc[0]) for m in MODEL_ORDER}
    observed_overall = {"all36": float(overall.query("space == 'information_cluster' and population == 'all36' and estimand == 'parameter_profile_model_equal_median'").value.iloc[0]), "exclude_simhyd": float(overall.query("space == 'information_cluster' and population == 'exclude_simhyd' and estimand == 'parameter_profile_model_equal_median'").value.iloc[0])}
    observed_anchor_overall = {"all36": float(overall.query("space == 'information_cluster' and population == 'all36' and estimand == 'ic_anchored_profile_model_equal_median'").value.iloc[0]), "exclude_simhyd": float(overall.query("space == 'information_cluster' and population == 'exclude_simhyd' and estimand == 'ic_anchored_profile_model_equal_median'").value.iloc[0])}
    cache_dir = R3_CACHE / "cross_estimator_permutation"; cache_dir.mkdir(parents=True, exist_ok=True); final = cache_dir / f"dpl_basin_correspondence_null_v3_{args.n_perm}_{args.seed}.npz"
    if final.exists():
        cached=np.load(final); full_null=cached["all36_full"]; strict_full=cached["exclude_simhyd_full"]; anchor_null=cached["all36_anchor"]; strict_anchor=cached["exclude_simhyd_anchor"]; full_model=cached["all36_model"]; strict_model=cached["exclude_simhyd_model"]; anchor_model=cached["all36_anchor_model"]; strict_anchor_model=cached["exclude_simhyd_anchor_model"]
    else:
        rng=np.random.default_rng(args.seed); full_values=[]; strict_values=[]; anchor_values=[]; strict_anchor_values=[]; full_models=[]; strict_models=[]; anchor_models=[]; strict_anchor_models=[]
        for _ in range(args.n_perm):
            full_row=[]; anchor_row=[]
            for model in MODEL_ORDER:
                idx=rng.permutation(N_BASINS); f,a=permuted_model_profiles(idx,dpl_rank[model],feature_rank,ic[model],anchor_mask[model]); full_row.append(f); anchor_row.append(a)
            strict_row=[v for v,m in zip(full_row,MODEL_ORDER) if m!='simhyd']; strict_anchor_row=[v for v,m in zip(anchor_row,MODEL_ORDER) if m!='simhyd']
            full_models.append(full_row); strict_models.append(strict_row); anchor_models.append(anchor_row); strict_anchor_models.append(strict_anchor_row); full_values.append(np.nanmedian(full_row)); strict_values.append(np.nanmedian(strict_row)); anchor_values.append(np.nanmedian(anchor_row)); strict_anchor_values.append(np.nanmedian(strict_anchor_row))
        full_null=np.asarray(full_values); strict_full=np.asarray(strict_values); anchor_null=np.asarray(anchor_values); strict_anchor=np.asarray(strict_anchor_values); full_model=np.asarray(full_models); strict_model=np.asarray(strict_models); anchor_model=np.asarray(anchor_models); strict_anchor_model=np.asarray(strict_anchor_models)
        np.savez_compressed(final,all36_full=full_null,exclude_simhyd_full=strict_full,all36_anchor=anchor_null,exclude_simhyd_anchor=strict_anchor,all36_model=full_model,exclude_simhyd_model=strict_model,all36_anchor_model=anchor_model,exclude_simhyd_anchor_model=strict_anchor_model,cluster_ids=np.asarray(cluster_ids),seed=args.seed,n_perm=args.n_perm)
    rows=[]
    for population, full, anchor, model_null, anchor_model_null in (("all36",full_null,anchor_null,full_model,anchor_model),("exclude_simhyd",strict_full,strict_anchor,strict_model,strict_anchor_model)):
        observed=observed_overall[population]; observed_anchor_value=observed_anchor_overall[population]
        for estimand, value, null_values in (("parameter_profile_model_equal_median",observed,full),("ic_anchored_profile_model_equal_median",observed_anchor_value,anchor)):
            rows.append({"space":"information_cluster","population":population,"estimand":estimand,"observed":value,"null_mean":float(null_values.mean()),"null_sd":float(null_values.std(ddof=1)),"null_q025":float(np.quantile(null_values,.025)),"null_q975":float(np.quantile(null_values,.975)),"empirical_p":float((1+np.sum(null_values>=value))/(args.n_perm+1)),"effect_above_null":float(value-null_values.mean()),"n_permutations":args.n_perm,"seed":args.seed,"null_definition":"joint permutation of dPL normalized parameter rows across basin labels within each model; IC-anchored variant restricts features by IC effect/sign-bootstrap gate"})
        max_full=model_null.max(axis=1); max_anchor=anchor_model_null.max(axis=1); observed_max=max(observed_full[m] for m in (MODEL_ORDER if population=='all36' else STRICT_MODELS)); observed_anchor_max=max(observed_anchor[m] for m in (MODEL_ORDER if population=='all36' else STRICT_MODELS))
        for estimand, value, null_values in (("maximum_model_reproducibility",observed_max,max_full),("maximum_ic_anchored_model_reproducibility",observed_anchor_max,max_anchor)):
            rows.append({"space":"information_cluster","population":population,"estimand":estimand,"observed":value,"null_mean":float(null_values.mean()),"null_sd":float(null_values.std(ddof=1)),"null_q025":float(np.quantile(null_values,.025)),"null_q975":float(np.quantile(null_values,.975)),"empirical_p":float((1+np.sum(null_values>=value))/(args.n_perm+1)),"effect_above_null":float(value-null_values.mean()),"n_permutations":args.n_perm,"seed":args.seed,"null_definition":"maximum across model-level profiles under the same dPL basin-label permutation"})
    result=pd.DataFrame(rows); write_csv(R3_TABLES/"R3_CROSS_ESTIMATOR_NULL.csv",result); write_json(R3_CACHE/"R3_CROSS_ESTIMATOR_NULL_MANIFEST.json",{"analysis":"R3-E cross-estimator permutation","n_permutations":args.n_perm,"seed":args.seed,"spaces":["information_cluster"],"cluster_threshold":CLUSTER_THRESHOLD,"estimands":["unfiltered","IC-anchored"],"dpl_parameter_marginals_preserved":True,"ic_dpl_basin_correspondence_broken":True,"attribute_clustering_unchanged":True,"runtime_seconds":time.time()-started,"cache":str(final)})
    primary=result[(result.population=='all36')&(result.estimand=='parameter_profile_model_equal_median')].iloc[0]; anchored_row=result[(result.population=='all36')&(result.estimand=='ic_anchored_profile_model_equal_median')].iloc[0]
    result_text=f"The primary unfiltered null used {args.n_perm} joint dPL basin-label permutations: observed={primary.observed:.6f}, null mean={primary.null_mean:.6f}, 95% interval=[{primary.null_q025:.6f},{primary.null_q975:.6f}], p={primary.empirical_p:.6f}. The IC-anchored sensitivity was observed={anchored_row.observed:.6f}, null mean={anchored_row.null_mean:.6f}, p={anchored_row.empirical_p:.6f}. Max-model and exclude_simhyd rows are also saved."
    write_audit(R3/"R3_CROSS_ESTIMATOR_NULL_AUDIT.md","R3-E Cross-Estimator Null Audit","Is IC--dPL relationship-profile correspondence higher than expected after destroying basin-level parameter correspondence while preserving estimator marginals?",f"IC information-cluster profiles `{R3 / 'tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv'}`; dPL normalized parameter rows from the canonical table; frozen cluster threshold {CLUSTER_THRESHOLD:.2f}.","The unfiltered primary is median over models of median over parameters of Spearman(IC profile,dPL profile). An IC-anchored sensitivity uses only information-cluster features with IC |rho|>=0.20 and IC bootstrap sign probability>=0.95; dPL does not select its denominator. Both are applied identically to observed and permuted data.","Basin denominator is 531 within each model; model denominator is 36 or 35. dPL rows are jointly permuted within model, preserving parameter marginals and eligible parameter counts; clustering and thresholds are not recomputed.","Permutation results are cached and report unfiltered/IC-anchored overall and maximum-model estimands, null intervals, empirical p, and effects above null for all36 and exclude_simhyd.",result_text,"The null does not test every possible estimator dependence and uses fixed-rank Spearman permutation computation. It is a correspondence null, not a causal null. The anchored sensitivity is the more defensible persistence check when dPL construction is the principal adversarial concern.","Because dPL is constructed from attributes, an above-null unfiltered result is not independent hydrological validation. Only IC-anchored/supportive cells can carry the narrower persistence interpretation.","R3_E_READY_WITH_ANCHORED_SENSITIVITY",time.time()-started)


if __name__ == "__main__": main()
