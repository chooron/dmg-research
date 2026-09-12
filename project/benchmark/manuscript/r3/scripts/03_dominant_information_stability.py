#!/usr/bin/env python3
"""Separate raw top-attribute disagreement from information-cluster agreement.

Top-1/3/5 overlap is computed in raw 35-attribute and frozen information-cluster
spaces. Bootstrap top-1 stability is computed independently for IC and dPL from
the saved relationship bootstrap tensors, so an unstable estimator is not mistaken
for cross-estimator disagreement. Raw top-1 mismatches within the same cluster are
flagged as correlated-proxy substitution.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from r3_common import CAMELS_35_ATTRIBUTES, MODEL_ORDER, R2, R3, R3_CACHE, R3_TABLES, relationship_matrix, write_audit, write_csv, write_json

SPACES = ("raw_attribute", "information_cluster")
CLUSTER_THRESHOLD = 0.70


def ordered_indices(values: np.ndarray) -> np.ndarray:
    return np.lexsort((np.arange(len(values)), -np.abs(values)))


def overlap(a: np.ndarray, b: np.ndarray, k: int) -> tuple[int, float]:
    sa = set(a[:k]); sb = set(b[:k]); intersection = len(sa & sb); union = len(sa | sb)
    return intersection, intersection / union if union else np.nan


def top1_bootstrap(path, space_index: int, point_index: int, point_top: int) -> float:
    cached = np.load(path, allow_pickle=True)
    boot = cached["bootstrap_rho"][:, :, point_index]
    if space_index == 0:
        feature_boot = boot[:, :35]
    else:
        feature_boot = boot[:, 35:]
    selected = np.argmax(np.abs(feature_boot), axis=1)
    return float(np.mean(selected == point_top))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-threshold", type=float, default=CLUSTER_THRESHOLD)
    args = parser.parse_args()
    started = time.time(); long = pd.read_csv(R3 / "tables/R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv")
    cluster_assign = pd.read_csv(R2 / "tables/R2_INFORMATION_CLUSTERS.csv")
    cluster_assign = cluster_assign[cluster_assign.threshold == args.cluster_threshold].set_index("attribute").cluster_id.to_dict()
    raw_rows = []; cluster_rows = []; proxy_rows = []
    for space in SPACES:
        ic = relationship_matrix(long, "IC", space); dpl = relationship_matrix(long, "dPL", space)
        feature_names = long[(long.space == space) & (long.method == "IC")].drop_duplicates("feature_index").sort_values("feature_index").feature.tolist()
        for model in MODEL_ORDER:
            ic_path = R3_CACHE / "ic_relationship_matrices" / f"{model}.npz"; dpl_path = R3_CACHE / "dpl_relationship_matrices" / f"{model}.npz"
            for p in range(ic[model].shape[0]):
                io = ordered_indices(ic[model][p]); do = ordered_indices(dpl[model][p])
                inter3, jac3 = overlap(io, do, 3); inter5, jac5 = overlap(io, do, 5)
                top1_ic = feature_names[io[0]]; top1_dpl = feature_names[do[0]]
                row = {"space": space, "model": model, "parameter_index": p, "parameter": long[(long.method == "IC") & (long.space == space) & (long.model == model) & (long.parameter_index == p)].parameter.iloc[0], "ic_top1": top1_ic, "dpl_top1": top1_dpl, "top1_exact_agreement": bool(io[0] == do[0]), "top3_overlap_count": inter3, "top3_jaccard": jac3, "top5_overlap_count": inter5, "top5_jaccard": jac5, "ic_top1_bootstrap_stability": top1_bootstrap(ic_path, 0 if space == "raw_attribute" else 1, p, io[0]), "dpl_top1_bootstrap_stability": top1_bootstrap(dpl_path, 0 if space == "raw_attribute" else 1, p, do[0]), "feature_count": len(feature_names), "population": "all36"}
                (raw_rows if space == "raw_attribute" else cluster_rows).append(row)
                if space == "raw_attribute":
                    ci = cluster_assign.get(top1_ic); cd = cluster_assign.get(top1_dpl)
                    proxy_rows.append({"model": model, "parameter_index": p, "parameter": row["parameter"], "ic_top1_attribute": top1_ic, "dpl_top1_attribute": top1_dpl, "ic_top1_cluster": ci, "dpl_top1_cluster": cd, "raw_top1_mismatch": not row["top1_exact_agreement"], "same_information_cluster": bool(ci == cd), "correlated_proxy_substitution": bool((not row["top1_exact_agreement"]) and ci == cd), "ic_top3_attributes": ";".join(feature_names[x] for x in io[:3]), "dpl_top3_attributes": ";".join(feature_names[x] for x in do[:3])})
    raw = pd.DataFrame(raw_rows); clusters = pd.DataFrame(cluster_rows); proxy = pd.DataFrame(proxy_rows)
    write_csv(R3_TABLES / "R3_DOMINANT_ATTRIBUTE_AGREEMENT.csv", raw)
    write_csv(R3_TABLES / "R3_DOMINANT_CLUSTER_AGREEMENT.csv", clusters)
    write_csv(R3_TABLES / "R3_CORRELATED_PROXY_SUBSTITUTION.csv", proxy)
    write_json(R3_CACHE / "R3_DOMINANT_MANIFEST.json", {"analysis": "R3-D dominant information stability", "cluster_threshold": args.cluster_threshold, "top_k": [1, 3, 5], "tie_rule": "stable feature-index order after descending absolute rho", "bootstrap_top1": "fraction of saved relationship bootstrap resamples selecting point top-1 feature", "proxy_substitution": "raw top-1 mismatch but same frozen information-cluster assignment", "runtime_seconds": time.time() - started})
    raw_top1 = raw.top1_exact_agreement.mean(); cluster_top1 = clusters.top1_exact_agreement.mean(); proxy_rate = proxy.correlated_proxy_substitution.mean(); raw_mismatch = proxy.raw_top1_mismatch.mean()
    result = f"Raw top-1 exact agreement was {raw_top1:.6f}; information-cluster top-1 agreement was {cluster_top1:.6f}. {proxy_rate:.6f} of all raw model-parameter pairs were within-cluster proxy substitutions, representing {proxy_rate / max(raw_mismatch, 1e-12):.6f} of raw top-1 mismatches. Bootstrap IC/dPL top-1 stability is reported per pair."
    write_audit(R3 / "R3_DOMINANT_INFORMATION_STABILITY_AUDIT.md", "R3-D Dominant Information Stability Audit", "How much raw dominant-attribute disagreement is genuine versus substitution among correlated proxies, and is each estimator's dominant choice itself stable?", f"Relationship matrix caches `{R3_CACHE / 'ic_relationship_matrices'}` and `{R3_CACHE / 'dpl_relationship_matrices'}`; cluster assignments `{R2 / 'tables/R2_INFORMATION_CLUSTERS.csv'}` at threshold {args.cluster_threshold:.2f}.", "For each model × common parameter, compare IC and dPL top-1 exact choice, Top-3/Top-5 overlap and Jaccard in raw and cluster spaces. Separately estimate IC and dPL bootstrap top-1 stability relative to each point top-1.", "Denominator is 271 model-parameter pairs for all36; raw feature count=35, cluster feature count is frozen from threshold 0.70. Proxy substitution is only a raw top-1 mismatch whose two descriptors share a precomputed cluster.", "Stable top-k calculations use deterministic descending absolute-rho order with feature-index tie break. Bootstrap top-1 selection is calculated from the saved 1000-replicate relationship tensors for IC and dPL independently.", result, "Raw top-1 can be low even when Top-3 or cluster agreement is high. Conversely, low estimator-specific bootstrap stability means a mismatch is not interpretable as estimator disagreement. Cluster-level agreement is a broader information estimand, not proof of identical parameter expression.", "A reviewer can argue that clusters were selected from the same attribute matrix and may overstate coherence; they were fixed before parameter analysis, and the raw/cluster/proxy tables remain separate. dPL-emergent cells are not treated as independent hydrological validation.", "R3_D_READY", time.time() - started)


if __name__ == "__main__":
    main()
