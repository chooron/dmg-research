#!/usr/bin/env python3
"""Build raw-attribute and information-cluster IC/dPL relationship matrices.

IC and dPL values come only from the R2 canonical table.  Both estimators use the
same 531-basin Spearman estimator and normalized parameter coordinate.  dPL's
attribute-to-parameter construction is documented as descriptive and is never
called independent validation.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from r3_common import (
    ATTRIBUTE_TYPES, CAMELS_35_ATTRIBUTES, MODEL_ORDER, N_BASINS, R2, R3, R3_CACHE,
    SEED, get_spec, load_attributes, load_canonical_table, load_ids, rank_bootstrap_matrix,
    spearman_rho_p, write_audit, write_csv, write_json,
)

CLUSTER_THRESHOLD = 0.70


def parse_models(value: str | None) -> tuple[str, ...]:
    if not value:
        return MODEL_ORDER
    models = tuple(x.strip() for x in value.split(",") if x.strip())
    if not models or not set(models).issubset(set(MODEL_ORDER)):
        raise ValueError(f"unknown model in {models}")
    return models


def feature_arrays(ids: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    attrs, _, _ = load_attributes(ids)
    raw_names = list(CAMELS_35_ATTRIBUTES)
    score = pd.read_parquet(R2 / "cache/information_cluster_scores.parquet")
    score = score[score.threshold == CLUSTER_THRESHOLD]
    wide = score.pivot(index="basin_id", columns="cluster_id", values="score_pc1").reindex(ids)
    if wide.shape[0] != N_BASINS or wide.isna().any().any():
        raise RuntimeError("cluster score matrix is not complete")
    cluster_names = sorted(wide.columns)
    return attrs, raw_names, wide[cluster_names].to_numpy(float), cluster_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=None)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--output-name", default="R3_IC_DPL_RELATIONSHIP_MATRICES_LONG.csv")
    args = parser.parse_args()
    if args.n_boot < 20 or (not args.models and args.n_boot < 1000):
        raise ValueError("formal run requires n_boot>=1000; smaller runs require --models smoke subset")
    started = time.time(); models = parse_models(args.models); ids = load_ids(); attrs, raw_names, clusters, cluster_names = feature_arrays(ids)
    canonical = load_canonical_table(); long_rows = []
    for model in models:
        spec = get_spec(model, device="cpu")
        matrices = {}
        for method, value_col in (("IC", "normalized_parameter_ic"), ("dPL", "normalized_parameter_dpl")):
            sub = canonical[canonical.model == model]
            theta = sub.pivot(index="basin_id", columns="parameter_index", values=value_col).reindex([str(int(x)).zfill(8) for x in ids]).to_numpy(float)
            if theta.shape != (N_BASINS, spec.dimension) or not np.isfinite(theta).all():
                raise RuntimeError(f"{model}/{method}: invalid normalized parameter matrix")
            feature_matrix = np.column_stack([attrs, clusters])
            feature_names = raw_names + cluster_names
            feature_spaces = (["raw_attribute"] * len(raw_names)) + (["information_cluster"] * len(cluster_names))
            boot = rank_bootstrap_matrix(feature_matrix, theta, n_boot=args.n_boot, seed=SEED + (1000 if method == "dPL" else 0) + list(MODEL_ORDER).index(model))
            point = np.empty((len(feature_names), spec.dimension)); pvalues = np.empty_like(point)
            for fi in range(len(feature_names)):
                for p in range(spec.dimension):
                    point[fi, p], pvalues[fi, p], n = spearman_rho_p(feature_matrix[:, fi], theta[:, p])
            matrices[method] = point
            cache_dir = R3_CACHE / ("dpl_relationship_matrices" if method == "dPL" else "ic_relationship_matrices")
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_dir / f"{model}.npz", bootstrap_rho=boot, point_rho=point, p_value=pvalues, feature_names=np.asarray(feature_names), feature_spaces=np.asarray(feature_spaces), parameter_names=np.asarray(spec.parameter_names), seed=SEED + (1000 if method == "dPL" else 0) + list(MODEL_ORDER).index(model), n_boot=args.n_boot, cluster_threshold=CLUSTER_THRESHOLD, method=method)
            for fi, (feature, space) in enumerate(zip(feature_names, feature_spaces)):
                space_index = fi if space == "raw_attribute" else fi - len(raw_names)
                for p, parameter in enumerate(spec.parameter_names):
                    ci_low, ci_high = np.nanquantile(boot[:, fi, p], [.025, .975])
                    pos = float(np.nanmean(boot[:, fi, p] > 0)); neg = float(np.nanmean(boot[:, fi, p] < 0))
                    long_rows.append({
                        "model": model, "method": method, "space": space, "feature_index": space_index, "feature": feature,
                        "parameter_index": p, "parameter": parameter, "rho": float(point[fi, p]), "p_value": float(pvalues[fi, p]), "n": N_BASINS,
                        "bootstrap_ci_low": float(ci_low), "bootstrap_ci_high": float(ci_high), "bootstrap_positive_probability": pos,
                        "bootstrap_negative_probability": neg, "bootstrap_sign_probability": max(pos, neg), "bootstrap_ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
                        "parameter_coordinate": "normalized_u", "n_boot": args.n_boot, "seed": SEED + (1000 if method == "dPL" else 0) + list(MODEL_ORDER).index(model),
                        "dpl_independence_status": "constructed_X_to_theta_descriptive_only" if method == "dPL" else "IC_independent_parameter_fit",
                    })
    output = pd.DataFrame(long_rows)
    write_csv(R3 / "tables" / args.output_name, output)
    write_json(R3_CACHE / (Path(args.output_name).stem + "_manifest.json"), {
        "analysis": "R3-A IC/dPL relationship matrices", "models": list(models), "n_models": len(models), "n_basins": N_BASINS,
        "spaces": ["raw_attribute", "information_cluster"], "cluster_threshold": CLUSTER_THRESHOLD, "n_boot": args.n_boot, "seed": SEED,
        "bootstrap": "basin resampling with fixed full-sample average ranks; exact scipy average-tie Spearman point estimate",
        "source_canonical_table": str(R2 / "cache/canonical_parameter_attribute_table.parquet"), "runtime_seconds": time.time() - started,
    })
    result = f"Built {len(output)} relationship rows for {len(models)} models, IC and dPL, in raw 35-attribute and primary 0.70 information-cluster spaces. Both estimators have exact point rho/p and {args.n_boot}-replicate basin-bootstrap CI/sign summaries."
    write_audit(R3 / "R3_RELATIONSHIP_MATRIX_AUDIT.md", "R3-A Relationship Matrix Audit", "Do IC and dPL produce comparable relationship matrices in the same raw-attribute and information-cluster spaces?", f"Canonical table `{R2 / 'cache/canonical_parameter_attribute_table.parquet'}`; canonical Caravan attributes and 531 basin IDs; dPL source is canonical v2 seed 42. IC is independently fitted; dPL is an X-to-theta construction and is not independent evidence by itself.", "For each estimator × model × parameter × feature, Spearman rho between normalized parameter u and feature, n, p, and basin-bootstrap CI/sign stability. Feature spaces are raw 35 attributes and frozen threshold-0.70 cluster PC1 scores.", "Each cell has n=531 finite basins; parameter count follows the current model registry. No missing cells are silently dropped. Raw and cluster spaces are kept as separate estimands.", "Bootstrap tensors are saved per model and estimator for profile, sign, and dominant-control audits. No checkpoint is retrained or modified.", result, "Monotone physical mapping does not alter rank rho, but normalized u makes the estimator coordinate explicit. dPL rows remain construction-linked and are used later only after IC anchoring.", "A reviewer can call any dPL relationship tautological because dPL was trained from attributes. The R3 artifact audit therefore classifies dPL-strong/IC-weak cells as dPL-emergent and excludes them from persistence claims.", "R3_A_READY", time.time() - started)


if __name__ == "__main__":
    main()
