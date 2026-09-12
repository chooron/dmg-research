#!/usr/bin/env python3
"""Shared R3 helpers and frozen paths; imports the audited R2 contract."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[5]
R2_SCRIPTS = REPO / "project/benchmark/manuscript/r2/scripts"
sys.path.insert(0, str(R2_SCRIPTS))
from r2_common import *  # noqa: F401,F403,E402

R3 = REPO / "project/benchmark/manuscript/r3"
R3_CACHE = R3 / "cache"
R3_TABLES = R3 / "tables"
R3_FIGURES = R3 / "figures"

PRIMARY_CLUSTER_THRESHOLD = 0.70
PRIMARY_EFFECT_THRESHOLD = 0.20
NONTRIVIAL_THRESHOLD = 0.10
IC_STABLE_SIGN_THRESHOLD = 0.95


def relationship_matrix(long: Any, method: str, space: str, value: str = "rho") -> dict[str, Any]:
    frame = long[(long.method == method) & (long.space == space)]
    out = {}
    for model in MODEL_ORDER:
        sub = frame[frame.model == model]
        if sub.empty:
            raise RuntimeError(f"missing relationship matrix for {model}/{method}/{space}")
        params = sorted(sub.parameter_index.unique())
        features = sorted(sub.feature_index.unique())
        pivot = sub.pivot(index="parameter_index", columns="feature_index", values=value).reindex(index=params, columns=features)
        if pivot.isna().any().any():
            raise RuntimeError(f"missing matrix values for {model}/{method}/{space}")
        out[model] = pivot.to_numpy(float)
    return out


def model_feature_names(long: Any, method: str, space: str) -> dict[int, str]:
    frame = long[(long.method == method) & (long.space == space)]
    return frame.drop_duplicates("feature_index").set_index("feature_index").feature.to_dict()


def spearman_profile(x: Any, y: Any) -> float:
    from scipy.stats import spearmanr
    import numpy as np
    a = np.asarray(x, dtype=float); b = np.asarray(y, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.unique(a[ok]).size < 2 or np.unique(b[ok]).size < 2:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).statistic)


def write_r3_audit(path: Path, title: str, question: str, data: str, estimand: str, denominator: str, method: str, result: str, sensitivity: str, adversarial: str, verdict: str, runtime_s: float) -> None:
    write_audit(path, title, question, data, estimand, denominator, method, result, sensitivity, adversarial, verdict, runtime_s)
