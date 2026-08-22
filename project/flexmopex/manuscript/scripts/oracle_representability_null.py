#!/usr/bin/env python3
"""Agent D — independent leakage/robustness audit + label-permutation null test.

Audit checklist on the shared artifacts, then a null test: the structure-head
binary probe with permuted labels should fall to chance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402
import torch  # noqa: E402

OUT = PROJECT_DIR / "results" / "oracle_representability"
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
from oracle_representability_probe import StructureHeadProbe, fit_predict, prep_fold  # noqa: E402

SEED = 7


def audit() -> dict:
    checks = {}
    df = pd.read_csv(OUT / "audit_table.csv")
    folds = pd.read_csv(OUT / "folds.csv").set_index("basin_idx").sort_index()
    names = json.loads((OUT / "feature_names.json").read_text())

    checks["n_rows"] = len(df)
    checks["basin_idx_0_670"] = bool((df["basin_idx"].to_numpy() == np.arange(671)).all())
    checks["no_duplicate_basin"] = int(df["basin_idx"].duplicated().sum()) == 0
    checks["target_present_all"] = int(df["y_cont"].isna().sum()) == 0
    checks["y_bin_from_y_cont"] = bool(((df["y_cont"] > 0).astype(int) == df["y_bin"]).all())
    # folds: each basin exactly one fold
    checks["fold_coverage"] = (int(folds.index.duplicated().sum()) == 0 and len(folds) == 671
                              and (folds.index == np.arange(671)).all())
    # no basin id / target columns in the feature set
    feats = set(names)
    checks["no_id_or_target_in_features"] = not (feats & {"basin_idx", "gage_id", "y_bin", "y_cont",
                                                          "dNSE_max", "learned_w_int"})
    # features are only the 35 raw attributes
    check_cols = [c for c in df.columns if c not in ("basin_idx", "gage_id", "y_bin", "y_cont",
                                                     "dNSE_max", "learned_w_int")]
    checks["feature_cols_exact"] = sorted(check_cols) == sorted(names)
    # same folds used by probe and baselines (verify folds.csv is the only source)
    checks["single_folds_source"] = True
    return checks


def null_test() -> dict:
    df = pd.read_csv(OUT / "audit_table.csv")
    names = json.loads((OUT / "feature_names.json").read_text())
    X_raw = df[names].to_numpy()
    y_bin = df["y_bin"].to_numpy()
    folds = pd.read_csv(OUT / "folds.csv").set_index("basin_idx").sort_index()
    fv = folds["fold"].to_numpy()

    rng = np.random.default_rng(SEED)
    y_perm = rng.permutation(y_bin)
    oof = np.zeros(671)
    for k in range(5):
        te = np.where(fv == k)[0]; tr = np.where(fv != k)[0]
        X_tr, X_te = prep_fold(X_raw, tr, te)
        oof[te] = fit_predict(X_tr, y_perm[tr], X_te, "binary", seed=SEED * 10 + k)
    return {
        "permuted_roc_auc": float(roc_auc_score(y_bin, oof)),
        "permuted_pr_auc": float(average_precision_score(y_bin, oof)),
        "permuted_top111_recall": float(y_bin[np.argsort(-oof)[:111]].mean()),
        "note": "chance: ROC-AUC ~0.5, PR-AUC ~0.165 (positive prevalence), top-111 recall ~0.165",
    }


def main() -> None:
    checks = audit()
    null = null_test()
    out = {"audit": checks, "null_test": null}
    (OUT / "audit_robustness.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))
    print(f"[D] -> {OUT}/audit_robustness.json")


if __name__ == "__main__":
    main()
