#!/usr/bin/env python3
"""Agent C — simple baselines + hydrologic organization of the oracle target.

Same folds/features as the structure-head probe (results/oracle_representability/).

Baselines (sklearn, no new deps):
  binary : LogisticRegression (balanced), RandomForestClassifier, GradientBoostingClassifier
  continuous: Ridge, RandomForestRegressor
  minimal-feature probe: logistic on the top-4 univariate attributes

Attribute organization: univariate Spearman (y_bin and y_cont), standardized
mean difference (Cohen's d) between oracle-positive and oracle-zero basins,
OOF permutation-importance-free descriptive summary.
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

from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    recall_score, precision_score, mean_absolute_error, mean_squared_error,
)
from scipy.stats import spearmanr  # noqa: E402

OUT = PROJECT_DIR / "results" / "oracle_representability"
SEED = 42


def prep_fold(X_raw, tr_idx, te_idx):
    col_mean = np.nanmean(X_raw[tr_idx], axis=0)
    X_imp = np.where(np.isnan(X_raw), col_mean, X_raw)
    m = X_imp[tr_idx].mean(axis=0)
    s = X_imp[tr_idx].std(axis=0) + 1e-8
    return (X_imp[tr_idx] - m) / s, (X_imp[te_idx] - m) / s


def main() -> None:
    df = pd.read_csv(OUT / "audit_table.csv")
    names = json.loads((OUT / "feature_names.json").read_text())
    X_raw = df[names].to_numpy()
    y_bin = df["y_bin"].to_numpy()
    y_cont = df["y_cont"].to_numpy()
    folds = pd.read_csv(OUT / "folds.csv").set_index("basin_idx").sort_index()
    fv = folds["fold"].to_numpy()

    # ---- attribute organization (univariate) ----
    org = []
    for j, a in enumerate(names):
        v = X_raw[:, j]
        rho_b, pb = spearmanr(v, y_bin)
        rho_c, pc = spearmanr(v, y_cont)
        pos, zero = v[y_bin == 1], v[y_bin == 0]
        sp = np.nanstd(pos, ddof=1) if len(pos) > 1 else 0.0
        sz = np.nanstd(zero, ddof=1) if len(zero) > 1 else 0.0
        pooled = np.sqrt(((len(pos) - 1) * sp**2 + (len(zero) - 1) * sz**2) /
                         (len(pos) + len(zero) - 2)) + 1e-12
        d = (np.nanmean(pos) - np.nanmean(zero)) / pooled
        org.append({"attr": a, "spearman_ybin": float(rho_b), "p_ybin": float(pb),
                    "spearman_ycont": float(rho_c), "p_ycont": float(pc),
                    "cohens_d": float(d)})
    org_df = pd.DataFrame(org).sort_values("spearman_ybin", key=lambda s: -s.abs())
    org_df.to_csv(OUT / "baseline_attribute_organization.csv", index=False)

    top4 = org_df.head(4)["attr"].tolist()
    X4 = df[top4].to_numpy()

    # ---- OOF baselines ----
    models_bin = {
        "logistic": LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                class_weight="balanced", random_state=SEED),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                                        random_state=SEED),
    }
    models_cont = {
        "ridge": Ridge(alpha=1.0),
        "random_forest_reg": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                   random_state=SEED),
    }
    res = {"attribute_organization": org_df.to_dict("records")[:12], "top4": top4}
    for name, mdl in models_bin.items():
        oof = np.zeros(671)
        for k in range(5):
            te = np.where(fv == k)[0]; tr = np.where(fv != k)[0]
            X_tr, X_te = prep_fold(X_raw, tr, te)
            m = mdl.__class__(**mdl.get_params())
            m.fit(X_tr, y_bin[tr])
            oof[te] = m.predict_proba(X_te)[:, 1]
        res[f"binary_{name}"] = {
            "roc_auc": float(roc_auc_score(y_bin, oof)),
            "pr_auc": float(average_precision_score(y_bin, oof)),
            "balanced_acc": float(balanced_accuracy_score(y_bin, oof > 0.5)),
            "recall_pos": float(recall_score(y_bin, oof > 0.5, zero_division=0)),
            "precision_pos": float(precision_score(y_bin, oof > 0.5, zero_division=0)),
            "top111_recall_pos": float(y_bin[np.argsort(-oof)[:111]].mean()),
        }
    for name, mdl in models_cont.items():
        oof = np.zeros(671)
        for k in range(5):
            te = np.where(fv == k)[0]; tr = np.where(fv != k)[0]
            X_tr, X_te = prep_fold(X_raw, tr, te)
            m = mdl.__class__(**mdl.get_params())
            m.fit(X_tr, y_cont[tr])
            oof[te] = m.predict(X_te)
        res[f"continuous_{name}"] = {
            "spearman": float(spearmanr(y_cont, oof).statistic),
            "mae": float(mean_absolute_error(y_cont, oof)),
            "rmse": float(mean_squared_error(y_cont, oof) ** 0.5),
            "top111_recall_pos": float(y_bin[np.argsort(-oof)[:111]].mean()),
        }
    # minimal-feature probe: logistic on top-4
    oof4 = np.zeros(671)
    for k in range(5):
        te = np.where(fv == k)[0]; tr = np.where(fv != k)[0]
        X_tr, X_te = prep_fold(X4, tr, te)
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
        m.fit(X_tr, y_bin[tr])
        oof4[te] = m.predict_proba(X_te)[:, 1]
    res["binary_logistic_top4"] = {
        "roc_auc": float(roc_auc_score(y_bin, oof4)),
        "pr_auc": float(average_precision_score(y_bin, oof4)),
        "top111_recall_pos": float(y_bin[np.argsort(-oof4)[:111]].mean()),
    }

    pd.DataFrame({"basin_idx": np.arange(671), "oof_logistic": oof if False else np.zeros(671)})
    (OUT / "baseline_results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    print(f"[C] -> {OUT}/baseline_results.json, baseline_attribute_organization.csv")


if __name__ == "__main__":
    main()
