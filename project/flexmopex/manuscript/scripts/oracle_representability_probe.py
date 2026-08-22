#!/usr/bin/env python3
"""Agent B — existing structure-head representability probe.

Supervised diagnostic probe that reuses the EXACT Flex structure-head
architecture and the w_int output parameterization:

  backbone: Linear(35,128) -> Tanh -> Dropout(0.5) -> Linear(128,128) -> Tanh
            -> Dropout(0.5)   (xavier init, as _BaseParameterNet)
  w_int head: Linear(128, 2)  (init normal(0, 0.001), as the weights head)
  output: w = softmax(logits)[..., 1]   (same 2-class off/on parameterization)

Tasks (no hydrologic objective, no AIC):
  T1 binary   : BCE on y_bin = 1(oracle w_int > 0)
  T2 continuous: MSE on y_cont = oracle w_int

Protocol: 5-fold stratified CV (folds.csv, fixed seed); preprocessing (NaN
imputation by train-fold column mean, z-score by train-fold stats) is fitted
per fold; fixed seeds per fold.  In-sample ceiling = fit on all 671.

Metrics on OOF predictions: ROC-AUC, PR-AUC, balanced accuracy, positive
recall/precision, confusion matrix, Spearman/MAE/RMSE, top-111 recall.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sklearn.metrics import (  # noqa: E402
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    recall_score, precision_score, confusion_matrix, mean_absolute_error,
    mean_squared_error,
)
from scipy.stats import spearmanr  # noqa: E402

OUT = PROJECT_DIR / "results" / "oracle_representability"
EPOCHS = 800
LR = 1e-3
SEED = 42


class StructureHeadProbe(nn.Module):
    """Exact replica of the Flex structure head (backbone + w_int logit pair)."""

    def __init__(self, input_dim: int = 35, hidden_dim: int = 128, dropout: float = 0.5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.w_int_head = nn.Linear(hidden_dim, 2)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.w_int_head:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        nn.init.normal_(self.w_int_head.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.w_int_head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.w_int_head(self.backbone(x)), dim=-1)[..., 1]


def fit_predict(X_tr, y_tr, X_te, task: str, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    model = StructureHeadProbe()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        w = model(xt)
        if task == "binary":
            loss = nn.functional.binary_cross_entropy(w, yt)
        else:
            loss = nn.functional.mse_loss(w, yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X_te, dtype=torch.float32)).numpy()


def prep_fold(X_raw: np.ndarray, tr_idx, te_idx) -> tuple[np.ndarray, np.ndarray]:
    """Fold-local preprocessing: impute NaN with train mean, z-score with train stats."""
    tr = X_raw[tr_idx]
    col_mean = np.nanmean(tr, axis=0)
    X_imp = np.where(np.isnan(X_raw), col_mean, X_raw)
    m = X_imp[tr_idx].mean(axis=0)
    s = X_imp[tr_idx].std(axis=0) + 1e-8
    return (X_imp[tr_idx] - m) / s, (X_imp[te_idx] - m) / s


def metrics_bin(y, p):
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "balanced_acc": float(balanced_accuracy_score(y, p > 0.5)),
        "recall_pos": float(recall_score(y, p > 0.5, zero_division=0)),
        "precision_pos": float(precision_score(y, p > 0.5, zero_division=0)),
        "confusion": confusion_matrix(y, p > 0.5).tolist(),
    }


def main() -> None:
    df = pd.read_csv(OUT / "audit_table.csv")
    names = json.loads((OUT / "feature_names.json").read_text())
    X_raw = df[names].to_numpy()
    y_bin = df["y_bin"].to_numpy()
    y_cont = df["y_cont"].to_numpy()
    folds = pd.read_csv(OUT / "folds.csv").set_index("basin_idx").sort_index()

    oof_bin = np.zeros(671); oof_cont = np.zeros(671)
    for k in range(5):
        te = np.where(folds["fold"].to_numpy() == k)[0]
        tr = np.where(folds["fold"].to_numpy() != k)[0]
        X_tr, X_te = prep_fold(X_raw, tr, te)
        oof_bin[te] = fit_predict(X_tr, y_bin[tr], X_te, "binary", seed=SEED + k)
        oof_cont[te] = fit_predict(X_tr, y_cont[tr], X_te, "continuous", seed=SEED + k)
        print(f"[B] fold {k}: done", flush=True)

    res = {"task_binary_oof": metrics_bin(y_bin, oof_bin)}
    res["task_continuous_oof"] = {
        "spearman": float(spearmanr(y_cont, oof_cont).statistic),
        "mae": float(mean_absolute_error(y_cont, oof_cont)),
        "rmse": float(mean_squared_error(y_cont, oof_cont) ** 0.5),
    }
    # top-111 (16.5%) ranking recall
    top = np.argsort(-oof_bin)[:111]
    res["task_binary_oof"]["top111_recall_pos"] = float(y_bin[top].mean())
    res["task_continuous_oof"]["top111_recall_pos"] = float(y_bin[np.argsort(-oof_cont)[:111]].mean())

    # in-sample ceiling (fit on all 671)
    X_all = prep_fold(X_raw, np.arange(671), np.arange(671))[0]
    is_bin = fit_predict(X_all, y_bin, X_all, "binary", seed=SEED + 100)
    is_cont = fit_predict(X_all, y_cont, X_all, "continuous", seed=SEED + 100)
    res["insample_ceiling"] = {
        "binary_roc_auc": float(roc_auc_score(y_bin, is_bin)),
        "binary_recall_pos": float(recall_score(y_bin, is_bin > 0.5, zero_division=0)),
        "binary_balanced_acc": float(balanced_accuracy_score(y_bin, is_bin > 0.5)),
        "continuous_spearman": float(spearmanr(y_cont, is_cont).statistic),
        "continuous_rmse": float(mean_squared_error(y_cont, is_cont) ** 0.5),
    }

    pd.DataFrame({"basin_idx": np.arange(671), "oof_bin": oof_bin,
                  "oof_cont": oof_cont, "y_bin": y_bin, "y_cont": y_cont}
                 ).to_csv(OUT / "probe_oof_predictions.csv", index=False)
    (OUT / "probe_results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    print(f"[B] -> {OUT}/probe_results.json, probe_oof_predictions.csv")


if __name__ == "__main__":
    main()
