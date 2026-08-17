#!/usr/bin/env python3
"""Part C: R14 vs R16 OOF AUC Discrepancy Audit and Unified 5-Fold Benchmark.

1. Historical Pipeline Audit Table:
   - Compares feature source, target formulation, model family, optimizer, scaling, and metrics.
2. Unified 5-Fold Stratified Benchmark:
   - Evaluates x35 (raw attributes) and h128 (frozen representation) under identical folds.
   - Evaluates 3 distinct targets:
       Target 1: DeltaJ_int > 0 (Endpoint counterfactual preference)
       Target 2: q_int > 0.5 (Soft-target threshold)
       Target 3: oracle_w_int > 0 (Continuous Oracle-positive status)
   - Evaluates model classes:
       - LogisticRegression(C=1.0) with fold-local StandardScaler (Canonical Probe)
       - LogisticRegression(C=0.1, 10.0)
       - PyTorch Linear Probe (BCE on continuous q vs BCE on binary label)

Outputs saved to:
  results/reconciliation_r16_5/part_c_pipeline_audit_table.csv
  results/reconciliation_r16_5/part_c_unified_oof_benchmark.csv
  results/reconciliation_r16_5/part_c_oof_predictions.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUT_DIR = Path("results/reconciliation_r16_5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-12


def compute_bce(p: np.ndarray, q: np.ndarray) -> float:
    p_c = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(q * np.log(p_c) + (1.0 - q) * np.log(1.0 - p_c)))


def main():
    print("=" * 80)
    print("Flex-MOPEX R16.5: Part C - Audit R14 vs R16 OOF AUC Discrepancy")
    print("=" * 80)

    # 1. Load Canonical Dataset
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    data = torch.load(OUT_DIR / "canonical_reconciliation_dataset.pt", map_location=dev, weights_only=False)
    x35 = data["x35"]                    # [671, 35]
    h128 = data["h128"]                  # [671, 128]
    DeltaJ_int = data["DeltaJ_int"]      # [671]
    q_int = data["q_int"]                # [671]
    target_pos = data["target_pos_binary"]  # [671]
    oracle_pos = data["oracle_pos_binary"]  # [671]
    oracle_w = data["oracle_w_int"]        # [671]
    B = len(target_pos)

    # 2. Historical Pipeline Audit Table
    audit_table = [
        {
            "aspect": "Checkpoint / State",
            "R14_Phase3_Probe": "R8 ep2, R8 ep10, Baseline ep10",
            "R16_Phase3_Probe": "R15 ep10",
            "Reconciliation_Status": "Aligned on R15 ep10 canonical checkpoint"
        },
        {
            "aspect": "Target formulation during training",
            "R14_Phase3_Probe": "Binary target y = (DeltaJ > 0) in {0, 1}",
            "R16_Phase3_Probe": "Continuous soft target q in (0, 1)",
            "Reconciliation_Status": "MAJOR DIVERGENCE: Classification vs Soft-Label Regression"
        },
        {
            "aspect": "Model architecture & solver",
            "R14_Phase3_Probe": "sklearn LogisticRegression(C=1.0, L-BFGS/lbfgs)",
            "R16_Phase3_Probe": "PyTorch Linear/MLP head (Adam lr=0.01, 800 steps)",
            "Reconciliation_Status": "MAJOR DIVERGENCE: Exact convex solver vs Adam neural probe"
        },
        {
            "aspect": "Feature standardization",
            "R14_Phase3_Probe": "StandardScaler() fitted inside each training fold",
            "R16_Phase3_Probe": "Raw unstandardized h fed directly to PyTorch net",
            "Reconciliation_Status": "MAJOR DIVERGENCE: Feature scaling crucial for unwhitened h"
        },
        {
            "aspect": "Loss function",
            "R14_Phase3_Probe": "Binary Log-Loss on y in {0, 1}",
            "R16_Phase3_Probe": "Soft-label BCE on continuous q across 4 joint processes",
            "Reconciliation_Status": "MAJOR DIVERGENCE: 4-process unweighted soft BCE vs 1-process binary"
        },
        {
            "aspect": "Evaluation metric task",
            "R14_Phase3_Probe": "ROC-AUC of predicted P(DeltaJ > 0 | h) vs True (DeltaJ > 0)",
            "R16_Phase3_Probe": "ROC-AUC of continuous regression score p vs True (q > 0.5)",
            "Reconciliation_Status": "Tested ranking of different targets under different calibration"
        }
    ]
    df_audit = pd.DataFrame(audit_table)
    df_audit.to_csv(OUT_DIR / "part_c_pipeline_audit_table.csv", index=False)
    print("Saved part_c_pipeline_audit_table.csv")

    # 3. Unified 5-Fold Stratified Benchmark
    print("\n[Unified Benchmark] Running 5-fold CV across features, models, and targets...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # We evaluate 3 targets:
    targets = {
        "Target1_DeltaJ_gt0": target_pos,
        "Target2_q_gt05": (q_int > 0.5).astype(int),
        "Target3_Oracle_gt0": oracle_pos,
    }

    feature_sets = {
        "x35_raw": x35,
        "h128_frozen": h128,
    }

    benchmark_rows = []
    oof_predictions = {"basin_idx": np.arange(B)}

    for t_name, y_t in targets.items():
        n_pos = int(np.sum(y_t))
        pos_frac = float(np.mean(y_t))
        print(f"\n--- Target: {t_name} (N_pos={n_pos}/{B} = {pos_frac*100:.1f}%) ---")

        for f_name, X_feat in feature_sets.items():
            # A. Canonical LogisticRegression(C=1.0) with fold-local StandardScaler
            oof_probs_lr1 = np.zeros(B)
            fold_aucs_lr1 = []
            fold_praucs_lr1 = []

            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_feat, y_t)):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_feat[tr_idx])
                X_val = scaler.transform(X_feat[val_idx])

                clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42 + fold)
                clf.fit(X_tr, y_t[tr_idx])
                preds = clf.predict_proba(X_val)[:, 1]
                oof_probs_lr1[val_idx] = preds

                f_roc = float(roc_auc_score(y_t[val_idx], preds))
                f_pr = float(average_precision_score(y_t[val_idx], preds))
                fold_aucs_lr1.append(f_roc)
                fold_praucs_lr1.append(f_pr)

            roc_lr1 = float(roc_auc_score(y_t, oof_probs_lr1))
            pr_lr1 = float(average_precision_score(y_t, oof_probs_lr1))
            sp_lr1, _ = spearmanr(oof_probs_lr1, y_t)
            sp_orc, _ = spearmanr(oof_probs_lr1, oracle_w)

            pred_key = f"oof_{f_name}_{t_name}_LogRegC1"
            oof_predictions[pred_key] = oof_probs_lr1

            benchmark_rows.append({
                "feature_set": f_name,
                "target_name": t_name,
                "model_class": "LogisticRegression(C=1.0, Scaler)",
                "n_pos": n_pos,
                "pos_fraction": pos_frac,
                "oof_roc_auc": roc_lr1,
                "oof_pr_auc": pr_lr1,
                "fold_roc_auc_mean": float(np.mean(fold_aucs_lr1)),
                "fold_roc_auc_std": float(np.std(fold_aucs_lr1)),
                "fold_pr_auc_mean": float(np.mean(fold_praucs_lr1)),
                "spearman_target": float(sp_lr1),
                "spearman_oracle_w": float(sp_orc),
            })
            print(f"  {f_name:<12s} | {t_name:<20s} | LogReg(C=1) : OOF ROC-AUC = {roc_lr1:.4f} (folds: {np.mean(fold_aucs_lr1):.4f} +/- {np.std(fold_aucs_lr1):.4f}) | PR-AUC = {pr_lr1:.4f} | Sp_orc = {sp_orc:+.4f}")

            # B. LogisticRegression with C=0.1 and C=10.0
            for C_val in [0.1, 10.0]:
                oof_probs_c = np.zeros(B)
                for fold, (tr_idx, val_idx) in enumerate(skf.split(X_feat, y_t)):
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_feat[tr_idx])
                    X_val = scaler.transform(X_feat[val_idx])
                    clf = LogisticRegression(C=C_val, max_iter=1000, random_state=42 + fold)
                    clf.fit(X_tr, y_t[tr_idx])
                    oof_probs_c[val_idx] = clf.predict_proba(X_val)[:, 1]

                roc_c = float(roc_auc_score(y_t, oof_probs_c))
                pr_c = float(average_precision_score(y_t, oof_probs_c))
                sp_orc_c, _ = spearmanr(oof_probs_c, oracle_w)

                benchmark_rows.append({
                    "feature_set": f_name,
                    "target_name": t_name,
                    "model_class": f"LogisticRegression(C={C_val}, Scaler)",
                    "n_pos": n_pos,
                    "pos_fraction": pos_frac,
                    "oof_roc_auc": roc_c,
                    "oof_pr_auc": pr_c,
                    "fold_roc_auc_mean": float(roc_c),
                    "fold_roc_auc_std": 0.0,
                    "fold_pr_auc_mean": float(pr_c),
                    "spearman_target": float(spearmanr(oof_probs_c, y_t)[0]),
                    "spearman_oracle_w": float(sp_orc_c),
                })

            # C. PyTorch Linear Probe trained with StandardScaler + Adam on binary target
            oof_probs_pt_bin = np.zeros(B)
            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_feat, y_t)):
                scaler = StandardScaler()
                X_tr = torch.from_numpy(scaler.fit_transform(X_feat[tr_idx])).float().to(dev)
                X_val = torch.from_numpy(scaler.transform(X_feat[val_idx])).float().to(dev)
                y_tr_t = torch.from_numpy(y_t[tr_idx]).float().to(dev)

                torch.manual_seed(42 + fold)
                lin_m = nn.Linear(X_tr.shape[1], 1).to(dev)
                opt_pt = torch.optim.Adam(lin_m.parameters(), lr=0.01, weight_decay=1e-3)
                for _ in range(500):
                    opt_pt.zero_grad()
                    p_pt = torch.sigmoid(lin_m(X_tr).squeeze(-1))
                    l_pt = F.binary_cross_entropy(p_pt, y_tr_t)
                    l_pt.backward()
                    opt_pt.step()

                with torch.no_grad():
                    oof_probs_pt_bin[val_idx] = torch.sigmoid(lin_m(X_val).squeeze(-1)).cpu().numpy()

            roc_pt_bin = float(roc_auc_score(y_t, oof_probs_pt_bin))
            pr_pt_bin = float(average_precision_score(y_t, oof_probs_pt_bin))
            sp_orc_pt, _ = spearmanr(oof_probs_pt_bin, oracle_w)

            benchmark_rows.append({
                "feature_set": f_name,
                "target_name": t_name,
                "model_class": "PyTorch Linear (Binary Adam, Scaler)",
                "n_pos": n_pos,
                "pos_fraction": pos_frac,
                "oof_roc_auc": roc_pt_bin,
                "oof_pr_auc": pr_pt_bin,
                "fold_roc_auc_mean": float(roc_pt_bin),
                "fold_roc_auc_std": 0.0,
                "fold_pr_auc_mean": float(pr_pt_bin),
                "spearman_target": float(spearmanr(oof_probs_pt_bin, y_t)[0]),
                "spearman_oracle_w": float(sp_orc_pt),
            })
            print(f"  {f_name:<12s} | {t_name:<20s} | PyTorch(Binary): OOF ROC-AUC = {roc_pt_bin:.4f} | PR-AUC = {pr_pt_bin:.4f} | Sp_orc = {sp_orc_pt:+.4f}")

            # D. PyTorch Linear Probe trained on CONTINUOUS q WITHOUT scaling (R16 Historical Probe Reproduction)
            oof_probs_pt_cont = np.zeros(B)
            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_feat, y_t)):
                X_tr = torch.from_numpy(X_feat[tr_idx]).float().to(dev)  # NO SCALER
                X_val = torch.from_numpy(X_feat[val_idx]).float().to(dev)
                q_tr_t = torch.from_numpy(q_int[tr_idx]).float().to(dev)

                torch.manual_seed(42 + fold)
                lin_m = nn.Linear(X_tr.shape[1], 1).to(dev)
                opt_pt = torch.optim.Adam(lin_m.parameters(), lr=0.01, weight_decay=1e-4)
                for _ in range(800):
                    opt_pt.zero_grad()
                    p_pt = torch.sigmoid(lin_m(X_tr).squeeze(-1))
                    l_pt = F.binary_cross_entropy(p_pt, q_tr_t)
                    l_pt.backward()
                    opt_pt.step()

                with torch.no_grad():
                    oof_probs_pt_cont[val_idx] = torch.sigmoid(lin_m(X_val).squeeze(-1)).cpu().numpy()

            roc_pt_cont = float(roc_auc_score(y_t, oof_probs_pt_cont))
            pr_pt_cont = float(average_precision_score(y_t, oof_probs_pt_cont))
            sp_orc_cont, _ = spearmanr(oof_probs_pt_cont, oracle_w)

            benchmark_rows.append({
                "feature_set": f_name,
                "target_name": t_name,
                "model_class": "PyTorch Linear (Cont-q Adam, NoScaler - R16)",
                "n_pos": n_pos,
                "pos_fraction": pos_frac,
                "oof_roc_auc": roc_pt_cont,
                "oof_pr_auc": pr_pt_cont,
                "fold_roc_auc_mean": float(roc_pt_cont),
                "fold_roc_auc_std": 0.0,
                "fold_pr_auc_mean": float(pr_pt_cont),
                "spearman_target": float(spearmanr(oof_probs_pt_cont, y_t)[0]),
                "spearman_oracle_w": float(sp_orc_cont),
            })
            print(f"  {f_name:<12s} | {t_name:<20s} | R16 Historical: OOF ROC-AUC = {roc_pt_cont:.4f} | PR-AUC = {pr_pt_cont:.4f} | Sp_orc = {sp_orc_cont:+.4f}")

    df_bench = pd.DataFrame(benchmark_rows)
    df_bench.to_csv(OUT_DIR / "part_c_unified_oof_benchmark.csv", index=False)

    df_preds = pd.DataFrame(oof_predictions)
    df_preds.to_csv(OUT_DIR / "part_c_oof_predictions.csv", index=False)
    print(f"\n[Part C Complete] Unified OOF benchmark saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
