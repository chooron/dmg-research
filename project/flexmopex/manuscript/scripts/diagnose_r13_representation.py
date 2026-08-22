#!/usr/bin/env python3
"""Phase 1: Representation Probing Diagnostic (Testing Hypothesis A).

Evaluates whether the shared learned backbone representation h (128-D) and h1 (128-D)
preserves, improves, or loses interception-relevant information compared to raw 35-D attributes x.

Uses strictly FIXED targets:
  1. Primary target: Canonical E-S0 ep10 exact oracle (146/671 positive, 21.8%)
  2. Sensitivity target: R8 ep2 exact oracle (103/671 positive, 15.4%)

Evaluates across checkpoints for:
  - Canonical E-S0 Baseline (ep0, 1, 2, 3, 5, 10)
  - R8 AIC-Delay (ep1, 2, 3, 4, 5, 10)
  - R10-B Sensitivity+Delay (ep1, 2, 3, 4, 5, 10)

Protocol:
  5-fold Stratified CV (fixed seed 42), fold-local StandardScaler (no leakage),
  deterministic LogisticRegression probe (C=1.0).
  Reports OOF ROC-AUC, OOF PR-AUC, fold-level statistics, and Delta vs raw-X.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config  # noqa: E402
from project.flexmopex.run_model import (  # noqa: E402
    apply_runtime_overrides, parse_args, _build_data_loader,
)
from scripts.diagnose_wint_collapse import build_handler  # noqa: E402

OUT_DIR = Path("results/root_cause_r13")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def evaluate_probe(X: np.ndarray, y: np.ndarray, cv_splits: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Evaluate logistic probe with strict fold-local scaling."""
    oof_probs = np.zeros(len(y), dtype=float)
    fold_roc_aucs = []
    fold_pr_aucs = []

    for fold_idx, (tr_idx, te_idx) in enumerate(cv_splits):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        y_tr = y[tr_idx]
        y_te = y[te_idx]

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        probs_te = clf.predict_proba(X_te)[:, 1]
        oof_probs[te_idx] = probs_te

        fold_roc_aucs.append(float(roc_auc_score(y_te, probs_te)))
        fold_pr_aucs.append(float(average_precision_score(y_te, probs_te)))

    oof_roc_auc = float(roc_auc_score(y, oof_probs))
    oof_pr_auc = float(average_precision_score(y, oof_probs))

    return {
        "oof_roc_auc": oof_roc_auc,
        "oof_pr_auc": oof_pr_auc,
        "fold_roc_auc_mean": float(np.mean(fold_roc_aucs)),
        "fold_roc_auc_std": float(np.std(fold_roc_aucs)),
        "fold_pr_auc_mean": float(np.mean(fold_pr_aucs)),
        "fold_pr_auc_std": float(np.std(fold_pr_aucs)),
        "fold_roc_aucs": fold_roc_aucs,
        "fold_pr_aucs": fold_pr_aucs,
        "oof_probs": oof_probs.tolist(),
    }


def main() -> None:
    # 1. Load targets
    manifest = json.load(open(OUT_DIR / "audit_manifest.json"))
    
    can_rows = list(csv.DictReader(open(manifest["targets"]["primary_canonical_ep10"]["source"])))
    can_ep10 = [r for r in can_rows if r["epoch"] == "10" and r["process"] == "w_int"]
    y_primary = np.array([float(r["w_star"]) > 0 for r in sorted(can_ep10, key=lambda x: int(x["basin_idx"]))], dtype=int)

    r8_rows = list(csv.DictReader(open(manifest["targets"]["sensitivity_r8_ep2"]["source"])))
    r8_ep2 = [r for r in r8_rows if r["epoch"] == "2" and r["process"] == "w_int"]
    y_sens = np.array([float(r["w_star"]) > 0 for r in sorted(r8_ep2, key=lambda x: int(x["basin_idx"]))], dtype=int)

    targets = {
        "primary_canonical_ep10": y_primary,
        "sensitivity_r8_ep2": y_sens,
    }

    # 2. Build data loader to get raw 35-D normalized attributes
    cfg = load_config("conf/config_dmopex_interceptE_S0.yaml")
    cli = parse_args(["--config", "conf/config_dmopex_interceptE_S0.yaml", "--gpu-id", "0"])
    apply_runtime_overrides(cfg, cli, config_path="conf/config_dmopex_interceptE_S0.yaml")
    cfg["mode"] = "train"
    cfg["model"]["phy"]["disable_compile"] = True
    dl = _build_data_loader(cfg)
    td = dl.train_dataset
    n_attr = td["xc_nn_norm"].shape[-1] - 3
    raw_attrs = td["xc_nn_norm"][0, :, -n_attr:].cpu().numpy()  # [671, 35]
    attrs_torch = torch.from_numpy(raw_attrs).float()

    print(f"[Phase 1] Raw attributes shape: {raw_attrs.shape}")

    # Build fixed CV splits for each target
    cv_splits_dict = {}
    for tgt_name, y_arr in targets.items():
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_splits_dict[tgt_name] = list(skf.split(raw_attrs, y_arr))

    # Evaluate baseline probe on Raw 35-D Attributes
    results = {"raw_attributes": {}, "runs": {}}
    table_rows = []

    for tgt_name, y_arr in targets.items():
        res_x = evaluate_probe(raw_attrs, y_arr, cv_splits_dict[tgt_name])
        results["raw_attributes"][tgt_name] = res_x
        print(f"Raw X -> {tgt_name:25s}: OOF ROC-AUC = {res_x['oof_roc_auc']:.4f}, PR-AUC = {res_x['oof_pr_auc']:.4f}")
        table_rows.append({
            "run": "Raw_Attributes_X",
            "epoch": "N/A",
            "feature_set": "raw_x_35d",
            "feature_dim": 35,
            "target": tgt_name,
            "oof_roc_auc": res_x["oof_roc_auc"],
            "oof_pr_auc": res_x["oof_pr_auc"],
            "fold_roc_auc_mean": res_x["fold_roc_auc_mean"],
            "fold_roc_auc_std": res_x["fold_roc_auc_std"],
            "delta_roc_auc_vs_x": 0.0,
            "delta_pr_auc_vs_x": 0.0,
        })

    # Evaluate representation probes across runs and checkpoints
    runs_to_probe = {
        "Baseline": {
            "config": "conf/config_dmopex_interceptE_S0.yaml",
            "output_root": "results/intercept_candidates",
            "run_name": "E_S0",
            "epochs": [0, 1, 2, 3, 5, 10],
        },
        "R8_AICDelay": {
            "config": "conf/config_dmopex_interceptE_S0_aicdelay2.yaml",
            "output_root": "results/intercept_aicdelay",
            "run_name": "E_S0_aicdelay2",
            "epochs": [1, 2, 3, 4, 5, 10],
        },
        "R10B_ReweightDelay": {
            "config": "conf/config_dmopex_interceptE_S0_reweight_aicdelay2.yaml",
            "output_root": "results/intercept_reweight",
            "run_name": "E_S0_reweight_delay2",
            "epochs": [1, 2, 3, 4, 5, 10],
        },
    }

    oof_preds_dict = {}

    for run_name, run_info in runs_to_probe.items():
        results["runs"][run_name] = {}
        c_path = run_info["config"]
        c = load_config(c_path)
        c_cli = parse_args(["--config", c_path, "--gpu-id", "0",
                            "--output-root", run_info["output_root"],
                            "--run-name", run_info["run_name"]])
        apply_runtime_overrides(c, c_cli, config_path=c_path)
        c["mode"] = "train"
        c["model"]["phy"]["disable_compile"] = True
        handler = build_handler(c)

        for ep in run_info["epochs"]:
            try:
                handler.load_model(ep)
            except Exception as e:
                print(f"[warn] Could not load {run_name} ep {ep}: {e}")
                continue

            for m in handler.model_dict.values():
                m.eval()
            model = next(iter(handler.model_dict.values()))
            nn = model.nn_model

            # Extract representations in deterministic eval mode (no dropout)
            with torch.no_grad():
                # Layer 1: Linear(35, 128) -> Tanh()
                h1 = nn.backbone[0:2](attrs_torch.to(model.device)).cpu().numpy()
                # Layer 2 (final h): Backbone full
                h2 = nn.backbone(attrs_torch.to(model.device)).cpu().numpy()

            feat_sets = {"h_final_128d": (h2, 128), "h1_layer1_128d": (h1, 128)}

            ep_res = {}
            for feat_name, (feat_mat, feat_dim) in feat_sets.items():
                ep_res[feat_name] = {}
                for tgt_name, y_arr in targets.items():
                    res = evaluate_probe(feat_mat, y_arr, cv_splits_dict[tgt_name])
                    delta_roc = res["oof_roc_auc"] - results["raw_attributes"][tgt_name]["oof_roc_auc"]
                    delta_pr = res["oof_pr_auc"] - results["raw_attributes"][tgt_name]["oof_pr_auc"]
                    res["delta_roc_auc_vs_x"] = delta_roc
                    res["delta_pr_auc_vs_x"] = delta_pr
                    ep_res[feat_name][tgt_name] = res

                    key_pred = f"{run_name}_ep{ep}_{feat_name}_{tgt_name}"
                    oof_preds_dict[key_pred] = res["oof_probs"]

                    table_rows.append({
                        "run": run_name,
                        "epoch": ep,
                        "feature_set": feat_name,
                        "feature_dim": feat_dim,
                        "target": tgt_name,
                        "oof_roc_auc": res["oof_roc_auc"],
                        "oof_pr_auc": res["oof_pr_auc"],
                        "fold_roc_auc_mean": res["fold_roc_auc_mean"],
                        "fold_roc_auc_std": res["fold_roc_auc_std"],
                        "delta_roc_auc_vs_x": delta_roc,
                        "delta_pr_auc_vs_x": delta_pr,
                    })

                    print(f"[{run_name:18s} ep{ep:>2d} {feat_name:14s}] -> {tgt_name:25s}: ROC-AUC = {res['oof_roc_auc']:.4f} (Δ={delta_roc:+.4f}), PR-AUC = {res['oof_pr_auc']:.4f} (Δ={delta_pr:+.4f})")

            results["runs"][run_name][str(ep)] = ep_res

    # Save summary json, table csv, and predictions
    summary_path = OUT_DIR / "representation_probe_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))

    df_table = pd.DataFrame(table_rows)
    table_path = OUT_DIR / "representation_probe_table.csv"
    df_table.to_csv(table_path, index=False)

    df_preds = pd.DataFrame(oof_preds_dict)
    df_preds["basin_idx"] = np.arange(len(y_primary))
    df_preds["y_primary"] = y_primary
    df_preds["y_sens"] = y_sens
    preds_path = OUT_DIR / "representation_probe_oof_predictions.csv"
    df_preds.to_csv(preds_path, index=False)

    print(f"\n[Phase 1 Complete] Summary saved to {summary_path} and {table_path}")


if __name__ == "__main__":
    main()
