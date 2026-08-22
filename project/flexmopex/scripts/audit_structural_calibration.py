#!/usr/bin/env python3
"""Part B: Nine-Lambda Structural Calibration Audit across 531 CAMELS basins.

Reconstructs online training counterfactual structural evidence:
  DeltaJ_{b,p} = J^{OFF}_{b,p} - J^{ON}_{b,p}
and binary ground truth preference:
  y_{b,p} = 1[DeltaJ_{b,p} > 0]

Evaluates:
  - Network positive rate P(p > 0.5)
  - Counterfactual positive rate P(DeltaJ > 0)
  - Difference P(p > 0.5) - P(DeltaJ > 0)
  - ROC-AUC
  - PR-AUC (Average Precision)
  - Spearman correlation between continuous p and continuous DeltaJ
  - Precision, Recall, F1 score (p > 0.5 vs DeltaJ > 0)
  - Confusion matrix counts (TP, FP, TN, FN)
"""
from __future__ import annotations

import os, json, torch, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent.parent
for p in (REPO_ROOT, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config
from project.flexmopex.models.learned_weight_mopex_candidates import (
    LearnedWeightMopexE, LearnedStructureNetPureAttrEncoder
)
from project.flexmopex.model_builder import build_phy_model, build_nn_model
from project.flexmopex.run_model import _build_data_loader

PROCESSES = ["w_phen", "w_int", "w_snow", "w_sub"]
GATE_IDX = {"w_phen": 0, "w_int": 1, "w_snow": 2, "w_sub": 3}
COSTS = {"w_phen": 2.0, "w_int": 2.0, "w_snow": 2.0, "w_sub": 1.0}

LAMBDA_CONFIGS = [
    ("λ=0.003", 0.003, "config_formal_531_flex_lambda0003/flex_alpha_config/seed_42"),
    ("λ=0.005", 0.005, "config_formal_531_flex_lambda0005/flex_alpha_config/seed_42"),
    ("λ=0.007", 0.007, "config_formal_531_flex_lambda0007/flex_alpha_config/seed_42"),
    ("λ=0.010", 0.010, "config_formal_531_flex_lambda0010/flex_alpha_config/seed_42"),
    ("λ=0.015", 0.015, "config_formal_531_flex_lambda0015/flex_alpha_config/seed_42"),
    ("λ=0.020", 0.020, "config_formal_531_flex_lambda0020/flex_alpha_config/seed_42"),
    ("λ=0.030", 0.030, "config_formal_531_flex_lambda0030/flex_alpha_config/seed_42"),
    ("λ=0.050", 0.050, "config_formal_531_flex_lambda0050/flex_alpha_config/seed_42"),
    ("λ=0.100", 0.100, "config_formal_531_flex_lambda0100/flex_alpha_config/seed_42"),
]


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 85)
    print(f"PART B: NINE-LAMBDA STRUCTURAL CALIBRATION AUDIT ({dev})")
    print("=" * 85)

    base_dir = PROJECT_DIR / "results" / "formal_531_parallel"
    out_dir = base_dir / "structural_consistency_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load training dataset (exact training window used for counterfactual objective)
    cfg_base = load_config("project/flexmopex/conf/config_formal_531_flex_lambda0007.yaml")
    cfg_base["device"] = dev
    dl = _build_data_loader(cfg_base)
    td = dl.train_dataset
    B = td["x_phy"].shape[1]
    assert B == 531, f"Expected exactly 531 basins, got {B}"

    n_attr = td["xc_nn_norm"].shape[-1] - 3
    attrs = td["xc_nn_norm"][0, :, -n_attr:].to(dev)

    y_obs_train = td["target"][:, :, 0].cpu().numpy()
    std_train = (np.nanstd(y_obs_train, axis=0) + 0.1).astype(np.float32)
    std_t = torch.from_numpy(std_train).to(dev)

    y_t_dev = td["target"][:, :, 0].to(dev)
    n_valid_b = (~torch.isnan(y_t_dev)).sum(dim=0).float()  # [531]
    N_tot = float(n_valid_b.sum().item())
    std_b_dev = std_t.view(1, B)

    # Prepare physics model template
    phy = build_phy_model(cfg_base, "LearnedWeightMopexE", device=dev)
    nn = build_nn_model(cfg_base, phy, device=dev)
    phy.eval()
    nn.eval()

    sample = {
        "x_phy": td["x_phy"].to(dev),
        "doy": td["doy"].to(dev),
        "c_nn_norm": attrs,
    }
    P, T_forcing, PET, doy, n_steps, _ = phy._prepare_forcings(sample)
    n_out_expected = n_steps - phy.warm_up
    obs_valid_window = y_t_dev[phy.warm_up:phy.warm_up + n_out_expected]  # [n_out, B]
    mask_valid = ~torch.isnan(obs_valid_window)
    n_v_b = mask_valid.sum(dim=0).clamp(min=1.0)  # [B]

    calibration_records = []
    four_process_records = []

    for tag, lmb, rel_path in LAMBDA_CONFIGS:
        print(f"\nProcessing calibration for {tag} (lambda={lmb})...")
        ckpt_path = base_dir / rel_path / "model" / "learnedweightmopexe_ep100.pt"
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        sd = {k.replace("nn_model.", ""): v for k, v in ckpt.items() if k.startswith("nn_model.")}
        nn.load_state_dict(sd, strict=False)

        with torch.no_grad():
            nn_out = nn({"c_nn_norm": attrs})
            mopex_params = phy._descale_mopex_params(nn_out["params"])
            routing = phy._descale_routing_params(nn_out["gamma_uh"])
            logits = nn_out["weights"].view(B, 4, 2).clamp(-10.0, 10.0)
            learned_probs = F.softmax(logits, dim=-1)[..., 1]  # [B, 4]

        p_learned_np = learned_probs.cpu().numpy()

        # Reconstruct counterfactual evidence DeltaJ for each process
        # Using exact CFTrainer formulation (S=2 per process: OFF and ON)
        DeltaJ_matrix = np.zeros((B, 4))
        y_true_matrix = np.zeros((B, 4), dtype=int)
        y_pred_matrix = (p_learned_np > 0.5).astype(int)

        for proc in PROCESSES:
            p_col = GATE_IDX[proc]
            cost_p = COSTS[proc]
            aic_pen = lmb * cost_p * (N_tot / (B * n_valid_b.cpu().numpy()))  # [B]

            # Evaluate OFF (w=0) and ON (w=1) holding baseline weights
            w_off = learned_probs.clone()
            w_off[:, p_col] = 0.0
            w_on = learned_probs.clone()
            w_on[:, p_col] = 1.0

            with torch.no_grad():
                Q_off = phy._run_weighted_loop(P, T_forcing, PET, doy, mopex_params, w_off, n_steps, B)
                Qr_off = phy._apply_routing(Q_off.mean(-1), routing)[:, :, 0]
                sq_off = torch.where(mask_valid, (Qr_off - obs_valid_window)**2 / (std_b_dev**2), torch.zeros_like(Qr_off))
                L_fit_off = (sq_off.sum(dim=0) / n_v_b).cpu().numpy()

                Q_on = phy._run_weighted_loop(P, T_forcing, PET, doy, mopex_params, w_on, n_steps, B)
                Qr_on = phy._apply_routing(Q_on.mean(-1), routing)[:, :, 0]
                sq_on = torch.where(mask_valid, (Qr_on - obs_valid_window)**2 / (std_b_dev**2), torch.zeros_like(Qr_on))
                L_fit_on = (sq_on.sum(dim=0) / n_v_b).cpu().numpy()

            fit_diff = L_fit_off - L_fit_on
            delta_J = fit_diff - aic_pen
            DeltaJ_matrix[:, p_col] = delta_J

            y_true = (delta_J > 0).astype(int)
            y_true_matrix[:, p_col] = y_true

            p_scores = p_learned_np[:, p_col]
            y_pred = (p_scores > 0.5).astype(int)

            n_pos_cf = int(np.sum(y_true))
            n_pos_net = int(np.sum(y_pred))
            cf_pos_rate = float(n_pos_cf / B * 100)
            net_pos_rate = float(n_pos_net / B * 100)
            rate_diff = float(net_pos_rate - cf_pos_rate)

            # Confusion matrix
            if len(np.unique(y_true)) > 1:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                roc_auc = float(roc_auc_score(y_true, p_scores))
                pr_auc = float(average_precision_score(y_true, p_scores))
            elif np.all(y_true == 0):
                tn = int(np.sum(y_pred == 0))
                fp = int(np.sum(y_pred == 1))
                fn = 0
                tp = 0
                roc_auc = np.nan
                pr_auc = float(np.mean(y_true))
            else:
                tn = 0
                fp = 0
                fn = int(np.sum(y_pred == 0))
                tp = int(np.sum(y_pred == 1))
                roc_auc = np.nan
                pr_auc = 1.0

            prec = float(precision_score(y_true, y_pred, zero_division=0))
            rec = float(recall_score(y_true, y_pred, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            sp_rho, _ = spearmanr(p_scores, delta_J)
            pr_r, _ = pearsonr(p_scores, delta_J)

            rec_proc = {
                "lambda_tag": tag,
                "lambda": lmb,
                "process": proc,
                "n_cf_positive": n_pos_cf,
                "n_net_positive": n_pos_net,
                "cf_pos_rate_pct": cf_pos_rate,
                "net_pos_rate_pct": net_pos_rate,
                "rate_diff_pct": rate_diff,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "spearman_rho": float(sp_rho),
                "pearson_r": float(pr_r),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
                "DeltaJ_mean": float(np.mean(delta_J)),
                "DeltaJ_median": float(np.median(delta_J)),
                "DeltaJ_std": float(np.std(delta_J)),
                "p_mean": float(np.mean(p_scores)),
                "p_median": float(np.median(p_scores)),
                "p_std": float(np.std(p_scores)),
            }
            calibration_records.append(rec_proc)

        # 4-Process micro/macro average
        all_y_true = y_true_matrix.flatten()
        all_y_pred = y_pred_matrix.flatten()
        all_p_scores = p_learned_np.flatten()
        all_DeltaJ = DeltaJ_matrix.flatten()

        tn4, fp4, fn4, tp4 = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1]).ravel()
        roc4 = float(roc_auc_score(all_y_true, all_p_scores))
        pr4 = float(average_precision_score(all_y_true, all_p_scores))
        prec4 = float(precision_score(all_y_true, all_y_pred, zero_division=0))
        rec4 = float(recall_score(all_y_true, all_y_pred, zero_division=0))
        f14 = float(f1_score(all_y_true, all_y_pred, zero_division=0))
        sp4, _ = spearmanr(all_p_scores, all_DeltaJ)

        four_process_records.append({
            "lambda_tag": tag,
            "lambda": lmb,
            "overall_cf_pos_rate_pct": float(np.mean(all_y_true) * 100),
            "overall_net_pos_rate_pct": float(np.mean(all_y_pred) * 100),
            "overall_rate_diff_pct": float((np.mean(all_y_pred) - np.mean(all_y_true)) * 100),
            "overall_roc_auc": roc4,
            "overall_pr_auc": pr4,
            "overall_spearman_rho": float(sp4),
            "overall_precision": prec4,
            "overall_recall": rec4,
            "overall_f1": f14,
            "TP": int(tp4),
            "FP": int(fp4),
            "TN": int(tn4),
            "FN": int(fn4),
        })

    df_cal = pd.DataFrame(calibration_records)
    csv_cal_path = out_dir / "structural_calibration_audit_by_process.csv"
    df_cal.to_csv(csv_cal_path, index=False)
    print(f"\nSaved calibration table -> {csv_cal_path}")

    df_4p = pd.DataFrame(four_process_records)
    csv_4p_path = out_dir / "structural_calibration_audit_four_process_summary.csv"
    df_4p.to_csv(csv_4p_path, index=False)
    print(f"Saved four-process summary table -> {csv_4p_path}")

    print("\n" + "=" * 110)
    print("STRUCTURAL CALIBRATION SUMMARY TABLE ACROSS 9 LAMBDAS (N=531)")
    print("=" * 110)
    cols_print = [
        "lambda", "process", "net_pos_rate_pct", "cf_pos_rate_pct", "rate_diff_pct",
        "roc_auc", "pr_auc", "spearman_rho", "precision", "recall", "f1"
    ]
    print(df_cal[cols_print].to_string(index=False))

    print("\n" + "=" * 110)
    print("FOUR-PROCESS OVERALL CALIBRATION SUMMARY ACROSS 9 LAMBDAS (Total 531 x 4 = 2,124 decisions)")
    print("=" * 110)
    print(df_4p.to_string(index=False))


if __name__ == "__main__":
    main()
