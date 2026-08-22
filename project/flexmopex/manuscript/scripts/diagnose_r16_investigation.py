#!/usr/bin/env python3
"""Flex-MOPEX R16: Diagnostic Investigation of Counterfactual Supervision.

Diagnoses why R15 learned an almost constant w_int (~0.284) across all 671 basins.
Distinguishes competing explanations:
  - Outcome A: Target Compression (q lacks basin discriminability)
  - Outcome B: Linear Head Capacity (linear weights_head cannot map h -> q, but MLP can)
  - Outcome C: Insufficient Optimization (70 steps too few; convergence separates)
  - Outcome D: Moving-Target Dynamics (online q shifts prevent convergence)
  - Outcome E: Representation Drift (online h shifts prevent convergence)
  - Outcome F: BCE Population/Prevalence Dominance (unweighted BCE bias matches mean)

Phases:
  Phase 0: Reconstruct exact R15 supervision dataset across epochs 1..10
  Phase 1: Target discriminability & constant-predictor baseline analysis
  Phase 2: Frozen-feature offline head fitting (R15 budget vs convergence)
  Phase 3: Linear vs MLP architecture capacity probe comparison (Full + 5-fold CV)
  Phase 4: Moving-target and representation-drift replay (Static vs Moving-q vs Moving-h+q)
  Phase 5: BCE gradient anatomy at R15 ep10
  Phase 6: Decision matrix synthesis

Outputs saved to: results/root_cause_r16/
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config
from project.flexmopex.run_model import apply_runtime_overrides, parse_args, _build_data_loader
from project.flexmopex.models.cf_trainer import CounterfactualTargetGenerator
from scripts.diagnose_wint_collapse import build_handler, build_forward, run_loop

OUT_DIR = Path("results/root_cause_r16")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSES = ["w_phen", "w_int", "w_snow", "w_sub"]
GATE_IDX = {"w_phen": 0, "w_int": 1, "w_snow": 2, "w_sub": 3}
COSTS = {"w_phen": 2.0, "w_int": 2.0, "w_snow": 2.0, "w_sub": 1.0}
AIC_ALPHA = 0.01
EPS = 1e-12


def compute_bce(p: np.ndarray, q: np.ndarray) -> float:
    p_c = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(q * np.log(p_c) + (1.0 - q) * np.log(1.0 - p_c)))


def main():
    print("=" * 80)
    print("Flex-MOPEX R16: Root Cause Diagnostic of R15 Counterfactual Supervision")
    print("=" * 80)

    dev = "cuda:0"
    cfg_path = "conf/config_dmopex_interceptE_S0_cf_supervision.yaml"
    c = load_config(cfg_path)
    c_cli = parse_args(["--config", cfg_path, "--gpu-id", "0",
                        "--output-root", "results/intercept_cf_supervision",
                        "--run-name", "E_S0_cf_supervision"])
    apply_runtime_overrides(c, c_cli, config_path=cfg_path)
    c["mode"] = "train"
    c["model"]["phy"]["disable_compile"] = True
    dl = _build_data_loader(c)

    td, ed = dl.train_dataset, dl.eval_dataset
    B = td["x_phy"].shape[1]
    n_attr = td["xc_nn_norm"].shape[-1] - 3
    attrs = td["xc_nn_norm"][0, :, -n_attr:].to(dev)

    # Load Oracle reference from R15 ep10 evaluation
    orc_path = "results/intercept_cf_supervision/E_S0_cf_supervision/process_oracle_table_ep10.csv"
    df_orc = pd.read_csv(orc_path)
    oracle_dict = {}
    for proc in PROCESSES:
        sub = df_orc[df_orc["process"] == proc]
        oracle_dict[proc] = sub["w_star"].values  # [671]

    handler = build_handler(c)
    target_gen = CounterfactualTargetGenerator(c, device=dev)

    # =========================================================================
    # Phase 0: Reconstruct Exact R15 Supervision Dataset Across Epochs 1..10
    # =========================================================================
    print("\n[Phase 0] Reconstructing exact R15 supervision states across epochs 1..10...")
    r15_states = {}
    phase0_records = []

    for ep in range(1, 11):
        handler.load_model(ep)
        for m in handler.model_dict.values():
            m.eval()
        model = next(iter(handler.model_dict.values()))
        phy_m, nn_m = model.phy_model, model.nn_model

        with torch.no_grad():
            h_repr = nn_m.backbone(attrs).detach()  # [671, 128]
            raw_w = nn_m.heads["weights"](h_repr)  # [671, 8]
            raw_w_clamped = torch.clamp(raw_w, min=-10.0, max=10.0)
            logits = raw_w_clamped.view(B, 4, 2)
            z_contrast = logits[..., 1] - logits[..., 0]  # [671, 4]
            p_struct = torch.sigmoid(z_contrast).cpu().numpy()  # [671, 4]

            # Reconstruct exact epoch targets
            q_targets, diag = target_gen.generate_targets(handler, td)
            q_np = q_targets.cpu().numpy()  # [671, 4]

            # Extract weights_head parameters
            w_head_W = nn_m.heads["weights"].weight.detach().cpu().numpy()  # [8, 128]
            w_head_b = nn_m.heads["weights"].bias.detach().cpu().numpy()    # [8]

        r15_states[ep] = {
            "h": h_repr.cpu().numpy(),
            "p_struct": p_struct,
            "q": q_np,
            "diag": diag,
            "W_head": w_head_W,
            "b_head": w_head_b,
        }

        for proc in PROCESSES:
            col = GATE_IDX[proc]
            for b_idx in range(B):
                phase0_records.append({
                    "epoch": ep,
                    "basin_idx": b_idx,
                    "process": proc,
                    "q": q_np[b_idx, col],
                    "p_struct": p_struct[b_idx, col],
                    "w_oracle": oracle_dict[proc][b_idx],
                })

    df_p0 = pd.DataFrame(phase0_records)
    df_p0.to_csv(OUT_DIR / "phase0_reconstructed_supervision.csv", index=False)
    print(f"[Phase 0 Complete] Reconstructed {len(df_p0)} basin-epoch records.")

    # =========================================================================
    # Phase 1: Target Discriminability & Constant-Predictor Baseline
    # =========================================================================
    print("\n[Phase 1] Analyzing target discriminability and constant-predictor baselines...")
    phase1_summary = []

    for ep in [1, 5, 10]:
        st = r15_states[ep]
        for proc in PROCESSES:
            col = GATE_IDX[proc]
            q_p = st["q"][:, col]
            p_p = st["p_struct"][:, col]
            w_star = oracle_dict[proc]
            orc_pos = (w_star > 0)
            n_orc_pos = int(np.sum(orc_pos))

            p_const = float(np.mean(q_p))
            bce_const = compute_bce(np.full_like(q_p, p_const), q_p)
            bce_actual = compute_bce(p_p, q_p)

            # Target statistics
            q_mean = float(np.mean(q_p))
            q_std = float(np.std(q_p))
            q_med = float(np.median(q_p))
            q_iqr = float(np.percentile(q_p, 75) - np.percentile(q_p, 25))
            q_p05 = float(np.percentile(q_p, 5))
            q_p95 = float(np.percentile(q_p, 95))
            frac_gt05 = float(np.mean(q_p > 0.5))

            # Group separation by Oracle
            q_pos_mean = float(np.mean(q_p[orc_pos])) if n_orc_pos > 0 else 0.0
            q_zero_mean = float(np.mean(q_p[~orc_pos])) if (B - n_orc_pos) > 0 else 0.0
            q_separation = q_pos_mean - q_zero_mean

            # Prediction separation by Oracle
            p_pos_mean = float(np.mean(p_p[orc_pos])) if n_orc_pos > 0 else 0.0
            p_zero_mean = float(np.mean(p_p[~orc_pos])) if (B - n_orc_pos) > 0 else 0.0
            p_separation = p_pos_mean - p_zero_mean

            # Ranking power of target q
            roc_q = float(roc_auc_score(orc_pos, q_p)) if n_orc_pos > 0 else 0.5
            pr_q = float(average_precision_score(orc_pos, q_p)) if n_orc_pos > 0 else 0.0
            sp_q, _ = spearmanr(q_p, w_star)

            # Ranking power of prediction p
            roc_p = float(roc_auc_score(orc_pos, p_p)) if n_orc_pos > 0 else 0.5
            pr_p = float(average_precision_score(orc_pos, p_p)) if n_orc_pos > 0 else 0.0
            sp_p, _ = spearmanr(p_p, w_star)

            phase1_summary.append({
                "epoch": ep,
                "process": proc,
                "T_scale": st["diag"][proc]["T_scale"],
                "q_mean": q_mean,
                "q_std": q_std,
                "q_median": q_med,
                "q_iqr": q_iqr,
                "q_p05": q_p05,
                "q_p95": q_p95,
                "frac_q_gt05": frac_gt05,
                "p_const": p_const,
                "bce_const": bce_const,
                "bce_actual": bce_actual,
                "q_oracle_pos_mean": q_pos_mean,
                "q_oracle_zero_mean": q_zero_mean,
                "q_oracle_separation": q_separation,
                "p_oracle_pos_mean": p_pos_mean,
                "p_oracle_zero_mean": p_zero_mean,
                "p_oracle_separation": p_separation,
                "roc_auc_q_for_oracle": roc_q,
                "pr_auc_q_for_oracle": pr_q,
                "spearman_q_oracle": float(sp_q),
                "roc_auc_p_for_oracle": roc_p,
                "pr_auc_p_for_oracle": pr_p,
                "spearman_p_oracle": float(sp_p),
            })

    df_p1 = pd.DataFrame(phase1_summary)
    df_p1.to_csv(OUT_DIR / "phase1_target_discriminability.csv", index=False)
    print("[Phase 1 Complete] Saved target discriminability table.")

    # =========================================================================
    # Phase 2: Frozen-Feature Offline Head Fitting (R15 Budget vs Convergence)
    # =========================================================================
    print("\n[Phase 2] Training offline isolated weights_head on (h_ep10, q_ep10)...")
    h_ep10 = torch.from_numpy(r15_states[10]["h"]).float().to(dev)  # [671, 128]
    q_ep10 = torch.from_numpy(r15_states[10]["q"]).float().to(dev)  # [671, 4]

    # Setup offline head
    def build_isolated_head(seed=42):
        torch.manual_seed(seed)
        head = nn.Linear(128, 8, bias=True).to(dev)
        nn.init.normal_(head.weight, mean=0.0, std=0.001)
        nn.init.constant_(head.bias, 0.0)
        return head

    def predict_probs(head, h_tensor):
        raw = torch.clamp(head(h_tensor), min=-10.0, max=10.0).view(-1, 4, 2)
        return torch.sigmoid(raw[..., 1] - raw[..., 0])

    # 2A: R15 Budget (70 steps on batch_size=100)
    head_2a = build_isolated_head(seed=42)
    opt_2a = torch.optim.Adadelta(head_2a.parameters(), lr=1.0)
    batch_size = 100
    n_steps_r15 = 70

    torch.manual_seed(42)
    loss_history_2a = []
    for step in range(n_steps_r15):
        idx = torch.randint(0, B, (batch_size,), device=dev)
        h_b = h_ep10[idx]
        q_b = q_ep10[idx]

        opt_2a.zero_grad()
        p_b = predict_probs(head_2a, h_b)
        loss = F.binary_cross_entropy(p_b, q_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_2a.parameters(), max_norm=1.0)
        opt_2a.step()
        loss_history_2a.append(float(loss.item()))

    # 2B: Full Convergence (Up to 2000 steps)
    head_2b = build_isolated_head(seed=42)
    opt_2b = torch.optim.Adadelta(head_2b.parameters(), lr=1.0)
    max_steps_conv = 2000
    patience = 50
    min_delta = 1e-6
    best_loss = float("inf")
    best_step = 0

    torch.manual_seed(42)
    loss_history_2b = []
    for step in range(max_steps_conv):
        idx = torch.randint(0, B, (batch_size,), device=dev)
        h_b = h_ep10[idx]
        q_b = q_ep10[idx]

        opt_2b.zero_grad()
        p_b = predict_probs(head_2b, h_b)
        loss = F.binary_cross_entropy(p_b, q_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_2b.parameters(), max_norm=1.0)
        opt_2b.step()
        l_val = float(loss.item())
        loss_history_2b.append(l_val)

        if l_val < best_loss - min_delta:
            best_loss = l_val
            best_step = step
        elif step - best_step > patience:
            print(f"  [Convergence] Plateau reached at step {step} (best step {best_step}, loss={best_loss:.5f})")
            break

    # Evaluate 2A and 2B
    with torch.no_grad():
        p_2a = predict_probs(head_2a, h_ep10).cpu().numpy()
        p_2b = predict_probs(head_2b, h_ep10).cpu().numpy()
        q_10_np = q_ep10.cpu().numpy()

    phase2_rows = []
    for proc in PROCESSES:
        col = GATE_IDX[proc]
        q_p = q_10_np[:, col]
        w_star = oracle_dict[proc]
        orc_pos = (w_star > 0)
        p_const = float(np.mean(q_p))
        bce_const = compute_bce(np.full_like(q_p, p_const), q_p)

        # 2A metrics
        p_a = p_2a[:, col]
        bce_2a = compute_bce(p_a, q_p)
        sp_2a_q, _ = spearmanr(p_a, q_p)
        pe_2a_q, _ = pearsonr(p_a, q_p)
        sp_2a_w, _ = spearmanr(p_a, w_star)
        roc_2a = float(roc_auc_score(orc_pos, p_a)) if np.sum(orc_pos) > 0 else 0.5

        # 2B metrics
        p_b = p_2b[:, col]
        bce_2b = compute_bce(p_b, q_p)
        sp_2b_q, _ = spearmanr(p_b, q_p)
        pe_2b_q, _ = pearsonr(p_b, q_p)
        sp_2b_w, _ = spearmanr(p_b, w_star)
        roc_2b = float(roc_auc_score(orc_pos, p_b)) if np.sum(orc_pos) > 0 else 0.5

        phase2_rows.append({
            "process": proc,
            "bce_const": bce_const,
            # 2A (R15 Budget 70 steps)
            "bce_2a_70steps": bce_2a,
            "p_2a_mean": float(np.mean(p_a)),
            "p_2a_std": float(np.std(p_a)),
            "p_2a_min": float(np.min(p_a)),
            "p_2a_max": float(np.max(p_a)),
            "p_2a_pos_mean": float(np.mean(p_a[orc_pos])),
            "p_2a_zero_mean": float(np.mean(p_a[~orc_pos])),
            "pearson_2a_q": float(pe_2a_q),
            "spearman_2a_q": float(sp_2a_q),
            "spearman_2a_oracle": float(sp_2a_w),
            "roc_auc_2a_oracle": roc_2a,
            # 2B (Convergence ~2000 steps)
            "steps_to_conv_2b": len(loss_history_2b),
            "bce_2b_conv": bce_2b,
            "p_2b_mean": float(np.mean(p_b)),
            "p_2b_std": float(np.std(p_b)),
            "p_2b_min": float(np.min(p_b)),
            "p_2b_max": float(np.max(p_b)),
            "p_2b_pos_mean": float(np.mean(p_b[orc_pos])),
            "p_2b_zero_mean": float(np.mean(p_b[~orc_pos])),
            "pearson_2b_q": float(pe_2b_q),
            "spearman_2b_q": float(sp_2b_q),
            "spearman_2b_oracle": float(sp_2b_w),
            "roc_auc_2b_oracle": roc_2b,
        })

    df_p2 = pd.DataFrame(phase2_rows)
    df_p2.to_csv(OUT_DIR / "phase2_offline_head_fitting.csv", index=False)
    print("[Phase 2 Complete] Saved offline head fitting results.")

    # =========================================================================
    # Phase 3: Linear vs MLP Capacity Probe Comparison
    # =========================================================================
    print("\n[Phase 3] Comparing Linear Probe L vs MLP Probe M (Full fit & 5-fold CV)...")
    # Probe L: Linear 128 -> 8
    # Probe M: MLP 128 -> 64 -> 8 (Tanh activation)
    class MLPProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 8),
            )
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

        def forward(self, x):
            raw = torch.clamp(self.net(x), min=-10.0, max=10.0).view(-1, 4, 2)
            return torch.sigmoid(raw[..., 1] - raw[..., 0])

    class LinearProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(128, 8)
            nn.init.xavier_uniform_(self.net.weight)
            nn.init.constant_(self.net.bias, 0.0)

        def forward(self, x):
            raw = torch.clamp(self.net(x), min=-10.0, max=10.0).view(-1, 4, 2)
            return torch.sigmoid(raw[..., 1] - raw[..., 0])

    def train_probe_model(model_cls, h_tr, q_tr, steps=500, lr=0.01, seed=42):
        torch.manual_seed(seed)
        m = model_cls().to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(steps):
            opt.zero_grad()
            p = m(h_tr)
            l = F.binary_cross_entropy(p, q_tr)
            l.backward()
            opt.step()
        return m

    # 1. Full Fit
    m_lin_full = train_probe_model(LinearProbe, h_ep10, q_ep10, steps=1000, lr=0.01)
    m_mlp_full = train_probe_model(MLPProbe, h_ep10, q_ep10, steps=1000, lr=0.01)

    with torch.no_grad():
        p_lin_full = m_lin_full(h_ep10).cpu().numpy()
        p_mlp_full = m_mlp_full(h_ep10).cpu().numpy()

    # 2. 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Stratify by w_int target q > 0.5
    y_strat = (q_10_np[:, GATE_IDX["w_int"]] > 0.5).astype(int)

    oof_p_lin = np.zeros((B, 4))
    oof_p_mlp = np.zeros((B, 4))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(B), y_strat)):
        h_tr, q_tr = h_ep10[tr_idx], q_ep10[tr_idx]
        h_val = h_ep10[val_idx]

        m_lin_fold = train_probe_model(LinearProbe, h_tr, q_tr, steps=800, lr=0.01, seed=42 + fold)
        m_mlp_fold = train_probe_model(MLPProbe, h_tr, q_tr, steps=800, lr=0.01, seed=42 + fold)

        with torch.no_grad():
            oof_p_lin[val_idx] = m_lin_fold(h_val).cpu().numpy()
            oof_p_mlp[val_idx] = m_mlp_fold(h_val).cpu().numpy()

    phase3_rows = []
    for proc in PROCESSES:
        col = GATE_IDX[proc]
        q_p = q_10_np[:, col]
        w_star = oracle_dict[proc]
        orc_pos = (w_star > 0)
        target_pos = (q_p > 0.5)

        # Full Fit Metrics
        bce_lin_full = compute_bce(p_lin_full[:, col], q_p)
        bce_mlp_full = compute_bce(p_mlp_full[:, col], q_p)
        sp_lin_full, _ = spearmanr(p_lin_full[:, col], q_p)
        sp_mlp_full, _ = spearmanr(p_mlp_full[:, col], q_p)
        pe_lin_full, _ = pearsonr(p_lin_full[:, col], q_p)
        pe_mlp_full, _ = pearsonr(p_mlp_full[:, col], q_p)

        # OOF 5-Fold Metrics
        bce_lin_oof = compute_bce(oof_p_lin[:, col], q_p)
        bce_mlp_oof = compute_bce(oof_p_mlp[:, col], q_p)
        sp_lin_oof, _ = spearmanr(oof_p_lin[:, col], q_p)
        sp_mlp_oof, _ = spearmanr(oof_p_mlp[:, col], q_p)
        pe_lin_oof, _ = pearsonr(oof_p_lin[:, col], q_p)
        pe_mlp_oof, _ = pearsonr(oof_p_mlp[:, col], q_p)

        roc_lin_oof = float(roc_auc_score(target_pos, oof_p_lin[:, col])) if np.sum(target_pos) > 0 else 0.5
        roc_mlp_oof = float(roc_auc_score(target_pos, oof_p_mlp[:, col])) if np.sum(target_pos) > 0 else 0.5
        pr_lin_oof = float(average_precision_score(target_pos, oof_p_lin[:, col])) if np.sum(target_pos) > 0 else 0.0
        pr_mlp_oof = float(average_precision_score(target_pos, oof_p_mlp[:, col])) if np.sum(target_pos) > 0 else 0.0

        roc_lin_orc = float(roc_auc_score(orc_pos, oof_p_lin[:, col])) if np.sum(orc_pos) > 0 else 0.5
        roc_mlp_orc = float(roc_auc_score(orc_pos, oof_p_mlp[:, col])) if np.sum(orc_pos) > 0 else 0.5

        phase3_rows.append({
            "process": proc,
            # Full Fit
            "bce_lin_full": bce_lin_full,
            "bce_mlp_full": bce_mlp_full,
            "pearson_lin_full": float(pe_lin_full),
            "pearson_mlp_full": float(pe_mlp_full),
            "spearman_lin_full": float(sp_lin_full),
            "spearman_mlp_full": float(sp_mlp_full),
            "std_lin_full": float(np.std(p_lin_full[:, col])),
            "std_mlp_full": float(np.std(p_mlp_full[:, col])),
            # OOF 5-Fold
            "bce_lin_oof": bce_lin_oof,
            "bce_mlp_oof": bce_mlp_oof,
            "pearson_lin_oof": float(pe_lin_oof),
            "pearson_mlp_oof": float(pe_mlp_oof),
            "spearman_lin_oof": float(sp_lin_oof),
            "spearman_mlp_oof": float(sp_mlp_oof),
            "std_lin_oof": float(np.std(oof_p_lin[:, col])),
            "std_mlp_oof": float(np.std(oof_p_mlp[:, col])),
            "roc_auc_lin_target_pos": roc_lin_oof,
            "roc_auc_mlp_target_pos": roc_mlp_oof,
            "pr_auc_lin_target_pos": pr_lin_oof,
            "pr_auc_mlp_target_pos": pr_mlp_oof,
            "roc_auc_lin_oracle_pos": roc_lin_orc,
            "roc_auc_mlp_oracle_pos": roc_mlp_orc,
        })

    df_p3 = pd.DataFrame(phase3_rows)
    df_p3.to_csv(OUT_DIR / "phase3_capacity_comparison.csv", index=False)
    print("[Phase 3 Complete] Saved Linear vs MLP capacity comparison table.")

    # =========================================================================
    # Phase 4: Moving-Target and Representation-Drift Replay
    # =========================================================================
    print("\n[Phase 4] Running offline replay: Static vs Moving-q vs Moving-h+q...")
    # 3 Conditions:
    # 1. Static: h = h_ep10, q = q_ep10 (all 10 replay epochs)
    # 2. Moving-q: h = h_ep10, q = q_ep1..10
    # 3. Moving-h + Moving-q: h = h_ep1..10, q = q_ep1..10
    def run_replay_regime(mode: str, seed=42):
        torch.manual_seed(seed)
        head = build_isolated_head(seed=seed)
        opt = torch.optim.Adadelta(head.parameters(), lr=1.0)
        epoch_stats = []

        for ep in range(1, 11):
            if mode == "static":
                h_cur = torch.from_numpy(r15_states[10]["h"]).float().to(dev)
                q_cur = torch.from_numpy(r15_states[10]["q"]).float().to(dev)
            elif mode == "moving_q":
                h_cur = torch.from_numpy(r15_states[10]["h"]).float().to(dev)
                q_cur = torch.from_numpy(r15_states[ep]["q"]).float().to(dev)
            elif mode == "moving_h_q":
                h_cur = torch.from_numpy(r15_states[ep]["h"]).float().to(dev)
                q_cur = torch.from_numpy(r15_states[ep]["q"]).float().to(dev)

            # 7 steps per epoch (same as R15)
            for _ in range(7):
                idx = torch.randint(0, B, (100,), device=dev)
                opt.zero_grad()
                p_b = predict_probs(head, h_cur[idx])
                loss = F.binary_cross_entropy(p_b, q_cur[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
                opt.step()

            # End of epoch evaluation against final q_ep10
            with torch.no_grad():
                h_eval = torch.from_numpy(r15_states[10]["h"]).float().to(dev)
                p_eval = predict_probs(head, h_eval).cpu().numpy()
                q_10 = r15_states[10]["q"]

            for proc in PROCESSES:
                col = GATE_IDX[proc]
                q_p = q_10[:, col]
                p_p = p_eval[:, col]
                w_star = oracle_dict[proc]
                orc_pos = (w_star > 0)
                pe_r, _ = pearsonr(p_p, q_p)
                epoch_stats.append({
                    "mode": mode,
                    "epoch": ep,
                    "process": proc,
                    "bce_vs_q10": compute_bce(p_p, q_p),
                    "p_mean": float(np.mean(p_p)),
                    "p_std": float(np.std(p_p)),
                    "p_pos_mean": float(np.mean(p_p[orc_pos])) if np.sum(orc_pos) > 0 else 0.0,
                    "p_zero_mean": float(np.mean(p_p[~orc_pos])) if np.sum(~orc_pos) > 0 else 0.0,
                    "pearson_vs_q10": float(pe_r),
                })
        return epoch_stats

    replay_static = run_replay_regime("static")
    replay_mov_q = run_replay_regime("moving_q")
    replay_mov_hq = run_replay_regime("moving_h_q")

    df_p4 = pd.DataFrame(replay_static + replay_mov_q + replay_mov_hq)
    df_p4.to_csv(OUT_DIR / "phase4_replay_trajectories.csv", index=False)
    print("[Phase 4 Complete] Saved moving-target and representation-drift replay trajectories.")

    # =========================================================================
    # Phase 5: BCE Gradient Anatomy at R15 ep10
    # =========================================================================
    print("\n[Phase 5] Decomposing BCE gradient anatomy for w_int at R15 ep10...")
    h_10 = torch.from_numpy(r15_states[10]["h"]).float().to(dev)  # [671, 128]
    q_10 = torch.from_numpy(r15_states[10]["q"]).float().to(dev)  # [671, 4]
    p_10 = torch.from_numpy(r15_states[10]["p_struct"]).float().to(dev)  # [671, 4]

    wint_col = GATE_IDX["w_int"]
    q_wint = q_10[:, wint_col]
    p_wint = p_10[:, wint_col]
    e_wint = p_wint - q_wint  # Residual / logit gradient [671]

    # Partition into 3 robust groups based on DeltaJ / q:
    # 1. Strong-ON: q > 0.60
    # 2. Strong-OFF: q < 0.20
    # 3. Weak/Ambiguous: 0.20 <= q <= 0.60
    q_np = q_wint.cpu().numpy()
    p_np = p_wint.cpu().numpy()
    e_np = e_wint.cpu().numpy()

    mask_on = (q_np > 0.60)
    mask_off = (q_np < 0.20)
    mask_weak = (~mask_on) & (~mask_off)

    groups = [
        ("Strong-ON (q > 0.60)", mask_on),
        ("Strong-OFF (q < 0.20)", mask_off),
        ("Weak/Ambiguous (0.20 <= q <= 0.60)", mask_weak),
        ("Full Population (All 671)", np.ones(B, dtype=bool)),
    ]

    # Per-basin gradient vectors in 129-D: G_i = e_i * [h_i, 1]
    G_W_i = e_wint.unsqueeze(-1) * h_10        # [671, 128]
    G_b_i = e_wint.unsqueeze(-1)               # [671, 1]
    G_param_i = torch.cat([G_W_i, G_b_i], dim=-1)  # [671, 129]

    v_full = G_param_i.sum(dim=0)  # [129]
    norm_v_full = float(v_full.norm())

    gradient_anatomy = []
    group_vectors = {}

    for g_name, g_mask in groups:
        g_t = torch.from_numpy(g_mask).to(dev)
        n_g = int(g_mask.sum())
        sub_param = G_param_i[g_t]
        v_g = sub_param.sum(dim=0)  # [129]
        norm_v_g = float(v_g.norm())
        group_vectors[g_name] = v_g

        # Bias vs Feature Weight energy
        v_W_g = G_W_i[g_t].sum(dim=0)
        v_b_g = G_b_i[g_t].sum(dim=0)
        norm_W_g = float(v_W_g.norm())
        norm_b_g = float(v_b_g.abs().item())
        bias_energy_frac = float((norm_b_g ** 2) / (norm_v_g ** 2 + EPS))

        cos_with_full = float(F.cosine_similarity(v_g.unsqueeze(0), v_full.unsqueeze(0))[0]) if norm_v_g > 0 and norm_v_full > 0 else 0.0

        gradient_anatomy.append({
            "group": g_name,
            "count": n_g,
            "frac_population": float(n_g / B),
            "mean_target_q": float(np.mean(q_np[g_mask])) if n_g > 0 else 0.0,
            "mean_pred_p": float(np.mean(p_np[g_mask])) if n_g > 0 else 0.0,
            "mean_residual_p_minus_q": float(np.mean(e_np[g_mask])) if n_g > 0 else 0.0,
            "sum_residual_bias_grad": float(v_b_g.item()),
            "norm_full_param_grad": norm_v_g,
            "norm_weight_grad": norm_W_g,
            "norm_bias_grad": norm_b_g,
            "bias_energy_frac": bias_energy_frac,
            "cos_with_full_update": cos_with_full,
        })

    # Cosine between Strong-ON and Strong-OFF group gradients
    cos_on_off = float(F.cosine_similarity(group_vectors["Strong-ON (q > 0.60)"].unsqueeze(0),
                                          group_vectors["Strong-OFF (q < 0.20)"].unsqueeze(0))[0])
    ratio_off_over_on = float(group_vectors["Strong-OFF (q < 0.20)"].norm() / (group_vectors["Strong-ON (q > 0.60)"].norm() + EPS))

    anatomy_summary = {
        "groups": gradient_anatomy,
        "cos_strong_on_vs_strong_off": cos_on_off,
        "norm_ratio_strong_off_over_strong_on": ratio_off_over_on,
        "constant_predictor_bce": float(df_p1[(df_p1["epoch"] == 10) & (df_p1["process"] == "w_int")]["bce_const"].iloc[0]),
        "actual_r15_ep10_bce": float(df_p1[(df_p1["epoch"] == 10) & (df_p1["process"] == "w_int")]["bce_actual"].iloc[0]),
        "bce_improvement_over_const": float(df_p1[(df_p1["epoch"] == 10) & (df_p1["process"] == "w_int")]["bce_const"].iloc[0] - df_p1[(df_p1["epoch"] == 10) & (df_p1["process"] == "w_int")]["bce_actual"].iloc[0]),
    }

    df_p5 = pd.DataFrame(gradient_anatomy)
    df_p5.to_csv(OUT_DIR / "phase5_gradient_anatomy.csv", index=False)
    p5_json = OUT_DIR / "phase5_gradient_summary.json"
    p5_json.write_text(json.dumps(anatomy_summary, indent=2))
    print(f"[Phase 5 Complete] Saved gradient anatomy summary to {p5_json}")

    print("\n" + "=" * 80)
    print("ALL R16 DIAGNOSTICS COMPLETED SUCCESSFULLY")
    print(f"Artifacts saved in {OUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
