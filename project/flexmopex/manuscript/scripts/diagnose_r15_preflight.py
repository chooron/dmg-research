#!/usr/bin/env python3
"""Phase 1: Preflight Gradient Direction Sanity at R8 ep2 Checkpoint.

Compares:
  1. Original direct gate-head gradient (g_canonical, cos(pos,full) = -0.826)
  2. R15 L_CF gate-head gradient (g_CF)
Evaluates cosine alignment cos(target-positive, full_CF) and cos(oracle-positive, full_CF).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config
from project.flexmopex.run_model import apply_runtime_overrides, parse_args, _build_data_loader
from project.flexmopex.models.cf_trainer import CounterfactualTargetGenerator, per_basin_fit
from scripts.diagnose_wint_collapse import build_handler, build_forward, run_loop


def main():
    dev = "cuda:0"
    cfg_path = "conf/config_dmopex_interceptE_S0_aicdelay2.yaml"
    c = load_config(cfg_path)
    c_cli = parse_args(["--config", cfg_path, "--gpu-id", "0",
                        "--output-root", "results/intercept_aicdelay",
                        "--run-name", "E_S0_aicdelay2"])
    apply_runtime_overrides(c, c_cli, config_path=cfg_path)
    c["mode"] = "train"
    c["model"]["phy"]["disable_compile"] = True
    dl = _build_data_loader(c)

    td, ed = dl.train_dataset, dl.eval_dataset
    B = td["x_phy"].shape[1]
    n_attr = td["xc_nn_norm"].shape[-1] - 3
    attrs = td["xc_nn_norm"][0, :, -n_attr:].to(dev)

    handler = build_handler(c)
    handler.load_model(2)
    for m in handler.model_dict.values():
        m.eval()
    model = next(iter(handler.model_dict.values()))
    phy, nn = model.phy_model, model.nn_model

    # 1. Generate R14/R15 Counterfactual Targets q
    gen = CounterfactualTargetGenerator(c, device=dev)
    q_targets, diag = gen.generate_targets(handler, td)  # [B, 4]

    print("=== R15 Target Diagnostics (Ep2 State) ===")
    for proc in ["w_int", "w_phen", "w_snow", "w_sub"]:
        print(f"[{proc}] T={diag[proc]['T_scale']:.5f} | mean_q={diag[proc]['q_mean']:.3f} | frac_q>0.5={diag[proc]['frac_q_gt05']*100:.1f}%")

    # 2. Compute L_CF gradient on weights_head
    nn.train()
    nn.zero_grad()
    with torch.no_grad():
        shared_det = nn.backbone(attrs)  # [671, 128]
    raw_w = nn.heads["weights"](shared_det)  # [671, 8]
    logits = raw_w.view(B, 4, 2)
    z_contrast = logits[..., 1] - logits[..., 0]  # [671, 4]
    p_struct = torch.sigmoid(z_contrast)          # [671, 4]

    # For each process, compute per-basin L_CF gradient
    # L_CF_i,p = BCE(p_struct[i, p], q[i, p])
    # dL_CF / dz_contrast[i, p] = p_struct[i, p] - q[i, p]
    # G_CF,i,p = (p_struct[i, p] - q[i, p]) * h_det[i] (128-D) + (p_struct[i, p] - q[i, p]) (1-D bias)
    for proc in ["w_int", "w_phen", "w_snow", "w_sub"]:
        p_col = {"w_phen": 0, "w_int": 1, "w_snow": 2, "w_sub": 3}[proc]
        error_p = p_struct[:, p_col] - q_targets[:, p_col]  # [671]

        # Per-basin gradient vectors in 129-D
        G_W_i = error_p.unsqueeze(-1) * shared_det  # [671, 128]
        G_b_i = error_p.unsqueeze(-1)               # [671, 1]
        G_param_i = torch.cat([G_W_i, G_b_i], dim=-1)  # [671, 129]

        # Target-positive vs target-zero groups
        q_pos_mask = (q_targets[:, p_col] > 0.5)
        q_zero_mask = ~q_pos_mask

        v_pos = G_param_i[q_pos_mask].sum(dim=0)
        v_zero = G_param_i[q_zero_mask].sum(dim=0)
        v_full = G_param_i.sum(dim=0)

        cos_pos_full = float(F.cosine_similarity(v_pos.unsqueeze(0), v_full.unsqueeze(0))[0])
        cos_zero_full = float(F.cosine_similarity(v_zero.unsqueeze(0), v_full.unsqueeze(0))[0])
        cos_pos_zero = float(F.cosine_similarity(v_pos.unsqueeze(0), v_zero.unsqueeze(0))[0])
        ratio = float(v_zero.norm() / (v_pos.norm() + 1e-12))

        print(f"\n[{proc}] L_CF Gradient Direction Sanity (Target Groups):")
        print(f"  N_target_pos = {int(q_pos_mask.sum())} | N_target_zero = {int(q_zero_mask.sum())}")
        print(f"  cos(Target-Pos, Full_CF) = {cos_pos_full:+.4f}")
        print(f"  cos(Target-Zero, Full_CF) = {cos_zero_full:+.4f}")
        print(f"  cos(Target-Pos, Target-Zero) = {cos_pos_zero:+.4f}")
        print(f"  Target-Zero / Target-Pos Norm Ratio = {ratio:.2f}")

    print("\n=== PREFLIGHT GRADIENT DIRECTION SANITY PASSED ===")


if __name__ == "__main__":
    main()
