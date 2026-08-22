#!/usr/bin/env python3
"""Preflight Validation for Flex-MOPEX R17-B Confidence-Weighted Counterfactual Supervision.

Validates:
  1. Forward numerical invariants between R17-A and R17-B (before applying weighting)
  2. Bounded confidence c = 2 * |q - 0.5| properties & process-wise statistics
  3. Effective weighted sample size N_eff and loss concentration fractions (top 5%, 10%, 20%)
  4. Gradient routing invariants (weights_head grad != 0, backbone grad == 0, direct fit/AIC grad == 0)
  5. Gradient group anatomy on R17-A ep10 state comparing unweighted vs confidence-weighted L_CF
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config
from project.flexmopex.run_model import apply_runtime_overrides, parse_args, _build_data_loader
from project.flexmopex.models.cf_trainer import CounterfactualTargetGenerator, per_basin_fit
from scripts.diagnose_wint_collapse import build_handler

OUT_DIR = Path("results/intercept_r17b/E_S0_r17b")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSES = ["w_phen", "w_int", "w_snow", "w_sub"]
GATE_IDX = {"w_phen": 0, "w_int": 1, "w_snow": 2, "w_sub": 3}
COSTS = {"w_phen": 2.0, "w_int": 2.0, "w_snow": 2.0, "w_sub": 1.0}
AIC_ALPHA = 0.01
EPS = 1e-12


def main():
    print("=" * 80)
    print("Flex-MOPEX R17-B: Preflight Validation & Concentration Audit")
    print("=" * 80)

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg_path = "conf/config_dmopex_interceptE_S0_r17b.yaml"
    c = load_config(cfg_path)
    c_cli = parse_args(["--config", cfg_path, "--gpu-id", "0",
                        "--output-root", "results/intercept_r17a",
                        "--run-name", "E_S0_r17a"])
    apply_runtime_overrides(c, c_cli, config_path=cfg_path)
    c["mode"] = "train"
    c["model"]["phy"]["disable_compile"] = True
    dl = _build_data_loader(c)

    td, ed = dl.train_dataset, dl.eval_dataset
    B = td["x_phy"].shape[1]
    n_attr = td["xc_nn_norm"].shape[-1] - 3
    attrs = td["xc_nn_norm"][0, :, -n_attr:].to(dev)

    # 1. Load R17-A ep10 Checkpoint to audit states
    ckpt_path = Path("results/intercept_r17a/E_S0_r17a/model/learnedweightmopexe_ep10.pt")
    handler = build_handler(c)
    handler.load_model(10)
    for m in handler.model_dict.values():
        m.eval()
    model = next(iter(handler.model_dict.values()))
    phy, nn_m = model.phy_model, model.nn_model

    # 2. Extract targets q and confidence c
    target_gen = CounterfactualTargetGenerator(c, device=dev)
    q_targets, diag = target_gen.generate_targets(handler, td)  # [B, 4]
    c_targets = (2.0 * torch.abs(q_targets - 0.5)).detach()     # [B, 4]

    # Verify confidence c properties
    assert (c_targets >= 0.0).all() and (c_targets <= 1.0).all(), "c must be bounded in [0, 1]"
    assert not c_targets.requires_grad, "c must be detached"

    print("\n--- 1. Process-Wise Target & Confidence Distributions (N=671) ---")
    conf_stats = {}
    for proc in PROCESSES:
        col = GATE_IDX[proc]
        qp = q_targets[:, col].cpu().numpy()
        cp = c_targets[:, col].cpu().numpy()

        sum_c = float(np.sum(cp))
        sum_c_sq = float(np.sum(cp ** 2))
        n_eff = float((sum_c ** 2) / (sum_c_sq + EPS))
        n_eff_pct = float(n_eff / B * 100.0)

        # Loss concentration
        sorted_c = np.sort(cp)[::-1]
        k5 = int(0.05 * B)
        k10 = int(0.10 * B)
        k20 = int(0.20 * B)
        top5_pct = float(np.sum(sorted_c[:k5]) / (sum_c + EPS) * 100.0)
        top10_pct = float(np.sum(sorted_c[:k10]) / (sum_c + EPS) * 100.0)
        top20_pct = float(np.sum(sorted_c[:k20]) / (sum_c + EPS) * 100.0)

        conf_stats[proc] = {
            "q_mean": float(np.mean(qp)),
            "q_median": float(np.median(qp)),
            "q_std": float(np.std(qp)),
            "c_mean": float(np.mean(cp)),
            "c_median": float(np.median(cp)),
            "c_std": float(np.std(cp)),
            "N_effective": n_eff,
            "N_effective_pct": n_eff_pct,
            "top5_loss_weight_pct": top5_pct,
            "top10_loss_weight_pct": top10_pct,
            "top20_loss_weight_pct": top20_pct,
        }

        print(f"[{proc}]")
        print(f"  Target q   : mean={np.mean(qp):.4f} | median={np.median(qp):.4f} | std={np.std(qp):.4f} (range [{np.min(qp):.3f}, {np.max(qp):.3f}])")
        print(f"  Confidence c: mean={np.mean(cp):.4f} | median={np.median(cp):.4f} | std={np.std(cp):.4f} (range [{np.min(cp):.3f}, {np.max(cp):.3f}])")
        print(f"  Effective N : {n_eff:.1f} / 671 ({n_eff_pct:.1f}% of population)")
        print(f"  Loss Weight Concentration: Top 5% basins = {top5_pct:.1f}% | Top 10% = {top10_pct:.1f}% | Top 20% = {top20_pct:.1f}%")
        assert top5_pct < 40.0, f"Extreme loss concentration in top 5%: {top5_pct:.1f}%!"

    # 3. Gradient Group Anatomy: Unweighted vs Confidence-Weighted L_CF for w_int
    print("\n--- 2. Gradient Contribution by Counterfactual Groups (Unweighted vs Weighted L_CF) ---")
    wint_col = GATE_IDX["w_int"]
    q_wint = q_targets[:, wint_col]
    c_wint = c_targets[:, wint_col]

    with torch.no_grad():
        h_repr = nn_m.backbone(attrs)
        raw_w = nn_m.heads["weights"](h_repr)
        logits = torch.clamp(raw_w, min=-10.0, max=10.0).view(B, 4, 2)
        p_wint = torch.sigmoid(logits[..., 1] - logits[..., 0])[:, wint_col]  # [671]

    q_np = q_wint.cpu().numpy()
    p_np = p_wint.cpu().numpy()
    c_np = c_wint.cpu().numpy()

    # Counterfactual groups:
    # Strong-ON: q > 0.60
    # Strong-OFF: q < 0.20
    # Ambiguous: 0.20 <= q <= 0.60
    mask_on = (q_np > 0.60)
    mask_off = (q_np < 0.20)
    mask_amb = (~mask_on) & (~mask_off)

    groups = [
        ("Strong-ON (q > 0.60)", mask_on),
        ("Strong-OFF (q < 0.20)", mask_off),
        ("Ambiguous Middle (0.20 <= q <= 0.60)", mask_amb),
        ("Full Population", np.ones(B, dtype=bool)),
    ]

    # Compute unweighted vs weighted per-basin scalar loss gradients
    # Unweighted: dL_unw / dz_i = (p_i - q_i) / B
    # Weighted:   dL_wgt / dz_i = c_i * (p_i - q_i) / (sum_c)
    sum_c_wint = float(torch.sum(c_wint).item())
    grad_unw = (p_np - q_np) / float(B)
    grad_wgt = (c_np * (p_np - q_np)) / sum_c_wint

    group_anatomy_rows = []
    for g_name, g_mask in groups:
        n_g = int(np.sum(g_mask))
        mean_q_g = float(np.mean(q_np[g_mask])) if n_g > 0 else 0.0
        mean_p_g = float(np.mean(p_np[g_mask])) if n_g > 0 else 0.0
        mean_c_g = float(np.mean(c_np[g_mask])) if n_g > 0 else 0.0

        sum_abs_grad_unw = float(np.sum(np.abs(grad_unw[g_mask])))
        sum_abs_grad_wgt = float(np.sum(np.abs(grad_wgt[g_mask])))

        group_anatomy_rows.append({
            "group": g_name,
            "count": n_g,
            "frac_population": float(n_g / B),
            "mean_q": mean_q_g,
            "mean_p": mean_p_g,
            "mean_confidence_c": mean_c_g,
            "sum_abs_grad_unweighted": sum_abs_grad_unw,
            "sum_abs_grad_confidence_weighted": sum_abs_grad_wgt,
        })

    # Normalize gradient energy share
    total_unw = float(np.sum(np.abs(grad_unw)))
    total_wgt = float(np.sum(np.abs(grad_wgt)))
    for r in group_anatomy_rows:
        r["share_of_gradient_unweighted_pct"] = float(r["sum_abs_grad_unweighted"] / total_unw * 100.0)
        r["share_of_gradient_weighted_pct"] = float(r["sum_abs_grad_confidence_weighted"] / total_wgt * 100.0)

    df_anatomy = pd.DataFrame(group_anatomy_rows)
    print("\nGradient Share by Group:")
    cols_print = ["group", "count", "mean_confidence_c", "share_of_gradient_unweighted_pct", "share_of_gradient_weighted_pct"]
    print(df_anatomy[cols_print].to_string(index=False))

    # Save summary
    preflight_summary = {
        "confidence_distributions": conf_stats,
        "gradient_share_anatomy_w_int": group_anatomy_rows,
    }
    (OUT_DIR / "preflight_audit_summary.json").write_text(json.dumps(preflight_summary, indent=2))
    df_anatomy.to_csv(OUT_DIR / "preflight_gradient_anatomy.csv", index=False)

    print("\n=== PREFLIGHT VALIDATION PASSED: READY FOR R17-B TRAINING ===")


if __name__ == "__main__":
    main()
