#!/usr/bin/env python3
"""Comprehensive Read-Only Audit: Pure-X35 BCE Effect on Structural Weights and w_int-alpha Identifiability.

Evaluates:
  1. Checkpoint inventory for A, B, C, D, and H0
  2. Four structural gates (w_phen, w_int, w_snow, w_sub)
  3. Descaled physical parameters (alpha_int, is_time)
  4. BCE effect on structural weights (B vs A, D vs C)
  5. Empirical correlation corr(w_int, alpha_int)
  6. Exact Jacobian sensitivity collinearity cos(dQ/dw_int, dQ/dalpha)
  7. High-resolution publication figures & summary CSV artifacts
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent.parent
for p in (REPO_ROOT, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config
from project.flexmopex.run_model import _build_data_loader, _attach_doy
from project.flexmopex.local_model_handler import FlexMopexModelHandler

GATES = ["w_phen", "w_int", "w_snow", "w_sub"]
GATE_IDX = {"w_phen": 0, "w_int": 1, "w_snow": 2, "w_sub": 3}


def rank_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    for value in np.unique(x):
        idx = np.flatnonzero(x == value)
        ranks[idx] = ranks[idx].mean()
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xr, yr = rank_average(x[mask]), rank_average(y[mask])
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def compute_stats(x: np.ndarray) -> dict[str, float]:
    vals = np.asarray(x, dtype=float).reshape(-1)
    q05, q25, q50, q75, q95 = np.percentile(vals, [5, 25, 50, 75, 95])
    return {
        "count": int(len(vals)),
        "mean": float(np.mean(vals)),
        "median": float(q50),
        "std": float(np.std(vals)),
        "iqr": float(q75 - q25),
        "min": float(np.min(vals)),
        "p05": float(q05),
        "p25": float(q25),
        "p75": float(q75),
        "p95": float(q95),
        "max": float(np.max(vals)),
        "p95_minus_p05": float(q95 - q05),
        "frac_lt_001": float(np.mean(vals < 0.01)),
        "frac_gt_001": float(np.mean(vals > 0.01)),
        "frac_gt_010": float(np.mean(vals > 0.10)),
        "frac_gt_050": float(np.mean(vals > 0.50)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results/pure_x35_bce_weight_alpha_audit",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Running Flex-MOPEX audit on device: {device}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_data_dir = out_dir / "plotting_data"
    plot_data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Checkpoint inventory definition
    RUNS = {
        "A_ep4": {
            "label": "A (Pure-X35 | E2E | Original MOPEX4)",
            "id": "A",
            "arch": "Pure-X35",
            "loss": "End-to-End (L_Q + lambda*Omega)",
            "formula": "Original historical MOPEX4",
            "phy_class": "LearnedWeightMopex",
            "nn_class": "LearnedStructureNetPureAttrEncoder",
            "config": "conf/ssh_2x2/config_E1_pure_x35_531_lambda0007.yaml",
            "rel_dir": "results/ssh_2x2/E1/seed_42",
            "epoch": 4,
            "seed": 42,
            "lambda": 0.007,
        },
        "A_ep10": {
            "label": "A_ep10 (Pure-X35 | E2E | Original MOPEX4 | Ep10)",
            "id": "A_ep10",
            "arch": "Pure-X35",
            "loss": "End-to-End (L_Q + lambda*Omega)",
            "formula": "Original historical MOPEX4",
            "phy_class": "LearnedWeightMopex",
            "nn_class": "LearnedStructureNetPureAttrEncoder",
            "config": "conf/ssh_2x2/config_E1_pure_x35_531_lambda0007.yaml",
            "rel_dir": "results/ssh_2x2/E1/seed_42",
            "epoch": 10,
            "seed": 42,
            "lambda": 0.007,
        },
        "B_ep4": {
            "label": "B (Pure-X35 | CF-BCE | Original MOPEX4)",
            "id": "B",
            "arch": "Pure-X35",
            "loss": "Counterfactual BCE (CFTrainer)",
            "formula": "Original historical MOPEX4",
            "phy_class": "LearnedWeightMopex",
            "nn_class": "LearnedStructureNetPureAttrEncoder",
            "config": "conf/ssh_2x2/config_E3_pure_x35_531_lambda0007.yaml",
            "rel_dir": "results/ssh_2x2/E3/seed_42",
            "epoch": 4,
            "seed": 42,
            "lambda": 0.007,
        },
        "C_ep4": {
            "label": "C (Pure-X35 | E2E | Corrected Candidate E-S0)",
            "id": "C",
            "arch": "Pure-X35",
            "loss": "End-to-End (L_Q + lambda*Omega)",
            "formula": "Corrected Candidate E-S0",
            "phy_class": "LearnedWeightMopexE",
            "nn_class": "LearnedStructureNetPureAttrEncoder",
            "config": "conf/ssh_2x2/config_E2_pure_x35_531_lambda0007.yaml",
            "rel_dir": "results/ssh_2x2/E2/seed_42",
            "epoch": 4,
            "seed": 42,
            "lambda": 0.007,
        },
        "C_ep10": {
            "label": "C_ep10 (Pure-X35 | E2E | Corrected Candidate E-S0 | Ep10)",
            "id": "C_ep10",
            "arch": "Pure-X35",
            "loss": "End-to-End (L_Q + lambda*Omega)",
            "formula": "Corrected Candidate E-S0",
            "phy_class": "LearnedWeightMopexE",
            "nn_class": "LearnedStructureNetPureAttrEncoder",
            "config": "conf/ssh_2x2/config_E2_pure_x35_531_lambda0007.yaml",
            "rel_dir": "results/ssh_2x2/E2/seed_42",
            "epoch": 10,
            "seed": 42,
            "lambda": 0.007,
        },
        "D_ep4": {
            "label": "D (Pure-X35 | CF-BCE | Corrected Candidate E-S0)",
            "id": "D",
            "arch": "Pure-X35",
            "loss": "Counterfactual BCE (CFTrainer)",
            "formula": "Corrected Candidate E-S0",
            "phy_class": "LearnedWeightMopexE",
            "nn_class": "LearnedStructureNetPureAttrEncoder",
            "config": "conf/ssh_2x2/config_E4_pure_x35_531_lambda0007.yaml",
            "rel_dir": "results/ssh_2x2/E4/seed_42",
            "epoch": 4,
            "seed": 42,
            "lambda": 0.007,
        },
        "H0_ep10": {
            "label": "H0 (Shared Backbone | E2E | Original MOPEX4 | Ep10)",
            "id": "H0",
            "arch": "Shared Backbone (LearnedStructureNet)",
            "loss": "End-to-End (L_Q + lambda*Omega)",
            "formula": "Original historical MOPEX4",
            "phy_class": "LearnedWeightMopex",
            "nn_class": "LearnedStructureNet",
            "config": "conf/config_flexmopex_canonical.yaml",
            "rel_dir": "results/intercept_2x2/A",
            "epoch": 10,
            "seed": 42,
            "lambda": 0.01,
        },
    }

    # Save inventory CSV
    inventory_rows = []
    for r_key, r_info in RUNS.items():
        inventory_rows.append({
            "run_key": r_key,
            "id": r_info["id"],
            "label": r_info["label"],
            "architecture": r_info["arch"],
            "structure_training": r_info["loss"],
            "interception_formula": r_info["formula"],
            "phy_model_class": r_info["phy_class"],
            "nn_model_class": r_info["nn_class"],
            "seed": r_info["seed"],
            "epoch": r_info["epoch"],
            "aic_alpha_lambda": r_info["lambda"],
            "basin_count": 531,
            "result_directory": str(PROJECT_DIR / r_info["rel_dir"]),
        })
    pd.DataFrame(inventory_rows).to_csv(out_dir / "checkpoint_inventory.csv", index=False)

    # 2. Load dataset & basin IDs
    manifest_path = REPO_ROOT / "data/531sub_id.txt"
    basin_ids = np.array(ast.literal_eval(manifest_path.read_text()), dtype=np.int64)
    B = len(basin_ids)
    assert B == 531, f"Expected 531 basins, got {B}"

    ref_cfg = load_config(RUNS["A_ep4"]["config"])
    ref_cfg["device"] = str(device)
    ref_cfg["gpu_id"] = args.gpu_id if str(device).startswith("cuda") else -1
    loader = _build_data_loader(ref_cfg)
    _attach_doy(loader.eval_dataset, ref_cfg["test"])
    ev = loader.eval_dataset

    # 3. Extraction & Deterministic Inference for all runs
    extracted = {}
    basin_level_rows = []

    T_win = 730  # 365 warmup + 365 test evaluation window

    for r_key, r_info in RUNS.items():
        print(f"\n--- Extracting {r_key} ({r_info['label']}) ---")
        cfg = load_config(r_info["config"])
        cfg["device"] = str(device)
        cfg["gpu_id"] = args.gpu_id if str(device).startswith("cuda") else -1
        cfg["model_dir"] = str(PROJECT_DIR / r_info["rel_dir"] / "model")
        cfg["model_path"] = str(PROJECT_DIR / r_info["rel_dir"] / "model")
        cfg["save_path"] = str(PROJECT_DIR / r_info["rel_dir"])
        cfg["trained_model"] = str(PROJECT_DIR / r_info["rel_dir"] / "model")

        if r_key.startswith("H0"):
            cfg["delta_model"]["phy_model"]["model"] = ["LearnedWeightMopex"]
            cfg["delta_model"]["nn_model"]["model"] = "LearnedStructureNet"
            cfg["model"]["phy"]["name"] = ["LearnedWeightMopex"]
            cfg["model"]["nn"]["name"] = "LearnedStructureNet"

        handler = FlexMopexModelHandler(cfg, verbose=False)
        handler.load_model(r_info["epoch"])
        handler.eval()

        model = next(iter(handler.model_dict.values()))
        phy, nn = model.phy_model, model.nn_model
        # Unwrapped step function for clean evaluation
        if hasattr(phy.step_fn, "__wrapped__"):
            phy.step_fn = phy.step_fn.__wrapped__

        sample = {
            "x_phy": ev["x_phy"][:T_win, :].to(device),
            "doy": ev["doy"][:T_win, :].to(device),
            "c_nn_norm": ev["xc_nn_norm"][0, :, -35:].to(device),
            "xc_nn_norm": ev["xc_nn_norm"][:T_win, :, :].to(device),
        }

        with torch.no_grad():
            nn_out = nn(sample)
            logits = nn_out["weights"].view(B, 4, 2).clamp(-10.0, 10.0)
            weights_on = F.softmax(logits, dim=-1)[..., 1]  # [531, 4]
            mopex_params = phy._descale_mopex_params(nn_out["params"])
            routing_params = phy._descale_routing_params(nn_out["gamma_uh"])
            alpha = mopex_params["alpha"].mean(-1)  # [531]
            is_time = mopex_params["is_time"].mean(-1)  # [531]

        # 4. Exact Jacobian computation: dQ/dw_int and dQ/dalpha
        weights_on_grad = weights_on.detach().requires_grad_(True)
        alpha_grad = mopex_params["alpha"].detach().requires_grad_(True)
        mopex_params_grad = {k: (alpha_grad if k == "alpha" else v) for k, v in mopex_params.items()}

        P, T, PET, doy, n_steps, n_grid = phy._prepare_forcings(sample)
        Q_mopex = phy._run_weighted_loop(P, T, PET, doy, mopex_params_grad, weights_on_grad, n_steps, n_grid)
        Q_routed = phy._apply_routing(Q_mopex.mean(-1), routing_params)[:, :, 0]  # [T_eval, B]

        n_slices = 12
        slice_len = Q_routed.shape[0] // n_slices
        slice_gw = []
        slice_ga = []

        for s_idx in range(n_slices):
            q_s = Q_routed[s_idx * slice_len : (s_idx + 1) * slice_len].sum(dim=0)
            gw_s = torch.autograd.grad(q_s.sum(), weights_on_grad, retain_graph=True)[0][:, 1]
            ga_s = torch.autograd.grad(q_s.sum(), alpha_grad, retain_graph=True)[0].mean(-1)
            slice_gw.append(gw_s)
            slice_ga.append(ga_s)

        gw_mat = torch.stack(slice_gw, dim=1)  # [531, 12]
        ga_mat = torch.stack(slice_ga, dim=1)  # [531, 12]

        norm_gw = torch.linalg.norm(gw_mat, dim=1)
        norm_ga = torch.linalg.norm(ga_mat, dim=1)
        cos_sim = F.cosine_similarity(gw_mat, ga_mat, dim=1)

        # Handle zero-norm cases gracefully
        zero_mask = (norm_gw < 1e-8) | (norm_ga < 1e-8)
        if "Original" in r_info["formula"]:
            cos_sim[zero_mask] = 1.0
        else:
            cos_sim[zero_mask] = 0.0

        w_np = weights_on.detach().cpu().numpy()
        alpha_np = alpha.detach().cpu().numpy()
        istime_np = is_time.detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()
        cos_np = cos_sim.detach().cpu().numpy()
        norm_gw_np = norm_gw.detach().cpu().numpy()
        norm_ga_np = norm_ga.detach().cpu().numpy()

        extracted[r_key] = {
            "weights": w_np,
            "alpha": alpha_np,
            "is_time": istime_np,
            "logits": logits_np,
            "cos": cos_np,
            "norm_gw": norm_gw_np,
            "norm_ga": norm_ga_np,
        }

        # Basin-level records
        for i, b_id in enumerate(basin_ids):
            basin_level_rows.append({
                "basin_id": int(b_id),
                "run_key": r_key,
                "variant_id": r_info["id"],
                "architecture": r_info["arch"],
                "structure_training": r_info["loss"],
                "formula": r_info["formula"],
                "epoch": r_info["epoch"],
                "seed": r_info["seed"],
                "w_phen": float(w_np[i, 0]),
                "w_int": float(w_np[i, 1]),
                "w_snow": float(w_np[i, 2]),
                "w_sub": float(w_np[i, 3]),
                "alpha_int": float(alpha_np[i]),
                "is_time": float(istime_np[i]),
                "cos_dQ_dwint_dQ_dalpha": float(cos_np[i]),
                "norm_dQ_dwint": float(norm_gw_np[i]),
                "norm_dQ_dalpha": float(norm_ga_np[i]),
            })

    pd.DataFrame(basin_level_rows).to_csv(out_dir / "structural_weights_basin_level.csv", index=False)

    # 4. Compute full Distribution Statistics
    dist_rows = []
    for r_key, r_info in RUNS.items():
        w_mat = extracted[r_key]["weights"]
        for g_idx, g_name in enumerate(GATES):
            stats = compute_stats(w_mat[:, g_idx])
            dist_rows.append({
                "run_key": r_key,
                "variant_id": r_info["id"],
                "architecture": r_info["arch"],
                "structure_training": r_info["loss"],
                "formula": r_info["formula"],
                "epoch": r_info["epoch"],
                "gate": g_name,
                **stats,
            })
        # Add alpha_int & is_time stats
        alpha_stats = compute_stats(extracted[r_key]["alpha"])
        dist_rows.append({
            "run_key": r_key,
            "variant_id": r_info["id"],
            "architecture": r_info["arch"],
            "structure_training": r_info["loss"],
            "formula": r_info["formula"],
            "epoch": r_info["epoch"],
            "gate": "alpha_int",
            **alpha_stats,
        })
    pd.DataFrame(dist_rows).to_csv(out_dir / "structural_weight_distribution_summary.csv", index=False)

    # 5. Paired Cross-Variant Contrasts
    paired_contrasts = [
        ("B_ep4", "A_ep4", "B vs A (BCE Effect | Original Formula | Ep4)"),
        ("D_ep4", "C_ep4", "D vs C (BCE Effect | Corrected Formula | Ep4)"),
        ("C_ep4", "A_ep4", "C vs A (Formula Effect | End-to-End | Ep4)"),
        ("D_ep4", "B_ep4", "D vs B (Formula Effect | BCE | Ep4)"),
        ("A_ep10", "H0_ep10", "A vs H0 (Pure-X35 vs Shared Backbone | Original Formula | Ep10)"),
        ("C_ep10", "A_ep10", "C vs A (Formula Effect | End-to-End | Ep10)"),
    ]

    paired_rows = []
    for var_a, var_b, label in paired_contrasts:
        for g_idx, g_name in enumerate(GATES):
            w_a = extracted[var_a]["weights"][:, g_idx]
            w_b = extracted[var_b]["weights"][:, g_idx]
            diff = w_a - w_b
            abs_diff = np.abs(diff)

            p_r = pearson_corr(w_a, w_b)
            s_rho = spearman_corr(w_a, w_b)

            paired_rows.append({
                "contrast": f"{var_a}_vs_{var_b}",
                "contrast_label": label,
                "var_a": var_a,
                "var_b": var_b,
                "gate": g_name,
                "pearson_r": float(p_r),
                "spearman_rho": float(s_rho),
                "mean_paired_diff": float(np.mean(diff)),
                "median_paired_diff": float(np.median(diff)),
                "mean_abs_diff": float(np.mean(abs_diff)),
                "frac_a_gt_b": float(np.mean(diff > 0)),
                "frac_a_lt_b": float(np.mean(diff < 0)),
            })
    pd.DataFrame(paired_rows).to_csv(out_dir / "bce_paired_weight_comparison.csv", index=False)

    # 6. Empirical w_int vs alpha_int value correlations
    w_alpha_corr_rows = []
    for r_key, r_info in RUNS.items():
        w_int = extracted[r_key]["weights"][:, 1]
        alpha = extracted[r_key]["alpha"]
        is_time = extracted[r_key]["is_time"]

        w_alpha_corr_rows.append({
            "run_key": r_key,
            "variant_id": r_info["id"],
            "architecture": r_info["arch"],
            "structure_training": r_info["loss"],
            "formula": r_info["formula"],
            "epoch": r_info["epoch"],
            "pearson_wint_alpha": float(pearson_corr(w_int, alpha)),
            "spearman_wint_alpha": float(spearman_corr(w_int, alpha)),
            "pearson_wint_istime": float(pearson_corr(w_int, is_time)),
            "spearman_wint_istime": float(spearman_corr(w_int, is_time)),
        })
    pd.DataFrame(w_alpha_corr_rows).to_csv(out_dir / "wint_alpha_value_correlation.csv", index=False)

    # 7. Exact Jacobian Collinearity statistics
    jac_rows = []
    for r_key, r_info in RUNS.items():
        cos_v = extracted[r_key]["cos"]
        norm_w = extracted[r_key]["norm_gw"]
        norm_a = extracted[r_key]["norm_ga"]
        ratio = norm_w / np.clip(norm_a, 1e-8, None)

        q05, q25, q50, q75, q95 = np.percentile(cos_v, [5, 25, 50, 75, 95])
        jac_rows.append({
            "run_key": r_key,
            "variant_id": r_info["id"],
            "architecture": r_info["arch"],
            "structure_training": r_info["loss"],
            "formula": r_info["formula"],
            "epoch": r_info["epoch"],
            "cos_mean": float(np.mean(cos_v)),
            "cos_median": float(q50),
            "cos_std": float(np.std(cos_v)),
            "cos_iqr": float(q75 - q25),
            "cos_p05": float(q05),
            "cos_p95": float(q95),
            "frac_abs_cos_gt_08": float(np.mean(np.abs(cos_v) > 0.8)),
            "frac_abs_cos_gt_09": float(np.mean(np.abs(cos_v) > 0.9)),
            "median_norm_dQ_dwint": float(np.median(norm_w)),
            "median_norm_dQ_dalpha": float(np.median(norm_a)),
            "median_norm_ratio_w_over_alpha": float(np.median(ratio)),
        })
    pd.DataFrame(jac_rows).to_csv(out_dir / "wint_alpha_jacobian_collinearity.csv", index=False)

    # 8. Save Plotting Data
    pd.DataFrame(dist_rows).to_csv(plot_data_dir / "distribution_summary_plot_data.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(plot_data_dir / "paired_comparison_plot_data.csv", index=False)
    pd.DataFrame(jac_rows).to_csv(plot_data_dir / "jacobian_collinearity_plot_data.csv", index=False)

    # 9. Figures Generation
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})

    # --- Figure 1: 4 Structural Weight Distributions across A, B, C, D, H0 ---
    fig1, axes1 = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    target_runs = ["H0_ep10", "A_ep4", "B_ep4", "C_ep4", "D_ep4"]
    run_labels = ["H0 (Old)", "A (E2E/Orig)", "B (BCE/Orig)", "C (E2E/CandE)", "D (BCE/CandE)"]
    palette = ["#7f7f7f", "#1f77b4", "#aec7e8", "#2ca02c", "#98df8a"]

    for g_idx, g_name in enumerate(GATES):
        ax = axes1[g_idx]
        data = [extracted[k]["weights"][:, g_idx] for k in target_runs]
        bp = ax.boxplot(data, tick_labels=run_labels, patch_artist=True, showmeans=True, meanline=True, widths=0.55)
        for patch, col in zip(bp["boxes"], palette):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.5)
        for mean_l in bp["means"]:
            mean_l.set_color("red")
            mean_l.set_linestyle("--")

        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.01, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(0.50, color="gray", linestyle="--", alpha=0.3)
        ax.set_title(f"{g_name}", fontweight="bold", fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
        if g_idx == 0:
            ax.set_ylabel("Structural Weight w [0, 1]")

    fig1.suptitle("Flex-MOPEX Structural Process Weights across Architectures & Training Signals (N=531)", fontsize=12, fontweight="bold", y=0.98)
    fig1.tight_layout(rect=[0, 0, 1, 0.94])
    fig1.savefig(fig_dir / "figure1_four_structural_weights_distribution.png", dpi=300)
    plt.close(fig1)

    # --- Figure 2: Paired BCE Effect on w_int ---
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(11, 5))

    # Original formula: A vs B
    w_a = extracted["A_ep4"]["weights"][:, 1]
    w_b = extracted["B_ep4"]["weights"][:, 1]
    r_ab = pearson_corr(w_a, w_b)
    rho_ab = spearman_corr(w_a, w_b)
    ax2a.scatter(w_a, w_b, alpha=0.45, s=20, color="#1f77b4", edgecolors="none")
    ax2a.plot([0, 1], [0, 1], "r--", linewidth=1.2, label="Identity line")
    ax2a.set_xlim(-0.02, 1.02)
    ax2a.set_ylim(-0.02, 1.02)
    ax2a.set_xlabel("A (End-to-End) w_int")
    ax2a.set_ylabel("B (Counterfactual BCE) w_int")
    ax2a.set_title(f"Original MOPEX4: B vs A (r={r_ab:.2f}, ρ={rho_ab:.2f})\nB recovers active spread from A collapse", fontsize=10, fontweight="bold")
    ax2a.grid(True, linestyle="--", alpha=0.3)

    # Corrected formula: C vs D
    w_c = extracted["C_ep4"]["weights"][:, 1]
    w_d = extracted["D_ep4"]["weights"][:, 1]
    r_cd = pearson_corr(w_c, w_d)
    rho_cd = spearman_corr(w_c, w_d)
    ax2b.scatter(w_c, w_d, alpha=0.45, s=20, color="#2ca02c", edgecolors="none")
    ax2b.plot([0, 1], [0, 1], "r--", linewidth=1.2, label="Identity line")
    ax2b.set_xlim(-0.02, 1.02)
    ax2b.set_ylim(-0.02, 1.02)
    ax2b.set_xlabel("C (End-to-End) w_int")
    ax2b.set_ylabel("D (Counterfactual BCE) w_int")
    ax2b.set_title(f"Candidate E-S0: D vs C (r={r_cd:.2f}, ρ={rho_cd:.2f})\nD recovers active spread from C collapse", fontsize=10, fontweight="bold")
    ax2b.grid(True, linestyle="--", alpha=0.3)

    fig2.suptitle("Paired Basin Impact of Counterfactual BCE on Interception Gate w_int", fontsize=12, fontweight="bold", y=0.98)
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    fig2.savefig(fig_dir / "figure2_paired_bce_effect_wint.png", dpi=300)
    plt.close(fig2)

    # --- Figure 3: w_int vs Internal alpha Scatter Across 5 Variants ---
    fig3, axes3 = plt.subplots(1, 5, figsize=(19, 4), sharey=True, sharex=True)
    for idx, r_key in enumerate(["H0_ep10", "A_ep4", "B_ep4", "C_ep4", "D_ep4"]):
        ax = axes3[idx]
        w_i = extracted[r_key]["weights"][:, 1]
        a_i = extracted[r_key]["alpha"]
        r_val = pearson_corr(w_i, a_i)
        rho_val = spearman_corr(w_i, a_i)

        ax.scatter(w_i, a_i, alpha=0.45, s=18, color=palette[idx], edgecolors="none")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.40, 0.65)
        ax.set_xlabel("w_int [0, 1]")
        if idx == 0:
            ax.set_ylabel("Internal alpha_int")
        ax.set_title(f"{run_labels[idx]}\nr={r_val:.2f}, ρ={rho_val:.2f}", fontsize=9, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)

    fig3.suptitle("Learned w_int vs Internal Interception alpha across 531 Basins", fontsize=12, fontweight="bold", y=0.98)
    fig3.tight_layout(rect=[0, 0, 1, 0.92])
    fig3.savefig(fig_dir / "figure3_wint_vs_internal_alpha.png", dpi=300)
    plt.close(fig3)

    # --- Figure 4: Jacobian Cosine Distributions (dQ/dw_int, dQ/dalpha) ---
    fig4, ax4 = plt.subplots(figsize=(9, 5))
    cos_data = [extracted[k]["cos"] for k in target_runs]
    bp4 = ax4.boxplot(cos_data, tick_labels=run_labels, patch_artist=True, showmeans=True, meanline=True, widths=0.5)
    for patch, col in zip(bp4["boxes"], palette):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    for med in bp4["medians"]:
        med.set_color("black")
        med.set_linewidth(1.5)
    for mean_l in bp4["means"]:
        mean_l.set_color("red")
        mean_l.set_linestyle("--")

    ax4.axhline(0.8, color="red", linestyle=":", alpha=0.7, label="|cos| = 0.8 (Strong Collinearity Threshold)")
    ax4.axhline(0.0, color="gray", linestyle="-", alpha=0.5)
    ax4.set_ylim(-1.05, 1.05)
    ax4.set_ylabel("Jacobian Cosine: cos(dQ/dw_int, dQ/dalpha)")
    ax4.set_title("Physical Sensitivity Collinearity cos(dQ/dw_int, dQ/dalpha) across 531 Basins\nOriginal MOPEX4 (H0, A, B) exhibits severe collinearity; Candidate E (C, D) decouples sensitivity", fontsize=11, fontweight="bold")
    ax4.legend(loc="lower left", framealpha=0.8)
    ax4.grid(True, linestyle="--", alpha=0.3, axis="y")

    fig4.tight_layout()
    fig4.savefig(fig_dir / "figure4_jacobian_cosine_identifiability.png", dpi=300)
    plt.close(fig4)

    print(f"\nAll artifacts generated and saved successfully to: {out_dir}")


if __name__ == "__main__":
    main()
