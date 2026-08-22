#!/usr/bin/env python3
"""Part A: Continuous Gate-Objective Sweep across 9 lambdas and 4 processes on 531 CAMELS basins.

Evaluates:
  - Sweep over w in {0.0, 0.10, 0.25, 0.50, 0.75, 1.00} holding other 3 gates fixed
  - Soft-gate objective J(p)
  - Hardened-gate objective J(1[p > 0.5])
  - Soft regret R_soft = J(p) - min_w J(w)
  - Hard regret R_hard = J(1[p > 0.5]) - min_w J(w)
  - Summary metrics: fraction w*=0, w*=1, interior w*, mean/median R_soft, R_hard, frac(R_soft < R_hard), rho(p, w*)
"""
from __future__ import annotations

import os, json, torch, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr

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
W_GRID = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]

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
    print(f"PART A: CONTINUOUS GATE-OBJECTIVE SWEEP AUDIT ({dev})")
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

    def eval_fit_for_weights(params_dict, routing_dict, w_tensor):
        # w_tensor: [B, 4]
        with torch.no_grad():
            Q = phy._run_weighted_loop(P, T_forcing, PET, doy, params_dict, w_tensor, n_steps, B)
            Qr = phy._apply_routing(Q.mean(-1), routing_dict)[:, :, 0]  # [n_out, B]
            sq_err = (Qr - obs_valid_window) ** 2 / (std_b_dev ** 2)
            sq_err = torch.where(mask_valid, sq_err, torch.zeros_like(sq_err))
            fit_b = sq_err.sum(dim=0) / n_v_b  # [B]
        return fit_b.cpu().numpy()

    sweep_summary_records = []
    basin_level_records = []

    for tag, lmb, rel_path in LAMBDA_CONFIGS:
        print(f"\nProcessing {tag} (lambda={lmb})...")
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

        for proc in PROCESSES:
            p_col = GATE_IDX[proc]
            cost_p = COSTS[proc]
            aic_unit = lmb * cost_p * (N_tot / (B * n_valid_b.cpu().numpy()))  # [B]

            # 1. Grid evaluations J(w)
            J_grid = np.zeros((B, len(W_GRID)))
            L_fit_grid = np.zeros((B, len(W_GRID)))

            for g_idx, w_val in enumerate(W_GRID):
                w_test = learned_probs.clone()
                w_test[:, p_col] = float(w_val)
                fit_b = eval_fit_for_weights(mopex_params, routing, w_test)
                L_fit_grid[:, g_idx] = fit_b
                J_grid[:, g_idx] = fit_b + aic_unit * w_val

            # 2. Soft-gate evaluation J(p)
            w_soft = learned_probs.clone()  # target is already learned_probs[:, p_col]
            fit_soft = eval_fit_for_weights(mopex_params, routing, w_soft)
            J_soft = fit_soft + aic_unit * p_learned_np[:, p_col]

            # 3. Hardened-gate evaluation J(1[p > 0.5])
            p_hard_np = (p_learned_np[:, p_col] > 0.5).astype(float)
            w_hard = learned_probs.clone()
            w_hard[:, p_col] = torch.from_numpy(p_hard_np).to(dev).float()
            fit_hard = eval_fit_for_weights(mopex_params, routing, w_hard)
            J_hard = fit_hard + aic_unit * p_hard_np

            # 4. Computations for every basin
            # Best grid value w*
            best_grid_idx = np.argmin(J_grid, axis=1)
            w_star = np.array([W_GRID[idx] for idx in best_grid_idx])
            min_J_grid = np.min(J_grid, axis=1)

            # Absolute minimum across all options
            all_J_stack = np.column_stack([J_grid, J_soft, J_hard])
            min_J_all = np.min(all_J_stack, axis=1)

            # Regrets
            R_soft = J_soft - min_J_grid
            R_hard = J_hard - min_J_grid

            # Endpoint preference
            delta_J_endpoint = J_grid[:, 0] - J_grid[:, -1]  # J(0) - J(1)

            # Basin classifications
            frac_w0 = float(np.mean(w_star == 0.0) * 100)
            frac_w1 = float(np.mean(w_star == 1.0) * 100)
            frac_interior = float(np.mean((w_star > 0.0) & (w_star < 1.0)) * 100)

            frac_soft_better = float(np.mean(R_soft < R_hard - 1e-6) * 100)
            frac_hard_better = float(np.mean(R_hard < R_soft - 1e-6) * 100)
            frac_equal = float(np.mean(np.abs(R_soft - R_hard) <= 1e-6) * 100)

            rho_p_wstar, _ = spearmanr(p_learned_np[:, p_col], w_star)
            r_p_wstar, _ = pearsonr(p_learned_np[:, p_col], w_star)
            mae_p_wstar = float(np.mean(np.abs(p_learned_np[:, p_col] - w_star)))

            rec_sum = {
                "lambda_tag": tag,
                "lambda": lmb,
                "process": proc,
                "p_mean": float(np.mean(p_learned_np[:, p_col])),
                "p_median": float(np.median(p_learned_np[:, p_col])),
                "p_std": float(np.std(p_learned_np[:, p_col])),
                "p_act_gt05": float(np.mean(p_learned_np[:, p_col] > 0.5) * 100),
                "frac_opt_w0": frac_w0,
                "frac_opt_w1": frac_w1,
                "frac_opt_interior": frac_interior,
                "mean_R_soft": float(np.mean(R_soft)),
                "median_R_soft": float(np.median(R_soft)),
                "mean_R_hard": float(np.mean(R_hard)),
                "median_R_hard": float(np.median(R_hard)),
                "diff_R_hard_minus_soft": float(np.mean(R_hard - R_soft)),
                "frac_R_soft_lt_R_hard": frac_soft_better,
                "frac_R_hard_lt_R_soft": frac_hard_better,
                "frac_R_equal": frac_equal,
                "spearman_p_wstar": float(rho_p_wstar),
                "pearson_p_wstar": float(r_p_wstar),
                "mae_p_wstar": mae_p_wstar,
                "mean_delta_J_endpoint": float(np.mean(delta_J_endpoint)),
                "frac_delta_J_gt0": float(np.mean(delta_J_endpoint > 0) * 100),
            }
            sweep_summary_records.append(rec_sum)

            # Store basin-level samples for export
            for b in range(B):
                basin_level_records.append({
                    "lambda": lmb,
                    "process": proc,
                    "basin_idx": b,
                    "p_learned": float(p_learned_np[b, p_col]),
                    "w_star": float(w_star[b]),
                    "J_0": float(J_grid[b, 0]),
                    "J_10": float(J_grid[b, 1]),
                    "J_25": float(J_grid[b, 2]),
                    "J_50": float(J_grid[b, 3]),
                    "J_75": float(J_grid[b, 4]),
                    "J_100": float(J_grid[b, 5]),
                    "J_soft": float(J_soft[b]),
                    "J_hard": float(J_hard[b]),
                    "R_soft": float(R_soft[b]),
                    "R_hard": float(R_hard[b]),
                    "Delta_J_endpoint": float(delta_J_endpoint[b]),
                })

    df_summary = pd.DataFrame(sweep_summary_records)
    csv_summary_path = out_dir / "continuous_gate_objective_sweep_summary.csv"
    df_summary.to_csv(csv_summary_path, index=False)
    print(f"\nSaved summary table -> {csv_summary_path}")

    df_basin = pd.DataFrame(basin_level_records)
    csv_basin_path = out_dir / "continuous_gate_objective_sweep_basin_level.csv.gz"
    df_basin.to_csv(csv_basin_path, index=False, compression="gzip")
    print(f"Saved basin-level table -> {csv_basin_path}")

    # Print formatted summary table
    print("\n" + "=" * 110)
    print("CONTINUOUS GATE-OBJECTIVE SWEEP SUMMARY ACROSS 9 LAMBDAS (N=531)")
    print("=" * 110)
    cols_print = [
        "lambda", "process", "p_mean", "frac_opt_w0", "frac_opt_w1", "frac_opt_interior",
        "mean_R_soft", "mean_R_hard", "frac_R_soft_lt_R_hard", "spearman_p_wstar"
    ]
    print(df_summary[cols_print].to_string(index=False))


if __name__ == "__main__":
    main()
