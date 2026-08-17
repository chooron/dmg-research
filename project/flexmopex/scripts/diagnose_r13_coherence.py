#!/usr/bin/env python3
"""Phase 2: Gradient Coherence, Bias, and Common-Mode (DC) Decomposition (Testing Hypothesis C).

Analyzes the shared structure head gradient at R8 epoch 2 (primary) and epoch 3 checkpoints across all 671 basins:
  2.1 Subgroup Gradient Coherence Ratios:
        C_G = ||sum_{i in G} g_param,i|| / sum_{i in G} ||g_param,i||
      for Oracle-positive vs Oracle-zero subgroups.
  2.2 Weight vs Bias Decomposition:
        Full head = [dL/dW, dL/db]
  2.3 Common-Feature (DC) vs Centered-Feature Decomposition:
        h_i = h_bar + (h_i - h_bar)
        dL/dW = (sum g_i) h_bar + sum g_i (h_i - h_bar) = G_W_DC + G_W_centered
  2.4 Counterfactual Gradient Replays (vector replay, no training):
        - Canonical full head
        - Weight-only (No-bias)
        - Centered-feature only (No-bias, No-DC)
  2.5 Controls: Coherence metrics for w_phen, w_snow, w_sub.

Outputs: results/root_cause_r13/gradient_coherence_decomposition.json
         results/root_cause_r13/gradient_coherence_table.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config  # noqa: E402
from project.flexmopex.run_model import (  # noqa: E402
    apply_runtime_overrides, parse_args, _build_data_loader,
)
from scripts.diagnose_wint_collapse import build_handler, build_forward, run_loop  # noqa: E402

OUT_DIR = Path("results/root_cause_r13")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSES = ["w_phen", "w_int", "w_snow", "w_sub"]
GATE_IDX = {"w_phen": 0, "w_int": 1, "w_snow": 2, "w_sub": 3}
COSTS = {"w_phen": 2.0, "w_int": 2.0, "w_snow": 2.0, "w_sub": 1.0}
AIC_ALPHA = 0.01
EPS = 1e-12


def per_basin_fit(q: torch.Tensor, obs: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    o = torch.nan_to_num(obs, nan=0.0)
    sq = (q - o) ** 2 / (std.view(1, -1, 1) ** 2)
    mask = ~torch.isnan(obs)
    n_valid = mask.sum(dim=0).clamp(min=1)
    sq = torch.where(mask, sq, torch.zeros_like(sq))
    return sq.sum(dim=0) / n_valid


def analyze_checkpoint(epoch: int, cfg: dict, dl, dev: str, oracle_dict: dict) -> dict:
    td, ed = dl.train_dataset, dl.eval_dataset
    B = td["x_phy"].shape[1]
    n_attr = td["xc_nn_norm"].shape[-1] - 3
    attrs = td["xc_nn_norm"][0, :, -n_attr:].to(dev)
    std_train = (np.nanstd(td["target"][:, :, 0].cpu().numpy(), axis=0) + 0.1).astype(np.float32)
    n_out = int(ed["x_phy"].shape[0]) - 365
    y_ev = ed["target"][365:365 + n_out, :, 0].cpu().numpy()
    n_valid_b = np.sum(~np.isnan(y_ev), axis=0).astype(float)
    N = float(n_valid_b.sum())

    handler = build_handler(cfg)
    handler.load_model(epoch)
    for m in handler.model_dict.values():
        m.eval()
    model = next(iter(handler.model_dict.values()))
    phy, nn = model.phy_model, model.nn_model

    with torch.no_grad():
        h = nn.backbone(attrs).detach().cpu()  # [671, 128]
    h_bar = torch.mean(h, dim=0, keepdim=True)  # [1, 128]
    h_centered = h - h_bar                      # [671, 128]

    x_phy_full = ed["x_phy"].to(dev)
    doy_full = ed["doy"].to(dev)
    std_t = torch.from_numpy(std_train).to(dev)
    nv = torch.from_numpy(n_valid_b).to(dev)

    # Compute fit gradients on weights_on
    g_w_all = {p_: [] for p_ in PROCESSES}
    jac_all = {p_: [] for p_ in PROCESSES}

    chunk_size = 128
    for c0 in range(0, B, chunk_size):
        c1 = min(c0 + chunk_size, B)
        sample = {"x_phy": x_phy_full[:, c0:c1], "doy": doy_full[:, c0:c1], "c_nn_norm": attrs[c0:c1]}
        params, logits, weights_on, mopex_params, routing = build_forward(phy, nn, sample)
        out = run_loop(phy, sample, weights_on, mopex_params, routing)
        q = out["streamflow"]
        obs = ed["target"][365:365 + n_out, c0:c1].to(dev)
        L_b = per_basin_fit(q, obs, std_t[c0:c1])
        L_fit_obj = (L_b * nv[c0:c1] / N).sum()
        g_w = torch.autograd.grad(L_fit_obj, weights_on, retain_graph=True)[0]
        jac = weights_on * (1 - weights_on)

        for p_ in PROCESSES:
            col = GATE_IDX[p_]
            g_w_all[p_].append(g_w[:, col].detach().cpu())
            jac_all[p_].append(jac[:, col].detach().cpu())

        del q, out, params, logits, weights_on, mopex_params, routing
        torch.cuda.empty_cache()

    for p_ in PROCESSES:
        g_w_all[p_] = torch.cat(g_w_all[p_])
        jac_all[p_] = torch.cat(jac_all[p_])

    epoch_res = {}
    for proc in PROCESSES:
        col = GATE_IDX[proc]
        g_w = g_w_all[proc]
        jac = jac_all[proc]
        g_z = g_w * jac  # [671] ON-logit gradient per basin

        pos_mask = np.array([oracle_dict[proc][b] > 0 for b in range(B)])
        opos = torch.from_numpy(pos_mask)
        ozero = ~opos
        n_pos = int(opos.sum().item())
        n_zero = int(ozero.sum().item())

        # -------------------------------------------------------------
        # 1. Per-basin gradient vectors in head parameter space
        # For ON logit: param vector i = [g_z_i * h_i (128-D), g_z_i (1-D bias)] -> 129-D
        # -------------------------------------------------------------
        G_W_i = g_z.unsqueeze(-1) * h          # [671, 128]
        G_b_i = g_z.unsqueeze(-1)              # [671, 1]
        G_param_i = torch.cat([G_W_i, G_b_i], dim=-1)  # [671, 129]

        G_W_DC_i = g_z.unsqueeze(-1) * h_bar   # [671, 128]
        G_W_cent_i = g_z.unsqueeze(-1) * h_centered  # [671, 128]

        # -------------------------------------------------------------
        # 2. Subgroup Coherence Ratios
        # C_G = ||sum g_i|| / sum ||g_i||
        # -------------------------------------------------------------
        def calc_coherence(G_mat, mask):
            sub = G_mat[mask]
            vec_sum = sub.sum(dim=0)
            norm_sum = torch.norm(vec_sum)
            indiv_norms = torch.norm(sub, dim=-1)
            sum_norms = indiv_norms.sum()
            coherence = float(norm_sum / (sum_norms + EPS))
            return {
                "vector_norm": float(norm_sum),
                "sum_of_norms": float(sum_norms),
                "mean_indiv_norm": float(indiv_norms.mean()),
                "median_indiv_norm": float(indiv_norms.median()),
                "coherence_ratio": coherence,
            }

        coh_param_pos = calc_coherence(G_param_i, opos)
        coh_param_zero = calc_coherence(G_param_i, ozero)
        coh_W_pos = calc_coherence(G_W_i, opos)
        coh_W_zero = calc_coherence(G_W_i, ozero)

        # -------------------------------------------------------------
        # 3. Weight vs Bias Decomposition of Head Gradient
        # -------------------------------------------------------------
        def decompose_head(mask):
            sub_W = G_W_i[mask].sum(dim=0)      # [128]
            sub_b = G_b_i[mask].sum(dim=0)      # [1]
            sub_param = torch.cat([sub_W, sub_b], dim=-1)  # [129]
            norm_param = float(torch.norm(sub_param))
            norm_W = float(torch.norm(sub_W))
            norm_b = float(torch.abs(sub_b[0]))
            bias_energy_frac = float((norm_b ** 2) / (norm_param ** 2 + EPS))

            # DC vs Centered
            sub_W_DC = G_W_DC_i[mask].sum(dim=0)      # [128]
            sub_W_cent = G_W_cent_i[mask].sum(dim=0)  # [128]
            norm_W_DC = float(torch.norm(sub_W_DC))
            norm_W_cent = float(torch.norm(sub_W_cent))
            dc_energy_frac = float((norm_W_DC ** 2) / (norm_W ** 2 + EPS))

            return {
                "norm_full_param": norm_param,
                "norm_W": norm_W,
                "norm_b": norm_b,
                "bias_sum": float(sub_b[0]),
                "bias_energy_frac": bias_energy_frac,
                "norm_W_DC": norm_W_DC,
                "norm_W_cent": norm_W_cent,
                "dc_energy_frac": dc_energy_frac,
                "vec_W": sub_W,
                "vec_b": sub_b,
                "vec_param": sub_param,
                "vec_W_DC": sub_W_DC,
                "vec_W_cent": sub_W_cent,
            }

        decomp_pos = decompose_head(opos)
        decomp_zero = decompose_head(ozero)
        decomp_full = decompose_head(slice(None))

        # -------------------------------------------------------------
        # 4. Counterfactual Gradient Replays (Vector Replay)
        # -------------------------------------------------------------
        # A. Full Parameter space [W, b] (129-D)
        v_full_param = decomp_full["vec_param"]
        cos_pos_full_param = float(F.cosine_similarity(decomp_pos["vec_param"].view(1, -1), v_full_param.view(1, -1))[0])
        cos_zero_full_param = float(F.cosine_similarity(decomp_zero["vec_param"].view(1, -1), v_full_param.view(1, -1))[0])
        cos_pos_zero_param = float(F.cosine_similarity(decomp_pos["vec_param"].view(1, -1), decomp_zero["vec_param"].view(1, -1))[0])

        # B. Weight-only [W] (128-D, No-Bias)
        v_full_W = decomp_full["vec_W"]
        cos_pos_full_W = float(F.cosine_similarity(decomp_pos["vec_W"].view(1, -1), v_full_W.view(1, -1))[0])
        cos_zero_full_W = float(F.cosine_similarity(decomp_zero["vec_W"].view(1, -1), v_full_W.view(1, -1))[0])
        cos_pos_zero_W = float(F.cosine_similarity(decomp_pos["vec_W"].view(1, -1), decomp_zero["vec_W"].view(1, -1))[0])

        # C. Centered-feature only [W_cent] (128-D, No-Bias + No-DC)
        v_full_cent = decomp_full["vec_W_cent"]
        cos_pos_full_cent = float(F.cosine_similarity(decomp_pos["vec_W_cent"].view(1, -1), v_full_cent.view(1, -1))[0]) if decomp_pos["norm_W_cent"] > EPS and float(v_full_cent.norm()) > EPS else 0.0
        cos_zero_full_cent = float(F.cosine_similarity(decomp_zero["vec_W_cent"].view(1, -1), v_full_cent.view(1, -1))[0]) if decomp_zero["norm_W_cent"] > EPS and float(v_full_cent.norm()) > EPS else 0.0
        cos_pos_zero_cent = float(F.cosine_similarity(decomp_pos["vec_W_cent"].view(1, -1), decomp_zero["vec_W_cent"].view(1, -1))[0]) if decomp_pos["norm_W_cent"] > EPS and decomp_zero["norm_W_cent"] > EPS else 0.0

        # D. DC-only component
        v_full_DC = decomp_full["vec_W_DC"]
        cos_pos_zero_DC = float(F.cosine_similarity(decomp_pos["vec_W_DC"].view(1, -1), decomp_zero["vec_W_DC"].view(1, -1))[0])

        def clean_dict(d):
            return {k: v for k, v in d.items() if not k.startswith("vec_")}

        epoch_res[proc] = {
            "n_pos": n_pos,
            "n_zero": n_zero,
            "coherence": {
                "pos_param_coherence": coh_param_pos["coherence_ratio"],
                "zero_param_coherence": coh_param_zero["coherence_ratio"],
                "pos_W_coherence": coh_W_pos["coherence_ratio"],
                "zero_W_coherence": coh_W_zero["coherence_ratio"],
                "pos_param_stats": coh_param_pos,
                "zero_param_stats": coh_param_zero,
            },
            "decomposition": {
                "pos": clean_dict(decomp_pos),
                "zero": clean_dict(decomp_zero),
                "full": clean_dict(decomp_full),
            },
            "counterfactual_replays": {
                "canonical_full_param_129d": {
                    "pos_norm": decomp_pos["norm_full_param"],
                    "zero_norm": decomp_zero["norm_full_param"],
                    "ratio_zero_over_pos": decomp_zero["norm_full_param"] / (decomp_pos["norm_full_param"] + EPS),
                    "cos_pos_zero": cos_pos_zero_param,
                    "cos_pos_full": cos_pos_full_param,
                    "cos_zero_full": cos_zero_full_param,
                },
                "no_bias_weight_only_128d": {
                    "pos_norm": decomp_pos["norm_W"],
                    "zero_norm": decomp_zero["norm_W"],
                    "ratio_zero_over_pos": decomp_zero["norm_W"] / (decomp_pos["norm_W"] + EPS),
                    "cos_pos_zero": cos_pos_zero_W,
                    "cos_pos_full": cos_pos_full_W,
                    "cos_zero_full": cos_zero_full_W,
                },
                "centered_feature_only_no_bias_no_dc_128d": {
                    "pos_norm": decomp_pos["norm_W_cent"],
                    "zero_norm": decomp_zero["norm_W_cent"],
                    "ratio_zero_over_pos": decomp_zero["norm_W_cent"] / (decomp_pos["norm_W_cent"] + EPS),
                    "cos_pos_zero": cos_pos_zero_cent,
                    "cos_pos_full": cos_pos_full_cent,
                    "cos_zero_full": cos_zero_full_cent,
                },
                "dc_common_feature_only_128d": {
                    "pos_norm": decomp_pos["norm_W_DC"],
                    "zero_norm": decomp_zero["norm_W_DC"],
                    "ratio_zero_over_pos": decomp_zero["norm_W_DC"] / (decomp_pos["norm_W_DC"] + EPS),
                    "cos_pos_zero": cos_pos_zero_DC,
                }
            }
        }

    return epoch_res


def main() -> None:
    manifest = json.load(open(OUT_DIR / "audit_manifest.json"))
    
    # Load targets / oracles
    can_rows = list(csv.DictReader(open(manifest["targets"]["primary_canonical_ep10"]["source"])))
    ep2_oracle = {}
    for proc in PROCESSES:
        p_rows = [r for r in can_rows if r["epoch"] == "10" and r["process"] == proc]
        ep2_oracle[proc] = {int(r["basin_idx"]): float(r["w_star"]) for r in p_rows}

    # Also load R8 ep2 state-conditional oracle specifically for R8
    r8_orc_file = manifest["targets"]["sensitivity_r8_ep2"]["source"]
    r8_rows = list(csv.DictReader(open(r8_orc_file)))
    r8_ep2_oracle = {}
    for proc in PROCESSES:
        p_rows = [r for r in r8_rows if r["epoch"] == "2" and r["process"] == proc]
        r8_ep2_oracle[proc] = {int(r["basin_idx"]): float(r["w_star"]) for r in p_rows}

    cfg_path = "conf/config_dmopex_interceptE_S0_aicdelay2.yaml"
    c = load_config(cfg_path)
    c_cli = parse_args(["--config", cfg_path, "--gpu-id", "0",
                        "--output-root", "results/intercept_aicdelay",
                        "--run-name", "E_S0_aicdelay2"])
    apply_runtime_overrides(c, c_cli, config_path=cfg_path)
    c["mode"] = "train"
    c["model"]["phy"]["disable_compile"] = True
    dl = _build_data_loader(c)

    print("[Phase 2] Analyzing R8 Epoch 2 (Primary Checkpoint) and Epoch 3...")
    res_ep2 = analyze_checkpoint(2, c, dl, "cuda:0", r8_ep2_oracle)
    res_ep3 = analyze_checkpoint(3, c, dl, "cuda:0", r8_ep2_oracle)

    all_res = {
        "r8_ep2_primary": res_ep2,
        "r8_ep3_persistence": res_ep3,
    }

    out_file = OUT_DIR / "gradient_coherence_decomposition.json"
    out_file.write_text(json.dumps(all_res, indent=2))
    print(f"[Phase 2] Saved full decomposition to {out_file}")

    # Create CSV table
    table_rows = []
    for ep_tag, ep_data in all_res.items():
        for proc, pdata in ep_data.items():
            cf = pdata["counterfactual_replays"]
            dec = pdata["decomposition"]
            coh = pdata["coherence"]
            table_rows.append({
                "checkpoint": ep_tag,
                "process": proc,
                "n_pos": pdata["n_pos"],
                "n_zero": pdata["n_zero"],
                "pos_param_coherence": coh["pos_param_coherence"],
                "zero_param_coherence": coh["zero_param_coherence"],
                "pos_bias_energy_frac": dec["pos"]["bias_energy_frac"],
                "zero_bias_energy_frac": dec["zero"]["bias_energy_frac"],
                "pos_dc_energy_frac": dec["pos"]["dc_energy_frac"],
                "zero_dc_energy_frac": dec["zero"]["dc_energy_frac"],
                # Canonical 129D
                "can_ratio_zero_over_pos": cf["canonical_full_param_129d"]["ratio_zero_over_pos"],
                "can_cos_pos_zero": cf["canonical_full_param_129d"]["cos_pos_zero"],
                "can_cos_pos_full": cf["canonical_full_param_129d"]["cos_pos_full"],
                # No-bias 128D
                "nobias_ratio_zero_over_pos": cf["no_bias_weight_only_128d"]["ratio_zero_over_pos"],
                "nobias_cos_pos_zero": cf["no_bias_weight_only_128d"]["cos_pos_zero"],
                "nobias_cos_pos_full": cf["no_bias_weight_only_128d"]["cos_pos_full"],
                # Centered 128D (No-bias, No-DC)
                "cent_ratio_zero_over_pos": cf["centered_feature_only_no_bias_no_dc_128d"]["ratio_zero_over_pos"],
                "cent_cos_pos_zero": cf["centered_feature_only_no_bias_no_dc_128d"]["cos_pos_zero"],
                "cent_cos_pos_full": cf["centered_feature_only_no_bias_no_dc_128d"]["cos_pos_full"],
            })

    df_table = pd.DataFrame(table_rows)
    table_csv = OUT_DIR / "gradient_coherence_table.csv"
    df_table.to_csv(table_csv, index=False)
    print(f"[Phase 2] Saved table to {table_csv}")


if __name__ == "__main__":
    main()
