#!/usr/bin/env python3
"""Preflight B: 3-Way Counterfactual Replay at R8 Epoch-2 Checkpoint (Canonical vs R10 vs R11).

Replays shared-head aggregation on the exact 5114-day evaluation window across all 671 basins:
  1. Canonical Aggregation (no reweighting)
  2. R10 Sensitivity-Only Reweighting (cap=5.0)
  3. R11 Direction-Balanced + Sensitivity-Weighted Reweighting (cap=5.0)

Outputs: results/intercept_aicdelay/E_S0_aicdelay2/R9_separability/r11_reweight_replay.json
"""
from __future__ import annotations

import argparse
import csv
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

from project.flexmopex import load_config  # noqa: E402
from project.flexmopex.run_model import (  # noqa: E402
    apply_runtime_overrides, parse_args, _build_data_loader,
)
from scripts.diagnose_wint_collapse import build_handler, build_forward, run_loop  # noqa: E402
from project.flexmopex.models.learned_weight_mopex_candidates import (  # noqa: E402
    reweight_fit_gradient,
    direction_balanced_reweight_fit_gradient,
)

PROCESSES = ["w_phen", "w_int", "w_snow", "w_sub"]
GATE_IDX = {"w_phen": 0, "w_int": 1, "w_snow": 2, "w_sub": 3}
COSTS = {"w_phen": 2.0, "w_int": 2.0, "w_snow": 2.0, "w_sub": 1.0}
AIC_ALPHA = 0.01
B_TOTAL = 671
EPS = 1e-12


def per_basin_fit(q: torch.Tensor, obs: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    o = torch.nan_to_num(obs, nan=0.0)
    sq = (q - o) ** 2 / (std.view(1, -1, 1) ** 2)
    mask = ~torch.isnan(obs)
    n_valid = mask.sum(dim=0).clamp(min=1)
    sq = torch.where(mask, sq, torch.zeros_like(sq))
    return sq.sum(dim=0) / n_valid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/config_dmopex_interceptE_S0_aicdelay2.yaml")
    ap.add_argument("--output-root", default="results/intercept_aicdelay")
    ap.add_argument("--run-name", default="E_S0_aicdelay2")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--cap", type=float, default=5.0)
    ap.add_argument("--chunk-size", type=int, default=128)
    args = ap.parse_args()

    dev = f"cuda:{args.gpu_id}"
    cli = parse_args(["--config", args.config, "--gpu-id", str(args.gpu_id),
                      "--output-root", str(args.output_root), "--run-name", args.run_name])
    cfg = load_config(args.config)
    apply_runtime_overrides(cfg, cli, config_path=args.config)
    cfg["mode"] = "train"
    cfg["model"]["phy"]["disable_compile"] = True
    if str(cfg["device"]).startswith("cuda"):
        torch.cuda.set_device(cfg["device"])

    dl = _build_data_loader(cfg)
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
    handler.load_model(2)
    for m in handler.model_dict.values():
        m.eval()
    model = next(iter(handler.model_dict.values()))
    phy, nn = model.phy_model, model.nn_model

    # Load R9 state-conditional oracle labels at epoch 2
    r9_oracle_path = Path(args.output_root) / args.run_name / "R9_separability" / "oracle_state_conditional.csv"
    oracle_rows = list(csv.DictReader(open(r9_oracle_path)))
    ep2_oracle = {p_: {} for p_ in PROCESSES}
    for r in oracle_rows:
        if r["epoch"] == "2":
            ep2_oracle[r["process"]][int(r["basin_idx"])] = float(r["w_star"])

    with torch.no_grad():
        h_all = nn.backbone(attrs).detach().cpu()  # [671, 128]

    x_phy_full = ed["x_phy"].to(dev)
    doy_full = ed["doy"].to(dev)
    std_t = torch.from_numpy(std_train).to(dev)
    nv = torch.from_numpy(n_valid_b).to(dev)

    # Compute canonical g_fit_w_obj for all 671 basins
    g_w_all = {p_: [] for p_ in PROCESSES}
    jac_all = {p_: [] for p_ in PROCESSES}
    w_all = {p_: [] for p_ in PROCESSES}

    for c0 in range(0, B, args.chunk_size):
        c1 = min(c0 + args.chunk_size, B)
        sample = {"x_phy": x_phy_full[:, c0:c1], "doy": doy_full[:, c0:c1], "c_nn_norm": attrs[c0:c1]}
        params, logits, weights_on, mopex_params, routing = build_forward(phy, nn, sample)
        out = run_loop(phy, sample, weights_on, mopex_params, routing)
        q = out["streamflow"]
        obs = ed["target"][365:365 + n_out, c0:c1].to(dev)
        L_b = per_basin_fit(q, obs, std_t[c0:c1])
        L_fit_obj = (L_b * nv[c0:c1] / N).sum()
        g_w = torch.autograd.grad(L_fit_obj, weights_on, retain_graph=True)[0]  # [C, 4]
        jac = weights_on * (1 - weights_on)  # [C, 4]

        for p_ in PROCESSES:
            col = GATE_IDX[p_]
            g_w_all[p_].append(g_w[:, col].detach().cpu())
            jac_all[p_].append(jac[:, col].detach().cpu())
            w_all[p_].append(weights_on[:, col].detach().cpu())

        del q, out, params, logits, weights_on, mopex_params, routing
        torch.cuda.empty_cache()

    for p_ in PROCESSES:
        g_w_all[p_] = torch.cat(g_w_all[p_])  # [671]
        jac_all[p_] = torch.cat(jac_all[p_])  # [671]
        w_all[p_] = torch.cat(w_all[p_])      # [671]

    # Stack for 4-process matrix
    G_w_matrix = torch.stack([g_w_all[p_] for p_ in PROCESSES], dim=1)  # [671, 4]
    Jac_matrix = torch.stack([jac_all[p_] for p_ in PROCESSES], dim=1)  # [671, 4]

    # 1. Canonical (unreweighted)
    G_w_canon = G_w_matrix
    G_z_canon = G_w_canon * Jac_matrix

    # 2. R10 Sensitivity-only reweighted
    # Compute R10 transform:
    s_r10 = torch.abs(G_w_matrix)
    s_r10_mean = torch.mean(s_r10, dim=0, keepdim=True) + EPS
    a_raw_r10 = s_r10 / s_r10_mean
    a_cap_r10 = torch.clamp(a_raw_r10, max=args.cap)
    a_r10 = a_cap_r10 / (torch.mean(a_cap_r10, dim=0, keepdim=True) + EPS)
    G_tmp_r10 = a_r10 * G_w_matrix
    scale_r10 = s_r10_mean / (torch.mean(torch.abs(G_tmp_r10), dim=0, keepdim=True) + EPS)
    G_w_r10 = G_tmp_r10 * scale_r10
    G_z_r10 = G_w_r10 * Jac_matrix

    # 3. R11 Direction-balanced + sensitivity reweighted
    G_w_r11 = torch.empty_like(G_w_matrix)
    B_factor = torch.ones_like(G_w_matrix)
    R_factor = torch.ones_like(G_w_matrix)

    for p in range(4):
        gp = G_w_matrix[:, p]
        s_all_mean = torch.mean(torch.abs(gp)) + EPS
        mask_on = gp < 0
        mask_off = gp > 0
        n_on = int(mask_on.sum().item())
        n_off = int(mask_off.sum().item())
        gtmp_p = torch.zeros_like(gp)

        if n_on > 0 and n_off > 0:
            b_on = float(B) / (2.0 * float(n_on))
            b_off = float(B) / (2.0 * float(n_off))
            B_factor[mask_on, p] = b_on
            B_factor[mask_off, p] = b_off

            # ON
            s_on = torch.abs(gp[mask_on])
            s_on_mean = torch.mean(s_on) + EPS
            r_on = torch.clamp(s_on / s_on_mean, max=args.cap)
            r_on = r_on / (torch.mean(r_on) + EPS)
            R_factor[mask_on, p] = r_on
            gtmp_p[mask_on] = b_on * r_on * gp[mask_on]

            # OFF
            s_off = torch.abs(gp[mask_off])
            s_off_mean = torch.mean(s_off) + EPS
            r_off = torch.clamp(s_off / s_off_mean, max=args.cap)
            r_off = r_off / (torch.mean(r_off) + EPS)
            R_factor[mask_off, p] = r_off
            gtmp_p[mask_off] = b_off * r_off * gp[mask_off]

        elif n_on > 0:
            s_on = torch.abs(gp[mask_on])
            s_on_mean = torch.mean(s_on) + EPS
            r_on = torch.clamp(s_on / s_on_mean, max=args.cap)
            r_on = r_on / (torch.mean(r_on) + EPS)
            R_factor[mask_on, p] = r_on
            gtmp_p[mask_on] = 1.0 * r_on * gp[mask_on]

        elif n_off > 0:
            s_off = torch.abs(gp[mask_off])
            s_off_mean = torch.mean(s_off) + EPS
            r_off = torch.clamp(s_off / s_off_mean, max=args.cap)
            r_off = r_off / (torch.mean(r_off) + EPS)
            R_factor[mask_off, p] = r_off
            gtmp_p[mask_off] = 1.0 * r_off * gp[mask_off]

        scale = s_all_mean / (torch.mean(torch.abs(gtmp_p)) + EPS)
        G_w_r11[:, p] = gtmp_p * scale

    G_z_r11 = G_w_r11 * Jac_matrix

    h = h_all  # [671, 128]

    # Replay analysis per process
    results = {}
    for proc in PROCESSES:
        col = GATE_IDX[proc]
        w_star = np.array([ep2_oracle[proc][b] for b in range(B)])
        pos_np = np.nan_to_num(w_star) > 0
        opos = torch.from_numpy(pos_np)
        ozero = ~opos

        gz_can = G_z_canon[:, col]
        gz_r10 = G_z_r10[:, col]
        gz_r11 = G_z_r11[:, col]

        # Endogenous ON/OFF partition based on g_w sign
        loc_on = G_w_matrix[:, col] < 0
        loc_off = G_w_matrix[:, col] > 0

        def calc_aggregates(gz):
            # Endogenous ON/OFF
            agg_on = (gz[loc_on].unsqueeze(-1) * h[loc_on]).sum(0) if loc_on.any() else torch.zeros(128)
            agg_off = (gz[loc_off].unsqueeze(-1) * h[loc_off]).sum(0) if loc_off.any() else torch.zeros(128)
            agg_full = agg_on + agg_off

            # Post-hoc Oracle Positive/Zero
            agg_pos = (gz[opos].unsqueeze(-1) * h[opos]).sum(0) if opos.any() else torch.zeros(128)
            agg_zero = (gz[ozero].unsqueeze(-1) * h[ozero]).sum(0) if ozero.any() else torch.zeros(128)

            return {
                "endogenous_on_off": {
                    "n_on": int(loc_on.sum().item()),
                    "n_off": int(loc_off.sum().item()),
                    "on_norm": float(agg_on.norm()),
                    "off_norm": float(agg_off.norm()),
                    "ratio_off_over_on": float(agg_off.norm() / (agg_on.norm() + EPS)),
                    "cos_on_off": float(F.cosine_similarity(agg_on.view(1, -1), agg_off.view(1, -1))[0]) if loc_on.any() and loc_off.any() else 0.0,
                    "cos_on_full": float(F.cosine_similarity(agg_on.view(1, -1), agg_full.view(1, -1))[0]) if loc_on.any() else 0.0,
                    "cos_off_full": float(F.cosine_similarity(agg_off.view(1, -1), agg_full.view(1, -1))[0]) if loc_off.any() else 0.0,
                    "on_bias_sum": float(gz[loc_on].sum()) if loc_on.any() else 0.0,
                    "off_bias_sum": float(gz[loc_off].sum()) if loc_off.any() else 0.0,
                    "net_bias_sum": float(gz.sum()),
                    "full_norm": float(agg_full.norm()),
                },
                "oracle_subgroups_post_hoc": {
                    "n_pos": int(opos.sum().item()),
                    "n_zero": int(ozero.sum().item()),
                    "pos_norm": float(agg_pos.norm()),
                    "zero_norm": float(agg_zero.norm()),
                    "ratio_zero_over_pos": float(agg_zero.norm() / (agg_pos.norm() + EPS)),
                    "cos_pos_zero": float(F.cosine_similarity(agg_pos.view(1, -1), agg_zero.view(1, -1))[0]),
                    "cos_pos_full": float(F.cosine_similarity(agg_pos.view(1, -1), agg_full.view(1, -1))[0]),
                    "cos_zero_full": float(F.cosine_similarity(agg_zero.view(1, -1), agg_full.view(1, -1))[0]),
                    "pos_bias_sum": float(gz[opos].sum()),
                    "zero_bias_sum": float(gz[ozero].sum()),
                }
            }

        res_can = calc_aggregates(gz_can)
        res_r10 = calc_aggregates(gz_r10)
        res_r11 = calc_aggregates(gz_r11)

        # Cross-comparison cosines
        agg_f_can = (gz_can.unsqueeze(-1) * h).sum(0)
        agg_f_r10 = (gz_r10.unsqueeze(-1) * h).sum(0)
        agg_f_r11 = (gz_r11.unsqueeze(-1) * h).sum(0)

        results[proc] = {
            "canonical": res_can,
            "r10_sensitivity": res_r10,
            "r11_direction_balanced": res_r11,
            "cos_full_can_vs_r10": float(F.cosine_similarity(agg_f_can.view(1, -1), agg_f_r10.view(1, -1))[0]),
            "cos_full_can_vs_r11": float(F.cosine_similarity(agg_f_can.view(1, -1), agg_f_r11.view(1, -1))[0]),
            "cos_full_r10_vs_r11": float(F.cosine_similarity(agg_f_r10.view(1, -1), agg_f_r11.view(1, -1))[0]),
            "r11_b_multiplier_on_mean": float(B_factor[loc_on, col].mean().item()) if loc_on.any() else 1.0,
            "r11_b_multiplier_off_mean": float(B_factor[loc_off, col].mean().item()) if loc_off.any() else 1.0,
            "r11_r_multiplier_on_mean": float(R_factor[loc_on, col].mean().item()) if loc_on.any() else 1.0,
            "r11_r_multiplier_off_mean": float(R_factor[loc_off, col].mean().item()) if loc_off.any() else 1.0,
        }

    out_path = Path(args.output_root) / args.run_name / "R9_separability" / "r11_reweight_replay.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"R11 Replay saved to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
