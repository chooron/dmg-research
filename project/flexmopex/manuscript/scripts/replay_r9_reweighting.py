#!/usr/bin/env python3
"""Preflight Check: Counterfactual R9 Shared-Head Aggregation Replay with Sensitivity Reweighting.

Uses the R8 epoch-2 checkpoint (results/intercept_aicdelay/E_S0_aicdelay2/model/learnedweightmopexe_ep2.pt)
and the exact 5114-day evaluation window to compute shared-head population gradient aggregation
under:
  1. Original Canonical Gradient Aggregation (reweighting OFF)
  2. Sensitivity-Reweighted Gradient Aggregation (reweighting ON, cap=5.0)

Outputs: results/intercept_aicdelay/E_S0_aicdelay2/R9_separability/r9_reweight_replay.json
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
from project.flexmopex.models.learned_weight_mopex_candidates import reweight_fit_gradient  # noqa: E402

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

    # Replay analysis
    results = {}
    for proc in PROCESSES:
        g_w = g_w_all[proc]
        jac = jac_all[proc]
        w_star = np.array([ep2_oracle[proc][b] for b in range(B)])
        pos_np = np.nan_to_num(w_star) > 0
        opos = torch.from_numpy(pos_np)
        ozero = ~opos

        # 1. Canonical (original) gate gradient
        g_z_orig = g_w * jac

        # 2. Reweighted structural gradient
        # Apply the exact mathematical transformation:
        s = torch.abs(g_w)
        s_mean = torch.mean(s) + EPS
        a_raw = s / s_mean
        a_cap = torch.clamp(a_raw, max=args.cap)
        a = a_cap / (torch.mean(a_cap) + EPS)
        g_tmp = a * g_w
        scale = s_mean / (torch.mean(torch.abs(g_tmp)) + EPS)
        g_w_rew = g_tmp * scale
        g_z_rew = g_w_rew * jac

        # Compute shared-head aggregates
        h = h_all
        # Original
        agg_p_orig = (g_z_orig[opos].unsqueeze(-1) * h[opos]).sum(0)
        agg_z_orig = (g_z_orig[ozero].unsqueeze(-1) * h[ozero]).sum(0)
        agg_f_orig = agg_p_orig + agg_z_orig

        # Reweighted
        agg_p_rew = (g_z_rew[opos].unsqueeze(-1) * h[opos]).sum(0)
        agg_z_rew = (g_z_rew[ozero].unsqueeze(-1) * h[ozero]).sum(0)
        agg_f_rew = agg_p_rew + agg_z_rew

        res = {
            "pos_n": int(opos.sum()),
            "zero_n": int(ozero.sum()),
            "original": {
                "pos_norm": float(agg_p_orig.norm()),
                "zero_norm": float(agg_z_orig.norm()),
                "ratio_zero_over_pos": float(agg_z_orig.norm() / (agg_p_orig.norm() + EPS)),
                "pos_norm_per_basin": float(agg_p_orig.norm() / max(int(opos.sum()), 1)),
                "zero_norm_per_basin": float(agg_z_orig.norm() / max(int(ozero.sum()), 1)),
                "pos_bias_sum": float(g_z_orig[opos].sum()),
                "zero_bias_sum": float(g_z_orig[ozero].sum()),
                "net_bias_sum": float(g_z_orig.sum()),
                "cos_pos_zero": float(F.cosine_similarity(agg_p_orig.view(1, -1), agg_z_orig.view(1, -1))[0]),
                "cos_pos_full": float(F.cosine_similarity(agg_p_orig.view(1, -1), agg_f_orig.view(1, -1))[0]),
                "cos_zero_full": float(F.cosine_similarity(agg_z_orig.view(1, -1), agg_f_orig.view(1, -1))[0]),
            },
            "reweighted": {
                "pos_norm": float(agg_p_rew.norm()),
                "zero_norm": float(agg_z_rew.norm()),
                "ratio_zero_over_pos": float(agg_z_rew.norm() / (agg_p_rew.norm() + EPS)),
                "pos_norm_per_basin": float(agg_p_rew.norm() / max(int(opos.sum()), 1)),
                "zero_norm_per_basin": float(agg_z_rew.norm() / max(int(ozero.sum()), 1)),
                "pos_bias_sum": float(g_z_rew[opos].sum()),
                "zero_bias_sum": float(g_z_rew[ozero].sum()),
                "net_bias_sum": float(g_z_rew.sum()),
                "cos_pos_zero": float(F.cosine_similarity(agg_p_rew.view(1, -1), agg_z_rew.view(1, -1))[0]),
                "cos_pos_full": float(F.cosine_similarity(agg_p_rew.view(1, -1), agg_f_rew.view(1, -1))[0]),
                "cos_zero_full": float(F.cosine_similarity(agg_z_rew.view(1, -1), agg_f_rew.view(1, -1))[0]),
                "cos_full_orig_vs_rew": float(F.cosine_similarity(agg_f_orig.view(1, -1), agg_f_rew.view(1, -1))[0]),
            },
            "mean_abs_gw_orig": float(torch.mean(torch.abs(g_w))),
            "mean_abs_gw_rew": float(torch.mean(torch.abs(g_w_rew))),
            "weight_multiplier_pos_mean": float(torch.mean(a[opos])),
            "weight_multiplier_zero_mean": float(torch.mean(a[ozero])),
        }
        results[proc] = res

    out_path = Path(args.output_root) / args.run_name / "R9_separability" / "r9_reweight_replay.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Replay saved to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
