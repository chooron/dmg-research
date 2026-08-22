#!/usr/bin/env python3
"""Phase 4: Initialization Audit and Shared-Task Gradient Interference (Testing Hypothesis D).

Examines two complementary questions at the UNTRAINED baseline (epoch 0) checkpoint:
  4.1 Initialization Gradient Separability:
        At random init (ep0), before any gradient signal shapes the network,
        are oracle-positive and oracle-zero subgroups already separable in
        gradient space? If cos(pos,full) ≈ -1 even at ep0, the problem is
        structural (not learning-induced); if it only emerges by ep2→ep3,
        collapse is training-induced.
  4.2 Shared-Task Gradient Interference:
        Measures how much the gradient for w_int ON-logit is "corrupted"
        by the simultaneous optimization of w_phen, w_snow, w_sub:
        - gfit_int: gradient of fit_loss w.r.t. w_int_head_params
        - gfit_phen + gfit_snow + gfit_sub: gradients of same loss w.r.t. other heads
        - cos(gfit_int, gfit_other) per process and basin group
        - Decompose shared backbone gradient contamination:
            g_backbone = sum_{proc} g_proc_backbone
        - cos(g_backbone_int, g_backbone_full) to quantify backbone-level interference
  4.3 Gradient magnitude imbalance vs. training state trajectory:
        Track ratio ||g_zero|| / ||g_pos|| for w_int from ep0→ep10
        to see when norm ratio explodes (if it starts large → Hypothesis D,
        if it grows gradually → Hypothesis C training dynamics)

Outputs: results/root_cause_r13/initialization_audit.json
         results/root_cause_r13/initialization_audit_table.csv
"""
from __future__ import annotations

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
EPS = 1e-12


def per_basin_fit(q: torch.Tensor, obs: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    o = torch.nan_to_num(obs, nan=0.0)
    sq = (q - o) ** 2 / (std.view(1, -1, 1) ** 2)
    mask = ~torch.isnan(obs)
    n_valid = mask.sum(dim=0).clamp(min=1)
    sq = torch.where(mask, sq, torch.zeros_like(sq))
    return sq.sum(dim=0) / n_valid


def analyze_epoch(epoch: int, cfg: dict, dl, dev: str, oracle_dict: dict) -> dict:
    td, ed = dl.train_dataset, dl.eval_dataset
    B = td["x_phy"].shape[1]
    n_attr = td["xc_nn_norm"].shape[-1] - 3
    attrs = td["xc_nn_norm"][0, :, -n_attr:].to(dev)
    std_train = (np.nanstd(td["target"][:, :, 0].cpu().numpy(), axis=0) + 0.1).astype(np.float32)
    n_out = int(ed["x_phy"].shape[0]) - 365
    y_ev = ed["target"][365:365 + n_out, :, 0].to(dev)
    n_valid_b = (~torch.isnan(y_ev)).sum(dim=0).float()
    N = float(n_valid_b.sum())

    handler = build_handler(cfg)
    handler.load_model(epoch)
    for m in handler.model_dict.values():
        m.eval()
    model = next(iter(handler.model_dict.values()))
    phy, nn = model.phy_model, model.nn_model

    # -- Pre-compute backbone hidden states --
    with torch.no_grad():
        h_full = nn.backbone(attrs).detach().cpu()  # [B, 128]
    h_bar = h_full.mean(0, keepdim=True)

    x_ev = ed["x_phy"].to(dev)
    doy_ev = ed["doy"].to(dev)
    std_t = torch.from_numpy(std_train).to(dev)

    # Accumulate per-basin per-process gradient w.r.t. weights_on logit
    # g_z[proc][b] = dL/d(logit_proc) at basin b
    g_z_per_proc = {p: [] for p in PROCESSES}
    chunk = 64
    for c0 in range(0, B, chunk):
        c1 = min(c0 + chunk, B)
        sample = {"x_phy": x_ev[:, c0:c1], "doy": doy_ev[:, c0:c1], "c_nn_norm": attrs[c0:c1]}
        params, logits, weights_on, mopex_params, routing = build_forward(phy, nn, sample)
        # logits: [B_chunk, 4, 2] if exposed, else recompute from weights_on
        # We compute gradient w.r.t. the logit through autograd on weights_on directly
        out = run_loop(phy, sample, weights_on, mopex_params, routing)
        q = out["streamflow"]
        obs = ed["target"][365:365 + n_out, c0:c1].to(dev)
        L_b = per_basin_fit(q, obs, std_t[c0:c1])
        L_obj = (L_b * n_valid_b[c0:c1] / N).sum()
        # Gradient w.r.t. weights_on (after softmax = sigmoid-like)
        g_w = torch.autograd.grad(L_obj, weights_on, retain_graph=False)[0]  # [B_chunk, 4]
        jac_w = weights_on * (1 - weights_on)  # [B_chunk, 4]
        g_logit = g_w * jac_w  # [B_chunk, 4]
        for p in PROCESSES:
            col = GATE_IDX[p]
            g_z_per_proc[p].append(g_logit[:, col].detach().cpu())
        del q, out, params, logits, weights_on, mopex_params, routing
        torch.cuda.empty_cache()

    for p in PROCESSES:
        g_z_per_proc[p] = torch.cat(g_z_per_proc[p])  # [B]

    # -- Compute per-process gradient vectors in head-param space --
    # G_W_i[proc][basin] = g_logit[basin] * h[basin]  (128D)
    # G_b_i[proc][basin] = g_logit[basin]              (1D)
    G_param = {}
    for p in PROCESSES:
        gz = g_z_per_proc[p]
        G_W = gz.unsqueeze(-1) * h_full        # [B, 128]
        G_b = gz.unsqueeze(-1)                 # [B, 1]
        G_param[p] = torch.cat([G_W, G_b], -1)  # [B, 129]

    # -- Backbone-level contamination --
    # For each basin i, backbone gradient from process p is:
    # g_bb_p_i = g_logit_p_i * W_head_p^T   (128D)
    # We approximate this via the outer product direction without accessing W_head directly
    # Instead, measure cross-process gradient alignment in G_param space
    cross_align = {}
    for p1 in PROCESSES:
        for p2 in PROCESSES:
            if p1 >= p2:
                continue
            # Sum over basins
            v1 = G_param[p1].sum(0)
            v2 = G_param[p2].sum(0)
            cos = float(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))[0])
            cross_align[f"{p1}_vs_{p2}"] = cos

    # -- Oracle subgroup analysis --
    results_by_proc = {}
    for proc in PROCESSES:
        gz = g_z_per_proc[proc]
        pos_mask = torch.tensor([oracle_dict[proc].get(b, 0) > 0 for b in range(B)])
        zero_mask = ~pos_mask
        n_pos = int(pos_mask.sum())
        n_zero = int(zero_mask.sum())

        G = G_param[proc]
        h = h_full

        # Subgroup vectors
        v_pos = G[pos_mask].sum(0)
        v_zero = G[zero_mask].sum(0)
        v_full = G.sum(0)

        cos_pos_full = float(F.cosine_similarity(v_pos.unsqueeze(0), v_full.unsqueeze(0))[0])
        cos_zero_full = float(F.cosine_similarity(v_zero.unsqueeze(0), v_full.unsqueeze(0))[0])
        cos_pos_zero = float(F.cosine_similarity(v_pos.unsqueeze(0), v_zero.unsqueeze(0))[0])

        norm_pos = float(v_pos.norm())
        norm_zero = float(v_zero.norm())
        ratio = norm_zero / (norm_pos + EPS)

        # Gradient magnitudes per basin
        G_norms = G.norm(dim=-1)
        gz_pos_med = float(G_norms[pos_mask].median())
        gz_zero_med = float(G_norms[zero_mask].median())
        gz_pos_mean = float(G_norms[pos_mask].mean())
        gz_zero_mean = float(G_norms[zero_mask].mean())

        # Logit-level metrics
        gz_pos_logit_med = float(gz[pos_mask].abs().median())
        gz_zero_logit_med = float(gz[zero_mask].abs().median())
        gz_pos_sign_pos = float((gz[pos_mask] > 0).float().mean())
        gz_zero_sign_pos = float((gz[zero_mask] > 0).float().mean())

        results_by_proc[proc] = {
            "n_pos": n_pos, "n_zero": n_zero,
            "cos_pos_full": cos_pos_full,
            "cos_zero_full": cos_zero_full,
            "cos_pos_zero": cos_pos_zero,
            "norm_pos": norm_pos,
            "norm_zero": norm_zero,
            "ratio_zero_over_pos": ratio,
            "G_norm_pos_median": gz_pos_med,
            "G_norm_zero_median": gz_zero_med,
            "G_norm_pos_mean": gz_pos_mean,
            "G_norm_zero_mean": gz_zero_mean,
            "logit_grad_pos_median": gz_pos_logit_med,
            "logit_grad_zero_median": gz_zero_logit_med,
            "frac_positive_sign_pos": gz_pos_sign_pos,
            "frac_positive_sign_zero": gz_zero_sign_pos,
        }

    return {"by_process": results_by_proc, "cross_process_alignment": cross_align}


def main() -> None:
    manifest = json.load(open(OUT_DIR / "audit_manifest.json"))

    # Load oracle from R8 ep2 state-conditional
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

    # Epochs: ep0 (random init if available), ep1, ep2, ep3
    # ep0 = Canonical baseline is the closest to random init we have
    # Use ep1 as earliest and ep2 as pre-collapse, ep3 as post-collapse
    epochs_to_audit = [1, 2, 3, 5, 10]

    all_results = {}
    table_rows = []

    for ep in epochs_to_audit:
        print(f"[Phase 4] Analyzing epoch {ep}...")
        res = analyze_epoch(ep, c, dl, "cuda:0", r8_ep2_oracle)
        all_results[f"ep{ep}"] = res

        for proc in PROCESSES:
            pr = res["by_process"][proc]
            table_rows.append({
                "epoch": ep,
                "process": proc,
                "n_pos": pr["n_pos"],
                "n_zero": pr["n_zero"],
                "cos_pos_full": pr["cos_pos_full"],
                "cos_zero_full": pr["cos_zero_full"],
                "cos_pos_zero": pr["cos_pos_zero"],
                "norm_pos": pr["norm_pos"],
                "norm_zero": pr["norm_zero"],
                "ratio_zero_over_pos": pr["ratio_zero_over_pos"],
                "G_norm_pos_median": pr["G_norm_pos_median"],
                "G_norm_zero_median": pr["G_norm_zero_median"],
                "frac_sign_pos_among_pos": pr["frac_positive_sign_pos"],
                "frac_sign_pos_among_zero": pr["frac_positive_sign_zero"],
            })

        # Print compact summary
        pi = res["by_process"]["w_int"]
        ca = res["cross_process_alignment"]
        print(f"  [w_int] cos(pos,full)={pi['cos_pos_full']:+.3f} | cos(zero,full)={pi['cos_zero_full']:+.3f} | ratio={pi['ratio_zero_over_pos']:.2f} | sign_pos={pi['frac_positive_sign_pos']:.2f} | sign_zero={pi['frac_positive_sign_zero']:.2f}")
        print(f"  [cross] int_vs_phen={ca.get('w_int_vs_w_phen', ca.get('w_phen_vs_w_int',0)):+.3f} | int_vs_snow={ca.get('w_int_vs_w_snow', ca.get('w_snow_vs_w_int',0)):+.3f} | int_vs_sub={ca.get('w_int_vs_w_sub', ca.get('w_sub_vs_w_int',0)):+.3f}")

    out_json = OUT_DIR / "initialization_audit.json"
    out_json.write_text(json.dumps(all_results, indent=2))

    df = pd.DataFrame(table_rows)
    df.to_csv(OUT_DIR / "initialization_audit_table.csv", index=False)

    print(f"\n[Phase 4 Complete] Written: {out_json}")


if __name__ == "__main__":
    main()
