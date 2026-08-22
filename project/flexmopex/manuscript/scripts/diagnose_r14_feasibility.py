#!/usr/bin/env python3
"""Flex-MOPEX R14: Counterfactual Structural-Target Feasibility Diagnostic.

Evaluates whether Flex-MOPEX can replace the fragile shared gate-gradient signal
with an oracle-free, basin-specific counterfactual structural target (Delta J):
  Delta J(i, p) = J_OFF(i, p) - J_ON(i, p) = (L_fit_OFF - L_fit_ON) - lambda_AIC * cost_p * (N / (B * n_valid_b))

Phases:
  Phase 1: Finite structural evidence computation across checkpoints & processes
  Phase 2: Signal validity, Oracle agreement, interior-optima check, gradient comparison & temporal stability
  Phase 3: Predictability of DeltaJ from raw attributes X vs learned representation h (5-fold CV)
  Phase 4: Parameter-state contamination & controlled parameter-swap decomposition
  Phase 5: Soft-target formulation evaluation (Candidates A, B, C)
  Phase 6: Real execution compute & memory benchmarking

Outputs saved to: results/feasibility_r14/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config  # noqa: E402
from project.flexmopex.run_model import (  # noqa: E402
    apply_runtime_overrides, parse_args, _build_data_loader,
)
from scripts.diagnose_wint_collapse import build_handler, build_forward, run_loop  # noqa: E402

OUT_DIR = Path("results/feasibility_r14")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSES = ["w_phen", "w_int", "w_snow", "w_sub"]
GATE_IDX = {"w_phen": 0, "w_int": 1, "w_snow": 2, "w_sub": 3}
COSTS = {"w_phen": 2.0, "w_int": 2.0, "w_snow": 2.0, "w_sub": 1.0}
AIC_ALPHA = 0.01
W_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
EPS = 1e-6


def per_basin_fit(q: torch.Tensor, obs: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    o = torch.nan_to_num(obs, nan=0.0)
    sq = (q - o) ** 2 / (std.view(1, -1, 1) ** 2)
    mask = ~torch.isnan(obs)
    n_valid = mask.sum(dim=0).clamp(min=1)
    sq = torch.where(mask, sq, torch.zeros_like(sq))
    return sq.sum(dim=0) / n_valid


def evaluate_checkpoint(
    cfg_path: str,
    run_name: str,
    output_root: str,
    epoch: int,
    dl,
    dev: str,
) -> Dict[str, Any]:
    c = load_config(cfg_path)
    c_cli = parse_args(["--config", cfg_path, "--gpu-id", "0",
                        "--output-root", output_root,
                        "--run-name", run_name])
    apply_runtime_overrides(c, c_cli, config_path=cfg_path)
    c["mode"] = "train"
    c["model"]["phy"]["disable_compile"] = True

    td, ed = dl.train_dataset, dl.eval_dataset
    B = td["x_phy"].shape[1]
    n_attr = td["xc_nn_norm"].shape[-1] - 3
    attrs = td["xc_nn_norm"][0, :, -n_attr:].to(dev)
    std_train = (np.nanstd(td["target"][:, :, 0].cpu().numpy(), axis=0) + 0.1).astype(np.float32)
    n_out = int(ed["x_phy"].shape[0]) - 365
    y_ev = ed["target"][365:365 + n_out, :, 0].cpu().numpy()
    n_valid_b = np.sum(~np.isnan(y_ev), axis=0).astype(float)
    N = float(n_valid_b.sum())

    handler = build_handler(c)
    handler.load_model(epoch)
    for m in handler.model_dict.values():
        m.eval()
    model = next(iter(handler.model_dict.values()))
    phy, nn = model.phy_model, model.nn_model

    with torch.no_grad():
        p_raw = nn({"c_nn_norm": attrs})
        w_learn = F.softmax(p_raw["weights"].view(B, 4, 2).clamp(-10, 10), dim=-1)[..., 1]
        mopex_params = phy._descale_mopex_params(p_raw["params"])
        routing = phy._descale_routing_params(p_raw["gamma_uh"])
        base_w = w_learn.detach().clone()
        h_repr = nn.backbone(attrs).detach().cpu().numpy()

    # 1. Compute local autograd gradient dL_fit / dw at learned w
    x_phy_full = ed["x_phy"].to(dev)
    doy_full = ed["doy"].to(dev)
    std_t = torch.from_numpy(std_train).to(dev)
    nv = torch.from_numpy(n_valid_b).to(dev)

    g_w_all = {p_: [] for p_ in PROCESSES}
    chunk_size = 128
    for c0 in range(0, B, chunk_size):
        c1 = min(c0 + chunk_size, B)
        sample = {"x_phy": x_phy_full[:, c0:c1], "doy": doy_full[:, c0:c1], "c_nn_norm": attrs[c0:c1]}
        params, logits, weights_on, m_params, rout = build_forward(phy, nn, sample)
        out = run_loop(phy, sample, weights_on, m_params, rout)
        q = out["streamflow"]
        obs = ed["target"][365:365 + n_out, c0:c1].to(dev)
        L_b = per_basin_fit(q, obs, std_t[c0:c1])
        L_fit_obj = (L_b * nv[c0:c1] / N).sum()
        g_w = torch.autograd.grad(L_fit_obj, weights_on, retain_graph=True)[0]
        for p_ in PROCESSES:
            g_w_all[p_].append(g_w[:, GATE_IDX[p_]].detach().cpu())
        del q, out, params, logits, weights_on, m_params, rout
        torch.cuda.empty_cache()

    for p_ in PROCESSES:
        g_w_all[p_] = torch.cat(g_w_all[p_]).numpy()

    # 2. Counterfactual sweeps over W_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    S = len(W_GRID)
    process_data = {}
    for proc in PROCESSES:
        col = GATE_IDX[proc]
        w_on = base_w.repeat(S, 1)
        for s in range(S):
            w_on[s * B:(s + 1) * B, col] = W_GRID[s]
        params_rep = {k: v.repeat(S, 1) for k, v in mopex_params.items()}
        routing_rep = {k: v.repeat(S) for k, v in routing.items()}
        sample_rep = {"x_phy": ed["x_phy"].repeat(1, S, 1).to(dev),
                      "doy": ed["doy"].repeat(1, S, 1).to(dev),
                      "c_nn_norm": attrs.repeat(S, 1).to(dev)}
        with torch.no_grad():
            P, T, PET, doy, n_steps, _ = phy._prepare_forcings(sample_rep)
            Q = phy._run_weighted_loop(P, T, PET, doy, params_rep, w_on, n_steps, B * S)
            Qr = phy._apply_routing(Q.mean(-1), routing_rep).cpu().numpy()[:, :, 0]

        Qr = Qr[:, :B * S].reshape(Qr.shape[0], S, B)
        Qs = np.transpose(Qr, (0, 2, 1))[:n_out]  # [n_out, B, S]

        fit_grid = np.full((B, S), np.nan)
        nse_grid = np.full((B, S), np.nan)
        for b in range(B):
            v = ~np.isnan(y_ev[:, b])
            if v.sum() < 30:
                continue
            o = y_ev[v, b]
            ss = Qs[v, b, :]  # [n_valid, S]
            fit_grid[b, :] = np.mean((ss - o[:, None]) ** 2 / (std_train[b] ** 2), axis=0)
            ss_tot = np.sum((o - o.mean()) ** 2)
            for s in range(S):
                ss_res = np.sum((ss[:, s] - o) ** 2)
                nse_grid[b, s] = 1.0 - (ss_res / (ss_tot + EPS))

        # Objective calculations
        cost = COSTS[proc]
        aic_unit = AIC_ALPHA * cost / B
        w_star = np.full(B, np.nan)
        for b in range(B):
            vals = [fit_grid[b, s] * n_valid_b[b] / N + aic_unit * W_GRID[s] for s in range(S)]
            if np.isfinite(vals).any():
                w_star[b] = W_GRID[int(np.nanargmin(vals))]

        # Exact DeltaJ per basin
        # DeltaJ = L_fit_OFF - L_fit_ON - AIC_ALPHA * cost * (N / (B * n_valid_b))
        aic_penalty_per_basin = AIC_ALPHA * cost * (N / (B * np.maximum(n_valid_b, 1.0)))
        L_fit_off = fit_grid[:, 0]
        L_fit_on = fit_grid[:, -1]
        J_off = L_fit_off
        J_on = L_fit_on + aic_penalty_per_basin
        delta_J = J_off - J_on

        # Finite fit improvement and NSE improvement
        fit_imp_max = L_fit_off - np.nanmin(fit_grid[:, 1:], axis=1)
        dNSE_max = np.nanmax(nse_grid[:, 1:], axis=1) - nse_grid[:, 0]
        fit_imp_endpoint = L_fit_off - L_fit_on

        # Endpoints
        endpoint = np.where(delta_J > 1e-7, "ON", np.where(delta_J < -1e-7, "OFF", "TIE"))

        # Interior optimum check: w* > 0 but ON (w=1) is worse than OFF (fit_imp_endpoint < 0 or DeltaJ <= 0)
        oracle_pos = (w_star > 0)
        interior_opt_w_star = (w_star > 0) & (w_star < 1.0)
        interior_misspec = oracle_pos & (delta_J <= 0)
        on_worse_than_off_while_opt = oracle_pos & (L_fit_on > L_fit_off)

        process_data[proc] = {
            "L_fit_off": L_fit_off,
            "L_fit_on": L_fit_on,
            "J_off": J_off,
            "J_on": J_on,
            "delta_J": delta_J,
            "abs_delta_J": np.abs(delta_J),
            "endpoint": endpoint,
            "learned_w": base_w[:, col].cpu().numpy(),
            "w_star": w_star,
            "oracle_pos": oracle_pos,
            "interior_misspec": interior_misspec,
            "on_worse_than_off_while_opt": on_worse_than_off_while_opt,
            "fit_imp_max": fit_imp_max,
            "fit_imp_endpoint": fit_imp_endpoint,
            "dNSE_max": dNSE_max,
            "g_fit_local": g_w_all[proc],
            "fit_grid": fit_grid,
            "nse_grid": nse_grid,
        }

    return {
        "epoch": epoch,
        "run_name": run_name,
        "attrs": td["xc_nn_norm"][0, :, -n_attr:].cpu().numpy(),
        "h_repr": h_repr,
        "mopex_params": {k: v.mean(-1).cpu().numpy() for k, v in mopex_params.items()},
        "routing_params": {k: v.cpu().numpy() for k, v in routing.items()},
        "process_data": process_data,
        "n_valid_b": n_valid_b,
        "std_train": std_train,
    }


def run_probes(
    X: np.ndarray,
    y_binary: np.ndarray,
    y_continuous: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    valid_mask = np.isfinite(y_binary) & np.isfinite(y_continuous) & np.all(np.isfinite(X), axis=1)
    X_v = X[valid_mask]
    yb_v = y_binary[valid_mask].astype(int)
    yc_v = y_continuous[valid_mask]

    n_pos = int(np.sum(yb_v == 1))
    n_total = len(yb_v)
    if n_pos < 5 or n_pos > n_total - 5:
        return {"error": "insufficient class balance", "n_pos": n_pos, "n_total": n_total}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_prob = np.zeros(n_total)
    oof_pred_c = np.zeros(n_total)

    for train_idx, val_idx in skf.split(X_v, yb_v):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_v[train_idx])
        X_val = scaler.transform(X_v[val_idx])

        # Classification
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        clf.fit(X_tr, yb_v[train_idx])
        oof_prob[val_idx] = clf.predict_proba(X_val)[:, 1]

        # Regression
        reg = Ridge(alpha=1.0, random_state=seed)
        reg.fit(X_tr, yc_v[train_idx])
        oof_pred_c[val_idx] = reg.predict(X_val)

    roc_auc = float(roc_auc_score(yb_v, oof_prob))
    pr_auc = float(average_precision_score(yb_v, oof_prob))

    # Top-k recall
    k = n_pos
    top_k_indices = np.argsort(oof_prob)[-k:]
    top_k_recall = float(np.sum(yb_v[top_k_indices] == 1) / n_pos)

    # Continuous metrics
    p_corr, p_val = pearsonr(yc_v, oof_pred_c)
    s_corr, s_val = spearmanr(yc_v, oof_pred_c)
    ss_res = np.sum((yc_v - oof_pred_c) ** 2)
    ss_tot = np.sum((yc_v - np.mean(yc_v)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + EPS))

    return {
        "n_pos": n_pos,
        "n_total": n_total,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "top_k_recall": top_k_recall,
        "continuous_pearson_r": float(p_corr),
        "continuous_spearman_r": float(s_corr),
        "continuous_r2": r2,
    }


def main() -> None:
    print("=" * 80)
    print("Flex-MOPEX R14: Counterfactual Structural-Target Feasibility Diagnostic")
    print("=" * 80)

    # Checkpoint configuration
    runs_to_eval = [
        # Baseline E-S0
        ("Baseline", "conf/config_dmopex_interceptE_S0.yaml", "E_S0", "results/intercept_candidates", [0, 2, 5, 10]),
        # R8 delayed-AIC
        ("R8_AICDelay", "conf/config_dmopex_interceptE_S0_aicdelay2.yaml", "E_S0_aicdelay2", "results/intercept_aicdelay", [1, 2, 3, 4, 10]),
        # R10-B Reweighted Delay
        ("R10B_Reweight", "conf/config_dmopex_interceptE_S0_reweight_aicdelay2.yaml", "E_S0_reweight_delay2", "results/intercept_reweight", [2, 4, 10]),
    ]

    # Shared dataloader
    base_cfg = load_config("conf/config_dmopex_interceptE_S0.yaml")
    base_cfg["mode"] = "train"
    base_cfg["model"]["phy"]["disable_compile"] = True
    dl = _build_data_loader(base_cfg)

    all_data = {}
    phase1_rows = []

    print("\n[Phase 1] Evaluating checkpoints across runs...")
    for run_tag, cfg_p, r_name, out_r, epochs in runs_to_eval:
        for ep in epochs:
            tag = f"{run_tag}_ep{ep}"
            print(f"  --> Loading {tag} ...")
            try:
                res = evaluate_checkpoint(cfg_p, r_name, out_r, ep, dl, "cuda:0")
                all_data[tag] = res
                for proc in PROCESSES:
                    pd_p = res["process_data"][proc]
                    for b in range(len(pd_p["delta_J"])):
                        phase1_rows.append({
                            "run_tag": run_tag,
                            "epoch": ep,
                            "basin_idx": b,
                            "process": proc,
                            "L_fit_off": pd_p["L_fit_off"][b],
                            "L_fit_on": pd_p["L_fit_on"][b],
                            "J_off": pd_p["J_off"][b],
                            "J_on": pd_p["J_on"][b],
                            "delta_J": pd_p["delta_J"][b],
                            "abs_delta_J": pd_p["abs_delta_J"][b],
                            "endpoint": pd_p["endpoint"][b],
                            "learned_w": pd_p["learned_w"][b],
                            "w_star": pd_p["w_star"][b],
                            "oracle_pos": pd_p["oracle_pos"][b],
                            "interior_misspec": pd_p["interior_misspec"][b],
                            "fit_imp_max": pd_p["fit_imp_max"][b],
                            "fit_imp_endpoint": pd_p["fit_imp_endpoint"][b],
                            "dNSE_max": pd_p["dNSE_max"][b],
                            "g_fit_local": pd_p["g_fit_local"][b],
                        })
            except Exception as e:
                print(f"      [warn] failed to load {tag}: {e}")

    df_p1 = pd.DataFrame(phase1_rows)
    p1_csv = OUT_DIR / "phase1_structural_evidence_per_basin.csv"
    df_p1.to_csv(p1_csv, index=False)
    print(f"[Phase 1 Complete] Wrote per-basin structural evidence to {p1_csv}")

    # =========================================================================
    # Phase 2: Signal Validity, Oracle Agreement, Gradient Comparison & Stability
    # =========================================================================
    print("\n[Phase 2] Analyzing signal validity, oracle agreement, gradient comparison & stability...")
    phase2_summary = {}
    p2_table_rows = []

    for tag, res in all_data.items():
        ep = res["epoch"]
        run_tag = res["run_name"]
        phase2_summary[tag] = {}
        for proc in PROCESSES:
            pd_p = res["process_data"][proc]
            dJ = pd_p["delta_J"]
            w_star = pd_p["w_star"]
            g_fit = pd_p["g_fit_local"]
            w_learn = pd_p["learned_w"]
            fit_imp = pd_p["fit_imp_max"]

            valid = np.isfinite(dJ) & np.isfinite(w_star)
            dJ_v = dJ[valid]
            w_star_v = w_star[valid]
            g_fit_v = g_fit[valid]
            fit_imp_v = fit_imp[valid]

            n_total = len(dJ_v)
            orc_pos = (w_star_v > 0)
            dJ_pos = (dJ_v > 0)

            n_orc_pos = int(np.sum(orc_pos))
            n_dJ_pos = int(np.sum(dJ_pos))

            # Agreement metrics
            tp = np.sum(dJ_pos & orc_pos)
            fp = np.sum(dJ_pos & ~orc_pos)
            fn = np.sum(~dJ_pos & orc_pos)
            tn = np.sum(~dJ_pos & ~orc_pos)

            precision = float(tp / (tp + fp + EPS))
            recall = float(tp / (tp + fn + EPS))
            fpr = float(fp / (fp + tn + EPS))

            sp_corr_dJ_wstar, _ = spearmanr(dJ_v, w_star_v)
            sp_corr_dJ_fitimp, _ = spearmanr(dJ_v, fit_imp_v)

            # Interior optimum misspecification
            interior_pos = (w_star_v > 0) & (w_star_v < 1.0)
            n_interior = int(np.sum(interior_pos))
            n_interior_dJ_neg = int(np.sum(interior_pos & (dJ_v <= 0)))
            n_orc_pos_on_worse_off = int(np.sum(pd_p["on_worse_than_off_while_opt"][valid]))

            # Comparison with local gradient
            # Local gradient dL_fit/dw < 0 means increasing w decreases loss (positive for ON)
            # Signal for ON is -g_fit
            g_sig = -g_fit_v
            sign_agreement = float(np.mean((dJ_v > 0) == (g_sig > 0)))

            roc_auc_dJ = float(roc_auc_score(orc_pos, dJ_v)) if n_orc_pos > 0 else 0.5
            pr_auc_dJ = float(average_precision_score(orc_pos, dJ_v)) if n_orc_pos > 0 else 0.0
            roc_auc_g = float(roc_auc_score(orc_pos, g_sig)) if n_orc_pos > 0 else 0.5
            pr_auc_g = float(average_precision_score(orc_pos, g_sig)) if n_orc_pos > 0 else 0.0

            # Separation ratio: median |signal| in positive vs zero
            med_dJ_pos = float(np.median(dJ_v[orc_pos])) if n_orc_pos > 0 else 0.0
            med_dJ_zero = float(np.median(dJ_v[~orc_pos])) if (n_total - n_orc_pos) > 0 else 0.0
            med_g_pos = float(np.median(np.abs(g_fit_v[orc_pos]))) if n_orc_pos > 0 else 0.0
            med_g_zero = float(np.median(np.abs(g_fit_v[~orc_pos]))) if (n_total - n_orc_pos) > 0 else 0.0

            p2_row = {
                "checkpoint": tag,
                "process": proc,
                "n_total": n_total,
                "frac_oracle_pos": float(n_orc_pos / n_total),
                "frac_delta_J_pos": float(n_dJ_pos / n_total),
                "precision": precision,
                "recall": recall,
                "fpr": fpr,
                "spearman_dJ_wstar": float(sp_corr_dJ_wstar),
                "spearman_dJ_fitimp": float(sp_corr_dJ_fitimp),
                "n_interior_optima": n_interior,
                "n_interior_dJ_le0": n_interior_dJ_neg,
                "frac_interior_misspec": float(n_interior_dJ_neg / (n_interior + EPS)),
                "n_orc_pos_on_worse_than_off": n_orc_pos_on_worse_off,
                "sign_agreement_dJ_gfit": sign_agreement,
                "roc_auc_dJ_for_oracle": roc_auc_dJ,
                "pr_auc_dJ_for_oracle": pr_auc_dJ,
                "roc_auc_gfit_for_oracle": roc_auc_g,
                "pr_auc_gfit_for_oracle": pr_auc_g,
                "med_dJ_pos": med_dJ_pos,
                "med_dJ_zero": med_dJ_zero,
                "med_g_pos": med_g_pos,
                "med_g_zero": med_g_zero,
            }
            p2_table_rows.append(p2_row)
            phase2_summary[tag][proc] = p2_row

    df_p2 = pd.DataFrame(p2_table_rows)
    p2_csv = OUT_DIR / "phase2_oracle_and_gradient_agreement.csv"
    df_p2.to_csv(p2_csv, index=False)

    # 2.C Temporal Stability across R8 trajectory (ep1 -> ep2 -> ep3 -> ep4 -> ep10)
    print("\n[Phase 2.C] Computing temporal stability across R8 trajectory...")
    r8_epochs = [1, 2, 3, 4, 10]
    stability_summary = {}
    for proc in PROCESSES:
        dJ_by_ep = [all_data[f"R8_AICDelay_ep{e}"]["process_data"][proc]["delta_J"] for e in r8_epochs]
        B_n = len(dJ_by_ep[0])

        # Adjacent sign flip rate & rank correlation
        adjacent_flips = []
        adjacent_spearman = []
        adjacent_jaccard = []
        for i in range(len(r8_epochs) - 1):
            s1 = (dJ_by_ep[i] > 0)
            s2 = (dJ_by_ep[i + 1] > 0)
            flips = float(np.mean(s1 != s2))
            sp, _ = spearmanr(dJ_by_ep[i], dJ_by_ep[i + 1])
            jacc = float(np.sum(s1 & s2) / (np.sum(s1 | s2) + EPS))
            adjacent_flips.append(flips)
            adjacent_spearman.append(float(sp))
            adjacent_jaccard.append(jacc)

        # Basins consistently ON / OFF / unstable across all 5 epochs
        all_on = np.all(np.array([d > 0 for d in dJ_by_ep]), axis=0)
        all_off = np.all(np.array([d <= 0 for d in dJ_by_ep]), axis=0)
        unstable = ~(all_on | all_off)

        margins = np.abs(np.array(dJ_by_ep))

        stability_summary[proc] = {
            "r8_epochs": r8_epochs,
            "adjacent_flip_rates": adjacent_flips,
            "mean_adjacent_flip_rate": float(np.mean(adjacent_flips)),
            "adjacent_spearman_corrs": adjacent_spearman,
            "mean_adjacent_spearman": float(np.mean(adjacent_spearman)),
            "adjacent_jaccard_overlaps": adjacent_jaccard,
            "mean_adjacent_jaccard": float(np.mean(adjacent_jaccard)),
            "frac_consistently_on": float(np.mean(all_on)),
            "frac_consistently_off": float(np.mean(all_off)),
            "frac_unstable": float(np.mean(unstable)),
            "margin_median": float(np.nanmedian(margins)),
            "margin_iqr": float(np.nanpercentile(margins, 75) - np.nanpercentile(margins, 25)),
        }

    stab_json = OUT_DIR / "phase2_temporal_stability.json"
    stab_json.write_text(json.dumps(stability_summary, indent=2))
    print(f"[Phase 2 Complete] Saved Oracle/gradient agreement to {p2_csv} and stability to {stab_json}")

    # =========================================================================
    # Phase 3: Predictability of DeltaJ from Attributes & Representations
    # =========================================================================
    print("\n[Phase 3] Running 5-fold cross-validation linear probes on DeltaJ...")
    probe_results = []
    # Checkpoints to probe: Baseline ep0 (untrained), Baseline ep10, R8 ep2 (pre-collapse), R8 ep10 (post-collapse), R10B ep4
    probe_ckpts = ["Baseline_ep0", "Baseline_ep10", "R8_AICDelay_ep2", "R8_AICDelay_ep10", "R10B_Reweight_ep4"]

    for tag in probe_ckpts:
        if tag not in all_data:
            continue
        res = all_data[tag]
        attrs_X = res["attrs"]     # [671, 35]
        h_repr = res["h_repr"]      # [671, 128]

        for proc in PROCESSES:
            pd_p = res["process_data"][proc]
            dJ = pd_p["delta_J"]
            w_star = pd_p["w_star"]
            dJ_binary = (dJ > 0).astype(int)
            w_star_binary = (w_star > 0).astype(int)

            # 1. Raw 35D -> DeltaJ (binary & continuous)
            p_raw_dJ = run_probes(attrs_X, dJ_binary, dJ)
            p_raw_wstar = run_probes(attrs_X, w_star_binary, w_star)

            # 2. Frozen 128D h -> DeltaJ (binary & continuous)
            p_h_dJ = run_probes(h_repr, dJ_binary, dJ)
            p_h_wstar = run_probes(h_repr, w_star_binary, w_star)

            probe_results.append({
                "checkpoint": tag,
                "process": proc,
                # Raw X predicting DeltaJ
                "raw_X_dJ_roc_auc": p_raw_dJ.get("roc_auc", np.nan),
                "raw_X_dJ_pr_auc": p_raw_dJ.get("pr_auc", np.nan),
                "raw_X_dJ_top_k_recall": p_raw_dJ.get("top_k_recall", np.nan),
                "raw_X_dJ_pearson_r": p_raw_dJ.get("continuous_pearson_r", np.nan),
                "raw_X_dJ_r2": p_raw_dJ.get("continuous_r2", np.nan),
                # Learned h predicting DeltaJ
                "h_dJ_roc_auc": p_h_dJ.get("roc_auc", np.nan),
                "h_dJ_pr_auc": p_h_dJ.get("pr_auc", np.nan),
                "h_dJ_top_k_recall": p_h_dJ.get("top_k_recall", np.nan),
                "h_dJ_pearson_r": p_h_dJ.get("continuous_pearson_r", np.nan),
                "h_dJ_r2": p_h_dJ.get("continuous_r2", np.nan),
                # Reference: Raw X and h predicting Oracle w*
                "raw_X_wstar_roc_auc": p_raw_wstar.get("roc_auc", np.nan),
                "raw_X_wstar_pr_auc": p_raw_wstar.get("pr_auc", np.nan),
                "h_wstar_roc_auc": p_h_wstar.get("roc_auc", np.nan),
                "h_wstar_pr_auc": p_h_wstar.get("pr_auc", np.nan),
                "n_pos_dJ": p_raw_dJ.get("n_pos", 0),
                "n_pos_wstar": p_raw_wstar.get("n_pos", 0),
            })

    df_p3 = pd.DataFrame(probe_results)
    p3_csv = OUT_DIR / "phase3_predictability_probes.csv"
    df_p3.to_csv(p3_csv, index=False)
    print(f"[Phase 3 Complete] Saved predictability probe results to {p3_csv}")

    # =========================================================================
    # Phase 4: Parameter-State Contamination & Controlled Swapping
    # =========================================================================
    print("\n[Phase 4] Checking parameter-state contamination & swapping...")
    # R8 ep2 vs ep10 controlled parameter swap:
    # Evaluate forced ON/OFF for R8 ep2 model using ep10 physical parameters, and vice versa!
    res_ep2 = all_data["R8_AICDelay_ep2"]
    res_ep10 = all_data["R8_AICDelay_ep10"]

    # Controlled Parameter Swap Experiment:
    # 1. State A: Ep2 Backbone/Gates + Ep2 Params (Normal ep2)
    # 2. State B: Ep2 Backbone/Gates + Ep10 Params (Ep2 gate evaluated on Ep10 physical state)
    # 3. State C: Ep10 Backbone/Gates + Ep2 Params (Ep10 gate evaluated on Ep2 physical state)
    # 4. State D: Ep10 Backbone/Gates + Ep10 Params (Normal ep10)

    # Let's perform the swap forward evaluations on CUDA
    dev = "cuda:0"
    td, ed = dl.train_dataset, dl.eval_dataset
    B = td["x_phy"].shape[1]
    n_out = int(ed["x_phy"].shape[0]) - 365
    y_ev = ed["target"][365:365 + n_out, :, 0].cpu().numpy()
    n_valid_b = np.sum(~np.isnan(y_ev), axis=0).astype(float)
    N = float(n_valid_b.sum())
    std_train = (np.nanstd(td["target"][:, :, 0].cpu().numpy(), axis=0) + 0.1).astype(np.float32)

    # Load handler for ep2
    c_r8 = load_config("conf/config_dmopex_interceptE_S0_aicdelay2.yaml")
    c_r8["mode"] = "train"
    c_r8["model"]["phy"]["disable_compile"] = True
    handler_r8 = build_handler(c_r8)
    handler_r8.load_model(2)
    m_ep2 = next(iter(handler_r8.model_dict.values()))
    phy = m_ep2.phy_model

    # Descaled param dicts
    # Convert numpy arrays back to tensors for forward pass
    def dict_to_tensor(d_np):
        return {k: torch.from_numpy(v).float().unsqueeze(-1).repeat(1, 16).to(dev) for k, v in d_np.items()}

    def routing_to_tensor(d_np):
        return {k: torch.from_numpy(v).float().to(dev) for k, v in d_np.items()}

    params_ep2_t = dict_to_tensor(res_ep2["mopex_params"])
    params_ep10_t = dict_to_tensor(res_ep10["mopex_params"])
    rout_ep2_t = routing_to_tensor(res_ep2["routing_params"])
    rout_ep10_t = routing_to_tensor(res_ep10["routing_params"])

    # Compute forced ON and OFF for w_int under Ep2 params vs Ep10 params
    def eval_forced_int(params_t, rout_t, base_gates):
        S = 2  # OFF=0, ON=1
        w_on = base_gates.repeat(S, 1)
        w_on[:B, GATE_IDX["w_int"]] = 0.0
        w_on[B:, GATE_IDX["w_int"]] = 1.0

        p_rep = {k: v.repeat(S, 1) for k, v in params_t.items()}
        r_rep = {k: v.repeat(S) for k, v in rout_t.items()}
        sample_rep = {"x_phy": ed["x_phy"].repeat(1, S, 1).to(dev),
                      "doy": ed["doy"].repeat(1, S, 1).to(dev)}

        with torch.no_grad():
            P, T, PET, doy, n_steps, _ = phy._prepare_forcings(sample_rep)
            Q = phy._run_weighted_loop(P, T, PET, doy, p_rep, w_on, n_steps, B * S)
            Qr = phy._apply_routing(Q.mean(-1), r_rep).cpu().numpy()[:, :, 0]

        Qr = Qr[:, :B * S].reshape(Qr.shape[0], S, B)
        Qs = np.transpose(Qr, (0, 2, 1))[:n_out]  # [n_out, B, 2]

        fit_off = np.full(B, np.nan)
        fit_on = np.full(B, np.nan)
        for b in range(B):
            v = ~np.isnan(y_ev[:, b])
            if v.sum() < 30:
                continue
            o = y_ev[v, b]
            fit_off[b] = np.mean((Qs[v, b, 0] - o) ** 2 / (std_train[b] ** 2))
            fit_on[b] = np.mean((Qs[v, b, 1] - o) ** 2 / (std_train[b] ** 2))

        aic_pen = AIC_ALPHA * COSTS["w_int"] * (N / (B * np.maximum(n_valid_b, 1.0)))
        dJ = fit_off - fit_on - aic_pen
        return dJ, fit_off, fit_on

    base_gates_ep2 = torch.from_numpy(res_ep2["process_data"]["w_int"]["learned_w"]).float().unsqueeze(-1).repeat(1, 4).to(dev)
    dJ_ep2params, _, _ = eval_forced_int(params_ep2_t, rout_ep2_t, base_gates_ep2)
    dJ_ep10params, _, _ = eval_forced_int(params_ep10_t, rout_ep10_t, base_gates_ep2)

    # Compare dJ under Ep2 params vs Ep10 params
    valid_swap = np.isfinite(dJ_ep2params) & np.isfinite(dJ_ep10params)
    sp_swap, _ = spearmanr(dJ_ep2params[valid_swap], dJ_ep10params[valid_swap])
    pe_swap, _ = pearsonr(dJ_ep2params[valid_swap], dJ_ep10params[valid_swap])
    sign_match_swap = float(np.mean((dJ_ep2params[valid_swap] > 0) == (dJ_ep10params[valid_swap] > 0)))

    # Track fixed cohort of 103 basins
    cohort_103 = np.where(res_ep2["process_data"]["w_int"]["oracle_pos"])[0]
    cohort_dJ_ep2 = dJ_ep2params[cohort_103]
    cohort_dJ_ep10 = dJ_ep10params[cohort_103]

    param_swap_summary = {
        "spearman_corr_ep2_vs_ep10_params": float(sp_swap),
        "pearson_corr_ep2_vs_ep10_params": float(pe_swap),
        "sign_match_rate_all_basins": sign_match_swap,
        "cohort_103_mean_dJ_under_ep2_params": float(np.nanmean(cohort_dJ_ep2)),
        "cohort_103_mean_dJ_under_ep10_params": float(np.nanmean(cohort_dJ_ep10)),
        "cohort_103_frac_dJ_gt0_under_ep2_params": float(np.nanmean(cohort_dJ_ep2 > 0)),
        "cohort_103_frac_dJ_gt0_under_ep10_params": float(np.nanmean(cohort_dJ_ep10 > 0)),
        "cohort_103_sign_retention_rate": float(np.nanmean((cohort_dJ_ep2 > 0) == (cohort_dJ_ep10 > 0))),
    }

    p4_json = OUT_DIR / "phase4_parameter_state_swap.json"
    p4_json.write_text(json.dumps(param_swap_summary, indent=2))
    print(f"[Phase 4 Complete] Saved parameter swap results to {p4_json}")

    # =========================================================================
    # Phase 5: Soft-Target Formulations Evaluation
    # =========================================================================
    print("\n[Phase 5] Evaluating candidate soft-target formulations (A, B, C)...")
    # Evaluate on R8 ep2 and Baseline ep10
    soft_target_evals = []

    for tag in ["R8_AICDelay_ep2", "R8_AICDelay_ep10", "Baseline_ep10", "R10B_Reweight_ep4"]:
        if tag not in all_data:
            continue
        res = all_data[tag]
        for proc in PROCESSES:
            pd_p = res["process_data"][proc]
            dJ = pd_p["delta_J"]
            w_star = pd_p["w_star"]
            orc_pos = (w_star > 0)
            valid = np.isfinite(dJ) & np.isfinite(w_star)

            dJ_v = dJ[valid]
            orc_v = orc_pos[valid]

            # Candidate A: Binary detached target q = 1[DeltaJ > 0]
            q_A = (dJ_v > 0).astype(float)
            entropy_A = float(-np.mean(q_A * np.log(np.maximum(q_A, 1e-12)) + (1 - q_A) * np.log(np.maximum(1 - q_A, 1e-12))))
            class_balance_A = float(np.mean(q_A))

            # Candidate B: Margin-aware confidence target
            # q_B = 0.5 + 0.5 * sign(dJ) * min(1.0, |dJ| / tau), where tau = 75th percentile of |dJ|
            tau_B = float(np.nanpercentile(np.abs(dJ_v[dJ_v != 0]), 75)) if np.sum(dJ_v != 0) > 0 else 0.01
            margin_ratio = np.clip(np.abs(dJ_v) / (tau_B + EPS), 0.0, 1.0)
            q_B = 0.5 + 0.5 * np.sign(dJ_v) * margin_ratio
            entropy_B = float(-np.mean(q_B * np.log(np.maximum(q_B, 1e-12)) + (1 - q_B) * np.log(np.maximum(1 - q_B, 1e-12))))
            class_balance_B = float(np.mean(q_B > 0.5))

            # Candidate C: Logistic soft target q_C = sigmoid(dJ / T) where T = median(|dJ|)
            T_C = float(np.nanmedian(np.abs(dJ_v[dJ_v != 0]))) if np.sum(dJ_v != 0) > 0 else 0.01
            q_C = 1.0 / (1.0 + np.exp(-np.clip(dJ_v / (T_C + EPS), -20, 20)))
            entropy_C = float(-np.mean(q_C * np.log(np.maximum(q_C, 1e-12)) + (1 - q_C) * np.log(np.maximum(1 - q_C, 1e-12))))
            class_balance_C = float(np.mean(q_C > 0.5))

            # Agreement with exact Oracle
            roc_A = float(roc_auc_score(orc_v, q_A)) if np.sum(orc_v) > 0 else 0.5
            roc_B = float(roc_auc_score(orc_v, q_B)) if np.sum(orc_v) > 0 else 0.5
            roc_C = float(roc_auc_score(orc_v, q_C)) if np.sum(orc_v) > 0 else 0.5

            pr_A = float(average_precision_score(orc_v, q_A)) if np.sum(orc_v) > 0 else 0.0
            pr_B = float(average_precision_score(orc_v, q_B)) if np.sum(orc_v) > 0 else 0.0
            pr_C = float(average_precision_score(orc_v, q_C)) if np.sum(orc_v) > 0 else 0.0

            soft_target_evals.append({
                "checkpoint": tag,
                "process": proc,
                "tau_B_scale": tau_B,
                "T_C_temperature": T_C,
                # Candidate A
                "A_class_balance_frac_on": class_balance_A,
                "A_entropy": entropy_A,
                "A_oracle_roc_auc": roc_A,
                "A_oracle_pr_auc": pr_A,
                # Candidate B
                "B_class_balance_frac_on": class_balance_B,
                "B_entropy": entropy_B,
                "B_oracle_roc_auc": roc_B,
                "B_oracle_pr_auc": pr_B,
                # Candidate C
                "C_class_balance_frac_on": class_balance_C,
                "C_entropy": entropy_C,
                "C_oracle_roc_auc": roc_C,
                "C_oracle_pr_auc": pr_C,
            })

    df_p5 = pd.DataFrame(soft_target_evals)
    p5_csv = OUT_DIR / "phase5_soft_target_formulations.csv"
    df_p5.to_csv(p5_csv, index=False)
    print(f"[Phase 5 Complete] Saved soft target evaluation to {p5_csv}")

    # =========================================================================
    # Phase 6: Compute / Memory Cost Benchmarking
    # =========================================================================
    print("\n[Phase 6] Benchmarking real training step vs counterfactual generation cost...")
    # Benchmark on CUDA
    sample_batch_size = 100
    n_trials = 10

    # 1. Standard training step (Forward + Backward on 100 basins)
    m_bench = next(iter(handler_r8.model_dict.values()))
    m_bench.train()
    optimizer = torch.optim.Adadelta(m_bench.parameters(), lr=1.0)

    sample_b100 = {
        "x_phy": ed["x_phy"][:, :sample_batch_size].to(dev),
        "doy": ed["doy"][:, :sample_batch_size].to(dev),
        "c_nn_norm": td["xc_nn_norm"][0, :sample_batch_size, -n_attr:].to(dev),
        "target": ed["target"][:, :sample_batch_size].to(dev),
    }

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_trials):
        optimizer.zero_grad()
        p, logits, w_on, m_p, r_p = build_forward(phy, m_bench.nn_model, sample_b100)
        out = run_loop(phy, sample_b100, w_on, m_p, r_p)
        q = out["streamflow"]
        obs = sample_b100["target"][365:365 + n_out]
        loss = per_basin_fit(q, obs, std_t[:sample_batch_size]).mean()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    t_std_step = (time.time() - t0) / n_trials

    # 2. Counterfactual evaluation: 1 process ON/OFF (S=2) on 100 basins
    m_bench.eval()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_trials):
        with torch.no_grad():
            p_raw = m_bench.nn_model({"c_nn_norm": sample_b100["c_nn_norm"]})
            w_learn = F.softmax(p_raw["weights"].view(sample_batch_size, 4, 2).clamp(-10, 10), dim=-1)[..., 1]
            m_p = phy._descale_mopex_params(p_raw["params"])
            r_p = phy._descale_routing_params(p_raw["gamma_uh"])

            S = 2
            w_cf = w_learn.repeat(S, 1)
            w_cf[:sample_batch_size, 1] = 0.0
            w_cf[sample_batch_size:, 1] = 1.0

            p_rep = {k: v.repeat(S, 1) for k, v in m_p.items()}
            r_rep = {k: v.repeat(S) for k, v in r_p.items()}
            s_rep = {"x_phy": sample_b100["x_phy"].repeat(1, S, 1), "doy": sample_b100["doy"].repeat(1, S, 1)}

            P, T, PET, doy, n_steps, _ = phy._prepare_forcings(s_rep)
            Q = phy._run_weighted_loop(P, T, PET, doy, p_rep, w_cf, n_steps, sample_batch_size * S)
            Qr = phy._apply_routing(Q.mean(-1), r_rep)
    torch.cuda.synchronize()
    t_1proc_cf = (time.time() - t0) / n_trials

    # 3. Counterfactual evaluation: all 4 processes (S=8: ON/OFF per process) vectorized on 100 basins
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_trials):
        with torch.no_grad():
            p_raw = m_bench.nn_model({"c_nn_norm": sample_b100["c_nn_norm"]})
            w_learn = F.softmax(p_raw["weights"].view(sample_batch_size, 4, 2).clamp(-10, 10), dim=-1)[..., 1]
            m_p = phy._descale_mopex_params(p_raw["params"])
            r_p = phy._descale_routing_params(p_raw["gamma_uh"])

            S = 8  # 4 processes * 2 endpoints
            w_cf = w_learn.repeat(S, 1)
            for p_idx in range(4):
                w_cf[(2 * p_idx) * sample_batch_size:(2 * p_idx + 1) * sample_batch_size, p_idx] = 0.0
                w_cf[(2 * p_idx + 1) * sample_batch_size:(2 * p_idx + 2) * sample_batch_size, p_idx] = 1.0

            p_rep = {k: v.repeat(S, 1) for k, v in m_p.items()}
            r_rep = {k: v.repeat(S) for k, v in r_p.items()}
            s_rep = {"x_phy": sample_b100["x_phy"].repeat(1, S, 1), "doy": sample_b100["doy"].repeat(1, S, 1)}

            P, T, PET, doy, n_steps, _ = phy._prepare_forcings(s_rep)
            Q = phy._run_weighted_loop(P, T, PET, doy, p_rep, w_cf, n_steps, sample_batch_size * S)
            Qr = phy._apply_routing(Q.mean(-1), r_rep)
    torch.cuda.synchronize()
    t_4proc_cf = (time.time() - t0) / n_trials

    # Memory usage
    mem_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)

    cost_benchmark = {
        "batch_size": sample_batch_size,
        "n_valid_days": n_out,
        "time_std_training_step_sec": float(t_std_step),
        "time_1proc_cf_sec": float(t_1proc_cf),
        "time_4proc_cf_vectorized_sec": float(t_4proc_cf),
        "ratio_1proc_cf_over_std_step": float(t_1proc_cf / t_std_step),
        "ratio_4proc_cf_over_std_step": float(t_4proc_cf / t_std_step),
        "gpu_max_mem_mb": float(mem_allocated),
        "cost_strategies": {
            "every_batch": {
                "additional_cost_per_epoch_sec": float(t_4proc_cf * 7),
                "epoch_overhead_pct": float(t_4proc_cf / t_std_step * 100),
            },
            "every_epoch_epoch_start": {
                "cost_per_epoch_sec": float(t_4proc_cf * (671 / 100)),
                "epoch_overhead_pct": float((t_4proc_cf * 6.71) / (t_std_step * 7) * 100),
            },
            "every_2_epochs": {
                "cost_per_epoch_sec": float(t_4proc_cf * 6.71 / 2),
                "epoch_overhead_pct": float((t_4proc_cf * 6.71 / 2) / (t_std_step * 7) * 100),
            }
        }
    }

    p6_json = OUT_DIR / "phase6_compute_memory_cost.json"
    p6_json.write_text(json.dumps(cost_benchmark, indent=2))
    print(f"[Phase 6 Complete] Saved compute benchmark to {p6_json}")

    print("\n" + "=" * 80)
    print("ALL R14 FEASIBILITY DIAGNOSTICS COMPLETED SUCCESSFULLY")
    print(f"Artifacts saved in {OUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
