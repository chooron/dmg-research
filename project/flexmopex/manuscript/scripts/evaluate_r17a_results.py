#!/usr/bin/env python3
"""Final Evaluation and 5-Way Benchmark for R17-A Dual-Optimizer Structural Supervision.

Evaluates R17-A across 5114-day window (671 basins, routed rows 0..5113 vs target[365:365+5114]):
  1. Full-period NSE and KGE metrics across ep1..ep10
  2. Four-process exact continuous Oracle grid sweep (w* in {0.0, 0.1, 0.25, 0.5, 0.75, 1.0}) at Ep 10
  3. Precision, Recall, FPR, FNR, and Spearman correlation with Oracle w*
  4. Comparison across Baseline E-S0, R8, R10-B, R15-A, and R17-A
  5. Weight norm and prediction distribution analysis

Outputs saved to:
  results/intercept_r17a/E_S0_r17a/eval_summary.json
  results/intercept_r17a/E_S0_r17a/process_oracle_table_ep10.csv
  results/intercept_r17a/E_S0_r17a/benchmark_comparison.csv
  results/intercept_r17a/E_S0_r17a/epoch_trajectory.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config
from project.flexmopex.run_model import apply_runtime_overrides, parse_args, _build_data_loader
from scripts.diagnose_wint_collapse import build_handler, build_forward, run_loop

OUT_ROOT = Path("results/intercept_r17a/E_S0_r17a")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

PROCESSES = ["w_phen", "w_int", "w_snow", "w_sub"]
GATE_IDX = {"w_phen": 0, "w_int": 1, "w_snow": 2, "w_sub": 3}
COSTS = {"w_phen": 2.0, "w_int": 2.0, "w_snow": 2.0, "w_sub": 1.0}
AIC_ALPHA = 0.01
W_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
EPS = 1e-6


def main():
    dev = "cuda:0"
    cfg_path = "conf/config_dmopex_interceptE_S0_r17a.yaml"
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
    n_out = int(ed["x_phy"].shape[0]) - 365
    y_ev = ed["target"][365:365 + n_out, :, 0].cpu().numpy()
    n_valid_b = np.sum(~np.isnan(y_ev), axis=0).astype(float)
    N = float(n_valid_b.sum())
    std_train = (np.nanstd(td["target"][:, :, 0].cpu().numpy(), axis=0) + 0.1).astype(np.float32)

    handler = build_handler(c)

    # 1. Evaluate Trajectory across epochs 1..10
    print("[1/4] Evaluating epoch trajectory (ep1..ep10)...")
    traj_rows = []
    for ep in range(1, 11):
        handler.load_model(ep)
        for m in handler.model_dict.values():
            m.eval()
        model = next(iter(handler.model_dict.values()))
        phy, nn = model.phy_model, model.nn_model

        with torch.no_grad():
            params_raw = nn({"c_nn_norm": attrs})
            w_learn = F.softmax(params_raw["weights"].view(B, 4, 2).clamp(-10, 10), dim=-1)[..., 1]
            mopex_params = phy._descale_mopex_params(params_raw["params"])
            routing = phy._descale_routing_params(params_raw["gamma_uh"])

            sample = {"x_phy": ed["x_phy"].to(dev), "doy": ed["doy"].to(dev), "c_nn_norm": attrs}
            p, logits, w_on, m_p, r_p = build_forward(phy, nn, sample)
            out = run_loop(phy, sample, w_on, m_p, r_p)
            q_stream = out["streamflow"][:n_out, :, 0].cpu().numpy()

            w_norm = float(torch.norm(nn.heads["weights"].weight).item())
            b_norm = float(torch.norm(nn.heads["weights"].bias).item())

        nses = []
        for b in range(B):
            v = ~np.isnan(y_ev[:, b])
            if v.sum() < 30: continue
            o = y_ev[v, b]
            s = q_stream[v, b]
            ss_res = np.sum((s - o)**2)
            ss_tot = np.sum((o - o.mean())**2)
            nses.append(1.0 - ss_res / ss_tot if ss_tot > EPS else np.nan)
        nses = np.array(nses)

        w_np = w_learn.cpu().numpy()
        t_row = {
            "epoch": ep,
            "median_nse": float(np.nanmedian(nses)),
            "mean_nse": float(np.nanmean(nses)),
            "frac_nse_gt0": float(np.nanmean(nses > 0)),
            "frac_nse_gt05": float(np.nanmean(nses > 0.5)),
            "weights_head_w_norm": w_norm,
            "weights_head_b_norm": b_norm,
        }
        for proc in PROCESSES:
            col = GATE_IDX[proc]
            pw = w_np[:, col]
            t_row[f"{proc}_mean"] = float(np.mean(pw))
            t_row[f"{proc}_std"] = float(np.std(pw))
            t_row[f"{proc}_median"] = float(np.median(pw))
            t_row[f"{proc}_min"] = float(np.min(pw))
            t_row[f"{proc}_max"] = float(np.max(pw))
            t_row[f"{proc}_frac_gt001"] = float(np.mean(pw > 0.01))
            t_row[f"{proc}_frac_gt01"] = float(np.mean(pw > 0.1))
            t_row[f"{proc}_frac_gt05"] = float(np.mean(pw > 0.5))
        traj_rows.append(t_row)
        print(f"  Ep {ep:>2d}: median_NSE={t_row['median_nse']:.4f}, mean_NSE={t_row['mean_nse']:.4f} | ||W||={w_norm:.3f} | "
              f"w_int: mean={t_row['w_int_mean']:.4f}, std={t_row['w_int_std']:.4f} (range [{t_row['w_int_min']:.3f}, {t_row['w_int_max']:.3f}])")

    df_traj = pd.DataFrame(traj_rows)
    df_traj.to_csv(OUT_ROOT / "epoch_trajectory.csv", index=False)

    # 2. Four-Process Exact Continuous Oracle Evaluation at Ep 10
    print("\n[2/4] Running 4-process exact continuous Oracle grid sweep at Epoch 10...")
    handler.load_model(10)
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

    S = len(W_GRID)
    process_eval_summary = {}
    oracle_table_rows = []

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
        kge_grid = np.full((B, S), np.nan)

        for b in range(B):
            v = ~np.isnan(y_ev[:, b])
            if v.sum() < 30: continue
            o = y_ev[v, b]
            ss = Qs[v, b, :]
            fit_grid[b, :] = np.mean((ss - o[:, None])**2 / (std_train[b]**2), axis=0)
            ss_tot = np.sum((o - o.mean())**2)
            for s in range(S):
                ss_res = np.sum((ss[:, s] - o)**2)
                nse_grid[b, s] = 1.0 - ss_res / (ss_tot + EPS) if ss_tot > EPS else np.nan
                r = np.corrcoef(ss[:, s], o)[0, 1] if np.std(ss[:, s]) > 0 and np.std(o) > 0 else 0.0
                alpha = np.std(ss[:, s]) / (np.std(o) + EPS)
                beta = np.mean(ss[:, s]) / (np.mean(o) + EPS)
                kge_grid[b, s] = 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

        # Exact Oracle w*
        cost = COSTS[proc]
        aic_unit = AIC_ALPHA * cost / B
        w_star = np.full(B, np.nan)
        for b in range(B):
            vals = [fit_grid[b, s] * n_valid_b[b] / N + aic_unit * W_GRID[s] for s in range(S)]
            if np.isfinite(vals).any():
                w_star[b] = W_GRID[int(np.nanargmin(vals))]

        w_l = base_w[:, col].cpu().numpy()
        dNSE_opt = np.nanmax(nse_grid[:, 1:], axis=1) - nse_grid[:, 0]
        fit_imp_opt = fit_grid[:, 0] - np.nanmin(fit_grid[:, 1:], axis=1)

        # Metrics
        valid = np.isfinite(w_star)
        orc_pos = (w_star[valid] > 0)
        learned_pos = (w_l[valid] > 0.01)
        learned_high = (w_l[valid] > 0.1)

        n_orc_pos = int(np.sum(orc_pos))
        n_orc_zero = int(np.sum(~orc_pos))
        n_learned_pos = int(np.sum(learned_pos))
        n_learned_high = int(np.sum(learned_high))

        tp = np.sum(learned_pos & orc_pos)
        fp = np.sum(learned_pos & ~orc_pos)
        fn = np.sum(~learned_pos & orc_pos)
        tn = np.sum(~learned_pos & ~orc_pos)

        tp_high = np.sum(learned_high & orc_pos)
        fp_high = np.sum(learned_high & ~orc_pos)

        precision = float(tp / (tp + fp + EPS))
        recall = float(tp / (tp + fn + EPS))
        fpr = float(fp / (fp + tn + EPS))
        fnr = float(fn / (fn + tp + EPS))
        precision_high = float(tp_high / (tp_high + fp_high + EPS))

        sp_corr, _ = spearmanr(w_l[valid], w_star[valid])
        pe_corr, _ = pearsonr(w_l[valid], w_star[valid])

        process_eval_summary[proc] = {
            "n_basins_valid": int(np.sum(valid)),
            "n_oracle_positive": n_orc_pos,
            "frac_oracle_positive": float(n_orc_pos / np.sum(valid)),
            "n_learned_active_gt001": n_learned_pos,
            "frac_learned_active_gt001": float(n_learned_pos / np.sum(valid)),
            "n_learned_high_gt01": n_learned_high,
            "frac_learned_high_gt01": float(n_learned_high / np.sum(valid)),
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
            "fnr": fnr,
            "precision_high_activation": precision_high,
            "tp_recovered": int(tp),
            "fp_activated": int(fp),
            "spearman_with_oracle_w": float(sp_corr),
            "pearson_with_oracle_w": float(pe_corr),
            "learned_w_mean": float(np.mean(w_l[valid])),
            "learned_w_std": float(np.std(w_l[valid])),
            "learned_w_median": float(np.median(w_l[valid])),
            "learned_w_min": float(np.min(w_l[valid])),
            "learned_w_max": float(np.max(w_l[valid])),
            "learned_w_p95": float(np.percentile(w_l[valid], 95)),
            "learned_w_oracle_pos_mean": float(np.mean(w_l[valid][orc_pos])) if n_orc_pos > 0 else 0.0,
            "learned_w_oracle_zero_mean": float(np.mean(w_l[valid][~orc_pos])) if n_orc_zero > 0 else 0.0,
        }

        print(f"\n[{proc}] Ep10 Oracle Evaluation (R17-A):")
        print(f"  Oracle Pos: {n_orc_pos} ({process_eval_summary[proc]['frac_oracle_positive']*100:.1f}%) | "
              f"Learned Active (>0.01): {n_learned_pos} | Learned High (>0.1): {n_learned_high}")
        print(f"  Recall: {recall*100:.1f}% ({tp}/{n_orc_pos}) | Precision: {precision*100:.1f}% | High-Act Precision: {precision_high*100:.1f}%")
        print(f"  FPR: {fpr*100:.1f}% | Spearman r: {sp_corr:+.4f}")
        print(f"  Learned w in Pos: {process_eval_summary[proc]['learned_w_oracle_pos_mean']:.4f} vs in Zero: {process_eval_summary[proc]['learned_w_oracle_zero_mean']:.4f} (Δ={process_eval_summary[proc]['learned_w_oracle_pos_mean']-process_eval_summary[proc]['learned_w_oracle_zero_mean']:+.4f})")

        for b in range(B):
            oracle_table_rows.append({
                "basin_idx": b,
                "process": proc,
                "learned_w": w_l[b],
                "w_star": w_star[b],
                "oracle_pos": bool(w_star[b] > 0),
                "dNSE_opt": dNSE_opt[b],
                "fit_imp_opt": fit_imp_opt[b],
            })

    df_orc = pd.DataFrame(oracle_table_rows)
    df_orc.to_csv(OUT_ROOT / "process_oracle_table_ep10.csv", index=False)

    # 3. 5-Way Benchmark Comparison Table
    print("\n[3/4] Compiling 5-Way Benchmark Comparison...")
    benchmarks = [
        ("Baseline (Canonical E-S0)", "results/intercept_candidates/E_S0/four_process/process_oracle_table.csv", 0.6317, 0.5544),
        ("R8 (AIC-Delay-2)", "results/intercept_aicdelay/E_S0_aicdelay2/four_process/process_oracle_table.csv", 0.6318, 0.5543),
        ("R10-B (Reweight + Delay-2)", "results/intercept_reweight/E_S0_reweight_delay2/four_process/process_oracle_table.csv", 0.6309, 0.5539),
        ("R15-A (CF Supervision Adadelta)", "results/intercept_cf_supervision/E_S0_cf_supervision/process_oracle_table_ep10.csv", 0.6400, 0.5604),
    ]

    bench_rows = []
    for name, fpath, med_nse, mn_nse in benchmarks:
        if Path(fpath).exists():
            df_b = pd.read_csv(fpath)
            if "epoch" in df_b.columns:
                df_b = df_b[df_b["epoch"] == 10]
            for proc in PROCESSES:
                sub = df_b[df_b["process"] == proc]
                wl = sub["w_learn"].values if "w_learn" in sub.columns else sub["learned_w"].values
                ws = sub["w_star"].values
                valid = np.isfinite(ws)
                orc_pos = (ws[valid] > 0)
                l_pos = (wl[valid] > 0.01)
                l_high = (wl[valid] > 0.1)

                tp = np.sum(l_pos & orc_pos)
                fp = np.sum(l_pos & ~orc_pos)
                fn = np.sum(~l_pos & orc_pos)
                tn = np.sum(~l_pos & ~orc_pos)
                tp_h = np.sum(l_high & orc_pos)
                fp_h = np.sum(l_high & ~orc_pos)

                rec = float(tp / (tp + fn + EPS))
                prec = float(tp / (tp + fp + EPS))
                prec_h = float(tp_h / (tp_h + fp_h + EPS))
                fpr = float(fp / (fp + tn + EPS))
                sp, _ = spearmanr(wl[valid], ws[valid])

                bench_rows.append({
                    "run": name,
                    "process": proc,
                    "median_nse": med_nse,
                    "mean_nse": mn_nse,
                    "oracle_pos_n": int(np.sum(orc_pos)),
                    "learned_active_n": int(np.sum(l_pos)),
                    "learned_high_n": int(np.sum(l_high)),
                    "recall": rec,
                    "precision": prec,
                    "precision_high": prec_h,
                    "fpr": fpr,
                    "spearman_r": float(sp),
                    "learned_w_mean": float(np.mean(wl[valid])),
                    "learned_w_std": float(np.std(wl[valid])),
                    "learned_w_pos_mean": float(np.mean(wl[valid][orc_pos])) if np.sum(orc_pos)>0 else 0.0,
                    "learned_w_zero_mean": float(np.mean(wl[valid][~orc_pos])) if np.sum(~orc_pos)>0 else 0.0,
                })

    # Add R17-A
    r17_ep10_traj = df_traj[df_traj["epoch"] == 10].iloc[0]
    for proc in PROCESSES:
        p_res = process_eval_summary[proc]
        bench_rows.append({
            "run": "R17-A (CF Supervision Dual-Optimizer)",
            "process": proc,
            "median_nse": r17_ep10_traj["median_nse"],
            "mean_nse": r17_ep10_traj["mean_nse"],
            "oracle_pos_n": p_res["n_oracle_positive"],
            "learned_active_n": p_res["n_learned_active_gt001"],
            "learned_high_n": p_res["n_learned_high_gt01"],
            "recall": p_res["recall"],
            "precision": p_res["precision"],
            "precision_high": p_res["precision_high_activation"],
            "fpr": p_res["fpr"],
            "spearman_r": p_res["spearman_with_oracle_w"],
            "learned_w_mean": p_res["learned_w_mean"],
            "learned_w_std": p_res["learned_w_std"],
            "learned_w_pos_mean": p_res["learned_w_oracle_pos_mean"],
            "learned_w_zero_mean": p_res["learned_w_oracle_zero_mean"],
        })

    df_bench = pd.DataFrame(bench_rows)
    df_bench.to_csv(OUT_ROOT / "benchmark_comparison.csv", index=False)

    full_summary = {
        "overall_evaluation": {
            "median_nse_ep10": float(r17_ep10_traj["median_nse"]),
            "mean_nse_ep10": float(r17_ep10_traj["mean_nse"]),
            "frac_nse_gt0": float(r17_ep10_traj["frac_nse_gt0"]),
            "frac_nse_gt05": float(r17_ep10_traj["frac_nse_gt05"]),
            "weights_head_w_norm_ep10": float(r17_ep10_traj["weights_head_w_norm"]),
        },
        "processes": process_eval_summary,
    }

    (OUT_ROOT / "eval_summary.json").write_text(json.dumps(full_summary, indent=2))
    print(f"\n[Evaluation Complete] Saved all artifacts to {OUT_ROOT}/")


if __name__ == "__main__":
    main()
