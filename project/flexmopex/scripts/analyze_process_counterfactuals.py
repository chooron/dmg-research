#!/usr/bin/env python3
"""Agent A — four-process oracle / fit-benefit comparison for the structural
weights (w_phen, w_int, w_snow, w_sub) at the standard Candidate E-S0 states.

For each process and epoch (10 primary; 0 for all four):
  * hold all other learned params/gates fixed, sweep ONLY that process weight
    over {0,.1,.25,.5,.75,1.0} on all 671 basins (canonical eval window,
    trainer alignment: routed rows 0..5113 vs target[365:365+5114]);
  * per (basin, w): predictive fit loss (training normalization),
    NSE, KGE;
  * exact total-objective oracle per process:
      w_p* = argmin_w [ fit_b(w)*n_valid_b/N + aic_alpha*cost_p*w/671 ]
    with the repository AIC costs {w_phen:2, w_int:2, w_snow:2, w_sub:1},
    aic_alpha = 0.01, N = sum_b n_valid_b (basin-separable; see prior round).

Outputs (results/intercept_candidates/E_S0/four_process/):
  process_grid_ep{epoch}_{process}.csv   per-(basin,w) fit/NSE/KGE
  process_oracle_table.csv               basin x process oracle + learned w + benefit
  process_summary.json                   matched four-process summary
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
from project.flexmopex.local_model_handler import FlexMopexModelHandler  # noqa: E402

PROCESSES = ["w_phen", "w_int", "w_snow", "w_sub"]
GATE_IDX = {"w_phen": 0, "w_int": 1, "w_snow": 2, "w_sub": 3}
COSTS = {"w_phen": 2.0, "w_int": 2.0, "w_snow": 2.0, "w_sub": 1.0}
AIC_ALPHA = 0.01
W_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
EPS = 1e-6


def q(v, ps=(0.05, 0.5, 0.9, 0.95)):
    return {f"p{int(p*100):02d}": float(np.nanquantile(v, p)) for p in ps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/config_dmopex_interceptE_S0.yaml")
    ap.add_argument("--output-root", default="results/intercept_candidates")
    ap.add_argument("--run-name", default="E_S0")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--epochs", default="10,0")
    args = ap.parse_args()
    root = Path(args.output_root)
    arm_dir = root / args.run_name
    out_dir = arm_dir / "four_process"
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(e) for e in args.epochs.split(",")]

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
    dev = cfg["device"]
    B = td["x_phy"].shape[1]
    n_attr = td["xc_nn_norm"].shape[-1] - 3
    attrs = td["xc_nn_norm"][0, :, -n_attr:].to(dev)

    # scored-window convention (trainer alignment): n_out = record - 365 = 5114
    n_out = int(ed["x_phy"].shape[0]) - 365
    y_ev = ed["target"][365:365 + n_out, :, 0].cpu().numpy()   # (n_out, 671)
    n_valid_b = np.sum(~np.isnan(y_ev), axis=0).astype(float)
    N = float(n_valid_b.sum())
    y_obs = td["target"][:, :, 0].cpu().numpy()
    std_train = np.nanstd(y_obs, axis=0) + 0.1
    print(f"[A] n_out={n_out}, N={N:.0f}, epochs={epochs}")

    handler = FlexMopexModelHandler(cfg, verbose=False)
    oracle_rows = []
    summary = {}
    for epoch in epochs:
        try:
            handler.load_model(epoch)
        except FileNotFoundError:
            print(f"[warn] ep{epoch} missing; skip")
            continue
        for m in handler.model_dict.values():
            m.eval()
        model = next(iter(handler.model_dict.values()))
        phy, nn = model.phy_model, model.nn_model
        with torch.no_grad():
            p = nn({"c_nn_norm": attrs})
            w_learn = F.softmax(p["weights"].view(B, 4, 2).clamp(-10, 10), dim=-1)[..., 1]
            mopex_params = phy._descale_mopex_params(p["params"])
            routing = phy._descale_routing_params(p["gamma_uh"])
            base_w = w_learn.detach().clone()

        S = len(W_GRID)
        for proc in PROCESSES:
            col = GATE_IDX[proc]
            w_on = base_w.repeat(S, 1)
            for s in range(S):
                w_on[s * B:(s + 1) * B, col] = W_GRID[s]
            params_rep = {k: v.repeat(S, 1) for k, v in mopex_params.items()}
            routing_rep = {k: v.repeat(S) for k, v in routing.items()}
            sample = {"x_phy": ed["x_phy"].repeat(1, S, 1).to(dev),
                      "doy": ed["doy"].repeat(1, S, 1).to(dev),
                      "c_nn_norm": attrs.repeat(S, 1).to(dev)}
            with torch.no_grad():
                P, T, PET, doy, n_steps, _ = phy._prepare_forcings(sample)
                Q = phy._run_weighted_loop(P, T, PET, doy, params_rep, w_on, n_steps, B * S)
                Qr = phy._apply_routing(Q.mean(-1), routing_rep).cpu().numpy()[:, :, 0]
            Qr = Qr[:, :B * S].reshape(Qr.shape[0], S, B)
            Qs = np.transpose(Qr, (0, 2, 1))[:n_out]           # (n_out, B, S)
            nse = np.full((B, S), np.nan); kge = np.full((B, S), np.nan)
            fit = np.full((B, S), np.nan)
            for b in range(B):
                v = ~np.isnan(y_ev[:, b])
                if v.sum() < 30:
                    continue
                o = y_ev[v, b]
                for s in range(S):
                    ss = Qs[v, b, s]
                    ss_res = np.sum((ss - o) ** 2)
                    ss_tot = np.sum((o - o.mean()) ** 2)
                    nse[b, s] = 1.0 - ss_res / (ss_tot + EPS) if ss_tot > EPS else np.nan
                    r = np.corrcoef(ss, o)[0, 1] if np.std(ss) > 0 and np.std(o) > 0 else 0.0
                    alpha = np.std(ss) / (np.std(o) + EPS)
                    beta = ss.mean() / (o.mean() + EPS)
                    kge[b, s] = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
                    fit[b, s] = float(np.mean((ss - o) ** 2 / (std_train[b] ** 2)))
            # save grid
            rows = [{"basin_idx": b, "w": W_GRID[s], "NSE": nse[b, s], "KGE": kge[b, s],
                     "fit": fit[b, s]} for b in range(B) for s in range(S)]
            with (out_dir / f"process_grid_ep{epoch}_{proc}.csv").open("w", newline="") as f:
                wcsv = csv.DictWriter(f, fieldnames=list(rows[0]))
                wcsv.writeheader(); wcsv.writerows(rows)
            print(f"[A] ep{epoch} {proc}: done", flush=True)

            # exact per-process oracle
            aic_unit = AIC_ALPHA * COSTS[proc] / B
            w_star = np.full(B, np.nan)
            for b in range(B):
                vals = [fit[b, s] * n_valid_b[b] / N + aic_unit * W_GRID[s] for s in range(S)]
                if not np.isfinite(vals).any():
                    continue
                w_star[b] = W_GRID[int(np.argmin(vals))]
            dNSE = np.array([np.nanmax(nse[b, 1:]) - nse[b, 0] for b in range(B)])
            dKGE = np.array([np.nanmax(kge[b, 1:]) - kge[b, 0] for b in range(B)])
            fit_imp = np.array([fit[b, 0] - np.nanmin(fit[b, 1:]) for b in range(B)])
            pos = w_star > 0
            summary[f"ep{epoch}_{proc}"] = {
                "cost": COSTS[proc],
                "frac_dNSE_gt0": float(np.nanmean(dNSE > 0)),
                "frac_dNSE_gt0005": float(np.nanmean(dNSE > 0.005)),
                "frac_dNSE_gt001": float(np.nanmean(dNSE > 0.01)),
                "frac_dNSE_gt002": float(np.nanmean(dNSE > 0.02)),
                "frac_dNSE_gt005": float(np.nanmean(dNSE > 0.05)),
                "dNSE_dist": q(dNSE),
                "frac_oracle_pos": float(np.nanmean(pos)),
                "oracle_w_dist": {f"w={w:g}": float(np.nanmean(w_star == w)) for w in W_GRID},
                "fit_gain_oracle_pos_mean": float(np.nanmean(fit_imp[pos])),
                "fit_gain_oracle_pos_median": float(np.nanmedian(fit_imp[pos])),
                "learned_frac_gt001": float((w_learn[:, col] > 0.01).float().mean()),
                "learned_frac_gt01": float((w_learn[:, col] > 0.1).float().mean()),
                "learned_median": float(w_learn[:, col].median()),
                "false_neg_frac": float((w_learn[:, col][pos] < 0.01).float().mean()) if pos.sum() else np.nan,
                "n_oracle_pos": int(np.nansum(pos)),
                "mean_fit_gain_oracle_pos_in_loss_units":
                    float(np.nanmean(fit_imp[pos] * n_valid_b[pos] / N)) if pos.sum() else np.nan,
                "total_obj_gain_oracle_pos_mean":
                    float(np.nanmean((fit_imp[pos] * n_valid_b[pos] / N) - (aic_unit * w_star[pos]))) if pos.sum() else np.nan,
            }
            for b in range(B):
                oracle_rows.append({"basin_idx": b, "process": proc, "epoch": epoch,
                                   "w_star": w_star[b], "learned_w": float(w_learn[b, col]),
                                   "dNSE_max": dNSE[b], "dKGE_max": dKGE[b],
                                   "fit_improvement": fit_imp[b], "n_valid": n_valid_b[b]})
        print(f"[A] epoch {epoch} complete", flush=True)

    with (out_dir / "process_oracle_table.csv").open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(oracle_rows[0]))
        wcsv.writeheader(); wcsv.writerows(oracle_rows)
    (out_dir / "process_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    print(f"[A] -> {out_dir}/process_oracle_table.csv, process_summary.json")


if __name__ == "__main__":
    main()
