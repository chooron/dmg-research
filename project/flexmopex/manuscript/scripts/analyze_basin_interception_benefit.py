#!/usr/bin/env python3
"""Agent A — basin-level interception benefit under Candidate E-S0.

For each saved E-S0 checkpoint (epochs 0/1/2/5/10) and every basin (all 671):

  * keep all learned parameters, kappa/phi and the other gates fixed;
  * override ONLY the effective w_int over {0, .1, .25, .5, .75, 1.0};
  * evaluate on the canonical eval window (1995-10-01..2010-09-30, scored days
    [365:]) and compute per-basin NSE, KGE and the training-normalized fit loss
    (mean (Q-O)^2 / (std_train_i + eps)^2 over valid days; no AIC).

Per basin/epoch derive:

  * delta_NSE_max = max_{w>0} NSE(w) - NSE(w=0)   (same for KGE)
  * best positive w (NSE and fit-loss based), fit-loss improvement vs w=0
  * monotonic vs interior-optimum shape

Validation: NSE(w=0) at epoch 10 must reproduce the canonical per-basin NSE
from the trained run's metrics.json (learned w_int ~= 0).

Outputs (CSV + JSON summary):
  results/intercept_candidates/E_S0/basin_benefit_ep{epoch}.csv   (per-epoch raw)
  results/intercept_candidates/E_S0/basin_benefit.csv             (merged, one row per basin per epoch)
  results/intercept_candidates/E_S0/basin_benefit_summary.json
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
from project.flexmopex.local_model_handler import FlexMopexModelHandler  # noqa: E402

GATE_NAMES = ["w_phen", "w_int", "w_snow", "w_sub"]
EPOCHS = [0, 1, 2, 5, 10]
W_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
EPS = 1e-6


def q(v):
    return {k: float(np.quantile(v, qq)) for k, qq in (("p05", .05), ("p10", .10),
            ("p25", .25), ("median", .50), ("p75", .75), ("p90", .90), ("p95", .95))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/config_dmopex_interceptE_S0.yaml")
    ap.add_argument("--output-root", default="results/intercept_candidates")
    ap.add_argument("--run-name", default="E_S0")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--epochs", default="0,1,2,5,10")
    ap.add_argument("--limit-basins", type=int, default=0, help="0 = all 671")
    ap.add_argument("--save-grid", action="store_true",
                    help="also dump per-basin NSE/KGE/fit at every w grid value")
    args = ap.parse_args()
    root = Path(args.output_root)
    arm_dir = root / args.run_name
    epochs = [int(e) for e in args.epochs.split(",")]

    cli = parse_args(["--config", args.config, "--gpu-id", str(args.gpu_id),
                      "--output-root", str(args.output_root), "--run-name", args.run_name])
    config = load_config(args.config)
    apply_runtime_overrides(config, cli, config_path=args.config)
    config["mode"] = "train"
    if str(config["device"]).startswith("cuda"):
        torch.cuda.set_device(config["device"])

    dl = _build_data_loader(config)
    td, ed = dl.train_dataset, dl.eval_dataset
    dev = config["device"]
    n_basin = td["x_phy"].shape[1]
    basins = list(range(n_basin if args.limit_basins <= 0 else min(args.limit_basins, n_basin)))
    n_attr = td["xc_nn_norm"].shape[-1] - 3
    attrs = td["xc_nn_norm"][0, basins, -n_attr:].to(dev)

    # training-normalized per-basin std (same as the loss object uses)
    y_obs = td["target"][:, :, 0].cpu().numpy()
    std_train = np.nanstd(y_obs, axis=0) + 0.1   # eps=0.1 as in NseDynAicBatchLoss

    # eval window: scored days [365:] (internal warmup 365 handled by the model)
    x_ev = ed["x_phy"].to(dev)
    doy_ev = ed["doy"].to(dev)
    y_ev = ed["target"][:, basins, 0].cpu().numpy()      # (T, B)
    x_ev = x_ev[:, basins, :]
    doy_ev = doy_ev[:, basins, :]
    t0 = 365
    sample = {"x_phy": x_ev, "doy": doy_ev,
              "c_nn_norm": attrs[basins].to(dev)}

    # eager steps only (torch.compile of a new batch shape is fragile);
    # vectorize the w grid as B*S pseudo-basins -> ONE forward per epoch.
    config["model"]["phy"]["disable_compile"] = True
    handler = FlexMopexModelHandler(config, verbose=False)
    merged = []
    summary = {}
    S = len(W_GRID)
    for epoch in epochs:
        try:
            handler.load_model(epoch)
        except FileNotFoundError:
            print(f"[warn] epoch {epoch} checkpoint missing; skipping")
            continue
        for m in handler.model_dict.values():
            m.eval()
        model = next(iter(handler.model_dict.values()))
        phy, nn = model.phy_model, model.nn_model
        with torch.no_grad():
            p = nn({"c_nn_norm": attrs.to(dev)})
            w_learn = F.softmax(p["weights"].view(len(basins), 4, 2).clamp(min=-10., max=10.), dim=-1)[..., 1]
            mopex_params = phy._descale_mopex_params(p["params"])
            routing = phy._descale_routing_params(p["gamma_uh"])
            base_w = F.softmax(p["weights"].view(len(basins), 4, 2).clamp(min=-10., max=10.), dim=-1)[..., 1].detach()

        # pseudo-basin layout: row b*S + s  ->  basin b with w_grid[s]
        B = len(basins)
        w_on = base_w.repeat(S, 1)
        for s in range(S):
            w_on[s * B:(s + 1) * B, 1] = W_GRID[s]
        params_rep = {k: v.repeat(S, 1) for k, v in mopex_params.items()}
        routing_rep = {k: v.repeat(S) for k, v in routing.items()}   # 1-D per-basin params
        xs = {k: (v.repeat(1, S, 1) if v.ndim == 3 and k in ("x_phy", "doy") else v)
              for k, v in sample.items()}
        with torch.no_grad():
            P, T, PET, doy, n_steps, _ = phy._prepare_forcings(xs)
            Q = phy._run_weighted_loop(P, T, PET, doy, params_rep, w_on, n_steps, B * S)
            Qr = phy._apply_routing(Q.mean(-1), routing_rep).cpu().numpy()[:, :, 0]  # (T, B*S)
        Qr = Qr[:, :B * S].reshape(Qr.shape[0], S, B)         # (T, S, B)
        Qr = np.transpose(Qr, (0, 2, 1))                      # (T, B, S)
        # The physics loop already excludes the internal 365-day warmup, so
        # Qr[t] corresponds to eval obs day (365 + t).  No second trim.
        Qs = Qr
        n_out = Qs.shape[0]                                   # routing trims ~15 days
        ys = y_ev[t0:t0 + n_out]                              # align (as trainer _trim)
        valid = ~np.isnan(ys)
        nse = np.full((B, S), np.nan); kge = np.full((B, S), np.nan); fit = np.full((B, S), np.nan)
        for b in range(B):
            v = valid[:, b]
            if v.sum() < 30:
                continue
            o = ys[v, b]
            for s in range(S):
                ss = Qs[v, b, s]
                ss_res = np.sum((ss - o) ** 2)
                ss_tot = np.sum((o - o.mean()) ** 2)
                nse[b, s] = 1.0 - ss_res / (ss_tot + EPS) if ss_tot > EPS else np.nan
                r = np.corrcoef(ss, o)[0, 1] if np.std(ss) > 0 and np.std(o) > 0 else 0.0
                alpha = np.std(ss) / (np.std(o) + EPS)
                beta = ss.mean() / (o.mean() + EPS)
                kge[b, s] = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
                fit[b, s] = float(np.mean((ss - o) ** 2 / (std_train[basins[b]] ** 2)))
        nse_w = {W_GRID[s]: nse[:, s] for s in range(S)}
        kge_w = {W_GRID[s]: kge[:, s] for s in range(S)}
        fit_w = {W_GRID[s]: fit[:, s] for s in range(S)}
        print(f"  ep{epoch}: forward done", flush=True)

        nse0, kge0, fit0 = nse_w[0.0], kge_w[0.0], fit_w[0.0]
        best_w_nse = np.full(len(basins), np.nan)
        d_nse = np.full(len(basins), np.nan)
        d_kge = np.full(len(basins), np.nan)
        best_w_fit = np.full(len(basins), np.nan)
        d_fit = np.full(len(basins), np.nan)
        mono = np.full(len(basins), np.nan)
        for b in range(len(basins)):
            nse_pos = [nse_w[w][b] for w in W_GRID[1:]]
            kge_pos = [kge_w[w][b] for w in W_GRID[1:]]
            fit_pos = [fit_w[w][b] for w in W_GRID[1:]]
            d_nse[b] = max(nse_pos) - nse0[b]
            d_kge[b] = max(kge_pos) - kge0[b]
            best_w_nse[b] = W_GRID[1 + int(np.argmax(nse_pos))]
            d_fit[b] = fit0[b] - min(fit_pos)          # positive = fit improvement
            best_w_fit[b] = W_GRID[1 + int(np.argmin(fit_pos))]
            mono[b] = 1.0 if all(np.diff(nse_pos) >= -1e-9) else (0.0 if all(np.diff(nse_pos) <= 1e-9) else -1.0)
        for b in basins:
            merged.append({
                "basin_idx": b, "epoch": epoch,
                "learned_w_int": float(w_learn[b, 1]),
                "NSE_w0": float(nse0[basins.index(b)]), "delta_NSE_max": float(d_nse[basins.index(b)]),
                "best_w_NSE": float(best_w_nse[basins.index(b)]),
                "KGE_w0": float(kge0[basins.index(b)]), "delta_KGE_max": float(d_kge[basins.index(b)]),
                "fitloss_w0": float(fit0[basins.index(b)]), "fit_improvement": float(d_fit[basins.index(b)]),
                "best_w_fit": float(best_w_fit[basins.index(b)]),
                "monotonic": float(mono[basins.index(b)]),
            })
        summary[epoch] = {
            "n_benefit_dNSE_gt0": float(np.nansum(d_nse > 0)),
            "frac_dNSE_gt0": float(np.nanmean(d_nse > 0)),
            "frac_dKGE_gt0": float(np.nanmean(d_kge > 0)),
            "dNSE_dist": q(d_nse[~np.isnan(d_nse)]),
            "dKGE_dist": q(d_kge[~np.isnan(d_kge)]),
            "frac_dNSE_gt_0005": float(np.nanmean(d_nse > 0.005)),
            "frac_dNSE_gt_001": float(np.nanmean(d_nse > 0.01)),
            "frac_dNSE_gt_002": float(np.nanmean(d_nse > 0.02)),
            "frac_dNSE_gt_005": float(np.nanmean(d_nse > 0.05)),
            "best_w_NSE_dist": q(best_w_nse[~np.isnan(best_w_nse)]),
            "mono_frac": float(np.nanmean(mono == 1.0)),
            "interior_opt_frac": float(np.nanmean(mono == -1.0)),
        }
        # per-epoch CSV
        with (arm_dir / f"basin_benefit_ep{epoch}.csv").open("w", newline="") as f:
            rows = [r for r in merged if r["epoch"] == epoch]
            wcsv = csv.DictWriter(f, fieldnames=list(rows[0]))
            wcsv.writeheader(); wcsv.writerows(rows)
        if args.save_grid:
            grid_rows = []
            for idx in range(len(basins)):
                for wv in W_GRID:
                    grid_rows.append({"basin_idx": basins[idx], "epoch": epoch, "w": wv,
                                      "NSE": float(nse_w[wv][idx]), "KGE": float(kge_w[wv][idx]),
                                      "fit": float(fit_w[wv][idx])})
            with (arm_dir / f"basin_fitgrid_ep{epoch}.csv").open("w", newline="") as f:
                wcsv = csv.DictWriter(f, fieldnames=list(grid_rows[0]))
                wcsv.writeheader(); wcsv.writerows(grid_rows)
        print(f"[ep{epoch}] frac dNSE>0: {summary[epoch]['frac_dNSE_gt0']:.3f} "
              f">0.01: {summary[epoch]['frac_dNSE_gt_001']:.3f} >0.02: {summary[epoch]['frac_dNSE_gt_002']:.3f} "
              f">0.05: {summary[epoch]['frac_dNSE_gt_005']:.3f} | median dNSE: {summary[epoch]['dNSE_dist']['median']:.5f}")

    with (arm_dir / "basin_benefit.csv").open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(merged[0]))
        wcsv.writeheader(); wcsv.writerows(merged)
    (arm_dir / "basin_benefit_summary.json").write_text(json.dumps(summary, indent=2))

    # validation vs canonical metrics.json (ep10, NSE at w=0)
    try:
        _raw = (arm_dir / "test1995-2010_Ep10" / "metrics.json").read_text()
        canon = json.loads(_raw)
        if isinstance(canon, str):
            canon = json.loads(canon)            # double-encoded dump
        canon_nse = canon.get("nse", canon.get("NSE"))
        if canon_nse is not None and len(canon_nse) == n_basin:
            ep10 = [r for r in merged if r["epoch"] == 10]
            mine = np.array([r["NSE_w0"] for r in ep10])
            valid = ~np.isnan(mine)
            rho = np.corrcoef(mine[valid], np.array(canon_nse)[valid])[0, 1]
            print(f"[validate] ep10 NSE(w=0) vs canonical per-basin NSE: corr={rho:.4f}")
            summary["_validation"] = {"canonical_nse_corr_ep10_w0": float(rho)}
            (arm_dir / "basin_benefit_summary.json").write_text(json.dumps(summary, indent=2))
    except Exception as e:
        print(f"[validate] canonical comparison skipped: {e}")
    print(f"[done] tables -> {arm_dir}/basin_benefit.csv (+_ep*.csv), summary -> basin_benefit_summary.json")


if __name__ == "__main__":
    main()
