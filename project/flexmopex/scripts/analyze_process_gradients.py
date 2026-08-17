#!/usr/bin/env python3
"""Agent B — four-process gradient and gate-dynamics audit at the standard
Candidate E-S0 checkpoints (epochs 0/1/2/5/10; 32-basin diagnostic window).

For every process and checkpoint, per-basin logit gradients:

  g_fit_i   = dL_fit/dz_i      (z = ON logit of the gate pair)
  g_AIC_i   = dL_AIC/dz_i
  g_total_i = g_fit_i + g_AIC_i

Sign convention (verified numerically in the script): g_total > 0 pushes the
ON logit down -> gate OFF (gradient descent).  AIC_z = aic*cost*(1/B)*w(1-w) > 0
always -> AIC always pushes OFF; fit gradient sign varies.

Oracle-positive/zero subgroup comparison uses the exact per-process oracle
labels (epoch 10; four_process/process_oracle_table.csv) and tests whether
basins that become oracle-positive already produce ON-directed gradients early
(H4) and whether the population aggregate dominates (H3).

Shared-head aggregation: for w_int and w_snow at epoch 0 and 2, the aggregate
weights-head gradient contribution of a basin is g_i * h_i (h = backbone
output); subgroup aggregates are compared by norm and cosine.
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
    apply_runtime_overrides, parse_args, _build_data_loader, _build_loss,
)
from scripts.diagnose_wint_collapse import (  # noqa: E402
    build_handler, diagnostic_sample, build_forward, run_loop,
)

PROCESSES = ["w_phen", "w_int", "w_snow", "w_sub"]
GATE_IDX = {"w_phen": 0, "w_int": 1, "w_snow": 2, "w_sub": 3}
EPOCHS = [0, 1, 2, 5, 10]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/config_dmopex_interceptE_S0.yaml")
    ap.add_argument("--output-root", default="results/intercept_candidates")
    ap.add_argument("--run-name", default="E_S0")
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.output_root)
    arm_dir = root / args.run_name
    out_dir = arm_dir / "four_process"
    out_dir.mkdir(parents=True, exist_ok=True)

    cli = parse_args(["--config", args.config, "--gpu-id", str(args.gpu_id),
                      "--output-root", str(args.output_root), "--run-name", args.run_name])
    cfg = load_config(args.config)
    apply_runtime_overrides(cfg, cli, config_path=args.config)
    cfg["mode"] = "train"
    if str(cfg["device"]).startswith("cuda"):
        torch.cuda.set_device(cfg["device"])

    dl = _build_data_loader(cfg)
    td = dl.train_dataset
    loss_tot = _build_loss(cfg, td)
    loss_fit = _build_loss(cfg, td)
    loss_fit.aic_alpha = 0.0
    sample = diagnostic_sample(td, cfg["device"])
    n_basin = sample["x_phy"].shape[1]
    handler = build_handler(cfg)

    oracle = pd.read_csv(out_dir / "process_oracle_table.csv")
    oracle = oracle[oracle["epoch"] == 10]

    results = {}
    rows_csv = []
    for epoch in EPOCHS:
        handler.load_model(epoch)
        for m in handler.model_dict.values():
            m.eval()
        model = next(iter(handler.model_dict.values()))
        phy, nn = model.phy_model, model.nn_model
        params, logits, weights_on, mopex_params, routing = build_forward(phy, nn, sample)
        out = run_loop(phy, sample, weights_on, mopex_params, routing)
        q = out["streamflow"]
        target = sample["target"]
        n = min(q.shape[0], target.shape[0])
        wdict = {g: out[g] for g in PROCESSES}
        L_tot = loss_tot(q[:n], target[:n], sample_ids=sample["batch_sample"], weights=wdict)
        L_fit = loss_fit(q[:n], target[:n], sample_ids=sample["batch_sample"], weights=wdict)

        def grad_of(y, x):
            g = torch.autograd.grad(y, x, retain_graph=True, allow_unused=True)[0]
            return g if g is not None else torch.zeros_like(x)

        gT_z = grad_of(L_tot, params["weights"]).view(n_basin, 4, 2)  # (B,4,2)
        gF_z = grad_of(L_fit, params["weights"]).view(n_basin, 4, 2)
        h = nn.backbone(sample["c_nn_norm"])                          # (B,128) shared repr

        ep_res = {}
        for proc in PROCESSES:
            col = GATE_IDX[proc]
            w = weights_on[:, col].detach()
            gF = gF_z[:, col, 1].detach()
            gT = gT_z[:, col, 1].detach()
            gA = gT - gF
            # sign convention check: gT > 0 -> OFF; verify AIC component ~ +aic*cost*(1/B)*w(1-w)
            aic_check = float((gA - 0.01 * 2.0 / n_basin * w * (1 - w)).abs().max()) if proc != "w_sub" else \
                float((gA - 0.01 * 1.0 / n_basin * w * (1 - w)).abs().max())
            opos = oracle.loc[oracle["process"] == proc, "w_star"].to_numpy() > 0

            def agg(mask, g):
                m = mask[:n_basin]
                return g[m]

            res = {
                "w_median": float(w.median()), "w_iqr": float(torch.quantile(w, 0.75) - torch.quantile(w, 0.25)),
                "frac_active_gt001": float((w > 0.01).float().mean()),
                "dw_dz_median": float((w * (1 - w)).median()),
                "median_abs_gfit": float(gF.abs().median()),
                "median_abs_gaic": float(gA.abs().median()),
                "median_abs_gtotal": float(gT.abs().median()),
                "R_fit_aic_median": float((gF.abs() / (gA.abs() + 1e-12)).median()),
                "frac_fit_pushes_ON": float((gF < 0).float().mean()),
                "frac_total_pushes_ON": float((gT < 0).float().mean()),
                "frac_total_pushes_OFF": float((gT > 0).float().mean()),
                "sign_convention_aic_check_max": float(aic_check),
            }
            # oracle-positive vs oracle-zero
            for tag, mask in (("oracle_pos", opos), ("oracle_zero", ~opos)):
                gFm, gTm = agg(mask, gF), agg(mask, gT)
                res[f"{tag}_n"] = int(mask.sum())
                res[f"{tag}_median_abs_gfit"] = float(gFm.abs().median()) if len(gFm) else float("nan")
                res[f"{tag}_median_abs_gtotal"] = float(gTm.abs().median()) if len(gTm) else float("nan")
                res[f"{tag}_frac_fit_ON"] = float((gFm < 0).float().mean()) if len(gFm) else float("nan")
                res[f"{tag}_frac_total_ON"] = float((gTm < 0).float().mean()) if len(gTm) else float("nan")
                res[f"{tag}_mean_gtotal"] = float(gTm.mean()) if len(gTm) else float("nan")
            # shared-head subgroup aggregates (w_int, w_snow at ep 0/2)
            if proc in ("w_int", "w_snow") and epoch in (0, 2):
                g_pos = gF[opos[:n_basin]]
                g_zero = gF[~opos[:n_basin]]
                h_pos = h[opos[:n_basin]]
                h_zero = h[~opos[:n_basin]]
                agg_pos = (g_pos.unsqueeze(-1) * h_pos).sum(0)
                agg_zero = (g_zero.unsqueeze(-1) * h_zero).sum(0)
                cos = float(F.cosine_similarity(agg_pos.view(1, -1), agg_zero.view(1, -1))[0])
                res["head_agg"] = {
                    "pos_norm": float(agg_pos.norm()),
                    "zero_norm": float(agg_zero.norm()),
                    "pos_bias_sum": float(g_pos.sum()),
                    "zero_bias_sum": float(g_zero.sum()),
                    "cos_pos_zero": cos,
                    "norm_ratio_pos_over_zero": float(agg_pos.norm() / (agg_zero.norm() + 1e-12)),
                }
            ep_res[proc] = res
            rows_csv.append({"epoch": epoch, "process": proc,
                             **{k: v for k, v in res.items() if k != "head_agg"}})  # head_agg stays in JSON
        results[epoch] = ep_res
        print(f"[B] epoch {epoch} done", flush=True)
        del q, out, L_tot, L_fit
        torch.cuda.empty_cache()

    (out_dir / "gradient_results.json").write_text(json.dumps(results, indent=2, default=float))
    with (out_dir / "gradient_decomposition.csv").open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        wcsv.writeheader(); wcsv.writerows(rows_csv)
    print(f"[B] -> {out_dir}/gradient_decomposition.csv, gradient_results.json")


if __name__ == "__main__":
    main()
