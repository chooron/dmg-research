#!/usr/bin/env python3
"""R9 — basin-level gate-gradient separability diagnostic on the R8 delayed-AIC
run (results/intercept_aicdelay/E_S0_aicdelay2, checkpoints ep1..ep4).

Questions: does a basin-specific interception ON signal exist after the R8
fit-only adaptation phase (epochs 1-2), and where is it lost — at the gate
parameterization (Jacobian/saturation), at shared-head population aggregation,
or is the local gradient itself not discriminative?

Conventions (reused from R7/R8, not redefined):

  * exact state-conditional total-objective oracle per (epoch, process):
        w*_b = argmin_s [ fit_b(s) * n_valid_b / N + aic_alpha * cost_p * s / 671 ]
    with fit_b(s) = mean over valid days of (Q - obs)^2 / std_train_b^2 on the
    5114-day eval window (routed rows 0..5113 vs target[365:365+5114]);
    N = sum_b n_valid_b = 3428084.  Oracle labels are recomputed at each
    checkpoint (they are state-conditional; never treated as fixed).
  * gate coordinate: 2-logit softmax pair, ON logit = index 1; w = softmax[...,1];
    the implemented Jacobian is dw/d(z_on - z_off) = w(1-w) (softmax-2); raw
    logits are clamped to [-10, 10] before softmax (recorded separately).
  * sign convention (R7-verified): g > 0 on the ON logit pushes OFF under
    gradient descent; AIC component = aic_alpha*cost*(1/671)*w(1-w) > 0.
  * shared-head aggregation (R7 definition): per-basin head contribution
    g_i * h_i with h = backbone(c_nn_norm) (B,128); subgroup aggregate =
    sum over group of g_i*h_i; R7 reported fit-based aggregates (gF), this
    script also reports total- and AIC-based aggregates (labeled).
  * gradient targets are real graph nodes: weights_on (structural weight),
    params["weights"] (raw gate logits).  Eval mode only; no parameter
    updates; disable_compile=True (eager).

The R7 32-basin gradient window cannot support full-basin group separation,
so gradients are computed on the full 671-basin x 5114-day window in basin
chunks (same loss/fit definition as the oracle); the R7 sign convention,
gate coordinate and head-aggregation definition are unchanged.  All gradient
quantities are at the exact-total-objective scale (fit weighted by
n_valid_b/N; AIC share aic*cost/671), i.e. the R7/R8 convention.  Note: the
live training loop applies the AIC share over a 100-basin minibatch
(aic*cost/100 per basin); this only rescales magnitudes uniformly and is
noted in the report.

Layers reported for w_int (all per basin):
  A. structural-weight level:  dL_fit/dw_int  (local per-basin fit and the
     objective-scaled version x n_valid_b/N), plus AIC derivative
     (aic_alpha*cost/671, constant) and canonical total.
  B. optimizer-facing gate-logit level (raw params["weights"], ON entry):
     g_fit, g_AIC, g_total; under the R8 mask the actual training gradient
     g_train = g_fit (epochs 1-2) — the counterfactual unmasked g_fit+g_AIC
     is reported at the same state with no parameter update.
  C. shared-head subgroup aggregates (oracle-positive vs oracle-zero):
     fit/total/AIC aggregate norms, size-normalized norms, ratios, cosines.
  D. gate Jacobian / saturation: w(1-w) and the clamp-active fraction.

Controls: the same summaries for w_phen, w_snow, w_sub (grouped by their own
state-conditional oracle) — summary-level only.

Validation (written to validation.json):
  * R7 AIC sign-convention check on unclamped basins:
        g_AIC_z == aic*cost*(1/671)*w(1-w)  (max abs err, per process);
  * masked training gradient == fit-only gradient at the gate (max abs err);
  * masked and unmasked total losses bit-identical at the same state;
  * determinism: epoch-2 chunk-0 gradient arrays recomputed -> bit-identical;
  * no parameter mutation: model parameter snapshot before/after each epoch.

Outputs (results/intercept_aicdelay/E_S0_aicdelay2/R9_separability/):
  oracle_state_conditional.csv, w_int_gradients.csv, controls_summary.csv,
  head_aggregation.json, validation.json, summary.json

Run: python scripts/analyze_gate_gradient_separability.py [--epochs 1,2,3,4]
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

PROCESSES = ["w_phen", "w_int", "w_snow", "w_sub"]
GATE_IDX = {"w_phen": 0, "w_int": 1, "w_snow": 2, "w_sub": 3}
COSTS = {"w_phen": 2.0, "w_int": 2.0, "w_snow": 2.0, "w_sub": 1.0}
AIC_ALPHA = 0.01
W_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
B_TOTAL = 671
EPS = 1e-12


def per_basin_fit(q: torch.Tensor, obs: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Mean normalized squared error per basin over valid days.

    q: [n, C, 1] routed rows; obs: [n, C, 1] aligned target; std: [C].
    Returns L_b [C].
    """
    o = torch.nan_to_num(obs, nan=0.0)
    sq = (q - o) ** 2 / (std.view(1, -1, 1) ** 2)
    mask = ~torch.isnan(obs)
    n_valid = mask.sum(dim=0).clamp(min=1)
    sq = torch.where(mask, sq, torch.zeros_like(sq))
    return sq.sum(dim=0) / n_valid  # [C,1] -> L_b per basin


def oracle_sweep(phy, x_phy, doy, w_learn, proc, mopex_params, routing,
                 n_out, y_ev, std_train, n_valid_b, N, dev):
    """State-conditional exact oracle for one process (R7 machinery, no_grad).

    Returns (w_star [B], fit_imp [B]).
    """
    B = w_learn.shape[0]
    S = len(W_GRID)
    col = GATE_IDX[proc]
    w_on = w_learn.detach().clone().repeat(S, 1)
    for s in range(S):
        w_on[s * B:(s + 1) * B, col] = W_GRID[s]
    params_rep = {k: v.detach().repeat(S, 1) for k, v in mopex_params.items()}
    routing_rep = {k: v.detach().repeat(S) for k, v in routing.items()}
    sample_rep = {"x_phy": x_phy.repeat(1, S, 1), "doy": doy.repeat(1, S, 1)}
    with torch.no_grad():
        P, T, PET, doy_r, n_steps, _ = phy._prepare_forcings(sample_rep)
        Q = phy._run_weighted_loop(P, T, PET, doy_r, params_rep, w_on, n_steps, B * S)
        Qr = phy._apply_routing(Q.mean(-1), routing_rep).cpu().numpy()[:, :, 0]
    Qs = Qr[:n_out].reshape(n_out, S, B)  # [n_out, S, B]
    fit = np.full((B, S), np.nan)
    for b in range(B):
        v = ~np.isnan(y_ev[:, b])
        if v.sum() < 30:
            continue
        o = y_ev[v, b]
        ss = Qs[v, :, b]  # [n_valid, S]
        fit[b, :] = np.nanmean((ss - o[:, None]) ** 2, axis=0) / (std_train[b] ** 2)
    aic_unit = AIC_ALPHA * COSTS[proc] / B
    w_star = np.full(B, np.nan)
    for b in range(B):
        vals = fit[b, :] * n_valid_b[b] / N + aic_unit * np.asarray(W_GRID)
        if not np.isfinite(vals).any():
            continue
        w_star[b] = W_GRID[int(np.argmin(vals))]
    fit_imp = np.array([fit[b, 0] - np.nanmin(fit[b, 1:]) if np.isfinite(fit[b, 0]) else np.nan
                        for b in range(B)])
    return w_star, fit_imp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/config_dmopex_interceptE_S0_aicdelay2.yaml")
    ap.add_argument("--output-root", default="results/intercept_aicdelay")
    ap.add_argument("--run-name", default="E_S0_aicdelay2")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--epochs", default="1,2,3,4")
    ap.add_argument("--chunk-size", type=int, default=128)
    args = ap.parse_args()

    root = Path(args.output_root)
    run_dir = root / args.run_name
    out_dir = run_dir / "R9_separability"
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(e) for e in args.epochs.split(",")]
    dev = f"cuda:{args.gpu_id}"

    cli = parse_args(["--config", args.config, "--gpu-id", str(args.gpu_id),
                      "--output-root", str(args.output_root), "--run-name", args.run_name])
    cfg = load_config(args.config)
    apply_runtime_overrides(cfg, cli, config_path=args.config)
    cfg["mode"] = "train"
    cfg["model"]["phy"]["disable_compile"] = True
    if str(cfg["device"]).startswith("cuda"):
        torch.cuda.set_device(cfg["device"])
    torch.manual_seed(42)

    dl = _build_data_loader(cfg)
    td, ed = dl.train_dataset, dl.eval_dataset
    B = td["x_phy"].shape[1]
    n_attr = td["xc_nn_norm"].shape[-1] - 3
    attrs = td["xc_nn_norm"][0, :, -n_attr:].to(dev)          # [671, 35]
    std_train = (np.nanstd(td["target"][:, :, 0].cpu().numpy(), axis=0) + 0.1).astype(np.float32)
    n_out = int(ed["x_phy"].shape[0]) - 365                   # 5114
    y_ev = ed["target"][365:365 + n_out, :, 0].cpu().numpy()  # [5114, 671]
    n_valid_b = np.sum(~np.isnan(y_ev), axis=0).astype(float)
    N = float(n_valid_b.sum())
    print(f"[R9] n_out={n_out}, N={N:.0f}, epochs={epochs}, chunk={args.chunk_size}", flush=True)

    handler = build_handler(cfg)
    x_phy_full = ed["x_phy"].to(dev)
    doy_full = ed["doy"].to(dev)

    oracle_rows, wint_rows, ctrl_rows = [], [], []
    head_agg, validation, summary = {}, {}, {}

    for epoch in epochs:
        handler.load_model(epoch)
        for m in handler.model_dict.values():
            m.eval()
        model = next(iter(handler.model_dict.values()))
        phy, nn = model.phy_model, model.nn_model
        # per-epoch no-mutation snapshot (before any diagnostic pass)
        param_snapshot = sum(float(p.detach().abs().sum()) for p in model.parameters())

        # shared eval-mode forward (no grad): learned gates + descaled params
        with torch.no_grad():
            params, logits, weights_on, mopex_params, routing = build_forward(
                phy, nn, {"x_phy": x_phy_full, "doy": doy_full, "c_nn_norm": attrs})
            w_learn = F.softmax(params["weights"].view(B, 4, 2).clamp(-10, 10), dim=-1)[..., 1]
        w_learn = w_learn.detach()

        # ---------- state-conditional oracle (all processes) ----------
        ep_oracle = {}
        for proc in PROCESSES:
            w_star, fit_imp = oracle_sweep(phy, x_phy_full, doy_full, w_learn, proc,
                                           mopex_params, routing, n_out, y_ev,
                                           std_train, n_valid_b, N, dev)
            ep_oracle[proc] = {"w_star": w_star, "fit_imp": fit_imp}
            for b in range(B):
                oracle_rows.append({"epoch": epoch, "process": proc, "basin_idx": b,
                                    "w_star": w_star[b], "learned_w": float(w_learn[b, GATE_IDX[proc]]),
                                    "fit_improvement": fit_imp[b], "n_valid": n_valid_b[b]})
            print(f"[R9] ep{epoch} {proc} oracle done (pos={int(np.nansum(w_star > 0))})", flush=True)

        # ---------- per-basin gradient decomposition (chunked, full window) ----------
        g = {proc: {k: [] for k in ("gfit_w_local", "gfit_w_obj", "gfit_z", "gtot_z",
                                    "gtrain_z", "gaic_z_an", "jac", "clamp", "w")}
             for proc in PROCESSES}
        ep_val = {}
        with torch.no_grad():
            h_all = nn.backbone(attrs)  # [671, 128] shared repr (R7 def)

        std_t = torch.from_numpy(std_train).to(dev)
        nv = torch.from_numpy(n_valid_b).to(dev)

        for c0 in range(0, B, args.chunk_size):
            c1 = min(c0 + args.chunk_size, B)
            C = c1 - c0
            sample = {"x_phy": x_phy_full[:, c0:c1], "doy": doy_full[:, c0:c1],
                      "c_nn_norm": attrs[c0:c1]}
            params, logits, weights_on, mopex_params, routing = build_forward(phy, nn, sample)
            out = run_loop(phy, sample, weights_on, mopex_params, routing)
            q = out["streamflow"]
            obs = ed["target"][365:365 + n_out, c0:c1].to(dev)
            L_b = per_basin_fit(q, obs, std_t[c0:c1])         # [C] local fit per basin
            wgt = nv[c0:c1] / N
            L_fit_obj = (L_b * wgt).sum()                      # exact-total-objective scale
            L_fit_loc = L_b.sum()                              # unweighted local sum
            # AIC at the exact-objective scale: aic*cost*mean over ALL 671 basins;
            # the off-chunk part is a constant (value only, no gradient).
            aic_live = AIC_ALPHA * sum(
                COSTS[p_] * weights_on[:, GATE_IDX[p_]].sum() / B_TOTAL for p_ in PROCESSES)
            L_unmasked = L_fit_obj + aic_live
            aic_det = AIC_ALPHA * sum(
                COSTS[p_] * weights_on[:, GATE_IDX[p_]].detach().sum() / B_TOTAL for p_ in PROCESSES)
            L_masked = L_fit_obj + aic_det                     # R8 mask: AIC path cut

            def grad_of(y, x):
                gg = torch.autograd.grad(y, x, retain_graph=True, allow_unused=True)[0]
                return gg if gg is not None else torch.zeros_like(x)

            g_fit_w_obj = grad_of(L_fit_obj, weights_on)       # [C,4] objective-scaled
            g_fit_w_loc = grad_of(L_fit_loc, weights_on)       # [C,4] local (unweighted)
            g_fit_z = grad_of(L_fit_obj, params["weights"]).view(C, 4, 2)
            g_tot_z = grad_of(L_unmasked, params["weights"]).view(C, 4, 2)
            g_train_z = grad_of(L_masked, params["weights"]).view(C, 4, 2)

            jac = (weights_on * (1 - weights_on)).detach()     # [C,4] softmax-2 Jacobian
            raw = params["weights"].view(C, 4, 2).detach()
            clamp_active = ((raw[:, :, 0] <= -10 + 1e-6) | (raw[:, :, 0] >= 10 - 1e-6) |
                            (raw[:, :, 1] <= -10 + 1e-6) | (raw[:, :, 1] >= 10 - 1e-6)).float()

            for proc in PROCESSES:
                col = GATE_IDX[proc]
                g[proc]["gfit_w_local"].append(g_fit_w_loc[:, col].detach().cpu())
                g[proc]["gfit_w_obj"].append(g_fit_w_obj[:, col].detach().cpu())
                g[proc]["gfit_z"].append(g_fit_z[:, col, 1].detach().cpu())
                g[proc]["gtot_z"].append(g_tot_z[:, col, 1].detach().cpu())
                g[proc]["gtrain_z"].append(g_train_z[:, col, 1].detach().cpu())
                g[proc]["gaic_z_an"].append(
                    (AIC_ALPHA * COSTS[proc] / B_TOTAL * jac[:, col]).detach().cpu())
                g[proc]["jac"].append(jac[:, col].detach().cpu())
                g[proc]["clamp"].append(clamp_active[:, col].detach().cpu())
                g[proc]["w"].append(weights_on[:, col].detach().cpu())

            # validation (first chunk only)
            if c0 == 0:
                ep_val["loss_masked_eq_unmasked"] = bool(torch.equal(L_masked, L_unmasked))
                ep_val["g_train_eq_g_fit_max"] = float((g_train_z - g_fit_z).abs().max().item())
                uncl = (clamp_active[:, :] < 0.5)
                errs = {}
                for proc in PROCESSES:
                    col = GATE_IDX[proc]
                    aic_an = AIC_ALPHA * COSTS[proc] / B_TOTAL * jac[:, col]
                    diff = (g_tot_z[:, col, 1] - g_fit_z[:, col, 1] - aic_an).abs()
                    errs[proc] = float(diff[uncl[:, col]].max().item()) if uncl[:, col].any() else float("nan")
                ep_val["aic_sign_convention_max_err_unclamped"] = errs
            del q, out, params, logits, weights_on, mopex_params, routing
            torch.cuda.empty_cache()

        # determinism: redo chunk 0 at epoch 2 and compare bit-identically
        if epoch == 2:
            c0 = 0; c1 = min(args.chunk_size, B)
            sample = {"x_phy": x_phy_full[:, c0:c1], "doy": doy_full[:, c0:c1],
                      "c_nn_norm": attrs[c0:c1]}
            params, logits, weights_on, mopex_params, routing = build_forward(phy, nn, sample)
            out = run_loop(phy, sample, weights_on, mopex_params, routing)
            obs = ed["target"][365:365 + n_out, c0:c1].to(dev)
            L_b2 = per_basin_fit(out["streamflow"], obs, std_t[c0:c1])
            wgt2 = nv[c0:c1] / N          # byte-identical expression to the main loop
            L2 = (L_b2 * wgt2).sum()
            g2 = torch.autograd.grad(L2, weights_on, retain_graph=True)[0].detach().cpu()
            ok = bool(torch.equal(g2[:, GATE_IDX["w_int"]], g["w_int"]["gfit_w_obj"][0]))
            ep_val["determinism_chunk0_bit_identical"] = ok
            del out, params, logits, weights_on, mopex_params, routing
            torch.cuda.empty_cache()

        # no-mutation snapshot
        s_after = sum(float(p.detach().abs().sum()) for p in model.parameters())
        ep_val["param_snapshot_unchanged"] = bool(abs(s_after - param_snapshot) < 1e-9)
        validation[str(epoch)] = ep_val

        # ---------- assemble per-epoch results ----------
        ep_head = {}
        for proc in PROCESSES:
            arr = {k: torch.cat(g[proc][k]) for k in g[proc]}
            w_star = ep_oracle[proc]["w_star"]
            pos_np = np.nan_to_num(w_star) > 0
            opos = torch.from_numpy(pos_np)
            ozero = ~opos

            # ---- w_int: basin-level export (mandatory) ----
            if proc == "w_int":
                for b in range(B):
                    wint_rows.append({
                        "epoch": epoch, "basin_idx": b,
                        "oracle_group": "pos" if pos_np[b] else "zero",
                        "w_star": w_star[b],
                        "fit_improvement": ep_oracle[proc]["fit_imp"][b],
                        "learned_w": float(arr["w"][b]),
                        "jacobian_w1mw": float(arr["jac"][b]),
                        "clamp_active": float(arr["clamp"][b]),
                        "gfit_w_local": float(arr["gfit_w_local"][b]),
                        "gfit_w_obj": float(arr["gfit_w_obj"][b]),
                        "gfit_z": float(arr["gfit_z"][b]),
                        "gaic_z_analytic": float(arr["gaic_z_an"][b]),
                        "gtot_z": float(arr["gtot_z"][b]),
                        "gtrain_z": float(arr["gtrain_z"][b]),
                        "fit_pushes_ON_w": float(arr["gfit_w_local"][b] < 0),
                        "fit_pushes_ON_z": float(arr["gfit_z"][b] < 0),
                        "total_pushes_ON_z": float(arr["gtot_z"][b] < 0),
                        "train_pushes_ON_z": float(arr["gtrain_z"][b] < 0),
                    })

            # ---- group summaries (all processes; summary level for controls) ----
            row = {"epoch": epoch, "process": proc}
            for tag, m in (("oracle_pos", opos), ("oracle_zero", ozero)):
                n_g = int(m.sum())
                row[f"{tag}_n"] = n_g
                if n_g == 0:
                    continue
                gfw = arr["gfit_w_local"][m]; gfz = arr["gfit_z"][m]
                gtz = arr["gtot_z"][m]; gtz_an = arr["gaic_z_an"][m]
                gtr = arr["gtrain_z"][m]; wm = arr["w"][m]
                row[f"{tag}_frac_fit_ON_w"] = float((gfw < 0).float().mean())
                row[f"{tag}_frac_fit_ON_z"] = float((gfz < 0).float().mean())
                row[f"{tag}_frac_total_ON_z"] = float((gtz < 0).float().mean())
                row[f"{tag}_frac_train_ON_z"] = float((gtr < 0).float().mean())
                row[f"{tag}_median_abs_gfit_w"] = float(gfw.abs().median())
                row[f"{tag}_median_abs_gfit_z"] = float(gfz.abs().median())
                row[f"{tag}_median_abs_gtot_z"] = float(gtz.abs().median())
                row[f"{tag}_mean_gfit_z"] = float(gfz.mean())
                row[f"{tag}_mean_gtot_z"] = float(gtz.mean())
                row[f"{tag}_R_fit_aic_median"] = float((gfz.abs() / (gtz_an.abs() + EPS)).median())
                row[f"{tag}_w_median"] = float(wm.median())
                row[f"{tag}_w_frac_gt001"] = float((wm > 0.01).float().mean())
            row["frac_fit_ON_w_all"] = float((arr["gfit_w_local"] < 0).float().mean())
            row["frac_fit_ON_z_all"] = float((arr["gfit_z"] < 0).float().mean())
            row["frac_total_ON_z_all"] = float((arr["gtot_z"] < 0).float().mean())
            ctrl_rows.append(row)

            # ---- shared-head subgroup aggregation (R7 definition) ----
            h = h_all.cpu()
            agg = {}
            for variant, gname in (("fit", "gfit_z"), ("total", "gtot_z"), ("aic", "gaic_z_an")):
                gg = arr[gname]
                agg_p = (gg[opos].unsqueeze(-1) * h[opos]).sum(0)
                agg_z = (gg[ozero].unsqueeze(-1) * h[ozero]).sum(0)
                agg_full = agg_p + agg_z
                agg[variant] = {
                    "pos_n": int(opos.sum()), "zero_n": int(ozero.sum()),
                    "pos_norm": float(agg_p.norm()), "zero_norm": float(agg_z.norm()),
                    "pos_norm_per_basin": float(agg_p.norm() / max(int(opos.sum()), 1)),
                    "zero_norm_per_basin": float(agg_z.norm() / max(int(ozero.sum()), 1)),
                    "norm_ratio_zero_over_pos": float(agg_z.norm() / (agg_p.norm() + EPS)),
                    "pos_bias_sum": float(gg[opos].sum()), "zero_bias_sum": float(gg[ozero].sum()),
                    "cos_pos_zero": float(F.cosine_similarity(agg_p.view(1, -1), agg_z.view(1, -1))[0]),
                    "cos_pos_full": float(F.cosine_similarity(agg_p.view(1, -1), agg_full.view(1, -1))[0]),
                    "cos_zero_full": float(F.cosine_similarity(agg_z.view(1, -1), agg_full.view(1, -1))[0]),
                }
            ep_head[proc] = agg
        head_agg[str(epoch)] = ep_head

        # ---- summary block for w_int ----
        a = {k: torch.cat(g["w_int"][k]) for k in g["w_int"]}
        w_star = ep_oracle["w_int"]["w_star"]
        pos_np = np.nan_to_num(w_star) > 0
        pm = torch.from_numpy(pos_np)
        summary[str(epoch)] = {
            "oracle_pos_n": int(pos_np.sum()), "oracle_zero_n": int((~pos_np).sum()),
            "learned_w_median": float(a["w"].median()),
            "learned_frac_gt001": float((a["w"] > 0.01).float().mean()),
            "jacobian_w1mw_median": float(a["jac"].median()),
            "clamp_active_frac": float(a["clamp"].mean()),
            "frac_fit_ON_w_pos": float((a["gfit_w_local"][pm] < 0).float().mean()) if pm.any() else float("nan"),
            "frac_fit_ON_w_zero": float((a["gfit_w_local"][~pm] < 0).float().mean()),
            "frac_fit_ON_z_pos": float((a["gfit_z"][pm] < 0).float().mean()) if pm.any() else float("nan"),
            "frac_fit_ON_z_zero": float((a["gfit_z"][~pm] < 0).float().mean()),
            "frac_total_ON_z_pos": float((a["gtot_z"][pm] < 0).float().mean()) if pm.any() else float("nan"),
            "frac_total_ON_z_zero": float((a["gtot_z"][~pm] < 0).float().mean()),
            "frac_train_ON_z_pos": float((a["gtrain_z"][pm] < 0).float().mean()) if pm.any() else float("nan"),
            "frac_train_ON_z_zero": float((a["gtrain_z"][~pm] < 0).float().mean()),
            "median_abs_gfit_w_pos": float(a["gfit_w_local"][pm].abs().median()) if pm.any() else float("nan"),
            "median_abs_gfit_w_zero": float(a["gfit_w_local"][~pm].abs().median()),
            "median_abs_gfit_z_pos": float(a["gfit_z"][pm].abs().median()) if pm.any() else float("nan"),
            "median_abs_gfit_z_zero": float(a["gfit_z"][~pm].abs().median()),
            "median_abs_gtot_z_pos": float(a["gtot_z"][pm].abs().median()) if pm.any() else float("nan"),
            "median_abs_gtot_z_zero": float(a["gtot_z"][~pm].abs().median()),
            "median_fit_imp_pos": float(np.nanmedian(ep_oracle["w_int"]["fit_imp"][pos_np])),
            "median_fit_imp_zero": float(np.nanmedian(ep_oracle["w_int"]["fit_imp"][~pos_np])),
        }
        print(f"[R9] epoch {epoch} complete", flush=True)

    # oracle positive-set overlap across epochs (w_int)
    pos_sets = {ep: {r["basin_idx"] for r in oracle_rows
                     if r["epoch"] == ep and r["process"] == "w_int"
                     and np.nan_to_num(r["w_star"]) > 0} for ep in epochs}
    overlap = {}
    for i in range(1, len(epochs)):
        a_, b_ = epochs[i - 1], epochs[i]
        inter = len(pos_sets[a_] & pos_sets[b_])
        union = len(pos_sets[a_] | pos_sets[b_])
        overlap[f"{a_}->{b_}"] = {"n_pos_a": len(pos_sets[a_]), "n_pos_b": len(pos_sets[b_]),
                                  "intersection": inter,
                                  "jaccard": float(inter / union) if union else float("nan")}
    summary["oracle_positive_overlap"] = overlap

    # ---------- write outputs ----------
    with (out_dir / "oracle_state_conditional.csv").open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(oracle_rows[0]))
        wcsv.writeheader(); wcsv.writerows(oracle_rows)
    with (out_dir / "w_int_gradients.csv").open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(wint_rows[0]))
        wcsv.writeheader(); wcsv.writerows(wint_rows)
    with (out_dir / "controls_summary.csv").open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(ctrl_rows[0]))
        wcsv.writeheader(); wcsv.writerows(ctrl_rows)
    (out_dir / "head_aggregation.json").write_text(json.dumps(head_agg, indent=2, default=float))
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2, default=float))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    print(f"[R9] -> {out_dir}/")


if __name__ == "__main__":
    main()
