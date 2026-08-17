#!/usr/bin/env python3
"""Mechanistic diagnosis of the E-S0 w_int gate collapse (Phases 3-6).

Uses the saved E-S0 checkpoints (epochs 0/1/2/5/10) and the SAME deterministic
diagnostic batch (basins 0..31, days 365..1095) at every epoch and gate.

For each gate (w_phen, w_int, w_snow, w_sub) at each epoch:

  * eval-softmax decomposition: g_fit,z, g_AIC,z, g_total,z wrt the ON logit
    (and wrt w), verifying g_total,z ~= g_fit,z + g_AIC,z numerically;
  * |dw/dz| = w(1-w) (eval, tau=1) and the chain check g_z ~= g_w * dw/dz;
  * R_z = |g_fit,z| / (|g_AIC,z| + eps); opposing-sign fraction; update direction.

  * Gumbel-matched control (epochs 0/1/2): identical Gumbel realization for the
    fit/AIC/total passes (manual Gumbel noise with a fixed generator).

Frozen w_int objective profiles per checkpoint (no_grad, only w_int overridden):

  * L_fit(w_int) and L_total(w_int) over the fixed grid; fit-only and total
    minimizers; Delta(L_fit)(w=0 vs best positive w); local slope near the
    learned gate value.

Compensation attribution: per-epoch gate/parameter drift and the fit-benefit of
interception across epochs.
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
    apply_runtime_overrides, parse_args, _build_data_loader, _build_loss,
)
from project.flexmopex.local_model_handler import FlexMopexModelHandler  # noqa: E402

GATE_NAMES = ["w_phen", "w_int", "w_snow", "w_sub"]
COSTS = {"w_phen": 2.0, "w_int": 2.0, "w_snow": 2.0, "w_sub": 1.0}
AIC_ALPHA = 0.01
DIAG_BASINS = list(range(32))
DIAG_T0, DIAG_T1 = 365, 1095
EPOCHS = [0, 1, 2, 5, 10]
W_GRID = [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
EPS = 1e-12


def qmed(x: torch.Tensor) -> float:
    return float(x.detach().float().median())


def qmean(x: torch.Tensor) -> float:
    return float(x.detach().float().mean())


def build_handler(config: dict) -> FlexMopexModelHandler:
    cfg = dict(config)
    cfg["mode"] = "train"
    return FlexMopexModelHandler(cfg, verbose=False)


def diagnostic_sample(train_dataset, device: str) -> dict:
    n_attr = train_dataset["xc_nn_norm"].shape[-1] - 3
    return {
        "x_phy": train_dataset["x_phy"][DIAG_T0:DIAG_T1, DIAG_BASINS, :].to(device),
        "doy": train_dataset["doy"][DIAG_T0:DIAG_T1, DIAG_BASINS, :].to(device),
        "c_nn_norm": train_dataset["xc_nn_norm"][0, DIAG_BASINS, -n_attr:].to(device),
        "target": train_dataset["target"][DIAG_T0 + 365:DIAG_T1, DIAG_BASINS, :].to(device),
        "batch_sample": np.asarray(DIAG_BASINS, dtype=np.int64),
    }


def build_forward(phy, nn, sample):
    """One shared eval-mode forward: nn -> params; returns all graph nodes."""
    params = nn(sample)
    logits = params["weights"].view(sample["c_nn_norm"].shape[0], 4, 2).clamp(min=-10.0, max=10.0)
    weights_on = F.softmax(logits, dim=-1)[..., 1]
    mopex_params = phy._descale_mopex_params(params["params"])
    routing = phy._descale_routing_params(params["gamma_uh"])
    return params, logits, weights_on, mopex_params, routing


def run_loop(phy, sample, weights_on, mopex_params, routing):
    """Physics loop with the given weights_on (same graph as build_forward)."""
    P, T, PET, doy, n_steps, n_grid = phy._prepare_forcings(sample)
    Q_mopex = phy._run_weighted_loop(P, T, PET, doy, mopex_params, weights_on, n_steps, n_grid)
    Qrouted = phy._apply_routing(Q_mopex.mean(-1), routing)
    out = {"streamflow": Qrouted}
    for i, name in enumerate(phy.weight_names):
        out[name] = weights_on[:, i].view(1, -1, 1).expand(Q_mopex.shape[0], -1, -1)
    return out


def eval_gate_decomposition(handler, loss_fit, loss_total, sample) -> dict:
    """Eval-softmax decomposition for all four gates."""
    model = next(iter(handler.model_dict.values()))
    phy, nn = model.phy_model, model.nn_model
    params, logits, weights_on, mopex_params, routing = build_forward(phy, nn, sample)

    out = run_loop(phy, sample, weights_on, mopex_params, routing)
    q = out["streamflow"]
    target = sample["target"]
    n = min(q.shape[0], target.shape[0])
    wdict = {g: out[g] for g in GATE_NAMES}
    L_total = loss_total(q[:n], target[:n], sample_ids=sample["batch_sample"], weights=wdict)
    L_fit = loss_fit(q[:n], target[:n], sample_ids=sample["batch_sample"], weights=wdict)

    def grad_of(y, x):
        g = torch.autograd.grad(y, x, retain_graph=True, allow_unused=True)[0]
        return g if g is not None else torch.zeros_like(x)

    gT_w = grad_of(L_total, weights_on)          # (B,4)
    gF_w = grad_of(L_fit, weights_on)            # (B,4)
    gT_z = grad_of(L_total, params["weights"]).view(-1, 4, 2)   # (B,4,2)
    gF_z = grad_of(L_fit, params["weights"]).view(-1, 4, 2)

    rows = {}
    for i, gname in enumerate(GATE_NAMES):
        w = weights_on[:, i]
        dlogit = w * (1.0 - w)                    # |dw/dz1| eval (tau=1)
        gF_w_i = gF_w[:, i]
        gT_w_i = gT_w[:, i]
        gA_w_i = gT_w_i - gF_w_i
        gF_z_i = gF_z[:, i, 1]
        gT_z_i = gT_z[:, i, 1]
        gA_z_i = gT_z_i - gF_z_i
        # chain check: g_z ~= g_w * dw/dz (median ratio over basins)
        chain_ratio = qmed((gF_z_i / (gF_w_i * dlogit + EPS)).abs())
        R = qmed((gF_z_i.abs() / (gA_z_i.abs() + EPS)))
        oppose = float((gF_z_i < 0).float().mean())   # AIC_z > 0 always -> opposition = fit wants ON
        rows[gname] = {
            "w_median": qmed(w),
            "dw_dz": qmed(dlogit),
            "dL_fit_dw_med": qmed(gF_w_i.abs()),
            "dL_AIC_dw_med": qmed(gA_w_i.abs()),
            "dL_total_dw_med": qmed(gT_w_i.abs()),
            "dL_fit_dz_med": qmed(gF_z_i.abs()),
            "dL_AIC_dz_med": qmed(gA_z_i.abs()),
            "dL_total_dz_med": qmed(gT_z_i.abs()),
            "g_fit_dz_mean": qmean(gF_z_i),
            "g_AIC_dz_mean": qmean(gA_z_i),
            "g_total_dz_mean": qmean(gT_z_i),
            "chain_gz_eq_gw_dwdz_ratio": chain_ratio,
            "R_z_median": R,
            "oppose_frac": oppose,
            "direction": "OFF" if qmean(gT_z_i) > 0 else "ON",
            "fit_sign_mean": qmean(torch.sign(gF_z_i)),
        }
    del q, out
    torch.cuda.empty_cache()
    return rows


def gumbel_matched_decomposition(handler, loss_fit, loss_total, sample, seed: int) -> dict:
    """Training-like decomposition: Gumbel-softmax with ONE fixed realization,
    used identically for the fit/AIC/total passes."""
    model = next(iter(handler.model_dict.values()))
    phy, nn = model.phy_model, model.nn_model
    params, logits, _, mopex_params, routing = build_forward(phy, nn, sample)
    gen = torch.Generator(device=logits.device).manual_seed(seed)
    u = torch.rand(logits.shape, device=logits.device, generator=gen)
    gumbel = -torch.log(-torch.log(u + EPS) + EPS)
    w_sample = F.softmax((logits + gumbel) / 1.0, dim=-1)[..., 1]

    out = run_loop(phy, sample, w_sample, mopex_params, routing)
    q = out["streamflow"]
    target = sample["target"]
    n = min(q.shape[0], target.shape[0])
    wdict = {g: out[g] for g in GATE_NAMES}
    L_total = loss_total(q[:n], target[:n], sample_ids=sample["batch_sample"], weights=wdict)
    L_fit = loss_fit(q[:n], target[:n], sample_ids=sample["batch_sample"], weights=wdict)

    def grad_of(y, x):
        g = torch.autograd.grad(y, x, retain_graph=True, allow_unused=True)[0]
        return g if g is not None else torch.zeros_like(x)

    gT_z = grad_of(L_total, params["weights"]).view(-1, 4, 2)
    gF_z = grad_of(L_fit, params["weights"]).view(-1, 4, 2)
    rows = {}
    for i, gname in enumerate(GATE_NAMES):
        gF_z_i = gF_z[:, i, 1]
        gT_z_i = gT_z[:, i, 1]
        gA_z_i = gT_z_i - gF_z_i
        rows[gname] = {
            "w_sample_median": qmed(w_sample[:, i]),
            "dL_fit_dz_med": qmed(gF_z_i.abs()),
            "dL_AIC_dz_med": qmed(gA_z_i.abs()),
            "dL_total_dz_med": qmed(gT_z_i.abs()),
            "g_total_dz_mean": qmean(gT_z_i),
            "R_z_median": qmed((gF_z_i.abs() / (gA_z_i.abs() + EPS))),
            "direction": "OFF" if qmean(gT_z_i) > 0 else "ON",
        }
    del q, out
    torch.cuda.empty_cache()
    return rows


def frozen_wint_profile(handler, loss_fit, loss_total, sample) -> dict:
    """L_fit(w_int) and L_total(w_int) with ONLY w_int overridden (no_grad)."""
    model = next(iter(handler.model_dict.values()))
    phy, nn = model.phy_model, model.nn_model
    params, logits, base_w, mopex_params, routing = build_forward(phy, nn, sample)
    base_w = base_w.detach()
    target = sample["target"]

    learned_w = float(base_w[:, 1].median())
    out = {}
    for wv in W_GRID:
        weights_on = base_w.clone()
        weights_on[:, 1] = wv
        with torch.no_grad():
            outd = run_loop(phy, sample, weights_on, mopex_params, routing)
            q = outd["streamflow"]
            n = min(q.shape[0], target.shape[0])
            wdict = {g: outd[g] for g in GATE_NAMES}
            lf = float(loss_fit(q[:n], target[:n], sample_ids=sample["batch_sample"], weights=wdict))
            lt = float(loss_total(q[:n], target[:n], sample_ids=sample["batch_sample"], weights=wdict))
        out[wv] = {"L_fit": lf, "L_total": lt}

    fit_best_w = min(W_GRID, key=lambda w: out[w]["L_fit"])
    total_best_w = min(W_GRID, key=lambda w: out[w]["L_total"])
    fit0 = out[0.0]["L_fit"]
    best_pos = min((w for w in W_GRID if w > 0), key=lambda w: out[w]["L_fit"])
    delta_fit = out[best_pos]["L_fit"] - fit0
    # local slope near learned gate value (FD on the grid neighbors)
    lo = max(w for w in W_GRID if w <= learned_w)
    hi = min(w for w in W_GRID if w >= learned_w)
    slope = 0.0 if lo == hi else (out[hi]["L_total"] - out[lo]["L_total"]) / (hi - lo)
    return {
        "learned_w_int_median": learned_w,
        "fit_best_w": fit_best_w,
        "total_best_w": total_best_w,
        "L_fit_w0": fit0,
        "L_fit_best_positive": out[best_pos]["L_fit"],
        "delta_Lfit_best_pos_minus_w0": delta_fit,
        "slope_Ltotal_near_learned": slope,
        "profile": {str(w): v for w, v in out.items()},
    }


def epoch_param_drift(handler, config, train_dataset) -> dict:
    model = next(iter(handler.model_dict.values()))
    phy, nn = model.phy_model, model.nn_model
    n_attr = len(config["model"]["nn"]["attributes"])
    attrs = train_dataset["xc_nn_norm"][0, :, -n_attr:].to(config["device"])
    with torch.no_grad():
        p = nn({"c_nn_norm": attrs})
        w = F.softmax(p["weights"].view(attrs.shape[0], 4, 2).clamp(min=-10.0, max=10.0), dim=-1)[..., 1]
        phys = phy._descale_mopex_params(p["params"])
    return {
        "gates": {g: float(w[:, i].median()) for i, g in enumerate(GATE_NAMES)},
        "params": {name: float(phys[name][:, 0].median()) for name in phy.mopex_param_names},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/config_dmopex_interceptE_S0.yaml")
    ap.add_argument("--output-root", default="results/intercept_candidates")
    ap.add_argument("--run-name", default="E_S0")
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.output_root)
    arm_dir = root / args.run_name

    cli = parse_args(["--config", args.config, "--gpu-id", str(args.gpu_id),
                      "--output-root", str(args.output_root), "--run-name", args.run_name])
    config = load_config(args.config)
    apply_runtime_overrides(config, cli, config_path=args.config)
    config["mode"] = "train"
    if str(config["device"]).startswith("cuda"):
        torch.cuda.set_device(config["device"])

    dl = _build_data_loader(config)
    td = dl.train_dataset
    loss_total = _build_loss(config, td)
    loss_fit = _build_loss(config, td)
    loss_fit.aic_alpha = 0.0
    sample = diagnostic_sample(td, config["device"])
    handler = build_handler(config)

    summary = {}
    for epoch in EPOCHS:
        try:
            handler.load_model(epoch)
        except FileNotFoundError as e:
            print(f"[warn] epoch {epoch} missing: {e}")
            continue
        for m in handler.model_dict.values():
            m.eval()
        print(f"[diag] epoch {epoch}: eval decomposition ...", flush=True)
        ev = eval_gate_decomposition(handler, loss_fit, loss_total, sample)
        gm = gumbel_matched_decomposition(handler, loss_fit, loss_total, sample, seed=1000 + epoch) \
            if epoch in (0, 1, 2) else None
        print(f"[diag] epoch {epoch}: frozen w_int profile ...", flush=True)
        prof = frozen_wint_profile(handler, loss_fit, loss_total, sample)
        drift = epoch_param_drift(handler, config, td)
        summary[epoch] = {"eval": ev, "gumbel": gm, "profile": prof, "drift": drift}
        print(f"  w_int: w={ev['w_int']['w_median']:.2e} |dL_fit/dw|={ev['w_int']['dL_fit_dw_med']:.2e} "
              f"|dL_fit/dz|={ev['w_int']['dL_fit_dz_med']:.2e} |dL_AIC/dz|={ev['w_int']['dL_AIC_dz_med']:.2e} "
              f"R={ev['w_int']['R_z_median']:.2f} dir={ev['w_int']['direction']} "
              f"|dw/dz|={ev['w_int']['dw_dz']:.2e}")
        print(f"  snow: w={ev['w_snow']['w_median']:.3f} |dL_fit/dw|={ev['w_snow']['dL_fit_dw_med']:.2e} "
              f"|dL_fit/dz|={ev['w_snow']['dL_fit_dz_med']:.2e} |dL_AIC/dz|={ev['w_snow']['dL_AIC_dz_med']:.2e} "
              f"R={ev['w_snow']['R_z_median']:.2f} dir={ev['w_snow']['direction']}")
        print(f"  profile: fit_best_w={prof['fit_best_w']} total_best_w={prof['total_best_w']} "
              f"delta_Lfit={prof['delta_Lfit_best_pos_minus_w0']:+.2e} slope={prof['slope_Ltotal_near_learned']:+.2e}")
        del ev, prof
        torch.cuda.empty_cache()

    (root / "collapse_diagnosis.json").write_text(json.dumps(summary, indent=2, default=float))

    # CSV: per-epoch per-gate eval decomposition
    rows = []
    for ep in EPOCHS:
        if ep not in summary:
            continue
        for g in GATE_NAMES:
            e = summary[ep]["eval"][g]
            rows.append({
                "epoch": ep, "gate": g,
                "w": e["w_median"], "dw_dz": e["dw_dz"],
                "dL_fit_dw": e["dL_fit_dw_med"], "dL_AIC_dw": e["dL_AIC_dw_med"],
                "dL_total_dw": e["dL_total_dw_med"],
                "dL_fit_dz": e["dL_fit_dz_med"], "dL_AIC_dz": e["dL_AIC_dz_med"],
                "dL_total_dz": e["dL_total_dz_med"],
                "chain_ratio": e["chain_gz_eq_gw_dwdz_ratio"],
                "R_z": e["R_z_median"], "oppose_frac": e["oppose_frac"],
                "direction": e["direction"],
            })
    with (arm_dir / "collapse_grad_decomposition.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"[diag] summary -> {root / 'collapse_diagnosis.json'}")
    print(f"[diag] csv     -> {arm_dir / 'collapse_grad_decomposition.csv'}")


if __name__ == "__main__":
    main()
