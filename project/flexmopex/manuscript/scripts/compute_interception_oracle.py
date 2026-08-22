#!/usr/bin/env python3
"""Agent B — correct basin-specific total-objective oracle for Candidate E-S0.

Uses the exact training-objective normalization (NseDynAicBatchLoss):

  L_total = L_fit + L_AIC
  L_fit    = mean over ALL valid (t,b) pairs of (Q-O)^2 / (std_train_b + 0.1)^2
           = sum_b S_b(w_b) / N,   S_b(w) = sum over valid t of norm. res^2,
           N = sum_b n_valid_b  (assignment-independent -> basin-separable)
  L_AIC    = aic_alpha * sum_gates mean_b(w_g) * cost_g
           = 0.01 * (2*mean(w_phen) + 2*mean(w_int) + 2*mean(w_snow) + 1*mean(w_sub))

With all other gates/params frozen, the per-basin total contribution is

  T_b(w) = fit_b(w) * n_valid_b / N  +  0.01*2*w / 671

and the oracle is w_b* = argmin over {0,.1,.25,.5,.75,1.0}.  This is the exact
full-population minimizer over the product grid (the objective is separable;
no fake per-basin decomposition).

Verification: one forward with the oracle assignment evaluated with the exact
NseDynAicBatchLoss on the full eval window, compared to the decomposition.
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
from project.flexmopex.local_model_handler import FlexMopexModelHandler  # noqa: E402

W_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
AIC_ALPHA = 0.01
COST_WINT = 2.0
B = 671
EPS = 1e-6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/config_dmopex_interceptE_S0.yaml")
    ap.add_argument("--output-root", default="results/intercept_candidates")
    ap.add_argument("--run-name", default="E_S0")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=10)
    args = ap.parse_args()
    root = Path(args.output_root)
    arm_dir = root / args.run_name
    ep = args.epoch

    cli = parse_args(["--config", args.config, "--gpu-id", str(args.gpu_id),
                      "--output-root", str(args.output_root), "--run-name", args.run_name])
    cfg = load_config(args.config)
    apply_runtime_overrides(cfg, cli, config_path=args.config)
    cfg["mode"] = "train"
    if str(cfg["device"]).startswith("cuda"):
        torch.cuda.set_device(cfg["device"])

    dl = _build_data_loader(cfg)
    td, ed = dl.train_dataset, dl.eval_dataset
    dev = cfg["device"]
    n_attr = td["xc_nn_norm"].shape[-1] - 3
    attrs = td["xc_nn_norm"][0, :, -n_attr:].to(dev)

    # ---- learned gates + params at the checkpoint ----
    handler = FlexMopexModelHandler(cfg, verbose=False)
    handler.load_model(ep)
    for m in handler.model_dict.values():
        m.eval()
    model = next(iter(handler.model_dict.values()))
    phy, nn = model.phy_model, model.nn_model
    with torch.no_grad():
        p = nn({"c_nn_norm": attrs})
        w_learn = F.softmax(p["weights"].view(B, 4, 2).clamp(-10, 10), dim=-1)[..., 1]
        mopex_params = phy._descale_mopex_params(p["params"])
        routing = phy._descale_routing_params(p["gamma_uh"])

    # ---- scored-window convention (same as the basin sweep): the model's
    # internal 365-day warmup is inside the loop, so the routed output rows
    # 0..n_out-1 correspond to eval-record days 365..365+n_out-1 and are
    # compared against target[365:365+n_out] (trainer alignment).  The sweep
    # currently uses Qs = Qr, n_out = Qr rows = record_length - 365 = 5114.
    n_out = int(ed["x_phy"].shape[0]) - 365
    print(f"[oracle] scored days n_out = {n_out} (record {ed['x_phy'].shape[0]})")

    y_ev = ed["target"][:, :, 0].cpu().numpy()
    y_scored = y_ev[365:365 + n_out, :]      # aligned with routed rows 0..n_out-1
    n_valid_b = np.sum(~np.isnan(y_scored), axis=0).astype(float)     # (671,)
    N = float(n_valid_b.sum())
    print(f"[oracle] N = {N:.0f} valid (t,b) pairs; per-basin valid min/med/max = "
          f"{n_valid_b.min():.0f}/{np.median(n_valid_b):.0f}/{n_valid_b.max():.0f}")

    # ---- fit grid from the sweep dump ----
    grid = pd.read_csv(arm_dir / f"basin_fitgrid_ep{ep}.csv")
    grid = grid[grid["epoch"] == ep].pivot_table(index="basin_idx", columns="w", values="fit")
    grid = grid.reindex(index=range(B), columns=W_GRID)               # NaN -> excluded basins
    learned = pd.read_csv(arm_dir / "basin_benefit.csv")
    learned = learned[learned["epoch"] == ep].set_index("basin_idx").sort_index()

    # ---- per-basin oracle: T_b(w) = fit_b(w)*n_valid_b/N + aic*2*w/671 ----
    aic_per_unit = AIC_ALPHA * COST_WINT / B                          # 2.98e-5 per unit w
    T = {}
    for w in W_GRID:
        T[w] = grid[w].to_numpy() * n_valid_b / N + aic_per_unit * w
    w_star = np.full(B, np.nan)
    for b in range(B):
        vals = {w: T[w][b] for w in W_GRID}
        if not np.isfinite(grid[0.1].iloc[b]) and not np.isfinite(grid[1.0].iloc[b]):
            continue                                                   # no valid fit data
        w_star[b] = min(vals, key=vals.get)

    def L_total(assign: np.ndarray) -> tuple[float, float, float]:
        # fit term uses the grid value at the nearest grid point (exact for
        # oracle/all-off/all-1; nearest-point for the learned assignment whose
        # values are off-grid); AIC uses the true assignment mean.
        grid_arr = np.array(W_GRID)
        fit_term = 0.0
        for b in range(B):
            w = assign[b]
            if not np.isfinite(w):
                continue
            wg = grid_arr[np.argmin(np.abs(grid_arr - w))]
            fw = grid[wg].iloc[b]
            if np.isfinite(fw):
                fit_term += fw * n_valid_b[b] / N
        w_int_mean = float(np.nanmean(assign))
        aic = AIC_ALPHA * (2.0 * w_int_mean
                           + 2.0 * float(w_learn[:, 0].mean())
                           + 2.0 * float(w_learn[:, 2].mean())
                           + 1.0 * float(w_learn[:, 3].mean()))
        return fit_term + aic, fit_term, aic

    learned_w = learned["learned_w_int"].to_numpy().clip(0, 1)
    L0, f0, a0 = L_total(np.zeros(B))
    L1, f1, a1 = L_total(np.ones(B))
    Lo, fo, ao = L_total(w_star)
    Llearn, flearn, alearn = L_total(learned_w)

    # ---- overlap + false negatives ----
    dNSE = learned["delta_NSE_max"].to_numpy()
    gt001 = dNSE > 0.01
    oracle_pos = w_star > 0
    overlap = float(np.mean(gt001[oracle_pos])) if oracle_pos.sum() > 0 else np.nan
    fn_lt001 = float(np.mean(learned_w[oracle_pos] < 0.01)) if oracle_pos.sum() > 0 else np.nan
    fn_lt01 = float(np.mean(learned_w[oracle_pos] < 0.1)) if oracle_pos.sum() > 0 else np.nan

    result = {
        "epoch": ep,
        "normalization": "L_fit = sum_b fit_b(w)*n_valid_b/N (N = sum n_valid_b); "
                         "L_AIC = 0.01*(2*mean(w_phen)+2*mean(w_int)+2*mean(w_snow)+1*mean(w_sub))",
        "basin_separable": True,
        "n_valid_min_med_max": [float(n_valid_b.min()), float(np.median(n_valid_b)), float(n_valid_b.max())],
        "N": N,
        "n_scored_days": int(n_out),
        "fraction_oracle_w_star_gt0": float(np.nanmean(w_star > 0)),
        "fraction_w_star_ge01": float(np.nanmean(w_star >= 0.1)),
        "fraction_w_star_ge05": float(np.nanmean(w_star >= 0.5)),
        "fraction_w_star_eq1": float(np.nanmean(w_star == 1.0)),
        "w_star_distribution": {f"w={w:g}": float(np.nanmean(w_star == w)) for w in W_GRID},
        "L_total": {"all_off": L0, "all_1": L1, "oracle": Lo, "learned_E_S0": Llearn},
        "L_fit": {"all_off": f0, "all_1": f1, "oracle": fo, "learned_E_S0": flearn},
        "L_AIC": {"all_off": a0, "all_1": a1, "oracle": ao, "learned_E_S0": alearn},
        "oracle_improvement_vs_all_off": L0 - Lo,
        "oracle_improvement_vs_learned": Llearn - Lo,
        "overlap_oracle_pos_with_dNSE_gt001": overlap,
        "frac_dNSE_gt001_within_oracle_pos": overlap,
        "frac_dNSE_gt001_overall": float(np.nanmean(gt001)),
        "learned_E_S0_false_negative_frac_lt001": fn_lt001,
        "learned_E_S0_false_negative_frac_lt01": fn_lt01,
        "n_oracle_pos": int(np.nansum(oracle_pos)),
    }

    # ---- exact full-population verification forward (also yields oracle NSE) ----
    cfg["model"]["phy"]["disable_compile"] = True
    loss_fn = _build_loss(cfg, td)
    w_on = torch.tensor(w_star, dtype=torch.float32, device=dev)
    w_on_full = w_learn.clone()
    w_on_full[:, 1] = w_on
    with torch.no_grad():
        P, T, PET, doy, n_steps, _ = phy._prepare_forcings(
            {"x_phy": ed["x_phy"].to(dev), "doy": ed["doy"].to(dev)})
        Q = phy._run_weighted_loop(P, T, PET, doy, mopex_params, w_on_full, n_steps, B)
        Qr = phy._apply_routing(Q.mean(-1), routing)
    q = Qr[:n_out]
    tgt = ed["target"][365:365 + n_out].to(dev)
    wdict = {}
    for i, g in enumerate(["w_phen", "w_int", "w_snow", "w_sub"]):
        wdict[g] = w_on_full[:, i].view(1, -1, 1).expand(n_out, -1, -1)
    L_exact = float(loss_fn(q, tgt, sample_ids=np.arange(B, dtype=np.int64), weights=wdict))
    result["exact_loss_verification"] = {
        "L_total_exact_forward": L_exact,
        "L_total_decomposition": Lo,
        "abs_diff": abs(L_exact - Lo),
    }

    # oracle table
    with (arm_dir / "oracle_table.csv").open("w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["basin_idx", "w_star", "learned_w_int", "dNSE_max", "fit_w0", "n_valid"])
        for b in range(B):
            wcsv.writerow([b, w_star[b], learned["learned_w_int"].iloc[b],
                           learned["delta_NSE_max"].iloc[b],
                           grid[0.0].iloc[b], n_valid_b[b]])
    (arm_dir / "AGENT_B_ORACLE.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))
    print(f"[oracle] table -> {arm_dir}/oracle_table.csv ; json -> AGENT_B_ORACLE.json")


if __name__ == "__main__":
    main()
