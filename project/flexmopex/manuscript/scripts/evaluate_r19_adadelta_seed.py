#!/usr/bin/env python3
"""Evaluate R19 Unified Adadelta for a given seed."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config
from project.flexmopex.run_model import apply_runtime_overrides, parse_args, _build_data_loader
from scripts.diagnose_wint_collapse import build_handler, build_forward, run_loop

PROCESSES = ["w_phen", "w_int", "w_snow", "w_sub"]
GATE_IDX  = {"w_phen": 0, "w_int": 1, "w_snow": 2, "w_sub": 3}
COSTS     = {"w_phen": 2.0, "w_int": 2.0, "w_snow": 2.0, "w_sub": 1.0}
AIC_ALPHA = 0.01
W_GRID    = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
EPS       = 1e-6


def evaluate(seed: int, cfg_path: str, out_root: Path, run_name: str, dev: str = "cuda:0"):
    c = load_config(cfg_path)
    c_cli = parse_args(["--config", cfg_path, "--gpu-id", "0",
                        "--output-root", str(out_root.parent),
                        "--run-name", run_name])
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
    handler.load_model(10)
    for m in handler.model_dict.values():
        m.eval()
    model = next(iter(handler.model_dict.values()))
    phy, nn = model.phy_model, model.nn_model

    with torch.no_grad():
        params_raw = nn({"c_nn_norm": attrs})
        w_learn = F.softmax(params_raw["weights"].view(B, 4, 2).clamp(-10, 10), dim=-1)[..., 1]
        mopex_params = phy._descale_mopex_params(params_raw["params"])
        routing = phy._descale_routing_params(params_raw["gamma_uh"])
        base_w = w_learn.detach().clone()
        sample = {"x_phy": ed["x_phy"].to(dev), "doy": ed["doy"].to(dev), "c_nn_norm": attrs}
        p, logits, w_on, m_p, r_p = build_forward(phy, nn, sample)
        out = run_loop(phy, sample, w_on, m_p, r_p)
        q_stream = out["streamflow"][:n_out, :, 0].cpu().numpy()

    nses = []
    for b in range(B):
        v = ~np.isnan(y_ev[:, b])
        if v.sum() < 30: continue
        o = y_ev[v, b]; s = q_stream[v, b]
        ss_res = np.sum((s - o)**2); ss_tot = np.sum((o - o.mean())**2)
        nses.append(1.0 - ss_res / ss_tot if ss_tot > EPS else np.nan)
    nses = np.array(nses)
    median_nse = float(np.nanmedian(nses))
    mean_nse   = float(np.nanmean(nses))
    frac_pos   = float(np.mean(nses > 0))
    frac_05    = float(np.mean(nses > 0.5))
    print(f"Seed {seed} (Ep10): Median NSE={median_nse:.4f} | Mean NSE={mean_nse:.4f} | >0: {frac_pos:.3f} | >0.5: {frac_05:.3f}")

    S = len(W_GRID)
    process_eval = {}
    for proc in PROCESSES:
        col = GATE_IDX[proc]
        w_on2 = base_w.repeat(S, 1)
        for s in range(S):
            w_on2[s * B:(s + 1) * B, col] = W_GRID[s]
        params_rep  = {k: v.repeat(S, 1) for k, v in mopex_params.items()}
        routing_rep = {k: v.repeat(S)    for k, v in routing.items()}
        sample_rep  = {"x_phy": ed["x_phy"].repeat(1, S, 1).to(dev),
                       "doy":   ed["doy"].repeat(1, S, 1).to(dev),
                       "c_nn_norm": attrs.repeat(S, 1).to(dev)}
        with torch.no_grad():
            P, T, PET, doy, n_steps, _ = phy._prepare_forcings(sample_rep)
            Q  = phy._run_weighted_loop(P, T, PET, doy, params_rep, w_on2, n_steps, B * S)
            Qr = phy._apply_routing(Q.mean(-1), routing_rep).cpu().numpy()[:, :, 0]
        Qr = Qr[:, :B * S].reshape(Qr.shape[0], S, B)
        Qs = np.transpose(Qr, (0, 2, 1))[:n_out]

        fit_grid = np.full((B, S), np.nan)
        for b in range(B):
            v = ~np.isnan(y_ev[:, b])
            if v.sum() < 30: continue
            o = y_ev[v, b]; ss = Qs[v, b, :]
            fit_grid[b, :] = np.mean((ss - o[:, None])**2 / (std_train[b]**2), axis=0)

        cost = COSTS[proc]; aic_unit = AIC_ALPHA * cost / B
        w_star = np.full(B, np.nan)
        for b in range(B):
            vals = [fit_grid[b, s] * n_valid_b[b] / N + aic_unit * W_GRID[s] for s in range(S)]
            if np.isfinite(vals).any():
                w_star[b] = W_GRID[int(np.nanargmin(vals))]

        w_l   = base_w[:, col].cpu().numpy()
        valid = np.isfinite(w_star)
        orc_pos  = (w_star[valid] > 0)
        pos_mean  = float(np.mean(w_l[valid][orc_pos]))  if orc_pos.any()  else 0.0
        zero_mean = float(np.mean(w_l[valid][~orc_pos])) if (~orc_pos).any() else 0.0
        sp_corr, _ = spearmanr(w_l[valid], w_star[valid])
        process_eval[proc] = {
            "n_oracle_pos": int(orc_pos.sum()),
            "mean": float(np.mean(w_l[valid])),
            "std":  float(np.std(w_l[valid])),
            "min":  float(np.min(w_l[valid])),
            "max":  float(np.max(w_l[valid])),
            "pos_mean":  pos_mean,
            "zero_mean": zero_mean,
            "Delta": pos_mean - zero_mean,
            "spearman":  float(sp_corr),
        }
        d = process_eval[proc]
        print(f"  [{proc}] n_pos={d['n_oracle_pos']} | mean={d['mean']:.4f} | std={d['std']:.4f} | "
              f"pos={d['pos_mean']:.4f} zero={d['zero_mean']:.4f} Δ={d['Delta']:+.4f} ρ={d['spearman']:+.4f}")

    result = {"seed": seed, "median_nse": median_nse, "mean_nse": mean_nse,
              "frac_pos": frac_pos, "frac_05": frac_05, "processes": process_eval}
    out_file = out_root / f"eval_summary_seed{seed}.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(f"Saved → {out_file}")
    return result


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 43
    cfg_map = {
        43: ("conf/config_dmopex_interceptE_S0_r19_unified_adadelta_seed43.yaml",
             Path("results/intercept_r19/E_S0_r19_unified_adadelta/seed_43"), "seed_43"),
        44: ("conf/config_dmopex_interceptE_S0_r19_unified_adadelta_seed44.yaml",
             Path("results/intercept_r19/E_S0_r19_unified_adadelta/seed_44"), "seed_44"),
    }
    cfg_path, out_root, run_name = cfg_map[seed]
    evaluate(seed, cfg_path, out_root, run_name)
