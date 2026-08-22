#!/usr/bin/env python3
"""Phase 3: Parameter Compensation and Finite Interception Benefit Audit (Testing Hypothesis B).

Tracks the primary fixed cohort of 103 state-conditional Oracle-positive interception basins
(defined at R8 epoch-2) across all available R8 checkpoints (ep1, 2, 3, 4, 5, 10):

  3.1 Evaluates finite interception benefit along trajectory:
        - NSE and fit loss with w_int = 0
        - NSE and fit loss with w_int = 1
        - NSE and fit loss at optimal grid w* in {0.0, 0.1, 0.25, 0.5, 0.75, 1.0}
        - Delta_fit = fit(0) - min_{w>0} fit(w)
        - Delta_NSE = max_{w>0} NSE(w) - NSE(0)
        - Current learned w_int
  3.2 Tracks physical parameter evolution for all 12 MOPEX parameters + 2 routing parameters:
        - Sb1, tw, tu, Se, tc, ddf, tcrit, Sb2, kappa, phi, tmin, tmax, rout_a, rout_b
        - (Averaged across nmul=16 realizations per basin)
        - Cohort-level shifts, distributions, and Spearman correlations with Delta_fit.

Outputs: results/root_cause_r13/compensation_audit_summary.json
         results/root_cause_r13/compensation_benefit_trajectory.csv
         results/root_cause_r13/compensation_parameter_trajectory.csv
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
from scipy.stats import spearmanr

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config  # noqa: E402
from project.flexmopex.run_model import (  # noqa: E402
    apply_runtime_overrides, parse_args, _build_data_loader,
)
from scripts.diagnose_wint_collapse import build_handler, build_forward  # noqa: E402

OUT_DIR = Path("results/root_cause_r13")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSES = ["w_phen", "w_int", "w_snow", "w_sub"]
GATE_IDX = {"w_phen": 0, "w_int": 1, "w_snow": 2, "w_sub": 3}
W_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
EPS = 1e-6


def main() -> None:
    # 1. Load primary fixed cohort (103 basins from R8 ep2 oracle)
    r8_orc_file = "results/intercept_aicdelay/E_S0_aicdelay2/R9_separability/oracle_state_conditional.csv"
    r8_rows = list(csv.DictReader(open(r8_orc_file)))
    r8_ep2 = [r for r in r8_rows if r["epoch"] == "2" and r["process"] == "w_int"]
    cohort_103 = [int(r["basin_idx"]) for r in sorted(r8_ep2, key=lambda x: int(x["basin_idx"])) if float(r["w_star"]) > 0]
    print(f"[Phase 3] Primary Fixed Cohort: {len(cohort_103)} basins (R8 ep2 w_int oracle > 0)")

    # 2. Setup model and data loader
    cfg_path = "conf/config_dmopex_interceptE_S0_aicdelay2.yaml"
    c = load_config(cfg_path)
    c_cli = parse_args(["--config", cfg_path, "--gpu-id", "0",
                        "--output-root", "results/intercept_aicdelay",
                        "--run-name", "E_S0_aicdelay2"])
    apply_runtime_overrides(c, c_cli, config_path=cfg_path)
    c["mode"] = "train"
    c["model"]["phy"]["disable_compile"] = True
    dl = _build_data_loader(c)

    td, ed = dl.train_dataset, dl.eval_dataset
    B = td["x_phy"].shape[1]
    n_attr = td["xc_nn_norm"].shape[-1] - 3
    attrs = td["xc_nn_norm"][0, :, -n_attr:].cuda()
    std_train = (np.nanstd(td["target"][:, :, 0].cpu().numpy(), axis=0) + 0.1).astype(np.float32)
    n_out = int(ed["x_phy"].shape[0]) - 365
    y_ev = ed["target"][365:365 + n_out, :, 0].cpu().numpy()

    handler = build_handler(c)
    epochs_to_audit = [1, 2, 3, 4, 5, 10]

    benefit_rows = []
    param_rows = []
    summary_by_epoch = {}

    for ep in epochs_to_audit:
        handler.load_model(ep)
        for m in handler.model_dict.values():
            m.eval()
        model = next(iter(handler.model_dict.values()))
        phy, nn = model.phy_model, model.nn_model

        with torch.no_grad():
            params_raw = nn({"c_nn_norm": attrs})
            w_learn = torch.softmax(params_raw["weights"].view(B, 4, 2).clamp(-10, 10), dim=-1)[..., 1]
            mopex_params = phy._descale_mopex_params(params_raw["params"])
            routing = phy._descale_routing_params(params_raw["gamma_uh"])

        # Extract descaled physical parameters (mean over nmul=16 per basin)
        basin_params = {}
        for p_name, p_tensor in mopex_params.items():
            basin_params[p_name] = p_tensor.mean(dim=-1).cpu().numpy()  # [671]
        for r_name, r_tensor in routing.items():
            basin_params[r_name] = r_tensor.cpu().numpy()                # [671]

        # Record parameters for cohort
        for b in cohort_103:
            row_p = {
                "epoch": ep,
                "basin_idx": b,
                "learned_w_int": float(w_learn[b, 1].item()),
            }
            for k, arr in basin_params.items():
                row_p[k] = float(arr[b])
            param_rows.append(row_p)

        # Compute finite benefit grid for cohort
        S = len(W_GRID)
        col = GATE_IDX["w_int"]
        w_on = w_learn.detach().clone().repeat(S, 1)
        for s in range(S):
            w_on[s * B:(s + 1) * B, col] = W_GRID[s]

        params_rep = {k: v.detach().repeat(S, 1) for k, v in mopex_params.items()}
        routing_rep = {k: v.detach().repeat(S) for k, v in routing.items()}
        sample_rep = {"x_phy": ed["x_phy"].repeat(1, S, 1).cuda(),
                      "doy": ed["doy"].repeat(1, S, 1).cuda()}

        with torch.no_grad():
            P, T, PET, doy_r, n_steps, _ = phy._prepare_forcings(sample_rep)
            Q = phy._run_weighted_loop(P, T, PET, doy_r, params_rep, w_on, n_steps, B * S)
            Qr = phy._apply_routing(Q.mean(-1), routing_rep).cpu().numpy()[:, :, 0]

        Qs = Qr[:n_out].reshape(n_out, S, B)  # [n_out, S, B]

        fit_grid = np.full((B, S), np.nan)
        nse_grid = np.full((B, S), np.nan)

        for b in range(B):
            v = ~np.isnan(y_ev[:, b])
            if v.sum() < 30:
                continue
            o = y_ev[v, b]
            ss = Qs[v, :, b]  # [n_valid, S]
            fit_grid[b, :] = np.nanmean((ss - o[:, None]) ** 2, axis=0) / (std_train[b] ** 2)

            # Compute NSE for each w
            ss_tot = np.sum((o - np.mean(o)) ** 2)
            for s_idx in range(S):
                ss_res = np.sum((ss[:, s_idx] - o) ** 2)
                nse_grid[b, s_idx] = 1.0 - (ss_res / (ss_tot + EPS))

        # Cohort benefit metrics
        cohort_dNSE = []
        cohort_dfit = []
        cohort_nse0 = []
        cohort_nse1 = []
        cohort_fit0 = []
        cohort_fit1 = []
        cohort_w_opt = []

        for b in cohort_103:
            f0 = fit_grid[b, 0]
            f1 = fit_grid[b, -1]
            n0 = nse_grid[b, 0]
            n1 = nse_grid[b, -1]

            best_s = int(np.nanargmin(fit_grid[b, :]))
            w_opt = W_GRID[best_s]

            dfit = f0 - np.nanmin(fit_grid[b, 1:])  # fit improvement from w>0
            dnse = np.nanmax(nse_grid[b, 1:]) - n0   # NSE improvement from w>0

            cohort_dfit.append(dfit)
            cohort_dNSE.append(dnse)
            cohort_nse0.append(n0)
            cohort_nse1.append(n1)
            cohort_fit0.append(f0)
            cohort_fit1.append(f1)
            cohort_w_opt.append(w_opt)

            benefit_rows.append({
                "epoch": ep,
                "basin_idx": b,
                "learned_w_int": float(w_learn[b, 1].item()),
                "nse_w0": n0,
                "nse_w1": n1,
                "nse_w_opt": nse_grid[b, best_s],
                "dNSE_max": dnse,
                "fit_w0": f0,
                "fit_w1": f1,
                "fit_w_opt": fit_grid[b, best_s],
                "fit_improvement": dfit,
                "w_opt_grid": w_opt,
            })

        summary_by_epoch[str(ep)] = {
            "epoch": ep,
            "cohort_n": len(cohort_103),
            "learned_w_median": float(np.median([float(w_learn[b, 1].item()) for b in cohort_103])),
            "learned_w_mean": float(np.mean([float(w_learn[b, 1].item()) for b in cohort_103])),
            "fit_improvement_median": float(np.nanmedian(cohort_dfit)),
            "fit_improvement_mean": float(np.nanmean(cohort_dfit)),
            "dNSE_max_median": float(np.nanmedian(cohort_dNSE)),
            "dNSE_max_mean": float(np.nanmean(cohort_dNSE)),
            "nse_w0_median": float(np.nanmedian(cohort_nse0)),
            "nse_w1_median": float(np.nanmedian(cohort_nse1)),
            "frac_positive_fit_imp": float(np.mean(np.array(cohort_dfit) > 0)),
            "frac_dNSE_gt001": float(np.mean(np.array(cohort_dNSE) > 0.01)),
            "frac_w_opt_gt0": float(np.mean(np.array(cohort_w_opt) > 0)),
        }

        print(f"[Phase 3 Ep {ep:>2d}] Cohort learned w_int = {summary_by_epoch[str(ep)]['learned_w_median']:.4f} | "
              f"Fit Imp median = {summary_by_epoch[str(ep)]['fit_improvement_median']:.4f} (mean={summary_by_epoch[str(ep)]['fit_improvement_mean']:.4f}) | "
              f"dNSE median = {summary_by_epoch[str(ep)]['dNSE_max_median']:.4f} (frac>0.01 = {summary_by_epoch[str(ep)]['frac_dNSE_gt001']*100:.1f}%)")

    # Parameter shifts from Ep 2 to Ep 10
    df_p = pd.DataFrame(param_rows)
    p_ep2 = df_p[df_p["epoch"] == 2].set_index("basin_idx")
    p_ep10 = df_p[df_p["epoch"] == 10].set_index("basin_idx")
    
    df_b = pd.DataFrame(benefit_rows)
    b_ep2 = df_b[df_b["epoch"] == 2].set_index("basin_idx")
    b_ep10 = df_b[df_b["epoch"] == 10].set_index("basin_idx")
    delta_fit_change = (b_ep10["fit_improvement"] - b_ep2["fit_improvement"]).values

    param_shifts = {}
    for p_name in list(mopex_params.keys()) + list(routing.keys()):
        v2 = p_ep2[p_name].values
        v10 = p_ep10[p_name].values
        diff = v10 - v2
        r_corr, p_val = spearmanr(diff, delta_fit_change)
        param_shifts[p_name] = {
            "mean_ep2": float(np.mean(v2)),
            "median_ep2": float(np.median(v2)),
            "std_ep2": float(np.std(v2)),
            "mean_ep10": float(np.mean(v10)),
            "median_ep10": float(np.median(v10)),
            "std_ep10": float(np.std(v10)),
            "mean_shift_ep10_minus_ep2": float(np.mean(diff)),
            "median_shift": float(np.median(diff)),
            "spearman_with_delta_fit_imp": float(r_corr) if np.isfinite(r_corr) else 0.0,
            "spearman_p_val": float(p_val) if np.isfinite(p_val) else 1.0,
        }

    full_summary = {
        "trajectory_by_epoch": summary_by_epoch,
        "parameter_shifts_ep2_to_ep10": param_shifts,
    }

    sum_path = OUT_DIR / "compensation_audit_summary.json"
    sum_path.write_text(json.dumps(full_summary, indent=2))

    df_b.to_csv(OUT_DIR / "compensation_benefit_trajectory.csv", index=False)
    df_p.to_csv(OUT_DIR / "compensation_parameter_trajectory.csv", index=False)

    print(f"\n[Phase 3 Complete] Summary written to {sum_path}")


if __name__ == "__main__":
    main()
