#!/usr/bin/env python3
"""Stage 1A: Calibrate default HBV per-basin using train window only.

For each of the 10 selected basins, calibrates default HBV parameters (12 phys)
independently.  Parameters optimized on train window; eval window used only for
final reporting.
"""
from __future__ import annotations

import argparse, csv, math, pickle, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from model.hbv_formula_static import HbvFormulaStatic

CAMELS_PATH = _PROJECT.parent.parent / "data" / "camels_dataset"
GAGE_ID_PATH = _PROJECT.parent.parent / "data" / "gage_id.npy"
OUT_BASE = _PROJECT / "validation_results" / "stage1_default_hbv_calibration_10basin"

NODE_ORDER = ["snow", "recharge", "aet", "response"]
DEFAULT_COMBO = {"snow": "S0", "recharge": "R0", "aet": "E0", "response": "Q0"}
_PARAM_BOUNDS = {
    "parBETA": [1.0, 6.0], "parFC": [50.0, 500.0], "parK0": [0.05, 0.5],
    "parK1": [0.01, 0.3], "parK2": [0.001, 0.1], "parLP": [0.3, 1.0],
    "parPERC": [0.0, 3.0], "parUZL": [0.0, 100.0], "parTT": [-2.5, 2.5],
    "parCFMAX": [1.0, 10.0], "parCFR": [0.0, 0.1], "parCWH": [0.0, 0.2],
}
_PHY_NAMES = list(_PARAM_BOUNDS)
N_PARAMS = len(_PARAM_BOUNDS)
_ALIAS_MAP = {"parBETA": "beta", "parFC": "FC", "parK0": "K_0", "parK1": "K_1",
              "parK2": "K_2", "parLP": "LP", "parUZL": "UZL", "parTT": "TT",
              "parCFMAX": "CFMAX", "parCFR": "CFR", "parCWH": "CWH", "parPERC": "PERC"}
_NODE_PARAMS = {
    "snow": ["parTT", "parCFMAX", "parCFR", "parCWH"],
    "recharge": ["parFC", "parBETA"],
    "aet": ["parFC", "parLP"],
    "response": ["parK0", "parK1", "parK2", "parUZL", "parPERC"],
}

def _to_np(x):
    if hasattr(x, 'cpu'):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def nse(qsim, qobs):
    qs = np.asarray(_to_np(qsim), dtype=np.float64)
    qo = np.asarray(_to_np(qobs), dtype=np.float64)
    mask = ~np.isnan(qo) & ~np.isnan(qs)
    if mask.sum() < 2: return float("nan")
    n = ((qs[mask] - qo[mask]) ** 2).sum()
    d = ((qo[mask] - qo[mask].mean()) ** 2).sum()
    return float(1.0 - n / d) if d > 1e-12 else float("nan")

def kge(qsim, qobs):
    qs = np.asarray(_to_np(qsim), dtype=np.float64)
    qo = np.asarray(_to_np(qobs), dtype=np.float64)
    mask = ~np.isnan(qo) & ~np.isnan(qs)
    if mask.sum() < 2: return float("nan")
    sqs, sqo = np.std(qs[mask]), np.std(qo[mask])
    if sqs < 1e-12 or sqo < 1e-12: return float("nan")
    r = np.corrcoef(qs[mask], qo[mask])[0, 1]
    return float(1.0 - math.sqrt((r - 1) ** 2 + (sqs / sqo - 1) ** 2 + (np.mean(qs[mask]) / max(np.mean(qo[mask]), 1e-12) - 1) ** 2))

def rmse(qsim, qobs):
    qs, qo = _to_np(qsim), _to_np(qobs)
    mask = ~np.isnan(qo) & ~np.isnan(qs)
    return float(np.sqrt(((qs[mask] - qo[mask]) ** 2).mean())) if mask.any() else float("nan")

def masked_mse(qsim, qobs):
    mask = ~(torch.isnan(qsim) | torch.isnan(qobs) | torch.isinf(qsim) | torch.isinf(qobs))
    if mask.sum() < 2: return torch.tensor(float("nan"), device=qsim.device)
    return F.mse_loss(qsim[mask], qobs[mask])

def flow_to_mmd(flow, area):
    return flow * 2.446575 / max(area, 1.0)

def _make_fparams(pv):
    params = {}
    for n in NODE_ORDER:
        nd = {}
        for hbv_name in _NODE_PARAMS.get(n, []):
            if hbv_name in pv:
                nd[_ALIAS_MAP[hbv_name]] = torch.as_tensor(pv[hbv_name], dtype=torch.float32)
        params[n] = nd
    if "parPERC" in pv:
        params["_perc"] = torch.as_tensor(pv["parPERC"], dtype=torch.float32)
    return params

def simulate_one(P, T, PET, fparams, warmup):
    m = HbvFormulaStatic(formula_config=DEFAULT_COMBO, warm_up=warmup, param_dicts=fparams)
    return m.simulate(P, T, PET)["Q_raw"]

# ===========================================================================
def run_calibration(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    with open(CAMELS_PATH, "rb") as f:
        forcings, target, attributes = pickle.load(f)
    gage_ids = np.load(GAGE_ID_PATH)

    # Read selected basins from previous round
    sel_csv = _PROJECT / "validation_results" / "static_router_10basin_conservative" / "recharge_anchor0.5_seed0" / "selected_basins.csv"
    if sel_csv.exists():
        basin_ids = [int(r["basin_id"]) for r in csv.DictReader(open(sel_csv))]
    else:
        basin_ids = [1013500, 1022500, 1030500, 1031500, 1047000, 1052500, 1054200, 1055000, 1057000, 1073000]
    B = len(basin_ids)
    print(f"Calibrating {B} basins with seed={args.seed}")

    # Find indices
    idx_list = [int(np.where(gage_ids == bid)[0][0]) for bid in basin_ids]

    wu = args.warmup_days
    tr_end = wu + args.train_days
    ev_end = tr_end + args.eval_days

    # Extract data
    forc = forcings[idx_list, :ev_end, :].astype(np.float32)
    targ_raw = target[idx_list, :ev_end, 0].astype(np.float32)
    areas = attributes[idx_list, 11]
    targ_mmd = np.zeros_like(targ_raw)
    for b in range(B):
        targ_mmd[b] = flow_to_mmd(targ_raw[b], areas[b])

    forcing_t = torch.from_numpy(forc).permute(1, 0, 2).to(device)
    targ_t = torch.from_numpy(targ_mmd.T).to(device)
    train_target = targ_t[wu:tr_end]
    eval_target = targ_t[tr_end:ev_end]

    # Per-basin calibration
    calib_steps_all = []
    params_rows = []
    train_metrics = []
    eval_metrics = []
    failures = []

    for b in range(B):
        print(f"\n  Basin {basin_ids[b]}...")
        bid = basin_ids[b]

        # Initialize normalized params [0,1]
        raw = torch.zeros(N_PARAMS, device=device)
        init_vals = torch.tensor([0.3, 0.4, 0.3, 0.5, 0.3, 0.5, 0.4, 0.5, 0.5, 0.3, 0.5, 0.5], device=device)
        raw = torch.logit(init_vals.clamp(1e-6, 1 - 1e-6)) + torch.randn_like(raw) * 0.1
        raw.requires_grad = True
        opt = torch.optim.Adam([raw], lr=args.lr)

        best_loss = float("inf")
        step_rows = []

        for step in range(args.steps):
            opt.zero_grad()
            norm = torch.sigmoid(raw.clamp(-5, 5))
            # Keep tensor form for gradient flow
            pv = {}
            for i, (name, (lo, hi)) in enumerate(_PARAM_BOUNDS.items()):
                pv[name] = lo + (hi - lo) * norm[i]
            fp = _make_fparams(pv)

            q_all = simulate_one(forcing_t[:, b, 0], forcing_t[:, b, 1], forcing_t[:, b, 2], fp, wu)
            q_train = q_all[:args.train_days]
            loss = masked_mse(q_train, train_target[:, b])

            if torch.isnan(loss) or torch.isinf(loss):
                failures.append({"basin_id": bid, "seed": args.seed, "step": step, "reason": f"NaN/Inf loss"})
                break

            loss.backward()
            g_norm = math.sqrt((raw.grad.norm().item() ** 2))
            torch.nn.utils.clip_grad_norm_([raw], max_norm=args.grad_clip)
            opt.step()

            step_rows.append({"basin_id": bid, "step": step, "loss": round(loss.item(), 8), "grad_norm": round(g_norm, 6)})
            if step % max(1, args.steps // 5) == 0:
                print(f"    step {step:4d}  loss={loss.item():.6f}  grad={g_norm:.4f}")

        # Final evaluation
        with torch.no_grad():
            norm_f = torch.sigmoid(raw.clamp(-5, 5))
            pv_f = {}
            for i, (name, (lo, hi)) in enumerate(_PARAM_BOUNDS.items()):
                pv_f[name] = float(lo + (hi - lo) * norm_f[i])
            fp_f = _make_fparams(pv_f)

            q_final = simulate_one(forcing_t[:, b, 0], forcing_t[:, b, 1], forcing_t[:, b, 2], fp_f, wu)

            # Train metrics
            qt = q_final[:args.train_days]
            qtr = train_target[:, b]
            n = min(len(qt), len(qtr))
            train_metrics.append({"basin_id": bid, "seed": args.seed, "nse": nse(qt[:n], qtr[:n]),
                                  "kge": kge(qt[:n], qtr[:n]), "rmse": rmse(qt[:n], qtr[:n]),
                                  "valid_ratio": 1.0})

            # Eval metrics
            qe = q_final[args.train_days:args.train_days + args.eval_days]
            qer = eval_target[:, b]
            ne = min(len(qe), len(qer))
            eval_nse_val = nse(qe[:ne], qer[:ne])
            eval_metrics.append({"basin_id": bid, "seed": args.seed, "eval_nse": eval_nse_val,
                                  "eval_kge": kge(qe[:ne], qer[:ne]), "eval_rmse": rmse(qe[:ne], qer[:ne]),
                                  "valid_eval_ratio": 1.0})

            print(f"    train_NSE={train_metrics[-1]['nse']:.4f}, eval_NSE={eval_nse_val:.4f}")

            for i, (name, (lo, hi)) in enumerate(_PARAM_BOUNDS.items()):
                params_rows.append({"basin_id": bid, "seed": args.seed, "parameter_name": name,
                                    "normalized_value": round(float(norm_f[i]), 6),
                                    "physical_value": round(float(lo + (hi - lo) * norm_f[i]), 6)})

        calib_steps_all.extend(step_rows)

    # Write outputs
    _w(calib_steps_all, out_dir / "calibration_steps.csv",
       ["basin_id", "step", "loss", "grad_norm"])
    _w(params_rows, out_dir / "params_default_calibrated.csv",
       ["basin_id", "seed", "parameter_name", "normalized_value", "physical_value"])
    _w(train_metrics, out_dir / "metrics_train_default.csv",
       ["basin_id", "seed", "nse", "kge", "rmse", "valid_ratio"])
    _w(eval_metrics, out_dir / "metrics_eval_default.csv",
       ["basin_id", "seed", "eval_nse", "eval_kge", "eval_rmse", "valid_eval_ratio"])
    _w(failures, out_dir / "failures.csv", ["basin_id", "seed", "step", "reason"])

    mean_ev = np.mean([m["eval_nse"] for m in eval_metrics if not np.isnan(m["eval_nse"])])
    report = [f"# Default HBV Calibration (seed={args.seed})",
              f"Basins: {B}, steps: {args.steps}",
              f"Mean eval NSE: {mean_ev:.4f}",
              f"Failures: {len(failures)}"]
    (out_dir / "run_report.md").write_text("\n".join(report))
    print(f"\nDone. Mean eval NSE: {mean_ev:.4f}")
    return True

def _w(rows, path, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-days", type=int, default=365)
    ap.add_argument("--train-days", type=int, default=60)
    ap.add_argument("--eval-days", type=int, default=60)
    ap.add_argument("--output-dir", default=str(OUT_BASE / "seed0"))
    args = ap.parse_args()
    run_calibration(args)
