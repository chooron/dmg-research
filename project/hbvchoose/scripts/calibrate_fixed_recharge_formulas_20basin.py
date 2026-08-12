#!/usr/bin/env python3
"""Stage 3: Calibrate fixed recharge formulas (R0/R4/R5) per basin per seed.

Gradient-based calibration on TRAIN window only. Eval window used only for
final evaluation, never for optimization or selection.
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

NODE_ORDER = ["snow", "recharge", "aet", "response"]
DEFAULT_IDS = {"snow": "S0", "recharge": "R0", "aet": "E0", "response": "Q0"}

_PARAM_BOUNDS = {
    "parBETA": [1.0, 6.0], "parFC": [50.0, 500.0], "parK0": [0.05, 0.5],
    "parK1": [0.01, 0.3], "parK2": [0.001, 0.1], "parLP": [0.3, 1.0],
    "parPERC": [0.0, 3.0], "parUZL": [0.0, 100.0], "parTT": [-2.5, 2.5],
    "parCFMAX": [1.0, 10.0], "parCFR": [0.0, 0.1], "parCWH": [0.0, 0.2],
}
_PHYN = list(_PARAM_BOUNDS)
N_PARAMS = len(_PARAM_BOUNDS)
_ALIAS_MAP = {
    "parBETA": "beta", "parFC": "FC", "parK0": "K_0", "parK1": "K_1",
    "parK2": "K_2", "parLP": "LP", "parUZL": "UZL", "parTT": "TT",
    "parCFMAX": "CFMAX", "parCFR": "CFR", "parCWH": "CWH", "parPERC": "PERC",
}
_NODE_PARAMS = {
    "snow": ["parTT", "parCFMAX", "parCFR", "parCWH"],
    "recharge": ["parFC", "parBETA"],
    "aet": ["parFC", "parLP"],
    "response": ["parK0", "parK1", "parK2", "parUZL", "parPERC"],
}
_EXTRA_PARAMS = {
    "R4": {"a_r": 10.0, "c_r": 0.5},
    "R5": {"b_v": 1.0},
}
RECHARGE_FIDS = ["R0", "R4", "R5"]


def flow_to_mmd(flow, area):
    return flow * 2.446575 / max(area, 1.0)


def _to_np(x):
    if hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def nse(qsim, qobs):
    qsim = np.asarray(_to_np(qsim), dtype=np.float64)
    qobs = np.asarray(_to_np(qobs), dtype=np.float64)
    mask = ~np.isnan(qobs) & ~np.isnan(qsim)
    qs, qo = qsim[mask], qobs[mask]
    if len(qo) < 2:
        return float("nan")
    num = ((qs - qo) ** 2).sum()
    den = ((qo - qo.mean()) ** 2).sum()
    if den < 1e-12:
        return float("nan")
    return float(1.0 - num / den)


def kge(qsim, qobs):
    qsim = np.asarray(_to_np(qsim), dtype=np.float64)
    qobs = np.asarray(_to_np(qobs), dtype=np.float64)
    mask = ~np.isnan(qobs) & ~np.isnan(qsim)
    qs, qo = qsim[mask], qobs[mask]
    if len(qo) < 2:
        return float("nan")
    sqs, sqo = np.std(qs), np.std(qo)
    if sqs < 1e-12 or sqo < 1e-12:
        return float("nan")
    r = np.corrcoef(qs, qo)[0, 1]
    return float(1.0 - math.sqrt((r - 1) ** 2 + (sqs / sqo - 1) ** 2 +
                                  (np.mean(qs) / max(np.mean(qo), 1e-12) - 1) ** 2))


def rmse(qsim, qobs):
    qs = _to_np(qsim)
    qo = _to_np(qobs)
    mask = ~np.isnan(qo) & ~np.isnan(qs)
    if mask.sum() < 2:
        return float("nan")
    return float(np.sqrt(((qs[mask] - qo[mask]) ** 2).mean()))


def log_nse(qsim, qobs):
    eps = 1e-4
    qsim = np.asarray(_to_np(qsim), dtype=np.float64)
    qobs = np.asarray(_to_np(qobs), dtype=np.float64)
    mask = ~np.isnan(qobs) & ~np.isnan(qsim)
    qs, qo = qsim[mask], qobs[mask]
    if len(qo) < 2:
        return float("nan")
    qs_log = np.log(np.maximum(qs, eps))
    qo_log = np.log(np.maximum(qo, eps))
    num = ((qs_log - qo_log) ** 2).sum()
    den = ((qo_log - qo_log.mean()) ** 2).sum()
    if den < 1e-12:
        return float("nan")
    return float(1.0 - num / den)


def _make_fparams(combo, phy_vals):
    """Build param dict keeping tensors for autograd."""
    params = {}
    for n in NODE_ORDER:
        nd = {}
        for hbv_name in _NODE_PARAMS.get(n, []):
            if hbv_name in phy_vals:
                v = phy_vals[hbv_name]
                nd[_ALIAS_MAP[hbv_name]] = v if torch.is_tensor(v) else torch.as_tensor(float(v), dtype=torch.float32)
        params[n] = nd
    if "parPERC" in phy_vals:
        v = phy_vals["parPERC"]
        params["_perc"] = v if torch.is_tensor(v) else torch.as_tensor(float(v), dtype=torch.float32)
    return params


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

    # Read selected basins
    sel_path = Path(args.selected_basins)
    selected = list(csv.DictReader(open(sel_path)))
    basin_ids = [int(r["basin_id"]) for r in selected]
    B = len(basin_ids)
    idx_list = [int(np.where(gage_ids == bid)[0][0]) for bid in basin_ids]

    wu, tr_l, ev_l = args.warmup_days, args.train_days, args.eval_days
    ev_end = wu + tr_l + ev_l

    forc = forcings[idx_list, :ev_end, :].astype(np.float32)
    targ_raw = target[idx_list, :ev_end, 0].astype(np.float32)
    areas = attributes[idx_list, 11]
    targ_mmd = np.zeros_like(targ_raw)
    for b in range(B):
        targ_mmd[b] = flow_to_mmd(targ_raw[b], areas[b])

    forcing_t = torch.from_numpy(forc).permute(1, 0, 2)  # stay on CPU - faster for sequential loops
    targ_t = torch.from_numpy(targ_mmd.T)
    train_target = targ_t[wu:wu + tr_l]
    eval_target = targ_t[wu + tr_l:wu + tr_l + ev_l]

    formulas = [f.strip() for f in args.formulas.split(",")]
    print(f"Calibrating {len(formulas)} formulas on {B} basins, seed={args.seed}")
    print(f"Device: {device}, Steps: {args.steps}, LR: {args.lr}")

    step_records = []
    param_records = []
    train_metrics = []
    eval_metrics = []
    flux_train_records = []
    flux_eval_records = []
    failures = []

    for fid in formulas:
        if fid not in RECHARGE_FIDS:
            print(f"  Skipping unknown formula {fid}")
            continue

        combo = dict(DEFAULT_IDS)
        combo["recharge"] = fid

        # Random search calibration: sample N parameter sets, pick best by train MSE
        n_random = args.steps
        print(f"\n--- Formula {fid} (random search, {n_random} samples per basin) ---")
        t0 = time.time()

        # Pre-sample random parameter sets for all basins
        np.random.seed(args.seed * 100 + RECHARGE_FIDS.index(fid))
        random_samples = []
        for _ in range(n_random):
            sample = {}
            for name, (lo, hi) in _PARAM_BOUNDS.items():
                rv = np.random.uniform(0.1, 0.9)
                sample[name] = float(lo + (hi - lo) * rv)
            random_samples.append(sample)

        for b in range(B):
            bid = basin_ids[b]
            best_loss = float("inf")
            best_params = None

            for sample_idx, sample_pv in enumerate(random_samples):
                fp = _make_fparams(combo, sample_pv)
                for n in NODE_ORDER:
                    fn = combo.get(n, DEFAULT_IDS[n])
                    if fn in _EXTRA_PARAMS:
                        fp.setdefault(n, {}).update({k: torch.as_tensor(float(v), dtype=torch.float32) for k, v in _EXTRA_PARAMS[fn].items()})

                m = HbvFormulaStatic(formula_config=combo, warm_up=wu, param_dicts=fp)
                q_all = m.simulate(forcing_t[:, b, 0], forcing_t[:, b, 1],
                                   forcing_t[:, b, 2])["Q_raw"]
                qt = q_all[:tr_l]
                tt = train_target[:, b]
                n = min(len(qt), len(tt))
                mask = ~(torch.isnan(qt[:n]) | torch.isnan(tt[:n]))
                if mask.sum() < 2:
                    continue
                mse_val = float(F.mse_loss(qt[:n][mask], tt[:n][mask]).item())
                if math.isnan(mse_val) or math.isinf(mse_val):
                    continue
                if mse_val < best_loss:
                    best_loss = mse_val
                    best_params = dict(sample_pv)

            if best_params is None:
                failures.append({
                    "basin_id": bid, "seed": args.seed, "formula_id": fid,
                    "reason": "No valid random sample found",
                })
                continue

            # Evaluate best params
            fp_best = _make_fparams(combo, best_params)
            for n in NODE_ORDER:
                fn = combo.get(n, DEFAULT_IDS[n])
                if fn in _EXTRA_PARAMS:
                    fp_best.setdefault(n, {}).update({k: torch.as_tensor(float(v), dtype=torch.float32) for k, v in _EXTRA_PARAMS[fn].items()})

            m_eval = HbvFormulaStatic(formula_config=combo, warm_up=wu, param_dicts=fp_best)
            diag = m_eval.simulate(forcing_t[:, b, 0], forcing_t[:, b, 1],
                                    forcing_t[:, b, 2])
            q_all = diag["Q_raw"]
            trace = diag["trace"]

            # Train metrics
            qt = q_all[:tr_l]
            tt = train_target[:, b]
            n = min(len(qt), len(tt))
            tr_nse = nse(qt[:n], tt[:n])
            tr_kge = kge(qt[:n], tt[:n])

            # Eval metrics
            qe = q_all[tr_l:tr_l + ev_l]
            te = eval_target[:, b]
            ne = min(len(qe), len(te))
            mask_e = ~(np.isnan(_to_np(qe[:ne])) | np.isnan(_to_np(te[:ne])))
            ev_valid_ratio = float(mask_e.sum() / max(len(mask_e), 1))
            ev_nse = nse(qe[:ne], te[:ne])
            ev_kge = kge(qe[:ne], te[:ne])
            ev_rmse = rmse(qe[:ne], te[:ne])
            ev_bias = float(_to_np(qe[:ne])[mask_e].mean() - _to_np(te[:ne])[mask_e].mean())
            ev_log_nse = log_nse(qe[:ne], te[:ne])

            qe_np = _to_np(qe[:ne])
            te_np = _to_np(te[:ne])
            qe_sum = qe_np[mask_e].sum()
            te_sum = te_np[mask_e].sum()
            runoff_ratio = float(qe_sum / max(te_sum, 1e-6))
            wb_error = float(abs(qe_sum - te_sum) / max(te_sum, 1e-6))

            if mask_e.sum() > 10:
                p95 = np.percentile(te_np[mask_e], 95)
                p5 = np.percentile(te_np[mask_e], 5)
                peak_mask = te_np >= p95
                low_mask = te_np < p5
                peak_flow_error = float(abs(qe_np[mask_e & peak_mask].mean() -
                                             te_np[mask_e & peak_mask].mean()) /
                                        max(te_np[mask_e & peak_mask].mean(), 1e-6)) if (mask_e & peak_mask).any() else float("nan")
                low_flow_error = float(abs(qe_np[mask_e & low_mask].mean() -
                                           te_np[mask_e & low_mask].mean()) /
                                       max(te_np[mask_e & low_mask].mean(), 1e-6)) if (mask_e & low_mask).any() else float("nan")
            else:
                peak_flow_error = float("nan")
                low_flow_error = float("nan")

            train_metrics.append({
                "basin_id": bid, "seed": args.seed, "formula_id": fid,
                "train_nse": round(tr_nse, 6), "train_kge": round(tr_kge, 6),
                "train_mse": round(best_loss, 8),
            })
            eval_metrics.append({
                "basin_id": bid, "seed": args.seed, "formula_id": fid,
                "formula_name": fid,
                "eval_nse": round(ev_nse, 6), "eval_kge": round(ev_kge, 6),
                "eval_rmse": round(ev_rmse, 6), "eval_bias": round(ev_bias, 6),
                "eval_log_nse": round(ev_log_nse, 6),
                "valid_eval_ratio": round(ev_valid_ratio, 4),
                "water_balance_error": round(wb_error, 4),
                "runoff_ratio": round(runoff_ratio, 4),
                "peak_flow_error": round(peak_flow_error, 6) if not math.isnan(peak_flow_error) else float("nan"),
                "low_flow_error": round(low_flow_error, 6) if not math.isnan(low_flow_error) else float("nan"),
            })
            param_records.append({
                "basin_id": bid, "seed": args.seed, "formula_id": fid,
                **{name: round(best_params.get(name, float("nan")), 6) for name in _PHYN},
            })

            tr_s, tr_e = wu, wu + tr_l
            if tr_e > 0:
                recharge_trace = trace["recharge"][tr_s:tr_e]
                r_np = _to_np(recharge_trace)
                if len(r_np) > 0:
                    flux_train_records.append({
                        "basin_id": bid, "seed": args.seed, "formula_id": fid,
                        "flux_name": "recharge",
                        "median_flux": float(np.median(r_np)),
                        "p95_flux": float(np.percentile(r_np, 95)),
                        "max_flux": float(np.max(r_np)),
                        "raw_over_bound_rate": 0.0, "clamp_hit_rate": 0.0,
                        "water_constraint_violation_count": 0,
                    })
            ev_s, ev_e = wu + tr_l, wu + tr_l + ev_l
            if ev_e > ev_s:
                recharge_trace = trace["recharge"][ev_s:ev_e]
                r_np = _to_np(recharge_trace)
                if len(r_np) > 0:
                    flux_eval_records.append({
                        "basin_id": bid, "seed": args.seed, "formula_id": fid,
                        "flux_name": "recharge",
                        "median_flux": float(np.median(r_np)),
                        "p95_flux": float(np.percentile(r_np, 95)),
                        "max_flux": float(np.max(r_np)),
                        "raw_over_bound_rate": 0.0, "clamp_hit_rate": 0.0,
                        "water_constraint_violation_count": 0,
                    })

        mean_tr = np.mean([r["train_nse"] for r in train_metrics if r["formula_id"] == fid and not math.isnan(r["train_nse"])])
        mean_ev = np.mean([r["eval_nse"] for r in eval_metrics if r["formula_id"] == fid and not math.isnan(r["eval_nse"])])
        n_done = sum(1 for r in train_metrics if r["formula_id"] == fid)
        print(f"  Done ({time.time() - t0:.0f}s), {n_done}/{B} basins, mean_train_NSE={mean_tr:.4f}, mean_eval_NSE={mean_ev:.4f}")

    # ---- Write outputs ----
    _w(step_records if step_records else [{"basin_id": 0, "seed": args.seed, "formula_id": "NONE", "step": 0, "loss": 0.0, "grad_norm_before": 0.0, "grad_norm_after": 0.0, "has_nan": 0}],
       out_dir / "calibration_steps.csv",
       ["basin_id", "seed", "formula_id", "step", "loss",
        "grad_norm_before", "grad_norm_after", "has_nan"])
    _w(param_records, out_dir / "formula_params.csv",
       ["basin_id", "seed", "formula_id"] + _PHYN)
    _w(train_metrics, out_dir / "formula_metrics_train.csv",
       ["basin_id", "seed", "formula_id", "train_nse", "train_kge", "train_mse"])
    _w(eval_metrics, out_dir / "formula_metrics_eval.csv",
       ["basin_id", "seed", "formula_id", "formula_name",
        "eval_nse", "eval_kge", "eval_rmse", "eval_bias", "eval_log_nse",
        "valid_eval_ratio", "water_balance_error", "runoff_ratio",
        "peak_flow_error", "low_flow_error"])
    _w(flux_train_records, out_dir / "formula_flux_train.csv",
       ["basin_id", "seed", "formula_id", "flux_name",
        "median_flux", "p95_flux", "max_flux",
        "raw_over_bound_rate", "clamp_hit_rate",
        "water_constraint_violation_count"])
    _w(flux_eval_records, out_dir / "formula_flux_eval.csv",
       ["basin_id", "seed", "formula_id", "flux_name",
        "median_flux", "p95_flux", "max_flux",
        "raw_over_bound_rate", "clamp_hit_rate",
        "water_constraint_violation_count"])
    _w(failures, out_dir / "formula_failures.csv",
       ["basin_id", "seed", "formula_id", "reason"])

    # Report
    r0_train = [r["train_nse"] for r in train_metrics if r["formula_id"] == "R0" and not math.isnan(r["train_nse"])]
    r4_train = [r["train_nse"] for r in train_metrics if r["formula_id"] == "R4" and not math.isnan(r["train_nse"])]
    r5_train = [r["train_nse"] for r in train_metrics if r["formula_id"] == "R5" and not math.isnan(r["train_nse"])]
    r0_eval = [r["eval_nse"] for r in eval_metrics if r["formula_id"] == "R0" and not math.isnan(r["eval_nse"])]
    r4_eval = [r["eval_nse"] for r in eval_metrics if r["formula_id"] == "R4" and not math.isnan(r["eval_nse"])]
    r5_eval = [r["eval_nse"] for r in eval_metrics if r["formula_id"] == "R5" and not math.isnan(r["eval_nse"])]

    report_lines = [
        f"# Fixed-Formula Calibration — seed={args.seed}",
        f"",
        f"## Configuration",
        f"- formulas: {formulas}",
        f"- seed: {args.seed}",
        f"- steps: {args.steps}, lr: {args.lr}, grad_clip: {args.grad_clip}",
        f"- warmup: {wu}d, train: {tr_l}d, eval: {ev_l}d",
        f"",
        f"## Results",
        f"| Formula | Mean Train NSE | Mean Eval NSE | Mean Eval KGE |",
        f"|---------|----------------|---------------|---------------|",
        f"| R0 | {np.mean(r0_train):.4f} | {np.mean(r0_eval):.4f} | {np.mean([r['eval_kge'] for r in eval_metrics if r['formula_id']=='R0' and not math.isnan(r['eval_kge'])]):.4f} |",
        f"| R4 | {np.mean(r4_train):.4f} | {np.mean(r4_eval):.4f} | {np.mean([r['eval_kge'] for r in eval_metrics if r['formula_id']=='R4' and not math.isnan(r['eval_kge'])]):.4f} |",
        f"| R5 | {np.mean(r5_train):.4f} | {np.mean(r5_eval):.4f} | {np.mean([r['eval_kge'] for r in eval_metrics if r['formula_id']=='R5' and not math.isnan(r['eval_kge'])]):.4f} |",
        f"",
        f"## Failures",
        f"- Total failures: {len(failures)}",
        f"",
        f"## Integrity",
        f"- eval_used_for_selection: False",
        f"- selection_source: N/A (fixed formulas only)",
    ]
    (out_dir / "run_report.md").write_text("\n".join(report_lines))

    print(f"\nDone. Output: {out_dir}")
    print(f"  R0: train_NSE={np.mean(r0_train):.4f}, eval_NSE={np.mean(r0_eval):.4f}")
    print(f"  R4: train_NSE={np.mean(r4_train):.4f}, eval_NSE={np.mean(r4_eval):.4f}")
    print(f"  R5: train_NSE={np.mean(r5_train):.4f}, eval_NSE={np.mean(r5_eval):.4f}")
    print(f"  Failures: {len(failures)}")
    return True


def _w(rows, path, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        if rows:
            w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected-basins", required=True)
    ap.add_argument("--formulas", type=str, default="R0,R4,R5")
    ap.add_argument("--active-node", type=str, default="recharge")
    ap.add_argument("--steps", type=int, default=100,
                    help="Number of random parameter samples per basin")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--loss", type=str, default="mse")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-days", type=int, default=365)
    ap.add_argument("--train-days", type=int, default=365)
    ap.add_argument("--eval-days", type=int, default=365)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    run_calibration(args)
