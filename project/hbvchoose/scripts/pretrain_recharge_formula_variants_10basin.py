#!/usr/bin/env python3
"""Stage 1B Fast: Pre-train recharge formulas with batch calibration.

For each recharge formula (R0,R4,R5), calibrate parameters jointly across all
10 basins on train window. Then compare per-basin train/eval NSE.
"""
from __future__ import annotations

import argparse, csv, math, pickle, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from model.hbv_formula_static import HbvFormulaStatic
import scripts.train_static_router_camels_10basin_conservative as _t

CAMELS_PATH = _PROJECT.parent.parent / "data" / "camels_dataset"
GAGE_ID_PATH = _PROJECT.parent.parent / "data" / "gage_id.npy"
OUT_BASE = _PROJECT / "validation_results" / "stage1_recharge_formula_pretraining_10basin"

NODE_ORDER = ["snow", "recharge", "aet", "response"]
DEFAULT_IDS = {"snow": "S0", "recharge": "R0", "aet": "E0", "response": "Q0"}
_PB = _t._PARAM_BOUNDS
_PHYN = _t._PHY_NAMES
_RECHARGE_FIDS = ["R0", "R4", "R5"]

def run_fast_pretrain(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    with open(CAMELS_PATH, "rb") as f: forcings, target, attributes = pickle.load(f)
    gage_ids = np.load(GAGE_ID_PATH)

    sel_csv = _PROJECT / "validation_results" / "static_router_10basin_conservative" / "recharge_anchor0.5_seed0" / "selected_basins.csv"
    if sel_csv.exists():
        basin_ids = [int(r["basin_id"]) for r in csv.DictReader(open(sel_csv))]
    else:
        basin_ids = [1013500, 1022500, 1030500, 1031500, 1047000, 1052500, 1054200, 1055000, 1057000, 1073000]
    B = len(basin_ids)
    idx_list = [int(np.where(gage_ids == bid)[0][0]) for bid in basin_ids]

    wu, tr_l, ev_l = args.warmup_days, args.train_days, args.eval_days
    ev_end = wu + tr_l + ev_l

    forc = forcings[idx_list, :ev_end, :].astype(np.float32)
    targ_raw = target[idx_list, :ev_end, 0].astype(np.float32)
    areas = attributes[idx_list, 11]
    targ_mmd = np.zeros_like(targ_raw)
    for b in range(B): targ_mmd[b] = targ_raw[b] * 2.446575 / max(areas[b], 1.0)

    forcing_t = torch.from_numpy(forc).permute(1, 0, 2).to(device)
    targ_t = torch.from_numpy(targ_mmd.T).to(device)
    train_target = targ_t[wu:wu + tr_l]
    eval_target = targ_t[wu + tr_l:wu + tr_l + ev_l]

    print(f"Pre-training {len(_RECHARGE_FIDS)} recharge formulas on {B} basins, seed={args.seed}")

    train_ranking = []
    eval_ranking = []
    failures = []

    for fid in _RECHARGE_FIDS:
        combo = dict(DEFAULT_IDS)
        combo["recharge"] = fid
        print(f"\n  Formula {fid}:")

        # Batch calibrate across all basins
        raw = torch.zeros(B, _t.N_PARAMS, device=device)
        init_vals = torch.tensor([0.3, 0.4, 0.3, 0.5, 0.3, 0.5, 0.4, 0.5, 0.5, 0.3, 0.5, 0.5], device=device).unsqueeze(0).repeat(B, 1)
        raw = torch.logit(init_vals.clamp(1e-6, 1 - 1e-6)) + torch.randn_like(raw) * 0.1
        raw.requires_grad = True
        opt = torch.optim.Adam([raw], lr=args.lr)

        for step in range(args.steps):
            opt.zero_grad()
            norm = torch.sigmoid(raw.clamp(-5, 5))
            Q_list = []
            for b in range(B):
                pv = {}
                for i, (name, (lo, hi)) in enumerate(_t._PARAM_BOUNDS.items()):
                    pv[name] = lo + (hi - lo) * norm[b, i]
                fp = _t._make_fparams(combo, pv)
                from model.hbv_formula_static import HbvFormulaStatic
                m = HbvFormulaStatic(formula_config=combo, warm_up=wu, param_dicts=fp)
                Q_list.append(m.simulate(forcing_t[:, b, 0], forcing_t[:, b, 1], forcing_t[:, b, 2])["Q_raw"])

            mx = max(q.shape[0] for q in Q_list)
            Qs = torch.zeros(mx, B, device=device)
            for b in range(B):
                L = Q_list[b].shape[0]
                Qs[:L, b] = Q_list[b]

            q_train = Qs[:tr_l]
            mask = ~(torch.isnan(q_train) | torch.isnan(train_target[:tr_l]))
            if mask.sum() < 2: break
            loss = F.mse_loss(q_train[mask], train_target[:tr_l][mask])
            if torch.isnan(loss) or torch.isinf(loss): break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw], max_norm=args.grad_clip)
            opt.step()

            if step % max(1, args.steps // 5) == 0:
                print(f"    step {step:4d} loss={loss.item():.6f}")

        # Evaluate per-basin
        with torch.no_grad():
            norm_f = torch.sigmoid(raw.clamp(-5, 5))
            for b in range(B):
                bid = basin_ids[b]
                pv = {}
                for i, (name, (lo, hi)) in enumerate(_t._PARAM_BOUNDS.items()):
                    pv[name] = float(lo + (hi - lo) * norm_f[b, i])
                fp = _t._make_fparams(combo, pv)
                m = HbvFormulaStatic(formula_config=combo, warm_up=wu, param_dicts=fp)
                q_all = m.simulate(forcing_t[:, b, 0], forcing_t[:, b, 1], forcing_t[:, b, 2])["Q_raw"]

                qt = q_all[:tr_l]; qtr = train_target[:, b]
                n = min(len(qt), len(qtr))
                tr_nse = _t.nse(qt[:n], qtr[:n])
                tr_mse = float(F.mse_loss(qt[:n], qtr[:n]).item()) if not np.isnan(tr_nse) else float("nan")

                qe = q_all[tr_l:tr_l + ev_l]; qer = eval_target[:, b]
                ne = min(len(qe), len(qer))
                ev_nse = _t.nse(qe[:ne], qer[:ne])

                train_ranking.append({"basin_id": bid, "seed": args.seed, "formula_id": fid,
                                       "train_mse": round(tr_mse, 8) if not math.isnan(tr_mse) else float("nan"),
                                       "train_nse": round(tr_nse, 6),
                                       "is_train_best": False})
                eval_ranking.append({"basin_id": bid, "seed": args.seed, "formula_id": fid,
                                      "eval_nse": round(ev_nse, 6),
                                      "is_eval_best": False})
                print(f"    Basin {bid}: train_NSE={tr_nse:.4f}  eval_NSE={ev_nse:.4f}")

    # Rank per basin
    for bid in basin_ids:
        tr_rows = [r for r in train_ranking if r["basin_id"] == bid]
        tr_rows.sort(key=lambda x: -x["train_nse"] if not np.isnan(x["train_nse"]) else -1e9)
        for i, r in enumerate(tr_rows):
            r["rank_by_train_nse"] = i + 1
            r["is_train_best"] = (i == 0)

        ev_rows = [r for r in eval_ranking if r["basin_id"] == bid]
        ev_rows.sort(key=lambda x: -x["eval_nse"] if not np.isnan(x["eval_nse"]) else -1e9)
        for i, r in enumerate(ev_rows):
            r["rank_by_eval_nse"] = i + 1
            r["is_eval_best"] = (i == 0)

    _w(train_ranking, out_dir / "formula_ranking_train.csv",
       ["basin_id", "seed", "formula_id", "train_mse", "train_nse", "rank_by_train_nse", "is_train_best"])
    _w(eval_ranking, out_dir / "formula_ranking_eval.csv",
       ["basin_id", "seed", "formula_id", "eval_nse", "rank_by_eval_nse", "is_eval_best"])

    # Summary
    for fid in _RECHARGE_FIDS:
        best_tr = sum(1 for r in train_ranking if r["formula_id"] == fid and r.get("is_train_best"))
        best_ev = sum(1 for r in eval_ranking if r["formula_id"] == fid and r.get("is_eval_best"))
        tr_nses = [r["train_nse"] for r in train_ranking if r["formula_id"] == fid and not np.isnan(r["train_nse"])]
        ev_nses = [r["eval_nse"] for r in eval_ranking if r["formula_id"] == fid and not np.isnan(r["eval_nse"])]
        print(f"\n  {fid}: train_best={best_tr}/{B}, eval_best={best_ev}/{B}, "
              f"mean_train_NSE={np.mean(tr_nses):.4f}, mean_eval_NSE={np.mean(ev_nses):.4f}")

    # Non-default check
    r0_best = sum(1 for r in train_ranking if r["formula_id"] == "R0" and r.get("is_train_best"))
    print(f"\n  R0 train best: {r0_best}/{B}")
    if r0_best < B:
        print(f"  *** Non-default formulas have train advantage in {B - r0_best} basins ***")
    else:
        print(f"  R0 remains best in all {B} basins after calibration")

    print(f"\nDone. Output: {out_dir}")
    return True

def _w(rows, path, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-days", type=int, default=365)
    ap.add_argument("--train-days", type=int, default=60)
    ap.add_argument("--eval-days", type=int, default=60)
    ap.add_argument("--output-dir", default=str(OUT_BASE / "seed0"))
    args = ap.parse_args()
    run_fast_pretrain(args)
