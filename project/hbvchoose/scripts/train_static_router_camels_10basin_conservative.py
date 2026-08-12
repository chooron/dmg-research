#!/usr/bin/env python3
"""10-basin conservative static router validation.

Strictly separates train/eval windows, explicitly screens basins, and supports
multi-seed experiments with auditable selection source.

Key invariants:
- Formula enumeration labels come from TRAIN window only
- Eval window is NEVER used for selection or label generation
- All selections come from StaticFormulaRouter.forward(argmax)
- Every selection_summary records selection_source
"""
from __future__ import annotations

import argparse
import csv
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from model.hbv_formula_static import HbvFormulaStatic
from model.static_formula_router import StaticFormulaRouter
from model.formula_pool import CandidateFormulaPool

CAMELS_PATH = _PROJECT.parent.parent / "data" / "camels_dataset"
GAGE_ID_PATH = _PROJECT.parent.parent / "data" / "gage_id.npy"
OUTPUT_BASE = _PROJECT / "validation_results" / "static_router_10basin_conservative"

NODE_ORDER = ["snow", "recharge", "aet", "response"]
DEFAULT_IDS = {"snow": "S0", "recharge": "R0", "aet": "E0", "response": "Q0"}
DEFAULT_COMBO = dict(DEFAULT_IDS)
_EXTRA_PARAMS = {
    "S4": {"a_s": 0.3, "phi_s": 172.0}, "S5": {"c_m": 0.3},
    "R4": {"a_r": 10.0, "c_r": 0.5}, "R5": {"b_v": 1.0},
    "E3": {"gamma_E": 1.2}, "E4": {"s_w": 0.1, "s_o": 0.6},
    "Q2": {"alpha_Q": 1.2},
}

_PARAM_BOUNDS = {
    "parBETA": [1.0, 6.0], "parFC": [50.0, 500.0], "parK0": [0.05, 0.5],
    "parK1": [0.01, 0.3], "parK2": [0.001, 0.1], "parLP": [0.3, 1.0],
    "parPERC": [0.0, 3.0], "parUZL": [0.0, 100.0], "parTT": [-2.5, 2.5],
    "parCFMAX": [1.0, 10.0], "parCFR": [0.0, 0.1], "parCWH": [0.0, 0.2],
}
_PHY_NAMES = list(_PARAM_BOUNDS.keys())
N_PARAMS = len(_PARAM_BOUNDS)
_DEFAULT_PARAM_VALS = [0.3, 0.4, 0.3, 0.5, 0.3, 0.5, 0.4, 0.5, 0.5, 0.3, 0.5, 0.5]
_ALIAS_MAP = {"parBETA": "beta", "parFC": "FC", "parK0": "K_0", "parK1": "K_1",
              "parK2": "K_2", "parLP": "LP", "parUZL": "UZL", "parTT": "TT",
              "parCFMAX": "CFMAX", "parCFR": "CFR", "parCWH": "CWH", "parPERC": "PERC"}
_NODE_PARAMS = {
    "snow": ["parTT", "parCFMAX", "parCFR", "parCWH"],
    "recharge": ["parFC", "parBETA"],
    "aet": ["parFC", "parLP"],
    "response": ["parK0", "parK1", "parK2", "parUZL", "parPERC"],
}


# ===========================================================================
# utilities
# ===========================================================================

def flow_to_mmd(flow_ft3s, area_km2):
    return flow_ft3s * 2.446575 / max(area_km2, 1.0)


def _to_np(x):
    if hasattr(x, 'cpu'):
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
    return float(1.0 - math.sqrt((r - 1) ** 2 + (sqs / sqo - 1) ** 2 + (np.mean(qs) / max(np.mean(qo), 1e-12) - 1) ** 2))


def rmse(qsim, qobs):
    qs = _to_np(qsim)
    qo = _to_np(qobs)
    mask = ~np.isnan(qo) & ~np.isnan(qs)
    return float(np.sqrt(((qs[mask] - qo[mask]) ** 2).mean())) if mask.any() else float("nan")


def masked_mse(qsim, qobs):
    mask = ~(torch.isnan(qsim) | torch.isnan(qobs) | torch.isinf(qsim) | torch.isinf(qobs))
    if mask.sum() < 2:
        return torch.tensor(float("nan"), device=qsim.device)
    return F.mse_loss(qsim[mask], qobs[mask])


def _make_fparams(combo, phy_vals):
    params = {}
    for n in NODE_ORDER:
        nd = {}
        for hbv_name in _NODE_PARAMS.get(n, []):
            if hbv_name in phy_vals:
                nd[_ALIAS_MAP[hbv_name]] = torch.as_tensor(phy_vals[hbv_name], dtype=torch.float32)
        params[n] = nd
    if "parPERC" in phy_vals:
        params["_perc"] = torch.as_tensor(phy_vals["parPERC"], dtype=torch.float32)
    return params


def physical_from_normalized(norm_vec, B):
    """Convert normalized [0,1] params to physical dicts."""
    result = []
    for b in range(B):
        pv = {}
        for i, (name, bounds) in enumerate(_PARAM_BOUNDS.items()):
            lo, hi = bounds
            pv[name] = float(lo + (hi - lo) * norm_vec[b, i])
        result.append(pv)
    return result


def simulate_one(P, T, PET, combo, fparams, warmup):
    m = HbvFormulaStatic(formula_config=combo, warm_up=warmup, param_dicts=fparams)
    return m.simulate(P, T, PET)["Q_raw"]


# ===========================================================================
# data loading and screening
# ===========================================================================

def load_camels_data():
    with open(CAMELS_PATH, "rb") as f:
        forcings, target, attributes = pickle.load(f)
    gage_ids = np.load(GAGE_ID_PATH)
    return forcings, target, attributes, gage_ids


def split_time_windows(total_length, warmup_days=365, train_days=365, eval_days=365):
    need = warmup_days + train_days + eval_days
    if total_length < need:
        # fallback
        train_days = min(180, total_length - warmup_days - 90)
        eval_days = min(180, total_length - warmup_days - train_days)
    w_start = 0
    w_end = warmup_days
    tr_start = w_end
    tr_end = w_end + train_days
    ev_start = tr_end
    ev_end = min(tr_end + eval_days, total_length)
    return {
        "warmup_start": w_start, "warmup_end": w_end,
        "train_start": tr_start, "train_end": tr_end,
        "eval_start": ev_start, "eval_end": ev_end,
        "warmup_days": warmup_days,
        "train_days": train_days,
        "eval_days": ev_end - ev_start,
        "total_used": ev_end - w_start,
    }


def screen_basins(forcings, target, attributes, gage_ids, warmup_days, train_days,
                  eval_days, max_basins=10, strict=True):
    n_basins = forcings.shape[0]
    min_valid_ratio = 0.90 if strict else 0.80

    selected = []
    excluded = []

    for idx in range(n_basins):
        total_len = forcings.shape[1]
        windows = split_time_windows(total_len, warmup_days, train_days, eval_days)
        need = windows["warmup_days"] + windows["train_days"] + windows["eval_days"]
        if total_len < need:
            excluded.append({"basin_id": int(gage_ids[idx]), "reason": f"Too short: {total_len} < {need}",
                             "valid_target_ratio": float("nan"), "eval_valid_ratio": float("nan"),
                             "forcing_nan_count": 0, "forcing_inf_count": 0,
                             "q_zero_ratio": float("nan"), "total_length": total_len})
            continue

        forc = forcings[idx]
        targ = target[idx, :, 0]
        area = attributes[idx, 11]

        # Forcing checks
        f_nan = int(np.isnan(forc).sum())
        f_inf = int(np.isinf(forc).sum())
        if f_nan > 0:
            excluded.append({"basin_id": int(gage_ids[idx]), "reason": f"Forcing has {f_nan} NaN",
                             "valid_target_ratio": float("nan"), "eval_valid_ratio": float("nan"),
                             "forcing_nan_count": f_nan, "forcing_inf_count": f_inf,
                             "q_zero_ratio": float("nan"), "total_length": total_len})
            continue
        if f_inf > 0:
            excluded.append({"basin_id": int(gage_ids[idx]), "reason": f"Forcing has {f_inf} Inf",
                             "valid_target_ratio": float("nan"), "eval_valid_ratio": float("nan"),
                             "forcing_nan_count": f_nan, "forcing_inf_count": f_inf,
                             "q_zero_ratio": float("nan"), "total_length": total_len})
            continue

        # Target checks — entire record
        targ_mmd = flow_to_mmd(targ, area)
        valid_all = ~np.isnan(targ) & ~np.isinf(targ)
        valid_ratio_all = float(valid_all.sum() / max(len(targ), 1))

        if valid_ratio_all < min_valid_ratio:
            excluded.append({"basin_id": int(gage_ids[idx]), "reason": f"Low overall valid ratio: {valid_ratio_all:.3f}",
                             "valid_target_ratio": round(valid_ratio_all, 4), "eval_valid_ratio": float("nan"),
                             "forcing_nan_count": f_nan, "forcing_inf_count": f_inf,
                             "q_zero_ratio": float("nan"), "total_length": total_len})
            continue

        # Eval window checks
        ev_start = windows["eval_start"]
        ev_end = windows["eval_end"]
        targ_eval = targ[ev_start:ev_end]
        valid_eval = ~np.isnan(targ_eval) & ~np.isinf(targ_eval)
        eval_valid_ratio = float(valid_eval.sum() / max(len(targ_eval), 1))

        if eval_valid_ratio < min_valid_ratio:
            excluded.append({"basin_id": int(gage_ids[idx]), "reason": f"Low eval valid ratio: {eval_valid_ratio:.3f}",
                             "valid_target_ratio": round(valid_ratio_all, 4), "eval_valid_ratio": round(eval_valid_ratio, 4),
                             "forcing_nan_count": f_nan, "forcing_inf_count": f_inf,
                             "q_zero_ratio": float("nan"), "total_length": total_len})
            continue

        # Q Inf check
        q_inf = int(np.isinf(targ_mmd).sum())
        if q_inf > 0:
            excluded.append({"basin_id": int(gage_ids[idx]), "reason": f"Q has {q_inf} Inf",
                             "valid_target_ratio": round(valid_ratio_all, 4), "eval_valid_ratio": round(eval_valid_ratio, 4),
                             "forcing_nan_count": f_nan, "forcing_inf_count": f_inf,
                             "q_zero_ratio": float("nan"), "total_length": total_len})
            continue

        # Q zero ratio
        q_zero = float((np.abs(targ_mmd) < 1e-8).sum() / max(len(targ_mmd), 1))
        if q_zero > 0.95:
            excluded.append({"basin_id": int(gage_ids[idx]), "reason": f"Q near-zero ratio: {q_zero:.3f}",
                             "valid_target_ratio": round(valid_ratio_all, 4), "eval_valid_ratio": round(eval_valid_ratio, 4),
                             "forcing_nan_count": f_nan, "forcing_inf_count": f_inf,
                             "q_zero_ratio": round(q_zero, 4), "total_length": total_len})
            continue

        q_nan = int(np.isnan(targ_mmd).sum())
        train_valid = ~np.isnan(targ[windows["train_start"]:windows["train_end"]])
        train_valid_ratio = float(train_valid.sum() / max(len(train_valid), 1))

        selected.append({
            "basin_id": int(gage_ids[idx]),
            "valid_target_ratio": round(valid_ratio_all, 4),
            "train_valid_ratio": round(train_valid_ratio, 4),
            "eval_valid_ratio": round(eval_valid_ratio, 4),
            "forcing_nan_count": f_nan,
            "forcing_inf_count": f_inf,
            "q_nan_count": q_nan,
            "q_inf_count": q_inf,
            "q_zero_ratio": round(q_zero, 4),
            "total_length": total_len,
            "warmup_days": windows["warmup_days"],
            "train_days": windows["train_days"],
            "eval_days": windows["eval_days"],
            "screening_mode": "strict" if strict else "fallback",
        })

        if len(selected) >= max_basins:
            break

    if len(selected) < max_basins:
        print(f"Strict screening found {len(selected)} basins, falling back to relaxed rules")
        return screen_basins(forcings, target, attributes, gage_ids,
                             warmup_days, train_days, eval_days, max_basins, strict=False)

    return selected[:max_basins], excluded


# ===========================================================================
# per-basin data extraction
# ===========================================================================

def extract_basin_data(forcings, target, attributes, gage_ids, selected, windows, device):
    B = len(selected)
    idx_list = []
    for s in selected:
        match = np.where(gage_ids == s["basin_id"])[0]
        idx_list.append(int(match[0]))

    ev_end = windows["eval_end"]
    forc = forcings[idx_list, :ev_end, :].astype(np.float32)
    targ_raw = target[idx_list, :ev_end, 0].astype(np.float32)
    areas = attributes[idx_list, 11]
    attr_raw = attributes[idx_list, :].astype(np.float32)

    targ_mmd = np.zeros_like(targ_raw)
    for b in range(B):
        targ_mmd[b] = flow_to_mmd(targ_raw[b], areas[b])

    a_min = attr_raw.min(axis=0, keepdims=True)
    a_rng = np.maximum(attr_raw.max(axis=0, keepdims=True) - a_min, 1e-8)
    attr_norm = (attr_raw - a_min) / a_rng

    forcing_t = torch.from_numpy(forc).permute(1, 0, 2).to(device)
    targ_t = torch.from_numpy(targ_mmd.T).to(device)
    attr_t = torch.from_numpy(attr_norm).to(device)

    return forcing_t, targ_t, attr_t, areas, idx_list


# ===========================================================================
# main experiment
# ===========================================================================

def run_10basin_experiment(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    active_nodes = [n.strip() for n in args.active_nodes.split(",") if n.strip()]
    inactive_nodes = [n for n in NODE_ORDER if n not in active_nodes]
    print(f"Active nodes: {active_nodes}  Inactive: {inactive_nodes}")

    # ---- Load and screen ---------------------------------------------------
    print("Loading CAMELS...")
    forcings, target, attributes, gage_ids = load_camels_data()
    n_total_basins = forcings.shape[0]

    print(f"Screening {n_total_basins} basins...")
    selected, excluded = screen_basins(
        forcings, target, attributes, gage_ids,
        args.warmup_days, args.train_days, args.eval_days, args.max_basins, strict=True)
    B = len(selected)
    print(f"Selected {B}/{args.max_basins}, excluded {len(excluded)}")
    for s in selected:
        print(f"  + {s['basin_id']}: valid={s['valid_target_ratio']:.3f}, "
              f"train={s['train_valid_ratio']:.3f}, eval={s['eval_valid_ratio']:.3f}")
    for e in excluded[:5]:
        print(f"  - {e['basin_id']}: {e['reason']}")

    if B < 2:
        print("ERROR: Too few basins. Aborting.")
        return False

    # Extract data with correct windows
    windows = split_time_windows(int(forcings.shape[1]), args.warmup_days, args.train_days, args.eval_days)
    print(f"Windows: warmup={windows['warmup_days']}d, train={windows['train_days']}d, eval={windows['eval_days']}d")

    forcing_t, targ_t, attr_t, areas, idx_list = extract_basin_data(
        forcings, target, attributes, gage_ids, selected, windows, device)

    wu, tr_e, ev_e = windows["warmup_end"], windows["train_end"], windows["eval_end"]
    warmup_d = windows["warmup_days"]

    # Slice forcing/target into train and eval windows
    train_forcing = forcing_t[wu:tr_e]
    train_target = targ_t[wu:tr_e]
    eval_forcing = forcing_t[tr_e:ev_e]
    eval_target = targ_t[tr_e:ev_e]

    print(f"Train window: {train_forcing.shape[0]} steps, eval window: {eval_forcing.shape[0]} steps")

    # ---- Pre-compute all candidate formula outputs (once per basin) ---------
    print(f"\n=== Pre-computing candidate formula outputs ({len(active_nodes)} nodes, {B} basins) ===")
    pool = CandidateFormulaPool()
    fids_dict = {n: pool.formulas(n, "main") for n in NODE_ORDER}

    calib_norm = torch.full((B, N_PARAMS), 0.4, device=device)

    # Pre-compute Q for each basin × each candidate formula
    # Structure: pre_q[b][node][fid_idx] = tensor of shape (eval_end - warmup,)
    pre_q = {}
    for b in range(B):
        pre_q[b] = {}
        for node in active_nodes:
            fids = fids_dict[node]
            pre_q[b][node] = {}
            for fid in fids:
                combo = dict(DEFAULT_COMBO)
                combo[node] = fid
                pv = physical_from_normalized(calib_norm, B)[b]
                fp = _make_fparams(combo, pv)
                for n2 in NODE_ORDER:
                    fn2 = combo[n2]
                    if fn2 in _EXTRA_PARAMS:
                        fp.setdefault(n2, {}).update(_EXTRA_PARAMS[fn2])
                q = simulate_one(forcing_t[:, b, 0], forcing_t[:, b, 1], forcing_t[:, b, 2],
                                 combo, fp, warmup_d)
                pre_q[b][node][fid] = q
    n_pre = B * sum(len(fids_dict[n]) for n in active_nodes)
    print(f"  Pre-computed {n_pre} formula outputs")

    # ---- Default HBV baseline (uses pre_q) ----------------------------------
    print("\n=== Default HBV baseline ===")
    default_metrics_train = []
    default_metrics_eval = []

    for b in range(B):
        q_all = pre_q[b]["recharge"]["R0"] if "recharge" in active_nodes else pre_q[b][active_nodes[0]][fids_dict[active_nodes[0]][0]]
        Tq_train = train_forcing.shape[0]
        qt = q_all[:Tq_train]
        qt_ref = train_target[:Tq_train, b]
        n = min(len(qt), len(qt_ref))
        default_metrics_train.append({
            "basin_id": selected[b]["basin_id"],
            "nse": nse(qt[:n], qt_ref[:n]),
            "kge": kge(qt[:n], qt_ref[:n]),
        })

        q_eval = q_all[Tq_train:min(Tq_train + eval_forcing.shape[0], len(q_all))]
        qe_ref = eval_target[:, b]
        ne = min(len(q_eval), len(qe_ref))
        default_metrics_eval.append({
            "basin_id": selected[b]["basin_id"],
            "nse": nse(q_eval[:ne], qe_ref[:ne]),
            "kge": kge(q_eval[:ne], qe_ref[:ne]),
        })

    print(f"  Train mean NSE: {np.mean([d['nse'] for d in default_metrics_train]):.4f}")
    print(f"  Eval  mean NSE: {np.mean([d['nse'] for d in default_metrics_eval]):.4f}")

    # ---- StaticFormulaRouter training (uses pre_q) --------------------------
    print(f"\n=== StaticRouter training ({args.steps} steps) ===")
    router = StaticFormulaRouter(
        attr_dim=attr_t.shape[1],
        temperature=args.temperature,
        default_bias=args.anchor_bias,
        hard_eval=False,
    ).to(device)

    trainable = list(router.parameters())
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    step_records = []

    # Pre-compute train-window MSE for each basin × formula combination
    pre_train_mse = {}
    for b in range(B):
        pre_train_mse[b] = {}
        for node in active_nodes:
            pre_train_mse[b][node] = {}
            for fid in fids_dict[node]:
                qs = pre_q[b][node][fid][:train_forcing.shape[0]]
                pre_train_mse[b][node][fid] = masked_mse(qs, train_target[:, b])

    for step in range(args.steps):
        router.train()
        optimizer.zero_grad()

        r_out = router(attr_t)

        for n in inactive_nodes:
            f = fids_dict[n]
            default_idx = f.index(DEFAULT_IDS[n]) if DEFAULT_IDS[n] in f else 0
            r_out["selected"][n] = torch.full((B,), default_idx, dtype=torch.long)

        total_loss = torch.tensor(0.0, device=device)
        for node in active_nodes:
            fids = fids_dict[node]
            n_f = len(fids)
            combo_losses = torch.stack([
                torch.stack([pre_train_mse[b][node][fid] for fid in fids])
                for b in range(B)
            ]).to(device)  # [B, n_f]
            logits = r_out["logits"][node]
            best = combo_losses.argmin(dim=-1)
            total_loss = total_loss + F.cross_entropy(logits, best)

        if active_nodes:
            total_loss = total_loss / len(active_nodes)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            step_records.append({"step": step, "loss": float("nan"), "grad_norm_before": 0.0,
                                 "grad_norm_after": 0.0, "has_nan": 1,
                                 "entropy_recharge": 0.5, "default_rate": 1.0})
            break

        total_loss.backward()

        g_before = math.sqrt(sum((p.grad.norm().item() ** 2) for p in trainable if p.grad is not None))
        has_nan_grad = any(torch.isnan(p.grad).any().item() or torch.isinf(p.grad).any().item()
                           for p in trainable if p.grad is not None)
        if has_nan_grad:
            step_records.append({"step": step, "loss": float(total_loss.item()), "grad_norm_before": g_before,
                                 "grad_norm_after": 0.0, "has_nan": 1,
                                 "entropy_recharge": 0.5, "default_rate": 1.0})
            continue

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip)
        g_after = math.sqrt(sum((p.grad.norm().item() ** 2) for p in trainable if p.grad is not None))
        optimizer.step()

        dr = _default_rate(r_out, active_nodes, fids_dict)
        ent = float(r_out.get("entropy_recharge", torch.tensor(0.5)).mean().item())
        step_records.append({"step": step, "loss": float(total_loss.item()), "grad_norm_before": round(g_before, 8),
                             "grad_norm_after": round(g_after, 8), "has_nan": 0,
                             "entropy_recharge": round(ent, 6), "default_rate": round(dr, 6)})

        if step % max(1, args.steps // 5) == 0:
            print(f"  step {step:4d}  loss={total_loss.item():.6f}  grad={g_before:.4f}  def_rate={dr:.4f}")

    # ---- Final evaluation (EVAL WINDOW ONLY, using pre_q) --------------------
    print("\n=== Final evaluation on EVAL window ===")
    router.eval()
    with torch.no_grad():
        r_out_final = router(attr_t)
        for n in inactive_nodes:
            f = fids_dict[n]
            r_out_final["selected"][n] = torch.full((B,), f.index(DEFAULT_IDS[n]), dtype=torch.long)

    metrics_train = []
    metrics_eval = []
    combo_records = {}

    for b in range(B):
        combo = dict(DEFAULT_COMBO)
        for node in NODE_ORDER:
            idx = int(r_out_final["selected"][node][b].item())
            combo[node] = fids_dict[node][idx]

        combo_str = "_".join(combo[n] for n in NODE_ORDER)
        combo_records[combo_str] = combo_records.get(combo_str, 0) + 1

        # Use pre-computed output for selected formula
        selected_fid = combo["recharge"] if "recharge" in active_nodes else combo[active_nodes[0]]
        node_key = "recharge" if "recharge" in active_nodes else active_nodes[0]
        q_full = pre_q[b][node_key][selected_fid]

        Tq_train = train_forcing.shape[0]
        qt_router = q_full[:Tq_train]
        qt_ref = train_target[:Tq_train, b]
        nt = min(len(qt_router), len(qt_ref))
        metrics_train.append({
            "basin_id": selected[b]["basin_id"],
            "nse": nse(qt_router[:nt], qt_ref[:nt]),
            "kge": kge(qt_router[:nt], qt_ref[:nt]),
        })

        q_eval = q_full[Tq_train:min(Tq_train + eval_forcing.shape[0], len(q_full))]
        qe_ref = eval_target[:, b]
        ne = min(len(q_eval), len(qe_ref))
        metrics_eval.append({
            "basin_id": selected[b]["basin_id"],
            "nse": nse(q_eval[:ne], qe_ref[:ne]),
            "kge": kge(q_eval[:ne], qe_ref[:ne]),
        })

        print(f"  Basin {selected[b]['basin_id']}: R={selected_fid}, "
              f"train_NSE={metrics_train[-1]['nse']:.4f}, eval_NSE={metrics_eval[-1]['nse']:.4f}")

    # ---- Write outputs -----------------------------------------------------
    _write_csv(step_records, out_dir / "training_steps.csv",
               ["step", "loss", "grad_norm_before", "grad_norm_after", "has_nan",
                "entropy_recharge", "default_rate"])

    sel_rows = [{"combo_id": c, "count": n, "selection_source": "router_logits",
                 "eval_used_for_selection": False, "label_source": "train_metric_enumeration"}
                for c, n in sorted(combo_records.items(), key=lambda x: -x[1])]
    _write_csv(sel_rows, out_dir / "selection_summary.csv",
               ["combo_id", "count", "selection_source", "eval_used_for_selection", "label_source"])

    _write_csv(metrics_train, out_dir / "metrics_train.csv",
               ["basin_id", "nse", "kge", "rmse"])
    _write_csv(metrics_eval, out_dir / "metrics_eval.csv",
               ["basin_id", "nse", "kge", "rmse"])

    failures = []
    for b in range(B):
        mn = metrics_eval[b]
        if np.isnan(mn["nse"]):
            failures.append({"basin_id": selected[b]["basin_id"], "reason": "eval_NSE=NaN",
                             "stage": "eval", "step": -1})
    _write_csv(failures, out_dir / "failures.csv", ["basin_id", "reason", "stage", "step"])

    # Write selected/excluded basin lists
    _write_csv(selected, out_dir / "selected_basins.csv",
               ["basin_id", "valid_target_ratio", "train_valid_ratio", "eval_valid_ratio",
                "forcing_nan_count", "forcing_inf_count", "q_nan_count", "q_inf_count",
                "q_zero_ratio", "total_length", "warmup_days", "train_days", "eval_days",
                "screening_mode"])
    _write_csv(excluded, out_dir / "excluded_basins.csv",
               ["basin_id", "reason", "valid_target_ratio", "eval_valid_ratio",
                "forcing_nan_count", "forcing_inf_count", "q_zero_ratio", "total_length"])

    # ---- Run report --------------------------------------------------------
    losses = [r["loss"] for r in step_records if not (math.isnan(r["loss"]) or math.isinf(r["loss"]))]
    mean_train_d = np.mean([m["nse"] for m in metrics_train if not np.isnan(m["nse"])])
    mean_eval_d = np.mean([m["nse"] for m in metrics_eval if not np.isnan(m["nse"])])
    mean_eval_delta = np.mean([metrics_eval[b]["nse"] - default_metrics_eval[b]["nse"]
                               for b in range(B)
                               if not np.isnan(metrics_eval[b]["nse"]) and not np.isnan(default_metrics_eval[b]["nse"])])

    final_dr = step_records[-1]["default_rate"] if step_records else 1.0

    report = [
        "# 10-Basin Conservative Static Router Run Report",
        "",
        f"## Configuration",
        f"- seed: {args.seed}",
        f"- active_nodes: {active_nodes}",
        f"- anchor_bias: {args.anchor_bias}",
        f"- temperature: {args.temperature}",
        f"- steps: {args.steps}, grad_clip: {args.grad_clip}",
        f"- warmup: {windows['warmup_days']}d, train: {windows['train_days']}d, eval: {windows['eval_days']}d",
        "",
        f"## Basins",
        f"- Selected: {B}/{args.max_basins}",
        f"- Excluded: {len(excluded)}",
        "",
        f"## Training",
        f"- Steps completed: {len(step_records)}",
        f"- Initial loss: {losses[0]:.6f}" if losses else "- Initial loss: N/A",
        f"- Final loss: {losses[-1]:.6f}" if losses else "",
        f"- NaN loss: {any(r['has_nan'] for r in step_records)}",
        f"- Final default_rate (recharge): {final_dr:.4f}",
        f"- Final entropy (recharge): {step_records[-1]['entropy_recharge']:.4f}" if step_records else "",
        "",
        f"## Metrics",
        f"- Mean train NSE (default): {np.mean([m['nse'] for m in default_metrics_train]):.4f}",
        f"- Mean train NSE (router): {mean_train_d:.4f}",
        f"- Mean eval NSE (default): {np.mean([m['nse'] for m in default_metrics_eval]):.4f}",
        f"- Mean eval NSE (router): {mean_eval_d:.4f}",
        f"- Mean eval ΔNSE: {mean_eval_delta:.4f}",
        "",
        f"## Integrity",
        f"- selection_source: router_logits",
        f"- label_source: train_metric_enumeration",
        f"- eval_used_for_selection: False",
        f"- leakage_risk: LOW",
    ]
    (out_dir / "run_report.md").write_text("\n".join(report))

    print(f"\nDone. Output: {out_dir}")
    return True


def _default_rate(r_out, active_nodes, fids_dict):
    if not active_nodes:
        return 1.0
    total = 0.0
    count = 0
    for node in active_nodes:
        if node not in r_out.get("selected", {}):
            continue
        sel = r_out["selected"][node]
        fids = fids_dict.get(node, [])
        default_id = DEFAULT_IDS.get(node)
        if default_id and default_id in fids:
            di = fids.index(default_id)
            total += float((sel == di).float().mean().item())
            count += 1
    return total / max(count, 1)


def _write_csv(rows, path, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        if rows:
            w.writerows(rows)


# ===========================================================================
# entry
# ===========================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-basins", type=int, default=10)
    ap.add_argument("--active-nodes", type=str, default="recharge")
    ap.add_argument("--anchor-bias", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--temperature-final", type=float, default=0.7)
    ap.add_argument("--temperature-anneal-steps", type=int, default=200)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-days", type=int, default=365)
    ap.add_argument("--train-days", type=int, default=365)
    ap.add_argument("--eval-days", type=int, default=365)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    run_10basin_experiment(args)
