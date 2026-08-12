#!/usr/bin/env python3
"""Stage 5: Train StaticFormulaRouter from train-window oracle labels.

Reads oracle_labels_train.csv, selected_basins.csv, and CAMELS static attributes.
Trains a StaticFormulaRouter with cross-entropy loss against oracle labels.
"""
from __future__ import annotations

import argparse, csv, math, pickle, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from model.static_formula_router import StaticFormulaRouter

CAMELS_PATH = _PROJECT.parent.parent / "data" / "camels_dataset"
GAGE_ID_PATH = _PROJECT.parent.parent / "data" / "gage_id.npy"

RECHARGE_FIDS = ["R0", "R4", "R5"]


def normalize_attrs(attr_raw):
    attr = attr_raw.astype(np.float32).copy()
    n_cols = attr.shape[1]
    for j in range(n_cols):
        col = attr[:, j]
        n_nan_before = int(np.isnan(col).sum())
        n_inf_before = int(np.isinf(col).sum())
        col[np.isinf(col)] = np.nan
        nan_mask = np.isnan(col)
        n_imp = int(nan_mask.sum())
        median_used = float("nan")
        if n_imp > 0 and n_imp < len(col):
            median_used = float(np.nanmedian(col))
            col[nan_mask] = median_used
        elif n_imp == len(col):
            col[:] = 0.0
        cmin, cmax = float(col.min()), float(col.max())
        constant = abs(cmax - cmin) < 1e-10
        attr[:, j] = col
    a_min = attr.min(axis=0, keepdims=True)
    a_rng = np.maximum(attr.max(axis=0, keepdims=True) - a_min, 1e-8)
    result = (attr - a_min) / a_rng
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    return result.astype(np.float32)


def nse(qsim, qobs):
    qsim = np.asarray(qsim, dtype=np.float64)
    qobs = np.asarray(qobs, dtype=np.float64)
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
    qsim = np.asarray(qsim, dtype=np.float64)
    qobs = np.asarray(qobs, dtype=np.float64)
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


def flow_to_mmd(flow, area):
    return flow * 2.446575 / max(area, 1.0)


def run_router_training(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {device}, Seed: {args.seed}")
    print(f"Anchor bias: {args.anchor_bias}, Temperature: {args.temperature}")

    # Load data
    with open(CAMELS_PATH, "rb") as f:
        forcings, target, attributes = pickle.load(f)
    gage_ids = np.load(GAGE_ID_PATH)

    # Read selected basins
    selected = list(csv.DictReader(open(args.selected_basins)))
    sel_basin_ids = [int(r["basin_id"]) for r in selected]
    B = len(sel_basin_ids)
    idx_list = [int(np.where(gage_ids == bid)[0][0]) for bid in sel_basin_ids]

    # Read oracle labels
    oracle_raw = list(csv.DictReader(open(args.oracle_labels)))
    # Filter to current seed
    oracle = [r for r in oracle_raw if int(r["seed"]) == args.seed]
    if not oracle:
        print(f"ERROR: No oracle labels found for seed {args.seed}")
        return False
    oracle_basin_ids = [int(r["basin_id"]) for r in oracle]

    # Align oracle with selected basins
    oracle_map = {int(r["basin_id"]): r for r in oracle}
    aligned_oracle = [oracle_map[bid] for bid in sel_basin_ids if bid in oracle_map]
    aligned_basin_ids = [int(r["basin_id"]) for r in aligned_oracle]
    assert len(aligned_oracle) == B, f"Oracle/selected basin mismatch: {len(aligned_oracle)} vs {B}"

    # Normalize attributes
    attr_sel = attributes[idx_list, :].astype(np.float32)
    attr_norm = normalize_attrs(attr_sel)
    attr_t = torch.from_numpy(attr_norm).float().to(device)

    # Build router
    router = StaticFormulaRouter(
        attr_dim=attr_t.shape[1],
        temperature=args.temperature,
        default_bias=args.anchor_bias,
        hard_eval=False,
    ).to(device)

    fids_registry = router.formula_ids["recharge"]
    label_map = {fid: fids_registry.index(fid) for fid in RECHARGE_FIDS if fid in fids_registry}
    labels = torch.tensor([label_map[r["best_train_formula"]] for r in aligned_oracle], device=device)

    # Training
    opt = torch.optim.Adam(router.parameters(), lr=args.lr)
    step_records = []
    selection_records = []
    failures = []

    for step in range(args.steps):
        router.train()
        opt.zero_grad()

        r_out = router(attr_t)
        logits = r_out["logits"]["recharge"]

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            failures.append({
                "step": step, "basin_id": "ALL", "seed": args.seed,
                "reason": f"NaN/Inf in router logits at step {step}",
            })
            step_records.append({
                "step": step, "loss": float("nan"),
                "grad_norm_before": 0.0, "grad_norm_after": 0.0,
                "has_nan": 1, "accuracy": 0.0, "default_rate": 1.0,
                "entropy_mean": 0.5,
            })
            break

        loss = F.cross_entropy(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            failures.append({
                "step": step, "basin_id": "ALL", "seed": args.seed,
                "reason": f"NaN/Inf loss at step {step}",
            })
            step_records.append({
                "step": step, "loss": float("nan"),
                "grad_norm_before": 0.0, "grad_norm_after": 0.0,
                "has_nan": 1, "accuracy": 0.0, "default_rate": 1.0,
                "entropy_mean": 0.5,
            })
            break

        loss.backward()

        params = list(router.parameters())
        g_before = math.sqrt(sum((p.grad.norm().item() ** 2) for p in params if p.grad is not None))
        torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip)
        g_after = math.sqrt(sum((p.grad.norm().item() ** 2) for p in params if p.grad is not None))
        opt.step()

        sel = [fids_registry[int(s)] for s in r_out["selected"]["recharge"]]
        correct = sum(1 for b in range(B) if sel[b] == aligned_oracle[b]["best_train_formula"])
        dr = float(sum(1 for s in sel if s == "R0") / B)
        ent_mean = float(r_out.get("entropy_recharge", torch.tensor(0.5)).mean().item())

        step_records.append({
            "step": step, "loss": round(float(loss.item()), 8),
            "grad_norm_before": round(g_before, 8),
            "grad_norm_after": round(g_after, 8),
            "has_nan": 0,
            "accuracy": round(correct / B, 4),
            "default_rate": round(dr, 4),
            "entropy_mean": round(ent_mean, 6),
        })

        if step % max(1, args.steps // 5) == 0:
            r0c = sel.count("R0")
            r4c = sel.count("R4")
            r5c = sel.count("R5")
            print(f"  step {step:3d} loss={loss.item():.4f} R0={r0c} R4={r4c} R5={r5c} correct={correct}/{B}")

    # Final router evaluation
    router.eval()
    with torch.no_grad():
        r_out_f = router(attr_t)
        sel_final = [fids_registry[int(s)] for s in r_out_f["selected"]["recharge"]]
        logits_f = r_out_f["logits"]["recharge"]
        probs_f = F.softmax(logits_f, dim=-1)
        entropies_f = r_out_f["entropy_recharge"]

    for b in range(B):
        bid = aligned_basin_ids[b]
        top1_idx = int(probs_f[b].argmax().item())
        selection_records.append({
            "basin_id": bid,
            "seed": args.seed,
            "selected_formula": fids_registry[top1_idx],
            "selected_formula_name": fids_registry[top1_idx],
            "selection_source": "router_logits",
            "label_source": "train_window_fixed_formula_calibration",
            "eval_used_for_selection": "False",
            "entropy": round(float(entropies_f[b].item()), 6),
            "top1_probability": round(float(probs_f[b].max().item()), 6),
        })

    # Build router vs oracle comparison
    router_vs_oracle = []
    for b in range(B):
        bid = aligned_basin_ids[b]
        oracle_fid = aligned_oracle[b]["best_train_formula"]
        router_fid = sel_final[b]
        router_vs_oracle.append({
            "basin_id": bid,
            "seed": args.seed,
            "oracle_formula": oracle_fid,
            "router_formula": router_fid,
            "match": router_fid == oracle_fid,
        })

    # Probability table
    prob_records = []
    for b in range(B):
        row = {"basin_id": aligned_basin_ids[b], "seed": args.seed}
        for i, fid in enumerate(fids_registry):
            row[f"prob_{fid}"] = round(float(probs_f[b, i].item()), 6)
        prob_records.append(row)

    # ---- Write outputs ----
    _w(step_records, out_dir / "router_training_steps.csv",
       ["step", "loss", "grad_norm_before", "grad_norm_after",
        "has_nan", "accuracy", "default_rate", "entropy_mean"])
    _w(selection_records, out_dir / "router_selection_summary.csv",
       ["basin_id", "seed", "selected_formula", "selected_formula_name",
        "selection_source", "label_source", "eval_used_for_selection",
        "entropy", "top1_probability"])
    _w(prob_records, out_dir / "router_probabilities.csv",
       ["basin_id", "seed"] + [f"prob_{fid}" for fid in fids_registry])
    _w(router_vs_oracle, out_dir / "router_vs_oracle_train.csv",
       ["basin_id", "seed", "oracle_formula", "router_formula", "match"])
    _w(failures, out_dir / "router_failures.csv",
       ["step", "basin_id", "seed", "reason"])

    # Report
    final_correct = sum(1 for r in router_vs_oracle if r["match"])
    report = [
        f"# StaticFormulaRouter Training — seed={args.seed}",
        f"",
        f"## Configuration",
        f"- anchor_bias: {args.anchor_bias}",
        f"- temperature: {args.temperature}",
        f"- steps: {args.steps}, lr: {args.lr}, grad_clip: {args.grad_clip}",
        f"- active_node: recharge",
        f"",
        f"## Results",
        f"- Router accuracy: {final_correct}/{B} ({final_correct/B*100:.1f}%)",
        f"- R0 selected: {sel_final.count('R0')}, R4: {sel_final.count('R4')}, R5: {sel_final.count('R5')}",
        f"",
        f"## Integrity",
        f"- selection_source: router_logits",
        f"- label_source: train_window_fixed_formula_calibration",
        f"- eval_used_for_selection: False",
        f"- eval_used_for_label: False",
    ]
    (out_dir / "run_report.md").write_text("\n".join(report))

    print(f"\nFinal: accuracy={final_correct}/{B}, R0={sel_final.count('R0')}, "
          f"R4={sel_final.count('R4')}, R5={sel_final.count('R5')}")
    print(f"Done. Output: {out_dir}")
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
    ap.add_argument("--oracle-labels", required=True)
    ap.add_argument("--active-nodes", type=str, default="recharge")
    ap.add_argument("--anchor-bias", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--temperature-final", type=float, default=0.7)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    run_router_training(args)
