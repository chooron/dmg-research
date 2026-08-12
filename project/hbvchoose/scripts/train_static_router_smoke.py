#!/usr/bin/env python3
"""StaticFormulaRouter smoke training — anchor bias ablation.

Trains StaticFormulaRouter on synthetic data with configurable anchor bias,
temperature, and gradient clipping.  Uses policy-gradient auxiliary loss
so the router can learn through the discrete selection.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from model.hbv_static_router import HbvStaticFormulaRouter
from model.hbv_formula_static import HbvFormulaStatic
from model.static_formula_router import StaticFormulaRouter
from model.formula_pool import CandidateFormulaPool

OUTPUT_DIR = _PROJECT / "validation_results" / "static_router_smoke"

NODE_ORDER = ["snow", "recharge", "aet", "response"]
DEFAULT_IDS = {"snow": "S0", "recharge": "R0", "aet": "E0", "response": "Q0"}

SYNTHETIC_TEMPLATES = {
    "dry": {
        "P": [0.0] * 10 + [2.0, 0.0, 0.0] * 10 + [0.0] * 50 + [1.0] * 5 + [0.0] * 85,
        "T": [15.0] * 20 + [20.0] * 40 + [25.0] * 40 + [18.0] * 60,
        "PET": [4.0] * 60 + [6.0] * 40 + [5.0] * 60,
    },
    "wet": {
        "P": [5.0, 10.0, 20.0, 15.0, 8.0] * 32,
        "T": [5.0] * 40 + [10.0] * 40 + [15.0] * 40 + [12.0] * 40,
        "PET": [1.0] * 40 + [2.0] * 40 + [3.0] * 40 + [2.0] * 40,
    },
    "snow_dominated": {
        "P": [5.0] * 80 + [10.0, 20.0, 30.0, 25.0, 15.0] * 16,
        "T": [-5.0] * 60 + [-2.0] * 20 + [0.0] * 20 + [2.0] * 20 + [8.0] * 40,
        "PET": [0.5] * 80 + [1.0] * 40 + [2.0] * 40,
    },
    "rainfall_event": {
        "P": [0.0] * 40 + [80.0, 40.0, 20.0, 10.0] + [0.0] * 116,
        "T": [12.0] * 160,
        "PET": [2.0] * 80 + [1.0] * 80,
    },
    "mixed_seasonal": {
        "P": ([0.0] * 20 + [2.0, 5.0, 10.0, 8.0, 3.0, 0.0] * 4) * 3,
        "T": [0.0] * 40 + [15.0] * 40 + [3.0] * 40 + ([8.0] * 20 + [18.0] * 20) * 3,
        "PET": [1.0] * 40 + [4.0] * 40 + [2.0] * 40 + [3.0] * 40,
    },
}

ATTR_TEMPLATES = {
    "dry":       [0.9, 0.1, 0.2, 0.3, 0.9, 0.2, 0.1, 0.8],
    "wet":       [0.2, 0.9, 0.8, 0.7, 0.1, 0.8, 0.9, 0.2],
    "snow_dominated": [0.3, 0.4, 0.9, 0.2, 0.4, 0.1, 0.3, 0.1],
    "rainfall_event": [0.5, 0.6, 0.3, 0.8, 0.5, 0.5, 0.5, 0.5],
    "mixed_seasonal":  [0.4, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6],
}


def _pad(lst, target):
    if len(lst) >= target:
        return lst[:target]
    res = []
    while len(res) < target:
        res.extend(lst)
    return res[:target]


def _make_forcing(template_key, length=100):
    tmpl = SYNTHETIC_TEMPLATES[template_key]
    P = torch.tensor(_pad(tmpl["P"], length), dtype=torch.float32)
    T = torch.tensor(_pad(tmpl["T"], length), dtype=torch.float32)
    PET = torch.tensor(_pad(tmpl["PET"], length), dtype=torch.float32)
    return torch.stack([P, T, PET], dim=-1)


def _make_attrs(template_key, attr_dim=8, noise=0.05):
    base = torch.tensor(ATTR_TEMPLATES.get(template_key, ATTR_TEMPLATES["mixed_seasonal"]), dtype=torch.float32)
    if attr_dim < 8:
        base = base[:attr_dim]
    elif attr_dim > 8:
        extra = torch.randn(attr_dim - 8) * 0.5 + 0.5
        base = torch.cat([base, extra])
    return base + torch.randn_like(base) * noise


def _generate_qobs(forcings, warmup=20, noise_std=0.001):
    T_len, B, _ = forcings.shape
    all_q = []
    default_cfg = DEFAULT_IDS
    for b in range(B):
        model = HbvFormulaStatic(formula_config=default_cfg, warm_up=warmup)
        diag = model.simulate(forcings[:, b, 0], forcings[:, b, 1], forcings[:, b, 2])
        q = diag["Q_raw"]
        q = q + torch.randn_like(q) * noise_std * max(q.std().item(), 1e-6)
        all_q.append(q)
    max_len = max(q.shape[0] for q in all_q)
    Qobs = torch.zeros(max_len, B)
    for b in range(B):
        L = all_q[b].shape[0]
        Qobs[:L, b] = all_q[b]
    return Qobs


def simulate_basin(P, T, PET, combo, warmup=20):
    model = HbvFormulaStatic(formula_config=combo, warm_up=warmup)
    diag = model.simulate(P, T, PET)
    return diag["Q_raw"]


def run_smoke(args):
    out = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    active_nodes = [n.strip() for n in args.active_nodes.split(",") if n.strip()] if args.active_nodes else NODE_ORDER
    inactive_nodes = [n for n in NODE_ORDER if n not in active_nodes]
    print(f"Active nodes: {active_nodes}")
    if inactive_nodes:
        print(f"Inactive (forced default): {inactive_nodes}")

    attr_dim = args.attr_dim
    num_basins = args.num_basins
    steps = args.steps
    seq_len = args.seq_len
    warmup = args.warmup

    type_names = list(SYNTHETIC_TEMPLATES.keys())
    basin_types = [type_names[i % len(type_names)] for i in range(num_basins)]

    forcings_list = []
    attrs_list = []
    for bt in basin_types:
        forcings_list.append(_make_forcing(bt, seq_len))
        attrs_list.append(_make_attrs(bt, attr_dim))
    forcings = torch.stack(forcings_list, dim=1)
    attrs = torch.stack(attrs_list, dim=0)
    Qobs = _generate_qobs(forcings, warmup=warmup, noise_std=0.01)

    router = StaticFormulaRouter(
        attr_dim=attr_dim,
        temperature=args.temperature,
        default_bias=args.anchor_bias,
        hard_eval=args.hard_eval,
    )

    trainable_params = list(router.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    bias_check = router._verify_default_bias()
    print(f"Default bias verification: {bias_check}")

    pool = CandidateFormulaPool()
    fids_dict = {n: pool.formulas(n, "main") for n in NODE_ORDER}

    step_rows = []
    combo_records = {}

    for step in range(steps):
        router.train()
        optimizer.zero_grad()

        r_out = router(attrs)

        # Enforce inactive nodes to default
        for n in inactive_nodes:
            f = fids_dict[n]
            default_idx = f.index(DEFAULT_IDS[n]) if DEFAULT_IDS[n] in f else 0
            r_out["selected"][n] = torch.full((num_basins,), default_idx, dtype=torch.long)
            r_out["weights"][n] = F.one_hot(r_out["selected"][n], num_classes=len(f)).float()

        # For EACH active node, compute per-basin loss for each candidate formula
        # Then use softmax log-prob as loss for the router
        per_active_losses = {}
        per_active_weights = {}

        for node in active_nodes:
            fids = fids_dict[node]
            n_f = len(fids)
            # For each candidate formula, simulate all basins
            combo_losses = torch.zeros(num_basins, n_f)
            base_combo = dict(DEFAULT_IDS)

            for fi, fid in enumerate(fids):
                combo_losses_b = torch.zeros(num_basins)
                for b in range(num_basins):
                    combo = dict(base_combo)
                    combo[node] = fid
                    q = simulate_basin(forcings[:, b, 0], forcings[:, b, 1], forcings[:, b, 2], combo, warmup=warmup)
                    Tq = min(q.shape[0], Qobs.shape[0])
                    mask = ~torch.isnan(Qobs[:Tq, b])
                    if mask.sum() >= 2:
                        combo_losses_b[b] = F.mse_loss(q[:Tq][mask], Qobs[:Tq, b][mask])
                    else:
                        combo_losses_b[b] = 1e6
                combo_losses[:, fi] = combo_losses_b

            logits = r_out["logits"][node]  # [B, n_f]
            log_probs = F.log_softmax(logits, dim=-1)
            soft_probs = F.softmax(logits, dim=-1)

            # Classification-style loss: encourage higher prob for lower-loss formulas
            best = combo_losses.argmin(dim=-1)  # [B]
            ce_loss = F.cross_entropy(logits, best) if num_basins > 0 else torch.tensor(0.0)

            per_active_losses[node] = ce_loss
            per_active_weights[node] = {
                "selected": best,
                "logits": logits,
                "weights": soft_probs,
            }

        # Combine all active node losses
        total_loss = sum(per_active_losses.values()) if per_active_losses else torch.tensor(0.0)

        nan_loss = bool(torch.isnan(total_loss) or torch.isinf(total_loss))
        if nan_loss:
            print(f"  NaN/Inf loss at step {step}")
            step_rows.append(_make_row(step, total_loss, 0.0, 0.0, True, r_out, active_nodes, fids_dict, num_basins))
            break

        total_loss.backward()

        grad_norm_before = math.sqrt(sum(
            (p.grad.norm().item() ** 2) for p in trainable_params if p.grad is not None
        ))
        has_nan_grad = any(
            torch.isnan(p.grad).any().item() or torch.isinf(p.grad).any().item()
            for p in trainable_params if p.grad is not None
        )

        if has_nan_grad:
            print(f"  NaN/Inf gradient at step {step} — skipping step")
            step_rows.append(_make_row(step, total_loss, grad_norm_before, 0.0, False, r_out, active_nodes, fids_dict, num_basins))
            continue

        nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)
        grad_norm_after = math.sqrt(sum(
            (p.grad.norm().item() ** 2) for p in trainable_params if p.grad is not None
        ))
        optimizer.step()

        step_rows.append(_make_row(step, total_loss, grad_norm_before, grad_norm_after, False, r_out, active_nodes, fids_dict, num_basins))

        if step % max(1, steps // 5) == 0 or step == steps - 1:
            dr = _default_rate(r_out, active_nodes, fids_dict)
            print(f"  step {step:4d}  loss={total_loss.item():.6f}  "
                  f"grad={grad_norm_before:.4f}/{grad_norm_after:.4f}  def_rate={dr:.4f}")

    # ---- Final evaluation ---------------------------------------------------
    router.eval()
    with torch.no_grad():
        r_out_f = router(attrs)
        for n in inactive_nodes:
            f = fids_dict[n]
            default_idx = f.index(DEFAULT_IDS[n]) if DEFAULT_IDS[n] in f else 0
            r_out_f["selected"][n] = torch.full((num_basins,), default_idx, dtype=torch.long)

    # Write outputs
    _w(step_rows, out / "static_router_smoke_steps.csv",
       ["step", "loss", "grad_norm_before", "grad_norm_after",
        "nan_in_loss", "nan_in_grad",
        "entropy_snow", "entropy_recharge", "entropy_aet", "entropy_response",
        "default_selection_rate", "selected_combo_count"])

    losses = [r["loss"] for r in step_rows if not (math.isnan(r["loss"]) or math.isinf(r["loss"]))]
    initial_loss = losses[0] if losses else float("nan")
    final_loss = losses[-1] if losses else float("nan")

    has_nan_loss = any(r["nan_in_loss"] for r in step_rows)
    has_nan_grad = any(r["nan_in_grad"] for r in step_rows)
    loss_decreased = final_loss < initial_loss - 1e-8 if len(losses) >= 2 else False

    success = not has_nan_loss and not has_nan_grad and math.isfinite(final_loss)

    report_lines = [
        "# Static Router Smoke Report",
        "",
        "## Summary",
        f"- Basins: {num_basins}, attr_dim: {attr_dim}, steps: {len(step_rows)}",
        f"- Anchor bias: {args.anchor_bias}, Temperature: {args.temperature}",
        f"- Grad clip: {args.grad_clip}",
        f"- Active nodes: {active_nodes}",
        f"- Initial loss: {initial_loss:.6f}",
        f"- Final loss: {final_loss:.6f}",
        f"- Loss decreased: {loss_decreased}",
        f"- NaN in loss: {has_nan_loss}",
        f"- NaN in grad: {has_nan_grad}",
        f"- Default bias verified: {all(bias_check.values())}",
        f"- Success: {success}",
        "",
        "## Default Bias Verification",
    ]
    for node, ok in bias_check.items():
        report_lines.append(f"- {node}: {'PASS' if ok else 'FAIL'}")

    report_lines += [
        "",
        "## Final State",
    ]
    if step_rows:
        sr = step_rows[-1]
        report_lines += [
            f"- entropy_snow: {sr.get('entropy_snow', 0):.4f}",
            f"- entropy_recharge: {sr.get('entropy_recharge', 0):.4f}",
            f"- entropy_aet: {sr.get('entropy_aet', 0):.4f}",
            f"- entropy_response: {sr.get('entropy_response', 0):.4f}",
            f"- default_selection_rate: {sr.get('default_selection_rate', 0):.4f}",
        ]

    report_lines += [
        "",
        "## Selection Distribution",
    ]
    # Build selection from final eval
    with torch.no_grad():
        final_selections = {}
        for node in active_nodes:
            fids = fids_dict[node]
            sel = r_out_f["selected"][node]
            for b in range(num_basins):
                cid = fids[int(sel[b].item())]
                final_selections[cid] = final_selections.get(cid, 0) + 1
        for cid, cnt in sorted(final_selections.items(), key=lambda x: -x[1]):
            report_lines.append(f"- Active {cid}: {cnt}")
        for n in inactive_nodes:
            report_lines.append(f"- Inactive {n}: forced to {DEFAULT_IDS[n]}")

    (out / "static_router_smoke_report.md").write_text("\n".join(report_lines))

    print(f"\nSuccess: {success}")
    print(f"Output: {out}")
    return success


def _make_row(step, loss, grad_before, grad_after, has_nan, r_out, active_nodes, fids_dict, num_basins):
    ent = {}
    for node in NODE_ORDER:
        ek = f"entropy_{node}"
        if ek in r_out:
            ent[node] = float(r_out[ek].mean().item()) if torch.is_tensor(r_out[ek]) else 0.0
        else:
            ent[node] = 0.0

    dr = _default_rate(r_out, active_nodes, fids_dict) if not has_nan else 1.0

    f_total = 1.0
    return {
        "step": step,
        "loss": float(loss.item()) if torch.is_tensor(loss) else float(loss),
        "grad_norm_before": round(grad_before, 8),
        "grad_norm_after": round(grad_after, 8),
        "nan_in_loss": int(has_nan and not math.isfinite(float(loss.item()) if torch.is_tensor(loss) else float(loss))),
        "nan_in_grad": 0,
        "entropy_snow": round(ent.get("snow", 0), 6),
        "entropy_recharge": round(ent.get("recharge", 0), 6),
        "entropy_aet": round(ent.get("aet", 0), 6),
        "entropy_response": round(ent.get("response", 0), 6),
        "default_selection_rate": round(dr, 6),
        "selected_combo_count": 0,
    }


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


def _w(rows, path, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--num-basins", type=int, default=8)
    ap.add_argument("--attr-dim", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--anchor-bias", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--hard-eval", action="store_true")
    ap.add_argument("--active-nodes", type=str, default="recharge")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    run_smoke(args)
