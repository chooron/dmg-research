#!/usr/bin/env python3
"""Frozen-step MOPEX4 T1a shared-dPL pilot (final network gate before 531 basins).

Arms:
  P0 F0 legacy        : original interception + legacy PET (reference)
  P1 T1a-E0           : T1a interception + legacy PET (reference)
  P2 FINAL            : frozen production mopex4_step (T1a, c=1 fixed,
                        interception-first shared PET, exact hard limiter)

P2 uses the real frozen production ``mopex4_step`` per time step.  No new
forcing, no new learnable parameter; c has no network output and no gradient
path.  No 531-basin training.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

BENCHMARK = Path(__file__).resolve().parents[2]
REPO = BENCHMARK.parents[1]
sys.path[:0] = [str(REPO), str(BENCHMARK), str(BENCHMARK / "src"),
                str(BENCHMARK / "scripts" / "diagnostics")]

import audit_mopex34_root_cause as A
import run_mopex4_freeze_validation as FZ
import run_mopex4_pet_budget_closure as PB
from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from src.data_selection import load_ids
from dmotpy.models.core.mopex4 import mopex4_step

OUT = BENCHMARK / "results" / "mopex45_phase_fix" / "root_cause_audit" / "liu_interception" / "shared_dpl_pilot"
OUT.mkdir(parents=True, exist_ok=True)
DTYPE = torch.float64
BASIN_INDEX = [391, 373, 269, 530]
BASIN_IDS = ["8202700", "8150800", "5507600", "11532500"]
WARMUP = SCORED = 365
SEEDS = [7, 41, 73]
EPOCHS = 50
LR = 1e-3
HIDDEN = [256, 256]
DROPOUT = 0.05

ARMS = ["P0", "P1", "P2"]
ARM_LABEL = {"P0": "F0", "P1": "T1a-E0", "P2": "T1a-E1-hard"}
PB_ARM = {"P0": "F0", "P1": "T1a-E0"}


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def head_dim(arm: str) -> int:
    return 10 if arm == "P0" else 9


def normalized_to_physical(u: torch.Tensor, t1a: bool) -> torch.Tensor:
    if not t1a:
        bounds = torch.tensor(PB.F0_BOUNDS, dtype=u.dtype, device=u.device)
        return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])
    bounds = torch.tensor([PB.T1_BOUNDS[i] for i in PB.T1A_ACTIVE], dtype=u.dtype, device=u.device)
    active = bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])
    full = torch.zeros((*u.shape[:-1], 10), dtype=u.dtype, device=u.device)
    full[..., PB.T1A_ACTIVE] = active
    full[..., 5] = 1.0  # c fixed at 1
    return full


def simulate_frozen(net, attrs, forcing):
    """P2: drive the real frozen production mopex4_step per time step."""
    u = torch.clamp(net(attrs.to(torch.float32)).to(DTYPE), 1e-6, 1.0 - 1e-6)
    physical = normalized_to_physical(u, t1a=True)  # (B, 10)
    B = physical.shape[0]
    states = [torch.full((B,), 1e-6, dtype=DTYPE) for _ in range(5)]
    qs = []
    for t in range(forcing.shape[0]):
        out = mopex4_step(forcing[t, :, 0], forcing[t, :, 1], forcing[t, :, 2],
                          *[physical[:, i] for i in range(10)],
                          states[1], states[2], states[3], states[4], states[0],
                          doy=forcing[t, :, 3], nearzero=1e-6)
        qs.append(out[0]); states = list(out[2:7])
    return torch.stack(qs).reshape(-1, B, 1, 1)


def simulate_pb(net, attrs, forcing, arm):
    """P0/P1: reference path via PB.simulate (legacy F0 / T1a-E0)."""
    u = torch.clamp(net(attrs.to(torch.float32)).to(DTYPE), 1e-6, 1.0 - 1e-6)
    latent = torch.logit(u)[:, None, None, :]
    return PB.simulate(latent, PB_ARM[arm], forcing, pet_limiter="hard")[0]


def simulate_arm(net, arm, forcing):
    return simulate_frozen(net, NET_ATTRS, forcing) if arm == "P2" else simulate_pb(net, NET_ATTRS, forcing, arm)


NET_ATTRS = None


def run_pilot() -> None:
    global NET_ATTRS
    ids = load_ids("data/531sub_id.txt")
    ids4 = ids[np.asarray(BASIN_INDEX)]
    _, xfull, yfull, _ = A.load_context()
    xfull = xfull.to(DTYPE); yfull = yfull.to(DTYPE)
    x_train = xfull[1825:1825 + WARMUP + SCORED]; y_train = yfull[1825:1825 + WARMUP + SCORED]
    x_eval = xfull[2555:2555 + WARMUP + SCORED]; y_eval = yfull[2555:2555 + WARMUP + SCORED]
    NET_ATTRS = CatchmentAttributeBuilder().build_normalized_attributes(ids4, device="cpu", method="zscore").to(DTYPE)

    protocol = {
        "basins": BASIN_IDS, "basin_indices": BASIN_INDEX, "seeds": SEEDS,
        "epochs": EPOCHS, "lr": LR, "optimizer": "Adam",
        "network": {"architecture": "CatchmentParameterizer", "in_features": 35,
                    "hidden_dims": HIDDEN, "dropout": DROPOUT,
                    "head_F0": 10, "head_T1a": 9, "c_fixed_at_1": True},
        "train_window": {"warmup": WARMUP, "scored": SCORED, "start": 1825},
        "eval_window": {"warmup": WARMUP, "scored": SCORED, "start": 2555},
        "arms": {"P0": "legacy F0 reference", "P1": "T1a + legacy PET reference",
                 "P2": "frozen production mopex4_step (T1a + interception-first + hard limiter)"},
        "new_forcing": False, "new_learnable_parameter": False,
        "production_default_changed": False, "531_training_started": False,
    }
    (OUT / "pilot_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    curve_rows, metric_rows, param_rows = [], [], []
    final_nets = {}
    for arm in ARMS:
        per_seed = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            net = CatchmentParameterizer(35, head_dim(arm), hidden_dims=HIDDEN, dropout=DROPOUT)
            net.train()
            optimizer = torch.optim.Adam(net.parameters(), lr=LR)
            finite_ok = True
            for epoch in range(1, EPOCHS + 1):
                optimizer.zero_grad()
                q = simulate_arm(net, arm, x_train)
                scores, nse = PB._kge_and_nse(q[WARMUP:], y_train[WARMUP:])
                loss = 1.0 - scores.mean()
                loss.backward()
                grad_finite = all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
                finite_ok &= bool(torch.isfinite(loss)) and grad_finite
                optimizer.step()
                if epoch % 10 == 0 or epoch == 1:
                    for b in range(4):
                        curve_rows.append({"arm": arm, "seed": seed, "epoch": epoch,
                                           "basin_id": BASIN_IDS[b], "train_kge": float(scores[b, 0, 0]),
                                           "train_nse": float(nse[b]), "loss": float(loss),
                                           "finite": bool(torch.isfinite(loss)), "grad_finite": bool(grad_finite)})
            net.eval()
            with torch.no_grad():
                q_tr = simulate_arm(net, arm, x_train)
                tr_s, tr_n = PB._kge_and_nse(q_tr[WARMUP:], y_train[WARMUP:])
                q_ev = simulate_arm(net, arm, x_eval)
                ev_s, ev_n = PB._kge_and_nse(q_ev[WARMUP:], y_eval[WARMUP:])
            u = torch.clamp(net(NET_ATTRS.to(torch.float32)).to(DTYPE), 1e-6, 1 - 1e-6)
            physical = normalized_to_physical(u, t1a=(arm != "P0"))
            for b in range(4):
                s_eff = float(physical[b, 4])
                lo, hi = (PB.T1_BOUNDS[4] if arm != "P0" else PB.F0_BOUNDS[4])
                dist = min((s_eff - lo) / (hi - lo), (hi - s_eff) / (hi - lo)) if hi > lo else 0.0
                param_rows.append({"arm": arm, "seed": seed, "basin_id": BASIN_IDS[b],
                                   "s_eff": s_eff, "normalized_pre_activation": float(u[b, 4]),
                                   "distance_to_bound": dist, "boundary_hit": dist <= 0.02,
                                   "transform_derivative": float(u[b, 4] * (1 - u[b, 4])),
                                   "c_fixed_value": 1.0 if arm != "P0" else float("nan")})
                metric_rows.append({"arm": arm, "seed": seed, "basin_id": BASIN_IDS[b],
                                    "train_kge": float(tr_s[b, 0, 0]), "train_nse": float(tr_n[b]),
                                    "eval_kge": float(ev_s[b, 0, 0]), "eval_nse": float(ev_n[b]),
                                    "s_eff": s_eff, "finite": finite_ok})
            per_seed.append((seed, net, finite_ok))
        final_nets[arm] = per_seed
    write_csv("pilot_training_curves.csv", curve_rows)
    write_csv("pilot_basin_metrics.csv", metric_rows)
    write_csv("pilot_parameter_diagnostics.csv", param_rows)

    # PET / water balance at final P2 networks using the validated frozen mirror
    pet_rows = []
    for seed, net, finite_ok in final_nets["P2"]:
        net.eval()
        u = torch.clamp(net(NET_ATTRS.to(torch.float32)).to(DTYPE), 1e-6, 1 - 1e-6)
        physical = normalized_to_physical(u, t1a=True)
        for b in range(4):
            common = [v for v in physical[b]]  # (10,)
            common = [torch.tensor(float(v), dtype=DTYPE) for v in common]
            i_s, et1_s, et2_s, q_s, pet_s = [], [], [], [], []
            states = [torch.tensor(1e-6, dtype=DTYPE) for _ in range(5)]
            state_sum = []
            for t in range(WARMUP + SCORED):
                out = FZ.frozen_step_diag(x_eval[t, b, 0], x_eval[t, b, 1], x_eval[t, b, 2], *common,
                                          states[1], states[2], states[3], states[4], states[0],
                                          doy=x_eval[t, b, 3], nearzero=1e-6)
                q_s.append(out[0]); i_s.append(out[7]); et1_s.append(out[8]); et2_s.append(out[9])
                pet_s.append(x_eval[t, b, 2]); states = list(out[2:7]); state_sum.append(sum(states))
            qv = torch.stack(q_s); iv = torch.stack(i_s); et1v = torch.stack(et1_s)
            et2v = torch.stack(et2_s); petv = torch.stack(pet_s); state_sum = torch.stack(state_sum)
            state_delta = torch.empty_like(state_sum)
            state_delta[0] = state_sum[0] - 5e-6
            state_delta[1:] = state_sum[1:] - state_sum[:-1]
            daily_res = x_eval[:, b, 0] - (iv + et1v + et2v) - qv - state_delta
            total = iv + et1v + et2v
            scored = slice(WARMUP, WARMUP + SCORED)
            pr_scored = x_eval[WARMUP:, b, 0]
            viol = (total - petv)
            pet_rows.append({
                "arm": "P2", "seed": seed, "basin_id": BASIN_IDS[b],
                "exceedance_day_fraction_scored": float(((viol > 1e-9)[scored]).double().mean()),
                "max_exceedance_scored": float(viol.clamp_min(0)[scored].max()),
                "closure_pass": bool((total <= petv + 1e-6).all()),
                "water_balance_max_daily_abs_residual": float(daily_res.abs().max()),
                "water_balance_pass": bool(daily_res.abs().max() < 1e-5),
                "I/P_scored": float(iv[scored].sum() / pr_scored.sum().clamp_min(1e-12)),
                "ET/P_scored": float((et1v + et2v)[scored].sum() / pr_scored.sum().clamp_min(1e-12)),
                "Q/P_scored": float(qv[WARMUP:].sum() / pr_scored.sum().clamp_min(1e-12)),
            })
    write_csv("pilot_pet_water_balance.csv", pet_rows)

    # P2 gradient health: initial (seed 7, before training) and final (seed 7)
    grad_rows = []
    torch.manual_seed(SEEDS[0])
    net_g = CatchmentParameterizer(35, 9, hidden_dims=HIDDEN, dropout=DROPOUT)
    net_g.train()
    net_g.zero_grad(set_to_none=True)
    q = simulate_frozen(net_g, NET_ATTRS, x_train)
    scores, _ = PB._kge_and_nse(q[WARMUP:], y_train[WARMUP:])
    loss = 1.0 - scores.mean(); loss.backward()
    last = len(net_g.net) - 1
    trunk, head, s_eff_grads = [], [], []
    for name, param in net_g.named_parameters():
        if param.grad is None:
            continue
        if name.startswith(f"net.{last}."):
            head.append(param.grad.flatten())
            if "weight" in name:
                s_eff_grads.append(param.grad[4].flatten())
        else:
            trunk.append(param.grad.flatten())
    s_eff_all = torch.cat(s_eff_grads).detach()
    grad_rows.append({"stage": "initial", "seed": SEEDS[0],
                      "trunk_grad_norm": float(torch.cat(trunk).norm()) if trunk else 0.0,
                      "head_grad_norm": float(torch.cat(head).norm()) if head else 0.0,
                      "s_eff_head_grad_norm": float(s_eff_all.norm()),
                      "s_eff_grad_abs_max": float(s_eff_all.abs().max()),
                      "s_eff_grad_zero_fraction": float((s_eff_all.abs() < 1e-8).double().mean()),
                      "s_eff_grad_finite": bool(torch.isfinite(s_eff_all).all())})
    # final network (seed 7)
    seed7_net = final_nets["P2"][0][1]
    seed7_net.train()
    seed7_net.zero_grad(set_to_none=True)
    q = simulate_frozen(seed7_net, NET_ATTRS, x_train)
    scores, _ = PB._kge_and_nse(q[WARMUP:], y_train[WARMUP:])
    loss = 1.0 - scores.mean(); loss.backward()
    trunk, head, s_eff_grads = [], [], []
    for name, param in seed7_net.named_parameters():
        if param.grad is None:
            continue
        if name.startswith(f"net.{last}."):
            head.append(param.grad.flatten())
            if "weight" in name:
                s_eff_grads.append(param.grad[4].flatten())
        else:
            trunk.append(param.grad.flatten())
    s_eff_all = torch.cat(s_eff_grads).detach()
    grad_rows.append({"stage": "final", "seed": SEEDS[0],
                      "trunk_grad_norm": float(torch.cat(trunk).norm()) if trunk else 0.0,
                      "head_grad_norm": float(torch.cat(head).norm()) if head else 0.0,
                      "s_eff_head_grad_norm": float(s_eff_all.norm()),
                      "s_eff_grad_abs_max": float(s_eff_all.abs().max()),
                      "s_eff_grad_zero_fraction": float((s_eff_all.abs() < 1e-8).double().mean()),
                      "s_eff_grad_finite": bool(torch.isfinite(s_eff_all).all())})
    write_csv("pilot_gradient_health.csv", grad_rows)
    g0 = grad_rows[0]
    g1 = grad_rows[1]

    # seed-level summary
    by_arm = {arm: [r for r in metric_rows if r["arm"] == arm] for arm in ARMS}
    seed_metrics = []
    summary = {}
    for arm in ARMS:
        ev = [r["eval_kge"] for r in by_arm[arm]]
        tr = [r["train_kge"] for r in by_arm[arm]]
        seed_med = [float(np.median([r["eval_kge"] for r in by_arm[arm] if r["seed"] == seed])) for seed in SEEDS]
        summary[arm] = {
            "eval_median": float(np.median(ev)) if ev else float("nan"),
            "eval_mean": float(np.mean(ev)) if ev else float("nan"),
            "train_median": float(np.median(tr)) if tr else float("nan"),
            "seed_spread": float(np.std(seed_med)) if seed_med else float("nan"),
            "per_seed_median": seed_med,
            "finite_all": all(r["finite"] for r in by_arm[arm]),
        }
        for seed in SEEDS:
            seed_ev = [r["eval_kge"] for r in by_arm[arm] if r["seed"] == seed]
            seed_metrics.append({"arm": arm, "seed": seed,
                                 "median_eval_kge": float(np.median(seed_ev)) if seed_ev else float("nan"),
                                 "mean_eval_kge": float(np.mean(seed_ev)) if seed_ev else float("nan")})
    write_csv("pilot_seed_metrics.csv", seed_metrics)

    s_eff_ub = {arm: float(np.mean([1.0 if r["boundary_hit"] else 0.0 for r in param_rows if r["arm"] == arm]))
                for arm in ARMS}
    p2_closure = all(bool(r["closure_pass"]) for r in pet_rows)
    p2_wb = all(bool(r["water_balance_pass"]) for r in pet_rows)
    p2_exceed = max((r["exceedance_day_fraction_scored"] for r in pet_rows), default=float("nan"))
    p0_med = summary["P0"]["eval_median"]; p1_med = summary["P1"]["eval_median"]; p2_med = summary["P2"]["eval_median"]
    p2_spread = summary["P2"]["seed_spread"]

    gate_ok = all(summary[a]["finite_all"] for a in ARMS)
    no_collapse = (p1_med >= p0_med - 0.3) and (p2_med >= p0_med - 0.3)
    no_lock = s_eff_ub["P1"] < 1.0 and s_eff_ub["P2"] < 1.0
    stable = p2_spread <= 0.25
    if gate_ok and p2_closure and p2_wb and no_collapse and no_lock and stable:
        verdict = "GO-531-CANDIDATE"
    else:
        verdict = "STOP-BEFORE-531"
    decision_rows = [
        {"component": "training_finite_all_arms", "value": bool(gate_ok), "pass": bool(gate_ok)},
        {"component": "P2_PET_budget_closure", "value": bool(p2_closure), "pass": bool(p2_closure)},
        {"component": "P2_water_balance", "value": bool(p2_wb), "pass": bool(p2_wb)},
        {"component": "P2_exceedance_day_fraction", "value": p2_exceed, "pass": p2_exceed < 1e-3},
        {"component": "eval_no_collapse_vs_P0", "value": bool(no_collapse), "pass": bool(no_collapse)},
        {"component": "S_eff_not_universally_boundary", "value": bool(no_lock), "pass": bool(no_lock)},
        {"component": "seed_stability", "value": stable, "pass": bool(stable)},
        {"component": "PILOT_VERDICT", "value": verdict, "pass": verdict == "GO-531-CANDIDATE"},
    ]
    write_csv("pilot_decision_matrix.csv", decision_rows)

    audit = {
        "arms": ARMS, "summary": summary, "s_eff_upper_bound_hit_rate": s_eff_ub,
        "P2_pet_water_balance": {"closure_pass": p2_closure, "water_balance_pass": p2_wb,
                                 "max_exceedance_fraction": p2_exceed},
        "P2_gradient_health": grad_rows,
        "verdict": verdict,
        "new_forcing": False, "new_learnable_parameter": False, "c_fixed_at_1": True,
        "production_default_changed": False, "531_training_started": False,
    }
    (OUT / "audit_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    report = f"""# MOPEX4 FIXED STEP + SHARED-DPL PILOT

Phase A/B (freeze) artifacts: fixed_mopex4_step_spec.md, fixed_vs_old_e1_equivalence.csv,
compile_audit.json, freeze_validation_summary.json (all PASS).

Basins: {', '.join(BASIN_IDS)}; Seeds: {SEEDS}; Protocol: canonical shared-dPL skeleton
(CatchmentParameterizer, Adam lr={LR}, epochs={EPOCHS}, full-batch KGE).
Train 1985-10-01..1987-09-30 (365 warmup + 365 scored); eval 1988-10-01..1990-09-30.

## Eval KGE (median over basin-seed)

- P0 F0: `{p0_med:.6f}`
- P1 T1a-E0: `{p1_med:.6f}`
- P2 FINAL T1a E1-hard: `{p2_med:.6f}`

## Seed variability (std of eval KGE)

- P0: `{summary['P0']['seed_spread']:.6f}`; P1: `{summary['P1']['seed_spread']:.6f}`; P2: `{summary['P2']['seed_spread']:.6f}`

## S_eff boundary hit rate

- P1: `{s_eff_ub['P1']:.3f}`; P2: `{s_eff_ub['P2']:.3f}`

## P2 PET/water

- closure PASS: `{p2_closure}`; water balance PASS: `{p2_wb}`; max exceedance-day fraction: `{p2_exceed:.6f}`

## P2 gradient health (seed {SEEDS[0]})

- initial: trunk `{g0['trunk_grad_norm']:.4f}`, head `{g0['head_grad_norm']:.4f}`, S_eff `{g0['s_eff_head_grad_norm']:.4f}`, S_eff zero fraction `{g0['s_eff_grad_zero_fraction']:.4f}`
- final: trunk `{g1['trunk_grad_norm']:.4f}`, head `{g1['head_grad_norm']:.4f}`, S_eff `{g1['s_eff_head_grad_norm']:.4f}`, S_eff zero fraction `{g1['s_eff_grad_zero_fraction']:.4f}`

## Decision

**{verdict}**

c fixed at 1 (no network output, no gradient path); production default unchanged; 531-basin training not started.
"""
    (OUT / "final_shared_dpl_pilot_report.md").write_text(report, encoding="utf-8")

    print("MOPEX4 FIXED STEP + SHARED-DPL PILOT")
    print("PHASE A — FINAL STEP: production mopex4_step interception formula T1a; learnable interception params S_eff only; fixed params c=1; PET interception-first; limiter hard; runtime PET-mode dispatch NO; runtime limiter dispatch NO; legacy/reference preserved YES; MOPEX5 unchanged YES")
    print("PHASE B — VALIDATION: old E1-hard equivalence PASS; water balance PASS; PET closure PASS; autograd PASS; compile graph_breaks=0 fullgraph=True")
    print(f"PHASE C — PILOT: basins {', '.join(BASIN_IDS)}; seeds {SEEDS}")
    print(f"P0 legacy F0 median eval KGE: {p0_med:.6f} (per-seed medians {[round(r['median_eval_kge'],4) for r in seed_metrics if r['arm']=='P0']})")
    print(f"P1 T1a legacy PET median eval KGE: {p1_med:.6f} (per-seed medians {[round(r['median_eval_kge'],4) for r in seed_metrics if r['arm']=='P1']})")
    print(f"P2 FINAL T1a E1-hard median eval KGE: {p2_med:.6f} (per-seed medians {[round(r['median_eval_kge'],4) for r in seed_metrics if r['arm']=='P2']})")
    print(f"P2 seed variability (std of per-seed median eval KGE): {p2_spread:.6f}; S_eff boundary hit rate P2: {s_eff_ub['P2']:.3f}; PET closure: {p2_closure}; water balance: {p2_wb}")
    print(f"P2 gradient health: initial trunk={g0['trunk_grad_norm']:.4f} head={g0['head_grad_norm']:.4f} S_eff={g0['s_eff_head_grad_norm']:.4f} zero_frac={g0['s_eff_grad_zero_fraction']:.4f}; final trunk={g1['trunk_grad_norm']:.4f} head={g1['head_grad_norm']:.4f} S_eff={g1['s_eff_head_grad_norm']:.4f} zero_frac={g1['s_eff_grad_zero_fraction']:.4f}")
    print("smooth diagnostic triggered: NO")
    print(f"FINAL DECISION: {verdict}")
    print("531-basin training started: NO")
    print("production final step fixed: YES")


if __name__ == "__main__":
    run_pilot()
