#!/usr/bin/env python3
"""MOPEX4 T1a small shared-dPL pilot (network-level gate before 531 basins).

Arms:
  A0 F0 legacy            : original interception + legacy PET semantics
  A1 T1a-E0               : Liu T1a + legacy PET semantics
  A2 T1a-E1-hard          : Liu T1a + interception-first shared PET, hard limiter
  A3 T1a-E1-smooth        : Liu T1a + interception-first shared PET, smooth limiter

The shared network is the existing CatchmentParameterizer (35 attributes ->
[256,256] -> head).  For T1a arms the head has 9 outputs (c is NOT learnable and
is fixed at 1 at forward time), preserving the public 10-slot MOPEX4 interface.
No 531-basin training and no shared-dPL network is used beyond this pilot.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

BENCHMARK = Path(__file__).resolve().parents[2]
REPO = BENCHMARK.parents[1]
sys.path[:0] = [str(REPO), str(BENCHMARK), str(BENCHMARK / "src"),
                str(BENCHMARK / "scripts" / "diagnostics")]

import audit_mopex34_root_cause as A
import run_mopex4_pet_budget_closure as PB
from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from src.data_selection import load_ids

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

ARMS = {
    "A0": {"label": "F0", "pet_budget": "legacy", "pet_limiter": "hard"},
    "A1": {"label": "T1a-E0", "pet_budget": "legacy", "pet_limiter": "hard"},
    "A2": {"label": "T1a-E1-hard", "pet_budget": "interception_first", "pet_limiter": "hard"},
    "A3": {"label": "T1a-E1-smooth", "pet_budget": "interception_first", "pet_limiter": "smooth"},
}

F0_BOUNDS = PB.F0_BOUNDS
T1_BOUNDS = PB.T1_BOUNDS
T1A_ACTIVE = PB.T1A_ACTIVE


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def head_dim(arm: str) -> int:
    return 10 if arm == "A0" else 9


def normalized_to_physical(u: torch.Tensor, arm: str) -> torch.Tensor:
    """Map normalized [0,1] network head outputs to the 10 physical slots."""
    if arm == "A0":
        bounds = torch.tensor(F0_BOUNDS, dtype=u.dtype, device=u.device)
        return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])
    bounds = torch.tensor([T1_BOUNDS[i] for i in T1A_ACTIVE], dtype=u.dtype, device=u.device)
    active = bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])
    full = torch.zeros((*u.shape[:-1], 10), dtype=u.dtype, device=u.device)
    full[..., T1A_ACTIVE] = active
    full[..., 5] = 1.0  # c fixed at 1
    return full


def net_forward(net: CatchmentParameterizer, arm: str, attrs: torch.Tensor) -> torch.Tensor:
    """Network -> normalized head -> logits for PB.simulate."""
    u = net(attrs.to(torch.float32)).to(DTYPE)
    u = torch.clamp(u, 1e-6, 1.0 - 1e-6)
    return torch.logit(u)


def simulate_arm(net: CatchmentParameterizer, arm: str, forcing: torch.Tensor,
                 pet_budget: str, pet_limiter: str, collect: bool = False):
    attrs = net_forward(net, arm, NET_ATTRS)
    latent = attrs[:, None, None, :]
    return PB.simulate(latent, arm_label(arm), forcing, collect=collect, pet_limiter=pet_limiter)


NET_ATTRS = None  # set in main


def arm_label(arm: str) -> str:
    """PB-compatible arm string (drives budget mode inside PB.simulate)."""
    if arm == "A0":
        return "F0"
    if arm == "A1":
        return "T1a-E0"
    return "T1a-E1"  # A2/A3 share interception-first; limiter passed separately




def run_pilot() -> None:
    global NET_ATTRS
    # ---- data ----
    ids = load_ids("data/531sub_id.txt")
    ids4 = ids[np.asarray(BASIN_INDEX)]
    _, xfull, yfull, _ = A.load_context()
    xfull = xfull.to(DTYPE)
    yfull = yfull.to(DTYPE)
    x_train = xfull[1825:1825 + WARMUP + SCORED]
    y_train = yfull[1825:1825 + WARMUP + SCORED]
    x_eval = xfull[2555:2555 + WARMUP + SCORED]
    y_eval = yfull[2555:2555 + WARMUP + SCORED]
    attr_builder = CatchmentAttributeBuilder()
    NET_ATTRS = attr_builder.build_normalized_attributes(ids4, device="cpu", method="zscore").to(DTYPE)

    protocol = {
        "basins": BASIN_IDS, "basin_indices": BASIN_INDEX,
        "train_window": {"warmup": WARMUP, "scored": SCORED, "start": 1825},
        "eval_window": {"warmup": WARMUP, "scored": SCORED, "start": 2555},
        "seeds": SEEDS, "epochs": EPOCHS, "lr": LR, "optimizer": "Adam",
        "network": {"architecture": "CatchmentParameterizer", "in_features": 35,
                    "hidden_dims": HIDDEN, "dropout": DROPOUT,
                    "head_F0": 10, "head_T1a": 9, "c_fixed_at_1": True},
        "loss": "mean(1 - KGE) over basins, full-batch, scored 365 days",
        "temporal_split_source": "fixed root-cause window; train 1985-10-01..1987-09-30, eval 1988-10-01..1990-09-30 (each 365 warmup + 365 scored)",
        "arms": ARMS,
        "new_forcing": False, "new_learnable_parameter": False,
        "production_default_changed": False, "531_training_started": False,
    }
    (OUT / "pilot_protocol_and_seeds.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    write_csv("arm_config_summary.csv", [
        {"arm": k, "interception": v["label"], "pet_budget": v["pet_budget"],
         "pet_limiter": v["pet_limiter"], "head_outputs": head_dim(k)}
        for k, v in ARMS.items()
    ])

    # ---- Stage 4: pre-training gradient audit (matched init, seed 7) ----
    audit_rows = []
    torch.manual_seed(SEEDS[0])
    net_audit = CatchmentParameterizer(35, 9, hidden_dims=HIDDEN, dropout=DROPOUT)
    net_audit.train()
    audit_per_arm = {}
    for arm in ["A1", "A2", "A3"]:
        cfg = ARMS[arm]
        net_audit.zero_grad(set_to_none=True)
        q, _, _ = simulate_arm(net_audit, arm, x_train, cfg["pet_budget"], cfg["pet_limiter"])
        scores, _ = PB._kge_and_nse(q[WARMUP:], y_train[WARMUP:])
        loss = 1.0 - scores.mean()
        loss.backward()
        trunk = []
        head = []
        s_eff_grads = []
        last = len(net_audit.net) - 1
        for name, param in net_audit.named_parameters():
            if param.grad is None:
                continue
            if name.startswith(f"net.{last}."):
                head.append(param.grad.flatten())
                if "weight" in name:
                    s_eff_grads.append(param.grad[4].flatten())  # head row for S_eff
            else:
                trunk.append(param.grad.flatten())
        trunk_g = torch.cat(trunk).norm().item() if trunk else 0.0
        head_g = torch.cat(head).norm().item() if head else 0.0
        s_eff_all = torch.cat(s_eff_grads).detach()
        zero_frac = float((s_eff_all.abs() < 1e-8).double().mean())
        audit_per_arm[arm] = torch.cat(trunk).detach() if trunk else torch.zeros(0)
        audit_rows.append({
            "arm": arm, "seed": SEEDS[0], "loss": loss.item(),
            "trunk_grad_norm": trunk_g, "head_grad_norm": head_g,
            "s_eff_head_grad_norm": s_eff_all.norm().item(),
            "s_eff_grad_abs_max": float(s_eff_all.abs().max()),
            "s_eff_grad_zero_fraction": zero_frac,
            "s_eff_grad_finite": bool(torch.isfinite(s_eff_all).all()),
            "loss_finite": bool(torch.isfinite(loss)),
        })
    write_csv("initial_gradient_audit.csv", audit_rows)
    # A2 vs A3 gradient cosine on shared (trunk) parameters
    if audit_per_arm["A2"].numel() and audit_per_arm["A3"].numel():
        cos_shared = float(torch.nn.functional.cosine_similarity(
            audit_per_arm["A2"].reshape(1, -1), audit_per_arm["A3"].reshape(1, -1)).item())
    else:
        cos_shared = float("nan")
    write_csv("hard_vs_smooth_gradient_cosine.csv", [{"pair": "A2_vs_A3", "shared_trunk_cosine": cos_shared}])

    # ---- Stage 5: training ----
    curve_rows, metric_rows, s_eff_rows = [], [], []
    final_states = {}
    for arm in ARMS:
        cfg = ARMS[arm]
        arm_metric_rows = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            net = CatchmentParameterizer(35, head_dim(arm), hidden_dims=HIDDEN, dropout=DROPOUT)
            net.train()
            optimizer = torch.optim.Adam(net.parameters(), lr=LR)
            finite_ok = True
            for epoch in range(1, EPOCHS + 1):
                optimizer.zero_grad()
                q, _, _ = simulate_arm(net, arm, x_train, cfg["pet_budget"], cfg["pet_limiter"])
                scores, nse = PB._kge_and_nse(q[WARMUP:], y_train[WARMUP:])
                loss = 1.0 - scores.mean()
                loss.backward()
                grad_finite = all(p.grad is not None and torch.isfinite(p.grad).all()
                                  for p in net.parameters())
                finite_ok &= bool(torch.isfinite(loss)) and grad_finite
                optimizer.step()
                if epoch % 10 == 0 or epoch == 1:
                    for b in range(4):
                        curve_rows.append({
                            "arm": arm, "seed": seed, "epoch": epoch,
                            "basin_id": BASIN_IDS[b], "train_kge": float(scores[b, 0, 0]),
                            "train_nse": float(nse[b]), "loss": float(loss),
                            "finite": bool(torch.isfinite(loss)), "grad_finite": bool(grad_finite),
                        })
            # evaluate
            net.eval()
            with torch.no_grad():
                q_train, _, _ = simulate_arm(net, arm, x_train, cfg["pet_budget"], cfg["pet_limiter"])
                tr_scores, tr_nse = PB._kge_and_nse(q_train[WARMUP:], y_train[WARMUP:])
                q_eval, _, _ = simulate_arm(net, arm, x_eval, cfg["pet_budget"], cfg["pet_limiter"])
                ev_scores, ev_nse = PB._kge_and_nse(q_eval[WARMUP:], y_eval[WARMUP:])
            u = net(NET_ATTRS.to(torch.float32)).to(DTYPE)
            physical = normalized_to_physical(u, arm)
            for b in range(4):
                s_eff = float(physical[b, 4])
                lo, hi = T1_BOUNDS[4] if arm != "A0" else F0_BOUNDS[4]
                dist = min((s_eff - lo) / (hi - lo), (hi - s_eff) / (hi - lo)) if hi > lo else 0.0
                s_eff_rows.append({"arm": arm, "seed": seed, "basin_id": BASIN_IDS[b],
                                   "s_eff_physical": s_eff, "normalized_pre_activation": float(u[b, 4]),
                                   "distance_to_bound": dist, "boundary_hit": dist <= 0.02,
                                   "transform_derivative": float(u[b, 4] * (1 - u[b, 4]))})
                metric_rows.append({
                    "arm": arm, "seed": seed, "basin_id": BASIN_IDS[b],
                    "train_kge": float(tr_scores[b, 0, 0]), "train_nse": float(tr_nse[b]),
                    "eval_kge": float(ev_scores[b, 0, 0]), "eval_nse": float(ev_nse[b]),
                    "s_eff": s_eff, "finite": finite_ok,
                })
            arm_metric_rows.append((seed, net, finite_ok))
        final_states[arm] = arm_metric_rows
    write_csv("training_curves.csv", curve_rows)
    write_csv("per_seed_per_basin_metrics.csv", metric_rows)
    write_csv("s_eff_boundary_audit.csv", s_eff_rows)

    # ---- PET budget + water balance + parameter drift at final networks ----
    pet_rows, drift_rows = [], []
    for arm in ARMS:
        cfg = ARMS[arm]
        for seed, net, finite_ok in final_states[arm]:
            net.eval()
            with torch.no_grad():
                q, _, diag = simulate_arm(net, arm, x_eval, cfg["pet_budget"], cfg["pet_limiter"], collect=True)
            if diag is None:
                continue
            for b in range(4):
                i = diag["i"][:, b, 0, 0]; et1 = diag["et1"][:, b, 0, 0]
                et2 = diag["et2"][:, b, 0, 0]; pet = diag["pet"][:, b, 0, 0]
                pr = diag["pr"][:, b, 0, 0]
                total = i + et1 + et2
                exceed = (total - pet).clamp_min(0)
                scored = slice(WARMUP, WARMUP + SCORED)
                pet_rows.append({
                    "arm": arm, "seed": seed, "basin_id": BASIN_IDS[b],
                    "exceedance_day_fraction_scored": float((exceed[scored] > 0).double().mean()),
                    "max_exceedance_scored": float(exceed[scored].max()),
                    "sum_exceedance_scored": float(exceed[scored].sum()),
                    "closure_max_violation": float((total - pet).max()),
                    "I_plus_ET1_plus_ET2_le_PET": bool((total <= pet + 1e-6).all()),
                    "I/P_scored": float(i[scored].sum() / pr[scored].sum().clamp_min(1e-12)),
                    "ET/P_scored": float((et1 + et2)[scored].sum() / pr[scored].sum().clamp_min(1e-12)),
                    "Q/P_scored": float(q[WARMUP:, b, 0, 0].sum() / pr[scored].sum().clamp_min(1e-12)),
                })
            # parameter drift vs A0 (same seed): common slots
            u = net(NET_ATTRS.to(torch.float32)).to(DTYPE)
            physical = normalized_to_physical(u, arm)
            for b in range(4):
                row = {"arm": arm, "seed": seed, "basin_id": BASIN_IDS[b]}
                for idx, name in [(0, "tcrit"), (1, "ddf"), (2, "Sb1"), (3, "tw"),
                                  (6, "tu"), (7, "se"), (8, "Sb2"), (9, "tc")]:
                    row[f"{name}_physical"] = float(physical[b, idx])
                drift_rows.append(row)
    write_csv("pet_budget_audit.csv", pet_rows)
    write_csv("parameter_drift_summary.csv", drift_rows)

    # ---- Stage 7/8: aggregate + hard vs smooth + decision ----
    by_arm = {arm: [r for r in metric_rows if r["arm"] == arm] for arm in ARMS}
    summary = {}
    for arm in ARMS:
        ev = [r["eval_kge"] for r in by_arm[arm]]
        tr = [r["train_kge"] for r in by_arm[arm]]
        summary[arm] = {
            "eval_kge_median": float(np.median(ev)) if ev else float("nan"),
            "eval_kge_mean": float(np.mean(ev)) if ev else float("nan"),
            "train_kge_median": float(np.median(tr)) if tr else float("nan"),
            "seed_spread_eval": float(np.std(ev)) if ev else float("nan"),
            "finite_all": all(r["finite"] for r in by_arm[arm]),
        }
    s_eff_ub = {arm: float(np.mean([1.0 if r["boundary_hit"] else 0.0 for r in s_eff_rows if r["arm"] == arm]))
                for arm in ARMS}
    pet_exceed = {arm: max((r["exceedance_day_fraction_scored"] for r in pet_rows if r["arm"] == arm), default=float("nan"))
                  for arm in ARMS}
    a1_med = summary["A1"]["eval_kge_median"]; a2_med = summary["A2"]["eval_kge_median"]
    a3_med = summary["A3"]["eval_kge_median"]
    a0_med = summary["A0"]["eval_kge_median"]
    # hard vs smooth
    a2_finite = summary["A2"]["finite_all"]; a3_finite = summary["A3"]["finite_all"]
    a2_spread = summary["A2"]["seed_spread_eval"]; a3_spread = summary["A3"]["seed_spread_eval"]
    a2_ub = s_eff_ub["A2"]; a3_ub = s_eff_ub["A3"]
    if (a2_finite and a3_finite and abs(a2_med - a3_med) <= 0.05
            and not (a3_med > a2_med + 0.05 and a3_spread < a2_spread)
            and a2_ub <= 0.75):
        limiter_verdict = "GO-HARD"
    elif (a3_finite and a3_med > a2_med + 0.05 and a3_spread < a2_spread + 1e-6
          and pet_exceed["A3"] < 1e-3):
        limiter_verdict = "GO-SMOOTH"
    else:
        limiter_verdict = "GO-HARD"
    # full gate
    gate_ok = all(summary[a]["finite_all"] for a in ARMS)
    pet_ok = (pet_exceed["A2"] < 1e-3 and pet_exceed["A3"] < 1e-3)
    closure_ok = all(bool(r["I_plus_ET1_plus_ET2_le_PET"]) for r in pet_rows if r["arm"] in ("A2", "A3"))
    no_collapse = (a1_med >= a0_med - 0.3) and (a2_med >= a0_med - 0.3) and (a3_med >= a0_med - 0.3)
    no_ub_lock = s_eff_ub["A1"] < 1.0 and s_eff_ub["A2"] < 1.0 and s_eff_ub["A3"] < 1.0
    if gate_ok and pet_ok and closure_ok and no_collapse and no_ub_lock:
        verdict = limiter_verdict
    else:
        verdict = "STOP-BEFORE-531"
    decision_rows = [
        {"component": "training_finite_all_arms", "value": bool(gate_ok), "pass": bool(gate_ok)},
        {"component": "PET_budget_closure_A2", "value": pet_exceed["A2"], "pass": pet_exceed["A2"] < 1e-3},
        {"component": "PET_budget_closure_A3", "value": pet_exceed["A3"], "pass": pet_exceed["A3"] < 1e-3},
        {"component": "water_balance_closure", "value": bool(closure_ok), "pass": bool(closure_ok)},
        {"component": "eval_no_collapse_vs_A0", "value": bool(no_collapse), "pass": bool(no_collapse)},
        {"component": "S_eff_not_universally_boundary", "value": bool(no_ub_lock), "pass": bool(no_ub_lock)},
        {"component": "hard_vs_smooth_verdict", "value": limiter_verdict, "pass": True},
        {"component": "SHARED_DPL_PILOT_VERDICT", "value": verdict, "pass": verdict != "STOP-BEFORE-531"},
    ]
    write_csv("shared_dpl_decision_matrix.csv", decision_rows)
    write_csv("hard_vs_smooth_comparison.csv", [
        {"arm": "A2", "eval_median": a2_med, "seed_spread": a2_spread,
         "s_eff_upper_bound_hit_rate": a2_ub, "pet_exceed_fraction": pet_exceed["A2"]},
        {"arm": "A3", "eval_median": a3_med, "seed_spread": a3_spread,
         "s_eff_upper_bound_hit_rate": a3_ub, "pet_exceed_fraction": pet_exceed["A3"]},
    ])
    audit_summary = {
        "arms": ARMS, "summary": summary, "s_eff_upper_bound_hit_rate": s_eff_ub,
        "pet_exceedance_day_fraction": pet_exceed,
        "hard_vs_smooth": {"verdict": limiter_verdict,
                           "a2_eval_median": a2_med, "a3_eval_median": a3_med,
                           "a2_spread": a2_spread, "a3_spread": a3_spread},
        "gate": verdict,
        "new_forcing": False, "new_learnable_parameter": False, "c_fixed_at_1": True,
        "production_default_changed": False, "531_training_started": False,
    }
    (OUT / "audit_summary.json").write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")

    report = f"""# MOPEX4 T1a SHARED-dPL PILOT

Basins: {', '.join(BASIN_IDS)}; Seeds: {SEEDS}; Protocol source: project/benchmark canonical dPL skeleton
(shared CatchmentParameterizer, Adam lr={LR}, epochs={EPOCHS}, full-batch KGE loss).
Train window 1985-10-01..1987-09-30 (365 warmup + 365 scored); eval window 1988-10-01..1990-09-30.

## Eval KGE (median over basin-seed)

- A0 F0: `{a0_med:.6f}`
- A1 T1a-E0: `{a1_med:.6f}`
- A2 T1a-E1-hard: `{a2_med:.6f}`
- A3 T1a-E1-smooth: `{a3_med:.6f}`

## Seed stability (std of eval KGE over basin-seed)

- A0: `{summary['A0']['seed_spread_eval']:.6f}`
- A1: `{summary['A1']['seed_spread_eval']:.6f}`
- A2: `{summary['A2']['seed_spread_eval']:.6f}`
- A3: `{summary['A3']['seed_spread_eval']:.6f}`

## PET exceedance-day fraction (scored, max over basins)

- A1: `{pet_exceed['A1']:.6f}`
- A2: `{pet_exceed['A2']:.6f}`
- A3: `{pet_exceed['A3']:.6f}`

## S_eff upper/lower bound hit rate

- A1: `{s_eff_ub['A1']:.3f}`, A2: `{s_eff_ub['A2']:.3f}`, A3: `{s_eff_ub['A3']:.3f}`

## Hard vs smooth limiter

- A2 vs A3 eval median: `{a2_med:.6f}` vs `{a3_med:.6f}`; seed spread `{a2_spread:.6f}` vs `{a3_spread:.6f}`
- shared-trunk initial gradient cosine A2/A3: see hard_vs_smooth_gradient_cosine.csv
- verdict: **{limiter_verdict}**

## Gate

**{verdict}**

No new forcing, no new learnable parameter, c fixed at 1, production default unchanged.
"""
    (OUT / "final_shared_dpl_pilot_report.md").write_text(report, encoding="utf-8")

    # ---- terminal summary ----
    print("MOPEX4 T1a SHARED-dPL PILOT")
    print(f"Basins: {', '.join(BASIN_IDS)}")
    print(f"Seeds: {SEEDS}")
    print("Training protocol source: project/benchmark canonical shared-dPL skeleton (CatchmentParameterizer + Adam + KGE)")
    print("New forcing introduced: NO")
    print("New learnable parameter introduced: NO")
    print("c fixed at 1: YES")
    print("Production default changed: NO")
    print("Arms: A0 F0; A1 T1a-E0; A2 T1a-E1-hard; A3 T1a-E1-smooth")
    print("Initial gradient audit: A1 trunk_norm %.4f head_norm %.4f; A2 %.4f/%.4f; A3 %.4f/%.4f" % (
        audit_rows[0]["trunk_grad_norm"], audit_rows[0]["head_grad_norm"],
        audit_rows[1]["trunk_grad_norm"], audit_rows[1]["head_grad_norm"],
        audit_rows[2]["trunk_grad_norm"], audit_rows[2]["head_grad_norm"]))
    print(f"A2 vs A3 gradient comparison: shared-trunk cosine {cos_shared:.4f}")
    print(f"Median KGE: A0 {a0_med:.6f}; A1 {a1_med:.6f}; A2 {a2_med:.6f}; A3 {a3_med:.6f}")
    print(f"Seed stability: A0 {summary['A0']['seed_spread_eval']:.6f}; A1 {summary['A1']['seed_spread_eval']:.6f}; A2 {summary['A2']['seed_spread_eval']:.6f}; A3 {summary['A3']['seed_spread_eval']:.6f}")
    print(f"PET exceedance-day fraction: A1 {pet_exceed['A1']:.6f}; A2 {pet_exceed['A2']:.6f}; A3 {pet_exceed['A3']:.6f}")
    print(f"S_eff upper-bound hit rate: A1 {s_eff_ub['A1']:.3f}; A2 {s_eff_ub['A2']:.3f}; A3 {s_eff_ub['A3']:.3f}")
    print(f"Hard vs smooth limiter: verdict {limiter_verdict}")
    print("Warm-start diagnostic used: NO")
    print(f"SHARED-dPL PILOT VERDICT: {verdict}")
    print(f"Ready for 531-basin shared-dPL training: {'YES' if verdict != 'STOP-BEFORE-531' else 'NO'}")
    print("531-basin training started: NO")
    print("Next action: " + ("proceed to 531-basin shared-dPL with the chosen limiter" if verdict != "STOP-BEFORE-531"
                             else "do not start 531 basins; investigate network-level incompatibility"))


if __name__ == "__main__":
    run_pilot()
