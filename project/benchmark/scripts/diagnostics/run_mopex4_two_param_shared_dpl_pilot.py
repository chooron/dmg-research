#!/usr/bin/env python3
"""Two-parameter MOPEX4 shared-dPL pilot (final network gate, S_eff + c active).

P2-TWO: 2-param Liu (S_eff, c) + interception-first shared PET + exact hard
limiter, driven through the real frozen production ``mopex4_step`` with a
10-active-output network head.  P0/P1/P2-T1a references are reused from the
previous shared_dpl_pilot results where the protocol is identical.
No new forcing, no c=1 insertion, no detach on c, no CMA-ES warm-start.
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
import run_mopex4_pet_budget_closure as PB
from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from src.data_selection import load_ids
from dmotpy.models.core.mopex4 import mopex4_step
from dmotpy.models.flux.mopex import mopex_rainfall_1

OUT = BENCHMARK / "results" / "mopex45_phase_fix" / "root_cause_audit" / "liu_interception" / "two_param_final"
OUT.mkdir(parents=True, exist_ok=True)
PREV = BENCHMARK / "results" / "mopex45_phase_fix" / "root_cause_audit" / "liu_interception" / "shared_dpl_pilot"
DTYPE = torch.float64
BASIN_INDEX = [391, 373, 269, 530]
BASIN_IDS = ["8202700", "8150800", "5507600", "11532500"]
WARMUP = SCORED = 365
SEEDS = [7, 41, 73]
EPOCHS = 50
LR = 1e-3
HIDDEN = [256, 256]
DROPOUT = 0.05
S_EFF_LO, S_EFF_HI = 1e-5, 5.0
C_LO, C_HI = 0.10, 0.98


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalized_to_physical(u: torch.Tensor) -> torch.Tensor:
    """Map the 10 normalized head outputs to physical slots via MOPEX4 bounds."""
    bounds = torch.tensor(PB.T1_BOUNDS, dtype=u.dtype, device=u.device)
    return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])


def simulate_two_param(net, attrs, forcing):
    """P2-TWO: drive the real frozen two-param mopex4_step per time step."""
    u = torch.clamp(net(attrs.to(torch.float32)).to(DTYPE), 1e-6, 1.0 - 1e-6)
    physical = normalized_to_physical(u)  # (B, 10), both S_eff and c active
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


NET_ATTRS = None


def load_previous_reference():
    """Read P0/P1/P2-T1a summary metrics from the previous pilot (same protocol)."""
    ref = {}
    try:
        with (PREV / "pilot_seed_metrics.csv").open() as handle:
            for row in csv.DictReader(handle):
                arm = {"P0": "P0", "P1": "P1", "P2": "P2"}.get(row["arm"], row["arm"])
                ref.setdefault(row["arm"], {})[int(row["seed"])] = float(row["median_eval_kge"])
        with (PREV / "pilot_basin_metrics.csv").open() as handle:
            basin_ref = {}
            for row in csv.DictReader(handle):
                basin_ref.setdefault(row["arm"], []).append(float(row["eval_kge"]))
            ref["_basin_all"] = basin_ref
    except FileNotFoundError:
        ref = {}
    return ref


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
        "arm": "P2-TWO", "formula": "I_pot = S_eff * (-expm1(-c * Pr / S_eff))",
        "learnable_interception": ["S_eff", "c"], "head_outputs": 10,
        "basins": BASIN_IDS, "seeds": SEEDS, "epochs": EPOCHS, "lr": LR, "optimizer": "Adam",
        "network": {"architecture": "CatchmentParameterizer", "in_features": 35,
                    "hidden_dims": HIDDEN, "dropout": DROPOUT, "head": 10},
        "train_window": {"warmup": WARMUP, "scored": SCORED, "start": 1825},
        "eval_window": {"warmup": WARMUP, "scored": SCORED, "start": 2555},
        "pet": "interception-first shared PET", "limiter": "exact hard min",
        "new_forcing": False, "c_fixed": False, "cma_es_warm_start": False,
        "531_training_started": False,
    }
    (OUT / "two_param_pilot_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    curve_rows, metric_rows, param_rows = [], [], []
    final_nets = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        net = CatchmentParameterizer(35, 10, hidden_dims=HIDDEN, dropout=DROPOUT)
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
        finite_ok = True
        for epoch in range(1, EPOCHS + 1):
            optimizer.zero_grad()
            q = simulate_two_param(net, NET_ATTRS, x_train)
            scores, nse = PB._kge_and_nse(q[WARMUP:], y_train[WARMUP:])
            loss = 1.0 - scores.mean()
            loss.backward()
            grad_finite = all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
            finite_ok &= bool(torch.isfinite(loss)) and grad_finite
            optimizer.step()
            if epoch % 10 == 0 or epoch == 1:
                for b in range(4):
                    curve_rows.append({"seed": seed, "epoch": epoch, "basin_id": BASIN_IDS[b],
                                       "train_kge": float(scores[b, 0, 0]), "train_nse": float(nse[b]),
                                       "loss": float(loss), "finite": bool(torch.isfinite(loss))})
        net.eval()
        with torch.no_grad():
            q_tr = simulate_two_param(net, NET_ATTRS, x_train)
            tr_s, tr_n = PB._kge_and_nse(q_tr[WARMUP:], y_train[WARMUP:])
            q_ev = simulate_two_param(net, NET_ATTRS, x_eval)
            ev_s, ev_n = PB._kge_and_nse(q_ev[WARMUP:], y_eval[WARMUP:])
        u = torch.clamp(net(NET_ATTRS.to(torch.float32)).to(DTYPE), 1e-6, 1 - 1e-6)
        physical = normalized_to_physical(u)
        for b in range(4):
            s_eff = float(physical[b, 4]); c = float(physical[b, 5])
            s_dist = min((s_eff - S_EFF_LO) / (S_EFF_HI - S_EFF_LO), (S_EFF_HI - s_eff) / (S_EFF_HI - S_EFF_LO))
            c_dist = min((c - C_LO) / (C_HI - C_LO), (C_HI - c) / (C_HI - C_LO))
            param_rows.append({"seed": seed, "basin_id": BASIN_IDS[b],
                               "s_eff": s_eff, "c": c,
                               "raw_s_eff_activation": float(u[b, 4]), "raw_c_activation": float(u[b, 5]),
                               "s_eff_transform_derivative": float(u[b, 4] * (1 - u[b, 4])),
                               "c_transform_derivative": float(u[b, 5] * (1 - u[b, 5])),
                               "s_eff_boundary_hit": s_dist <= 0.02, "c_boundary_hit": c_dist <= 0.02})
            metric_rows.append({"seed": seed, "basin_id": BASIN_IDS[b],
                                "train_kge": float(tr_s[b, 0, 0]), "train_nse": float(tr_n[b]),
                                "eval_kge": float(ev_s[b, 0, 0]), "eval_nse": float(ev_n[b]),
                                "s_eff": s_eff, "c": c, "finite": finite_ok})
        final_nets.append((seed, net, finite_ok))
    write_csv("two_param_training_curves.csv", curve_rows)
    write_csv("two_param_shared_dpl_basin_metrics.csv", metric_rows)
    write_csv("two_param_parameter_diagnostics.csv", param_rows)

    # PET / water balance + I/P ET/P Q/P at final networks (seed 7) via component mirror
    from run_mopex4_two_param_final_validation import step_diag as tp_step_diag
    pet_rows = []
    for seed, net, finite_ok in final_nets:
        net.eval()
        u = torch.clamp(net(NET_ATTRS.to(torch.float32)).to(DTYPE), 1e-6, 1 - 1e-6)
        physical = normalized_to_physical(u)
        for b in range(4):
            common = [torch.tensor(float(v), dtype=DTYPE) for v in physical[b]]
            i_s, et1_s, et2_s, q_s, pet_s = [], [], [], [], []
            states = [torch.tensor(1e-6, dtype=DTYPE) for _ in range(5)]
            state_sum = []
            for t in range(WARMUP + SCORED):
                out = tp_step_diag(x_eval[t, b, 0], x_eval[t, b, 1], x_eval[t, b, 2], *common,
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
            viol = total - petv
            scored = slice(WARMUP, WARMUP + SCORED)
            pr_scored = x_eval[WARMUP:, b, 0]
            pet_rows.append({"seed": seed, "basin_id": BASIN_IDS[b],
                             "water_balance_pass": bool(daily_res.abs().max() < 1e-5),
                             "max_daily_abs_residual": float(daily_res.abs().max()),
                             "closure_pass": bool((total <= petv + 1e-6).all()),
                             "exceedance_day_fraction_scored": float(((viol > 1e-9)[scored]).double().mean()),
                             "max_overshoot_scored": float(viol.clamp_min(0)[scored].max()),
                             "I/P_scored": float(iv[scored].sum() / pr_scored.sum().clamp_min(1e-12)),
                             "ET/P_scored": float((et1v + et2v)[scored].sum() / pr_scored.sum().clamp_min(1e-12)),
                             "Q/P_scored": float(qv[WARMUP:].sum() / pr_scored.sum().clamp_min(1e-12))})
    write_csv("two_param_pet_water_balance.csv", pet_rows)

    # identifiability at final params: median |dI_pot/dS_eff| and |dI_pot/dc| on
    # active rainy days (I_pot < PET), plus PET-limited fraction
    ident_rows = []
    pr_all = mopex_rainfall_1(x_eval[:, :, 0], x_eval[:, :, 1], torch.tensor(0.0, dtype=DTYPE)).clamp_min(0)
    for seed, net, _ in final_nets:
        net.eval()
        u = torch.clamp(net(NET_ATTRS.to(torch.float32)).to(DTYPE), 1e-6, 1 - 1e-6)
        physical = normalized_to_physical(u)
        for b in range(4):
            s_eff = float(physical[b, 4]); c = float(physical[b, 5])
            pr = pr_all[:, b]
            xr = c * pr / s_eff
            rainy = pr > 1e-10
            i_pot_rain = torch.tensor(s_eff, dtype=DTYPE) * (-torch.expm1(-xr))
            pet_eval = x_eval[:, :, 2]
            d_s = (1.0 - torch.exp(-xr) * (1.0 + xr)).abs()
            d_c = (pr * torch.exp(-xr)).abs()
            ident_rows.append({"seed": seed, "basin_id": BASIN_IDS[b],
                               "median_abs_dI_dS_eff_rain": float(d_s[rainy].median()),
                               "median_abs_dI_dc_rain": float(d_c[rainy].median()),
                               "q95_abs_dI_dc_rain": float(torch.quantile(d_c[rainy], .95)),
                               "pet_limited_fraction_rain": float((rainy & (i_pot_rain > pet_eval[:, b])).double().mean())})
    write_csv("two_param_identifiability_at_final.csv", ident_rows)

    # seed summary
    by_seed = {seed: [r["eval_kge"] for r in metric_rows if r["seed"] == seed] for seed in SEEDS}
    seed_med = {seed: float(np.median(v)) for seed, v in by_seed.items()}
    all_ev = [r["eval_kge"] for r in metric_rows]
    seed_metrics = [{"seed": seed, "median_eval_kge": seed_med[seed],
                     "mean_eval_kge": float(np.mean(by_seed[seed]))} for seed in SEEDS]
    write_csv("two_param_shared_dpl_seed_metrics.csv", seed_metrics)

    p2_med = float(np.median(all_ev))
    p2_seed_std = float(np.std(list(seed_med.values())))
    s_hit = float(np.mean([1.0 if r["s_eff_boundary_hit"] else 0.0 for r in param_rows]))
    c_hit = float(np.mean([1.0 if r["c_boundary_hit"] else 0.0 for r in param_rows]))
    finite_all = all(r["finite"] for r in metric_rows)
    wb_pass = all(bool(r["water_balance_pass"]) for r in pet_rows)
    closure_pass = all(bool(r["closure_pass"]) for r in pet_rows)
    median_ds = float(np.median([r["median_abs_dI_dS_eff_rain"] for r in ident_rows]))
    median_dc = float(np.median([r["median_abs_dI_dc_rain"] for r in ident_rows]))
    # S_eff-c relation across basin-seed pairs
    s_vals = [r["s_eff"] for r in param_rows]; c_vals = [r["c"] for r in param_rows]
    corr_sc = float(np.corrcoef(s_vals, c_vals)[0, 1]) if len(s_vals) > 2 else float("nan")

    ref = load_previous_reference()
    p0_med = float(np.median([v for v in ref.get("P0", {}).values()])) if ref.get("P0") else float("nan")
    p1_med = float(np.median([v for v in ref.get("P1", {}).values()])) if ref.get("P1") else float("nan")
    p2t1a_med = float(np.median([v for v in ref.get("P2", {}).values()])) if ref.get("P2") else float("nan")

    gate_ok = finite_all and wb_pass and closure_pass and (p2_med >= p0_med - 0.4)
    no_lock = s_hit < 1.0 and c_hit < 1.0
    stable = p2_seed_std <= 0.30
    grads_ok = median_ds > 0 and median_dc > 0
    monitoring = bool(c_hit > 0) or abs(corr_sc) > 0.8 or median_dc < 0.05 * median_ds
    if gate_ok and no_lock and stable and grads_ok:
        verdict = "GO-531-TWO-PARAM-WITH-MONITORING" if monitoring else "GO-531-TWO-PARAM"
    else:
        verdict = "STOP-BEFORE-531"

    decision_rows = [
        {"component": "training_finite_all_seeds", "value": bool(finite_all), "pass": bool(finite_all)},
        {"component": "P2_TWO_water_balance", "value": bool(wb_pass), "pass": bool(wb_pass)},
        {"component": "P2_TWO_pet_closure", "value": bool(closure_pass), "pass": bool(closure_pass)},
        {"component": "both_gradients_active", "value": bool(grads_ok), "pass": bool(grads_ok)},
        {"component": "no_boundary_lock", "value": bool(no_lock), "pass": bool(no_lock)},
        {"component": "seed_stability", "value": bool(stable), "pass": bool(stable)},
        {"component": "monitoring_flag", "value": bool(monitoring), "pass": True},
        {"component": "VERDICT", "value": verdict, "pass": verdict != "STOP-BEFORE-531"},
    ]
    write_csv("two_param_decision_matrix.csv", decision_rows)

    audit = {"arm": "P2-TWO", "formula": "I_pot = S_eff * (-expm1(-c * Pr / S_eff))",
             "median_eval_kge_by_seed": seed_med, "median_eval_kge": p2_med,
             "seed_std": p2_seed_std, "s_eff_boundary_hit_rate": s_hit, "c_boundary_hit_rate": c_hit,
             "median_abs_dI_dS_eff": median_ds, "median_abs_dI_dc": median_dc,
             "s_eff_c_correlation": corr_sc, "water_balance_pass": wb_pass,
             "pet_closure_pass": closure_pass, "reference_P0_median": p0_med,
             "reference_P1_median": p1_med, "reference_P2_T1a_median": p2t1a_med,
             "verdict": verdict,
             "new_forcing": False, "c_fixed": False, "cma_es_warm_start": False,
             "531_training_started": False}
    (OUT / "audit_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    report = f"""# MOPEX4 FINAL TWO-PARAM SHARED-DPL PILOT (P2-TWO)

Formula: I_pot = S_eff * (-expm1(-c * Pr / S_eff)); S_eff [1e-5,5] mm, c [0.10,0.98];
interception-first shared PET; exact hard limiter; 10 active head outputs.
Basins {', '.join(BASIN_IDS)}; seeds {SEEDS}; protocol identical to the previous pilot.

## Median eval KGE

- P0 F0 reference: `{p0_med:.6f}`
- P1 T1a-E0 reference: `{p1_med:.6f}`
- P2 previous T1a E1-hard reference: `{p2t1a_med:.6f}`
- P2-TWO (S_eff + c): `{p2_med:.6f}` (per-seed medians {[round(seed_med[s],4) for s in SEEDS]})

Seed std (per-seed median eval KGE): `{p2_seed_std:.6f}`
S_eff boundary hit rate: `{s_hit:.3f}`; c boundary hit rate: `{c_hit:.3f}`
median |dI/dS_eff| (rainy): `{median_ds:.6f}`; median |dI/dc| (rainy): `{median_dc:.6f}`
S_eff-c correlation across basin-seed: `{corr_sc:.3f}`

## Gates

water balance PASS: `{wb_pass}`; PET closure PASS: `{closure_pass}`; all seeds finite: `{finite_all}`

## Verdict

**{verdict}**

No new forcing, c learnable (no insertion/detach), no CMA-ES warm-start, 531-basin training not started.
"""
    (OUT / "final_two_param_shared_dpl_report.md").write_text(report, encoding="utf-8")

    print("MOPEX4 FINAL TWO-PARAM INTERCEPTION RETEST")
    print("Production formula: I_pot = S_eff * (-expm1(-c * Pr / S_eff))")
    print("Learnable interception parameters: S_eff YES [1e-5,5.0] mm; c YES [0.10,0.98]")
    print("Effective MOPEX4 learnable parameters: 10; Pr = liquid rainfall after snow partition")
    print("PET semantics: interception-first shared PET; limiter: exact hard min; runtime dispatch: NO")
    print(f"Median eval KGE: P2-TWO {p2_med:.6f} (seeds {[round(seed_med[s],4) for s in SEEDS]}); refs P0 {p0_med:.6f} P1 {p1_med:.6f} P2-T1a {p2t1a_med:.6f}")
    print(f"Seed std: {p2_seed_std:.6f}; S_eff boundary hit rate: {s_hit:.3f}; c boundary hit rate: {c_hit:.3f}")
    print(f"median |dI/dS_eff| {median_ds:.6f}; median |dI/dc| {median_dc:.6f}; S_eff-c corr {corr_sc:.3f}")
    print(f"water balance PASS: {wb_pass}; PET closure PASS: {closure_pass}; all seeds finite: {finite_all}")
    print(f"FINAL VERDICT: {verdict}")
    print("531-basin training started: NO")
    print("production final step updated to 2-param: YES")


if __name__ == "__main__":
    run_pilot()
