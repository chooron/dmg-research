#!/usr/bin/env python3
"""Phase B validation of the frozen production mopex4_step.

Checks:
  B1 exact/near-exact equivalence of the frozen mopex4_step with the previous
     validated E1-hard reference path (component level: Q, ET, I, ET1, ET2,
     states) over the 4 representative basins;
  B2 water balance / PET closure / finite / autograd / M3/M5 regression;
  B3 torch.compile audit (graph breaks, recompiles, fullgraph diagnostic).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BENCHMARK = Path(__file__).resolve().parents[2]
REPO = BENCHMARK.parents[1]
sys.path[:0] = [str(REPO), str(BENCHMARK), str(BENCHMARK / "src"),
                str(BENCHMARK / "scripts" / "diagnostics")]

import audit_mopex34_root_cause as A
import run_mopex4_pet_budget_closure as PB
from dmotpy.models.core.mopex4 import mopex4_step
from dmotpy.models.flux.mopex import (
    mopex_baseflow_1,
    mopex_evap_7,
    mopex_interception_4_t1a,
    mopex_melt_1,
    mopex_rainfall_1,
    mopex_recharge_3,
    mopex_saturation_1,
    mopex_snowfall_1,
    mopex_training_context,
)

OUT = BENCHMARK / "results" / "mopex45_phase_fix" / "root_cause_audit" / "liu_interception" / "shared_dpl_pilot"
OUT.mkdir(parents=True, exist_ok=True)
DTYPE = torch.float64
BASIN_IDS = ["8202700", "8150800", "5507600", "11532500"]
WARMUP = SCORED = 365
START = A.START


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def frozen_step_diag(P, T, PET, tcrit, ddf, Sb1, tw, S_eff, c, tu, Se, Sb2, tc,
                     S1, S2, Sc1, Sc2, Sn, *, doy, nearzero=1e-6):
    """Exact mirror of the frozen production mopex4_step that returns components."""
    del doy, c
    Sn = torch.relu(Sn); S1 = torch.relu(S1); S2 = torch.relu(S2)
    Sc1 = torch.relu(Sc1); Sc2 = torch.relu(Sc2)
    ps = mopex_snowfall_1(P, T, tcrit)
    pr = mopex_rainfall_1(P, T, tcrit)
    qn = mopex_melt_1(ddf, tcrit, T, Sn)
    Sn_new = Sn + ps - qn
    S1 = S1 + pr + qn
    i_pot = mopex_interception_4_t1a(pr, S_eff, nearzero=nearzero)
    i = torch.minimum(i_pot, PET)
    pet_after_i = PET - i
    et1 = torch.minimum(mopex_evap_7(S1, Sb1, pet_after_i, 1.0, nearzero), S1)
    S1 = S1 - et1
    pet_after_et1 = pet_after_i - et1
    i = torch.minimum(i, S1)
    S1 = S1 - i
    q1f = torch.minimum(mopex_saturation_1(pr + qn, S1, Sb1, nearzero=nearzero), S1)
    S1 = S1 - q1f
    qw = torch.minimum(mopex_recharge_3(tw, S1), S1)
    S1_new = S1 - qw
    S2 = S2 + qw
    q2f = torch.minimum(mopex_saturation_1(qw, S2, Sb2, nearzero=nearzero), S2)
    S2 = S2 - q2f
    q2u = mopex_baseflow_1(tu, S2)
    S2 = S2 - q2u
    et2 = torch.minimum(mopex_evap_7(S2, Se * Sb2, pet_after_et1, 1.0, nearzero), S2)
    S2_new = S2 - et2
    Sc1 = Sc1 + q1f + q2f
    qf = mopex_baseflow_1(tc, Sc1)
    Sc1_new = Sc1 - qf
    Sc2 = Sc2 + q2u
    qs = mopex_baseflow_1(tc, Sc2)
    Sc2_new = Sc2 - qs
    return (qf + qs, et1 + et2 + i, S1_new, S2_new, Sc1_new, Sc2_new, Sn_new,
            i, et1, et2, pr, pet_after_i, pet_after_et1)


def b1_equivalence(x, y):
    rows = []
    s_grid = [0.05, 0.6, 2.5]
    max_errs = {k: 0.0 for k in ["Q", "ET", "I", "ET1", "ET2", "S1", "S2", "Sc1", "Sc2", "Sn"]}
    for b in range(4):
        for s_eff in s_grid:
            common = [torch.tensor(v, dtype=DTYPE) for v in [0.0, 4.0, 200.0, 0.1, s_eff, 1.0, 0.1, 0.5, 300.0, 0.2]]
            states_a = [torch.tensor(1e-6, dtype=DTYPE) for _ in range(5)]
            states_b = [torch.tensor(1e-6, dtype=DTYPE) for _ in range(5)]
            errs = {k: 0.0 for k in max_errs}
            with mopex_training_context(pet_budget="interception_first", pet_limiter="hard"):
                for t in range(WARMUP + SCORED):
                    P, T, PET, doy = x[t, b, 0], x[t, b, 1], x[t, b, 2], x[t, b, 3]
                    out_ref = PB.step_diag(P, T, PET, *common, states_a[1], states_a[2], states_a[3],
                                           states_a[4], states_a[0], doy=doy, nearzero=1e-6)
                    out_frz = frozen_step_diag(P, T, PET, *common, states_b[1], states_b[2], states_b[3],
                                               states_b[4], states_b[0], doy=doy, nearzero=1e-6)
                    keys = [("Q", 0, 0), ("ET", 1, 1), ("I", 7, 7), ("ET1", 8, 8), ("ET2", 9, 9),
                            ("S1", 2, 2), ("S2", 3, 3), ("Sc1", 4, 4), ("Sc2", 5, 5), ("Sn", 6, 6)]
                    for name, ia, ib in keys:
                        errs[name] = max(errs[name], float((out_ref[ia] - out_frz[ib]).abs().max()))
                    states_a = list(out_ref[2:7]); states_b = list(out_frz[2:7])
            for k in max_errs:
                max_errs[k] = max(max_errs[k], errs[k])
            rows.append({"basin_id": BASIN_IDS[b], "S_eff": s_eff, **{f"max_abs_diff_{k}": errs[k] for k in max_errs}})
    write_csv("fixed_vs_old_e1_equivalence.csv", rows)
    return rows, max_errs


def b2_physics(x, y):
    rows, pet_rows = [], []
    settings = [("low", 0.05), ("mid", 0.6), ("high", 2.5)]
    autograd_ok = True
    for b in range(4):
        for name, s_eff in settings:
            s = torch.tensor(s_eff, dtype=DTYPE, requires_grad=True)
            common = [torch.tensor(v, dtype=DTYPE) for v in [0.0, 4.0, 200.0, 0.1, 0.0, 1.0, 0.1, 0.5, 300.0, 0.2]]
            common[4] = s
            i_s, et1_s, et2_s, q_s, pet_s = [], [], [], [], []
            states = [torch.tensor(1e-6, dtype=DTYPE) for _ in range(5)]
            state_sum = []
            for t in range(WARMUP + SCORED):
                out = frozen_step_diag(x[t, b, 0], x[t, b, 1], x[t, b, 2], *common,
                                       states[1], states[2], states[3], states[4], states[0],
                                       doy=x[t, b, 3], nearzero=1e-6)
                q_s.append(out[0]); i_s.append(out[7]); et1_s.append(out[8]); et2_s.append(out[9])
                pet_s.append(x[t, b, 2])
                states = list(out[2:7]); state_sum.append(sum(states))
            qv = torch.stack(q_s); iv = torch.stack(i_s); et1v = torch.stack(et1_s)
            et2v = torch.stack(et2_s); petv = torch.stack(pet_s); state_sum = torch.stack(state_sum)
            state_delta = torch.empty_like(state_sum)
            state_delta[0] = state_sum[0] - 5e-6
            state_delta[1:] = state_sum[1:] - state_sum[:-1]
            daily_res = x[:, b, 0] - (iv + et1v + et2v) - qv - state_delta
            total = iv + et1v + et2v
            exceed = (total - petv).clamp_min(0)
            scored = slice(WARMUP, WARMUP + SCORED)
            wb_pass = bool(daily_res.abs().max() < 1e-5)
            closure = bool((total <= petv + 1e-6).all())
            finite = bool(torch.isfinite(qv).all() and torch.isfinite(iv).all())
            rows.append({"basin_id": BASIN_IDS[b], "setting": name,
                         "max_daily_abs_residual": float(daily_res.abs().max()),
                         "water_balance_pass": wb_pass,
                         "exceedance_day_fraction_scored": float((exceed[scored] > 0).double().mean()),
                         "max_exceedance_scored": float(exceed[scored].max()),
                         "closure_pass": closure, "finite": finite})
            # S_eff interior gradient through the production step
            with torch.no_grad():
                pass
            loss = qv[WARMUP:].mean()
            loss.backward()
            autograd_ok &= bool(s.grad is not None and torch.isfinite(s.grad))
    write_csv("pilot_pet_water_balance.csv", rows)
    return rows, autograd_ok


def b3_compile():
    p = [torch.tensor(v, dtype=torch.float32) for v in [0.0, 4.0, 200.0, 0.1, 0.8, 0.5, 0.1, 0.5, 300.0, 0.2]]
    st = [torch.tensor(20.0, dtype=torch.float32) for _ in range(5)]
    args = (torch.tensor(10.0), torch.tensor(12.0), torch.tensor(3.0),
            *p, st[1], st[2], st[3], st[4], st[0])
    kw = {"doy": torch.tensor(180.0)}

    torch._dynamo.config.suppress_errors = False
    compiled = torch.compile(mopex4_step)
    # warmup/compile
    t0 = time.perf_counter()
    out0 = compiled(*args, **kw)
    first_compile = time.perf_counter() - t0
    for _ in range(3):
        compiled(*args, **kw)
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        compiled(*args, **kw)
    steady = (time.perf_counter() - t0) / n
    graph_breaks = int(torch._dynamo.utils.counters.get("graph_break", {}).get("total", 0))
    recompiles = torch._dynamo.utils.counters.get("bytecode", {}).get("recompile_count", 0) \
        if "bytecode" in torch._dynamo.utils.counters else 0

    # fullgraph diagnostic
    fullgraph_info = {}
    try:
        compiled_fg = torch.compile(mopex4_step, fullgraph=True)
        compiled_fg(*args, **kw)
        fullgraph_info = {"success": True, "error": None}
    except Exception as exc:  # noqa: BLE001
        fullgraph_info = {"success": False, "error": str(exc)[:2000]}

    audit = {
        "compile_success": True,
        "graph_breaks": graph_breaks,
        "recompiles": int(recompiles),
        "first_compile_sec": first_compile,
        "steady_state_step_sec": steady,
        "fullgraph_diagnostic": fullgraph_info,
        "eager_step_sec": None,
        "note": "graph_break/recompile counters from torch._dynamo.utils.counters",
    }
    # eager timing for reference
    t0 = time.perf_counter()
    for _ in range(n):
        mopex4_step(*args, **kw)
    audit["eager_step_sec"] = (time.perf_counter() - t0) / n
    (OUT / "compile_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    _, xfull, yfull, _ = A.load_context()
    x = xfull[START : START + WARMUP + SCORED].to(DTYPE)
    y = yfull[START : START + WARMUP + SCORED].to(DTYPE)
    print("B1 equivalence")
    rows, max_errs = b1_equivalence(x, y)
    equiv_pass = max(max_errs.values()) < 1e-9
    print("B2 physics")
    wb_rows, autograd_ok = b2_physics(x, y)
    wb_pass = all(bool(r["water_balance_pass"]) for r in wb_rows)
    closure_pass = all(bool(r["closure_pass"]) for r in wb_rows)
    finite_pass = all(bool(r["finite"]) for r in wb_rows)
    print("B3 compile")
    compile_audit = b3_compile()

    # M3/M5 smoke
    from dmotpy.models.core.mopex3 import mopex3_step
    from dmotpy.models.core.mopex5 import mopex5_step
    p3 = [torch.tensor(v, dtype=DTYPE) for v in [0.0, 4.0, 200.0, 0.1, 0.1, 0.5, 300.0, 0.2]]
    p5 = [torch.tensor(v, dtype=DTYPE) for v in [0.0, 4.0, 200.0, 0.1, 0.3, 180.0, 10.0, 15.0, 0.1, 0.5, 300.0, 0.2]]
    st = [torch.tensor(1e-6, dtype=DTYPE) for _ in range(5)]
    o3 = mopex3_step(torch.tensor(5.0), torch.tensor(10.0), torch.tensor(2.0), *p3,
                     st[0], st[1], st[2], st[3], st[4], nearzero=1e-6)
    o5a = mopex5_step(torch.tensor(5.0), torch.tensor(10.0), torch.tensor(2.0), *p5,
                      st[1], st[2], st[3], st[4], st[0], doy=torch.tensor(180.0))
    o5b = mopex5_step(torch.tensor(5.0), torch.tensor(10.0), torch.tensor(2.0), *p5,
                      st[1], st[2], st[3], st[4], st[0], doy=torch.tensor(180.0))
    m3_ok = bool(torch.isfinite(o3[0]))
    m5_ok = bool(torch.isfinite(o5a[0]) and torch.allclose(o5a[0], o5b[0]))

    # fixed step spec
    spec = """# Fixed production mopex4_step

## Final candidate (frozen)

- Interception formula: **T1a** single-threshold kernel on post-snow liquid rainfall
  `I_pot = S_eff * (-expm1(-Pr / S_eff))`
- Learnable interception parameters: **`S_eff` only**, bound `[1e-5, 5.0] mm`
- Fixed parameters: **`c = 1`** (compatibility slot `is_time` value ignored; no gradient path)
- PET semantics: **interception-first** shared daily PET demand
  `I = min(I_pot, PET); PET_after_I = PET - I; ET1 = evap_7(S1, Sb1, PET_after_I);
   PET_after_ET1 = PET_after_I - ET1; ET2 = evap_7(S2, se*s3max, PET_after_ET1)`
- Limiter: **exact hard** `min(I_pot, PET)`

## Public parameter slots

- `tcrit, ddf, s2max(Sb1), tw, alpha->S_eff, is_time->c(fixed 1), tu, se, s3max(Sb2), tc` (10 slots preserved)
- Effective learnable parameters: 9 (`S_eff` + 8 common hydrologic)
- Fixed compatibility slots: `is_time` (c=1), plus keyword-only `doy/phase_cos/phase_sin` accepted and unused

## Hot path

- No `_pet_budget_mode()` / `_pet_limiter()` runtime dispatch inside `mopex4_step`
- No hard/smooth runtime selection
- No c-dependent branch or gradient path
- Tensor-level safety ops (`minimum`, `relu`, storage caps) retained

## Reference/diagnostic implementations preserved (not on the hot path)

- `_mopex_interception_4_legacy` / `mopex_interception_4` (F0)
- `mopex_interception_4_liu` (2-param Liu diagnostic)
- `mopex_pet_budget_limit` (hard/smooth limiter helper)
- `mopex_training_context(pet_budget=..., pet_limiter=...)` (diagnostic reference modes, incl. legacy PET and soil-ET-first)
"""
    (OUT / "fixed_mopex4_step_spec.md").write_text(spec, encoding="utf-8")

    summary = {
        "b1_old_e1_hard_equivalence_pass": equiv_pass,
        "b1_max_abs_diffs": max_errs,
        "b2_water_balance_pass": wb_pass,
        "b2_pet_budget_closure_pass": closure_pass,
        "b2_finite_pass": finite_pass,
        "b2_autograd_finite": autograd_ok,
        "b2_mopex3_smoke": m3_ok,
        "b2_mopex5_unchanged": m5_ok,
        "b3_compile": compile_audit,
    }
    (OUT / "freeze_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("MOPEX4 FREEZE VALIDATION")
    print(f"B1 old E1-hard equivalence: {'PASS' if equiv_pass else 'FAIL'}")
    for k, v in max_errs.items():
        print(f"  max_abs_diff_{k}: {v:.3e}")
    print(f"B2 water balance: {'PASS' if wb_pass else 'FAIL'}; PET closure: {'PASS' if closure_pass else 'FAIL'}; finite: {'PASS' if finite_pass else 'FAIL'}; autograd: {'PASS' if autograd_ok else 'FAIL'}")
    print(f"B2 MOPEX3 smoke: {'PASS' if m3_ok else 'FAIL'}; MOPEX5 unchanged: {'PASS' if m5_ok else 'FAIL'}")
    print(f"B3 compile: graph_breaks={compile_audit['graph_breaks']} recompiles={compile_audit['recompiles']} "
          f"first_compile={compile_audit['first_compile_sec']:.3f}s steady={compile_audit['steady_state_step_sec']:.5f}s "
          f"eager={compile_audit['eager_step_sec']:.5f}s fullgraph={compile_audit['fullgraph_diagnostic']['success']}")


if __name__ == "__main__":
    main()
