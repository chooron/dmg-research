#!/usr/bin/env python3
"""Final two-parameter MOPEX4 validation (Phase D/E of the two-param retest).

Formula boundary tests, analytic-vs-autograd gradients for S_eff and c,
water/PET audit, snow-partition regression, and the torch.compile audit for the
frozen two-parameter production ``mopex4_step`` (T1a-family Liu + S_eff + c +
interception-first shared PET + exact hard limiter).
"""
from __future__ import annotations

import csv
import json
import math
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
from dmotpy.models.core.mopex4 import mopex4_step
from dmotpy.models.flux.mopex import (
    mopex_baseflow_1,
    mopex_evap_7,
    mopex_interception_4_liu2,
    mopex_melt_1,
    mopex_rainfall_1,
    mopex_recharge_3,
    mopex_saturation_1,
    mopex_snowfall_1,
)

OUT = BENCHMARK / "results" / "mopex45_phase_fix" / "root_cause_audit" / "liu_interception" / "two_param_final"
OUT.mkdir(parents=True, exist_ok=True)
DTYPE = torch.float64
BASIN_IDS = ["8202700", "8150800", "5507600", "11532500"]
WARMUP = SCORED = 365
START = A.START
S_EFF_BOUNDS = (1e-5, 5.0)
C_BOUNDS = (0.10, 0.98)


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def step_diag(P, T, PET, tcrit, ddf, Sb1, tw, S_eff, c, tu, Se, Sb2, tc,
              S1, S2, Sc1, Sc2, Sn, *, doy, nearzero=1e-6):
    """Mirror of the frozen two-parameter production step returning components."""
    del doy
    Sn = torch.relu(Sn); S1 = torch.relu(S1); S2 = torch.relu(S2)
    Sc1 = torch.relu(Sc1); Sc2 = torch.relu(Sc2)
    ps = mopex_snowfall_1(P, T, tcrit)
    pr = mopex_rainfall_1(P, T, tcrit)
    qn = mopex_melt_1(ddf, tcrit, T, Sn)
    Sn_new = Sn + ps - qn
    # interception-first water path: I is removed from Pr before soil entry
    i_pot = mopex_interception_4_liu2(pr, S_eff, c, nearzero=nearzero)
    i = torch.minimum(i_pot, PET)
    pr_net = pr - i
    pet_after_i = PET - i
    soil_input = pr_net + qn
    S1 = S1 + soil_input
    et1 = torch.minimum(mopex_evap_7(S1, Sb1, pet_after_i, 1.0, nearzero), S1)
    S1 = S1 - et1
    pet_after_et1 = pet_after_i - et1
    q1f = torch.minimum(mopex_saturation_1(soil_input, S1, Sb1, nearzero=nearzero), S1)
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


def stage_boundary() -> bool:
    rows = []
    all_pass = True
    for s_eff in [1e-5, 0.5, 2.0, 5.0]:
        for c in [0.10, 0.5, 0.98]:
            for pr in [0.0, 1e-4, s_eff / c, s_eff / c * 3.0, 50.0]:
                for pet in [0.0, 0.01, 1.0, 20.0]:
                    i_pot = mopex_interception_4_liu2(torch.tensor(pr, dtype=DTYPE),
                                                      torch.tensor(s_eff, dtype=DTYPE),
                                                      torch.tensor(c, dtype=DTYPE))
                    i = torch.minimum(i_pot, torch.tensor(pet, dtype=DTYPE))
                    pet_after = torch.tensor(pet, dtype=DTYPE) - i
                    ok = (bool(torch.isfinite(i_pot))
                          and 0.0 <= float(i_pot) <= pr + 1e-9
                          and 0.0 <= float(i) <= min(float(i_pot), pet) + 1e-9
                          and float(pet_after) >= -1e-9)
                    all_pass &= ok
                    rows.append({"S_eff": s_eff, "c": c, "Pr": pr, "PET": pet,
                                 "I_pot": float(i_pot), "I": float(i),
                                 "PET_after_I": float(pet_after),
                                 "bounds_ok": ok})
    write_csv("two_param_formula_boundary_tests.csv", rows)
    return all_pass


def stage_gradients(x, y) -> bool:
    rows = []
    all_pass = True
    for s_eff in [1e-5, 0.5, 2.0, 5.0]:
        for c in [0.10, 0.5, 0.98]:
            for pr in [0.01, 0.5, 3.0, 20.0]:
                s = torch.tensor(s_eff, dtype=DTYPE, requires_grad=True)
                cv = torch.tensor(c, dtype=DTYPE, requires_grad=True)
                pv = torch.tensor(pr, dtype=DTYPE)
                pot = mopex_interception_4_liu2(pv, s, cv)
                gs, gc = torch.autograd.grad(pot, (s, cv))
                xr = c * pr / s_eff
                exp_x = math.exp(-xr)
                ana_s = 1.0 - exp_x * (1.0 + xr)
                ana_c = pr * exp_x
                err_s = abs(float(gs) - ana_s)
                err_c = abs(float(gc) - ana_c)
                ok = torch.isfinite(gs) and torch.isfinite(gc) and max(err_s, err_c) < 1e-9
                all_pass &= bool(ok)
                rows.append({"space": "physical", "S_eff": s_eff, "c": c, "Pr": pr,
                             "dI_dS_analytic": ana_s, "dI_dS_autograd": float(gs),
                             "dI_dc_analytic": ana_c, "dI_dc_autograd": float(gc),
                             "abs_err_S": err_s, "abs_err_c": err_c,
                             "sign_S": math.copysign(1.0, float(gs)), "sign_c": math.copysign(1.0, float(gc)),
                             "finite": bool(torch.isfinite(gs) and torch.isfinite(gc)), "pass": ok})
    # step-level autograd on actual rainfall: zero-gradient fraction over days
    pr_all = mopex_rainfall_1(x[:, :, 0], x[:, :, 1], torch.tensor(0.0, dtype=DTYPE)).clamp_min(0)
    for b in range(4):
        s = torch.tensor(0.8, dtype=DTYPE, requires_grad=True)
        cv = torch.tensor(0.5, dtype=DTYPE, requires_grad=True)
        common = [torch.tensor(v, dtype=DTYPE) for v in [0.0, 4.0, 200.0, 0.1, 0.0, 0.0, 0.1, 0.5, 300.0, 0.2]]
        common[4] = s; common[5] = cv
        states = [torch.tensor(1e-6, dtype=DTYPE) for _ in range(5)]
        qs = []
        for t in range(WARMUP + SCORED):
            out = step_diag(x[t, b, 0], x[t, b, 1], x[t, b, 2], *common,
                            states[1], states[2], states[3], states[4], states[0], doy=x[t, b, 3])
            qs.append(out[0]); states = list(out[2:7])
        q = torch.stack(qs)
        loss = q[WARMUP:].mean()
        loss.backward()
        rows.append({"space": "step_autograd_actual_rain", "basin_id": BASIN_IDS[b],
                     "dLoss_dS_eff": float(s.grad), "dLoss_dc": float(cv.grad),
                     "S_eff_grad_finite": bool(torch.isfinite(s.grad)),
                     "c_grad_finite": bool(torch.isfinite(cv.grad)),
                     "S_eff_grad_zero": bool(float(s.grad) == 0.0),
                     "c_grad_zero": bool(float(cv.grad) == 0.0),
                     "rainy_days": int((pr_all[:, b] > 0).sum())})
        all_pass &= bool(torch.isfinite(s.grad) and torch.isfinite(cv.grad))
    write_csv("two_param_gradient_audit.csv", rows)
    return all_pass


def stage_water_pet(x, y):
    rows = []
    settings = [("low", 0.05, 0.30), ("mid", 0.8, 0.6), ("high", 3.0, 0.9)]
    for b in range(4):
        for name, s_eff, c in settings:
            common = [torch.tensor(v, dtype=DTYPE) for v in [0.0, 4.0, 200.0, 0.1, s_eff, c, 0.1, 0.5, 300.0, 0.2]]
            i_s, et1_s, et2_s, q_s, pet_s = [], [], [], [], []
            states = [torch.tensor(1e-6, dtype=DTYPE) for _ in range(5)]
            state_sum = []
            for t in range(WARMUP + SCORED):
                out = step_diag(x[t, b, 0], x[t, b, 1], x[t, b, 2], *common,
                                states[1], states[2], states[3], states[4], states[0], doy=x[t, b, 3])
                q_s.append(out[0]); i_s.append(out[7]); et1_s.append(out[8]); et2_s.append(out[9])
                pet_s.append(x[t, b, 2]); states = list(out[2:7]); state_sum.append(sum(states))
            qv = torch.stack(q_s); iv = torch.stack(i_s); et1v = torch.stack(et1_s)
            et2v = torch.stack(et2_s); petv = torch.stack(pet_s); state_sum = torch.stack(state_sum)
            state_delta = torch.empty_like(state_sum)
            state_delta[0] = state_sum[0] - 5e-6
            state_delta[1:] = state_sum[1:] - state_sum[:-1]
            daily_res = x[:, b, 0] - (iv + et1v + et2v) - qv - state_delta
            total = iv + et1v + et2v
            viol = total - petv
            scored = slice(WARMUP, WARMUP + SCORED)
            rows.append({"basin_id": BASIN_IDS[b], "setting": name, "S_eff": s_eff, "c": c,
                         "water_balance_pass": bool(daily_res.abs().max() < 1e-5),
                         "max_daily_abs_residual": float(daily_res.abs().max()),
                         "closure_pass": bool((total <= petv + 1e-6).all()),
                         "exceedance_day_fraction_scored": float(((viol > 1e-9)[scored]).double().mean()),
                         "max_overshoot_scored": float(viol.clamp_min(0)[scored].max()),
                         "I/P_scored": float(iv[scored].sum() / x[WARMUP:, b, 0].sum().clamp_min(1e-12)),
                         "ET/P_scored": float((et1v + et2v)[scored].sum() / x[WARMUP:, b, 0].sum().clamp_min(1e-12)),
                         "Q/P_scored": float(qv[WARMUP:].sum() / x[WARMUP:, b, 0].sum().clamp_min(1e-12))})
    write_csv("two_param_water_pet_audit.csv", rows)
    return rows


def stage_snow_partition():
    """Snowfall must not be intercepted: T below tcrit => Pr ~ 0 => I ~ 0."""
    rows = []
    ok = True
    for T in [-5.0, -1.0, 1.0, 10.0]:
        P = torch.tensor(20.0, dtype=DTYPE)
        Tt = torch.tensor(T, dtype=DTYPE)
        tcrit = torch.tensor(0.0, dtype=DTYPE)
        pr = mopex_rainfall_1(P, Tt, tcrit)
        ps = mopex_snowfall_1(P, Tt, tcrit)
        s = torch.tensor(1.0, dtype=DTYPE)
        c = torch.tensor(0.5, dtype=DTYPE)
        i_pot = mopex_interception_4_liu2(pr, s, c)
        rows.append({"T": T, "P": 20.0, "Pr": float(pr), "Ps": float(ps),
                     "I_pot": float(i_pot), "pr_plus_ps_equals_P": bool(torch.isclose(pr + ps, P)),
                     "cold_no_interception": bool(float(i_pot) < 1e-6 if T <= 0 else True)})
        ok &= bool(torch.isclose(pr + ps, P))
        if T <= 0.0:
            ok &= bool(float(i_pot) < 1e-6)
    write_csv("two_param_snow_partition_regression.csv", rows)
    return ok


def stage_compile():
    p = [torch.tensor(v, dtype=torch.float32) for v in [0.0, 4.0, 200.0, 0.1, 0.8, 0.5, 0.1, 0.5, 300.0, 0.2]]
    st = [torch.tensor(20.0, dtype=torch.float32) for _ in range(5)]
    args = (torch.tensor(10.0), torch.tensor(12.0), torch.tensor(3.0),
            *p, st[1], st[2], st[3], st[4], st[0])
    kw = {"doy": torch.tensor(180.0)}
    compiled = torch.compile(mopex4_step)
    t0 = time.perf_counter()
    compiled(*args, **kw)
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
    fullgraph_info = {}
    try:
        compiled_fg = torch.compile(mopex4_step, fullgraph=True)
        compiled_fg(*args, **kw)
        fullgraph_info = {"success": True, "error": None}
    except Exception as exc:  # noqa: BLE001
        fullgraph_info = {"success": False, "error": str(exc)[:2000]}
    t0 = time.perf_counter()
    for _ in range(n):
        mopex4_step(*args, **kw)
    eager = (time.perf_counter() - t0) / n
    audit = {"graph_breaks": graph_breaks, "recompiles": int(recompiles),
             "first_compile_sec": first_compile, "steady_state_step_sec": steady,
             "eager_step_sec": eager, "fullgraph": fullgraph_info,
             "note": "torch._dynamo counters; fixed-shape repeated execution"}
    (OUT / "two_param_compile_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    _, xfull, yfull, _ = A.load_context()
    x = xfull[START:START + WARMUP + SCORED].to(DTYPE)
    y = yfull[START:START + WARMUP + SCORED].to(DTYPE)
    print("boundary")
    boundary_pass = stage_boundary()
    print("gradients")
    grad_pass = stage_gradients(x, y)
    print("water/pet")
    wb_rows = stage_water_pet(x, y)
    wb_pass = all(bool(r["water_balance_pass"]) for r in wb_rows)
    closure_pass = all(bool(r["closure_pass"]) for r in wb_rows)
    print("snow partition")
    snow_pass = stage_snow_partition()
    print("compile")
    compile_audit = stage_compile()

    spec = """# Final two-parameter MOPEX4 step

## Production formula

```
I_pot = S_eff * (-expm1(-c * Pr / S_eff))     Pr = liquid rainfall after snow partition
```

## Learnable interception parameters

- `S_eff`: effective daily interception threshold, bounds [1e-5, 5.0] mm
- `c`: effective canopy closure / wetting efficiency, bounds [0.10, 0.98]

Both carry active gradient paths.  Effective MOPEX4 learnable parameters: 10.

## PET semantics (unchanged validated design)

- interception-first shared daily PET demand
- `I = min(I_pot, PET)` (exact hard limiter)
- `PET_after_I = PET - I`; ET1 uses `PET_after_I`; `PET_after_ET1 = PET_after_I - ET1`; ET2 uses `PET_after_ET1`
- `I + ET1 + ET2 <= PET` by construction

## Hot path

- single fixed branch-free forward (T1a-family two-param Liu + interception-first + hard limiter)
- no runtime PET/limiter/T1a-vs-T1/seasonal-phase dispatch
- positional slots: `tcrit ddf s2max tw alpha(S_eff) is_time(c) tu se s3max tc` (10 preserved)
- `doy/phase_cos/phase_sin` accepted and unused

## Reference/diagnostic helpers preserved (not on hot path)

- `_mopex_interception_4_legacy` / `mopex_interception_4` (F0)
- `mopex_interception_4_t1a` (T1a c=1 reference)
- `mopex_interception_4_liu` (context-scaled 2-param diagnostic)
- `mopex_pet_budget_limit` (hard/smooth limiter helper)
"""
    (OUT / "two_param_final_step_spec.md").write_text(spec, encoding="utf-8")

    summary = {"formula": "I_pot = S_eff * (-expm1(-c * Pr / S_eff))",
               "learnable_interception": ["S_eff", "c"],
               "effective_learnable_parameters": 10,
               "boundary_pass": boundary_pass,
               "gradient_pass": grad_pass,
               "water_balance_pass": wb_pass,
               "pet_closure_pass": closure_pass,
               "snow_partition_pass": snow_pass,
               "compile": compile_audit,
               "MOPEX3_regression": None, "MOPEX5_unchanged": None, "test_count": None}
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("MOPEX4 FINAL TWO-PARAM VALIDATION")
    print(f"formula boundaries: {'PASS' if boundary_pass else 'FAIL'}")
    print(f"S_eff/c gradients (analytic vs autograd + step autograd): {'PASS' if grad_pass else 'FAIL'}")
    print(f"water balance: {'PASS' if wb_pass else 'FAIL'}; PET closure: {'PASS' if closure_pass else 'FAIL'}")
    print(f"snow-partition regression (snowfall not intercepted): {'PASS' if snow_pass else 'FAIL'}")
    print(f"compile: graph_breaks={compile_audit['graph_breaks']} recompiles={compile_audit['recompiles']} "
          f"fullgraph={compile_audit['fullgraph']['success']} steady={compile_audit['steady_state_step_sec']:.5f}s "
          f"eager={compile_audit['eager_step_sec']:.5f}s")


if __name__ == "__main__":
    main()
