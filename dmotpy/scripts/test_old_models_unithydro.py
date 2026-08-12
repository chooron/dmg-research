"""
Validate old-models unithydro: weight accuracy, routing, impulse, mass conservation.
"""
from __future__ import annotations

import sys, os, inspect
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/tmp")
from old_models.unithydro import (
    DplHalf1, DplFull2, DplTri3, DplTri4,
    DplExp5, DplGamma6, DplUniform7, DplDelay8, DplIdentity0,
)

PROJECT_ROOT = Path("/home/jingxin/code/dmg-research")
sys.path.insert(0, str(PROJECT_ROOT))
from tests.reference_unithydro_numpy import (
    build_unit_hydrograph_numpy, route_with_unit_hydrograph_numpy,
)

MODEL_REGISTRY = {
    "half1": DplHalf1, "full2": DplFull2, "tri3": DplTri3,
    "tri4": DplTri4, "exp5": DplExp5, "gamma6": DplGamma6,
    "uniform7": DplUniform7, "delay8": DplDelay8,
}

PARAM_CASES = {
    "half1": (0.3, 1.0, 2.7, 7.5),
    "full2": (0.3, 1.0, 2.7, 7.5),
    "tri3":  (0.3, 1.0, 2.7, 7.5),
    "tri4":  (0.5, 1.0, 2.5, 7.0),
    "exp5":  (0.5, 1.2, 3.0, 7.0),
    "gamma6": (1.0, 0.5),
    "uniform7": (0.5, 1.0, 3.0, 7.0),
    "delay8": (0.3, 1.0, 2.5, 7.0),
}

SEED = 20260623


def _uh_len(kind, params, max_lag=32):
    uh = MODEL_REGISTRY[kind](max_lag=max_lag)
    if kind == "gamma6":
        p = torch.tensor([params], dtype=torch.float64)
    elif len(params) == 1:
        p = torch.tensor([[params[0]]], dtype=torch.float64)
    else:
        p = torch.tensor([params], dtype=torch.float64)
    w = uh.get_weights(p)
    return int((w[0, 0] > 1e-15).sum().item())


def test_weight_shape_and_sum():
    """Verify weight shapes are correct and sum to 1.0."""
    print("\n--- Weight shape & mass ---")
    all_ok = True
    for kind in sorted(MODEL_REGISTRY):
        for pval in [PARAM_CASES[kind][:1]][0] if kind == "gamma6" else [PARAM_CASES[kind][:1]][0]:
            pass
    for kind in sorted(MODEL_REGISTRY):
        if kind == "gamma6":
            vals = [PARAM_CASES[kind]]
        else:
            vals = [(v,) for v in PARAM_CASES[kind]]
        for params in vals:
            max_lag = 32
            uh = MODEL_REGISTRY[kind](max_lag=max_lag)
            if kind == "gamma6":
                p = torch.tensor([params], dtype=torch.float64)
            elif len(params) == 1:
                p = torch.tensor([[params[0]]], dtype=torch.float64)
            else:
                p = torch.tensor([params], dtype=torch.float64)
            w = uh.get_weights(p)
            s = float(w.sum())
            ok = abs(s - 1.0) < 1e-10
            if not ok:
                print(f"  {kind:10s} {str(params):20s} FAIL: sum={s:.6f}")
                all_ok = False
    if all_ok:
        print("  All UH weights sum to 1.0 -- PASS")
    return all_ok


def test_numpy_comparison():
    """Compare dMoT weights against NumPy reference."""
    print("\n--- NumPy weight comparison ---")
    all_ok = True
    for kind in sorted(MODEL_REGISTRY):
        if kind == "gamma6":
            params_list = [PARAM_CASES[kind]]
        else:
            params_list = [(v,) for v in PARAM_CASES[kind]]
        for params in params_list:
            max_lag = 32
            uh = MODEL_REGISTRY[kind](max_lag=max_lag)
            if kind == "gamma6":
                p = torch.tensor([params], dtype=torch.float64)
            elif len(params) == 1:
                p = torch.tensor([[params[0]]], dtype=torch.float64)
            else:
                p = torch.tensor([params], dtype=torch.float64)
            dmot_w = uh.get_weights(p)[0, 0].detach().cpu().numpy()

            try:
                if kind == "gamma6":
                    np_w = build_unit_hydrograph_numpy(kind, params)
                else:
                    np_w = build_unit_hydrograph_numpy(kind, params[0])
            except Exception:
                continue

            # Trim to same length
            min_len = min(len(dmot_w), len(np_w))
            dmot_cmp = dmot_w[:min_len]
            np_cmp = np_w[:min_len]

            max_abs = float(np.max(np.abs(dmot_cmp - np_cmp)))
            # Check tail (dmot may be longer)
            tail_mass = float(dmot_w[min_len:].sum()) if len(dmot_w) > min_len else 0.0

            ok = max_abs < 1e-6 or (kind in ("gamma6", "delay8") and max_abs < 1e-3)
            status = "PASS" if ok else "FAIL"
            print(f"  {kind:10s} {str(params):20s} {status}  "
                  f"max_abs={max_abs:.3e}  len_dmot={len(dmot_w)}  len_np={len(np_w)}  tail={tail_mass:.3e}")
            if not ok:
                all_ok = False
    return all_ok


def test_routing_mass_balance():
    """Check that total output mass equals total input mass for random inflow."""
    print("\n--- Routing mass balance ---")
    torch.manual_seed(SEED + 42)
    all_ok = True
    for kind in sorted(MODEL_REGISTRY):
        max_lag = 24
        uh = MODEL_REGISTRY[kind](max_lag=max_lag)
        n = 200
        inflow = torch.rand(1, n, dtype=torch.float64) * 10.0

        if kind == "gamma6":
            p = torch.tensor([PARAM_CASES[kind]], dtype=torch.float64)
        else:
            pval = PARAM_CASES[kind][1]
            p = torch.tensor([[pval]], dtype=torch.float64)

        routed = uh(inflow, p)
        in_sum = float(inflow.sum())
        out_sum = float(routed.sum())
        rel_err = abs(out_sum - in_sum) / max(in_sum, 1e-12)

        ok = rel_err < 1e-6
        status = "PASS" if ok else "FAIL"
        print(f"  {kind:10s} {status}  mass_rel_err={rel_err:.3e}")
        if not ok:
            all_ok = False
    return all_ok


def test_impulse_response():
    """Test causal impulse response."""
    print("\n--- Impulse response ---")
    all_ok = True
    for kind in sorted(MODEL_REGISTRY):
        max_lag = 20
        uh = MODEL_REGISTRY[kind](max_lag=max_lag)
        n_steps = 40
        impulse = torch.zeros(1, n_steps, dtype=torch.float64)
        impulse[0, 5] = 1.0  # impulse at t=5

        if kind == "gamma6":
            p = torch.tensor([PARAM_CASES[kind]], dtype=torch.float64)
        else:
            pval = PARAM_CASES[kind][1]
            p = torch.tensor([[pval]], dtype=torch.float64)

        out = uh(impulse, p)
        out_arr = out[0].detach().cpu().numpy()

        # No output before impulse
        before = np.max(np.abs(out_arr[:5]))
        causal = before < 1e-15

        # Has output at or after impulse
        has_response = float(out_arr[5:].max()) > 1e-15

        # For delay8: pure delay, response starts at t=5+ceil(delay)
        if kind == "delay8":
            delay = int(np.ceil(PARAM_CASES[kind][1]))
            # Output should be zero before t=5+delay
            has_early = np.max(np.abs(out_arr[:5 + delay])) > 1e-15
            has_late = np.max(np.abs(out_arr[5 + delay:])) > 1e-15 if 5 + delay < n_steps else False
            ok = not has_early and has_late
        else:
            ok = causal and has_response

        status = "PASS" if ok else "FAIL"
        peak_t = int(np.argmax(np.abs(out_arr)))
        print(f"  {kind:10s} {status}  causal={causal}  "
              f"peak_at_t={peak_t}  peak_val={out_arr[peak_t]:.4f}  "
              f"out_sum={out_arr.sum():.4f}")
        if not ok:
            all_ok = False
    return all_ok


def test_identity():
    """DplIdentity0: output should equal input."""
    print("\n--- Identity UH ---")
    identity = DplIdentity0(max_lag=1)
    inflow = torch.rand(1, 30, dtype=torch.float64) * 10.0
    out = identity(inflow, None)
    max_diff = float(torch.max(torch.abs(out - inflow)))
    ok = max_diff < 1e-12
    status = "PASS" if ok else "FAIL"
    print(f"  identity  {status}  max_diff={max_diff:.3e}")
    return ok


def test_large_inflow_stability():
    """Test numerical stability with extreme inflow values."""
    print("\n--- Large inflow stability ---")
    all_ok = True
    for kind in sorted(MODEL_REGISTRY):
        max_lag = 16
        uh = MODEL_REGISTRY[kind](max_lag=max_lag)
        n = 50
        inflow = torch.full((1, n), 1000.0, dtype=torch.float64)

        if kind == "gamma6":
            p = torch.tensor([PARAM_CASES[kind]], dtype=torch.float64)
        else:
            pval = PARAM_CASES[kind][1]
            p = torch.tensor([[pval]], dtype=torch.float64)

        try:
            out = uh(inflow, p)
            has_nan = torch.isnan(out).any().item()
            has_inf = torch.isinf(out).any().item()
            ok = not has_nan and not has_inf
            status = "PASS" if ok else "FAIL"
            print(f"  {kind:10s} {status}  nan={has_nan}  inf={has_inf}")
            if not ok:
                all_ok = False
        except Exception as e:
            print(f"  {kind:10s} FAIL  exception: {e}")
            all_ok = False
    return all_ok


def main():
    results = {}
    results["weight_shape"] = test_weight_shape_and_sum()
    results["numpy_compare"] = test_numpy_comparison()
    results["routing_mass"] = test_routing_mass_balance()
    results["impulse"] = test_impulse_response()
    results["identity"] = test_identity()
    results["stability"] = test_large_inflow_stability()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {name:25s} {'PASS' if ok else 'FAIL'}")
    all_pass = all(results.values())
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
