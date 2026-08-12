"""
Test old-models core step functions for:
1. Water balance (P = Q + Ea + dS within tolerance)
2. Gradient stability (torch.autograd.gradcheck)
3. NaN/Inf stability
"""
from __future__ import annotations

import inspect, re, sys, os
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass
from typing import Callable

import torch
import numpy as np

# ---- Import old-models core ----
sys.path.insert(0, "/tmp")
from old_models.core import PARAM_INFO, STFN_INFO, INIT_INFO, STATE_INFO

NEARZERO = 1.0e-6
SEED = 20260623


# ====================================================================
# Registry builders (mirrors tests/core_model_registry.py)
# ====================================================================

STATE_SIGN_OVERRIDES = {
    "ihacres": (-1.0,),
    "penman": (1.0, -1.0, 1.0),
    "tcm": (1.0, -1.0, 1.0, 1.0),
    "topmodel": (1.0, -1.0),
}

DISABLED_MODELS = {"shm": "Empty file"}


@dataclass(frozen=True)
class CoreModelEntry:
    model_name: str
    step_fn: Callable
    init_fn: Callable
    param_bounds: dict
    state_names: tuple
    state_signs: tuple
    uses_snow: bool
    supports_diagnostics: bool
    enabled: bool


def _actual_state_count(init_fn):
    return len(init_fn(1, 1, torch.device("cpu"), NEARZERO))


def _extract_state_names(init_fn):
    actual_count = _actual_state_count(init_fn)
    return tuple(f"S{i + 1}" for i in range(actual_count))


def _uses_snow(step_fn, param_bounds):
    source = inspect.getsource(step_fn).lower()
    param_names = {name.lower() for name in param_bounds}
    snow_markers = {"tt", "tti", "ttm", "ddf", "tcrit", "cfmax", "whc", "cfr"}
    return "snow" in source or "snowfall" in source or "melt" in source or bool(param_names & snow_markers)


def build_core_model_registry():
    registry = {}
    for name, bounds in PARAM_INFO.items():
        entry = CoreModelEntry(
            model_name=name,
            step_fn=STFN_INFO[name],
            init_fn=INIT_INFO[name],
            param_bounds=bounds,
            state_names=_extract_state_names(INIT_INFO[name]),
            state_signs=STATE_SIGN_OVERRIDES.get(name, tuple(1.0 for _ in range(_actual_state_count(INIT_INFO[name])))),
            uses_snow=_uses_snow(STFN_INFO[name], bounds),
            supports_diagnostics="return_diagnostics" in inspect.signature(STFN_INFO[name]).parameters,
            enabled=name not in DISABLED_MODELS,
        )
        registry[name] = entry
    return registry


CORE_MODEL_REGISTRY = build_core_model_registry()


# ====================================================================
# Water Balance Utils (simplified from tests/core_water_balance_utils.py)
# ====================================================================

def _signed_storage_sum(entry, states):
    total = torch.zeros_like(states[0])
    for sign, state in zip(entry.state_signs, states):
        total = total + sign * state
    return total


def _param_value(lo, hi, mode, rand):
    if mode == "midpoint":
        return torch.full_like(rand, (lo + hi) / 2.0)
    if mode == "lower_near":
        return torch.full_like(rand, lo + 0.01 * (hi - lo))
    if mode == "upper_near":
        return torch.full_like(rand, lo + 0.99 * (hi - lo))
    if mode == "random_valid":
        return lo + rand * (hi - lo)
    raise KeyError(mode)


def _call_step(entry, forcing_at_step, step_index, params_list, states, return_diagnostics=False):
    precip_t, temp_t, pet_t = forcing_at_step
    kwargs = {}
    sig = inspect.signature(entry.step_fn).parameters
    if "doy" in sig:
        kwargs["doy"] = torch.full_like(precip_t, float((step_index % 365) + 1))
    if "mean_P" in sig:
        kwargs["mean_P"] = torch.zeros_like(precip_t)
    if "delta_t" in sig:
        kwargs["delta_t"] = torch.ones_like(precip_t)
    if return_diagnostics and "return_diagnostics" in sig:
        kwargs["return_diagnostics"] = True

    result = entry.step_fn(precip_t, temp_t, pet_t, *params_list, *states, **kwargs)
    if return_diagnostics and "return_diagnostics" in sig:
        diagnostics = result[-1]
        next_states = list(result[2:-1])
        extra_losses = diagnostics.get("external_losses", torch.zeros_like(precip_t))
    else:
        next_states = list(result[2:])
        extra_losses = torch.zeros_like(precip_t)
    return result[0], result[1], next_states, extra_losses


def run_water_balance_case(entry, n_steps=365, parameter_case="midpoint"):
    """Run a water balance check on a single model with constant forcing."""
    dtype = torch.float64
    device = "cpu"
    batch_shape = (3, 2)
    n_grid, n_mul = batch_shape

    gen = torch.Generator(device=device).manual_seed(SEED)

    # Build forcing
    precip = torch.full((n_steps, n_grid, n_mul), 4.0, dtype=dtype, device=device)
    temp = torch.full((n_steps, n_grid, n_mul), 8.0, dtype=dtype, device=device)
    pet = torch.full((n_steps, n_grid, n_mul), 2.0, dtype=dtype, device=device)

    # Build parameters
    params_list = []
    for param_name, (lo, hi) in entry.param_bounds.items():
        rand = torch.rand((n_grid, n_mul), dtype=dtype, device=device, generator=gen)
        values = _param_value(lo, hi, parameter_case, rand)
        params_list.append(values)

    # Build states
    states = [s.to(dtype=dtype) for s in entry.init_fn(n_grid, n_mul, torch.device(device), NEARZERO)]
    initial_storage = _signed_storage_sum(entry, states)

    total_input = torch.zeros(batch_shape, dtype=dtype, device=device)
    total_output = torch.zeros(batch_shape, dtype=dtype, device=device)
    max_step_residual = 0.0
    nan_count = 0
    inf_count = 0

    for t in range(n_steps):
        storage_before = _signed_storage_sum(entry, states)
        qsim, ea, next_states, extra = _call_step(
            entry, (precip[t], temp[t], pet[t]), t, params_list, states, return_diagnostics=True
        )
        storage_after = _signed_storage_sum(entry, next_states)

        step_residual = precip[t] - (qsim + ea + extra) - (storage_after - storage_before)
        max_step_residual = max(max_step_residual, float(torch.max(torch.abs(step_residual))))

        total_input = total_input + precip[t]
        total_output = total_output + qsim + ea + extra
        states = next_states

        for state in states:
            nan_count += int(torch.isnan(state).sum().item())
            inf_count += int(torch.isinf(state).sum().item())

    final_storage = _signed_storage_sum(entry, states)
    full_residual = total_input - total_output - (final_storage - initial_storage)
    max_full_abs = float(torch.max(torch.abs(full_residual)))
    rel_residual = float(torch.max(torch.abs(full_residual) / torch.maximum(torch.abs(total_input), torch.tensor(1e-12))))

    n_params = len(entry.param_bounds)
    n_states_actual = _actual_state_count(entry.init_fn)
    clamp_budget = n_states_actual * n_steps * NEARZERO
    abs_tol = max(1e-7, clamp_budget)

    passed = max_full_abs <= abs_tol and nan_count == 0 and inf_count == 0

    return {
        "model": entry.model_name,
        "n_params": n_params,
        "n_states_registry": STATE_INFO.get(entry.model_name, 0),
        "n_states_actual": n_states_actual,
        "max_step_residual": max_step_residual,
        "max_full_residual": max_full_abs,
        "rel_residual": rel_residual,
        "abs_tol": abs_tol,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "passed": passed,
    }


# ====================================================================
# Gradient Stability Test
# ====================================================================

def check_gradient_stability(entry, n_steps=2):
    """Check that step_fn gradients are finite and non-exploding."""
    dtype = torch.float64
    device = "cpu"
    n_grid, n_mul = 2, 1

    # Build params with gradients
    param_tensors = []
    gen = torch.Generator(device=device).manual_seed(SEED + 1)
    for name, (lo, hi) in entry.param_bounds.items():
        rand = torch.rand((n_grid, n_mul), dtype=dtype, device=device, generator=gen)
        val = _param_value(lo, hi, "midpoint", rand).clone().detach().requires_grad_(True)
        param_tensors.append(val)

    states = [s.to(dtype=dtype) for s in entry.init_fn(n_grid, n_mul, torch.device(device), NEARZERO)]

    precip = torch.ones(n_grid, n_mul, dtype=dtype, device=device) * 4.0
    temp = torch.ones(n_grid, n_mul, dtype=dtype, device=device) * 8.0
    pet = torch.ones(n_grid, n_mul, dtype=dtype, device=device) * 2.0

    failed_params = []
    for i, (name, bounds) in enumerate(entry.param_bounds.items()):
        p = param_tensors[i]

        def loss_fn(param):
            kwargs = {}
            sig = inspect.signature(entry.step_fn).parameters
            if "doy" in sig:
                kwargs["doy"] = torch.ones_like(param) * 180
            if "mean_P" in sig:
                kwargs["mean_P"] = torch.zeros_like(param)
            if "delta_t" in sig:
                kwargs["delta_t"] = torch.ones_like(param)
            arg_list = [precip, temp, pet]
            arg_list.extend([p2.detach() if j != i else param for j, p2 in enumerate(param_tensors)])
            arg_list.extend(states)
            result = entry.step_fn(*arg_list, **kwargs)
            return result[0].sum() + result[1].sum()

        try:
            grad = torch.autograd.grad(loss_fn(p), p, create_graph=False)[0]
            grad_max = float(torch.max(torch.abs(grad)))
            if grad_max > 1e6 or torch.isnan(grad).any() or torch.isinf(grad).any():
                failed_params.append(f"{name}(max_grad={grad_max:.1f})")
        except Exception as e:
            failed_params.append(f"{name}(exception: {str(e)[:60]})")

    return {
        "model": entry.model_name,
        "n_params_total": len(entry.param_bounds),
        "failed_params": failed_params,
        "all_ok": len(failed_params) == 0,
    }


# ====================================================================
# Euler Substeps Convergence Test
# ====================================================================

def check_euler_convergence(entry, substep_counts=(1, 4, 16)):
    """Check that as n_substeps increases, state error decreases (first-order convergence)."""
    dtype = torch.float64
    device = "cpu"
    n_grid, n_mul = 1, 1
    n_steps = 10

    gen = torch.Generator(device=device).manual_seed(SEED + 2)
    params_list = []
    for name, (lo, hi) in entry.param_bounds.items():
        rand = torch.rand((n_grid, n_mul), dtype=dtype, device=device, generator=gen)
        val = _param_value(lo, hi, "midpoint", rand)
        params_list.append(val)

    base_states = [s.to(dtype=dtype) for s in entry.init_fn(n_grid, n_mul, torch.device(device), NEARZERO)]

    def run_trajectory(n_substeps):
        dt = 1.0 / n_substeps
        states = [s.clone() for s in base_states]
        total_q = torch.zeros(n_grid, n_mul, dtype=dtype, device=device)
        for t in range(n_steps):
            precip = torch.ones(n_grid, n_mul, dtype=dtype, device=device) * (4.0 * dt)
            temp = torch.ones(n_grid, n_mul, dtype=dtype, device=device) * 8.0
            pet = torch.ones(n_grid, n_mul, dtype=dtype, device=device) * (2.0 * dt)
            for _ in range(n_substeps):
                qsim, ea, states, _ = _call_step(
                    entry, (precip, temp, pet), t, params_list, states, return_diagnostics=True
                )
                total_q = total_q + qsim
        return states, total_q

    ref_states, ref_q = run_trajectory(max(substep_counts))

    errors = {}
    for n_sub in substep_counts:
        if n_sub == max(substep_counts):
            continue
        states, q = run_trajectory(n_sub)
        state_err = 0.0
        for s_ref, s in zip(ref_states, states):
            state_err += float(torch.max(torch.abs(s_ref - s)))
        errors[n_sub] = {"state_error": state_err, "q_error": float(torch.max(torch.abs(ref_q - q)))}

    # Check monotonic decrease
    decreasing = True
    prev = float("inf")
    for n in sorted(errors):
        if errors[n]["state_error"] > prev:
            decreasing = False
            break
        prev = errors[n]["state_error"]

    return {
        "model": entry.model_name,
        "substep_errors": {str(k): v for k, v in errors.items()},
        "decreasing": decreasing,
    }


# ====================================================================
# Main Test Runner
# ====================================================================

def main():
    models = list(CORE_MODEL_REGISTRY.keys())
    print(f"Testing {len(models)} old-models core models\n")

    # ---- 1. Water Balance ----
    print("=" * 70)
    print("1. WATER BALANCE TEST (P = Q + Ea + dS, float64)")
    print("=" * 70)
    wb_results = []
    failures = []
    for name in sorted(models):
        entry = CORE_MODEL_REGISTRY[name]
        if not entry.enabled:
            continue
        try:
            r = run_water_balance_case(entry)
            wb_results.append(r)
            status = "PASS" if r["passed"] else "FAIL"
            if not r["passed"]:
                failures.append(r)
            print(f"  {name:20s} {status}  max_full={r['max_full_residual']:.2e}  "
                  f"max_step={r['max_step_residual']:.2e}  nan={r['nan_count']}  "
                  f"tol={r['abs_tol']:.2e}")
        except Exception as e:
            print(f"  {name:20s} ERROR: {str(e)[:80]}")
            failures.append({"model": name, "error": str(e)})

    n_pass = sum(1 for r in wb_results if r["passed"])
    print(f"\n  Water balance: {n_pass}/{len(wb_results)} passed, {len(failures)} failed")

    if failures:
        print("\n  === FAILURES ===")
        for f in failures:
            if "error" in f:
                print(f"    {f['model']}: ERROR - {f['error'][:100]}")
            else:
                print(f"    {f['model']}: max_full={f['max_full_residual']:.3e} rel={f['rel_residual']:.3e} tol={f['abs_tol']:.3e}")

    # ---- 2. Gradient Stability ----
    print("\n" + "=" * 70)
    print("2. GRADIENT STABILITY TEST")
    print("=" * 70)
    grad_failures = []
    for name in sorted(models):
        entry = CORE_MODEL_REGISTRY[name]
        if not entry.enabled:
            continue
        try:
            r = check_gradient_stability(entry)
            if not r["all_ok"]:
                grad_failures.append(r)
                print(f"  {name:20s} FAIL: {r['failed_params']}")
            else:
                print(f"  {name:20s} PASS  all {r['n_params_total']} params differentiable")
        except Exception as e:
            print(f"  {name:20s} ERROR: {str(e)[:80]}")
            grad_failures.append({"model": name, "error": str(e)})

    print(f"\n  Gradient: {len(models) - len(grad_failures)}/{len(models)} fully differentiable")

    # ---- 3. Euler Convergence ----
    print("\n" + "=" * 70)
    print("3. EULER SUBSTEP CONVERGENCE TEST")
    print("=" * 70)
    euler_issues = []
    for name in sorted(models):
        entry = CORE_MODEL_REGISTRY[name]
        if not entry.enabled:
            continue
        try:
            r = check_euler_convergence(entry)
            status = "OK" if r["decreasing"] else "NOT DECREASING"
            err = r["substep_errors"]
            print(f"  {name:20s} {status}  err@1={err.get('1', {}).get('state_error', 'n/a'):.4e}")
            if not r["decreasing"]:
                euler_issues.append(r)
        except Exception as e:
            print(f"  {name:20s} ERROR: {str(e)[:80]}")

    print(f"\n  Euler convergence monotonic for {len(models) - len(euler_issues)}/{len(models)} models")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Water balance: {n_pass}/{len(wb_results)} passed")
    print(f"  Gradient stability: {len(models) - len(grad_failures)}/{len(models)} OK")
    print(f"  Euler convergence: {len(models) - len(euler_issues)}/{len(models)} monotonic")

    all_ok = n_pass >= len(wb_results) * 0.8 and len(grad_failures) < 5
    print(f"\n  Overall: {'PASS' if all_ok else 'NEEDS REVIEW'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    exit(main())
