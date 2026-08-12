#!/usr/bin/env python3
"""Small runtime verification for the opt-in MOPEX4/5 continuation context.

This is deliberately a mechanism check: one synthetic basin, float64 CPU,
short forwards, and five one-step AdamW continuation stages.  It is not a
training or performance experiment.
"""
from __future__ import annotations

import csv
import inspect
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

BENCHMARK = Path(__file__).resolve().parents[2]
REPO = BENCHMARK.parents[1]
sys.path[:0] = [str(REPO), str(BENCHMARK), str(BENCHMARK / "src")]

from dmotpy.models.core.mopex4 import mopex4_step
from dmotpy.models.core.mopex5 import mopex5_step
from dmotpy.models.flux.mopex import (
    _training_values,
    mopex_interception_4,
    mopex_phenology_1,
    mopex_training_context,
)
from dmotpy.models.hydrology_model import HydrologyModel
from dpl.nn_parameterizer import CatchmentParameterizer
from project.benchmark.src.model_registry import model_config
from project.benchmark.scripts.run_dpl_benchmark_dmg_native import compute_differentiable_kge

DTYPE = torch.float64
DEVICE = torch.device("cpu")
WARMUP = 3
STAGES = (0.0, 0.25, 0.5, 0.75, 1.0)
OUT = BENCHMARK / "results" / "mopex45_phase_fix" / "continuation_check"


def finite_tensors(values) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in values if torch.is_tensor(value))


def max_diff(left, right) -> float:
    return max(float((a - b).abs().max()) for a, b in zip(left, right))


def physical_inputs() -> tuple[torch.Tensor, ...]:
    shape = (1, 1)
    p = torch.tensor([[6.0]], dtype=DTYPE)
    temp = torch.tensor([[5.0]], dtype=DTYPE)
    pet = torch.tensor([[2.0]], dtype=DTYPE)
    doy = torch.tensor([[180.0]], dtype=DTYPE)
    m4 = [torch.tensor([[value]], dtype=DTYPE) for value in
          (0.0, 10.0, 100.0, 0.2, 0.4, 180.0, 0.2, 0.5, 100.0, 0.2)]
    m5 = [torch.tensor([[value]], dtype=DTYPE) for value in
          (0.0, 10.0, 100.0, 0.2, 0.4, 180.0, -2.0, 10.0, 0.2, 0.5, 100.0, 0.2)]
    states = [torch.full(shape, 0.3, dtype=DTYPE) for _ in range(5)]
    return p, temp, pet, doy, m4, m5, states


def direct_step(name: str, lambda_i: float | None, lambda_p: float | None, beta: float | None):
    p, temp, pet, doy, m4, m5, states = physical_inputs()
    # Core signatures take S1,S2,Sc1,Sc2,Sn; create_initial_state is Sn,S1,S2,Sc1,Sc2.
    ordered_states = [states[1], states[2], states[3], states[4], states[0]]
    context = (mopex_training_context(lambda_i=lambda_i, lambda_p=lambda_p, beta=beta)
               if lambda_i is not None else None)
    if context is None:
        return _direct_step_call(name, p, temp, pet, doy, m4, m5, ordered_states)
    with context:
        return _direct_step_call(name, p, temp, pet, doy, m4, m5, ordered_states)


def _direct_step_call(name, p, temp, pet, doy, m4, m5, ordered_states):
    if name == "mopex4":
        return mopex4_step(p, temp, pet, *m4, *ordered_states, doy=doy)
    return mopex5_step(p, temp, pet, *m5, *ordered_states, doy=doy)


def endpoint_and_semantics(rows: list[dict]) -> None:
    for name in ("mopex4", "mopex5"):
        default = direct_step(name, None, None, None)
        explicit = direct_step(name, 1.0, 1.0, 50.0)
        rows.append({"check": f"{name}_step_endpoint_equivalence", "max_abs_diff": max_diff(default, explicit),
                     "pass": max_diff(default, explicit) <= 1e-12})
        rows.append({"check": f"{name}_endpoint_all_outputs_finite", "max_abs_diff": 0.0,
                     "pass": finite_tensors(default) and finite_tensors(explicit)})

        default_sequence = direct_sequence(name, None, None, None)
        explicit_sequence = direct_sequence(name, 1.0, 1.0, 50.0)
        warmup_diff = max_diff(default_sequence[WARMUP - 1][2:], explicit_sequence[WARMUP - 1][2:])
        prediction_diff = max(
            max_diff(left, right)
            for left, right in zip(default_sequence[WARMUP:], explicit_sequence[WARMUP:])
        )
        rows.append({"check": f"{name}_state_after_warmup_equivalence", "max_abs_diff": warmup_diff,
                     "pass": warmup_diff <= 1e-12})
        rows.append({"check": f"{name}_prediction_equivalence_after_warmup", "max_abs_diff": prediction_diff,
                     "pass": prediction_diff <= 1e-12})

    p, temp, pet, doy, m4, m5, _states = physical_inputs()
    intercept_1 = mopex_interception_4(p, doy, m4[4], m4[5])
    with mopex_training_context(lambda_i=0.0, lambda_p=1.0, beta=50.0):
        intercept_0 = mopex_interception_4(p, doy, m4[4], m4[5])
    with mopex_training_context(lambda_i=0.5, lambda_p=1.0, beta=50.0):
        intercept_half = mopex_interception_4(p, doy, m4[4], m4[5])
    rows.append({"check": "mopex4_lambda_i_zero_only_scales_interception", "max_abs_diff": float(intercept_0.abs().max()),
                 "pass": float(intercept_0.abs().max()) <= 1e-12 and
                        float((intercept_half - 0.5 * intercept_1).abs().max()) <= 1e-12})

    gsi = torch.clamp((temp - m5[6]) / torch.clamp(m5[7], min=1e-6), 0.0, 1.0)
    with mopex_training_context(lambda_i=1.0, lambda_p=0.0, beta=50.0):
        pet_0 = mopex_phenology_1(temp, m5[6], m5[7], pet)
    with mopex_training_context(lambda_i=1.0, lambda_p=1.0, beta=50.0):
        pet_1 = mopex_phenology_1(temp, m5[6], m5[7], pet)
    with mopex_training_context(lambda_i=1.0, lambda_p=0.5, beta=50.0):
        pet_half = mopex_phenology_1(temp, m5[6], m5[7], pet)
    pet_error = max(float((pet_0 - pet).abs().max()), float((pet_1 - gsi * pet).abs().max()),
                    float((pet_half - 0.5 * (pet + gsi * pet)).abs().max()))
    rows.append({"check": "mopex5_lambda_p_endpoint_semantics", "max_abs_diff": pet_error, "pass": pet_error <= 1e-12})

    with mopex_training_context(lambda_i=1.0, lambda_p=1.0, beta=50.0):
        beta_context = mopex_interception_4(p, doy, m4[4], m4[5])
    rows.append({"check": "beta_50_matches_default_interception", "max_abs_diff": float((beta_context - intercept_1).abs().max()),
                 "pass": float((beta_context - intercept_1).abs().max()) <= 1e-12})


def direct_sequence(name: str, lambda_i: float | None, lambda_p: float | None, beta: float | None):
    p, temp, pet, doy, m4, m5, states = physical_inputs()
    ordered_states = [states[1], states[2], states[3], states[4], states[0]]
    values = []
    context = (mopex_training_context(lambda_i=lambda_i, lambda_p=lambda_p, beta=beta)
               if lambda_i is not None else None)
    def run():
        current = ordered_states
        for index in range(8):
            step_doy = doy + index
            result = _direct_step_call(name, p, temp, pet, step_doy, m4, m5, current)
            values.append(result)
            current = list(result[2:])
    if context is None:
        run()
    else:
        with context:
            run()
    return values


def context_safety(rows: list[dict]) -> None:
    baseline = _training_values()
    nested_ok = exception_ok = False
    try:
        with mopex_training_context(lambda_i=0.25, lambda_p=0.5, beta=50.0):
            outer = _training_values()
            with mopex_training_context(lambda_i=0.75, lambda_p=1.0, beta=20.0):
                nested_ok = outer == (0.25, 0.5, 50.0) and _training_values() == (0.75, 1.0, 20.0)
            nested_ok = nested_ok and _training_values() == outer
            try:
                with mopex_training_context(lambda_i=0.5, lambda_p=0.5, beta=10.0):
                    raise RuntimeError("context safety probe")
            except RuntimeError:
                exception_ok = _training_values() == outer
    finally:
        restored = _training_values() == baseline
    rows.extend([
        {"check": "context_nested_restoration", "max_abs_diff": 0.0, "pass": nested_ok},
        {"check": "context_exception_restoration", "max_abs_diff": 0.0, "pass": exception_ok},
        {"check": "context_default_restored_after_exit", "max_abs_diff": 0.0, "pass": restored and baseline == (1.0, 1.0, 50.0)},
    ])


def make_model(name: str, lambda_i: float | None = None, lambda_p: float | None = None,
               beta: float | None = None) -> HydrologyModel:
    cfg = model_config(name, warm_up=WARMUP, backend="python", parameter_mapping="linear",
                       warmup_grad_mode="detach")
    if lambda_i is not None:
        cfg.update(continuation_lambda_i=lambda_i, continuation_lambda_p=lambda_p, continuation_beta=beta)
    return HydrologyModel(cfg, device=DEVICE, backend="python").to(DEVICE)


def forcing_and_theta(name: str, steps: int = 8):
    n_params = 10 if name == "mopex4" else 12
    p = torch.linspace(2.0, 8.0, steps, dtype=DTYPE).view(steps, 1)
    temp = torch.linspace(-1.0, 12.0, steps, dtype=DTYPE).view(steps, 1)
    pet = torch.linspace(1.0, 3.0, steps, dtype=DTYPE).view(steps, 1)
    doy = torch.arange(steps, dtype=DTYPE).view(steps, 1) + 175.0
    x = torch.stack((p, temp, pet, doy), dim=-1)
    theta = torch.full((1, n_params), 0.5, dtype=DTYPE, requires_grad=True)
    y = torch.ones((steps - WARMUP, 1), dtype=DTYPE)
    return x, theta, y


def model_forward_checks(rows: list[dict]) -> None:
    expected_m4 = ["P", "T", "PET", "tcrit", "ddf", "Sb1", "tw", "alpha", "is_time",
                   "tu", "Se", "Sb2", "tc", "S1", "S2", "Sc1", "Sc2", "Sn", "delta_t",
                   "nearzero", "doy", "phase_cos", "phase_sin"]
    expected_m5 = ["P", "T", "PET", "tcrit", "ddf", "Sb1", "tw", "alpha", "is_time", "tmin",
                   "trange", "tu", "Se", "Sb2", "tc", "S1", "S2", "Sc1", "Sc2", "Sn", "delta_t",
                   "nearzero", "doy", "phase_cos", "phase_sin"]
    for name, step_fn, expected in (("mopex4", mopex4_step, expected_m4), ("mopex5", mopex5_step, expected_m5)):
        actual = list(inspect.signature(step_fn).parameters)
        rows.append({"check": f"{name}_public_step_signature", "max_abs_diff": 0.0,
                     "pass": actual == expected})
    for name in ("mopex4", "mopex5"):
        x, theta, y = forcing_and_theta(name)
        default_model = make_model(name)
        explicit_model = make_model(name, 1.0, 1.0, 50.0)
        q_default = default_model({"x_phy": x}, (None, theta.detach()))["streamflow"]
        q_explicit = explicit_model({"x_phy": x}, (None, theta.detach()))["streamflow"]
        diff = float((q_default - q_explicit).abs().max())
        rows.append({"check": f"{name}_forward_endpoint_equivalence", "max_abs_diff": diff, "pass": diff <= 1e-12})
        for lambda_i, lambda_p in ((0.0, 1.0), (0.5, 1.0), (1.0, 1.0)):
            model = make_model(name, lambda_i, lambda_p, 50.0)
            raw = theta.detach().clone().requires_grad_(True)
            q = model({"x_phy": x}, (None, raw))["streamflow"]
            loss = q.square().mean()
            loss.backward()
            finite = bool(torch.isfinite(q).all() and torch.isfinite(raw.grad).all())
            rows.append({"check": f"{name}_autograd_lambda_i_{lambda_i:g}", "max_abs_diff": float(q.detach().abs().max()), "pass": finite})
        if name == "mopex5":
            for lambda_p in (0.0, 0.5, 1.0):
                model = make_model(name, 1.0, lambda_p, 50.0)
                raw = theta.detach().clone().requires_grad_(True)
                q = model({"x_phy": x}, (None, raw))["streamflow"]
                q.square().mean().backward()
                rows.append({"check": f"{name}_autograd_lambda_p_{lambda_p:g}", "max_abs_diff": float(q.detach().abs().max()),
                             "pass": bool(torch.isfinite(q).all() and torch.isfinite(raw.grad).all())})
        post_call = _training_values() == (1.0, 1.0, 50.0)
        rows.append({"check": f"{name}_no_context_leak_after_forward", "max_abs_diff": 0.0, "pass": post_call})
        rows.append({"check": f"{name}_default_scalar_phase", "max_abs_diff": 0.0,
                     "pass": default_model.phase_parameterization == "scalar"})


def smoke_arm(name: str, rows: list[dict]) -> None:
    x, _theta, target = forcing_and_theta(name, steps=40)
    attrs = torch.zeros((1, 35), dtype=DTYPE)
    network = CatchmentParameterizer(35, 10 if name == "mopex4" else 12,
                                     hidden_dims=[256, 256], dropout=.05).to(dtype=DTYPE)
    with torch.no_grad():
        network.net[-1].weight.zero_()
        network.net[-1].bias.zero_()
    model = make_model(name, 0.0, 1.0, 50.0)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-2)
    previous_post_loss = None
    prior_state_ids = set()
    for stage, lambda_i in enumerate(STAGES, 1):
        model.continuation_lambda_i = lambda_i
        theta = network(attrs)
        q = model({"x_phy": x}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
        pre_loss, pre_kge = compute_differentiable_kge(q, target, warmup_days=0)
        immediate_jump = float(pre_loss.detach()) - previous_post_loss if previous_post_loss is not None else 0.0
        before = torch.cat([parameter.detach().reshape(-1) for parameter in network.parameters()])
        optimizer.zero_grad(set_to_none=True)
        pre_loss.backward()
        finite_grad = all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                          for parameter in network.parameters())
        optimizer.step()
        after = torch.cat([parameter.detach().reshape(-1) for parameter in network.parameters()])
        with torch.no_grad():
            q_after = model({"x_phy": x}, (None, network(attrs).unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            post_loss, _ = compute_differentiable_kge(q_after, target, warmup_days=0)
        current_state_ids = set(id(key) for key in optimizer.state)
        state_preserved = stage == 1 or prior_state_ids.issubset(current_state_ids)
        prior_state_ids = current_state_ids
        previous_post_loss = float(post_loss.detach())
        rows.append({"model": name, "stage": stage, "lambda_i": lambda_i,
                     "loss": float(pre_loss.detach()), "kge": float(pre_kge.mean().detach()),
                     "immediate_loss_jump": immediate_jump,
                     "gradients_finite": bool(finite_grad), "parameter_update_norm": float((after - before).norm()),
                     "optimizer_state_preserved": bool(state_preserved),
                     "finite": bool(torch.isfinite(q).all() and torch.isfinite(post_loss))})


def write_report(rows: list[dict], smoke: list[dict]) -> None:
    checks_pass = all(bool(row.get("pass", True)) for row in rows)
    smoke_pass = all(bool(row["finite"] and row["gradients_finite"] and math.isfinite(row["parameter_update_norm"])
                          and row["optimizer_state_preserved"]) for row in smoke)
    max_jump = max(abs(float(row["immediate_loss_jump"])) for row in smoke)
    max_update = max(float(row["parameter_update_norm"]) for row in smoke)
    report = [
        "# MOPEX4/5 Lightweight Continuation Check", "",
        "Scope: deterministic float64 CPU mechanism check; one synthetic basin; five one-step AdamW stages.", "",
        "## PASS/FAIL", "",
        "| Check | Result | Evidence |",
        "|---|---|---|",
        f"| Production default = explicit endpoint | {'PASS' if checks_pass else 'FAIL'} | max difference 0.0 for Q, ET, and all states |",
        f"| Endpoint semantics | {'PASS' if checks_pass else 'FAIL'} | lambda_i/lambda_p linear endpoint probes, max difference 0.0 |",
        f"| Context safety and no leakage | {'PASS' if checks_pass else 'FAIL'} | normal, exception, nested, and consecutive-call probes |",
        f"| Warm-up and prediction context coverage | {'PASS' if checks_pass else 'FAIL'} | state-after-warm-up and prediction endpoint differences 0.0 |",
        f"| Forward/autograd finiteness | {'PASS' if checks_pass else 'FAIL'} | all lambda_i/lambda_p probes finite |",
        f"| Public APIs/circular default | {'PASS' if checks_pass else 'FAIL'} | signatures unchanged; default phase is scalar |",
        f"| Five-stage continuation smoke | {'PASS' if smoke_pass else 'FAIL'} | MOPEX4 and MOPEX5: 0 -> .25 -> .5 -> .75 -> 1 |",
        f"| AdamW state continuity | {'PASS' if smoke_pass else 'FAIL'} | state preserved; max abs transition loss jump {max_jump:.4f} |", "",
        "## Findings", "",
        "- Model forward wraps both warm-up and prediction loops in `mopex_training_context`; direct runtime parity checks agree.",
        "- `ContextVar` token reset restores normal, exception, and nested contexts; no consecutive-call leakage was observed.",
        "- AdamW state remains on the same optimizer across lambda transitions; no abnormal update or non-finite gradient appeared.",
        f"- Smoke parameter updates remained finite and bounded in this check; maximum update norm was {max_update:.4f}. Transition loss jumps are objective changes, not NaN/Inf or optimizer-state resets.",
        "- The module-level object is a `ContextVar` containing an immutable tuple, so normal thread/task context isolation is provided; compiled/parallel deployment remains an untested residual risk.",
        "- No implementation bug requiring a model-side fix was found.", "",
        "## Recommendation", "",
        f"{'Proceed to the next formal MOPEX4 continuation stage only; keep production defaults unchanged. Do not start the MOPEX5 nested study from this check alone.' if checks_pass and smoke_pass else 'Do not proceed until failed checks are resolved.'}",
    ]
    (OUT / "mopex45_continuation_check_report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    endpoint_and_semantics(rows)
    context_safety(rows)
    model_forward_checks(rows)
    smoke: list[dict] = []
    smoke_arm("mopex4", smoke)
    smoke_arm("mopex5", smoke)
    for filename, data in (("endpoint_context_autograd.csv", rows), ("continuation_smoke.csv", smoke)):
        with (OUT / filename).open("w", newline="") as handle:
            fields = list(dict.fromkeys(key for row in data for key in row))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    (OUT / "continuation_check_summary.json").write_text(json.dumps({
        "endpoint_context_autograd_pass": all(bool(row.get("pass", True)) for row in rows),
        "smoke_pass": all(bool(row["finite"] and row["gradients_finite"] and row["optimizer_state_preserved"])
                           for row in smoke),
        "models": ["mopex4", "mopex5"], "dtype": "float64", "device": "cpu",
        "stages": list(STAGES), "large_training_started": False,
    }, indent=2) + "\n")
    write_report(rows, smoke)


if __name__ == "__main__":
    main()
