from __future__ import annotations

from functools import lru_cache
import math

import pytest
import torch

from scripts.review_focused_flux_formula_stability import (
    TARGET_CONTEXTS,
    build_realistic_inputs,
    capture_realistic_domain,
)


TARGET_IDS = [f"{target.formula}-{target.active_model}" for target in TARGET_CONTEXTS]


@lru_cache(maxsize=None)
def _captured_domain(formula: str, active_model: str):
    target = next(item for item in TARGET_CONTEXTS if item.formula == formula and item.active_model == active_model)
    return capture_realistic_domain(target)


def _evaluate_target(target, captured_domain):
    module = __import__(target.flux_module, fromlist=[target.formula])
    flux_fn = getattr(module, target.formula)
    arg_order = [name for name in flux_fn.__code__.co_varnames[: flux_fn.__code__.co_argcount] if name != "nearzero"]
    inputs = build_realistic_inputs(target, captured_domain, "mid")
    output = flux_fn(*[inputs[name] for name in arg_order], nearzero=1.0e-6)
    grads = torch.autograd.grad(
        output.sum(),
        [inputs[name] for name in target.active_grad_inputs if inputs[name].requires_grad],
        allow_unused=True,
        retain_graph=False,
    )
    return output, grads, inputs


@pytest.mark.parametrize("target", TARGET_CONTEXTS, ids=TARGET_IDS)
def test_target_fluxes_have_finite_realistic_outputs_and_gradients(target):
    torch.manual_seed(20260624)
    captured_domain = _captured_domain(target.formula, target.active_model)
    output, grads, inputs = _evaluate_target(target, captured_domain)

    assert not torch.isnan(output).any(), f"{target.formula}/{target.active_model} produced NaN output in realistic domain."
    assert not torch.isinf(output).any(), f"{target.formula}/{target.active_model} produced Inf output in realistic domain."

    for grad, grad_name in zip(grads, [name for name in target.active_grad_inputs if inputs[name].requires_grad]):
        assert grad is not None, f"{target.formula}/{target.active_model} missing gradient for {grad_name}."
        assert not torch.isnan(grad).any(), f"{target.formula}/{target.active_model} produced NaN gradient for {grad_name}."
        assert not torch.isinf(grad).any(), f"{target.formula}/{target.active_model} produced Inf gradient for {grad_name}."

    if target.output_storage_cap is not None:
        cap = inputs[target.output_storage_cap]
        assert not (output > cap + 1.0e-8).any(), f"{target.formula}/{target.active_model} exceeded storage cap in realistic domain."

    assert not (output < -1.0e-10).any(), f"{target.formula}/{target.active_model} produced unexpected negative output."


@pytest.mark.parametrize(
    ("target", "case_name"),
    [
        (target, case_name)
        for target in TARGET_CONTEXTS
        if target.formula in {"baseflow_6", "interflow_10"}
        for case_name in ("threshold_at", "threshold_plus")
    ],
    ids=[
        f"{target.formula}-{target.active_model}-{case_name}"
        for target in TARGET_CONTEXTS
        if target.formula in {"baseflow_6", "interflow_10"}
        for case_name in ("threshold_at", "threshold_plus")
    ],
)
def test_threshold_sensitive_targets_are_finite_near_activation_regions(target, case_name):
    captured_domain = _captured_domain(target.formula, target.active_model)
    inputs = build_realistic_inputs(target, captured_domain, case_name)
    module = __import__(target.flux_module, fromlist=[target.formula])
    flux_fn = getattr(module, target.formula)
    arg_order = [name for name in flux_fn.__code__.co_varnames[: flux_fn.__code__.co_argcount] if name != "nearzero"]
    output = flux_fn(*[inputs[name] for name in arg_order], nearzero=1.0e-6)
    assert torch.isfinite(output).all(), f"{target.formula}/{target.active_model}/{case_name} produced non-finite output."
    grad = torch.autograd.grad(output.sum(), inputs[target.probe_input], allow_unused=True)[0]
    assert grad is not None, f"{target.formula}/{target.active_model}/{case_name} missing probe gradient."
    assert torch.isfinite(grad).all(), f"{target.formula}/{target.active_model}/{case_name} produced non-finite probe gradient."
    assert not math.isnan(float(torch.abs(grad).max().item()))
