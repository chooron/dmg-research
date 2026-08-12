from __future__ import annotations

import math

import torch

from tests.flux_gradient_wrappers import (
    build_wrapper_inputs,
    evaluate_wrapper,
    iter_all_flux_wrappers,
)


REPRESENTATIVE_PARAMETER_CASES = ("mid", "near_lower", "near_upper")
REPRESENTATIVE_STATE_CASES = ("mid", "nearzero")


def test_active_flux_wrappers_run_without_nonfinite_outputs_or_gradients() -> None:
    wrappers = [wrapper for wrapper in iter_all_flux_wrappers() if wrapper.context.active_usage_status != "unused"]
    assert wrappers, "No active flux wrappers were discovered."

    for wrapper in wrappers:
        for parameter_case in REPRESENTATIVE_PARAMETER_CASES:
            for state_case in REPRESENTATIVE_STATE_CASES:
                inputs = build_wrapper_inputs(
                    wrapper,
                    parameter_case=parameter_case,
                    state_case=state_case,
                    dtype=torch.float64,
                    device="cpu",
                    shape=(7,),
                )
                output = evaluate_wrapper(wrapper, inputs)
                assert not torch.isnan(output).any(), f"{wrapper.flux_info.function_name} produced NaN output"
                assert not torch.isinf(output).any(), f"{wrapper.flux_info.function_name} produced Inf output"

                scalar = output.sum()
                grads = torch.autograd.grad(
                    scalar,
                    [tensor for tensor in inputs.values() if tensor.requires_grad],
                    allow_unused=True,
                )
                for grad in grads:
                    if grad is None:
                        continue
                    assert not torch.isnan(grad).any(), f"{wrapper.flux_info.function_name} produced NaN gradient"
                    assert not torch.isinf(grad).any(), f"{wrapper.flux_info.function_name} produced Inf gradient"
                    if wrapper.threshold_inputs:
                        continue
                    assert float(torch.abs(grad).max().item()) < 1.0e6, (
                        f"{wrapper.flux_info.function_name} gradient exploded under representative range"
                    )
