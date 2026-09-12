from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.unithydro_validation_utils import (
    MODEL_REGISTRY,
    evaluate_kernel_property_cases,
    evaluate_routing_cases,
    evaluate_weight_cases,
    extract_dmot_weights,
    get_parameter_cases,
    run_dmot_model,
)


def _assert_rows_pass(rows: list[dict]) -> None:
    failures = [row for row in rows if not row["pass_fail"]]
    if failures:
        details = "\n".join(
            (
                f"{row['dmot_function']} {row['mode']} {row['test_case']} "
                f"{row['parameter_case']} {row['dtype']} {row['device']} "
                f"max_abs={row['max_abs_error']:.3e} rel_l2={row['relative_l2_error']:.3e} "
                f"tol={row['tolerance']:.3e} cause={row['suspected_cause_if_failed'] or 'n/a'}"
            )
            for row in failures[:20]
        )
        pytest.fail(f"{len(failures)} unit-hydrograph validation cases failed:\n{details}")


@pytest.mark.parametrize("kind", list(MODEL_REGISTRY))
def test_weight_consistency_cpu_float64(kind: str) -> None:
    _assert_rows_pass(evaluate_weight_cases(kind, torch.float64, "cpu"))


@pytest.mark.parametrize("kind", list(MODEL_REGISTRY))
def test_routing_consistency_cpu_float64(kind: str) -> None:
    _assert_rows_pass(evaluate_routing_cases(kind, torch.float64, "cpu"))


@pytest.mark.parametrize("kind", list(MODEL_REGISTRY))
def test_routing_consistency_cpu_float32(kind: str) -> None:
    _assert_rows_pass(evaluate_routing_cases(kind, torch.float32, "cpu"))


@pytest.mark.parametrize("kind", list(MODEL_REGISTRY))
def test_kernel_properties_cpu_float64(kind: str) -> None:
    _assert_rows_pass(evaluate_kernel_property_cases(kind, torch.float64, "cpu"))


@pytest.mark.parametrize("kind", list(MODEL_REGISTRY))
def test_shifted_impulse_alignment_cpu_float64(kind: str) -> None:
    parameter_case = get_parameter_cases()[kind][2]
    params = np.asarray([parameter_case.params], dtype=np.float64)
    weights = extract_dmot_weights(kind, params, max_lag=32, dtype=torch.float64, device="cpu")[0]
    impulse = np.zeros((1, 40), dtype=np.float64)
    impulse[0, 2] = 1.0
    output = run_dmot_model(kind, impulse, params, max_lag=32, dtype=torch.float64, device="cpu")[0]

    expected = np.zeros_like(output)
    length = min(len(weights), len(output) - 2)
    expected[2 : 2 + length] = weights[:length]

    assert output.shape == impulse.shape[1:]
    assert np.allclose(output, expected, atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("kind", list(MODEL_REGISTRY))
def test_routing_consistency_cuda(kind: str) -> None:
    _assert_rows_pass(evaluate_routing_cases(kind, torch.float32, "cuda"))
    _assert_rows_pass(evaluate_weight_cases(kind, torch.float64, "cuda"))
