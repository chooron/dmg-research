from __future__ import annotations

import pytest
import torch

from tests.core_model_registry import CORE_MODEL_REGISTRY
from tests.core_water_balance_utils import evaluate_model


def _assert_rows_pass(rows: list[dict]) -> None:
    failures = [row for row in rows if not row["pass_fail"]]
    if failures:
        details = "\n".join(
            (
                f"{row['model_name']} {row['test_case']} {row['parameter_case']} {row['initial_state_case']} "
                f"{row['dtype']} {row['device']} full_abs={row['max_absolute_full_period_residual']:.3e} "
                f"full_rel={row['full_period_relative_residual']:.3e} step_abs={row['max_stepwise_residual']:.3e} "
                f"tol={row['tolerance']:.3e} cause={row['suspected_cause_if_failed'] or 'n/a'}"
            )
            for row in failures[:20]
        )
        pytest.fail(f"{len(failures)} core water-balance cases failed:\n{details}")


ENABLED_MODELS = [name for name, entry in CORE_MODEL_REGISTRY.items() if entry.enabled]


@pytest.mark.parametrize("model_name", ENABLED_MODELS)
def test_core_water_balance_cpu_float64(model_name: str) -> None:
    _assert_rows_pass(evaluate_model(CORE_MODEL_REGISTRY[model_name], torch.float64, "cpu", "pytest"))


@pytest.mark.parametrize("model_name", ENABLED_MODELS)
def test_core_water_balance_cpu_float32_smoke(model_name: str) -> None:
    _assert_rows_pass(evaluate_model(CORE_MODEL_REGISTRY[model_name], torch.float32, "cpu", "float32_smoke"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("model_name", ENABLED_MODELS)
def test_core_water_balance_cuda_float32_smoke(model_name: str) -> None:
    _assert_rows_pass(evaluate_model(CORE_MODEL_REGISTRY[model_name], torch.float32, "cuda", "float32_smoke"))
