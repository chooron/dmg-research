from __future__ import annotations

import math

import pytest

from tests.training_regression_utils import PYTEST_SMOKE_CONFIG, run_calibration_case


TARGET_MODELS = ("flexb", "tcm", "hbv96")


@pytest.mark.parametrize("model_name", TARGET_MODELS)
def test_selected_models_support_short_training_smoke(model_name: str) -> None:
    row = run_calibration_case(model_name, PYTEST_SMOKE_CONFIG)
    assert row["status"] == "passed", f"{model_name} failed during {row['failed_stage']}: {row['notes']}"
    assert bool(row["optimizer_step_success"])
    assert int(row["loss_nan_count"]) == 0
    assert int(row["loss_inf_count"]) == 0
    assert int(row["grad_nan_count"]) == 0
    assert int(row["grad_inf_count"]) == 0
    assert int(row["output_nan_count"]) == 0
    assert int(row["output_inf_count"]) == 0
    assert math.isfinite(float(row["initial_loss"]))
    assert math.isfinite(float(row["final_loss"]))

