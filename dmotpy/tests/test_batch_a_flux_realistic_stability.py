from __future__ import annotations

from functools import lru_cache

import pytest

from scripts.review_batch_a_flux_realistic_stability import BATCH_A_TARGETS, run_batch_a_review


@lru_cache(maxsize=1)
def _batch_a_artifacts():
    return run_batch_a_review()


TARGET_IDS = [f"{target.formula}-{target.active_model}" for target in BATCH_A_TARGETS]


@pytest.mark.parametrize("target_id", TARGET_IDS)
def test_batch_a_realistic_cases_are_finite_and_bounded(target_id):
    artifacts = _batch_a_artifacts()
    gradient_rows = artifacts["gradient_rows"]
    formula, active_model = target_id.split("-", 1)
    realistic_rows = [
        row
        for row in gradient_rows
        if row["formula"] == formula and row["active_model"] == active_model and row["case_group"] == "realistic_domain"
    ]
    assert realistic_rows, f"No realistic-domain rows found for {target_id}."

    for row in realistic_rows:
        assert row["output_nan_count"] == 0, f"{target_id}/{row['case_name']} produced NaN output."
        assert row["output_inf_count"] == 0, f"{target_id}/{row['case_name']} produced Inf output."
        assert row["grad_nan_count"] == 0, f"{target_id}/{row['case_name']} produced NaN gradients."
        assert row["grad_inf_count"] == 0, f"{target_id}/{row['case_name']} produced Inf gradients."
        assert row["output_negative_count"] == 0, f"{target_id}/{row['case_name']} produced negative output."
        assert row["output_bound_violation_count"] == 0, f"{target_id}/{row['case_name']} exceeded the expected physical bound."


def test_batch_a_boundary_probes_are_finite_and_bounded_after_stable_rewrite():
    artifacts = _batch_a_artifacts()
    gradient_rows = artifacts["gradient_rows"]
    boundary_rows = [row for row in gradient_rows if row["case_group"] == "boundary_parameter_probe"]
    assert boundary_rows, "Expected boundary-parameter probe rows for Batch A."

    for row in boundary_rows:
        key = (row["formula"], row["active_model"])
        total_nonfinite_output = row["output_nan_count"] + row["output_inf_count"]
        total_nonfinite_grad = row["grad_nan_count"] + row["grad_inf_count"]
        assert total_nonfinite_output == 0, f"{key} unexpectedly produced non-finite boundary-probe outputs."
        assert total_nonfinite_grad == 0, f"{key} unexpectedly produced non-finite boundary-probe gradients."
        assert row["output_bound_violation_count"] == 0, f"{key} unexpectedly exceeded its physical bound in the boundary probe."
