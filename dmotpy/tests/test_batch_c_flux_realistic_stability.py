from __future__ import annotations

from functools import lru_cache

import pytest

from scripts.review_batch_c_flux_realistic_stability import BATCH_C_TARGETS, run_batch_c_review


@lru_cache(maxsize=1)
def _batch_c_artifacts():
    return run_batch_c_review()


TARGET_IDS = [f"{target.formula}-{target.active_model}" for target in BATCH_C_TARGETS]


@pytest.mark.parametrize("target_id", TARGET_IDS)
def test_batch_c_realistic_cases_are_finite_and_physically_safe(target_id):
    artifacts = _batch_c_artifacts()
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
        assert (
            row["output_bound_violation_count"] == 0
        ), f"{target_id}/{row['case_name']} violated its physically meaningful bound in the realistic-domain review."
