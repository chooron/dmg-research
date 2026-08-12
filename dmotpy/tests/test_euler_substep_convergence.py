from __future__ import annotations

from functools import lru_cache

import pytest

from tests.euler_convergence_utils import (
    ERRORS_CSV_PATH,
    PASS_BAND,
    TARGET_MODELS,
    run_euler_convergence_validation,
)


@lru_cache(maxsize=1)
def _artifacts() -> dict:
    return run_euler_convergence_validation(write_outputs=True)


def test_euler_substep_convergence_has_no_nan_or_inf() -> None:
    artifacts = _artifacts()
    bad_rows = [
        row
        for row in artifacts["error_rows"]
        if int(row["output_nan_count"]) > 0
        or int(row["output_inf_count"]) > 0
        or int(row["state_nan_count"]) > 0
        or int(row["state_inf_count"]) > 0
    ]
    assert not bad_rows


def test_euler_substep_convergence_supported_models_pass_first_order_band() -> None:
    artifacts = _artifacts()
    supported_rows = [row for row in artifacts["summary_rows"] if row["model"] in TARGET_MODELS]
    failures = [
        row
        for row in supported_rows
        if not bool(row["state_convergence_pass"])
        or not bool(row["state_error_monotone"])
        or not (PASS_BAND[0] <= float(row["median_p_state"]) <= PASS_BAND[1])
    ]
    if failures:
        details = "\n".join(
            f"{row['model']} median_p_state={row['median_p_state']} "
            f"monotone={row['state_error_monotone']} classification={row['classification']} notes={row['notes']}"
            for row in failures
        )
        pytest.fail(f"Euler substep convergence failed for supported models:\n{details}")


def test_euler_substep_convergence_writes_csv_outputs() -> None:
    _artifacts()
    assert ERRORS_CSV_PATH.exists()
    assert ERRORS_CSV_PATH.stat().st_size > 0
