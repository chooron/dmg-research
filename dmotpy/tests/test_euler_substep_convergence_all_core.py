from __future__ import annotations

from functools import lru_cache

import pytest

from tests.euler_convergence_all_core_utils import (
    ALL_CORE_ERRORS_CSV_PATH,
    ALL_CORE_ORDERS_CSV_PATH,
    ALL_CORE_SUMMARY_CSV_PATH,
    ALL_CORE_TARGET_MODELS,
    CAVEAT_MODELS,
    EXCLUDED_MODELS,
    PASS_BAND,
    run_euler_convergence_validation_all_core,
)


@lru_cache(maxsize=1)
def _artifacts() -> dict:
    return run_euler_convergence_validation_all_core(write_outputs=True)


def test_euler_all_core_no_nan_or_inf() -> None:
    """No NaN or Inf values should appear in any state or flux output during
    substepping across all target models."""
    artifacts = _artifacts()
    bad_rows = [
        row
        for row in artifacts["error_rows"]
        if int(row["output_nan_count"]) > 0
        or int(row["output_inf_count"]) > 0
        or int(row["state_nan_count"]) > 0
        or int(row["state_inf_count"]) > 0
    ]
    if bad_rows:
        details = "\n".join(
            f"{row['model']} k={row['n_substeps']} output_nan={row['output_nan_count']} "
            f"output_inf={row['output_inf_count']} state_nan={row['state_nan_count']} "
            f"state_inf={row['state_inf_count']}"
            for row in bad_rows
        )
        pytest.fail(f"NaN/Inf detected in substep simulation outputs:\n{details}")


def test_euler_all_core_smooth_models_pass_first_order_band() -> None:
    """Models classified substep_supported (no caveat) must have state errors
    monotone-decreasing and median empirical order within PASS_BAND = (0.85, 1.15)."""
    artifacts = _artifacts()
    smooth_models = set(ALL_CORE_TARGET_MODELS) - CAVEAT_MODELS
    failures = [
        row
        for row in artifacts["summary_rows"]
        if row["model"] in smooth_models
        and (
            not bool(row["state_convergence_pass"])
            or not bool(row["state_error_monotone"])
            or not (PASS_BAND[0] <= float(row["median_p_state"]) <= PASS_BAND[1])
        )
    ]
    if failures:
        details = "\n".join(
            f"{row['model']} median_p_state={row['median_p_state']} "
            f"monotone={row['state_error_monotone']} classification={row['classification']} "
            f"notes={row['notes']}"
            for row in failures
        )
        pytest.fail(
            f"Euler substep first-order convergence failed for smooth-supported models:\n{details}"
        )


def test_euler_all_core_caveat_models_run_without_crash() -> None:
    """Models marked substep_supported_with_caveat must run to completion without
    exceptions; we do not enforce a strict first-order convergence band for them."""
    artifacts = _artifacts()
    caveat_failures = [
        row
        for row in artifacts["summary_rows"]
        if row["model"] in CAVEAT_MODELS
        and row["classification"] == "fail_unexpected"
    ]
    if caveat_failures:
        details = "\n".join(
            f"{row['model']} classification={row['classification']} notes={row['notes']}"
            for row in caveat_failures
        )
        pytest.fail(
            f"Caveat models crashed during substep simulation:\n{details}"
        )


def test_euler_all_core_excluded_models_not_in_results() -> None:
    """Excluded models must not appear in any error or summary rows."""
    artifacts = _artifacts()
    included_in_errors = {row["model"] for row in artifacts["error_rows"]} & EXCLUDED_MODELS
    included_in_summary = {row["model"] for row in artifacts["summary_rows"]} & EXCLUDED_MODELS
    all_unexpected = included_in_errors | included_in_summary
    assert not all_unexpected, (
        f"Excluded models appeared in convergence results: {sorted(all_unexpected)}"
    )


def test_euler_all_core_writes_csv_outputs() -> None:
    """Validation must write all three CSV output files with non-empty content."""
    _artifacts()
    for path in (ALL_CORE_ERRORS_CSV_PATH, ALL_CORE_ORDERS_CSV_PATH, ALL_CORE_SUMMARY_CSV_PATH):
        assert path.exists(), f"Expected CSV output not found: {path}"
        assert path.stat().st_size > 0, f"CSV output is empty: {path}"


def test_euler_all_core_summary_covers_all_target_models() -> None:
    """Summary must contain exactly one row per target model."""
    artifacts = _artifacts()
    summary_models = {row["model"] for row in artifacts["summary_rows"]}
    missing = set(ALL_CORE_TARGET_MODELS) - summary_models
    extra = summary_models - set(ALL_CORE_TARGET_MODELS)
    assert not missing, f"Missing models in summary: {sorted(missing)}"
    assert not extra, f"Unexpected extra models in summary: {sorted(extra)}"
