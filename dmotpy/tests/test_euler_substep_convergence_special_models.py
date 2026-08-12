from __future__ import annotations

from functools import lru_cache

import pytest

from tests.euler_convergence_special_models_utils import (
    CAVEAT_MODELS,
    PERMANENTLY_EXCLUDED_MODELS,
    SPECIAL_ERRORS_CSV_PATH,
    SPECIAL_ORDERS_CSV_PATH,
    SPECIAL_SUMMARY_CSV_PATH,
    SPECIAL_TARGET_MODELS,
    PASS_BAND,
    run_euler_convergence_validation_special_models,
)


@lru_cache(maxsize=1)
def _artifacts() -> dict:
    return run_euler_convergence_validation_special_models(write_outputs=True)


def test_euler_special_models_no_nan_or_inf() -> None:
    """No NaN or Inf values should appear in any state or flux output during
    substepping across all 7 special target models."""
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


def test_euler_special_models_smooth_models_pass_first_order_band() -> None:
    """Models classified substep_supported (no caveat) must have state errors
    monotone-decreasing and median empirical order within PASS_BAND = (0.85, 1.15).

    Smooth special models (no caveat): topmodel only.
    gr4j uses closed-form analytical equations (not Euler ODE); gsfb/tcm have
    threshold-based saturation functions — all three are in CAVEAT_MODELS.
    """
    artifacts = _artifacts()
    smooth_models = set(SPECIAL_TARGET_MODELS) - CAVEAT_MODELS
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
            f"Euler substep first-order convergence failed for smooth-supported special models:\n{details}"
        )


def test_euler_special_models_caveat_models_run_without_crash() -> None:
    """Models marked substep_supported_with_caveat must run to completion without
    exceptions; we do not enforce a strict first-order convergence band for them.

    Caveat special models: mopex4, mopex5 (snow threshold), tank (multi-threshold).
    """
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
            f"Caveat special models crashed during substep simulation:\n{details}"
        )


def test_euler_special_models_excluded_not_in_results() -> None:
    """shm (permanently excluded: empty file) must not appear in any results."""
    artifacts = _artifacts()
    included_in_errors = {row["model"] for row in artifacts["error_rows"]} & PERMANENTLY_EXCLUDED_MODELS
    included_in_summary = {row["model"] for row in artifacts["summary_rows"]} & PERMANENTLY_EXCLUDED_MODELS
    all_unexpected = included_in_errors | included_in_summary
    assert not all_unexpected, (
        f"Permanently excluded models appeared in convergence results: {sorted(all_unexpected)}"
    )


def test_euler_special_models_writes_csv_outputs() -> None:
    """Validation must write all three CSV output files with non-empty content."""
    _artifacts()
    for path in (SPECIAL_ERRORS_CSV_PATH, SPECIAL_ORDERS_CSV_PATH, SPECIAL_SUMMARY_CSV_PATH):
        assert path.exists(), f"Expected CSV output not found: {path}"
        assert path.stat().st_size > 0, f"CSV output is empty: {path}"


def test_euler_special_models_summary_covers_all_target_models() -> None:
    """Summary must contain exactly one row per target model."""
    artifacts = _artifacts()
    summary_models = {row["model"] for row in artifacts["summary_rows"]}
    missing = set(SPECIAL_TARGET_MODELS) - summary_models
    extra = summary_models - set(SPECIAL_TARGET_MODELS)
    assert not missing, f"Missing models in summary: {sorted(missing)}"
    assert not extra, f"Unexpected extra models in summary: {sorted(extra)}"


def test_euler_special_models_topmodel_deficit_state_sign_corrected() -> None:
    """topmodel S2 is a deficit store (sign override -1). Confirm topmodel appears
    in summary with a non-nan median_p_state, verifying the sign correction was applied
    (a wrong sign would produce diverging errors and NaN convergence orders)."""
    artifacts = _artifacts()
    topmodel_rows = [r for r in artifacts["summary_rows"] if r["model"] == "topmodel"]
    assert topmodel_rows, "topmodel missing from summary"
    row = topmodel_rows[0]
    assert row["classification"] != "fail_unexpected", (
        f"topmodel simulation failed unexpectedly: {row['notes']}"
    )
    import math
    assert math.isfinite(float(row["median_p_state"])), (
        f"topmodel median_p_state is not finite: {row['median_p_state']} — "
        "deficit-state sign correction may not be applied correctly."
    )


def test_euler_special_models_mopex_doy_kwarg_supplied() -> None:
    """mopex4 and mopex5 require a keyword-only `doy` argument. Confirm both
    models appear in summary without a TypeError crash (verifying the doy kwarg
    wrapper is correctly wired)."""
    artifacts = _artifacts()
    for model in ("mopex4", "mopex5"):
        rows = [r for r in artifacts["summary_rows"] if r["model"] == model]
        assert rows, f"{model} missing from summary"
        row = rows[0]
        assert row["classification"] != "fail_unexpected", (
            f"{model} simulation raised an unexpected error (possibly missing doy kwarg): {row['notes']}"
        )


def test_euler_special_models_tcm_mean_p_kwarg_supplied() -> None:
    """tcm requires a keyword-only `mean_P` argument. Confirm tcm appears in
    summary without a TypeError crash (verifying the mean_P kwarg wrapper is
    correctly wired)."""
    artifacts = _artifacts()
    rows = [r for r in artifacts["summary_rows"] if r["model"] == "tcm"]
    assert rows, "tcm missing from summary"
    row = rows[0]
    assert row["classification"] != "fail_unexpected", (
        f"tcm simulation raised an unexpected error (possibly missing mean_P kwarg): {row['notes']}"
    )
