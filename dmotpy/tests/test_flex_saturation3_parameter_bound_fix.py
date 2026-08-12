from __future__ import annotations

from functools import lru_cache
import math

import pytest

from models import core
from scripts.validate_flex_saturation3_bound_fix import TARGET_MODELS, TARGET_PARAM, main


@lru_cache(maxsize=1)
def _validation_rows():
    main()
    import csv
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "validation_results" / "flex_saturation3_parameter_bound_fix" / "flex_saturation3_bound_fix_validation.csv"
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("model_name", TARGET_MODELS)
def test_flex_beta_lower_bound_is_restored_to_zero(model_name):
    bounds = core.PARAM_INFO[model_name]
    assert bounds[TARGET_PARAM][0] == pytest.approx(0.0)
    assert bounds[TARGET_PARAM][1] == pytest.approx(10.0)


@pytest.mark.parametrize("model_name", TARGET_MODELS)
def test_flex_saturation3_validation_rows_are_finite_from_beta_zero_through_midpoint(model_name):
    rows = [row for row in _validation_rows() if row["model"] == model_name]
    assert rows, f"No validation rows found for {model_name}."
    assert all(row["pass_fail"] == "pass" for row in rows), (
        f"{model_name} should remain finite for the tested zero, near-zero, and midpoint beta probes after the stable rewrite."
    )

    for row in rows:
        tested_beta = float(row["tested_beta"])
        assert any(
            math.isclose(tested_beta, expected, rel_tol=0.0, abs_tol=1.0e-12)
            for expected in (0.0, 1.0e-12, 1.0e-9, 1.0e-6, 1.0e-5, 1.0e-4, 5.0)
        )
        assert int(row["output_nan_count"]) == 0
        assert int(row["output_inf_count"]) == 0
        assert int(row["output_bound_violation_count"]) == 0
        assert int(row["grad_nan_count"]) == 0
        assert int(row["grad_inf_count"]) == 0

    beta0_row = next(row for row in rows if math.isclose(float(row["tested_beta"]), 0.0, rel_tol=0.0, abs_tol=1.0e-15))
    beta1e6_row = next(row for row in rows if math.isclose(float(row["tested_beta"]), 1.0e-6, rel_tol=0.0, abs_tol=1.0e-15))
    assert float(beta0_row["output_diff_vs_beta0_if_available"]) == pytest.approx(0.0)
    assert float(beta1e6_row["output_diff_vs_beta0_if_available"]) == pytest.approx(0.0)
