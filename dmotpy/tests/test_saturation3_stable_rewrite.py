from __future__ import annotations

from functools import lru_cache
import inspect
import math

from models.flux.saturation import saturation_3
from scripts.report_saturation3_stable_rewrite import (
    OUTPUT_DIR,
    TEST_BETAS,
    build_rows,
    main,
)


@lru_cache(maxsize=1)
def _artifact_rows():
    main()
    return build_rows()


def test_saturation3_uses_stable_sigmoid_without_internal_beta_clamp():
    source = inspect.getsource(saturation_3)
    assert "torch.sigmoid" in source
    assert "torch.exp(" not in source
    assert "torch.clamp" not in source


def test_saturation3_forward_equivalence_on_requested_float64_domains():
    forward_rows, _ = _artifact_rows()
    finite_rows = [row for row in forward_rows if row["old_output_finite"]]
    assert finite_rows, "Expected finite reference rows for saturation_3 forward equivalence."

    max_abs_diff = max(float(row["abs_diff"]) for row in finite_rows)
    old_sq = sum(float(row["old_output"]) ** 2 for row in finite_rows)
    diff_sq = sum(float(row["abs_diff"]) ** 2 for row in finite_rows)
    relative_l2 = math.sqrt(diff_sq / old_sq) if old_sq > 0.0 else 0.0

    assert max_abs_diff <= 1.0e-12
    assert relative_l2 <= 1.0e-12

    for row in finite_rows:
        assert row["new_output_finite"], f"Non-finite new output for {row['anchor']}"
        assert row["new_output_bounded"], f"Output escaped expected physical range for {row['anchor']}"


def test_saturation3_gradients_are_finite_on_requested_domains():
    _, gradient_rows = _artifact_rows()
    assert gradient_rows, "Expected gradient rows for saturation_3 stable rewrite."

    synthetic_rows = [row for row in gradient_rows if row["dataset"] == "synthetic_grid"]
    realistic_rows = [row for row in gradient_rows if row["dataset"] == "realistic_trace_anchor"]

    for row in synthetic_rows + realistic_rows:
        assert row["new_output_finite"], f"New output is non-finite for {row['anchor']}"
        assert row["new_grad_S_finite"], f"d/dS is non-finite for {row['anchor']}"
        assert row["new_grad_beta_finite"], f"d/dbeta is non-finite for {row['anchor']}"
        assert row["new_output_bounded"], f"Output escaped expected physical range for {row['anchor']}"

    requested_beta_strings = {f"{beta:.12g}" for beta in TEST_BETAS}
    seen_beta_strings = {row["beta"] for row in synthetic_rows}
    assert requested_beta_strings.issubset(seen_beta_strings)


def test_saturation3_report_artifacts_are_written():
    _artifact_rows()
    expected = (
        OUTPUT_DIR / "saturation3_forward_equivalence.csv",
        OUTPUT_DIR / "saturation3_gradient_stability.csv",
        OUTPUT_DIR / "saturation3_stable_rewrite_report.md",
    )
    for path in expected:
        assert path.exists(), f"Expected artifact {path} to exist."
        assert path.stat().st_size > 0
