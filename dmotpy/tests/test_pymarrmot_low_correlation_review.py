"""
QC test for low_correlation_review debug artifacts.
Validates completeness of targeted review per Round 3.
"""
import csv
import json
from pathlib import Path

import pytest

ARTIFACT_DIR = Path(__file__).parent.parent / 'validation_results' / 'pymarrmot_crosscheck'
REVIEW_DIR = ARTIFACT_DIR / 'low_correlation_review'

# Expected 7 cases (from Round 2 LOW_CORRELATION_REVIEW)
EXPECTED_CASES = [
    ('australia', 'p50'),
    ('collie1', 'p50'),
    ('modhydrolog', 'p50'),
    ('penman', 'p50'),
    ('plateau', 'p50'),
    ('susannah2', 'p50'),
    ('susannah2', 'p05'),
]


class TestLowCorrelationReview:
    """Validate Round 3 low-correlation review artifacts."""

    @pytest.mark.parametrize("model,quantile", EXPECTED_CASES)
    def test_timeseries_csv_exists_and_nonempty(self, model, quantile):
        fname = REVIEW_DIR / f'{model}_{quantile}_timeseries.csv'
        assert fname.exists(), f"Missing: {fname}"
        with open(fname) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 365, (
            f"{fname}: expected 365 days, got {len(rows)} rows"
        )

    @pytest.mark.parametrize("model,quantile", EXPECTED_CASES)
    def test_timeseries_no_nan_inf_in_q(self, model, quantile):
        fname = REVIEW_DIR / f'{model}_{quantile}_timeseries.csv'
        with open(fname) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in ['Q_dmot', 'Q_pymarrmot']:
                    val = float(row[col])
                    assert not (val != val), f"NaN in {fname} {col} at day {row['day']}"
                    # Inf check (unlikely but guard)
                    assert abs(val) < 1e100, f"Inf in {fname} {col} at day {row['day']}"

    @pytest.mark.parametrize("model,quantile", EXPECTED_CASES)
    def test_debug_md_exists(self, model, quantile):
        fname = REVIEW_DIR / f'{model}_{quantile}_debug.md'
        assert fname.exists(), f"Missing: {fname}"
        assert fname.stat().st_size > 0, f"Empty: {fname}"

    def test_all_results_json_exists(self):
        fname = REVIEW_DIR / '_all_results.json'
        assert fname.exists(), "Missing: _all_results.json"
        with open(fname) as f:
            data = json.load(f)
        assert len(data) == 7, f"Expected 7 results, got {len(data)}"

    def test_no_low_correlation_review_remaining_in_summary(self):
        """After Round 3, crosscheck_summary.csv must have zero LOW_CORRELATION_REVIEW."""
        fname = ARTIFACT_DIR / 'crosscheck_summary.csv'
        with open(fname) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        low_corr = [r for r in rows
                    if r.get('mismatch_subtype') == 'LOW_CORRELATION_REVIEW']
        assert len(low_corr) == 0, (
            f"Still have {len(low_corr)} LOW_CORRELATION_REVIEW entries: "
            f"{[(r['model'], r['quantile']) for r in low_corr]}"
        )

    def test_original_7_cases_have_revised_subtype(self):
        """All 7 original cases must have revised subtype (not LOW_CORRELATION_REVIEW)."""
        fname = ARTIFACT_DIR / 'crosscheck_summary.csv'
        with open(fname) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for model, quantile in EXPECTED_CASES:
            found = [r for r in rows
                     if r['model'] == model and r['quantile'] == quantile]
            assert len(found) == 1, f"Missing: {model}/{quantile}"
            row = found[0]
            subtype = row.get('mismatch_subtype', '')
            assert subtype != 'LOW_CORRELATION_REVIEW', (
                f"{model}/{quantile} still has LOW_CORRELATION_REVIEW"
            )
            assert subtype, f"{model}/{quantile} has no mismatch_subtype"
            assert row['comparison_status'] == 'MATCH_CAVEAT', (
                f"{model}/{quantile} status={row['comparison_status']}, "
                f"expected MATCH_CAVEAT"
            )

    def test_near_zero_variance_not_in_issue_list(self):
        """NEAR_CONSTANT_OR_ZERO_VARIANCE cases must NOT appear as blocking issues."""
        fname = ARTIFACT_DIR / 'pymarrmot_crosscheck_report.md'
        with open(fname) as f:
            content = f.read()

        # NEAR_CONSTANT cases should appear in the "resolved" section, not as "must-fix"
        assert 'NEAR_CONSTANT_OR_ZERO_VARIANCE' in content, (
            "Report should mention NEAR_CONSTANT cases in resolution section"
        )
        # The report should NOT list them as must-fix
        assert 'No dMoT model formula bugs identified' in content or \
               'No dMoT model formula bug' in content
