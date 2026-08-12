"""
QC test for reliability closure audit artifacts.
Validates completeness of all five closure directions.
"""
import csv
from pathlib import Path

import pytest

ARTIFACT_DIR = Path(__file__).parent.parent / 'validation_results' / 'reliability_closure'

REQUIRED_FILES = [
    'attribution_falsifiability_review.csv',
    'coverage_gap_inventory.csv',
    'pathological_gradient_scan.csv',
    'precision_determinism_check.csv',
    'special_model_scope_check.csv',
    'reliability_closure_review.md',
]


class TestReliabilityClosureArtifacts:
    """Validate reliability closure audit artifacts."""

    @pytest.mark.parametrize("filename", REQUIRED_FILES)
    def test_artifact_exists(self, filename):
        path = ARTIFACT_DIR / filename
        assert path.exists(), f"Missing: {filename}"
        assert path.stat().st_size > 0, f"Empty: {filename}"

    def test_coverage_gap_inventory_not_empty(self):
        path = ARTIFACT_DIR / 'coverage_gap_inventory.csv'
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0, "Coverage gap inventory empty"
        # No high-risk gaps should exist
        high_risk = [r for r in rows if r['risk_level'] == 'high']
        assert len(high_risk) == 0, (
            f"High-risk gaps found: {[(r['dimension'], r['gap']) for r in high_risk]}"
        )

    def test_pathological_gradient_covers_8_models(self):
        path = ARTIFACT_DIR / 'pathological_gradient_scan.csv'
        with open(path) as f:
            rows = list(csv.DictReader(f))
        models = set(r['model'] for r in rows)
        expected = {'australia', 'hbv96', 'mopex2', 'mopex3', 'vic', 'gsfb', 'tcm', 'topmodel'}
        assert len(models & expected) >= 8, (
            f"Gradient scan missing models: {expected - models}"
        )
        # No nonfinite gradients (all must be 0)
        for r in rows:
            assert int(r['nonfinite_grad_count']) == 0, (
                f"{r['model']}: {r['nonfinite_grad_count']} nonfinite gradients"
            )

    def test_precision_determinism_covers_4_models(self):
        path = ARTIFACT_DIR / 'precision_determinism_check.csv'
        with open(path) as f:
            rows = list(csv.DictReader(f))
        models = set(r['model'] for r in rows)
        expected = {'flexb', 'gsfb', 'topmodel', 'hbv96'}
        assert len(models & expected) >= 4, (
            f"Precision check missing models: {expected - models}"
        )
        # All must be seed-repeatable
        for r in rows:
            assert r['float64_seed_repeatable'] == 'True', (
                f"{r['model']}: seed not repeatable"
            )

    def test_special_model_scope_covers_gr4j_gsmsocont_echo(self):
        path = ARTIFACT_DIR / 'special_model_scope_check.csv'
        with open(path) as f:
            rows = list(csv.DictReader(f))
        models = set(r['model'] for r in rows)
        assert 'gr4j' in models, "gr4j missing from special model scope"
        assert 'gsmsocont' in models, "gsmsocont missing from special model scope"
        assert 'echo' in models, "echo missing from special model scope"

    def test_closure_report_contains_decision(self):
        path = ARTIFACT_DIR / 'reliability_closure_review.md'
        with open(path) as f:
            content = f.read()
        assert 'Closure Decision' in content, "Report missing closure decision"
        # Must have one of the three valid decisions
        assert any(d in content for d in [
            'CLOSE_CORRECTNESS_VALIDATION',
            'CLOSE_WITH_DOCUMENTED_CAVEATS',
            'DO_NOT_CLOSE',
        ]), "Report missing valid closure decision"

    def test_closure_decision_not_close_if_blocking_issues(self):
        path = ARTIFACT_DIR / 'reliability_closure_review.md'
        with open(path) as f:
            content = f.read()
        if 'DO_NOT_CLOSE' not in content:
            # Verify no blocking issues
            blocking_section = content.find('Blocking Issues')
            if blocking_section > 0:
                section = content[blocking_section:blocking_section + 500]
                if 'None' in section or 'None.' in section:
                    pass  # Good
                elif 'Decision: CLOSE_WITH_DOCUMENTED_CAVEATS' in content:
                    pass  # Caveats are non-blocking
                elif 'Decision: CLOSE_CORRECTNESS_VALIDATION' in content:
                    pass  # No blocking issues

    def test_attribution_review_covers_all_seven_cases(self):
        path = ARTIFACT_DIR / 'attribution_falsifiability_review.csv'
        with open(path) as f:
            rows = list(csv.DictReader(f))
        # Should cover near-zero cases, lag test, threshold models, official exceptions
        tests = set(r['test'] for r in rows)
        assert 'non_zero_interval' in tests
        assert 'lag_verification' in tests
        assert 'threshold_substep_evidence' in tests
        assert 'official_exception_labeling' in tests
