"""
QC test for pymarrmot crosscheck artifacts.

Verifies completeness and integrity of the crosscheck output files.
Does NOT validate model-level numerical results (those are documented in the reports).
"""

import csv
import os
from pathlib import Path

import pytest

ARTIFACT_DIR = Path(__file__).parent.parent / 'validation_results' / 'pymarrmot_crosscheck'

REQUIRED_FILES = [
    'forcing_diagnostics.md',
    'model_mapping.csv',
    'model_mapping.md',
    'parameter_quantiles.csv',
    'official_pymarrmot_matlab_exceptions.csv',
    'crosscheck_summary.csv',
    'crosscheck_summary.md',
    'model_level_crosscheck_matrix.csv',
    'model_level_crosscheck_matrix.md',
    'pymarrmot_crosscheck_report.md',
]

# The 36 dMoT core models (from dmotpy.models.core.PARAM_INFO)
EXPECTED_MODELS = [
    "alpine1", "alpine2", "australia", "collie1", "collie2", "collie3",
    "flexb", "flexi", "flexis", "gr4j", "gsfb", "hbv96",
    "hillslope", "hymod", "ihacres", "modhydrolog",
    "mopex1", "mopex2", "mopex3", "mopex4", "mopex5",
    "newzealand1", "newzealand2", "penman", "plateau", "simhyd",
    "smar", "susannah1", "susannah2", "tank", "tcm", "topmodel",
    "us1", "vic", "wetland", "xinanjiang",
]


class TestPymarrmotCrosscheckArtifacts:
    """Test the integrity of pymarrmot crosscheck artifacts."""

    @pytest.mark.parametrize("filename", REQUIRED_FILES)
    def test_artifact_file_exists(self, filename):
        """Each required artifact file must exist."""
        path = ARTIFACT_DIR / filename
        assert path.exists(), f"Missing artifact: {filename}"
        assert os.path.getsize(path) > 0, f"Empty artifact: {filename}"

    def test_model_mapping_contains_all_36_models(self):
        """model_mapping.csv must contain all 36 dMoT core models."""
        path = ARTIFACT_DIR / 'model_mapping.csv'
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        dmot_models_in_csv = [r['dmot_model'] for r in rows]
        for model in EXPECTED_MODELS:
            assert model in dmot_models_in_csv, (
                f"Model '{model}' missing from model_mapping.csv"
            )

        # All should be mapped
        for row in rows:
            assert row['mapping_status'] in ('mapped', 'ambiguous',
                                              'reference_not_available')

    def test_crosscheck_summary_not_empty(self):
        """crosscheck_summary.csv must have data."""
        path = ARTIFACT_DIR / 'crosscheck_summary.csv'
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0, "crosscheck_summary.csv is empty"

        # Each mapped model should have at least p50 and p05
        models_tested = {}
        for row in rows:
            m = row['model']
            q = row['quantile']
            if m not in models_tested:
                models_tested[m] = set()
            models_tested[m].add(q)

        for model in EXPECTED_MODELS:
            if model in models_tested:
                assert 'p50' in models_tested[model], (
                    f"Model '{model}' missing p50 quantile in summary"
                )
                assert 'p05' in models_tested[model], (
                    f"Model '{model}' missing p05 quantile in summary"
                )

    def test_run_failures_have_notes(self):
        """All run failures must have notes."""
        path = ARTIFACT_DIR / 'crosscheck_summary.csv'
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            if row['comparison_status'] == 'RUN_FAILED' or row['run_status'] != 'success':
                assert row.get('notes', '').strip(), (
                    f"Run failure for {row['model']}/{row['quantile']} "
                    f"has no notes"
                )

    def test_mismatch_review_have_notes(self):
        """All MISMATCH_REVIEW entries must have notes."""
        path = ARTIFACT_DIR / 'crosscheck_summary.csv'
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            if row['comparison_status'] == 'MISMATCH_REVIEW':
                assert row.get('notes', '').strip(), (
                    f"MISMATCH_REVIEW for {row['model']}/{row['quantile']} "
                    f"has no notes"
                )

    def test_model_matrix_contains_36_models(self):
        """model_level_crosscheck_matrix.csv must contain 36 models."""
        path = ARTIFACT_DIR / 'model_level_crosscheck_matrix.csv'
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        matrix_models = [r['model'] for r in rows]
        for model in EXPECTED_MODELS:
            assert model in matrix_models, (
                f"Model '{model}' missing from model_level_crosscheck_matrix.csv"
            )

        assert len(rows) == 36, (
            f"Expected 36 models in matrix, got {len(rows)}"
        )

    def test_parameter_quantiles_have_correct_structure(self):
        """parameter_quantiles.csv must have the right structure."""
        path = ARTIFACT_DIR / 'parameter_quantiles.csv'
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0, "parameter_quantiles.csv is empty"

        required_cols = {'model', 'quantile', 'param_name', 'lower', 'upper',
                         'value', 'source'}
        actual_cols = set(rows[0].keys())
        assert required_cols.issubset(actual_cols), (
            f"Missing columns: {required_cols - actual_cols}"
        )

    def test_official_exceptions_csv_exists(self):
        """official exception list must exist (can be empty but file should be there)."""
        path = ARTIFACT_DIR / 'official_pymarrmot_matlab_exceptions.csv'
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Can be empty or have entries - both are valid
        # Just verify the structure is valid CSV
        assert reader.fieldnames is not None

    def test_official_exception_models_marked_correctly(self):
        """Models in official exception list must be correctly marked."""
        path = ARTIFACT_DIR / 'crosscheck_summary.csv'
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Models that should have official exceptions
        expected_exc_models = {
            'gr4j', 'us1', 'newzealand2', 'gsfb', 'tcm', 'hbv96'
        }

        for row in rows:
            if row['model'] in expected_exc_models:
                assert row['in_official_exception_list'] == 'yes', (
                    f"Model {row['model']} ({row['quantile']}) should be "
                    f"in official exception list but isn't"
                )
                assert row.get('official_exception_note', '').strip(), (
                    f"Model {row['model']} has no exception note"
                )

    def test_no_models_without_mapping_status(self):
        """Every model in summary should have a mapping_status."""
        path = ARTIFACT_DIR / 'crosscheck_summary.csv'
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            assert row.get('mapping_status', ''), (
                f"Missing mapping_status for {row['model']}/{row['quantile']}"
            )

    def test_mismatch_review_has_subtype(self):
        """Every MISMATCH_REVIEW entry must have a mismatch_subtype."""
        path = ARTIFACT_DIR / 'crosscheck_summary.csv'
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            if row['comparison_status'] == 'MISMATCH_REVIEW':
                subtype = row.get('mismatch_subtype', '')
                assert subtype, (
                    f"MISMATCH_REVIEW for {row['model']}/{row['quantile']} "
                    f"missing mismatch_subtype"
                )
                assert row.get('notes', '').strip(), (
                    f"MISMATCH_REVIEW for {row['model']}/{row['quantile']} "
                    f"has no notes"
                )

    def test_low_correlation_review_has_detailed_notes(self):
        """LOW_CORRELATION_REVIEW entries must have detailed notes."""
        path = ARTIFACT_DIR / 'crosscheck_summary.csv'
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            if row.get('mismatch_subtype') == 'LOW_CORRELATION_REVIEW':
                notes = row.get('notes', '')
                assert 'review' in notes.lower() or 'manual' in notes.lower() or \
                       'investigation' in notes.lower() or 'discrepancy' in notes.lower(), (
                    f"LOW_CORRELATION_REVIEW for {row['model']}/{row['quantile']} "
                    f"should have detailed notes"
                )

    def test_state_info_audit_exists(self):
        """state_info_audit.csv must exist."""
        path = ARTIFACT_DIR / 'state_info_audit.csv'
        assert path.exists(), "state_info_audit.csv not found"
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 36, f"Expected 36 models in audit, got {len(rows)}"

    def test_state_info_mismatches_are_zero(self):
        """After fixes, state_info_audit should show zero mismatches."""
        path = ARTIFACT_DIR / 'state_info_audit.csv'
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        mismatches = [r for r in rows if r['status'] == 'mismatch']
        assert len(mismatches) == 0, (
            f"STATE_INFO still has {len(mismatches)} mismatches: "
            f"{[r['model'] for r in mismatches]}"
        )
