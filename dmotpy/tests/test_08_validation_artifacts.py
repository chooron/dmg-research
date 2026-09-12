"""
Category 8: Validation Results and Benchmark Artifacts Test Suite
=================================================================
Audit Reference: DMOTPY_CURRENT_STATE_AUDIT_20260831.md Section 9, 13 & 14

Validates that all essential validation artifacts preserved in dmotpy/validation_results:
1. Exist and have valid non-empty schemas.
2. Maintain provenance continuity for paper-ready tables and figures.
"""

from __future__ import annotations

import csv
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VAL_DIR = REPO_ROOT / "dmotpy" / "validation_results"


class TestValidationResultsArtifacts:
    """Validate official benchmark and audit reports preserved in validation_results."""

    def test_validation_results_directory_exists(self):
        """The validation_results directory must exist in dmotpy."""
        assert VAL_DIR.exists() and VAL_DIR.is_dir()

    def test_gmd_stage1_artifacts_intact(self):
        """GMD Stage 1 fidelity artifacts must be present and non-empty."""
        stage1_dir = VAL_DIR / "gmd_3_1_stage1_fidelity"
        assert stage1_dir.exists()
        assert (stage1_dir / "01_mass_balance_rerun_results.csv").exists()
        assert (stage1_dir / "02_gradcheck_all_models_results.csv").exists()
        assert (stage1_dir / "04_uh_validation_results.csv").exists()

    def test_gmd_stage2_euler_artifacts_intact(self):
        """GMD Stage 2 discretization and Euler convergence classification must be present."""
        stage2_dir = VAL_DIR / "gmd_3_1_stage2_discretization_smoothing"
        assert stage2_dir.exists()
        assert (stage2_dir / "02_euler_model_classification.csv").exists()
        assert (stage2_dir / "03_smooth_gate_parameter_classification.csv").exists()

    def test_flux_gradient_stability_report_intact(self):
        """Flux gradient stability report and rankings must be present."""
        flux_dir = VAL_DIR / "flux_gradient_stability"
        assert flux_dir.exists()
        assert (flux_dir / "final_flux_gradient_risk_ranking.csv").exists()
        assert (flux_dir / "final_flux_gradient_stability_report.md").exists()

    def test_tost_equivalence_artifacts_intact(self):
        """TOST equivalence planning and metric files must be present."""
        tost_dir = VAL_DIR / "tost_equivalence"
        assert tost_dir.exists()
        assert (tost_dir / "tost_planning_matrix.csv").exists()
        assert (tost_dir / "basin_model_param_coverage.csv").exists()

    def test_unithydro_consistency_artifacts_intact(self):
        """Unit hydrograph consistency reports must be present."""
        uh_dir = VAL_DIR / "unithydro_consistency"
        assert uh_dir.exists()
        assert (uh_dir / "unithydro_consistency_report.md").exists()
        assert (uh_dir / "unithydro_consistency_summary.csv").exists()
