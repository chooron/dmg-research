"""Validate UH-inclusive TOST artifacts exist and are consistent.

Note: The paired TOST (dMoT vs pymarrmot) was not executed due to critical
pymarrmot issues (UH stubs, flux group bug, get_output crash). This test
validates that dMoT-side readiness artifacts exist and the report accurately
documents the blockers.
"""

from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "validation_results"
TOST_UH = BASE / "tost_equivalence_uh_inclusive"
TOST_ORIG = BASE / "tost_equivalence"


def test_uh_inclusive_design_files_exist():
    assert (TOST_UH / "tost_uh_inclusive_design_lock.md").exists()


def test_old_uh_disabled_design_unchanged():
    assert (TOST_ORIG / "tost_design_lock.md").exists()
    assert (TOST_ORIG / "tost_design.yaml").exists()


def test_dmot_cache_manifest_exists():
    assert (TOST_UH / "dmot_cache").is_dir()
    cache_files = list((TOST_UH / "dmot_cache").glob("*_eval_outputs.npz"))
    assert len(cache_files) >= 36, f"Expected >=36 cached models, got {len(cache_files)}"


def test_alignment_check_exists():
    assert (TOST_UH / "pymarrmot_uh_alignment_check.json").exists()


def test_results_directory():
    assert (TOST_UH / "results").is_dir()
    assert (TOST_UH / "results" / "tost_uh_basin_level_metrics.csv").exists()
    assert (TOST_UH / "results" / "tost_uh_model_metric_summary.csv").exists()


def test_report_documents_blockers():
    report = TOST_UH / "results" / "tost_uh_equivalence_report.md"
    assert report.exists()
    text = report.read_text()
    # Must state it's supplementary, not a replacement
    assert "not a replacement" in text.lower() or "supplement" in text.lower()
    # Must document pymarrmot issues
    assert "pymarrmot" in text.lower()
    assert "stub" in text.lower() or "flux_group" in text.lower() or "identity" in text.lower()


def test_report_separates_readiness_from_tost():
    report = TOST_UH / "results" / "tost_uh_equivalence_report.md"
    text = report.read_text()
    assert "readiness" in text.lower() or "readiness assessment" in text.lower() or "dmot readiness" in text.lower()
    assert "paired tost" in text.lower() or "not executed" in text.lower() or "blocked" in text.lower()


def test_ihacres_param_mismatch_documented():
    for f in [TOST_UH / "pymarrmot_uh_alignment_check.json",
              TOST_UH / "results" / "tost_uh_equivalence_report.md"]:
        if f.exists():
            text = f.read_text() if f.suffix == '.md' else open(f).read()
            if "ihacres" in text.lower() and ("mismatch" in text.lower() or "PARAM_DIMENSION" in text):
                return
    raise AssertionError("ihacres parameter dimension mismatch not documented")
