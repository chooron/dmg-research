"""Test: robust comparison artifacts."""
from pathlib import Path
import pandas as pd
import pytest

ROBUST_DIR = Path(__file__).parent.parent / 'validation_results' / 'tost_equivalence' / 'robust_comparison'


def test_robust_dir_exists():
    assert ROBUST_DIR.is_dir()


def test_model_csv_exists():
    assert (ROBUST_DIR / 'robust_kge_comparison_by_model.csv').exists()


def test_basin_csv_exists():
    assert (ROBUST_DIR / 'robust_kge_comparison_by_basin.csv').exists()


def test_artifact_csv_exists():
    assert (ROBUST_DIR / 'calibration_artifact_cases.csv').exists()


def test_diagnostics_csv_exists():
    assert (ROBUST_DIR / 'large_deviation_model_diagnostics.csv').exists()


def test_report_md_exists():
    assert (ROBUST_DIR / 'robust_comparison_report.md').exists()


def test_model_csv_has_29_models():
    df = pd.read_csv(ROBUST_DIR / 'robust_kge_comparison_by_model.csv')
    assert len(df) == 29, f"Expected 29 models, got {len(df)}"


def test_artifact_csv_nonempty():
    df = pd.read_csv(ROBUST_DIR / 'calibration_artifact_cases.csv')
    assert len(df) > 0, "calibration_artifact_cases.csv should be non-empty"


def test_diagnostics_covers_8_large_deviation():
    df = pd.read_csv(ROBUST_DIR / 'large_deviation_model_diagnostics.csv')
    assert len(df) == 8, f"Expected 8 LARGE_DEVIATION diagnostics, got {len(df)}"


def test_large_deviation_diagnostics_have_likely_source():
    df = pd.read_csv(ROBUST_DIR / 'large_deviation_model_diagnostics.csv')
    missing = df[df['primary_likely_source'].isna() | (df['primary_likely_source'] == '')]
    assert len(missing) == 0, f"Missing likely_source for: {missing['model'].tolist()}"


def test_no_formula_bug_claimed_without_evidence():
    df = pd.read_csv(ROBUST_DIR / 'large_deviation_model_diagnostics.csv')
    df_model = pd.read_csv(ROBUST_DIR / 'robust_kge_comparison_by_model.csv')
    
    # Check diagnostics
    for _, r in df.iterrows():
        if r['formula_bug_suspected'] == 'yes':
            # If yes, must have concrete evidence in reason
            assert 'parameter' in str(r['reason']).lower() or 'formula' in str(r['reason']).lower() or 'mismatch' in str(r['reason']).lower(), \
                f"{r['model']}: formula_bug_suspected=yes but no concrete evidence in reason"
    
    # Check model-level
    for _, r in df_model.iterrows():
        if r['formula_bug_suspected'] == 'yes':
            assert False, f"{r['model']}: formula_bug_suspected=yes without corresponding diagnostic evidence"


def test_report_contains_claim_language():
    text = (ROBUST_DIR / 'robust_comparison_report.md').read_text()
    assert 'Claim Language' in text or 'RECOMMENDED CLAIM' in text
    assert 'not a replacement' in text.lower() or 'NOT a replacement' in text


def test_report_states_robust_is_post_hoc():
    text = (ROBUST_DIR / 'robust_comparison_report.md').read_text()
    assert 'post-hoc' in text.lower() or 'POST-HOC' in text or 'post_hoc' in text


def test_report_contains_recommendation():
    text = (ROBUST_DIR / 'robust_comparison_report.md').read_text()
    assert 'recommend' in text.lower() or 'Recommend' in text


def test_report_does_not_claim_global_equivalence():
    text = (ROBUST_DIR / 'robust_comparison_report.md').read_text()
    # These phrases may appear in "NOT ALLOWED" section — that's OK
    # They should NOT appear as positive claims outside the NOT ALLOWED block
    not_allowed_section_start = text.find('NOT ALLOWED')
    not_allowed_section_end = text.find('ALLOWED', not_allowed_section_start + 20) if not_allowed_section_start >= 0 else -1
    
    check_text = text
    if not_allowed_section_start >= 0 and not_allowed_section_end >= 0:
        check_text = text[:not_allowed_section_start] + text[not_allowed_section_end:]
    
    assert 'numerically equivalent' not in check_text.lower()


def test_no_uh_models_in_robust_comparison():
    df = pd.read_csv(ROBUST_DIR / 'robust_kge_comparison_by_model.csv')
    excluded = {'gr4j', 'hbv96', 'hillslope', 'ihacres', 'newzealand2', 'plateau', 'smar'}
    present = set(df['model']) & excluded
    assert len(present) == 0, f"UH models should not be in robust comparison: {present}"
