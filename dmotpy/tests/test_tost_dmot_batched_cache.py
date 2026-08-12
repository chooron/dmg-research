"""Test: dMoT batched cache artifacts."""
import json, csv
from pathlib import Path

import pandas as pd
import pytest

CACHE_DIR = Path(__file__).parent.parent / 'validation_results' / 'tost_equivalence' / 'dmot_cache'

EXCLUDED_UH = {'gr4j', 'hbv96', 'hillslope', 'ihacres', 'newzealand2', 'plateau', 'smar'}
TESTABLE_29 = {'alpine1','alpine2','australia','collie1','collie2','collie3','flexb','flexi','flexis',
               'gsfb','hymod','modhydrolog','mopex1','mopex2','mopex3','mopex4','mopex5',
               'newzealand1','penman','simhyd','susannah1','susannah2','tank','tcm','topmodel',
               'us1','vic','wetland','xinanjiang'}


def test_cache_dir_exists():
    assert CACHE_DIR.is_dir()


def test_manifest_exists():
    assert (CACHE_DIR / 'dmot_batched_outputs_manifest.json').exists()


def test_run_summary_exists():
    assert (CACHE_DIR / 'dmot_batched_run_summary.csv').exists()


def test_qobs_metrics_exists():
    assert (CACHE_DIR / 'dmot_basin_level_qobs_metrics.csv').exists()


def test_manifest_contains_all_testable_models():
    with open(CACHE_DIR / 'dmot_batched_outputs_manifest.json') as f:
        m = json.load(f)
    successful = set(m.get('successful_models', []))
    missing = TESTABLE_29 - successful
    assert len(missing) == 0, f"Missing testable models: {missing}"


def test_manifest_excludes_uh_models():
    with open(CACHE_DIR / 'dmot_batched_outputs_manifest.json') as f:
        m = json.load(f)
    excluded = set(m.get('excluded_uh_models', []))
    assert EXCLUDED_UH == excluded, f"UH exclusion mismatch: {EXCLUDED_UH ^ excluded}"


def test_run_summary_all_ok():
    df = pd.read_csv(CACHE_DIR / 'dmot_batched_run_summary.csv')
    failures = df[df['run_status'] != 'OK']
    assert len(failures) == 0, f"Run failures: {failures['model'].tolist()}"


def test_all_models_559_basins():
    df = pd.read_csv(CACHE_DIR / 'dmot_batched_run_summary.csv')
    wrong = df[df['n_basins'] != 559]
    assert len(wrong) == 0, f"Models with wrong basin count: {wrong['model'].tolist()}"


def test_warmup_eval_lengths():
    df = pd.read_csv(CACHE_DIR / 'dmot_batched_run_summary.csv')
    assert (df['n_days_warmup'] == 3652).all()
    assert (df['n_days_eval'] == 4018).all()


def test_no_nan_inf_q():
    df = pd.read_csv(CACHE_DIR / 'dmot_batched_run_summary.csv')
    bad = df[df['any_nan_inf_q'] == True]
    assert len(bad) == 0, f"NaN/Inf in Q: {bad['model'].tolist()}"


def test_quality_checks_all_pass():
    df = pd.read_csv(CACHE_DIR / 'dmot_output_quality_checks.csv')
    failures = df[df['status'] == 'FAIL']
    assert len(failures) == 0, f"Failed quality checks:\n{failures.to_string()}"


def test_area_index_recorded_as_11():
    with open(CACHE_DIR / 'dmot_batched_outputs_manifest.json') as f:
        m = json.load(f)
    assert m.get('area_index') == 11, f"AREA_INDEX wrong: {m.get('area_index')}"


def test_first9_recheck_exists():
    assert (CACHE_DIR / 'recheck_first9_qobs_metrics_after_area_fix.csv').exists()


def test_first9_recheck_all_models_present():
    df = pd.read_csv(CACHE_DIR / 'recheck_first9_qobs_metrics_after_area_fix.csv')
    expected = {'alpine1','alpine2','australia','collie1','collie2','collie3','flexb','flexi','flexis'}
    assert set(df['model']) == expected


def test_qobs_metrics_has_all_models():
    df = pd.read_csv(CACHE_DIR / 'dmot_basin_level_qobs_metrics.csv')
    models_in = set(df['model'].unique())
    missing = TESTABLE_29 - models_in
    assert len(missing) == 0, f"Missing from metrics: {missing}"
    # No excluded models
    extra = models_in - TESTABLE_29
    assert len(extra) == 0, f"Unexpected models in metrics: {extra}"


def test_qobs_metrics_basin_count():
    df = pd.read_csv(CACHE_DIR / 'dmot_basin_level_qobs_metrics.csv')
    for model in TESTABLE_29:
        n = len(df[df['model'] == model])
        assert n == 559, f"{model}: expected 559 basins, got {n}"


def test_no_final_tost_results():
    """Planning stage: no final TOST result file should exist."""
    tost_dir = CACHE_DIR.parent
    assert not (tost_dir / 'tost_global_summary.csv').exists(), \
        "Final TOST results should not exist yet"
    assert not (tost_dir / 'tost_equivalence_report.md').exists(), \
        "Final TOST report should not exist yet"
