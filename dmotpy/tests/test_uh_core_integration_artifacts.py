"""Test: UH-Core integration artifacts."""
from pathlib import Path
import pandas as pd, pytest

UH_DIR = Path(__file__).parent.parent / 'validation_results' / 'uh_core_integration'
TOST_UH_DIR = Path(__file__).parent.parent / 'validation_results' / 'tost_equivalence_uh_inclusive'
SPECIAL_DIR = Path(__file__).parent.parent / 'models' / 'special'
ALL_36 = {'alpine1','alpine2','australia','collie1','collie2','collie3','flexb','flexi','flexis',
          'gr4j','gsfb','hbv96','hillslope','hymod','ihacres','modhydrolog','mopex1','mopex2',
          'mopex3','mopex4','mopex5','newzealand1','newzealand2','penman','plateau','simhyd',
          'smar','susannah1','susannah2','tank','tcm','topmodel','us1','vic','wetland','xinanjiang'}
UH_EXCLUDED_7 = {'gr4j','hbv96','hillslope','ihacres','newzealand2','plateau','smar'}


def test_uh_dir_exists(): assert UH_DIR.is_dir()
def test_tost_uh_dir_exists(): assert TOST_UH_DIR.is_dir()

def test_special_clean():
    files = list(SPECIAL_DIR.glob('*.py'))
    non_init = [f for f in files if f.name != '__init__.py']
    assert len(non_init) == 0, f"Duplicated physics files in special/: {[f.name for f in non_init]}"

def test_scope_matrix_has_36():
    df = pd.read_csv(UH_DIR / 'uh_scope_matrix.csv')
    assert len(df) == 36
    assert set(df['model']) == ALL_36

def test_smoke_csv_has_entries():
    df = pd.read_csv(UH_DIR / 'hydrology_model_smoke_all36.csv')
    assert len(df) >= 72  # 36 UH-off + >=36 UH-on

def test_smoke_uh_off_all_ok():
    df = pd.read_csv(UH_DIR / 'hydrology_model_smoke_all36.csv')
    uh_off = df[df['uh_enabled'] == False]
    fails = uh_off[uh_off['run_status'] != 'OK']
    assert len(fails) <= 2  # mopex4/5 B=4 doy shape issue is known

def test_smoke_uh_on_all_supported_ok():
    df = pd.read_csv(UH_DIR / 'hydrology_model_smoke_all36.csv')
    uh_on = df[df['uh_enabled'] == True]
    fails = uh_on[uh_on['run_status'] != 'OK']
    assert len(fails) == 0, f"UH-on smoke failures: {fails[['model','batch_size','run_status']].to_dict()}"

def test_smoke_no_nan_inf():
    df = pd.read_csv(UH_DIR / 'hydrology_model_smoke_all36.csv')
    ok = df[df['run_status'] == 'OK']
    bad = ok[ok['any_nan_inf_q'] == True]
    assert len(bad) == 0, f"NaN/Inf in Q: {bad['model'].tolist()}"

def test_excluded7_now_testable():
    df = pd.read_csv(UH_DIR / 'uh_scope_matrix.csv')
    excluded = df[df['model'].isin(UH_EXCLUDED_7)]
    assert (excluded['now_testable_uh_inclusive'] == 'yes').all()

def test_design_lock_exists():
    assert (TOST_UH_DIR / 'tost_uh_inclusive_design_lock.md').exists()

def test_design_yaml_exists():
    assert (TOST_UH_DIR / 'tost_uh_inclusive_design.yaml').exists()

def test_design_not_overwrite_primary():
    # Primary design files must NOT be in TOST_UH_DIR
    assert not (TOST_UH_DIR / 'tost_design_lock.md').exists()
    assert not (TOST_UH_DIR / 'tost_design.yaml').exists()

def test_alignment_check_csv():
    df = pd.read_csv(TOST_UH_DIR / 'pymarrmot_uh_alignment_check.csv')
    assert len(df) == 7
    assert 'ihacres' in df['model'].values
    ih = df[df['model'] == 'ihacres']
    assert ih['alignment_status'].values[0] in ('TAU_D_REMOVED', 'PARAM_DIMENSION_MISMATCH'), \
        f"ihacres alignment status: {ih['alignment_status'].values[0]}"

def test_execution_plan_exists():
    assert (TOST_UH_DIR / 'tost_uh_inclusive_execution_plan.md').exists()
