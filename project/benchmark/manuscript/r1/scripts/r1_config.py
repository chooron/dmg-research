"""Frozen paths and configuration for the JoH R1 analysis.

This module is intentionally local to manuscript/r1.  It only defines paths and
constants; it does not create directories or modify benchmark artifacts.
"""
from __future__ import annotations

import sys
from pathlib import Path

# /.../project/benchmark/manuscript/r1/scripts -> repository root.
REPO = Path(__file__).resolve().parents[5]
BENCHMARK = REPO / "project" / "benchmark"
DATA_ROOT = REPO / "data"

R1_ROOT = BENCHMARK / "manuscript" / "r1"
SCRIPTS_DIR = R1_ROOT / "scripts"
FIGURES_DIR = R1_ROOT / "figures"
TABLES_DIR = R1_ROOT / "tables"
CACHE_DIR = R1_ROOT / "cache"

FORMAL_ROOT = BENCHMARK / "results" / "ic_dpl_seenbasin_formal_20260901"
IC_ROOT = BENCHMARK / "results" / "ic_dpl_aligned_full300_20260819_final"
IC_CHECKPOINT_ROOT = IC_ROOT / "checkpoints" / "ic_dpl_aligned_full300_20260819"
DPL_ROOT = BENCHMARK / "results" / "dpl_canonical_v2_20260831"
VIC_AUDIT_ROOT = BENCHMARK / "results" / "ic_vic_full300_dynamic_doy_20260901"
FORENSIC_ROOT = BENCHMARK / "results" / "canonical_v2_remote_forensics_20260901"

IDS_PATH = DATA_ROOT / "531sub_id.txt"
GAGE_IDS_PATH = DATA_ROOT / "gage_id.npy"
DATASET_PATH = DATA_ROOT / "camels_dataset"
CARAVAN_PATH = DATA_ROOT / "caravan_671_attributes.npy"
FORCING_V2_PATH = DATA_ROOT / "camels_forcing_v2.pkl"
DATE_INDEX_PATH = DATA_ROOT / "camels_dates.npy"

FORMAL_PAIRED_SOURCE = FORMAL_ROOT / "02_BASIN_PAIRED_KGE_LONG.csv"
FORMAL_MODEL_SOURCE = FORMAL_ROOT / "03_MODEL_PERFORMANCE_SUMMARY.csv"
FORMAL_MANIFEST = FORMAL_ROOT / "00_INPUT_MANIFEST.csv"
IC_STATUS_PATH = IC_ROOT / "status_summary.json"

# This is the current canonical registry order.  It is not reordered by any
# performance statistic in R1 figures or tables.
MODEL_REGISTRY = (
    "alpine1", "alpine2", "australia", "collie1", "collie2", "collie3",
    "flexb", "flexi", "flexis", "gr4j", "gsfb", "hbv96", "hillslope",
    "hymod", "ihacres", "modhydrolog", "mopex1", "mopex2", "mopex3",
    "mopex4", "mopex5", "newzealand1", "newzealand2", "penman", "plateau",
    "simhyd", "smar", "susannah1", "susannah2", "tank", "tcm", "topmodel",
    "us1", "vic", "wetland", "xinanjiang",
)

TEST_START = "1995-10-01"
TEST_END = "2010-09-30"
EVAL_WARMUP_DAYS = 365
CANONICAL_DPL_SEED = 42
CANONICAL_DPL_SOURCE_SHA = "7d1132bf5c9ee0114a1a12dd10720101f6ca3b74"

# Reused, previously implemented R1 temporal definition.  The halves are
# date-based and are applied to post-warmup TEST outputs; no new split is
# designed here.
TEMPORAL_AB = (
    {"partition": "A", "start_date": "1995-10-01", "end_date": "2003-03-31"},
    {"partition": "B", "start_date": "2003-04-01", "end_date": "2010-09-30"},
)
TEMPORAL_SOURCE_SCRIPT = BENCHMARK / "scripts" / "diagnostics" / "r1_r2_nontraining_complete.py"
TEMPORAL_PRIOR_RESULT = BENCHMARK / "results" / "r1_r2_nontraining_complete_20260829" / "r1_temporal_ab_summary_strict34.csv"
TEMPORAL_QUICK_RESULT = BENCHMARK / "results" / "r1_r2_quick_survey_20260829" / "temporal_ab_feasibility.csv"

# Checksums independently recorded in the canonical remote/local data-contract
# forensic report.  They are audit references, not generated result values.
EXPECTED_DATA = {
    "531sub_id.txt": (4882, "140dd33776049b3c970a2fab0568c36d7a8f72ab12d45521040aef91de68b444"),
    "camels_dataset": (266827436, "2c4666e2ca5ece74028d5d2aa898754abf25e9a877701b218826547cb4e7a264"),
    "gage_id.npy": (5496, "81117b75abf870fbc9195a17d663307dba482621b4d81493ddeb282c437a8144"),
    "caravan_671_attributes.npy": (188008, "686366653e5cbcac00ac24ecb20b710b4940ccf73fe1957f27f7da1969dfd825"),
    "camels_forcing_v2.pkl": (100096795, "f48d82b2cd3d023a2e89aa7060df2929dba3c8dd73ea3b11241f7cff5b04eb1d"),
    "camels_dates.npy": (99472, "abd5b0cfff04a3b66a08a5f54557a6974851a2e98e42d7e7e523202783261126"),
}

# Add project source roots without importing any training entry point.  In
# particular, do not import scripts that create benchmark result directories.
sys.path[:0] = [str(REPO), str(BENCHMARK), str(BENCHMARK / "src")]
