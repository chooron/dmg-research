"""Configuration, directory paths, and constants for canonical R2 parameter analysis."""
from __future__ import annotations

from pathlib import Path

# Base Paths
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
REPO_ROOT = PROJECT_ROOT.parent.parent if (PROJECT_ROOT.parent.parent / ".git").exists() else PROJECT_ROOT

DATA_DIR = PROJECT_ROOT.parents[1] / "data" if (PROJECT_ROOT.parents[1] / "data").exists() else REPO_ROOT / "data"
RESULTS_DIR = HERE / "results"
RESULTS_ROOT = PROJECT_ROOT / "results"

# Input Artifact Paths
BASIN_FILE = DATA_DIR / "531sub_id.txt"
BOUNDS_FILE = PROJECT_ROOT / "manuscript" / "supplement" / "results" / "s2_parameter_bounds_from_code.csv"
SNOW_FILE = PROJECT_ROOT / "manuscript" / "results" / "R1" / "r1_snow_attributes.csv"
CANONICAL_R1_BASIN_TABLE = PROJECT_ROOT / "manuscript" / "analysis" / "R1" / "results" / "canonical_basin_level.csv"

# Raw IC Directories (10 restarts each)
IC_RAW_DIRS = {
    "Base": RESULTS_ROOT / "xaj_base_cmaes_531_batched_paired_v2" / "raw" / "xaj",
    "CN": RESULTS_ROOT / "xaj_cn_cmaes_531_batched_paired_v2" / "raw" / "xaj_cn",
    "TGD": RESULTS_ROOT / "xaj_tgd2_cmaes_531_batched_v1" / "raw" / "xaj_tgd2",
}

# Raw dPL Directories (3 seeds: 42, 123, 2026)
DPL_ROOT_V2 = RESULTS_ROOT / "dpl_camels_531_lite_v2"
DPL_ROOT_TGD2 = RESULTS_ROOT / "dpl_camels_531_lite_v3_tgd2_dpl_audited" / "XAJ_TGD2"
DPL_SEED_DIRS = {
    "Base": DPL_ROOT_V2 / "XAJ",
    "CN": DPL_ROOT_V2 / "XAJ_CN",
    "TGD": DPL_ROOT_TGD2,
}

# Analysis Constants
BASE_SEED = 20260730
DEFAULT_DRAWS = 10_000
TOTAL_BASINS = 531
PARADIGMS = ("IC", "dPL")
STRUCTURES = ("Base", "CN", "TGD")
DPL_SEEDS = (42, 123, 2026)
IC_STARTS = tuple(range(10))

# Snow Strata
STRATA = ("S1", "S2", "S3", "S4", "S5")
STRATA_BOUNDS = {
    "S1": (0.00, 0.05),
    "S2": (0.05, 0.15),
    "S3": (0.15, 0.30),
    "S4": (0.30, 0.50),
    "S5": (0.50, 1.00),
}
STRATA_COUNTS = {"S1": 165, "S2": 156, "S3": 121, "S4": 34, "S5": 55}
SUBSETS = ("Full531", "ExcludeS5")
