"""Configuration, constants, and schema contracts for canonical R1 analysis."""
from __future__ import annotations

import os
from pathlib import Path

# Paths
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
REPO_ROOT = PROJECT_ROOT.parent.parent if (PROJECT_ROOT.parent.parent / ".git").exists() else PROJECT_ROOT
STAGED_DIR = PROJECT_ROOT / "manuscript" / "cache" / "r1_rebuild_audit_staged"
RESULTS_DIR = HERE / "results"
MANIFEST_PATH = HERE / "staged_compact_manifest.json"

# Analysis parameters
DEFAULT_DRAWS = 10_000
BASE_SEED = 20260730
PARADIGMS = ("IC-CMA-ES", "dPL-MLP")
STRUCTURES = ("Base", "TGD", "CN")
PERIODS = ("test", "train")
EVAL_PERIOD = "test"
DPL_SEEDS = (42, 123, 2026)

# Snow strata definitions and frozen counts
STRATA = ("S1", "S2", "S3", "S4", "S5")
STRATA_BOUNDS = {
    "S1": (0.00, 0.05),
    "S2": (0.05, 0.15),
    "S3": (0.15, 0.30),
    "S4": (0.30, 0.50),
    "S5": (0.50, 1.00),
}
STRATA_COUNTS = {"S1": 165, "S2": 156, "S3": 121, "S4": 34, "S5": 55}
TOTAL_BASINS = 531

# Expected staged files and schemas
EXPECTED_TABLES = {
    "r1_basin_level_performance_rebuilt.csv": {
        "rows": 6372,
        "sha256": "f9c30837aab2c68f544e00b7b03efe6bf7c3d0cecf9bc8284d5d354356e8c0f9",
        "schema": [
            "basin_id", "paradigm", "structure", "model", "period",
            "seed_or_restart", "selected_run", "KGE", "NSE", "PBIAS", "RMSE",
            "valid_observation_count", "valid_simulation_count", "valid_days",
            "valid_metric", "basin_median_Delta_CT"
        ],
    },
    "r1_basin_level_ct.csv": {
        "rows": 6372,
        "sha256": "c19ab2e7fc9d7d5f68008abc2d976c8777322e2f3d1dcf05ab79c6af5432d14c",
        "schema": [
            "basin_id", "paradigm", "structure", "period", "seed_or_restart",
            "valid_year_count", "basin_median_Delta_CT", "basin_CT_q25_years",
            "basin_CT_q75_years", "CT_obs_median_years", "CT_sim_median_years",
            "frac_snow", "snow_stratum", "KGE_pass_0p60", "KGE", "basin_test_KGE"
        ],
    },
    "r1_basin_year_ct.csv": {
        "rows": 92394,
        "sha256": "cf63c7c70787a23835b38f71a586cda8aeed2c6f7a9882de442f70f89dde2cae",
        "schema": [
            "basin_id", "paradigm", "structure", "model", "period",
            "seed_or_restart", "water_year", "complete_year", "valid_year",
            "invalid_reason", "n_valid_days", "CT_obs", "CT_sim", "Delta_CT",
            "seed_count", "frac_snow", "snow_stratum"
        ],
    },
    "r1_basin_year_ct_runs.csv": {
        "rows": 184788,
        "schema": [
            "basin_id", "paradigm", "structure", "model", "period",
            "seed_or_restart", "water_year", "complete_year", "valid_year",
            "invalid_reason", "n_valid_days", "CT_obs", "CT_sim", "Delta_CT",
            "frac_snow", "snow_stratum"
        ],
    },
}

UPSTREAM_MANIFESTS = {
    "r1_streaming_validation.json": "4d1bbce371811684a41b58bbfa14d1917a149c16c7b7893db2b1945a56016407",
    "r1_audit_manifest.json": "c005798c3c5f3b727da4c0ca2fcbfdd4f58a13e9e59c578140ac578ac298bf04",
}
