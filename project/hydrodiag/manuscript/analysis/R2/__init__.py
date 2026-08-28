"""Canonical R2 Parameter Analysis Package.

Modular analyses:
  - shared_parameter_specs: 15 shared parameters identities, bounds, and normalizations
  - parameter_ledger: Raw long-form parameter ledger from lowest-level artifacts (310,635 rows)
  - canonical_vectors: Canonical parameter vector reductions (3,186 rows)
  - macro_whole_space: Macro whole-space 15-D displacement and ensemble within/between/excess
  - parameter_shifts_all15: All 15 signed parameter shifts across Full, Strata, and Robustness subsets
  - tgd_attribution_control: Macro TGD attribution control and paired Delta_beta bootstrap
  - diagnostics_and_safeguards: IC restart quality, dPL seed stability, and boundary mass safeguards
  - canonical_gates: 12 Canonical promotion gates verification
  - run_all: End-to-end orchestration runner
"""
from __future__ import annotations

from r2_config import (
    BASE_SEED,
    DEFAULT_DRAWS,
    DPL_SEEDS,
    IC_STARTS,
    PARADIGMS,
    STRATA,
    STRATA_COUNTS,
    STRUCTURES,
    TOTAL_BASINS,
)
from shared_parameter_specs import (
    PARAMETER_METADATA,
    SHARED_15_PARAMETERS,
    normalize_parameters,
    physical_from_normalized,
)
from run_r2 import run_r2_pipeline

__all__ = [
    "BASE_SEED",
    "DEFAULT_DRAWS",
    "DPL_SEEDS",
    "IC_STARTS",
    "PARADIGMS",
    "PARAMETER_METADATA",
    "SHARED_15_PARAMETERS",
    "STRATA",
    "STRATA_COUNTS",
    "STRUCTURES",
    "TOTAL_BASINS",
    "normalize_parameters",
    "physical_from_normalized",
    "run_r2_pipeline",
]
