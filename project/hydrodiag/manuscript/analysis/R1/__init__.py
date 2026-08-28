"""Canonical R1 Downstream Analysis Package.

Modular analyses:
  - canonical_basin_table: Basin-level data preparation & input contract verification
  - paired_contrasts: Same-basin paired estimands with strict alignment checks
  - snow_activity_analysis: S1-S5 strata distributions, continuous Spearman rho, S5-S1 endpoints
  - secondary_tgd_control: Secondary TGD structural control analyses
  - threshold_prevalence_audit: Metric cutoff prevalence across denominator types
  - robustness_analysis: Regional LORO and seed/restart sensitivity
  - canonical_gates: 5 Canonical promotion gates enforcement
  - run_all: End-to-end pipeline runner
"""
from __future__ import annotations

from config import (
    BASE_SEED,
    DEFAULT_DRAWS,
    DPL_SEEDS,
    EVAL_PERIOD,
    PARADIGMS,
    STRATA,
    STRATA_COUNTS,
    STRUCTURES,
    TOTAL_BASINS,
)
from cuda_engine import (
    bootstrap_median_ci,
    derive_seed,
    endpoint_activity_contrast,
    gpu_median,
    gpu_quantile,
    require_cuda,
    spearman,
    spearman_bootstrap,
)
from run_all import run_pipeline

__all__ = [
    "BASE_SEED",
    "DEFAULT_DRAWS",
    "DPL_SEEDS",
    "EVAL_PERIOD",
    "PARADIGMS",
    "STRATA",
    "STRATA_COUNTS",
    "STRUCTURES",
    "TOTAL_BASINS",
    "bootstrap_median_ci",
    "derive_seed",
    "endpoint_activity_contrast",
    "gpu_median",
    "gpu_quantile",
    "require_cuda",
    "run_pipeline",
    "spearman",
    "spearman_bootstrap",
]
