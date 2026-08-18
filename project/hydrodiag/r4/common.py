"""R4 common constants and path resolution.

All R4 code reuses the R1/R2 canonical period definitions and the
repository-standard CAMELS-531 bundle (``ablation.ic_core.data_adapter``).
Nothing here re-implements data loading; it only pins the R4 conventions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# r4/ -> project/hydrodiag/r4: parents[1] is the hydrodiag project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]

# The main checkout holds the physical results root (worktrees share `data`
# but NOT `results`).  Prefer an explicit override, then the main checkout,
# then the worktree-local results dir.
_MAIN_CHECKOUT_RESULTS = Path.home() / "code/dmg-research/project/hydrodiag/results"


def default_results_root() -> Path:
    env = os.environ.get("R4_RESULTS_ROOT")
    if env:
        return Path(env).resolve()
    if _MAIN_CHECKOUT_RESULTS.exists():
        return _MAIN_CHECKOUT_RESULTS
    return (PROJECT_ROOT / "results").resolve()


def default_data_root() -> Path:
    env = os.environ.get("R4_DATA_ROOT")
    if env:
        return Path(env).resolve()
    candidates = [
        WORKSPACE_ROOT / "data",          # orca worktree shared data symlink
        PROJECT_ROOT / "data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


# ---------------------------------------------------------------------------
# R1/R2 canonical periods (frozen; identical to manuscript/scripts/r1_daily_inference.py)
# ---------------------------------------------------------------------------

PERIODS = {
    "warmup": {"start": "1980-10-01", "end": "1981-09-30", "days": 365},
    "train": {"start": "1981-10-01", "end": "1995-09-30", "days": 5113},
    "test": {"start": "1995-10-01", "end": "2010-09-30", "days": 5479},
}

# Indices on the 12418-day axis (verified against R3 manifests and
# `ablation.ic_core.periods.resolve_periods`):
#   train_forcing = [0, 5478)      (warmup + train)
#   test_forcing  = [5113, 10957)  (365 d preceding warmup + test)
PERIOD_INDEX = {
    "train_start": 365,
    "train_end": 5477,          # inclusive
    "test_start": 5478,
    "test_end": 10956,          # inclusive
    "train_forcing_start": 0,
    "train_forcing_end": 5478,  # exclusive
    "test_forcing_start": 5113,
    "test_forcing_end": 10957,  # exclusive
}

# ---------------------------------------------------------------------------
# Model identity (canonical repository keys)
# ---------------------------------------------------------------------------

# model_key -> (parameter count, short label)
MODEL_KEYS = {
    "XAJ": (15, "Base"),
    "XAJ_CN": (17, "CN"),
    "XAJ_TGD2": (17, "TGD2"),
}

# Legacy/other keys seen on the remote training node (never canonical for R4).
LEGACY_MODEL_KEYS = ("XAJ_TGD", "XAJ_PD", "GR4J", "GR4J_CN", "SIMHYD", "SIMHYD_CN", "HBV")

# ---------------------------------------------------------------------------
# R1/R2 canonical result run identities (recovery clues, not assumptions)
# ---------------------------------------------------------------------------

# IC runs: canonical local names -> raw record subdir name.
IC_CANONICAL_RUNS = {
    "XAJ": ("xaj_base_cmaes_531_batched_paired_v2", "xaj"),
    "XAJ_CN": ("xaj_cn_cmaes_531_batched_paired_v2", "xaj_cn"),
    "XAJ_TGD2": ("xaj_tgd2_cmaes_531_batched_v1", "xaj_tgd2"),
}

# dPL runs: canonical local directory names.
DPL_CANONICAL_RUNS = {
    "XAJ": "dpl_camels_531_lite_v2",
    "XAJ_CN": "dpl_camels_531_lite_v2",
    "XAJ_TGD2": "dpl_camels_531_lite_v3_tgd2_dpl_audited",
}

DPL_SEEDS = (42, 123, 2026)

# ---------------------------------------------------------------------------
# R3 dev-only run identities (local results root, synthetic q* trained)
# ---------------------------------------------------------------------------

R3_IC_RUNS = {
    "XAJ": "r3_misspec_ic_xaj_531_v1",
    "XAJ_CN": "r3_gate_ic_xaj_cn_531_v1",
    "XAJ_TGD2": "r3_misspec_ic_xaj_tgd2_531_v1",
}

R3_DPL_RUNS = {
    "XAJ": "r3_misspec_dpl_xaj_seed_",
    "XAJ_CN": "r3_gate_dpl_xaj_cn_seed_",
    "XAJ_TGD2": "r3_misspec_dpl_xaj_tgd2_seed_",
}

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def zfill8(basin_id: Any) -> str:
    """Normalise a basin identifier to the canonical 8-digit string."""
    return str(basin_id).zfill(8)


def bundle_config(data_root: Path) -> dict[str, Any]:
    """Repository-standard 531-bundle config (same keys as ic_foundation_531_v1)."""
    return {
        "project_root": str(PROJECT_ROOT),
        "dataset_path": str(data_root / "camels_dataset"),
        "gage_ids_path": str(data_root / "gage_id.npy"),
        "dates_path": str(data_root / "camels_dates.npy"),
        "basin_list_path": str(data_root / "531sub_id.txt"),
        "periods": {
            "warmup": {"start": "1980-10-01", "end": "1981-09-30"},
            "train": {"start": "1981-10-01", "end": "1995-09-30"},
            "test": {"start": "1995-10-01", "end": "2010-09-30"},
        },
    }


def load_bundle(data_root: Path):
    """Load the canonical CAMELS-531 bundle (reuses ablation.ic_core)."""
    from ablation.ic_core.data_adapter import load_531_bundle

    return load_531_bundle(bundle_config(data_root))


def period_slices(bundle: Any) -> dict[str, slice]:
    """R1/R2 canonical train/test slices on the full 12418-day axis."""
    p = PERIOD_INDEX
    return {
        "full": slice(0, bundle.forcing.shape[1]),
        "train": slice(p["train_start"], p["train_end"] + 1),
        "test": slice(p["test_start"], p["test_end"] + 1),
        "train_forcing": slice(p["train_forcing_start"], p["train_forcing_end"]),
        "test_forcing": slice(p["test_forcing_start"], p["test_forcing_end"]),
    }
