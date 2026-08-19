"""Shared R3 constants, data loading, metrics, and pilot-basin selection.

All scientific constants (periods, basin list, attribute order, metric
definitions) are inherited from the repository's canonical sources:

- 531 basin list, forcing, attributes, dates: ``data/531sub_id.txt``,
  ``data/camels_dataset``, ``data/gage_id.npy``, ``data/camels_dates.npy``
- periods: warmup 1980-10-01..1981-09-30, train 1981-10-01..1995-09-30,
  test 1995-10-01..2010-09-30 (flexmopex protocol used by both IC and dPL)
- KGE: repository standard KGE (``training/ic/gpu_kge.py`` semantics)
- CT/AMJJ water-year signatures: definitions from
  ``manuscript/scripts/r1_statistics.py`` (water year starts Oct 1; CT is
  the day cumulative flow first reaches 50% of annual flow; AMJJ is the
  April-July fraction of annual flow).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

# Canonical roots.
# Canonical roots are worktree-local by default. External read-only sources
# require an explicit environment override.
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "HYDRODIAG_DATA_ROOT",
        str(DEFAULT_PROJECT_ROOT / "data"),
    )
)
DEFAULT_RESULTS_ROOT = Path(
    os.environ.get(
        "HYDRODIAG_RESULTS_ROOT",
        str(DEFAULT_PROJECT_ROOT / "results"),
    )
)

PERIODS = {
    "warmup": {"start": "1980-10-01", "end": "1981-09-30"},
    "train": {"start": "1981-10-01", "end": "1995-09-30"},
    "test": {"start": "1995-10-01", "end": "2010-09-30"},
}

# R2 canonical shared-XAJ parameter order (15 parameters common to Base,
# CN and TGD2), reused verbatim from run_r2_within_structure_baseline.py.
COMMON_XAJ = [
    "xaj_k",
    "xaj_b",
    "xaj_im",
    "xaj_um",
    "xaj_lm",
    "xaj_dm",
    "xaj_c",
    "xaj_sm",
    "xaj_ex",
    "xaj_ki",
    "xaj_kg",
    "xaj_ci",
    "xaj_cg",
    "xaj_a",
    "xaj_theta",
]

FRAC_SNOW_INDEX = 3  # CAMELS attribute order, ``frac_snow`` position

IC_RESULT_ROOTS = {
    "XAJ": "xaj_base_cmaes_531_batched_paired_v2",
    "XAJ_CN": "xaj_cn_cmaes_531_batched_paired_v2",
    "XAJ_TGD2": "xaj_tgd2_cmaes_531_batched_v1",
}
IC_RAW_SUBDIRS = {"XAJ": "xaj", "XAJ_CN": "xaj_cn", "XAJ_TGD2": "xaj_tgd2"}


def git_commit(project_root: Path) -> dict[str, str]:
    """Best-effort repository provenance (works in a git worktree)."""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip()
    except Exception:
        head = "UNVERIFIED"
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=project_root, text=True
            ).strip()
        )
    except Exception:
        dirty = True
    return {"commit": head, "dirty": dirty}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle(
    project_root: Path = DEFAULT_PROJECT_ROOT, data_root: Path = DEFAULT_DATA_ROOT
):
    """Load the canonical 531-basin IC bundle (reuses ablation.ic_core)."""
    import sys

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from ablation.ic_core.data_adapter import load_531_bundle

    config = {
        "project_root": str(project_root),
        "dataset_path": str(data_root / "camels_dataset"),
        "gage_ids_path": str(data_root / "gage_id.npy"),
        "dates_path": str(data_root / "camels_dates.npy"),
        "basin_list_path": str(data_root / "531sub_id.txt"),
        "periods": PERIODS,
    }
    return load_531_bundle(config), config


def bundle_with_synthetic_target(bundle, target_mm_day: np.ndarray) -> Any:
    """Return a bundle whose calibration target is the synthetic Q*.

    Uses ``dataclasses.replace`` so the frozen IC foundation dataclass and
    all downstream runtime code remain untouched.
    """
    from dataclasses import replace

    target = np.asarray(target_mm_day, dtype=np.float64)
    if target.shape != bundle.target_mm_day.shape:
        raise ValueError(
            f"synthetic target shape {target.shape} != bundle target "
            f"{bundle.target_mm_day.shape}"
        )
    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("synthetic target must be finite and non-negative")
    return replace(
        bundle,
        target_mm_day=target,
        valid_target_mask=np.ones(target.shape, dtype=bool),
        target_unit_ic="mm/day (synthetic Q*)",
    )


def frac_snow_series(bundle) -> pd.DataFrame:
    values = np.asarray(bundle.raw_attributes[:, FRAC_SNOW_INDEX], dtype=np.float64)
    return pd.DataFrame({"basin_id": list(bundle.basin_ids), "frac_snow": values})


def period_indices(bundle) -> dict[str, tuple[int, int]]:
    p = bundle.periods
    return {
        "warmup": (p.warmup.start_index, p.warmup.end_index),
        "train": (p.train.start_index, p.train.end_index),
        "test": (p.test.start_index, p.test.end_index),
        "train_forcing": (p.train_forcing_start_index, p.train_forcing_end_index),
        "test_forcing": (p.test_forcing_start_index, p.test_forcing_end_index),
    }


def standard_kge(sim: np.ndarray, obs: np.ndarray, min_valid: int = 30) -> float:
    """Repository-standard KGE (same mask/floor semantics as gpu_kge)."""
    sim = np.asarray(sim, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    mask = np.isfinite(sim) & np.isfinite(obs) & (sim >= 0) & (obs >= 0)
    count = int(mask.sum())
    if count < min_valid:
        return math.nan
    s = sim[mask]
    o = obs[mask]
    o_std = o.std()
    if o_std < 1e-10:
        return math.nan
    r = np.corrcoef(s, o)[0, 1]
    alpha = s.std() / o_std
    beta = s.mean() / o.mean()
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def nse(sim: np.ndarray, obs: np.ndarray, min_valid: int = 30) -> float:
    sim = np.asarray(sim, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    mask = np.isfinite(sim) & np.isfinite(obs) & (sim >= 0) & (obs >= 0)
    if int(mask.sum()) < min_valid:
        return math.nan
    s, o = sim[mask], obs[mask]
    denom = ((o - o.mean()) ** 2).sum()
    if denom <= 0:
        return math.nan
    return float(1.0 - ((s - o) ** 2).sum() / denom)


def pbias(sim: np.ndarray, obs: np.ndarray, min_valid: int = 30) -> float:
    sim = np.asarray(sim, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    mask = np.isfinite(sim) & np.isfinite(obs) & (sim >= 0) & (obs >= 0)
    if int(mask.sum()) < min_valid:
        return math.nan
    s, o = sim[mask], obs[mask]
    total = o.sum()
    if total <= 0:
        return math.nan
    return float(100.0 * (s.sum() - total) / total)


def water_year_series(dates: np.ndarray, flow: np.ndarray) -> pd.DataFrame:
    """Rows of (water_year, ct_day, amjj) for complete water years.

    Definition reused from ``manuscript/scripts/r1_statistics.py``
    (``signature_rows``): water year starts Oct 1; CT is the first day the
    cumulative flow reaches 50% of the annual total; AMJJ is the April-July
    flow divided by the annual total.  Only complete water years are kept.
    """
    dates = pd.to_datetime(np.asarray(dates)).to_numpy(dtype="datetime64[D]")
    flow = np.asarray(flow, dtype=np.float64)
    year = pd.to_datetime(dates).year + (pd.to_datetime(dates).month >= 10).astype(int)
    rows = []
    for wy in np.unique(year):
        sel = year == wy
        sub = flow[sel]
        if len(sub) != 365 and not (len(sub) == 366):
            continue  # incomplete water year
        # the dataset is contiguous daily; a complete water year is 365 days
        # (1980-2010 contains no leap water years for the canonical periods,
        # but keep the explicit check for robustness)
        if len(sub) != 365:
            continue
        total = float(sub.sum())
        if total <= 0 or not np.isfinite(sub).all() or (sub < 0).any():
            continue
        ct = int(np.argmax(np.cumsum(sub) >= 0.5 * total) + 1)
        month = pd.to_datetime(dates[sel]).month.to_numpy()
        amjj = float(sub[(month >= 4) & (month <= 7)].sum() / total)
        rows.append({"water_year": int(wy), "ct_day": ct, "amjj": amjj})
    return pd.DataFrame(rows)


def water_year_errors(
    dates: np.ndarray, sim: np.ndarray, obs: np.ndarray
) -> dict[str, float]:
    """Median |error| of CT (days) and AMJJ (fraction) across complete years."""
    sim_years = water_year_series(dates, sim)
    obs_years = water_year_series(dates, obs)
    if obs_years.empty or sim_years.empty:
        return {"ct_error_absolute": math.nan, "amjj_error_absolute": math.nan}
    joined = sim_years.merge(obs_years, on="water_year", suffixes=("_sim", "_obs"))
    if joined.empty:
        return {"ct_error_absolute": math.nan, "amjj_error_absolute": math.nan}
    return {
        "ct_error_absolute": float(
            np.median(np.abs(joined["ct_day_sim"] - joined["ct_day_obs"]))
        ),
        "amjj_error_absolute": float(
            np.median(np.abs(joined["amjj_sim"] - joined["amjj_obs"]))
        ),
    }


def pilot_basin_subset(
    basin_ids: Iterable[str], frac_snow: np.ndarray, per_tercile: int = 4
) -> list[str]:
    """Deterministic stratified pilot subset across the frac_snow terciles.

    Engineering gate only: covers low/mid/high frac_snow but is not a final
    scientific sample.  No RNG: within each tercile the basins nearest the
    (1/8, 3/8, 5/8, 7/8) quantile positions of the sorted tercile are chosen.
    """
    ids = np.asarray(list(basin_ids))
    fs = np.asarray(frac_snow, dtype=np.float64)
    order = np.argsort(fs, kind="stable")
    sorted_ids = ids[order]
    n = len(ids)
    if n < 3:
        raise ValueError("pilot subset requires at least 3 basins")
    q1, q2 = np.quantile(fs, [1 / 3, 2 / 3])
    tercile = np.digitize(fs, [q1, q2])  # 0, 1, 2
    selected: list[str] = []
    for t in range(3):
        positions = np.flatnonzero(tercile == t)
        if len(positions) == 0:
            raise ValueError(f"empty frac_snow tercile {t}")
        # spread picks inside the tercile by its sorted index positions
        fracs = np.linspace(1 / 8, 7 / 8, per_tercile)
        picks = sorted({int(round(f * (len(positions) - 1))) for f in fracs})
        while len(picks) < per_tercile:
            picks.append(
                picks[-1] + 1 if picks[-1] + 1 < len(positions) else picks[-1] - 1
            )
        for p in picks[:per_tercile]:
            selected.append(str(sorted_ids[positions[p]]).zfill(8))
    return selected


def reordered_531_list(basin_ids: Iterable[str], first: Iterable[str]) -> list[str]:
    """Full 531-basin list with the pilot basins first, rest in original order."""
    first_ids = [str(b).zfill(8) for b in first]
    rest = [str(b).zfill(8) for b in basin_ids if str(b).zfill(8) not in set(first_ids)]
    if len(first_ids) + len(rest) != len(set(basin_ids)):
        raise ValueError("pilot basin reorder lost or duplicated basins")
    return first_ids + rest


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
