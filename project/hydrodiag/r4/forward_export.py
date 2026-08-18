"""R4 post-hoc forward + state export orchestration.

Pipeline (no re-training, no re-calibration):

    trained parameters/checkpoint
        -> continuous full-axis recorded forward (production kernels)
        -> per-period state export (train / test, R1/R2 period definitions)
        -> per-basin daily CSV + run manifest (protocol, provenance, tags)

The manifest records the CN ``psol_annual`` / ``g_thresh`` values actually
used by the forward (full-axis window) *and* the R1-era per-period inference
windows (train_forcing / test_forcing) for auditability.

Outputs are written under ``<results_root>/r4_<run_id>/`` and are tagged
``DEV_ONLY``/``SYNTHETIC_TRAINED`` or ``OFFICIAL_OBSERVATION_TRAINED``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import (
    DEV_ONLY,
    DEV_ONLY_SYNTHETIC_TRAINED,
    IC_FUSED_5x200_SENSITIVITY,
    OFFICIAL_DPL_OBSERVATION_TRAINED,
    OFFICIAL_OBSERVATION_TRAINED,
    SYNTHETIC_TRAINED,
)
from .common import MODEL_KEYS, PERIOD_INDEX, bundle_config, load_bundle, zfill8
from .state_export import (
    cn_psol_gthresh,
    continuous_forward,
    model_instances,
    period_slices_full,
)

# Columns written to the daily export tables, per structure.
EXPORT_COLUMNS = {
    "XAJ": ("discharge", "wu", "wl", "wd", "s", "fr", "qi", "qg"),
    "XAJ_CN": ("discharge", "snow_pack", "thermal_state", "sca", "rain", "melt",
               "effective_precip", "wu", "wl", "wd", "s", "fr", "qi", "qg", "evap"),
    "XAJ_TGD2": ("discharge", "effective_precip", "tgd_storage", "tgd_tau", "tgd_retention",
                 "wu", "wl", "wd", "s", "fr", "qi", "qg"),
}

STATE_TO_COLUMN = {
    "G": "snow_pack",
    "eTG": "thermal_state",
    "effective_precip": "effective_precip",
    "tgd_storage": "tgd_storage",
    "tgd_tau": "tgd_tau",
    "tgd_retention": "tgd_retention",
    "rs_instant": "rs_instant",
    "evap": "evap",
    "rain": "rain",
    "melt": "melt",
    "sca": "sca",
    "wu": "wu", "wl": "wl", "wd": "wd", "s": "s", "fr": "fr", "qi": "qi", "qg": "qg",
}


def build_daily_tables(
    structure: str,
    basin_ids: list[str],
    dates: Any,
    q_full: np.ndarray,
    states: dict[str, np.ndarray],
    periods: tuple[str, ...] = ("train", "test"),
) -> dict[str, list[dict[str, Any]]]:
    """Slice the continuous simulation into per-period long tables.

    Returns {period: [row, ...]} with one row per (basin, day).
    """
    slices = period_slices_full(q_full.shape[1])
    columns = EXPORT_COLUMNS[structure]
    tables: dict[str, list[dict[str, Any]]] = {}
    for period in periods:
        sl = slices[period]
        rows: list[dict[str, Any]] = []
        for i, basin in enumerate(basin_ids):
            for t in range(sl.start, sl.stop):
                row: dict[str, Any] = {
                    "basin_id": basin,
                    "date": str(dates[t]),
                    "discharge": float(q_full[i, t]),
                }
                for col in columns:
                    if col == "discharge":
                        continue
                    key = col if col in states else STATE_TO_COLUMN.get(col)
                    if key in states:
                        row[col] = float(states[key][i, t])
                    else:
                        row[col] = np.nan  # structure has no such state
                rows.append(row)
        tables[period] = rows
    return tables


def write_daily_csv(tables: dict[str, list[dict[str, Any]]], out_dir: Path, run_id: str) -> list[Path]:
    import csv

    paths: list[Path] = []
    for period, rows in tables.items():
        path = out_dir / f"{run_id}_daily_{period}.csv"
        if not rows:
            path.write_text("", encoding="utf-8")
            paths.append(path)
            continue
        columns = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
    return paths


def psol_gthresh_manifest(bundle: Any, basin_ids: list[str], device: Any, dtype: Any) -> dict[str, dict[str, Any]]:
    """Per-basin psol_annual / g_thresh for the relevant windows (CN only).

    - ``full``: the window used by the continuous R4 export forward;
    - ``train_forcing`` / ``test_forcing``: the windows the R1-era per-period
      inference forwards used (informational; R1/R2 historical semantics).
    """
    from models.cemaneige import _estimate_psol_annual

    import torch

    forcing = bundle.forcing.astype(np.float32)
    slices = period_slices_full(forcing.shape[1])
    index = {b: i for i, b in enumerate(zfill8(x) for x in bundle.basin_ids)}
    result: dict[str, dict[str, Any]] = {}
    windows = {
        "full": slice(0, forcing.shape[1]),
        "train_forcing": slices["train_forcing"],
        "test_forcing": slices["test_forcing"],
    }
    for basin in basin_ids:
        i = index[basin]
        entry: dict[str, Any] = {}
        for label, sl in windows.items():
            p = torch.as_tensor(forcing[i, sl, 0], device=device, dtype=dtype)
            t = torch.as_tensor(forcing[i, sl, 1], device=device, dtype=dtype)
            psol = _estimate_psol_annual(p.unsqueeze(0), t.unsqueeze(0))
            entry[label] = {
                "psol_annual_mm": float(psol.detach().cpu().numpy()[0]),
                "g_thresh_mm": float((0.9 * psol).detach().cpu().numpy()[0]),
            }
        result[basin] = entry
    return result


def export_run(
    *,
    structure: str,
    parameters: np.ndarray,
    parameter_meta: dict[str, Any],
    basin_ids: list[str],
    data_root: Path,
    results_root: Path,
    run_id: str,
    tag: str,
    provenance: dict[str, Any],
    device: Any,
    dtype: Any = None,
    batch: int = 64,
    validate_subset: int = 8,
    periods: tuple[str, ...] = ("train", "test"),
    csv_basin_subset: Optional[list[str]] = None,
    save_npz: bool = True,
    extra_notes: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run the R4 export pipeline and write per-run artifacts + a manifest.

    ``tag`` must be one of r4.DEV_ONLY / r4.SYNTHETIC_TRAINED /
    r4.OFFICIAL_OBSERVATION_TRAINED.  ``run_id`` is the output directory
    name under ``results_root``.  ``save_npz`` (default True) writes the
    compact full-axis arrays (q + per-day states, float32); daily CSVs are
    written only when ``csv_basin_subset`` is given.
    """
    import torch

    ALLOWED_TAGS = (
        DEV_ONLY,
        SYNTHETIC_TRAINED,
        DEV_ONLY_SYNTHETIC_TRAINED,
        OFFICIAL_OBSERVATION_TRAINED,
        OFFICIAL_DPL_OBSERVATION_TRAINED,
        IC_FUSED_5x200_SENSITIVITY,
    )
    if tag not in ALLOWED_TAGS:
        raise ValueError(f"unknown output tag: {tag!r}")
    if structure not in MODEL_KEYS:
        raise ValueError(f"unknown structure: {structure!r}")
    if dtype is None:
        dtype = torch.float32
    device = torch.device(device)

    bundle = load_bundle(data_root)
    expected = [zfill8(b) for b in basin_ids]
    if len(expected) != len(set(expected)):
        raise ValueError("basin_ids contains duplicates")
    if len(expected) != parameters.shape[0]:
        raise ValueError(
            f"parameters rows {parameters.shape[0]} != basins {len(expected)}"
        )
    if len(expected) != 531:
        raise ValueError(
            f"R4 export requires the canonical 531-basin set, got {len(expected)}"
        )

    # basin order must equal the bundle order (no silent re-alignment)
    bundle_ids = [zfill8(b) for b in bundle.basin_ids]
    if expected != bundle_ids:
        raise ValueError(
            "basin_ids order differs from the canonical bundle order; "
            "R4 requires the canonical 531 order (no positional remap)"
        )

    models = model_instances(device, dtype)
    q_full, states = continuous_forward(
        structure, models[structure], parameters, bundle.forcing.astype(np.float32),
        device, dtype, batch=batch, validate_subset=validate_subset,
    )

    dates = bundle.dates
    out_dir = results_root / f"r4_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths: list[Path] = []
    if csv_basin_subset:
        csv_basins = [zfill8(b) for b in csv_basin_subset]
        if set(csv_basins) - set(expected):
            raise ValueError("csv_basin_subset contains basins outside the exported set")
        tables = build_daily_tables(structure, csv_basins, dates, q_full, states, periods)
        csv_paths = write_daily_csv(tables, out_dir, run_id)

    npz_path = None
    if save_npz:
        npz_path = out_dir / f"{run_id}_full_arrays.npz"
        np.savez_compressed(
            npz_path,
            basin_ids=np.asarray(expected),
            dates=np.asarray(dates, dtype="datetime64[D]"),
            q_full=q_full.astype(np.float32),
            **{key: states[key].astype(np.float32) for key in states},
        )

    psol = None
    if structure == "XAJ_CN":
        psol = psol_gthresh_manifest(bundle, expected, device, dtype)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "structure": structure,
        "structure_label": MODEL_KEYS[structure][1],
        "parameter_meta": parameter_meta,
        "provenance": provenance,
        "periods": {
            "warmup": {"start": "1980-10-01", "end": "1981-09-30"},
            "train": {"start": "1981-10-01", "end": "1995-09-30"},
            "test": {"start": "1995-10-01", "end": "2010-09-30"},
            "continuous_forward": "full 12418-day axis from zero initial states",
        },
        "psol_gthresh_semantics": {
            "note": "window-based psol_annual (R1/R2 historical semantics); "
                    "the R3 canonical_cn_psol_annual path is NOT used",
            "export_used_window": "full",
            "r1_inference_windows": ["train_forcing", "test_forcing"],
        },
        "cn_psol_gthresh": psol,
        "has_snow_module": structure == "XAJ_CN",
        "pseudo_swe": False,
        "state_columns": list(EXPORT_COLUMNS[structure]),
        "outputs": [str(p) for p in csv_paths],
        "arrays": str(npz_path) if npz_path is not None else None,
        "n_basins": len(expected),
        "n_days_full": q_full.shape[1],
        "q_finite": bool(np.isfinite(q_full).all()),
        "q_nonnegative": bool((q_full >= 0).all()),
        "device": str(device),
        "extra_notes": extra_notes or {},
    }
    manifest_path = out_dir / f"{run_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def kge_vs_reference(q_full: np.ndarray, reference: np.ndarray, periods: tuple[str, ...] = ("train", "test")) -> dict[str, np.ndarray]:
    """Standard repository KGE of the exported discharge vs a reference.

    ``reference`` must be [n, time] on the full axis (e.g. q* for R3 dev
    checks, or observed Q for official runs).  Slices use the R1/R2 periods.
    """
    from training.dpl.run_dpl_model import compute_kge_fp64

    slices = period_slices_full(q_full.shape[1])
    result: dict[str, np.ndarray] = {}
    for period in periods:
        sl = slices[period]
        result[period] = np.array([
            compute_kge_fp64(q_full[i, sl], reference[i, sl])
            for i in range(q_full.shape[0])
        ])
    return result
