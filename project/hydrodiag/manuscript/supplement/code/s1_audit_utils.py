from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ATTRIBUTE_NAMES = [
    "p_mean", "pet_mean", "p_seasonality", "frac_snow", "aridity",
    "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur",
    "elev_mean", "slope_mean", "area_gages2", "frac_forest", "lai_max",
    "lai_diff", "gvf_max", "gvf_diff", "dom_land_cover_frac", "dom_land_cover",
    "root_depth_50", "soil_depth_pelletier", "soil_depth_statsgo", "soil_porosity",
    "soil_conductivity", "max_water_content", "sand_frac", "silt_frac", "clay_frac",
    "geol_1st_class", "glim_1st_class_frac", "geol_2nd_class", "glim_2nd_class_frac",
    "carbonate_rocks_frac", "geol_porosity", "geol_permeability",
]
CATEGORICAL_ATTRIBUTES = {"dom_land_cover", "geol_1st_class", "geol_2nd_class"}
CONTINUOUS_ATTRIBUTES = [x for x in ATTRIBUTE_NAMES if x not in CATEGORICAL_ATTRIBUTES]
STRATA = [
    ("S1", 0.00, 0.05, "[0, 0.05)"),
    ("S2", 0.05, 0.15, "[0.05, 0.15)"),
    ("S3", 0.15, 0.30, "[0.15, 0.30)"),
    ("S4", 0.30, 0.50, "[0.30, 0.50)"),
    ("S5", 0.50, 1.00, "[0.50, 1.00]"),
]
PRIMARY_PERIODS = {
    "warmup": {"start": "1980-10-01", "end": "1981-09-30"},
    "calibration": {"start": "1981-10-01", "end": "1995-09-30"},
    "test": {"start": "1995-10-01", "end": "2010-09-30"},
}
CANDIDATE_PERIODS = {
    "warmup": {"start": "1988-01-01", "end": "1988-12-31"},
    "calibration": {"start": "1989-01-01", "end": "1998-12-31"},
    "test": {"start": "1999-01-01", "end": "2009-12-31"},
}


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "status": "UNRESOLVED"}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "format": path.suffix.lstrip(".") or "extensionless",
        "bytes": int(stat.st_size),
        "modified_time": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(),
        "sha256": sha256_file(path),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g")


def qvalues(values: np.ndarray, probabilities=(0, 10, 25, 50, 75, 90, 100)) -> dict[str, float | None]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {f"P{p}": None for p in probabilities}
    q = np.percentile(x, probabilities)
    return {f"P{p}": float(v) for p, v in zip(probabilities, q)}


def read_basin_list(path: Path) -> list[str]:
    values = json.loads(path.read_text())
    ids = [str(v).zfill(8) for v in values]
    return ids


def load_data(data_dir: Path, basin_list: Path) -> dict[str, Any]:
    forcing_path = data_dir / "camels_forcing_v2.pkl"
    dataset_path = data_dir / "camels_dataset"
    with forcing_path.open("rb") as handle:
        metadata = pickle.load(handle)
    with dataset_path.open("rb") as handle:
        dataset_forcing, dataset_target, attributes = pickle.load(handle)
    metadata_forcing = np.asarray(metadata["forcing"])
    dates = np.asarray(metadata["dates"]).astype("datetime64[D]")
    full_ids = [str(v).zfill(8) for v in metadata["basin_ids"]]
    selected_ids = read_basin_list(basin_list)
    # Reproduce the project adapter's P/T signature alignment. PET is not used
    # as an identity key because the two project files have different PET data.
    def sig(row: np.ndarray) -> str:
        return hashlib.sha256(np.asarray(row[:, :2], dtype=np.float32).tobytes()).hexdigest()
    metadata_sig = {sig(metadata_forcing[i]): i for i in range(metadata_forcing.shape[0])}
    dataset_to_metadata = [metadata_sig[sig(np.asarray(dataset_forcing[i]))] for i in range(len(dataset_forcing))]
    dataset_to_metadata_map = {m: i for i, m in enumerate(dataset_to_metadata)}
    metadata_indices = np.array([full_ids.index(b) for b in selected_ids], dtype=np.int64)
    source_indices = np.array([dataset_to_metadata_map[i] for i in metadata_indices], dtype=np.int64)
    selected_forcing = np.asarray(dataset_forcing)[source_indices].astype(np.float64)
    selected_target = np.asarray(dataset_target)[source_indices, :, 0].astype(np.float64)
    selected_attributes = np.asarray(attributes)[source_indices].astype(np.float64)
    return {
        "forcing_path": forcing_path,
        "dataset_path": dataset_path,
        "metadata": metadata,
        "metadata_forcing": metadata_forcing,
        "dataset_forcing": np.asarray(dataset_forcing),
        "dataset_target": np.asarray(dataset_target),
        "attributes": np.asarray(attributes),
        "dates": dates,
        "full_ids": full_ids,
        "basin_ids": selected_ids,
        "source_indices": source_indices,
        "metadata_indices": metadata_indices,
        "forcing": selected_forcing,
        "target_cfs": selected_target,
        "attributes_selected": selected_attributes,
        "dataset_to_metadata": np.asarray(dataset_to_metadata, dtype=np.int64),
    }


def date_slice(dates: np.ndarray, spec: dict[str, str]) -> tuple[int, int, int]:
    start = np.datetime64(spec["start"], "D")
    end = np.datetime64(spec["end"], "D")
    idx = np.flatnonzero((dates >= start) & (dates <= end))
    expected = int((end - start).astype("timedelta64[D]").astype(int) + 1)
    if len(idx) != expected or not np.all(np.diff(dates[idx].astype("int64")) == 1):
        raise ValueError(f"Non-contiguous or incomplete date range: {spec}")
    return int(idx[0]), int(idx[-1]), expected


def period_frame(dates: np.ndarray, routes: dict[str, dict[str, dict[str, str]]]) -> pd.DataFrame:
    rows = []
    for route, periods in routes.items():
        for name, spec in periods.items():
            try:
                first, last, days = date_slice(dates, spec)
                rows.append({"route": route, "period": name, **spec, "first_index": first,
                             "last_index": last, "days": days, "status": "VERIFIED"})
            except Exception as exc:
                rows.append({"route": route, "period": name, **spec, "first_index": None,
                             "last_index": None, "days": None, "status": f"UNRESOLVED: {exc}"})
    return pd.DataFrame(rows)


def convert_flow(q_cfs: np.ndarray, area_km2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Same physical conversion as ablation/ic_core/units.py and dPL's loader.
    factor = 0.028316846592 * 86400.0 * 1000.0 / 1_000_000.0
    q = np.asarray(q_cfs, dtype=float)
    area = np.asarray(area_km2, dtype=float)
    valid = np.isfinite(q) & (q >= 0)
    out = np.full(q.shape, np.nan, dtype=float)
    out = q * (factor / area[:, None])
    out[~valid] = np.nan
    return out, valid


def stratum(value: float) -> str:
    if not np.isfinite(value):
        return "UNRESOLVED"
    if 0 <= value < 0.05:
        return "S1"
    if 0.05 <= value < 0.15:
        return "S2"
    if 0.15 <= value < 0.30:
        return "S3"
    if 0.30 <= value < 0.50:
        return "S4"
    if 0.50 <= value <= 1.0:
        return "S5"
    return "OUT_OF_RANGE"


def fixed_strata(values: np.ndarray) -> np.ndarray:
    return np.array([stratum(float(v)) for v in values], dtype=object)


def git_head(project_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    except Exception:
        return "UNVERIFIED"


def source_line(path: Path, needle: str) -> str:
    try:
        for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if needle in line:
                return f"{path}:{i}"
    except Exception:
        pass
    return f"{path}:UNRESOLVED_LINE"


def safe_mean(values: np.ndarray) -> float | None:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else None


def safe_sum(values: np.ndarray) -> float | None:
    x = np.asarray(values, dtype=float)
    return float(np.sum(x)) if np.all(np.isfinite(x)) else None
