from __future__ import annotations

import csv
import hashlib
import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from training.data_contract import FORCING_NAMES, load_dates, load_gage_ids

from .periods import resolve_periods
from .schemas import ICDataBundle
from .units import conversion_metadata, convert_ft3s_to_mm_day

ATTRIBUTE_NAMES = (
    "p_mean",
    "pet_mean",
    "p_seasonality",
    "frac_snow",
    "aridity",
    "high_prec_freq",
    "high_prec_dur",
    "low_prec_freq",
    "low_prec_dur",
    "elev_mean",
    "slope_mean",
    "area_gages2",
    "frac_forest",
    "lai_max",
    "lai_diff",
    "gvf_max",
    "gvf_diff",
    "dom_land_cover_frac",
    "dom_land_cover",
    "root_depth_50",
    "soil_depth_pelletier",
    "soil_depth_statsgo",
    "soil_porosity",
    "soil_conductivity",
    "max_water_content",
    "sand_frac",
    "silt_frac",
    "clay_frac",
    "geol_1st_class",
    "glim_1st_class_frac",
    "geol_2nd_class",
    "glim_2nd_class_frac",
    "carbonate_rocks_frac",
    "geol_porosity",
    "geol_permeability",
)
AREA_FIELD = "area_gages2"
AREA_ATTRIBUTE_INDEX = 11


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_basin_ids(path: str | Path) -> list[str]:
    source = Path(path)
    if "559" in source.name:
        raise ValueError("531 adapter refuses a 559 basin list")
    text = source.read_text().strip()
    try:
        values = json.loads(text)
    except json.JSONDecodeError:
        values = [line.strip() for line in text.splitlines() if line.strip()]
    if not isinstance(values, list):
        raise ValueError("basin list must be a JSON list or one ID per line")
    ids = [str(value).zfill(8) for value in values]
    if len(ids) != 531:
        raise ValueError(f"531 adapter requires 531 IDs, got {len(ids)}")
    if len(set(ids)) != len(ids):
        duplicates = sorted({value for value in ids if ids.count(value) > 1})
        raise ValueError(f"duplicate basin IDs: {duplicates}")
    return ids


def _git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip()
    except Exception:
        return "UNVERIFIED"


def load_531_bundle(config: dict[str, Any]) -> ICDataBundle:
    basin_ids = read_basin_ids(config["basin_list_path"])
    with Path(config["dataset_path"]).open("rb") as handle:
        dataset = pickle.load(handle)
    if not isinstance(dataset, tuple) or len(dataset) != 3:
        raise ValueError(
            "camels_dataset must be a pickle tuple (forcing,target,attributes)"
        )
    dataset_forcing, dataset_target, attributes = dataset
    dataset_forcing = np.asarray(dataset_forcing)
    dataset_target = np.asarray(dataset_target)
    attributes = np.asarray(attributes)
    full_ids = load_gage_ids(config["gage_ids_path"])
    if len(full_ids) != dataset_forcing.shape[0]:
        raise ValueError("basin ID count does not match source forcing axis")
    id_to_index = {basin_id: index for index, basin_id in enumerate(full_ids)}
    missing = [basin_id for basin_id in basin_ids if basin_id not in id_to_index]
    if missing:
        raise KeyError(f"531 basin IDs missing from gage_id.npy: {missing}")
    metadata_indices = np.asarray(
        [id_to_index[basin_id] for basin_id in basin_ids], dtype=np.int64
    )
    if dataset_forcing.ndim != 3 or dataset_forcing.shape[2] != 3:
        raise ValueError(
            f"dataset forcing must be [basin,time,3], got {dataset_forcing.shape}"
        )
    if dataset_target.ndim != 3 or dataset_target.shape[2] != 1:
        raise ValueError(
            f"dataset target must be [basin,time,1], got {dataset_target.shape}"
        )
    if attributes.ndim != 2 or attributes.shape[1] != 35:
        raise ValueError(
            f"dataset attributes must be [basin,35], got {attributes.shape}"
        )
    if (
        dataset_forcing.shape[:2] != dataset_target.shape[:2]
        or dataset_forcing.shape[0] != attributes.shape[0]
    ):
        raise ValueError(
            "forcing, target, and attributes do not share source basin/time axes"
        )
    forcing_names = FORCING_NAMES
    # gage_id.npy defines the dataset basin axis.  The tuple has no IDs, so
    # this replaces the old P/T signature join against forcing metadata while
    # preserving the verified source row order.
    source_indices = metadata_indices.copy()
    dates = load_dates(config["dates_path"])
    if dates.shape != (dataset_forcing.shape[1],):
        raise ValueError("date axis does not match dataset time axis")

    selected_forcing = dataset_forcing[source_indices].astype(np.float32, copy=True)
    target_cfs = dataset_target[source_indices, :, 0].astype(np.float64, copy=True)
    raw_attributes = attributes[source_indices].astype(np.float64, copy=True)
    area_km2 = raw_attributes[:, AREA_ATTRIBUTE_INDEX]
    if not np.isfinite(area_km2).all() or (area_km2 <= 0).any():
        raise ValueError("area_gages2 must be finite and positive for all 531 basins")
    target_mm_day, valid_target_mask = convert_ft3s_to_mm_day(
        target_cfs, area_km2, return_valid_mask=True
    )
    if (
        "periods" not in config
        or "warmup" not in config["periods"]
        or "train" not in config["periods"]
    ):
        raise ValueError(
            "Formal runner config must include explicit 'periods' (warmup and train)."
        )

    w_start = np.datetime64(config["periods"]["warmup"]["start"], "D")
    w_end = np.datetime64(config["periods"]["warmup"]["end"], "D")
    calc_warmup_days = int((w_end - w_start).astype("timedelta64[D]").astype(int) + 1)

    periods = resolve_periods(
        dates,
        config["periods"],
        warmup_days=calc_warmup_days,
    )

    input_length = periods.train_forcing_end_index - periods.train_forcing_start_index
    warmup_length = periods.warmup.days
    train_length = periods.train.days

    if input_length <= 0 or warmup_length <= 0 or train_length <= 0:
        raise ValueError(
            f"Hard fail: Invalid period lengths input_length={input_length}, warmup_length={warmup_length}, train_length={train_length}"
        )

    train_temp = selected_forcing[
        :, periods.train.start_index : periods.train.end_index + 1, 1
    ].astype(np.float64)
    temp_mean_train = np.nanmean(train_temp, axis=1).astype(np.float32)
    temp_std_train = np.nanstd(train_temp, axis=1).astype(np.float32)
    if not np.isfinite(temp_mean_train).all() or not np.isfinite(temp_std_train).all():
        raise ValueError("training temperature statistics contain non-finite values")

    source_metadata = {
        "serialization": "pickle tuple (forcing,target,attributes)",
        "forcing_names": list(forcing_names),
        "n_source_basins": int(len(full_ids)),
        "basin_axis_source": "gage_id.npy; dataset row order verified against the 671-basin source",
        "date_axis_source": "camels_dates.npy",
        "forcing_names_source": "training/data_contract.py",
        "dataset_dtypes": {
            "forcing": str(dataset_forcing.dtype),
            "target": str(dataset_target.dtype),
            "attributes": str(attributes.dtype),
        },
        "target_raw_unit_evidence": "training/dpl/run_dpl_model.py:118-124",
        "area_evidence": "training/dpl/run_dpl_model.py:118-124; benchmark/core/data.py area_gages2",
    }
    return ICDataBundle(
        basin_ids=tuple(basin_ids),
        source_indices=source_indices,
        metadata_indices=metadata_indices,
        dates=dates,
        forcing=selected_forcing,
        target_cfs=target_cfs,
        target_mm_day=target_mm_day,
        valid_target_mask=valid_target_mask,
        raw_attributes=raw_attributes,
        raw_area_km2=area_km2.astype(np.float64),
        forcing_names=forcing_names,
        target_unit_raw="ft3/s",
        target_unit_ic="mm/day",
        area_field=AREA_FIELD,
        area_unit="km2",
        periods=periods,
        temp_mean_train=temp_mean_train,
        temp_std_train=temp_std_train,
        source_metadata=source_metadata,
    )


def manifest_for_bundle(bundle: ICDataBundle, config: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(config["project_root"])
    order_hash = hashlib.sha256("\n".join(bundle.basin_ids).encode()).hexdigest()
    return {
        "dataset_path": config["dataset_path"],
        "dataset_file_fingerprint": sha256_file(config["dataset_path"]),
        "gage_ids_path": config["gage_ids_path"],
        "gage_ids_fingerprint": sha256_file(config["gage_ids_path"]),
        "dates_path": config["dates_path"],
        "dates_fingerprint": sha256_file(config["dates_path"]),
        "basin_list_path": config["basin_list_path"],
        "basin_list_fingerprint": sha256_file(config["basin_list_path"]),
        "n_source_basins": int(bundle.source_metadata["n_source_basins"]),
        "n_selected_basins": len(bundle.basin_ids),
        "n_timesteps": int(bundle.forcing.shape[1]),
        "forcing_names": list(bundle.forcing_names),
        "target_raw_unit": bundle.target_unit_raw,
        "target_ic_unit": bundle.target_unit_ic,
        "date_start": str(bundle.dates[0]),
        "date_end": str(bundle.dates[-1]),
        "selected_shapes": {
            "forcing": list(bundle.forcing.shape),
            "target_cfs": list(bundle.target_cfs.shape),
            "target_mm_day": list(bundle.target_mm_day.shape),
            "valid_target_mask": list(bundle.valid_target_mask.shape),
            "attributes": list(bundle.raw_attributes.shape),
        },
        "source_indices_first_last": [
            bundle.source_indices[:5].tolist(),
            bundle.source_indices[-5:].tolist(),
        ],
        "metadata_indices_first_last": [
            bundle.metadata_indices[:5].tolist(),
            bundle.metadata_indices[-5:].tolist(),
        ],
        "area_field": bundle.area_field,
        "area_attribute_index": AREA_ATTRIBUTE_INDEX,
        "area_unit": bundle.area_unit,
        "missing_value_policy": conversion_metadata()["missing_policy"],
        "basin_order_hash": order_hash,
        "periods": bundle.periods.as_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(project_root),
        "source_metadata": bundle.source_metadata,
    }


def write_basin_index_csv(bundle: ICDataBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["selected_position", "basin_id", "source_index"])
        writer.writerows(
            (position, basin_id, int(source_index))
            for position, (basin_id, source_index) in enumerate(
                zip(bundle.basin_ids, bundle.source_indices)
            )
        )


def write_manifest(
    bundle: ICDataBundle, config: dict[str, Any], path: str | Path
) -> dict[str, Any]:
    manifest = manifest_for_bundle(bundle, config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
