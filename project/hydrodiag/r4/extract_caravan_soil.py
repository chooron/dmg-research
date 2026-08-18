"""Extract and cache Caravan v1.1 CAMELS-US ERA5-Land soil moisture for 531 basins.

Extracts daily volumetric soil water (layers 1..4) and ERA5-Land SWE,
aligns them onto the project 12418-day axis (1980-10-01 .. 2014-09-30),
computes the depth-weighted composites:

    SM100 = 0.07 * L1 + 0.21 * L2 + 0.72 * L3
    SM289 = (0.07 * L1 + 0.21 * L2 + 0.72 * L3 + 1.89 * L4) / 2.89

Outputs:
    results/r4_caravan_soil_reference_v1/
        caravan_soil_ensemble.npz
        manifest.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from r4.common import (  # noqa: E402
    default_data_root,
    default_results_root,
    load_bundle,
    zfill8,
)

STAGE_ROOT = Path("/mnt/c/r4caravan_staging")
OUT_ROOT = default_results_root() / "r4_caravan_soil_reference_v1"


def extract_all(stage_dir: Path, data_root: Path, out_dir: Path) -> dict:
    bundle = load_bundle(data_root)
    basin_ids = [zfill8(b) for b in bundle.basin_ids]
    project_dates = bundle.dates.astype("datetime64[D]")
    n_basins = len(basin_ids)
    n_days = len(project_dates)
    test_sl = slice(5478, 10957)

    print(f"Extracting Caravan soil moisture for {n_basins} basins x {n_days} days...")

    # Allocate arrays
    L1 = np.full((n_basins, n_days), np.nan, dtype=np.float32)
    L2 = np.full((n_basins, n_days), np.nan, dtype=np.float32)
    L3 = np.full((n_basins, n_days), np.nan, dtype=np.float32)
    L4 = np.full((n_basins, n_days), np.nan, dtype=np.float32)
    SWE = np.full((n_basins, n_days), np.nan, dtype=np.float32)

    missing_basins = []

    for i, b in enumerate(basin_ids):
        nc_path = stage_dir / f"camels_{b}.nc"
        if not nc_path.is_file():
            missing_basins.append(b)
            continue
        ds = xr.open_dataset(nc_path)
        file_dates = pd.to_datetime(ds["date"].values).to_numpy(dtype="datetime64[D]")

        # Find overlapping slice
        idx = np.searchsorted(project_dates, file_dates)
        in_range = idx < len(project_dates)
        valid = in_range & (
            project_dates[np.minimum(idx, len(project_dates) - 1)] == file_dates
        )

        proj_indices = idx[valid]

        L1[i, proj_indices] = (
            ds["volumetric_soil_water_layer_1_mean"].values[valid].astype(np.float32)
        )
        L2[i, proj_indices] = (
            ds["volumetric_soil_water_layer_2_mean"].values[valid].astype(np.float32)
        )
        L3[i, proj_indices] = (
            ds["volumetric_soil_water_layer_3_mean"].values[valid].astype(np.float32)
        )
        L4[i, proj_indices] = (
            ds["volumetric_soil_water_layer_4_mean"].values[valid].astype(np.float32)
        )
        SWE[i, proj_indices] = (
            ds["snow_depth_water_equivalent_mean"].values[valid].astype(np.float32)
        )
        ds.close()

    if missing_basins:
        raise RuntimeError(
            f"Missing {len(missing_basins)} Caravan basin files in {stage_dir}: {missing_basins[:5]}"
        )

    # Compute depth-weighted composites
    SM100 = 0.07 * L1 + 0.21 * L2 + 0.72 * L3
    SM289 = (0.07 * L1 + 0.21 * L2 + 0.72 * L3 + 1.89 * L4) / 2.89

    # Validation on test period (1995-10-01 .. 2010-09-30, 5479 days)
    test_days = test_sl.stop - test_sl.start
    nan_l1_test = int(np.isnan(L1[:, test_sl]).sum())
    nan_l2_test = int(np.isnan(L2[:, test_sl]).sum())
    nan_l3_test = int(np.isnan(L3[:, test_sl]).sum())
    nan_l4_test = int(np.isnan(L4[:, test_sl]).sum())

    print(f"Validation on test period ({test_days} days):")
    print(
        f"  L1 NaNs: {nan_l1_test}, L2 NaNs: {nan_l2_test}, L3 NaNs: {nan_l3_test}, L4 NaNs: {nan_l4_test}"
    )
    print(
        f"  SM100 range: [{np.nanmin(SM100[:, test_sl]):.4f}, {np.nanmax(SM100[:, test_sl]):.4f}] m3/m3"
    )
    print(
        f"  SM289 range: [{np.nanmin(SM289[:, test_sl]):.4f}, {np.nanmax(SM289[:, test_sl]):.4f}] m3/m3"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "caravan_soil_ensemble.npz"
    np.savez_compressed(
        npz_path,
        basin_ids=np.asarray(basin_ids),
        dates=project_dates,
        test_slice_start=test_sl.start,
        test_slice_stop=test_sl.stop,
        L1_mean=L1,
        L2_mean=L2,
        L3_mean=L3,
        L4_mean=L4,
        SM100=SM100,
        SM289=SM289,
        caravan_swe=SWE,
    )

    manifest = {
        "dataset": "Caravan v1.1 CAMELS-US subset",
        "description": "ERA5-Land daily basin-averaged volumetric soil water content and SWE",
        "source_path": "G:\\Dataset\\Caravan\\timeseries\\netcdf\\camels\\",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_basins": n_basins,
        "n_days": n_days,
        "test_period": {
            "start": "1995-10-01",
            "end": "2010-09-30",
            "days": test_days,
            "test_slice": [test_sl.start, test_sl.stop],
            "nan_count_test": {
                "L1": nan_l1_test,
                "L2": nan_l2_test,
                "L3": nan_l3_test,
                "L4": nan_l4_test,
            },
        },
        "variables": {
            "L1_mean": "volumetric_soil_water_layer_1_mean (0-7 cm) [m3/m3]",
            "L2_mean": "volumetric_soil_water_layer_2_mean (7-28 cm) [m3/m3]",
            "L3_mean": "volumetric_soil_water_layer_3_mean (28-100 cm) [m3/m3]",
            "L4_mean": "volumetric_soil_water_layer_4_mean (100-289 cm) [m3/m3]",
            "SM100": "0.07 * L1 + 0.21 * L2 + 0.72 * L3 [m3/m3] (0-100 cm depth-weighted composite)",
            "SM289": "(0.07 * L1 + 0.21 * L2 + 0.72 * L3 + 1.89 * L4) / 2.89 [m3/m3] (0-289 cm composite)",
            "caravan_swe": "snow_depth_water_equivalent_mean [mm]",
        },
        "reference_semantics": "external process-state consistency reference; NOT ground-truth soil moisture",
        "file_size_bytes": npz_path.stat().st_size,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Saved: {npz_path} ({manifest['file_size_bytes'] / (1024 * 1024):.2f} MB)")
    return manifest


if __name__ == "__main__":
    extract_all(STAGE_ROOT, default_data_root(), OUT_ROOT)
