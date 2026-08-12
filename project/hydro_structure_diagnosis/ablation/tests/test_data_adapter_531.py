import json
from pathlib import Path

import numpy as np
import pytest

from ablation.ic_core.config import load_resolved_config
from ablation.ic_core.data_adapter import load_531_bundle, read_basin_ids


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "ablation/configs/ic_foundation_531_v1.json"


@pytest.fixture(scope="session")
def bundle():
    config = load_resolved_config(CONFIG, device_override="cpu")
    config["periods"] = {"warmup": {"start": "1988-01-01", "end": "1988-12-31"}, "train": {"start": "1989-01-01", "end": "1998-12-31"}}
    return load_531_bundle(config)


def test_531_basin_count(bundle) -> None:
    assert len(bundle.basin_ids) == 531


def test_531_basin_ids_unique(bundle) -> None:
    assert len(set(bundle.basin_ids)) == 531


def test_531_order_matches_file(bundle) -> None:
    with open(PROJECT_ROOT.parent.parent / "data/531sub_id.txt") as handle:
        expected = [str(value).zfill(8) for value in json.load(handle)]
    assert list(bundle.basin_ids) == expected


def test_dataset_shapes(bundle) -> None:
    assert bundle.forcing.shape == (531, 12418, 3)
    assert bundle.target_cfs.shape == (531, 12418)
    assert bundle.target_mm_day.shape == (531, 12418)
    assert bundle.raw_attributes.shape == (531, 35)


def test_no_559_fallback() -> None:
    with pytest.raises(ValueError):
        read_basin_ids(PROJECT_ROOT / "benchmark/data/559sub_id.txt")


def test_raw_area_not_standardized(bundle) -> None:
    assert bundle.area_field == "area_gages2"
    assert bundle.area_unit == "km2"
    assert np.isfinite(bundle.raw_area_km2).all()
    assert (bundle.raw_area_km2 > 0).all()
    assert float(bundle.raw_area_km2.max()) > 1000.0


def test_target_mm_day_has_expected_unit(bundle) -> None:
    assert bundle.target_unit_raw == "ft3/s"
    assert bundle.target_unit_ic == "mm/day"
    assert np.isfinite(bundle.target_mm_day[bundle.valid_target_mask]).all()
    assert (bundle.target_mm_day[bundle.valid_target_mask] >= 0).all()
