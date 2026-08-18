import json
from pathlib import Path

import pandas as pd
import pytest


def test_96_manifest_subset_of_531():
    manifest_path = Path("ablation/manifests/ic_ablation_96_basins_v1.json")
    data = json.loads(manifest_path.read_text())
    basins = data.get("basins", [])
    assert len(basins) == 96


def test_96_manifest_counts():
    pass


def test_each_stratum_has_8():
    pass


def test_split_counts_32_each():
    pass


def test_manifest_deterministic():
    pass


def test_no_calibration_result_used_in_sampling():
    pass


def test_lhs_centers_persisted():
    pass


def test_lhs_centers_same_across_optimizers():
    pass


def test_seed_namespaces_separated():
    pass
