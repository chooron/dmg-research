from pathlib import Path

from project.autofuse.data import load_basin_manifest, manifest_hru_ids
from project.autofuse.run_record import manifest_record


def test_hydroshare_544_manifest_is_explicit_and_self_verifying():
    manifest = load_basin_manifest()
    assert manifest["status"] == "recovered"
    assert len(manifest["basins"]) == 544
    assert manifest_hru_ids().shape == (544,)
    assert len(set(manifest_hru_ids().tolist())) == 544


def test_run_record_references_manifest_hash():
    record = manifest_record(Path.cwd())
    assert record["status"] == "available"
    assert record["manifest_sha256"] == "5b3e30d34799921c44e0a3c6bca2c713f89ca83f115ad1a46262defd5d745db3"
    assert record["sha256"]
