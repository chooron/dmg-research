"""Correctness tests for the local basin-held-out OOB implementation."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

OOB_DIR = Path(__file__).resolve().parents[1] / "scripts" / "oob"
sys.path.insert(0, str(OOB_DIR))

from oob_common import (  # noqa: E402
    FOLD_SEED,
    N_FOLDS,
    NormalizationStats,
    BasinDataRepository,
    FoldDataCache,
    LeakageError,
    Phase,
    TargetAccessPolicy,
    fold_ids,
    kge_from_statistics,
    kge_numpy,
    load_oob_config,
    make_fold_assignment,
    require_phase,
    set_phase,
    validate_fold_contract,
)

from run_oob_queue import resolve_devices  # noqa: E402


def test_fold_contract_is_deterministic_and_disjoint() -> None:
    ids = np.arange(1000, 1531, dtype=np.int64)
    rows = make_fold_assignment(ids, n_folds=N_FOLDS, random_state=FOLD_SEED)
    contract = validate_fold_contract(ids, rows, n_folds=N_FOLDS, expected_count=531)
    assert contract.fold_sizes == (107, 106, 106, 106, 106)
    seen: set[int] = set()
    for fold in range(N_FOLDS):
        train, heldout = fold_ids(ids, rows, fold)
        assert not set(train).intersection(heldout)
        assert len(heldout) == contract.fold_sizes[fold]
        seen.update(map(int, heldout))
    assert seen == set(map(int, ids))
    bad = [dict(row) for row in rows]
    first_zero = next(index for index, row in enumerate(bad) if row["fold"] == 0)
    first_one = next(index for index, row in enumerate(bad) if row["fold"] == 1)
    bad[first_zero]["fold"], bad[first_one]["fold"] = 1, 0
    with pytest.raises(ValueError, match="does not match frozen KFold"):
        validate_fold_contract(ids, bad, n_folds=N_FOLDS, expected_count=531)

def test_queue_allows_four_process_slots_on_one_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("run_oob_queue.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("run_oob_queue.torch.cuda.device_count", lambda: 1)
    assert resolve_devices(None, 4) == ["cuda:0"] * 4
    assert resolve_devices("cuda:0,cuda:0,cuda:0,cuda:0", 4) == ["cuda:0"] * 4
    with pytest.raises(ValueError, match="only 1 physical GPU"):
        resolve_devices("cuda:1,cuda:1,cuda:1,cuda:1", 4)

def test_normalization_statistics_are_fit_on_train_only(tmp_path: Path) -> None:
    train = np.tile(np.arange(35, dtype=np.float64), (4, 1))
    train[:, 11] -= 2.0
    heldout = np.tile(np.arange(35, dtype=np.float64) + 100.0, (2, 1))
    stats = NormalizationStats.fit(train)
    transformed_train = stats.transform(train)
    transformed_heldout = stats.transform(heldout)
    assert np.allclose(transformed_train.mean(axis=0), 0.0, atol=1.0e-10)
    assert not np.allclose(transformed_heldout.mean(axis=0), 0.0, atol=1.0e-3)
    stats.save(tmp_path)
    assert np.array_equal(np.load(tmp_path / "attribute_mean.npy"), stats.mean)
    assert json_scope(tmp_path / "normalization_metadata.json") == "train_basins_only"


def json_scope(path: Path) -> str:
    import json

    return json.loads(path.read_text())["scope"]


def test_target_access_policy_blocks_outer_holdout_in_training() -> None:
    train = [1, 2, 3]
    heldout = [4]
    policy = TargetAccessPolicy(Phase.TRAIN, train)
    policy.authorize(train, "test training access")
    with pytest.raises(LeakageError):
        policy.authorize(heldout, "test training access")
    set_phase(Phase.TRAIN)
    with pytest.raises(LeakageError):
        require_phase(Phase.EVAL, "held-out test evaluation")
    set_phase(Phase.TRAIN)


def test_repository_training_loader_returns_only_train_targets(tmp_path: Path) -> None:
    ids = np.arange(10, 16, dtype=np.int64)
    np.save(tmp_path / "gage_id.npy", ids)
    np.savetxt(tmp_path / "basins.txt", ids, fmt="%d")
    attributes = np.ones((len(ids), 35), dtype=np.float64)
    attributes[:, 11] = 100.0
    np.save(tmp_path / "caravan_671_attributes.npy", attributes)
    dates = np.arange(np.datetime64("1980-10-01"), np.datetime64("2010-10-01"), np.timedelta64(1, "D"))
    forcing = np.zeros((len(ids), len(dates), 3), dtype=np.float32)
    target = np.empty((len(ids), len(dates), 1), dtype=np.float64)
    for index, basin_id in enumerate(ids):
        target[index, :, 0] = float(basin_id)
    with (tmp_path / "bundle.pkl").open("wb") as handle:
        pickle.dump((forcing, target, attributes), handle)

    config = {
        "data": {
            "data_path": str(tmp_path / "bundle.pkl"),
            "reference_ids": str(tmp_path / "gage_id.npy"),
            "attributes_path": str(tmp_path / "caravan_671_attributes.npy"),
            "source_start": "1980-10-01",
            "source_end": "2010-09-30",
            "train": {"start_time": "1980-10-01", "end_time": "1995-09-30"},
            "test": {"start_time": "1995-10-01", "end_time": "2010-09-30"},
        },
        "protocol": {"evaluation_warmup_days": 365},
    }
    repository = BasinDataRepository(config)
    train_ids = np.asarray([10, 11], dtype=np.int64)
    heldout_ids = np.asarray([12], dtype=np.int64)
    cache_dir = tmp_path / "cache" / "fold_0"
    cache_dir.mkdir(parents=True)
    train_days, test_days = 5478, 5479
    np.save(cache_dir / "train_x.npy", np.zeros((train_days, 2, 3), dtype=np.float64))
    np.save(cache_dir / "train_y.npy", np.ones((train_days, 2), dtype=np.float64))
    np.save(cache_dir / "heldout_x.npy", np.zeros((365 + test_days, 1, 3), dtype=np.float64))
    np.save(cache_dir / "heldout_y.npy", np.ones((test_days, 1), dtype=np.float64))
    (cache_dir / "cache_metadata.json").write_text(
        '{"train_basin_ids": [10, 11], "heldout_basin_ids": [12]}\n'
    )
    cache = FoldDataCache(tmp_path / "cache", 0, config)
    set_phase(Phase.TRAIN)
    with pytest.raises(LeakageError, match="monolithic canonical bundle"):
        repository.load_training_period(train_ids, TargetAccessPolicy(Phase.TRAIN, train_ids))
    data = cache.load_training_period(train_ids, TargetAccessPolicy(Phase.TRAIN, train_ids))
    assert data.basin_ids.tolist() == [10, 11]
    assert data.x.shape[1] == 2 and data.y.shape[1] == 2
    with pytest.raises(LeakageError):
        repository.load_heldout_evaluation(heldout_ids, TargetAccessPolicy(Phase.EVAL, heldout_ids))
    set_phase(Phase.EVAL)
    heldout = cache.load_heldout_evaluation(heldout_ids, TargetAccessPolicy(Phase.EVAL, heldout_ids))
    assert heldout.basin_ids.tolist() == [12]
    assert heldout.warmup_days == 365
    assert heldout.x.shape[0] == heldout.y.shape[0] + 365
    set_phase(Phase.TRAIN)


def test_protocol_config_freezes_requested_primary8() -> None:
    config = load_oob_config("oob_primary8_5fold_20260902.yaml")
    assert tuple(config["models"]) == (
        "alpine2", "hbv96", "xinanjiang", "newzealand2",
        "ihacres", "us1", "mopex4", "hillslope",
    )
    assert config["folds"] == {
        "n_splits": 5,
        "shuffle": True,
        "random_state": 20260902,
        "assignment_file": "project/benchmark/results/oob_primary8_5fold_20260902/OOB_FOLD_ASSIGNMENT.csv",
        "expected_basin_count": 531,
    }
    assert config["protocol"]["training_warmup_days"] == 730
    assert config["protocol"]["scored_horizon_days"] == 365
    assert config["protocol"]["evaluation_warmup_days"] == 365
    assert config["protocol"]["selection_metric"] == "train_loss"


def test_pooled_kge_uses_combined_sufficient_statistics() -> None:
    observation = np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    prediction = observation * 0.9
    scores, invalid, stats = kge_numpy(prediction, observation)
    assert not invalid.any()
    combined = kge_from_statistics(stats)
    pooled_scores, pooled_invalid, pooled_stats = kge_numpy(prediction.reshape(-1, 1), observation.reshape(-1, 1))
    assert not pooled_invalid.any()
    assert np.isclose(combined, pooled_scores[0])
    assert np.isfinite(scores).all()
