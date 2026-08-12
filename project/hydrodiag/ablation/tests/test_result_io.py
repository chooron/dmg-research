import json

from ablation.ic_core.checkpoint import CheckpointStore
from ablation.ic_core.result_io import atomic_write_json, atomic_write_text


def test_atomic_result_write(tmp_path) -> None:
    target = tmp_path / "nested" / "result.json"
    atomic_write_json(target, {"status": "pass", "value": 1})
    assert json.loads(target.read_text())["value"] == 1


def test_resume_does_not_duplicate_completed_work(tmp_path) -> None:
    store = CheckpointStore(tmp_path / "checkpoint")
    state = {"run_id": "smoke", "completed_evaluations": 8}
    assert not store.is_complete()
    store.mark_complete(state)
    assert store.is_complete()
    assert json.loads(store.complete_path.read_text())["completed_evaluations"] == 8
    store.mark_complete(state)
    assert json.loads(store.state_path.read_text())["completed_evaluations"] == 8


def test_failed_marker_is_atomic(tmp_path) -> None:
    store = CheckpointStore(tmp_path / "checkpoint")
    store.mark_failed({"run_id": "smoke"}, "test failure")
    assert store.failed_path.exists()
    assert json.loads(store.failed_path.read_text())["failure_reason"] == "test failure"
