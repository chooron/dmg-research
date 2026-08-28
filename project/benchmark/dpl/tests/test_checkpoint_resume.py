from __future__ import annotations

import sys
from pathlib import Path

import pytest

import torch

# The production runner imports its sibling src package as a top-level module.
BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from scripts import run_36model_benchmark as runner  # noqa: E402
from src.batched_cmaes import BatchedCMAES  # noqa: E402
from src.checkpoint_guard import validate_canonical_checkpoint  # noqa: E402


def test_checkpoint_generation_and_chunk_sorting(tmp_path: Path) -> None:
    names = [
        "chunk_0_gen_300.pt",
        "chunk_0_gen_95.pt",
        "chunk_0_gen_5.pt",
    ]
    paths = [tmp_path / name for name in names]
    assert [runner.checkpoint_generation(path) for path in sorted(paths, key=runner.checkpoint_generation)] == [5, 95, 300]

    for left in (0, 128, 256, 384, 512):
        (tmp_path / f"chunk_{left}_gen_300.pt").touch()
    assert runner.existing_checkpoint_chunk_size(tmp_path) == 128


def _patch_runner(monkeypatch, tmp_path: Path, generation: int, history: list[float]) -> None:
    monkeypatch.setattr(runner, "BENCHMARK_ROOT", tmp_path)
    monkeypatch.setattr(runner, "load_resolved_config", lambda _path: {
        "optimization": {"generations": 3, "starts": 1},
        "global_seed": 7,
        "data": {"basin_ids": "unused"},
    })
    monkeypatch.setattr(runner, "validate_full_run_config", lambda _resolved: None)
    monkeypatch.setattr(runner, "load_ids", lambda _path: [0, 1])
    monkeypatch.setattr(
        runner,
        "load_repeated_warmup_and_train",
        lambda _ids, _resolved, _device: (
            torch.zeros(6, 2, 3),
            torch.zeros(3, 2),
            {"warmup_total_days": 3},
        ),
    )

    class DummyModel:
        model_name = "flexi"

    monkeypatch.setattr(runner, "build_model", lambda *_args, **_kwargs: DummyModel())
    monkeypatch.setattr(runner.torch, "compile", lambda fn, **_kwargs: fn)
    monkeypatch.setattr(
        runner,
        "compute_streaming_fitness",
        lambda _model, _x, _y, latent, **_kwargs: (
            torch.ones(latent.shape[0], latent.shape[1], latent.shape[2]),
            torch.zeros(latent.shape[0], latent.shape[1], latent.shape[2], dtype=torch.bool),
        ),
    )

    checkpoint_root = tmp_path / "checkpoints" / "resume" / "flexi"
    checkpoint_root.mkdir(parents=True)
    solver = BatchedCMAES(2, 10, 16, stdev_init=0.1, active=True, seed=7, device="cpu")
    solver.set_centers(torch.zeros(2, 10, dtype=torch.float64))
    torch.save(
        {
            "generation": generation,
            "solver": solver.state_dict(),
            "basin_ids": [0, 1],
            "history": history,
        },
        checkpoint_root / f"chunk_0_gen_{generation}.pt",
    )


def test_midrun_resume_advances_only_remaining_generations(monkeypatch, tmp_path: Path) -> None:
    _patch_runner(monkeypatch, tmp_path, generation=1, history=[0.25])
    summary = runner.run_single_model(
        "flexi", "resume", tmp_path / "config.yaml", chunk_size=2, backend="compile", device="cpu"
    )
    assert summary["generations_completed"] == 3
    assert summary["seconds_per_generation"] >= 0.0
    payload = torch.load(
        tmp_path / "checkpoints" / "resume" / "flexi" / "chunk_0_gen_3.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["generation"] == 3
    assert len(payload["history"]) == 3


def test_final_generation_checkpoint_without_done_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    _patch_runner(monkeypatch, tmp_path, generation=3, history=[0.25, 0.5, 0.75])
    summary = runner.run_single_model(
        "flexi", "resume", tmp_path / "config.yaml", chunk_size=2, backend="compile", device="cpu"
    )
    assert summary["generations_completed"] == 3
    assert summary["seconds_per_generation"] == 0.0
    assert summary["candidates_per_second"] == 0.0
    assert (tmp_path / "checkpoints" / "resume" / "flexi" / "DONE").is_file()


def test_resume_rejects_mismatched_basin_ids(monkeypatch, tmp_path: Path) -> None:
    _patch_runner(monkeypatch, tmp_path, generation=1, history=[0.25])
    path = tmp_path / "checkpoints" / "resume" / "flexi" / "chunk_0_gen_1.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["basin_ids"] = [10, 11]
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="basin IDs"):
        runner.run_single_model(
            "flexi", "resume", tmp_path / "config.yaml", chunk_size=2, backend="compile", device="cpu"
        )


def test_canonical_guard_rejects_same_size_wrong_basin_set(tmp_path: Path) -> None:
    model_dir = tmp_path / "flexi"
    model_dir.mkdir()
    path = model_dir / "chunk_0_gen_300.pt"
    torch.save(
        {
            "generation": 300,
            "model": "flexi",
            "basin_ids": [1, 2],
            "solver": {"state": {"best_latent": torch.zeros(1, 10)}},
        },
        path,
    )
    (model_dir / "DONE").write_text("{}\n")
    assert validate_canonical_checkpoint(
        model_dir, model_name="flexi", required_basins=2, required_basin_ids=[1, 2]
    )["passed"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["basin_ids"] = [1, 3]
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="basin ID coverage mismatch"):
        validate_canonical_checkpoint(
            model_dir, model_name="flexi", required_basins=2, required_basin_ids=[1, 2]
        )


def test_legacy_fp32_solver_state_is_promoted_to_fp64() -> None:
    solver = BatchedCMAES(2, 4, 4, stdev_init=0.1, active=True, seed=3, device="cpu")
    payload = solver.state_dict()
    payload["state"] = {
        key: value.float() if isinstance(value, torch.Tensor) and value.is_floating_point() else value
        for key, value in payload["state"].items()
    }
    restored = BatchedCMAES(2, 4, 4, stdev_init=0.1, active=True, seed=3, device="cpu")
    restored.load_state_dict(payload)
    assert all(
        not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.dtype == torch.float64
        for value in restored.state.__dict__.values()
    )


def test_cmaes_repairs_non_positive_covariance() -> None:
    solver = BatchedCMAES(2, 4, 4, stdev_init=0.1, active=True, seed=5, device="cpu")
    solver.state.C[0, 0, 0] = -1.0
    solver._refresh_factor()
    assert torch.isfinite(solver.state.A).all()
    assert torch.linalg.eigvalsh(solver.state.C)[0, 0] > 0.0
