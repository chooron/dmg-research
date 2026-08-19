from __future__ import annotations

import sys
from pathlib import Path

import torch

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCHMARK_ROOT.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(BENCHMARK_ROOT), str(BENCHMARK_ROOT / "src")]

from scripts.run_dpl_benchmark_dmg_native import compute_differentiable_kge  # noqa: E402
from src.objective import streaming_kge  # noqa: E402


def test_dpl_kge_matches_canonical_ic_objective() -> None:
    generator = torch.Generator().manual_seed(42)
    observation = torch.rand((64, 4), generator=generator, dtype=torch.float32) + 0.2
    prediction = observation * 0.85 + 0.03 * torch.rand(
        (64, 4), generator=generator, dtype=torch.float32
    )

    expected, invalid = streaming_kge(
        prediction.unsqueeze(-1).unsqueeze(-1), observation, eps=0.1
    )
    actual_loss, actual = compute_differentiable_kge(
        prediction, observation, warmup_days=0
    )

    assert not bool(invalid.any())
    assert torch.allclose(actual, expected.squeeze(-1).squeeze(-1), atol=1e-10, rtol=0.0)
    assert torch.allclose(actual_loss, 1.0 - expected.mean(), atol=1e-10, rtol=0.0)


def test_dpl_kge_keeps_gradient_finite() -> None:
    generator = torch.Generator().manual_seed(7)
    observation = torch.rand((96, 3), generator=generator, dtype=torch.float32) + 0.2
    prediction = (
        torch.rand((96, 3), generator=generator, dtype=torch.float32) + 0.2
    ).requires_grad_()

    loss, _ = compute_differentiable_kge(prediction, observation, warmup_days=0)
    loss.backward()

    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
