from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

from benchmark.metrics import kge_components, log_nse, nse
from benchmark.objectives import objective_loss


def test_metric_perfect_scores() -> None:
    obs = np.array([1.0, 2.0, 3.0])
    pred = obs.copy()
    assert nse(pred, obs) == 1.0
    assert log_nse(pred, obs) == 1.0
    assert kge_components(pred, obs)["KGE"] == 1.0


def test_objective_loss_accepts_multistart_predictions() -> None:
    prediction = torch.tensor([[1.0, 1.1], [2.0, 1.9], [3.0, 3.2]])
    target = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    loss = objective_loss(prediction, target, "nse")
    assert torch.isfinite(loss)
