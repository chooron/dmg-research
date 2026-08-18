import numpy as np
import torch
from ablation.ic_core.objective_adapter import KGEObjective


def test_objective_uses_mm_day_and_is_maximized() -> None:
    target = torch.linspace(0.1, 2.0, 100, dtype=torch.float64)
    simulation = target.reshape(1, 1, -1)
    fitness, diagnostics = KGEObjective().evaluate(simulation, target.reshape(1, -1))
    assert np.isclose(fitness.item(), 1.0)
    assert bool(diagnostics["valid"].item())


def test_objective_invalid_candidate_gets_sentinel() -> None:
    target = torch.ones((1, 40), dtype=torch.float64)
    simulation = torch.full((1, 1, 40), float("nan"), dtype=torch.float32)
    fitness, diagnostics = KGEObjective().evaluate(simulation, target)
    assert fitness.item() == -999.0
    assert not bool(diagnostics["valid"].item())


def test_fp32_forward_fp64_metric_path() -> None:
    target = torch.linspace(0.1, 2.0, 100, dtype=torch.float64)
    simulation = target.to(torch.float32).reshape(1, 1, -1)
    fitness, diagnostics = KGEObjective().evaluate(simulation, target.reshape(1, -1))
    assert fitness.dtype == torch.float64
    assert diagnostics["metric_dtype"].item() == 64
