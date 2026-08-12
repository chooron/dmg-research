from __future__ import annotations

import pytest
import torch

from tests.dmot_formula_wrappers import WRAPPERS


@pytest.mark.parametrize("candidate_id", sorted(WRAPPERS))
def test_formula_wrapper_outputs_are_finite(candidate_id: str) -> None:
    x = torch.linspace(-2.0, 102.0, 257, dtype=torch.float64, requires_grad=True)
    y = WRAPPERS[candidate_id](x)
    assert torch.isfinite(y).all(), candidate_id
    assert y.shape == x.shape


@pytest.mark.parametrize("candidate_id", sorted(WRAPPERS))
def test_formula_wrapper_gradients_are_finite(candidate_id: str) -> None:
    x = torch.linspace(-2.0, 102.0, 257, dtype=torch.float64, requires_grad=True)
    y = WRAPPERS[candidate_id](x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), candidate_id


@pytest.mark.parametrize("candidate_id", sorted(WRAPPERS))
def test_formula_wrapper_autograd_matches_finite_difference(candidate_id: str) -> None:
    x = torch.tensor([-1.3, 0.3, 3.7, 17.0, 33.0, 67.0, 91.0], dtype=torch.float64, requires_grad=True)
    y = WRAPPERS[candidate_id](x)
    y.sum().backward()
    autograd = x.grad.detach()

    h = 1.0e-5
    with torch.no_grad():
        fd = (WRAPPERS[candidate_id](x.detach() + h) - WRAPPERS[candidate_id](x.detach() - h)) / (2.0 * h)
    assert torch.max(torch.abs(autograd - fd)).item() < 1.0e-4, candidate_id

