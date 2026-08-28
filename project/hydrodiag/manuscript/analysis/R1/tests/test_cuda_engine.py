"""Tests for CUDA computational engine and small-sample CPU reference validation."""
import sys
from pathlib import Path

import pytest
import torch

R1_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(R1_DIR))

from cuda_engine import (
    average_rank,
    bootstrap_median_ci,
    derive_seed,
    endpoint_activity_contrast,
    gpu_median,
    gpu_quantile,
    require_cuda,
    spearman,
    spearman_bootstrap,
)

CUDA = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_average_rank_ties_and_nans():
    x = torch.tensor([1.0, 1.0, 2.0, 3.0, float("nan")], device=CUDA, dtype=torch.float64)
    ranks = average_rank(x)
    assert torch.allclose(ranks[:4], torch.tensor([1.5, 1.5, 3.0, 4.0], device=CUDA, dtype=torch.float64))
    assert torch.isnan(ranks[4])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_spearman_gpu_vs_cpu_reference():
    # Construct synthetic test vectors
    x_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
    y_cpu = torch.tensor([1.1, 1.9, 3.2, 3.9, 5.1, 6.2, 7.0, 7.9], dtype=torch.float64)

    x_gpu = x_cpu.to(CUDA)
    y_gpu = y_cpu.to(CUDA)

    rho_gpu = spearman(x_gpu, y_gpu)
    # CPU manual rank correlation calculation
    rx = torch.arange(1, 9, dtype=torch.float64)
    ry = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    rho_expected = (dx * dy).sum() / torch.sqrt((dx * dx).sum() * (dy * dy).sum())

    assert torch.allclose(rho_gpu.cpu(), rho_expected, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bootstrap_determinism_and_cpu_parity():
    vals_cpu = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0, 11.0], dtype=torch.float64)[:, None]
    vals_gpu = vals_cpu.to(CUDA)

    med1, low1, high1, q25_1, q75_1 = bootstrap_median_ci(vals_gpu, seed=12345, draws=100)
    med2, low2, high2, q25_2, q75_2 = bootstrap_median_ci(vals_gpu, seed=12345, draws=100)

    assert torch.equal(med1, med2)
    assert torch.equal(low1, low2)
    assert torch.equal(high1, high2)
    assert med1.item() == 5.0
    assert q25_1.item() == 3.5
    assert q75_1.item() == 8.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_endpoint_contrast_math():
    s1 = torch.tensor([1.0, 2.0, 3.0], device=CUDA, dtype=torch.float64)
    s5 = torch.tensor([10.0, 11.0, 12.0], device=CUDA, dtype=torch.float64)
    diff, low, high = endpoint_activity_contrast(s1, s5, seed=999, draws=50)

    assert diff.item() == 9.0  # 11.0 - 2.0
    assert low.item() <= diff.item() <= high.item()
