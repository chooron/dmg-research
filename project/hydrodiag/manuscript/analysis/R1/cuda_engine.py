"""Vectorized CUDA computing engine for canonical R1 inferential statistics.

Performs paired basin-level bootstraps, Spearman rank correlations with average-tie
handling, and quantile/median reductions entirely on GPU tensors.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

import torch

# Explicitly limit CPU threads to avoid consuming host resources
torch.set_num_threads(1)


def require_cuda() -> torch.device:
    """Ensure CUDA is available and return the device."""
    if not torch.cuda.is_available():
        raise RuntimeError("Canonical R1 analysis requires CUDA; CPU fallback is disabled")
    return torch.device("cuda")


def file_sha256(path: Path) -> str:
    """Compute the SHA-256 digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def derive_seed(base_seed: int, label: str) -> int:
    """Deterministically derive an integer seed from a base seed and a label string."""
    offset = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "little") % 100_000
    return base_seed + offset


def paired_bootstrap_indices(
    n: int,
    seed: int,
    draws: int = 10_000,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate paired resampling indices of shape (draws, n) on GPU."""
    dev = device or torch.device("cuda")
    generator = torch.Generator(device=dev)
    generator.manual_seed(seed)
    return torch.randint(n, (draws, n), generator=generator, device=dev)


def gpu_median(values: torch.Tensor) -> torch.Tensor:
    """Compute the median of finite values in a 1D tensor."""
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return torch.tensor(float("nan"), device=values.device, dtype=torch.float64)
    return torch.median(finite)


def gpu_quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    """Compute a quantile of finite values in a 1D tensor."""
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return torch.tensor(float("nan"), device=values.device, dtype=torch.float64)
    return torch.quantile(finite, q)


def bootstrap_median_ci(
    values: torch.Tensor,
    seed: int,
    draws: int = 10_000,
    batch: int = 256,
    indices: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paired basin bootstrap for column medians on GPU.

    Returns:
        (median, ci_low_0.025, ci_high_0.975, q25, q75)
    """
    if values.ndim == 1:
        values = values[:, None]
    finite_mask = torch.isfinite(values).all(dim=1)
    val = values[finite_mask]
    n = val.shape[0]
    k = val.shape[1]

    if n == 0:
        nan_vec = torch.full((k,), float("nan"), device=values.device, dtype=torch.float64)
        return nan_vec, nan_vec, nan_vec, nan_vec, nan_vec

    point_median = torch.median(val, dim=0).values
    q25 = torch.quantile(val, 0.25, dim=0)
    q75 = torch.quantile(val, 0.75, dim=0)

    if indices is None:
        indices = paired_bootstrap_indices(n, seed, draws, val.device)
    elif indices.shape != (draws, n):
        raise ValueError(f"indices shape {indices.shape} does not match (draws={draws}, n={n})")

    boot_stats = torch.empty((draws, k), dtype=val.dtype, device=val.device)
    for start in range(0, draws, batch):
        stop = min(start + batch, draws)
        boot_stats[start:stop] = torch.median(val[indices[start:stop]], dim=1).values

    ci_low = torch.quantile(boot_stats, 0.025, dim=0)
    ci_high = torch.quantile(boot_stats, 0.975, dim=0)
    return point_median, ci_low, ci_high, q25, q75


def average_rank(values: torch.Tensor) -> torch.Tensor:
    """Compute average ranks on GPU with tied-rank resolution and NaN preservation.

    Supports 1D (n,) and 2D (batch, n).
    """
    finite = torch.isfinite(values)
    if values.ndim == 1:
        if int(finite.sum()) < 2:
            return torch.full_like(values, float("nan"))
        x = values[finite]
        sx, order = torch.sort(x)
        starts = torch.ones_like(sx, dtype=torch.bool)
        starts[1:] = sx[1:] != sx[:-1]
        group = torch.cumsum(starts.to(torch.int64), dim=0) - 1
        count = int(group[-1].item()) + 1
        counts = torch.zeros(count, device=values.device, dtype=torch.float64).scatter_add_(
            0, group, torch.ones_like(sx, dtype=torch.float64)
        )
        sums = torch.zeros_like(counts).scatter_add_(
            0, group, torch.arange(1, sx.numel() + 1, device=values.device, dtype=torch.float64)
        )
        ranks_sorted = sums[group] / counts[group]
        ranks = torch.full_like(values, float("nan"), dtype=torch.float64)
        ranks[finite] = torch.empty_like(x, dtype=torch.float64).scatter(0, order, ranks_sorted)
        if bool(torch.all(sx == sx[0])):
            ranks[:] = float("nan")
        return ranks

    # 2D case: values shape is (batch, n)
    if values.shape[1] < 2:
        return torch.full_like(values, float("nan"))
    sorted_values, order = torch.sort(values, dim=1)
    starts = torch.ones_like(sorted_values, dtype=torch.bool)
    starts[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    group = torch.cumsum(starts.to(torch.int64), dim=1) - 1
    max_groups = values.shape[1]
    counts = torch.zeros(values.shape[0], max_groups, device=values.device, dtype=torch.float64)
    sums = torch.zeros_like(counts)
    positions = torch.arange(1, values.shape[1] + 1, device=values.device, dtype=torch.float64).expand_as(values)
    counts.scatter_add_(1, group, torch.ones_like(values, dtype=torch.float64))
    sums.scatter_add_(1, group, positions)
    ranked_sorted = sums.gather(1, group) / counts.gather(1, group)
    ranks = torch.empty_like(values, dtype=torch.float64).scatter(1, order, ranked_sorted)
    ranks[~finite] = float("nan")
    constant = finite.sum(1).lt(2) | ((sorted_values[:, 1:] == sorted_values[:, :-1]) | ~finite[:, 1:] | ~finite[:, :-1]).all(1)
    ranks[constant] = float("nan")
    return ranks


def spearman(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Spearman rank correlation between two 1D tensors on GPU."""
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum()) < 2:
        return torch.tensor(float("nan"), device=x.device, dtype=torch.float64)
    rx, ry = average_rank(x[mask]), average_rank(y[mask])
    if not bool(torch.isfinite(rx).all() & torch.isfinite(ry).all()):
        return torch.tensor(float("nan"), device=x.device, dtype=torch.float64)
    dx, dy = rx - rx.mean(), ry - ry.mean()
    denom = torch.sqrt(torch.sum(dx * dx) * torch.sum(dy * dy))
    return torch.sum(dx * dy) / denom if bool(denom > 0) else torch.tensor(float("nan"), device=x.device, dtype=torch.float64)


def spearman_bootstrap(
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    draws: int = 10_000,
    batch: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Spearman rank correlation and its 95% bootstrap CI on GPU."""
    finite = torch.isfinite(x) & torch.isfinite(y)
    x_val, y_val = x[finite], y[finite]
    estimate = spearman(x_val, y_val)
    if x_val.numel() < 2 or not bool(torch.isfinite(estimate)):
        nan = torch.tensor(float("nan"), device=x.device, dtype=torch.float64)
        return estimate, nan, nan

    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    boot = torch.empty(draws, dtype=torch.float64, device=x.device)

    for start in range(0, draws, batch):
        stop = min(start + batch, draws)
        cur_batch = stop - start
        ix = torch.randint(x_val.numel(), (cur_batch, x_val.numel()), generator=generator, device=x.device)
        bx, by = x_val[ix], y_val[ix]
        rx, ry = average_rank(bx), average_rank(by)
        dx = rx - rx.mean(1, keepdim=True)
        dy = ry - ry.mean(1, keepdim=True)
        den = torch.sqrt(torch.sum(dx * dx, 1) * torch.sum(dy * dy, 1))
        boot[start:stop] = torch.where(den > 0, torch.sum(dx * dy, 1) / den, torch.full_like(den, float("nan")))

    return estimate, gpu_quantile(boot, 0.025), gpu_quantile(boot, 0.975)


def endpoint_activity_contrast(
    s1_values: torch.Tensor,
    s5_values: torch.Tensor,
    seed: int,
    draws: int = 10_000,
    batch: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute D_activity = median(S5) - median(S1) and its 95% bootstrap CI on GPU."""
    v1 = s1_values[torch.isfinite(s1_values)]
    v5 = s5_values[torch.isfinite(s5_values)]
    if v1.numel() == 0 or v5.numel() == 0:
        nan = torch.tensor(float("nan"), device=s1_values.device, dtype=torch.float64)
        return nan, nan, nan

    estimate = torch.median(v5) - torch.median(v1)
    generator = torch.Generator(device=s1_values.device)
    generator.manual_seed(seed)
    boot = torch.empty(draws, dtype=torch.float64, device=s1_values.device)

    for start in range(0, draws, batch):
        stop = min(start + batch, draws)
        cur_batch = stop - start
        i1 = torch.randint(v1.numel(), (cur_batch, v1.numel()), generator=generator, device=s1_values.device)
        i5 = torch.randint(v5.numel(), (cur_batch, v5.numel()), generator=generator, device=s5_values.device)
        boot[start:stop] = torch.median(v5[i5], 1).values - torch.median(v1[i1], 1).values

    return estimate, gpu_quantile(boot, 0.025), gpu_quantile(boot, 0.975)
