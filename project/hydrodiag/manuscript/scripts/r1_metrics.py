"""Small deterministic statistical helpers for the R1 audit."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def kge_prime(sim: np.ndarray, obs: np.ndarray, min_valid: int = 30) -> tuple[float, int]:
    """Return CV-ratio KGE' and valid count using one shared finite mask."""
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = np.isfinite(sim) & np.isfinite(obs) & (sim >= 0) & (obs >= 0)
    n = int(mask.sum())
    if n < min_valid:
        return math.nan, n
    sim_valid, obs_valid = sim[mask], obs[mask]
    mean_obs = float(obs_valid.mean())
    std_obs = float(obs_valid.std())
    if mean_obs == 0.0 or std_obs == 0.0:
        return math.nan, n
    r = float(np.corrcoef(sim_valid, obs_valid)[0, 1])
    cv_sim = float(sim_valid.std() / sim_valid.mean()) if sim_valid.mean() != 0 else math.nan
    cv_obs = float(std_obs / mean_obs)
    if not np.isfinite(r) or not np.isfinite(cv_sim) or not np.isfinite(cv_obs):
        return math.nan, n
    gamma = cv_sim / cv_obs
    beta = float(sim_valid.mean() / mean_obs)
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (gamma - 1.0) ** 2 + (beta - 1.0) ** 2)), n


def finite_values(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return np.sort(array[np.isfinite(array)])


def bootstrap_mean_ci(values: Iterable[float], rng: np.random.Generator, n_resamples: int = 10_000) -> tuple[float, float]:
    values = finite_values(values)
    if values.size == 0:
        return math.nan, math.nan
    indices = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def bootstrap_median_ci(values: Iterable[float], rng: np.random.Generator, n_resamples: int = 10_000) -> tuple[float, float]:
    """Return a paired basin bootstrap interval for the sample median."""
    values = finite_values(values)
    if values.size == 0:
        return math.nan, math.nan
    indices = rng.integers(0, values.size, size=(n_resamples, values.size))
    medians = np.median(values[indices], axis=1)
    return tuple(float(x) for x in np.percentile(medians, [2.5, 97.5]))


def block_bootstrap_mean_ci(
    values: Iterable[float],
    blocks: Iterable[object],
    rng: np.random.Generator,
    n_resamples: int = 10_000,
) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    blocks = np.asarray(list(blocks), dtype=object)
    mask = np.isfinite(values)
    values, blocks = values[mask], blocks[mask]
    if values.size == 0:
        return math.nan, math.nan
    unique_blocks = np.asarray(sorted(set(blocks.tolist())), dtype=object)
    block_values = [values[blocks == block] for block in unique_blocks]
    means = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sampled = rng.integers(0, len(block_values), size=len(block_values))
        means[index] = np.concatenate([block_values[item] for item in sampled]).mean()
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def block_bootstrap_median_ci(
    values: Iterable[float],
    blocks: Iterable[object],
    rng: np.random.Generator,
    n_resamples: int = 10_000,
) -> tuple[float, float]:
    """Return a region-block bootstrap interval for the sample median."""
    values = np.asarray(list(values), dtype=float)
    blocks = np.asarray(list(blocks), dtype=object)
    mask = np.isfinite(values)
    values, blocks = values[mask], blocks[mask]
    if values.size == 0:
        return math.nan, math.nan
    unique_blocks = np.asarray(sorted(set(blocks.tolist())), dtype=object)
    block_values = [values[blocks == block] for block in unique_blocks]
    medians = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sampled = rng.integers(0, len(block_values), size=len(block_values))
        medians[index] = np.median(np.concatenate([block_values[item] for item in sampled]))
    return tuple(float(x) for x in np.percentile(medians, [2.5, 97.5]))


def support_status(ci_low: float, ci_high: float) -> str:
    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        return "descriptive_only"
    if ci_low > 0:
        return "supported_positive"
    if ci_high < 0:
        return "supported_negative"
    return "inconclusive"


def summary(values: Iterable[float], rng: np.random.Generator | None = None) -> dict[str, float]:
    values = finite_values(values)
    result = {
        "valid_basin_count": int(values.size),
        "median": math.nan,
        "p25": math.nan,
        "p75": math.nan,
        "mean": math.nan,
        "sd": math.nan,
        "minimum": math.nan,
        "maximum": math.nan,
        "fraction_positive": math.nan,
        "bootstrap_ci_low": math.nan,
        "bootstrap_ci_high": math.nan,
    }
    if values.size:
        result.update({
            "median": float(np.median(values)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if values.size > 1 else math.nan,
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "fraction_positive": float((values > 0).mean()),
        })
        if rng is not None:
            result["bootstrap_ci_low"], result["bootstrap_ci_high"] = bootstrap_mean_ci(values, rng)
    return result


def rank_relationship(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    """Return Spearman rho and a two-sided p-value without hidden imputation."""
    from scipy.stats import spearmanr

    x, y = np.asarray(list(x), dtype=float), np.asarray(list(y), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return math.nan, math.nan
    result = spearmanr(x[mask], y[mask])
    return float(result.statistic), float(result.pvalue)


def bh_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan)
    mask = np.isfinite(values)
    if not mask.any():
        return adjusted.tolist()
    finite = values[mask]
    order = np.argsort(finite)
    ranked = finite[order] * finite.size / np.arange(1, finite.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    restored = np.empty_like(finite)
    restored[order] = ranked
    adjusted[mask] = restored
    return adjusted.tolist()
