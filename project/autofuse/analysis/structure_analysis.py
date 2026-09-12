"""Analysis helpers for a [basin, structure] score matrix."""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr


def regret(scores: np.ndarray) -> np.ndarray:
    """Per-basin regret relative to the best available structure."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2:
        raise ValueError("scores must have shape [basin, structure]")
    return np.nanmax(values, axis=1, keepdims=True) - values


def equivalent_set(scores: np.ndarray, *, tolerance: float = 0.01) -> list[np.ndarray]:
    """Return near-optimal structure indices for every basin."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2:
        raise ValueError("scores must have shape [basin, structure]")
    best = np.nanmax(values, axis=1, keepdims=True)
    return [np.flatnonzero(row >= best[i, 0] - tolerance) for i, row in enumerate(values)]

def coverage_set(scores: np.ndarray, *, tolerance: float = 0.01) -> np.ndarray:
    """Return the non-dominated near-optimal structure indices across basins."""
    sets = equivalent_set(scores, tolerance=tolerance)
    return np.asarray(sorted({int(index) for values in sets for index in values}), dtype=np.int64)


def rank_correlations(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Spearman/Kendall agreement across paired structure score matrices."""
    left, right = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("reference and candidate must have equal [basin, structure] shapes")
    spearman_values, kendall_values = [], []
    for ref_row, candidate_row in zip(left, right):
        mask = np.isfinite(ref_row) & np.isfinite(candidate_row)
        if mask.sum() < 2:
            continue
        spearman_values.append(float(spearmanr(ref_row[mask], candidate_row[mask]).statistic))
        kendall_values.append(float(kendalltau(ref_row[mask], candidate_row[mask]).statistic))
    return {
        "spearman_median": float(np.median(spearman_values)) if spearman_values else float("nan"),
        "kendall_median": float(np.median(kendall_values)) if kendall_values else float("nan"),
        "n_basins": len(spearman_values),
    }
