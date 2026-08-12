"""Torch-only differentiable streamflow signatures for physical-theta sensitivity."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

import torch

DEFAULT_MEAN_ANNUAL_PEAK_TAU = 4096.0
DEFAULT_BASEFLOW_ALPHA = 0.925
DEFAULT_RECESSION_WEIGHT_SCALE = 5.0e-4
DEFAULT_EPS = 1.0e-6


def _validate_q(q: torch.Tensor) -> torch.Tensor:
    if q.ndim != 3 or q.shape[-1] != 1:
        raise ValueError(f"Expected Q with shape [T, B, 1], got {tuple(q.shape)}.")
    return q[..., 0]


def _default_group_ids(length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(length, device=device, dtype=torch.long) // 365


def _normalize_group_ids(group_ids: torch.Tensor | None, length: int, device: torch.device) -> torch.Tensor:
    if group_ids is None:
        return _default_group_ids(length, device)
    if group_ids.ndim != 1 or group_ids.shape[0] != length:
        raise ValueError(
            "group_ids must have shape [T]. "
            f"Got {tuple(group_ids.shape)} for T={length}."
        )
    return group_ids.to(device=device, dtype=torch.long)


def water_year_ids_from_dates(dates: Sequence[date | datetime]) -> torch.Tensor:
    values = []
    for item in dates:
        current = item.date() if isinstance(item, datetime) else item
        water_year = current.year + 1 if current.month >= 10 else current.year
        values.append(water_year)
    return torch.tensor(values, dtype=torch.long)


def _iter_group_masks(group_ids: torch.Tensor) -> list[torch.Tensor]:
    unique_ids = torch.unique_consecutive(group_ids)
    return [group_ids == group_id for group_id in unique_ids]


def total_runoff_volume(q: torch.Tensor) -> torch.Tensor:
    return _validate_q(q).sum(dim=0)


def annual_peak_reference(
    q: torch.Tensor,
    water_year_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    q2 = _validate_q(q)
    group_ids = _normalize_group_ids(water_year_ids, q2.shape[0], q2.device)
    peaks = [q2[mask].amax(dim=0) for mask in _iter_group_masks(group_ids)]
    return torch.stack(peaks, dim=0).mean(dim=0)


def mean_annual_peak(
    q: torch.Tensor,
    water_year_ids: torch.Tensor | None = None,
    tau: float = DEFAULT_MEAN_ANNUAL_PEAK_TAU,
) -> torch.Tensor:
    q2 = _validate_q(q)
    group_ids = _normalize_group_ids(water_year_ids, q2.shape[0], q2.device)
    peaks = [torch.logsumexp(tau * q2[mask], dim=0) / tau for mask in _iter_group_masks(group_ids)]
    return torch.stack(peaks, dim=0).mean(dim=0)


def calibrate_mean_annual_peak_tau(
    q: torch.Tensor,
    water_year_ids: torch.Tensor | None = None,
    candidate_taus: Sequence[float] = (
        8.0,
        16.0,
        32.0,
        64.0,
        128.0,
        256.0,
        512.0,
        1024.0,
        2048.0,
        4096.0,
        8192.0,
        16384.0,
    ),
    relative_tolerance: float = 0.01,
    eps: float = DEFAULT_EPS,
) -> dict[str, float | list[dict[str, float]]]:
    hard = annual_peak_reference(q, water_year_ids=water_year_ids)
    candidate_rows: list[dict[str, float]] = []
    chosen = None
    for tau in candidate_taus:
        soft = mean_annual_peak(q, water_year_ids=water_year_ids, tau=float(tau))
        rel = torch.abs(soft - hard) / hard.clamp_min(eps)
        row = {
            "tau": float(tau),
            "max_relative_error": float(rel.max().item()),
            "mean_relative_error": float(rel.mean().item()),
            "median_relative_error": float(rel.median().item()),
        }
        candidate_rows.append(row)
        if chosen is None and row["max_relative_error"] <= relative_tolerance:
            chosen = row
    if chosen is None:
        chosen = min(candidate_rows, key=lambda row: row["max_relative_error"])
    return {
        "selected_tau": float(chosen["tau"]),
        "selected_max_relative_error": float(chosen["max_relative_error"]),
        "selected_mean_relative_error": float(chosen["mean_relative_error"]),
        "selected_median_relative_error": float(chosen["median_relative_error"]),
        "candidates": candidate_rows,
    }


def recession_constant(
    q: torch.Tensor,
    weight_scale: float = DEFAULT_RECESSION_WEIGHT_SCALE,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    q2 = _validate_q(q)
    if q2.shape[0] < 3:
        raise ValueError("recession_constant requires at least three timesteps.")

    q_prev = q2[:-1]
    q_next = q2[1:]
    dq = q_next - q_prev
    scale = weight_scale * q_prev.mean(dim=0, keepdim=True).clamp_min(eps)
    weights = torch.sigmoid(-dq / scale)

    x = torch.arange(1, q2.shape[0], device=q2.device, dtype=q2.dtype).unsqueeze(-1)
    x = x.expand_as(q_next)
    y = torch.log(q_next.clamp_min(eps))

    w_sum = weights.sum(dim=0).clamp_min(eps)
    x_mean = (weights * x).sum(dim=0) / w_sum
    y_mean = (weights * y).sum(dim=0) / w_sum

    x_centered = x - x_mean.unsqueeze(0)
    y_centered = y - y_mean.unsqueeze(0)
    slope = (weights * x_centered * y_centered).sum(dim=0) / (
        (weights * x_centered.pow(2)).sum(dim=0).clamp_min(eps)
    )
    return -slope


def lyne_hollick_baseflow(
    q: torch.Tensor,
    alpha: float = DEFAULT_BASEFLOW_ALPHA,
) -> torch.Tensor:
    q2 = _validate_q(q)
    quickflow = torch.zeros_like(q2)
    for index in range(1, q2.shape[0]):
        quickflow[index] = (
            alpha * quickflow[index - 1]
            + 0.5 * (1.0 + alpha) * (q2[index] - q2[index - 1])
        )
    return q2 - quickflow


def baseflow_index(
    q: torch.Tensor,
    alpha: float = DEFAULT_BASEFLOW_ALPHA,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    q2 = _validate_q(q)
    baseflow = lyne_hollick_baseflow(q, alpha=alpha)
    return baseflow.sum(dim=0) / q2.sum(dim=0).clamp_min(eps)
