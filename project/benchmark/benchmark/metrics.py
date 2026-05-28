"""Evidence metrics for independent calibration outputs."""

from __future__ import annotations

import math

import numpy as np


def _valid(pred: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(obs)
    return pred[mask], obs[mask]


def nse(pred: np.ndarray, obs: np.ndarray) -> float:
    pred, obs = _valid(pred, obs)
    if len(obs) < 2:
        return math.nan
    den = np.sum((obs - np.mean(obs)) ** 2)
    if den <= 0:
        return math.nan
    return float(1.0 - np.sum((pred - obs) ** 2) / den)


def log_nse(pred: np.ndarray, obs: np.ndarray, eps: float = 1e-6) -> float:
    pred, obs = _valid(pred, obs)
    return nse(np.log(np.clip(pred, 0.0, None) + eps), np.log(np.clip(obs, 0.0, None) + eps))


def kge_components(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    pred, obs = _valid(pred, obs)
    if len(obs) < 2:
        return {"KGE": math.nan, "KGE_r": math.nan, "KGE_alpha": math.nan, "KGE_beta": math.nan}

    pred_mean = float(np.mean(pred))
    obs_mean = float(np.mean(obs))
    pred_std = float(np.std(pred))
    obs_std = float(np.std(obs))
    r = float(np.corrcoef(pred, obs)[0, 1]) if pred_std > 0 and obs_std > 0 else math.nan
    alpha = pred_std / obs_std if obs_std > 0 else math.nan
    beta = pred_mean / obs_mean if abs(obs_mean) > 0 else math.nan
    if any(math.isnan(value) for value in (r, alpha, beta)):
        kge = math.nan
    else:
        kge = 1.0 - math.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return {"KGE": float(kge), "KGE_r": float(r), "KGE_alpha": float(alpha), "KGE_beta": float(beta)}


def flow_diagnostics(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    pred, obs = _valid(pred, obs)
    if len(obs) < 10:
        return {
            "high_flow_bias": math.nan,
            "low_flow_bias": math.nan,
            "rmse_high_flow": math.nan,
            "rmse_low_flow": math.nan,
        }

    high_threshold = np.nanquantile(obs, 0.98)
    low_threshold = np.nanquantile(obs, 0.30)
    high = obs >= high_threshold
    low = obs <= low_threshold
    return {
        "high_flow_bias": _percent_bias(pred[high], obs[high]),
        "low_flow_bias": _percent_bias(pred[low], obs[low], offset=1e-4),
        "rmse_high_flow": _rmse(pred[high], obs[high]),
        "rmse_low_flow": _rmse(pred[low], obs[low]),
    }


def _percent_bias(pred: np.ndarray, obs: np.ndarray, offset: float = 0.0) -> float:
    if len(obs) == 0:
        return math.nan
    den = float(np.sum(obs) + offset)
    if abs(den) <= 0:
        return math.nan
    return float(100.0 * np.sum(pred - obs) / den)


def _rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    if len(obs) == 0:
        return math.nan
    return float(np.sqrt(np.mean((pred - obs) ** 2)))
