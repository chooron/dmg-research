#!/usr/bin/env python3
"""Train one hydrological model with the fixed HBV-selected dPL config."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# The shared model implementations use torch.compile for their recurrent
# process kernels.  PyTorch 2.8 can create separate graphs when training and
# validation toggle grad mode and when the final batch has a different size.
# Raise only the compiler cache limits; this does not change the dPL model,
# optimizer, or ablation hyperparameters.
try:
    import torch._dynamo as _dynamo
    _dynamo.config.recompile_limit = max(_dynamo.config.recompile_limit, 256)
    _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 256)
except (ImportError, AttributeError):
    pass

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from models import (  # noqa: E402
    GR4J,
    GR4JLite,
    GR4JWithCemaNeige,
    GR4JWithCemaNeigeLite,
    GR4JWithPrecipitationDelay,
    GR4JWithPrecipitationDelayLite,
    GR4JWithTGD2,
    GR4JWithTGD2Lite,
    HBV,
    HBVLite,
    SIMHYD,
    SIMHYDLite,
    SIMHYDWithCemaNeige,
    SIMHYDWithCemaNeigeLite,
    SIMHYDWithPrecipitationDelay,
    SIMHYDWithPrecipitationDelayLite,
    SIMHYDWithTGD2,
    SIMHYDWithTGD2Lite,
    XAJ,
    XAJLite,
    XAJWithCemaNeige,
    XAJWithCemaNeigeLite,
    XAJWithPrecipitationDelay,
    XAJWithPrecipitationDelayLite,
    XAJWithTGD2,
    XAJWithTGD2Lite,
    XAJ2SWithCemaNeige,
    XAJ2SWithCemaNeigeLite,
    XAJRWPEWithCemaNeige,
    XAJRWPEWithCemaNeigeLite,
    XAJDEWithCemaNeige,
    XAJDEWithCemaNeigeLite,
    XAJGEWithCemaNeige,
    XAJGEWithCemaNeigeLite,
    XAJDRWithCemaNeige,
    XAJDRWithCemaNeigeLite,
    XAJGRWithCemaNeige,
    XAJGRWithCemaNeigeLite,
)
from models.parameter_specs import (  # noqa: E402
    GR4J_CN_PARAM_SPECS,
    GR4J_PD_PARAM_SPECS,
    GR4J_PARAM_SPECS,
    GR4J_TGD2_PARAM_SPECS,
    HBV_PARAM_SPECS,
    SIMHYD_CN_PARAM_SPECS,
    SIMHYD_PD_PARAM_SPECS,
    SIMHYD_PARAM_SPECS,
    SIMHYD_TGD2_PARAM_SPECS,
    TGD2_STRUCTURE_VERSION,
    XAJ_CN_PARAM_SPECS,
    XAJ_2S_PARAM_SPECS,
    XAJ_RWPE_PARAM_SPECS,
    XAJ_PD_PARAM_SPECS,
    XAJ_LITE_PARAM_SPECS,
    XAJ_PARAM_SPECS,
    XAJ_TGD2_PARAM_SPECS,
    CEMANEIGE_PARAM_SPECS,
    XAJ_DE_PARAM_SPECS,
    XAJ_GE_PARAM_SPECS,
    XAJ_DR_PARAM_SPECS,
    XAJ_GR_PARAM_SPECS,
)
from training.data_contract import FORCING_NAMES, load_dates, load_gage_ids  # noqa: E402


MODEL_REGISTRY: dict[str, tuple[type[nn.Module], dict[str, dict[str, Any]]]] = {
    "HBV": (HBV, HBV_PARAM_SPECS),
    "GR4J": (GR4J, GR4J_PARAM_SPECS),
    "XAJ": (XAJ, XAJ_PARAM_SPECS),
    "GR4J_CN": (GR4JWithCemaNeige, GR4J_CN_PARAM_SPECS),
    "GR4J_PD": (GR4JWithPrecipitationDelay, GR4J_PD_PARAM_SPECS),
    "GR4J_TGD2": (GR4JWithTGD2, GR4J_TGD2_PARAM_SPECS),
    "XAJ_CN": (XAJWithCemaNeige, XAJ_CN_PARAM_SPECS),
    "XAJ_D_E_CN": (XAJDEWithCemaNeige, {**CEMANEIGE_PARAM_SPECS, **XAJ_DE_PARAM_SPECS}),
    "XAJ_G_E_CN": (XAJGEWithCemaNeige, {**CEMANEIGE_PARAM_SPECS, **XAJ_GE_PARAM_SPECS}),
    "XAJ_D_R_CN": (XAJDRWithCemaNeige, {**CEMANEIGE_PARAM_SPECS, **XAJ_DR_PARAM_SPECS}),
    "XAJ_G_R_CN": (XAJGRWithCemaNeige, {**CEMANEIGE_PARAM_SPECS, **XAJ_GR_PARAM_SPECS}),
    "XAJ_2S": (XAJ2SWithCemaNeige, XAJ_2S_PARAM_SPECS),
    "XAJ_RWPE": (XAJRWPEWithCemaNeige, XAJ_RWPE_PARAM_SPECS),
    "XAJ_PD": (XAJWithPrecipitationDelay, XAJ_PD_PARAM_SPECS),
    "SIMHYD": (SIMHYD, SIMHYD_PARAM_SPECS),
    "SIMHYD_CN": (SIMHYDWithCemaNeige, SIMHYD_CN_PARAM_SPECS),
    "SIMHYD_PD": (SIMHYDWithPrecipitationDelay, SIMHYD_PD_PARAM_SPECS),
    "SIMHYD_TGD2": (SIMHYDWithTGD2, SIMHYD_TGD2_PARAM_SPECS),
    "XAJ_TGD2": (XAJWithTGD2, XAJ_TGD2_PARAM_SPECS),
}

LITE_MODEL_REGISTRY: dict[str, tuple[type[nn.Module], dict[str, dict[str, Any]]]] = {
    "HBV": (HBVLite, HBV_PARAM_SPECS),
    "GR4J": (GR4JLite, GR4J_PARAM_SPECS),
    "XAJ": (XAJLite, XAJ_LITE_PARAM_SPECS),
    "GR4J_CN": (GR4JWithCemaNeigeLite, GR4J_CN_PARAM_SPECS),
    "GR4J_PD": (GR4JWithPrecipitationDelayLite, GR4J_PD_PARAM_SPECS),
    "GR4J_TGD2": (GR4JWithTGD2Lite, GR4J_TGD2_PARAM_SPECS),
    "XAJ_CN": (XAJWithCemaNeigeLite, XAJ_CN_PARAM_SPECS),
    "XAJ_D_E_CN": (XAJDEWithCemaNeigeLite, {**CEMANEIGE_PARAM_SPECS, **XAJ_DE_PARAM_SPECS}),
    "XAJ_G_E_CN": (XAJGEWithCemaNeigeLite, {**CEMANEIGE_PARAM_SPECS, **XAJ_GE_PARAM_SPECS}),
    "XAJ_D_R_CN": (XAJDRWithCemaNeigeLite, {**CEMANEIGE_PARAM_SPECS, **XAJ_DR_PARAM_SPECS}),
    "XAJ_G_R_CN": (XAJGRWithCemaNeigeLite, {**CEMANEIGE_PARAM_SPECS, **XAJ_GR_PARAM_SPECS}),
    "XAJ_2S": (XAJ2SWithCemaNeigeLite, XAJ_2S_PARAM_SPECS),
    "XAJ_RWPE": (XAJRWPEWithCemaNeigeLite, XAJ_RWPE_PARAM_SPECS),
    "XAJ_PD": (XAJWithPrecipitationDelayLite, XAJ_PD_PARAM_SPECS),
    "SIMHYD": (SIMHYDLite, SIMHYD_PARAM_SPECS),
    "SIMHYD_CN": (SIMHYDWithCemaNeigeLite, SIMHYD_CN_PARAM_SPECS),
    "SIMHYD_PD": (SIMHYDWithPrecipitationDelayLite, SIMHYD_PD_PARAM_SPECS),
    "SIMHYD_TGD2": (SIMHYDWithTGD2Lite, SIMHYD_TGD2_PARAM_SPECS),
    "XAJ_TGD2": (XAJWithTGD2Lite, XAJ_TGD2_PARAM_SPECS),
}


# CAMELS static attribute order stores ``area_gages2`` (km²) at index 11.
# ``camels_dataset`` keeps observed streamflow in the source ft³/s unit,
# whereas the differentiable rainfall-runoff models produce basin-average
# runoff depth in mm/day.
AREA_GAGES2_ATTRIBUTE_INDEX = 11
FT3S_TO_MMD_NUMERATOR = 0.0283168 * 86400.0 * 1000.0


class StaticParameterNet(nn.Module):
    """Static CAMELS attributes to normalized physical-model parameters."""

    def __init__(self, n_attributes: int, specs: dict[str, dict[str, Any]],
                 hidden_sizes: list[int], dropout: float,
                 output_epsilon: float) -> None:
        super().__init__()
        self.parameter_names = list(specs)
        self.output_epsilon = float(output_epsilon)
        if not hidden_sizes or any(int(width) <= 0 for width in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive widths")
        self.hidden_sizes = [int(width) for width in hidden_sizes]
        layers: list[nn.Module] = []
        input_size = n_attributes
        for index, width in enumerate(self.hidden_sizes):
            layers.extend([nn.Linear(input_size, width), nn.LayerNorm(width), nn.SiLU()])
            if index < len(self.hidden_sizes) - 1:
                layers.append(nn.Dropout(dropout))
            input_size = width
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(self.hidden_sizes[-1], len(self.parameter_names))
        self._initialize(specs)

    def _initialize(self, specs: dict[str, dict[str, Any]]) -> None:
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        defaults = []
        for name in self.parameter_names:
            spec = specs[name]
            # Match the released Lite-v2 dPL geometry exactly.  The network
            # always emits a sigmoid-normalized coordinate.  Ordinary
            # parameters are normalized linearly; residence times use their
            # log-physical coordinate before the inverse transform below.
            # This is a dPL convention, distinct from the CMA-ES adapter.
            if name in DPL_LOG_RESIDENCE_PARAMETERS:
                value = (
                    math.log(spec["default"]) - math.log(spec["lower"])
                ) / (math.log(spec["upper"]) - math.log(spec["lower"]))
            else:
                value = (spec["default"] - spec["lower"]) / (spec["upper"] - spec["lower"])
            defaults.append(np.clip(value, self.output_epsilon, 1.0 - self.output_epsilon))
        self.head.bias.data.copy_(torch.logit(torch.tensor(defaults, dtype=torch.float32)))

    def forward(self, attributes: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.trunk(attributes))
        return torch.sigmoid(logits).clamp(self.output_epsilon, 1.0 - self.output_epsilon)


# The historical Lite-v2 artifacts prove that `tgd_tau` used log-space
# denormalization after the sigmoid.  TGD2 has two positive residence times,
# so both use that same dPL-only geometry.  Keep this declaration adjacent to
# the network/physical mapping rather than duplicating it in launch scripts.
DPL_LOG_RESIDENCE_PARAMETERS = frozenset({"tgd_tau_warm", "tgd_delta_tau_cold"})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gate_time_index(config: dict) -> dict[str, tuple[int, int]]:
    dates = pd.to_datetime(load_dates(config["dates_path"]))

    def bounds(name: str) -> tuple[int, int]:
        period = config["time_periods"][name]
        start = pd.Timestamp(period["start"])
        end = pd.Timestamp(period["end"])
        si = int((dates >= start).argmax())
        ei = len(dates) - 1 - int((dates <= end)[::-1].argmax())
        assert dates[si].date() == start.date()
        assert dates[ei].date() == end.date()
        assert ei - si + 1 == period["days"]
        return si, ei

    indices = {name: bounds(name) for name in ("warmup", "calibration", "evaluation")}
    assert indices["calibration"][0] - indices["warmup"][0] == config["window"]["warmup_days"]
    if config.get("allow_evaluation_gap", False):
        assert indices["evaluation"][0] > indices["calibration"][1]
        assert indices["evaluation"][0] >= config["window"]["warmup_days"]
    else:
        assert indices["evaluation"][0] == indices["calibration"][1] + 1
    return indices


def load_data(config: dict, indices: dict[str, tuple[int, int]], max_basins: int | None):
    with open(config["data_basin_ids"]) as handle:
        configured_basin_ids = [str(value).zfill(8) for value in json.load(handle)]
    basin_ids = configured_basin_ids
    if max_basins is not None:
        basin_ids = basin_ids[:max_basins]

    full_ids = load_gage_ids(config["gage_ids_path"])
    id_to_index = {basin_id: index for index, basin_id in enumerate(full_ids)}
    missing_ids = [basin_id for basin_id in basin_ids if basin_id not in id_to_index]
    if missing_ids:
        raise ValueError(f"Basin IDs missing from forcing metadata: {missing_ids[:5]}")
    selected = np.array([id_to_index[basin_id] for basin_id in basin_ids], dtype=np.int64)

    with open(config["data_pkl_dataset"], "rb") as handle:
        dataset_forcing, dataset_target, all_attributes = pickle.load(handle)
    attributes = np.asarray(all_attributes, dtype=np.float32)[selected]

    axis = {"precip": FORCING_NAMES.index("P"),
            "temp": FORCING_NAMES.index("T"),
            "pet": FORCING_NAMES.index("PET")}
    wi_s, _ = indices["warmup"]
    ci_s, ci_e = indices["calibration"]
    ei_s, ei_e = indices["evaluation"]
    assert ci_s - wi_s == config["window"]["warmup_days"]

    data_source = config.get("data_source", "npz_559")
    if data_source == "camels_dataset_pickle":
        # This is the flexmopex CAMELS tensor: [671 basins, time, variables].
        # Select by gauge ID before slicing time so the 531-basin subset remains
        # aligned across forcings, observations, and static attributes.
        forcing = np.asarray(dataset_forcing, dtype=np.float32)[selected]
        target = np.asarray(dataset_target, dtype=np.float32)[selected]
        if forcing.shape[:2] != target.shape[:2]:
            raise ValueError("camels_dataset forcing and target basin/time axes differ")
        if forcing.shape[0] != len(basin_ids):
            raise ValueError("camels_dataset basin selection did not preserve requested IDs")

        train_forcing = {
            key: forcing[:, wi_s:ci_e + 1, axis[key]].copy()
            for key in ("precip", "pet", "temp")
        }
        calibration_obs = convert_streamflow_ft3s_to_mm_day(
            target[:, ci_s:ci_e + 1, 0],
            attributes[:, AREA_GAGES2_ATTRIBUTE_INDEX],
        )
        eval_warmup_start = ei_s - config["window"]["warmup_days"]
        evaluation_forcing = {
            key: forcing[:, eval_warmup_start:ei_e + 1, axis[key]].copy()
            for key in ("precip", "pet", "temp")
        }
        evaluation_obs = convert_streamflow_ft3s_to_mm_day(
            target[:, ei_s:ei_e + 1, 0],
            attributes[:, AREA_GAGES2_ATTRIBUTE_INDEX],
        )
    elif data_source == "npz_559":
        raw = np.load(config["data_npz"], allow_pickle=True)
        forcing = np.asarray(raw["forcing"], dtype=np.float32)
        target = np.asarray(raw["target"], dtype=np.float32)
        if forcing.shape[1] != len(configured_basin_ids):
            raise ValueError(
                "NPZ basin axis does not match requested basin IDs. "
                "Use data_source='camels_dataset_pickle' for CAMELS-531."
            )
        train_forcing = {
            key: forcing[wi_s:ci_e + 1, :len(basin_ids), axis[key]].transpose().copy()
            for key in ("precip", "pet", "temp")
        }
        calibration_obs = target[ci_s:ci_e + 1, :len(basin_ids), 0].transpose().copy()
        eval_warmup_start = ei_s - config["window"]["warmup_days"]
        evaluation_forcing = {
            key: forcing[eval_warmup_start:ei_e + 1, :len(basin_ids), axis[key]].transpose().copy()
            for key in ("precip", "pet", "temp")
        }
        evaluation_obs = target[ei_s:ei_e + 1, :len(basin_ids), 0].transpose().copy()
    else:
        raise ValueError(f"Unsupported data_source: {data_source}")

    # Freeze basin-wise statistics from the calibration portion only.  The
    # forcing windows include warmup, so remove that prefix before computing
    # the statistics.  The same values are carried into evaluation windows.
    warmup_days = ci_s - wi_s
    temp_train = train_forcing["temp"][:, warmup_days:]
    temp_mean_train = temp_train.mean(axis=1).astype(np.float32)
    temp_std_train = temp_train.std(axis=1).astype(np.float32)
    train_forcing["temp_mean_train"] = temp_mean_train
    train_forcing["temp_std_train"] = temp_std_train
    evaluation_forcing["temp_mean_train"] = temp_mean_train.copy()
    evaluation_forcing["temp_std_train"] = temp_std_train.copy()
    return basin_ids, attributes, train_forcing, calibration_obs, evaluation_forcing, evaluation_obs


def robust_normalize(attributes: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values = np.asarray(attributes, dtype=np.float32).copy()
    median = np.nanmedian(values, axis=0)
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = np.take(median, np.where(missing)[1])
    q25, q75 = np.percentile(values, [25, 75], axis=0)
    scale = q75 - q25
    fallback = values.std(axis=0)
    scale[scale < 1e-6] = fallback[scale < 1e-6]
    scale[scale < 1e-6] = 1.0
    return np.clip((values - median) / scale, -5.0, 5.0).astype(np.float32), {
        "median": median, "scale": scale,
    }


def convert_streamflow_ft3s_to_mm_day(
    streamflow: np.ndarray,
    area_km2: np.ndarray,
) -> np.ndarray:
    """Convert CAMELS observed flow from ft³/s to basin-average mm/day."""
    values = np.asarray(streamflow, dtype=np.float32)
    areas = np.asarray(area_km2, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"streamflow must have shape [basin, time], got {values.shape}")
    if areas.ndim != 1 or areas.shape[0] != values.shape[0]:
        raise ValueError(
            f"area_km2 must have shape [{values.shape[0]}], got {areas.shape}"
        )
    if not np.isfinite(areas).all() or (areas <= 0.0).any():
        raise ValueError("area_km2 must contain finite positive drainage areas")
    factor = FT3S_TO_MMD_NUMERATOR / (areas * 1.0e6)
    return values * factor[:, None]


def build_windows(calibration_days: int, warmup_days: int,
                  prediction_days: int, stride_days: int) -> list[tuple[int, int]]:
    windows = []
    start = 0
    while start + prediction_days <= calibration_days:
        windows.append((start, start + prediction_days))
        start += stride_days
    if not windows:
        raise ValueError("No complete training windows fit in calibration period")
    return windows


def bettermodel_training_iterations(
    n_basins: int,
    n_time: int,
    batch_size: int,
    warmup_days: int,
    prediction_days: int,
) -> int:
    """Match bettermodel's random-window iterations-per-epoch calculation."""
    if min(n_basins, n_time, batch_size, prediction_days) < 1:
        raise ValueError("training dimensions must be positive")
    available_time = n_time - warmup_days
    if n_time - prediction_days <= warmup_days:
        raise ValueError("training data cannot fit a warmup plus prediction window")
    probability = batch_size * prediction_days / (n_basins * available_time)
    if probability >= 1.0:
        return 1
    return max(1, int(math.ceil(math.log(0.01) / math.log(1.0 - probability))))


def sample_bettermodel_window(
    n_basins: int,
    n_time: int,
    batch_size: int,
    warmup_days: int,
    prediction_days: int,
    window_catalog: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample basin/time indices for a dPL training batch.

    ``target_starts`` are absolute indices in the train forcing array.  Each
    sample consumes [start-warmup:start+prediction], while its loss target is
    [start:start+prediction].  With ``window_catalog`` supplied, basins are
    still sampled uniformly, but each basin's time index is sampled uniformly
    from its precomputed valid-window catalogue.  This prevents a handful of
    zero-variance observation windows from dominating the mean KGE loss while
    preserving equal basin probability.
    """
    upper = n_time - prediction_days
    if upper <= warmup_days:
        raise ValueError("training data cannot fit a random warmup window")
    basin_index = np.random.randint(0, n_basins, size=batch_size)
    if window_catalog is None:
        target_starts = np.random.randint(warmup_days, upper, size=batch_size)
    else:
        if len(window_catalog) != n_basins:
            raise ValueError("window_catalog must contain one entry per basin")
        target_starts = np.empty(batch_size, dtype=np.int64)
        for i, basin in enumerate(basin_index):
            choices = np.asarray(window_catalog[int(basin)], dtype=np.int64)
            if choices.size == 0:
                raise ValueError(f"window_catalog[{int(basin)}] is empty")
            target_starts[i] = choices[np.random.randint(choices.size)]
    return basin_index.astype(np.int64), target_starts.astype(np.int64)


def build_valid_window_catalog(
    observations: np.ndarray,
    warmup_days: int,
    prediction_days: int,
    *,
    min_valid_points: int = 30,
    min_observation_std: float = 0.05,
) -> tuple[list[np.ndarray], dict[str, float | int]]:
    """Build balanced per-basin catalogues for informative KGE windows.

    The catalogue is based only on calibration observations.  A window is
    eligible when it contains at least ``min_valid_points`` finite,
    non-negative observations and its observed standard deviation is at least
    ``min_observation_std`` (mm/day).  Basin selection remains uniform; only
    the within-basin time selection is filtered.  This is deliberately a
    sampler-level policy, leaving the public KGE formula unchanged.

    The returned starts use the same absolute forcing coordinates as
    :func:`sample_bettermodel_window`.
    """
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"observations must have shape [basin, time], got {values.shape}")
    n_basins, calibration_days = values.shape
    if min_valid_points < 1 or min_valid_points > prediction_days:
        raise ValueError("min_valid_points must be within [1, prediction_days]")
    if not math.isfinite(float(min_observation_std)) or min_observation_std < 0.0:
        raise ValueError("min_observation_std must be finite and non-negative")
    n_windows = calibration_days - prediction_days
    if n_windows < 1:
        raise ValueError("calibration data cannot fit a prediction window")

    valid = np.isfinite(values) & (values >= 0.0)
    clean = np.where(valid, values, 0.0)
    count_cs = np.concatenate(
        (np.zeros((n_basins, 1), dtype=np.float64), np.cumsum(valid, axis=1, dtype=np.float64)),
        axis=1,
    )
    sum_cs = np.concatenate(
        (np.zeros((n_basins, 1), dtype=np.float64), np.cumsum(clean, axis=1)),
        axis=1,
    )
    square_cs = np.concatenate(
        (np.zeros((n_basins, 1), dtype=np.float64), np.cumsum(clean * clean, axis=1)),
        axis=1,
    )
    count = count_cs[:, prediction_days:] - count_cs[:, :-prediction_days]
    total = sum_cs[:, prediction_days:] - sum_cs[:, :-prediction_days]
    square_total = square_cs[:, prediction_days:] - square_cs[:, :-prediction_days]
    safe_count = np.maximum(count, 1.0)
    variance = np.maximum(square_total / safe_count - (total / safe_count) ** 2, 0.0)
    eligible = (count >= min_valid_points) & (variance >= float(min_observation_std) ** 2)

    # Keep every basin represented.  The fallback is only for a basin with no
    # informative window; choose its highest-variance valid windows rather
    # than silently dropping that basin from the uniform basin sampler.
    catalog: list[np.ndarray] = []
    fallback_basins = 0
    for basin in range(n_basins):
        starts = np.flatnonzero(eligible[basin])
        if starts.size == 0:
            fallback_basins += 1
            score = np.where(count[basin] >= min_valid_points, variance[basin], -1.0)
            starts = np.asarray([int(np.argmax(score))], dtype=np.int64)
        catalog.append((starts + warmup_days).astype(np.int64))
    summary = {
        "n_basins": n_basins,
        "candidate_windows_per_basin": n_windows,
        "eligible_windows": int(eligible.sum()),
        "eligible_fraction": float(eligible.mean()),
        "fallback_basins": fallback_basins,
        "min_catalog_windows": int(min(len(item) for item in catalog)),
        "median_catalog_windows": float(np.median([len(item) for item in catalog])),
        "max_catalog_windows": int(max(len(item) for item in catalog)),
        "min_valid_points": int(min_valid_points),
        "min_observation_std": float(min_observation_std),
    }
    return catalog, summary


def kge_per_basin(
    qsim: torch.Tensor,
    qobs: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return differentiable KGE for each basin.

    The epsilon is placed *inside* every square root.  An epsilon added only
    to a quotient denominator makes the forward value finite, but does not
    remove the ``sqrt(0)`` derivative singularity for constant simulations.
    """
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be a finite positive float, got {eps!r}")
    eps_sq = eps * eps
    mask = torch.isfinite(qsim) & torch.isfinite(qobs) & (qobs >= 0.0) & (qsim >= 0.0)
    mask_f = mask.to(qsim.dtype)
    count = mask_f.sum(dim=1).clamp_min(1.0)
    p = torch.where(mask, qsim, torch.zeros_like(qsim))
    o = torch.where(mask, qobs, torch.zeros_like(qobs))
    mean_p = p.sum(dim=1) / count
    mean_o = o.sum(dim=1) / count
    dp = (p - mean_p[:, None]) * mask_f
    do = (o - mean_o[:, None]) * mask_f
    sim_ss = dp.square().sum(dim=1)
    obs_ss = do.square().sum(dim=1)
    std_p = torch.sqrt(sim_ss / count + eps_sq)
    std_o = torch.sqrt(obs_ss / count + eps_sq)
    covariance = (dp * do).sum(dim=1) / count
    # Reuse the floored standard deviations, so correlation is also defined
    # when either series has zero variance.
    r = covariance / (std_p * std_o)
    alpha = std_p / std_o
    beta = mean_p / (mean_o + eps)
    distance_sq = (
        (r - 1.0).square()
        + (alpha - 1.0).square()
        + (beta - 1.0).square()
        + eps_sq
    )
    return 1.0 - torch.sqrt(distance_sq)


def physical_parameters(theta: torch.Tensor, names: list[str], lower: torch.Tensor,
                        parameter_range: torch.Tensor) -> dict[str, torch.Tensor]:
    """Map normalized network outputs to physical parameters.

    dPL follows Lite-v2: sigmoid-normalized outputs are linearly denormalized
    except positive TGD2 residence times, which are inverse-log-normalized.
    This is not CMA-ES coordinate logic: the sigmoid is part of the dPL net.
    """
    physical = lower + theta * parameter_range
    for index, name in enumerate(names):
        if name in DPL_LOG_RESIDENCE_PARAMETERS:
            upper = lower[index] + parameter_range[index]
            physical[:, index] = torch.exp(
                torch.log(lower[index])
                + theta[:, index] * (torch.log(upper) - torch.log(lower[index]))
            )
    result = {name: physical[:, index] for index, name in enumerate(names)}
    return result


def compute_kge_fp64(sim: np.ndarray, obs: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim) & (obs >= 0) & (sim >= 0)
    if int(mask.sum()) < 30:
        return -999.0
    s = sim[mask].astype(np.float64)
    o = obs[mask].astype(np.float64)
    o_std = o.std()
    if o_std < 1e-10:
        return -999.0
    r = np.corrcoef(s, o)[0, 1]
    alpha = s.std() / o_std
    beta = s.mean() / o.mean()
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def evaluate(net: nn.Module, model_cls: type[nn.Module], specs: dict[str, dict[str, Any]],
             attributes: torch.Tensor, forcing: dict[str, np.ndarray], observations: np.ndarray,
             batch_size: int, device: torch.device, warmup_days: int):
    net.eval()
    model = model_cls().to(device)
    names = list(specs)
    lower = torch.tensor([specs[name]["lower"] for name in names], device=device, dtype=torch.float64)
    upper = torch.tensor([specs[name]["upper"] for name in names], device=device, dtype=torch.float64)
    parameter_range = upper - lower
    kges = np.full(attributes.shape[0], np.nan, dtype=np.float64)
    parameters = np.full((attributes.shape[0], len(names)), np.nan, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, attributes.shape[0], batch_size):
            stop = min(start + batch_size, attributes.shape[0])
            theta = net(attributes[start:stop].to(device)).to(torch.float64)
            params = physical_parameters(theta, names, lower, parameter_range)
            fc = {key: torch.from_numpy(value[start:stop].copy()).to(device=device, dtype=torch.float64)
                  for key, value in forcing.items()}
            qsim, _ = model(forcings=fc, params=params)
            parameters[start:stop] = theta.cpu().numpy()
            q_np = qsim[:, warmup_days:].cpu().numpy()
            for local, basin in enumerate(range(start, stop)):
                kges[basin] = compute_kge_fp64(q_np[local], observations[basin])
    return kges, parameters, model


def latest_checkpoint(output_dir: Path) -> Path | None:
    """Return the newest periodic checkpoint, falling back to best_checkpoint."""
    periodic = []
    for path in output_dir.glob("checkpoint_epoch_*.pt"):
        try:
            epoch = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        periodic.append((epoch, path))
    if periodic:
        return max(periodic, key=lambda item: item[0])[1]
    best = output_dir / "best_checkpoint.pt"
    return best if best.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), required=True)
    parser.add_argument("--max-basins", type=int)
    parser.add_argument("--max-windows", type=int,
                        help="Limit random training batches per epoch for a fast smoke run.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int,
                        help="Override training.seed for an independent run.")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the newest checkpoint in output_dir.")
    parser.add_argument("--lite", action="store_true",
                        help="Use streamflow-only Lite model implementations.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    config["model_name"] = args.model
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    output_dir = PROJECT_DIR / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = LITE_MODEL_REGISTRY if args.lite else MODEL_REGISTRY
    model_cls, specs = registry[args.model]
    config["lite_mode"] = bool(args.lite)
    existing_config_path = output_dir / "config.json"
    if args.resume and existing_config_path.exists():
        existing_config = json.loads(existing_config_path.read_text())
        existing_model = existing_config.get("model_name")
        existing_lite = existing_config.get("lite_mode")
        if existing_model is not None and existing_model != args.model:
            raise RuntimeError(
                f"Cannot resume {args.model} from output configured for {existing_model}"
            )
        if existing_lite is not None and bool(existing_lite) != bool(args.lite):
            raise RuntimeError(
                "Cannot resume with a different Lite mode than the output config"
            )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    data_dir = PROJECT_DIR.parents[1] / "data"
    if not Path(config.get("data_pkl_dataset", "")).exists():
        if (data_dir / "camels_dataset").exists():
            config["data_pkl_dataset"] = str(data_dir / "camels_dataset")
            config["gage_ids_path"] = str(data_dir / "gage_id.npy")
            config["dates_path"] = str(data_dir / "camels_dates.npy")
            config["data_basin_ids"] = str(data_dir / "531sub_id.txt")
        elif Path("/autodl-fs/data/camels_dataset").exists():
            config["data_pkl_dataset"] = "/autodl-fs/data/camels_dataset"
            config["gage_ids_path"] = "/autodl-fs/data/gage_id.npy"
            config["dates_path"] = "/autodl-fs/data/camels_dates.npy"
            config["data_basin_ids"] = "/autodl-fs/data/531sub_id.txt"
    set_seed(config["training"]["seed"])
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    indices = gate_time_index(config)
    basin_ids, raw_attrs, train_forcing, calibration_obs, eval_forcing, eval_obs = load_data(
        config, indices, args.max_basins
    )
    attrs_np, attr_stats = robust_normalize(raw_attrs)
    np.savez_compressed(output_dir / "attribute_normalization.npz", **attr_stats)
    attributes = torch.from_numpy(attrs_np)

    win = config["window"]
    warmup_days = int(win["warmup_days"])
    prediction_days = int(win["prediction_days"])
    assert train_forcing["precip"].shape[1] == calibration_obs.shape[1] + win["warmup_days"]
    sampling_cfg = config.get("sampling", {})
    window_catalog = None
    if sampling_cfg.get("strategy") == "balanced_valid_kge_windows":
        window_catalog, sampling_summary = build_valid_window_catalog(
            calibration_obs,
            warmup_days,
            prediction_days,
            min_valid_points=int(sampling_cfg.get("min_valid_points", 30)),
            min_observation_std=float(sampling_cfg.get("min_observation_std", 0.05)),
        )
        config["sampling_summary"] = sampling_summary
        (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(
            "Sampling strategy=balanced_valid_kge_windows "
            f"eligible={sampling_summary['eligible_fraction']:.3%} "
            f"catalog_windows={sampling_summary['min_catalog_windows']}"
            f"/{sampling_summary['median_catalog_windows']:.0f}"
            f"/{sampling_summary['max_catalog_windows']} "
            f"fallback_basins={sampling_summary['fallback_basins']} "
            f"min_valid_points={sampling_summary['min_valid_points']} "
            f"min_observation_std={sampling_summary['min_observation_std']:.6g}",
            flush=True,
        )
    names = list(specs)
    config["parameter_names"] = names
    config["parameter_specs"] = specs
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    tgd_structure_version = config.get("tgd_structure_version")
    if tgd_structure_version is None:
        tgd_structure_version = config.get("network", {}).get("tgd_structure_version")
    if args.model.endswith("_TGD2"):
        tgd_structure_version = TGD2_STRUCTURE_VERSION
    checkpoint_metadata = {
        "model_name": args.model,
        "model_class": model_cls.__name__,
        "lite_mode": bool(args.lite),
        "tgd_structure_version": tgd_structure_version if args.model.endswith("_TGD2") else None,
        "parameter_names": names,
        "parameter_specs": specs,
        "model_structure_version": getattr(model_cls, "checkpoint_schema", None),
    }
    lower = torch.tensor([specs[name]["lower"] for name in names], device=device, dtype=torch.float32)
    upper = torch.tensor([specs[name]["upper"] for name in names], device=device, dtype=torch.float32)
    parameter_range = upper - lower
    net_cfg = config["network"]
    hidden_sizes = [int(v) for v in net_cfg.get("hidden_sizes", [net_cfg["hidden_size"]] * net_cfg.get("depth", 2))]
    net = StaticParameterNet(attributes.shape[1], specs, hidden_sizes, net_cfg["dropout"], net_cfg["output_epsilon"]).to(device)
    model = model_cls().to(device)
    train_cfg = config["training"]
    sampling_batch_size = min(int(train_cfg["batch_size"]), len(basin_ids))
    iterations_per_epoch = bettermodel_training_iterations(
        len(basin_ids), train_forcing["precip"].shape[1], sampling_batch_size,
        warmup_days, prediction_days,
    )
    if args.max_windows is not None:
        if args.max_windows < 1:
            parser.error("--max-windows must be positive")
        iterations_per_epoch = min(iterations_per_epoch, args.max_windows)
    optimizer = torch.optim.AdamW(net.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"], eta_min=train_cfg["min_lr"]
    )

    start_epoch = 1
    history = []
    best_state = None
    best_validation = -np.inf
    if args.resume:
        checkpoint_path = latest_checkpoint(output_dir)
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            def validate_checkpoint_metadata(candidate: dict[str, Any], path: Path) -> None:
                if candidate.get("model_name") not in (None, args.model):
                    raise RuntimeError(
                        f"Checkpoint {path.name} belongs to model "
                        f"{candidate.get('model_name')}, not {args.model}"
                    )
                checkpoint_lite = candidate.get("lite_mode")
                if checkpoint_lite is None:
                    if args.lite:
                        raise RuntimeError(
                            f"Checkpoint {path.name} has no Lite-mode metadata; "
                            "refusing to resume it as a Lite model"
                        )
                elif bool(checkpoint_lite) != bool(args.lite):
                    raise RuntimeError(
                        f"Checkpoint {path.name} Lite mode does not match the requested mode"
                    )
                checkpoint_class = candidate.get("model_class")
                if checkpoint_class is not None and checkpoint_class != model_cls.__name__:
                    raise RuntimeError(
                        f"Checkpoint {path.name} class {checkpoint_class} does not match "
                        f"requested class {model_cls.__name__}"
                    )
                if args.model.endswith("_TGD2") and candidate.get("tgd_structure_version") != TGD2_STRUCTURE_VERSION:
                    raise RuntimeError(
                        f"Checkpoint {path.name} has incompatible TGD structure version "
                        f"{candidate.get('tgd_structure_version')!r}; expected {TGD2_STRUCTURE_VERSION!r}"
                    )
                expected_structure = getattr(model_cls, "checkpoint_schema", None)
                if expected_structure is not None and candidate.get("model_structure_version") != expected_structure:
                    validator = getattr(model_cls, "validate_checkpoint_schema", None)
                    if validator is not None:
                        validator(candidate.get("model_structure_version"))
                    raise RuntimeError(
                        f"Checkpoint {path.name} has incompatible model structure version "
                        f"{candidate.get('model_structure_version')!r}; expected {expected_structure!r}"
                    )
                if candidate.get("parameter_names") not in (None, names):
                    raise RuntimeError(f"Checkpoint {path.name} parameter names do not match")
                if candidate.get("parameter_specs") not in (None, specs):
                    raise RuntimeError(f"Checkpoint {path.name} parameter specs do not match")

            validate_checkpoint_metadata(checkpoint, checkpoint_path)
            net.load_state_dict(checkpoint["state_dict"])
            checkpoint_epoch = int(checkpoint.get("epoch", 0))
            start_epoch = checkpoint_epoch + 1

            history_path = output_dir / "epoch_history.csv"
            if history_path.exists():
                history = pd.read_csv(history_path).to_dict("records")

            best_checkpoint_path = output_dir / "best_checkpoint.pt"
            best_checkpoint = checkpoint
            if best_checkpoint_path.exists():
                best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
                validate_checkpoint_metadata(best_checkpoint, best_checkpoint_path)
            best_validation = float(best_checkpoint.get("val_kge_median", -np.inf))
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in best_checkpoint["state_dict"].items()
            }

            # New checkpoints carry optimizer/scheduler state.  Older remote
            # checkpoints only carry the network, so reconstruct the cosine
            # learning-rate position for those files.
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                scheduler.last_epoch = checkpoint_epoch
                scheduler._step_count = checkpoint_epoch + 1
                for group, lr in zip(
                    optimizer.param_groups, scheduler._get_closed_form_lr()
                ):
                    group["lr"] = lr
            print(
                f"RESUME checkpoint={checkpoint_path.name} "
                f"checkpoint_epoch={checkpoint_epoch} start_epoch={start_epoch}",
                flush=True,
            )
        else:
            print("RESUME requested but no checkpoint was found; starting fresh", flush=True)

    print(f"DPL model={args.model} basins={len(basin_ids)} params={len(names)} device={device}", flush=True)
    print(
        f"Random windows/epoch={iterations_per_epoch} × "
        f"warmup={warmup_days} + prediction={prediction_days} "
        f"(batch={sampling_batch_size})",
        flush=True,
    )
    print(f"Network=35→{'→'.join(map(str, hidden_sizes))}→{len(names)} lr={train_cfg['lr']} epochs={train_cfg['epochs']}", flush=True)

    started = time.time()
    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        net.train()
        losses = []
        finite_batches = 0
        for _ in range(iterations_per_epoch):
            batch_index, target_start = sample_bettermodel_window(
                len(basin_ids), train_forcing["precip"].shape[1],
                sampling_batch_size, warmup_days, prediction_days,
                window_catalog=window_catalog,
            )
            forcing_offsets = np.arange(-warmup_days, prediction_days, dtype=np.int64)
            target_offsets = np.arange(prediction_days, dtype=np.int64)
            forcing_index = target_start[:, None] + forcing_offsets[None, :]
            target_index = (target_start - warmup_days)[:, None] + target_offsets[None, :]
            optimizer.zero_grad(set_to_none=True)
            x = attributes[batch_index].to(device)
            theta = net(x)
            params = physical_parameters(theta, names, lower, parameter_range)
            fc = {
                key: torch.from_numpy(train_forcing[key][batch_index[:, None], forcing_index].copy()).to(device)
                for key in ("precip", "pet", "temp")
            }
            for key in ("temp_mean_train", "temp_std_train"):
                fc[key] = torch.from_numpy(train_forcing[key][batch_index].copy()).to(device)
            obs = torch.from_numpy(calibration_obs[batch_index[:, None], target_index].copy()).to(device)
            qsim, _ = model(forcings=fc, params=params)
            kge = kge_per_basin(qsim[:, warmup_days:], obs)
            valid = torch.isfinite(kge)
            if not valid.any():
                continue
            loss = (1.0 - kge[valid]).mean()
            if not torch.isfinite(loss):
                continue
            loss.backward()
            # Never silently replace invalid gradients with zero: that
            # can leave individual physical parameters permanently
            # untrained.  Fail at the first offending batch instead.
            invalid_gradients = []
            for name, parameter in net.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    invalid_gradients.append(name)
            if invalid_gradients:
                raise FloatingPointError(
                    "Non-finite dPL gradients in batch: "
                    + ", ".join(invalid_gradients)
                )
            torch.nn.utils.clip_grad_norm_(net.parameters(), train_cfg["grad_clip_norm"])
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            finite_batches += 1
        scheduler.step()
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else np.nan,
               "finite_batches": finite_batches, "lr": optimizer.param_groups[0]["lr"],
               "elapsed_s": time.time() - started}
        if epoch == 1 or epoch % train_cfg["validation_interval"] == 0 or epoch == train_cfg["epochs"]:
            val, _, _ = evaluate(net, model_cls, specs, attributes, eval_forcing, eval_obs,
                                 train_cfg["batch_size"], device, warmup_days)
            row["val_kge_mean"] = float(np.nanmean(val))
            row["val_kge_median"] = float(np.nanmedian(val))
            if row["val_kge_median"] > best_validation:
                best_validation = row["val_kge_median"]
                best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
                torch.save({**checkpoint_metadata, "epoch": epoch, "state_dict": best_state,
                            "val_kge_median": best_validation,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict()},
                           output_dir / "best_checkpoint.pt")
        history.append(row)
        checkpoint_interval = int(train_cfg.get("checkpoint_interval", 0))
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
            torch.save(
                {
                    **checkpoint_metadata,
                    "epoch": epoch,
                    "state_dict": {
                        key: value.detach().cpu().clone()
                        for key, value in net.state_dict().items()
                    },
                    "train_loss": row["train_loss"],
                    "val_kge_median": row.get("val_kge_median"),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
            )
            pd.DataFrame(history).to_csv(output_dir / "epoch_history.csv", index=False)
        print(f"epoch={epoch:03d} train_loss={row['train_loss']:.5f} batches={finite_batches} "
              f"lr={row['lr']:.2e}" +
              (f" val_median={row['val_kge_median']:.4f}" if "val_kge_median" in row else ""), flush=True)

    if best_state is not None:
        net.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(output_dir / "epoch_history.csv", index=False)
    val, params_norm, _ = evaluate(net, model_cls, specs, attributes, eval_forcing, eval_obs,
                                   train_cfg["batch_size"], device, warmup_days)
    np.savez_compressed(output_dir / "best_parameters_normalized.npz", params=params_norm)
    # Keep exported physical parameters identical to the forward-pass mapping
    # (notably the dPL inverse-log map for positive TGD2 residence times).
    with torch.no_grad():
        exported = physical_parameters(
            torch.from_numpy(params_norm).to(device=device, dtype=lower.dtype),
            names,
            lower,
            parameter_range,
        )
        physical_np = np.column_stack([
            exported[name].detach().cpu().numpy() for name in names
        ])
    np.savez_compressed(output_dir / "best_parameters_physical.npz", params=physical_np)
    with (output_dir / "basin_final_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["basin_id", "basin_index", "val_kge"])
        writer.writeheader()
        for index, basin_id in enumerate(basin_ids):
            writer.writerow({"basin_id": basin_id, "basin_index": index, "val_kge": val[index]})
    (output_dir / "report.md").write_text(
        f"# dPL {args.model} unified configuration\n\n"
        f"Basins={len(basin_ids)}\n\n"
        f"Random windows/epoch={iterations_per_epoch} × {warmup_days} warmup + "
        f"{prediction_days} prediction\n\n"
        f"Epochs={train_cfg['epochs']}\n\n"
        f"Validation KGE mean={np.nanmean(val):.4f}, median={np.nanmedian(val):.4f}\n"
    )
    (output_dir / "COMPLETE").touch()
    print(f"COMPLETE model={args.model} mean={np.nanmean(val):.4f} median={np.nanmedian(val):.4f}", flush=True)


if __name__ == "__main__":
    main()
