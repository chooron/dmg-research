from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
import sys

REPO_TOP = Path(__file__).resolve().parents[3]
if str(REPO_TOP) not in sys.path:
    sys.path.insert(0, str(REPO_TOP))

import numpy as np
import pytest
import torch
from dmg.core.data.loaders import HydroLoader
from dmg.core.utils.utils import initialize_config
from omegaconf import OmegaConf

from project.parameterize.implements import build_paper_dpl
from project.parameterize.implements.basin_utils import (
    basin_subset_indices,
    load_basin_ids,
    subset_dataset_by_indices,
)
from project.parameterize.implements.differentiable_signatures import (
    DEFAULT_BASEFLOW_ALPHA,
    calibrate_mean_annual_peak_tau,
    mean_annual_peak,
    recession_constant,
    total_runoff_volume,
    water_year_ids_from_dates,
    baseflow_index,
)
from project.parameterize.train_dmotpy import (
    _build_loader_config,
    _normalize_runtime_paths,
    _resolve_path,
)
from project.parameterize.paper_variants import normalize_paper_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "config_param_paper.yaml"
CHECKPOINT_PATH = (
    REPO_ROOT
    / "outputs"
    / "distributional-531"
    / "HybridNseBatchLoss"
    / "seed_111"
    / "model"
    / "model_epoch100.pt"
)
TEST_BASIN_COUNT = 16
SYNTHETIC_TAU_CANDIDATES = (8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0)
REAL_TAU_TARGET = 0.01


@dataclass(frozen=True)
class LoadedRun:
    model: torch.nn.Module
    dataset: dict[str, torch.Tensor]
    eval_dates: tuple[datetime, ...]


def _load_runtime_config() -> dict:
    raw_config = OmegaConf.load(_resolve_path(str(CONFIG_PATH)))
    raw_config["mode"] = "test"
    raw_config["seed"] = 111
    raw_config["device"] = "cpu"
    raw_config["gpu_id"] = 0
    raw_config.setdefault("paper", {})
    raw_config["paper"]["variant"] = "distributional"
    raw_config.setdefault("train", {}).setdefault("loss_function", {})
    raw_config["train"]["loss_function"]["name"] = "HybridNseBatchLoss"
    _normalize_runtime_paths(raw_config)
    normalize_paper_config(raw_config)
    return initialize_config(raw_config)


def _subset_eval_dataset(config: dict) -> tuple[dict[str, torch.Tensor], tuple[datetime, ...]]:
    loader = HydroLoader(_build_loader_config(config), test_split=True, overwrite=False)
    reference_ids = load_basin_ids(config["data"]["basin_ids_reference_path"])
    subset_ids = load_basin_ids(config["data"]["basin_ids_path"])
    subset_idx = basin_subset_indices(reference_ids, subset_ids)
    dataset = subset_dataset_by_indices(loader.eval_dataset, subset_idx)

    start = datetime.strptime(config["test_time"][0], "%Y/%m/%d")
    total_steps = int(dataset["x_phy"].shape[0])
    dates = tuple(start + timedelta(days=index) for index in range(total_steps))
    return dataset, dates


@lru_cache(maxsize=1)
def _loaded_run() -> LoadedRun:
    config = _load_runtime_config()
    dataset, eval_dates = _subset_eval_dataset(config)
    model = build_paper_dpl(config).to("cpu")
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return LoadedRun(model=model, dataset=dataset, eval_dates=eval_dates)


@lru_cache(maxsize=1)
def _acceptance_artifacts() -> dict[str, object]:
    loaded = _loaded_run()
    with torch.no_grad():
        torch.manual_seed(111000)
        parameters = loaded.model.nn_model(loaded.dataset["xc_nn_norm"])
        q_ref = loaded.model.phy_model(loaded.dataset, parameters)["streamflow"]
        physical_theta = loaded.model.phy_model.physical_parameters_from_normalized(parameters)
        q_new = loaded.model.phy_model.forward_from_physical(
            loaded.dataset,
            physical_theta,
        )["streamflow"]
    diff = (q_ref - q_new).abs()
    return {
        "parameters": parameters.detach().clone(),
        "physical_theta": physical_theta.detach().clone(),
        "q_ref": q_ref.detach().clone(),
        "q_new": q_new.detach().clone(),
        "max_diff": float(diff.max().item()),
    }


def _numpy_total_runoff_volume(q: np.ndarray) -> np.ndarray:
    return q[..., 0].sum(axis=0)


def _numpy_mean_annual_peak(q: np.ndarray, water_year_ids: np.ndarray) -> np.ndarray:
    q2 = q[..., 0]
    peaks = []
    for year in np.unique(water_year_ids):
        mask = water_year_ids == year
        peaks.append(q2[mask].max(axis=0))
    return np.stack(peaks, axis=0).mean(axis=0)


def _numpy_recession_constant(q: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    q2 = q[..., 0]
    q_next = q2[1:]
    dq = q_next - q2[:-1]
    weights = (dq < 0.0).astype(np.float64)
    x = np.arange(1, q2.shape[0], dtype=np.float64)[:, None]
    y = np.log(np.clip(q_next, eps, None))
    w_sum = np.clip(weights.sum(axis=0), eps, None)
    x_mean = (weights * x).sum(axis=0) / w_sum
    y_mean = (weights * y).sum(axis=0) / w_sum
    x_centered = x - x_mean[None, :]
    y_centered = y - y_mean[None, :]
    denom = np.clip((weights * x_centered**2).sum(axis=0), eps, None)
    slope = (weights * x_centered * y_centered).sum(axis=0) / denom
    return -slope


def _numpy_baseflow_index(q: np.ndarray, alpha: float = DEFAULT_BASEFLOW_ALPHA) -> np.ndarray:
    q2 = q[..., 0]
    quick = np.zeros_like(q2)
    for index in range(1, q2.shape[0]):
        quick[index] = alpha * quick[index - 1] + 0.5 * (1.0 + alpha) * (q2[index] - q2[index - 1])
    base = q2 - quick
    return base.sum(axis=0) / np.clip(q2.sum(axis=0), 1.0e-6, None)


def _synthetic_signature_series() -> tuple[torch.Tensor, torch.Tensor]:
    years = 4
    days_per_year = 365
    total_days = years * days_per_year
    day = torch.arange(total_days, dtype=torch.float64)
    seasonal = 4.0 + 1.2 * torch.sin(2.0 * np.pi * day / 365.0)
    pulses = torch.zeros_like(day)
    for offset in (45, 410, 775, 1140):
        pulse_day = day - float(offset)
        pulses = pulses + 8.0 * torch.exp(-(pulse_day**2) / (2.0 * 12.0**2))
    basin_a = seasonal + pulses
    basin_b = 0.8 * seasonal + 0.6 * pulses + 0.1 * torch.cos(2.0 * np.pi * day / 90.0)
    q = torch.stack([basin_a, basin_b], dim=1).unsqueeze(-1)

    dates = tuple(datetime(2000, 10, 1) + timedelta(days=index) for index in range(total_days))
    water_year_ids = water_year_ids_from_dates(dates)
    return q, water_year_ids


def _relative_error(actual: torch.Tensor, reference: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    actual_np = actual.detach().cpu().numpy()
    return np.abs(actual_np - reference) / np.clip(np.abs(reference), eps, None)


def _signature_gradients(signature_name: str) -> dict[str, object]:
    loaded = _loaded_run()
    basin_count = min(TEST_BASIN_COUNT, loaded.dataset["x_phy"].shape[1])
    dataset = {
        key: value[:, :basin_count].clone()
        for key, value in loaded.dataset.items()
        if torch.is_tensor(value)
    }
    eval_dates = loaded.eval_dates

    torch.manual_seed(111000)
    parameters = loaded.model.nn_model(dataset["xc_nn_norm"]).detach()
    theta = loaded.model.phy_model.physical_parameters_from_normalized(parameters)
    theta = theta[:basin_count].clone().detach().requires_grad_(True)
    q = loaded.model.phy_model.forward_from_physical(dataset, theta)["streamflow"]

    effective_dates = eval_dates[loaded.model.phy_model.warm_up :]
    water_year_ids = water_year_ids_from_dates(effective_dates).to(q.device)

    if signature_name == "total_runoff_volume":
        signature = total_runoff_volume(q).sum()
    elif signature_name == "mean_annual_peak":
        tau_info = calibrate_mean_annual_peak_tau(
            q.detach(),
            water_year_ids=water_year_ids,
            candidate_taus=SYNTHETIC_TAU_CANDIDATES,
        )
        signature = mean_annual_peak(
            q,
            water_year_ids=water_year_ids,
            tau=float(tau_info["selected_tau"]),
        ).sum()
    elif signature_name == "recession_constant":
        signature = recession_constant(q).sum()
    elif signature_name == "baseflow_index":
        signature = baseflow_index(q).sum()
    else:
        raise KeyError(signature_name)

    grads = torch.autograd.grad(signature, theta)[0]
    per_parameter_max = grads.abs().amax(dim=0).detach().cpu().numpy()
    parameter_names = loaded.model.phy_model.ALL_PARAMETER_NAMES
    nonzero = [name for name, value in zip(parameter_names, per_parameter_max) if value > 1.0e-10]
    dead = [name for name, value in zip(parameter_names, per_parameter_max) if value <= 1.0e-10]
    return {
        "grads": grads.detach(),
        "per_parameter_max": per_parameter_max,
        "nonzero": nonzero,
        "dead": dead,
    }


def test_forward_from_physical_matches_existing_forward_on_checkpoint():
    artifacts = _acceptance_artifacts()
    assert artifacts["q_ref"].shape == artifacts["q_new"].shape
    assert artifacts["max_diff"] < 1.0e-5, f"Acceptance-gate max diff was {artifacts['max_diff']:.8e}."


def test_total_runoff_volume_matches_numpy_reference():
    q, _ = _synthetic_signature_series()
    expected = _numpy_total_runoff_volume(q.numpy())
    actual = total_runoff_volume(q)
    rel = _relative_error(actual, expected)
    assert float(rel.max()) < 1.0e-12


def test_mean_annual_peak_matches_numpy_reference_with_calibrated_tau():
    q, water_year_ids = _synthetic_signature_series()
    expected = _numpy_mean_annual_peak(q.numpy(), water_year_ids.numpy())
    tau_info = calibrate_mean_annual_peak_tau(
        q,
        water_year_ids=water_year_ids,
        candidate_taus=SYNTHETIC_TAU_CANDIDATES,
    )
    actual = mean_annual_peak(q, water_year_ids=water_year_ids, tau=float(tau_info["selected_tau"]))
    rel = _relative_error(actual, expected)
    assert float(rel.max()) <= 0.01


def test_mean_annual_peak_real_checkpoint_calibration_stays_within_one_percent():
    loaded = _loaded_run()
    artifacts = _acceptance_artifacts()
    water_year_ids = water_year_ids_from_dates(
        loaded.eval_dates[loaded.model.phy_model.warm_up :]
    ).to(artifacts["q_ref"].device)
    tau_info = calibrate_mean_annual_peak_tau(
        artifacts["q_ref"],
        water_year_ids=water_year_ids,
    )
    assert float(tau_info["selected_max_relative_error"]) <= REAL_TAU_TARGET


def test_recession_constant_matches_boolean_reference():
    q, _ = _synthetic_signature_series()
    expected = _numpy_recession_constant(q.numpy())
    actual = recession_constant(q)
    rel = _relative_error(actual, expected)
    assert float(rel.max()) < 0.05


def test_baseflow_index_matches_numpy_reference():
    q, _ = _synthetic_signature_series()
    expected = _numpy_baseflow_index(q.numpy())
    actual = baseflow_index(q)
    rel = _relative_error(actual, expected)
    assert float(rel.max()) < 1.0e-12


@pytest.mark.parametrize(
    "signature_name",
    ["total_runoff_volume", "mean_annual_peak", "recession_constant", "baseflow_index"],
)
def test_signature_gradients_are_finite_for_physical_theta(signature_name):
    grad_info = _signature_gradients(signature_name)
    grads = grad_info["grads"]
    assert grads is not None
    assert torch.isfinite(grads).all()
    assert grad_info["nonzero"], f"{signature_name} produced no non-zero parameter gradients."
    assert any(
        name in grad_info["nonzero"]
        for name in ("parBETA", "parFC", "parPERC", "parUZL")
    ), f"{signature_name} did not reach any production parameters."
