from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.autograd import gradcheck

from tests.core_model_registry import CORE_MODEL_REGISTRY, CoreModelEntry
from tests.core_water_balance_utils import _call_step, build_initial_states


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "validation_results" / "model_gradcheck_water_balance_tests"
SUMMARY_CSV_PATH = OUTPUT_DIR / "model_gradcheck_representative_summary.csv"

REPRESENTATIVE_MODELS = ("flexb", "flexi", "flexis", "tcm", "gsfb", "topmodel", "hbv96", "vic", "hymod")
GRADCHECK_EPS = 1.0e-6
GRADCHECK_ATOL = 1.0e-4
GRADCHECK_RTOL = 1.0e-3
GRADCHECK_NONDET_TOL = 0.0

EXPECTED_NONDIFF_REASONS: dict[str, str] = {}
EXPECTED_API_NOT_SUITABLE_REASONS: dict[str, str] = {}


def _make_forcing(entry: CoreModelEntry, n_timesteps: int, dtype: torch.dtype, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (n_timesteps, 1, 1)
    time_axis = torch.arange(n_timesteps, dtype=dtype, device=device).view(n_timesteps, 1, 1)
    precip = (3.5 + 0.3 * time_axis).expand(shape)
    pet = (1.0 + 0.1 * time_axis).expand(shape)

    if entry.model_name in {"flexis", "hbv96"}:
        temperature = torch.full(shape, 8.0, dtype=dtype, device=device)
    else:
        temperature = torch.full(shape, 5.5, dtype=dtype, device=device)
    return precip, temperature, pet


def _parameter_fraction(param_name: str, lower_bound: float) -> float:
    special = {
        "tt": 0.20,
        "ttm": 0.20,
        "tti": 0.30,
        "lp": 0.40,
        "st": 0.40,
        "ndc": 0.40,
        "fsm": 0.40,
        "ishift": 0.20,
    }
    if param_name in special:
        return special[param_name]
    if lower_bound == 0.0:
        return 0.35
    return 0.45


def _base_raw_vector(entry: CoreModelEntry, dtype: torch.dtype, device: str) -> torch.Tensor:
    fractions = [_parameter_fraction(name, float(bounds[0])) for name, bounds in entry.param_bounds.items()]
    return torch.tensor(fractions, dtype=dtype, device=device, requires_grad=True)


def _raw_to_physical_parameters(entry: CoreModelEntry, raw_vector: torch.Tensor) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    params_list: list[torch.Tensor] = []
    params_map: dict[str, torch.Tensor] = {}
    for raw_value, (param_name, (lower_bound, upper_bound)) in zip(raw_vector, entry.param_bounds.items()):
        value = (float(lower_bound) + raw_value * (float(upper_bound) - float(lower_bound))).reshape(1, 1)
        params_list.append(value)
        params_map[param_name.lower()] = value
    return params_list, params_map


def _manual_gradient_magnitude(fn, raw_vector: torch.Tensor) -> float:
    probe = raw_vector.detach().clone().requires_grad_(True)
    value = fn(probe)
    grad = torch.autograd.grad(value, probe, allow_unused=False)[0]
    return float(torch.max(torch.abs(grad)).item())


def _classify_exception(model_name: str, exc: Exception) -> tuple[str, str, str]:
    message = str(exc)
    lowered = message.lower()

    if model_name in EXPECTED_NONDIFF_REASONS:
        return (
            "gradcheck_expected_nondifferentiable_point",
            "expected_nondifferentiable_threshold",
            EXPECTED_NONDIFF_REASONS[model_name],
        )
    if model_name in EXPECTED_API_NOT_SUITABLE_REASONS:
        return (
            "gradcheck_api_not_suitable",
            "expected_discrete_or_stateful_operation",
            EXPECTED_API_NOT_SUITABLE_REASONS[model_name],
        )
    if "nan" in lowered or "inf" in lowered:
        return ("gradcheck_failed_unexpectedly", "unexpected_nan_or_inf", "")
    if "jacobian mismatch" in lowered:
        return ("gradcheck_failed_unexpectedly", "unexpected_gradient_mismatch", "")
    if isinstance(exc, (TypeError, ValueError)):
        return ("gradcheck_failed_unexpectedly", "wrapper_api_issue", "")
    return ("gradcheck_failed_unexpectedly", "unexpected_exception", "")


def evaluate_gradcheck_model(model_name: str) -> dict[str, Any]:
    entry = CORE_MODEL_REGISTRY[model_name]
    dtype = torch.float64
    device = "cpu"
    n_timesteps = 4
    forcing = _make_forcing(entry, n_timesteps, dtype, device)
    raw_vector = _base_raw_vector(entry, dtype, device)
    base_params_list, base_params_map = _raw_to_physical_parameters(entry, raw_vector.detach())
    initial_states = build_initial_states(
        entry,
        "small",
        (1, 1),
        dtype,
        device,
        base_params_map,
        forcing,
        base_params_list,
    )
    initial_states = [state.detach().clone() for state in initial_states]
    mean_precip = forcing[0].mean(dim=0)

    def wrapped_loss(raw: torch.Tensor) -> torch.Tensor:
        params_list, _ = _raw_to_physical_parameters(entry, raw)
        states = [state.clone() for state in initial_states]
        discharge = []
        for step_index in range(n_timesteps):
            qsim, _, states, _ = _call_step(
                entry=entry,
                forcing_at_step=(forcing[0][step_index], forcing[1][step_index], forcing[2][step_index]),
                step_index=step_index,
                params_list=params_list,
                states=states,
                mean_precip=mean_precip,
                return_diagnostics=False,
            )
            discharge.append(qsim)
        discharge_tensor = torch.stack(discharge, dim=0)
        target = torch.full_like(discharge_tensor, 1.25)
        return torch.mean((discharge_tensor - target) ** 2)

    row: dict[str, Any] = {
        "model": model_name,
        "gradcheck_status": "gradcheck_pass",
        "eps": GRADCHECK_EPS,
        "atol": GRADCHECK_ATOL,
        "rtol": GRADCHECK_RTOL,
        "n_timesteps": n_timesteps,
        "n_basins": 1,
        "checked_variables": "raw_parameter_vector(all_params)",
        "max_abs_grad_if_available": float("nan"),
        "failure_type": "",
        "failure_message": "",
        "notes": "float64 cpu interior-point raw parameter vector",
    }

    try:
        row["max_abs_grad_if_available"] = _manual_gradient_magnitude(wrapped_loss, raw_vector)
        gradcheck(
            wrapped_loss,
            (raw_vector,),
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            rtol=GRADCHECK_RTOL,
            nondet_tol=GRADCHECK_NONDET_TOL,
            raise_exception=True,
        )
    except Exception as exc:
        status, failure_type, notes = _classify_exception(model_name, exc)
        row["gradcheck_status"] = status
        row["failure_type"] = failure_type
        row["failure_message"] = str(exc)
        if notes:
            row["notes"] = notes

    return row


def _write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "gradcheck_status",
        "eps",
        "atol",
        "rtol",
        "n_timesteps",
        "n_basins",
        "checked_variables",
        "max_abs_grad_if_available",
        "failure_type",
        "failure_message",
        "notes",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


@lru_cache(maxsize=1)
def _artifact_rows() -> list[dict[str, Any]]:
    rows = [evaluate_gradcheck_model(model_name) for model_name in REPRESENTATIVE_MODELS]
    _write_summary_csv(rows, SUMMARY_CSV_PATH)
    return rows


@pytest.mark.parametrize("model_name", REPRESENTATIVE_MODELS)
def test_model_gradcheck_representative(model_name: str) -> None:
    row = next(result for result in _artifact_rows() if result["model"] == model_name)

    if row["gradcheck_status"] == "gradcheck_expected_nondifferentiable_point":
        pytest.xfail(row["notes"] or row["failure_message"])
    if row["gradcheck_status"] == "gradcheck_api_not_suitable":
        pytest.xfail(row["notes"] or row["failure_message"])

    assert row["gradcheck_status"] == "gradcheck_pass", (
        f"{model_name} gradcheck classified as {row['gradcheck_status']} "
        f"({row['failure_type']}): {row['failure_message']}"
    )


def test_model_gradcheck_representative_has_expected_model_coverage() -> None:
    rows = _artifact_rows()
    assert {row["model"] for row in rows} == set(REPRESENTATIVE_MODELS)


def test_model_gradcheck_representative_summary_csv_is_written() -> None:
    _artifact_rows()
    assert SUMMARY_CSV_PATH.exists()
    assert SUMMARY_CSV_PATH.stat().st_size > 0
