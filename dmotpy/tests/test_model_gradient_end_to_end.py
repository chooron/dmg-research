from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.core_model_registry import CORE_MODEL_REGISTRY, CoreModelEntry
from tests.core_water_balance_utils import _call_step, build_initial_states


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "validation_results" / "model_gradcheck_water_balance_tests"
SUMMARY_CSV_PATH = OUTPUT_DIR / "model_gradient_end_to_end_summary.csv"

DEFAULT_DTYPE = torch.float64
DEFAULT_DEVICE = "cpu"
DEFAULT_TIMESTEPS = 5
DEFAULT_BASINS = 2
EXPECTED_SKIPS: dict[str, str] = {}


def discover_runnable_models() -> dict[str, CoreModelEntry]:
    return {name: entry for name, entry in CORE_MODEL_REGISTRY.items() if entry.enabled}


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float64:
        return "float64"
    if dtype == torch.float32:
        return "float32"
    return str(dtype).replace("torch.", "")


def make_synthetic_forcing(
    entry: CoreModelEntry,
    n_timesteps: int,
    n_basins: int,
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (n_timesteps, n_basins, 1)
    time_axis = torch.arange(n_timesteps, dtype=dtype, device=device).view(n_timesteps, 1, 1)
    basin_axis = torch.arange(n_basins, dtype=dtype, device=device).view(1, n_basins, 1)

    precip = 3.5 + 0.45 * time_axis + 0.2 * basin_axis
    pet = 1.1 + 0.15 * (time_axis % 3.0) + 0.05 * basin_axis

    if entry.uses_snow:
        temperature = 6.0 + 0.4 * basin_axis + 0.1 * time_axis
    else:
        temperature = 5.0 + 0.3 * basin_axis + 0.2 * time_axis

    return precip.expand(shape), temperature.expand(shape), pet.expand(shape)


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


def make_interior_parameters(
    entry: CoreModelEntry,
    n_basins: int,
    dtype: torch.dtype,
    device: str,
    *,
    requires_grad: bool,
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    params_list: list[torch.Tensor] = []
    params_map: dict[str, torch.Tensor] = {}
    shape = (n_basins, 1)

    for param_name, (lower_bound, upper_bound) in entry.param_bounds.items():
        fraction = _parameter_fraction(param_name, float(lower_bound))
        value = torch.full(
            shape,
            float(lower_bound) + fraction * (float(upper_bound) - float(lower_bound)),
            dtype=dtype,
            device=device,
        )
        if requires_grad:
            value.requires_grad_(True)
        params_list.append(value)
        params_map[param_name.lower()] = value

    return params_list, params_map


def run_model_forward(
    entry: CoreModelEntry,
    forcing: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    params_list: list[torch.Tensor],
    initial_states: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor], int, int]:
    discharge: list[torch.Tensor] = []
    states = [state.clone() for state in initial_states]
    state_nan_count = 0
    state_inf_count = 0
    mean_precip = forcing[0].mean(dim=0)

    for step_index in range(forcing[0].shape[0]):
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
        state_nan_count += sum(int(torch.isnan(state).sum().item()) for state in states)
        state_inf_count += sum(int(torch.isinf(state).sum().item()) for state in states)

    return torch.stack(discharge, dim=0), states, state_nan_count, state_inf_count


def extract_discharge(forward_output: torch.Tensor) -> torch.Tensor:
    return forward_output


def compute_scalar_loss(discharge: torch.Tensor) -> torch.Tensor:
    n_timesteps, n_basins = discharge.shape[:2]
    time_target = torch.linspace(0.75, 1.35, n_timesteps, dtype=discharge.dtype, device=discharge.device).view(
        n_timesteps, 1, 1
    )
    basin_scale = torch.linspace(0.95, 1.05, n_basins, dtype=discharge.dtype, device=discharge.device).view(
        1, n_basins, 1
    )
    target = time_target * basin_scale
    return torch.mean((discharge - target) ** 2)


def _gradient_statistics(params_list: list[torch.Tensor]) -> tuple[int, int, float, float, int]:
    gradients = [tensor.grad for tensor in params_list if tensor.grad is not None]
    if not gradients:
        return 0, 0, 0.0, 0.0, 0

    flat_abs = torch.cat([grad.detach().abs().reshape(-1) for grad in gradients])
    grad_nan_count = sum(int(torch.isnan(grad).sum().item()) for grad in gradients)
    grad_inf_count = sum(int(torch.isinf(grad).sum().item()) for grad in gradients)
    return (
        grad_nan_count,
        grad_inf_count,
        float(flat_abs.max().item()),
        float(flat_abs.mean().item()),
        len(gradients),
    )


def evaluate_model_gradient_end_to_end(entry: CoreModelEntry) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": entry.model_name,
        "status": "passed",
        "dtype": _dtype_name(DEFAULT_DTYPE),
        "device": DEFAULT_DEVICE,
        "n_timesteps": DEFAULT_TIMESTEPS,
        "n_basins": DEFAULT_BASINS,
        "loss": float("nan"),
        "output_nan_count": 0,
        "output_inf_count": 0,
        "grad_nan_count": 0,
        "grad_inf_count": 0,
        "max_abs_grad": 0.0,
        "mean_abs_grad": 0.0,
        "failed_stage": "",
        "notes": "",
    }

    if entry.model_name in EXPECTED_SKIPS:
        row["status"] = "expected_skip"
        row["failed_stage"] = "skip"
        row["notes"] = EXPECTED_SKIPS[entry.model_name]
        return row

    try:
        forcing = make_synthetic_forcing(entry, DEFAULT_TIMESTEPS, DEFAULT_BASINS, DEFAULT_DTYPE, DEFAULT_DEVICE)
        base_params_list, base_params_map = make_interior_parameters(
            entry,
            DEFAULT_BASINS,
            DEFAULT_DTYPE,
            DEFAULT_DEVICE,
            requires_grad=False,
        )
        initial_states = build_initial_states(
            entry,
            "small",
            (DEFAULT_BASINS, 1),
            DEFAULT_DTYPE,
            DEFAULT_DEVICE,
            base_params_map,
            forcing,
            base_params_list,
        )
        params_list = [tensor.detach().clone().requires_grad_(True) for tensor in base_params_list]
    except Exception as exc:
        row["status"] = "failed"
        row["failed_stage"] = "setup"
        row["notes"] = f"{type(exc).__name__}: {exc}"
        return row

    try:
        discharge, final_states, state_nan_count, state_inf_count = run_model_forward(
            entry,
            forcing,
            params_list,
            initial_states,
        )
        discharge = extract_discharge(discharge)
        row["output_nan_count"] = int(torch.isnan(discharge).sum().item())
        row["output_inf_count"] = int(torch.isinf(discharge).sum().item())

        if state_nan_count or state_inf_count:
            row["status"] = "failed"
            row["failed_stage"] = "forward_state_check"
            row["notes"] = f"nonfinite_states nan={state_nan_count} inf={state_inf_count}"
            return row

        if any(torch.isnan(state).any() or torch.isinf(state).any() for state in final_states):
            row["status"] = "failed"
            row["failed_stage"] = "final_state_check"
            row["notes"] = "final state tensor contains NaN/Inf"
            return row

        if row["output_nan_count"] or row["output_inf_count"]:
            row["status"] = "failed"
            row["failed_stage"] = "forward_output_check"
            row["notes"] = "discharge contains NaN/Inf"
            return row
    except Exception as exc:
        row["status"] = "failed"
        row["failed_stage"] = "forward"
        row["notes"] = f"{type(exc).__name__}: {exc}"
        return row

    try:
        loss = compute_scalar_loss(discharge)
        row["loss"] = float(loss.detach().item())
        if not torch.isfinite(loss):
            row["status"] = "failed"
            row["failed_stage"] = "loss"
            row["notes"] = "loss is NaN/Inf"
            return row
    except Exception as exc:
        row["status"] = "failed"
        row["failed_stage"] = "loss"
        row["notes"] = f"{type(exc).__name__}: {exc}"
        return row

    try:
        loss.backward()
        grad_nan_count, grad_inf_count, max_abs_grad, mean_abs_grad, gradient_count = _gradient_statistics(params_list)
        row["grad_nan_count"] = grad_nan_count
        row["grad_inf_count"] = grad_inf_count
        row["max_abs_grad"] = max_abs_grad
        row["mean_abs_grad"] = mean_abs_grad

        if gradient_count == 0:
            row["status"] = "failed"
            row["failed_stage"] = "backward_gradient_check"
            row["notes"] = "no parameter gradients were produced"
            return row
        if grad_nan_count or grad_inf_count:
            row["status"] = "failed"
            row["failed_stage"] = "backward_gradient_check"
            row["notes"] = "parameter gradients contain NaN/Inf"
            return row
        row["notes"] = f"grads_present={gradient_count}"
    except Exception as exc:
        row["status"] = "failed"
        row["failed_stage"] = "backward"
        row["notes"] = f"{type(exc).__name__}: {exc}"
        return row

    return row


def _write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "status",
        "dtype",
        "device",
        "n_timesteps",
        "n_basins",
        "loss",
        "output_nan_count",
        "output_inf_count",
        "grad_nan_count",
        "grad_inf_count",
        "max_abs_grad",
        "mean_abs_grad",
        "failed_stage",
        "notes",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


@lru_cache(maxsize=1)
def _artifact_rows() -> list[dict[str, Any]]:
    rows = [evaluate_model_gradient_end_to_end(entry) for entry in discover_runnable_models().values()]
    _write_summary_csv(rows, SUMMARY_CSV_PATH)
    return rows


@pytest.mark.parametrize("model_name", sorted(discover_runnable_models()))
def test_model_gradient_end_to_end_per_model(model_name: str) -> None:
    row = next(result for result in _artifact_rows() if result["model"] == model_name)
    if row["status"] == "expected_skip":
        pytest.skip(row["notes"])

    assert row["status"] == "passed", (
        f"{model_name} failed during {row['failed_stage'] or 'validation'}: {row['notes']}"
    )
    assert int(row["output_nan_count"]) == 0
    assert int(row["output_inf_count"]) == 0
    assert int(row["grad_nan_count"]) == 0
    assert int(row["grad_inf_count"]) == 0
    assert torch.isfinite(torch.tensor(float(row["loss"]), dtype=torch.float64))


def test_model_gradient_end_to_end_has_full_runnable_model_coverage() -> None:
    rows = _artifact_rows()
    seen_models = {row["model"] for row in rows}
    expected_models = set(discover_runnable_models())
    assert seen_models == expected_models


def test_model_gradient_end_to_end_summary_csv_is_written() -> None:
    _artifact_rows()
    assert SUMMARY_CSV_PATH.exists()
    assert SUMMARY_CSV_PATH.stat().st_size > 0
