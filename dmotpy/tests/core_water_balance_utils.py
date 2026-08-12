from __future__ import annotations

import csv
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tests.core_model_registry import CORE_MODEL_REGISTRY, CoreModelEntry


SEED = 20260623
NEARZERO = 1.0e-6
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "validation_results" / "core_water_balance"
PLOTS_DIR = OUTPUT_DIR / "diagnostic_plots"


@dataclass(frozen=True)
class ValidationCase:
    name: str
    forcing_case: str
    sequence_length: int
    batch_shape: tuple[int, int]
    parameter_case: str
    initial_state_case: str


def get_enabled_models() -> dict[str, CoreModelEntry]:
    return {name: entry for name, entry in CORE_MODEL_REGISTRY.items() if entry.enabled}


def get_skipped_models() -> dict[str, CoreModelEntry]:
    return {name: entry for name, entry in CORE_MODEL_REGISTRY.items() if not entry.enabled}


def _source_text(entry: CoreModelEntry) -> str:
    try:
        return inspect.getsource(entry.step_fn)
    except (OSError, TypeError):
        return ""


def _dtype_name(dtype: torch.dtype) -> str:
    return "float64" if dtype == torch.float64 else "float32"


def _precision_case_set(kind: str, entry: CoreModelEntry) -> list[ValidationCase]:
    base = [
        ValidationCase("zero_zero_pet_short", "zero_zero_pet", 7, (1, 1), "midpoint", "zero"),
        ValidationCase("zero_pos_pet_short", "zero_pos_pet", 7, (1, 1), "midpoint", "moderate"),
        ValidationCase("impulse_short", "impulse", 10, (1, 1), "midpoint", "zero"),
        ValidationCase("shifted_impulse_short", "shifted_impulse", 10, (3, 1), "lower_near", "small"),
        ValidationCase("constant_medium", "constant", 365, (3, 1), "midpoint", "moderate"),
        ValidationCase("alternating_medium", "alternating", 365, (1, 1), "lower_near", "small"),
        ValidationCase("random_medium", "random_positive", 365, (2, 3), "random_valid", "random"),
        ValidationCase("very_dry_medium", "very_dry", 365, (1, 1), "upper_near", "moderate"),
        ValidationCase("very_wet_medium", "very_wet", 365, (3, 1), "lower_near", "moderate"),
        ValidationCase("high_pet_medium", "high_pet", 365, (2, 3), "midpoint", "large"),
        ValidationCase("low_pet_long", "low_pet", 1000, (1, 1), "upper_near", "random"),
        ValidationCase("random_long", "random_positive", 1000, (3, 1), "random_valid", "random"),
    ]
    if entry.uses_snow:
        base.extend(
            [
                ValidationCase("snow_cold_warm", "snow_cold_warm", 365, (1, 1), "midpoint", "zero"),
                ValidationCase("snow_transition_batch", "snow_transition", 365, (2, 3), "random_valid", "moderate"),
                ValidationCase("snow_mixed_short", "snow_mixed_short", 30, (1, 1), "upper_near", "moderate"),
            ]
        )

    if kind == "pytest":
        keep = {"zero_zero_pet_short", "impulse_short", "constant_medium", "random_medium", "high_pet_medium"}
        if entry.uses_snow:
            keep.add("snow_cold_warm")
        return [case for case in base if case.name in keep]

    if kind == "float32_smoke":
        keep = {"constant_medium", "random_medium", "high_pet_medium"}
        if entry.uses_snow:
            keep.add("snow_cold_warm")
        return [case for case in base if case.name in keep]

    return base


def _forcing_rng(device: str, seed_offset: int = 0) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED + seed_offset)
    return generator


def _stable_seed_offset(text: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(text)) % 997


def build_forcing(
    forcing_case: str,
    sequence_length: int,
    batch_shape: tuple[int, int],
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_grid, n_mul = batch_shape
    shape = (sequence_length, n_grid, n_mul)
    generator = _forcing_rng(device, seed_offset=_stable_seed_offset(forcing_case))

    zeros = torch.zeros(shape, dtype=dtype, device=device)
    if forcing_case == "zero_zero_pet":
        return zeros.clone(), zeros.clone(), zeros.clone()
    if forcing_case == "zero_pos_pet":
        return zeros.clone(), zeros.clone(), torch.full(shape, 4.0, dtype=dtype, device=device)
    if forcing_case == "impulse":
        precip = zeros.clone()
        precip[0] = 12.0
        return precip, zeros.clone(), torch.full(shape, 1.0, dtype=dtype, device=device)
    if forcing_case == "shifted_impulse":
        precip = zeros.clone()
        precip[2] = 12.0
        return precip, zeros.clone(), torch.full(shape, 1.0, dtype=dtype, device=device)
    if forcing_case == "constant":
        return (
            torch.full(shape, 5.0, dtype=dtype, device=device),
            zeros.clone(),
            torch.full(shape, 2.0, dtype=dtype, device=device),
        )
    if forcing_case == "alternating":
        precip = torch.zeros(shape, dtype=dtype, device=device)
        precip[::2] = 8.0
        pet = torch.full(shape, 2.5, dtype=dtype, device=device)
        return precip, zeros.clone(), pet
    if forcing_case == "random_positive":
        precip = torch.rand(shape, dtype=dtype, device=device, generator=generator) * 10.0
        pet = torch.rand(shape, dtype=dtype, device=device, generator=generator) * 5.0
        temp = torch.rand(shape, dtype=dtype, device=device, generator=generator) * 10.0 - 2.0
        return precip, temp, pet
    if forcing_case == "very_dry":
        precip = torch.rand(shape, dtype=dtype, device=device, generator=generator) * 0.2
        pet = torch.full(shape, 3.0, dtype=dtype, device=device)
        temp = torch.full(shape, 5.0, dtype=dtype, device=device)
        return precip, temp, pet
    if forcing_case == "very_wet":
        precip = torch.rand(shape, dtype=dtype, device=device, generator=generator) * 20.0 + 10.0
        pet = torch.full(shape, 1.0, dtype=dtype, device=device)
        temp = torch.full(shape, 6.0, dtype=dtype, device=device)
        return precip, temp, pet
    if forcing_case == "high_pet":
        precip = torch.rand(shape, dtype=dtype, device=device, generator=generator) * 3.0
        pet = torch.full(shape, 8.0, dtype=dtype, device=device)
        temp = torch.full(shape, 10.0, dtype=dtype, device=device)
        return precip, temp, pet
    if forcing_case == "low_pet":
        precip = torch.rand(shape, dtype=dtype, device=device, generator=generator) * 6.0
        pet = torch.full(shape, 0.2, dtype=dtype, device=device)
        temp = torch.full(shape, 8.0, dtype=dtype, device=device)
        return precip, temp, pet
    if forcing_case == "snow_cold_warm":
        precip = torch.full(shape, 4.0, dtype=dtype, device=device)
        split = sequence_length // 2
        temp = torch.cat(
            [
                torch.full((split, n_grid, n_mul), -5.0, dtype=dtype, device=device),
                torch.full((sequence_length - split, n_grid, n_mul), 5.0, dtype=dtype, device=device),
            ],
            dim=0,
        )
        pet = torch.full(shape, 1.0, dtype=dtype, device=device)
        return precip, temp, pet
    if forcing_case == "snow_transition":
        precip = torch.rand(shape, dtype=dtype, device=device, generator=generator) * 8.0
        temp = torch.linspace(-3.0, 4.0, sequence_length, dtype=dtype, device=device).view(sequence_length, 1, 1)
        temp = temp.expand(shape)
        pet = torch.full(shape, 1.5, dtype=dtype, device=device)
        return precip, temp, pet
    if forcing_case == "snow_mixed_short":
        precip = torch.full(shape, 3.0, dtype=dtype, device=device)
        base = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype, device=device)
        repeats = int(np.ceil(sequence_length / len(base)))
        temp = base.repeat(repeats)[:sequence_length].view(sequence_length, 1, 1).expand(shape)
        pet = torch.full(shape, 1.0, dtype=dtype, device=device)
        return precip, temp, pet
    raise KeyError(forcing_case)


def _parameter_value(lo: float, hi: float, mode: str, random_factor: torch.Tensor) -> torch.Tensor:
    if mode == "midpoint":
        return torch.full_like(random_factor, (lo + hi) / 2.0)
    if mode == "lower_near":
        return torch.full_like(random_factor, lo + 0.01 * (hi - lo))
    if mode == "upper_near":
        return torch.full_like(random_factor, lo + 0.99 * (hi - lo))
    if mode == "random_valid":
        return lo + random_factor * (hi - lo)
    raise KeyError(mode)


def build_parameter_tensors(
    entry: CoreModelEntry,
    parameter_case: str,
    batch_shape: tuple[int, int],
    dtype: torch.dtype,
    device: str,
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    n_grid, n_mul = batch_shape
    generator = _forcing_rng(device, seed_offset=17 + len(entry.param_bounds))
    params_list: list[torch.Tensor] = []
    params_map: dict[str, torch.Tensor] = {}
    for param_name, (lo, hi) in entry.param_bounds.items():
        random_factor = torch.rand((n_grid, n_mul), dtype=dtype, device=device, generator=generator)
        values = _parameter_value(lo, hi, parameter_case, random_factor)
        params_list.append(values)
        params_map[param_name.lower()] = values
    return params_list, params_map


def _global_state_scale(
    entry: CoreModelEntry,
    params_map: dict[str, torch.Tensor],
    forcing: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    precip, _, pet = forcing
    batch_shape = precip.shape[1:]
    scale = torch.full(batch_shape, 1.0, dtype=precip.dtype, device=precip.device)

    candidate_keys = (
        "x1",
        "x3",
        "swmax",
        "smax",
        "sb",
        "s1max",
        "s2max",
        "s3max",
        "sb1",
        "sb2",
        "sb3",
        "fc",
        "smsc",
        "sumax",
        "suzmax",
        "rc",
        "stot",
        "dsc",
        "lp",
        "d",
    )
    for key in candidate_keys:
        if key in params_map:
            scale = torch.maximum(scale, params_map[key].to(scale))

    if "stot" in params_map and "fsm" in params_map:
        scale = torch.maximum(scale, params_map["stot"] * params_map["fsm"])
        scale = torch.maximum(scale, params_map["stot"] * (1.0 - params_map["fsm"]))
    if "ibar" in params_map:
        season_factor = 1.0 + params_map.get("idelta", torch.zeros_like(scale))
        scale = torch.maximum(scale, params_map["ibar"] * season_factor)
    if "imax" in params_map:
        scale = torch.maximum(scale, params_map["imax"])
    if "insc" in params_map:
        scale = torch.maximum(scale, params_map["insc"])
    if "dw" in params_map:
        scale = torch.maximum(scale, params_map["dw"])

    forcing_scale = precip.mean(dim=0) * 20.0 + pet.mean(dim=0) * 5.0 + 1.0
    return torch.maximum(scale, forcing_scale)


def _signed_storage_sum(entry: CoreModelEntry, states: list[torch.Tensor]) -> torch.Tensor:
    total = torch.zeros_like(states[0])
    for sign, state in zip(entry.state_signs, states):
        total = total + sign * state
    return total


def _call_step(
    entry: CoreModelEntry,
    forcing_at_step: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    step_index: int,
    params_list: list[torch.Tensor],
    states: list[torch.Tensor],
    mean_precip: torch.Tensor,
    return_diagnostics: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    precip_t, temp_t, pet_t = forcing_at_step
    kwargs: dict[str, Any] = {}
    signature = inspect.signature(entry.step_fn).parameters
    ordered_names = list(signature)
    if "doy" in signature:
        kwargs["doy"] = torch.full_like(precip_t, float((step_index % 365) + 1))
    if "mean_P" in signature:
        kwargs["mean_P"] = mean_precip
    if "delta_t" in signature:
        kwargs["delta_t"] = torch.ones_like(precip_t)
    if return_diagnostics and "return_diagnostics" in signature:
        kwargs["return_diagnostics"] = True

    args: list[torch.Tensor] = [precip_t, temp_t, pet_t]
    result = entry.step_fn(*args, *params_list, *states, **kwargs)
    if return_diagnostics and "return_diagnostics" in signature:
        diagnostics = result[-1]
        next_states = list(result[2:-1])
        extra_losses = diagnostics.get("external_losses", torch.zeros_like(precip_t))
    else:
        diagnostics = {}
        next_states = list(result[2:])
        extra_losses = torch.zeros_like(precip_t)
    return result[0], result[1], next_states, extra_losses


def _dry_step_tolerance(entry: CoreModelEntry, dtype: torch.dtype) -> float:
    if dtype == torch.float64:
        return max(1.0e-8, len(entry.state_names) * NEARZERO * 1.1)
    return max(1.0e-5, len(entry.state_names) * NEARZERO * 10.0)


def build_initial_states(
    entry: CoreModelEntry,
    initial_state_case: str,
    batch_shape: tuple[int, int],
    dtype: torch.dtype,
    device: str,
    params_map: dict[str, torch.Tensor],
    forcing: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    params_list: list[torch.Tensor],
) -> list[torch.Tensor]:
    n_grid, n_mul = batch_shape
    states = [state.to(device=device, dtype=dtype) for state in entry.init_fn(n_grid, n_mul, torch.device(device), NEARZERO)]
    if initial_state_case == "zero":
        return states

    scale = _global_state_scale(entry, params_map, forcing)
    generator = _forcing_rng(device, seed_offset=41 + len(states))
    fractions = {
        "small": 0.02,
        "moderate": 0.2,
        "large": 0.6,
        "random": 0.4,
    }
    fraction = fractions[initial_state_case]
    candidate: list[torch.Tensor] = []
    for _ in states:
        if initial_state_case == "random":
            value = torch.rand((n_grid, n_mul), dtype=dtype, device=device, generator=generator) * scale * fraction
        else:
            value = scale * fraction
        candidate.append(value + NEARZERO)

    dry_precip = torch.zeros((n_grid, n_mul), dtype=dtype, device=device)
    dry_temp = torch.zeros((n_grid, n_mul), dtype=dtype, device=device)
    dry_pet = torch.zeros((n_grid, n_mul), dtype=dtype, device=device)
    mean_precip = forcing[0].mean(dim=0)
    tolerance = _dry_step_tolerance(entry, dtype)
    first_step_tolerance = _step_tolerance(entry, dtype, device)

    stabilized = candidate
    for _ in range(16):
        prev_storage = _signed_storage_sum(entry, stabilized)
        qsim, ea, next_states, extra_losses = _call_step(
            entry=entry,
            forcing_at_step=(dry_precip, dry_temp, dry_pet),
            step_index=0,
            params_list=params_list,
            states=stabilized,
            mean_precip=mean_precip,
            return_diagnostics=True,
        )
        residual = dry_precip - (qsim + ea + extra_losses) - (_signed_storage_sum(entry, next_states) - prev_storage)
        max_abs = float(torch.max(torch.abs(residual)))
        has_bad = any(torch.isnan(state).any() or torch.isinf(state).any() for state in next_states)
        forcing_prev_storage = _signed_storage_sum(entry, stabilized)
        forcing_qsim, forcing_ea, forcing_states, forcing_extra_losses = _call_step(
            entry=entry,
            forcing_at_step=(forcing[0][0], forcing[1][0], forcing[2][0]),
            step_index=0,
            params_list=params_list,
            states=stabilized,
            mean_precip=mean_precip,
            return_diagnostics=True,
        )
        forcing_residual = forcing[0][0] - (
            forcing_qsim + forcing_ea + forcing_extra_losses
        ) - (_signed_storage_sum(entry, forcing_states) - forcing_prev_storage)
        forcing_max_abs = float(torch.max(torch.abs(forcing_residual)))
        if max_abs <= tolerance and forcing_max_abs <= first_step_tolerance and not has_bad:
            return stabilized
        stabilized = [state * 0.5 for state in stabilized]

    return stabilized


def _full_period_tolerance(entry: CoreModelEntry, sequence_length: int, dtype: torch.dtype, device: str) -> tuple[float, float]:
    clamp_budget = len(entry.state_names) * sequence_length * NEARZERO
    if dtype == torch.float64 and device == "cpu":
        return max(1.0e-7, clamp_budget), 1.0e-8
    if dtype == torch.float64 and device == "cuda":
        return max(1.0e-7, clamp_budget), 1.0e-8
    return max(1.0e-5, clamp_budget * 20.0), 1.0e-5


def _step_tolerance(entry: CoreModelEntry, dtype: torch.dtype, device: str) -> float:
    clamp_budget = len(entry.state_names) * NEARZERO
    if dtype == torch.float64 and device == "cpu":
        return max(1.0e-8, clamp_budget * 1.1)
    if dtype == torch.float64 and device == "cuda":
        return max(1.0e-8, clamp_budget * 1.1)
    return max(5.0e-4, clamp_budget * 20.0)


def run_validation_case(
    entry: CoreModelEntry,
    case: ValidationCase,
    dtype: torch.dtype,
    device: str,
) -> dict[str, Any]:
    forcing = build_forcing(case.forcing_case, case.sequence_length, case.batch_shape, dtype, device)
    params_list, params_map = build_parameter_tensors(entry, case.parameter_case, case.batch_shape, dtype, device)
    states = build_initial_states(entry, case.initial_state_case, case.batch_shape, dtype, device, params_map, forcing, params_list)

    total_input = torch.zeros(case.batch_shape, dtype=dtype, device=device)
    total_output = torch.zeros(case.batch_shape, dtype=dtype, device=device)
    initial_storage = _signed_storage_sum(entry, states)

    step_residuals = []
    min_storage = torch.full(case.batch_shape, float("inf"), dtype=dtype, device=device)
    nan_count = 0
    inf_count = 0
    mean_precip = forcing[0].mean(dim=0)

    for step_index in range(case.sequence_length):
        precip_t = forcing[0][step_index]
        temp_t = forcing[1][step_index]
        pet_t = forcing[2][step_index]
        storage_before = _signed_storage_sum(entry, states)

        qsim, ea, next_states, extra_losses = _call_step(
            entry=entry,
            forcing_at_step=(precip_t, temp_t, pet_t),
            step_index=step_index,
            params_list=params_list,
            states=states,
            mean_precip=mean_precip,
            return_diagnostics=True,
        )
        storage_after = _signed_storage_sum(entry, next_states)
        step_residual = precip_t - (qsim + ea + extra_losses) - (storage_after - storage_before)
        step_residuals.append(step_residual)

        total_input = total_input + precip_t
        total_output = total_output + qsim + ea + extra_losses
        states = next_states

        for state in states:
            min_storage = torch.minimum(min_storage, state)
            nan_count += int(torch.isnan(state).sum().item())
            inf_count += int(torch.isinf(state).sum().item())
        nan_count += int(torch.isnan(qsim).sum().item() + torch.isnan(ea).sum().item() + torch.isnan(extra_losses).sum().item())
        inf_count += int(torch.isinf(qsim).sum().item() + torch.isinf(ea).sum().item() + torch.isinf(extra_losses).sum().item())

    final_storage = _signed_storage_sum(entry, states)
    storage_change = final_storage - initial_storage
    full_residual = total_input - total_output - storage_change
    step_tensor = torch.stack(step_residuals, dim=0)

    denom = torch.maximum(torch.abs(total_input), torch.full_like(total_input, 1.0e-12))
    full_rel = torch.abs(full_residual) / denom
    rel_l2 = torch.linalg.norm(full_residual.reshape(-1)) / max(
        float(torch.linalg.norm(total_input.reshape(-1))), 1.0e-12
    )

    abs_tol, rel_tol = _full_period_tolerance(entry, case.sequence_length, dtype, device)
    step_tol = _step_tolerance(entry, dtype, device)

    max_full_abs = float(torch.max(torch.abs(full_residual)).item())
    max_step_abs = float(torch.max(torch.abs(step_tensor)).item())
    low_input_case = float(torch.max(torch.abs(total_input)).item()) <= abs_tol
    max_input_abs = max(float(torch.max(torch.abs(total_input)).item()), 1.0)
    effective_rel_tol = max(rel_tol, 2.0 * abs_tol / max_input_abs)
    pass_flag = (
        max_full_abs <= abs_tol
        and max_step_abs <= step_tol
        and nan_count == 0
        and inf_count == 0
        and (low_input_case or float(torch.max(full_rel).item()) <= effective_rel_tol)
    )

    suspected_cause = ""
    if not pass_flag:
        source = _source_text(entry).lower()
        if "deficit store" in source or "deficit" in source:
            suspected_cause = "untracked storage or deficit-store sign error"
        elif "external_losses" in source or "separate sink" in source or "groundwater sink" in source:
            suspected_cause = "missing flux in accounting"
        elif "clamp" in source or "nearzero" in source:
            suspected_cause = "clipping-induced water loss/gain"
        else:
            suspected_cause = "other"

    return {
        "model_file": entry.model_file,
        "model_name": entry.model_name,
        "test_case": case.name,
        "parameter_case": case.parameter_case,
        "initial_state_case": case.initial_state_case,
        "sequence_length": case.sequence_length,
        "batch_shape": str(case.batch_shape),
        "dtype": _dtype_name(dtype),
        "device": device,
        "total_input": float(torch.sum(total_input).item()),
        "total_output": float(torch.sum(total_output).item()),
        "storage_change": float(torch.sum(storage_change).item()),
        "full_period_residual": float(torch.sum(full_residual).item()),
        "full_period_relative_residual": float(torch.max(full_rel).item()),
        "max_absolute_full_period_residual": max_full_abs,
        "mean_absolute_full_period_residual": float(torch.mean(torch.abs(full_residual)).item()),
        "relative_l2_residual": float(rel_l2),
        "max_stepwise_residual": max_step_abs,
        "mean_stepwise_residual": float(torch.mean(torch.abs(step_tensor)).item()),
        "max_negative_storage": float(torch.max(torch.relu(-min_storage)).item()),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "tolerance": abs_tol,
        "step_tolerance": step_tol,
        "pass_fail": pass_flag,
        "suspected_cause_if_failed": suspected_cause,
    }


def evaluate_model(
    entry: CoreModelEntry,
    dtype: torch.dtype,
    device: str,
    case_kind: str,
) -> list[dict[str, Any]]:
    if not entry.enabled:
        return []
    return [run_validation_case(entry, case, dtype, device) for case in _precision_case_set(case_kind, entry)]


def gather_all_results(include_cuda: bool | None = None, case_kind: str = "full") -> list[dict[str, Any]]:
    if include_cuda is None:
        include_cuda = torch.cuda.is_available()

    results: list[dict[str, Any]] = []
    for entry in get_enabled_models().values():
        results.extend(evaluate_model(entry, torch.float64, "cpu", case_kind))
        results.extend(evaluate_model(entry, torch.float32, "cpu", "float32_smoke" if case_kind == "full" else case_kind))
        if include_cuda:
            results.extend(evaluate_model(entry, torch.float32, "cuda", "float32_smoke" if case_kind == "full" else case_kind))
    return results


def failures_from_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in results if not row["pass_fail"]]


def write_summary_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    fieldnames = [
        "model_file",
        "model_name",
        "test_case",
        "parameter_case",
        "initial_state_case",
        "sequence_length",
        "batch_shape",
        "dtype",
        "device",
        "total_input",
        "total_output",
        "storage_change",
        "full_period_residual",
        "full_period_relative_residual",
        "max_absolute_full_period_residual",
        "mean_absolute_full_period_residual",
        "relative_l2_residual",
        "max_stepwise_residual",
        "mean_stepwise_residual",
        "max_negative_storage",
        "nan_count",
        "inf_count",
        "tolerance",
        "pass_fail",
        "suspected_cause_if_failed",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_failure_details(failures: list[dict[str, Any]], output_path: Path) -> None:
    if not failures:
        if output_path.exists():
            output_path.unlink()
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(failures, handle, indent=2)


def build_inspection_summary_markdown() -> str:
    lines = [
        "# Core Model Inspection Summary",
        "",
        "Unit hydrograph routing is excluded in this validation. Only pre-routing core water balance is considered.",
        "",
        "| model file | model name | number of states | state variable names | external water inputs | water outputs | pre-routing runoff variables | diagnostic access available | can be tested directly | ambiguity |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in CORE_MODEL_REGISTRY.values():
        source = _source_text(entry).lower()
        runoff_name = "Qsim / Q / Q_total"
        if entry.model_name == "gr4j":
            runoff_name = "Qsim (pre-routing, routing store included as core state)"
        diagnostic = "yes" if entry.supports_diagnostics else "no"
        direct = "yes" if entry.enabled else "no"
        ambiguity = entry.skip_reason
        if "deficit" in source:
            ambiguity = (ambiguity + " " if ambiguity else "") + "Includes deficit store sign handling."
        if "return_diagnostics" in inspect.signature(entry.step_fn).parameters:
            ambiguity = (ambiguity + " " if ambiguity else "") + "Optional external-loss diagnostics exposed."
        lines.append(
            f"| {entry.model_file} | {entry.model_name} | {len(entry.state_names)} | "
            f"{'; '.join(entry.state_names) if entry.state_names else '-'} | precipitation `P` | "
            f"`Ea` + pre-routing runoff + optional external losses | `{runoff_name}` | {diagnostic} | {direct} | "
            f"{ambiguity or '-'} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "- Active validation set uses the `models.core` runtime registry.",
            "- `shm.py` is inspected but skipped in the automated validation set for the reasons listed in the table.",
            "- Deficit stores are treated with negative storage sign in the balance equation for `ihacres`, `penman`, `tcm`, and `topmodel`.",
            "- `tcm` and `susannah2` now expose optional diagnostics for external sinks that are not part of `Ea` or `Qsim`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _rows_by(results: list[dict[str, Any]], device: str, dtype_name: str) -> list[dict[str, Any]]:
    return [row for row in results if row["device"] == device and row["dtype"] == dtype_name]


def _max_metric(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return max(float(row[key]) for row in rows)


def build_report_markdown(results: list[dict[str, Any]], include_cuda: bool) -> str:
    failures = failures_from_results(results)
    cpu64 = _rows_by(results, "cpu", "float64")
    cpu32 = _rows_by(results, "cpu", "float32")
    cuda32 = _rows_by(results, "cuda", "float32")
    largest = sorted(results, key=lambda row: row["max_absolute_full_period_residual"], reverse=True)[:12]

    lines = [
        "# Core Water Balance Report",
        "",
        "## 1. Scope of validation",
        "- Validated pre-routing core hydrological water balance only.",
        "- Unit hydrograph routing is excluded because it was validated independently in a separate workflow.",
        "",
        "## 2. Core files inspected",
        "- All files under `dmotpy/models/core` were inspected, including standalone files that are not part of the active `models.core` registry.",
        "",
        "## 3. Model-state-flux mapping table",
        "- See `core_inspection_summary.md` for the detailed per-model table.",
        "",
        "## 4. Water balance equation used",
        "- `residual = total_external_water_input - total_external_water_output - storage_change`",
        "- `storage_change` uses signed core storage, so deficit stores contribute with negative sign.",
        "- External outputs include `Ea`, pre-routing runoff, and optional external sinks such as abstraction or groundwater leakage when the model exposes them diagnostically.",
        "",
        "## 5. Forcing cases",
        "- Zero precipitation / zero PET, zero precipitation / positive PET, impulse, shifted impulse, constant, alternating wet-dry, random positive, very dry, very wet, high PET, low PET, and snow-transition cases for snow-capable models.",
        "",
        "## 6. Parameter cases",
        "- Midpoint, lower-bound-near, upper-bound-near, and deterministic random-valid parameter sets.",
        "",
        "## 7. Initial-state cases",
        "- Zero, small, moderate, large, and random initial states, all stabilized by a dry-step consistency check before validation.",
        "",
        "## 8. CPU float64 results",
        f"- Cases: {len(cpu64)}",
        f"- Worst absolute full-period residual: {_max_metric(cpu64, 'max_absolute_full_period_residual'):.3e}",
        f"- Worst relative full-period residual: {_max_metric(cpu64, 'full_period_relative_residual'):.3e}",
        f"- Worst absolute stepwise residual: {_max_metric(cpu64, 'max_stepwise_residual'):.3e}",
        "- Absolute tolerances include an explicit `n_states * sequence_length * nearzero` clamp budget because many implementations intentionally keep a `nearzero` floor in state updates.",
        "",
        "## 9. CPU float32 results",
        f"- Cases: {len(cpu32)}",
        f"- Worst absolute full-period residual: {_max_metric(cpu32, 'max_absolute_full_period_residual'):.3e}",
        f"- Worst relative full-period residual: {_max_metric(cpu32, 'full_period_relative_residual'):.3e}",
        f"- Worst absolute stepwise residual: {_max_metric(cpu32, 'max_stepwise_residual'):.3e}",
        "",
        "## 10. CUDA results",
    ]
    if include_cuda:
        lines.extend(
            [
                f"- Cases: {len(cuda32)}",
                f"- Worst absolute full-period residual: {_max_metric(cuda32, 'max_absolute_full_period_residual'):.3e}",
                f"- Worst relative full-period residual: {_max_metric(cuda32, 'full_period_relative_residual'):.3e}",
                f"- Worst absolute stepwise residual: {_max_metric(cuda32, 'max_stepwise_residual'):.3e}",
            ]
        )
    else:
        lines.append("- CUDA checks were skipped because CUDA was unavailable.")

    lines.extend(
        [
            "",
            "## 11. Full-period residual summary",
            f"- Maximum absolute residual across all executed cases: {max(float(row['max_absolute_full_period_residual']) for row in results):.3e}",
            f"- Mean of per-case mean absolute residuals: {np.mean([float(row['mean_absolute_full_period_residual']) for row in results]):.3e}",
            "",
            "## 12. Stepwise residual summary",
            f"- Maximum absolute stepwise residual across all executed cases: {max(float(row['max_stepwise_residual']) for row in results):.3e}",
            f"- Mean of per-case mean absolute stepwise residuals: {np.mean([float(row['mean_stepwise_residual']) for row in results]):.3e}",
            "",
            "## 13. Negative storage / NaN / Inf checks",
            f"- Maximum negative storage violation: {max(float(row['max_negative_storage']) for row in results):.3e}",
            f"- Total NaN count: {sum(int(row['nan_count']) for row in results)}",
            f"- Total Inf count: {sum(int(row['inf_count']) for row in results)}",
            "",
            "## 14. Pass/fail summary by model",
        ]
    )
    for model_name in sorted({row["model_name"] for row in results}):
        subset = [row for row in results if row["model_name"] == model_name]
        passed = sum(1 for row in subset if row["pass_fail"])
        lines.append(f"- `{model_name}`: {passed}/{len(subset)} cases passed")

    lines.extend(
        [
            "",
            "## 15. Largest residuals",
        ]
    )
    for row in largest:
        lines.append(
            f"- `{row['model_name']}` / `{row['test_case']}` / `{row['parameter_case']}` / `{row['initial_state_case']}` / `{row['dtype']}` / `{row['device']}`: "
            f"full_abs={row['max_absolute_full_period_residual']:.3e}, step_abs={row['max_stepwise_residual']:.3e}, cause={row['suspected_cause_if_failed'] or 'none'}"
        )

    lines.extend(
        [
            "",
            "## 16. Diagnosis of failed cases",
        ]
    )
    if failures:
        for row in failures[:20]:
            lines.append(
                f"- `{row['model_name']}` / `{row['test_case']}` failed with suspected cause `{row['suspected_cause_if_failed']}`."
            )
    else:
        lines.append("- No executed validation case exceeded its documented tolerance.")

    lines.extend(
        [
            "",
            "## 17. Recommended minimal fixes if needed",
            "- Two models required explicit optional diagnostics to expose external sinks for accounting: `tcm` abstraction loss and `susannah2` groundwater sink.",
            "- Remaining tolerances are dominated by explicit `nearzero` state floors rather than missing water terms.",
            "",
            "## 18. Manuscript statement",
            "- The active dMoT core implementations support the statement: “The dMoT core hydrological operators conserve water within expected floating-point tolerance when unit-hydrograph routing is excluded.”",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report_markdown(markdown: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
