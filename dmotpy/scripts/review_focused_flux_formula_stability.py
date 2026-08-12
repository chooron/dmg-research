from __future__ import annotations

import ast
import csv
import importlib
import inspect
import json
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.core_model_registry import CORE_MODEL_REGISTRY  # noqa: E402
from tests.core_water_balance_utils import (  # noqa: E402
    ValidationCase,
    build_forcing,
    build_initial_states,
    build_parameter_tensors,
    run_validation_case,
)
from tests.flux_gradient_wrappers import (  # noqa: E402
    FIXED_SEED,
    build_flux_wrapper,
    evaluate_wrapper,
    load_flux_inventory,
    load_flux_usage_contexts,
)


OUTPUT_DIR = REPO_ROOT / "validation_results" / "focused_flux_formula_review"
PLOTS_DIR = OUTPUT_DIR / "plots"
PREVIOUS_RESULTS_DIR = REPO_ROOT / "validation_results" / "flux_gradient_stability"
DEFAULT_DTYPE = torch.float64
DEFAULT_DEVICE = "cpu"
DEFAULT_NEARZERO = 1.0e-6
FD_EPS = 1.0e-6
REALISTIC_SHAPE = (41,)


@dataclass(frozen=True)
class TargetContext:
    formula: str
    flux_module: str
    active_model: str
    previous_risk: str
    previous_max_abs_grad: float
    expected_issue_type: str
    probe_input: str
    active_grad_inputs: tuple[str, ...]
    output_storage_cap: str | None
    call_site_labels: tuple[str, ...] = ()


TARGET_CONTEXTS: tuple[TargetContext, ...] = (
    TargetContext(
        formula="baseflow_6",
        flux_module="models.flux.baseflow",
        active_model="tcm",
        previous_risk="large gradient and dead region",
        previous_max_abs_grad=1.0e6,
        expected_issue_type="threshold / quadratic baseflow sensitivity",
        probe_input="S",
        active_grad_inputs=("p1", "S"),
        output_storage_cap="S",
    ),
    TargetContext(
        formula="interflow_10",
        flux_module="models.flux.interflow",
        active_model="topmodel",
        previous_risk="large gradient and dead region",
        previous_max_abs_grad=2.0e4,
        expected_issue_type="threshold activation and denominator sensitivity",
        probe_input="S",
        active_grad_inputs=("S", "p1", "p2", "p3"),
        output_storage_cap=None,
    ),
    TargetContext(
        formula="baseflow_2",
        flux_module="models.flux.baseflow",
        active_model="susannah1",
        previous_risk="non-finite output / non-finite gradient",
        previous_max_abs_grad=8748.48035918326,
        expected_issue_type="power-law real-domain safety",
        probe_input="S",
        active_grad_inputs=("S", "p1", "p2"),
        output_storage_cap="S",
    ),
    TargetContext(
        formula="interflow_2",
        flux_module="models.flux.interflow",
        active_model="hbv96",
        previous_risk="non-finite output / non-finite gradient",
        previous_max_abs_grad=1318.2567399273944,
        expected_issue_type="power-law real-domain safety",
        probe_input="S",
        active_grad_inputs=("S", "p1", "p2"),
        output_storage_cap="S",
    ),
    TargetContext(
        formula="interflow_3",
        flux_module="models.flux.interflow",
        active_model="australia",
        previous_risk="non-finite output / non-finite gradient",
        previous_max_abs_grad=1318.2567399273944,
        expected_issue_type="power-law real-domain safety",
        probe_input="S",
        active_grad_inputs=("S", "p1", "p2"),
        output_storage_cap="S",
        call_site_labels=("subsurface_flow", "groundwater_baseflow"),
    ),
    TargetContext(
        formula="interflow_3",
        flux_module="models.flux.interflow",
        active_model="susannah2",
        previous_risk="non-finite output / non-finite gradient",
        previous_max_abs_grad=1318.2567399273944,
        expected_issue_type="power-law real-domain safety",
        probe_input="S",
        active_grad_inputs=("S", "p1", "p2"),
        output_storage_cap="S",
        call_site_labels=("subsurface_flow", "groundwater_sink"),
    ),
    TargetContext(
        formula="baseflow_5",
        flux_module="models.flux.baseflow",
        active_model="vic",
        previous_risk="non-finite output / non-finite gradient",
        previous_max_abs_grad=999.9990010010001,
        expected_issue_type="scaled power-law / near-zero ratio safety",
        probe_input="S",
        active_grad_inputs=("S", "Smax", "p1", "p2"),
        output_storage_cap="S",
    ),
)


CAPTURE_CASES: dict[str, tuple[ValidationCase, ...]] = {
    "tcm": (
        ValidationCase("impulse_short", "impulse", 20, (1, 1), "lower_near", "zero"),
        ValidationCase("constant_short", "constant", 60, (1, 1), "midpoint", "moderate"),
        ValidationCase("random_short", "random_positive", 60, (2, 2), "random_valid", "random"),
        ValidationCase("very_wet_short", "very_wet", 60, (1, 1), "upper_near", "moderate"),
    ),
    "topmodel": (
        ValidationCase("impulse_short", "impulse", 20, (1, 1), "lower_near", "zero"),
        ValidationCase("constant_short", "constant", 60, (1, 1), "midpoint", "moderate"),
        ValidationCase("random_short", "random_positive", 60, (2, 2), "random_valid", "random"),
        ValidationCase("very_wet_short", "very_wet", 60, (1, 1), "upper_near", "moderate"),
    ),
    "susannah1": (
        ValidationCase("impulse_short", "impulse", 20, (1, 1), "lower_near", "zero"),
        ValidationCase("constant_short", "constant", 60, (1, 1), "midpoint", "moderate"),
        ValidationCase("random_short", "random_positive", 60, (2, 2), "random_valid", "random"),
        ValidationCase("very_wet_short", "very_wet", 60, (1, 1), "upper_near", "moderate"),
    ),
    "hbv96": (
        ValidationCase("constant_short", "constant", 60, (1, 1), "midpoint", "moderate"),
        ValidationCase("random_short", "random_positive", 60, (2, 2), "random_valid", "random"),
        ValidationCase("snow_cold_warm", "snow_cold_warm", 60, (1, 1), "midpoint", "zero"),
        ValidationCase("snow_transition", "snow_transition", 60, (1, 1), "upper_near", "moderate"),
    ),
    "australia": (
        ValidationCase("impulse_short", "impulse", 20, (1, 1), "lower_near", "zero"),
        ValidationCase("constant_short", "constant", 60, (1, 1), "midpoint", "moderate"),
        ValidationCase("random_short", "random_positive", 60, (2, 2), "random_valid", "random"),
        ValidationCase("very_wet_short", "very_wet", 60, (1, 1), "upper_near", "moderate"),
    ),
    "susannah2": (
        ValidationCase("impulse_short", "impulse", 20, (1, 1), "lower_near", "zero"),
        ValidationCase("constant_short", "constant", 60, (1, 1), "midpoint", "moderate"),
        ValidationCase("random_short", "random_positive", 60, (2, 2), "random_valid", "random"),
        ValidationCase("very_wet_short", "very_wet", 60, (1, 1), "upper_near", "moderate"),
    ),
    "vic": (
        ValidationCase("impulse_short", "impulse", 20, (1, 1), "lower_near", "zero"),
        ValidationCase("constant_short", "constant", 60, (1, 1), "midpoint", "moderate"),
        ValidationCase("random_short", "random_positive", 60, (2, 2), "random_valid", "random"),
        ValidationCase("very_wet_short", "very_wet", 60, (1, 1), "upper_near", "moderate"),
        ValidationCase("high_pet_short", "high_pet", 60, (1, 1), "midpoint", "large"),
    ),
}


REALISTIC_CASES_BY_FORMULA: dict[str, tuple[str, ...]] = {
    "baseflow_6": ("lower", "mid", "upper", "threshold_at", "threshold_plus", "random"),
    "interflow_10": ("lower", "mid", "upper", "threshold_minus", "threshold_at", "threshold_plus", "random"),
    "baseflow_2": ("lower", "mid", "upper", "near_zero", "random"),
    "interflow_2": ("lower", "mid", "upper", "near_zero", "random"),
    "interflow_3": ("lower", "mid", "upper", "near_zero", "random"),
    "baseflow_5": ("lower", "mid", "upper", "near_zero", "random"),
}


def _target_id(target: TargetContext) -> str:
    return f"{target.formula}::{target.active_model}"


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_previous_summary_rows() -> list[dict[str, Any]]:
    path = PREVIOUS_RESULTS_DIR / "flux_gradient_stability_summary.csv"
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_previous_ranking_rows() -> list[dict[str, Any]]:
    path = PREVIOUS_RESULTS_DIR / "flux_gradient_risk_ranking.csv"
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite_difference(
    callable_fn: Callable[..., torch.Tensor],
    arg_order: list[str],
    base_inputs: dict[str, torch.Tensor],
    input_name: str,
    eps: float = FD_EPS,
) -> tuple[float, int]:
    center = {name: tensor.detach().clone() for name, tensor in base_inputs.items()}
    flat = center[input_name].reshape(-1)
    if flat.numel() == 0:
        return math.nan, 0
    index = flat.numel() // 2
    plus = {name: tensor.detach().clone() for name, tensor in center.items()}
    minus = {name: tensor.detach().clone() for name, tensor in center.items()}
    plus[input_name].reshape(-1)[index] += eps
    minus[input_name].reshape(-1)[index] -= eps
    with torch.no_grad():
        y_plus = callable_fn(*[plus[name] for name in arg_order], nearzero=DEFAULT_NEARZERO).reshape(-1)[index].item()
        y_minus = callable_fn(*[minus[name] for name in arg_order], nearzero=DEFAULT_NEARZERO).reshape(-1)[index].item()
    return (y_plus - y_minus) / (2.0 * eps), index


def _autograd_gradients(output: torch.Tensor, inputs: dict[str, torch.Tensor], grad_inputs: tuple[str, ...]) -> dict[str, torch.Tensor | None]:
    ordered = [inputs[name] for name in grad_inputs if inputs[name].requires_grad]
    grads = torch.autograd.grad(output.sum(), ordered, allow_unused=True, retain_graph=False)
    result: dict[str, torch.Tensor | None] = {}
    grad_iter = iter(grads)
    for name in grad_inputs:
        tensor = inputs[name]
        if tensor.requires_grad:
            result[name] = next(grad_iter)
        else:
            result[name] = None
    return result


def _sign_flip_count(grad: torch.Tensor | None) -> int:
    if grad is None or grad.numel() < 2:
        return 0
    flat = grad.detach().reshape(-1)
    signs = torch.sign(flat)
    return int((signs[1:] * signs[:-1] < 0).sum().item())


def _risk_level_from_metrics(
    output_nan_count: int,
    output_inf_count: int,
    grad_nan_count: int,
    grad_inf_count: int,
    output_bound_violation_count: int,
    max_abs_grad: float,
    zero_gradient_fraction: float,
) -> str:
    if any(
        [
            output_nan_count > 0,
            output_inf_count > 0,
            grad_nan_count > 0,
            grad_inf_count > 0,
            output_bound_violation_count > 0,
        ]
    ):
        return "high"
    if max_abs_grad > 1.0e4 or zero_gradient_fraction > 0.8:
        return "medium"
    return "low"


def _line_bounds_for_function(path: Path, function_name: str) -> tuple[int, int]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node.lineno, getattr(node, "end_lineno", node.lineno)
    raise KeyError(function_name)


def _source_snippet(path: Path, line_start: int, line_end: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[line_start - 1:line_end])


def _usage_rows_for_target(target: TargetContext) -> list[dict[str, Any]]:
    usage = load_flux_usage_contexts()
    rows = []
    for ctx in usage:
        if ctx.flux_function != target.formula or ctx.model_name != target.active_model:
            continue
        rows.append(
            {
                "call_site": ctx.call_site,
                "parameter_mapping": ctx.parameter_mapping,
                "state_variable_mapping": ctx.state_variable_mapping,
                "forcing_variable_mapping": ctx.forcing_variable_mapping,
                "parameter_bounds": ctx.parameter_bounds,
                "inferred_or_exact": ctx.inferred_or_exact,
            }
        )
    return rows


def _context_parameter_bounds(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    merged: dict[str, tuple[float, float]] = {}
    for row in rows:
        for key, bounds in row["parameter_bounds"].items():
            lo, hi = float(bounds[0]), float(bounds[1])
            if key not in merged:
                merged[key] = (lo, hi)
            else:
                merged[key] = (min(merged[key][0], lo), max(merged[key][1], hi))
    return merged


def _call_sites_summary(rows: list[dict[str, Any]]) -> tuple[str, str]:
    if not rows:
        return "", ""
    first_call = rows[0]["call_site"]
    file_path, _, line_text = first_call.partition(":")
    line_numbers = [row["call_site"].partition(":")[2] for row in rows]
    return file_path, ",".join(line_numbers)


def _arg_order_for_target(target: TargetContext) -> list[str]:
    fn = getattr(importlib.import_module(target.flux_module), target.formula)
    return [name for name in inspect.signature(fn).parameters if name != "nearzero"]


class _RangeRecorder:
    def __init__(self, arg_order: list[str]):
        self.arg_order = arg_order
        self.ranges: dict[str, dict[str, float]] = {}
        self.call_count = 0

    def update(self, *args: torch.Tensor) -> None:
        self.call_count += 1
        for name, tensor in zip(self.arg_order, args):
            flat = tensor.detach().reshape(-1)
            finite = flat[torch.isfinite(flat)]
            if finite.numel() == 0:
                continue
            lo = float(finite.min().item())
            hi = float(finite.max().item())
            if name not in self.ranges:
                self.ranges[name] = {"min": lo, "max": hi}
            else:
                self.ranges[name]["min"] = min(self.ranges[name]["min"], lo)
                self.ranges[name]["max"] = max(self.ranges[name]["max"], hi)


@contextmanager
def _patched_target_flux(target: TargetContext, recorder: _RangeRecorder):
    core_module = importlib.import_module(f"models.core.{target.active_model}")
    flux_module = importlib.import_module(target.flux_module)
    original_core_fn = getattr(core_module, target.formula)
    original_flux_fn = getattr(flux_module, target.formula)

    def wrapped(*args: torch.Tensor, nearzero: float = DEFAULT_NEARZERO) -> torch.Tensor:
        recorder.update(*args)
        return original_flux_fn(*args, nearzero=nearzero)

    setattr(core_module, target.formula, wrapped)
    try:
        yield
    finally:
        setattr(core_module, target.formula, original_core_fn)


def capture_realistic_domain(target: TargetContext) -> dict[str, dict[str, float]]:
    recorder = _RangeRecorder(_arg_order_for_target(target))
    entry = CORE_MODEL_REGISTRY[target.active_model]
    torch.manual_seed(FIXED_SEED)
    with _patched_target_flux(target, recorder):
        for case in CAPTURE_CASES[target.active_model]:
            run_validation_case(entry, case, DEFAULT_DTYPE, DEFAULT_DEVICE)
    return recorder.ranges


def _state_domain_summary(target: TargetContext, captured_ranges: dict[str, dict[str, float]]) -> str:
    ordered = {name: captured_ranges[name] for name in _arg_order_for_target(target) if name in captured_ranges}
    return _json_dump(ordered)


def _realistic_case_value(case_name: str, arg_name: str, domain: dict[str, dict[str, float]], target: TargetContext, shape: tuple[int, ...]) -> torch.Tensor:
    lo = domain[arg_name]["min"]
    hi = domain[arg_name]["max"]
    span = max(hi - lo, 1.0e-12)
    if case_name == "lower":
        value = lo
    elif case_name == "upper":
        value = hi
    elif case_name == "mid":
        value = 0.5 * (lo + hi)
    elif case_name == "near_zero":
        value = max(0.0, min(hi, DEFAULT_NEARZERO))
    elif case_name == "random":
        generator = torch.Generator(device=DEFAULT_DEVICE)
        generator.manual_seed(FIXED_SEED + sum(ord(ch) for ch in f"{target.formula}:{target.active_model}:{arg_name}"))
        return (lo + (hi - lo) * torch.rand(shape, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE, generator=generator))
    elif case_name in {"threshold_minus", "threshold_at", "threshold_plus"} and arg_name == target.probe_input:
        threshold = domain.get("p2", {"min": lo, "max": lo})
        threshold_value = 0.5 * (threshold["min"] + threshold["max"])
        eps = max(1.0e-3, 0.01 * span)
        if case_name == "threshold_minus":
            value = max(0.0, threshold_value - eps)
        elif case_name == "threshold_at":
            value = max(0.0, threshold_value)
        else:
            value = max(0.0, threshold_value + eps)
    else:
        value = 0.5 * (lo + hi)
    return torch.full(shape, float(value), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)


def build_realistic_inputs(target: TargetContext, captured_ranges: dict[str, dict[str, float]], case_name: str, shape: tuple[int, ...] = REALISTIC_SHAPE) -> dict[str, torch.Tensor]:
    inputs: dict[str, torch.Tensor] = {}
    for arg_name in _arg_order_for_target(target):
        tensor = _realistic_case_value(case_name, arg_name, captured_ranges, target, shape)
        tensor = tensor.requires_grad_(arg_name in target.active_grad_inputs)
        inputs[arg_name] = tensor
    return inputs


def representative_realistic_inputs(target: TargetContext, shape: tuple[int, ...] = REALISTIC_SHAPE) -> dict[str, torch.Tensor]:
    if target.formula == "baseflow_6":
        values = {
            "p1": torch.full(shape, 0.3, dtype=DEFAULT_DTYPE),
            "p2": torch.zeros(shape, dtype=DEFAULT_DTYPE),
            "S": torch.linspace(0.0, 50.0, steps=shape[0], dtype=DEFAULT_DTYPE),
        }
    elif target.formula == "interflow_10":
        values = {
            "S": torch.linspace(70.0, 95.0, steps=shape[0], dtype=DEFAULT_DTYPE),
            "p1": torch.full(shape, 0.5, dtype=DEFAULT_DTYPE),
            "p2": torch.full(shape, 80.0, dtype=DEFAULT_DTYPE),
            "p3": torch.full(shape, 20.0, dtype=DEFAULT_DTYPE),
        }
    elif target.formula == "baseflow_2":
        values = {
            "S": torch.linspace(DEFAULT_NEARZERO, 40.0, steps=shape[0], dtype=DEFAULT_DTYPE),
            "p1": torch.full(shape, 10.0, dtype=DEFAULT_DTYPE),
            "p2": torch.full(shape, 0.5, dtype=DEFAULT_DTYPE),
        }
    elif target.formula == "interflow_2":
        values = {
            "p1": torch.full(shape, 0.3, dtype=DEFAULT_DTYPE),
            "S": torch.linspace(DEFAULT_NEARZERO, 50.0, steps=shape[0], dtype=DEFAULT_DTYPE),
            "p2": torch.full(shape, 1.0, dtype=DEFAULT_DTYPE),
        }
    elif target.formula == "interflow_3":
        values = {
            "p1": torch.full(shape, 0.4, dtype=DEFAULT_DTYPE),
            "p2": torch.full(shape, 2.0, dtype=DEFAULT_DTYPE),
            "S": torch.linspace(DEFAULT_NEARZERO, 50.0, steps=shape[0], dtype=DEFAULT_DTYPE),
        }
    elif target.formula == "baseflow_5":
        values = {
            "p1": torch.full(shape, 0.3, dtype=DEFAULT_DTYPE),
            "p2": torch.full(shape, 2.0, dtype=DEFAULT_DTYPE),
            "S": torch.linspace(DEFAULT_NEARZERO, 100.0, steps=shape[0], dtype=DEFAULT_DTYPE),
            "Smax": torch.full(shape, 500.0, dtype=DEFAULT_DTYPE),
        }
    else:
        raise KeyError(target.formula)

    result = {}
    for name, tensor in values.items():
        result[name] = tensor.to(device=DEFAULT_DEVICE).requires_grad_(name in target.active_grad_inputs)
    return result


def _evaluate_direct_case(
    target: TargetContext,
    case_name: str,
    domain_type: str,
    inputs: dict[str, torch.Tensor],
) -> dict[str, Any]:
    callable_fn = getattr(importlib.import_module(target.flux_module), target.formula)
    arg_order = _arg_order_for_target(target)
    output = callable_fn(*[inputs[name] for name in arg_order], nearzero=DEFAULT_NEARZERO)
    grads = _autograd_gradients(output, inputs, target.active_grad_inputs)

    output_nan_count = int(torch.isnan(output).sum().item())
    output_inf_count = int(torch.isinf(output).sum().item())
    negative_output_count = int((output < -1.0e-12).sum().item())
    output_exceeds_storage_count = 0
    if target.output_storage_cap is not None:
        cap = inputs[target.output_storage_cap]
        output_exceeds_storage_count = int((output > cap + 1.0e-10).sum().item())

    grad_nan_count = 0
    grad_inf_count = 0
    max_abs_grad = 0.0
    mean_abs_grad = 0.0
    median_abs_grad = 0.0
    grad_l2_norm = 0.0
    sign_flip_count = 0
    zero_gradient_fraction = 1.0
    gradient_saturation_ratio = 1.0

    grad_values = []
    grad_zero_fractions = []
    grad_sat_fractions = []
    for grad in grads.values():
        if grad is None:
            continue
        grad_nan_count += int(torch.isnan(grad).sum().item())
        grad_inf_count += int(torch.isinf(grad).sum().item())
        abs_grad = torch.abs(grad)
        if abs_grad.numel() > 0:
            grad_values.append(abs_grad.reshape(-1))
            max_abs_grad = max(max_abs_grad, float(abs_grad.max().item()))
            grad_zero_fractions.append(float((abs_grad < 1.0e-12).float().mean().item()))
            grad_sat_fractions.append(float((abs_grad < 1.0e-8).float().mean().item()))
        sign_flip_count += _sign_flip_count(grad)

    if grad_values:
        flat = torch.cat(grad_values)
        mean_abs_grad = float(flat.mean().item())
        median_abs_grad = float(flat.median().item())
        grad_l2_norm = float(torch.linalg.norm(flat).item())
        zero_gradient_fraction = max(grad_zero_fractions)
        gradient_saturation_ratio = max(grad_sat_fractions)

    autograd_fd_relative_error = math.nan
    autograd_fd_max_error = math.nan
    probe_name = target.probe_input
    if probe_name in inputs and inputs[probe_name].requires_grad and grads.get(probe_name) is not None:
        fd_grad, fd_index = _finite_difference(callable_fn, arg_order, inputs, probe_name)
        ad_grad = grads[probe_name].reshape(-1)[fd_index].item()
        autograd_fd_max_error = abs(ad_grad - fd_grad)
        autograd_fd_relative_error = abs(ad_grad - fd_grad) / max(abs(fd_grad), 1.0e-12)

    row = {
        "formula": target.formula,
        "active_model": target.active_model,
        "domain_type": domain_type,
        "test_case": case_name,
        "output_nan_count": output_nan_count,
        "output_inf_count": output_inf_count,
        "grad_nan_count": grad_nan_count,
        "grad_inf_count": grad_inf_count,
        "max_abs_grad": max_abs_grad,
        "mean_abs_grad": mean_abs_grad,
        "median_abs_grad": median_abs_grad,
        "grad_l2_norm": grad_l2_norm,
        "zero_gradient_fraction": zero_gradient_fraction,
        "dead_region_fraction": zero_gradient_fraction,
        "gradient_saturation_ratio": gradient_saturation_ratio,
        "autograd_fd_relative_error": autograd_fd_relative_error,
        "autograd_fd_max_error": autograd_fd_max_error,
        "output_bound_violation_count": output_exceeds_storage_count,
        "negative_output_count": negative_output_count,
        "output_exceeds_storage_count": output_exceeds_storage_count,
        "max_output": float(torch.max(output).item()),
        "min_output": float(torch.min(output).item()),
        "mean_output": float(torch.mean(output).item()),
        "sign_flip_count": sign_flip_count,
    }
    row["risk_level"] = _risk_level_from_metrics(
        row["output_nan_count"],
        row["output_inf_count"],
        row["grad_nan_count"],
        row["grad_inf_count"],
        row["output_bound_violation_count"],
        row["max_abs_grad"],
        row["zero_gradient_fraction"],
    )
    return row


def build_realistic_summary_rows(target: TargetContext, captured_ranges: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for case_name in REALISTIC_CASES_BY_FORMULA[target.formula]:
        inputs = build_realistic_inputs(target, captured_ranges, case_name)
        rows.append(_evaluate_direct_case(target, case_name, "realistic", inputs))
    return rows


def build_broad_summary_rows(target: TargetContext) -> list[dict[str, Any]]:
    previous_summary = _load_previous_summary_rows()
    rows = []
    for row in previous_summary:
        if row["flux_function"] != target.formula or row["model_context"] != target.active_model:
            continue
        rows.append(
            {
                "formula": target.formula,
                "active_model": target.active_model,
                "domain_type": "broad",
                "test_case": row["test_case"],
                "output_nan_count": int(row["output_nan_count"]),
                "output_inf_count": int(row["output_inf_count"]),
                "grad_nan_count": int(row["grad_nan_count"]),
                "grad_inf_count": int(row["grad_inf_count"]),
                "max_abs_grad": _float(row["max_abs_grad"], 0.0),
                "mean_abs_grad": _float(row["mean_abs_grad"], 0.0),
                "median_abs_grad": _float(row["median_abs_grad"], 0.0),
                "grad_l2_norm": _float(row["grad_l2_norm"], 0.0),
                "zero_gradient_fraction": _float(row["zero_gradient_fraction"], 1.0),
                "dead_region_fraction": _float(row["zero_gradient_fraction"], 1.0),
                "gradient_saturation_ratio": _float(row["gradient_saturation_ratio"], 1.0),
                "autograd_fd_relative_error": _float(row["autograd_fd_relative_error"], math.nan),
                "autograd_fd_max_error": _float(row["autograd_fd_max_error"], math.nan),
                "output_bound_violation_count": int(row["output_bound_violation_count"]),
                "negative_output_count": int(row["output_negative_count"]),
                "output_exceeds_storage_count": int(row["output_bound_violation_count"]),
                "max_output": _float(row["max_output"], math.nan),
                "min_output": _float(row["min_output"], math.nan),
                "mean_output": _float(row["mean_output"], math.nan),
                "sign_flip_count": int(row["sign_flip_count"]),
                "risk_level": "high" if row["pass_fail"] == "False" else "low",
            }
        )
    return rows


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Expected non-empty rows.")
    output_nan = sum(int(row["output_nan_count"]) for row in rows)
    output_inf = sum(int(row["output_inf_count"]) for row in rows)
    grad_nan = sum(int(row["grad_nan_count"]) for row in rows)
    grad_inf = sum(int(row["grad_inf_count"]) for row in rows)
    max_abs_grad = max(float(row["max_abs_grad"]) for row in rows)
    zero_gradient_fraction = max(float(row["zero_gradient_fraction"]) for row in rows)
    output_bound_violations = sum(int(row["output_bound_violation_count"]) for row in rows)
    if any([output_nan, output_inf, grad_nan, grad_inf, output_bound_violations]):
        risk = "high"
    elif max_abs_grad > 1.0e4 or zero_gradient_fraction > 0.8:
        risk = "medium"
    else:
        risk = "low"
    return {
        "risk": risk,
        "output_nan_count": output_nan,
        "output_inf_count": output_inf,
        "grad_nan_count": grad_nan,
        "grad_inf_count": grad_inf,
        "output_bound_violation_count": output_bound_violations,
        "max_abs_grad": max_abs_grad,
        "zero_gradient_fraction": zero_gradient_fraction,
        "autograd_fd_relative_error": max(
            [_float(row["autograd_fd_relative_error"], 0.0) for row in rows if not math.isnan(_float(row["autograd_fd_relative_error"], math.nan))],
            default=math.nan,
        ),
    }


def _previous_risk_level(target: TargetContext) -> str:
    ranking_rows = _load_previous_ranking_rows()
    for row in ranking_rows:
        if row["flux_function"] == target.formula and row["called_by_models"] == target.active_model:
            return row["risk_level"]
    return "unknown"


def _recommended_action(target: TargetContext, realistic_summary: dict[str, Any], broad_summary: dict[str, Any]) -> tuple[str, str, str]:
    realistic_has_nonfinite = any(
        [
            realistic_summary["output_nan_count"] > 0,
            realistic_summary["output_inf_count"] > 0,
            realistic_summary["grad_nan_count"] > 0,
            realistic_summary["grad_inf_count"] > 0,
        ]
    )
    if realistic_has_nonfinite:
        return "add_safety_clamp_later", "high", "Non-finite values persist in realistic model-domain tests."
    if realistic_summary["output_bound_violation_count"] > 0:
        return "check_parameter_range", "high", "Realistic-domain outputs exceed expected physical caps."
    if target.formula in {"baseflow_6", "interflow_10"}:
        if realistic_summary["risk"] == "medium":
            return "keep_but_document", "medium", "Finite but threshold-sensitive gradients remain under realistic domains."
        return "keep_but_document", "low", "Realistic-domain behavior is finite; broad risk is dominated by threshold-domain diagnostics."
    if broad_summary["risk"] == "high" and realistic_summary["risk"] == "low":
        return "revise_state_domain_test", "low", "Broad-domain failures do not persist when inputs are constrained to active model domains."
    return "safe_no_action", "low", "Realistic-domain outputs and gradients are finite and physically bounded."


def build_risk_decision_rows(
    combined_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    decision_rows = []
    for target in TARGET_CONTEXTS:
        target_rows = [row for row in combined_rows if row["formula"] == target.formula and row["active_model"] == target.active_model]
        broad_rows = [row for row in target_rows if row["domain_type"] == "broad"]
        realistic_rows = [row for row in target_rows if row["domain_type"] == "realistic"]
        broad_summary = _aggregate_rows(broad_rows)
        realistic_summary = _aggregate_rows(realistic_rows)

        if broad_summary["risk"] == "high" and realistic_summary["risk"] in {"low", "medium"} and realistic_summary["output_nan_count"] == 0 and realistic_summary["grad_nan_count"] == 0:
            artifact_label = "broad_domain_artifact"
        elif realistic_summary["risk"] == "high":
            artifact_label = "realistic_domain_concern"
        else:
            artifact_label = "mixed_threshold_behavior"

        recommended_action, human_review_priority, short_reason = _recommended_action(target, realistic_summary, broad_summary)
        decision_rows.append(
            {
                "formula": target.formula,
                "active_model": target.active_model,
                "previous_risk": target.previous_risk,
                "broad_domain_risk": broad_summary["risk"],
                "realistic_domain_risk": realistic_summary["risk"],
                "output_nan_inf_realistic": realistic_summary["output_nan_count"] + realistic_summary["output_inf_count"],
                "grad_nan_inf_realistic": realistic_summary["grad_nan_count"] + realistic_summary["grad_inf_count"],
                "max_abs_grad_realistic": realistic_summary["max_abs_grad"],
                "main_failure_mode": short_reason,
                "likely_artifact_or_real": artifact_label,
                "recommended_action": recommended_action,
                "human_review_priority": human_review_priority,
                "short_reason": short_reason,
            }
        )
    return decision_rows


def build_target_context_rows(captured_domains: dict[str, dict[str, dict[str, float]]]) -> list[dict[str, Any]]:
    rows = []
    for target in TARGET_CONTEXTS:
        usage_rows = _usage_rows_for_target(target)
        parameter_mapping = {}
        state_mapping = {}
        for row in usage_rows:
            parameter_mapping.update(row["parameter_mapping"])
            state_mapping.update(row["state_variable_mapping"])
        call_site_file, call_site_lines = _call_sites_summary(usage_rows)
        flux_rel_path = Path(target.flux_module.replace(".", "/") + ".py")
        flux_abs_path = REPO_ROOT / flux_rel_path
        flux_line_start, flux_line_end = _line_bounds_for_function(flux_abs_path, target.formula)
        rows.append(
            {
                "formula": target.formula,
                "flux_file": str(flux_rel_path),
                "flux_lines": f"{flux_line_start}-{flux_line_end}",
                "active_model": target.active_model,
                "call_site_file": call_site_file,
                "call_site_lines": call_site_lines,
                "parameter_mapping": _json_dump(parameter_mapping),
                "parameter_bounds": _json_dump(_context_parameter_bounds(usage_rows)),
                "state_mapping": _json_dump(state_mapping),
                "inferred_state_domain": _state_domain_summary(target, captured_domains[_target_id(target)]),
                "exact_or_inferred_domain": "captured_active_model_rollout",
                "notes": target.expected_issue_type,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_source_context_markdown(captured_domains: dict[str, dict[str, dict[str, float]]]) -> str:
    lines = [
        "# Focused Flux Formula Source Context",
        "",
        "This file records the exact target flux snippets, the active core-model call context,",
        "and the captured realistic call domains used in the focused review.",
        "",
    ]
    for target in TARGET_CONTEXTS:
        flux_path = REPO_ROOT / f"{target.flux_module.replace('.', '/')}.py"
        flux_start, flux_end = _line_bounds_for_function(flux_path, target.formula)
        flux_snippet = _source_snippet(flux_path, flux_start, flux_end)
        usage_rows = _usage_rows_for_target(target)
        lines.append(f"## {target.formula} / {target.active_model}")
        lines.append("")
        lines.append(f"- Previous risk: {target.previous_risk}")
        lines.append(f"- Previous max_abs_grad: {target.previous_max_abs_grad}")
        lines.append(f"- Expected issue type: {target.expected_issue_type}")
        lines.append(f"- Realistic captured domain: `{_state_domain_summary(target, captured_domains[_target_id(target)])}`")
        lines.append("")
        lines.append("### Flux code")
        lines.append("")
        lines.append(f"`{flux_path.relative_to(REPO_ROOT)}:{flux_start}-{flux_end}`")
        lines.append("")
        lines.append("```python")
        lines.append(flux_snippet)
        lines.append("```")
        lines.append("")
        lines.append("### Core call sites")
        lines.append("")
        for row in usage_rows:
            call_site = row["call_site"]
            file_path_str, _, line_text = call_site.partition(":")
            file_path = REPO_ROOT / file_path_str
            line_no = int(line_text)
            snippet = _source_snippet(file_path, max(1, line_no - 2), line_no + 3)
            lines.append(f"`{call_site}`")
            lines.append("")
            lines.append("```python")
            lines.append(snippet)
            lines.append("```")
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def _plot_target(target: TargetContext, captured_domain: dict[str, dict[str, float]]) -> None:
    callable_fn = getattr(importlib.import_module(target.flux_module), target.formula)
    arg_order = _arg_order_for_target(target)
    broad_inputs = representative_realistic_inputs(target)
    realistic_inputs = build_realistic_inputs(target, captured_domain, "mid")
    broad_probe = broad_inputs[target.probe_input].detach()
    realistic_probe = realistic_inputs[target.probe_input].detach()
    broad_grid = torch.linspace(float(broad_probe.min().item()), float(broad_probe.max().item()), steps=128, dtype=DEFAULT_DTYPE)
    realistic_grid = torch.linspace(float(realistic_probe.min().item()), float(realistic_probe.max().item()), steps=128, dtype=DEFAULT_DTYPE)

    def compute_curve(base_inputs: dict[str, torch.Tensor], grid: torch.Tensor) -> tuple[list[float], list[float]]:
        outputs = []
        grads = []
        for value in grid:
            current_inputs = {}
            for name, tensor in base_inputs.items():
                if name == target.probe_input:
                    current_inputs[name] = torch.full_like(tensor, float(value), requires_grad=(name in target.active_grad_inputs))
                else:
                    current_inputs[name] = tensor.detach().clone().requires_grad_(name in target.active_grad_inputs)
            output = callable_fn(*[current_inputs[name] for name in arg_order], nearzero=DEFAULT_NEARZERO)
            outputs.append(float(output.mean().item()))
            if target.probe_input in target.active_grad_inputs:
                grad = torch.autograd.grad(output.sum(), current_inputs[target.probe_input], allow_unused=True)[0]
                grads.append(float(torch.abs(grad).mean().item()) if grad is not None else 0.0)
            else:
                grads.append(0.0)
        return outputs, grads

    broad_outputs, broad_grads = compute_curve(broad_inputs, broad_grid)
    realistic_outputs, realistic_grads = compute_curve(realistic_inputs, realistic_grid)

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=False)
    axes[0].plot(broad_grid.cpu().numpy(), broad_outputs, label="broad")
    axes[0].plot(realistic_grid.cpu().numpy(), realistic_outputs, label="realistic")
    axes[0].set_ylabel("output")
    axes[0].set_title(f"{target.formula} [{target.active_model}]")
    axes[0].legend()

    axes[1].plot(broad_grid.cpu().numpy(), broad_grads, label="broad")
    axes[1].plot(realistic_grid.cpu().numpy(), realistic_grads, label="realistic")
    axes[1].set_ylabel("|grad|")
    axes[1].set_xlabel(target.probe_input)
    axes[1].legend()

    if target.formula in {"baseflow_6", "interflow_10"} and "p2" in captured_domain:
        threshold_value = 0.5 * (captured_domain["p2"]["min"] + captured_domain["p2"]["max"])
        axes[0].axvline(threshold_value, color="black", linestyle="--", linewidth=1.0)
        axes[1].axvline(threshold_value, color="black", linestyle="--", linewidth=1.0)

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / f"{target.formula}_{target.active_model}.png", dpi=160)
    plt.close(fig)


def _build_review_report(
    target_context_rows: list[dict[str, Any]],
    combined_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Focused Flux Formula Review Report",
        "",
        "## 1. Scope",
        "- Focused review of six previously high-risk active flux formulas from the earlier flux-gradient-stability diagnostic.",
        "- This review did not change any hydrological formulas, smoothing defaults, unit hydrograph code, or water-balance fixes.",
        "",
        "## 2. Previous high-risk flags",
        "- The previous workflow intentionally used broad diagnostic domains to stress formulas outside normal active-model operating ranges.",
        "- This focused review separates those broad-domain findings from realistic active-model call domains captured during short synthetic model rollouts.",
        "",
        "## 3. Source and call context",
        "- Source and call snippets are recorded in `source_context_summary.md`.",
        "- Captured realistic domains are recorded in `target_formula_context.csv` under `inferred_state_domain`.",
        "",
        "## 4. Broad vs realistic domain design",
        "- Broad domain: reused the previous generic stress-test results from `flux_gradient_stability_summary.csv`.",
        "- Realistic domain: traced actual arguments passed by active core models during short deterministic rollouts with model-specific parameter bounds and stabilized initial states.",
        "",
    ]

    for target in TARGET_CONTEXTS:
        decision = next(row for row in decision_rows if row["formula"] == target.formula and row["active_model"] == target.active_model)
        realistic_rows = [
            row for row in combined_rows
            if row["formula"] == target.formula and row["active_model"] == target.active_model and row["domain_type"] == "realistic"
        ]
        realistic_max_grad = max(float(row["max_abs_grad"]) for row in realistic_rows)
        realistic_nonfinite = sum(
            int(row["output_nan_count"]) + int(row["output_inf_count"]) + int(row["grad_nan_count"]) + int(row["grad_inf_count"])
            for row in realistic_rows
        )
        lines.extend(
            [
                f"## {5 + list(TARGET_CONTEXTS).index(target)}. Results for {target.formula} / {target.active_model}",
                f"- Previous risk: {target.previous_risk}",
                f"- Broad-domain risk: {decision['broad_domain_risk']}",
                f"- Realistic-domain risk: {decision['realistic_domain_risk']}",
                f"- Realistic-domain max_abs_grad: {realistic_max_grad:.6g}",
                f"- Realistic-domain NaN/Inf count: {realistic_nonfinite}",
                f"- Recommended action: {decision['recommended_action']}",
                f"- Human review priority: {decision['human_review_priority']}",
                f"- Short reason: {decision['short_reason']}",
                "",
            ]
        )

    broad_artifacts = [row for row in decision_rows if row["likely_artifact_or_real"] == "broad_domain_artifact"]
    realistic_concerns = [row for row in decision_rows if row["likely_artifact_or_real"] == "realistic_domain_concern"]
    lines.extend(
        [
            "## 11. Summary table",
            "",
            "| formula | model | broad risk | realistic risk | recommended action |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in decision_rows:
        lines.append(
            f"| {row['formula']} | {row['active_model']} | {row['broad_domain_risk']} | "
            f"{row['realistic_domain_risk']} | {row['recommended_action']} |"
        )

    lines.extend(
        [
            "",
            "## 12. Which risks were artifacts of broad test domains",
        ]
    )
    if broad_artifacts:
        for row in broad_artifacts:
            lines.append(f"- `{row['formula']}` / `{row['active_model']}`: {row['short_reason']}")
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## 13. Which risks remain under realistic model domains",
        ]
    )
    if realistic_concerns:
        for row in realistic_concerns:
            lines.append(f"- `{row['formula']}` / `{row['active_model']}`: {row['short_reason']}")
    else:
        lines.append("- No non-finite realistic-domain failures remained in this focused review.")

    lines.extend(
        [
            "",
            "## 14. Recommended next actions",
            "- Keep formulas unchanged in this task.",
            "- Document threshold-gradient limitations for formulas that remain finite but have broad dead regions around activation thresholds.",
            "- If future calibration explores threshold neighborhoods aggressively, prioritize human review of the threshold-sensitive formulas before considering any smoothing or safety reformulation.",
            "",
            "## 15. Whether active flux formulas are safe enough for large-scale gradient-based calibration",
            "- Active formulas that only failed in broad impossible domains appear numerically usable under current active-model parameter and state domains.",
            "- Threshold-sensitive formulas still deserve documentation and targeted review before any large benchmark recalibration campaign.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_focus_review() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    torch.manual_seed(FIXED_SEED)
    captured_domains = {_target_id(target): capture_realistic_domain(target) for target in TARGET_CONTEXTS}
    target_context_rows = build_target_context_rows(captured_domains)

    combined_rows: list[dict[str, Any]] = []
    for target in TARGET_CONTEXTS:
        combined_rows.extend(build_broad_summary_rows(target))
        combined_rows.extend(build_realistic_summary_rows(target, captured_domains[_target_id(target)]))
        _plot_target(target, captured_domains[_target_id(target)])

    decision_rows = build_risk_decision_rows(combined_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "target_formula_context.csv", target_context_rows)
    _write_csv(OUTPUT_DIR / "focused_formula_stability_summary.csv", combined_rows)
    _write_csv(OUTPUT_DIR / "focused_formula_risk_decision.csv", decision_rows)
    (OUTPUT_DIR / "source_context_summary.md").write_text(_build_source_context_markdown(captured_domains), encoding="utf-8")
    (OUTPUT_DIR / "focused_flux_formula_review_report.md").write_text(
        _build_review_report(target_context_rows, combined_rows, decision_rows),
        encoding="utf-8",
    )
    return target_context_rows, combined_rows, decision_rows


def main() -> None:
    run_focus_review()


if __name__ == "__main__":
    main()
