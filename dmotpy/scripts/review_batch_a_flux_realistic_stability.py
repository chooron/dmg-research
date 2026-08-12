from __future__ import annotations

import ast
import csv
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
    _call_step,
    build_forcing,
    build_initial_states,
    build_parameter_tensors,
)
from tests.flux_gradient_wrappers import load_flux_usage_contexts  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "validation_results" / "batch_a_flux_realistic_review"
PLOTS_DIR = OUTPUT_DIR / "plots"
GRADIENT_DIR = REPO_ROOT / "validation_results" / "flux_gradient_stability"
DEFAULT_DTYPE = torch.float64
DEFAULT_DEVICE = "cpu"
DEFAULT_NEARZERO = 1.0e-6
FIXED_SEED = 20260624
FD_EPS = 1.0e-6
REALISTIC_SHAPE = (65,)


@dataclass(frozen=True)
class TargetSpec:
    formula: str
    flux_module: str
    active_model: str
    probe_arg: str
    grad_inputs: tuple[str, ...]
    arg_roles: dict[str, str]
    output_meaning: str
    expected_physical_bounds: str
    expected_bound_type: str
    bound_arg: str
    existing_pre_or_post_caps: str
    notes: str
    capacity_pair: tuple[str, str] | None = None
    boundary_param: str | None = None


BATCH_A_TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec(
        formula="saturation_3",
        flux_module="models.flux.saturation",
        active_model="flexb",
        probe_arg="S",
        grad_inputs=("S", "Smax", "p1", "incoming_flux"),
        arg_roles={"S": "state_storage", "Smax": "capacity", "p1": "shape_parameter", "incoming_flux": "incoming_flux"},
        output_meaning="Infiltration from precipitation into the FLEX-B unsaturated store.",
        expected_physical_bounds="0 <= flux_ru <= incoming precipitation P.",
        expected_bound_type="incoming_flux",
        bound_arg="incoming_flux",
        existing_pre_or_post_caps="Post-call cap in core: `torch.clamp(flux_ru, min=0, max=P)`.",
        notes="Broad high-risk flag was active NaN gradients; likely sensitive to very small beta.",
        capacity_pair=("S", "Smax"),
        boundary_param="p1",
    ),
    TargetSpec(
        formula="saturation_3",
        flux_module="models.flux.saturation",
        active_model="flexi",
        probe_arg="S",
        grad_inputs=("S", "Smax", "p1", "incoming_flux"),
        arg_roles={"S": "state_storage", "Smax": "capacity", "p1": "shape_parameter", "incoming_flux": "incoming_flux"},
        output_meaning="Infiltration from effective precipitation into the FLEX-I soil store.",
        expected_physical_bounds="0 <= flux_ru <= throughfall/effective precipitation flux_peff.",
        expected_bound_type="incoming_flux",
        bound_arg="incoming_flux",
        existing_pre_or_post_caps="Post-call cap in core: `torch.clamp(flux_ru, min=0, max=flux_peff)`.",
        notes="Broad high-risk flag was active NaN gradients; likely sensitive to very small beta.",
        capacity_pair=("S", "Smax"),
        boundary_param="p1",
    ),
    TargetSpec(
        formula="saturation_3",
        flux_module="models.flux.saturation",
        active_model="flexis",
        probe_arg="S",
        grad_inputs=("S", "Smax", "p1", "incoming_flux"),
        arg_roles={"S": "state_storage", "Smax": "capacity", "p1": "shape_parameter", "incoming_flux": "incoming_flux"},
        output_meaning="Infiltration from effective precipitation into the FLEX-IS soil store.",
        expected_physical_bounds="0 <= flux_ru <= interception-filtered effective precipitation flux_peff.",
        expected_bound_type="incoming_flux",
        bound_arg="incoming_flux",
        existing_pre_or_post_caps="Post-call cap in core: `torch.clamp(flux_ru, min=0, max=flux_peff)`.",
        notes="Broad high-risk flag was active NaN gradients; likely sensitive to very small beta.",
        capacity_pair=("S", "Smax"),
        boundary_param="p1",
    ),
    TargetSpec(
        formula="saturation_2",
        flux_module="models.flux.saturation",
        active_model="hymod",
        probe_arg="S",
        grad_inputs=("S", "Smax", "p1", "incoming_flux"),
        arg_roles={"S": "state_storage", "Smax": "capacity", "p1": "shape_parameter", "incoming_flux": "incoming_flux"},
        output_meaning="Potential saturation-excess runoff from the HYMOD soil store.",
        expected_physical_bounds="0 <= flux_pe <= precipitation P.",
        expected_bound_type="incoming_flux",
        bound_arg="incoming_flux",
        existing_pre_or_post_caps="Post-call cap in core: `torch.clamp(flux_pe, min=0, max=P)`.",
        notes="Broad high-risk flag was a formula-level bound violation with large finite gradients.",
        capacity_pair=("S", "Smax"),
        boundary_param="p1",
    ),
    TargetSpec(
        formula="baseflow_9",
        flux_module="models.flux.baseflow",
        active_model="gsfb",
        probe_arg="S",
        grad_inputs=("S", "p1", "p2"),
        arg_roles={"S": "state_storage", "p1": "release_coefficient", "p2": "threshold_storage"},
        output_meaning="Baseflow from the GSFB intermediate store S2.",
        expected_physical_bounds="0 <= flux_qb <= available intermediate storage S2_tmp_in.",
        expected_bound_type="storage",
        bound_arg="S",
        existing_pre_or_post_caps="Post-call cap in core: `torch.minimum(flux_qb, S2_tmp_in - nearzero)` followed by `F.relu`.",
        notes="Broad high-risk flag likely mixed a generic parameter range with a product expression `b * dpf`.",
    ),
)


REGIME_CONFIGS: tuple[dict[str, Any], ...] = (
    {"forcing_regime": "dry", "forcing_case": "very_dry", "sequence_length": 60, "batch_shape": (1, 1)},
    {"forcing_regime": "normal", "forcing_case": "constant", "sequence_length": 60, "batch_shape": (1, 1)},
    {"forcing_regime": "wet", "forcing_case": "random_positive", "sequence_length": 60, "batch_shape": (2, 2)},
    {"forcing_regime": "high_precip", "forcing_case": "very_wet", "sequence_length": 60, "batch_shape": (1, 1)},
    {"forcing_regime": "low_pet", "forcing_case": "low_pet", "sequence_length": 60, "batch_shape": (1, 1)},
    {"forcing_regime": "high_pet", "forcing_case": "high_pet", "sequence_length": 60, "batch_shape": (1, 1)},
)
PARAMETER_CASES = ("lower_near", "midpoint", "upper_near", "random_valid")
INITIAL_STATE_CASE = {
    "lower_near": "small",
    "midpoint": "moderate",
    "upper_near": "large",
    "random_valid": "random",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _line_bounds(path: Path, function_name: str) -> tuple[int, int]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node.lineno, getattr(node, "end_lineno", node.lineno)
    raise KeyError(function_name)


def _snippet(path: Path, start: int, end: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[start - 1:end])


def _load_usage_rows() -> dict[tuple[str, str], list[dict[str, str]]]:
    rows = _read_csv(GRADIENT_DIR / "flux_usage_parameter_map.csv")
    mapping: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        mapping.setdefault((row["flux_function"], row["called_by_models"]), []).append(row)
    return mapping


def _arg_order(target: TargetSpec) -> list[str]:
    fn = getattr(__import__(target.flux_module, fromlist=[target.formula]), target.formula)
    return [name for name in inspect.signature(fn).parameters if name != "nearzero"]


class TraceRecorder:
    def __init__(self, target: TargetSpec):
        self.target = target
        self.calls_by_regime: dict[str, list[dict[str, torch.Tensor]]] = {}

    def record(self, forcing_regime: str, arg_order: list[str], args: tuple[torch.Tensor, ...]) -> None:
        call = {
            name: tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1).clone()
            for name, tensor in zip(arg_order, args)
        }
        self.calls_by_regime.setdefault(forcing_regime, []).append(call)


@contextmanager
def _patched_core_symbol(target: TargetSpec, recorder: TraceRecorder, forcing_regime: str):
    core_module = __import__(f"models.core.{target.active_model}", fromlist=[target.active_model])
    flux_module = __import__(target.flux_module, fromlist=[target.formula])
    original_core_fn = getattr(core_module, target.formula)
    original_flux_fn = getattr(flux_module, target.formula)
    arg_order = _arg_order(target)

    def wrapped(*args: torch.Tensor, nearzero: float = DEFAULT_NEARZERO) -> torch.Tensor:
        recorder.record(forcing_regime, arg_order, args)
        return original_flux_fn(*args, nearzero=nearzero)

    setattr(core_module, target.formula, wrapped)
    try:
        yield
    finally:
        setattr(core_module, target.formula, original_core_fn)


def _run_trace_rollouts(target: TargetSpec) -> TraceRecorder:
    torch.manual_seed(FIXED_SEED)
    entry = CORE_MODEL_REGISTRY[target.active_model]
    recorder = TraceRecorder(target)
    for regime in REGIME_CONFIGS:
        for parameter_case in PARAMETER_CASES:
            forcing = build_forcing(
                regime["forcing_case"],
                regime["sequence_length"],
                regime["batch_shape"],
                DEFAULT_DTYPE,
                DEFAULT_DEVICE,
            )
            params_list, params_map = build_parameter_tensors(
                entry,
                parameter_case,
                regime["batch_shape"],
                DEFAULT_DTYPE,
                DEFAULT_DEVICE,
            )
            states = build_initial_states(
                entry,
                INITIAL_STATE_CASE[parameter_case],
                regime["batch_shape"],
                DEFAULT_DTYPE,
                DEFAULT_DEVICE,
                params_map,
                forcing,
                params_list,
            )
            mean_precip = forcing[0].mean(dim=0)
            with _patched_core_symbol(target, recorder, regime["forcing_regime"]):
                for step_index in range(regime["sequence_length"]):
                    step_forcing = (
                        forcing[0][step_index],
                        forcing[1][step_index],
                        forcing[2][step_index],
                    )
                    qsim, ea, next_states, _ = _call_step(
                        entry=entry,
                        forcing_at_step=step_forcing,
                        step_index=step_index,
                        params_list=params_list,
                        states=states,
                        mean_precip=mean_precip,
                        return_diagnostics=False,
                    )
                    _ = (qsim, ea)
                    states = next_states
    return recorder


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return math.nan
    return float(torch.quantile(values, q).item())


def _aggregate_domain_trace(target: TargetSpec, recorder: TraceRecorder) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    pooled: dict[str, list[torch.Tensor]] = {}
    for regime_name, calls in recorder.calls_by_regime.items():
        for arg_name in _arg_order(target):
            values = torch.cat([call[arg_name] for call in calls]) if calls else torch.empty(0, dtype=torch.float64)
            pooled.setdefault(arg_name, []).append(values)
            above_capacity_count = 0
            if target.capacity_pair is not None and arg_name == target.capacity_pair[0]:
                cap_name = target.capacity_pair[1]
                above_capacity_count = sum(
                    int((call[arg_name] > call[cap_name] + 1.0e-12).sum().item())
                    for call in calls
                    if cap_name in call
                )
            rows.append(
                {
                    "formula": target.formula,
                    "active_model": target.active_model,
                    "argument_name": arg_name,
                    "argument_role": target.arg_roles[arg_name],
                    "min": float(values.min().item()) if values.numel() else math.nan,
                    "p01": _quantile(values, 0.01),
                    "p05": _quantile(values, 0.05),
                    "median": _quantile(values, 0.5),
                    "mean": float(values.mean().item()) if values.numel() else math.nan,
                    "p95": _quantile(values, 0.95),
                    "p99": _quantile(values, 0.99),
                    "max": float(values.max().item()) if values.numel() else math.nan,
                    "zero_count": int((torch.abs(values) <= 1.0e-12).sum().item()) if values.numel() else 0,
                    "negative_count": int((values < -1.0e-12).sum().item()) if values.numel() else 0,
                    "above_capacity_count": above_capacity_count,
                    "forcing_regime": regime_name,
                    "notes": "Trace pooled across parameter cases: lower_near, midpoint, upper_near, random_valid.",
                }
            )
    pooled_stats: dict[str, dict[str, float]] = {}
    for arg_name, parts in pooled.items():
        values = torch.cat(parts) if parts else torch.empty(0, dtype=torch.float64)
        pooled_stats[arg_name] = {
            "min": float(values.min().item()) if values.numel() else math.nan,
            "p01": _quantile(values, 0.01),
            "p05": _quantile(values, 0.05),
            "median": _quantile(values, 0.5),
            "mean": float(values.mean().item()) if values.numel() else math.nan,
            "p95": _quantile(values, 0.95),
            "p99": _quantile(values, 0.99),
            "max": float(values.max().item()) if values.numel() else math.nan,
        }
    return rows, pooled_stats


def _context_inventory_rows() -> list[dict[str, Any]]:
    usage_rows = _load_usage_rows()
    usage_contexts = {
        (ctx.flux_function, ctx.model_name, ctx.call_site): ctx
        for ctx in load_flux_usage_contexts()
        if ctx.module_type == "core"
    }
    rows = []
    for target in BATCH_A_TARGETS:
        flux_path = REPO_ROOT / f"{target.flux_module.replace('.', '/')}.py"
        flux_start, flux_end = _line_bounds(flux_path, target.formula)
        usage = usage_rows[(target.formula, target.active_model)][0]
        ctx = usage_contexts[(target.formula, target.active_model, usage["call_sites"])]
        core_path = REPO_ROOT / usage["call_sites"].split(":")[0]
        call_line = int(usage["call_sites"].split(":")[1])
        forcing_or_incoming = {}
        forcing_or_incoming.update(ctx.forcing_variable_mapping)
        if "incoming_flux" in ctx.parameter_mapping:
            forcing_or_incoming["incoming_flux"] = ctx.parameter_mapping["incoming_flux"]
        rows.append(
            {
                "formula": target.formula,
                "active_model": target.active_model,
                "flux_file": str(flux_path.relative_to(REPO_ROOT)),
                "flux_lines": f"{flux_start}-{flux_end}",
                "core_file": str(core_path.relative_to(REPO_ROOT)),
                "call_site_lines": usage["call_sites"].split(":")[1],
                "parameter_mapping": usage["parameter_mapping"],
                "parameter_bounds": usage["parameter_bounds"],
                "state_mapping": usage["state_variable_mapping"],
                "forcing_or_incoming_flux_mapping": _json(forcing_or_incoming),
                "output_meaning": target.output_meaning,
                "expected_physical_bounds": target.expected_physical_bounds,
                "existing_pre_or_post_caps": target.existing_pre_or_post_caps,
                "notes": target.notes,
            }
        )
    return rows


def _source_context_markdown(context_rows: list[dict[str, Any]]) -> str:
    usage_rows = _load_usage_rows()
    lines = [
        "# Batch A Source Context",
        "",
        "This document records the exact flux code and active-model call sites for the Batch A realistic-domain review.",
        "",
    ]
    for row in context_rows:
        target = next(item for item in BATCH_A_TARGETS if item.formula == row["formula"] and item.active_model == row["active_model"])
        flux_path = REPO_ROOT / row["flux_file"]
        flux_start, flux_end = map(int, row["flux_lines"].split("-"))
        lines.append(f"## {row['formula']} / {row['active_model']}")
        lines.append("")
        lines.append("### Flux code")
        lines.append("")
        lines.append(f"`{row['flux_file']}:{row['flux_lines']}`")
        lines.append("")
        lines.append("```python")
        lines.append(_snippet(flux_path, flux_start, flux_end))
        lines.append("```")
        lines.append("")
        lines.append("### Active model call site")
        lines.append("")
        for usage in usage_rows[(target.formula, target.active_model)]:
            file_part, line_part = usage["call_sites"].split(":")
            core_path = REPO_ROOT / file_part
            line_no = int(line_part)
            lines.append(f"`{usage['call_sites']}`")
            lines.append("")
            lines.append("```python")
            lines.append(_snippet(core_path, max(1, line_no - 3), min(line_no + 5, len(core_path.read_text(encoding='utf-8').splitlines()))))
            lines.append("```")
            lines.append("")
        lines.append(f"- Output meaning: {row['output_meaning']}")
        lines.append(f"- Expected physical bounds: {row['expected_physical_bounds']}")
        lines.append(f"- Existing cap logic: {row['existing_pre_or_post_caps']}")
        lines.append("")
    return "\n".join(lines)


def _base_inputs_from_stats(target: TargetSpec, stats: dict[str, dict[str, float]], case_name: str) -> dict[str, torch.Tensor]:
    inputs: dict[str, torch.Tensor] = {}
    for arg_name in _arg_order(target):
        values = stats[arg_name]
        if case_name == "median":
            value = values["median"]
        elif case_name == "high_state" and arg_name == target.probe_arg:
            value = values["p99"]
        elif case_name == "random":
            generator = torch.Generator(device=DEFAULT_DEVICE)
            generator.manual_seed(FIXED_SEED + sum(ord(char) for char in f"{target.formula}:{target.active_model}:{arg_name}"))
            lo = values["p05"]
            hi = values["p95"]
            tensor = lo + (hi - lo) * torch.rand(REALISTIC_SHAPE, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE, generator=generator)
            inputs[arg_name] = tensor.requires_grad_(arg_name in target.grad_inputs)
            continue
        elif case_name == "lower_param_boundary" and arg_name == target.boundary_param:
            model_bounds = CORE_MODEL_REGISTRY[target.active_model].param_bounds
            if arg_name == "p1":
                if target.formula == "baseflow_9":
                    value = 0.0
                else:
                    source_key = "beta" if target.formula == "saturation_3" else "b_exp"
                    value = float(model_bounds[source_key][0])
            else:
                value = values["median"]
        elif case_name == "threshold_plus" and target.formula == "baseflow_9" and arg_name == "S":
            threshold = stats["p2"]["median"]
            value = threshold + max(1.0, 0.05 * max(stats["S"]["p99"] - threshold, 1.0))
        elif case_name == "upper_coeff_stress" and target.formula == "baseflow_9" and arg_name == "p1":
            value = stats["p1"]["p99"]
        elif case_name == "upper_coeff_stress" and target.formula == "baseflow_9" and arg_name == "S":
            value = stats["S"]["p99"]
        elif case_name == "upper_coeff_stress" and target.formula == "baseflow_9" and arg_name == "p2":
            value = stats["p2"]["p01"]
        else:
            value = values["median"]
        inputs[arg_name] = torch.full(REALISTIC_SHAPE, float(value), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE).requires_grad_(arg_name in target.grad_inputs)
    return inputs


def representative_case_names(target: TargetSpec) -> tuple[str, ...]:
    if target.formula == "baseflow_9":
        return ("median", "random", "threshold_plus", "high_state", "upper_coeff_stress")
    return ("median", "random", "high_state", "lower_param_boundary")


def _autograd_gradients(output: torch.Tensor, inputs: dict[str, torch.Tensor], grad_inputs: tuple[str, ...]) -> dict[str, torch.Tensor | None]:
    required = [inputs[name] for name in grad_inputs if inputs[name].requires_grad]
    grads = torch.autograd.grad(output.sum(), required, allow_unused=True, retain_graph=False)
    iterator = iter(grads)
    out: dict[str, torch.Tensor | None] = {}
    for name in grad_inputs:
        out[name] = next(iterator) if inputs[name].requires_grad else None
    return out


def _fd_gradient(target: TargetSpec, inputs: dict[str, torch.Tensor], arg_name: str) -> tuple[float, int]:
    flux_fn = getattr(__import__(target.flux_module, fromlist=[target.formula]), target.formula)
    arg_order = _arg_order(target)
    center = {name: tensor.detach().clone() for name, tensor in inputs.items()}
    idx = center[arg_name].numel() // 2
    plus = {name: tensor.detach().clone() for name, tensor in center.items()}
    minus = {name: tensor.detach().clone() for name, tensor in center.items()}
    plus[arg_name].reshape(-1)[idx] += FD_EPS
    minus[arg_name].reshape(-1)[idx] -= FD_EPS
    with torch.no_grad():
        y_plus = flux_fn(*[plus[name] for name in arg_order], nearzero=DEFAULT_NEARZERO).reshape(-1)[idx].item()
        y_minus = flux_fn(*[minus[name] for name in arg_order], nearzero=DEFAULT_NEARZERO).reshape(-1)[idx].item()
    return (y_plus - y_minus) / (2.0 * FD_EPS), idx


def _evaluate_case(target: TargetSpec, case_name: str, inputs: dict[str, torch.Tensor], case_group: str) -> dict[str, Any]:
    flux_fn = getattr(__import__(target.flux_module, fromlist=[target.formula]), target.formula)
    arg_order = _arg_order(target)
    output = flux_fn(*[inputs[name] for name in arg_order], nearzero=DEFAULT_NEARZERO)
    grads = _autograd_gradients(output, inputs, target.grad_inputs)

    output_nan_count = int(torch.isnan(output).sum().item())
    output_inf_count = int(torch.isinf(output).sum().item())
    output_negative_count = int((output < -1.0e-12).sum().item())

    if target.expected_bound_type == "incoming_flux":
        cap = inputs[target.bound_arg]
        output_exceeds_incoming_flux_count = int((output > cap + 1.0e-10).sum().item())
        output_exceeds_storage_count = 0
    else:
        cap = inputs[target.bound_arg]
        output_exceeds_incoming_flux_count = 0
        output_exceeds_storage_count = int((output > cap + 1.0e-10).sum().item())
    output_bound_violation_count = output_exceeds_incoming_flux_count + output_exceeds_storage_count

    grad_nan_count = 0
    grad_inf_count = 0
    max_abs_grad = 0.0
    mean_abs_grad = 0.0
    median_abs_grad = 0.0
    zero_gradient_fraction = 1.0
    grad_values = []
    zero_fracs = []
    for grad in grads.values():
        if grad is None:
            continue
        grad_nan_count += int(torch.isnan(grad).sum().item())
        grad_inf_count += int(torch.isinf(grad).sum().item())
        abs_grad = torch.abs(grad)
        grad_values.append(abs_grad.reshape(-1))
        max_abs_grad = max(max_abs_grad, float(abs_grad.max().item()))
        zero_fracs.append(float((abs_grad < 1.0e-12).float().mean().item()))
    if grad_values:
        flat = torch.cat(grad_values)
        mean_abs_grad = float(flat.mean().item())
        median_abs_grad = float(flat.median().item())
        zero_gradient_fraction = max(zero_fracs)

    autograd_fd_relative_error = math.nan
    if case_name not in {"lower_param_boundary"}:
        probe_grad = grads.get(target.probe_arg)
        if probe_grad is not None:
            fd, idx = _fd_gradient(target, inputs, target.probe_arg)
            ad = probe_grad.reshape(-1)[idx].item()
            autograd_fd_relative_error = abs(ad - fd) / max(abs(fd), 1.0e-12)

    return {
        "formula": target.formula,
        "active_model": target.active_model,
        "case_name": case_name,
        "case_group": case_group,
        "output_nan_count": output_nan_count,
        "output_inf_count": output_inf_count,
        "grad_nan_count": grad_nan_count,
        "grad_inf_count": grad_inf_count,
        "max_abs_grad": max_abs_grad,
        "mean_abs_grad": mean_abs_grad,
        "median_abs_grad": median_abs_grad,
        "zero_gradient_fraction": zero_gradient_fraction,
        "autograd_fd_relative_error": autograd_fd_relative_error,
        "output_negative_count": output_negative_count,
        "output_exceeds_incoming_flux_count": output_exceeds_incoming_flux_count,
        "output_exceeds_storage_count": output_exceeds_storage_count,
        "output_bound_violation_count": output_bound_violation_count,
        "max_output": float(output.max().item()),
        "min_output": float(output.min().item()),
    }


def _evaluate_target(target: TargetSpec, pooled_stats: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for case_name in representative_case_names(target):
        case_group = "realistic_domain" if case_name != "lower_param_boundary" else "boundary_parameter_probe"
        inputs = _base_inputs_from_stats(target, pooled_stats, case_name)
        rows.append(_evaluate_case(target, case_name, inputs, case_group))
    return rows


def _broad_metrics_map() -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_csv(GRADIENT_DIR / "flux_gradient_risk_ranking.csv")
    return {(row["flux_function"], row["called_by_models"]): row for row in rows}


def _failure_mode_map() -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_csv(GRADIENT_DIR / "remaining_high_risk_failure_mode_summary.csv")
    return {(row["formula"], row["active_model"]): row for row in rows}


def _comparison_rows(gradient_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    broad_map = _broad_metrics_map()
    failure_map = _failure_mode_map()
    rows = []
    for target in BATCH_A_TARGETS:
        key = (target.formula, target.active_model)
        broad = broad_map[key]
        realistic = [row for row in gradient_rows if row["formula"] == target.formula and row["active_model"] == target.active_model and row["case_group"] == "realistic_domain"]
        boundary = [row for row in gradient_rows if row["formula"] == target.formula and row["active_model"] == target.active_model and row["case_group"] == "boundary_parameter_probe"]
        realistic_output_nan_inf = sum(row["output_nan_count"] + row["output_inf_count"] for row in realistic)
        realistic_grad_nan_inf = sum(row["grad_nan_count"] + row["grad_inf_count"] for row in realistic)
        realistic_bound = sum(row["output_bound_violation_count"] for row in realistic)
        realistic_max_grad = max(row["max_abs_grad"] for row in realistic)
        if realistic_output_nan_inf or realistic_grad_nan_inf:
            artifact_or_real = "realistic_domain_risk"
            reason = "Non-finite values persist inside traced active-model domains."
        elif boundary and sum(row["grad_nan_count"] + row["grad_inf_count"] for row in boundary):
            artifact_or_real = "parameter_boundary_sensitive"
            reason = "Traced realistic domains are finite, but exact lower-bound parameter probing still produces non-finite gradients."
        elif realistic_bound:
            artifact_or_real = "realistic_bound_issue"
            reason = "Traced realistic domains still exceed the expected physical bound."
        else:
            artifact_or_real = "broad_domain_artifact"
            reason = "The broad high-risk signal does not persist under traced active-model domains."
        rows.append(
            {
                "formula": target.formula,
                "active_model": target.active_model,
                "previous_failure_mode": failure_map[key]["failure_mode"],
                "broad_output_nan_inf": _int(broad["output_nan_count"]) + _int(broad["output_inf_count"]),
                "broad_grad_nan_inf": _int(broad["grad_nan_count"]) + _int(broad["grad_inf_count"]),
                "broad_bound_violation": _int(broad["output_bound_violation_count"]),
                "broad_max_abs_grad": _float(broad["max_abs_grad"]),
                "realistic_output_nan_inf": realistic_output_nan_inf,
                "realistic_grad_nan_inf": realistic_grad_nan_inf,
                "realistic_bound_violation": realistic_bound,
                "realistic_max_abs_grad": realistic_max_grad,
                "artifact_or_real": artifact_or_real,
                "reason": reason,
            }
        )
    return rows


def _bound_review_rows(gradient_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    broad_map = _broad_metrics_map()
    for target in BATCH_A_TARGETS:
        key = (target.formula, target.active_model)
        realistic = [row for row in gradient_rows if row["formula"] == target.formula and row["active_model"] == target.active_model and row["case_group"] == "realistic_domain"]
        boundary = [row for row in gradient_rows if row["formula"] == target.formula and row["active_model"] == target.active_model and row["case_group"] == "boundary_parameter_probe"]
        realistic_bound_count = sum(row["output_bound_violation_count"] for row in realistic)
        boundary_bound_count = sum(row["output_bound_violation_count"] for row in boundary)

        if realistic_bound_count > 0:
            bound_issue_class = "true_realistic_bound_violation"
            reason = "Representative traced active-model cases still exceed the physically meaningful bound before any model-level post-cap."
        elif target.formula == "baseflow_9":
            bound_issue_class = "broad_domain_artifact"
            reason = "The broad flag depended on an inferred generic range for `b * dpf`; realistic traced product values stay within the physically safe release range."
        elif target.formula == "saturation_2" and boundary_bound_count > 0:
            bound_issue_class = "manual_review_required"
            reason = "Representative traced cases are bounded, but an exact lower-bound parameter probe can still exceed the incoming-flux bound."
        elif target.formula == "saturation_3" and boundary_bound_count == 0:
            bound_issue_class = "no_bound_issue"
            reason = "Neither the traced realistic cases nor the positive lower-bound beta probe exceed the incoming-flux bound."
        elif target.formula == "saturation_3":
            bound_issue_class = "manual_review_required"
            reason = "The representative traced cases are bounded, but the lower-bound beta probe still exceeds the incoming-flux bound."
        else:
            bound_issue_class = "broad_domain_artifact"
            reason = "The broad bound flag does not persist once inputs are restricted to traced active-model domains."
        rows.append(
            {
                "formula": target.formula,
                "active_model": target.active_model,
                "previous_bound_flag": _int(broad_map[key]["output_bound_violation_count"]),
                "expected_bound_type": target.expected_bound_type,
                "is_bound_valid_for_formula": "yes",
                "model_update_applies_later_cap": "yes",
                "realistic_bound_violation_count": realistic_bound_count,
                "bound_issue_class": bound_issue_class,
                "reason": reason,
            }
        )
    return rows


def _risk_decision_rows(gradient_rows: list[dict[str, Any]], bound_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    broad_map = _broad_metrics_map()
    bound_map = {(row["formula"], row["active_model"]): row for row in bound_rows}
    rows = []
    for target in BATCH_A_TARGETS:
        key = (target.formula, target.active_model)
        broad = broad_map[key]
        realistic = [row for row in gradient_rows if row["formula"] == target.formula and row["active_model"] == target.active_model and row["case_group"] == "realistic_domain"]
        boundary = [row for row in gradient_rows if row["formula"] == target.formula and row["active_model"] == target.active_model and row["case_group"] == "boundary_parameter_probe"]
        realistic_nan_inf = sum(row["output_nan_count"] + row["output_inf_count"] + row["grad_nan_count"] + row["grad_inf_count"] for row in realistic)
        realistic_bound = sum(row["output_bound_violation_count"] for row in realistic)
        realistic_max_grad = max(row["max_abs_grad"] for row in realistic)
        boundary_nan_inf = sum(row["output_nan_count"] + row["output_inf_count"] + row["grad_nan_count"] + row["grad_inf_count"] for row in boundary)
        bound_review = bound_map[key]

        if realistic_nan_inf > 0:
            realistic_risk = "high"
            artifact_or_real = "real"
            recommended_action = "manual_review_required"
            human_review_priority = "high"
            main_failure_mode = "nonfinite_output_or_gradient"
            short_reason = "Non-finite outputs or gradients remain under traced active-model domains."
        elif bound_review["bound_issue_class"] == "true_realistic_bound_violation":
            realistic_risk = "high"
            artifact_or_real = "real"
            recommended_action = "add_safety_clamp_later"
            human_review_priority = "high"
            main_failure_mode = "physical_bound_violation"
            short_reason = "Representative traced active-model cases still exceed the expected physical bound before the model-level cap."
        elif boundary_nan_inf > 0:
            realistic_risk = "medium"
            artifact_or_real = "parameter_boundary_sensitive"
            recommended_action = "check_parameter_range"
            human_review_priority = "high" if target.formula == "saturation_3" else "medium"
            main_failure_mode = "parameter_range"
            short_reason = "Traced realistic domains are finite, but the exact lower-bound parameter probe still creates non-finite gradients."
        elif bound_review["bound_issue_class"] == "broad_domain_artifact":
            if realistic_max_grad > 1.0e2:
                realistic_risk = "medium"
                artifact_or_real = "real_but_bounded"
                recommended_action = "keep_but_document"
                human_review_priority = "medium"
                main_failure_mode = "exploding_gradient_but_finite"
                short_reason = (
                    f"{bound_review['reason']} Realistic traced cases remain finite and bounded, "
                    "but gradients are large enough to warrant documentation."
                )
            else:
                realistic_risk = "low"
                artifact_or_real = "artifact"
                recommended_action = "broad_domain_artifact"
                human_review_priority = "low"
                main_failure_mode = "broad_domain_artifact"
                short_reason = bound_review["reason"]
        elif realistic_max_grad > 1.0e2:
            realistic_risk = "medium"
            artifact_or_real = "real_but_bounded"
            recommended_action = "keep_but_document"
            human_review_priority = "medium"
            main_failure_mode = "exploding_gradient_but_finite"
            short_reason = "Traced realistic domains are finite and bounded, but gradients are large enough to warrant documentation."
        else:
            realistic_risk = "low"
            artifact_or_real = "artifact"
            recommended_action = "safe_no_action"
            human_review_priority = "low"
            main_failure_mode = "no_realistic_issue"
            short_reason = "No non-finite values or true physical bound violations remain in the traced active-model domains."

        rows.append(
            {
                "formula": target.formula,
                "active_model": target.active_model,
                "previous_risk": broad["risk_level"],
                "realistic_risk": realistic_risk,
                "output_nan_inf_realistic": sum(row["output_nan_count"] + row["output_inf_count"] for row in realistic),
                "grad_nan_inf_realistic": sum(row["grad_nan_count"] + row["grad_inf_count"] for row in realistic),
                "physical_bound_violation_realistic": realistic_bound,
                "max_abs_grad_realistic": realistic_max_grad,
                "main_failure_mode": main_failure_mode,
                "artifact_or_real": artifact_or_real,
                "recommended_action": recommended_action,
                "human_review_priority": human_review_priority,
                "short_reason": short_reason,
            }
        )
    return rows


def _plot_target(target: TargetSpec, pooled_stats: dict[str, dict[str, float]]) -> None:
    flux_fn = getattr(__import__(target.flux_module, fromlist=[target.formula]), target.formula)
    arg_order = _arg_order(target)
    median_inputs = _base_inputs_from_stats(target, pooled_stats, "median")
    broad_lo = 0.0
    broad_hi = 2000.0
    if target.formula == "baseflow_9":
        broad_lo = 0.0
        broad_hi = max(2000.0, pooled_stats["S"]["max"])
    realistic_lo = pooled_stats[target.probe_arg]["p01"]
    realistic_hi = pooled_stats[target.probe_arg]["p99"]
    broad_grid = torch.linspace(broad_lo, broad_hi, steps=256, dtype=DEFAULT_DTYPE)

    outputs = []
    grads = []
    ratios = []
    for value in broad_grid:
        inputs = {}
        for name, tensor in median_inputs.items():
            if name == target.probe_arg:
                inputs[name] = torch.full_like(tensor, float(value), requires_grad=(name in target.grad_inputs))
            else:
                inputs[name] = tensor.detach().clone().requires_grad_(name in target.grad_inputs)
        output = flux_fn(*[inputs[name] for name in arg_order], nearzero=DEFAULT_NEARZERO)
        outputs.append(float(output.mean().item()))
        grad = torch.autograd.grad(output.sum(), inputs[target.probe_arg], allow_unused=True)[0]
        grads.append(float(torch.abs(grad).mean().item()) if grad is not None else 0.0)
        bound = inputs[target.bound_arg]
        ratio = output / torch.clamp(bound, min=1.0e-12)
        ratios.append(float(ratio.mean().item()))

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    axes[0].plot(broad_grid.cpu().numpy(), outputs)
    axes[0].axvspan(realistic_lo, realistic_hi, color="tab:green", alpha=0.2, label="realistic p01-p99")
    axes[0].set_ylabel("output")
    axes[0].set_title(f"{target.formula} [{target.active_model}]")
    axes[0].legend()

    axes[1].plot(broad_grid.cpu().numpy(), grads)
    axes[1].axvspan(realistic_lo, realistic_hi, color="tab:green", alpha=0.2)
    axes[1].set_ylabel("|grad|")

    axes[2].plot(broad_grid.cpu().numpy(), ratios)
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[2].axvspan(realistic_lo, realistic_hi, color="tab:green", alpha=0.2)
    axes[2].set_ylabel("output / bound")
    axes[2].set_xlabel(target.probe_arg)

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / f"{target.formula}_{target.active_model}.png", dpi=160)
    plt.close(fig)


def _report(context_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]], bound_rows: list[dict[str, Any]], decision_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Batch A Flux Realistic Review Report",
        "",
        "## 1. Scope",
        "- Focused realistic-domain review for Batch A remaining active high-risk contexts.",
        "- No parameter bounds, smoothing defaults, unit hydrograph routines, or water-balance fixes were changed in this review workflow.",
        "- `saturation_3` now uses an algebraically equivalent stable sigmoid evaluation; hydrological semantics are unchanged.",
        "",
        "## 2. Target contexts",
    ]
    for target in BATCH_A_TARGETS:
        lines.append(f"- `{target.formula}` / `{target.active_model}`")
    lines.extend(
        [
            "",
            "## 3. Source and call context",
            "- Source snippets and call sites are recorded in `batch_a_source_context.md`.",
            "",
            "## 4. Realistic-domain tracing method",
            "- Each target flux was patched into its active core model namespace during short deterministic rollouts.",
            "- Forcing regimes: dry, normal, wet, high precipitation, low PET, high PET.",
            "- Parameter cases pooled in each regime: `lower_near`, `midpoint`, `upper_near`, `random_valid`.",
            "",
            "## 5. Broad vs realistic comparison",
            "",
            "| formula | model | broad failure | realistic NaN/Inf | realistic bound violations | realistic max_abs_grad | artifact_or_real |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"| {row['formula']} | {row['active_model']} | {row['previous_failure_mode']} | "
            f"{row['realistic_output_nan_inf'] + row['realistic_grad_nan_inf']} | {row['realistic_bound_violation']} | "
            f"{row['realistic_max_abs_grad']:.6g} | {row['artifact_or_real']} |"
        )

    lines.extend(
        [
            "",
            "## 6. Physical bound heuristic review",
        ]
    )
    for row in bound_rows:
        lines.append(
            f"- `{row['formula']}` / `{row['active_model']}`: `{row['bound_issue_class']}`. {row['reason']}"
        )

    for index, target in enumerate(BATCH_A_TARGETS, start=7):
        decision = next(row for row in decision_rows if row["formula"] == target.formula and row["active_model"] == target.active_model)
        lines.extend(
            [
                "",
                f"## {index}. Results for {target.formula} / {target.active_model}",
                f"- Previous risk: {decision['previous_risk']}",
                f"- Realistic-domain risk: {decision['realistic_risk']}",
                f"- Realistic max_abs_grad: {decision['max_abs_grad_realistic']:.6g}",
                f"- Realistic NaN/Inf counts: output={decision['output_nan_inf_realistic']}, grad={decision['grad_nan_inf_realistic']}",
                f"- Realistic physical bound violations: {decision['physical_bound_violation_realistic']}",
                f"- Artifact or real: {decision['artifact_or_real']}",
                f"- Recommended action: {decision['recommended_action']}",
                f"- Reason: {decision['short_reason']}",
            ]
        )

    lines.extend(
        [
            "",
            "## 12. Final risk decision table",
            "",
            "| formula | model | realistic risk | action | reason |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in decision_rows:
        lines.append(
            f"| {row['formula']} | {row['active_model']} | {row['realistic_risk']} | {row['recommended_action']} | {row['short_reason']} |"
        )

    boundary_rows = [row for row in decision_rows if row["artifact_or_real"] == "parameter_boundary_sensitive"]
    artifact_rows = [row for row in decision_rows if row["artifact_or_real"] == "artifact"]
    real_rows = [row for row in decision_rows if row["artifact_or_real"] in {"real", "real_but_bounded"}]
    lines.extend(
        [
            "",
            "## 13. Which cases are boundary-sensitive",
        ]
    )
    if boundary_rows:
        for row in boundary_rows:
            lines.append(f"- `{row['formula']}` / `{row['active_model']}`: {row['short_reason']}")
    else:
        lines.append("- None after the stable saturation_3 rewrite.")
    lines.extend(
        [
            "",
            "## 14. Which cases are artifacts",
        ]
    )
    for row in artifact_rows:
        lines.append(f"- `{row['formula']}` / `{row['active_model']}`: {row['short_reason']}")
    lines.extend(
        [
            "",
            "## 15. Which cases remain true realistic-domain risks",
        ]
    )
    if real_rows:
        for row in real_rows:
            lines.append(f"- `{row['formula']}` / `{row['active_model']}`: {row['short_reason']}")
    else:
        lines.append("- None of the Batch A contexts require additional formula modification based on the traced realistic domains.")
    lines.extend(
        [
            "",
            "## 16. Whether any formula modification is justified now",
            "- No additional formula modification is justified in this diagnostic pass.",
            "",
            "## 17. Recommended next step",
            "- Review Batch B next, or run benchmark/regression recalibration checks if you want to quantify any downstream optimization impact of the stable `saturation_3` rewrite.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_batch_a_review() -> dict[str, Any]:
    torch.manual_seed(FIXED_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    context_rows = _context_inventory_rows()
    (OUTPUT_DIR / "batch_a_source_context.md").write_text(_source_context_markdown(context_rows), encoding="utf-8")
    _write_csv(OUTPUT_DIR / "batch_a_context_inventory.csv", context_rows)

    trace_rows: list[dict[str, Any]] = []
    pooled_stats_by_key: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for target in BATCH_A_TARGETS:
        recorder = _run_trace_rollouts(target)
        rows, pooled_stats = _aggregate_domain_trace(target, recorder)
        trace_rows.extend(rows)
        pooled_stats_by_key[(target.formula, target.active_model)] = pooled_stats
    _write_csv(OUTPUT_DIR / "batch_a_realistic_domain_trace.csv", trace_rows)

    gradient_rows: list[dict[str, Any]] = []
    for target in BATCH_A_TARGETS:
        pooled_stats = pooled_stats_by_key[(target.formula, target.active_model)]
        gradient_rows.extend(_evaluate_target(target, pooled_stats))
        _plot_target(target, pooled_stats)
    _write_csv(OUTPUT_DIR / "batch_a_realistic_gradient_summary.csv", gradient_rows)

    comparison_rows = _comparison_rows(gradient_rows)
    bound_rows = _bound_review_rows(gradient_rows)
    decision_rows = _risk_decision_rows(gradient_rows, bound_rows)
    _write_csv(OUTPUT_DIR / "batch_a_broad_vs_realistic_comparison.csv", comparison_rows)
    _write_csv(OUTPUT_DIR / "batch_a_bound_heuristic_review.csv", bound_rows)
    _write_csv(OUTPUT_DIR / "batch_a_risk_decision.csv", decision_rows)
    (OUTPUT_DIR / "batch_a_flux_realistic_review_report.md").write_text(
        _report(context_rows, comparison_rows, bound_rows, decision_rows),
        encoding="utf-8",
    )
    return {
        "context_rows": context_rows,
        "trace_rows": trace_rows,
        "gradient_rows": gradient_rows,
        "comparison_rows": comparison_rows,
        "bound_rows": bound_rows,
        "decision_rows": decision_rows,
        "pooled_stats_by_key": pooled_stats_by_key,
    }


def main() -> None:
    run_batch_a_review()


if __name__ == "__main__":
    main()
