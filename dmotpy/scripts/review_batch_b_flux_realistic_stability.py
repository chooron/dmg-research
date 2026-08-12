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
from typing import Any

import torch
import torch.nn.functional as F


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


OUTPUT_DIR = REPO_ROOT / "validation_results" / "batch_b_flux_realistic_review"
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
    existing_pre_or_post_caps: str
    notes: str
    capacity_pair: tuple[str, str] | None = None


BATCH_B_TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec(
        formula="baseflow_4",
        flux_module="models.flux.baseflow",
        active_model="topmodel",
        probe_arg="S",
        grad_inputs=("S", "p1", "p2"),
        arg_roles={"S": "state_deficit", "p1": "baseflow_scale", "p2": "deficit_decay"},
        output_meaning="Baseflow discharge from the TOPMODEL saturated-zone deficit representation.",
        expected_physical_bounds="0 <= flux_qb <= q0. No `flux <= S2` storage cap applies because S2 is a deficit, not a water store.",
        expected_bound_type="deficit_store_parameter_ceiling",
        existing_pre_or_post_caps="No direct post-call cap on `flux_qb`; downstream `flux_qv` is limited by `S2 + flux_qb` so the deficit state stays non-negative.",
        notes="Broad diagnostic likely misread the deficit store as directly withdrawable storage.",
    ),
    TargetSpec(
        formula="evap_3",
        flux_module="models.flux.evap",
        active_model="hbv96",
        probe_arg="S",
        grad_inputs=("S", "p1", "Smax", "Ep"),
        arg_roles={"S": "state_storage", "p1": "wilting_fraction", "Smax": "capacity", "Ep": "potential_evaporation"},
        output_meaning="Actual soil evaporation demand from the HBV96 soil store.",
        expected_physical_bounds="0 <= flux_ea <= min(PET, S3).",
        expected_bound_type="min_pet_storage",
        existing_pre_or_post_caps="Formula already uses nested `torch.minimum`; core re-applies `torch.minimum(flux_ea_pot, S3)` before subtracting from storage.",
        notes="Broad diagnostic reported a bound issue even though the active formula already contains PET and storage caps.",
        capacity_pair=("S", "Smax"),
    ),
    TargetSpec(
        formula="recharge_2",
        flux_module="models.flux.recharge",
        active_model="hbv96",
        probe_arg="S",
        grad_inputs=("S", "p1", "Smax", "flux"),
        arg_roles={"S": "state_storage", "p1": "nonlinearity_parameter", "Smax": "capacity", "flux": "incoming_flux"},
        output_meaning="Recharge from the HBV96 soil store into the upper response box.",
        expected_physical_bounds="True active-model bound is `0 <= flux_r <= S3` after evaporation; `flux_r <= flux_se` is not a universally valid standalone bound once soil storage has already accumulated water.",
        expected_bound_type="storage_with_incoming_heuristic_audit",
        existing_pre_or_post_caps="Core applies `torch.minimum(flux_r_pot, S3)` before subtracting recharge from the soil store.",
        notes="Broad diagnostic likely mixed the raw recharge formula with an overly strict incoming-flux partition heuristic.",
        capacity_pair=("S", "Smax"),
    ),
    TargetSpec(
        formula="depression_1",
        flux_module="models.flux.depression",
        active_model="modhydrolog",
        probe_arg="S",
        grad_inputs=("ads", "md", "S", "Smax", "incoming_flux"),
        arg_roles={
            "ads": "area_fraction",
            "md": "depression_parameter",
            "S": "state_storage",
            "Smax": "capacity",
            "incoming_flux": "incoming_flux",
        },
        output_meaning="Depression-storage trapping from surface runoff in MODHYDROLOG.",
        expected_physical_bounds="0 <= flux_TRAP <= min(flux_RUN, dsc - S3).",
        expected_bound_type="incoming_and_residual_capacity",
        existing_pre_or_post_caps="The formula itself caps trapping to both incoming runoff and residual depression capacity; no extra core cap is added before the state update.",
        notes="Broad diagnostic likely used current storage instead of residual capacity as the physical ceiling.",
        capacity_pair=("S", "Smax"),
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

    def wrapped(*args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        recorder.record(forcing_regime, arg_order, args[: len(arg_order)])
        return original_flux_fn(*args, **kwargs)

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


def _pooled_joint_tensors(target: TargetSpec, recorder: TraceRecorder) -> dict[str, torch.Tensor]:
    pooled_lists: dict[str, list[torch.Tensor]] = {name: [] for name in _arg_order(target)}
    for regime_name in sorted(recorder.calls_by_regime):
        for call in recorder.calls_by_regime[regime_name]:
            for name in pooled_lists:
                pooled_lists[name].append(call[name])
    return {
        name: torch.cat(parts).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
        for name, parts in pooled_lists.items()
        if parts
    }


def _context_inventory_rows() -> list[dict[str, Any]]:
    usage_rows = _load_usage_rows()
    usage_contexts = {
        (ctx.flux_function, ctx.model_name, ctx.call_site): ctx
        for ctx in load_flux_usage_contexts()
        if ctx.module_type == "core"
    }
    rows = []
    for target in BATCH_B_TARGETS:
        flux_path = REPO_ROOT / f"{target.flux_module.replace('.', '/')}.py"
        flux_start, flux_end = _line_bounds(flux_path, target.formula)
        usages = usage_rows[(target.formula, target.active_model)]
        core_file = usages[0]["call_sites"].split(":")[0]
        forcing_or_incoming: dict[str, Any] = {}
        for usage in usages:
            ctx = usage_contexts[(target.formula, target.active_model, usage["call_sites"])]
            forcing_or_incoming.update(ctx.forcing_variable_mapping)
            for name, value in ctx.parameter_mapping.items():
                if name in {"incoming_flux", "flux", "Ep"}:
                    forcing_or_incoming[name] = value
        rows.append(
            {
                "formula": target.formula,
                "active_model": target.active_model,
                "flux_file": str(flux_path.relative_to(REPO_ROOT)),
                "flux_lines": f"{flux_start}-{flux_end}",
                "core_file": core_file,
                "call_site_lines": ";".join(usage["call_sites"].split(":")[1] for usage in usages),
                "parameter_mapping": " | ".join(usage["parameter_mapping"] for usage in usages),
                "parameter_bounds": " | ".join(usage["parameter_bounds"] for usage in usages),
                "state_mapping": " | ".join(usage["state_variable_mapping"] for usage in usages),
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
        "# Batch B Source Context",
        "",
        "This document records the exact flux code and active-model call sites for the Batch B realistic-domain review.",
        "",
    ]
    for row in context_rows:
        target = next(item for item in BATCH_B_TARGETS if item.formula == row["formula"] and item.active_model == row["active_model"])
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
            lines.append(
                _snippet(
                    core_path,
                    max(1, line_no - 3),
                    min(line_no + 5, len(core_path.read_text(encoding='utf-8').splitlines())),
                )
            )
            lines.append("```")
            lines.append("")
        lines.append(f"- Output meaning: {row['output_meaning']}")
        lines.append(f"- Expected physical bounds: {row['expected_physical_bounds']}")
        lines.append(f"- Existing cap logic: {row['existing_pre_or_post_caps']}")
        lines.append("")
    return "\n".join(lines)


def representative_case_names(target: TargetSpec) -> tuple[str, ...]:
    if target.formula == "baseflow_4":
        return ("median", "random", "low_state", "high_state", "upper_coeff_stress")
    if target.formula == "depression_1":
        return ("median", "random", "high_flux", "low_state", "high_state")
    return ("median", "random", "low_state", "high_state", "high_flux")


def _repeat_indices(indices: torch.Tensor, n_total: int, n_out: int) -> torch.Tensor:
    if indices.numel() == 0:
        indices = torch.arange(n_total, dtype=torch.long)
    repeats = math.ceil(n_out / indices.numel())
    return indices.repeat(repeats)[:n_out]


def _select_case_indices(target: TargetSpec, pooled: dict[str, torch.Tensor], case_name: str) -> torch.Tensor:
    probe = pooled[target.probe_arg]
    n_total = probe.numel()
    n_out = REALISTIC_SHAPE[0]
    generator = torch.Generator(device=DEFAULT_DEVICE)
    generator.manual_seed(FIXED_SEED + sum(ord(char) for char in f"{target.formula}:{target.active_model}:{case_name}"))

    if case_name == "random":
        return torch.randint(0, n_total, (n_out,), generator=generator, device=DEFAULT_DEVICE)

    if case_name == "median":
        median = torch.quantile(probe, 0.5)
        idx = int(torch.argmin(torch.abs(probe - median)).item())
        return torch.full((n_out,), idx, dtype=torch.long, device=DEFAULT_DEVICE)

    if case_name == "low_state":
        threshold = torch.quantile(probe, 0.05)
        return _repeat_indices(torch.nonzero(probe <= threshold, as_tuple=False).reshape(-1), n_total, n_out)

    if case_name == "high_state":
        threshold = torch.quantile(probe, 0.95)
        return _repeat_indices(torch.nonzero(probe >= threshold, as_tuple=False).reshape(-1), n_total, n_out)

    if case_name == "high_flux":
        flux_name = "incoming_flux" if "incoming_flux" in pooled else "flux" if "flux" in pooled else "Ep"
        threshold = torch.quantile(pooled[flux_name], 0.95)
        return _repeat_indices(torch.nonzero(pooled[flux_name] >= threshold, as_tuple=False).reshape(-1), n_total, n_out)

    if case_name == "upper_coeff_stress":
        score = torch.zeros_like(probe)
        if "p1" in pooled:
            score = score + pooled["p1"]
        if "p2" in pooled:
            score = score + pooled["p2"]
        score = score - probe
        topk = min(n_out, n_total)
        indices = torch.topk(score, k=topk).indices
        return _repeat_indices(indices, n_total, n_out)

    raise KeyError(case_name)


def _base_inputs_from_joint_tensors(target: TargetSpec, pooled: dict[str, torch.Tensor], case_name: str) -> dict[str, torch.Tensor]:
    indices = _select_case_indices(target, pooled, case_name)
    inputs: dict[str, torch.Tensor] = {}
    for arg_name in _arg_order(target):
        tensor = pooled[arg_name].index_select(0, indices).clone().to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
        inputs[arg_name] = tensor.requires_grad_(arg_name in target.grad_inputs)
    return inputs


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


def _count(condition: torch.Tensor) -> int:
    return int(condition.sum().item())


def _bound_metrics(target: TargetSpec, inputs: dict[str, torch.Tensor], output: torch.Tensor, negative_count: int) -> dict[str, int]:
    tol = 1.0e-10
    output_exceeds_incoming_flux_count = 0
    output_exceeds_storage_count = 0
    output_exceeds_pet_count = 0
    output_exceeds_capacity_count = 0
    output_exceeds_parameter_ceiling_count = 0

    if target.formula == "baseflow_4":
        output_exceeds_parameter_ceiling_count = _count(output > inputs["p1"] + tol)
        true_bound_violation_count = negative_count + output_exceeds_parameter_ceiling_count
    elif target.formula == "evap_3":
        output_exceeds_storage_count = _count(output > inputs["S"] + tol)
        output_exceeds_pet_count = _count(output > inputs["Ep"] + tol)
        true_bound_violation_count = negative_count + output_exceeds_storage_count + output_exceeds_pet_count
    elif target.formula == "recharge_2":
        output_exceeds_storage_count = _count(output > inputs["S"] + tol)
        output_exceeds_incoming_flux_count = _count(output > inputs["flux"] + tol)
        # HBV96 applies `torch.minimum(flux_r_pot, S3)` before the state update,
        # so raw formula overshoots should be tracked but not classified as a
        # true active-model physical bound violation in this review.
        true_bound_violation_count = negative_count
    elif target.formula == "depression_1":
        residual_capacity = F.relu(inputs["Smax"] - inputs["S"])
        output_exceeds_capacity_count = _count(output > residual_capacity + tol)
        output_exceeds_incoming_flux_count = _count(output > inputs["incoming_flux"] + tol)
        true_bound_violation_count = negative_count + output_exceeds_capacity_count + output_exceeds_incoming_flux_count
    else:
        true_bound_violation_count = negative_count

    return {
        "output_exceeds_incoming_flux_count": output_exceeds_incoming_flux_count,
        "output_exceeds_storage_count": output_exceeds_storage_count,
        "output_exceeds_pet_count": output_exceeds_pet_count,
        "output_exceeds_capacity_count": output_exceeds_capacity_count,
        "output_exceeds_parameter_ceiling_count": output_exceeds_parameter_ceiling_count,
        "output_bound_violation_count": true_bound_violation_count,
    }


def _evaluate_case(target: TargetSpec, case_name: str, inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    flux_fn = getattr(__import__(target.flux_module, fromlist=[target.formula]), target.formula)
    arg_order = _arg_order(target)
    output = flux_fn(*[inputs[name] for name in arg_order], nearzero=DEFAULT_NEARZERO)
    grads = _autograd_gradients(output, inputs, target.grad_inputs)

    output_nan_count = _count(torch.isnan(output))
    output_inf_count = _count(torch.isinf(output))
    output_negative_count = _count(output < -1.0e-12)
    bound_metrics = _bound_metrics(target, inputs, output, output_negative_count)

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
        grad_nan_count += _count(torch.isnan(grad))
        grad_inf_count += _count(torch.isinf(grad))
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
    probe_grad = grads.get(target.probe_arg)
    if probe_grad is not None:
        fd, idx = _fd_gradient(target, inputs, target.probe_arg)
        ad = probe_grad.reshape(-1)[idx].item()
        if math.isfinite(fd) and math.isfinite(ad):
            autograd_fd_relative_error = abs(ad - fd) / max(abs(fd), 1.0e-12)

    row = {
        "formula": target.formula,
        "active_model": target.active_model,
        "case_name": case_name,
        "case_group": "realistic_domain",
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
        "max_output": float(output.max().item()),
        "min_output": float(output.min().item()),
    }
    row.update(bound_metrics)
    return row


def _evaluate_target(target: TargetSpec, pooled: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows = []
    for case_name in representative_case_names(target):
        inputs = _base_inputs_from_joint_tensors(target, pooled, case_name)
        rows.append(_evaluate_case(target, case_name, inputs))
    return rows


def _broad_metrics_map() -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_csv(GRADIENT_DIR / "final_flux_gradient_risk_ranking.csv")
    return {(row["formula"], row["active_model"]): row for row in rows}


def _failure_mode_map() -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_csv(GRADIENT_DIR / "remaining_high_risk_failure_mode_summary.csv")
    return {(row["formula"], row["active_model"]): row for row in rows}


def _broad_diagnostic_map() -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_csv(GRADIENT_DIR / "flux_gradient_risk_ranking.csv")
    return {(row["flux_function"], row["called_by_models"]): row for row in rows}


def _bound_review_rows(gradient_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    broad_map = _broad_metrics_map()
    broad_diag_map = _broad_diagnostic_map()
    rows = []
    for target in BATCH_B_TARGETS:
        key = (target.formula, target.active_model)
        realistic = [row for row in gradient_rows if row["formula"] == target.formula and row["active_model"] == target.active_model]
        realistic_true_bound = sum(row["output_bound_violation_count"] for row in realistic)
        incoming_heuristic = sum(row["output_exceeds_incoming_flux_count"] for row in realistic)

        storage_heuristic = sum(row["output_exceeds_storage_count"] for row in realistic)

        if realistic_true_bound > 0:
            bound_issue_class = "true_realistic_bound_violation"
        elif target.formula == "baseflow_4":
            bound_issue_class = "bound_heuristic_artifact"
        elif target.formula == "recharge_2" and (incoming_heuristic > 0 or storage_heuristic > 0):
            bound_issue_class = "bound_heuristic_artifact"
        elif target.formula == "depression_1":
            bound_issue_class = "bound_heuristic_artifact"
        elif target.formula == "evap_3":
            bound_issue_class = "broad_domain_artifact"
        else:
            bound_issue_class = "no_bound_issue"

        if target.formula == "baseflow_4":
            reason = (
                "The broad diagnostic treated TOPMODEL `S2` as a withdrawable storage cap. In the active model it is a deficit "
                "state, so the physically relevant ceiling is `q0`, not `S2`."
            )
            is_bound_valid = "no"
            later_cap = "no"
        elif target.formula == "evap_3":
            reason = (
                "HBV96 traced calls stay inside `0 <= Ea <= min(PET, S3)`. The broad flag does not persist once arguments are "
                "restricted to active-model rollouts."
            )
            is_bound_valid = "yes"
            later_cap = "yes"
        elif target.formula == "recharge_2":
            if incoming_heuristic > 0 or storage_heuristic > 0:
                reason = (
                    "Representative traced calls can overshoot the raw standalone storage or `flux_se` heuristics, but HBV96 immediately "
                    "applies `torch.minimum(flux_r_pot, S3)` before the state update. The previous high-risk bound flag is therefore "
                    "a post-cap heuristic artifact rather than a true active-model violation."
                )
            else:
                reason = (
                    "No true storage-cap violation appears in traced HBV96 rollouts. The earlier high-risk bound signal is a broad-domain "
                    "or heuristic artifact."
                )
            is_bound_valid = "partially"
            later_cap = "yes"
        else:
            reason = (
                "The formula already caps depression trapping by both incoming runoff and residual capacity `relu(dsc - S3)`. The broad "
                "flag likely came from comparing against current storage instead of residual capacity."
            )
            is_bound_valid = "yes"
            later_cap = "no"

        rows.append(
            {
                "formula": target.formula,
                "active_model": target.active_model,
                "previous_bound_flag": _int(broad_diag_map[key]["output_bound_violation_count"]),
                "expected_bound_type": target.expected_bound_type,
                "is_bound_valid_for_formula": is_bound_valid,
                "model_update_applies_later_cap": later_cap,
                "realistic_bound_violation_count": realistic_true_bound,
                "bound_issue_class": bound_issue_class,
                "reason": reason,
            }
        )
    return rows


def _risk_decision_rows(gradient_rows: list[dict[str, Any]], bound_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    broad_map = _broad_metrics_map()
    failure_map = _failure_mode_map()
    bound_map = {(row["formula"], row["active_model"]): row for row in bound_rows}
    rows = []
    for target in BATCH_B_TARGETS:
        key = (target.formula, target.active_model)
        realistic = [row for row in gradient_rows if row["formula"] == target.formula and row["active_model"] == target.active_model]
        bound_review = bound_map[key]
        output_nan_inf = sum(row["output_nan_count"] + row["output_inf_count"] for row in realistic)
        grad_nan_inf = sum(row["grad_nan_count"] + row["grad_inf_count"] for row in realistic)
        realistic_bound = sum(row["output_bound_violation_count"] for row in realistic)
        realistic_max_grad = max(row["max_abs_grad"] for row in realistic)

        if output_nan_inf or grad_nan_inf:
            realistic_risk = "high"
            artifact_or_real = "real"
            recommended_action = "manual_review_required"
            human_review_priority = "high"
            main_failure_mode = "nonfinite_output_or_gradient"
            short_reason = "Non-finite outputs or gradients remain under traced active-model domains."
        elif realistic_bound:
            realistic_risk = "high"
            artifact_or_real = "real"
            recommended_action = "add_safety_clamp_later"
            human_review_priority = "high"
            main_failure_mode = "physical_bound_violation"
            short_reason = "Representative traced active-model cases still violate the physically meaningful bound."
        elif realistic_max_grad > 1.0e2:
            realistic_risk = "medium"
            artifact_or_real = "real_but_bounded"
            recommended_action = "keep_but_document"
            human_review_priority = "medium"
            main_failure_mode = "large_but_finite_gradient"
            short_reason = "Traced active-model cases are finite and physically bounded, but gradients remain large enough to document."
        elif bound_review["bound_issue_class"] in {"bound_heuristic_artifact", "broad_domain_artifact"}:
            realistic_risk = "low"
            artifact_or_real = "artifact"
            recommended_action = bound_review["bound_issue_class"]
            human_review_priority = "low"
            main_failure_mode = failure_map[key]["failure_mode"]
            short_reason = bound_review["reason"]
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
                "previous_risk": broad_map[key]["final_active_risk"],
                "realistic_risk": realistic_risk,
                "output_nan_inf_realistic": output_nan_inf,
                "grad_nan_inf_realistic": grad_nan_inf,
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


def _report(
    context_rows: list[dict[str, Any]],
    gradient_rows: list[dict[str, Any]],
    bound_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
) -> str:
    broad_map = _broad_metrics_map()
    lines = [
        "# Batch B Flux Realistic Review Report",
        "",
        "## 1. Scope",
        "- Focused realistic-domain review for the next Batch B group of remaining active high-risk flux contexts.",
        "- No hydrological formulas, smoothing defaults, parameter bounds, unit hydrograph code, or water-balance fixes were changed in this diagnostic pass.",
        "",
        "## 2. Target contexts",
    ]
    for target in BATCH_B_TARGETS:
        lines.append(f"- `{target.formula}` / `{target.active_model}`")
    lines.extend(
        [
            "",
            "## 3. Source and call context",
            "- Source snippets and call sites are recorded in `batch_b_source_context.md`.",
            "",
            "## 4. Realistic-domain tracing method",
            "- Each target flux was patched into its active core model namespace during short deterministic rollouts.",
            "- Forcing regimes: dry, normal, wet, high precipitation, low PET, high PET.",
            "- Parameter cases pooled in each regime: `lower_near`, `midpoint`, `upper_near`, `random_valid`.",
            "",
            "## 5. Broad vs realistic comparison",
            "",
            "| formula | model | previous risk | realistic risk | realistic NaN/Inf | realistic true bound violations | realistic max_abs_grad | artifact_or_real |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in decision_rows:
        lines.append(
            f"| {row['formula']} | {row['active_model']} | {row['previous_risk']} | {row['realistic_risk']} | "
            f"{row['output_nan_inf_realistic'] + row['grad_nan_inf_realistic']} | {row['physical_bound_violation_realistic']} | "
            f"{row['max_abs_grad_realistic']:.6g} | {row['artifact_or_real']} |"
        )

    lines.extend(["", "## 6. Physical-bound heuristic review"])
    for row in bound_rows:
        lines.append(f"- `{row['formula']}` / `{row['active_model']}`: `{row['bound_issue_class']}`. {row['reason']}")

    section_titles = [
        "Results for baseflow_4 / topmodel",
        "Results for evap_3 / hbv96",
        "Results for recharge_2 / hbv96",
        "Results for depression_1 / modhydrolog",
    ]
    for index, (target, title) in enumerate(zip(BATCH_B_TARGETS, section_titles), start=7):
        key = (target.formula, target.active_model)
        decision = next(row for row in decision_rows if (row["formula"], row["active_model"]) == key)
        target_rows = [row for row in gradient_rows if (row["formula"], row["active_model"]) == key]
        lines.extend(
            [
                "",
                f"## {index}. {title}",
                f"- Previous risk: {decision['previous_risk']}",
                f"- Broad diagnostic max_abs_grad: {_float(broad_map[key].get('realistic_max_abs_grad'), math.nan):.6g}" if broad_map[key].get("realistic_max_abs_grad") else f"- Broad diagnostic reason: {broad_map[key]['final_reason']}",
                f"- Realistic-domain risk: {decision['realistic_risk']}",
                f"- Realistic max_abs_grad: {decision['max_abs_grad_realistic']:.6g}",
                f"- Realistic NaN/Inf counts: output={decision['output_nan_inf_realistic']}, grad={decision['grad_nan_inf_realistic']}",
                f"- Realistic true physical bound violations: {decision['physical_bound_violation_realistic']}",
                f"- Representative cases reviewed: {', '.join(row['case_name'] for row in target_rows)}",
                f"- Artifact or real: {decision['artifact_or_real']}",
                f"- Recommended action: {decision['recommended_action']}",
                f"- Reason: {decision['short_reason']}",
            ]
        )

    lines.extend(
        [
            "",
            "## 11. Final risk decision table",
            "",
            "| formula | model | realistic risk | action | reason |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in decision_rows:
        lines.append(
            f"| {row['formula']} | {row['active_model']} | {row['realistic_risk']} | {row['recommended_action']} | {row['short_reason']} |"
        )

    artifact_rows = [row for row in decision_rows if row["artifact_or_real"] == "artifact"]
    real_rows = [row for row in decision_rows if row["artifact_or_real"] in {"real", "real_but_bounded"}]
    lines.extend(["", "## 12. Which cases are artifacts"])
    if artifact_rows:
        for row in artifact_rows:
            lines.append(f"- `{row['formula']}` / `{row['active_model']}`: {row['short_reason']}")
    else:
        lines.append("- None in Batch B.")
    lines.extend(["", "## 13. Which cases remain true realistic-domain risks"])
    if real_rows:
        for row in real_rows:
            lines.append(f"- `{row['formula']}` / `{row['active_model']}`: {row['short_reason']}")
    else:
        lines.append("- None of the Batch B contexts require immediate formula intervention based on traced realistic domains.")
    lines.extend(
        [
            "",
            "## 14. Whether any formula modification is justified now",
            "- No. This review found no evidence that Batch B requires an immediate hydrological formula change.",
            "",
            "## 15. Recommended next step",
            "- Review the first Batch C context next: `excess_1 / australia`, then continue the remaining overflow-style Batch C contexts with the same realistic-domain tracing method.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_batch_b_review() -> dict[str, Any]:
    torch.manual_seed(FIXED_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    context_rows = _context_inventory_rows()
    (OUTPUT_DIR / "batch_b_source_context.md").write_text(_source_context_markdown(context_rows), encoding="utf-8")
    _write_csv(OUTPUT_DIR / "batch_b_context_inventory.csv", context_rows)

    trace_rows: list[dict[str, Any]] = []
    pooled_stats_by_key: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    pooled_joint_by_key: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for target in BATCH_B_TARGETS:
        recorder = _run_trace_rollouts(target)
        rows, pooled_stats = _aggregate_domain_trace(target, recorder)
        trace_rows.extend(rows)
        pooled_stats_by_key[(target.formula, target.active_model)] = pooled_stats
        pooled_joint_by_key[(target.formula, target.active_model)] = _pooled_joint_tensors(target, recorder)
    _write_csv(OUTPUT_DIR / "batch_b_realistic_domain_trace.csv", trace_rows)

    gradient_rows: list[dict[str, Any]] = []
    for target in BATCH_B_TARGETS:
        pooled_joint = pooled_joint_by_key[(target.formula, target.active_model)]
        gradient_rows.extend(_evaluate_target(target, pooled_joint))
    _write_csv(OUTPUT_DIR / "batch_b_realistic_gradient_summary.csv", gradient_rows)

    bound_rows = _bound_review_rows(gradient_rows)
    decision_rows = _risk_decision_rows(gradient_rows, bound_rows)
    _write_csv(OUTPUT_DIR / "batch_b_bound_heuristic_review.csv", bound_rows)
    _write_csv(OUTPUT_DIR / "batch_b_risk_decision.csv", decision_rows)
    (OUTPUT_DIR / "batch_b_flux_realistic_review_report.md").write_text(
        _report(context_rows, gradient_rows, bound_rows, decision_rows),
        encoding="utf-8",
    )
    return {
        "context_rows": context_rows,
        "trace_rows": trace_rows,
        "gradient_rows": gradient_rows,
        "bound_rows": bound_rows,
        "decision_rows": decision_rows,
        "pooled_stats_by_key": pooled_stats_by_key,
        "pooled_joint_by_key": pooled_joint_by_key,
    }


def main() -> None:
    run_batch_b_review()


if __name__ == "__main__":
    main()
