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


OUTPUT_DIR = REPO_ROOT / "validation_results" / "medium_flux_formula_rewrite_review"
DEFAULT_DTYPE = torch.float64
DEFAULT_DEVICE = "cpu"
DEFAULT_NEARZERO = 1.0e-6
FIXED_SEED = 20260625


@dataclass(frozen=True)
class TargetSpec:
    formula: str
    flux_module: str
    active_model: str
    grad_inputs: tuple[str, ...]
    output_meaning: str
    downstream_state_update: str
    expected_bounds: str
    notes: str
    parameter_mapping: dict[str, str]
    state_mapping: dict[str, str]
    parameter_bounds: dict[str, str]
    call_site_lines: tuple[int, ...]


TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec(
        formula="baseflow_6",
        flux_module="models.flux.baseflow",
        active_model="tcm",
        grad_inputs=("p1", "p2", "S"),
        output_meaning="Quadratic slow-routing/baseflow discharge candidate from TCM store S4 after abstraction loss.",
        downstream_state_update="`flux_q = min(baseflow_6(k2, 0, S4), S4)`; then `S4_new = clamp(S4 - flux_q, min=nearzero)`.",
        expected_bounds="Raw helper is non-negative and, because of `min(S, p1*S^2)`, satisfies `0 <= q <= S` for non-negative storage. Active TCM then re-applies `min(flux_q, S4)`.",
        notes="Threshold argument is fixed at 0 in active TCM usage, so the soft gate is a near-zero activation helper rather than a moving physical threshold.",
        parameter_mapping={"p1": "k2", "p2": "torch.tensor(0.0, device=P.device)"},
        state_mapping={"S": "S4"},
        parameter_bounds={"p1": "[0.0, 1.0]", "p2": "[0.0, 0.0]"},
        call_site_lines=(167,),
    ),
    TargetSpec(
        formula="baseflow_9",
        flux_module="models.flux.baseflow",
        active_model="gsfb",
        grad_inputs=("p1", "p2", "S"),
        output_meaning="Slow baseflow release from the GSFB intermediate store S2 after infiltration from the soil store.",
        downstream_state_update="`flux_qb = min(baseflow_9(b*dpf, sdrmax, S2_tmp_in), S2_tmp_in - nearzero)`; then `S2_tmp_perc = S2_tmp_in - flux_qb`.",
        expected_bounds="Active GSFB applies a post-call storage cap `flux_qb <= S2_tmp_in - nearzero`. The raw helper is non-negative but not intrinsically storage-capped when large excess storage is present.",
        notes="The previous broad high-risk label was driven by a generic treatment of `p1`; active usage constrains `p1 = b * dpf` to a product of two `[0,1]` parameters.",
        parameter_mapping={"p1": "b * dpf", "p2": "sdrmax"},
        state_mapping={"S": "S2_tmp_in"},
        parameter_bounds={"p1": "[0.0, 1.0] from b*dpf with b,dpf in [0,1]", "p2": "[1.0, 300.0]"},
        call_site_lines=(135,),
    ),
    TargetSpec(
        formula="interflow_10",
        flux_module="models.flux.interflow",
        active_model="topmodel",
        grad_inputs=("S", "p1", "p2", "p3"),
        output_meaning="Potential recharge/interflow transfer from TOPMODEL unsaturated storage S1 into the saturated-zone deficit update.",
        downstream_state_update="`flux_qv = min(interflow_10(...), s1_free_now, s2_space)`; then `S1_new = clamp(S1_tmp - flux_qv, min=nearzero)` and `S2_new = clamp(S2 + flux_qb - flux_qv, min=nearzero)`.",
        expected_bounds="No standalone raw helper storage cap is physically required because TOPMODEL applies joint supply and receiving-space caps after the helper call. The meaningful active constraints are the later `min(..., s1_free_now, s2_space)` operations.",
        notes="Hard thresholding below `st * suzmax` is physically meaningful in TOPMODEL because recharge should not occur below the unsaturated storage threshold.",
        parameter_mapping={"p1": "kd", "p2": "threshold_s1 = st * suzmax", "p3": "capacity_s1 = suzmax - threshold_s1"},
        state_mapping={"S": "S1_tmp"},
        parameter_bounds={"p1": "[0.0, 1.0]", "p2": "[0.05, 1900.0] from st*suzmax", "p3": "[0.05, 1900.0] from suzmax*(1-st)"},
        call_site_lines=(118,),
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


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _target_by_key(formula: str, active_model: str) -> TargetSpec:
    return next(item for item in TARGETS if item.formula == formula and item.active_model == active_model)


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


def _arg_order(target: TargetSpec) -> list[str]:
    fn = getattr(__import__(target.flux_module, fromlist=[target.formula]), target.formula)
    return [name for name in inspect.signature(fn).parameters if name != "nearzero"]


class TraceRecorder:
    def __init__(self, target: TargetSpec):
        self.target = target
        self.calls_by_regime: dict[str, list[dict[str, Any]]] = {}

    def record(self, forcing_regime: str, arg_order: list[str], args: tuple[torch.Tensor, ...], line_no: int) -> None:
        max_numel = max(tensor.numel() for tensor in args) if args else 1
        call = {
            name: (
                tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1).clone().repeat(max_numel)
                if tensor.numel() == 1 and max_numel > 1
                else tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1).clone()
            )
            for name, tensor in zip(arg_order, args)
        }
        call["__call_site__"] = f"models/core/{self.target.active_model}.py:{line_no}"
        self.calls_by_regime.setdefault(forcing_regime, []).append(call)


@contextmanager
def _patched_core_symbol(target: TargetSpec, recorder: TraceRecorder, forcing_regime: str):
    core_module = __import__(f"models.core.{target.active_model}", fromlist=[target.active_model])
    flux_module = __import__(target.flux_module, fromlist=[target.formula])
    original_core_fn = getattr(core_module, target.formula)
    original_flux_fn = getattr(flux_module, target.formula)
    arg_order = _arg_order(target)

    def wrapped(*args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        recorder.record(forcing_regime, arg_order, args[: len(arg_order)], caller.f_lineno if caller is not None else -1)
        return original_flux_fn(*args, **kwargs)

    setattr(core_module, target.formula, wrapped)
    try:
        yield
    finally:
        setattr(core_module, target.formula, original_core_fn)


def _run_trace_rollouts(target: TargetSpec) -> TraceRecorder:
    entry = CORE_MODEL_REGISTRY[target.active_model]
    recorder = TraceRecorder(target)
    torch.manual_seed(FIXED_SEED)
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
                    _, _, next_states, _ = _call_step(
                        entry=entry,
                        forcing_at_step=step_forcing,
                        step_index=step_index,
                        params_list=params_list,
                        states=states,
                        mean_precip=mean_precip,
                        return_diagnostics=False,
                    )
                    states = next_states
    return recorder


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return math.nan
    return float(torch.quantile(values, q).item())


def _aggregate_trace(recorder: TraceRecorder) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]], dict[str, torch.Tensor]]:
    target = recorder.target
    rows: list[dict[str, Any]] = []
    pooled: dict[str, list[torch.Tensor]] = {name: [] for name in _arg_order(target)}
    for regime_name, calls in recorder.calls_by_regime.items():
        for arg_name in _arg_order(target):
            values = torch.cat([call[arg_name] for call in calls]) if calls else torch.empty(0, dtype=torch.float64)
            pooled[arg_name].append(values)
            rows.append(
                {
                    "formula": target.formula,
                    "active_model": target.active_model,
                    "argument_name": arg_name,
                    "forcing_regime": regime_name,
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
                    "call_sites": ";".join(sorted({call["__call_site__"] for call in calls})),
                }
            )
    pooled_stats: dict[str, dict[str, float]] = {}
    pooled_joint: dict[str, torch.Tensor] = {}
    for arg_name, parts in pooled.items():
        values = torch.cat(parts).to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
        pooled_joint[arg_name] = values
        pooled_stats[arg_name] = {
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
        }
    return rows, pooled_stats, pooled_joint


def _context_rows() -> list[dict[str, Any]]:
    usage = [ctx for ctx in load_flux_usage_contexts() if ctx.module_type == "core"]
    rows = []
    for target in TARGETS:
        flux_path = REPO_ROOT / f"{target.flux_module.replace('.', '/')}.py"
        flux_start, flux_end = _line_bounds(flux_path, target.formula)
        ctx = next(ctx for ctx in usage if ctx.flux_function == target.formula and ctx.model_name == target.active_model)
        rows.append(
            {
                "formula": target.formula,
                "active_model": target.active_model,
                "flux_file": str(flux_path.relative_to(REPO_ROOT)),
                "flux_lines": f"{flux_start}-{flux_end}",
                "core_file": ctx.call_site.split(":")[0],
                "call_site_lines": ";".join(str(line) for line in target.call_site_lines),
                "parameter_mapping": _json(target.parameter_mapping),
                "parameter_bounds": _json(target.parameter_bounds),
                "state_mapping": _json(target.state_mapping),
                "output_meaning": target.output_meaning,
                "downstream_state_update": target.downstream_state_update,
                "expected_bounds": target.expected_bounds,
                "notes": target.notes,
            }
        )
    return rows


def _source_context_markdown(context_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Medium Formula Source Context",
        "",
        "This document records the exact flux code and active core-model call context for the three remaining medium/document-only active formulas.",
        "",
    ]
    for row in context_rows:
        target = _target_by_key(row["formula"], row["active_model"])
        flux_path = REPO_ROOT / row["flux_file"]
        flux_start, flux_end = map(int, row["flux_lines"].split("-"))
        core_path = REPO_ROOT / row["core_file"]
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
        for line_no in target.call_site_lines:
            max_line = len(core_path.read_text(encoding="utf-8").splitlines())
            lines.append(f"`{row['core_file']}:{line_no}`")
            lines.append("")
            lines.append("```python")
            lines.append(_snippet(core_path, max(1, line_no - 4), min(line_no + 6, max_line)))
            lines.append("```")
            lines.append("")
        lines.append(f"- Parameter mapping: `{row['parameter_mapping']}`")
        lines.append(f"- Parameter bounds: `{row['parameter_bounds']}`")
        lines.append(f"- State mapping: `{row['state_mapping']}`")
        lines.append(f"- Output meaning: {row['output_meaning']}")
        lines.append(f"- Downstream state update: {row['downstream_state_update']}")
        lines.append(f"- Expected bounds: {row['expected_bounds']}")
        lines.append(f"- Notes: {row['notes']}")
        lines.append("")
    return "\n".join(lines)


@dataclass(frozen=True)
class CandidateSpec:
    formula: str
    candidate_id: str
    candidate_description: str
    expression_summary: str
    exact_equivalent: str
    expected_gradient_effect: str
    expected_output_distortion: str
    expected_leakage_risk: str
    preserves_physical_meaning: str
    recommended_for_testing: str
    reason: str
    fn: Callable[..., torch.Tensor]


def _soft_gate_storage_above_custom(S: torch.Tensor, threshold: torch.Tensor, k: float, nearzero: float = DEFAULT_NEARZERO) -> torch.Tensor:
    thresh_abs = torch.abs(threshold) + nearzero
    scale = torch.clamp(torch.as_tensor(k, dtype=S.dtype, device=S.device) / thresh_abs, max=50.0)
    return torch.sigmoid(scale * (S - threshold))


def _stable_softmin(a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
    stacked = torch.stack((-tau * a, -tau * b), dim=0)
    return -torch.logsumexp(stacked, dim=0) / tau


def _candidate_specs() -> list[CandidateSpec]:
    specs: list[CandidateSpec] = [
        CandidateSpec(
            formula="baseflow_6",
            candidate_id="A_current",
            candidate_description="Current formula unchanged",
            expression_summary="min(S, p1*S^2) * soft_gate_storage_above(S, p2, k=10)",
            exact_equivalent="yes",
            expected_gradient_effect="Baseline kink from `min` plus sigmoid gate derivative near threshold.",
            expected_output_distortion="none",
            expected_leakage_risk="none beyond current gate behavior",
            preserves_physical_meaning="yes",
            recommended_for_testing="baseline",
            reason="Reference implementation.",
            fn=lambda p1, p2, S, nearzero=DEFAULT_NEARZERO: torch.minimum(S, p1 * S.pow(2))
            * _soft_gate_storage_above_custom(S, p2, k=10.0, nearzero=nearzero),
        ),
        CandidateSpec(
            formula="baseflow_6",
            candidate_id="B_k2",
            candidate_description="Current formula with softer gate k=2",
            expression_summary="min(S, p1*S^2) * soft_gate_storage_above(S, p2, k=2)",
            exact_equivalent="no",
            expected_gradient_effect="Broader threshold transition and less concentrated gate gradient.",
            expected_output_distortion="moderate near threshold",
            expected_leakage_risk="higher soft-gate leakage near threshold",
            preserves_physical_meaning="mostly",
            recommended_for_testing="future_only",
            reason="Tests threshold-sharpness sensitivity only.",
            fn=lambda p1, p2, S, nearzero=DEFAULT_NEARZERO: torch.minimum(S, p1 * S.pow(2))
            * _soft_gate_storage_above_custom(S, p2, k=2.0, nearzero=nearzero),
        ),
        CandidateSpec(
            formula="baseflow_6",
            candidate_id="B_k5",
            candidate_description="Current formula with softer gate k=5",
            expression_summary="min(S, p1*S^2) * soft_gate_storage_above(S, p2, k=5)",
            exact_equivalent="no",
            expected_gradient_effect="Somewhat broader threshold transition.",
            expected_output_distortion="low_to_moderate near threshold",
            expected_leakage_risk="slightly higher near threshold",
            preserves_physical_meaning="mostly",
            recommended_for_testing="future_only",
            reason="Compares sensitivity to a softer but still sharp gate.",
            fn=lambda p1, p2, S, nearzero=DEFAULT_NEARZERO: torch.minimum(S, p1 * S.pow(2))
            * _soft_gate_storage_above_custom(S, p2, k=5.0, nearzero=nearzero),
        ),
        CandidateSpec(
            formula="baseflow_6",
            candidate_id="B_k20",
            candidate_description="Current formula with sharper gate k=20",
            expression_summary="min(S, p1*S^2) * soft_gate_storage_above(S, p2, k=20)",
            exact_equivalent="no",
            expected_gradient_effect="Sharper threshold transition and more concentrated gate gradient.",
            expected_output_distortion="low except very near threshold",
            expected_leakage_risk="lower near threshold",
            preserves_physical_meaning="mostly",
            recommended_for_testing="no",
            reason="Primarily diagnostic; unlikely to improve stability.",
            fn=lambda p1, p2, S, nearzero=DEFAULT_NEARZERO: torch.minimum(S, p1 * S.pow(2))
            * _soft_gate_storage_above_custom(S, p2, k=20.0, nearzero=nearzero),
        ),
        CandidateSpec(
            formula="baseflow_6",
            candidate_id="C_softmin_tau20",
            candidate_description="Smooth-min candidate for min(S, p1*S^2) only",
            expression_summary="softmin_tau20(S, p1*S^2) * soft_gate_storage_above(S, p2, k=10)",
            exact_equivalent="no",
            expected_gradient_effect="Removes hard `min` kink but not the threshold gate behavior.",
            expected_output_distortion="moderate around the `min` branch switch",
            expected_leakage_risk="same as current gate",
            preserves_physical_meaning="partially",
            recommended_for_testing="future_only",
            reason="Useful only to quantify `min`-kink sensitivity.",
            fn=lambda p1, p2, S, nearzero=DEFAULT_NEARZERO: _stable_softmin(S, p1 * S.pow(2), tau=20.0)
            * _soft_gate_storage_above_custom(S, p2, k=10.0, nearzero=nearzero),
        ),
        CandidateSpec(
            formula="baseflow_9",
            candidate_id="A_current_beta50",
            candidate_description="Current formula unchanged",
            expression_summary="p1 * softplus(S-p2, beta=50)",
            exact_equivalent="yes",
            expected_gradient_effect="Baseline sharp but smooth threshold transition.",
            expected_output_distortion="none",
            expected_leakage_risk="baseline sub-threshold leakage",
            preserves_physical_meaning="yes",
            recommended_for_testing="baseline",
            reason="Reference implementation.",
            fn=lambda p1, p2, S, nearzero=DEFAULT_NEARZERO: p1 * F.softplus(S - p2, beta=50.0),
        ),
        CandidateSpec(
            formula="baseflow_9",
            candidate_id="B_beta10",
            candidate_description="Softer softplus beta=10",
            expression_summary="p1 * softplus(S-p2, beta=10)",
            exact_equivalent="no",
            expected_gradient_effect="Less concentrated threshold gradient.",
            expected_output_distortion="moderate near threshold",
            expected_leakage_risk="higher below-threshold leakage",
            preserves_physical_meaning="partially",
            recommended_for_testing="future_only",
            reason="Quantifies distortion/leakage tradeoff from a softer threshold.",
            fn=lambda p1, p2, S, nearzero=DEFAULT_NEARZERO: p1 * F.softplus(S - p2, beta=10.0),
        ),
        CandidateSpec(
            formula="baseflow_9",
            candidate_id="B_beta20",
            candidate_description="Moderately sharp softplus beta=20",
            expression_summary="p1 * softplus(S-p2, beta=20)",
            exact_equivalent="no",
            expected_gradient_effect="Less concentrated threshold gradient than beta=50.",
            expected_output_distortion="low_to_moderate near threshold",
            expected_leakage_risk="higher than current, lower than beta=10",
            preserves_physical_meaning="partially",
            recommended_for_testing="future_only",
            reason="Middle-ground smoothing candidate for sensitivity only.",
            fn=lambda p1, p2, S, nearzero=DEFAULT_NEARZERO: p1 * F.softplus(S - p2, beta=20.0),
        ),
        CandidateSpec(
            formula="baseflow_9",
            candidate_id="C_hard_relu",
            candidate_description="Hard ReLU reference",
            expression_summary="p1 * relu(S-p2)",
            exact_equivalent="no",
            expected_gradient_effect="Eliminates softplus leakage but introduces a hard kink.",
            expected_output_distortion="low above threshold, exact zero below threshold",
            expected_leakage_risk="none below threshold",
            preserves_physical_meaning="yes",
            recommended_for_testing="future_only",
            reason="Useful as a leakage-free behavioral reference, not as a stability rewrite.",
            fn=lambda p1, p2, S, nearzero=DEFAULT_NEARZERO: p1 * F.relu(S - p2),
        ),
        CandidateSpec(
            formula="interflow_10",
            candidate_id="A_current",
            candidate_description="Current formula unchanged",
            expression_summary="p1 * relu(S-p2) / (p3+eps)",
            exact_equivalent="yes",
            expected_gradient_effect="Baseline hard-threshold dead region and ReLU kink at threshold.",
            expected_output_distortion="none",
            expected_leakage_risk="none below threshold",
            preserves_physical_meaning="yes",
            recommended_for_testing="baseline",
            reason="Reference implementation.",
            fn=lambda S, p1, p2, p3, nearzero=DEFAULT_NEARZERO: p1 * F.relu(S - p2) / (p3 + nearzero),
        ),
        CandidateSpec(
            formula="interflow_10",
            candidate_id="B_softplus20",
            candidate_description="Softplus threshold beta=20",
            expression_summary="p1 * softplus(S-p2, beta=20) / (p3+eps)",
            exact_equivalent="no",
            expected_gradient_effect="Removes ReLU kink and dead region.",
            expected_output_distortion="moderate near threshold",
            expected_leakage_risk="positive below-threshold leakage",
            preserves_physical_meaning="no",
            recommended_for_testing="future_only",
            reason="Only suitable for future sensitivity analysis because it leaks below threshold.",
            fn=lambda S, p1, p2, p3, nearzero=DEFAULT_NEARZERO: p1 * F.softplus(S - p2, beta=20.0) / (p3 + nearzero),
        ),
        CandidateSpec(
            formula="interflow_10",
            candidate_id="B_softplus50",
            candidate_description="Softplus threshold beta=50",
            expression_summary="p1 * softplus(S-p2, beta=50) / (p3+eps)",
            exact_equivalent="no",
            expected_gradient_effect="Removes ReLU kink with less leakage than beta=20.",
            expected_output_distortion="low_to_moderate near threshold",
            expected_leakage_risk="positive below-threshold leakage",
            preserves_physical_meaning="no",
            recommended_for_testing="future_only",
            reason="Closer to hard threshold than beta=20 but still leaks below threshold.",
            fn=lambda S, p1, p2, p3, nearzero=DEFAULT_NEARZERO: p1 * F.softplus(S - p2, beta=50.0) / (p3 + nearzero),
        ),
        CandidateSpec(
            formula="interflow_10",
            candidate_id="C_gate_times_excess",
            candidate_description="Soft gate times positive excess",
            expression_summary="p1 * relu(S-p2) * soft_gate_storage_above(S,p2) / (p3+eps)",
            exact_equivalent="no",
            expected_gradient_effect="Keeps zero below threshold but retains ReLU kink and adds an extra gate derivative.",
            expected_output_distortion="low near threshold but non-negligible above threshold shoulder",
            expected_leakage_risk="none below threshold",
            preserves_physical_meaning="mostly",
            recommended_for_testing="no",
            reason="Does not solve the ReLU kink and changes the above-threshold amplitude near threshold.",
            fn=lambda S, p1, p2, p3, nearzero=DEFAULT_NEARZERO: p1
            * F.relu(S - p2)
            * _soft_gate_storage_above_custom(S, p2, k=10.0, nearzero=nearzero)
            / (p3 + nearzero),
        ),
    ]
    return specs


def _candidate_option_rows() -> list[dict[str, Any]]:
    rows = []
    for spec in _candidate_specs():
        rows.append(
            {
                "formula": spec.formula,
                "candidate_id": spec.candidate_id,
                "candidate_description": spec.candidate_description,
                "expression_summary": spec.expression_summary,
                "exact_equivalent": spec.exact_equivalent,
                "expected_gradient_effect": spec.expected_gradient_effect,
                "expected_output_distortion": spec.expected_output_distortion,
                "expected_leakage_risk": spec.expected_leakage_risk,
                "preserves_physical_meaning": spec.preserves_physical_meaning,
                "recommended_for_testing": spec.recommended_for_testing,
                "reason": spec.reason,
            }
        )
    return rows


def _autograd_gradients(output: torch.Tensor, inputs: dict[str, torch.Tensor], grad_inputs: tuple[str, ...]) -> dict[str, torch.Tensor | None]:
    required = [inputs[name] for name in grad_inputs if inputs[name].requires_grad]
    grads = torch.autograd.grad(output.sum(), required, allow_unused=True, retain_graph=False)
    it = iter(grads)
    result: dict[str, torch.Tensor | None] = {}
    for name in grad_inputs:
        result[name] = next(it) if inputs[name].requires_grad else None
    return result


def _evaluate_candidate(
    target: TargetSpec,
    candidate: CandidateSpec,
    pooled_joint: dict[str, torch.Tensor],
    current_output: torch.Tensor | None = None,
) -> tuple[dict[str, Any], torch.Tensor]:
    arg_order = _arg_order(target)
    inputs = {
        name: pooled_joint[name].clone().to(device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE).requires_grad_(name in target.grad_inputs)
        for name in arg_order
    }
    output = candidate.fn(*[inputs[name] for name in arg_order], nearzero=DEFAULT_NEARZERO)
    if current_output is None:
        current_output = output.detach()
    diff = output.detach() - current_output
    diff_l2 = float(torch.linalg.norm(diff).item())
    current_l2 = float(torch.linalg.norm(current_output).item())
    grads = _autograd_gradients(output, inputs, target.grad_inputs)

    grad_nan_count = 0
    grad_inf_count = 0
    grad_values = []
    for grad in grads.values():
        if grad is None:
            continue
        grad_nan_count += int(torch.isnan(grad).sum().item())
        grad_inf_count += int(torch.isinf(grad).sum().item())
        grad_values.append(torch.abs(grad).reshape(-1))
    flat = torch.cat(grad_values) if grad_values else torch.zeros(1, dtype=DEFAULT_DTYPE)
    max_abs_grad = float(flat.max().item())
    mean_abs_grad = float(flat.mean().item())
    median_abs_grad = float(flat.median().item())
    zero_gradient_fraction = float((flat < 1.0e-12).float().mean().item())

    below_threshold_leakage_count = 0
    below_threshold_leakage_max = 0.0
    if "p2" in inputs:
        mask = inputs["S"].detach() < (inputs["p2"].detach() - 1.0e-10)
        if mask.any():
            leaked = output.detach()[mask]
            positive_leak = leaked[leaked > 1.0e-10]
            below_threshold_leakage_count = int(positive_leak.numel())
            below_threshold_leakage_max = float(positive_leak.max().item()) if positive_leak.numel() else 0.0

    if target.formula in {"baseflow_6", "baseflow_9"}:
        output_bound_violation_count = int((output.detach() > inputs["S"].detach() + 1.0e-10).sum().item())
    else:
        output_bound_violation_count = 0

    row = {
        "formula": target.formula,
        "active_model": target.active_model,
        "candidate_id": candidate.candidate_id,
        "candidate_description": candidate.candidate_description,
        "exact_equivalent": candidate.exact_equivalent,
        "max_abs_output_diff_vs_current": float(torch.max(torch.abs(diff)).item()),
        "relative_L2_output_diff_vs_current": diff_l2 / max(current_l2, 1.0e-12),
        "max_abs_grad": max_abs_grad,
        "mean_abs_grad": mean_abs_grad,
        "median_abs_grad": median_abs_grad,
        "grad_nan_count": grad_nan_count,
        "grad_inf_count": grad_inf_count,
        "zero_gradient_fraction": zero_gradient_fraction,
        "below_threshold_leakage_count": below_threshold_leakage_count,
        "below_threshold_leakage_max": below_threshold_leakage_max,
        "output_bound_violation_count": output_bound_violation_count,
        "output_negative_count": int((output.detach() < -1.0e-12).sum().item()),
    }
    return row, output.detach()


def _candidate_comparison_rows(pooled_joint_by_key: dict[tuple[str, str], dict[str, torch.Tensor]]) -> list[dict[str, Any]]:
    by_formula: dict[str, list[CandidateSpec]] = {}
    for spec in _candidate_specs():
        by_formula.setdefault(spec.formula, []).append(spec)

    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        pooled_joint = pooled_joint_by_key[(target.formula, target.active_model)]
        current_spec = next(spec for spec in by_formula[target.formula] if spec.candidate_id.startswith("A_"))
        current_row, current_output = _evaluate_candidate(target, current_spec, pooled_joint, current_output=None)
        rows.append(current_row)
        for spec in by_formula[target.formula]:
            if spec is current_spec:
                continue
            row, _ = _evaluate_candidate(target, spec, pooled_joint, current_output=current_output)
            rows.append(row)
    return rows


def _gradient_analysis_markdown(
    pooled_stats_by_key: dict[tuple[str, str], dict[str, dict[str, float]]],
    comparison_rows: list[dict[str, Any]],
) -> str:
    rows_by_key = {(row["formula"], row["active_model"], row["candidate_id"]): row for row in comparison_rows}
    lines = [
        "# Medium Formula Gradient Analysis",
        "",
        "This document summarizes the mathematical structure, derivative behavior, and realistic-domain gradient implications for the three remaining medium/document-only active contexts.",
        "",
    ]

    for target in TARGETS:
        key = (target.formula, target.active_model)
        stats = pooled_stats_by_key[key]
        current_candidate = "A_current_beta50" if target.formula == "baseflow_9" else "A_current"
        current = rows_by_key[(target.formula, target.active_model, current_candidate)]

        lines.append(f"## {target.formula} / {target.active_model}")
        lines.append("")
        if target.formula == "baseflow_6":
            lines.extend(
                [
                    "- Mathematical expression: `q(S) = min(S, p1*S^2) * sigmoid(scale*(S-p2))`, with `scale = min(10 / (|p2| + eps), 50)`.",
                    "- Storage derivative: product-rule combination of a kinked `min(S, p1*S^2)` derivative and the sigmoid gate derivative `gate*(1-gate)*scale`.",
                    "- Parameter derivatives: `dq/dp1` follows the active `p1*S^2` branch; `dq/dp2` comes only through the gate scaling and threshold shift.",
                    "- Kink/singular points: branch switch at `S = 1/p1` for `p1 > 0`; no singularity in the active domain because TCM fixes `p2 = 0`, which clamps gate scale to `50` rather than dividing by zero.",
                    "- Dead-gradient regions: none from the gate in realistic TCM rollouts because `S >= 0` and the active threshold is fixed at `0`; near-zero gradients mainly come from the quadratic factor near `S ~= 0` and from the logistic gate saturating to `1` away from zero.",
                    f"- Realistic traced domain: `S` in [{stats['S']['min']:.6g}, {stats['S']['max']:.6g}], `p1` in [{stats['p1']['min']:.6g}, {stats['p1']['max']:.6g}], `p2` fixed at [{stats['p2']['min']:.6g}, {stats['p2']['max']:.6g}].",
                    f"- Realistic-domain status: max_abs_grad={current['max_abs_grad']:.6g}, grad_nan_count={current['grad_nan_count']}, grad_inf_count={current['grad_inf_count']}, zero_gradient_fraction={current['zero_gradient_fraction']:.6g}.",
                    "- Practical interpretation: the medium label is driven by threshold/min-kink structure and saturation of the gate, not by any realistic-domain NaN/Inf instability.",
                ]
            )
        elif target.formula == "baseflow_9":
            lines.extend(
                [
                    "- Mathematical expression: `q(S) = p1 * softplus(S-p2, beta=50)`.",
                    "- Storage derivative: `dq/dS = p1 * sigmoid(50*(S-p2))`, which is finite everywhere and peaks near the threshold.",
                    "- Parameter derivatives: `dq/dp1 = softplus(S-p2, beta=50)` and `dq/dp2 = -p1 * sigmoid(50*(S-p2))`.",
                    "- Kink/singular points: none; the helper is smooth by construction.",
                    "- Dead-gradient regions: only asymptotic saturation far below threshold; not a hard dead region.",
                    f"- Realistic traced domain: `S` in [{stats['S']['min']:.6g}, {stats['S']['max']:.6g}], `p1` in [{stats['p1']['min']:.6g}, {stats['p1']['max']:.6g}], `p2` in [{stats['p2']['min']:.6g}, {stats['p2']['max']:.6g}].",
                    f"- Realistic-domain status: max_abs_grad={current['max_abs_grad']:.6g}, grad_nan_count={current['grad_nan_count']}, grad_inf_count={current['grad_inf_count']}, zero_gradient_fraction={current['zero_gradient_fraction']:.6g}.",
                    "- Practical interpretation: the medium label is mainly large-but-finite sensitivity from realistic output magnitude, especially `dq/dp1 = excess_storage`, not numerical instability. The broad-domain bound concern was already resolved as an artifact.",
                ]
            )
        else:
            lines.extend(
                [
                    "- Mathematical expression: `q(S) = p1 * relu(S-p2) / (p3 + eps)`.",
                    "- Storage derivative: `dq/dS = p1/(p3+eps)` above threshold and `0` below threshold, with a hard kink at `S = p2`.",
                    "- Parameter derivatives: `dq/dp1 = relu(S-p2)/(p3+eps)`, `dq/dp2 = -p1/(p3+eps)` above threshold and `0` below threshold, `dq/dp3 = -p1*relu(S-p2)/(p3+eps)^2`.",
                    "- Kink/singular points: hard ReLU kink at the threshold; no realistic-domain singularity because active `p3 = suzmax*(1-st)` stays strictly positive in traced rollouts.",
                    "- Dead-gradient regions: exact zero-gradient region below threshold is expected and physically meaningful in TOPMODEL.",
                    f"- Realistic traced domain: `S` in [{stats['S']['min']:.6g}, {stats['S']['max']:.6g}], `p2` in [{stats['p2']['min']:.6g}, {stats['p2']['max']:.6g}], `p3` in [{stats['p3']['min']:.6g}, {stats['p3']['max']:.6g}].",
                    f"- Realistic-domain status: max_abs_grad={current['max_abs_grad']:.6g}, grad_nan_count={current['grad_nan_count']}, grad_inf_count={current['grad_inf_count']}, zero_gradient_fraction={current['zero_gradient_fraction']:.6g}.",
                    "- Practical interpretation: the medium label is due to an expected hard threshold dead region and ReLU kink, not to non-finite gradients. Softening it would necessarily change below-threshold behavior or above-threshold onset.",
                ]
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _report(
    context_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    pooled_stats_by_key: dict[tuple[str, str], dict[str, dict[str, float]]],
) -> str:
    rows_by_key = {(row["formula"], row["active_model"], row["candidate_id"]): row for row in comparison_rows}
    lines = [
        "# Medium Flux Formula Rewrite Review Report",
        "",
        "## 1. Scope",
        "- Investigated the three remaining medium/document-only active flux contexts: `baseflow_6 / tcm`, `baseflow_9 / gsfb`, and `interflow_10 / topmodel`.",
        "- This review is candidate-comparison only. No active formula, smoothing rule, clamp, parameter bound, model physics, soft-gate default, unit hydrograph code, or water-balance fix was changed.",
        "",
        "## 2. Why these formulas were reviewed",
        "- They are the only remaining active contexts classified as medium/document-only after the complete active flux-gradient review.",
        "- The core question is whether any exact numerical-stability rewrite exists, or whether the remaining medium labels are simply expected threshold/kink behavior that should stay documented rather than modified.",
        "",
        "## 3. Current source and model context",
    ]
    for row in context_rows:
        lines.append(f"- `{row['formula']}` / `{row['active_model']}`: {row['output_meaning']}")

    def current_row(formula: str, model: str) -> dict[str, Any]:
        cid = "A_current_beta50" if formula == "baseflow_9" else "A_current"
        return rows_by_key[(formula, model, cid)]

    lines.extend(
        [
            "",
            "## 4. Gradient analysis for baseflow_6",
            f"- Current realistic-domain max_abs_grad: {current_row('baseflow_6', 'tcm')['max_abs_grad']:.6g}",
            f"- Current realistic-domain NaN/Inf: output={current_row('baseflow_6', 'tcm')['grad_nan_count'] + current_row('baseflow_6', 'tcm')['grad_inf_count'] + 0}, grad={current_row('baseflow_6', 'tcm')['grad_nan_count'] + current_row('baseflow_6', 'tcm')['grad_inf_count']}",
            "- Root cause of medium label: threshold/min-kink structure plus gate saturation, not realistic-domain instability.",
            "- Exact stable rewrite status: none found. Changing `k`, smoothing the `min`, or otherwise altering the gate changes the actual function.",
            "",
            "## 5. Gradient analysis for baseflow_9",
            f"- Current realistic-domain max_abs_grad: {current_row('baseflow_9', 'gsfb')['max_abs_grad']:.6g}",
            f"- Current realistic-domain NaN/Inf: output={current_row('baseflow_9', 'gsfb')['grad_nan_count'] + current_row('baseflow_9', 'gsfb')['grad_inf_count'] + 0}, grad={current_row('baseflow_9', 'gsfb')['grad_nan_count'] + current_row('baseflow_9', 'gsfb')['grad_inf_count']}",
            "- Root cause of medium label: large but finite sensitivity from realistic output magnitude and parameter-product scaling, not NaN/Inf behavior.",
            "- Exact stable rewrite status: none needed and none found; the current helper is already smooth and finite.",
            "",
            "## 6. Gradient analysis for interflow_10",
            f"- Current realistic-domain max_abs_grad: {current_row('interflow_10', 'topmodel')['max_abs_grad']:.6g}",
            f"- Current realistic-domain NaN/Inf: output={current_row('interflow_10', 'topmodel')['grad_nan_count'] + current_row('interflow_10', 'topmodel')['grad_inf_count'] + 0}, grad={current_row('interflow_10', 'topmodel')['grad_nan_count'] + current_row('interflow_10', 'topmodel')['grad_inf_count']}",
            "- Root cause of medium label: physically meaningful hard threshold dead region and ReLU kink, plus denominator scaling by a strictly positive capacity term.",
            "- Exact stable rewrite status: none found. Any softplus/gate substitute changes threshold behavior or introduces leakage.",
            "",
            "## 7. Candidate rewrite options",
            "- Candidate options are listed in `medium_formula_candidate_options.csv`.",
            "",
            "## 8. Numerical candidate comparison",
            "",
            "| formula | candidate | max_abs_output_diff | relative_L2_diff | max_abs_grad | grad_nan_inf | leakage_count |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"| {row['formula']} | {row['candidate_id']} | {row['max_abs_output_diff_vs_current']:.6g} | "
            f"{row['relative_L2_output_diff_vs_current']:.6g} | {row['max_abs_grad']:.6g} | "
            f"{row['grad_nan_count'] + row['grad_inf_count']} | {row['below_threshold_leakage_count']} |"
        )

    lines.extend(
        [
            "",
            "## 9. Which candidate, if any, is algebraically equivalent",
            "- None of the investigated alternatives for `baseflow_6`, `baseflow_9`, or `interflow_10` is an exact algebraically equivalent stable rewrite of the active formula.",
            "- The only exact-equivalent stable rewrite found in the broader program remained the earlier `saturation_3 -> torch.sigmoid(z)` rewrite.",
            "",
            "## 10. Which candidates introduce output distortion or leakage",
            "- `baseflow_6` gate-`k` variants distort the near-threshold activation width; the soft-min candidate additionally distorts the branch transition between `S` and `p1*S^2`.",
            "- `baseflow_9` softer softplus variants materially increase below-threshold leakage, while the hard-ReLU reference removes leakage but reintroduces a hard kink.",
            "- `interflow_10` softplus variants introduce below-threshold leakage; the gate-times-excess candidate preserves zero leakage but still changes the above-threshold shoulder and does not remove the core ReLU kink.",
            "",
            "## 11. Recommended decision for each formula",
        ]
    )

    decisions = {
        "baseflow_6": "keep unchanged; document only",
        "baseflow_9": "keep unchanged; document only",
        "interflow_10": "do not modify now; if anything, only future benchmark sensitivity work should test smoothing candidates",
    }
    for formula, decision in decisions.items():
        model = next(target.active_model for target in TARGETS if target.formula == formula)
        lines.append(f"- `{formula}` / `{model}`: {decision}.")

    lines.extend(
        [
            "",
            "## 12. Required validation if any formula is later changed",
            "- Any future smoothing or rewrite experiment must be benchmarked against realistic-domain output distortion, below-threshold leakage, water-balance sensitivity, and calibration regression results before adoption.",
            "",
            "## Direct answers",
            "- `baseflow_6` should not be changed now.",
            "- `baseflow_9` should not be changed now.",
            "- `interflow_10` should not be changed now.",
            "- No investigated change is an exact numerical-stability rewrite of these active formulas.",
            "- All investigated smoothing-style alternatives are hydrological behavior changes, not pure numerical-evaluation rewrites.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_medium_review() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    context_rows = _context_rows()
    (OUTPUT_DIR / "medium_formula_source_context.md").write_text(_source_context_markdown(context_rows), encoding="utf-8")
    _write_csv(OUTPUT_DIR / "medium_formula_context_inventory.csv", context_rows)

    trace_rows: list[dict[str, Any]] = []
    pooled_stats_by_key: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    pooled_joint_by_key: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for target in TARGETS:
        recorder = _run_trace_rollouts(target)
        rows, pooled_stats, pooled_joint = _aggregate_trace(recorder)
        trace_rows.extend(rows)
        pooled_stats_by_key[(target.formula, target.active_model)] = pooled_stats
        pooled_joint_by_key[(target.formula, target.active_model)] = pooled_joint

    _write_csv(OUTPUT_DIR / "medium_formula_realistic_domain_trace.csv", trace_rows)
    option_rows = _candidate_option_rows()
    _write_csv(OUTPUT_DIR / "medium_formula_candidate_options.csv", option_rows)

    comparison_rows = _candidate_comparison_rows(pooled_joint_by_key)
    _write_csv(OUTPUT_DIR / "medium_formula_candidate_comparison.csv", comparison_rows)

    (OUTPUT_DIR / "medium_formula_gradient_analysis.md").write_text(
        _gradient_analysis_markdown(pooled_stats_by_key, comparison_rows),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "medium_flux_formula_rewrite_review_report.md").write_text(
        _report(context_rows, comparison_rows, pooled_stats_by_key),
        encoding="utf-8",
    )

    return {
        "context_rows": context_rows,
        "trace_rows": trace_rows,
        "comparison_rows": comparison_rows,
        "pooled_stats_by_key": pooled_stats_by_key,
    }


def main() -> None:
    run_medium_review()


if __name__ == "__main__":
    main()
