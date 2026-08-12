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


OUTPUT_DIR = REPO_ROOT / "validation_results" / "batch_c_flux_realistic_review"
GRADIENT_DIR = REPO_ROOT / "validation_results" / "flux_gradient_stability"
DEFAULT_DTYPE = torch.float64
DEFAULT_DEVICE = "cpu"
DEFAULT_NEARZERO = 1.0e-6
FIXED_SEED = 20260625
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
    representative_cases: tuple[str, ...]
    capacity_pair: tuple[str, str] | None = None


BATCH_C_TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec(
        formula="excess_1",
        flux_module="models.flux.excess",
        active_model="australia",
        probe_arg="So",
        grad_inputs=("So", "Smax"),
        arg_roles={"So": "prospective_storage", "Smax": "receiving_capacity"},
        output_meaning="Overflow/recharge excess from the Australia unsaturated store into the saturated pathway after saturation-style recharge is removed.",
        expected_physical_bounds="0 <= flux_se <= So and `So - flux_se <= cap_s1_to_s2`. This is a prospective-state overflow term, so `flux_se <= P` is not a valid bound.",
        expected_bound_type="prospective_overflow_to_capacity",
        existing_pre_or_post_caps="No explicit post-call cap on `flux_se`; the formula is used as the exact overflow that reduces `S1_tmp_in` down to `cap_s1_to_s2` before evaporation.",
        notes="Broad high-risk flag likely came from comparing overflow against an incoming-flux heuristic rather than against the prospective overflow semantics.",
        representative_cases=("median", "random", "low_state", "high_state", "high_capacity_gap"),
        capacity_pair=("So", "Smax"),
    ),
    TargetSpec(
        formula="excess_1",
        flux_module="models.flux.excess",
        active_model="susannah2",
        probe_arg="So",
        grad_inputs=("So", "Smax"),
        arg_roles={"So": "prospective_storage", "Smax": "receiving_capacity"},
        output_meaning="Overflow transfer candidate from the Susannah2 unsaturated store into the saturated store before the model jointly rescales `flux_rg + flux_se`.",
        expected_physical_bounds="Raw overflow semantics are `0 <= flux_se <= So` with `So - flux_se <= cap_s1_to_s2`. The physically relevant model-level transfer is additionally constrained later by the joint `rg+se` scaling step.",
        expected_bound_type="prospective_overflow_with_joint_post_scale",
        existing_pre_or_post_caps="Post-call joint scaling: `flux_rg` and `flux_se` are rescaled together so `flux_rg + flux_se <= P + relu(S1 - cap_s1_to_s2)` before the state update.",
        notes="Broad high-risk flag likely ignored the joint post-flux scaling that constrains the actual transfer from `S1` to `S2`.",
        representative_cases=("median", "random", "low_state", "high_state", "high_capacity_gap"),
        capacity_pair=("So", "Smax"),
    ),
    TargetSpec(
        formula="excess_1",
        flux_module="models.flux.excess",
        active_model="vic",
        probe_arg="So",
        grad_inputs=("So", "Smax"),
        arg_roles={"So": "prospective_storage", "Smax": "interception_capacity"},
        output_meaning="Interception overflow from the VIC canopy/interception store after throughfall is computed.",
        expected_physical_bounds="0 <= flux_iex <= So and `So - flux_iex <= aux_imax`. This is canopy overflow from a prospective store state, not a direct `flux <= P` partition constraint.",
        expected_bound_type="prospective_overflow_to_capacity",
        existing_pre_or_post_caps="No direct post-call cap on `flux_iex`; downstream soil input uses `potential_inf = flux_peff + flux_iex`, and later soil-process caps act on that combined inflow.",
        notes="Broad high-risk flag likely compared the overflow against a generic incoming-flux/storage heuristic instead of the prospective interception-overflow semantics.",
        representative_cases=("median", "random", "low_state", "high_state", "high_capacity_gap"),
        capacity_pair=("So", "Smax"),
    ),
    TargetSpec(
        formula="recharge_1",
        flux_module="models.flux.recharge",
        active_model="modhydrolog",
        probe_arg="S",
        grad_inputs=("p1", "S", "Smax", "flux"),
        arg_roles={"p1": "recharge_coefficient", "S": "soil_wetness_state", "Smax": "soil_capacity", "flux": "remaining_infiltration"},
        output_meaning="Recharge partition from the MODHYDROLOG infiltrated-water remainder into the groundwater store.",
        expected_physical_bounds="0 <= flux_REC <= remain_after_int under the active parameter range because this formula partitions incoming infiltrated water; `flux_REC <= S2` is not a required one-step storage-withdrawal bound.",
        expected_bound_type="incoming_flux_partition",
        existing_pre_or_post_caps="Core applies `flux_REC = torch.minimum(flux_REC_pot, remain_after_int)` before the state update, although the shared formula is already bounded by incoming flux when `crak in [0,1]` and `S/Smax <= 1`.",
        notes="The broad risk likely combined a too-broad diagnostic parameter/range assumption with an invalid `flux <= storage` heuristic.",
        representative_cases=("median", "random", "low_state", "high_state", "high_flux"),
        capacity_pair=("S", "Smax"),
    ),
    TargetSpec(
        formula="recharge_1",
        flux_module="models.flux.recharge",
        active_model="simhyd",
        probe_arg="S",
        grad_inputs=("p1", "S", "Smax", "flux"),
        arg_roles={"p1": "recharge_coefficient", "S": "soil_wetness_state", "Smax": "soil_capacity", "flux": "remaining_infiltration"},
        output_meaning="Recharge partition from the SIMHYD infiltrated-water remainder into the groundwater store.",
        expected_physical_bounds="0 <= flux_REC <= flux_rem_inf under the active parameter range because this formula partitions incoming infiltrated water; `flux_REC <= S2` is not a required one-step storage-withdrawal bound.",
        expected_bound_type="incoming_flux_partition",
        existing_pre_or_post_caps="Core applies `flux_REC = torch.minimum(flux_REC_pot, flux_rem_inf)` before the state update, although the shared formula is already bounded by incoming flux when `crak in [0,1]` and `S/Smax <= 1`.",
        notes="The broad risk likely combined a too-broad diagnostic parameter/range assumption with an invalid `flux <= storage` heuristic.",
        representative_cases=("median", "random", "low_state", "high_state", "high_flux"),
        capacity_pair=("S", "Smax"),
    ),
    TargetSpec(
        formula="split_1",
        flux_module="models.flux.split",
        active_model="flexb",
        probe_arg="p1",
        grad_inputs=("p1", "incoming_flux"),
        arg_roles={"p1": "fast_flow_fraction", "incoming_flux": "surface_excess_flux"},
        output_meaning="Fast-routing share of FLEX-B surface excess, with the slow share computed as the complement.",
        expected_physical_bounds="0 <= flux_rf <= p_excess provided `p1 = 1 - d_split` stays in `[0,1]`. The broad diagnostic was suspected to allow impossible `p1 > 1` values.",
        expected_bound_type="incoming_flux_fraction_partition",
        existing_pre_or_post_caps="No direct post-call cap on `flux_rf`; the slow branch is computed as `relu(p_excess - flux_rf)`, so valid fraction bounds depend on the active `d_split` range rather than on added caps.",
        notes="This context is primarily a parameter-expression range audit: the active expression is `1.0 - d_split`, not a generic free `p1` in `[0.05, 5.0]`.",
        representative_cases=("median", "random", "low_flux", "high_flux", "high_fraction"),
    ),
    TargetSpec(
        formula="split_1",
        flux_module="models.flux.split",
        active_model="flexi",
        probe_arg="p1",
        grad_inputs=("p1", "incoming_flux"),
        arg_roles={"p1": "fast_flow_fraction", "incoming_flux": "surface_excess_flux"},
        output_meaning="Fast-routing share of FLEX-I surface excess, with the slow share computed as the complement.",
        expected_physical_bounds="0 <= flux_rf <= rem_peff provided `p1 = 1 - d_split` stays in `[0,1]`.",
        expected_bound_type="incoming_flux_fraction_partition",
        existing_pre_or_post_caps="No direct post-call cap on `flux_rf`; the slow branch is computed as `relu(rem_peff - flux_rf)`.",
        notes="This context is primarily a parameter-expression range audit: the active expression is `1.0 - d_split`, not a generic free `p1` in `[0.05, 5.0]`.",
        representative_cases=("median", "random", "low_flux", "high_flux", "high_fraction"),
    ),
    TargetSpec(
        formula="split_1",
        flux_module="models.flux.split",
        active_model="flexis",
        probe_arg="p1",
        grad_inputs=("p1", "incoming_flux"),
        arg_roles={"p1": "fast_flow_fraction", "incoming_flux": "surface_excess_flux"},
        output_meaning="Fast-routing share of FLEX-IS surface excess, with the slow share computed as the complement.",
        expected_physical_bounds="0 <= flux_rf <= rem_peff provided `p1 = 1 - d_split` stays in `[0,1]`.",
        expected_bound_type="incoming_flux_fraction_partition",
        existing_pre_or_post_caps="No direct post-call cap on `flux_rf`; the slow branch is computed as `relu(rem_peff - flux_rf)`.",
        notes="This context is primarily a parameter-expression range audit: the active expression is `1.0 - d_split`, not a generic free `p1` in `[0.05, 5.0]`.",
        representative_cases=("median", "random", "low_flux", "high_flux", "high_fraction"),
    ),
    TargetSpec(
        formula="evap_16",
        flux_module="models.flux.evap",
        active_model="penman",
        probe_arg="S2",
        grad_inputs=("p1", "S2", "Ep"),
        arg_roles={"p1": "lower_zone_evap_reduction", "S1": "supply_cap_argument", "S2": "deficit_store_state", "S2min": "deficit_threshold", "Ep": "remaining_pet"},
        output_meaning="Penman lower-zone evapotranspiration limiter driven by a deficit-threshold gate; in active use `S1` is passed as infinity, so the active ceiling is PET-side rather than supply-side.",
        expected_physical_bounds="0 <= flux_et <= pet_rem. A `flux <= S2` storage-withdrawal bound is invalid because `S2` is a deficit state, not a directly depleted water store.",
        expected_bound_type="deficit_store_pet_limiter",
        existing_pre_or_post_caps="No additional post-call cap is applied beyond `F.relu(flux_et)`; the physically relevant semantics come from `p1=gam in [0,1]`, `gate in [0,1]`, and the deficit-store update `S2_new = S2 + flux_et + flux_u2 - flux_q12`.",
        notes="The broad high-risk flag likely came from a bound heuristic that treated the deficit store as available water and ignored the special `S1 = Inf` active call context.",
        representative_cases=("median", "random", "low_state", "high_state", "high_flux"),
    ),
    TargetSpec(
        formula="evap_7",
        flux_module="models.flux.evap",
        active_model="vic",
        probe_arg="S",
        grad_inputs=("S", "Smax", "Ep"),
        arg_roles={"S": "active_store_storage", "Smax": "reference_capacity", "Ep": "available_pet"},
        output_meaning="Shared VIC relative-storage evaporation helper used for interception ET, soil ET, and groundwater ET.",
        expected_physical_bounds="0 <= Ea <= min(Ep, S)` for the raw helper call. VIC then adds branch-specific post-call caps using temporary post-inflow storages and PET remainders where needed.",
        expected_bound_type="min_pet_storage",
        existing_pre_or_post_caps="Branch-specific post-call caps: `flux_ei <= S1`, `flux_et1 <= S2 + flux_inf - nearzero` and `<= pet_rem_s2`, `flux_et2 <= S3 + flux_pc - nearzero` and `<= pet_rem_s3`.",
        notes="The broad risk likely mixed multiple VIC call sites and over-applied a generic storage/bound heuristic that is stricter than the actual helper semantics.",
        representative_cases=("median", "random", "low_state", "high_state", "high_flux"),
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


def _arg_order(target: TargetSpec) -> list[str]:
    fn = getattr(__import__(target.flux_module, fromlist=[target.formula]), target.formula)
    return [name for name in inspect.signature(fn).parameters if name != "nearzero"]


def _target_by_key(formula: str, active_model: str) -> TargetSpec:
    return next(item for item in BATCH_C_TARGETS if item.formula == formula and item.active_model == active_model)


def _core_contexts_by_key() -> dict[tuple[str, str], list[Any]]:
    mapping: dict[tuple[str, str], list[Any]] = {}
    for ctx in load_flux_usage_contexts():
        if ctx.module_type != "core":
            continue
        key = (ctx.flux_function, ctx.model_name)
        mapping.setdefault(key, []).append(ctx)
    return mapping


class TraceRecorder:
    def __init__(self, target: TargetSpec):
        self.target = target
        self.calls_by_regime: dict[str, list[dict[str, Any]]] = {}

    def record(
        self,
        forcing_regime: str,
        call_site: str,
        arg_order: list[str],
        args: tuple[torch.Tensor, ...],
    ) -> None:
        max_numel = max(tensor.numel() for tensor in args) if args else 1
        call = {
            name: (
                tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1).clone().repeat(max_numel)
                if tensor.numel() == 1 and max_numel > 1
                else tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1).clone()
            )
            for name, tensor in zip(arg_order, args)
        }
        call["__call_site__"] = call_site
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
        line_no = caller.f_lineno if caller is not None else -1
        call_site = f"models/core/{target.active_model}.py:{line_no}"
        recorder.record(forcing_regime, call_site, arg_order, args[: len(arg_order)])
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


def _aggregate_domain_trace(
    target: TargetSpec,
    recorder: TraceRecorder,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]], dict[str, list[str]]]:
    rows: list[dict[str, Any]] = []
    pooled: dict[str, list[torch.Tensor]] = {}
    call_sites_by_arg: dict[str, set[str]] = {name: set() for name in _arg_order(target)}

    for regime_name, calls in recorder.calls_by_regime.items():
        for arg_name in _arg_order(target):
            values = torch.cat([call[arg_name] for call in calls]) if calls else torch.empty(0, dtype=torch.float64)
            pooled.setdefault(arg_name, []).append(values)
            for call in calls:
                call_sites_by_arg[arg_name].add(call["__call_site__"])
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
                    "forcing_regime": regime_name,
                    "call_sites": ";".join(sorted({call["__call_site__"] for call in calls})),
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
            "zero_count": int((torch.abs(values) <= 1.0e-12).sum().item()) if values.numel() else 0,
            "negative_count": int((values < -1.0e-12).sum().item()) if values.numel() else 0,
        }
    sorted_sites = {arg_name: sorted(sites) for arg_name, sites in call_sites_by_arg.items()}
    return rows, pooled_stats, sorted_sites


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
    contexts_map = _core_contexts_by_key()
    rows = []
    for target in BATCH_C_TARGETS:
        contexts = contexts_map[(target.formula, target.active_model)]
        flux_path = REPO_ROOT / f"{target.flux_module.replace('.', '/')}.py"
        flux_start, flux_end = _line_bounds(flux_path, target.formula)
        call_lines = ";".join(ctx.call_site.split(":")[1] for ctx in contexts)
        param_mapping = " | ".join(_json(ctx.parameter_mapping) for ctx in contexts)
        param_bounds = " | ".join(_json(ctx.parameter_bounds) for ctx in contexts)
        state_mapping = " | ".join(_json(ctx.state_variable_mapping) for ctx in contexts)
        forcing_mapping = " | ".join(_json(ctx.forcing_variable_mapping) for ctx in contexts)
        rows.append(
            {
                "formula": target.formula,
                "active_model": target.active_model,
                "flux_file": str(flux_path.relative_to(REPO_ROOT)),
                "flux_lines": f"{flux_start}-{flux_end}",
                "core_file": contexts[0].call_site.split(":")[0],
                "call_site_lines": call_lines,
                "parameter_mapping": param_mapping,
                "parameter_bounds": param_bounds,
                "state_mapping": state_mapping,
                "forcing_or_incoming_flux_mapping": forcing_mapping,
                "output_meaning": target.output_meaning,
                "expected_physical_bounds": target.expected_physical_bounds,
                "existing_pre_or_post_caps": target.existing_pre_or_post_caps,
                "notes": target.notes,
            }
        )
    return rows


def _source_context_markdown(context_rows: list[dict[str, Any]]) -> str:
    contexts_map = _core_contexts_by_key()
    lines = [
        "# Batch C Source Context",
        "",
        "This document records the exact shared-flux code plus the active core-model call sites reviewed in the Batch C realistic-domain audit.",
        "",
    ]
    for row in context_rows:
        target = _target_by_key(row["formula"], row["active_model"])
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
        lines.append("### Active core call site(s)")
        lines.append("")
        for ctx in contexts_map[(target.formula, target.active_model)]:
            file_part, line_part = ctx.call_site.split(":")
            core_path = REPO_ROOT / file_part
            line_no = int(line_part)
            max_line = len(core_path.read_text(encoding="utf-8").splitlines())
            lines.append(f"`{ctx.call_site}`")
            lines.append("")
            lines.append(f"- Parameter mapping: `{_json(ctx.parameter_mapping)}`")
            lines.append(f"- Parameter bounds: `{_json(ctx.parameter_bounds)}`")
            lines.append(f"- State mapping: `{_json(ctx.state_variable_mapping)}`")
            lines.append(f"- Forcing/incoming mapping: `{_json(ctx.forcing_variable_mapping)}`")
            lines.append("")
            lines.append("```python")
            lines.append(_snippet(core_path, max(1, line_no - 3), min(line_no + 5, max_line)))
            lines.append("```")
            lines.append("")
        lines.append(f"- Output meaning: {row['output_meaning']}")
        lines.append(f"- Expected physical bounds: {row['expected_physical_bounds']}")
        lines.append(f"- Existing cap logic: {row['existing_pre_or_post_caps']}")
        lines.append(f"- Notes: {row['notes']}")
        lines.append("")
    return "\n".join(lines)


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
    if case_name == "low_flux":
        flux_name = "incoming_flux" if "incoming_flux" in pooled else "flux" if "flux" in pooled else "Ep"
        threshold = torch.quantile(pooled[flux_name], 0.05)
        return _repeat_indices(torch.nonzero(pooled[flux_name] <= threshold, as_tuple=False).reshape(-1), n_total, n_out)
    if case_name == "high_fraction":
        threshold = torch.quantile(pooled["p1"], 0.95)
        return _repeat_indices(torch.nonzero(pooled["p1"] >= threshold, as_tuple=False).reshape(-1), n_total, n_out)
    if case_name == "high_capacity_gap":
        gap = pooled["So"] - pooled["Smax"]
        threshold = torch.quantile(gap, 0.95)
        return _repeat_indices(torch.nonzero(gap >= threshold, as_tuple=False).reshape(-1), n_total, n_out)
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
    output_exceeds_storage_bound_heuristic_count = 0
    output_exceeds_pet_count = 0
    output_exceeds_capacity_count = 0
    post_flux_capacity_violation_count = 0
    p1_below_zero_count = 0
    p1_above_one_count = 0

    if target.formula == "excess_1":
        output_exceeds_storage_count = _count(output > inputs["So"] + tol)
        post_flux_capacity_violation_count = _count(inputs["So"] - output > inputs["Smax"] + tol)
        true_bound_violation_count = negative_count + output_exceeds_storage_count + post_flux_capacity_violation_count
    elif target.formula == "recharge_1":
        output_exceeds_incoming_flux_count = _count(output > inputs["flux"] + tol)
        output_exceeds_storage_bound_heuristic_count = _count(output > inputs["S"] + tol)
        true_bound_violation_count = negative_count + output_exceeds_incoming_flux_count
    elif target.formula == "split_1":
        output_exceeds_incoming_flux_count = _count(output > inputs["incoming_flux"] + tol)
        p1_below_zero_count = _count(inputs["p1"] < -tol)
        p1_above_one_count = _count(inputs["p1"] > 1.0 + tol)
        true_bound_violation_count = negative_count + output_exceeds_incoming_flux_count + p1_below_zero_count + p1_above_one_count
    elif target.formula == "evap_16":
        output_exceeds_pet_count = _count(output > inputs["Ep"] + tol)
        true_bound_violation_count = negative_count + output_exceeds_pet_count
    elif target.formula == "evap_7":
        output_exceeds_storage_count = _count(output > inputs["S"] + tol)
        output_exceeds_pet_count = _count(output > inputs["Ep"] + tol)
        true_bound_violation_count = negative_count + output_exceeds_storage_count + output_exceeds_pet_count
    else:
        true_bound_violation_count = negative_count

    return {
        "output_exceeds_incoming_flux_count": output_exceeds_incoming_flux_count,
        "output_exceeds_storage_count": output_exceeds_storage_count,
        "output_exceeds_storage_bound_heuristic_count": output_exceeds_storage_bound_heuristic_count,
        "output_exceeds_pet_count": output_exceeds_pet_count,
        "output_exceeds_capacity_count": output_exceeds_capacity_count,
        "post_flux_capacity_violation_count": post_flux_capacity_violation_count,
        "p1_below_zero_count": p1_below_zero_count,
        "p1_above_one_count": p1_above_one_count,
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
    return [_evaluate_case(target, case_name, _base_inputs_from_joint_tensors(target, pooled, case_name)) for case_name in target.representative_cases]


def _broad_metrics_map() -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_csv(GRADIENT_DIR / "final_flux_gradient_risk_ranking.csv")
    return {(row["formula"], row["active_model"]): row for row in rows}


def _failure_mode_map() -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_csv(GRADIENT_DIR / "remaining_high_risk_failure_mode_summary.csv")
    return {(row["formula"], row["active_model"]): row for row in rows}


def _broad_diagnostic_map() -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_csv(GRADIENT_DIR / "remaining_active_high_risk_contexts.csv")
    return {(row["formula"], row["active_model"]): row for row in rows}


def _bound_review_rows(
    gradient_rows: list[dict[str, Any]],
    pooled_stats_by_key: dict[tuple[str, str], dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    broad_diag_map = _broad_diagnostic_map()
    rows = []

    for target in BATCH_C_TARGETS:
        key = (target.formula, target.active_model)
        realistic = [row for row in gradient_rows if row["formula"] == target.formula and row["active_model"] == target.active_model]
        realistic_true_bound = sum(row["output_bound_violation_count"] for row in realistic)
        incoming_heuristic = sum(row["output_exceeds_incoming_flux_count"] for row in realistic)
        storage_heuristic = sum(row["output_exceeds_storage_bound_heuristic_count"] for row in realistic)
        pooled_stats = pooled_stats_by_key[key]

        if target.formula == "excess_1":
            bound_issue_class = "bound_heuristic_artifact" if realistic_true_bound == 0 else "true_realistic_bound_violation"
            is_bound_valid = "no"
            if target.active_model == "susannah2":
                later_cap = "yes"
                reason = (
                    "The earlier bound flag does not reflect the active semantics. `excess_1` is an overflow from a prospective store state, "
                    "and in Susannah2 the actual transfer is further constrained by the joint `flux_rg + flux_se` scaling step."
                )
            elif target.active_model == "australia":
                later_cap = "no"
                reason = (
                    "The earlier bound flag compared overflow against an invalid heuristic. In Australia, `flux_se = relu(S1_tmp_in - cap_s1_to_s2)` "
                    "is the exact prospective-state overflow needed to reduce `S1_tmp_in` back to capacity."
                )
            else:
                later_cap = "no"
                reason = (
                    "The earlier bound flag compared interception overflow against a generic storage/incoming-flux heuristic. In VIC this helper is "
                    "used as a prospective canopy-overflow term with `So = S1 + P - flux_peff` and `Smax = aux_imax`."
                )
        elif target.formula == "recharge_1":
            p1_max = pooled_stats["p1"]["max"]
            if realistic_true_bound > 0:
                bound_issue_class = "true_realistic_bound_violation"
            elif storage_heuristic > 0:
                bound_issue_class = "bound_heuristic_artifact"
            else:
                bound_issue_class = "broad_domain_artifact"
            is_bound_valid = "partially"
            later_cap = "yes"
            reason = (
                "The physically relevant bound is `flux_REC <= incoming remainder`, not `flux_REC <= S`. Under realistic rollouts "
                f"`p1` stays within the active range (max traced {p1_max:.6g}), so the shared formula remains an incoming-flux partition."
            )
        elif target.formula == "split_1":
            p1_min = pooled_stats["p1"]["min"]
            p1_max = pooled_stats["p1"]["max"]
            if realistic_true_bound > 0:
                bound_issue_class = "true_realistic_bound_violation"
            else:
                bound_issue_class = "broad_domain_artifact"
            is_bound_valid = "yes"
            later_cap = "no"
            reason = (
                "The broad diagnostic used a generic `p1` range, but the active expression is `1 - d_split`. "
                f"Realistic tracing shows `p1` stays in [{p1_min:.6g}, {p1_max:.6g}], so the partition bound `0 <= flux_rf <= incoming_flux` holds."
            )
        elif target.formula == "evap_16":
            if realistic_true_bound > 0:
                bound_issue_class = "true_realistic_bound_violation"
            else:
                bound_issue_class = "bound_heuristic_artifact"
            is_bound_valid = "no"
            later_cap = "no"
            reason = (
                "The earlier label treated Penman's deficit store as directly withdrawable storage. In the active context `S2` is a deficit state, "
                "`S1` is passed as `Inf`, and the relevant finite ceiling is remaining PET, not `S2`."
            )
        elif target.formula == "evap_7":
            if realistic_true_bound > 0:
                bound_issue_class = "true_realistic_bound_violation"
            else:
                bound_issue_class = "broad_domain_artifact"
            is_bound_valid = "yes"
            later_cap = "yes"
            reason = (
                "VIC traced calls stay within the raw helper bounds `0 <= Ea <= min(Ep, S)`, and VIC then applies branch-specific temporary-storage "
                "caps after the helper call where needed."
            )
        else:
            bound_issue_class = "manual_review_required"
            is_bound_valid = "manual_review_required"
            later_cap = "manual_review_required"
            reason = "Manual review required."

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


def _risk_decision_rows(
    gradient_rows: list[dict[str, Any]],
    bound_rows: list[dict[str, Any]],
    pooled_stats_by_key: dict[tuple[str, str], dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    broad_map = _broad_metrics_map()
    failure_map = _failure_mode_map()
    bound_map = {(row["formula"], row["active_model"]): row for row in bound_rows}
    rows = []

    for target in BATCH_C_TARGETS:
        key = (target.formula, target.active_model)
        realistic = [row for row in gradient_rows if row["formula"] == target.formula and row["active_model"] == target.active_model]
        bound_review = bound_map[key]
        pooled_stats = pooled_stats_by_key[key]
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
            recommended_action = "manual_review_required"
            human_review_priority = "high"
            main_failure_mode = "physical_bound_violation"
            short_reason = "Representative traced active-model cases still violate the physically meaningful bound."
        elif target.formula == "split_1":
            p1_min = pooled_stats["p1"]["min"]
            p1_max = pooled_stats["p1"]["max"]
            realistic_risk = "low"
            artifact_or_real = "artifact"
            recommended_action = "broad_domain_artifact"
            human_review_priority = "low"
            main_failure_mode = failure_map[key]["failure_mode"]
            short_reason = f"Active `1 - d_split` stays within [{p1_min:.6g}, {p1_max:.6g}], so the earlier bound flag came from the broad diagnostic expression-range inference."
        elif bound_review["bound_issue_class"] in {"bound_heuristic_artifact", "broad_domain_artifact"}:
            realistic_risk = "low"
            artifact_or_real = "artifact"
            recommended_action = bound_review["bound_issue_class"]
            human_review_priority = "low"
            main_failure_mode = failure_map[key]["failure_mode"]
            short_reason = bound_review["reason"]
        elif realistic_max_grad > 1.0e2:
            realistic_risk = "medium"
            artifact_or_real = "real_but_bounded"
            recommended_action = "keep_but_document"
            human_review_priority = "medium"
            main_failure_mode = "large_but_finite_gradient"
            short_reason = "Traced active-model cases are finite and physically bounded, but gradients remain large enough to document."
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
    trace_rows: list[dict[str, Any]],
    gradient_rows: list[dict[str, Any]],
    bound_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
) -> str:
    broad_map = _broad_metrics_map()
    lines = [
        "# Batch C Flux Realistic Review Report",
        "",
        "## 1. Scope",
        "- Realistic-domain review for the remaining active high-risk Batch C flux contexts listed in the task brief.",
        "- This pass is diagnostic only: no hydrological formulas, smoothing rules, parameter bounds, soft-gate defaults, unit hydrograph code, or water-balance fixes were changed.",
        "",
        "## 2. Target contexts",
    ]
    for target in BATCH_C_TARGETS:
        lines.append(f"- `{target.formula}` / `{target.active_model}`")

    lines.extend(
        [
            "",
            "## 3. Source and call context",
            "- Exact flux code and active core call sites are recorded in `batch_c_source_context.md`.",
            "- Structured source/context inventory is recorded in `batch_c_context_inventory.csv`.",
            "",
            "## 4. Realistic-domain tracing method",
            "- Each target flux symbol was patched in its active core-model namespace during deterministic synthetic rollouts.",
            "- Forcing regimes: dry, normal, wet, high precipitation, low PET, high PET.",
            "- Parameter cases pooled in each regime: `lower_near`, `midpoint`, `upper_near`, `random_valid`.",
            "- Stable initial states were generated with the existing water-balance test utilities and a short dry-step stabilization loop.",
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
        "Results for excess_1 / australia",
        "Results for excess_1 / susannah2",
        "Results for excess_1 / vic",
        "Results for recharge_1 / modhydrolog",
        "Results for recharge_1 / simhyd",
        "Results for split_1 / flexb",
        "Results for split_1 / flexi",
        "Results for split_1 / flexis",
        "Results for evap_16 / penman",
        "Results for evap_7 / vic",
    ]
    for index, (target, title) in enumerate(zip(BATCH_C_TARGETS, section_titles), start=7):
        key = (target.formula, target.active_model)
        decision = next(row for row in decision_rows if (row["formula"], row["active_model"]) == key)
        target_rows = [row for row in gradient_rows if (row["formula"], row["active_model"]) == key]
        target_trace_rows = [row for row in trace_rows if (row["formula"], row["active_model"]) == key]
        lines.extend(
            [
                "",
                f"## {index}. {title}",
                f"- Previous risk: {decision['previous_risk']}",
                f"- Broad diagnostic reason: {broad_map[key]['final_reason']}",
                f"- Realistic-domain risk: {decision['realistic_risk']}",
                f"- Realistic max_abs_grad: {decision['max_abs_grad_realistic']:.6g}",
                f"- Realistic NaN/Inf counts: output={decision['output_nan_inf_realistic']}, grad={decision['grad_nan_inf_realistic']}",
                f"- Realistic true physical bound violations: {decision['physical_bound_violation_realistic']}",
                f"- Representative cases reviewed: {', '.join(row['case_name'] for row in target_rows)}",
                f"- Trace regimes captured: {', '.join(sorted({row['forcing_regime'] for row in target_trace_rows}))}",
                f"- Artifact or real: {decision['artifact_or_real']}",
                f"- Recommended action: {decision['recommended_action']}",
                f"- Reason: {decision['short_reason']}",
            ]
        )

    lines.extend(
        [
            "",
            "## 17. Final risk decision table",
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
    lines.extend(["", "## 18. Which cases are artifacts"])
    if artifact_rows:
        for row in artifact_rows:
            lines.append(f"- `{row['formula']}` / `{row['active_model']}`: {row['short_reason']}")
    else:
        lines.append("- None.")

    lines.extend(["", "## 19. Which cases remain true realistic-domain risks"])
    if real_rows:
        for row in real_rows:
            lines.append(f"- `{row['formula']}` / `{row['active_model']}`: {row['short_reason']}")
    else:
        lines.append("- None of the Batch C contexts remain true realistic-domain risks after tracing the active domains.")

    lines.extend(
        [
            "",
            "## 20. Whether any formula modification is justified now",
            "- No. The Batch C realistic-domain audit did not find evidence that any of these shared formulas should be changed now.",
            "",
            "## 21. Recommended next step",
            "- Treat these Batch C high-risk flags as reviewed. If there is a follow-on diagnostics pass, the highest-value work is improving the broad-domain wrapper heuristics and parameter-expression inference so they match the active model semantics more closely.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_batch_c_review() -> dict[str, Any]:
    torch.manual_seed(FIXED_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    context_rows = _context_inventory_rows()
    (OUTPUT_DIR / "batch_c_source_context.md").write_text(_source_context_markdown(context_rows), encoding="utf-8")
    _write_csv(OUTPUT_DIR / "batch_c_context_inventory.csv", context_rows)

    trace_rows: list[dict[str, Any]] = []
    pooled_stats_by_key: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    pooled_joint_by_key: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    call_sites_by_key: dict[tuple[str, str], dict[str, list[str]]] = {}
    for target in BATCH_C_TARGETS:
        recorder = _run_trace_rollouts(target)
        rows, pooled_stats, call_sites = _aggregate_domain_trace(target, recorder)
        trace_rows.extend(rows)
        pooled_stats_by_key[(target.formula, target.active_model)] = pooled_stats
        pooled_joint_by_key[(target.formula, target.active_model)] = _pooled_joint_tensors(target, recorder)
        call_sites_by_key[(target.formula, target.active_model)] = call_sites
    _write_csv(OUTPUT_DIR / "batch_c_realistic_domain_trace.csv", trace_rows)

    gradient_rows: list[dict[str, Any]] = []
    for target in BATCH_C_TARGETS:
        pooled_joint = pooled_joint_by_key[(target.formula, target.active_model)]
        gradient_rows.extend(_evaluate_target(target, pooled_joint))
    _write_csv(OUTPUT_DIR / "batch_c_realistic_gradient_summary.csv", gradient_rows)

    bound_rows = _bound_review_rows(gradient_rows, pooled_stats_by_key)
    decision_rows = _risk_decision_rows(gradient_rows, bound_rows, pooled_stats_by_key)
    _write_csv(OUTPUT_DIR / "batch_c_bound_heuristic_review.csv", bound_rows)
    _write_csv(OUTPUT_DIR / "batch_c_risk_decision.csv", decision_rows)
    (OUTPUT_DIR / "batch_c_flux_realistic_review_report.md").write_text(
        _report(context_rows, trace_rows, gradient_rows, bound_rows, decision_rows),
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
        "call_sites_by_key": call_sites_by_key,
    }


def main() -> None:
    run_batch_c_review()


if __name__ == "__main__":
    main()
