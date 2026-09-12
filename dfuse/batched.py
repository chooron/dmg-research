"""GPU-batched execution for one structure across multiple catchments.

Structure choices remain literals in one generated scalar step.  ``torch.vmap``
adds only the leading basin dimension, so the time loop remains the single Python
loop shared by a batch and no catchment loop runs inside a timestep.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .kernel import (
    COUPLED_RK2_DIAGNOSTIC_NAMES,
    FLUX_NAMES,
    _capacity,
    _day_of_year,
    _initial_state,
    _parameter_values,
    _parameter_vector,
    _routing_fractions,
    _sequential_union_state,
    _structure_context,
    _topographic_mean,
)
from .runtime import get_generated_step
from .spec import PARAMETER_NAMES, STATE_NAMES, StructureSpec, get_structure

WATER_BALANCE_TOLERANCE = 1.0e-8
SNOW_BALANCE_TOLERANCE = 1.0e-8
CAPACITY_TOLERANCE = 1.0e-12
_STATE_CAPACITY = {
    "TENS_1A": "MAXTENS_1A",
    "TENS_1B": "MAXTENS_1B",
    "TENS_1": "MAXTENS_1",
    "FREE_1": "MAXFREE_1",
    "WATR_1": "MAXWATR_1",
    "TENS_2": "MAXTENS_2",
    "FREE_2A": "MAXFREE_2A",
    "FREE_2B": "MAXFREE_2B",
    "WATR_2": "MAXWATR_2",
}
_FAILURE_BITS = {"finite": 1, "negative_state": 2, "capacity": 4, "water_balance": 8, "snow_balance": 16}


@dataclass
class BatchedSimulationResult:
    model_id: int
    basin_ids: tuple[str, ...]
    q: Tensor
    q_instantaneous: Tensor | None
    states: Tensor
    state_names: tuple[str, ...]
    fluxes: dict[str, Tensor]
    water_balance_residual: Tensor
    snow: Tensor | None
    snow_balance_residual: Tensor
    sequential_diagnostics: dict[str, Tensor]
    monitoring: dict[str, Any]
    output_mode: str = "full"

    @property
    def final_states(self) -> Tensor:
        """Final active state for both output modes.

        Full mode stores ``[B, T + 1, S]``; Lite stores only ``[B, 1, S]``.
        This property gives callers a shape-stable ``[B, S]`` final state.
        """
        return self.states[:, -1]
    @property
    def max_abs_water_balance_error(self) -> Tensor:
        return self.water_balance_residual.abs().amax()


_BATCH_COMPILED: dict[tuple[Any, ...], tuple[Any, dict[str, Any]]] = {}


def _counter_snapshot() -> dict[str, dict[str, int]]:
    try:
        from torch._dynamo.utils import counters
    except Exception:
        return {}
    return {
        str(group): {str(name): int(value) for name, value in counter.items()}
        for group, counter in counters.items()
        if group in {"graph_break", "stats", "inductor", "aot_autograd", "guard_failures"} and counter
    }


def _counter_delta(before: Mapping[str, Mapping[str, int]], after: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for group in set(before) | set(after):
        values = {}
        for name in set(before.get(group, {})) | set(after.get(group, {})):
            delta = int(after.get(group, {}).get(name, 0)) - int(before.get(group, {}).get(name, 0))
            if delta:
                values[name] = delta
        if values:
            result[group] = values
    return result


class _AuditedBatchedStep:
    def __init__(self, compiled: Any, record: dict[str, Any]) -> None:
        self.compiled = compiled
        self.record = record
        self.record["counter_baseline"] = _counter_snapshot()

    def __call__(self, *args: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        self.record["calls"] += 1
        started = time.perf_counter() if not self.record["warmed"] else None
        try:
            result = self.compiled(*args)
        except Exception as exc:
            self.record["fallbacks"] = 1
            self.record["failures"].append(f"{type(exc).__name__}: {str(exc)[:500]}")
            raise
        if started is not None:
            self.record["warmed"] = True
            self.record["compile_successes"] = 1
            self.record["cold_compile_seconds"] = [time.perf_counter() - started]
        after = _counter_snapshot()
        delta = _counter_delta(self.record["counter_baseline"], after)
        self.record["counter_baseline"] = after
        for group, values in delta.items():
            accumulated = self.record["dynamo_counters"].setdefault(group, {})
            for name, value in values.items():
                accumulated[name] = int(accumulated.get(name, 0)) + int(value)
        self.record["graph_breaks"] += sum(delta.get("graph_break", {}).values())
        self.record["forward_unique_graphs"] = int(self.record["dynamo_counters"].get("stats", {}).get("unique_graphs", 0))
        self.record["recompilations"] = max(self.record["forward_unique_graphs"] - 1, 0)
        return result


def reset_batched_compile_diagnostics() -> None:
    _BATCH_COMPILED.clear()


def batched_compile_diagnostics() -> dict[str, Any]:
    records = {}
    for key, (_, record) in _BATCH_COMPILED.items():
        records[str(key)] = {name: value for name, value in record.items() if name not in {"counter_baseline", "warmed"}}
    return {"registry_size": len(_BATCH_COMPILED), "records": records}


def _compiled_step(spec: StructureSpec, *, batch_size: int, device: torch.device, dtype: torch.dtype, backend: str, fullgraph: bool) -> _AuditedBatchedStep:
    signature, generated = get_generated_step(spec, order=("coupled_rhs",), n_substeps=1, execution_mode="coupled_rk2")
    key = (signature.digest, str(device), str(dtype), int(batch_size), backend, bool(fullgraph))
    cached = _BATCH_COMPILED.get(key)
    if cached is not None:
        return cached[0]
    record: dict[str, Any] = {
        "model_id": spec.model_id,
        "graph_signature_digest": signature.digest,
        "batch_size": batch_size,
        "backend": backend,
        "fullgraph": bool(fullgraph),
        "compile_attempts": 1,
        "compile_successes": 0,
        "calls": 0,
        "warmed": False,
        "fallbacks": 0,
        "graph_breaks": 0,
        "recompilations": 0,
        "forward_unique_graphs": 0,
        "cold_compile_seconds": [],
        "failures": [],
        "dynamo_counters": {},
        "graph_signature": signature.to_dict(),
    }
    mapped = torch.vmap(generated, in_dims=(0, 0, 0, 0, 0, 0), out_dims=(0, 0, 0))
    compiled = torch.compile(mapped, backend=backend, fullgraph=bool(fullgraph))
    entry = _AuditedBatchedStep(compiled, record)
    _BATCH_COMPILED[key] = (entry, record)
    return entry


def _forcing_batch(forcing: Tensor | Mapping[str, object], *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if isinstance(forcing, Mapping):
        values = [torch.as_tensor(forcing[name], device=device, dtype=dtype) for name in ("ppt", "pet", "temp")]
        if any(value.ndim != 2 for value in values):
            raise ValueError("batched forcing mapping values must have shape [basin, time]")
        if any(value.shape != values[0].shape for value in values[1:]):
            raise ValueError("batched forcing columns must have equal shapes")
        return torch.stack(values, dim=-1)
    tensor = torch.as_tensor(forcing, device=device, dtype=dtype)
    if tensor.ndim != 3 or tensor.shape[-1] != 3:
        raise ValueError("batched forcing must have shape [basin, time, 3]")
    return tensor


def _theta_batch(params: Sequence[Mapping[str, object]] | Tensor, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, list[dict[str, Tensor]]]:
    if isinstance(params, Tensor):
        theta = params.to(device=device, dtype=dtype)
        if theta.ndim != 2 or tuple(theta.shape) != (batch_size, len(PARAMETER_NAMES)):
            raise ValueError(f"batched parameter tensor must have shape [{batch_size}, {len(PARAMETER_NAMES)}]")
        maps = [{name: theta[index, position] for position, name in enumerate(PARAMETER_NAMES)} for index in range(batch_size)]
        return theta, maps
    if len(params) != batch_size:
        raise ValueError("one parameter mapping is required for each basin")
    maps = [_parameter_values(item, dtype=dtype, device=device) for item in params]
    return torch.stack([_parameter_vector(item) for item in maps], dim=0), maps


def _dates_for_batch(n_steps: int, dates: Sequence[Any] | Tensor | None, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    if dates is None:
        return _day_of_year(n_steps, None, dtype=dtype, device=device)
    return _day_of_year(n_steps, dates, dtype=dtype, device=device)


def _initial_batch(spec: StructureSpec, parameter_maps: Sequence[Mapping[str, Tensor]], *, fraction: float) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    states = []
    means = []
    maxima = []
    fractions = []
    capacities = []
    context = _structure_context(spec, dtype=next(iter(parameter_maps[0].values())).dtype, device=next(iter(parameter_maps[0].values())).device)
    for params in parameter_maps:
        cap = _capacity(params)
        states.append(_sequential_union_state(_initial_state(spec, params, cap, fraction), spec))
        capacities.append(torch.stack([cap[name] for name in ("MAXTENS_1A", "MAXTENS_1B", "MAXTENS_1", "MAXFREE_1", "MAXWATR_1", "MAXTENS_2", "MAXFREE_2A", "MAXFREE_2B", "MAXWATR_2")]))
        if spec.decisions["QSURF"] == "tmdl_param" or spec.decisions["ARCH2"] == "unlimpow_2":
            mean, maximum = _topographic_mean(params)
        else:
            zero = params["RFERR_MLT"] * 0.0
            mean, maximum = zero, zero
        means.append(mean)
        maxima.append(maximum)
        fractions.append(_routing_fractions(params["TIMEDELAY"], dtype=params["TIMEDELAY"].dtype, device=params["TIMEDELAY"].device))
    return torch.stack(states), torch.stack(means), torch.stack(maxima), torch.stack(fractions), torch.stack(capacities)


def _monitor_code(packed: Tensor, routed: Tensor, diagnostics: Tensor, capacities: Tensor, bounded: Tensor) -> Tensor:
    finite = torch.isfinite(packed).all(dim=1) & torch.isfinite(routed) & torch.isfinite(diagnostics).all(dim=1)
    negative = (packed[:, : len(STATE_NAMES)] < 0.0).any(dim=1)
    capacity = ((packed[:, : len(STATE_NAMES)] - capacities) > CAPACITY_TOLERANCE).logical_and(bounded).any(dim=1)
    water = diagnostics[:, len(FLUX_NAMES)].abs() > WATER_BALANCE_TOLERANCE
    snow = diagnostics[:, len(FLUX_NAMES) + 1].abs() > SNOW_BALANCE_TOLERANCE
    return (~finite).to(torch.int64) * _FAILURE_BITS["finite"] + negative.to(torch.int64) * _FAILURE_BITS["negative_state"] + capacity.to(torch.int64) * _FAILURE_BITS["capacity"] + water.to(torch.int64) * _FAILURE_BITS["water_balance"] + snow.to(torch.int64) * _FAILURE_BITS["snow_balance"]


def _category(code: int) -> str | None:
    for name, bit in _FAILURE_BITS.items():
        if code & bit:
            return name
    return None

def simulate_coupled_rk2_batched(
    model_id: int | str,
    forcing: Tensor | Mapping[str, object],
    params: Sequence[Mapping[str, object]] | Tensor,
    *,
    basin_ids: Sequence[str] | None = None,
    initial_fraction: float = 0.25,
    dates: Sequence[Any] | Tensor | None = None,
    dt_days: float = 1.0,
    compile_step: bool = True,
    compile_backend: str = "inductor",
    compile_fullgraph: bool = True,
    monitor_chunk_size: int = 64,
    output_mode: str = "full",
 ) -> BatchedSimulationResult:
    """Run one structure for a basin batch on the leading GPU dimension.

    ``output_mode="full"`` retains the complete diagnostic contract: active
    states, accepted fluxes, routed and instantaneous Q, balance residuals,
    snow history, and coupled-RK2 diagnostics.  ``output_mode="lite"`` runs
    the identical RK2/FIX_STATES recurrence and retains only routed Q plus the
    final active state; optional diagnostic fields are ``None``/empty.  ``q_only``
    remains an alias for ``lite`` for callers from the frozen v1 execution layer;
    new code should use ``lite``.
    """
    if dt_days <= 0.0:
        raise ValueError("dt_days must be positive")
    if output_mode == "q_only":
        # Backward-compatible spelling used by the frozen execution layer.
        output_mode = "lite"
    if output_mode not in {"full", "lite"}:
        raise ValueError("output_mode must be either 'full' or 'lite'")
    if not isinstance(monitor_chunk_size, int) or isinstance(monitor_chunk_size, bool) or monitor_chunk_size < 1:
        raise ValueError("monitor_chunk_size must be a positive integer")
    spec = get_structure(model_id)
    device = forcing.device if isinstance(forcing, Tensor) else next((value.device for value in forcing.values() if isinstance(value, Tensor)), torch.device("cpu"))
    dtype = forcing.dtype if isinstance(forcing, Tensor) and forcing.is_floating_point() else torch.float64
    forcing_tensor = _forcing_batch(forcing, device=device, dtype=dtype)
    batch_size, n_steps, _ = forcing_tensor.shape
    if batch_size < 1 or n_steps < 1:
        raise ValueError("batched forcing must contain at least one basin and one timestep")
    theta, parameter_maps = _theta_batch(params, batch_size=batch_size, device=device, dtype=dtype)
    initial_states, topo_mean, topo_max, fractions, capacities = _initial_batch(spec, parameter_maps, fraction=initial_fraction)
    bounded = torch.ones_like(capacities, dtype=torch.bool)
    if spec.decisions["ARCH2"] != "fixedsiz_2":
        bounded[:, STATE_NAMES.index("WATR_2")] = False
    days, leap_years = _dates_for_batch(n_steps, dates, device=device, dtype=dtype)
    packed = torch.cat((initial_states, torch.zeros((batch_size, 1 + 500), device=device, dtype=dtype)), dim=1)
    active_indices = torch.as_tensor([STATE_NAMES.index(name) for name in spec.state_names], device=device, dtype=torch.long)
    is_full = output_mode == "full"
    states = [packed[:, active_indices]] if is_full else []
    flux_history: dict[str, list[Tensor]] = {name: [] for name in FLUX_NAMES} if is_full else {}
    q_history: list[Tensor] = []
    q_instantaneous_history: list[Tensor] = [] if is_full else []
    balance_history: list[Tensor] = [] if is_full else []
    snow_balance_history: list[Tensor] = [] if is_full else []
    snow_history = [packed[:, len(STATE_NAMES)]] if is_full else []
    coupled_history: dict[str, list[Tensor]] = {name: [] for name in COUPLED_RK2_DIAGNOSTIC_NAMES} if is_full else {}
    first_failure = torch.full((batch_size,), -1, dtype=torch.int64, device=device)
    first_code = torch.zeros((batch_size,), dtype=torch.int64, device=device)
    negative_counts = torch.zeros((batch_size,), dtype=torch.int64, device=device)
    capacity_counts = torch.zeros((batch_size,), dtype=torch.int64, device=device)
    nonfinite_counts = torch.zeros((batch_size,), dtype=torch.int64, device=device)
    max_water = torch.zeros((batch_size,), dtype=dtype, device=device)
    max_snow = torch.zeros((batch_size,), dtype=dtype, device=device)
    if compile_step:
        step_callable = _compiled_step(spec, batch_size=batch_size, device=device, dtype=dtype, backend=compile_backend, fullgraph=compile_fullgraph)
    else:
        _, generated = get_generated_step(spec, order=("coupled_rhs",), n_substeps=1, execution_mode="coupled_rk2")
        step_callable = torch.vmap(generated, in_dims=(0, 0, 0, 0, 0, 0), out_dims=(0, 0, 0))
    stopped = False
    for index in range(n_steps):
        step_forcing = torch.cat((forcing_tensor[:, index, :], days[index].expand(batch_size, 1), leap_years[index].to(dtype).expand(batch_size, 1), torch.full((batch_size, 1), dt_days, dtype=dtype, device=device)), dim=1)
        packed, routed, diagnostics = step_callable(packed, step_forcing, theta, topo_mean, topo_max, fractions)
        flux = diagnostics[:, : len(FLUX_NAMES)]
        q_history.append(routed)
        if is_full:
            q_instantaneous_history.append(diagnostics[:, len(FLUX_NAMES) + 2])
            balance_history.append(diagnostics[:, len(FLUX_NAMES)])
            snow_balance_history.append(diagnostics[:, len(FLUX_NAMES) + 1])
            for flux_index, name in enumerate(FLUX_NAMES):
                flux_history[name].append(flux[:, flux_index])
            for diagnostic_index, name in enumerate(COUPLED_RK2_DIAGNOSTIC_NAMES):
                coupled_history[name].append(diagnostics[:, len(FLUX_NAMES) + diagnostic_index])
            states.append(packed[:, active_indices])
            snow_history.append(packed[:, len(STATE_NAMES)])
        code = _monitor_code(packed, routed, diagnostics, capacities, bounded)
        new_failure = (first_failure < 0) & (code > 0)
        first_failure = torch.where(new_failure, torch.full_like(first_failure, index), first_failure)
        first_code = torch.where(new_failure, code, first_code)
        negative_counts += (packed[:, : len(STATE_NAMES)] < 0.0).any(dim=1).to(torch.int64)
        capacity_counts += (((packed[:, : len(STATE_NAMES)] - capacities) > CAPACITY_TOLERANCE).logical_and(bounded)).sum(dim=1).to(torch.int64)
        nonfinite_counts += (~(torch.isfinite(packed).all(dim=1) & torch.isfinite(routed) & torch.isfinite(diagnostics).all(dim=1))).to(torch.int64)
        max_water = torch.maximum(max_water, diagnostics[:, len(FLUX_NAMES)].detach().abs())
        max_snow = torch.maximum(max_snow, diagnostics[:, len(FLUX_NAMES) + 1].detach().abs())
        if (index + 1) % monitor_chunk_size == 0 or index == n_steps - 1:
            if bool((first_failure >= 0).any().item()):
                stopped = True
                break
    actual_steps = len(q_history)
    q = torch.stack(q_history, dim=1)
    if is_full:
        q_instantaneous = torch.stack(q_instantaneous_history, dim=1)
        state_tensor = torch.stack(states, dim=1)
        fluxes = {name: torch.stack(values, dim=1) for name, values in flux_history.items()}
        balance = torch.stack(balance_history, dim=1)
        snow_balance = torch.stack(snow_balance_history, dim=1)
        snow = torch.stack(snow_history, dim=1)
        coupled = {name: torch.stack(values, dim=1) for name, values in coupled_history.items()}
    else:
        q_instantaneous = None
        state_tensor = packed[:, active_indices].unsqueeze(1)
        fluxes = {}
        balance = torch.zeros((batch_size, 0), dtype=dtype, device=device)
        snow = None
        snow_balance = torch.zeros((batch_size, 0), dtype=dtype, device=device)
        coupled = {}
    ids = tuple(str(value) for value in basin_ids) if basin_ids is not None else tuple(str(index) for index in range(batch_size))
    if len(ids) != batch_size:
        raise ValueError("basin_ids length must equal forcing batch size")
    first_indices = first_failure.detach().cpu().tolist()
    first_codes = first_code.detach().cpu().tolist()
    per_basin = []
    for basin_index, basin_id in enumerate(ids):
        failure_index = int(first_indices[basin_index])
        item: dict[str, Any] = {"basin_id": basin_id, "first_failure_index": None if failure_index < 0 else failure_index, "first_failure_category": _category(int(first_codes[basin_index]))}
        if failure_index >= 0 and is_full:
            item["failure_trace"] = {"state_start": state_tensor[basin_index, failure_index].detach().cpu().tolist(), "state_next": state_tensor[basin_index, failure_index + 1].detach().cpu().tolist(), "accepted_flux": torch.stack([fluxes[name][basin_index, failure_index] for name in FLUX_NAMES]).detach().cpu().tolist(), "water_balance_residual": float(balance[basin_index, failure_index].detach().cpu()), "snow_balance_residual": float(snow_balance[basin_index, failure_index].detach().cpu())}
        per_basin.append(item)
    monitoring = {"completed_full_period": not stopped and actual_steps == n_steps, "actual_steps": actual_steps, "monitor_chunk_size": monitor_chunk_size, "stopped_on_failure": stopped, "per_basin": per_basin, "nonfinite_count": nonfinite_counts.detach().cpu().tolist(), "negative_active_state_count": negative_counts.detach().cpu().tolist(), "capacity_violation_count": capacity_counts.detach().cpu().tolist(), "water_balance_max_abs": max_water.detach().cpu().tolist(), "snow_balance_max_abs": max_snow.detach().cpu().tolist()}
    return BatchedSimulationResult(spec.model_id, ids, q, q_instantaneous, state_tensor, spec.state_names, fluxes, balance, snow, snow_balance, coupled, monitoring, output_mode)
