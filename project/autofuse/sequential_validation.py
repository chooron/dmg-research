"""Bounded validation of compiled FUSE process-sequential time steps.

This is an experiment harness, not a calibration or training driver.  The
selection set is loaded from the locked JSON artifact before any candidate
scores are computed.  Each order/substep variant is compiled in isolation so
validation does not retain 24 large Inductor graphs simultaneously.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from dfuse import (
    SEQUENTIAL_ORDERS,
    compile_diagnostics,
    enumerate_structures,
    reset_compile_diagnostics,
    simulate,
    simulate_sequential,
)
from dfuse.kernel import (
    _capacity,
    _compiled_flux_rhs,
    _sequential_raw_process,
    _sequential_union_state,
    _SEQUENTIAL_FLUX_MASKS,
    _day_of_year,
    _initial_state,
    _parameter_values,
    _parameter_vector,
    _structure_context,
    _topographic_mean,
)
from dfuse.spec import FLUX_NAMES, STATE_NAMES, default_parameters, get_structure

from .fidelity import _as_tensor_forcing, synthetic_forcing
from .metrics import kgecomp
from .reference_oracle import ReferenceResult, run_reference


ORDERS = tuple(SEQUENTIAL_ORDERS)
N_SUBSTEPS = (1, 2, 4, 8, 12, 24)
PROCESS_FLUXES = {
    "ET": ("EVAP_1", "EVAP_2"),
    "qsurf": ("QSURF",),
    "qperc": ("QPERC_12",),
    "qintf": ("QINTF_1",),
    "qbase": ("QBASE_2",),
}


def _number(value: Any) -> float | None:
    result = float(value.detach() if isinstance(value, torch.Tensor) else value)
    return result if math.isfinite(result) else None


def _max_abs(value: torch.Tensor) -> float | None:
    return _number(value.detach().abs().amax())


def _rmse(value: torch.Tensor) -> float | None:
    return _number(torch.sqrt(torch.mean(value.detach() * value.detach())))


def _p95(value: torch.Tensor) -> float | None:
    return _number(torch.quantile(value.detach().abs().reshape(-1), 0.95))


def _kgecomp_value(left: torch.Tensor, right: torch.Tensor) -> float | None:
    try:
        return _number(kgecomp(left, right).detach())
    except (RuntimeError, ValueError, FloatingPointError):
        return None


def _finite(result) -> bool:
    return bool(
        torch.isfinite(result.q).all()
        and torch.isfinite(result.states).all()
        and torch.isfinite(result.water_balance_residual).all()
        and torch.isfinite(result.snow_balance_residual).all()
        and all(torch.isfinite(value).all() for value in result.fluxes.values())
    )


def _flux_vector(result, names: tuple[str, ...]) -> torch.Tensor:
    return sum((result.fluxes[name] for name in names), result.q * 0.0)


def _reference_flux_vector(reference: ReferenceResult, names: tuple[str, ...]) -> torch.Tensor:
    return sum(
        (torch.as_tensor(reference.fluxes[name], dtype=torch.float64) for name in names),
        torch.zeros(reference.q_routed.size, dtype=torch.float64),
    )


def _load_development_set(path: Path) -> tuple[list[int], dict[str, Any]]:
    payload = json.loads(path.read_text())
    model_ids = [int(value) for value in payload["model_ids"]]
    catalogue_ids = {spec.model_id for spec in enumerate_structures()}
    if len(model_ids) != 18 or len(set(model_ids)) != 18 or not set(model_ids) <= catalogue_ids:
        raise ValueError("development set must contain 18 unique catalogue IDs")
    if not payload.get("selection_locked_before_results", False):
        raise ValueError("development set is not marked as locked before results")
    return model_ids, payload


def _topographic_for(spec, params_tensor: Mapping[str, torch.Tensor], theta: torch.Tensor):
    if spec.decisions["QSURF"] == "tmdl_param" or spec.decisions["ARCH2"] == "unlimpow_2":
        return _topographic_mean(params_tensor)
    return theta[0] * 0.0, theta[0] * 0.0


def _raw_flux_audit(model_ids: list[int], forcing: torch.Tensor) -> dict[str, Any]:
    rows = []
    params_tensor = _parameter_values(
        default_parameters(), dtype=forcing.dtype, device=forcing.device
    )
    theta = _parameter_vector(params_tensor)
    cap = _capacity(params_tensor)
    for model_id in model_ids:
        spec = get_structure(model_id)
        state = _initial_state(spec, params_tensor, cap, 0.25)
        union = _sequential_union_state(state, spec)
        context = _structure_context(spec, dtype=forcing.dtype, device=forcing.device)
        topo = _topographic_for(spec, params_tensor, theta)
        full = _compiled_flux_rhs(
            union, forcing[0, 0], forcing[0, 1], theta, context, topo[0], topo[1]
        )[1]
        process_errors = {}
        for process, mask_values in _SEQUENTIAL_FLUX_MASKS.items():
            process_raw = _sequential_raw_process(
                union, forcing[0, 0], forcing[0, 1], theta, context, topo[0], topo[1], process
            )
            mask = torch.as_tensor(
                mask_values[: len(FLUX_NAMES)], dtype=forcing.dtype, device=forcing.device
            )
            process_errors[process] = float(((process_raw - full) * mask).abs().max())
        pairwise = max(process_errors.values(), default=0.0)
        rows.append(
            {
                "model_id": model_id,
                "max_raw_flux_process_abs": pairwise,
                "raw_flux_process_abs": process_errors,
            }
        )
    return {
        "equations": "all orders call dfuse.kernel._compiled_flux_rhs; only state update order differs",
        "max_raw_flux_process_abs": max((row["max_raw_flux_process_abs"] for row in rows), default=0.0),
        "shared_projection_rule": "_sequential_project_union",
        "shared_clamp_spill_capacity": True,
        "rows": rows,
    }


def _baseline(
    executable: str, model_ids: list[int], forcing_map: Mapping[str, np.ndarray], forcing: torch.Tensor
):
    implicit = {}
    references = {}
    for model_id in model_ids:
        implicit[model_id] = simulate(model_id, forcing, implicit_iterations=16)
        references[model_id] = run_reference(executable, model_id, forcing_map)
    return implicit, references


def _state_error(result, baseline, reference: ReferenceResult, model_id: int):
    spec = get_structure(model_id)
    seq = result.states[:-1]
    imp = baseline.states[:-1]
    ref = torch.stack(
        [torch.as_tensor(reference.states[name], dtype=seq.dtype) for name in spec.state_names], dim=1
    )
    imp_error = (seq - imp).abs()
    ref_error = (seq - ref).abs()
    return {
        "implicit": {"max": _max_abs(imp_error), "median": _number(torch.median(imp_error)), "p95": _p95(imp_error)},
        "reference": {"max": _max_abs(ref_error), "median": _number(torch.median(ref_error)), "p95": _p95(ref_error)},
    }


def _flux_errors(result, baseline, reference: ReferenceResult):
    by_process = {}
    for process, names in PROCESS_FLUXES.items():
        seq = _flux_vector(result, names)
        imp = _flux_vector(baseline, names)
        ref = _reference_flux_vector(reference, names).to(dtype=seq.dtype)
        by_process[process] = {
            "implicit": {"max": _max_abs(seq - imp), "median": _number(torch.median((seq - imp).abs())), "p95": _p95(seq - imp)},
            "reference": {"max": _max_abs(seq - ref), "median": _number(torch.median((seq - ref).abs())), "p95": _p95(seq - ref)},
        }
    return by_process


def _model_metrics(result, baseline, reference: ReferenceResult, model_id: int) -> dict[str, Any]:
    q_reference = torch.as_tensor(reference.q_routed, dtype=result.q.dtype)
    metrics = {
        "model_id": model_id,
        "decisions": dict(get_structure(model_id).decisions),
        "finite": _finite(result),
        "negative_storage": bool((result.states < 0.0).any()),
        "min_state": _number(result.states.min()),
        "water_balance_max_abs": _max_abs(result.water_balance_residual),
        "snow_balance_max_abs": _max_abs(result.snow_balance_residual),
        "q": {
            "implicit_max_abs": _max_abs(result.q - baseline.q),
            "implicit_rmse": _rmse(result.q - baseline.q),
            "reference_max_abs": _max_abs(result.q - q_reference),
            "reference_rmse": _rmse(result.q - q_reference),
            "kgecomp_implicit": _kgecomp_value(result.q, baseline.q),
            "kgecomp_reference": _kgecomp_value(result.q, q_reference),
        },
        "state": _state_error(result, baseline, reference, model_id),
        "flux": _flux_errors(result, baseline, reference),
    }
    if result.sequential_diagnostics is not None:
        metrics["projection"] = {
            name: _number(value.sum()) for name, value in result.sequential_diagnostics.items()
        }
    return metrics


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(path: tuple[str, ...]):
        values = []
        for row in rows:
            value: Any = row
            for key in path:
                value = value[key]
            if value is not None:
                values.append(float(value))
        return max(values) if values else None

    return {
        "count": len(rows),
        "finite": sum(bool(row["finite"]) for row in rows),
        "negative_storage": sum(bool(row["negative_storage"]) for row in rows),
        "max_water_balance": collect(("water_balance_max_abs",)),
        "max_snow_balance": collect(("snow_balance_max_abs",)),
        "max_q_implicit": collect(("q", "implicit_max_abs")),
        "max_q_reference": collect(("q", "reference_max_abs")),
        "max_state_implicit": collect(("state", "implicit", "max")),
        "max_state_reference": collect(("state", "reference", "max")),
        "max_flux_implicit": max(
            (collect(("flux", process, "implicit", "max")) or 0.0 for process in PROCESS_FLUXES),
            default=0.0,
        ),
        "max_flux_reference": max(
            (collect(("flux", process, "reference", "max")) or 0.0 for process in PROCESS_FLUXES),
            default=0.0,
        ),
        "worst_q_reference_models": sorted(
            ((row["q"]["reference_max_abs"] or -1.0, row["model_id"]) for row in rows),
            reverse=True,
        )[:5],
    }


def _parity(compiled_rows, eager_rows):
    by_id = {row["model_id"]: row for row in eager_rows}
    return {
        "count": len(compiled_rows),
        "finite": sum(bool(row["finite"]) for row in compiled_rows),
        "max_q_abs": max(
            (float((row["q"] - by_id[row["model_id"]]["q"]).abs().max()) for row in compiled_rows),
            default=0.0,
        ),
    }


def _parity_result(compiled, eager):
    flux_abs = max(
        (float((compiled.fluxes[name] - eager.fluxes[name]).abs().max()) for name in FLUX_NAMES),
        default=0.0,
    )
    return {
        "q_max_abs": float((compiled.q - eager.q).abs().max()),
        "state_max_abs": float((compiled.states - eager.states).abs().max()),
        "flux_max_abs": flux_abs,
        "water_balance_max_abs": float((compiled.water_balance_residual - eager.water_balance_residual).abs().max()),
        "diagnostics_max_abs": max(
            (
                float((compiled.sequential_diagnostics[name] - eager.sequential_diagnostics[name]).abs().max())
                for name in compiled.sequential_diagnostics
            ),
            default=0.0,
        ),
    }


def _gradient_probe(
    model_ids: tuple[int, ...], forcing: torch.Tensor, order: str, n_substeps: int
) -> dict[str, Any]:
    rows = []
    defaults = default_parameters()
    for model_id in model_ids:
        raw = {
            name: torch.tensor(value, dtype=forcing.dtype, requires_grad=True)
            for name, value in defaults.items()
        }
        result = simulate_sequential(
            model_id,
            forcing[: min(4, forcing.shape[0])],
            raw,
            order=order,
            n_substeps=n_substeps,
            compile_step=True,
        )
        result.q.sum().backward()
        spec = get_structure(model_id)
        active = []
        inactive = []
        for name, value in raw.items():
            grad = value.grad
            if name in spec.parameter_names:
                active.append(grad is not None and bool(torch.isfinite(grad).all()))
            elif grad is not None:
                inactive.append(float(grad.abs().max()))
        rows.append(
            {
                "model_id": model_id,
                "active_finite": all(active),
                "inactive_max_abs": max(inactive, default=0.0),
                "inactive_zero_or_none": max(inactive, default=0.0) == 0.0,
            }
        )
    return {
        "count": len(rows),
        "active_finite": sum(bool(row["active_finite"]) for row in rows),
        "inactive_zero_or_none": sum(bool(row["inactive_zero_or_none"]) for row in rows),
        "rows": rows,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(callable_, device: torch.device, repeats: int = 2) -> dict[str, float]:
    values = []
    for _ in range(repeats):
        _synchronize(device)
        started = time.perf_counter()
        result = callable_()
        _synchronize(device)
        values.append(time.perf_counter() - started)
        del result
    return {
        "min_seconds": min(values),
        "median_seconds": float(np.median(values)),
        "max_seconds": max(values),
    }


def _compile_ok(order: str, n_substeps: int, device: str = "cpu") -> bool:
    variants = compile_diagnostics()["sequential"]["variants"].values()
    return any(
        record.get("order") == order
        and int(record.get("n_substeps", -1)) == n_substeps
        and record.get("key", [None, None, None, None, None, device])[5] == device
        and int(record.get("compile_successes", 0)) >= 1
        and int(record.get("fallbacks", 0)) == 0
        for record in variants
    )

def _benchmark(order: str, n_substeps: int, device: torch.device, forcing_cpu: torch.Tensor):
    forcing = forcing_cpu.to(device=device)
    model_id = 210
    # Warm the complete step before timing; cold Inductor time is kept in the graph audit.
    warm = simulate_sequential(model_id, forcing[:1], order=order, n_substeps=n_substeps, compile_step=True)
    _synchronize(device)
    del warm
    eager = _timed(
        lambda: simulate_sequential(model_id, forcing[:1], order=order, n_substeps=n_substeps, compile_step=False),
        device,
    )
    compiled = _timed(
        lambda: simulate_sequential(model_id, forcing[:1], order=order, n_substeps=n_substeps, compile_step=True),
        device,
    )
    eager_series = _timed(
        lambda: simulate_sequential(model_id, forcing, order=order, n_substeps=n_substeps, compile_step=False),
        device,
    )
    compiled_series = _timed(
        lambda: simulate_sequential(model_id, forcing, order=order, n_substeps=n_substeps, compile_step=True),
        device,
    )

    def make_raw():
        return {
            name: torch.tensor(value, dtype=forcing.dtype, device=device, requires_grad=True)
            for name, value in default_parameters().items()
        }

    def eager_backward():
        raw = make_raw()
        result = simulate_sequential(model_id, forcing, raw, order=order, n_substeps=n_substeps, compile_step=False)
        result.q.sum().backward()
        return result

    def compiled_backward():
        raw = make_raw()
        result = simulate_sequential(model_id, forcing, raw, order=order, n_substeps=n_substeps, compile_step=True)
        result.q.sum().backward()
        return result

    eager_bwd = _timed(eager_backward, device)
    compiled_bwd = _timed(compiled_backward, device)
    eager_total = _timed(
        lambda: (lambda raw: (simulate_sequential(model_id, forcing, raw, order=order, n_substeps=n_substeps, compile_step=False).q.sum().backward()))(make_raw()),
        device,
    )
    compiled_total = _timed(
        lambda: (lambda raw: (simulate_sequential(model_id, forcing, raw, order=order, n_substeps=n_substeps, compile_step=True).q.sum().backward()))(make_raw()),
        device,
    )
    return {
        "device": str(device),
        "dtype": str(forcing.dtype),
        "one_step_forward": {"eager": eager, "compiled": compiled, "speedup": eager["median_seconds"] / compiled["median_seconds"]},
        "time_series_forward": {"eager": eager_series, "compiled": compiled_series, "speedup": eager_series["median_seconds"] / compiled_series["median_seconds"]},
        "time_series_backward": {"eager": eager_bwd, "compiled": compiled_bwd, "speedup": eager_bwd["median_seconds"] / compiled_bwd["median_seconds"]},
        "time_series_forward_backward": {"eager": eager_total, "compiled": compiled_total, "speedup": eager_total["median_seconds"] / compiled_total["median_seconds"]},
    }


def run_validation(
    executable: str,
    output: Path,
    development_path: Path,
    n_steps: int,
    orders: tuple[str, ...] = ORDERS,
    n_substeps_values: tuple[int, ...] = N_SUBSTEPS,
    skip_cuda: bool = False,
 ) -> dict[str, Any]:
    torch.set_num_threads(1)
    model_ids, development_metadata = _load_development_set(development_path)
    forcing_map = synthetic_forcing(n_steps)
    forcing = _as_tensor_forcing(forcing_map)
    implicit, references = _baseline(executable, model_ids, forcing_map, forcing)
    raw_audit = _raw_flux_audit(model_ids, forcing)
    candidates = []
    for order in orders:
        for n_substeps in n_substeps_values:
            reset_compile_diagnostics()
            eager_results = {
                model_id: simulate_sequential(
                    model_id, forcing, order=order, n_substeps=n_substeps, compile_step=False
                )
                for model_id in model_ids
            }
            compiled_results = {
                model_id: simulate_sequential(
                    model_id, forcing, order=order, n_substeps=n_substeps, compile_step=True
                )
                for model_id in model_ids
            }
            eager_rows = [
                _model_metrics(eager_results[model_id], implicit[model_id], references[model_id], model_id)
                for model_id in model_ids
            ]
            compiled_rows = [
                _model_metrics(compiled_results[model_id], implicit[model_id], references[model_id], model_id)
                for model_id in model_ids
            ]
            parity = {
                "max": {
                    key: max(
                        (_number(_parity_result(compiled_results[mid], eager_results[mid])[key]) or 0.0 for mid in model_ids),
                        default=0.0,
                    )
                    for key in ("q_max_abs", "state_max_abs", "flux_max_abs", "water_balance_max_abs", "diagnostics_max_abs")
                },
            }
            parity["within_existing_tolerance"] = all(
                value <= 1.0e-5 for value in parity["max"].values()
            )
            compile_ok = _compile_ok(order, n_substeps, "cpu")
            if compile_ok:
                cpu_benchmark = _benchmark(order, n_substeps, forcing_cpu=forcing, device=forcing.device)
            else:
                cpu_benchmark = {"status": "compile_failed"}
            cuda_benchmark = None
            if compile_ok and torch.cuda.is_available() and not skip_cuda:
                cuda_benchmark = _benchmark(
                    order, n_substeps, forcing_cpu=forcing, device=torch.device("cuda")
                )
            gradient = (
                _gradient_probe((2, 108, 178, 210), forcing, order, n_substeps)
                if compile_ok
                else {"status": "compile_failed"}
            )
            device_info = torch.cuda.get_device_name(torch.device("cuda")) if cuda_benchmark is not None else None
            diagnostics = copy.deepcopy(compile_diagnostics()["sequential"])
            candidates.append(
                {
                    "order": order,
                    "order_definition": list(SEQUENTIAL_ORDERS[order]),
                    "n_substeps": n_substeps,
                    "compiled_step_usable": compile_ok,
                    "development": _aggregate(compiled_rows),
                    "development_rows": compiled_rows,
                    "eager_vs_compiled_parity": parity,
                    "gradient": gradient,
                    "compile": diagnostics,
                    "benchmark": {
                        "device_name": device_info,
                        "cpu": cpu_benchmark,
                        "cuda": cuda_benchmark,
                    },
                }
            )
            del eager_results, compiled_results, eager_rows, compiled_rows
            reset_compile_diagnostics()
    payload = {
        "schema": "autofuse-sequential-validation-v1",
        "selection": development_metadata,
        "orders": {name: list(SEQUENTIAL_ORDERS[name]) for name in orders},
        "n_substeps": list(n_substeps_values),
        "skip_cuda": skip_cuda,
        "forcing_steps": n_steps,
        "raw_flux_audit": raw_audit,
        "candidates": candidates,
        "compile_boundary": {
            "kind": "one complete hydrological macro-step",
            "outer_time_loop": "ordinary Python",
            "packed_state": "9 hydro union states + SWE + 500 routing-future bins",
            "snow_inside_step": True,
            "routing_inside_step": True,
            "newton_inside_step": False,
            "formal_training_started": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--development-set",
        type=Path,
        default=Path("project/autofuse/docs/sequential_development_set.json"),
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--orders", nargs="+", choices=ORDERS, default=ORDERS)
    parser.add_argument("--n-substeps", nargs="+", type=int, default=N_SUBSTEPS)
    parser.add_argument("--skip-cuda", action="store_true")
    args = parser.parse_args()
    payload = run_validation(
        args.executable,
        args.output,
        args.development_set,
        args.steps,
        orders=tuple(args.orders),
        n_substeps_values=tuple(args.n_substeps),
        skip_cuda=args.skip_cuda,
    )
    print(
        json.dumps(
            {
                "candidates": len(payload["candidates"]),
                "development_size": len(payload["selection"]["model_ids"]),
                "raw_flux_max": payload["raw_flux_audit"]["max_raw_flux_process_abs"],
            }
        )
    )


if __name__ == "__main__":
    main()
