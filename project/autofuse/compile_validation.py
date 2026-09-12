"""Bounded validation and benchmark for the optional compiled inner RHS."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path
from typing import Iterable

import torch

from dfuse import (
    compile_diagnostics,
    enumerate_structures,
    evaluate_implicit_residual,
    reset_compile_diagnostics,
    simulate,
)
from dfuse.kernel import (
    _CompiledInnerKernel,
    _eager_derivatives,
    _initial_state,
    _parameter_values,
    _snow_step,
    _topographic_mean,
    _capacity,
)
from dfuse.spec import STATE_NAMES, default_parameters, get_structure

from .fidelity import _as_tensor_forcing, regress_all_78, synthetic_forcing


SENSITIVE_MODELS = (164, 188, 212, 214, 166, 190)
GRADIENT_MODELS = (2, 108, 178, 210, *SENSITIVE_MODELS)


def _diagnostics_from_flux(flux: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(
        (
            flux["SATAREA"],
            flux["QPERC_12"],
            flux["QBASE_2"],
            flux["QSURF"],
            flux["OFLOW_1"] + flux["OFLOW_2"],
            flux["EVAP_1"] + flux["EVAP_2"],
        )
    )


def _inner_case(model_id: int, forcing: torch.Tensor):
    spec = get_structure(model_id)
    params = _parameter_values(default_parameters(), dtype=forcing.dtype, device=forcing.device)
    cap = _capacity(params)
    state = _initial_state(spec, params, cap, 0.25)
    effective, _, _ = _snow_step(
        forcing[0, 0],
        forcing[0, 2],
        params,
        torch.zeros((), dtype=forcing.dtype, device=forcing.device),
        torch.ones((), dtype=forcing.dtype, device=forcing.device),
        torch.zeros((), dtype=torch.bool, device=forcing.device),
        1.0,
    )
    topographic = None
    if spec.decisions["QSURF"] == "tmdl_param" or spec.decisions["ARCH2"] == "unlimpow_2":
        topographic = _topographic_mean(params)
    return spec, params, cap, state, effective, topographic


def _inner_parity(model_ids: Iterable[int], forcing: torch.Tensor) -> dict[str, object]:
    rows = []
    for model_id in model_ids:
        spec, params, cap, state, effective, topographic = _inner_case(model_id, forcing)
        eager_rhs, eager_flux = _eager_derivatives(
            state, spec, params, cap, effective, forcing[0, 1], topographic
        )
        runner = _CompiledInnerKernel(
            spec,
            params,
            cap,
            topographic,
            backend="inductor",
            fullgraph=True,
        )
        compiled_rhs, compiled_flux, compiled_diag = runner.evaluate(
            state, effective, forcing[0, 1]
        )
        eager_diag = _diagnostics_from_flux(eager_flux)
        flux_difference = max(
            float((compiled_flux[name] - eager_flux[name]).abs().detach())
            for name in eager_flux
        )
        rows.append(
            {
                "model_id": model_id,
                "rhs_max_abs": float((compiled_rhs - eager_rhs).abs().max().detach()),
                "flux_max_abs": flux_difference,
                "diagnostics_max_abs": float((compiled_diag - eager_diag).abs().max().detach()),
                "finite": bool(
                    torch.isfinite(compiled_rhs).all()
                    and torch.isfinite(compiled_diag).all()
                    and all(torch.isfinite(value) for value in compiled_flux.values())
                ),
            }
        )
    return {
        "count": len(rows),
        "finite": sum(bool(row["finite"]) for row in rows),
        "max_rhs_abs": max(row["rhs_max_abs"] for row in rows),
        "max_flux_abs": max(row["flux_max_abs"] for row in rows),
        "max_diagnostics_abs": max(row["diagnostics_max_abs"] for row in rows),
        "sensitive": [row for row in rows if row["model_id"] in SENSITIVE_MODELS],
    }


def _full_forward_parity(model_ids: Iterable[int], forcing: torch.Tensor) -> dict[str, object]:
    rows = []
    with torch.no_grad():
        for model_id in model_ids:
            eager = simulate(model_id, forcing, implicit_iterations=16)
            compiled = simulate(
                model_id,
                forcing,
                implicit_iterations=16,
                compile_inner=True,
                compile_inner_backend="inductor",
                compile_inner_fullgraph=True,
            )
            flux_difference = max(
                float((compiled.fluxes[name] - eager.fluxes[name]).abs().max().detach())
                for name in eager.fluxes
            )
            rows.append(
                {
                    "model_id": model_id,
                    "q_max_abs": float((compiled.q - eager.q).abs().max().detach()),
                    "state_max_abs": float((compiled.states - eager.states).abs().max().detach()),
                    "flux_max_abs": flux_difference,
                    "water_balance_max_abs": float(
                        (compiled.water_balance_residual - eager.water_balance_residual).abs().max().detach()
                    ),
                    "finite": bool(
                        torch.isfinite(compiled.q).all()
                        and torch.isfinite(compiled.states).all()
                        and all(torch.isfinite(value).all() for value in compiled.fluxes.values())
                    ),
                }
            )
    return {
        "count": len(rows),
        "finite": sum(bool(row["finite"]) for row in rows),
        "max_q_abs": max(row["q_max_abs"] for row in rows),
        "max_state_abs": max(row["state_max_abs"] for row in rows),
        "max_flux_abs": max(row["flux_max_abs"] for row in rows),
        "max_water_balance_abs": max(row["water_balance_max_abs"] for row in rows),
        "sensitive": [row for row in rows if row["model_id"] in SENSITIVE_MODELS],
    }


def _jacobian_parity(model_ids: Iterable[int], forcing: torch.Tensor) -> dict[str, object]:
    rows = []
    for model_id in model_ids:
        spec, params, cap, state, effective, topographic = _inner_case(model_id, forcing)
        runner = _CompiledInnerKernel(
            spec,
            params,
            cap,
            topographic,
            backend="inductor",
            fullgraph=True,
        )

        def eager_fn(candidate: torch.Tensor) -> torch.Tensor:
            return _eager_derivatives(
                candidate, spec, params, cap, effective, forcing[0, 1], topographic
            )[0]

        def compiled_fn(candidate: torch.Tensor) -> torch.Tensor:
            return runner.evaluate(candidate, effective, forcing[0, 1])[0]

        eager_jacobian = torch.autograd.functional.jacobian(eager_fn, state)
        compiled_jacobian = torch.autograd.functional.jacobian(compiled_fn, state)
        rows.append(
            {
                "model_id": model_id,
                "max_abs": float((compiled_jacobian - eager_jacobian).abs().max().detach()),
                "max_rel": float(
                    ((compiled_jacobian - eager_jacobian).abs() / (eager_jacobian.abs() + 1.0e-12))
                    .max()
                    .detach()
                ),
            }
        )
    return {
        "count": len(rows),
        "max_abs": max(row["max_abs"] for row in rows),
        "max_rel": max(row["max_rel"] for row in rows),
        "rows": rows,
    }


def _gradient_parity(model_ids: Iterable[int], forcing: torch.Tensor) -> dict[str, object]:
    rows = []
    for model_id in model_ids:
        raw_eager = {
            name: torch.tensor(value, dtype=forcing.dtype, requires_grad=True)
            for name, value in default_parameters().items()
        }
        eager = simulate(model_id, forcing, raw_eager, implicit_iterations=16)
        eager_grads = torch.autograd.grad(eager.q.sum(), tuple(raw_eager.values()), allow_unused=True)
        raw_compiled = {
            name: torch.tensor(value, dtype=forcing.dtype, requires_grad=True)
            for name, value in default_parameters().items()
        }
        compiled = simulate(
            model_id,
            forcing,
            raw_compiled,
            implicit_iterations=16,
            compile_inner=True,
            compile_inner_backend="inductor",
            compile_inner_fullgraph=True,
        )
        compiled_grads = torch.autograd.grad(
            compiled.q.sum(), tuple(raw_compiled.values()), allow_unused=True
        )
        active = set(get_structure(model_id).parameter_names)
        active_abs = []
        active_rel = []
        inactive_ok = True
        finite = True
        for name, eager_grad, compiled_grad in zip(raw_eager, eager_grads, compiled_grads):
            if compiled_grad is not None:
                finite = finite and bool(torch.isfinite(compiled_grad).all())
            if name in active:
                eager_value = torch.zeros((), dtype=forcing.dtype) if eager_grad is None else eager_grad
                compiled_value = torch.zeros((), dtype=forcing.dtype) if compiled_grad is None else compiled_grad
                difference = (compiled_value - eager_value).abs()
                active_abs.append(float(difference.detach()))
                active_rel.append(float((difference / (eager_value.abs() + 1.0e-12)).detach()))
            elif compiled_grad is not None:
                inactive_ok = inactive_ok and bool((compiled_grad == 0.0).all())
        rows.append(
            {
                "model_id": model_id,
                "active_max_abs": max(active_abs, default=0.0),
                "active_max_rel": max(active_rel, default=0.0),
                "inactive_zero_or_none": inactive_ok,
                "finite": finite,
            }
        )
    return {
        "count": len(rows),
        "max_active_abs": max(row["active_max_abs"] for row in rows),
        "max_active_rel": max(row["active_max_rel"] for row in rows),
        "inactive_zero_or_none": all(bool(row["inactive_zero_or_none"]) for row in rows),
        "finite": sum(bool(row["finite"]) for row in rows),
        "rows": rows,
    }


def _double_gradient_parity(forcing: torch.Tensor) -> dict[str, object]:
    cases = ((2, "MAXWATR_1"), (210, "TISHAPE"))
    rows = []
    for model_id, parameter_name in cases:
        raw_eager = {
            name: torch.tensor(
                value, dtype=forcing.dtype, requires_grad=(name == parameter_name)
            )
            for name, value in default_parameters().items()
        }
        eager = simulate(model_id, forcing[:1], raw_eager, implicit_iterations=1)
        eager_first = torch.autograd.grad(
            eager.q.sum(), raw_eager[parameter_name], create_graph=True
        )[0]
        eager_second = torch.autograd.grad(
            eager_first, raw_eager[parameter_name], allow_unused=True
        )[0]
        raw_compiled = {
            name: torch.tensor(
                value, dtype=forcing.dtype, requires_grad=(name == parameter_name)
            )
            for name, value in default_parameters().items()
        }
        compiled = simulate(
            model_id,
            forcing[:1],
            raw_compiled,
            implicit_iterations=1,
            compile_inner=True,
            compile_inner_backend="inductor",
            compile_inner_fullgraph=True,
        )
        compiled_first = torch.autograd.grad(
            compiled.q.sum(), raw_compiled[parameter_name], create_graph=True
        )[0]
        compiled_second = torch.autograd.grad(
            compiled_first, raw_compiled[parameter_name], allow_unused=True
        )[0]
        second_eager = torch.zeros((), dtype=forcing.dtype) if eager_second is None else eager_second
        second_compiled = torch.zeros((), dtype=forcing.dtype) if compiled_second is None else compiled_second
        rows.append(
            {
                "model_id": model_id,
                "parameter": parameter_name,
                "first_max_abs": float((compiled_first - eager_first).abs().detach()),
                "second_max_abs": float((second_compiled - second_eager).abs().detach()),
                "finite": bool(
                    torch.isfinite(compiled_first).all()
                    and torch.isfinite(second_compiled).all()
                ),
            }
        )
    return {
        "count": len(rows),
        "max_first_abs": max(row["first_max_abs"] for row in rows),
        "max_second_abs": max(row["second_max_abs"] for row in rows),
        "finite": sum(bool(row["finite"]) for row in rows),
        "rows": rows,
    }



def _level_b_parity(forcing: torch.Tensor) -> dict[str, object]:
    spec, params, cap, state, effective, topographic = _inner_case(2, forcing)
    candidate = state + 0.1
    eager_residual, eager_flux, eager_diag = evaluate_implicit_residual(
        2, candidate, state, effective, forcing[0, 1], dt_days=1.0
    )
    compiled_residual, compiled_flux, compiled_diag = evaluate_implicit_residual(
        2,
        candidate,
        state,
        effective,
        forcing[0, 1],
        dt_days=1.0,
        compile_residual=True,
        compile_backend="inductor",
        compile_fullgraph=True,
    )
    return {
        "model_id": 2,
        "residual_max_abs": float((compiled_residual - eager_residual).abs().max().detach()),
        "flux_max_abs": max(
            float((compiled_flux[name] - eager_flux[name]).abs().detach()) for name in eager_flux
        ),
        "diagnostics_max_abs": float((compiled_diag - eager_diag).abs().max().detach()),
        "finite": bool(
            torch.isfinite(compiled_residual).all()
            and torch.isfinite(compiled_diag).all()
            and all(torch.isfinite(value) for value in compiled_flux.values())
        ),
    }


def _timed(callable_, repeats: int, *, synchronize: bool = False) -> dict[str, float]:
    if synchronize:
        torch.cuda.synchronize()
    callable_()
    if synchronize:
        torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        callable_()
        values.append(time.perf_counter() - started)
    return {"median_seconds": statistics.median(values), "min_seconds": min(values), "max_seconds": max(values)}


def _cpu_benchmark(forcing: torch.Tensor) -> dict[str, object]:
    one_step = forcing[:1]
    with torch.no_grad():
        eager_one = _timed(lambda: simulate(2, one_step, implicit_iterations=16), 3)
        compiled_one = _timed(
            lambda: simulate(
                2,
                one_step,
                implicit_iterations=16,
                compile_inner=True,
                compile_inner_backend="inductor",
                compile_inner_fullgraph=True,
            ),
            3,
        )
        eager_series = _timed(lambda: simulate(2, forcing, implicit_iterations=16), 3)
        compiled_series = _timed(
            lambda: simulate(
                2,
                forcing,
                implicit_iterations=16,
                compile_inner=True,
                compile_inner_backend="inductor",
                compile_inner_fullgraph=True,
            ),
            3,
        )
        spec, params, cap, state, effective, topographic = _inner_case(2, one_step)
        runner = _CompiledInnerKernel(
            spec,
            params,
            cap,
            topographic,
            backend="inductor",
            fullgraph=True,
        )
        eager_inner = _timed(
            lambda: _eager_derivatives(
                state, spec, params, cap, effective, one_step[0, 1], topographic
            ),
            50,
        )
        compiled_inner = _timed(
            lambda: runner.evaluate(state, effective, one_step[0, 1]),
            50,
        )
    return {
        "device": "cpu",
        "dtype": str(forcing.dtype),
        "steps": int(forcing.shape[0]),
        "eager_one_step": eager_one,
        "compiled_inner_one_step": compiled_one,
        "one_step_speedup": eager_one["median_seconds"] / compiled_one["median_seconds"],
        "eager_time_series": eager_series,
        "compiled_inner_time_series": compiled_series,
        "time_series_speedup": eager_series["median_seconds"] / compiled_series["median_seconds"],
        "eager_rhs": eager_inner,
        "compiled_rhs": compiled_inner,
        "rhs_speedup": eager_inner["median_seconds"] / compiled_inner["median_seconds"],
    }


def _cuda_benchmark() -> dict[str, object] | None:
    if not torch.cuda.is_available():
        return None
    forcing = _as_tensor_forcing(synthetic_forcing(8)).cuda()
    with torch.no_grad():
        # One short warm-up/measurement avoids retaining a large CUDA graph or history.
        eager = _timed(lambda: simulate(2, forcing, implicit_iterations=16), 2, synchronize=True)
        compiled = _timed(
            lambda: simulate(
                2,
                forcing,
                implicit_iterations=16,
                compile_inner=True,
                compile_inner_backend="inductor",
                compile_inner_fullgraph=True,
            ),
            2,
            synchronize=True,
        )
    return {
        "device": torch.cuda.get_device_name(0),
        "dtype": str(forcing.dtype),
        "steps": int(forcing.shape[0]),
        "eager_time_series": eager,
        "compiled_inner_time_series": compiled,
        "time_series_speedup": eager["median_seconds"] / compiled["median_seconds"],
    }


def _reference_summary(executable: str, forcing: dict[str, object]) -> dict[str, object]:
    rows = regress_all_78(executable, forcing, implicit_iterations=16)
    return {
        "count": len(rows),
        "dfuse_finite": sum(bool(row["dfuse_finite"]) for row in rows),
        "reference_finite": sum(bool(row["reference_finite"]) for row in rows),
        "max_q_abs": max(float(row["q_max_abs"]) for row in rows),
        "max_state_abs": max(float(row["state_max_abs"]) for row in rows),
        "max_flux_abs": max(float(row["flux_max_abs"]) for row in rows),
        "max_dfuse_water_balance": max(float(row["dfuse_water_balance_max_abs"]) for row in rows),
        "sensitive": [row for row in rows if int(row["model_id"]) in SENSITIVE_MODELS],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    forcing = _as_tensor_forcing(synthetic_forcing(4))
    all_ids = [spec.model_id for spec in enumerate_structures()]

    reset_compile_diagnostics()
    inner_parity = _inner_parity(all_ids, forcing)
    forward_parity = _full_forward_parity(all_ids, forcing)
    jacobian_parity = _jacobian_parity(all_ids, forcing)
    gradient_parity = _gradient_parity(GRADIENT_MODELS, forcing)
    double_gradient_parity = _double_gradient_parity(forcing)
    level_a_diagnostics = copy.deepcopy(compile_diagnostics()["level_a"])
    cpu_benchmark = _cpu_benchmark(_as_tensor_forcing(synthetic_forcing(24)))
    cuda_benchmark = _cuda_benchmark()

    reset_compile_diagnostics()
    level_b_parity = _level_b_parity(forcing)
    level_b_diagnostics = copy.deepcopy(compile_diagnostics()["level_b"])
    reference = _reference_summary(args.executable, synthetic_forcing(4))

    result = {
        "compile_boundary": {
            "level_a": "unified fixed-shape tensor flux/RHS only",
            "state_union": list(STATE_NAMES),
            "structure_context": "tensor flags, state mask, parameter mask",
            "eager_boundaries": ["time loop", "Newton loop", "Jacobian", "linear solve", "line search"],
            "backend": "inductor",
            "fullgraph": True,
            "formal_training_started": False,
        },
        "level_a": {
            "inner_parity_78": inner_parity,
            "full_forward_parity_78": forward_parity,
            "jacobian_parity": jacobian_parity,
            "gradient_parity": gradient_parity,
            "double_gradient_parity": double_gradient_parity,
            "diagnostics": level_a_diagnostics,
        },
        "level_b": {"parity": level_b_parity, "diagnostics": level_b_diagnostics},
        "benchmark": {"cpu": cpu_benchmark, "cuda": cuda_benchmark},
        "reference_regression_implicit_eager": reference,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "inner_finite": inner_parity["finite"],
        "forward_finite": forward_parity["finite"],
        "level_a_compile_successes": level_a_diagnostics["compile_successes"],
        "level_a_graphs": level_a_diagnostics["unique_graphs"],
        "level_b_compile_successes": level_b_diagnostics["compile_successes"],
        "reference_count": reference["count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
