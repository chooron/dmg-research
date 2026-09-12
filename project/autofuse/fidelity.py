"""Small, reproducible reference-vs-dFUSE fidelity harnesses.

The harness intentionally runs one catchment at a time.  It is a validation
utility, not a calibration driver: no SCE or dPL loop is started here.
"""

from __future__ import annotations

import time
from typing import Iterable, Mapping

import numpy as np
import torch

from dfuse import enumerate_structures, get_structure, simulate

from .metrics import kge, kgecomp
from .reference_oracle import ReferenceResult, run_reference


# These are canonical representatives present in the paper's 78-row list.
# Their fixed decisions (RFERR/ESOIL/QINTF/Q_TDH/SNOWM) remain those of the
# list; only the four open structural choices differ.
MOTHER_MODELS = {
    "VIC": 2,
    "PRMS": 108,
    "SAC-SMA": 178,
    "TOPMODEL": 210,
}


def synthetic_forcing(n_steps: int = 24) -> dict[str, np.ndarray]:
    """Return a deterministic warm synthetic forcing for oracle comparisons."""
    if n_steps < 4:
        raise ValueError("n_steps must be at least four")
    index = np.arange(n_steps, dtype=np.float64)
    return {
        "ppt": 1.0 + 4.0 * (np.mod(index, 7.0) == 0.0) + 0.5 * np.sin(index / 3.0),
        "pet": 1.5 + 0.25 * np.cos(index / 4.0),
        "temp": np.full(n_steps, 10.0, dtype=np.float64),
        "q_obs": np.zeros(n_steps, dtype=np.float64),
    }


def _as_tensor_forcing(forcing: Mapping[str, np.ndarray]) -> torch.Tensor:
    return torch.as_tensor(
        np.stack((forcing["ppt"], forcing["pet"], forcing["temp"]), axis=1),
        dtype=torch.float64,
    )


def _reference_storage(reference: ReferenceResult, model_id: int) -> np.ndarray:
    spec = get_structure(model_id)
    n = reference.q_routed.size
    storage = np.zeros(n, dtype=np.float64)
    for name in spec.state_names:
        storage += reference.states[name]
    if "SWE_TOT" in reference.states:
        storage += reference.states["SWE_TOT"]
    return storage

def _reference_water_balance(
    reference: ReferenceResult,
    model_id: int,
    forcing: Mapping[str, np.ndarray] | None = None,
 ) -> np.ndarray:
    """Compute residual using raw precipitation when SWE is part of storage."""
    storage = _reference_storage(reference, model_id)
    effective = reference.fluxes["EFF_PPT"]
    if "SWE_TOT" in reference.states and forcing is not None:
        input_rate = np.asarray(forcing["ppt"], dtype=np.float64).reshape(-1)
    else:
        input_rate = effective
    evaporation = reference.fluxes["EVAP_1"] + reference.fluxes["EVAP_2"]
    residual = input_rate[:-1] - evaporation[:-1] - reference.q_instantaneous[:-1]
    residual -= storage[1:] - storage[:-1]
    return residual


def _finite_reference(reference: ReferenceResult) -> bool:
    return bool(
        np.isfinite(reference.q_routed).all()
        and np.isfinite(reference.q_instantaneous).all()
        and all(np.isfinite(value).all() for value in reference.states.values())
        and all(np.isfinite(value).all() for value in reference.fluxes.values())
    )


def compare_explicit_one(
    executable: str,
    model_id: int,
    forcing: Mapping[str, np.ndarray],
    n_substeps: int,
    *,
    reference: ReferenceResult | None = None,
    implicit=None,
    implicit_iterations: int = 16,
) -> dict[str, object]:
    """Compare fixed-substep explicit Euler with both existing baselines."""
    if reference is None:
        reference = run_reference(executable, model_id, forcing)
    tensor_forcing = _as_tensor_forcing(forcing)
    if implicit is None:
        implicit = simulate(model_id, tensor_forcing, initial_fraction=0.25, implicit_iterations=implicit_iterations)
    started = time.perf_counter()
    explicit = simulate(
        model_id,
        tensor_forcing,
        initial_fraction=0.25,
        solver="explicit",
        n_substeps=n_substeps,
    )
    elapsed = time.perf_counter() - started
    spec = get_structure(model_id)
    q_explicit = explicit.q.detach().cpu().numpy()
    q_implicit = implicit.q.detach().cpu().numpy()
    q_reference = reference.q_routed
    q_error_reference = q_explicit - q_reference
    q_error_implicit = q_explicit - q_implicit
    q_error_implicit_reference = q_implicit - q_reference
    state_errors_reference = {
        name: float(np.max(np.abs(explicit.states[:-1, index].detach().cpu().numpy() - reference.states[name])))
        for index, name in enumerate(spec.state_names)
    }
    state_errors_implicit = {
        name: float(np.max(np.abs((explicit.states[:-1, index] - implicit.states[:-1, index]).detach().cpu().numpy())))
        for index, name in enumerate(spec.state_names)
    }
    flux_errors_reference = {
        name: float(np.max(np.abs(explicit.fluxes[name].detach().cpu().numpy() - reference.fluxes[name])))
        for name in explicit.fluxes
        if name in reference.fluxes
    }
    flux_errors_implicit = {
        name: float(np.max(np.abs((explicit.fluxes[name] - implicit.fluxes[name]).detach().cpu().numpy())))
        for name in explicit.fluxes
    }
    explicit_finite = bool(
        torch.isfinite(explicit.q).all()
        and torch.isfinite(explicit.states).all()
        and torch.isfinite(explicit.water_balance_residual).all()
        and torch.isfinite(explicit.snow_balance_residual).all()
        and all(torch.isfinite(value).all() for value in explicit.fluxes.values())
    )
    implicit_finite = bool(
        torch.isfinite(implicit.q).all()
        and torch.isfinite(implicit.states).all()
        and all(torch.isfinite(value).all() for value in implicit.fluxes.values())
    )
    return {
        "model_id": model_id,
        "decisions": dict(spec.decisions),
        "n_substeps": int(n_substeps),
        "dt_sub_days": 1.0 / float(n_substeps),
        "explicit_vs_reference_q_max_abs": float(np.max(np.abs(q_error_reference))),
        "explicit_vs_reference_q_rmse": float(np.sqrt(np.mean(q_error_reference * q_error_reference))),
        "explicit_vs_implicit_q_max_abs": float(np.max(np.abs(q_error_implicit))),
        "implicit_vs_reference_q_max_abs": float(np.max(np.abs(q_error_implicit_reference))),
        "explicit_vs_implicit_q_rmse": float(np.sqrt(np.mean(q_error_implicit * q_error_implicit))),
        "explicit_vs_reference_state_max_abs": max(state_errors_reference.values(), default=0.0),
        "explicit_vs_implicit_state_max_abs": max(state_errors_implicit.values(), default=0.0),
        "state_errors_reference": state_errors_reference,
        "state_errors_implicit": state_errors_implicit,
        "explicit_vs_reference_flux_max_abs": max(flux_errors_reference.values(), default=0.0),
        "explicit_vs_implicit_flux_max_abs": max(flux_errors_implicit.values(), default=0.0),
        "flux_errors_reference": flux_errors_reference,
        "flux_errors_implicit": flux_errors_implicit,
        "kgecomp_explicit_vs_reference": float(kgecomp(explicit.q, torch.as_tensor(q_reference, dtype=explicit.q.dtype)).detach()),
        "kgecomp_explicit_vs_implicit": float(kgecomp(explicit.q, implicit.q).detach()),
        "explicit_water_balance_max_abs": float(explicit.max_abs_water_balance_error.detach()),
        "explicit_conservation_ok_existing_tolerance": float(explicit.max_abs_water_balance_error.detach()) < 1.0e-5,
        "implicit_water_balance_max_abs": float(implicit.max_abs_water_balance_error.detach()),
        "explicit_snow_balance_max_abs": float(explicit.snow_balance_residual.abs().amax().detach()),
        "reference_water_balance_max_abs": float(np.max(np.abs(_reference_water_balance(reference, model_id, forcing)))) if q_reference.size else 0.0,
        "explicit_min_state": float(explicit.states.min().detach()),
        "explicit_finite": explicit_finite,
        "implicit_finite": implicit_finite,
        "reference_finite": _finite_reference(reference),
        "explicit_elapsed_seconds": elapsed,
        "reference_executable_sha256": reference.metadata["executable_sha256"],
    }


def compare_one(
    executable: str,
    model_id: int,
    forcing: Mapping[str, np.ndarray],
    *,
    implicit_iterations: int = 64,
) -> dict[str, object]:
    """Compare one structure using the same forcing/default parameter protocol."""
    reference = run_reference(executable, model_id, forcing)
    tensor_forcing = _as_tensor_forcing(forcing)
    started = time.perf_counter()
    differentiable = simulate(model_id, tensor_forcing, initial_fraction=0.25, implicit_iterations=implicit_iterations)
    elapsed = time.perf_counter() - started
    q = differentiable.q.detach().cpu().numpy()
    q_delta = q - reference.q_routed
    spec = get_structure(model_id)
    state_errors = {
        name: float(np.max(np.abs(differentiable.states[:-1, i].detach().cpu().numpy() - reference.states[name])))
        for i, name in enumerate(spec.state_names)
    }
    flux_errors = {
        name: float(np.max(np.abs(differentiable.fluxes[name].detach().cpu().numpy() - reference.fluxes[name])))
        for name in differentiable.fluxes
        if name in reference.fluxes
    }
    reference_balance = _reference_water_balance(reference, model_id, forcing)
    return {
        "model_id": model_id,
        "decisions": dict(spec.decisions),
        "implicit_iterations": implicit_iterations,
        "q_max_abs": float(np.max(np.abs(q_delta))),
        "q_rmse": float(np.sqrt(np.mean(q_delta * q_delta))),
        "state_max_abs": max(state_errors.values(), default=0.0),
        "state_errors": state_errors,
        "flux_max_abs": max(flux_errors.values(), default=0.0),
        "flux_errors": flux_errors,
        "kge_vs_reference": float(kge(differentiable.q, torch.as_tensor(reference.q_routed)).detach()),
        "kgecomp_vs_reference": float(kgecomp(differentiable.q, torch.as_tensor(reference.q_routed)).detach()),
        "dfuse_water_balance_max_abs": float(differentiable.max_abs_water_balance_error.detach()),
        "reference_water_balance_max_abs": float(np.max(np.abs(reference_balance))) if reference_balance.size else 0.0,
        "dfuse_finite": bool(
            torch.isfinite(differentiable.q).all()
            and torch.isfinite(differentiable.states).all()
            and all(torch.isfinite(value).all() for value in differentiable.fluxes.values())
        ),
        "reference_finite": bool(
            np.isfinite(reference.q_routed).all()
            and np.isfinite(reference.q_instantaneous).all()
            and all(np.isfinite(value).all() for value in reference.states.values())
            and all(np.isfinite(value).all() for value in reference.fluxes.values())
        ),
        "dfuse_elapsed_seconds": elapsed,
        "reference_executable_sha256": reference.metadata["executable_sha256"],
    }


def compare_mothers(
    executable: str,
    forcing: Mapping[str, np.ndarray] | None = None,
    *,
    implicit_iterations: int = 64,
) -> dict[str, dict[str, object]]:
    """Run the four canonical mother-configuration comparisons."""
    forcing = synthetic_forcing() if forcing is None else forcing
    return {
        label: compare_one(executable, model_id, forcing, implicit_iterations=implicit_iterations)
        for label, model_id in MOTHER_MODELS.items()
    }


def regress_all_78(
    executable: str,
    forcing: Mapping[str, np.ndarray] | None = None,
    *,
    implicit_iterations: int = 64,
) -> list[dict[str, object]]:
    """Run the same finite/conservation/fidelity checks for every paper model."""
    forcing = synthetic_forcing() if forcing is None else forcing
    return [
        compare_one(executable, spec.model_id, forcing, implicit_iterations=implicit_iterations)
        for spec in enumerate_structures()
    ]


def explicit_scan(
    executable: str,
    forcing: Mapping[str, np.ndarray] | None = None,
    *,
    n_substeps: Iterable[int] = (1, 2, 4, 8, 12, 24, 48),
    implicit_iterations: int = 16,
    model_ids: Iterable[int] | None = None,
) -> list[dict[str, object]]:
    """Run explicit substep fidelity checks with one cached baseline per structure."""
    forcing = synthetic_forcing() if forcing is None else forcing
    tensor_forcing = _as_tensor_forcing(forcing)
    ids = [spec.model_id for spec in enumerate_structures()] if model_ids is None else [int(model_id) for model_id in model_ids]
    references = {model_id: run_reference(executable, model_id, forcing) for model_id in ids}
    implicit = {
        model_id: simulate(model_id, tensor_forcing, initial_fraction=0.25, implicit_iterations=implicit_iterations)
        for model_id in ids
    }
    rows: list[dict[str, object]] = []
    for substep_count in n_substeps:
        if not isinstance(substep_count, int) or isinstance(substep_count, bool) or substep_count < 1:
            raise ValueError("all n_substeps values must be positive integers")
        for model_id in ids:
            rows.append(
                compare_explicit_one(
                    executable,
                    model_id,
                    forcing,
                    substep_count,
                    reference=references[model_id],
                    implicit=implicit[model_id],
                    implicit_iterations=implicit_iterations,
                )
            )
    return rows

def explicit_stress_scan(n_substeps: Iterable[int] = (1, 2, 4, 8, 12, 24, 48, 96, 192)) -> dict[str, dict[str, object]]:
    """Check all structures under the fixed 5000 mm/day bound stress."""
    forcing = torch.tensor([[5000.0, 0.0, 10.0]] * 3, dtype=torch.float64)
    result: dict[str, dict[str, object]] = {}
    for substep_count in n_substeps:
        checks = []
        for spec in enumerate_structures():
            output = simulate(spec.model_id, forcing, solver="explicit", n_substeps=int(substep_count))
            checks.append(
                {
                    "model_id": spec.model_id,
                    "finite": bool(
                        torch.isfinite(output.q).all()
                        and torch.isfinite(output.states).all()
                        and all(torch.isfinite(value).all() for value in output.fluxes.values())
                    ),
                    "max_water_balance": float(output.max_abs_water_balance_error.detach()),
                    "min_storage": float(output.states.min().detach()),
                }
            )
        result[str(int(substep_count))] = {
            "count": len(checks),
            "finite": sum(bool(item["finite"]) for item in checks),
            "max_water_balance": max(float(item["max_water_balance"]) for item in checks),
            "min_storage": min(float(item["min_storage"]) for item in checks),
            "worst_ids": [
                int(item["model_id"])
                for item in sorted(checks, key=lambda item: float(item["max_water_balance"]), reverse=True)[:10]
            ],
        }
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return result


def _spearman(a: Iterable[float], b: Iterable[float]) -> float:
    first = _average_ranks(np.asarray(list(a), dtype=np.float64))
    second = _average_ranks(np.asarray(list(b), dtype=np.float64))
    if first.size < 2 or np.std(first) == 0.0 or np.std(second) == 0.0:
        return 1.0 if np.array_equal(first, second) else 0.0
    return float(np.corrcoef(first, second)[0, 1])


def summarize_explicit_scan(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate explicit scan distributions and numerical-error rankings."""
    rows = [dict(row) for row in rows]
    by_substeps: dict[str, object] = {}
    for substep_count in sorted({int(row["n_substeps"]) for row in rows}):
        subset = [row for row in rows if int(row["n_substeps"]) == substep_count]
        def distribution(key: str) -> dict[str, float]:
            values = np.asarray([float(row[key]) for row in subset], dtype=np.float64)
            return {"max": float(np.max(values)), "median": float(np.median(values)), "p95": float(np.quantile(values, 0.95))}
        explicit_errors = {
            "q": distribution("explicit_vs_reference_q_max_abs"),
            "state": distribution("explicit_vs_reference_state_max_abs"),
            "flux": distribution("explicit_vs_reference_flux_max_abs"),
            "q_rmse": distribution("explicit_vs_reference_q_rmse"),
            "kgecomp_reference": distribution("kgecomp_explicit_vs_reference"),
            "kgecomp_implicit": distribution("kgecomp_explicit_vs_implicit"),
            "water_balance": distribution("explicit_water_balance_max_abs"),
            "snow_balance": distribution("explicit_snow_balance_max_abs"),
        }
        by_substeps[str(substep_count)] = {
            "finite": sum(bool(row["explicit_finite"]) for row in subset),
            "count": len(subset),
            "conservation_ok_existing_tolerance": sum(bool(row["explicit_conservation_ok_existing_tolerance"]) for row in subset),
            "errors": explicit_errors,
            "min_state": float(min(float(row["explicit_min_state"]) for row in subset)),
            "slowest_structures": [
                {"model_id": row["model_id"], "decisions": row["decisions"], "q_error": row["explicit_vs_reference_q_max_abs"]}
                for row in sorted(subset, key=lambda item: float(item["explicit_vs_reference_q_max_abs"]), reverse=True)[:10]
            ],
        }
        ids = [int(row["model_id"]) for row in subset]
        explicit_order = sorted(subset, key=lambda item: float(item["explicit_vs_reference_q_max_abs"]), reverse=True)
        implicit_error = {int(row["model_id"]): float(row["implicit_vs_reference_q_max_abs"]) for row in subset}
        implicit_order = sorted(ids, key=lambda model_id: implicit_error[model_id], reverse=True)
        explicit_values = [float(row["explicit_vs_reference_q_max_abs"]) for row in explicit_order]
        implicit_values = [implicit_error[model_id] for model_id in [int(row["model_id"]) for row in explicit_order]]
        by_substeps[str(substep_count)]["ranking"] = {
            "spearman_explicit_vs_implicit_q_error": _spearman(explicit_values, implicit_values),
            "worst10_overlap": len(set(int(row["model_id"]) for row in explicit_order[:10]) & set(implicit_order[:10])),
            "explicit_worst10": [int(row["model_id"]) for row in explicit_order[:10]],
            "implicit_worst10": implicit_order[:10],
        }
    return {"count": len(rows), "by_n_substeps": by_substeps}


def worst_structures(results: Iterable[Mapping[str, object]], *, limit: int = 10) -> list[dict[str, object]]:
    """Return worst structures ordered by discharge then state/flux error."""
    rows = [dict(row) for row in results]
    rows.sort(key=lambda row: (float(row["q_max_abs"]), float(row["state_max_abs"]), float(row["flux_max_abs"])), reverse=True)
    return rows[:limit]


def _gradient_probe(model_id: int, forcing: torch.Tensor, implicit_iterations: int) -> dict[str, float | bool]:
    name = "MAXWATR_1"
    base = 100.0
    step = 1.0e-3
    parameter = torch.tensor(base, dtype=forcing.dtype, requires_grad=True)
    output = simulate(model_id, forcing, {name: parameter}, initial_fraction=0.25, implicit_iterations=implicit_iterations)
    output.q.sum().backward()
    gradient = float(parameter.grad.detach())
    plus = simulate(model_id, forcing, {name: base + step}, initial_fraction=0.25, implicit_iterations=implicit_iterations).q.sum()
    minus = simulate(model_id, forcing, {name: base - step}, initial_fraction=0.25, implicit_iterations=implicit_iterations).q.sum()
    finite_difference = float(((plus - minus) / (2.0 * step)).detach())
    relative_error = abs(gradient - finite_difference) / (1.0 + abs(finite_difference))
    return {
        "gradient": gradient,
        "finite_difference": finite_difference,
        "relative_error": relative_error,
        "finite": bool(np.isfinite(gradient) and np.isfinite(finite_difference)),
    }


def solver_scan(
    executable: str,
    forcing: Mapping[str, np.ndarray] | None = None,
    *,
    iterations: Iterable[int] = (4, 8, 16, 32, 64),
) -> list[dict[str, object]]:
    """Scan fixed Newton iteration counts against one reference run per mother.

    The explicit fixed-substep path is evaluated separately by ``explicit_scan``;
    this function retains the implicit solver control scan for the existing path.
    """
    forcing = synthetic_forcing() if forcing is None else forcing
    tensor_forcing = _as_tensor_forcing(forcing)
    rows: list[dict[str, object]] = []
    for label, model_id in MOTHER_MODELS.items():
        reference = run_reference(executable, model_id, forcing)
        for iteration_count in iterations:
            started = time.perf_counter()
            output = simulate(model_id, tensor_forcing, initial_fraction=0.25, implicit_iterations=int(iteration_count))
            elapsed = time.perf_counter() - started
            q_delta = output.q.detach().cpu().numpy() - reference.q_routed
            rows.append(
                {
                    "mother": label,
                    "model_id": model_id,
                    "implicit_iterations": int(iteration_count),
                    "q_max_abs": float(np.max(np.abs(q_delta))),
                    "q_rmse": float(np.sqrt(np.mean(q_delta * q_delta))),
                    "water_balance_max_abs": float(output.max_abs_water_balance_error.detach()),
                    "gradient": _gradient_probe(model_id, tensor_forcing, int(iteration_count)),
                    "finite": bool(torch.isfinite(output.q).all() and torch.isfinite(output.states).all()),
                    "elapsed_seconds": elapsed,
                    "reference_executable_sha256": reference.metadata["executable_sha256"],
                    "explicit_substep_scan": "see explicit_scan",
                }
            )
    return rows
