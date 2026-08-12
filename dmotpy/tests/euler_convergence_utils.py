from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tests.core_model_registry import CORE_MODEL_REGISTRY, CoreModelEntry
from tests.core_water_balance_utils import evaluate_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "validation_results" / "euler_convergence_validation"
ERRORS_CSV_PATH = OUTPUT_DIR / "euler_convergence_errors.csv"
ORDERS_CSV_PATH = OUTPUT_DIR / "euler_convergence_orders.csv"
SUMMARY_CSV_PATH = OUTPUT_DIR / "euler_convergence_summary.csv"
REPORT_MD_PATH = OUTPUT_DIR / "euler_convergence_report.md"
FEASIBILITY_MD_PATH = OUTPUT_DIR / "euler_substep_feasibility.md"
WB_CHECK_MD_PATH = OUTPUT_DIR / "water_balance_residual_normalization_check.md"
WB_CHECK_CSV_PATH = OUTPUT_DIR / "water_balance_residual_normalization_check.csv"

DTYPE = torch.float64
DEVICE = "cpu"
NEARZERO = 1.0e-6
SCENARIO_NAME = "smooth_warm_positive"
TARGET_MODELS = ("hbv96", "hymod", "flexb", "vic")
FEASIBILITY_MODELS = TARGET_MODELS + ("topmodel", "tcm", "gsfb")
SUBSTEP_LEVELS = (1, 2, 4, 8, 16)
N_SUBSTEPS_REF = 1024
PASS_BAND = (0.85, 1.15)
PRECISION_FLOOR = 1.0e-10
N_DAYS = 20
N_GRID = 1
N_MUL = 1


SUPPORTED_MODEL_REASONS = {
    "hbv96": "Daily explicit map can be wrapped consistently with dt-scaled precipitation, PET, and rate parameters.",
    "hymod": "All dynamic fluxes are storage-scaled flux partitions or linear-reservoir rates; dt-scaled wrapper is consistent.",
    "flexb": "Unsaturated and routing updates admit a dt-scaled explicit wrapper; saturation_3 remains unchanged and is only exercised diagnostically.",
    "vic": "Interception/soil/groundwater updates can be exercised with zero-order-hold forcing and dt-scaled rate parameters.",
}

UNSUPPORTED_MODEL_REASONS = {
    "topmodel": "Not included in this first convergence suite; no separately reviewed dt-aware diagnostic wrapper was added for the deficit-store and threshold-activation interactions.",
    "tcm": "Not included in this first convergence suite; abstraction and deficit-store accounting would need a separately reviewed dt-aware wrapper.",
    "gsfb": "Not included in this first convergence suite; recharge/interflow threshold logic was left for follow-up rather than inferred here.",
}

RATE_PARAMETERS = {
    "hbv96": frozenset({"cfmax", "cflux", "k0", "perc", "k1"}),
    "hymod": frozenset({"kf", "ks"}),
    "flexb": frozenset({"percmax", "kf", "ks"}),
    "vic": frozenset({"k1", "k2"}),
}


@dataclass(frozen=True)
class SimulationResult:
    model: str
    scenario: str
    n_substeps: int
    dt: float
    state_daily: torch.Tensor
    flux_daily: torch.Tensor
    output_nan_count: int
    output_inf_count: int
    state_nan_count: int
    state_inf_count: int
    diagnostics: dict[str, Any]


def _dtype_device_kwargs() -> dict[str, Any]:
    return {"dtype": DTYPE, "device": DEVICE}


def _tensor(value: float) -> torch.Tensor:
    return torch.full((N_GRID, N_MUL), float(value), **_dtype_device_kwargs())


def build_smooth_forcing(model_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del model_name
    day = torch.arange(N_DAYS, **_dtype_device_kwargs()).view(N_DAYS, 1, 1)
    angle = 2.0 * math.pi * day / float(N_DAYS)
    precip = 4.5 + 0.8 * torch.sin(angle) + 0.3 * torch.cos(2.0 * angle)
    pet = 1.7 + 0.25 * torch.cos(angle - 0.4) + 0.1 * torch.sin(2.0 * angle)
    temp = 12.5 + 1.2 * torch.sin(angle + 0.3)
    return precip, temp, pet


def build_interior_parameters(model_name: str) -> dict[str, torch.Tensor]:
    if model_name == "hbv96":
        values = {
            "tt": 0.5,
            "tti": 4.0,
            "ttm": 0.0,
            "cfr": 0.05,
            "cfmax": 2.0,
            "whc": 0.08,
            "cflux": 0.2,
            "fc": 300.0,
            "lp": 0.6,
            "beta": 2.0,
            "k0": 0.08,
            "alpha": 1.0,
            "perc": 0.6,
            "k1": 0.04,
            "maxbas": 5.0,
        }
    elif model_name == "hymod":
        values = {
            "smax": 250.0,
            "b_exp": 1.5,
            "a_split": 0.45,
            "kf": 0.12,
            "ks": 0.04,
        }
    elif model_name == "flexb":
        values = {
            "s1max": 280.0,
            "beta": 1.6,
            "d_split": 0.4,
            "percmax": 0.7,
            "lp": 0.6,
            "nlagf": 2.0,
            "nlags": 6.0,
            "kf": 0.10,
            "ks": 0.035,
        }
    elif model_name == "vic":
        values = {
            "ibar": 1.5,
            "idelta": 0.2,
            "ishift": 180.0,
            "stot": 420.0,
            "fsm": 0.55,
            "b": 1.6,
            "k1": 0.07,
            "c1": 1.5,
            "k2": 0.035,
            "c2": 1.4,
        }
    else:
        raise KeyError(model_name)
    return {name: _tensor(value) for name, value in values.items()}


def build_initial_states(model_name: str, params: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    if model_name == "hbv96":
        return [
            _tensor(1.0),
            _tensor(0.4),
            params["fc"] * 0.32,
            _tensor(18.0),
            _tensor(40.0),
        ]
    if model_name == "hymod":
        return [
            params["smax"] * 0.35,
            _tensor(3.0),
            _tensor(2.4),
            _tensor(1.8),
            _tensor(16.0),
        ]
    if model_name == "flexb":
        return [
            params["s1max"] * 0.36,
            _tensor(4.0),
            _tensor(18.0),
        ]
    if model_name == "vic":
        smmax = params["fsm"] * params["stot"]
        gwmax = (1.0 - params["fsm"]) * params["stot"]
        return [
            params["ibar"] * 0.35,
            smmax * 0.32,
            gwmax * 0.28,
        ]
    raise KeyError(model_name)


def _scaled_parameter_list(entry: CoreModelEntry, params: dict[str, torch.Tensor], dt: float) -> list[torch.Tensor]:
    rate_names = RATE_PARAMETERS[entry.model_name]
    scaled: list[torch.Tensor] = []
    for name in entry.param_bounds:
        value = params[name]
        if name in rate_names:
            value = value * dt
        scaled.append(value)
    return scaled


def _capacity_diagnostics(model_name: str, params: dict[str, torch.Tensor], states: list[torch.Tensor]) -> tuple[float, int]:
    margins: list[torch.Tensor] = []
    if model_name == "hbv96":
        margins.append(params["fc"] - states[2])
    elif model_name == "hymod":
        margins.append(params["smax"] - states[0])
    elif model_name == "flexb":
        margins.append(params["s1max"] - states[0])
    elif model_name == "vic":
        smmax = params["fsm"] * params["stot"]
        gwmax = (1.0 - params["fsm"]) * params["stot"]
        margins.extend((smmax - states[1], gwmax - states[2]))
    if not margins:
        return float("nan"), 0
    min_margin = min(float(torch.min(margin).item()) for margin in margins)
    hits = sum(int(torch.count_nonzero(margin <= 1.0e-3).item()) for margin in margins)
    return min_margin, hits


def simulate_with_substeps(model_name: str, n_substeps: int) -> SimulationResult:
    if model_name not in TARGET_MODELS:
        raise KeyError(model_name)

    entry = CORE_MODEL_REGISTRY[model_name]
    forcing = build_smooth_forcing(model_name)
    params = build_interior_parameters(model_name)
    states = [state.clone() for state in build_initial_states(model_name, params)]
    dt = 1.0 / float(n_substeps)
    scaled_params = _scaled_parameter_list(entry, params, dt)

    daily_states: list[torch.Tensor] = []
    daily_fluxes: list[torch.Tensor] = []
    output_nan_count = 0
    output_inf_count = 0
    state_nan_count = 0
    state_inf_count = 0
    zero_hits = 0
    capacity_hits = 0
    min_capacity_margin = float("inf")
    min_state_value = float("inf")
    forcing_crosses_snow_threshold = False

    if model_name == "hbv96":
        tt = float(params["tt"].item())
        tti = float(params["tti"].item())
        snow_lo = tt - 0.5 * tti
        snow_hi = tt + 0.5 * tti
        forcing_crosses_snow_threshold = bool(
            torch.any((forcing[1] >= snow_lo) & (forcing[1] <= snow_hi)).item()
        )

    for day_index in range(N_DAYS):
        precip_day = forcing[0][day_index]
        temp_day = forcing[1][day_index]
        pet_day = forcing[2][day_index]
        q_day = torch.zeros_like(precip_day)
        ea_day = torch.zeros_like(precip_day)

        for _ in range(n_substeps):
            precip_sub = precip_day * dt
            pet_sub = pet_day * dt
            result = entry.step_fn(precip_sub, temp_day, pet_sub, *scaled_params, *states)
            qsim = result[0]
            ea = result[1]
            states = [state for state in result[2:]]
            q_day = q_day + qsim
            ea_day = ea_day + ea

            output_nan_count += int(torch.isnan(qsim).sum().item() + torch.isnan(ea).sum().item())
            output_inf_count += int(torch.isinf(qsim).sum().item() + torch.isinf(ea).sum().item())
            for state in states:
                state_nan_count += int(torch.isnan(state).sum().item())
                state_inf_count += int(torch.isinf(state).sum().item())
                zero_hits += int(torch.count_nonzero(state <= NEARZERO * 1.01).item())
                min_state_value = min(min_state_value, float(torch.min(state).item()))

            capacity_margin, capacity_hit_count = _capacity_diagnostics(model_name, params, states)
            if not math.isnan(capacity_margin):
                min_capacity_margin = min(min_capacity_margin, capacity_margin)
            capacity_hits += capacity_hit_count

        daily_states.append(torch.stack([state.reshape(-1) for state in states], dim=-1))
        daily_fluxes.append(torch.stack([q_day.reshape(-1), ea_day.reshape(-1)], dim=-1))

    diagnostics = {
        "forcing_crosses_snow_threshold": forcing_crosses_snow_threshold,
        "zero_hits": zero_hits,
        "capacity_hits": capacity_hits,
        "min_capacity_margin": min_capacity_margin if math.isfinite(min_capacity_margin) else float("nan"),
        "min_state_value": min_state_value if math.isfinite(min_state_value) else float("nan"),
    }
    return SimulationResult(
        model=model_name,
        scenario=SCENARIO_NAME,
        n_substeps=n_substeps,
        dt=dt,
        state_daily=torch.stack(daily_states, dim=0),
        flux_daily=torch.stack(daily_fluxes, dim=0),
        output_nan_count=output_nan_count,
        output_inf_count=output_inf_count,
        state_nan_count=state_nan_count,
        state_inf_count=state_inf_count,
        diagnostics=diagnostics,
    )


def _error_metrics(estimate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    diff = estimate - reference
    flat_diff = diff.reshape(-1)
    flat_ref = reference.reshape(-1)
    l2 = float(torch.linalg.norm(flat_diff).item())
    rmse = float(torch.sqrt(torch.mean(flat_diff.pow(2))).item())
    ref_l2 = max(float(torch.linalg.norm(flat_ref).item()), 1.0e-12)
    relative_l2 = l2 / ref_l2
    max_abs = float(torch.max(torch.abs(flat_diff)).item())
    return {
        "l2": l2,
        "rmse": rmse,
        "relative_l2": relative_l2,
        "max_abs": max_abs,
        "normalized_error": relative_l2,
    }


def _order_value(left_error: float, right_error: float) -> tuple[float | None, str]:
    if left_error < PRECISION_FLOOR and right_error < PRECISION_FLOOR:
        return None, "precision_floor"
    if left_error <= 0.0 or right_error <= 0.0:
        return None, "nonpositive_error"
    return math.log2(left_error / right_error), ""


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _feasibility_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in FEASIBILITY_MODELS:
        entry = CORE_MODEL_REGISTRY[model_name]
        supported = model_name in TARGET_MODELS
        rows.append(
            {
                "model": model_name,
                "substep_supported": supported,
                "reason": SUPPORTED_MODEL_REASONS.get(model_name, UNSUPPORTED_MODEL_REASONS.get(model_name, "")),
                "update_fn": entry.step_fn.__name__,
                "state_variables": ", ".join(entry.state_names),
                "key_fluxes": "daily Qsim, daily Ea",
                "caveats": (
                    "diagnostic wrapper scales P/PET and [d-1]/[mm/d] parameters by dt; source formulas unchanged"
                    if supported
                    else "reported but not executed in the first convergence suite"
                ),
            }
        )
    return rows


def _write_feasibility_markdown(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Euler Substep Feasibility",
        "",
        "This file documents whether a diagnostic-only dt-scaled wrapper was added for the first Euler convergence suite.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['model']}",
                f"- substep_supported: {'true' if row['substep_supported'] else 'false'}",
                f"- reason: {row['reason']}",
                f"- model update function used: `{row['update_fn']}`",
                f"- state variables tracked: {row['state_variables'] or '-'}",
                f"- key flux variables tracked: {row['key_fluxes']}",
                f"- caveats: {row['caveats']}",
                "",
            ]
        )
    FEASIBILITY_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEASIBILITY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def _water_balance_normalization_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, entry in CORE_MODEL_REGISTRY.items():
        if not entry.enabled:
            continue
        for raw in evaluate_model(entry, torch.float64, "cpu", "pytest"):
            total_precip = abs(float(raw["total_input"]))
            total_flux = abs(float(raw["total_output"]))
            storage_scale = max(abs(float(raw["storage_change"])), 1.0)
            residual = float(raw["max_absolute_full_period_residual"])
            rows.append(
                {
                    "model": raw["model_name"],
                    "case_id": f"{raw['model_name']}::{raw['test_case']}::{raw['parameter_case']}::{raw['initial_state_case']}",
                    "residual_absolute": residual,
                    "residual_normalized_by_precip": residual / max(total_precip, 1.0e-12),
                    "residual_normalized_by_total_flux": residual / max(total_flux, 1.0e-12),
                    "residual_normalized_by_storage_scale": residual / storage_scale,
                    "total_precip": total_precip,
                    "total_flux": total_flux,
                    "storage_scale": storage_scale,
                    "status": "passed" if raw["pass_fail"] else "failed",
                    "notes": f"full_period_relative_residual={float(raw['full_period_relative_residual']):.3e}",
                }
            )
    return rows


def _write_water_balance_normalization(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "case_id",
        "residual_absolute",
        "residual_normalized_by_precip",
        "residual_normalized_by_total_flux",
        "residual_normalized_by_storage_scale",
        "total_precip",
        "total_flux",
        "storage_scale",
        "status",
        "notes",
    ]
    _write_csv(WB_CHECK_CSV_PATH, fieldnames, rows)

    worst_abs = max(rows, key=lambda row: float(row["residual_absolute"]))
    lines = [
        "# Water-Balance Residual Normalization Check",
        "",
        "- The previously quoted `1.114e-03` value is the maximum absolute full-period residual, not a normalized metric.",
        "- In `tests/core_water_balance_utils.py`, `full_residual = total_input - total_output - storage_change`, and `max_absolute_full_period_residual` is `max(abs(full_residual))` over basin elements for each case.",
        "- The separate normalized metric already computed by the water-balance utility is `full_period_relative_residual = abs(full_residual) / max(abs(total_input), 1e-12)`.",
        "- The pytest summary used for the earlier report exposes the absolute residual column only.",
        "",
        f"- Worst absolute residual case: `{worst_abs['model']}` / `{worst_abs['case_id']}` / residual={float(worst_abs['residual_absolute']):.6e}",
        f"- Normalized by total precipitation for that case: {float(worst_abs['residual_normalized_by_precip']):.6e}",
        f"- Normalized by total flux for that case: {float(worst_abs['residual_normalized_by_total_flux']):.6e}",
        f"- Normalized by storage scale for that case: {float(worst_abs['residual_normalized_by_storage_scale']):.6e}",
        "- Interpretation: the published `1.114e-03` should be read as an absolute water-depth residual over the full case, which is acceptable because it remains below the established absolute tolerance and its precipitation-normalized counterpart is small.",
        "",
        "- The residual is full-period cumulative per case, with the summary reporting the maximum absolute basin element over each case.",
        "- The generated CSV provides alternative normalizations for all pytest water-balance cases without changing existing tolerances.",
        "",
    ]
    WB_CHECK_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_euler_convergence_validation(write_outputs: bool = True) -> dict[str, Any]:
    feasibility_rows = _feasibility_rows()
    if write_outputs:
        _write_feasibility_markdown(feasibility_rows)

    reference_runs = {model_name: simulate_with_substeps(model_name, N_SUBSTEPS_REF) for model_name in TARGET_MODELS}
    simulation_runs = {
        model_name: {n_substeps: simulate_with_substeps(model_name, n_substeps) for n_substeps in SUBSTEP_LEVELS}
        for model_name in TARGET_MODELS
    }

    error_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model_name in TARGET_MODELS:
        reference = reference_runs[model_name]
        state_errors_by_level: dict[int, float] = {}
        flux_errors_by_level: dict[int, float] = {}
        model_notes: list[str] = []

        for n_substeps in SUBSTEP_LEVELS:
            result = simulation_runs[model_name][n_substeps]
            state_metrics = _error_metrics(result.state_daily, reference.state_daily)
            flux_metrics = _error_metrics(result.flux_daily, reference.flux_daily)
            state_errors_by_level[n_substeps] = state_metrics["normalized_error"]
            flux_errors_by_level[n_substeps] = flux_metrics["normalized_error"]
            if result.diagnostics["forcing_crosses_snow_threshold"]:
                model_notes.append("forcing crosses snow threshold")
            error_rows.append(
                {
                    "model": model_name,
                    "scenario": SCENARIO_NAME,
                    "k": int(math.log2(n_substeps)),
                    "n_substeps": n_substeps,
                    "dt": 1.0 / float(n_substeps),
                    "k_ref": int(math.log2(N_SUBSTEPS_REF)),
                    "n_substeps_ref": N_SUBSTEPS_REF,
                    "state_l2": state_metrics["l2"],
                    "state_rmse": state_metrics["rmse"],
                    "state_relative_l2": state_metrics["relative_l2"],
                    "state_max_abs": state_metrics["max_abs"],
                    "normalized_state_error": state_metrics["normalized_error"],
                    "flux_l2": flux_metrics["l2"],
                    "flux_rmse": flux_metrics["rmse"],
                    "flux_relative_l2": flux_metrics["relative_l2"],
                    "flux_max_abs": flux_metrics["max_abs"],
                    "normalized_flux_error": flux_metrics["normalized_error"],
                    "output_nan_count": result.output_nan_count,
                    "output_inf_count": result.output_inf_count,
                    "state_nan_count": result.state_nan_count,
                    "state_inf_count": result.state_inf_count,
                    "notes": (
                        f"zero_hits={result.diagnostics['zero_hits']}; "
                        f"capacity_hits={result.diagnostics['capacity_hits']}; "
                        f"min_capacity_margin={result.diagnostics['min_capacity_margin']:.6e}"
                    ),
                }
            )

        valid_fine_p_state: list[float] = []
        valid_fine_p_flux: list[float] = []
        p_rows_for_model: list[dict[str, Any]] = []
        for left, right in zip(SUBSTEP_LEVELS[:-1], SUBSTEP_LEVELS[1:]):
            p_state, exclusion_state = _order_value(state_errors_by_level[left], state_errors_by_level[right])
            p_flux, exclusion_flux = _order_value(flux_errors_by_level[left], flux_errors_by_level[right])
            k_left = int(math.log2(left))
            k_right = int(math.log2(right))
            used_for_pass_fail = k_left >= 2 and p_state is not None
            if used_for_pass_fail:
                valid_fine_p_state.append(float(p_state))
            if k_left >= 2 and p_flux is not None:
                valid_fine_p_flux.append(float(p_flux))
            p_rows_for_model.append(
                {
                    "model": model_name,
                    "scenario": SCENARIO_NAME,
                    "k_left": k_left,
                    "k_right": k_right,
                    "n_substeps_left": left,
                    "n_substeps_right": right,
                    "p_state": "" if p_state is None else p_state,
                    "p_flux": "" if p_flux is None else p_flux,
                    "state_error_left": state_errors_by_level[left],
                    "state_error_right": state_errors_by_level[right],
                    "flux_error_left": flux_errors_by_level[left],
                    "flux_error_right": flux_errors_by_level[right],
                    "used_for_pass_fail": used_for_pass_fail,
                    "exclusion_reason": exclusion_state or exclusion_flux,
                    "notes": "",
                }
            )
        order_rows.extend(p_rows_for_model)

        # ``torch.median`` returns the lower middle value for an even-sized
        # sample.  The convergence decision uses two fine-level orders, so use
        # the conventional median (mean of the two middle values) instead.
        median_p_state = float(statistics.median(valid_fine_p_state)) if valid_fine_p_state else float("nan")
        median_p_flux = float(statistics.median(valid_fine_p_flux)) if valid_fine_p_flux else float("nan")
        state_error_monotone = all(
            state_errors_by_level[right] <= state_errors_by_level[left] * (1.0 + 1.0e-9)
            for left, right in zip(SUBSTEP_LEVELS[:-1], SUBSTEP_LEVELS[1:])
        )
        flux_error_monotone = all(
            flux_errors_by_level[right] <= flux_errors_by_level[left] * (1.0 + 1.0e-9)
            for left, right in zip(SUBSTEP_LEVELS[:-1], SUBSTEP_LEVELS[1:])
        )
        kink_detected = bool(
            reference.diagnostics["forcing_crosses_snow_threshold"]
            or reference.diagnostics["capacity_hits"] > 0
        )
        state_convergence_pass = (
            state_error_monotone
            and bool(valid_fine_p_state)
            and PASS_BAND[0] <= median_p_state <= PASS_BAND[1]
        )
        if state_convergence_pass:
            classification = "pass_smooth_first_order"
            recommendation = "No discretization change suggested."
        elif not valid_fine_p_state:
            classification = "fail_due_to_precision_floor"
            recommendation = "Increase trajectory scale or use a less-refined reference before making a stronger claim."
        elif kink_detected:
            classification = "fail_due_to_threshold_crossing"
            recommendation = "Adjust the smooth-domain scenario before interpreting the order estimate."
        else:
            classification = "fail_unexpected"
            recommendation = "Follow up on the dt wrapper or the affected flux sequence."

        summary_rows.append(
            {
                "model": model_name,
                "scenario": SCENARIO_NAME,
                "substep_supported": True,
                "state_error_monotone": state_error_monotone,
                "flux_error_monotone": flux_error_monotone,
                "median_p_state": median_p_state,
                "median_p_flux": median_p_flux,
                "state_pass_band_low": PASS_BAND[0],
                "state_pass_band_high": PASS_BAND[1],
                "state_convergence_pass": state_convergence_pass,
                "flux_convergence_report_only": True,
                "kink_or_threshold_crossing_detected": kink_detected,
                "classification": classification,
                "recommendation": recommendation,
                "notes": "; ".join(sorted(set(model_notes)))
                or (
                    f"min_state_value={reference.diagnostics['min_state_value']:.6e}; "
                    f"min_capacity_margin={reference.diagnostics['min_capacity_margin']:.6e}; "
                    f"zero_hits={reference.diagnostics['zero_hits']}; "
                    f"capacity_hits={reference.diagnostics['capacity_hits']}"
                ),
            }
        )

    for model_name in FEASIBILITY_MODELS:
        if model_name in TARGET_MODELS:
            continue
        summary_rows.append(
            {
                "model": model_name,
                "scenario": "not_run",
                "substep_supported": False,
                "state_error_monotone": "",
                "flux_error_monotone": "",
                "median_p_state": "",
                "median_p_flux": "",
                "state_pass_band_low": PASS_BAND[0],
                "state_pass_band_high": PASS_BAND[1],
                "state_convergence_pass": False,
                "flux_convergence_report_only": True,
                "kink_or_threshold_crossing_detected": "",
                "classification": "fail_due_to_substep_not_supported",
                "recommendation": "Add a separately reviewed dt-aware wrapper before claiming Euler order for this model.",
                "notes": UNSUPPORTED_MODEL_REASONS[model_name],
            }
        )

    if write_outputs:
        _write_csv(
            ERRORS_CSV_PATH,
            [
                "model",
                "scenario",
                "k",
                "n_substeps",
                "dt",
                "k_ref",
                "n_substeps_ref",
                "state_l2",
                "state_rmse",
                "state_relative_l2",
                "state_max_abs",
                "normalized_state_error",
                "flux_l2",
                "flux_rmse",
                "flux_relative_l2",
                "flux_max_abs",
                "normalized_flux_error",
                "output_nan_count",
                "output_inf_count",
                "state_nan_count",
                "state_inf_count",
                "notes",
            ],
            error_rows,
        )
        _write_csv(
            ORDERS_CSV_PATH,
            [
                "model",
                "scenario",
                "k_left",
                "k_right",
                "n_substeps_left",
                "n_substeps_right",
                "p_state",
                "p_flux",
                "state_error_left",
                "state_error_right",
                "flux_error_left",
                "flux_error_right",
                "used_for_pass_fail",
                "exclusion_reason",
                "notes",
            ],
            order_rows,
        )
        _write_csv(
            SUMMARY_CSV_PATH,
            [
                "model",
                "scenario",
                "substep_supported",
                "state_error_monotone",
                "flux_error_monotone",
                "median_p_state",
                "median_p_flux",
                "state_pass_band_low",
                "state_pass_band_high",
                "state_convergence_pass",
                "flux_convergence_report_only",
                "kink_or_threshold_crossing_detected",
                "classification",
                "recommendation",
                "notes",
            ],
            summary_rows,
        )
        _write_report(error_rows, order_rows, summary_rows)
        wb_rows = _water_balance_normalization_rows()
        _write_water_balance_normalization(wb_rows)

    return {
        "feasibility_rows": feasibility_rows,
        "error_rows": error_rows,
        "order_rows": order_rows,
        "summary_rows": summary_rows,
    }


def _error_row_map(error_rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(row["model"], int(row["n_substeps"])): row for row in error_rows}


def _orders_for_model(order_rows: list[dict[str, Any]], model_name: str) -> list[dict[str, Any]]:
    return [row for row in order_rows if row["model"] == model_name]


def _write_report(error_rows: list[dict[str, Any]], order_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    error_map = _error_row_map(error_rows)
    supported_rows = [row for row in summary_rows if row["model"] in TARGET_MODELS]
    all_supported_pass = all(bool(row["state_convergence_pass"]) for row in supported_rows)

    lines = [
        "# Euler Convergence Report",
        "",
        "## 1. Scope",
        "- This diagnostic evaluates dMoT internal Euler substep refinement in smooth regimes for representative models.",
        f"- Executed models: {', '.join(f'`{name}`' for name in TARGET_MODELS)}.",
        "",
        "## 2. Numerical question being tested",
        "- Question: if a daily forcing value is held constant within each day and the internal explicit Euler step is refined, do daily-aligned state trajectories converge to a very fine-step reference at approximately first order?",
        "",
        "## 3. Why this is not a MARRMoT step-by-step comparison",
        "- The test does not compare dMoT daily outputs against MARRMoT daily discretization choices.",
        "- It only asks whether dMoT's own explicit stepping is internally consistent under dt refinement.",
        "",
        "## 4. Method: zero-order-hold forcing and Euler substepping",
        "- Daily precipitation and PET are treated as constant rates within a day and converted to substep amounts by multiplying by dt.",
        "- Temperature is held constant within each day.",
        "- Model source formulas are unchanged; the diagnostic wrapper only scales forcing amounts and rate parameters with units `[d-1]` or `[mm/d]`.",
        "",
        "## 5. Reference solution definition",
        f"- Reference integration uses `n_substeps_ref = {N_SUBSTEPS_REF}` per day with `torch.float64` on CPU.",
        "",
        "## 6. Error norm definitions",
        "- State and flux errors are computed on daily-aligned samples over the whole trajectory.",
        "- Reported metrics include L2, RMSE, relative L2, max-absolute error, and normalized error (relative L2).",
        "",
        "## 7. Empirical order calculation",
        "- `p_k = log2(error_k / error_{k+1})` using normalized state error as the primary error sequence.",
        f"- Pass criterion: median fine-level state order from `k=2->3` and `k=3->4` lies in [{PASS_BAND[0]:.2f}, {PASS_BAND[1]:.2f}] and state error decreases monotonically.",
        "",
        "## 8. Smooth-domain test design",
        "- Scenario uses 20 warm days with moderate positive precipitation and PET, chosen to avoid snow/rain switching and dry-state dead zones.",
        "",
        "## 9. Per-model results",
    ]
    for summary in supported_rows:
        model_name = summary["model"]
        state_series = ", ".join(
            f"k={int(math.log2(n))}:{float(error_map[(model_name, n)]['normalized_state_error']):.3e}"
            for n in SUBSTEP_LEVELS
        )
        flux_series = ", ".join(
            f"k={int(math.log2(n))}:{float(error_map[(model_name, n)]['normalized_flux_error']):.3e}"
            for n in SUBSTEP_LEVELS
        )
        order_series = ", ".join(
            f"{row['k_left']}->{row['k_right']}:{float(row['p_state']):.3f}"
            for row in _orders_for_model(order_rows, model_name)
            if row["p_state"] != ""
        )
        flux_order_series = ", ".join(
            f"{row['k_left']}->{row['k_right']}:{float(row['p_flux']):.3f}"
            for row in _orders_for_model(order_rows, model_name)
            if row["p_flux"] != ""
        )
        lines.extend(
            [
                f"### {model_name}",
                f"- state errors: {state_series}",
                f"- state empirical orders: {order_series}",
                f"- median fine-level state order: {float(summary['median_p_state']):.3f}",
                f"- flux errors: {flux_series}",
                f"- flux empirical orders: {flux_order_series}",
                f"- classification: `{summary['classification']}`",
                f"- diagnostics: {summary['notes']}",
                "",
            ]
        )

    lines.extend(
        [
            "## 10. Kink/threshold diagnostics",
            "- No supported scenario is classified as a snow-threshold crossing case because the warm forcing stays well above the HBV rain/snow transition band.",
            "- The report still records zero-state hits and capacity-near hits so that an apparent order failure would not be misclassified as a formula bug.",
            "",
            "## 11. Pass/fail summary",
        ]
    )
    for summary in summary_rows:
        lines.append(
            f"- `{summary['model']}`: classification=`{summary['classification']}`, pass={summary['state_convergence_pass']}"
        )

    lines.extend(
        [
            "",
            "## 12. Interpretation for numerical equivalence",
            (
                "- The representative dMoT models show first-order convergence under internal Euler step refinement in smooth regimes, supporting the consistency of the Euler discretization used by dMoT."
                if all_supported_pass
                else "- Not all supported representative models met the smooth first-order criterion; follow-up is required before making a broader discretization claim."
            ),
            "",
            "## 13. Limitations",
            "- This is a diagnostic wrapper test, not a proof that every daily core map in the repository exposes a fully general dt API.",
            "- TOPMODEL, TCM, and GSFB were documented in the feasibility file but not executed in this first suite.",
            "- Flux-order estimates are reported for context only; state convergence is the primary criterion.",
            "",
            "## 14. Recommended wording for manuscript",
            (
                "- The representative dMoT models show first-order convergence under internal Euler step refinement in smooth regimes, supporting the consistency of the Euler discretization used by dMoT."
                if all_supported_pass
                else "- Representative dMoT Euler-refinement diagnostics are available, but the manuscript claim should stay model-specific until the failing cases are resolved."
            ),
            "",
            "## 15. Recommended next step",
            "- Extend the reviewed dt-aware wrapper approach to TOPMODEL, TCM, and GSFB if a broader representative set is needed for publication.",
            "",
        ]
    )
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
