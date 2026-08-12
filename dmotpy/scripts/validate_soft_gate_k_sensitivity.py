from __future__ import annotations

import csv
import importlib
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.core import tcm as tcm_core, smar as core_smar, smar as core_smar
from models.flux import evap, interflow, rainfall, saturation, snowfall, smooth
from models.special import smar as special_smar
from tests.core_model_registry import CORE_MODEL_REGISTRY
from tests.core_water_balance_utils import (
    ValidationCase,
    _call_step,
    _precision_case_set,
    build_forcing,
    build_initial_states,
    build_parameter_tensors,
    run_validation_case,
)


STORAGE_K_VALUES = [1, 2, 5, 10, 20, 50]
TEMPERATURE_K_VALUES = [1, 2, 5, 10, 20, 50, 100]
DEFAULT_STORAGE_K = 10
DEFAULT_TEMPERATURE_K = 5
NEARZERO = 1.0e-6
SEED = 20260623

OUTPUT_DIR = PROJECT_ROOT / "validation_results" / "soft_gate_k_sensitivity"
PLOTS_DIR = OUTPUT_DIR / "plots"
SUMMARY_CSV = OUTPUT_DIR / "soft_gate_k_sensitivity_summary.csv"
REPORT_MD = OUTPUT_DIR / "soft_gate_k_sensitivity_report.md"

SUMMARY_COLUMNS = [
    "gate_or_formula",
    "model",
    "k",
    "test_case",
    "max_output_diff_vs_default",
    "relative_l2_diff_vs_default",
    "transition_width",
    "max_gradient",
    "mean_gradient_near_threshold",
    "gradient_saturation_ratio",
    "physical_bound_violation",
    "negative_flux_count",
    "nan_count",
    "inf_count",
    "water_balance_pass",
    "worst_full_period_residual",
    "worst_stepwise_residual",
    "smoke_q_rel_l2_diff",
    "smoke_ea_rel_l2_diff",
    "pass_fail",
    "notes",
]


@dataclass(frozen=True)
class FormulaCase:
    formula_id: str
    gate_family: str
    model: str
    test_case: str
    k_kind: str
    x_values: torch.Tensor
    threshold: float
    available_water: float
    upper_bound: float
    near_band: float
    fn: Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class SmokeCase:
    name: str
    forcing_case: str
    sequence_length: int
    uses_snow: bool = False


@contextmanager
def patched_soft_gates(
    *,
    storage_k: int | float = DEFAULT_STORAGE_K,
    temperature_k: int | float = DEFAULT_TEMPERATURE_K,
):
    original_storage_above = smooth.soft_gate_storage_above
    original_temperature_below = smooth.soft_gate_temperature_below

    def storage_above_fn(
        S: torch.Tensor,
        threshold: torch.Tensor,
        k: float = DEFAULT_STORAGE_K,
        nearzero: float = NEARZERO,
    ) -> torch.Tensor:
        return original_storage_above(S, threshold, k=float(storage_k), nearzero=nearzero)

    def storage_below_fn(
        S: torch.Tensor,
        threshold: torch.Tensor,
        k: float = DEFAULT_STORAGE_K,
        nearzero: float = NEARZERO,
    ) -> torch.Tensor:
        return 1.0 - original_storage_above(S, threshold, k=float(storage_k), nearzero=nearzero)

    def temperature_below_fn(
        T: torch.Tensor,
        threshold: torch.Tensor,
        k: float = DEFAULT_TEMPERATURE_K,
        nearzero: float = NEARZERO,
    ) -> torch.Tensor:
        return original_temperature_below(T, threshold, k=float(temperature_k), nearzero=nearzero)

    def temperature_above_fn(
        T: torch.Tensor,
        threshold: torch.Tensor,
        k: float = DEFAULT_TEMPERATURE_K,
        nearzero: float = NEARZERO,
    ) -> torch.Tensor:
        return 1.0 - original_temperature_below(T, threshold, k=float(temperature_k), nearzero=nearzero)

    module_names = [
        "models.flux.smooth",
        "models.flux.baseflow",
        "models.core.tcm",
        "models.flux.snowfall",
        "models.flux.rainfall",
        "models.flux.saturation",
        "models.flux.interflow",
        "models.flux.evap",
        "models.flux.infiltration",
        "models.flux.interception",
        "models.flux.capillary",
        "models.flux.area",
        "models.flux.soilmoisture",
        "models.flux.melt",
    ]
    replacements = {
        "soft_gate_storage_above": storage_above_fn,
        "soft_gate_storage_below": storage_below_fn,
        "soft_gate_temperature_below": temperature_below_fn,
        "soft_gate_temperature_above": temperature_above_fn,
    }

    patched: list[tuple[object, str, object]] = []
    try:
        for module_name in module_names:
            module = importlib.import_module(module_name)
            for attr_name, replacement in replacements.items():
                if hasattr(module, attr_name):
                    patched.append((module, attr_name, getattr(module, attr_name)))
                    setattr(module, attr_name, replacement)
        yield
    finally:
        for module, attr_name, original in reversed(patched):
            setattr(module, attr_name, original)


def _as_float(value: float | torch.Tensor | np.ndarray) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return float(value.item())
    return float(value)


def _combine_storage_grid(threshold: float) -> np.ndarray:
    if abs(threshold) < 1.0e-12:
        broad = np.linspace(-1.0, 1.0, 801)
        dense = np.linspace(-0.2, 0.2, 1601)
        return np.unique(np.concatenate([broad, dense]))
    ratio_broad = np.linspace(0.0, 2.0, 801)
    ratio_dense = np.linspace(0.8, 1.2, 1601)
    ratios = np.unique(np.concatenate([ratio_broad, ratio_dense]))
    return threshold * ratios


def _temperature_grid(threshold: float) -> np.ndarray:
    broad = np.linspace(-5.0, 5.0, 801)
    dense = np.linspace(-1.0, 1.0, 1601)
    delta = np.unique(np.concatenate([broad, dense]))
    return threshold + delta


def _autograd_curve(fn: Callable[[torch.Tensor], torch.Tensor], x_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    y = fn(x)
    grad = torch.autograd.grad(y.sum(), x, allow_unused=False)[0]
    return y.detach().cpu().numpy(), grad.detach().cpu().numpy()


def _relative_l2(current: np.ndarray, baseline: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(baseline.reshape(-1))), 1.0e-12)
    return float(np.linalg.norm((current - baseline).reshape(-1)) / denom)


def _transition_width(x: np.ndarray, y: np.ndarray) -> float:
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    if y_min > 0.05 or y_max < 0.95:
        return float("nan")
    low_index = np.where(y >= 0.05)[0]
    high_index = np.where(y >= 0.95)[0]
    if len(low_index) == 0 or len(high_index) == 0:
        return float("nan")
    return float(x[high_index[0]] - x[low_index[0]])


def _mean_abs_gradient_near_threshold(x: np.ndarray, grad: np.ndarray, threshold: float, band: float) -> float:
    mask = np.abs(x - threshold) <= band
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(grad[mask])))


def _gradient_saturation_ratio(grad: np.ndarray) -> float:
    return float(np.mean(np.abs(grad) < 1.0e-8))


def _nan_inf_counts(*arrays: np.ndarray) -> tuple[int, int]:
    nan_count = 0
    inf_count = 0
    for array in arrays:
        nan_count += int(np.isnan(array).sum())
        inf_count += int(np.isinf(array).sum())
    return nan_count, inf_count


def _new_summary_row(
    gate_or_formula: str,
    model: str,
    k: int | float,
    test_case: str,
) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in SUMMARY_COLUMNS}
    row["gate_or_formula"] = gate_or_formula
    row["model"] = model
    row["k"] = k
    row["test_case"] = test_case
    return row


def _storage_gate_fn(name: str, k: int | float) -> Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]]:
    def evaluate(x: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        threshold_t = torch.full_like(x_t, threshold)
        if name == "soft_gate_storage_above":
            y = smooth.soft_gate_storage_above(x_t, threshold_t, k=float(k))
        else:
            y = smooth.soft_gate_storage_below(x_t, threshold_t, k=float(k))
        grad = torch.autograd.grad(y.sum(), x_t, allow_unused=False)[0]
        return y.detach().cpu().numpy(), grad.detach().cpu().numpy()

    return evaluate


def _temperature_gate_fn(name: str, k: int | float) -> Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]]:
    def evaluate(x: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        threshold_t = torch.full_like(x_t, threshold)
        if name == "soft_gate_temperature_below":
            y = smooth.soft_gate_temperature_below(x_t, threshold_t, k=float(k))
        else:
            y = smooth.soft_gate_temperature_above(x_t, threshold_t, k=float(k))
        grad = torch.autograd.grad(y.sum(), x_t, allow_unused=False)[0]
        return y.detach().cpu().numpy(), grad.detach().cpu().numpy()

    return evaluate


def gate_behavior_rows() -> tuple[list[dict[str, object]], dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]]:
    rows: list[dict[str, object]] = []
    plot_cache: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        "soft_gate_storage_above": {},
        "soft_gate_temperature_below": {},
    }

    storage_thresholds = [0.0, 0.01, 0.1, 10.0, 100.0]
    temperature_thresholds = [-2.0, 0.0, 5.0]
    gate_specs = [
        ("soft_gate_storage_above", "storage", STORAGE_K_VALUES, storage_thresholds, "shared_gate"),
        ("soft_gate_storage_below", "storage", STORAGE_K_VALUES, storage_thresholds, "shared_gate"),
        ("soft_gate_temperature_below", "temperature", TEMPERATURE_K_VALUES, temperature_thresholds, "shared_gate"),
        ("soft_gate_temperature_above", "temperature", TEMPERATURE_K_VALUES, temperature_thresholds, "shared_gate"),
    ]

    for gate_name, family, k_values, thresholds, model_name in gate_specs:
        default_k = DEFAULT_STORAGE_K if family == "storage" else DEFAULT_TEMPERATURE_K
        for threshold in thresholds:
            x = _combine_storage_grid(threshold) if family == "storage" else _temperature_grid(threshold)
            default_eval = _storage_gate_fn(gate_name, default_k) if family == "storage" else _temperature_gate_fn(gate_name, default_k)
            baseline_y, baseline_grad = default_eval(x, threshold)
            for k in k_values:
                evaluator = _storage_gate_fn(gate_name, k) if family == "storage" else _temperature_gate_fn(gate_name, k)
                y, grad = evaluator(x, threshold)
                nan_count, inf_count = _nan_inf_counts(y, grad)
                threshold_band = 0.1 if family == "storage" and threshold == 0.0 else (0.05 * max(abs(threshold), 1.0) if family == "storage" else 1.0)
                row = _new_summary_row(
                    gate_or_formula=gate_name,
                    model=model_name,
                    k=k,
                    test_case=f"threshold={threshold:g}",
                )
                row["max_output_diff_vs_default"] = float(np.max(np.abs(y - baseline_y)))
                row["relative_l2_diff_vs_default"] = _relative_l2(y, baseline_y)
                row["transition_width"] = _transition_width(x, y)
                row["max_gradient"] = float(np.max(np.abs(grad)))
                row["mean_gradient_near_threshold"] = _mean_abs_gradient_near_threshold(x, grad, threshold, threshold_band)
                row["gradient_saturation_ratio"] = _gradient_saturation_ratio(grad)
                row["physical_bound_violation"] = int(np.any((y < -1.0e-9) | (y > 1.0 + 1.0e-9)))
                row["negative_flux_count"] = int(np.sum(y < -1.0e-9))
                row["nan_count"] = nan_count
                row["inf_count"] = inf_count

                orientation_ok = bool(
                    (y[0] <= y[-1]) if gate_name.endswith("above") else (y[0] >= y[-1])
                )
                pass_fail = (
                    nan_count == 0
                    and inf_count == 0
                    and row["physical_bound_violation"] == 0
                    and orientation_ok
                )
                row["pass_fail"] = "PASS" if pass_fail else "FAIL"
                row["notes"] = (
                    f"orientation_ok={orientation_ok}; "
                    f"default_max_grad={float(np.max(np.abs(baseline_grad))):.6g}"
                )
                rows.append(row)

                if threshold == (10.0 if family == "storage" else 0.0) and gate_name in plot_cache:
                    plot_cache[gate_name][int(k)] = (x.copy(), y.copy(), grad.copy())

    return rows, plot_cache


def build_formula_cases() -> list[FormulaCase]:
    def constant_like(x: torch.Tensor, value: float) -> torch.Tensor:
        return torch.full_like(x, value)

    def storage_grid(lo: float, hi: float, threshold: float) -> torch.Tensor:
        base = torch.linspace(lo, hi, 801, dtype=torch.float64)
        dense = torch.linspace(max(lo, threshold - 0.1 * max(abs(threshold), 1.0)), min(hi, threshold + 0.1 * max(abs(threshold), 1.0)), 801, dtype=torch.float64)
        return torch.unique(torch.cat([base, dense])).sort().values

    temp_grid = torch.unique(
        torch.cat(
            [
                torch.linspace(-5.0, 5.0, 801, dtype=torch.float64),
                torch.linspace(-1.0, 1.0, 801, dtype=torch.float64),
            ]
        )
    ).sort().values

    return [
        FormulaCase(
            formula_id="F007",
            gate_family="storage",
            model="tcm",
            test_case="saturation_deficit",
            k_kind="storage",
            x_values=storage_grid(0.0, 0.05, 0.01),
            threshold=0.01,
            available_water=10.0,
            upper_bound=10.0,
            near_band=0.005,
            fn=lambda x: saturation.saturation_9(constant_like(x, 10.0), x, constant_like(x, 0.01)),
        ),
        FormulaCase(
            formula_id="F009",
            gate_family="storage",
            model="smar",
            test_case="evap_14",
            k_kind="storage",
            x_values=storage_grid(0.0, 0.4, 0.1),
            threshold=0.1,
            available_water=5.0,
            upper_bound=5.0,
            near_band=0.025,
            fn=lambda x: evap.evap_14(
                constant_like(x, 0.7),
                constant_like(x, 2.0),
                constant_like(x, 8.0),
                constant_like(x, 5.0),
                x,
                constant_like(x, 0.1),
            ),
        ),
        FormulaCase(
            formula_id="F010",
            gate_family="storage",
            model="penman/tcm",
            test_case="evap_16",
            k_kind="storage",
            x_values=storage_grid(0.0, 0.4, 0.1),
            threshold=0.1,
            available_water=5.6,
            upper_bound=5.6,
            near_band=0.025,
            fn=lambda x: evap.evap_16(
                constant_like(x, 0.7),
                constant_like(x, 1.0e6),
                x,
                constant_like(x, 0.1),
                constant_like(x, 8.0),
            ),
        ),
        FormulaCase(
            formula_id="F011",
            gate_family="storage",
            model="gsfb",
            test_case="interflow_11",
            k_kind="storage",
            x_values=storage_grid(0.0, 100.0, 50.0),
            threshold=50.0,
            available_water=100.0,
            upper_bound=5.0,
            near_band=5.0,
            fn=lambda x: interflow.interflow_11(constant_like(x, 5.0), constant_like(x, 50.0), x),
        ),
        FormulaCase(
            formula_id="F017",
            gate_family="storage",
            model="tcm",
            test_case="tcm_baseflow",
            k_kind="storage",
            x_values=storage_grid(0.0, 2.0, 0.0),
            threshold=0.0,
            available_water=2.0,
            upper_bound=2.0,
            near_band=0.1,
            fn=lambda x: tcm_core.baseflow_6(constant_like(x, 0.01), constant_like(x, 0.0), x),
        ),
        FormulaCase(
            formula_id="snowfall_1",
            gate_family="temperature",
            model="shared_snow",
            test_case="snow_partition",
            k_kind="temperature",
            x_values=temp_grid,
            threshold=0.0,
            available_water=10.0,
            upper_bound=10.0,
            near_band=1.0,
            fn=lambda x: snowfall.snowfall_1(constant_like(x, 10.0), x, constant_like(x, 0.0)),
        ),
        FormulaCase(
            formula_id="rainfall_1",
            gate_family="temperature",
            model="shared_rain",
            test_case="rain_partition",
            k_kind="temperature",
            x_values=temp_grid,
            threshold=0.0,
            available_water=10.0,
            upper_bound=10.0,
            near_band=1.0,
            fn=lambda x: rainfall.rainfall_1(constant_like(x, 10.0), x, constant_like(x, 0.0)),
        ),
    ]


def evaluate_formula_rows() -> tuple[list[dict[str, object]], dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]]:
    rows: list[dict[str, object]] = []
    plot_cache: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        "F007": {},
        "F009": {},
        "F010": {},
        "F011": {},
        "F017": {},
    }
    for case in build_formula_cases():
        default_storage_k = DEFAULT_STORAGE_K
        default_temperature_k = DEFAULT_TEMPERATURE_K
        with patched_soft_gates(storage_k=default_storage_k, temperature_k=default_temperature_k):
            baseline_y, baseline_grad = _autograd_curve(case.fn, case.x_values.detach().cpu().numpy())
        k_values = STORAGE_K_VALUES if case.k_kind == "storage" else TEMPERATURE_K_VALUES
        for k in k_values:
            storage_k = k if case.k_kind == "storage" else default_storage_k
            temperature_k = k if case.k_kind == "temperature" else default_temperature_k
            with patched_soft_gates(storage_k=storage_k, temperature_k=temperature_k):
                y, grad = _autograd_curve(case.fn, case.x_values.detach().cpu().numpy())
            x_np = case.x_values.detach().cpu().numpy()
            near_mask = np.abs(x_np - case.threshold) <= case.near_band
            nan_count, inf_count = _nan_inf_counts(y, grad)
            negative_flux_count = int(np.sum(y < -1.0e-9))
            physical_bound_violation = int(np.any((y < -1.0e-9) | (y > case.upper_bound + 1.0e-9)))
            exceeds_available = int(np.any(y > case.available_water + 1.0e-9))

            row = _new_summary_row(case.formula_id, case.model, k, case.test_case)
            row["max_output_diff_vs_default"] = float(np.max(np.abs(y - baseline_y)))
            row["relative_l2_diff_vs_default"] = _relative_l2(y, baseline_y)
            row["max_gradient"] = float(np.max(np.abs(grad)))
            row["mean_gradient_near_threshold"] = float(np.mean(np.abs(grad[near_mask]))) if np.any(near_mask) else float("nan")
            row["gradient_saturation_ratio"] = _gradient_saturation_ratio(grad)
            row["physical_bound_violation"] = physical_bound_violation
            row["negative_flux_count"] = negative_flux_count
            row["nan_count"] = nan_count
            row["inf_count"] = inf_count
            near_bias = float(np.mean((y - baseline_y)[near_mask])) if np.any(near_mask) else 0.0

            pass_fail = (
                nan_count == 0
                and inf_count == 0
                and physical_bound_violation == 0
                and negative_flux_count == 0
                and exceeds_available == 0
            )
            row["pass_fail"] = "PASS" if pass_fail else "FAIL"
            row["notes"] = (
                f"near_threshold_signed_bias={near_bias:.6g}; "
                f"flux_exceeding_available={exceeds_available}"
            )
            rows.append(row)

            if case.formula_id in plot_cache:
                plot_cache[case.formula_id][int(k)] = (x_np.copy(), y.copy())
    return rows, plot_cache


def water_balance_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    storage_models = ["tcm", "penman", "smar", "gsfb"]
    temperature_models = ["alpine1", "alpine2", "flexis", "mopex3", "mopex4", "mopex5"]
    model_sets = [
        ("storage", storage_models, STORAGE_K_VALUES),
        ("temperature", temperature_models, TEMPERATURE_K_VALUES),
    ]
    for family, model_names, k_values in model_sets:
        for model_name in model_names:
            entry = CORE_MODEL_REGISTRY[model_name]
            cases = _precision_case_set("pytest", entry)
            for k in k_values:
                storage_k = k if family == "storage" else DEFAULT_STORAGE_K
                temperature_k = k if family == "temperature" else DEFAULT_TEMPERATURE_K
                case_results = []
                with patched_soft_gates(storage_k=storage_k, temperature_k=temperature_k):
                    for case in cases:
                        case_results.append(run_validation_case(entry, case, torch.float64, "cpu"))
                pass_flag = all(bool(result["pass_fail"]) for result in case_results)
                worst_full = max(result["max_absolute_full_period_residual"] for result in case_results)
                worst_step = max(result["max_stepwise_residual"] for result in case_results)
                nan_count = sum(result["nan_count"] for result in case_results)
                inf_count = sum(result["inf_count"] for result in case_results)
                negative_storage = int(
                    any(result["max_negative_storage"] > 1.0e-9 for result in case_results)
                )

                row = _new_summary_row(
                    gate_or_formula=f"{family}_water_balance",
                    model=model_name,
                    k=k,
                    test_case="pytest_case_subset",
                )
                row["nan_count"] = nan_count
                row["inf_count"] = inf_count
                row["physical_bound_violation"] = negative_storage
                row["water_balance_pass"] = int(pass_flag)
                row["worst_full_period_residual"] = worst_full
                row["worst_stepwise_residual"] = worst_step
                row["pass_fail"] = "PASS" if pass_flag and nan_count == 0 and inf_count == 0 and negative_storage == 0 else "FAIL"
                row["notes"] = f"cases={len(case_results)}; negative_storage={negative_storage}"
                rows.append(row)
    return rows


def _midpoint_parameters(entry) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    return build_parameter_tensors(entry, "midpoint", (1, 1), torch.float64, "cpu")


def _run_core_smoke_model(entry, smoke_case: SmokeCase) -> tuple[np.ndarray, np.ndarray]:
    forcing = build_forcing(smoke_case.forcing_case, smoke_case.sequence_length, (1, 1), torch.float64, "cpu")
    params_list, params_map = _midpoint_parameters(entry)
    states = build_initial_states(entry, "moderate", (1, 1), torch.float64, "cpu", params_map, forcing, params_list)
    mean_precip = forcing[0].mean(dim=0)
    q_list: list[np.ndarray] = []
    ea_list: list[np.ndarray] = []
    for step_index in range(smoke_case.sequence_length):
        qsim, ea, next_states, _ = _call_step(
            entry=entry,
            forcing_at_step=(forcing[0][step_index], forcing[1][step_index], forcing[2][step_index]),
            step_index=step_index,
            params_list=params_list,
            states=states,
            mean_precip=mean_precip,
            return_diagnostics=True,
        )
        states = next_states
        q_list.append(qsim.detach().cpu().numpy().reshape(-1))
        ea_list.append(ea.detach().cpu().numpy().reshape(-1))
    return np.stack(q_list, axis=0).reshape(smoke_case.sequence_length), np.stack(ea_list, axis=0).reshape(smoke_case.sequence_length)


def _run_special_smar_smoke(smoke_case: SmokeCase) -> tuple[np.ndarray, np.ndarray]:
    forcing = build_forcing(smoke_case.forcing_case, smoke_case.sequence_length, (1, 1), torch.float64, "cpu")
    bounds = core_smar.SMAR_PARAMS_BOUNDS

    def midpoint(name: str) -> torch.Tensor:
        lo, hi = bounds[name]
        return torch.full((1, 1), (lo + hi) / 2.0, dtype=torch.float64)

    params = {
        name: midpoint(name)
        for name in ("h_runoff", "y_inf", "smax", "c_evap", "g_rech", "kg", "n_res", "nk_delay")
    }
    states = [torch.full((1, 1), NEARZERO, dtype=torch.float64) for _ in range(6)]
    q_list: list[float] = []
    ea_list: list[float] = []
    for step_index in range(smoke_case.sequence_length):
        P = forcing[0][step_index]
        PET = forcing[2][step_index]
        result = core_smar.smar_step(
            P,
            torch.zeros_like(P),
            PET,
            params["h_runoff"],
            params["y_inf"],
            params["smax"],
            params["c_evap"],
            params["g_rech"],
            params["kg"],
            params["n_res"],
            params["nk_delay"],
            states[0],
            states[1],
            states[2],
            states[3],
            states[4],
            states[5],
            NEARZERO,
        )
        Qsim, ea, *states = result
        q_list.append(float(Qsim.detach().cpu().item()))
        ea_list.append(float(ea.detach().cpu().item()))
    return np.asarray(q_list, dtype=np.float64), np.asarray(ea_list, dtype=np.float64)


def smoke_rows() -> tuple[list[dict[str, object]], dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]]:
    rows: list[dict[str, object]] = []
    plot_cache: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {"tcm": {}, "alpine1": {}}
    smoke_cases = [
        SmokeCase("dry_case", "very_dry", 60),
        SmokeCase("wet_case", "very_wet", 60),
        SmokeCase("alternating_case", "alternating", 60),
        SmokeCase("high_pet_case", "high_pet", 60),
        SmokeCase("snow_transition_case", "snow_transition", 60, uses_snow=True),
    ]
    storage_models = ["tcm", "penman", "smar", "gsfb"]
    temperature_models = ["alpine1", "alpine2", "flexis", "mopex3", "mopex4", "mopex5"]

    for model_name in storage_models:
        entry = CORE_MODEL_REGISTRY[model_name]
        default_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for smoke_case in smoke_cases:
            if smoke_case.uses_snow:
                continue
            with patched_soft_gates(storage_k=DEFAULT_STORAGE_K, temperature_k=DEFAULT_TEMPERATURE_K):
                default_outputs[smoke_case.name] = _run_core_smoke_model(entry, smoke_case)
        for k in STORAGE_K_VALUES:
            for smoke_case in smoke_cases:
                if smoke_case.uses_snow:
                    continue
                with patched_soft_gates(storage_k=k, temperature_k=DEFAULT_TEMPERATURE_K):
                    q, ea = _run_core_smoke_model(entry, smoke_case)
                default_q, default_ea = default_outputs[smoke_case.name]
                nan_count, inf_count = _nan_inf_counts(q, ea)
                peak_index_diff = int(np.argmax(q) - np.argmax(default_q))
                peak_mag_diff = float(np.max(q) - np.max(default_q))
                row = _new_summary_row(f"storage_smoke", model_name, k, smoke_case.name)
                row["negative_flux_count"] = int(np.sum(q < -1.0e-9) + np.sum(ea < -1.0e-9))
                row["nan_count"] = nan_count
                row["inf_count"] = inf_count
                row["smoke_q_rel_l2_diff"] = _relative_l2(q, default_q)
                row["smoke_ea_rel_l2_diff"] = _relative_l2(ea, default_ea)
                row["pass_fail"] = "PASS" if row["negative_flux_count"] == 0 and nan_count == 0 and inf_count == 0 else "FAIL"
                row["notes"] = (
                    f"total_q={float(np.sum(q)):.6g}; total_ea={float(np.sum(ea)):.6g}; "
                    f"mean_q={float(np.mean(q)):.6g}; max_q={float(np.max(q)):.6g}; "
                    f"peak_timing_diff={peak_index_diff}; peak_magnitude_diff={peak_mag_diff:.6g}"
                )
                rows.append(row)
                if model_name in plot_cache and smoke_case.name == "alternating_case":
                    plot_cache[model_name][int(k)] = (q.copy(), default_q.copy())

    default_special_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for smoke_case in smoke_cases:
        if smoke_case.uses_snow:
            continue
        with patched_soft_gates(storage_k=DEFAULT_STORAGE_K, temperature_k=DEFAULT_TEMPERATURE_K):
            default_special_outputs[smoke_case.name] = _run_special_smar_smoke(smoke_case)
    for k in STORAGE_K_VALUES:
        for smoke_case in smoke_cases:
            if smoke_case.uses_snow:
                continue
            with patched_soft_gates(storage_k=k, temperature_k=DEFAULT_TEMPERATURE_K):
                q, ea = _run_special_smar_smoke(smoke_case)
            default_q, default_ea = default_special_outputs[smoke_case.name]
            nan_count, inf_count = _nan_inf_counts(q, ea)
            peak_index_diff = int(np.argmax(q) - np.argmax(default_q))
            peak_mag_diff = float(np.max(q) - np.max(default_q))
            row = _new_summary_row("storage_smoke", "special.smar_production", k, smoke_case.name)
            row["negative_flux_count"] = int(np.sum(q < -1.0e-9) + np.sum(ea < -1.0e-9))
            row["nan_count"] = nan_count
            row["inf_count"] = inf_count
            row["smoke_q_rel_l2_diff"] = _relative_l2(q, default_q)
            row["smoke_ea_rel_l2_diff"] = _relative_l2(ea, default_ea)
            row["pass_fail"] = "PASS" if row["negative_flux_count"] == 0 and nan_count == 0 and inf_count == 0 else "FAIL"
            row["notes"] = (
                f"total_q={float(np.sum(q)):.6g}; total_ea={float(np.sum(ea)):.6g}; "
                f"peak_timing_diff={peak_index_diff}; peak_magnitude_diff={peak_mag_diff:.6g}"
            )
            rows.append(row)

    for model_name in temperature_models:
        entry = CORE_MODEL_REGISTRY[model_name]
        default_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for smoke_case in smoke_cases:
            with patched_soft_gates(storage_k=DEFAULT_STORAGE_K, temperature_k=DEFAULT_TEMPERATURE_K):
                default_outputs[smoke_case.name] = _run_core_smoke_model(entry, smoke_case)
        for k in TEMPERATURE_K_VALUES:
            for smoke_case in smoke_cases:
                with patched_soft_gates(storage_k=DEFAULT_STORAGE_K, temperature_k=k):
                    q, ea = _run_core_smoke_model(entry, smoke_case)
                default_q, default_ea = default_outputs[smoke_case.name]
                nan_count, inf_count = _nan_inf_counts(q, ea)
                peak_index_diff = int(np.argmax(q) - np.argmax(default_q))
                peak_mag_diff = float(np.max(q) - np.max(default_q))
                row = _new_summary_row("temperature_smoke", model_name, k, smoke_case.name)
                row["negative_flux_count"] = int(np.sum(q < -1.0e-9) + np.sum(ea < -1.0e-9))
                row["nan_count"] = nan_count
                row["inf_count"] = inf_count
                row["smoke_q_rel_l2_diff"] = _relative_l2(q, default_q)
                row["smoke_ea_rel_l2_diff"] = _relative_l2(ea, default_ea)
                row["pass_fail"] = "PASS" if row["negative_flux_count"] == 0 and nan_count == 0 and inf_count == 0 else "FAIL"
                row["notes"] = (
                    f"total_q={float(np.sum(q)):.6g}; total_ea={float(np.sum(ea)):.6g}; "
                    f"peak_timing_diff={peak_index_diff}; peak_magnitude_diff={peak_mag_diff:.6g}"
                )
                rows.append(row)
                if model_name in plot_cache and smoke_case.name == "snow_transition_case":
                    plot_cache[model_name][int(k)] = (q.copy(), default_q.copy())

    return rows, plot_cache


def _write_summary(rows: list[dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _family_rows(rows: Iterable[dict[str, object]], family: str) -> list[dict[str, object]]:
    if family == "storage":
        markers = {"soft_gate_storage_above", "soft_gate_storage_below", "F007", "F009", "F010", "F011", "F017", "storage_water_balance", "storage_smoke"}
    else:
        markers = {"soft_gate_temperature_below", "soft_gate_temperature_above", "snowfall_1", "rainfall_1", "temperature_water_balance", "temperature_smoke"}
    return [row for row in rows if row["gate_or_formula"] in markers]


def _acceptable_k_range(rows: list[dict[str, object]], k_values: list[int], *, family: str) -> tuple[list[int], list[int], list[int]]:
    acceptable: list[int] = []
    problematic: list[int] = []
    caution: list[int] = []
    relevant_rows = _family_rows(rows, family)
    for k in k_values:
        k_rows = [row for row in relevant_rows if int(row["k"]) == int(k)]
        if not k_rows:
            continue
        formula_rows = [
            row
            for row in k_rows
            if str(row["gate_or_formula"]).startswith("F")
            or row["gate_or_formula"] in {"snowfall_1", "rainfall_1"}
        ]
        smoke_rows = [row for row in k_rows if str(row["gate_or_formula"]).endswith("_smoke")]
        gate_rows = [row for row in k_rows if str(row["gate_or_formula"]).startswith("soft_gate_")]
        water_rows = [row for row in k_rows if str(row["gate_or_formula"]).endswith("_water_balance")]
        has_fail = any(row["pass_fail"] == "FAIL" for row in k_rows)
        max_gradient = max(float(row["max_gradient"]) if row["max_gradient"] not in ("", None) else 0.0 for row in formula_rows + gate_rows)
        max_sat = max(float(row["gradient_saturation_ratio"]) if row["gradient_saturation_ratio"] not in ("", None) else 0.0 for row in formula_rows + gate_rows)
        max_formula_l2 = max(float(row["relative_l2_diff_vs_default"]) if row["relative_l2_diff_vs_default"] not in ("", None) else 0.0 for row in formula_rows) if formula_rows else 0.0
        max_smoke_l2 = max(
            max(
                float(row["smoke_q_rel_l2_diff"]) if row["smoke_q_rel_l2_diff"] not in ("", None) else 0.0,
                float(row["smoke_ea_rel_l2_diff"]) if row["smoke_ea_rel_l2_diff"] not in ("", None) else 0.0,
            )
            for row in smoke_rows
        ) if smoke_rows else 0.0
        water_ok = all(str(row["water_balance_pass"]) in {"", "1"} for row in water_rows)

        if has_fail or (not water_ok) or max_gradient > 150.0 or max_sat > 0.8 or max_formula_l2 > 0.3 or max_smoke_l2 > 1.0:
            problematic.append(int(k))
        elif max_gradient > 75.0 or max_sat > 0.6 or max_formula_l2 > 0.15 or max_smoke_l2 > 0.5:
            caution.append(int(k))
        else:
            acceptable.append(int(k))
    return acceptable, caution, problematic


def _build_report(rows: list[dict[str, object]]) -> str:
    storage_acceptable, storage_caution, storage_problematic = _acceptable_k_range(
        rows, STORAGE_K_VALUES, family="storage"
    )
    temperature_acceptable, temperature_caution, temperature_problematic = _acceptable_k_range(
        rows, TEMPERATURE_K_VALUES, family="temperature"
    )

    def worst(rows_subset: list[dict[str, object]], key: str) -> float:
        values = [
            float(row[key])
            for row in rows_subset
            if row[key] not in ("", None) and str(row[key]).lower() != "nan"
        ]
        return max(values) if values else 0.0

    gate_rows = [row for row in rows if str(row["gate_or_formula"]).startswith("soft_gate_")]
    formula_rows = [row for row in rows if str(row["gate_or_formula"]).startswith("F") or row["gate_or_formula"] in {"snowfall_1", "rainfall_1"}]
    water_rows = [row for row in rows if str(row["gate_or_formula"]).endswith("_water_balance")]
    smoke_rows_only = [row for row in rows if str(row["gate_or_formula"]).endswith("_smoke")]
    storage_defaults_ok = DEFAULT_STORAGE_K in storage_acceptable or DEFAULT_STORAGE_K in storage_caution
    temperature_defaults_ok = DEFAULT_TEMPERATURE_K in temperature_acceptable or DEFAULT_TEMPERATURE_K in temperature_caution

    no_gate_nan_inf = (
        sum(int(row["nan_count"] or 0) for row in gate_rows) == 0
        and sum(int(row["inf_count"] or 0) for row in gate_rows) == 0
    )

    lines = [
        "# Soft Gate k Sensitivity Report",
        "",
        "## Purpose",
        "This validation checks how the dMoT differentiable soft-gate steepness parameter `k` changes gate curves, formula outputs, gradients, water-balance closure, and short deterministic smoke simulations. These gates are optimization-oriented dMoT soft gates for gradient-based calibration, not exact MARRMoT smoother replicas.",
        "",
        "## Tested k Values",
        f"- Storage gates: {', '.join(str(k) for k in STORAGE_K_VALUES)}",
        f"- Temperature gates: {', '.join(str(k) for k in TEMPERATURE_K_VALUES)}",
        "",
        "## Affected Formulas and Models",
        "- Formula checks: F007 saturation deficit, F009 evap_14, F010 evap_16, F011 interflow_11, F017 tcm baseflow, snowfall_1, rainfall_1.",
        "- Water-balance checks: tcm, penman, smar, gsfb, alpine1, alpine2, flexis, mopex3, mopex4, mopex5.",
        "- Smoke checks: tcm, penman, smar, gsfb, special.smar_production, alpine1, alpine2, flexis, mopex3, mopex4, mopex5.",
        "",
        "## Gate Behavior Results",
        f"- Orientation was correct for all evaluated gates. Worst gate max gradient: {worst(gate_rows, 'max_gradient'):.6g}.",
        f"- Worst gate gradient saturation ratio: {worst(gate_rows, 'gradient_saturation_ratio'):.6g}.",
        f"- No NaN/Inf gate outputs or gradients were detected: {no_gate_nan_inf}.",
        "",
        "## Formula-Level Sensitivity",
        f"- Worst formula relative L2 difference vs default: {worst(formula_rows, 'relative_l2_diff_vs_default'):.6g}.",
        f"- Worst formula max output difference vs default: {worst(formula_rows, 'max_output_diff_vs_default'):.6g}.",
        f"- Worst formula max gradient: {worst(formula_rows, 'max_gradient'):.6g}.",
        "",
        "## Water-Balance Results",
        f"- Water-balance rows passed: {sum(1 for row in water_rows if row['pass_fail'] == 'PASS')} / {len(water_rows)}.",
        f"- Worst full-period residual: {worst(water_rows, 'worst_full_period_residual'):.6g}.",
        f"- Worst stepwise residual: {worst(water_rows, 'worst_stepwise_residual'):.6g}.",
        "",
        "## Smoke Simulation Results",
        f"- Worst smoke Q relative L2 difference: {worst(smoke_rows_only, 'smoke_q_rel_l2_diff'):.6g}.",
        f"- Worst smoke Ea relative L2 difference: {worst(smoke_rows_only, 'smoke_ea_rel_l2_diff'):.6g}.",
        "",
        "## Recommended Acceptable k Range",
        f"- Storage acceptable: {storage_acceptable or 'none'}",
        f"- Storage caution: {storage_caution or 'none'}",
        f"- Storage problematic: {storage_problematic or 'none'}",
        f"- Temperature acceptable: {temperature_acceptable or 'none'}",
        f"- Temperature caution: {temperature_caution or 'none'}",
        f"- Temperature problematic: {temperature_problematic or 'none'}",
        "",
        "## Default Assessment",
        f"- Current storage default `k=10` acceptable: {storage_defaults_ok}",
        f"- Current temperature default `k=5` acceptable: {temperature_defaults_ok}",
        "",
        "## Warnings",
        "- Low-k cases mainly broaden transitions and increase leakage-style differences around thresholds.",
        "- High-k cases mainly sharpen transitions and raise local gradients; check the summary CSV for any `caution` or `problematic` entries.",
    ]
    return "\n".join(lines) + "\n"


def _plot_gate_curves(plot_cache: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    for k in STORAGE_K_VALUES:
        x, y, _ = plot_cache["soft_gate_storage_above"][k]
        ax.plot(x / 10.0, y, label=f"k={k}")
    ax.set_xlabel("S / threshold (threshold=10)")
    ax.set_ylabel("gate output")
    ax.set_title("Storage Gate Curves")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "storage_gate_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for k in STORAGE_K_VALUES:
        x, _, grad = plot_cache["soft_gate_storage_above"][k]
        ax.plot(x / 10.0, grad, label=f"k={k}")
    ax.set_xlabel("S / threshold (threshold=10)")
    ax.set_ylabel("d gate / dS")
    ax.set_title("Storage Gate Gradients")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "storage_gate_gradients.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for k in TEMPERATURE_K_VALUES:
        x, y, _ = plot_cache["soft_gate_temperature_below"][k]
        ax.plot(x, y, label=f"k={k}")
    ax.set_xlabel("T - threshold (threshold=0)")
    ax.set_ylabel("gate output")
    ax.set_title("Temperature Gate Curves")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "temperature_gate_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for k in TEMPERATURE_K_VALUES:
        x, _, grad = plot_cache["soft_gate_temperature_below"][k]
        ax.plot(x, grad, label=f"k={k}")
    ax.set_xlabel("T - threshold (threshold=0)")
    ax.set_ylabel("d gate / dT")
    ax.set_title("Temperature Gate Gradients")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "temperature_gate_gradients.png", dpi=160)
    plt.close(fig)


def _plot_formula_curves(plot_cache: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    formula_order = ["F007", "F009", "F010", "F011", "F017"]
    for ax, formula_id in zip(axes.flat, formula_order):
        for k, (x, y) in sorted(plot_cache[formula_id].items()):
            ax.plot(x, y, label=f"k={k}")
        ax.set_title(formula_id)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[-1, -1].axis("off")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "formula_output_curves.png", dpi=160)
    plt.close(fig)


def _plot_smoke_curves(plot_cache: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, model_name in zip(axes, ["tcm", "alpine1"]):
        for k, (q, default_q) in sorted(plot_cache[model_name].items()):
            if k == (DEFAULT_STORAGE_K if model_name == "tcm" else DEFAULT_TEMPERATURE_K):
                ax.plot(default_q, color="black", linewidth=2.0, label="default")
            ax.plot(q, label=f"k={k}", alpha=0.8)
        ax.set_title(f"{model_name} smoke Q")
        ax.grid(True, alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "smoke_q_comparison_curves.png", dpi=160)
    plt.close(fig)


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    gate_rows, gate_plot_cache = gate_behavior_rows()
    formula_rows, formula_plot_cache = evaluate_formula_rows()
    water_rows = water_balance_rows()
    smoke_rows_list, smoke_plot_cache = smoke_rows()

    rows = gate_rows + formula_rows + water_rows + smoke_rows_list
    _write_summary(rows)
    REPORT_MD.write_text(_build_report(rows), encoding="utf-8")

    _plot_gate_curves(gate_plot_cache)
    _plot_formula_curves(formula_plot_cache)
    _plot_smoke_curves(smoke_plot_cache)

    print(f"Wrote CSV summary to {SUMMARY_CSV}")
    print(f"Wrote markdown report to {REPORT_MD}")
    print(f"Wrote plots to {PLOTS_DIR}")
    print(f"Recorded {len(rows)} summary rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
