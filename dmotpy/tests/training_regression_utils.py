from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.benchmark.core.metrics import kge_components, nse  # noqa: E402
from project.benchmark.core.objectives import objective_loss  # noqa: E402
from tests.core_model_registry import CORE_MODEL_REGISTRY, CoreModelEntry  # noqa: E402
from tests.core_water_balance_utils import _call_step, build_initial_states  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "dmotpy" / "validation_results" / "training_regression_after_validation"
TRAINING_SMOKE_SUMMARY_CSV = OUTPUT_DIR / "training_smoke_summary.csv"
FLEX_SATURATION3_TRAINING_CSV = OUTPUT_DIR / "flex_saturation3_training_regression.csv"
MEDIUM_CONTEXT_MONITORING_CSV = OUTPUT_DIR / "medium_context_training_monitoring.csv"
ALL_MODEL_SMOKE_SUMMARY_CSV = OUTPUT_DIR / "all_model_calibration_smoke_summary.csv"
REPORT_MD_PATH = OUTPUT_DIR / "training_regression_after_validation_report.md"

DEFAULT_DTYPE = torch.float64
DEFAULT_DEVICE = "cpu"
DEFAULT_OBJECTIVE = "KGE"
DEFAULT_SEED = 20260625

STAGE1_TARGET_MODELS = ("flexb", "flexi", "flexis", "tcm", "gsfb", "topmodel", "hbv96", "hymod", "vic")
FLEX_MODELS = ("flexb", "flexi", "flexis")
MEDIUM_CONTEXTS = {
    "tcm": ("baseflow_6 / tcm", ("k2",)),
    "gsfb": ("baseflow_9 / gsfb", ("b", "dpf", "sdrmax")),
    "topmodel": ("interflow_10 / topmodel", ("suzmax", "st", "kd")),
}


@dataclass(frozen=True)
class TrainingConfig:
    n_timesteps: int
    n_basins: int
    n_optimizer_steps: int
    learning_rate: float
    objective: str = DEFAULT_OBJECTIVE


TRAINING_SMOKE_CONFIG = TrainingConfig(n_timesteps=30, n_basins=2, n_optimizer_steps=8, learning_rate=0.1)
FLEX_TRAINING_CONFIG = TrainingConfig(n_timesteps=30, n_basins=2, n_optimizer_steps=8, learning_rate=0.1)
MEDIUM_CONTEXT_CONFIG = TrainingConfig(n_timesteps=30, n_basins=2, n_optimizer_steps=8, learning_rate=0.1)
ALL_MODEL_SMOKE_CONFIG = TrainingConfig(n_timesteps=30, n_basins=2, n_optimizer_steps=6, learning_rate=0.1)
PYTEST_SMOKE_CONFIG = TrainingConfig(n_timesteps=20, n_basins=2, n_optimizer_steps=4, learning_rate=0.1)


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float64:
        return "float64"
    if dtype == torch.float32:
        return "float32"
    return str(dtype).replace("torch.", "")


def discover_runnable_models() -> dict[str, CoreModelEntry]:
    return {name: entry for name, entry in CORE_MODEL_REGISTRY.items() if entry.enabled}


def make_synthetic_forcing(
    entry: CoreModelEntry,
    n_timesteps: int,
    n_basins: int,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (n_timesteps, n_basins, 1)
    time_axis = torch.arange(n_timesteps, dtype=dtype, device=device).view(n_timesteps, 1, 1)
    basin_axis = torch.arange(n_basins, dtype=dtype, device=device).view(1, n_basins, 1)

    precip = (4.0 + 0.22 * time_axis + 0.12 * basin_axis).expand(shape)
    pet = (1.2 + 0.07 * (time_axis % 5.0) + 0.04 * basin_axis).expand(shape)

    if entry.model_name in {"flexis", "hbv96"}:
        temperature = (8.0 + 0.2 * basin_axis + 0.03 * time_axis).expand(shape)
    else:
        temperature = (5.5 + 0.2 * basin_axis + 0.02 * time_axis).expand(shape)

    return precip, temperature, pet


def _parameter_fraction_base(param_name: str, lower_bound: float) -> float:
    special = {
        "tt": 0.20,
        "ttm": 0.20,
        "tti": 0.30,
        "lp": 0.40,
        "st": 0.40,
        "ndc": 0.40,
        "fsm": 0.40,
        "ishift": 0.20,
    }
    if param_name in special:
        return special[param_name]
    if lower_bound == 0.0:
        return 0.30
    return 0.45


def _fraction_matrix(
    entry: CoreModelEntry,
    mode: str,
    n_basins: int,
    *,
    beta_near_zero: bool = False,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    basin_offset = torch.linspace(-0.04, 0.04, n_basins, dtype=dtype).view(1, n_basins)
    rows = []

    for index, (param_name, (lower_bound, _)) in enumerate(entry.param_bounds.items()):
        base = _parameter_fraction_base(param_name, float(lower_bound))
        if mode == "true":
            frac = torch.full((1, n_basins), base, dtype=dtype) + basin_offset
        elif mode == "init":
            delta = 0.18 if index % 2 == 0 else -0.14
            frac = torch.full((1, n_basins), base + delta, dtype=dtype) - 0.5 * basin_offset
        else:
            raise KeyError(mode)

        if beta_near_zero and param_name == "beta":
            frac = torch.full((1, n_basins), 1.0e-8, dtype=dtype)

        rows.append(frac.clamp(1.0e-12, 1.0 - 1.0e-6))

    return torch.cat(rows, dim=0)


def _logit(fraction: torch.Tensor) -> torch.Tensor:
    return torch.log(fraction) - torch.log1p(-fraction)


def _raw_to_physical_parameters(
    entry: CoreModelEntry,
    raw_parameters: torch.Tensor,
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    params_list: list[torch.Tensor] = []
    params_map: dict[str, torch.Tensor] = {}

    for index, (param_name, (lower_bound, upper_bound)) in enumerate(entry.param_bounds.items()):
        fraction = torch.sigmoid(raw_parameters[index]).unsqueeze(-1)
        physical = float(lower_bound) + fraction * (float(upper_bound) - float(lower_bound))
        params_list.append(physical)
        params_map[param_name.lower()] = physical

    return params_list, params_map


def _run_rollout(
    entry: CoreModelEntry,
    forcing: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    params_list: list[torch.Tensor],
    initial_states: list[torch.Tensor],
) -> tuple[torch.Tensor, int, int]:
    discharge = []
    output_nan_count = 0
    output_inf_count = 0
    states = [state.clone() for state in initial_states]
    mean_precip = forcing[0].mean(dim=0)

    for step_index in range(forcing[0].shape[0]):
        qsim, _, states, _ = _call_step(
            entry=entry,
            forcing_at_step=(forcing[0][step_index], forcing[1][step_index], forcing[2][step_index]),
            step_index=step_index,
            params_list=params_list,
            states=states,
            mean_precip=mean_precip,
            return_diagnostics=False,
        )
        discharge.append(qsim)
        output_nan_count += int(torch.isnan(qsim).sum().item())
        output_inf_count += int(torch.isinf(qsim).sum().item())

    return torch.stack(discharge, dim=0), output_nan_count, output_inf_count


def _make_target_discharge(
    entry: CoreModelEntry,
    forcing: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    config: TrainingConfig,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    true_raw = _logit(_fraction_matrix(entry, "true", config.n_basins))
    true_params, true_param_map = _raw_to_physical_parameters(entry, true_raw)
    initial_states = build_initial_states(
        entry,
        "small",
        (config.n_basins, 1),
        DEFAULT_DTYPE,
        DEFAULT_DEVICE,
        true_param_map,
        forcing,
        true_params,
    )
    target_discharge, _, _ = _run_rollout(entry, forcing, true_params, initial_states)
    return target_discharge, [state.detach().clone() for state in initial_states]


def _final_metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    pred_np = prediction.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    nse_values = []
    kge_values = []

    for basin_index in range(pred_np.shape[1]):
        pred_basin = pred_np[:, basin_index]
        target_basin = target_np[:, basin_index]
        nse_values.append(nse(pred_basin, target_basin))
        kge_values.append(kge_components(pred_basin, target_basin)["KGE"])

    mean_nse = sum(value for value in nse_values if math.isfinite(value)) / max(
        sum(math.isfinite(value) for value in nse_values), 1
    )
    mean_kge = sum(value for value in kge_values if math.isfinite(value)) / max(
        sum(math.isfinite(value) for value in kge_values), 1
    )
    return float(mean_nse), float(mean_kge)


def _failed_basin_count(prediction: torch.Tensor) -> int:
    nonfinite = ~torch.isfinite(prediction)
    if prediction.dim() == 2:
        return int(nonfinite.any(dim=0).sum().item())
    return int(nonfinite.reshape(prediction.shape[0], prediction.shape[1], -1).any(dim=0).any(dim=-1).sum().item())


def _count_parameter_bound_hits(
    entry: CoreModelEntry,
    physical_params: list[torch.Tensor],
    *,
    tolerance_fraction: float = 1.0e-6,
) -> int:
    hit_count = 0
    for physical, (param_name, (lower_bound, upper_bound)) in zip(physical_params, entry.param_bounds.items()):
        span = float(upper_bound) - float(lower_bound)
        tolerance = max(tolerance_fraction * max(span, 1.0), 1.0e-12)
        hit_count += int((physical <= float(lower_bound) + tolerance).sum().item())
        hit_count += int((physical >= float(upper_bound) - tolerance).sum().item())
    return hit_count


def run_calibration_case(
    model_name: str,
    config: TrainingConfig,
    *,
    beta_near_zero: bool = False,
    monitored_params: tuple[str, ...] = (),
) -> dict[str, Any]:
    torch.manual_seed(DEFAULT_SEED)
    entry = CORE_MODEL_REGISTRY[model_name]
    forcing = make_synthetic_forcing(entry, config.n_timesteps, config.n_basins)
    target_discharge, initial_states = _make_target_discharge(entry, forcing, config)

    raw_parameters = torch.nn.Parameter(_logit(_fraction_matrix(entry, "init", config.n_basins, beta_near_zero=beta_near_zero)))
    optimizer = torch.optim.Adam([raw_parameters], lr=config.learning_rate)

    monitored_indices = [index for index, name in enumerate(entry.param_bounds) if name in monitored_params]
    beta_index = list(entry.param_bounds).index("beta") if "beta" in entry.param_bounds else None

    loss_history: list[float] = []
    loss_nan_count = 0
    loss_inf_count = 0
    grad_nan_count = 0
    grad_inf_count = 0
    max_abs_grad = 0.0
    mean_abs_grad_samples: list[float] = []
    max_abs_grad_monitored = 0.0
    mean_abs_grad_monitored_samples: list[float] = []
    parameter_bound_hit_count = 0
    beta_values: list[float] = []
    beta_near_zero_step_count = 0
    beta_near_zero_grad_nan_count = 0
    beta_near_zero_grad_inf_count = 0
    beta_near_zero_grad_finite = True
    output_nan_count = 0
    output_inf_count = 0
    optimizer_step_success = True
    failed_stage = ""
    notes: list[str] = []
    start_time = time.perf_counter()

    initial_loss = math.nan
    final_loss = math.nan
    final_prediction = target_discharge.detach().clone()
    final_params: list[torch.Tensor] = []

    for step_index in range(config.n_optimizer_steps):
        optimizer.zero_grad()
        params_list, _ = _raw_to_physical_parameters(entry, raw_parameters)
        prediction, step_output_nan, step_output_inf = _run_rollout(entry, forcing, params_list, initial_states)
        output_nan_count += step_output_nan
        output_inf_count += step_output_inf
        loss = objective_loss(prediction.squeeze(-1), target_discharge, config.objective)

        if not torch.isfinite(loss):
            optimizer_step_success = False
            failed_stage = "loss"
            loss_nan_count += int(torch.isnan(loss).item())
            loss_inf_count += int(torch.isinf(loss).item())
            notes.append("nonfinite loss encountered")
            break

        loss_value = float(loss.detach().item())
        loss_history.append(loss_value)
        if step_index == 0:
            initial_loss = loss_value

        try:
            loss.backward()
        except Exception as exc:
            optimizer_step_success = False
            failed_stage = "backward"
            notes.append(f"{type(exc).__name__}: {exc}")
            break

        if raw_parameters.grad is None:
            optimizer_step_success = False
            failed_stage = "gradient_missing"
            notes.append("optimizer parameter gradient is missing")
            break

        grad_nan_count += int(torch.isnan(raw_parameters.grad).sum().item())
        grad_inf_count += int(torch.isinf(raw_parameters.grad).sum().item())
        if grad_nan_count or grad_inf_count:
            optimizer_step_success = False
            failed_stage = "gradient_nonfinite"
            notes.append("nonfinite gradient encountered")
            break

        grad_abs = raw_parameters.grad.detach().abs()
        max_abs_grad = max(max_abs_grad, float(grad_abs.max().item()))
        mean_abs_grad_samples.append(float(grad_abs.mean().item()))

        if monitored_indices:
            monitored_grad_abs = grad_abs[monitored_indices]
            max_abs_grad_monitored = max(max_abs_grad_monitored, float(monitored_grad_abs.max().item()))
            mean_abs_grad_monitored_samples.append(float(monitored_grad_abs.mean().item()))

        if beta_index is not None:
            beta_values.extend(params_list[beta_index].detach().reshape(-1).tolist())
            beta_near_zero_now = float(params_list[beta_index].detach().min().item()) <= 1.0e-6
            if beta_near_zero_now:
                beta_near_zero_step_count += 1
                beta_grad = raw_parameters.grad[beta_index]
                beta_near_zero_grad_nan_count += int(torch.isnan(beta_grad).sum().item())
                beta_near_zero_grad_inf_count += int(torch.isinf(beta_grad).sum().item())
                beta_near_zero_grad_finite = (
                    beta_near_zero_grad_finite
                    and bool(torch.isfinite(beta_grad).all())
                )

        parameter_bound_hit_count = max(parameter_bound_hit_count, _count_parameter_bound_hits(entry, params_list))

        try:
            optimizer.step()
        except Exception as exc:
            optimizer_step_success = False
            failed_stage = "optimizer_step"
            notes.append(f"{type(exc).__name__}: {exc}")
            break

        final_prediction = prediction.detach().clone()
        final_params = [param.detach().clone() for param in params_list]
        final_loss = loss_value

    runtime_seconds = time.perf_counter() - start_time

    if optimizer_step_success:
        params_list, _ = _raw_to_physical_parameters(entry, raw_parameters)
        final_prediction, final_output_nan, final_output_inf = _run_rollout(entry, forcing, params_list, initial_states)
        output_nan_count += final_output_nan
        output_inf_count += final_output_inf
        final_params = [param.detach().clone() for param in params_list]
        if loss_history:
            final_loss = loss_history[-1]
        if beta_index is not None and not beta_values:
            beta_values.extend(final_params[beta_index].detach().reshape(-1).tolist())

    parameter_nan_count = sum(int(torch.isnan(param).sum().item()) for param in final_params)
    parameter_inf_count = sum(int(torch.isinf(param).sum().item()) for param in final_params)
    failed_basin_count = _failed_basin_count(final_prediction)
    synthetic_nse, synthetic_kge = _final_metrics(final_prediction.squeeze(-1), target_discharge.squeeze(-1))

    status = "passed"
    if not optimizer_step_success or output_nan_count or output_inf_count or grad_nan_count or grad_inf_count:
        status = "failed"
    if not loss_history:
        notes.append("no successful optimization step completed")

    mean_abs_grad = sum(mean_abs_grad_samples) / len(mean_abs_grad_samples) if mean_abs_grad_samples else 0.0
    mean_abs_grad_monitored = (
        sum(mean_abs_grad_monitored_samples) / len(mean_abs_grad_monitored_samples)
        if mean_abs_grad_monitored_samples
        else 0.0
    )

    row: dict[str, Any] = {
        "model": model_name,
        "status": status,
        "objective": config.objective,
        "dtype": _dtype_name(DEFAULT_DTYPE),
        "device": DEFAULT_DEVICE,
        "n_timesteps": config.n_timesteps,
        "n_basins": config.n_basins,
        "n_optimizer_steps": config.n_optimizer_steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_change": final_loss - initial_loss if math.isfinite(initial_loss) and math.isfinite(final_loss) else math.nan,
        "loss_nan_count": loss_nan_count,
        "loss_inf_count": loss_inf_count,
        "grad_nan_count": grad_nan_count,
        "grad_inf_count": grad_inf_count,
        "max_abs_grad": max_abs_grad,
        "mean_abs_grad": mean_abs_grad,
        "parameter_nan_count": parameter_nan_count,
        "parameter_inf_count": parameter_inf_count,
        "parameter_bound_hit_count": parameter_bound_hit_count,
        "optimizer_step_success": optimizer_step_success,
        "failed_basin_count": failed_basin_count,
        "failed_stage": failed_stage,
        "output_nan_count": output_nan_count,
        "output_inf_count": output_inf_count,
        "synthetic_nse": synthetic_nse,
        "synthetic_kge": synthetic_kge,
        "runtime_seconds": runtime_seconds,
        "loss_curve": json.dumps(loss_history),
        "beta_min": min(beta_values) if beta_values else math.nan,
        "beta_max": max(beta_values) if beta_values else math.nan,
        "beta_reaches_zero": any(value == 0.0 for value in beta_values),
        "beta_near_zero_step_count": beta_near_zero_step_count,
        "beta_near_zero_grad_nan_count": beta_near_zero_grad_nan_count,
        "beta_near_zero_grad_inf_count": beta_near_zero_grad_inf_count,
        "beta_near_zero_grad_finite": beta_near_zero_grad_finite,
        "max_abs_grad_monitored": max_abs_grad_monitored,
        "mean_abs_grad_monitored": mean_abs_grad_monitored,
        "notes": "; ".join(notes) if notes else "synthetic target calibration smoke",
    }
    return row


def write_csv(rows: list[dict[str, Any]], output_path: Path, fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


TRAINING_SMOKE_FIELDNAMES = [
    "model",
    "status",
    "objective",
    "dtype",
    "device",
    "n_timesteps",
    "n_basins",
    "n_optimizer_steps",
    "initial_loss",
    "final_loss",
    "loss_change",
    "loss_nan_count",
    "loss_inf_count",
    "grad_nan_count",
    "grad_inf_count",
    "max_abs_grad",
    "mean_abs_grad",
    "parameter_nan_count",
    "parameter_inf_count",
    "parameter_bound_hit_count",
    "optimizer_step_success",
    "failed_basin_count",
    "failed_stage",
    "notes",
]

FLEX_FIELDNAMES = [
    "model",
    "status",
    "objective",
    "dtype",
    "device",
    "n_timesteps",
    "n_basins",
    "n_optimizer_steps",
    "initial_loss",
    "final_loss",
    "loss_change",
    "loss_curve",
    "beta_min",
    "beta_max",
    "beta_reaches_zero",
    "beta_near_zero_step_count",
    "beta_near_zero_grad_nan_count",
    "beta_near_zero_grad_inf_count",
    "beta_near_zero_grad_finite",
    "synthetic_nse",
    "synthetic_kge",
    "failed_basin_count",
    "baseline_reference",
    "notes",
]

MEDIUM_FIELDNAMES = [
    "model",
    "context_id",
    "status",
    "objective",
    "dtype",
    "device",
    "n_timesteps",
    "n_basins",
    "n_optimizer_steps",
    "initial_loss",
    "final_loss",
    "loss_change",
    "output_finite",
    "gradient_finite",
    "output_nan_count",
    "output_inf_count",
    "grad_nan_count",
    "grad_inf_count",
    "max_abs_grad_all",
    "max_abs_grad_monitored",
    "mean_abs_grad_monitored",
    "optimizer_step_success",
    "failed_stage",
    "notes",
]

ALL_MODEL_FIELDNAMES = [
    "model",
    "status",
    "objective",
    "dtype",
    "device",
    "n_timesteps",
    "n_basins",
    "n_optimizer_steps",
    "initial_loss",
    "final_loss",
    "loss_change",
    "output_nan_count",
    "output_inf_count",
    "grad_nan_count",
    "grad_inf_count",
    "failed_basin_count",
    "runtime_seconds",
    "synthetic_nse",
    "synthetic_kge",
    "notes",
]


def build_training_smoke_rows() -> list[dict[str, Any]]:
    return [run_calibration_case(model_name, TRAINING_SMOKE_CONFIG) for model_name in STAGE1_TARGET_MODELS]


def build_flex_regression_rows() -> list[dict[str, Any]]:
    rows = []
    for model_name in FLEX_MODELS:
        row = run_calibration_case(model_name, FLEX_TRAINING_CONFIG, beta_near_zero=True)
        row["baseline_reference"] = "no_prior_baseline_available"
        row["notes"] = (
            f"{row['notes']}; beta initialized near the restored 0.0 lower bound; "
            "NSE/KGE are against the synthetic calibration target"
        )
        rows.append(row)
    return rows


def build_medium_context_rows() -> list[dict[str, Any]]:
    rows = []
    for model_name, (context_id, monitored_params) in MEDIUM_CONTEXTS.items():
        run_row = run_calibration_case(
            model_name,
            MEDIUM_CONTEXT_CONFIG,
            monitored_params=monitored_params,
        )
        rows.append(
            {
                "model": model_name,
                "context_id": context_id,
                "status": run_row["status"],
                "objective": run_row["objective"],
                "dtype": run_row["dtype"],
                "device": run_row["device"],
                "n_timesteps": run_row["n_timesteps"],
                "n_basins": run_row["n_basins"],
                "n_optimizer_steps": run_row["n_optimizer_steps"],
                "initial_loss": run_row["initial_loss"],
                "final_loss": run_row["final_loss"],
                "loss_change": run_row["loss_change"],
                "output_finite": run_row["output_nan_count"] == 0 and run_row["output_inf_count"] == 0,
                "gradient_finite": run_row["grad_nan_count"] == 0 and run_row["grad_inf_count"] == 0,
                "output_nan_count": run_row["output_nan_count"],
                "output_inf_count": run_row["output_inf_count"],
                "grad_nan_count": run_row["grad_nan_count"],
                "grad_inf_count": run_row["grad_inf_count"],
                "max_abs_grad_all": run_row["max_abs_grad"],
                "max_abs_grad_monitored": run_row["max_abs_grad_monitored"],
                "mean_abs_grad_monitored": run_row["mean_abs_grad_monitored"],
                "optimizer_step_success": run_row["optimizer_step_success"],
                "failed_stage": run_row["failed_stage"],
                "notes": f"monitored_params={','.join(monitored_params)}; {run_row['notes']}",
            }
        )
    return rows


def build_all_model_smoke_rows() -> list[dict[str, Any]]:
    return [run_calibration_case(model_name, ALL_MODEL_SMOKE_CONFIG) for model_name in sorted(discover_runnable_models())]
