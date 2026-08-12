from __future__ import annotations

import ast
import csv
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.core import PARAM_INFO
from models.flux.saturation import saturation_3
from models.hydrology_model import HydrologyModel
from scripts.review_batch_a_flux_realistic_stability import BATCH_A_TARGETS, run_batch_a_review
from tests.core_model_registry import CORE_MODEL_REGISTRY
from tests.core_water_balance_utils import build_parameter_tensors
from tests.flux_gradient_wrappers import load_flux_usage_contexts


OUTPUT_DIR = REPO_ROOT / "validation_results" / "flex_saturation3_parameter_bound_fix"
DEFAULT_DTYPE = torch.float64
DEFAULT_DEVICE = "cpu"
DEFAULT_NEARZERO = 1.0e-6
FIXED_SEED = 20260624
TARGET_MODELS = ("flexb", "flexi", "flexis")
TARGET_PARAM = "beta"
PREVIOUS_WORKAROUND_LOWER_BOUND = 1.0e-6
ORIGINAL_LOWER_BOUND = 0.0
TESTED_BETAS = (0.0, 1.0e-12, 1.0e-9, 1.0e-6, 1.0e-5, 1.0e-4)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _module_bounds_name(model_name: str) -> str:
    return f"{model_name.upper()}_PARAMS_BOUNDS"


def _model_module_path(model_name: str) -> Path:
    return REPO_ROOT / "models" / "core" / f"{model_name}.py"


def _dict_line_bounds(path: Path, dict_name: str) -> tuple[int, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == dict_name:
                    return node.lineno, getattr(node, "end_lineno", node.lineno)
    raise KeyError(dict_name)


def _target_inventory_rows() -> list[dict[str, Any]]:
    usage_rows = {}
    for ctx in load_flux_usage_contexts():
        if ctx.flux_function != "saturation_3" or ctx.model_name not in TARGET_MODELS:
            continue
        if ctx.module_type != "core":
            continue
        if ctx.active_usage_status != "active_registered_model":
            continue
        current = usage_rows.get((ctx.flux_function, ctx.model_name))
        if current is None or ctx.parameter_bounds.get("p1", (0.0, 0.0))[0] > current.parameter_bounds.get("p1", (0.0, 0.0))[0]:
            usage_rows[(ctx.flux_function, ctx.model_name)] = ctx
    rows: list[dict[str, Any]] = []
    for model_name in TARGET_MODELS:
        bounds = PARAM_INFO[model_name]
        index = list(bounds.keys()).index(TARGET_PARAM)
        path = _model_module_path(model_name)
        line_start, line_end = _dict_line_bounds(path, _module_bounds_name(model_name))
        usage = usage_rows[("saturation_3", model_name)]
        rows.append(
            {
                "model": model_name,
                "parameter_name": TARGET_PARAM,
                "parameter_index": index,
                "previous_lower_bound": PREVIOUS_WORKAROUND_LOWER_BOUND,
                "current_lower_bound": float(bounds[TARGET_PARAM][0]),
                "current_upper_bound": float(bounds[TARGET_PARAM][1]),
                "bound_source_file": str(path.relative_to(REPO_ROOT)),
                "bound_source_lines": f"{line_start}-{line_end}",
                "saturation3_argument_position": 3,
                "notes": (
                    "0-based parameter index in *_PARAMS_BOUNDS order; `beta` is passed as the third "
                    "positional argument (`p1`) to `saturation_3`. Current lower bound reflects the "
                    "post-stable-rewrite decision."
                ),
            }
        )
        assert math.isclose(usage.parameter_bounds["p1"][0], ORIGINAL_LOWER_BOUND, rel_tol=0.0, abs_tol=1.0e-12)
    return rows


def _pooled_stats_by_model() -> dict[str, dict[str, dict[str, float]]]:
    artifacts = run_batch_a_review()
    pooled = {}
    for target in BATCH_A_TARGETS:
        if target.formula != "saturation_3" or target.active_model not in TARGET_MODELS:
            continue
        pooled[target.active_model] = artifacts["pooled_stats_by_key"][(target.formula, target.active_model)]
    return pooled


def _base_inputs(model_name: str, pooled_stats: dict[str, dict[str, float]]) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=DEFAULT_DEVICE)
    generator.manual_seed(FIXED_SEED + sum(ord(ch) for ch in model_name))
    inputs: dict[str, torch.Tensor] = {}
    for arg_name in ("S", "Smax", "incoming_flux"):
        values = pooled_stats[arg_name]
        lo = values["p05"]
        hi = values["p95"]
        tensor = lo + (hi - lo) * torch.rand((65,), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE, generator=generator)
        inputs[arg_name] = tensor
    return inputs


def _gradient_metrics(output: torch.Tensor, beta: torch.Tensor, storage: torch.Tensor) -> tuple[int, int, float, float]:
    grads = torch.autograd.grad(output.sum(), (storage, beta), allow_unused=True)
    flat = []
    grad_nan_count = 0
    grad_inf_count = 0
    for grad in grads:
        if grad is None:
            continue
        grad_nan_count += int(torch.isnan(grad).sum().item())
        grad_inf_count += int(torch.isinf(grad).sum().item())
        flat.append(torch.abs(grad).reshape(-1))
    if not flat:
        return grad_nan_count, grad_inf_count, 0.0, 0.0
    cat = torch.cat(flat)
    return grad_nan_count, grad_inf_count, float(cat.max().item()), float(cat.mean().item())


def _validate_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pooled = _pooled_stats_by_model()
    for model_name in TARGET_MODELS:
        bounds = PARAM_INFO[model_name][TARGET_PARAM]
        midpoint = (float(bounds[0]) + float(bounds[1])) / 2.0
        tested_betas = TESTED_BETAS + (midpoint,)

        entry = CORE_MODEL_REGISTRY[model_name]
        _, params_map_lower = build_parameter_tensors(entry, "lower_near", (2, 2), DEFAULT_DTYPE, DEFAULT_DEVICE)
        _, params_map_random = build_parameter_tensors(entry, "random_valid", (8, 8), DEFAULT_DTYPE, DEFAULT_DEVICE)
        sampled_min = min(
            float(params_map_lower[TARGET_PARAM].min().item()),
            float(params_map_random[TARGET_PARAM].min().item()),
        )

        model = HydrologyModel({"model_name": model_name, "backend": "eager"}, device=torch.device(DEFAULT_DEVICE), backend="eager")
        transform_zero = model._change_param_range(
            torch.zeros((1,), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE),
            model.parameter_bounds[TARGET_PARAM],
            model.parameter_mapping,
            model.log_mapping_span_threshold,
        )
        transform_grid = model._change_param_range(
            torch.linspace(0.0, 1.0, steps=257, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE),
            model.parameter_bounds[TARGET_PARAM],
            model.parameter_mapping,
            model.log_mapping_span_threshold,
        )
        base = _base_inputs(model_name, pooled[model_name])
        base_beta0 = torch.zeros((65,), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE, requires_grad=False)
        base_beta1e6 = torch.full((65,), 1.0e-6, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE, requires_grad=False)
        beta0_output = saturation_3(
            base["S"],
            base["Smax"],
            base_beta0,
            base["incoming_flux"],
            nearzero=DEFAULT_NEARZERO,
        )
        beta1e6_output = saturation_3(
            base["S"],
            base["Smax"],
            base_beta1e6,
            base["incoming_flux"],
            nearzero=DEFAULT_NEARZERO,
        )
        beta0_finite = torch.isfinite(beta0_output).all().item()
        beta1e6_finite = torch.isfinite(beta1e6_output).all().item()

        for tested_beta in tested_betas:
            S = base["S"].detach().clone().requires_grad_(True)
            Smax = base["Smax"].detach().clone()
            incoming_flux = base["incoming_flux"].detach().clone()
            beta = torch.full((65,), float(tested_beta), dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE, requires_grad=True)
            output = saturation_3(S, Smax, beta, incoming_flux, nearzero=DEFAULT_NEARZERO)
            output_nan_count = int(torch.isnan(output).sum().item())
            output_inf_count = int(torch.isinf(output).sum().item())
            output_bound_violation_count = int(
                ((output < -1.0e-12) | (output > incoming_flux + 1.0e-12)).sum().item()
            )
            grad_nan_count, grad_inf_count, max_abs_grad, mean_abs_grad = _gradient_metrics(output, beta, S)
            diff_vs_beta0 = math.nan
            if beta0_finite:
                diff_vs_beta0 = float(torch.max(torch.abs(output.detach() - beta0_output)).item())
            diff_vs_beta1e6 = math.nan
            if beta1e6_finite:
                diff_vs_beta1e6 = float(torch.max(torch.abs(output.detach() - beta1e6_output)).item())

            pass_fail = (
                output_nan_count == 0
                and output_inf_count == 0
                and output_bound_violation_count == 0
                and grad_nan_count == 0
                and grad_inf_count == 0
                and sampled_min > 0.0
            )

            notes = (
                f"sampled_min_beta={sampled_min:.6g}; raw0_maps_to={float(transform_zero.item()):.6g}; "
                f"grid_min_beta={float(transform_grid.min().item()):.6g}"
            )
            rows.append(
                {
                    "model": model_name,
                    "parameter_name": TARGET_PARAM,
                    "tested_beta": tested_beta,
                    "output_nan_count": output_nan_count,
                    "output_inf_count": output_inf_count,
                    "output_bound_violation_count": output_bound_violation_count,
                    "grad_nan_count": grad_nan_count,
                    "grad_inf_count": grad_inf_count,
                    "max_abs_grad": max_abs_grad,
                    "mean_abs_grad": mean_abs_grad,
                    "output_diff_vs_beta0_if_available": diff_vs_beta0,
                    "output_diff_vs_beta_1e6_if_available": diff_vs_beta1e6,
                    "pass_fail": "pass" if pass_fail else "fail",
                    "notes": notes,
                }
            )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory_rows = _target_inventory_rows()
    validation_rows = _validate_rows()
    _write_csv(OUTPUT_DIR / "flex_saturation3_bound_target.csv", inventory_rows)
    _write_csv(OUTPUT_DIR / "flex_saturation3_bound_fix_validation.csv", validation_rows)


if __name__ == "__main__":
    main()
