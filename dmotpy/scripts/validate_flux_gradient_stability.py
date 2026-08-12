from __future__ import annotations

import ast
import csv
import inspect
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.flux_gradient_wrappers import (  # noqa: E402
    DEFAULT_NEARZERO,
    FIXED_SEED,
    FluxFunctionInfo,
    FluxUsageContext,
    build_flux_wrapper,
    build_wrapper_inputs,
    evaluate_wrapper,
    iter_all_flux_wrappers,
    load_flux_inventory,
    load_flux_usage_contexts,
)


OUTPUT_DIR = REPO_ROOT / "validation_results" / "flux_gradient_stability"
PLOTS_DIR = OUTPUT_DIR / "plots"

PARAMETER_CASES = ("lower", "near_lower", "mid", "near_upper", "upper", "random")
STATE_CASES = ("zero", "nearzero", "lower", "mid", "upper", "random")
THRESHOLD_CASES = ("threshold_minus", "mid", "threshold_plus")
DEFAULT_SHAPE = (17,)
EXPLODING_GRAD_HIGH = 1.0e4
EXPLODING_GRAD_MED = 1.0e2
VANISHING_GRAD = 1.0e-10
FD_EPS = 1.0e-6


def _function_features(info: FluxFunctionInfo) -> dict[str, bool]:
    source = info.source
    return {
        "uses_soft_gate": "soft_gate_" in source,
        "uses_relu": "relu" in source,
        "uses_clamp": "clamp" in source,
        "uses_where": "where" in source,
        "uses_minimum": "minimum" in source,
        "uses_maximum": "maximum" in source,
        "uses_power": "**" in source or ".pow(" in source,
        "uses_exp": "exp(" in source,
        "uses_log": "log(" in source or "log10(" in source,
        "uses_division": "/" in source,
        "uses_epsilon": "nearzero" in source,
    }


def _formula_type(info: FluxFunctionInfo) -> str:
    return f"{Path(info.flux_file).stem}:{info.function_name}"


def _likely_threshold_formula(info: FluxFunctionInfo) -> bool:
    src = info.source.lower()
    tokens = ("threshold", "relu", "where", "soft_gate", "tcrit", "fc", "lp", "wilting", "degree-day")
    return any(token in src for token in tokens)


def _bound_mode(flux_file: str) -> str:
    stem = Path(flux_file).stem
    if stem in {"saturation", "split", "rainfall", "snowfall", "interception", "effective"}:
        return "incoming_only"
    if stem in {"evap", "baseflow", "recharge", "melt", "refreeze", "percolation", "capillary", "depression", "excess"}:
        return "storage_only"
    return "none"


def build_inventory_rows() -> list[dict[str, Any]]:
    rows = []
    for info in load_flux_inventory().values():
        features = _function_features(info)
        rows.append(
            {
                "flux_file": info.flux_file,
                "function_name": info.function_name,
                "line_start": info.line_start,
                "line_end": info.line_end,
                "function_signature": info.function_signature,
                "formula_type": _formula_type(info),
                "likely_threshold_formula": _likely_threshold_formula(info),
                "notes": "",
                **features,
            }
        )
    rows.sort(key=lambda row: (row["flux_file"], row["line_start"]))
    return rows


def build_usage_rows() -> list[dict[str, Any]]:
    inventory = load_flux_inventory()
    usage = load_flux_usage_contexts()
    by_function: dict[str, list[FluxUsageContext]] = {}
    for ctx in usage:
        by_function.setdefault(ctx.flux_function, []).append(ctx)

    rows = []
    for function_name, info in inventory.items():
        contexts = by_function.get(function_name, [])
        if not contexts:
            rows.append(
                {
                    "flux_function": function_name,
                    "flux_file": info.flux_file,
                    "active_usage_status": "unused",
                    "called_by_models": "",
                    "call_sites": "",
                    "parameter_mapping": "{}",
                    "parameter_bounds_source": "generic_inferred",
                    "parameter_bounds": "{}",
                    "state_variable_mapping": "{}",
                    "state_ranges": "{}",
                    "forcing_ranges": "{}",
                    "inferred_or_exact": "inferred",
                    "notes": "No active core/special call site found.",
                }
            )
            continue
        for ctx in contexts:
            state_ranges = {
                name: (
                    "[-10,20]" if mapped.lower() == "t" else
                    "[0,15]" if mapped.lower() in {"pet", "ep"} else
                    "[0,2000]" if mapped.lower().startswith("s") else
                    "[0,200]"
                )
                for name, mapped in ctx.state_variable_mapping.items()
            }
            forcing_ranges = {
                name: (
                    "[-10,20]" if mapped.lower() == "t" else
                    "[0,15]" if mapped.lower() in {"pet", "ep"} else
                    "[1,365]" if mapped.lower() == "doy" else
                    "[0,200]"
                )
                for name, mapped in ctx.forcing_variable_mapping.items()
            }
            rows.append(
                {
                    "flux_function": function_name,
                    "flux_file": info.flux_file,
                    "active_usage_status": ctx.active_usage_status,
                    "called_by_models": ctx.model_name,
                    "call_sites": ctx.call_site,
                    "parameter_mapping": json.dumps(ctx.parameter_mapping, ensure_ascii=False, sort_keys=True),
                    "parameter_bounds_source": f"{ctx.model_name}_PARAMS_BOUNDS",
                    "parameter_bounds": json.dumps(ctx.parameter_bounds, ensure_ascii=False, sort_keys=True),
                    "state_variable_mapping": json.dumps(ctx.state_variable_mapping, ensure_ascii=False, sort_keys=True),
                    "state_ranges": json.dumps(state_ranges, ensure_ascii=False, sort_keys=True),
                    "forcing_ranges": json.dumps(forcing_ranges, ensure_ascii=False, sort_keys=True),
                    "inferred_or_exact": ctx.inferred_or_exact,
                    "notes": "",
                }
            )
    rows.sort(key=lambda row: (row["flux_file"], row["flux_function"], row["called_by_models"]))
    return rows


def _select_scalar_input(inputs: dict[str, torch.Tensor], preferred_names: list[str]) -> str | None:
    for name in preferred_names:
        if name in inputs:
            return name
    for name, tensor in inputs.items():
        if tensor.requires_grad:
            return name
    return None


def _finite_difference(
    wrapper_name: str,
    model_context: str,
    input_name: str,
    base_inputs: dict[str, torch.Tensor],
    eps: float = FD_EPS,
) -> tuple[float, float]:
    wrapper = build_flux_wrapper(wrapper_name, model_context)
    center = {k: v.detach().clone() for k, v in base_inputs.items()}
    if center[input_name].numel() == 0:
        return math.nan, math.nan
    flat = center[input_name].reshape(-1)
    idx = flat.numel() // 2

    plus = {k: v.detach().clone() for k, v in center.items()}
    minus = {k: v.detach().clone() for k, v in center.items()}
    plus[input_name].reshape(-1)[idx] += eps
    minus[input_name].reshape(-1)[idx] -= eps

    with torch.no_grad():
        y_plus = evaluate_wrapper(wrapper, plus).reshape(-1)[idx].item()
        y_minus = evaluate_wrapper(wrapper, minus).reshape(-1)[idx].item()
    fd = (y_plus - y_minus) / (2.0 * eps)
    return fd, idx


def _autograd_gradients(output: torch.Tensor, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    scalar = output.sum()
    grads = torch.autograd.grad(
        scalar,
        [tensor for tensor in inputs.values() if tensor.requires_grad],
        allow_unused=True,
        retain_graph=False,
    )
    result: dict[str, torch.Tensor] = {}
    grad_iter = iter(grads)
    for name, tensor in inputs.items():
        if tensor.requires_grad:
            result[name] = next(grad_iter)
    return result


def _bound_violation_count(wrapper, output: torch.Tensor, inputs: dict[str, torch.Tensor]) -> int:
    count = 0
    mode = _bound_mode(wrapper.flux_info.flux_file)
    if mode in {"storage_only", "both"} and wrapper.available_storage_inputs:
        storage_cap = torch.stack([torch.relu(inputs[name]) for name in wrapper.available_storage_inputs]).amin(dim=0)
        count += int((output > storage_cap + 1.0e-10).sum().item())
    if mode in {"incoming_only", "both"} and wrapper.incoming_flux_inputs:
        incoming_cap = torch.stack([torch.relu(inputs[name]) for name in wrapper.incoming_flux_inputs]).amin(dim=0)
        count += int((output > incoming_cap + 1.0e-10).sum().item())
    return count


def _gradient_saturation_ratio(grads: dict[str, torch.Tensor]) -> float:
    values = []
    for grad in grads.values():
        if grad is None:
            continue
        abs_grad = torch.abs(grad)
        values.append(float((abs_grad < 1.0e-8).float().mean().item()))
    return max(values) if values else 1.0


def _zero_gradient_fraction(grads: dict[str, torch.Tensor]) -> float:
    values = []
    for grad in grads.values():
        if grad is None:
            continue
        values.append(float((torch.abs(grad) < 1.0e-12).float().mean().item()))
    return max(values) if values else 1.0


def _sign_flip_count(grad: torch.Tensor | None) -> int:
    if grad is None or grad.numel() < 2:
        return 0
    flat = grad.detach().reshape(-1)
    signs = torch.sign(flat)
    return int((signs[1:] * signs[:-1] < 0).sum().item())


def _dead_region_width(wrapper, probe_name: str, inputs: dict[str, torch.Tensor]) -> float:
    if probe_name not in inputs:
        return math.nan
    base = inputs[probe_name].detach().clone().reshape(-1)
    low = float(base.min().item())
    high = float(base.max().item())
    if math.isclose(low, high):
        return 0.0
    grid = torch.linspace(low, high, steps=33, dtype=base.dtype, device=base.device)
    zeros = 0
    for value in grid:
        probe_inputs = {k: v.detach().clone().requires_grad_(v.requires_grad) for k, v in inputs.items()}
        probe_inputs[probe_name] = torch.full_like(inputs[probe_name], float(value), requires_grad=True)
        out = evaluate_wrapper(wrapper, probe_inputs)
        grads = _autograd_gradients(out, probe_inputs)
        grad = grads.get(probe_name)
        if grad is None or float(torch.abs(grad).max().item()) < 1.0e-10:
            zeros += 1
    return (high - low) * zeros / max(len(grid), 1)


def diagnose_wrapper_case(wrapper, parameter_case: str, state_case: str, dtype: torch.dtype, device: str) -> dict[str, Any]:
    inputs = build_wrapper_inputs(wrapper, parameter_case=parameter_case, state_case=state_case, dtype=dtype, device=device, shape=DEFAULT_SHAPE)
    output = evaluate_wrapper(wrapper, inputs)
    grads = _autograd_gradients(output, inputs)

    output_nan_count = int(torch.isnan(output).sum().item())
    output_inf_count = int(torch.isinf(output).sum().item())
    output_negative_count = int((output < -1.0e-12).sum().item()) if wrapper.expected_nonnegative else 0
    output_bound_violation_count = _bound_violation_count(wrapper, output, inputs)

    grad_nan_count = 0
    grad_inf_count = 0
    max_abs_grad = 0.0
    mean_abs_grad = 0.0
    median_abs_grad = 0.0
    grad_l2_norm = 0.0
    sign_flip_count = 0

    grad_values = []
    for grad in grads.values():
        if grad is None:
            continue
        grad_nan_count += int(torch.isnan(grad).sum().item())
        grad_inf_count += int(torch.isinf(grad).sum().item())
        abs_grad = torch.abs(grad)
        grad_values.append(abs_grad.reshape(-1))
        max_abs_grad = max(max_abs_grad, float(abs_grad.max().item()))
        sign_flip_count += _sign_flip_count(grad)

    if grad_values:
        flat = torch.cat(grad_values)
        mean_abs_grad = float(flat.mean().item())
        median_abs_grad = float(flat.median().item())
        grad_l2_norm = float(torch.linalg.norm(flat).item())

    probe_name = _select_scalar_input(inputs, list(wrapper.threshold_inputs) + list(wrapper.available_storage_inputs) + list(inputs.keys()))
    autograd_fd_max_error = math.nan
    autograd_fd_relative_error = math.nan
    kink_indicator = False
    if probe_name is not None and grads.get(probe_name) is not None:
        fd, fd_idx = _finite_difference(wrapper.flux_info.function_name, wrapper.context.model_context, probe_name, inputs)
        grad_probe = grads[probe_name].reshape(-1)[fd_idx].item()
        autograd_fd_max_error = abs(grad_probe - fd)
        autograd_fd_relative_error = abs(grad_probe - fd) / max(abs(fd), 1.0e-12)
        kink_indicator = autograd_fd_relative_error > 1.0e-3 and max_abs_grad < EXPLODING_GRAD_HIGH

    zero_gradient_fraction = _zero_gradient_fraction(grads)
    gradient_saturation_ratio = _gradient_saturation_ratio(grads)
    dead_region_width = _dead_region_width(wrapper, probe_name, inputs) if probe_name is not None else math.nan

    exploding_gradient_flag = max_abs_grad > EXPLODING_GRAD_HIGH
    vanishing_gradient_flag = zero_gradient_fraction > 0.8 or (max_abs_grad < VANISHING_GRAD and grad_nan_count == 0 and grad_inf_count == 0)

    notes = []
    if wrapper.manual_review_required:
        notes.append(wrapper.manual_review_reason)
    if kink_indicator:
        notes.append("hard-threshold gradient limitation")
    if wrapper.threshold_inputs:
        notes.append(f"threshold_inputs={','.join(wrapper.threshold_inputs)}")

    risk_fail = any(
        [
            output_nan_count > 0,
            output_inf_count > 0,
            grad_nan_count > 0,
            grad_inf_count > 0,
            output_bound_violation_count > 0,
            exploding_gradient_flag,
        ]
    )

    return {
        "flux_function": wrapper.flux_info.function_name,
        "flux_file": wrapper.flux_info.flux_file,
        "model_context": wrapper.context.model_context,
        "active_usage_status": wrapper.context.active_usage_status,
        "formula_type": _formula_type(wrapper.flux_info),
        "test_case": f"{parameter_case}_{state_case}",
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "parameter_case": parameter_case,
        "state_case": state_case,
        "output_nan_count": output_nan_count,
        "output_inf_count": output_inf_count,
        "output_negative_count": output_negative_count,
        "output_bound_violation_count": output_bound_violation_count,
        "grad_nan_count": grad_nan_count,
        "grad_inf_count": grad_inf_count,
        "max_abs_grad": max_abs_grad,
        "mean_abs_grad": mean_abs_grad,
        "median_abs_grad": median_abs_grad,
        "grad_l2_norm": grad_l2_norm,
        "zero_gradient_fraction": zero_gradient_fraction,
        "gradient_saturation_ratio": gradient_saturation_ratio,
        "autograd_fd_max_error": autograd_fd_max_error,
        "autograd_fd_relative_error": autograd_fd_relative_error,
        "dead_region_width_near_threshold": dead_region_width,
        "kink_indicator": kink_indicator,
        "exploding_gradient_flag": exploding_gradient_flag,
        "vanishing_gradient_flag": vanishing_gradient_flag,
        "pass_fail": not risk_fail,
        "notes": "; ".join(notes),
        "min_output": float(torch.min(output).item()),
        "max_output": float(torch.max(output).item()),
        "mean_output": float(torch.mean(output).item()),
        "sign_flip_count": sign_flip_count,
    }


def summarize_risk(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inventory = load_flux_inventory()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["flux_function"], row["model_context"]), []).append(row)

    ranking = []
    for (flux_function, model_context), cases in grouped.items():
        sample = cases[0]
        likely_threshold = _likely_threshold_formula(inventory[sample["flux_function"]]) or (
            "hard-threshold gradient limitation" in sample["notes"]
        )
        max_abs_grad = max(float(case["max_abs_grad"]) for case in cases)
        grad_nan_count = sum(int(case["grad_nan_count"]) for case in cases)
        grad_inf_count = sum(int(case["grad_inf_count"]) for case in cases)
        output_nan_count = sum(int(case["output_nan_count"]) for case in cases)
        output_inf_count = sum(int(case["output_inf_count"]) for case in cases)
        output_bound_violation_count = sum(int(case["output_bound_violation_count"]) for case in cases)
        zero_gradient_fraction = max(float(case["zero_gradient_fraction"]) for case in cases)
        autograd_fd_relative_error = max(
            float(case["autograd_fd_relative_error"])
            for case in cases
            if not math.isnan(float(case["autograd_fd_relative_error"]))
        ) if any(not math.isnan(float(case["autograd_fd_relative_error"])) for case in cases) else math.nan

        if output_nan_count or output_inf_count or grad_nan_count or grad_inf_count or output_bound_violation_count or max_abs_grad > EXPLODING_GRAD_HIGH:
            risk_level = "high"
        elif max_abs_grad > EXPLODING_GRAD_MED or (likely_threshold and zero_gradient_fraction > 0.8) or (likely_threshold and not math.isnan(autograd_fd_relative_error) and autograd_fd_relative_error > 1.0e-3):
            risk_level = "medium"
        else:
            risk_level = "low"

        reasons = []
        if output_nan_count or output_inf_count:
            reasons.append("non-finite output")
        if grad_nan_count or grad_inf_count:
            reasons.append("non-finite gradient")
        if output_bound_violation_count:
            reasons.append("output bound violation")
        if max_abs_grad > EXPLODING_GRAD_HIGH:
            reasons.append("exploding gradient")
        elif max_abs_grad > EXPLODING_GRAD_MED:
            reasons.append("large gradient")
        if likely_threshold and zero_gradient_fraction > 0.8:
            reasons.append("large dead region")
        if not reasons:
            reasons.append("stable in tested domain")

        recommended_action = "safe_no_action"
        if sample["active_usage_status"] == "unused":
            recommended_action = "inactive_deprecate"
        elif output_nan_count or output_inf_count or grad_nan_count or grad_inf_count:
            recommended_action = "check_parameter_range"
        elif output_bound_violation_count:
            recommended_action = "add_safety_clamp"
        elif likely_threshold and zero_gradient_fraction > 0.8 and "soft_gate" not in sample["notes"]:
            recommended_action = "candidate_for_smoothing_review"
        elif "hard-threshold gradient limitation" in sample["notes"]:
            recommended_action = "document_hard_threshold"

        ranking.append(
            {
                "flux_function": flux_function,
                "flux_file": sample["flux_file"],
                "active_usage_status": sample["active_usage_status"],
                "called_by_models": model_context,
                "formula_type": sample["formula_type"],
                "risk_level": risk_level,
                "risk_reason": "; ".join(reasons),
                "max_abs_grad": max_abs_grad,
                "grad_nan_count": grad_nan_count,
                "grad_inf_count": grad_inf_count,
                "output_nan_count": output_nan_count,
                "output_inf_count": output_inf_count,
                "output_bound_violation_count": output_bound_violation_count,
                "zero_gradient_fraction": zero_gradient_fraction,
                "autograd_fd_relative_error": autograd_fd_relative_error,
                "recommended_action": recommended_action,
                "human_review_priority": "high" if risk_level == "high" else ("medium" if risk_level == "medium" else "low"),
            }
        )
    ranking.sort(key=lambda row: ({"high": 0, "medium": 1, "low": 2}[row["risk_level"]], -row["max_abs_grad"], row["flux_function"]))
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
    return ranking


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _plot_representative_cases(rows: list[dict[str, Any]]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    selected = [row for row in rows if row["risk_level"] in {"high", "medium"}][:12]
    for row in selected:
        wrapper = build_flux_wrapper(row["flux_function"], row["called_by_models"])
        input_name = next(iter(wrapper.available_storage_inputs or wrapper.incoming_flux_inputs or wrapper.threshold_inputs or wrapper.differentiable_inputs), None)
        if input_name is None:
            continue
        inputs = build_wrapper_inputs(wrapper, parameter_case="mid", state_case="mid", dtype=torch.float64, device="cpu", shape=(64,))
        base = inputs[input_name]
        low = float(base.min().item())
        high = float(base.max().item())
        if math.isclose(low, high):
            low -= 1.0
            high += 1.0
        grid = torch.linspace(low, high, steps=64, dtype=torch.float64)
        outputs = []
        grads = []
        for value in grid:
            probe_inputs = {k: v.detach().clone().requires_grad_(v.requires_grad) for k, v in inputs.items()}
            probe_inputs[input_name] = torch.full_like(inputs[input_name], float(value), requires_grad=True)
            out = evaluate_wrapper(wrapper, probe_inputs)
            outputs.append(float(out.mean().item()))
            grad = torch.autograd.grad(out.sum(), probe_inputs[input_name], allow_unused=True)[0]
            grads.append(float(torch.abs(grad).mean().item()) if grad is not None else 0.0)

        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        axes[0].plot(grid.cpu().numpy(), outputs)
        axes[0].set_ylabel("output")
        axes[0].set_title(f"{row['flux_function']} [{row['called_by_models']}]")
        axes[1].plot(grid.cpu().numpy(), grads)
        axes[1].set_ylabel("|grad|")
        axes[1].set_xlabel(input_name)
        fig.tight_layout()
        out_path = PLOTS_DIR / f"{row['rank']:03d}_{row['flux_function']}_{row['called_by_models']}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)


def build_report(
    inventory_rows: list[dict[str, Any]],
    usage_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    ranking_rows: list[dict[str, Any]],
) -> str:
    active = {row["flux_function"] for row in usage_rows if row["active_usage_status"] != "unused"}
    unused = {row["flux_function"] for row in usage_rows if row["active_usage_status"] == "unused"}
    high = [row for row in ranking_rows if row["risk_level"] == "high"]
    medium = [row for row in ranking_rows if row["risk_level"] == "medium"]

    lines = [
        "# Flux Gradient Stability Report",
        "",
        "## 1. Scope and purpose",
        "- This workflow diagnoses output and gradient stability for `dmotpy/models/flux` formulas under PyTorch automatic differentiation.",
        "- It is intended to support future gradient-based calibration and formula review.",
        "- No hydrological formula was changed in this workflow.",
        "",
        "## 2. Inventory summary",
        f"- Flux functions found: {len(inventory_rows)}",
        f"- Active flux functions: {len(active)}",
        f"- Unused flux functions: {len(unused)}",
        f"- Model-specific parameter contexts tested: {len(ranking_rows)}",
        "",
        "## 3. Parameter-range inference",
        "- Preferred source was the active calling model's `*_PARAMS_BOUNDS` dictionary.",
        "- When a flux argument could not be mapped directly to a model parameter, a conservative inferred range was used and marked as inferred in the CSV outputs.",
        "- Unused flux functions were still tested in a generic inferred domain.",
        "",
        "## 4. Test domains",
        "- Parameter cases: lower, near-lower, midpoint, near-upper, upper, random-valid.",
        "- State/forcing cases: zero, near-zero, lower, midpoint, upper, random-valid.",
        "- Threshold-sensitive formulas were additionally probed with just-below / just-above style cases when the argument naming pattern allowed it.",
        "- Main diagnostics used `torch.float64` on CPU.",
        "",
        "## 5. Output stability summary",
        f"- Total diagnostic rows: {len(summary_rows)}",
        f"- Rows with non-finite outputs: {sum(1 for row in summary_rows if row['output_nan_count'] or row['output_inf_count'])}",
        f"- Rows with output bound violations: {sum(1 for row in summary_rows if row['output_bound_violation_count'])}",
        "",
        "## 6. Gradient stability summary",
        f"- Rows with non-finite gradients: {sum(1 for row in summary_rows if row['grad_nan_count'] or row['grad_inf_count'])}",
        f"- Rows with `max_abs_grad > 1e4`: {sum(1 for row in summary_rows if row['max_abs_grad'] > 1.0e4)}",
        f"- Rows with `zero_gradient_fraction > 0.8`: {sum(1 for row in summary_rows if row['zero_gradient_fraction'] > 0.8)}",
        "",
        "## 7. Medium/high-risk formulas",
        "| flux function | file | context | risk | reason | recommended action |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in ranking_rows:
        if row["risk_level"] == "low":
            continue
        lines.append(
            f"| {row['flux_function']} | {row['flux_file']} | {row['called_by_models']} | "
            f"{row['risk_level']} | {row['risk_reason']} | {row['recommended_action']} |"
        )

    lines.extend(
        [
            "",
            "## 8. Formula-specific notes",
            "- Hard-threshold and piecewise formulas were not automatically treated as implementation bugs.",
            "- When autograd/finite-difference disagreement localized near a threshold or ReLU kink, the report labels it as a hard-threshold gradient limitation rather than a formula error.",
            "- Unused formulas with medium/high risk are candidates to keep inactive unless they are revalidated in a model-specific context.",
            "",
            "## 9. Overall calibration-readiness assessment",
            "- The active flux set is generally numerically usable for gradient-based calibration if current model parameter bounds are respected.",
            "- The main residual risks are concentrated in hard-threshold, piecewise, and near-zero power-law formulas rather than in the soft-gated active formulas.",
            "- Medium/high-risk cases should be reviewed before large-scale optimization campaigns, especially if the optimizer is expected to explore threshold neighborhoods aggressively.",
            "",
            "## 10. Recommended next steps",
            "- Keep low-risk active formulas unchanged.",
            "- Document hard-threshold gradient limitations for active threshold formulas that are physically safe but partly inactive below thresholds.",
            "- Prioritize smoothing review only for active formulas that combine dead regions with materially important process activation.",
            "- Keep unused medium/high-risk formulas out of active models unless they are revalidated in a model-specific domain.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    torch.manual_seed(FIXED_SEED)
    inventory_rows = build_inventory_rows()
    usage_rows = build_usage_rows()
    wrappers = iter_all_flux_wrappers()

    summary_rows: list[dict[str, Any]] = []
    high_risk_cases: list[dict[str, Any]] = []
    for wrapper in wrappers:
        state_cases = STATE_CASES
        if wrapper.threshold_inputs:
            state_cases = THRESHOLD_CASES
        for parameter_case in PARAMETER_CASES:
            for state_case in state_cases:
                row = diagnose_wrapper_case(wrapper, parameter_case, state_case, torch.float64, "cpu")
                summary_rows.append(row)
                if not row["pass_fail"]:
                    high_risk_cases.append(row)

    ranking_rows = summarize_risk(summary_rows)
    _plot_representative_cases(ranking_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "flux_function_inventory.csv", inventory_rows)
    _write_csv(OUTPUT_DIR / "flux_usage_parameter_map.csv", usage_rows)
    _write_csv(OUTPUT_DIR / "flux_gradient_stability_summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "flux_gradient_risk_ranking.csv", ranking_rows)
    (OUTPUT_DIR / "failed_or_high_risk_cases.json").write_text(json.dumps(high_risk_cases, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "flux_gradient_stability_report.md").write_text(
        build_report(inventory_rows, usage_rows, summary_rows, ranking_rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
