from __future__ import annotations

import csv
import inspect
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.flux.saturation import saturation_3  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "validation_results" / "flex_saturation3_formula_redesign_review"
TRACE_CSV = (
    REPO_ROOT
    / "validation_results"
    / "batch_a_flux_realistic_review"
    / "batch_a_realistic_domain_trace.csv"
)
DEFAULT_NEARZERO = 1.0e-6
DEFAULT_DTYPE = torch.float64
MAX_LOG_FLOAT64 = float(math.log(sys.float_info.max))
REALISTIC_QUANTILES = ("min", "p01", "p05", "median", "mean", "p95", "p99", "max")
BOUNDARY_X_VALUES = (
    0.0,
    1.0e-12,
    1.0e-9,
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    0.05,
    0.1,
    0.5,
    0.9,
    1.0,
    1.1,
)
BOUNDARY_BETA_VALUES = (
    0.0,
    1.0e-12,
    1.0e-9,
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    0.05,
    0.1,
    0.5,
    1.0,
    2.0,
)


@dataclass(frozen=True)
class ContextSpec:
    model: str
    core_file: str
    call_line: int
    call_context_start: int
    call_context_end: int
    saturation3_arguments: str
    beta_parameter_name: str
    beta_parameter_index: int
    storage_argument: str
    capacity_argument: str
    incoming_flux_argument: str
    output_variable: str
    downstream_state_update: str
    expected_output_bounds: str
    notes: str


CONTEXTS = (
    ContextSpec(
        model="flexb",
        core_file="models/core/flexb.py",
        call_line=90,
        call_context_start=88,
        call_context_end=106,
        saturation3_arguments="S1, s1max, beta, P, nearzero=nearzero",
        beta_parameter_name="beta",
        beta_parameter_index=1,
        storage_argument="S1",
        capacity_argument="s1max",
        incoming_flux_argument="P",
        output_variable="flux_ru",
        downstream_state_update=(
            "flux_ru = clamp(flux_ru, 0, P); p_excess = relu(P - flux_ru); "
            "S1_tmp = clamp(S1 + flux_ru, min=nearzero)"
        ),
        expected_output_bounds="0 <= flux_ru <= P",
        notes=(
            "No pre-call cap. Post-call clamp enforces incoming-flux bound before the "
            "soil-moisture state update."
        ),
    ),
    ContextSpec(
        model="flexi",
        core_file="models/core/flexi.py",
        call_line=119,
        call_context_start=117,
        call_context_end=131,
        saturation3_arguments="S2, smax, beta, flux_peff, nearzero=nearzero",
        beta_parameter_name="beta",
        beta_parameter_index=1,
        storage_argument="S2",
        capacity_argument="smax",
        incoming_flux_argument="flux_peff",
        output_variable="flux_ru",
        downstream_state_update=(
            "flux_ru = clamp(flux_ru, 0, flux_peff); rem_peff = relu(flux_peff - flux_ru); "
            "S2_tmp = clamp(S2 + flux_ru, min=nearzero)"
        ),
        expected_output_bounds="0 <= flux_ru <= flux_peff",
        notes=(
            "No pre-call cap. Post-call clamp enforces the throughfall bound before the "
            "soil store update."
        ),
    ),
    ContextSpec(
        model="flexis",
        core_file="models/core/flexis.py",
        call_line=151,
        call_context_start=149,
        call_context_end=162,
        saturation3_arguments="S3, smax, beta, flux_peff, nearzero=nearzero",
        beta_parameter_name="beta",
        beta_parameter_index=1,
        storage_argument="S3",
        capacity_argument="smax",
        incoming_flux_argument="flux_peff",
        output_variable="flux_ru",
        downstream_state_update=(
            "flux_ru = clamp(flux_ru, 0, flux_peff); rem_peff = relu(flux_peff - flux_ru); "
            "S3_tmp = clamp(S3 + flux_ru, min=nearzero)"
        ),
        expected_output_bounds="0 <= flux_ru <= flux_peff",
        notes=(
            "No pre-call cap. Post-call clamp enforces the effective-precipitation bound "
            "before the soil store update."
        ),
    ),
)


@dataclass(frozen=True)
class Candidate:
    option_id: str
    option_name: str
    formula_description: str
    expected_gradient_behavior: str
    expected_output_distortion: str
    preserves_bounds: str
    preserves_hydrological_meaning: str
    implementation_scope: str
    risk_level: str
    recommended_for_testing: str
    reason: str
    implementation_tag: str | None = None


CANDIDATES = (
    Candidate(
        option_id="A0",
        option_name="Current exp form",
        formula_description="In * (1 - 1 / (1 + exp(z))) with z = (S/(Smax+eps) + 0.5) / (beta+eps).",
        expected_gradient_behavior="Finite mathematically, but autograd can return NaN when exp(z) overflows.",
        expected_output_distortion="none",
        preserves_bounds="yes",
        preserves_hydrological_meaning="yes",
        implementation_scope="status_quo",
        risk_level="known_high_at_small_beta",
        recommended_for_testing="baseline_only",
        reason="Baseline for comparison; this is the currently shipped expression.",
        implementation_tag="current_exp_form",
    ),
    Candidate(
        option_id="A1",
        option_name="Stable sigmoid rewrite",
        formula_description="In * sigmoid(z), algebraically equivalent to the current formula.",
        expected_gradient_behavior="Finite for all tested x >= 0 and beta >= 0 because the stable sigmoid path avoids inf * 0 in backward.",
        expected_output_distortion="machine_precision_only",
        preserves_bounds="yes",
        preserves_hydrological_meaning="yes",
        implementation_scope="local_saturation3_only_or_shared_helper",
        risk_level="low",
        recommended_for_testing="yes",
        reason="Exact same mathematics, minimal output drift, removes the observed autograd overflow mechanism.",
        implementation_tag="stable_sigmoid",
    ),
    Candidate(
        option_id="A2",
        option_name="Stable logsigmoid rewrite",
        formula_description="In * exp(logsigmoid(z)), algebraically equivalent to sigmoid(z).",
        expected_gradient_behavior="Finite and numerically stable; same ideal derivatives as A1.",
        expected_output_distortion="machine_precision_only",
        preserves_bounds="yes",
        preserves_hydrological_meaning="yes",
        implementation_scope="local_saturation3_only_or_shared_helper",
        risk_level="low",
        recommended_for_testing="yes",
        reason="Equivalent stable alternative to A1; worth testing only if a log-domain implementation is preferred.",
        implementation_tag="stable_logsigmoid",
    ),
    Candidate(
        option_id="B1",
        option_name="Smooth relative-storage floor",
        formula_description="Replace x with softplus(x / x0) * x0 before the logistic calculation.",
        expected_gradient_behavior="Finite if paired with a stable sigmoid path, but it does not target the real failure term.",
        expected_output_distortion="nonzero_near_empty_store",
        preserves_bounds="yes",
        preserves_hydrological_meaning="partial",
        implementation_scope="local_saturation3_only",
        risk_level="medium",
        recommended_for_testing="no",
        reason="The failure is not caused by x=0 or log(x), so smoothing x adds behavior change without addressing the root cause.",
    ),
    Candidate(
        option_id="C1",
        option_name="Piecewise beta->0 limit",
        formula_description="Use In directly below a beta threshold and the stable sigmoid form above it.",
        expected_gradient_behavior="Finite, but introduces a hard kink in beta at the switching threshold.",
        expected_output_distortion="tiny_on_tested_domains_but_nonzero_in_principle",
        preserves_bounds="yes",
        preserves_hydrological_meaning="mostly",
        implementation_scope="local_saturation3_only",
        risk_level="medium",
        recommended_for_testing="maybe",
        reason="Works numerically but changes the formula and creates a non-smooth beta transition.",
        implementation_tag="piecewise_beta_limit",
    ),
    Candidate(
        option_id="C2",
        option_name="Smooth beta->0 blend",
        formula_description="Blend between In and the stable sigmoid form over a narrow beta transition band.",
        expected_gradient_behavior="Finite and smooth, but the beta derivative now includes the blend schedule.",
        expected_output_distortion="tiny_on_tested_domains_but_model_dependent",
        preserves_bounds="yes",
        preserves_hydrological_meaning="mostly",
        implementation_scope="local_saturation3_only",
        risk_level="medium",
        recommended_for_testing="maybe",
        reason="Removes the hard kink of C1, but it is still a deliberate formula change rather than a pure numerical rewrite.",
        implementation_tag="smooth_beta_blend",
    ),
    Candidate(
        option_id="D1",
        option_name="Beta reparameterization",
        formula_description="Map a raw parameter through a positive transform before passing beta into saturation_3.",
        expected_gradient_behavior="Potentially finite, depending on the downstream formula, but the flux code still contains the same exp overflow path.",
        expected_output_distortion="parameter_space_change_not_formula_change",
        preserves_bounds="n/a",
        preserves_hydrological_meaning="parameterization_only",
        implementation_scope="parameter_transform_layer",
        risk_level="medium",
        recommended_for_testing="no_for_formula_task",
        reason="Outside the flux-formula scope of this review and not sufficient by itself if the exp form is retained.",
        implementation_tag="beta_floor_reparameterized",
    ),
    Candidate(
        option_id="E1",
        option_name="State-domain exclusion",
        formula_description="Handle only x near zero specially while leaving the rest of the formula unchanged.",
        expected_gradient_behavior="Does not solve the observed failures because overflow also occurs at moderate and high relative storage.",
        expected_output_distortion="depends_on_special_case",
        preserves_bounds="yes_if_careful",
        preserves_hydrological_meaning="weak",
        implementation_scope="local_saturation3_only",
        risk_level="high",
        recommended_for_testing="no",
        reason="Rejected by the diagnostics: x=0 is not the trigger; the numerator x+0.5 stays positive everywhere in the tested domain.",
    ),
    Candidate(
        option_id="F1",
        option_name="Shared safe logistic helper",
        formula_description="Introduce a helper that evaluates sigmoid-like hydrologic formulas through stable primitives.",
        expected_gradient_behavior="Same finite behavior as A1/A2 when saturation_3 uses it.",
        expected_output_distortion="machine_precision_only_if_helper_is_exact",
        preserves_bounds="yes",
        preserves_hydrological_meaning="yes",
        implementation_scope="shared_helper_possible",
        risk_level="low_to_medium",
        recommended_for_testing="yes_after_A1",
        reason="Implementation pattern rather than a new formula; useful only if the team wants the stable rewrite reusable elsewhere.",
    ),
)


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _extract_block(path: Path, start_line: int, end_line: int) -> str:
    lines = _read_lines(path)
    return "\n".join(lines[start_line - 1 : end_line])


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.12g}"


def _sigmoid(z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(z)


def _current_exp_form(
    S: torch.Tensor,
    Smax: torch.Tensor,
    beta: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = DEFAULT_NEARZERO,
) -> torch.Tensor:
    ratio = S / (Smax + nearzero)
    return (1.0 - (1.0 / (1.0 + torch.exp((ratio + 0.5) / (beta + nearzero))))) * incoming_flux


def _stable_sigmoid(
    S: torch.Tensor,
    Smax: torch.Tensor,
    beta: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = DEFAULT_NEARZERO,
) -> torch.Tensor:
    ratio = S / (Smax + nearzero)
    z = (ratio + 0.5) / (beta + nearzero)
    return torch.sigmoid(z) * incoming_flux


def _stable_logsigmoid(
    S: torch.Tensor,
    Smax: torch.Tensor,
    beta: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = DEFAULT_NEARZERO,
) -> torch.Tensor:
    ratio = S / (Smax + nearzero)
    z = (ratio + 0.5) / (beta + nearzero)
    return torch.exp(F.logsigmoid(z)) * incoming_flux


def _piecewise_beta_limit(
    S: torch.Tensor,
    Smax: torch.Tensor,
    beta: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = DEFAULT_NEARZERO,
    beta_limit: float = 1.0e-3,
) -> torch.Tensor:
    ratio = S / (Smax + nearzero)
    z = (ratio + 0.5) / (beta + nearzero)
    stable = torch.sigmoid(z) * incoming_flux
    return torch.where(beta <= beta_limit, incoming_flux, stable)


def _smooth_beta_blend(
    S: torch.Tensor,
    Smax: torch.Tensor,
    beta: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = DEFAULT_NEARZERO,
    center: float = 1.0e-3,
    width: float = 2.0e-4,
) -> torch.Tensor:
    ratio = S / (Smax + nearzero)
    z = (ratio + 0.5) / (beta + nearzero)
    stable = torch.sigmoid(z) * incoming_flux
    alpha = torch.sigmoid((beta - center) / width)
    return alpha * stable + (1.0 - alpha) * incoming_flux


def _beta_floor_reparameterized(
    S: torch.Tensor,
    Smax: torch.Tensor,
    beta: torch.Tensor,
    incoming_flux: torch.Tensor,
    nearzero: float = DEFAULT_NEARZERO,
    beta_floor: float = 1.0e-3,
) -> torch.Tensor:
    ratio = S / (Smax + nearzero)
    beta_eff = torch.sqrt(beta * beta + beta_floor * beta_floor)
    z = (ratio + 0.5) / (beta_eff + nearzero)
    return torch.sigmoid(z) * incoming_flux


FORMULA_IMPLS: dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor]] = {
    "current_exp_form": _current_exp_form,
    "stable_sigmoid": _stable_sigmoid,
    "stable_logsigmoid": _stable_logsigmoid,
    "piecewise_beta_limit": _piecewise_beta_limit,
    "smooth_beta_blend": _smooth_beta_blend,
    "beta_floor_reparameterized": _beta_floor_reparameterized,
}


def _relative_storage(S: torch.Tensor, Smax: torch.Tensor, nearzero: float = DEFAULT_NEARZERO) -> torch.Tensor:
    return S / (Smax + nearzero)


def _z_value(S: torch.Tensor, Smax: torch.Tensor, beta: torch.Tensor, nearzero: float = DEFAULT_NEARZERO) -> torch.Tensor:
    ratio = _relative_storage(S, Smax, nearzero=nearzero)
    return (ratio + 0.5) / (beta + nearzero)


def _evaluate_scalar_formula(
    formula_name: str,
    S_value: float,
    Smax_value: float,
    beta_value: float,
    incoming_flux_value: float,
) -> dict[str, float | bool]:
    fn = FORMULA_IMPLS[formula_name]
    S = torch.tensor([S_value], dtype=DEFAULT_DTYPE, requires_grad=True)
    Smax = torch.tensor([Smax_value], dtype=DEFAULT_DTYPE)
    beta = torch.tensor([beta_value], dtype=DEFAULT_DTYPE, requires_grad=True)
    incoming_flux = torch.tensor([incoming_flux_value], dtype=DEFAULT_DTYPE)
    output = fn(S, Smax, beta, incoming_flux, DEFAULT_NEARZERO)
    grad_S, grad_beta = torch.autograd.grad(output.sum(), (S, beta), allow_unused=True)
    out_value = float(output.detach().item())
    grad_S_value = float("nan") if grad_S is None else float(grad_S.detach().item())
    grad_beta_value = float("nan") if grad_beta is None else float(grad_beta.detach().item())
    ratio_value = float(_relative_storage(S.detach(), Smax.detach()).item())
    z_value = float(_z_value(S.detach(), Smax.detach(), beta.detach()).item())
    return {
        "output": out_value,
        "grad_storage": grad_S_value,
        "grad_beta": grad_beta_value,
        "ratio": ratio_value,
        "z": z_value,
        "output_finite": math.isfinite(out_value),
        "grad_storage_finite": math.isfinite(grad_S_value),
        "grad_beta_finite": math.isfinite(grad_beta_value),
        "physically_bounded": (-1.0e-12 <= out_value <= incoming_flux_value + 1.0e-12),
    }


def _analytic_terms(x: float, beta: float, incoming_flux: float = 1.0) -> dict[str, float]:
    z = (x + 0.5) / (beta + DEFAULT_NEARZERO)
    sigma = 1.0 / (1.0 + math.exp(-z))
    sigma_prime = sigma * (1.0 - sigma)
    grad_x = incoming_flux * sigma_prime / (beta + DEFAULT_NEARZERO)
    grad_beta = -incoming_flux * sigma_prime * (x + 0.5) / ((beta + DEFAULT_NEARZERO) ** 2)
    return {
        "stable_output": incoming_flux * sigma,
        "stable_grad_x": grad_x,
        "stable_grad_beta": grad_beta,
    }


def _build_boundary_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    Smax_value = 1.0
    for x in BOUNDARY_X_VALUES:
        S_value = x * (Smax_value + DEFAULT_NEARZERO)
        for beta in BOUNDARY_BETA_VALUES:
            current = _evaluate_scalar_formula("current_exp_form", S_value, Smax_value, beta, 1.0)
            stable = _analytic_terms(x, beta)
            z_value = float(current["z"])
            rows.append(
                {
                    "x": _format_float(x),
                    "beta": _format_float(beta),
                    "output": _format_float(float(current["output"])),
                    "grad_x": _format_float(float(current["grad_storage"])),
                    "grad_beta": _format_float(float(current["grad_beta"])),
                    "finite_output_flag": bool(current["output_finite"]),
                    "finite_gradient_flag": bool(current["grad_storage_finite"] and current["grad_beta_finite"]),
                    "log_x": "-inf" if x == 0.0 else _format_float(math.log(x)),
                    "physically_bounded": bool(current["physically_bounded"]),
                    "z": _format_float(z_value),
                    "expected_exp_overflow_float64": bool(z_value > MAX_LOG_FLOAT64),
                    "stable_output_reference": _format_float(float(stable["stable_output"])),
                    "stable_grad_x_reference": _format_float(float(stable["stable_grad_x"])),
                    "stable_grad_beta_reference": _format_float(float(stable["stable_grad_beta"])),
                }
            )
    return rows


def _load_realistic_anchor_points() -> list[dict[str, object]]:
    rows = list(csv.DictReader(TRACE_CSV.open(encoding="utf-8")))
    points: list[dict[str, object]] = []
    for model in ("flexb", "flexi", "flexis"):
        model_rows = [row for row in rows if row["formula"] == "saturation_3" and row["active_model"] == model]
        for regime in sorted({row["forcing_regime"] for row in model_rows}):
            grouped = {row["argument_name"]: row for row in model_rows if row["forcing_regime"] == regime}
            if set(grouped) != {"S", "Smax", "p1", "incoming_flux"}:
                continue
            for quantile in REALISTIC_QUANTILES:
                points.append(
                    {
                        "dataset": "realistic_trace_anchor",
                        "model": model,
                        "forcing_regime": regime,
                        "anchor": quantile,
                        "S": float(grouped["S"][quantile]),
                        "Smax": float(grouped["Smax"][quantile]),
                        "beta": float(grouped["p1"][quantile]),
                        "incoming_flux": float(grouped["incoming_flux"][quantile]),
                    }
                )
    return points


def _build_boundary_points_for_comparison() -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    Smax_value = 1.0
    for x in BOUNDARY_X_VALUES:
        S_value = x * (Smax_value + DEFAULT_NEARZERO)
        for beta in BOUNDARY_BETA_VALUES:
            points.append(
                {
                    "dataset": "boundary_grid",
                    "model": "synthetic",
                    "forcing_regime": "synthetic",
                    "anchor": f"x={x:.12g}|beta={beta:.12g}",
                    "S": S_value,
                    "Smax": Smax_value,
                    "beta": beta,
                    "incoming_flux": 1.0,
                }
            )
    return points


def _evaluate_dataset(
    formula_name: str,
    dataset_points: list[dict[str, object]],
) -> list[dict[str, object]]:
    evaluated: list[dict[str, object]] = []
    for point in dataset_points:
        result = _evaluate_scalar_formula(
            formula_name=formula_name,
            S_value=float(point["S"]),
            Smax_value=float(point["Smax"]),
            beta_value=float(point["beta"]),
            incoming_flux_value=float(point["incoming_flux"]),
        )
        evaluated.append({**point, **result})
    return evaluated


def _candidate_comparison_rows(
    boundary_points: list[dict[str, object]],
    realistic_points: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    rows: list[dict[str, object]] = []
    summary: dict[tuple[str, str], dict[str, object]] = {}
    datasets = {
        "boundary_grid": boundary_points,
        "realistic_trace_anchor": realistic_points,
    }
    baseline_by_dataset = {
        name: _evaluate_dataset("current_exp_form", points) for name, points in datasets.items()
    }
    candidate_tags = [
        "current_exp_form",
        "stable_sigmoid",
        "stable_logsigmoid",
        "piecewise_beta_limit",
        "smooth_beta_blend",
        "beta_floor_reparameterized",
    ]
    for dataset_name, points in datasets.items():
        baseline = baseline_by_dataset[dataset_name]
        for formula_name in candidate_tags:
            evaluated = _evaluate_dataset(formula_name, points)
            diffs = [
                abs(float(row["output"]) - float(base_row["output"]))
                for row, base_row in zip(evaluated, baseline)
                if math.isfinite(float(row["output"])) and math.isfinite(float(base_row["output"]))
            ]
            base_sq = sum(float(base_row["output"]) ** 2 for base_row in baseline if math.isfinite(float(base_row["output"])))
            diff_sq = sum(
                (float(row["output"]) - float(base_row["output"])) ** 2
                for row, base_row in zip(evaluated, baseline)
                if math.isfinite(float(row["output"])) and math.isfinite(float(base_row["output"]))
            )
            grad_beta_values = [float(row["grad_beta"]) for row in evaluated if math.isfinite(float(row["grad_beta"]))]
            outputs = [float(row["output"]) for row in evaluated if math.isfinite(float(row["output"]))]
            physical_bound_violations = sum(
                1
                for row, point in zip(evaluated, points)
                if not (-1.0e-12 <= float(row["output"]) <= float(point["incoming_flux"]) + 1.0e-12)
            )
            leakage_count = sum(
                1
                for row, base_row in zip(evaluated, baseline)
                if abs(float(base_row["output"])) <= 1.0e-15 and float(row["output"]) > 0.0
            )
            row = {
                "dataset": dataset_name,
                "formula_name": formula_name,
                "max_abs_output_diff_vs_current": _format_float(max(diffs) if diffs else float("nan")),
                "relative_L2_output_diff_vs_current": _format_float(math.sqrt(diff_sq / base_sq) if base_sq > 0.0 else 0.0),
                "max_abs_grad_beta": _format_float(max(abs(value) for value in grad_beta_values) if grad_beta_values else float("nan")),
                "grad_beta_nan_count": sum(
                    1 for result in evaluated if math.isnan(float(result["grad_beta"]))
                ),
                "grad_beta_inf_count": sum(
                    1 for result in evaluated if math.isinf(float(result["grad_beta"]))
                ),
                "grad_storage_nan_count": sum(
                    1 for result in evaluated if math.isnan(float(result["grad_storage"]))
                ),
                "grad_storage_inf_count": sum(
                    1 for result in evaluated if math.isinf(float(result["grad_storage"]))
                ),
                "physical_bound_violation_count": physical_bound_violations,
                "leakage_count": leakage_count,
                "output_min": _format_float(min(outputs) if outputs else float("nan")),
                "output_max": _format_float(max(outputs) if outputs else float("nan")),
            }
            rows.append(row)
            summary[(dataset_name, formula_name)] = row
    return rows, summary


def _source_and_context_markdown() -> str:
    saturation_source = inspect.getsource(saturation_3).rstrip()
    parts = [
        "# saturation_3 Source And FLEX Context",
        "",
        "## Flux source",
        "",
        "`models/flux/saturation.py:52-64`",
        "",
        "```python",
        saturation_source,
        "```",
        "",
        "### Input variables",
        "- `S`: current soil-moisture storage state for the FLEX soil store.",
        "- `Smax`: soil-moisture storage capacity parameter.",
        "- `p1`: FLEX `beta` parameter.",
        "- `incoming_flux`: precipitation or effective precipitation entering the store.",
        "- `nearzero`: dMoT numerical epsilon shared across denominator terms.",
        "",
        "### Output variable",
        "- Return value: `flux_ru`, the infiltration flux entering the soil-moisture store before downstream clamping.",
        "",
        "### Formula type and physical meaning",
        "- Formula type: logistic saturation partition in relative storage `S / Smax`.",
        "- Physical meaning in FLEX contexts: a saturation-dependent infiltration fraction multiplied by the incoming water flux.",
        "- Expected bounds before post-call clamps: mathematically `0 < out_frac < 1` for finite positive `beta + nearzero`, so `0 <= flux_ru <= incoming_flux` for non-negative forcing.",
        "",
    ]
    for context in CONTEXTS:
        core_path = REPO_ROOT / context.core_file
        block = _extract_block(core_path, context.call_context_start, context.call_context_end)
        parts.extend(
            [
                f"## {context.model}",
                "",
                f"`{context.core_file}:{context.call_context_start}-{context.call_context_end}`",
                "",
                "```python",
                block,
                "```",
                "",
                f"- `saturation_3` call: `{context.output_variable} = saturation_3({context.saturation3_arguments})`",
                f"- Beta parameter passed: `{context.beta_parameter_name}` (parameter index `{context.beta_parameter_index}` in the model bounds order).",
                f"- Storage argument: `{context.storage_argument}`.",
                f"- Capacity argument: `{context.capacity_argument}`.",
                f"- Incoming flux argument: `{context.incoming_flux_argument}`.",
                f"- Downstream use: `{context.downstream_state_update}`.",
                f"- Expected output bounds: `{context.expected_output_bounds}`.",
                f"- Notes: {context.notes}",
                "",
            ]
        )
    return "\n".join(parts).rstrip() + "\n"


def _call_context_rows() -> list[dict[str, object]]:
    return [
        {
            "model": context.model,
            "core_file": context.core_file,
            "call_site_lines": f"{context.call_context_start}-{context.call_context_end}",
            "saturation3_arguments": context.saturation3_arguments,
            "beta_parameter_name": context.beta_parameter_name,
            "beta_parameter_index": context.beta_parameter_index,
            "storage_argument": context.storage_argument,
            "capacity_argument": context.capacity_argument,
            "incoming_flux_argument": context.incoming_flux_argument,
            "output_variable": context.output_variable,
            "downstream_state_update": context.downstream_state_update,
            "expected_output_bounds": context.expected_output_bounds,
            "notes": context.notes,
        }
        for context in CONTEXTS
    ]


def _gradient_derivation_markdown(boundary_rows: list[dict[str, object]]) -> str:
    overflow_rows = [
        row for row in boundary_rows if row["expected_exp_overflow_float64"] and row["finite_output_flag"] and not row["finite_gradient_flag"]
    ]
    finite_output_nonfinite_grad = len(overflow_rows)
    lines = [
        "# saturation_3 Gradient Derivation",
        "",
        "## Expression",
        "",
        "Let",
        "",
        "- `eps = nearzero`",
        "- `x = S / (Smax + eps)`",
        "- `z = (x + 0.5) / (beta + eps)`",
        "- `sigma(z) = 1 / (1 + exp(-z))`",
        "- `I = incoming_flux`",
        "",
        "Then the implemented formula is",
        "",
        "`y = I * (1 - 1 / (1 + exp(z))) = I * sigma(z)`",
        "",
        "The current code uses the first algebraic form, but the second form is mathematically equivalent and numerically more stable.",
        "",
        "## Derivatives",
        "",
        "Using `sigma'(z) = sigma(z) * (1 - sigma(z))`,",
        "",
        "- `dy / dI = sigma(z)`",
        "- `dy / dx = I * sigma(z) * (1 - sigma(z)) / (beta + eps)`",
        "- `dy / dS = I * sigma(z) * (1 - sigma(z)) / ((Smax + eps) * (beta + eps))`",
        "- `dy / dSmax = -I * sigma(z) * (1 - sigma(z)) * S / ((Smax + eps)^2 * (beta + eps))`",
        "- `dy / dbeta = -I * sigma(z) * (1 - sigma(z)) * (x + 0.5) / (beta + eps)^2`",
        "",
        "## Alternative backward expression that matches the current exp implementation",
        "",
        "If `e = exp(z)`, then",
        "",
        "- `y = I * e / (1 + e)`",
        "- `dy / dz = I * e / (1 + e)^2`",
        "- `dy / dbeta = -I * e / (1 + e)^2 * (x + 0.5) / (beta + eps)^2`",
        "",
        "This is the same derivative mathematically, but it is the form that exposes the floating-point failure: when `e = inf`, the factor `e / (1 + e)^2` is an `inf * 0` type cancellation in backward.",
        "",
        "## Singular points and failure conditions",
        "",
        "- There is no `x ** beta` term anywhere in `saturation_3`.",
        "- There is no `log(x)` term anywhere in `saturation_3`.",
        "- `x = 0` is not itself singular because the formula depends on `x + 0.5`, not `log(x)` or `x^beta`.",
        "- The only true denominator singularities of the dMoT formula would be `Smax = -eps` or `beta = -eps`, which are outside the physical parameter/state domain.",
        "- The practical non-finite-gradient condition is computational overflow of `exp(z)` in the current implementation.",
        "",
        "## Why finite outputs can coexist with non-finite gradients",
        "",
        f"- Boundary-grid cases with finite output but non-finite gradient: `{finite_output_nonfinite_grad}`.",
        "- For large positive `z`, the forward value saturates to `sigma(z) ~= 1`, so the output stays finite and bounded by `incoming_flux`.",
        "- Backward still differentiates through `exp(z)`, so once `exp(z)` overflows, PyTorch can produce `NaN` gradients even though the forward output is exactly `incoming_flux`.",
        "",
        "## Why beta = 1e-6, 1e-5, and 1e-4 still fail",
        "",
        f"- In float64, `exp(z)` overflows once `z > {MAX_LOG_FLOAT64:.2f}`.",
        "- Because `x + 0.5 >= 0.5` for every realistic non-negative relative storage, small beta makes `z` huge even when storage is empty.",
        "- Approximate float64 overflow threshold: `beta_crit(x) ~= (x + 0.5) / 709.78 - eps`.",
        "- Example thresholds with `eps = 1e-6`:",
        "  - `x = 0.0  -> beta_crit ~= 7.03e-4`",
        "  - `x = 0.5  -> beta_crit ~= 1.41e-3`",
        "  - `x = 1.1  -> beta_crit ~= 2.25e-3`",
        "- Therefore beta values `1e-6`, `1e-5`, and `1e-4` are deep inside the overflow region, and even `1e-3` still fails for moderate-to-high relative storage.",
        "",
        "## Root-cause decision for the review questions",
        "",
        "- `x ** beta` near zero: not applicable.",
        "- `log(x)` in `dbeta`: not applicable.",
        "- Zero or near-zero relative storage: not required for failure.",
        "- Exact boundary beta values: they worsen the problem, but the failure is not limited to the exact boundary because `beta = 1e-4` and some `beta = 1e-3` cases still overflow.",
        "- Clamp/ReLU discontinuity: not the cause inside `saturation_3`; the non-finite gradients arise before post-call clamps matter.",
        "- Division by storage capacity: not singular in the realistic domain because `Smax > 0`; it only feeds the ratio term.",
        "- Actual cause: overflow of `exp((S/(Smax+eps) + 0.5) / (beta+eps))` in the current algebraic form, causing unstable backward cancellation.",
        "",
    ]
    return "\n".join(lines)


def _reference_formula_markdown() -> str:
    matlab_flux = _extract_block(
        Path("/home/jingxin/code/dmg-research/MARRMoT/Flux files/saturation_3.m"),
        1,
        20,
    )
    julia_flux = _extract_block(
        Path("/home/jingxin/code/dmg-research/MARRMoT/Julia File/saturation.jl"),
        28,
        36,
    )
    lines = [
        "# saturation_3 Reference Formula Review",
        "",
        "## Reference files inspected",
        "- `/home/jingxin/code/dmg-research/MARRMoT/Flux files/saturation_3.m`",
        "- `/home/jingxin/code/dmg-research/MARRMoT/Julia File/saturation.jl`",
        "- `/home/jingxin/code/dmg-research/MARRMoT/Model files/m_21_flexb_9p_3s.m`",
        "- `/home/jingxin/code/dmg-research/MARRMoT/Model files/m_26_flexi_10p_4s.m`",
        "- `/home/jingxin/code/dmg-research/MARRMoT/Model files/m_34_flexis_12p_5s.m`",
        "",
        "## MATLAB saturation_3",
        "",
        "```matlab",
        matlab_flux,
        "```",
        "",
        "## Julia saturation_3",
        "",
        "```julia",
        julia_flux,
        "```",
        "",
        "## Findings",
        "",
        "- The original MARRMoT flux uses the same logistic expression: `out = (1 - 1 / (1 + exp((S / Smax + 0.5) / beta))) .* In`.",
        "- The original FLEXB, FLEXI, and FLEXIS model files all set the beta parameter range to `[0, 10]`.",
        "- The MATLAB and Julia references do not add epsilon to `Smax` or `beta` inside `saturation_3`.",
        "- No MATLAB-side special case for `beta = 0` is present in the reference flux or model files.",
        "- No explicit guard for zero storage, zero relative storage, or large positive `z` is present.",
        "",
        "## Interpretation",
        "",
        "- In the original forward-only MARRMoT context, `beta = 0` is effectively a degenerate limit that drives `z` to positive infinity for the FLEX storage domain, so the flux saturates to `In`.",
        "- That makes beta zero mathematically degenerate for the reference formula, even though forward evaluation can still return a finite limit in floating-point arithmetic.",
        "- The reference implementation gives no autograd guidance because MARRMoT does not differentiate the formula with respect to beta.",
        "- The dMoT `nearzero` additions make the forward expression defined at `beta = 0`, but they do not by themselves stabilize the current `exp(z)` backward path.",
        "",
    ]
    return "\n".join(lines)


def _candidate_rows() -> list[dict[str, object]]:
    return [
        {
            "option_id": candidate.option_id,
            "option_name": candidate.option_name,
            "formula_description": candidate.formula_description,
            "expected_gradient_behavior": candidate.expected_gradient_behavior,
            "expected_output_distortion": candidate.expected_output_distortion,
            "preserves_bounds": candidate.preserves_bounds,
            "preserves_hydrological_meaning": candidate.preserves_hydrological_meaning,
            "implementation_scope": candidate.implementation_scope,
            "risk_level": candidate.risk_level,
            "recommended_for_testing": candidate.recommended_for_testing,
            "reason": candidate.reason,
        }
        for candidate in CANDIDATES
    ]


def _redesign_report(
    boundary_rows: list[dict[str, object]],
    comparison_summary: dict[tuple[str, str], dict[str, object]],
) -> str:
    boundary_nan_rows = sum(1 for row in boundary_rows if not row["finite_gradient_flag"])
    current_boundary = comparison_summary[("boundary_grid", "current_exp_form")]
    stable_boundary = comparison_summary[("boundary_grid", "stable_sigmoid")]
    stable_realistic = comparison_summary[("realistic_trace_anchor", "stable_sigmoid")]
    beta_floor_realistic = comparison_summary[("realistic_trace_anchor", "beta_floor_reparameterized")]
    lines = [
        "# saturation_3 Formula Redesign Report",
        "",
        "## 1. Scope",
        "- Investigated `models/flux/saturation.py::saturation_3` as used by `flexb`, `flexi`, and `flexis`.",
        "- No active hydrological source code, parameter bounds, unit hydrograph code, soft-gate defaults, or water-balance fixes were changed.",
        "",
        "## 2. Current formula and FLEX call context",
        "- `saturation_3` receives `(S, Smax, beta, incoming_flux, nearzero)` and returns a soil-infiltration flux used before downstream clamping.",
        "- FLEXB uses `(S1, s1max, beta, P)`, FLEXI uses `(S2, smax, beta, flux_peff)`, and FLEXIS uses `(S3, smax, beta, flux_peff)`.",
        "- All three cores clamp the result to `[0, incoming_flux]` immediately after the call and then add it into the soil state.",
        "",
        "## 3. Mathematical expression",
        "- Define `x = S / (Smax + eps)` and `z = (x + 0.5) / (beta + eps)` with `eps = nearzero`.",
        "- Current dMoT implementation: `y = incoming_flux * (1 - 1 / (1 + exp(z)))`.",
        "- Equivalent stable form: `y = incoming_flux * sigmoid(z)`.",
        "",
        "## 4. Gradient derivation",
        "- `dy/dbeta = -incoming_flux * sigmoid(z) * (1 - sigmoid(z)) * (x + 0.5) / (beta + eps)^2`.",
        "- `dy/dS = incoming_flux * sigmoid(z) * (1 - sigmoid(z)) / ((Smax + eps) * (beta + eps))`.",
        "- The analytic derivative is finite across the realistic domain whenever `Smax + eps > 0` and `beta + eps > 0`.",
        "",
        "## 5. Singularities and failure conditions",
        f"- Boundary-grid non-finite gradient cases in the current implementation: `{boundary_nan_rows}` out of `{len(boundary_rows)}`.",
        f"- Boundary-grid current-form NaN counts: grad_beta=`{current_boundary['grad_beta_nan_count']}`, grad_storage=`{current_boundary['grad_storage_nan_count']}`.",
        "- The root cause is overflow of `exp(z)` in the current algebraic form, not a physical-domain bound violation.",
        "- `x = 0` is not required for failure; the numerator `x + 0.5` stays positive even at empty storage.",
        "- `x ** beta`, `log(x)`, and clamp/ReLU kinks are not part of this formula’s pathology.",
        "",
        "## 6. Reference/MARRMoT comparison",
        "- Local MATLAB and Julia MARRMoT references use the same logistic formula without epsilon guards inside the flux.",
        "- Original FLEX model files allow beta in `[0, 10]`.",
        "- The reference implementation offers no special handling for `beta = 0` or gradient-based calibration.",
        "",
        "## 7. Boundary-grid results",
        f"- Stable sigmoid rewrite boundary NaN counts: grad_beta=`{stable_boundary['grad_beta_nan_count']}`, grad_storage=`{stable_boundary['grad_storage_nan_count']}`.",
        f"- Stable sigmoid max absolute output difference vs current on the boundary grid: `{stable_boundary['max_abs_output_diff_vs_current']}`.",
        "- The stable rewrite matches the current forward output to machine precision where the current output is finite, while removing the non-finite gradients.",
        "",
        "## 8. Why parameter-bound change alone was insufficient",
        "- Increasing the lower beta bound from `0` to `1e-6` avoids exact zero sampling, but it does not prevent `z` from becoming extremely large.",
        "- With `x >= 0`, `z >= 0.5 / (beta + eps)`, so beta values such as `1e-6`, `1e-5`, and `1e-4` remain far inside the `exp(z)` overflow region.",
        "- Even `beta = 1e-3` still overflows for moderate and high relative storage.",
        "",
        "## 9. Candidate rewrite options",
        "- Exact stable algebraic rewrites: `sigmoid(z)` and `exp(logsigmoid(z))`.",
        "- Beta-limit special casing: piecewise or smooth blend to the `beta -> 0` limit `incoming_flux`.",
        "- More intrusive alternatives such as storage-floor smoothing or beta reparameterization were reviewed but are not root-cause fixes for this formula.",
        "",
        "## 10. Candidate numerical comparison",
        f"- Stable sigmoid on realistic trace anchors: max output diff=`{stable_realistic['max_abs_output_diff_vs_current']}`, relative L2 diff=`{stable_realistic['relative_L2_output_diff_vs_current']}`, grad_beta NaN count=`{stable_realistic['grad_beta_nan_count']}`.",
        f"- Beta-floor reparameterization on realistic trace anchors: max output diff=`{beta_floor_realistic['max_abs_output_diff_vs_current']}`, relative L2 diff=`{beta_floor_realistic['relative_L2_output_diff_vs_current']}`.",
        "- Piecewise and smooth beta-limit rewrites were also finite on the tested domains, but they add avoidable formula changes.",
        "",
        "## 11. Recommended candidate(s) for later implementation test",
        "- `A1 stable sigmoid rewrite` is the best first candidate.",
        "- `A2 stable logsigmoid rewrite` is an acceptable alternative if the team prefers a log-domain primitive.",
        "- `F1 shared safe logistic helper` is worth considering only if the stable rewrite should be reused across other formulas.",
        "",
        "## 12. Candidate(s) not recommended",
        "- `B1 smooth relative-storage floor`: rejected because the issue is not caused by zero storage or `log(x)`.",
        "- `E1 state-domain exclusion`: rejected because failures also occur at moderate and high relative storage.",
        "- `D1 beta reparameterization`: outside the scope of a flux-formula fix and unnecessary if the formula itself is stabilized.",
        "",
        "## 13. Whether any immediate source-code change is justified now",
        "- No active source-code change is made in this task.",
        "- The investigation is strong enough to justify a later controlled implementation test of the exact stable sigmoid rewrite.",
        "",
        "## 14. Required validation if a rewrite is later implemented",
        "- Re-run the FLEX saturation boundary diagnostics and realistic-domain trace checks.",
        "- Re-run `tests/test_flex_saturation3_parameter_bound_fix.py`, `tests/test_batch_a_flux_realistic_stability.py`, `tests/test_core_water_balance.py`, and `tests/test_formula_smoothing_diagnostics.py`.",
        "- Compare calibrated or benchmark FLEX outputs before and after the change to confirm that only numerical stability changes, not hydrological behavior.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_context_path = OUTPUT_DIR / "saturation3_source_and_context.md"
    call_context_csv_path = OUTPUT_DIR / "saturation3_call_context.csv"
    gradient_md_path = OUTPUT_DIR / "saturation3_gradient_derivation.md"
    reference_md_path = OUTPUT_DIR / "saturation3_reference_formula_review.md"
    boundary_csv_path = OUTPUT_DIR / "saturation3_boundary_singularity_grid.csv"
    candidates_csv_path = OUTPUT_DIR / "saturation3_candidate_rewrites.csv"
    comparison_csv_path = OUTPUT_DIR / "saturation3_candidate_rewrite_comparison.csv"
    redesign_md_path = OUTPUT_DIR / "saturation3_formula_redesign_report.md"

    boundary_rows = _build_boundary_rows()
    realistic_points = _load_realistic_anchor_points()
    boundary_points = _build_boundary_points_for_comparison()
    comparison_rows, comparison_summary = _candidate_comparison_rows(boundary_points, realistic_points)

    _write_text(source_context_path, _source_and_context_markdown())
    _write_csv(call_context_csv_path, _call_context_rows())
    _write_text(gradient_md_path, _gradient_derivation_markdown(boundary_rows))
    _write_text(reference_md_path, _reference_formula_markdown())
    _write_csv(boundary_csv_path, boundary_rows)
    _write_csv(candidates_csv_path, _candidate_rows())
    _write_csv(comparison_csv_path, comparison_rows)
    _write_text(redesign_md_path, _redesign_report(boundary_rows, comparison_summary))

    print("Wrote review artifacts to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
