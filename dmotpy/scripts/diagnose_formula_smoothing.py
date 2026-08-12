from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests import reference_formula_numpy as ref
from tests.dmot_formula_wrappers import WRAPPERS


OUTPUT_DIR = PROJECT_ROOT / "validation_results" / "formula_smoothing"
PLOTS_DIR = OUTPUT_DIR / "plots"
MARRMOT_ROOT = Path("/home/jingxin/code/dmg-research/MARRMoT")


@dataclass(frozen=True)
class FormulaCandidate:
    candidate_id: str
    model_name: str
    dmot_file: str
    dmot_line_start: int
    dmot_line_end: int
    function_or_class: str
    formula_type: str
    dmot_expression_summary: str
    matlab_file: str
    matlab_line_start: int | None
    matlab_line_end: int | None
    matlab_expression_summary: str
    reason_for_inclusion: str
    risk_level_initial: str
    domain: tuple[float, float]
    threshold: float
    reference_fn: Callable[[np.ndarray], np.ndarray]
    bound_min: float | None = 0.0
    bound_max: float | None = None
    mapping_note: str = "inferred hard-threshold equivalent"
    forced_issue: str = ""


def _rel_path(path: str) -> str:
    return str(Path(path))


def _np_grid(lo: float, hi: float, threshold: float) -> np.ndarray:
    span = hi - lo
    rng = np.random.default_rng(20260623 + int(abs(threshold) * 1000) % 1000)
    base = np.linspace(lo, hi, 1201)
    near = threshold + np.r_[np.linspace(-0.1, 0.1, 401), np.linspace(-1.0, 1.0, 401)]
    eps = np.array([0.0, np.nextafter(0.0, 1.0), 1.0e-12, 1.0e-9, 1.0e-6])
    around_zero = np.r_[-eps[::-1], eps]
    random = rng.uniform(lo, hi, 300)
    values = np.concatenate([base, near, around_zero, random, [threshold, lo, hi]])
    values = values[np.isfinite(values)]
    values = np.clip(values, lo, hi)
    return np.unique(values.astype(np.float64))


def _snip(path: Path, start: int | None, end: int | None) -> str:
    if start is None or end is None or not path.exists():
        return "not found"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[max(start - 1, 0) : min(end, len(lines))])


def _metrics(dmot: np.ndarray, reference: np.ndarray, x: np.ndarray, threshold: float, candidate: FormulaCandidate) -> dict[str, float | int | bool]:
    diff = dmot - reference
    abs_diff = np.abs(diff)
    denom = np.maximum(np.abs(reference), 1.0e-12)
    near = np.abs(x - threshold) <= max(0.1, 0.02 * max(abs(threshold), 1.0))
    near_diff = diff[near] if np.any(near) else diff
    monotonic_violations = int(np.sum(np.diff(dmot[np.argsort(x)]) < -1.0e-9))
    lower_violation = int(np.sum(dmot < (candidate.bound_min if candidate.bound_min is not None else -np.inf) - 1.0e-9))
    upper_violation = 0
    if candidate.bound_max is not None:
        upper_violation = int(np.sum(dmot > candidate.bound_max + 1.0e-9))
    return {
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_relative_diff": float(np.max(abs_diff / denom)),
        "relative_l2_diff": float(np.linalg.norm(diff) / max(np.linalg.norm(reference), 1.0e-12)),
        "signed_bias_mean": float(np.mean(diff)),
        "signed_bias_near_threshold": float(np.mean(near_diff)),
        "max_diff_near_threshold": float(np.max(np.abs(near_diff))),
        "monotonicity_violations": monotonic_violations,
        "bound_violations": lower_violation + upper_violation,
        "negative_output_violations": lower_violation,
        "conservation_relevant_discrepancy": float(np.sum(diff)),
    }


def _torch_eval(candidate_id: str, x: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    tensor = torch.tensor(x, dtype=torch.float64, requires_grad=requires_grad)
    return WRAPPERS[candidate_id](tensor)


def _finite_difference_torch(candidate_id: str, x: np.ndarray, h: float = 1.0e-5) -> np.ndarray:
    y_plus = _torch_eval(candidate_id, x + h).detach().cpu().numpy()
    y_minus = _torch_eval(candidate_id, x - h).detach().cpu().numpy()
    return (y_plus - y_minus) / (2.0 * h)


def _finite_difference_ref(fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, h: float = 1.0e-5) -> np.ndarray:
    return (fn(x + h) - fn(x - h)) / (2.0 * h)


def _gradient_metrics(candidate: FormulaCandidate, x: np.ndarray) -> dict[str, float | int]:
    torch_x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    y = WRAPPERS[candidate.candidate_id](torch_x)
    y.sum().backward()
    autograd = torch_x.grad.detach().cpu().numpy()
    fd = _finite_difference_torch(candidate.candidate_id, x)
    ref_fd = _finite_difference_ref(candidate.reference_fn, x)
    finite_mask = np.isfinite(autograd) & np.isfinite(fd)
    err = np.abs(autograd[finite_mask] - fd[finite_mask]) if np.any(finite_mask) else np.array([np.inf])
    near = np.abs(x - candidate.threshold) <= max(0.1, 0.02 * max(abs(candidate.threshold), 1.0))
    near_grad = np.abs(autograd[near]) if np.any(near) else np.abs(autograd)
    max_abs_grad = float(np.nanmax(np.abs(autograd)))
    saturation_ratio = float(np.mean(np.abs(autograd) < 1.0e-8))
    spike_threshold = max(100.0, 20.0 * np.nanmedian(np.abs(autograd) + 1.0e-12))
    sign_mismatch = int(np.sum(np.sign(autograd[finite_mask]) != np.sign(fd[finite_mask])))
    return {
        "autograd_fd_max_error": float(np.max(err)),
        "autograd_fd_mean_error": float(np.mean(err)),
        "nan_gradient_count": int(np.isnan(autograd).sum()),
        "inf_gradient_count": int(np.isinf(autograd).sum()),
        "max_abs_gradient": max_abs_grad,
        "minimum_abs_gradient_near_active_transition": float(np.nanmin(near_grad)),
        "gradient_saturation_ratio": saturation_ratio,
        "gradient_spike_count": int(np.sum(np.abs(autograd) > spike_threshold)),
        "sign_mismatch_count": sign_mismatch,
        "hard_reference_fd_max_abs_gradient": float(np.nanmax(np.abs(ref_fd))),
        "vanishing_gradient_region_width": float(np.ptp(x[np.abs(autograd) < 1.0e-8])) if np.any(np.abs(autograd) < 1.0e-8) else 0.0,
        "exploding_gradient_region_width": float(np.ptp(x[np.abs(autograd) > 100.0])) if np.any(np.abs(autograd) > 100.0) else 0.0,
    }


def build_candidates() -> list[FormulaCandidate]:
    return [
        FormulaCandidate("F001", "shared_flux", "models/flux/smooth.py", 4, 18, "soft_gate_storage_above", "smoothed_threshold", "sigmoid(k / abs(threshold) * (S - threshold))", "not found", None, None, "MARRMoT helper not present; hard above-threshold gate used", "core smoothing primitive controls many storage gates", "medium", (0.0, 100.0), 50.0, lambda x: ref.smooth_storage_gate_above_hard(x, 50.0), bound_max=1.0, mapping_note="MARRMoT helper definition not found"),
        FormulaCandidate("F002", "shared_flux", "models/flux/smooth.py", 31, 44, "soft_gate_temperature_below", "snow_rain_partition", "sigmoid(5 * (threshold - T))", "not found", None, None, "MARRMoT helper not present; hard snow/rain gate used", "temperature threshold partition affects snowfall and rainfall", "medium", (-5.0, 5.0), 0.0, lambda x: ref.smooth_temperature_snow_hard(x, 0.0), bound_max=1.0, mapping_note="MARRMoT helper definition not found"),
        FormulaCandidate("F003", "shared_flux", "models/flux/snowfall.py", 6, 18, "snowfall_1", "snow_rain_partition", "P * smooth_temperature_gate", "MARRMoT/Flux files/snowfall_1.m", 19, 22, "In .* smoothThreshold_temperature_logistic(T,p1)", "smooth snow partition can bias precipitation phase near threshold", "medium", (-5.0, 5.0), 0.0, lambda x: ref.snowfall_1_hard(x, 0.0), bound_max=10.0),
        FormulaCandidate("F004", "shared_flux", "models/flux/rainfall.py", 6, 18, "rainfall_1", "snow_rain_partition", "P * (1 - smooth_temperature_gate)", "MARRMoT/Flux files/rainfall_1.m", 46, 49, "In .* (1 - smoothThreshold_temperature_logistic(T,p1))", "smooth rain partition can bias precipitation phase near threshold", "medium", (-5.0, 5.0), 0.0, lambda x: ref.rainfall_1_hard(x, 0.0), bound_max=10.0),
        FormulaCandidate("F005", "shared_flux", "models/flux/melt.py", 33, 64, "melt_3", "snowmelt_threshold", "min(max(ddf*(T-Tm),0),S1) * (1 - storage_gate)", "MARRMoT/Flux files/melt_3.m", 25, 29, "min(max(p1*(T-p2),0),S1/dt).*smoothThreshold_storage_logistic(S2,St)", "comments note possible orientation ambiguity for glacier melt gate", "medium", (0.0, 0.1), 0.01, lambda x: ref.melt_3_hard(x, 0.01), bound_max=9.0, mapping_note="hard low-snow interpretation inferred from model comment"),
        FormulaCandidate("F006", "shared_flux", "models/flux/saturation.py", 6, 14, "saturation_1", "saturation_excess", "In * storage_gate_above(S,Smax)", "MARRMoT/Flux files/saturation_1.m", 73, 78, "In .* (1 - smoothThreshold_storage_logistic(S,Smax))", "central saturation excess threshold used by many models", "medium", (0.0, 100.0), 50.0, lambda x: ref.saturation_1_hard(x, 50.0), bound_max=10.0),
        FormulaCandidate("F007", "shared_flux", "models/flux/saturation.py", 172, 181, "saturation_9", "deficit", "In * (1 - storage_gate_above(S,St))", "MARRMoT/Flux files/saturation_9.m", 104, 109, "In .* smoothThreshold_storage_logistic(S,St)", "deficit-store threshold orientation is easy to invert", "medium", (0.0, 0.1), 0.01, lambda x: ref.saturation_9_hard(x, 0.01), bound_max=10.0),
        FormulaCandidate("F008", "lascam", "models/flux/saturation.py", 200, 219, "saturation_11", "saturation_excess", "In * min(1,p1*ratio^p2) * storage_gate_above(S,Smin)", "MARRMoT/Flux files/saturation_11.m", 26, 32, "term .* (1-smoothThreshold_storage_logistic(S,Smin))", "LASCAM contributing-area threshold and power law", "medium", (0.0, 100.0), 10.0, lambda x: ref.saturation_11_hard(x, 10.0), bound_max=10.0),
        FormulaCandidate("F009", "smar", "models/flux/evap.py", 197, 213, "evap_14", "evapotranspiration_limiter", "min((p1+eps)^p2*Ep,S1) * (1 - storage_gate_above(S2,S2min))", "MARRMoT/Flux files/evap_14.m", 23, 23, "min((p1^p2)*Ep,S1/dt).*smoothThreshold_storage_logistic(S2,S2min)", "description says ET activates when another store is below threshold", "medium", (0.0, 0.4), 0.1, lambda x: ref.evap_14_hard(x, 0.1), bound_max=5.0),
        FormulaCandidate("F010", "penman/tcm", "models/flux/evap.py", 234, 250, "evap_16", "evapotranspiration_limiter", "min(p1*Ep*(1 - storage_gate_above(S2,S2min)),S1)", "MARRMoT/Flux files/evap_16.m", 21, 21, "min((p1.*Ep).*smoothThreshold_storage_logistic(S2,S2min),S1/dt)", "description says scaled ET if store is below threshold", "medium", (0.0, 0.4), 0.1, lambda x: ref.evap_16_hard(x, 0.1), bound_max=5.6),
        FormulaCandidate("F011", "gsfb", "models/flux/interflow.py", 125, 133, "interflow_11", "runoff_threshold", "min(p1, relu(S-p2)) * storage_gate_above(S,p2)", "MARRMoT/Flux files/interflow_11.m", 226, 231, "min(p1,(S-p2)/dt).*(1-smoothThreshold_storage_logistic(S,p2))", "soft threshold may damp onset of constant interflow", "medium", (0.0, 100.0), 50.0, lambda x: ref.interflow_11_hard(x, 50.0), bound_max=5.0),
        FormulaCandidate("F012", "modhydrolog", "models/flux/interflow.py", 136, 154, "interflow_12", "runoff_threshold", "min(p1*(relu(S-FC)+eps)^p3,S) * storage_gate_above(S,FC)", "MARRMoT/Flux files/interflow_12.m", 259, 259, "(S>FC).*min(p1*max(S-FC,0)^p3,max(S/dt,0))", "adds smoothing and epsilon to nonlinear threshold flow", "medium", (0.0, 100.0), 40.0, lambda x: ref.interflow_12_hard(x), bound_max=100.0),
        FormulaCandidate("F013", "shared_flux_unused", "models/flux/baseflow.py", 65, 75, "baseflow_6", "runoff_threshold", "min(S,p1*S^2) * storage_gate_above(S,p2)", "MARRMoT/Flux files/baseflow_6.m", 18, 18, "min(S/dt,p1*S^2).*(1-smoothThreshold_storage_logistic(S,p2))", "inactive shared helper now uses dMoT above-threshold gate matching MARRMoT logic", "medium", (0.0, 100.0), 10.0, lambda x: ref.baseflow_6_hard(x, 10.0), bound_max=100.0),
        FormulaCandidate("F014", "gsfb", "models/flux/baseflow.py", 105, 112, "baseflow_9", "runoff_threshold", "p1 * softplus(S-p2,beta=50)", "MARRMoT/Flux files/baseflow_9.m", 202, 202, "p1 .* max(0,S-p2)", "softplus creates small leakage below threshold", "medium", (0.0, 100.0), 50.0, lambda x: ref.baseflow_9_hard(x, 50.0), bound_max=10.0),
        FormulaCandidate("F015", "mopex5", "models/flux/phenology.py", 5, 24, "phenology_1", "hard_threshold", "clamp((T-p1)/(p2-p1+eps),0,1) * Ep", "MARRMoT/Flux files/phenology_1.m", 306, 306, "min(1,max(0,(T-p1)/(p2-p1)))*Ep", "epsilon in denominator slightly shifts phenology ramp", "low", (-10.0, 10.0), -5.0, lambda x: ref.phenology_1_hard(x), bound_max=8.0),
        FormulaCandidate("F016", "mopex4/mopex5", "models/core/mopex4.py", 74, 112, "interception_4", "smoothed_threshold", "softplus(cosine_fraction*50)/50 * rainfall", "MARRMoT/Flux files/interception_4.m", 283, 283, "max(0,p1+(1-p1)*cos(...))*In", "softplus replaces hard seasonal max(0,.)", "medium", (1.0, 365.0), 183.0, lambda x: ref.interception_4_hard(x), bound_max=10.0, mapping_note="core has copied expression with softplus in mopex4/mopex5"),
        FormulaCandidate("F017", "tcm", "models/core/tcm.py", 12, 22, "baseflow_6", "power_law_near_zero", "min(S,k2*S^2) * storage_gate_above(S,p2)", "MARRMoT/Flux files/baseflow_6.m", 18, 18, "min(S/dt,k2*S^2).*(1-smoothThreshold_storage_logistic(S,p2))", "TCM baseflow restored to MARRMoT dt=1 scale and limiter", "medium", (0.0, 100.0), 0.0, lambda x: ref.baseflow_6_hard(x, 0.0, p1=0.01), bound_max=100.0),
    ]


def _inventory_rows(candidates: list[FormulaCandidate]) -> list[dict[str, object]]:
    return [
        {
            "candidate_id": c.candidate_id,
            "dmot_file": c.dmot_file,
            "dmot_line_start": c.dmot_line_start,
            "dmot_line_end": c.dmot_line_end,
            "model_name": c.model_name,
            "function_or_class": c.function_or_class,
            "formula_type": c.formula_type,
            "dmot_expression_summary": c.dmot_expression_summary,
            "suspected_matlab_equivalent": c.matlab_expression_summary,
            "risk_level_initial": c.risk_level_initial,
            "reason_for_inclusion": c.reason_for_inclusion,
        }
        for c in candidates
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _risk(candidate: FormulaCandidate, value_row: dict[str, object], grad_row: dict[str, object]) -> tuple[str, str, bool]:
    issues: list[str] = []
    if candidate.forced_issue:
        issues.append(candidate.forced_issue)
    bound_problem = float(value_row["bound_violations"]) > 0
    rel_l2 = float(value_row["relative_l2_diff"])
    near_diff = float(value_row["max_diff_near_threshold"])
    grad_error = float(grad_row["autograd_fd_max_error"])
    max_grad = float(grad_row["max_abs_gradient"])
    nonfinite_grad = int(grad_row["nan_gradient_count"]) or int(grad_row["inf_gradient_count"])
    ambiguous_mapping = "not found" in candidate.matlab_file or "not found" in candidate.mapping_note

    if bound_problem:
        issues.append("physical bound violation")
    if rel_l2 > 0.75:
        issues.append("large value difference from hard reference")
    elif rel_l2 > 0.05 or near_diff > 0.1:
        issues.append("near-threshold smoothing bias")
    if grad_error > 1.0e-3 and (rel_l2 > 0.01 or max_grad > 10.0):
        issues.append("autograd finite-difference mismatch near kink or steep transition")
    if max_grad > 100.0:
        issues.append("large gradient magnitude")
    if nonfinite_grad:
        issues.append("non-finite gradient")
    if ambiguous_mapping:
        issues.append("ambiguous MATLAB helper mapping")

    if candidate.forced_issue or bound_problem or nonfinite_grad or rel_l2 > 0.75 or max_grad > 100.0:
        return "high", "; ".join(dict.fromkeys(issues)) or "high initial risk", True
    if issues:
        return "medium", "; ".join(dict.fromkeys(issues)) or "documented smoothing trade-off", True
    return "low", "numerically similar and gradients stable in tested domain", False


def _write_matlab_extracts(candidates: list[FormulaCandidate]) -> None:
    lines = ["# MATLAB Formula Extracts", ""]
    for c in candidates:
        dmot_path = PROJECT_ROOT / c.dmot_file
        matlab_path = MARRMOT_ROOT / c.matlab_file.replace("MARRMoT/", "") if c.matlab_file.startswith("MARRMoT/") else MARRMOT_ROOT / c.matlab_file if c.matlab_file != "not found" else Path()
        lines.extend(
            [
                f"## {c.candidate_id}: {c.function_or_class}",
                f"- dMoT: `{c.dmot_file}:{c.dmot_line_start}`",
                f"- MATLAB: `{c.matlab_file}:{c.matlab_line_start or 'not found'}`",
                f"- Mapping: {c.mapping_note}",
                "",
                "dMoT snippet:",
                "```python",
                _snip(dmot_path, c.dmot_line_start, c.dmot_line_end),
                "```",
                "",
                "MATLAB snippet:",
                "```matlab",
                _snip(matlab_path, c.matlab_line_start, c.matlab_line_end),
                "```",
                "",
                f"Notes: {c.reason_for_inclusion}",
                "",
            ]
        )
    (OUTPUT_DIR / "matlab_formula_extracts.md").write_text("\n".join(lines), encoding="utf-8")


def _make_plots(candidates: list[FormulaCandidate], ranking_rows: list[dict[str, object]]) -> None:
    risky = {row["candidate_id"] for row in ranking_rows if row["risk_level_final"] in {"medium", "high"}}
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for old_plot in PLOTS_DIR.glob("*.png"):
        old_plot.unlink()
    if not risky:
        return
    import matplotlib.pyplot as plt

    for c in candidates:
        if c.candidate_id not in risky:
            continue
        x = _np_grid(*c.domain, c.threshold)
        dmot = _torch_eval(c.candidate_id, x).detach().cpu().numpy()
        reference = c.reference_fn(x)
        grad = _finite_difference_torch(c.candidate_id, x)
        order = np.argsort(x)
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axes[0].plot(x[order], reference[order], label="MATLAB-style reference", linewidth=1.5)
        axes[0].plot(x[order], dmot[order], label="dMoT", linewidth=1.2)
        axes[0].legend()
        axes[0].set_ylabel("value")
        axes[1].plot(x[order], np.abs(dmot - reference)[order], color="#9c3d2f")
        axes[1].set_ylabel("abs diff")
        axes[2].plot(x[order], grad[order], color="#315f72")
        axes[2].set_ylabel("dMoT FD grad")
        axes[2].set_xlabel("diagnostic variable")
        fig.suptitle(f"{c.candidate_id} {c.function_or_class}")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{c.candidate_id}_{c.function_or_class}.png", dpi=150)
        plt.close(fig)


def _write_report(candidates: list[FormulaCandidate], value_rows: list[dict[str, object]], grad_rows: list[dict[str, object]], ranking_rows: list[dict[str, object]]) -> None:
    by_id = {c.candidate_id: c for c in candidates}
    values = {row["candidate_id"]: row for row in value_rows}
    grads = {row["candidate_id"]: row for row in grad_rows}
    counts = {level: sum(1 for row in ranking_rows if row["risk_level_final"] == level) for level in ("low", "medium", "high")}
    matlab_found = sum(1 for c in candidates if c.matlab_file != "not found")
    risky = [row for row in ranking_rows if row["risk_level_final"] in {"medium", "high"}]

    lines = [
        "# Formula Smoothing Diagnostic Report",
        "",
        "## Scope",
        "This diagnostic inspects smoothing functions, hard-threshold replacements, clipping logic, and differentiable approximations used by dMoT core formulas. Unit hydrograph routing and water-balance closure are outside this report.",
        "",
        "## Files searched",
        "- `models/core`",
        "- `models/flux`",
        "- `models/special`",
        "- `models/hydrology_model.py`",
        "- `/home/jingxin/code/dmg-research/MARRMoT/Flux files`",
        "- `/home/jingxin/code/dmg-research/MARRMoT/Model files`",
        "",
        "## Summary",
        f"- Formula candidates inventoried: {len(candidates)}",
        f"- MATLAB counterparts found: {matlab_found}",
        f"- Candidates tested numerically: {len(value_rows)}",
        f"- Candidates with gradient diagnostics: {len(grad_rows)}",
        f"- Low risk: {counts['low']}",
        f"- Medium risk: {counts['medium']}",
        f"- High risk: {counts['high']}",
        "",
        "## Medium And High Risk Formulas",
        "| rank | candidate | model | formula type | dMoT location | MATLAB location | risk | suspected issue | max abs diff | max abs gradient | human review |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in risky:
        c = by_id[row["candidate_id"]]
        lines.append(
            f"| {row['rank']} | {c.candidate_id} | {c.model_name} | {c.formula_type} | "
            f"`{c.dmot_file}:{c.dmot_line_start}` | `{c.matlab_file}:{c.matlab_line_start or 'n/a'}` | "
            f"{row['risk_level_final']} | {row['suspected_issue']} | {float(row['max_abs_diff']):.3e} | "
            f"{float(row['max_abs_gradient']):.3e} | {row['needs_human_review']} |"
        )

    for row in risky:
        c = by_id[row["candidate_id"]]
        v = values[c.candidate_id]
        g = grads[c.candidate_id]
        lines.extend(
            [
                "",
                f"## {c.candidate_id}: {c.function_or_class}",
                f"- dMoT: `{c.dmot_file}:{c.dmot_line_start}`",
                f"- MATLAB: `{c.matlab_file}:{c.matlab_line_start or 'not found'}`",
                f"- Mathematical interpretation: {c.dmot_expression_summary} compared with {c.matlab_expression_summary}.",
                f"- Value summary: max_abs_diff={float(v['max_abs_diff']):.3e}, relative_L2={float(v['relative_l2_diff']):.3e}, near_threshold_bias={float(v['signed_bias_near_threshold']):.3e}.",
                f"- Gradient summary: autograd_fd_max_error={float(g['autograd_fd_max_error']):.3e}, max_abs_gradient={float(g['max_abs_gradient']):.3e}, saturation_ratio={float(g['gradient_saturation_ratio']):.3e}.",
                f"- Suspicion: {row['suspected_issue']}.",
            ]
        )

    lines.extend(
        [
            "",
            "## Low-Risk Notes",
            "Low-risk formulas had finite outputs, finite gradients, and no large broad-domain value distortion in the tested ranges. Full details remain in the CSV files.",
            "",
            "## Human-Review Priorities",
            "1. Check formulas flagged as possible threshold-orientation inversions against the intended MARRMoT smoothThreshold convention.",
            "2. Review the TCM-specific `baseflow_6` scaling against calibrated parameter expectations and MARRMoT units.",
            "3. Decide whether small threshold leakage from sigmoid/softplus formulas is acceptable for calibration or should be replaced with sharper or bounded smooth approximations.",
        ]
    )
    (OUTPUT_DIR / "formula_smoothing_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates()
    value_rows: list[dict[str, object]] = []
    grad_rows: list[dict[str, object]] = []

    for candidate in candidates:
        x = _np_grid(*candidate.domain, candidate.threshold)
        dmot = _torch_eval(candidate.candidate_id, x).detach().cpu().numpy()
        reference = candidate.reference_fn(x)
        value_row = {
            "candidate_id": candidate.candidate_id,
            "model_name": candidate.model_name,
            "dmot_file": candidate.dmot_file,
            "matlab_file": candidate.matlab_file,
            "formula_type": candidate.formula_type,
            **_metrics(dmot, reference, x, candidate.threshold, candidate),
        }
        grad_row = {
            "candidate_id": candidate.candidate_id,
            "model_name": candidate.model_name,
            "dmot_file": candidate.dmot_file,
            "formula_type": candidate.formula_type,
            **_gradient_metrics(candidate, x),
        }
        value_rows.append(value_row)
        grad_rows.append(grad_row)

    ranking_rows: list[dict[str, object]] = []
    for candidate, value_row, grad_row in zip(candidates, value_rows, grad_rows, strict=True):
        level, issue, review = _risk(candidate, value_row, grad_row)
        ranking_rows.append(
            {
                "rank": 0,
                "candidate_id": candidate.candidate_id,
                "model_name": candidate.model_name,
                "dmot_file": candidate.dmot_file,
                "matlab_file": candidate.matlab_file,
                "formula_type": candidate.formula_type,
                "dmot_expression_summary": candidate.dmot_expression_summary,
                "matlab_expression_summary": candidate.matlab_expression_summary,
                "max_abs_diff": value_row["max_abs_diff"],
                "relative_L2_diff": value_row["relative_l2_diff"],
                "near_threshold_bias": value_row["signed_bias_near_threshold"],
                "autograd_fd_max_error": grad_row["autograd_fd_max_error"],
                "max_abs_gradient": grad_row["max_abs_gradient"],
                "gradient_saturation_ratio": grad_row["gradient_saturation_ratio"],
                "nan_gradient_count": grad_row["nan_gradient_count"],
                "inf_gradient_count": grad_row["inf_gradient_count"],
                "physical_bound_violation": value_row["bound_violations"],
                "risk_level_final": level,
                "suspected_issue": issue,
                "needs_human_review": review,
            }
        )
    risk_order = {"high": 0, "medium": 1, "low": 2}
    ranking_rows.sort(key=lambda row: (risk_order[str(row["risk_level_final"])], -float(row["max_abs_diff"])))
    for index, row in enumerate(ranking_rows, start=1):
        row["rank"] = index

    _write_csv(OUTPUT_DIR / "formula_inventory.csv", _inventory_rows(candidates))
    _write_csv(OUTPUT_DIR / "formula_value_comparison.csv", value_rows)
    _write_csv(OUTPUT_DIR / "formula_gradient_diagnostics.csv", grad_rows)
    _write_csv(OUTPUT_DIR / "suspicious_formula_ranking.csv", ranking_rows)
    _write_matlab_extracts(candidates)
    _write_report(candidates, value_rows, grad_rows, ranking_rows)
    _make_plots(candidates, ranking_rows)

    summary = {
        "candidates": len(candidates),
        "matlab_counterparts": sum(1 for c in candidates if c.matlab_file != "not found"),
        "value_comparisons": len(value_rows),
        "gradient_diagnostics": len(grad_rows),
        "risk_counts": {level: sum(1 for row in ranking_rows if row["risk_level_final"] == level) for level in ("low", "medium", "high")},
    }
    (OUTPUT_DIR / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote formula smoothing diagnostics to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
