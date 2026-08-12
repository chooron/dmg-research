from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnose_formula_smoothing import (  # noqa: E402
    FormulaCandidate,
    _finite_difference_ref,
    _finite_difference_torch,
    _gradient_metrics,
    _torch_eval,
    build_candidates,
)


OUTPUT_DIR = PROJECT_ROOT / "validation_results" / "formula_smoothing" / "focused_formula_review"
PLOTS_DIR = OUTPUT_DIR / "plots"
MARRMOT_ROOT = Path("/home/jingxin/code/dmg-research/MARRMoT")

FOCUSED_IDS = ("F017", "F010", "F009", "F013", "F007", "F005", "F011")

USAGE: dict[str, dict[str, str]] = {
    "F017": {
        "active_usage_status": "active_registered_core",
        "called_by_models": "tcm: models/core/tcm.py::tcm_step lines 202-204",
        "registered_model": "tcm",
        "used_in_validation": "yes",
        "notes": "Inline TCM-specific baseflow_6 shadows the shared flux helper.",
    },
    "F010": {
        "active_usage_status": "active_registered_core",
        "called_by_models": "penman: models/core/penman.py::penman_step lines 209-216; tcm: models/core/tcm.py::tcm_step lines 164-171",
        "registered_model": "penman;tcm",
        "used_in_validation": "yes",
        "notes": "Shared evap_16 is used by two registered core models.",
    },
    "F009": {
        "active_usage_status": "active_registered_core_and_special",
        "called_by_models": "smar: models/core/smar.py::smar_step lines 131-180; special/smar.py lines 158-195",
        "registered_model": "smar",
        "used_in_validation": "yes",
        "notes": "Shared evap_14 is active in registered core SMAR and duplicated in special SMAR.",
    },
    "F013": {
        "active_usage_status": "unused_shared_helper",
        "called_by_models": "none found outside diagnostics/tests",
        "registered_model": "no",
        "used_in_validation": "yes",
        "notes": "models/flux/baseflow.py::baseflow_6 is not imported by registered core/special models; TCM uses its own local baseflow_6.",
    },
    "F007": {
        "active_usage_status": "active_registered_core",
        "called_by_models": "tcm: models/core/tcm.py::tcm_step line 178; penman imports saturation_9 but active call is commented/replaced",
        "registered_model": "tcm",
        "used_in_validation": "yes",
        "notes": "Deficit-store pass-through used in TCM; Penman currently uses a custom relu expression instead.",
    },
    "F005": {
        "active_usage_status": "inactive_in_registered_dmot_models",
        "called_by_models": "none found in models/core or models/special",
        "registered_model": "no",
        "used_in_validation": "yes",
        "notes": "MARRMoT uses melt_3 in m_43_gsmsocont, but that model is not present in the registered dMoT core set.",
    },
    "F011": {
        "active_usage_status": "active_registered_core",
        "called_by_models": "gsfb: models/core/gsfb.py::gsfb_step line 120",
        "registered_model": "gsfb",
        "used_in_validation": "yes",
        "notes": "Constant interflow threshold in registered GSFB core model.",
    },
}

PREVIOUS_RISK = {
    "F017": "high",
    "F010": "high",
    "F009": "high",
    "F013": "high",
    "F007": "high",
    "F005": "high",
    "F011": "medium",
}

CLASSIFICATION: dict[str, dict[str, str]] = {
    "F017": {
        "likely_issue_type": "fixed_scale_and_limiter_restored",
        "value_difference_severity": "low_after_fix",
        "gradient_risk_severity": "low",
        "physical_bound_risk": "low_after_tcm_external_clamp",
        "implementation_mismatch_likelihood": "low_after_fix",
        "recommended_action": "keep_as_smoothing",
        "human_review_priority": "low",
        "short_reason": "TCM baseflow now removes the empirical /1000 scale and restores min(S,p1*S^2) with dMoT's above-threshold gate.",
    },
    "F010": {
        "likely_issue_type": "fixed_threshold_orientation",
        "value_difference_severity": "medium_near_threshold_only",
        "gradient_risk_severity": "medium_sharp_threshold",
        "physical_bound_risk": "low",
        "implementation_mismatch_likelihood": "low_after_fix",
        "recommended_action": "keep_but_document",
        "human_review_priority": "medium",
        "short_reason": "evap_16 now uses 1 - dMoT's above-threshold gate, matching the inferred MARRMoT below-threshold activation with smoothing near the threshold.",
    },
    "F009": {
        "likely_issue_type": "fixed_threshold_orientation",
        "value_difference_severity": "medium_near_threshold_only",
        "gradient_risk_severity": "medium_sharp_threshold",
        "physical_bound_risk": "low",
        "implementation_mismatch_likelihood": "low_after_fix",
        "recommended_action": "keep_but_document",
        "human_review_priority": "medium",
        "short_reason": "evap_14 now uses 1 - dMoT's above-threshold gate, matching the inferred MARRMoT below-threshold activation with smoothing near the threshold.",
    },
    "F013": {
        "likely_issue_type": "inactive_helper_fixed",
        "value_difference_severity": "low_to_medium_near_threshold",
        "gradient_risk_severity": "low_to_medium",
        "physical_bound_risk": "low",
        "implementation_mismatch_likelihood": "low_after_fix_but_inactive",
        "recommended_action": "inactive_no_action",
        "human_review_priority": "low",
        "short_reason": "Shared baseflow_6 is still unused by registered models, but now uses dMoT's above-threshold gate consistently with MARRMoT baseflow_6.",
    },
    "F007": {
        "likely_issue_type": "acceptable_smoothing_tradeoff",
        "value_difference_severity": "medium_near_threshold_only",
        "gradient_risk_severity": "medium_sharp_threshold",
        "physical_bound_risk": "low",
        "implementation_mismatch_likelihood": "low",
        "recommended_action": "keep_but_document",
        "human_review_priority": "medium",
        "short_reason": "Using 1 - dMoT above-gate reproduces MARRMoT's below-threshold deficit-store gate; differences are concentrated at the smooth transition.",
    },
    "F005": {
        "likely_issue_type": "inactive_acceptable_smoothing",
        "value_difference_severity": "medium_near_threshold_only",
        "gradient_risk_severity": "medium_sharp_threshold",
        "physical_bound_risk": "low",
        "implementation_mismatch_likelihood": "low",
        "recommended_action": "inactive_no_action",
        "human_review_priority": "low",
        "short_reason": "Formula orientation matches the inferred MARRMoT below-threshold snowpack gate, but no registered dMoT model uses melt_3.",
    },
    "F011": {
        "likely_issue_type": "acceptable_smoothing_plus_relu_guard",
        "value_difference_severity": "medium",
        "gradient_risk_severity": "medium_kink_at_threshold",
        "physical_bound_risk": "low",
        "implementation_mismatch_likelihood": "low_to_medium",
        "recommended_action": "keep_but_document",
        "human_review_priority": "medium",
        "short_reason": "dMoT uses ReLU and an above-threshold smooth gate; this prevents negative leakage and approximates MARRMoT's above-threshold interflow.",
    },
}

DMOT_SNIP_RANGES = {
    "F017": (12, 22),
    "F010": (234, 250),
    "F009": (197, 213),
    "F013": (65, 75),
    "F007": (172, 183),
    "F005": (33, 64),
    "F011": (125, 133),
}

MATLAB_SNIP_RANGES = {
    "F017": (1, 20),
    "F010": (1, 23),
    "F009": (1, 25),
    "F013": (1, 20),
    "F007": (1, 30),
    "F005": (1, 32),
    "F011": (1, 29),
}

MATLAB_FORMULA_LINES = {
    "F017": "18",
    "F010": "21",
    "F009": "23",
    "F013": "18",
    "F007": "23-27",
    "F005": "25-29",
    "F011": "22-26",
}

FORMULA_DETAILS = {
    "F017": {
        "variables": "S: slow-routing storage S4 [mm]; p1/k2: quadratic coefficient [mm-1 d-1]; p2: threshold [mm], TCM passes 0.",
        "parameters": "dMoT TCM k2 bounds [0,1], matching MARRMoT.",
        "unit_scale": "MARRMoT allows q=min(S/dt,k2*S^2); dMoT uses dt=1 and computes min(S,k2*S^2).",
        "translation_type": "fixed MARRMoT dt=1 smoothing approximation",
    },
    "F010": {
        "variables": "S1: source storage limit, often Inf in Penman/TCM; S2: controlling store; S2min: activation threshold; Ep: PET.",
        "parameters": "p1/gam: evaporation reduction coefficient [-].",
        "unit_scale": "No unit rescale visible; threshold orientation has been corrected.",
        "translation_type": "fixed below-threshold smoothing approximation",
    },
    "F009": {
        "variables": "S1: evaporating layer storage; S2: overlying controlling store; S2min: threshold; Ep: remaining PET.",
        "parameters": "p1/c: evaporation coefficient; p2: layer exponent.",
        "unit_scale": "No unit rescale visible; threshold orientation has been corrected.",
        "translation_type": "fixed below-threshold smoothing approximation",
    },
    "F013": {
        "variables": "S: storage [mm]; p1: quadratic coefficient; p2: threshold.",
        "parameters": "Shared helper parameters only; no active registered caller found.",
        "unit_scale": "No rescale; orientation has been corrected to dMoT's above-threshold gate convention.",
        "translation_type": "inactive helper fixed to MARRMoT dt=1 smoothing approximation",
    },
    "F007": {
        "variables": "incoming_flux: water to pass/overflow; S: deficit-store magnitude; St: near-zero threshold.",
        "parameters": "St often 0.01 in TCM.",
        "unit_scale": "Small threshold makes logistic derivative large because scale is clamped at 50.",
        "translation_type": "smoothed approximation, orientation consistent with MARRMoT",
    },
    "F005": {
        "variables": "S1: glacier ice storage; S2: snowpack storage; St: snowpack threshold; T: temperature.",
        "parameters": "p1: degree-day factor; p2: melt temperature threshold.",
        "unit_scale": "Small St causes sharp storage-gate gradient; output remains bounded by melt_actual.",
        "translation_type": "smoothed approximation, inactive in registered dMoT models",
    },
    "F011": {
        "variables": "S: source storage; p1: maximum interflow; p2: threshold.",
        "parameters": "GSFB passes frate and ndc*smax.",
        "unit_scale": "No unit rescale; ReLU explicitly enforces nonnegative excess.",
        "translation_type": "smoothed approximation with hard ReLU guard",
    },
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _snip(path: Path, start: int, end: int) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[start - 1 : end])


def _matlab_path(candidate: FormulaCandidate) -> Path:
    return MARRMOT_ROOT / candidate.matlab_file.replace("MARRMoT/", "")


def _focused_grid(candidate: FormulaCandidate) -> np.ndarray:
    lo, hi = candidate.domain
    threshold = candidate.threshold
    span = hi - lo
    base = np.linspace(lo, hi, 2401)
    near_width = max(0.005 * max(abs(threshold), 1.0), 0.02 * span)
    near = np.linspace(threshold - near_width, threshold + near_width, 2401)
    tiny = threshold + np.array(
        [-1e-3, -1e-4, -1e-6, -1e-8, 0.0, 1e-8, 1e-6, 1e-4, 1e-3],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260623 + int(candidate.candidate_id[1:]))
    random = rng.uniform(lo, hi, 400)
    x = np.concatenate([base, near, tiny, [lo, threshold, hi, 0.0], random])
    x = x[np.isfinite(x)]
    return np.unique(np.clip(x, lo, hi).astype(np.float64))


def _physical_violation(candidate: FormulaCandidate, x: np.ndarray, y: np.ndarray) -> tuple[float, bool, bool]:
    negative_amount = float(np.max(np.maximum(-y, 0.0)))
    available_amount = np.inf
    if candidate.candidate_id in {"F017", "F013", "F011"}:
        available_amount = x
    elif candidate.candidate_id == "F005":
        available_amount = 20.0
    elif candidate.candidate_id == "F009":
        available_amount = 5.0
    elif candidate.candidate_id == "F010":
        available_amount = np.inf
    exceeds = np.maximum(y - available_amount, 0.0) if np.ndim(available_amount) else np.maximum(y - available_amount, 0.0)
    exceed_amount = float(np.nanmax(exceeds)) if np.size(exceeds) else 0.0
    max_violation = max(negative_amount, exceed_amount)
    return max_violation, bool(exceed_amount > 1e-10), bool(negative_amount > 1e-10)


def _opposite_monotonic_direction(candidate: FormulaCandidate, x: np.ndarray, y: np.ndarray, ref: np.ndarray) -> bool:
    width = max(0.01 * max(abs(candidate.threshold), 1.0), 0.01 * (candidate.domain[1] - candidate.domain[0]))
    below = x < candidate.threshold - width
    above = x > candidate.threshold + width
    if not np.any(below) or not np.any(above):
        return False
    dmot_delta = float(np.mean(y[above]) - np.mean(y[below]))
    ref_delta = float(np.mean(ref[above]) - np.mean(ref[below]))
    if abs(dmot_delta) < 1e-9 or abs(ref_delta) < 1e-9:
        return False
    return dmot_delta * ref_delta < 0.0


def _value_row(candidate: FormulaCandidate) -> dict[str, Any]:
    x = _focused_grid(candidate)
    y = _torch_eval(candidate.candidate_id, x).detach().cpu().numpy()
    ref = candidate.reference_fn(x)
    diff = y - ref
    abs_diff = np.abs(diff)
    near = np.abs(x - candidate.threshold) <= max(0.02 * (candidate.domain[1] - candidate.domain[0]), 0.005)
    if not np.any(near):
        near = np.ones_like(x, dtype=bool)
    max_phys, exceeds_available, negative = _physical_violation(candidate, x, y)
    return {
        "candidate_id": candidate.candidate_id,
        "dmot_function": candidate.function_or_class,
        "matlab_function": Path(candidate.matlab_file).stem,
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "relative_L2_diff": float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-12)),
        "signed_bias_mean": float(np.mean(diff)),
        "near_threshold_max_abs_diff": float(np.max(abs_diff[near])),
        "near_threshold_signed_bias": float(np.mean(diff[near])),
        "max_physical_bound_violation": max_phys,
        "dmot_output_exceeds_available_water": exceeds_available,
        "dmot_output_negative_when_not_expected": negative,
        "opposite_monotonic_direction_from_matlab": _opposite_monotonic_direction(candidate, x, y, ref),
    }


def _gradient_row(candidate: FormulaCandidate) -> dict[str, Any]:
    x = _focused_grid(candidate)
    torch_x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    y = _torch_eval(candidate.candidate_id, x, requires_grad=False)
    del y
    y_grad = _torch_eval(candidate.candidate_id, x, requires_grad=True)
    del y_grad

    y2 = __import__("tests.dmot_formula_wrappers", fromlist=["WRAPPERS"]).WRAPPERS[candidate.candidate_id](torch_x)
    y2.sum().backward()
    autograd = torch_x.grad.detach().cpu().numpy()
    fd = _finite_difference_torch(candidate.candidate_id, x)
    hard_fd = _finite_difference_ref(candidate.reference_fn, x)
    finite = np.isfinite(autograd) & np.isfinite(fd)
    err = np.abs(autograd[finite] - fd[finite]) if np.any(finite) else np.array([np.inf])
    near = np.abs(x - candidate.threshold) <= max(0.02 * (candidate.domain[1] - candidate.domain[0]), 0.005)
    if not np.any(near):
        near = np.ones_like(x, dtype=bool)
    max_abs_grad = float(np.nanmax(np.abs(autograd)))
    spike_threshold = max(100.0, 20.0 * float(np.nanmedian(np.abs(autograd) + 1e-12)))
    return {
        "candidate_id": candidate.candidate_id,
        "dmot_function": candidate.function_or_class,
        "max_autograd_vs_fd_error": float(np.max(err)),
        "mean_autograd_vs_fd_error": float(np.mean(err)),
        "max_abs_gradient": max_abs_grad,
        "max_abs_gradient_near_threshold": float(np.nanmax(np.abs(autograd[near]))),
        "mean_gradient_near_threshold": float(np.nanmean(autograd[near])),
        "gradient_sign_mismatch_count": int(np.sum(np.sign(autograd[finite]) != np.sign(fd[finite]))),
        "gradient_spike_count": int(np.sum(np.abs(autograd) > spike_threshold)),
        "vanishing_gradient_region_width": float(np.ptp(x[np.abs(autograd) < 1e-8])) if np.any(np.abs(autograd) < 1e-8) else 0.0,
        "exploding_gradient_region_width": float(np.ptp(x[np.abs(autograd) > 100.0])) if np.any(np.abs(autograd) > 100.0) else 0.0,
        "nan_gradient_count": int(np.isnan(autograd).sum()),
        "inf_gradient_count": int(np.isinf(autograd).sum()),
        "hard_reference_fd_max_abs_gradient": float(np.nanmax(np.abs(hard_fd))),
    }


def _active_usage_rows(candidates: list[FormulaCandidate]) -> list[dict[str, Any]]:
    rows = []
    for c in candidates:
        usage = USAGE[c.candidate_id]
        rows.append(
            {
                "candidate_id": c.candidate_id,
                "risk_level_previous": PREVIOUS_RISK[c.candidate_id],
                "dmot_file": c.dmot_file,
                "dmot_lines": f"{DMOT_SNIP_RANGES[c.candidate_id][0]}-{DMOT_SNIP_RANGES[c.candidate_id][1]}",
                "dmot_function": c.function_or_class,
                "matlab_file": c.matlab_file,
                "matlab_lines": MATLAB_FORMULA_LINES[c.candidate_id],
                "matlab_function": Path(c.matlab_file).stem,
                **usage,
            }
        )
    return rows


def _ranking_rows(candidates: list[FormulaCandidate], value_rows: list[dict[str, Any]], grad_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = {r["candidate_id"]: r for r in value_rows}
    grads = {r["candidate_id"]: r for r in grad_rows}
    rows = []
    priority = {"high": 0, "medium": 1, "low": 2}
    for c in candidates:
        cls = CLASSIFICATION[c.candidate_id]
        rows.append(
            {
                "candidate_id": c.candidate_id,
                "model_or_function": c.model_name,
                "active_usage_status": USAGE[c.candidate_id]["active_usage_status"],
                "dmot_file": f"{c.dmot_file}:{c.dmot_line_start}",
                "matlab_file": f"{c.matlab_file}:{MATLAB_FORMULA_LINES[c.candidate_id]}",
                "formula_type": c.formula_type,
                **cls,
                "max_abs_diff": values[c.candidate_id]["max_abs_diff"],
                "max_abs_gradient": grads[c.candidate_id]["max_abs_gradient"],
            }
        )
    rows.sort(key=lambda r: (priority[r["human_review_priority"]], -float(r["max_abs_diff"])))
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def _write_formula_cards(candidates: list[FormulaCandidate], value_rows: list[dict[str, Any]], grad_rows: list[dict[str, Any]]) -> None:
    values = {r["candidate_id"]: r for r in value_rows}
    grads = {r["candidate_id"]: r for r in grad_rows}
    lines = ["# Focused Formula Cards", ""]
    for c in candidates:
        details = FORMULA_DETAILS[c.candidate_id]
        dmot_start, dmot_end = DMOT_SNIP_RANGES[c.candidate_id]
        matlab_start, matlab_end = MATLAB_SNIP_RANGES[c.candidate_id]
        lines.extend(
            [
                f"## {c.candidate_id}: {c.model_name} / {c.function_or_class}",
                f"- Active usage: {USAGE[c.candidate_id]['active_usage_status']}",
                f"- dMoT formula type: {details['translation_type']}",
                f"- dMoT mathematical form: {c.dmot_expression_summary}",
                f"- MATLAB mathematical form: {c.matlab_expression_summary}",
                f"- Variables: {details['variables']}",
                f"- Parameters: {details['parameters']}",
                f"- Unit/scale assumptions: {details['unit_scale']}",
                f"- Value result: max_abs_diff={values[c.candidate_id]['max_abs_diff']:.6g}, near_threshold_max_abs_diff={values[c.candidate_id]['near_threshold_max_abs_diff']:.6g}, opposite_monotonic={values[c.candidate_id]['opposite_monotonic_direction_from_matlab']}",
                f"- Gradient result: max_abs_gradient={grads[c.candidate_id]['max_abs_gradient']:.6g}, autograd_fd_max_error={grads[c.candidate_id]['max_autograd_vs_fd_error']:.6g}",
                "",
                f"dMoT snippet `{c.dmot_file}:{dmot_start}-{dmot_end}`:",
                "```python",
                _snip(PROJECT_ROOT / c.dmot_file, dmot_start, dmot_end),
                "```",
                "",
                f"MATLAB snippet `{c.matlab_file}:{matlab_start}-{matlab_end}`:",
                "```matlab",
                _snip(_matlab_path(c), matlab_start, matlab_end),
                "```",
                "",
            ]
        )
    (OUTPUT_DIR / "formula_cards.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_plots(candidates: list[FormulaCandidate]) -> None:
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for old in PLOTS_DIR.glob("*.png"):
        old.unlink()

    names = {
        "F017": "tcm_baseflow",
        "F010": "evap_16",
        "F009": "evap_14",
        "F013": "shared_baseflow_6",
        "F007": "saturation_9",
        "F005": "melt_3",
        "F011": "interflow_11",
    }
    for c in candidates:
        x = _focused_grid(c)
        y = _torch_eval(c.candidate_id, x).detach().cpu().numpy()
        ref = c.reference_fn(x)
        order = np.argsort(x)
        near_width = max(0.05 * (c.domain[1] - c.domain[0]), 0.02)
        near = np.abs(x - c.threshold) <= near_width

        fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.5))
        axes[0].plot(x[order], ref[order], label="MATLAB-style hard/reference", linewidth=1.7)
        axes[0].plot(x[order], y[order], label="dMoT", linewidth=1.3)
        axes[0].set_ylabel("value")
        axes[0].legend()
        axes[1].plot(x[order], np.abs(y - ref)[order], color="#9b3a2f")
        axes[1].set_ylabel("abs diff")
        if np.any(near):
            norder = np.argsort(x[near])
            axes[2].plot(x[near][norder], ref[near][norder], label="MATLAB-style", linewidth=1.7)
            axes[2].plot(x[near][norder], y[near][norder], label="dMoT", linewidth=1.3)
        axes[2].axvline(c.threshold, color="0.35", linestyle="--", linewidth=1)
        axes[2].set_ylabel("near threshold")
        axes[2].set_xlabel("diagnostic variable")
        fig.suptitle(f"{c.candidate_id} {c.function_or_class}: value comparison")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{c.candidate_id}_{names[c.candidate_id]}_value.png", dpi=160)
        plt.close(fig)

        autograd_x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        wrappers = __import__("tests.dmot_formula_wrappers", fromlist=["WRAPPERS"]).WRAPPERS
        out = wrappers[c.candidate_id](autograd_x)
        out.sum().backward()
        autograd = autograd_x.grad.detach().cpu().numpy()
        fd = _finite_difference_torch(c.candidate_id, x)
        hard_fd = _finite_difference_ref(c.reference_fn, x)

        fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)
        axes[0].plot(x[order], autograd[order], label="dMoT autograd", linewidth=1.4)
        axes[0].plot(x[order], fd[order], label="dMoT central FD", linestyle="--", linewidth=1.2)
        axes[0].legend()
        axes[0].set_ylabel("dMoT gradient")
        axes[1].plot(x[order], hard_fd[order], label="MATLAB-style hard FD", color="#586f35", linewidth=1.2)
        axes[1].axvline(c.threshold, color="0.35", linestyle="--", linewidth=1)
        axes[1].set_ylabel("hard/reference FD")
        axes[1].set_xlabel("diagnostic variable")
        fig.suptitle(f"{c.candidate_id} {c.function_or_class}: gradient comparison")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{c.candidate_id}_{names[c.candidate_id]}_gradient.png", dpi=160)
        plt.close(fig)


def _write_report(candidates: list[FormulaCandidate], value_rows: list[dict[str, Any]], grad_rows: list[dict[str, Any]], ranking_rows: list[dict[str, Any]]) -> None:
    values = {r["candidate_id"]: r for r in value_rows}
    grads = {r["candidate_id"]: r for r in grad_rows}
    lines = [
        "# Focused Formula Smoothing Review",
        "",
        "## Scope",
        "This review investigates seven formulas flagged by the previous smoothing diagnostics. The current report is regenerated after applying the accepted fixes for F017, F010, F009, and inactive helper F013.",
        "",
        "## Files Inspected",
        "- `models/core/tcm.py`, `models/core/penman.py`, `models/core/smar.py`, `models/core/gsfb.py`",
        "- `models/flux/baseflow.py`, `models/flux/evap.py`, `models/flux/saturation.py`, `models/flux/melt.py`, `models/flux/interflow.py`, `models/flux/smooth.py`",
        "- `models/special/smar.py`",
        "- `models/core/__init__.py`, `models/hydrology_model.py`",
        "- `MARRMoT/Flux files/baseflow_6.m`, `evap_16.m`, `evap_14.m`, `saturation_9.m`, `melt_3.m`, `interflow_11.m`, `saturation_1.m`",
        "- `MARRMoT/Model files/m_17_penman_4p_3s.m`, `m_25_tcm_6p_4s.m`, `m_40_smar_8p_6s.m`, `m_20_gsfb_8p_3s.m`, `m_43_gsmsocont_12p_6s.m`",
        "",
        "## Candidate List",
        "| candidate | model/function | previous risk | active usage | recommended action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in candidates:
        lines.append(
            f"| {c.candidate_id} | {c.model_name}/{c.function_or_class} | {PREVIOUS_RISK[c.candidate_id]} | "
            f"{USAGE[c.candidate_id]['active_usage_status']} | {CLASSIFICATION[c.candidate_id]['recommended_action']} |"
        )

    lines.extend(
        [
            "",
            "## Active Usage Summary",
            "The active registered model set in this checkout contains 36 core models via `models/core/__init__.py`. F009, F010, F011, and F017 are active in registered core models. F007 is active in TCM. F013 and F005 are not called by registered dMoT models found in `models/core` or `models/special`.",
            "",
            "## Interpretation Of MARRMoT Storage Gate",
            "The MATLAB helper definition `smoothThreshold_storage_logistic` was not present in this checkout, but its orientation is inferable from MARRMoT usage. For example, `saturation_1.m` uses `1-smoothThreshold_storage_logistic(S,Smax)` for saturation excess above capacity, while `saturation_9.m`, `evap_14.m`, and `evap_16.m` use `smoothThreshold_storage_logistic` for below-threshold/deficit logic. Therefore the dMoT helper, which explicitly returns approximately 1 when `S > threshold`, has the opposite storage-gate orientation from the MATLAB helper name.",
            "",
            "## Value Comparison Summary",
            "| candidate | max abs diff | near-threshold max diff | mean bias | opposite monotonic | exceeds water | negative output |",
            "| --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for c in candidates:
        v = values[c.candidate_id]
        lines.append(
            f"| {c.candidate_id} | {v['max_abs_diff']:.6g} | {v['near_threshold_max_abs_diff']:.6g} | "
            f"{v['signed_bias_mean']:.6g} | {v['opposite_monotonic_direction_from_matlab']} | "
            f"{v['dmot_output_exceeds_available_water']} | {v['dmot_output_negative_when_not_expected']} |"
        )

    lines.extend(
        [
            "",
            "## Gradient Diagnostic Summary",
            "| candidate | max abs gradient | near-threshold max gradient | autograd-FD max error | spike count | NaN | Inf |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for c in candidates:
        g = grads[c.candidate_id]
        lines.append(
            f"| {c.candidate_id} | {g['max_abs_gradient']:.6g} | {g['max_abs_gradient_near_threshold']:.6g} | "
            f"{g['max_autograd_vs_fd_error']:.6g} | {g['gradient_spike_count']} | {g['nan_gradient_count']} | {g['inf_gradient_count']} |"
        )

    lines.extend(["", "## Formula Cards"])
    for c in candidates:
        cls = CLASSIFICATION[c.candidate_id]
        lines.extend(
            [
                f"### {c.candidate_id}: {c.function_or_class}",
                f"- Active usage: {USAGE[c.candidate_id]['active_usage_status']}; called by {USAGE[c.candidate_id]['called_by_models']}.",
                f"- dMoT form: {c.dmot_expression_summary}.",
                f"- MATLAB form: {c.matlab_expression_summary}.",
                f"- Diagnosis: {cls['short_reason']}",
                f"- Recommended action: `{cls['recommended_action']}`.",
            ]
        )

    plot_files = sorted(p.name for p in PLOTS_DIR.glob("*.png"))
    lines.extend(["", "## Plot Index"])
    lines.extend(f"- `plots/{name}`" for name in plot_files)

    lines.extend(
        [
            "",
            "## Final Ranking",
            "| rank | candidate | active usage | issue type | mismatch likelihood | action | reason |",
            "| ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in ranking_rows:
        lines.append(
            f"| {row['rank']} | {row['candidate_id']} | {row['active_usage_status']} | {row['likely_issue_type']} | "
            f"{row['implementation_mismatch_likelihood']} | `{row['recommended_action']}` | {row['short_reason']} |"
        )

    lines.extend(
        [
            "",
            "## Recommended Next Actions",
            "- Fixed in the current workspace: F017, F010, F009, and inactive helper F013.",
            "- Keep but document as smoothing approximations: F007 and F011.",
            "- Inactive/no immediate model effect: F005.",
            "- Remaining gradient-impact formulas: F007 and F005 have sharp storage-gate gradients because the tested threshold is 0.01; F010/F009 now have the corrected activation direction but still retain a sharp differentiable transition.",
            "",
            "Model implementation was modified in this follow-up fix pass; this report is regenerated from the modified formulas.",
        ]
    )
    (OUTPUT_DIR / "focused_formula_review_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates_by_id = {c.candidate_id: c for c in build_candidates()}
    candidates = [candidates_by_id[cid] for cid in FOCUSED_IDS]

    active_rows = _active_usage_rows(candidates)
    value_rows = [_value_row(c) for c in candidates]
    grad_rows = [_gradient_row(c) for c in candidates]
    ranking_rows = _ranking_rows(candidates, value_rows, grad_rows)

    _write_csv(OUTPUT_DIR / "active_usage_map.csv", active_rows)
    _write_csv(OUTPUT_DIR / "focused_value_comparison.csv", value_rows)
    _write_csv(OUTPUT_DIR / "focused_gradient_diagnostics.csv", grad_rows)
    _write_csv(OUTPUT_DIR / "final_formula_review_ranking.csv", ranking_rows)
    _write_formula_cards(candidates, value_rows, grad_rows)
    _make_plots(candidates)
    _write_report(candidates, value_rows, grad_rows, ranking_rows)

    summary = {
        "focused_candidates": len(candidates),
        "active_registered_or_special": sum(
            USAGE[c.candidate_id]["active_usage_status"].startswith("active") for c in candidates
        ),
        "recommended_actions": {
            action: sum(1 for c in candidates if CLASSIFICATION[c.candidate_id]["recommended_action"] == action)
            for action in sorted({CLASSIFICATION[c.candidate_id]["recommended_action"] for c in candidates})
        },
        "outputs": str(OUTPUT_DIR),
    }
    (OUTPUT_DIR / "focused_review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
