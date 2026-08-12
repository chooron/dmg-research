from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
GRADIENT_DIR = REPO_ROOT / "validation_results" / "flux_gradient_stability"
FOCUSED_DIR = REPO_ROOT / "validation_results" / "focused_flux_formula_review"

FINAL_RANKING = GRADIENT_DIR / "final_flux_gradient_risk_ranking.csv"
BROAD_RANKING = GRADIENT_DIR / "flux_gradient_risk_ranking.csv"
USAGE_MAP = GRADIENT_DIR / "flux_usage_parameter_map.csv"

EXCLUDED_ALREADY_REVIEWED = {
    ("baseflow_2", "susannah1"),
    ("interflow_2", "hbv96"),
    ("interflow_3", "australia"),
    ("interflow_3", "susannah2"),
    ("baseflow_5", "vic"),
    ("baseflow_6", "tcm"),
    ("interflow_10", "topmodel"),
}

CONTEXT_REVIEW_CONFIG: dict[tuple[str, str], dict[str, str]] = {
    ("baseflow_4", "topmodel"): {
        "failure_mode": "physical_bound_violation",
        "likely_cause": "Deficit-store baseflow is compared against a generic storage cap in the broad diagnostic, but TOPMODEL uses downstream space accounting rather than direct storage depletion.",
        "needs_realistic_domain_review": "yes",
        "priority": "high",
        "short_reason": "Likely semantic mismatch between deficit-store behavior and the generic bound-check heuristic.",
        "recommended_initial_review_type": "deficit_store_semantics_and_realistic_rollout",
    },
    ("baseflow_9", "gsfb"): {
        "failure_mode": "physical_bound_violation",
        "likely_cause": "Thresholded baseflow can exceed immediately available S2 in the standalone broad diagnostic and is then capped in core code.",
        "needs_realistic_domain_review": "yes",
        "priority": "high",
        "short_reason": "Potentially real active-model sensitivity because the core model applies a post-call cap.",
        "recommended_initial_review_type": "realistic_rollout_with_post_flux_cap_check",
    },
    ("depression_1", "modhydrolog"): {
        "failure_mode": "physical_bound_violation",
        "likely_cause": "Depression trapping uses incoming runoff and residual capacity with a broad-domain overflow flag plus a large dead region.",
        "needs_realistic_domain_review": "yes",
        "priority": "high",
        "short_reason": "Overflow-style process with possible post-state semantics that the generic bound test may misread.",
        "recommended_initial_review_type": "realistic_rollout_with_capacity_semantics_check",
    },
    ("evap_16", "penman"): {
        "failure_mode": "diagnostic_domain_uncertain",
        "likely_cause": "The broad wrapper likely misinterprets deficit-store and threshold arguments as available storage, so its bound violation heuristic is not semantically aligned with Penman usage.",
        "needs_realistic_domain_review": "yes",
        "priority": "medium",
        "short_reason": "Likely wrapper/heuristic artifact, but active in a calibration-relevant model and should be verified under real domains.",
        "recommended_initial_review_type": "realistic_rollout_with_argument_semantics_audit",
    },
    ("evap_3", "hbv96"): {
        "failure_mode": "physical_bound_violation",
        "likely_cause": "Standalone evaporation can exceed available storage before the HBV96 core step applies `min(flux_ea_pot, S3)`.",
        "needs_realistic_domain_review": "yes",
        "priority": "high",
        "short_reason": "Common active model and broad failure depends on a post-call cap.",
        "recommended_initial_review_type": "realistic_rollout_with_post_flux_cap_check",
    },
    ("evap_7", "vic"): {
        "failure_mode": "physical_bound_violation",
        "likely_cause": "Standalone relative-storage evaporation can exceed active store supply before VIC applies downstream storage/PET caps.",
        "needs_realistic_domain_review": "yes",
        "priority": "medium",
        "short_reason": "Likely cap-dependent rather than intrinsically unstable, but active in three VIC ET branches.",
        "recommended_initial_review_type": "realistic_rollout_with_post_flux_cap_check",
    },
    ("excess_1", "australia"): {
        "failure_mode": "physical_bound_violation",
        "likely_cause": "Overflow is computed on a prospective storage state `So`, so the generic bound heuristic may not match the actual overflow semantics.",
        "needs_realistic_domain_review": "yes",
        "priority": "medium",
        "short_reason": "Possible broad-domain artifact driven by prospective-state semantics.",
        "recommended_initial_review_type": "realistic_rollout_with_overflow_semantics_check",
    },
    ("excess_1", "susannah2"): {
        "failure_mode": "physical_bound_violation",
        "likely_cause": "Overflow is compared against a generic storage bound, but Susannah2 later rescales total recharge plus excess jointly.",
        "needs_realistic_domain_review": "yes",
        "priority": "medium",
        "short_reason": "Could be a broad-domain artifact because the core step constrains combined outflow after the raw formula call.",
        "recommended_initial_review_type": "realistic_rollout_with_joint_scaling_check",
    },
    ("excess_1", "vic"): {
        "failure_mode": "physical_bound_violation",
        "likely_cause": "Overflow from interception uses a prospective store state and is later combined with throughfall into a bounded soil-input calculation.",
        "needs_realistic_domain_review": "yes",
        "priority": "medium",
        "short_reason": "Semantics of `So` differ from the generic available-storage heuristic used in the broad diagnostic.",
        "recommended_initial_review_type": "realistic_rollout_with_overflow_semantics_check",
    },
    ("recharge_1", "modhydrolog"): {
        "failure_mode": "physical_bound_violation",
        "likely_cause": "Recharge is explicitly capped by remaining infiltration after the formula call, so the raw broad-domain bound violation may be post-cap dependent.",
        "needs_realistic_domain_review": "yes",
        "priority": "medium",
        "short_reason": "Likely broad-domain artifact but still worth tracing because the core model depends on downstream capping.",
        "recommended_initial_review_type": "realistic_rollout_with_post_flux_cap_check",
    },
    ("recharge_1", "simhyd"): {
        "failure_mode": "physical_bound_violation",
        "likely_cause": "Recharge is explicitly capped by remaining infiltration after the formula call, so the raw broad-domain bound violation may be post-cap dependent.",
        "needs_realistic_domain_review": "yes",
        "priority": "medium",
        "short_reason": "Same formula pattern as MODHYDROLOG; likely cap-dependent rather than inherently unstable.",
        "recommended_initial_review_type": "realistic_rollout_with_post_flux_cap_check",
    },
    ("recharge_2", "hbv96"): {
        "failure_mode": "physical_bound_violation",
        "likely_cause": "Non-linear recharge scaling can exceed available water in the broad diagnostic before HBV96 applies `min(flux_r_pot, S3)`.",
        "needs_realistic_domain_review": "yes",
        "priority": "high",
        "short_reason": "Active in HBV96 and combines non-linearity with a post-call cap.",
        "recommended_initial_review_type": "realistic_rollout_with_post_flux_cap_check",
    },
    ("saturation_2", "hymod"): {
        "failure_mode": "physical_bound_violation",
        "likely_cause": "For low `b_exp`, the outflow fraction can exceed one in the standalone formula, after which HYMOD clamps runoff to precipitation.",
        "needs_realistic_domain_review": "yes",
        "priority": "high",
        "short_reason": "Likely genuine formula-level risk if realistic domains still approach the low-parameter broad-domain cases.",
        "recommended_initial_review_type": "realistic_rollout_with_fraction_bound_check",
    },
    ("saturation_3", "flexb"): {
        "failure_mode": "nonfinite_output_or_gradient",
        "likely_cause": "The exponential argument contains division by `p1 + nearzero`, so very small `beta` can create overflow and non-finite gradients even when outputs remain finite.",
        "needs_realistic_domain_review": "yes",
        "priority": "high",
        "short_reason": "This is the strongest remaining true autograd concern because the broad diagnostic already produced active-context NaN gradients.",
        "recommended_initial_review_type": "realistic_rollout_with_autograd_overflow_trace",
    },
    ("saturation_3", "flexi"): {
        "failure_mode": "nonfinite_output_or_gradient",
        "likely_cause": "The exponential argument contains division by `p1 + nearzero`, so very small `beta` can create overflow and non-finite gradients even when outputs remain finite.",
        "needs_realistic_domain_review": "yes",
        "priority": "high",
        "short_reason": "Same autograd-overflow mechanism as FLEXB, with an active production context.",
        "recommended_initial_review_type": "realistic_rollout_with_autograd_overflow_trace",
    },
    ("saturation_3", "flexis"): {
        "failure_mode": "nonfinite_output_or_gradient",
        "likely_cause": "The exponential argument contains division by `p1 + nearzero`, so very small `beta` can create overflow and non-finite gradients even when outputs remain finite.",
        "needs_realistic_domain_review": "yes",
        "priority": "high",
        "short_reason": "Same autograd-overflow mechanism as FLEXB/FLEXI and currently the highest-priority remaining family.",
        "recommended_initial_review_type": "realistic_rollout_with_autograd_overflow_trace",
    },
    ("split_1", "flexb"): {
        "failure_mode": "diagnostic_domain_uncertain",
        "likely_cause": "The usage mapper inferred a generic `p1` range for the expression `1.0 - d_split`, so broad-domain bound violations likely include impossible `p1 > 1` cases.",
        "needs_realistic_domain_review": "yes",
        "priority": "medium",
        "short_reason": "Very likely a parameter-expression range artifact rather than a true formula instability.",
        "recommended_initial_review_type": "parameter_expression_range_audit",
    },
    ("split_1", "flexi"): {
        "failure_mode": "diagnostic_domain_uncertain",
        "likely_cause": "The usage mapper inferred a generic `p1` range for the expression `1.0 - d_split`, so broad-domain bound violations likely include impossible `p1 > 1` cases.",
        "needs_realistic_domain_review": "yes",
        "priority": "medium",
        "short_reason": "Likely an inference artifact; realistic domain tracing should verify the true `d_split` range directly.",
        "recommended_initial_review_type": "parameter_expression_range_audit",
    },
    ("split_1", "flexis"): {
        "failure_mode": "diagnostic_domain_uncertain",
        "likely_cause": "The usage mapper inferred a generic `p1` range for the expression `1.0 - d_split`, so broad-domain bound violations likely include impossible `p1 > 1` cases.",
        "needs_realistic_domain_review": "yes",
        "priority": "medium",
        "short_reason": "Likely an inference artifact; realistic domain tracing should verify the true `d_split` range directly.",
        "recommended_initial_review_type": "parameter_expression_range_audit",
    },
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_int(value: str) -> int:
    return int(float(value))


def _to_float(value: str) -> float:
    return float(value)


def _json_merge(values: list[str]) -> str:
    payloads = []
    for value in values:
        if value:
            try:
                payloads.append(json.loads(value))
            except json.JSONDecodeError:
                payloads.append(value)
    return json.dumps(payloads, ensure_ascii=False, sort_keys=False)


def _context_key(row: dict[str, str]) -> tuple[str, str]:
    return row["formula"], row["active_model"]


def _usage_key(row: dict[str, str]) -> tuple[str, str]:
    return row["flux_function"], row["called_by_models"]


def _priority_rank(priority: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(priority, 9)


def build_remaining_high_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final_rows = _read_csv(FINAL_RANKING)
    broad_rows = {_usage_key(row): row for row in _read_csv(BROAD_RANKING)}
    usage_rows = _read_csv(USAGE_MAP)

    usage_map: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in usage_rows:
        usage_map.setdefault(_usage_key(row), []).append(row)

    remaining_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for row in final_rows:
        key = _context_key(row)
        if row["final_active_risk"] != "high":
            continue
        if key in EXCLUDED_ALREADY_REVIEWED:
            continue
        broad = broad_rows[key]
        config = CONTEXT_REVIEW_CONFIG[key]
        usages = usage_map.get(key, [])
        remaining_rows.append(
            {
                "formula": row["formula"],
                "flux_file": row["flux_file"],
                "active_model": row["active_model"],
                "previous_broad_risk": row["previous_broad_risk"],
                "final_active_risk": row["final_active_risk"],
                "main_risk_reason": broad["risk_reason"],
                "max_abs_grad": _to_float(broad["max_abs_grad"]),
                "output_nan_count": _to_int(broad["output_nan_count"]),
                "output_inf_count": _to_int(broad["output_inf_count"]),
                "grad_nan_count": _to_int(broad["grad_nan_count"]),
                "grad_inf_count": _to_int(broad["grad_inf_count"]),
                "output_bound_violation_count": _to_int(broad["output_bound_violation_count"]),
                "zero_gradient_fraction": _to_float(broad["zero_gradient_fraction"]),
                "active_usage_context": _json_merge(
                    [
                        json.dumps(
                            {
                                "call_site": usage["call_sites"],
                                "parameter_mapping": usage["parameter_mapping"],
                                "parameter_bounds": usage["parameter_bounds"],
                                "state_mapping": usage["state_variable_mapping"],
                            },
                            ensure_ascii=False,
                        )
                        for usage in usages
                    ]
                ),
                "recommended_initial_review_type": config["recommended_initial_review_type"],
            }
        )
        failure_rows.append(
            {
                "formula": row["formula"],
                "active_model": row["active_model"],
                "failure_mode": config["failure_mode"],
                "likely_cause": config["likely_cause"],
                "needs_realistic_domain_review": config["needs_realistic_domain_review"],
                "priority": config["priority"],
                "short_reason": config["short_reason"],
            }
        )

    remaining_rows.sort(key=lambda item: (_priority_rank(CONTEXT_REVIEW_CONFIG[(item["formula"], item["active_model"])]["priority"]), item["formula"], item["active_model"]))
    failure_rows.sort(key=lambda item: (_priority_rank(item["priority"]), item["failure_mode"], item["formula"], item["active_model"]))
    return remaining_rows, failure_rows


def build_review_plan(remaining_rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]]) -> str:
    failure_groups: dict[str, list[dict[str, Any]]] = {}
    for row in failure_rows:
        failure_groups.setdefault(row["failure_mode"], []).append(row)

    high_priority = [row for row in failure_rows if row["priority"] == "high"]
    medium_priority = [row for row in failure_rows if row["priority"] == "medium"]
    needs_rollout = [row for row in failure_rows if row["needs_realistic_domain_review"] == "yes"]

    lines = [
        "# Remaining High-Risk Review Plan",
        "",
        "## 1. Remaining active high-risk contexts",
    ]
    for row in remaining_rows:
        lines.append(
            f"- `{row['formula']}` / `{row['active_model']}`: {row['main_risk_reason']} "
            f"(max_abs_grad={row['max_abs_grad']}, bound_violations={row['output_bound_violation_count']}, "
            f"grad_nan={row['grad_nan_count']})"
        )

    lines.extend(
        [
            "",
            "## 2. Failure-mode grouping",
        ]
    )
    for failure_mode, rows in sorted(failure_groups.items()):
        lines.append(f"- `{failure_mode}`: {len(rows)} contexts")
        for row in rows:
            lines.append(f"  - {row['formula']} / {row['active_model']}: {row['short_reason']}")

    lines.extend(
        [
            "",
            "## 3. Priority order",
            "### High priority",
        ]
    )
    for row in high_priority:
        lines.append(f"- `{row['formula']}` / `{row['active_model']}`: {row['short_reason']}")
    lines.append("### Medium priority")
    for row in medium_priority:
        lines.append(f"- `{row['formula']}` / `{row['active_model']}`: {row['short_reason']}")

    lines.extend(
        [
            "",
            "## 4. Contexts that need realistic-domain rollout tracing",
        ]
    )
    for row in needs_rollout:
        lines.append(f"- `{row['formula']}` / `{row['active_model']}`")

    lines.extend(
        [
            "",
            "## 5. Contexts that likely only need documentation or range-audit follow-up",
            "- `evap_16 / penman`: likely broad-diagnostic heuristic mismatch involving deficit-store and threshold semantics.",
            "- `split_1 / flexb`, `split_1 / flexi`, `split_1 / flexis`: likely parameter-expression range artifacts because the review mapper inferred `1 - d_split` too loosely.",
            "",
            "## 6. Contexts that may eventually need safety clamps",
            "- `baseflow_9 / gsfb`",
            "- `depression_1 / modhydrolog`",
            "- `evap_3 / hbv96`",
            "- `evap_7 / vic`",
            "- `recharge_1 / modhydrolog`, `recharge_1 / simhyd`",
            "- `recharge_2 / hbv96`",
            "- `saturation_2 / hymod`",
            "",
            "## 7. Contexts that may eventually need smoothing review",
            "- None should be moved to smoothing review yet without realistic-domain evidence.",
            "- If realistic-domain tracing shows hard activation plus material calibration dead regions, revisit `depression_1`, `excess_1`, and `recharge_2` first.",
            "",
            "## 8. Contexts that should not be modified without hydrological interpretation",
            "- `baseflow_4 / topmodel`: deficit-store semantics make naive storage-cap fixes risky.",
            "- `evap_16 / penman`: the diagnostic wrapper does not reflect the Penman deficit-store interpretation cleanly.",
            "- `saturation_3 / flexb`, `saturation_3 / flexi`, `saturation_3 / flexis`: any numerical fix must preserve the intended FLEX-family infiltration partition behavior.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_report(remaining_rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]]) -> str:
    failure_counts = Counter(row["failure_mode"] for row in failure_rows)
    high_priority = [row for row in failure_rows if row["priority"] == "high"]
    likely_artifacts = [
        row for row in failure_rows
        if row["failure_mode"] in {"diagnostic_domain_uncertain"}
        or (row["formula"], row["active_model"]) in {
            ("baseflow_4", "topmodel"),
            ("excess_1", "australia"),
            ("excess_1", "susannah2"),
            ("excess_1", "vic"),
            ("recharge_1", "modhydrolog"),
            ("recharge_1", "simhyd"),
        }
    ]
    real_risk_candidates = [
        row for row in failure_rows
        if (row["formula"], row["active_model"]) in {
            ("saturation_3", "flexb"),
            ("saturation_3", "flexi"),
            ("saturation_3", "flexis"),
            ("saturation_2", "hymod"),
            ("baseflow_9", "gsfb"),
            ("recharge_2", "hbv96"),
            ("evap_3", "hbv96"),
        }
    ]

    lines = [
        "# Remaining Active High-Risk Review Report",
        "",
        "## 1. Scope",
        "- This report isolates the remaining active high-risk flux contexts that were not covered by the previous focused realistic-domain review.",
        "- It does not modify hydrological formulas, smoothing, unit hydrograph code, or water-balance fixes.",
        "",
        f"## 2. Remaining active high-risk count",
        f"- Remaining active high-risk contexts: {len(remaining_rows)}",
        "",
        "## 3. Excluded already-reviewed contexts",
        "- `baseflow_2 / susannah1`",
        "- `interflow_2 / hbv96`",
        "- `interflow_3 / australia`",
        "- `interflow_3 / susannah2`",
        "- `baseflow_5 / vic`",
        "- `baseflow_6 / tcm`",
        "- `interflow_10 / topmodel`",
        "",
        "## 4. Remaining high-risk table",
        "",
        "| formula | model | broad reason | max_abs_grad | bound violations | grad_nan | review type |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in remaining_rows:
        lines.append(
            f"| {row['formula']} | {row['active_model']} | {row['main_risk_reason']} | "
            f"{row['max_abs_grad']} | {row['output_bound_violation_count']} | {row['grad_nan_count']} | "
            f"{row['recommended_initial_review_type']} |"
        )

    lines.extend(
        [
            "",
            "## 5. Failure-mode summary",
        ]
    )
    for failure_mode, count in sorted(failure_counts.items()):
        lines.append(f"- `{failure_mode}`: {count}")

    lines.extend(
        [
            "",
            "## 6. Priority ranking",
            "- Highest-priority realistic-domain review batch:",
        ]
    )
    for row in high_priority:
        lines.append(f"  - `{row['formula']}` / `{row['active_model']}`: {row['short_reason']}")

    lines.extend(
        [
            "",
            "## 7. Recommended focused-review batches",
            "- Batch A: `saturation_3 / flexb`, `saturation_3 / flexi`, `saturation_3 / flexis`, `saturation_2 / hymod`, `baseflow_9 / gsfb`.",
            "- Batch B: `baseflow_4 / topmodel`, `evap_3 / hbv96`, `recharge_2 / hbv96`, `depression_1 / modhydrolog`.",
            "- Batch C: `excess_1 / australia`, `excess_1 / susannah2`, `excess_1 / vic`, `recharge_1 / modhydrolog`, `recharge_1 / simhyd`, `evap_7 / vic`.",
            "- Batch D: `evap_16 / penman`, `split_1 / flexb`, `split_1 / flexi`, `split_1 / flexis`.",
            "",
            "## 8. Which formulas should not be changed yet",
            "- No immediate formula modification is justified at this stage.",
            "- `baseflow_4 / topmodel` and `evap_16 / penman` should not be edited before a realistic-domain semantic audit because the broad diagnostics likely mis-handle deficit-store semantics.",
            "- `split_1` FLEX contexts should not be changed before verifying the true `d_split` range directly from model bounds.",
            "",
            "## 9. Whether any immediate formula modification is justified",
            "- No. The remaining contexts still need realistic-domain evidence before any hydrological or numerical intervention is justified.",
            "",
            "## 10. Next commands to run",
            "- `python scripts/review_remaining_high_risk_flux_contexts.py`",
            "- Then implement and run a second focused realistic-domain tracer for Batch A and Batch B contexts.",
        ]
    )

    lines.extend(
        [
            "",
            "## Likely broad-domain artifacts",
        ]
    )
    for row in likely_artifacts:
        lines.append(f"- `{row['formula']}` / `{row['active_model']}`: {row['short_reason']}")

    lines.extend(
        [
            "",
            "## Likely real calibration risks",
        ]
    )
    for row in real_risk_candidates:
        lines.append(f"- `{row['formula']}` / `{row['active_model']}`: {row['short_reason']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    remaining_rows, failure_rows = build_remaining_high_rows()
    _write_csv(GRADIENT_DIR / "remaining_active_high_risk_contexts.csv", remaining_rows)
    _write_csv(GRADIENT_DIR / "remaining_high_risk_failure_mode_summary.csv", failure_rows)
    (GRADIENT_DIR / "remaining_high_risk_review_plan.md").write_text(
        build_review_plan(remaining_rows, failure_rows),
        encoding="utf-8",
    )
    (GRADIENT_DIR / "remaining_active_high_risk_review_report.md").write_text(
        build_report(remaining_rows, failure_rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
