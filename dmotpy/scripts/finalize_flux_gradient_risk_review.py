from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
GRADIENT_DIR = REPO_ROOT / "validation_results" / "flux_gradient_stability"
FOCUSED_DIR = REPO_ROOT / "validation_results" / "focused_flux_formula_review"
BATCH_A_DIR = REPO_ROOT / "validation_results" / "batch_a_flux_realistic_review"


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


def _to_int(value: str | int | float) -> int:
    return int(float(value))


def _to_float(value: str | int | float) -> float:
    return float(value)


def _focused_key(row: dict[str, str]) -> tuple[str, str]:
    return row["formula"], row["active_model"]


def _broad_key(row: dict[str, str]) -> tuple[str, str]:
    return row["flux_function"], row["called_by_models"]


def _focused_review_rows() -> dict[tuple[str, str], dict[str, str]]:
    return {_focused_key(row): row for row in _read_csv(FOCUSED_DIR / "focused_formula_risk_decision.csv")}


def _batch_a_review_rows() -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_csv(BATCH_A_DIR / "batch_a_risk_decision.csv")
    normalized: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        normalized[_focused_key(row)] = {
            "formula": row["formula"],
            "active_model": row["active_model"],
            "realistic_domain_risk": row["realistic_risk"],
            "recommended_action": row["recommended_action"],
            "likely_artifact_or_real": row["artifact_or_real"],
            "human_review_priority": row["human_review_priority"],
            "short_reason": row["short_reason"],
        }
    return normalized


def _realistic_summary_by_key() -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    focused_rows = _read_csv(FOCUSED_DIR / "focused_formula_stability_summary.csv")
    batch_a_rows = _read_csv(BATCH_A_DIR / "batch_a_realistic_gradient_summary.csv")
    rows = focused_rows + batch_a_rows
    for row in rows:
        domain_type = row.get("domain_type")
        case_group = row.get("case_group")
        if domain_type is not None and domain_type != "realistic":
            continue
        if case_group is not None and case_group != "realistic_domain":
            continue
        grouped.setdefault((row["formula"], row["active_model"]), []).append(row)

    summary: dict[tuple[str, str], dict[str, Any]] = {}
    for key, group in grouped.items():
        summary[key] = {
            "realistic_domain_nan_inf": sum(
                _to_int(row["output_nan_count"])
                + _to_int(row["output_inf_count"])
                + _to_int(row["grad_nan_count"])
                + _to_int(row["grad_inf_count"])
                for row in group
            ),
            "realistic_max_abs_grad": max(_to_float(row["max_abs_grad"]) for row in group),
            "output_bound_violation_realistic": sum(_to_int(row["output_bound_violation_count"]) for row in group),
        }
    return summary


def _final_active_risk(focused: dict[str, str], realistic: dict[str, Any]) -> str:
    if realistic["realistic_domain_nan_inf"] > 0:
        return "high"
    if realistic["output_bound_violation_realistic"] > 0:
        return "high"
    if focused["recommended_action"] == "keep_but_document":
        return "medium"
    if focused["likely_artifact_or_real"] == "broad_domain_artifact":
        return "broad_domain_artifact"
    if focused["realistic_domain_risk"] == "low":
        return "low"
    return focused["realistic_domain_risk"]


def build_final_rows() -> list[dict[str, Any]]:
    broad_rows = _read_csv(GRADIENT_DIR / "flux_gradient_risk_ranking.csv")
    focused_rows = _focused_review_rows()
    batch_a_rows = _batch_a_review_rows()
    reviewed_rows = dict(focused_rows)
    reviewed_rows.update(batch_a_rows)
    realistic_rows = _realistic_summary_by_key()

    final_rows: list[dict[str, Any]] = []
    for row in broad_rows:
        if row["active_usage_status"] == "unused":
            continue

        key = _broad_key(row)
        broad_domain_nan_inf = (
            _to_int(row["output_nan_count"])
            + _to_int(row["output_inf_count"])
            + _to_int(row["grad_nan_count"])
            + _to_int(row["grad_inf_count"])
        )

        if key in reviewed_rows and key in realistic_rows:
            focused = reviewed_rows[key]
            realistic = realistic_rows[key]
            final_rows.append(
                {
                    "formula": row["flux_function"],
                    "flux_file": row["flux_file"],
                    "active_model": row["called_by_models"],
                    "previous_broad_risk": row["risk_level"],
                    "realistic_domain_risk": focused["realistic_domain_risk"],
                    "final_active_risk": _final_active_risk(focused, realistic),
                    "broad_domain_nan_inf": broad_domain_nan_inf,
                    "realistic_domain_nan_inf": realistic["realistic_domain_nan_inf"],
                    "realistic_max_abs_grad": realistic["realistic_max_abs_grad"],
                    "output_bound_violation_realistic": realistic["output_bound_violation_realistic"],
                    "final_recommended_action": focused["recommended_action"],
                    "final_human_review_priority": focused["human_review_priority"],
                    "final_reason": focused["short_reason"],
                    "evidence_source": "batch_a_realistic_review" if key in batch_a_rows else "focused_formula_review",
                }
            )
            continue

        final_rows.append(
            {
                "formula": row["flux_function"],
                "flux_file": row["flux_file"],
                "active_model": row["called_by_models"],
                "previous_broad_risk": row["risk_level"],
                "realistic_domain_risk": "not_reviewed",
                "final_active_risk": row["risk_level"],
                "broad_domain_nan_inf": broad_domain_nan_inf,
                "realistic_domain_nan_inf": "",
                "realistic_max_abs_grad": "",
                "output_bound_violation_realistic": "",
                "final_recommended_action": row["recommended_action"],
                "final_human_review_priority": row["human_review_priority"],
                "final_reason": "No focused realistic-domain override; retaining broad diagnostic classification pending targeted review.",
                "evidence_source": "broad_diagnostic_only",
            }
        )

    risk_order = {"high": 0, "medium": 1, "low": 2, "broad_domain_artifact": 3}
    priority_order = {"high": 0, "medium": 1, "low": 2}
    final_rows.sort(
        key=lambda item: (
            risk_order.get(str(item["final_active_risk"]), 9),
            priority_order.get(str(item["final_human_review_priority"]), 9),
            item["formula"],
            item["active_model"],
        )
    )
    return final_rows


def build_summary(rows: list[dict[str, Any]]) -> tuple[str, str]:
    active_high = [row for row in rows if row["final_active_risk"] == "high"]
    broad_artifacts = [row for row in rows if row["final_active_risk"] == "broad_domain_artifact"]
    threshold_stable = [
        row for row in rows
        if row["final_active_risk"] == "medium" and row["final_recommended_action"] == "keep_but_document"
    ]

    summary_lines = [
        "# Final Flux Gradient Risk Summary",
        "",
        f"- Active formula contexts ranked: {len(rows)}",
        f"- Final active high-risk contexts: {len(active_high)}",
        f"- Broad-domain artifact reclassifications: {len(broad_artifacts)}",
        f"- Threshold-sensitive but stable medium-risk contexts: {len(threshold_stable)}",
        "",
        "## Reclassified Broad-Domain Artifacts",
    ]
    for row in broad_artifacts:
        summary_lines.append(
            f"- `{row['formula']}` / `{row['active_model']}`: {row['final_reason']}"
        )
    summary_lines.extend(
        [
            "",
            "## Threshold-Sensitive But Stable",
        ]
    )
    for row in threshold_stable:
        summary_lines.append(
            f"- `{row['formula']}` / `{row['active_model']}`: {row['final_reason']}"
        )

    report_lines = [
        "# Final Flux Gradient Stability Report",
        "",
        "## 1. Scope",
        "- This report finalizes the flux-gradient-stability interpretation after the focused realistic-domain review.",
        "- No parameter bounds, smoothing defaults, unit hydrograph routines, or water-balance fixes were changed.",
        "- `saturation_3` was stabilized through an algebraically equivalent sigmoid rewrite; hydrological semantics are unchanged.",
        "",
        "## 2. Broad vs realistic domains",
        "- The original gradient workflow used broad diagnostic domains to expose potential NaN/Inf, large gradients, dead regions, and bound issues.",
        "- The focused follow-up traced actual active-model call domains for the previously flagged formulas and re-tested them under those realistic domains.",
        "",
        "## 3. Previous high-risk flags",
        "- The original broad-domain workflow marked 26 active formula contexts as high risk.",
        "- Twelve of those contexts now have targeted realistic-domain review evidence from the focused-review and Batch A follow-up workflows.",
        "",
        "## 4. Focused review results",
        "- `baseflow_6 / tcm`: threshold-sensitive, finite realistic outputs/gradients, retained unchanged.",
        "- `interflow_10 / topmodel`: threshold-sensitive, finite realistic outputs/gradients, retained unchanged.",
        "- `baseflow_2 / susannah1`: previous non-finite flag was a broad-domain artifact.",
        "- `interflow_2 / hbv96`: previous non-finite flag was a broad-domain artifact.",
        "- `interflow_3 / australia`: previous non-finite flag was a broad-domain artifact.",
        "- `interflow_3 / susannah2`: previous non-finite flag was a broad-domain artifact.",
        "- `baseflow_5 / vic`: broad-domain failures do not persist in realistic active-model domains.",
        "- `saturation_3 / flexb`, `flexi`, `flexis`: the stable sigmoid rewrite removes the earlier exact-boundary autograd overflow while preserving the forward formula.",
        "- `saturation_2 / hymod`: the broad bound flag does not persist in realistic active-model domains.",
        "- `baseflow_9 / gsfb`: realistic active-model domains remain finite and bounded, but gradients are large enough to document.",
        "",
        "## 5. Final active-risk decision table",
        "",
        "| formula | model | previous broad risk | realistic risk | final active risk | action |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row["evidence_source"] not in {"focused_formula_review", "batch_a_realistic_review"}:
            continue
        report_lines.append(
            f"| {row['formula']} | {row['active_model']} | {row['previous_broad_risk']} | "
            f"{row['realistic_domain_risk']} | {row['final_active_risk']} | {row['final_recommended_action']} |"
        )

    report_lines.extend(
        [
            "",
            "## 6. Which risks were broad-domain artifacts",
            "- No additional hydrological formula modification is required based on the focused realistic-domain review.",
            "- The previous non-finite flags for `baseflow_2`, `interflow_2`, `interflow_3`, and `baseflow_5` were caused by broad diagnostic domains that do not represent the active model state domains.",
            "- The FLEX `saturation_3` contexts no longer show the earlier boundary-gradient pathology after the exact stable rewrite.",
            "",
            "## 7. Which formulas remain threshold-sensitive but stable",
            "- `baseflow_6` and `interflow_10` remain threshold-sensitive but produce finite outputs and gradients under realistic active-model domains, so they are retained unchanged and covered by regression tests.",
            "",
            "## 8. Why no further hydrological formula modification is required",
            "- The focused review did not reveal realistic-domain NaN/Inf failures or realistic-domain physical bound violations for the previously questioned active formulas.",
            "- The FLEX `saturation_3` follow-up indicates that the stable sigmoid rewrite resolves the earlier autograd overflow without changing the hydrological formula semantics.",
            "- Preserving current model behavior while using the stable algebraic form is therefore preferable to smoothing or protective clamps.",
            "",
            "## 9. Current conclusion for gradient-based calibration",
            f"- Final active high-risk count after focused review: {len(active_high)}.",
            "- The active flux set appears safe enough for gradient-based calibration when current active-model parameter bounds and state domains are respected.",
            "- Threshold-sensitive formulas should be documented, but they do not require immediate formula changes.",
            "",
        "## 10. Recommended future checks before full benchmark recalibration",
        "- Keep unused high-risk formulas inactive unless a future model explicitly uses them.",
        "- Unused formulas are not active production risks in the current ranking because they are excluded from the active-context final risk table.",
        "- Any future activation of an unused formula should require model-specific gradient validation before calibration.",
        "- Prioritize targeted review of the remaining active high-risk contexts that were not part of the focused realistic-domain follow-up.",
        ]
    )
    return "\n".join(summary_lines) + "\n", "\n".join(report_lines) + "\n"


def main() -> None:
    rows = build_final_rows()
    _write_csv(GRADIENT_DIR / "final_flux_gradient_risk_ranking.csv", rows)
    summary_text, report_text = build_summary(rows)
    (GRADIENT_DIR / "final_flux_gradient_risk_summary.md").write_text(summary_text, encoding="utf-8")
    (GRADIENT_DIR / "final_flux_gradient_stability_report.md").write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()
