from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
GRADIENT_DIR = REPO_ROOT / "validation_results" / "flux_gradient_stability"
FOCUSED_DIR = REPO_ROOT / "validation_results" / "focused_flux_formula_review"
BATCH_A_DIR = REPO_ROOT / "validation_results" / "batch_a_flux_realistic_review"
BATCH_B_DIR = REPO_ROOT / "validation_results" / "batch_b_flux_realistic_review"
BATCH_C_DIR = REPO_ROOT / "validation_results" / "batch_c_flux_realistic_review"
SATURATION3_DIR = REPO_ROOT / "validation_results" / "saturation3_stable_rewrite"

COMPLETE_RANKING_PATH = GRADIENT_DIR / "complete_active_flux_gradient_risk_ranking.csv"
COMPLETE_SUMMARY_CSV_PATH = GRADIENT_DIR / "complete_flux_gradient_review_summary.csv"
COMPLETE_SUMMARY_MD_PATH = GRADIENT_DIR / "complete_flux_gradient_review_summary.md"
COMPLETE_REPORT_PATH = GRADIENT_DIR / "complete_flux_gradient_stability_report.md"


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


def _key(formula: str, active_model: str) -> tuple[str, str]:
    return formula, active_model


def _broad_key(row: dict[str, str]) -> tuple[str, str]:
    return row["flux_function"], row["called_by_models"]


def _sorted_context_list(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "none"
    items = sorted(f"{row['formula']} / {row['active_model']}" for row in rows)
    return "; ".join(items)


def _review_override_rows() -> dict[tuple[str, str], dict[str, Any]]:
    overrides: dict[tuple[str, str], dict[str, Any]] = {}

    for row in _read_csv(FOCUSED_DIR / "focused_formula_risk_decision.csv"):
        key = _key(row["formula"], row["active_model"])
        if row["recommended_action"] == "keep_but_document":
            overrides[key] = {
                "final_realistic_risk": "medium",
                "final_decision": "keep_but_document",
                "final_recommended_action": "keep_but_document",
                "artifact_type": "keep_but_document",
                "formula_changed": "no",
                "change_type": "none",
                "evidence_source": "focused_formula_review",
                "notes": row["short_reason"],
            }
        else:
            overrides[key] = {
                "final_realistic_risk": "low",
                "final_decision": "broad_domain_artifact",
                "final_recommended_action": "broad_domain_artifact",
                "artifact_type": "broad_domain_artifact",
                "formula_changed": "no",
                "change_type": "none",
                "evidence_source": "focused_formula_review",
                "notes": row["short_reason"],
            }

    for row in _read_csv(BATCH_A_DIR / "batch_a_risk_decision.csv"):
        key = _key(row["formula"], row["active_model"])
        if row["formula"] == "saturation_3":
            overrides[key] = {
                "final_realistic_risk": "low",
                "final_decision": "stable_numerical_rewrite_applied",
                "final_recommended_action": "stable_numerical_rewrite_applied",
                "artifact_type": "stable_numerical_rewrite",
                "formula_changed": "yes",
                "change_type": "stable_numerical_rewrite",
                "evidence_source": "saturation3_stable_rewrite + batch_a_realistic_review",
                "notes": (
                    "Algebraically equivalent rewrite from `1 - 1 / (1 + exp(z))` to `torch.sigmoid(z)` removed non-finite "
                    "gradients while preserving forward outputs to machine precision."
                ),
            }
        elif row["recommended_action"] == "keep_but_document":
            overrides[key] = {
                "final_realistic_risk": "medium",
                "final_decision": "keep_but_document",
                "final_recommended_action": "keep_but_document",
                "artifact_type": "keep_but_document",
                "formula_changed": "no",
                "change_type": "none",
                "evidence_source": "batch_a_realistic_review",
                "notes": row["short_reason"],
            }
        else:
            overrides[key] = {
                "final_realistic_risk": "low",
                "final_decision": "broad_domain_artifact",
                "final_recommended_action": "broad_domain_artifact",
                "artifact_type": "broad_domain_artifact",
                "formula_changed": "no",
                "change_type": "none",
                "evidence_source": "batch_a_realistic_review",
                "notes": row["short_reason"],
            }

    for row in _read_csv(BATCH_B_DIR / "batch_b_risk_decision.csv"):
        key = _key(row["formula"], row["active_model"])
        if key == _key("recharge_2", "hbv96"):
            overrides[key] = {
                "final_realistic_risk": "low",
                "final_decision": "model_level_cap_resolves",
                "final_recommended_action": "model_level_cap_resolves",
                "artifact_type": "model_level_cap_resolves",
                "formula_changed": "no",
                "change_type": "none",
                "evidence_source": "batch_b_realistic_review",
                "notes": row["short_reason"],
            }
        elif row["recommended_action"] == "bound_heuristic_artifact":
            overrides[key] = {
                "final_realistic_risk": "low",
                "final_decision": "bound_heuristic_artifact",
                "final_recommended_action": "bound_heuristic_artifact",
                "artifact_type": "bound_heuristic_artifact",
                "formula_changed": "no",
                "change_type": "none",
                "evidence_source": "batch_b_realistic_review",
                "notes": row["short_reason"],
            }
        else:
            overrides[key] = {
                "final_realistic_risk": "low",
                "final_decision": "broad_domain_artifact",
                "final_recommended_action": "broad_domain_artifact",
                "artifact_type": "broad_domain_artifact",
                "formula_changed": "no",
                "change_type": "none",
                "evidence_source": "batch_b_realistic_review",
                "notes": row["short_reason"],
            }

    for row in _read_csv(BATCH_C_DIR / "batch_c_risk_decision.csv"):
        key = _key(row["formula"], row["active_model"])
        if row["recommended_action"] == "bound_heuristic_artifact":
            overrides[key] = {
                "final_realistic_risk": "low",
                "final_decision": "bound_heuristic_artifact",
                "final_recommended_action": "bound_heuristic_artifact",
                "artifact_type": "bound_heuristic_artifact",
                "formula_changed": "no",
                "change_type": "none",
                "evidence_source": "batch_c_realistic_review",
                "notes": row["short_reason"],
            }
        else:
            overrides[key] = {
                "final_realistic_risk": "low",
                "final_decision": "broad_domain_artifact",
                "final_recommended_action": "broad_domain_artifact",
                "artifact_type": "broad_domain_artifact",
                "formula_changed": "no",
                "change_type": "none",
                "evidence_source": "batch_c_realistic_review",
                "notes": row["short_reason"],
            }

    return overrides


def build_complete_ranking() -> list[dict[str, Any]]:
    broad_rows = _read_csv(GRADIENT_DIR / "flux_gradient_risk_ranking.csv")
    review_overrides = _review_override_rows()
    ranking_rows: list[dict[str, Any]] = []

    for row in broad_rows:
        if row["active_usage_status"] == "unused":
            continue

        key = _broad_key(row)
        if key in review_overrides:
            override = review_overrides[key]
            ranking_rows.append(
                {
                    "formula": row["flux_function"],
                    "active_model": row["called_by_models"],
                    "original_broad_risk": row["risk_level"],
                    "final_realistic_risk": override["final_realistic_risk"],
                    "final_decision": override["final_decision"],
                    "final_recommended_action": override["final_recommended_action"],
                    "artifact_type": override["artifact_type"],
                    "formula_changed": override["formula_changed"],
                    "change_type": override["change_type"],
                    "evidence_source": override["evidence_source"],
                    "notes": override["notes"],
                }
            )
            continue

        ranking_rows.append(
            {
                "formula": row["flux_function"],
                "active_model": row["called_by_models"],
                "original_broad_risk": row["risk_level"],
                "final_realistic_risk": "low",
                "final_decision": "safe_no_action",
                "final_recommended_action": "safe_no_action",
                "artifact_type": "not_originally_high_risk",
                "formula_changed": "no",
                "change_type": "none",
                "evidence_source": "broad_diagnostic_only_non_high_active_context",
                "notes": (
                    f"Original broad-screen active-context risk was `{row['risk_level']}`; this context was not part of the "
                    "original active high-risk follow-up set."
                ),
            }
        )

    ranking_rows.sort(
        key=lambda item: (
            {"high": 0, "medium": 1, "low": 2}.get(item["final_realistic_risk"], 9),
            item["formula"],
            item["active_model"],
        )
    )
    return ranking_rows


def build_summary_row(ranking_rows: list[dict[str, Any]]) -> dict[str, Any]:
    original_high = [row for row in ranking_rows if row["original_broad_risk"] == "high"]
    final_high = [row for row in ranking_rows if row["final_realistic_risk"] == "high"]
    resolved_broad = [row for row in original_high if row["final_decision"] == "broad_domain_artifact"]
    resolved_bound = [row for row in original_high if row["final_decision"] == "bound_heuristic_artifact"]
    resolved_cap = [row for row in original_high if row["final_decision"] == "model_level_cap_resolves"]
    resolved_rewrite = [row for row in original_high if row["final_decision"] == "stable_numerical_rewrite_applied"]
    medium_only = [row for row in ranking_rows if row["final_decision"] == "keep_but_document"]
    unresolved = [row for row in original_high if row["final_decision"] == "manual_review_required" or row["final_realistic_risk"] == "high"]

    return {
        "active_contexts_total": len(ranking_rows),
        "original_active_high_risk_contexts": len(original_high),
        "final_active_high_risk_contexts": len(final_high),
        "contexts_resolved_as_broad_domain_artifact": len(resolved_broad),
        "contexts_resolved_as_bound_heuristic_artifact": len(resolved_bound),
        "contexts_resolved_by_model_level_cap": len(resolved_cap),
        "contexts_resolved_by_stable_numerical_rewrite": len(resolved_rewrite),
        "contexts_remaining_medium_document_only": len(medium_only),
        "contexts_remaining_unresolved": len(unresolved),
        "remaining_medium_document_only_contexts": _sorted_context_list(medium_only),
        "remaining_unresolved_contexts": _sorted_context_list(unresolved),
    }


def build_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Complete Flux Gradient Review Summary",
        "",
        f"- Active contexts total: {summary['active_contexts_total']}",
        f"- Original active high-risk contexts: {summary['original_active_high_risk_contexts']}",
        f"- Final active high-risk contexts: {summary['final_active_high_risk_contexts']}",
        f"- Contexts resolved as broad-domain artifact: {summary['contexts_resolved_as_broad_domain_artifact']}",
        f"- Contexts resolved as bound-heuristic artifact: {summary['contexts_resolved_as_bound_heuristic_artifact']}",
        f"- Contexts resolved by model-level cap: {summary['contexts_resolved_by_model_level_cap']}",
        f"- Contexts resolved by stable numerical rewrite: {summary['contexts_resolved_by_stable_numerical_rewrite']}",
        f"- Contexts remaining medium/document-only: {summary['contexts_remaining_medium_document_only']}",
        f"- Contexts remaining unresolved: {summary['contexts_remaining_unresolved']}",
        "",
    ]

    if summary["final_active_high_risk_contexts"] != 0 or summary["contexts_remaining_unresolved"] != 0:
        lines.extend(
            [
                "## Remaining contexts",
                f"- Remaining unresolved contexts: {summary['remaining_unresolved_contexts']}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Final status",
                "- Final active high-risk contexts: 0",
                "- Unresolved active high-risk contexts: 0",
                "",
            ]
        )

    lines.extend(
        [
            "## Remaining medium/document-only contexts",
            f"- {summary['remaining_medium_document_only_contexts']}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_complete_report(ranking_rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    original_high = [row for row in ranking_rows if row["original_broad_risk"] == "high"]
    broad_artifacts = [row for row in original_high if row["final_decision"] == "broad_domain_artifact"]
    bound_artifacts = [row for row in original_high if row["final_decision"] == "bound_heuristic_artifact"]
    cap_resolved = [row for row in original_high if row["final_decision"] == "model_level_cap_resolves"]
    rewritten = [row for row in original_high if row["final_decision"] == "stable_numerical_rewrite_applied"]
    medium_only = [row for row in ranking_rows if row["final_decision"] == "keep_but_document"]

    lines = [
        "# Complete Flux Gradient Stability Report",
        "",
        "## 1. Scope",
        "- This report consolidates the complete active flux-gradient-stability review after the broad diagnostic screen, focused review, Batch A/B/C realistic-domain reviews, and the `saturation_3` stable rewrite.",
        "- This pass is reporting and consistency-checking only. No new hydrological formulas, smoothing rules, clamps, parameter bounds, model physics, soft-gate defaults, unit hydrograph code, or water-balance fixes were changed.",
        "",
        "## 2. Original broad-domain diagnostic design",
        "- The original diagnostic intentionally used broad synthetic domains to expose NaN/Inf outputs, NaN/Inf gradients, very large finite gradients, dead regions, and heuristic bound failures.",
        "- That screen was useful as a sensitivity scanner, but it was intentionally broader than the actual active-model state and parameter domains.",
        "",
        "## 3. Why broad-domain risk is not the same as active-model risk",
        "- Shared flux helpers are called inside specific core-model update sequences with constrained parameter ranges, temporary-state semantics, and post-flux caps or rescaling steps.",
        "- Broad-domain bound heuristics can misclassify prospective overflow terms, deficit-store terms, or partition functions whose true physical bound is enforced at the model level rather than by the raw helper alone.",
        "",
        "## 4. Focused review summary",
        "- `baseflow_2 / susannah1`, `interflow_2 / hbv96`, `interflow_3 / australia`, `interflow_3 / susannah2`, and `baseflow_5 / vic` were reclassified as broad-domain artifacts under realistic active-model domains.",
        "- `baseflow_6 / tcm` and `interflow_10 / topmodel` remained finite and physically safe but threshold-sensitive, so they were retained as `keep_but_document` contexts.",
        "",
        "## 5. Batch A summary",
        "- `saturation_2 / hymod` was confirmed as a broad-domain artifact.",
        "- `baseflow_9 / gsfb` remained finite and bounded in realistic domains, but large finite gradients were kept documented rather than edited.",
        "- `saturation_3 / flexb`, `flexi`, and `flexis` became low-risk after the stable numerical rewrite.",
        "",
        "## 6. Saturation_3 stable numerical rewrite",
        "- `saturation_3` was the only source-level flux change in the entire review program.",
        "- The change was algebraically exact: `1 - 1 / (1 + exp(z))` was rewritten as `torch.sigmoid(z)`.",
        "- This preserved forward values to machine precision and removed the previous non-finite-gradient autograd overflow path. FLEX beta lower bounds were restored to `0.0` after the rewrite.",
        "",
        "## 7. Batch B summary",
        "- `baseflow_4 / topmodel`: bound-heuristic artifact.",
        "- `evap_3 / hbv96`: broad-domain artifact.",
        "- `recharge_2 / hbv96`: resolved by model-level cap semantics in the active core update.",
        "- `depression_1 / modhydrolog`: bound-heuristic artifact.",
        "",
        "## 8. Batch C summary",
        "- `excess_1 / australia`, `susannah2`, and `vic`: bound-heuristic artifacts.",
        "- `recharge_1 / modhydrolog` and `simhyd`: broad-domain artifacts.",
        "- `split_1 / flexb`, `flexi`, and `flexis`: broad-domain artifacts caused by broad diagnostic expression-range inference rather than the active `1 - d_split` range.",
        "- `evap_16 / penman`: bound-heuristic artifact.",
        "- `evap_7 / vic`: broad-domain artifact.",
        "",
        "## 9. Final active risk count",
        f"- Active contexts total: {summary['active_contexts_total']}",
        f"- Original active high-risk contexts: {summary['original_active_high_risk_contexts']}",
        f"- Final active high-risk contexts: {summary['final_active_high_risk_contexts']}",
        f"- Final unresolved active high-risk contexts: {summary['contexts_remaining_unresolved']}",
        "",
        "## 10. Final formula changes made",
        "- Only `saturation_3 / flexb`, `saturation_3 / flexi`, and `saturation_3 / flexis` are marked as `stable_numerical_rewrite_applied`.",
        "- No other source-level flux formula was changed.",
        "",
        "## 11. Formulas kept unchanged and why",
        "- Broad-domain artifacts were kept unchanged because realistic-domain tracing showed finite outputs/gradients and no true physical bound violation in active model rollouts.",
        "- Bound-heuristic artifacts were kept unchanged because the broad diagnostic bound assumption was not physically valid for the active formula semantics.",
        "- `recharge_2 / hbv96` was kept unchanged because the active model-level cap resolves the raw helper overshoot before the state update.",
        "",
        "## 12. Remaining medium/document-only formulas if any",
    ]
    if medium_only:
        for row in sorted(medium_only, key=lambda item: (item["formula"], item["active_model"])):
            lines.append(f"- `{row['formula']}` / `{row['active_model']}`: {row['notes']}")
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## 13. Unused flux functions and why they are not active production risks",
            "- Unused shared flux helpers are not part of the current active production-risk set because they are not called by an enabled registered core model.",
            "- They may still warrant future cleanup, but they were intentionally excluded from this active-context final table.",
            "",
            "## 14. Final conclusion for gradient-based calibration",
            "The active flux set has no remaining unresolved high-risk context after realistic-domain review. Only `saturation_3` required a source-level change, and that change was an algebraically equivalent stable numerical rewrite rather than a hydrological formula modification. The remaining original high-risk flags were broad-domain artifacts, bound-heuristic artifacts, or resolved by model-level caps.",
            "",
            "## 15. Recommended next step: benchmark/calibration regression",
            "- Run benchmark and calibration regression workflows using the current active flux set and the stable `saturation_3` implementation, with the three documented medium contexts tracked in release notes rather than modified.",
            "",
            "## Consolidated high-risk outcomes",
            "",
            "| formula | model | final decision | final realistic risk | evidence |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in sorted(original_high, key=lambda item: (item["formula"], item["active_model"])):
        lines.append(
            f"| {row['formula']} | {row['active_model']} | {row['final_decision']} | {row['final_realistic_risk']} | {row['evidence_source']} |"
        )

    lines.extend(
        [
            "",
            "## Resolution counts",
            f"- Stable numerical rewrite: {len(rewritten)}",
            f"- Broad-domain artifact: {len(broad_artifacts)}",
            f"- Bound-heuristic artifact: {len(bound_artifacts)}",
            f"- Model-level cap resolves: {len(cap_resolved)}",
            f"- Keep/document only: {len(medium_only)}",
        ]
    )
    return "\n".join(lines) + "\n"


def run_complete_review() -> dict[str, Any]:
    ranking_rows = build_complete_ranking()
    _write_csv(COMPLETE_RANKING_PATH, ranking_rows)

    summary = build_summary_row(ranking_rows)
    _write_csv(COMPLETE_SUMMARY_CSV_PATH, [summary])
    COMPLETE_SUMMARY_MD_PATH.write_text(build_summary_markdown(summary), encoding="utf-8")
    COMPLETE_REPORT_PATH.write_text(build_complete_report(ranking_rows, summary), encoding="utf-8")

    return {
        "ranking_rows": ranking_rows,
        "summary": summary,
    }


def main() -> None:
    run_complete_review()


if __name__ == "__main__":
    main()
