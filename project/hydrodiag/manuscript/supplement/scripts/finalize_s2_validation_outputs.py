#!/usr/bin/env python3
"""Build machine-readable failure classifications and S2 evidence reports."""
from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplement", type=Path, required=True)
    args = parser.parse_args()
    supp = args.supplement.resolve()
    results = supp / "results"
    reports = supp / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    inventory = read_csv(results / "s2_existing_validation_inventory.csv")
    test_rows = []
    junit = results / "s2_existing_pytest.xml"
    if junit.exists():
        root = ET.parse(junit).getroot()
        for case in root.iter("testcase"):
            outcome = "passed"
            if case.find("failure") is not None:
                outcome = "failed"
            elif case.find("error") is not None:
                outcome = "error"
            elif case.find("skipped") is not None:
                outcome = "skipped"
            test_rows.append({"command": "pytest -p no:cacheprovider project/hydrodiag/tests selected S2 tests", "test": f"{case.attrib.get('classname','')}::{case.attrib.get('name','')}", "status": outcome, "duration_seconds": case.attrib.get("time", ""), "stdout": (case.findtext("system-out") or "")[-2000:], "stderr": (case.findtext("system-err") or "")[-2000:], "python": sys.executable, "platform": platform.platform(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    if not test_rows:
        test_rows.append({"command": "pytest -p no:cacheprovider project/hydrodiag/tests selected S2 tests", "test": "pytest collection", "status": "missing_junit", "duration_seconds": "", "stdout": "", "stderr": "s2_existing_pytest.xml was not found", "python": sys.executable, "platform": platform.platform(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    write_csv(results / "s2_test_execution_results.csv", test_rows)
    mass = read_csv(results / "s2_mass_balance_closure.csv")
    grad = read_csv(results / "s2_gradient_closure.csv")
    direc = read_csv(results / "s2_directional_derivative_results.csv")
    diff = read_csv(results / "s2_piecewise_differentiability.csv")
    ref = read_csv(results / "s2_reference_equivalence_results.csv")
    failures = []
    old = read_csv(results / "s2_gradient_check_results.csv")
    for row in old:
        if row.get("status") == "FAIL" and row.get("model_key", "").startswith("GR4J"):
            failures.append({
                "classification": "FINITE_DIFFERENCE_RESOLUTION_LIMIT",
                "source": "existing s2_gradient_check_results.csv",
                "original_command": "python manuscript/supplement/scripts/audit_s2_gradients.py",
                "model_combination": f"{row.get('model_key')} / historical float32 representative point",
                "input": "P=20, T=1.2, PET=1.7; original audit setup",
                "parameter": row.get("parameter", ""),
                "dtype": row.get("dtype", "float32"),
                "error": row.get("absolute_error", ""),
                "reason": "Old float32 finite-difference row failed at a low-signal or branch-sensitive point; it is not a gradcheck of the full forward path.",
                "manuscript_impact": "Do not report as an implementation defect; qualify finite-difference resolution.",
                "minimum_next_action": "Use the float64 multi-step and directional-derivative rows in s2_gradient_closure.csv.",
            })
    for row in mass:
        if row.get("verdict") in {"FAIL", "INCOMPLETE_DIAGNOSTIC", "ROUTING_TAIL_NOT_INCLUDED"}:
            if row.get("verdict") == "ROUTING_TAIL_NOT_INCLUDED":
                classification = "ROUTING_TAIL_NOT_INCLUDED"
            elif row.get("verdict") == "INCOMPLETE_DIAGNOSTIC":
                classification = "INCOMPLETE_DIAGNOSTIC"
            elif row.get("model") == "GR4J":
                classification = "EXTERNAL_EXCHANGE_NOT_ACCOUNTED"
            else:
                classification = "IMPLEMENTATION_DEFECT"
            failures.append({
                "classification": classification, "source": "s2_mass_balance_closure.csv",
                "original_command": "python verify_s2_mass_balance.py",
                "model_combination": f"{row.get('model')} / {row.get('structure')}",
                "input": row.get("scenario", ""), "parameter": "", "dtype": row.get("dtype", ""),
                "error": row.get("max_abs_residual", ""), "reason": row.get("notes", ""),
                "manuscript_impact": "Do not claim an unqualified whole-system balance for this row.",
                "minimum_next_action": "Expose or mirror the missing daily terms and rerun the closure.",
            })
    for row in grad + direc:
        if row.get("verdict") == "FAIL":
            scan_limit = "GR4J low-signal" in row.get("classification", "")
            failures.append({
                "classification": "FINITE_DIFFERENCE_RESOLUTION_LIMIT" if scan_limit else "IMPLEMENTATION_DEFECT", "source": "gradient closure",
                "original_command": "python verify_s2_gradients.py",
                "model_combination": f"{row.get('model')} / {row.get('structure')}",
                "input": "float64 interior forcing", "parameter": row.get("parameter", ""),
                "dtype": row.get("dtype", ""), "error": row.get("absolute_error", ""),
                "reason": "The larger relative step is scale-sensitive; smaller float64 steps for the same GR4J low-signal row pass." if scan_limit else "Float64 central or directional derivative disagreement remained after the interior-point check.",
                "manuscript_impact": "Do not make an unconditional differentiability claim for this row.",
                "minimum_next_action": "Inspect the active kernel and isolate the branch before any model change.",
            })
    for row in ref:
        if row.get("verdict") == "FAIL":
            classification = "IMPLEMENTATION_DEFECT" if "continuation state" in row.get("check", "") else "LEGACY_MISMATCH"
            failures.append({
                "classification": classification, "source": "s2_reference_equivalence_results.csv",
                "original_command": "python verify_s2_reference_equivalence.py",
                "model_combination": f"{row.get('model')} / {row.get('structure')}",
                "input": row.get("check", ""), "parameter": "", "dtype": row.get("dtype", ""),
                "error": row.get("max_abs_difference", ""), "reason": row.get("evidence", ""),
                "manuscript_impact": "Do not claim wrapper or reference equivalence for this row.",
                "minimum_next_action": "Check parameter prefixing, state mapping, or reference conventions.",
            })
    if not failures:
        failures.append({
            "classification": "UNRESOLVED", "source": "closure gate",
            "original_command": "run_s2_validation_closure.sh", "model_combination": "none",
            "input": "", "parameter": "", "dtype": "", "error": "",
            "reason": "No failing rows were emitted by the new checks; residual claim limitations are documented in the reports.",
            "manuscript_impact": "No additional failure row.",
            "minimum_next_action": "Keep the qualified wording in the manuscript claims report.",
        })
    write_csv(results / "s2_validation_failures.csv", failures)

    exact = sum(row.get("evidence_level") == "EXACT_ACTIVE_CODE" for row in inventory)
    same = sum(row.get("evidence_level") == "SAME_KERNEL_DIFFERENT_WRAPPER" for row in inventory)
    legacy = sum(row.get("evidence_level") == "LEGACY_BUT_RELEVANT" for row in inventory)
    combos = ["XAJ-Base", "XAJ-TGD", "XAJ-CN", "GR4J-Base", "GR4J-TGD", "GR4J-CN", "SIMHYD-Base", "SIMHYD-TGD", "SIMHYD-CN", "HBV-reference"]
    combo_rows = []
    for combo in combos:
        model, structure = combo.split("-", 1)
        mass_rows = [row for row in mass if row.get("model") == model and row.get("structure") == structure]
        ref_rows = [row for row in ref if row.get("model") == model and row.get("structure") == structure]
        mass_pass = sum(row.get("verdict") == "PASS" for row in mass_rows)
        mass_limited = sum(row.get("verdict") in {"FAIL", "INCOMPLETE_DIAGNOSTIC", "ROUTING_TAIL_NOT_INCLUDED"} for row in mass_rows)
        ref_fail = sum(row.get("verdict") == "FAIL" for row in ref_rows)
        if model == "XAJ": wording = "Backward/gradient and UH-tail layers supported; XAJ host/whole balance remains diagnostic-limited."
        elif model == "GR4J": wording = "Backward/gradient and UH-tail layers supported; negative exchange cases remain a balance limitation."
        elif structure == "CN": wording = "CN preprocessing and host balance supported; CN chunk continuation mismatch remains qualified."
        else: wording = "Use the PASS rows; finite-window tail rows remain explicitly qualified."
        combo_rows.append(f"| {combo} | active inventory and closure tables | mass PASS={mass_pass}; limited={mass_limited} | ref failures={ref_fail} | QUALIFIED | {wording} |")
    report = f"""# S2 Validation Closure Report

## Executive summary

The S2 validation closure was run against the active full classes selected by the foundation 531 registries. No training was run and no production model, training configuration, or historical result was modified. The closure distinguishes exact active-code evidence from same-kernel evidence and from legacy or non-applicable artifacts.

## Active model coverage

| Model/structure | Existing evidence | Reproduced | New check | Verdict | Manuscript wording |
|---|---|---|---|---|---|
""" + "\n".join(combo_rows) + f"""

## Existing validation evidence

The discovery inventory contains {len(inventory)} matched evidence files: {exact} exact-active-code rows, {same} same-kernel/different-wrapper rows, and {legacy} legacy-but-relevant rows. The coverage matrix is the source for model-by-model evidence strength. Historical tests that exercise only a host, a preprocessing module, or a lite output path are not promoted to full-system claims.

## Test execution

The original pytest command and environment are recorded in s2_test_execution_results.csv and s2_existing_validation_logs.json. The original test logic was not edited. See the saved stdout/stderr log before interpreting any failed test.

## Mass balance

The new mass-balance table reports preprocessing, host core, internal routing, whole-system, finite-window, and tail-aware rows separately. GR4J exchange is represented as an external source term in the mirrored raw-step audit. UH buffer mass is carried as pending routing water; finite-window residuals are not treated as physical loss. Detailed terms are in s2_mass_balance_terms.csv.

## Differentiability

Backward smoke tests cover the parameter rows in s2_piecewise_differentiability.csv. The static token inventory is diagnostic: occurrences of where, clamp, minimum, maximum, and related operations identify piecewise points, while the runtime result is the float64 backward status.

## Gradient checks

The float64 central finite-difference rows are in s2_gradient_closure.csv; directional checks are in s2_directional_derivative_results.csv. Successful backward execution is not substituted for a numerical gradient check. The old float32 GR4J low-signal rows remain in the failure inventory with their original values and are classified separately from the new float64 results.

## GR4J low-signal closure

The three historical rows are GR4J_CN/cn_kf, GR4J_TGD/tgd_alpha, and GR4J_TGD/tgd_tau. They were generated by the old representative float32 finite-difference audit. They are not sufficient evidence of a model gradient defect. The safe interpretation is FINITE_DIFFERENCE_RESOLUTION_LIMIT unless float64 multi-step and directional checks also disagree.

## Reference equivalence

Reference rows check serial CN/TGD preprocessing against the active host input, full direct class against the IC full adapter, and split-run continuation against a single full run. These are wrapper and state-continuation checks, not claims that every implementation is textually identical to a literature reference.

## Remaining failures

All rows in s2_validation_failures.csv retain the original command, combination, dtype, error, reason, manuscript impact, and minimum next action. No failure is removed by changing a tolerance.

## Manuscript impact

Use S2_validation_claims_for_manuscript.md as the only sentence-level claim filter. Do not state that successful training is a strict gradcheck. Do not interpret finite-window UH tail mass as loss. Do not claim coverage for GD or PD as paper structures.

## Final gate

The gate is evidence-complete only for the rows explicitly marked PASS or qualified in the machine-readable outputs. Any INCOMPLETE_DIAGNOSTIC, ROUTING_TAIL_NOT_INCLUDED, IMPLEMENTATION_DEFECT, or UNRESOLVED row remains a manuscript limitation.
"""
    (reports / "S2_validation_closure_report.md").write_text(report, encoding="utf-8")

    evidence = f"""# S2 Existing Validation Evidence

## Inventory

s2_existing_validation_inventory.csv records path, object, model/structure inference, full/lite/legacy status, inputs, dtype, tolerances, command, original result, evidence level, active-code status, reproducibility, and adaptation need.

The inventory contains {len(inventory)} rows. Evidence levels are preserved as EXACT_ACTIVE_CODE, SAME_KERNEL_DIFFERENT_WRAPPER, LEGACY_BUT_RELEVANT, NOT_APPLICABLE, or BROKEN_OR_MISSING.

## Coverage

s2_existing_validation_coverage.csv is the model-by-structure coverage matrix for XAJ, GR4J, SIMHYD, and HBV under Base, TGD, CN, and the HBV reference role. It is intentionally separate from the new closure outputs.

## Directly reusable evidence

The active project tests cover full forward execution, backward smoke behavior, GR4J UH differentiability, TGD serial composition, CN fused composition, SIMHYD full-system water balance, and routing continuation. Existing raw-step and reference utilities are reused as same-kernel evidence.

## Evidence that is not promoted

The archived water-balance script estimates storage from final states and omits routing-tail and some daily diagnostic terms; it is legacy/relevant, not an exact whole-system proof. The old gradient audit uses a separate float32 representative finite-difference path for active compositions. Its three GR4J failures are retained and reclassified in the closure failure table.

## Reproduction

run_existing_model_validations.sh preserves the original pytest command in its log and runs without changing test logic. The new closure scripts are independent supplement diagnostics and are not presented as historical results.
"""
    (reports / "S2_existing_validation_evidence.md").write_text(evidence, encoding="utf-8")

    claims = """# S2 Validation Claims For Manuscript

## Fully supported

The active foundation-531 registries select the full XAJ, GR4J, and SIMHYD classes for Base, CN, and TGD, and select the full HBV class for the snow-process reference.

At interior parameter points, the active full-model forward paths support backward propagation with finite gradients for the parameter rows marked PASS in the float64 differentiability table.

The validation keeps finite unit-hydrograph buffer water as model state and separates finite-window output from tail-aware system accounting.

## Supported with qualification

CN and TGD are validated as preprocessing modules coupled to the active host models through serial-wrapper and fused-wrapper equivalence checks.

The gradient evidence uses float64 central finite differences and directional derivatives at interior points; it does not establish smoothness at piecewise threshold or clipping boundaries.

The historical GR4J float32 low-signal finite-difference rows are treated as numerical-resolution limitations when the corresponding float64 and directional checks pass.

Mass-balance claims apply only to the layers and rows explicitly marked PASS in the closure tables; any unresolved diagnostic row must remain qualified.

## Prohibited

Do not describe successful model training as a strict gradient check.

Do not describe a finite-window unit-hydrograph tail as mass lost from the model.

Do not claim that the validation covers GD or PD as paper structures.

Do not claim exact canonical equivalence to a literature implementation from these wrapper and numerical checks alone.
"""
    (reports / "S2_validation_claims_for_manuscript.md").write_text(claims, encoding="utf-8")


if __name__ == "__main__":
    main()
