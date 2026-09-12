#!/usr/bin/env python3
"""Deterministic QC for the MOPEX-redundancy revision and blocker."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results/oob_model_selection_20260901"


def main() -> None:
    original = pd.read_csv(OUT / "OOB_PRIMARY_8_MODEL_TABLE.csv")
    quality = pd.read_csv(OUT / "OOB_SCENARIO_SELECTION_QUALITY.csv")
    prov = json.loads((OUT / "OOB_MOPEX_REVISION_PROVENANCE.json").read_text())
    checks = []
    checks.append({"check": "original selection preserved", "pass": len(original) == 8 and {"mopex2", "mopex4", "mopex5"}.issubset(set(original.model)), "value": original.model.tolist()})
    checks.append({"check": "both scenarios are represented", "pass": len(quality) == 2 and set(quality.scenario) == {"SCENARIO_A_KEEP_MOPEX5", "SCENARIO_B_KEEP_MOPEX4"}, "value": quality.scenario.tolist()})
    for name in quality.scenario:
        row = quality[quality.scenario == name].iloc[0]
        path = OUT / ("OOB_SCENARIO_A_KEEP_MOPEX5.csv" if name.startswith("SCENARIO_A") else "OOB_SCENARIO_B_KEEP_MOPEX4.csv")
        candidate = pd.read_csv(path)
        checks.append({"check": f"{name} candidate", "pass": len(candidate) == 8 and candidate.model.nunique() == 8 and candidate.status.eq("BLOCKED").all() and candidate.model.str.startswith("mopex").sum() == 1 and "mopex2" not in set(candidate.model) and bool(candidate.performance_gate_pass.all()), "value": {"models": candidate.model.tolist(), "failed_gates": row.failed_gates}})
    checks.append({"check": "both scenarios fail only unchanged complexity gate", "pass": bool((quality.selection_status == "BLOCKED").all()) and set(quality.failed_gates) == {"complexity_tertile_coverage"} and bool(quality.complexity_coverage_pass.eq(False).all()), "value": quality[["scenario", "P_low_count", "P_medium_count", "P_high_count", "failed_gates"]].to_dict(orient="records")})
    checks.append({"check": "no adopted revised list", "pass": prov["final_scenario"] == "BLOCKED_NO_FINAL_PRIMARY_8" and prov["primary_8"] == [] and prov["fallback_6"] == [] and (OUT / "OOB_PRIMARY_8_MODELS_REVISED.txt").read_text().strip() == "BLOCKED_NO_FINAL_PRIMARY_8", "value": prov["final_scenario"]})
    checks.append({"check": "MOPEX gate and coverage metadata", "pass": prov["mopex_primary_count"] == 0 and prov["manual_override"] is False and prov["oob_information_used"] is False and prov["training_started"] is False, "value": {"manual_override": prov["manual_override"], "oob_information_used": prov["oob_information_used"], "training_started": prov["training_started"]}})
    checks.append({"check": "feature diversity improves for both legal candidates", "pass": bool((quality.min_pairwise_delta_vs_original > 0).all()) and bool((quality.mean_pairwise_delta_vs_original > 0).all()), "value": quality[["scenario", "min_pairwise_delta_vs_original", "mean_pairwise_delta_vs_original"]].to_dict(orient="records")})
    structure = pd.read_csv(OUT / "OOB_STRUCTURAL_COVERAGE_QC.csv")
    checks.append({"check": "structural auxiliary QC", "pass": set(structure.population) == {"ORIGINAL_PRIMARY_8", "SCENARIO_A_KEEP_MOPEX5_LEGAL_CANDIDATE", "SCENARIO_B_KEEP_MOPEX4_LEGAL_CANDIDATE"} and int(structure.loc[structure.population == "ORIGINAL_PRIMARY_8", "mopex_count"].iloc[0]) == 3 and bool((structure.loc[structure.population != "ORIGINAL_PRIMARY_8", "mopex_count"] == 1).all()), "value": structure[["population", "mopex_count", "state_count_unique", "routing_forms"]].to_dict(orient="records")})
    forbidden = {"backward", "step", "optimizer", "train", "zero_grad"}
    called = {}
    for path in HERE.glob("*.py"):
        tree = ast.parse(path.read_text())
        names = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))}
        called[path.name] = sorted(names & forbidden)
    checks.append({"check": "static no-training calls", "pass": all(not value for value in called.values()), "value": called})
    result = {"qc": "PASS" if all(item["pass"] for item in checks) else "FAIL", "checks": checks, "blocker": "both legal scenarios fail parameter-complexity tertile coverage under same-quadrant performance-valid replacements"}
    (OUT / "OOB_MOPEX_REVISION_QC.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    (OUT / "OOB_MOPEX_REVISION_QC.md").write_text(f"# MOPEX revision QC\n\n**{result['qc']}**\n\nBoth legal scenarios were checked for 8-model/quadrant structure, exactly one MOPEX member, no mopex2, performance validity, unchanged A/P gates, original-list preservation, feature-space diversity, structural descriptors, and no training-like calls. Both are blocked only by the unchanged parameter-complexity tertile requirement.\n")
    print("OOB_MOPEX_REVISION_QC", result["qc"], "checks", len(checks))


if __name__ == "__main__":
    main()
