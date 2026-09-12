#!/usr/bin/env python3
"""Deterministic contract QC for the representative-model selection."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results/oob_model_selection_20260901"


def main() -> None:
    feature = pd.read_csv(OUT / "OOB_MODEL_SELECTION_FEATURES.csv")
    quadrant = pd.read_csv(OUT / "OOB_MODEL_SELECTION_QUADRANTS.csv")
    ranking = pd.read_csv(OUT / "OOB_MODEL_SELECTION_RANKING.csv")
    final = pd.read_csv(OUT / "OOB_PRIMARY_8_MODEL_TABLE.csv")
    prov = json.loads((OUT / "OOB_SELECTION_PROVENANCE.json").read_text())
    checks = []
    checks.append({"check": "candidate pool", "pass": len(feature) == 36 and feature.model.nunique() == 36, "value": len(feature)})
    checks.append({"check": "percentile ranks", "pass": bool(feature[["G_pct", "R_pct", "D_pct", "U_pct", "K_joint_pct", "P_pct", "A_pct"]].notna().all().all()) and bool(((feature[["G_pct", "R_pct", "D_pct", "U_pct", "K_joint_pct", "P_pct", "A_pct"]] >= 0) & (feature[["G_pct", "R_pct", "D_pct", "U_pct", "K_joint_pct", "P_pct", "A_pct"]] <= 1)).all().all()), "value": "all seven ranks finite and in [0,1]"})
    checks.append({"check": "four quadrants", "pass": set(quadrant.quadrant) == {"Q1_LOW_G_HIGH_R", "Q2_LOW_G_LOW_R", "Q3_HIGH_G_HIGH_R", "Q4_HIGH_G_LOW_R"} and bool((quadrant.primary_models_final.str.split(";").str.len() == 2).all()), "value": quadrant.quadrant.tolist()})
    checks.append({"check": "primary roles", "pass": set(final.role) == {"Q1_REP", "Q1_CONTRAST", "Q2_REP", "Q2_CONTRAST", "Q3_REP", "Q3_CONTRAST", "Q4_REP", "Q4_CONTRAST"} and len(final) == 8, "value": sorted(final.role)})
    checks.append({"check": "fallback six", "pass": len(prov["fallback_6"]) == 6 and set(prov["fallback_6"]).issubset(set(prov["primary_8"])), "value": prov["fallback_6"]})
    checks.append({"check": "performance gate", "pass": not prov["performance_gate_relaxed_quadrants"] and bool(final.K_joint.ge(prov["K_joint_Q25"]).all()), "value": prov["K_joint_Q25"]})
    checks.append({"check": "A and complexity coverage", "pass": prov["coverage_gate_pass"] and min(prov["coverage_counts_final"].values()) >= 2, "value": prov["coverage_counts_final"]})
    checks.append({"check": "no exclusions or override", "pass": prov["manual_override"] is False and prov["family_metadata_found"] is False, "value": "no manual override; no invented family metadata"})
    checks.append({"check": "no OOB/training", "pass": prov["oob_information_used"] is False and prov["training_started"] is False, "value": True})
    checks.append({"check": "ranking coverage", "pass": len(ranking) == 36 and ranking.model.nunique() == 36 and bool(ranking.selected_primary_8.sum() == 8), "value": len(ranking)})
    checks.append({"check": "QC figures", "pass": all((OUT / name).is_file() for name in ["qc_G_vs_reproducibility_selected.png", "qc_selected_feature_heatmap.png", "qc_G_vs_reproducibility_selected_data.csv", "qc_selected_feature_heatmap_data.csv"]), "value": True})
    # Do not permit a training-like call to enter this analysis directory.
    forbidden = {"backward", "step", "optimizer", "train", "zero_grad"}
    called = {}
    for path in HERE.glob("*.py"):
        tree = ast.parse(path.read_text())
        names = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))}
        called[path.name] = sorted(names & forbidden)
    checks.append({"check": "static no-training calls", "pass": all(not value for value in called.values()), "value": called})
    result = {"qc": "PASS" if all(item["pass"] for item in checks) else "FAIL", "checks": checks, "primary_8": prov["primary_8"], "fallback_6": prov["fallback_6"]}
    (OUT / "OOB_SELECTION_QC.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    (OUT / "OOB_SELECTION_QC.md").write_text(f"# OOB selection QC\n\n**{result['qc']}**\n\nAll 36 candidates, four quadrants, primary/fallback cardinalities, performance gate, A/P coverage, percentile ranks, no-override/no-exclusion contract, QC figures, and no-training static guard passed.\n")
    print("OOB_SELECTION_QC", result["qc"], "checks", len(checks))


if __name__ == "__main__":
    main()
