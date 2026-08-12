#!/usr/bin/env python3
"""Summarize formula audit results into master report."""
import csv, math, sys
from pathlib import Path

import numpy as np

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

def main():
    master = list(csv.DictReader(open(_PROJECT / "validation_results" / "formula_audit_master_summary.csv")))
    print("# Formula Audit Master Report\n")

    for node in ["snow", "recharge", "aet", "response"]:
        rows = [r for r in master if r["node"] == node]
        print(f"## {node}\n")
        for r in rows:
            print(f"- **{r['formula_id']}** ({r['formula_name']}): risk={r['overall_risk']}, "
                  f"action={r['recommended_action']}, scale={r['scale_status']}, water={r['water_constraint_status']}")
        print()

    # Critical findings
    print("## Key Findings\n")

    # R4 analysis
    r4 = [r for r in master if r["formula_id"] == "R4"]
    if r4:
        print(f"### R4 (saturation_threshold_recharge)")
        print(f"- Overall risk: {r4[0]['overall_risk']}")
        print(f"- Scale status: {r4[0]['scale_status']}")
        print(f"- Water constraint: {r4[0]['water_constraint_status']}")
        print(f"- Recommended action: {r4[0]['recommended_action']}")

        # Check pairwise with R0
        pw = list(csv.DictReader(open(_PROJECT / "validation_results/formula_scale_audit_v2/recharge_pairwise_ratios.csv")))
        for p in pw:
            if (p["formula_a"] == "R0" and p["formula_b"] == "R4") or (p["formula_a"] == "R4" and p["formula_b"] == "R0"):
                print(f"- R0 vs R4: max_log10_ratio={p['max_log10_ratio']}, median_log10_ratio={p['median_log10_ratio']}, severity={p['severity']}")
                print()
                if float(p["max_log10_ratio"]) >= 1.5:
                    print("  **WARNING**: R0-R4 have SEVERE/CRITICAL scale mismatch. R4 advantage may be scale artifact.")
                else:
                    print("  R0-R4 scale mismatch is manageable. R4 advantage is likely real.")

    print("\n## Recommended Actions Summary\n")
    for r in master:
        if r["recommended_action"] not in ["KEEP"]:
            print(f"- **{r['node']}/{r['formula_id']}**: {r['recommended_action']} (risk: {r['overall_risk']})")


if __name__ == "__main__":
    main()
