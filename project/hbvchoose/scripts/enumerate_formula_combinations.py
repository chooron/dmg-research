#!/usr/bin/env python3
"""Enumerate all main-pool formula combinations from FORMULA_REGISTRY."""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

from model.formula_pool import CandidateFormulaPool

OUTPUT_DIR = _PROJECT / "validation_results" / "formula_combination_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NODES = ["snow", "recharge", "aet", "response"]


def main():
    pool = CandidateFormulaPool()
    node_formulas = {n: pool.formulas(n, "main") for n in NODES}
    policies = {n: pool.routing_policy(n) for n in NODES}

    print("Formula counts per node:")
    for n in NODES:
        print(f"  {n}: {len(node_formulas[n])} -> {node_formulas[n]}")

    total = 1
    for n in NODES:
        total *= len(node_formulas[n])
    print(f"Total combinations: {total}")

    combos = []
    for idx, combo in enumerate(itertools.product(*[node_formulas[n] for n in NODES])):
        combo_dict = dict(zip(NODES, combo))
        combo_id = "_".join(combo_dict[n] for n in NODES)
        is_default = combo_id == "S0_R0_E0_Q0"
        combos.append({
            "combo_id": combo_id,
            "index": idx,
            **{f"{n}_id": combo_dict[n] for n in NODES},
            **{f"routing_policy_{n}": policies[n] for n in NODES},
            "is_default_hbv": is_default,
        })

    path = OUTPUT_DIR / "formula_combinations.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(combos[0].keys()))
        w.writeheader()
        w.writerows(combos)

    print(f"Combinations written to {path}")
    print(f"Default HBV combo: {'FOUND' if any(c['is_default_hbv'] for c in combos) else 'MISSING'}")


if __name__ == "__main__":
    main()
