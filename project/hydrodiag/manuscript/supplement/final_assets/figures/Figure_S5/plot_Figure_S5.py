#!/usr/bin/env python3
"""Stage the frozen R4 external-state figure under final Figure S5 numbering."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[5]
SOURCE = PROJECT / "manuscript" / "supplement" / "figures" / "FigureS1_R4_multibasin_validation.png"
AUDIT_JSON = PROJECT / "manuscript" / "supplement" / "figures" / "FigureS1_R4_selection_audit.json"
AUDIT_CSV = PROJECT / "manuscript" / "supplement" / "figures" / "FigureS1_R4_population_audit.csv"
OUT = Path(__file__).resolve().parent / "Figure_S5.png"


def main() -> None:
    if not SOURCE.exists() or not AUDIT_JSON.exists() or not AUDIT_CSV.exists():
        raise FileNotFoundError("Frozen R4 figure or its selection audit is missing")
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    examples = audit.get("selected_example_basins", [])
    if len(examples) != 6 or sorted(e["group"] for e in examples) != ["High", "High", "Low", "Low", "Middle", "Middle"]:
        raise ValueError("Unexpected external-information example selection")
    shutil.copy2(SOURCE, OUT)
    print(f"Copied frozen R4 external-state asset to {OUT}")


if __name__ == "__main__":
    main()
