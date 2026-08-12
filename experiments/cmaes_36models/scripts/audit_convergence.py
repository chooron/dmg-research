from __future__ import annotations
from _common import EXPERIMENT
from pathlib import Path


def main() -> int:
    missing=[]
    for path in (EXPERIMENT/"results/unit_convergence.parquet", EXPERIMENT/"results/escalation_history.parquet"):
        if not path.exists(): missing.append(str(path.relative_to(EXPERIMENT)))
    print("no completed production convergence artifacts" if missing else "production artifacts found")
    if missing: print("missing: " + ", ".join(missing))

if __name__ == "__main__": main()
