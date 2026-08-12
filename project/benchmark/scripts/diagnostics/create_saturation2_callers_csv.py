"""Generate saturation2_callers.csv listing all models that invoke saturation_2."""
from __future__ import annotations
import csv
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/dpl_round13_20260805/vic_saturation_fix"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    callers = [
        {
            "model": "hymod",
            "file": "dmotpy/models/core/hymod.py",
            "line": 87,
            "parameter_corresponding_to_p1": "b_exp",
            "Smax_construction": "smax"
        },
        {
            "model": "xinanjiang",
            "file": "dmotpy/models/core/xinanjiang.py",
            "line": 125,
            "parameter_corresponding_to_p1": "ex",
            "Smax_construction": "smax"
        },
        {
            "model": "xinanjiang",
            "file": "dmotpy/models/core/xinanjiang.py",
            "line": 129,
            "parameter_corresponding_to_p1": "ex",
            "Smax_construction": "smax"
        },
        {
            "model": "xinanjiang",
            "file": "dmotpy/models/core/xinanjiang.py",
            "line": 134,
            "parameter_corresponding_to_p1": "ex",
            "Smax_construction": "smax"
        },
        {
            "model": "vic",
            "file": "dmotpy/models/core/vic.py",
            "line": 129,
            "parameter_corresponding_to_p1": "b",
            "Smax_construction": "smmax = fsm * stot"
        },
        {
            "model": "wetland",
            "file": "dmotpy/models/core/wetland.py",
            "line": 69,
            "parameter_corresponding_to_p1": "betaw",
            "Smax_construction": "swmax"
        },
        {
            "model": "hillslope",
            "file": "dmotpy/models/core/hillslope.py",
            "line": 85,
            "parameter_corresponding_to_p1": "betaw",
            "Smax_construction": "swmax"
        }
    ]

    df = pd.DataFrame(callers)
    df.to_csv(OUT_DIR / "saturation2_callers.csv", index=False)
    print(f"Saved saturation2_callers.csv with {len(callers)} records:")
    print(df.to_string())

if __name__ == "__main__":
    main()
