#!/usr/bin/env python3
"""Run the remaining frozen seen-basin analyses serially."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for name in [
    "agent_a_gap_and_admissibility.py",
    "agent_b_model_place.py",
    "agent_c_distance_restart.py",
    "agent_d_reliability_confound.py",
    "coordinator_sensitivity_and_master.py",
    "coordinator_figures.py",
    "coordinator_final_qc.py",
]:
    print(f"RUN {name}", flush=True)
    subprocess.run([sys.executable, str(HERE / name)], check=True, cwd=HERE.parents[3])
print("ALL_SEENBASIN_REMAINING_ANALYSES_COMPLETE")
