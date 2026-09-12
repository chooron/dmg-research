#!/usr/bin/env python3
"""Run rule-based model selection and its two QC plots; never runs OOB or training."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for name in ("select_models.py", "plot_selection_qc.py", "verify_selection.py"):
    print(f"RUN {name}", flush=True)
    subprocess.run([sys.executable, str(HERE / name)], check=True, cwd=HERE.parents[3])
print("OOB_MODEL_SELECTION_COMPLETE_NO_OOB_RUN")
