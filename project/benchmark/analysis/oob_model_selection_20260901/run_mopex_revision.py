#!/usr/bin/env python3
"""Run the MOPEX redundancy revision and its blocker QC only."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for name in ("revise_mopex_redundancy.py", "verify_mopex_revision.py"):
    print(f"RUN {name}", flush=True)
    subprocess.run([sys.executable, str(HERE / name)], check=True, cwd=HERE.parents[3])
print("OOB_MOPEX_REVISION_COMPLETE_NO_OOB_RUN")
