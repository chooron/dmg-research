#!/usr/bin/env python3
"""R3 main-text summary table generator (Table 2).

Generates Table2_controlled_recovery.md and Table2_controlled_recovery.tex.
Delegates to manuscript.scripts.shared.generate_table2_controlled_recovery.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from generate_table2_controlled_recovery import main as run_generate_table2

if __name__ == "__main__":
    run_generate_table2()
