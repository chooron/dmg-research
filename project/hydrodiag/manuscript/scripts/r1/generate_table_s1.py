#!/usr/bin/env python3
"""Table S1 Generator Script: Model parameter definitions and bounds.

Generates TableS1_parameter_bounds.md and TableS1_parameter_bounds.tex.
Delegates to manuscript.scripts.shared.generate_table_s1_parameter_bounds.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from generate_table_s1_parameter_bounds import main as run_generate_table_s1

if __name__ == "__main__":
    run_generate_table_s1()
