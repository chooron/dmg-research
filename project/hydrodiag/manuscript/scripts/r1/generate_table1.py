#!/usr/bin/env python3
"""Table 1 Generator Script: Structural configurations and diagnostic roles.

Generates Table1_structural_configurations.md and Table1_structural_configurations.tex.
Delegates to manuscript.scripts.shared.generate_table1_structural_configurations.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from generate_table1_structural_configurations import main as run_generate_table1

if __name__ == "__main__":
    run_generate_table1()
