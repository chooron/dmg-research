#!/usr/bin/env python3
"""Compatibility entry point for the active HBV dPL ablation runner.

The implementation lives in ``training/dpl``; this shim keeps old launch
commands reproducible without treating the historical ablation directory as
the active training module.
"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(
            Path(__file__).resolve().parents[3]
            / "training"
            / "dpl"
            / "run_hbv_window_ablation.py"
        ),
        run_name="__main__",
    )
