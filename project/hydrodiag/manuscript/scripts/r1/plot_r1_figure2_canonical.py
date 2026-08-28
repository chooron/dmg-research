#!/usr/bin/env python3
"""Canonical R1 Figure 2 renderer wrapper.

This is a plot-only renderer wrapper that delegates to plot_r1_figure2.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HYDRODIAG_ROOT = HERE.parents[2]
if str(HYDRODIAG_ROOT) not in sys.path:
    sys.path.insert(0, str(HYDRODIAG_ROOT))

from manuscript.scripts.r1.plot_r1_figure2 import (
    main as render_figure2,
    PLOTS_FIG_DIR,
)


def render(out_dir: Path | None = None) -> Path:
    return render_figure2(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    print(render(args.out_dir))
