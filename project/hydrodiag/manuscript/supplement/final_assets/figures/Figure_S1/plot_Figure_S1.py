#!/usr/bin/env python3
"""Render final Figure S1 from frozen HUC-2 LORO CSVs.

The source labels HUC_11-HUC_18 are intentionally displayed as HUC_01-HUC_08;
source HUC_01-HUC_10 are the random ten-fold partitions and are excluded.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[5]
SOURCE_SCRIPT = PROJECT / "manuscript" / "scripts" / "supplement" / "plot_huc2_loro_robustness.py"
OUT = Path(__file__).resolve().parent / "Figure_S1.png"


def main() -> None:
    spec = importlib.util.spec_from_file_location("huc2_loro_renderer", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {SOURCE_SCRIPT}")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    expected = [f"HUC_{i:02d}" for i in range(11, 19)]
    displayed = [f"HUC_{i:02d}" for i in range(1, 9)]
    if renderer.REGIONS != expected or renderer.DISPLAY_REGION_LABELS != displayed:
        raise ValueError("HUC-2 source/display mapping is not the frozen HUC_11-HUC_18 to HUC_01-HUC_08 mapping")
    renderer.build_figure(OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
