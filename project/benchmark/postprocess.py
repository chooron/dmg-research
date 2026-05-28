#!/usr/bin/env python
"""Top-level postprocess entry point for dual-evidence benchmark.

Usage:
    python postprocess.py --output-dir outputs
    python postprocess.py --output-dir outputs --calib-kge-threshold 0.5 --pl-kge-threshold 0.4 --top-n 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from benchmark root
sys.path.insert(0, str(Path(__file__).parent))

from benchmark.postprocess import run_postprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs",
                        help="Root outputs directory (default: outputs)")
    parser.add_argument("--calib-kge-threshold", type=float, default=0.5,
                        help="KGE threshold for high calibration evidence (default: 0.5)")
    parser.add_argument("--pl-kge-threshold", type=float, default=0.4,
                        help="KGE threshold for high param-learning evidence (default: 0.4)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top models for comparison table (default: 5)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    args = parse_args()
    saved = run_postprocess(
        output_dir=args.output_dir,
        calib_kge_threshold=args.calib_kge_threshold,
        pl_kge_threshold=args.pl_kge_threshold,
        top_n=args.top_n,
    )
    print("\nSaved tables:")
    for name, path in saved.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
