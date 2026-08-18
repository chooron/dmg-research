"""Build the official R4 real-basin soil-state statistics package.

Orchestrates the formal R4 soil-water state consistency evaluation:
1. Verifies/loads the Caravan v1.1 CAMELS-US ERA5-Land soil moisture reference
   (results/r4_caravan_soil_reference_v1/caravan_soil_ensemble.npz)
2. Runs the formal state consistency & timing pipeline on observation-trained
   canonical dPL Base/CN (seeds 42 & 123) and IC fused sensitivity
3. Runs the 4 strict robustness modules (similar-performance subsets,
   controlled regressions, leave-one-region-out, extreme-SWE trimming,
   SWE decile shape audit, 4-phase process breakdown, timing sensitivity)
4. Writes all figure-ready tables to results/r4_phase1_soil_official/

Usage:
    python manuscript/scripts/r4/build_r4_soil_statistics.py [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript.r4.common import default_data_root, default_results_root  # noqa: E402
from manuscript.r4.extract_caravan_soil import STAGE_ROOT, extract_all  # noqa: E402
from manuscript.r4.robustness_analysis import run_all_robustness_checks  # noqa: E402
from manuscript.r4.soil_analysis import run_soil_consistency_analysis  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("R4_Build")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu"
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--force-extract-caravan", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root or default_data_root()
    results_root = args.results_root or default_results_root()

    logger.info("Starting R4 formal soil-state statistics build...")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Results root: {results_root}")
    logger.info(f"Device: {args.device}")

    # 1. Caravan cache check
    caravan_cache = (
        results_root / "r4_caravan_soil_reference_v1" / "caravan_soil_ensemble.npz"
    )
    if not caravan_cache.is_file() or args.force_extract_caravan:
        logger.info(
            f"Caravan cache missing or force re-extraction requested; extracting from {STAGE_ROOT}..."
        )
        extract_all(
            STAGE_ROOT, data_root, results_root / "r4_caravan_soil_reference_v1"
        )
    else:
        logger.info(
            f"Caravan soil cache found: {caravan_cache} ({caravan_cache.stat().st_size / (1024 * 1024):.2f} MB)"
        )

    # 2. Run formal soil consistency analysis
    logger.info("Executing formal Base vs CN soil-state consistency analysis...")
    soil_report = run_soil_consistency_analysis()
    logger.info(
        f"Soil consistency analysis complete. Tag: {soil_report.get('regimes', {}).get('dPL_seed42', {}).get('tag', 'OFFICIAL')}"
    )

    # 3. Run full robustness suite
    logger.info("Executing full suite of 4 robustness modules...")
    robustness_report = run_all_robustness_checks()
    logger.info("Robustness checks complete. Verdict: R4_ROBUSTNESS_COMPLETE")

    # 4. Summary of artifacts
    out_dir = results_root / "r4_phase1_soil_official"
    tables = sorted(p.name for p in out_dir.glob("*.csv"))
    reports = sorted(p.name for p in out_dir.glob("*.json"))

    logger.info(
        f"Successfully generated {len(tables)} figure-ready tables and {len(reports)} JSON reports in {out_dir}/:"
    )
    for t in tables:
        n_rows = len(pd.read_csv(out_dir / t))
        logger.info(f"  [Table] {t:<42s} ({n_rows} rows)")
    for r in reports:
        logger.info(f"  [Report] {r}")

    logger.info(
        "R4 build completed successfully. Ready for manuscript table & figure generation."
    )


if __name__ == "__main__":
    main()
