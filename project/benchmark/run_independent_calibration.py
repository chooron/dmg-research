#!/usr/bin/env python
"""Run one independent calibration benchmark task."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from benchmark.config import load_benchmark_config
from benchmark.independent_calibration import run_independent_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="project/benchmark/conf/benchmark.yaml")
    parser.add_argument("--basin-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--objective", required=True,
                        choices=["nse", "log_nse", "logNSE", "log-transformed-nse",
                                 "NSE", "LOG_NSE", "KGE", "KGE_LOG"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-starts", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    args = parse_args()
    config = load_benchmark_config(args.config)
    if args.num_starts is not None:
        config["calibration"]["num_random_starts"] = args.num_starts
    if args.epochs is not None:
        config["calibration"]["epochs"] = args.epochs
    if args.device is not None:
        config["calibration"]["device"] = args.device

    result = run_independent_calibration(
        config,
        basin_id=args.basin_id,
        model_id=args.model_id,
        objective=args.objective,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(result)


if __name__ == "__main__":
    main()
