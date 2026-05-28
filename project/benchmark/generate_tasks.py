#!/usr/bin/env python
"""Generate basin-model-objective task tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.config import load_benchmark_config
from benchmark.tasks import generate_independent_calibration_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="project/benchmark/conf/benchmark.yaml")
    parser.add_argument("--output", default="project/benchmark/outputs/tasks/independent_calibration_tasks.csv")
    parser.add_argument("--limit-basins", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_benchmark_config(args.config)
    path = generate_independent_calibration_tasks(config, Path(args.output), limit_basins=args.limit_basins)
    print(path)


if __name__ == "__main__":
    main()
