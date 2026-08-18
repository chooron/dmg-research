#!/usr/bin/env python3
"""Launch the formal GR4J_TGD2 and SIMHYD_TGD2 CMA-ES calibrations sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from training.ic.run_tgd2_batched_cmaes_531 import (
    DEFAULT_GENERATIONS,
    DEFAULT_STARTS,
    MODEL_DIMENSIONS,
    population_for_dimension,
)

PROJECT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT / "training/ic/run_tgd2_batched_cmaes_531.py"
MODELS = ("GR4J_TGD2", "SIMHYD_TGD2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--starts", type=int, default=DEFAULT_STARTS)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--chunk-basins", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        min(args.starts, args.generations, args.chunk_basins, args.checkpoint_interval)
        < 1
    ):
        parser.error(
            "starts, generations, chunk-basins, and checkpoint-interval must be positive"
        )

    for model in args.models:
        dimension = MODEL_DIMENSIONS[model]
        population = population_for_dimension(dimension)
        output = args.output_root / f"{model.lower()}_cmaes_531_batched_v1"
        command = [
            sys.executable,
            str(RUNNER),
            "--model",
            model,
            "--output",
            str(output),
            "--starts",
            str(args.starts),
            "--population",
            str(population),
            "--generations",
            str(args.generations),
            "--chunk-basins",
            str(args.chunk_basins),
            "--checkpoint-interval",
            str(args.checkpoint_interval),
            "--device",
            args.device,
        ]
        print("COMMAND:", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
