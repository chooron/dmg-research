#!/usr/bin/env python3
"""Run the missing Base/CN IC calibrations serially under the 531-basin protocol."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))

from training.ic.run_tgd2_batched_cmaes_531 import (
    ADDITIONAL_MODEL_DIMENSIONS,
    DEFAULT_GENERATIONS,
    DEFAULT_STARTS,
    population_for_dimension,
)


RUNNER = PROJECT / "training/ic/run_tgd2_batched_cmaes_531.py"
MODELS = ("GR4J", "GR4J_CN", "SIMHYD", "SIMHYD_CN")
# Tuned from the completed SIMHYD_TGD2 run: approximately 10 GiB peak VRAM
# while retaining room for CUDA/WSL driver overhead.
CHUNK_BASINS = {"GR4J": 100, "GR4J_CN": 90, "SIMHYD": 85, "SIMHYD_CN": 75}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--starts", type=int, default=DEFAULT_STARTS)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.starts, args.generations, args.checkpoint_interval) < 1:
        parser.error("starts, generations, and checkpoint-interval must be positive")

    for model in args.models:
        population = population_for_dimension(ADDITIONAL_MODEL_DIMENSIONS[model])
        output = args.output_root / f"{model.lower()}_cmaes_531_batched_v1"
        command = [
            sys.executable, str(RUNNER), "--model", model, "--output", str(output),
            "--starts", str(args.starts), "--population", str(population),
            "--generations", str(args.generations),
            "--chunk-basins", str(CHUNK_BASINS[model]),
            "--checkpoint-interval", str(args.checkpoint_interval), "--device", args.device,
        ]
        print("COMMAND:", " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
