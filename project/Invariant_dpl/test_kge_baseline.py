#!/usr/bin/env python3
"""Evaluate a trained KGE-baseline checkpoint on test basins with MC dropout."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from implements import BaselineTrainer
from test_causal_dpl import build_parser, run_evaluation


def parse_args():
    return build_parser(
        description="Test a trained KGE-baseline checkpoint with MC dropout.",
        default_config="conf/config_kge_dhbv.yaml",
    ).parse_args()


def main() -> None:
    run_evaluation(
        args=parse_args(),
        trainer_cls=BaselineTrainer,
        logger_name="test_kge_baseline",
    )


if __name__ == "__main__":
    main()
