#!/usr/bin/env python
"""Merge per-task calibration outputs into one evidence table."""

from __future__ import annotations

import argparse

from benchmark.evidence import write_evidence_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="project/benchmark/outputs/independent_calibration")
    parser.add_argument("--output", default="project/benchmark/outputs/evidence/independent_calibration_evidence.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(write_evidence_table(args.root, args.output))


if __name__ == "__main__":
    main()
