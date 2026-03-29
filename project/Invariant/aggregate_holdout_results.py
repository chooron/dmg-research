"""Aggregate basin-level held-out KGE results across repeated seeds."""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input-glob',
        required=True,
        help='Glob for results_held_out_{cluster}_seed{seed}.csv files.',
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to aggregated summary csv.',
    )
    parser.add_argument(
        '--expect-repeats',
        type=int,
        default=5,
        help='Expected number of repeats per basin.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(f"No result files matched: {args.input_glob}")

    frames = [pd.read_csv(path) for path in paths]
    results = pd.concat(frames, ignore_index=True)

    summary = (
        results.groupby('basin_id', as_index=False)
        .agg(
            kge_mean=('kge', 'mean'),
            kge_std=('kge', lambda s: s.std(ddof=0)),
            n_repeats=('kge', 'count'),
            effective_cluster=('effective_cluster', 'first'),
            gauge_cluster=('gauge_cluster', 'first'),
        )
        .sort_values('basin_id')
    )

    missing = summary.loc[summary['n_repeats'] != args.expect_repeats, ['basin_id', 'n_repeats']]
    if not missing.empty:
        raise ValueError(
            "Unexpected repeat count for some basins:\n"
            f"{missing.to_string(index=False)}"
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary.to_csv(args.output, index=False)

    print(f"Aggregated {len(paths)} files into {args.output}")
    print(f"Basins: {len(summary)}")
    print(f"Expected repeats per basin: {args.expect_repeats}")


if __name__ == '__main__':
    main()
