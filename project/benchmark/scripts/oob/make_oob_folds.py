#!/usr/bin/env python3
"""Freeze and validate the shared 531-basin five-fold assignment."""
from __future__ import annotations

import argparse
from pathlib import Path

from oob_common import (
    EXPECTED_BASINS,
    FOLD_SEED,
    N_FOLDS,
    BENCHMARK_ROOT,
    fold_ids,
    git_sha,
    load_ids,
    load_oob_config,
    make_fold_assignment,
    read_fold_assignment,
    resolve_repo_path,
    sha256_file,
    validate_fold_contract,
    write_fold_assignment,
    write_json,
)


def assignment_path(config: dict, explicit: str | None) -> Path:
    path = explicit or config["folds"]["assignment_file"]
    return resolve_repo_path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="oob_primary8_5fold_20260902.yaml")
    parser.add_argument("--output", default=None, help="Override OOB_FOLD_ASSIGNMENT.csv path")
    parser.add_argument("--force", action="store_true", help="Regenerate only after explicit confirmation")
    args = parser.parse_args()

    config = load_oob_config(args.config)
    ids = load_ids(config["data"]["basin_ids"])
    expected_count = int(config["folds"].get("expected_basin_count", EXPECTED_BASINS))
    if ids.size != expected_count:
        raise SystemExit(f"fold gate failed: expected {expected_count} configured basins, found {ids.size}")
    destination = assignment_path(config, args.output)

    expected_rows = make_fold_assignment(ids, n_folds=N_FOLDS, random_state=FOLD_SEED)
    if destination.exists() and not args.force:
        rows = read_fold_assignment(destination)
        if {int(row["basin_id"]): int(row["fold"]) for row in rows} != {int(row["basin_id"]): int(row["fold"]) for row in expected_rows}:
            raise SystemExit("existing assignment does not match frozen KFold random_state=20260902")
        contract = validate_fold_contract(ids, rows, n_folds=N_FOLDS, expected_count=expected_count)
        print(f"FROZEN assignment retained: {destination}")
    else:
        rows = make_fold_assignment(ids, n_folds=N_FOLDS, random_state=FOLD_SEED)
        contract = validate_fold_contract(ids, rows, n_folds=N_FOLDS, expected_count=expected_count)
        digest = write_fold_assignment(destination, rows)
        print(f"FROZEN assignment written: {destination}")
        print(f"assignment_sha256={digest}")

    digest = sha256_file(destination)
    metadata = {
        "experiment_id": config["experiment_id"],
        "assignment_file": str(destination),
        "assignment_sha256": digest,
        "basin_count": contract.basin_count,
        "n_folds": contract.n_folds,
        "fold_sizes": list(contract.fold_sizes),
        "shuffle": True,
        "random_state": FOLD_SEED,
        "kfold_mechanics": "sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=20260902)",
        "train_heldout_disjoint": True,
        "heldout_union_exact": True,
        "git_sha": git_sha(),
    }
    write_json(destination.with_name("OOB_FOLD_ASSIGNMENT_METADATA.json"), metadata)
    print(f"fold_sizes={contract.fold_sizes}; sha256={digest}")


if __name__ == "__main__":
    main()
