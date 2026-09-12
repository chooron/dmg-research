#!/usr/bin/env python3
"""Materialize fold-partitioned OOB data before any training worker starts.

The canonical source pickle contains all basins in one object.  To make the
training access boundary enforceable, this one-time PREPARE step reads that
source and writes separate per-fold arrays.  TRAIN workers then open only
``train_x.npy``/``train_y.npy``; held-out target arrays are not opened until
Phase.EVAL.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from oob_common import (
    EXPECTED_BASINS,
    N_FOLDS,
    BasinDataRepository,
    Phase,
    fold_ids,
    load_ids,
    load_oob_config,
    read_fold_assignment,
    resolve_repo_path,
    set_phase,
    sha256_file,
    validate_fold_contract,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="oob_primary8_5fold_20260902.yaml")
    parser.add_argument("--out", default=None, help="Fold cache root; defaults to outputs.data_cache_root")
    parser.add_argument("--fold-assignment", default=None)
    args = parser.parse_args()
    config = load_oob_config(args.config)
    output_root = resolve_repo_path(args.out or config["outputs"]["data_cache_root"])
    assignment_path = resolve_repo_path(args.fold_assignment or config["folds"]["assignment_file"])
    all_ids = load_ids(config["data"]["basin_ids"])
    rows = read_fold_assignment(assignment_path)
    contract = validate_fold_contract(
        all_ids,
        rows,
        n_folds=N_FOLDS,
        expected_count=int(config["folds"].get("expected_basin_count", EXPECTED_BASINS)),
    )

    set_phase(Phase.PREPARE)
    try:
        repository = BasinDataRepository(config)
        forcings, target = repository._load_bundle()  # PREPARE-only monolithic source access
        dates = repository._dates()
        train_left, train_right = repository._bounds(dates, config["data"]["train"])
        test_left, test_right = repository._bounds(dates, config["data"]["test"])
        eval_warmup = int(config["protocol"]["evaluation_warmup_days"])
        forcing_left = test_left - eval_warmup
        if forcing_left < 0:
            raise ValueError("canonical source has insufficient held-out evaluation warm-up forcing")

        output_root.mkdir(parents=True, exist_ok=True)
        common_metadata = {
            "experiment_id": config["experiment_id"],
            "cache_contract": "train target and held-out target are separate files",
            "source_data_path": str(repository.data_path),
            "source_data_sha256": sha256_file(repository.data_path),
            "source_reference_ids_sha256": sha256_file(repository.reference_ids_path),
            "source_attributes_sha256": sha256_file(repository.attributes_path),
            "assignment_path": str(assignment_path),
            "assignment_sha256": sha256_file(assignment_path),
            "fold_sizes": list(contract.fold_sizes),
            "training_target_access_during_prepare": "allowed; preparation is outside TRAIN",
        }
        for fold in range(N_FOLDS):
            train_ids, heldout_ids = fold_ids(all_ids, rows, fold)
            train_indices = repository._indices(train_ids)
            heldout_indices = repository._indices(heldout_ids)
            fold_dir = output_root / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            np.save(fold_dir / "train_x.npy", np.asarray(forcings[train_indices, train_left:train_right, :3], dtype=np.float64).transpose(1, 0, 2))
            np.save(fold_dir / "train_y.npy", repository._convert_target(np.asarray(target[train_indices, train_left:train_right], dtype=np.float64), train_ids).T)
            np.save(fold_dir / "heldout_x.npy", np.asarray(forcings[heldout_indices, forcing_left:test_right, :3], dtype=np.float64).transpose(1, 0, 2))
            np.save(fold_dir / "heldout_y.npy", repository._convert_target(np.asarray(target[heldout_indices, test_left:test_right], dtype=np.float64), heldout_ids).T)
            (fold_dir / "train_basin_ids.txt").write_text("\n".join(map(str, train_ids.tolist())) + "\n", encoding="utf-8")
            (fold_dir / "heldout_basin_ids.txt").write_text("\n".join(map(str, heldout_ids.tolist())) + "\n", encoding="utf-8")
            write_json(
                fold_dir / "cache_metadata.json",
                {
                    **common_metadata,
                    "fold": fold,
                    "train_basin_ids": train_ids.tolist(),
                    "heldout_basin_ids": heldout_ids.tolist(),
                    "train_x_file": "train_x.npy",
                    "train_y_file": "train_y.npy",
                    "heldout_x_file": "heldout_x.npy",
                    "heldout_y_file": "heldout_y.npy",
                    "heldout_target_available_to_train_worker": False,
                    "training_period": [config["data"]["train"]["start_time"], config["data"]["train"]["end_time"]],
                    "test_period": [config["data"]["test"]["start_time"], config["data"]["test"]["end_time"]],
                    "evaluation_warmup_days": eval_warmup,
                },
            )
        write_json(output_root / "CACHE_COMPLETE.json", {**common_metadata, "n_folds": N_FOLDS, "complete": True})
        print(f"prepared {N_FOLDS} fold caches at {output_root}; fold_sizes={contract.fold_sizes}")
    finally:
        set_phase(Phase.TRAIN)


if __name__ == "__main__":
    main()
