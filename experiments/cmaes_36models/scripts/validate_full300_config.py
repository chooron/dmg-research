#!/usr/bin/env python3
"""Validate the frozen 10-start/300-generation configuration without training."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/cmaes_36models"
sys.path[:0] = [str(ROOT), str(EXPERIMENT)]

from src.data_selection import load_ids, load_repeated_warmup_and_train
from src.model_registry import audit_registry
from src.production_config import load_resolved_config, validate_full_run_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    config = load_resolved_config(args.config)
    validate_full_run_config(config)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = EXPERIMENT / manifest_path
    manifest = json.loads(manifest_path.read_text())
    manifest_config = {key: value for key, value in manifest["resolved_config"].items() if key != "_resolved_from"}
    runtime_config = {key: value for key, value in config.items() if key != "_resolved_from"}
    if manifest_config != runtime_config:
        raise RuntimeError("frozen manifest config differs from requested production config")
    for relative, expected in manifest["source_hashes_sha256"].items():
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"frozen source mismatch: {relative}")

    basin_ids = load_ids(config["data"]["basin_ids"])
    forcing, targets, metadata = load_repeated_warmup_and_train(basin_ids, config, "cpu")
    warmup_days, train_days = metadata["warmup_total_days"], metadata["train_days"]
    if forcing.shape[0] != warmup_days + train_days or targets.shape[0] != train_days:
        raise RuntimeError("warm-up/train shape contract failed")
    source_days = metadata["warmup_source_days"]
    for repeat in range(1, metadata["warmup_repetitions"]):
        torch.testing.assert_close(forcing[:source_days], forcing[repeat * source_days : (repeat + 1) * source_days])
    result = {
        "passed": True,
        "registry_models": len(audit_registry()),
        "basins": int(len(basin_ids)),
        "forcing_shape": list(forcing.shape),
        "target_shape": list(targets.shape),
        "warmup": metadata,
        "objective_period": config["data"]["train"],
        "test_period_not_used_for_selection": config["data"]["test"],
        "manifest": str(manifest_path.resolve()),
    }
    report = EXPERIMENT / "reports/full300_configuration_validation.json"
    tmp = report.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(report)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
