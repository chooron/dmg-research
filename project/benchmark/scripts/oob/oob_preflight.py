#!/usr/bin/env python3
"""Run local OOB gates without training or starting a queue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from oob_common import (
    EXPECTED_BASINS,
    N_FOLDS,
    PRIMARY8,
    BasinDataRepository,
    Phase,
    fold_ids,
    load_ids,
    load_oob_config,
    read_fold_assignment,
    resolve_repo_path,
    set_phase,
    validate_fold_contract,
    validate_oob_protocol,
    write_json,
)
from dmotpy.data_contract import CALENDAR_MODELS, add_calendar_forcing
from src.model_registry import get_spec


def run_gate(name: str, function, results: dict[str, dict]) -> None:
    try:
        detail = function() or {}
        results[name] = {"status": "PASS", **detail}
        print(f"{name}: PASS")
    except Exception as exc:
        results[name] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        print(f"{name}: FAIL: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="oob_primary8_5fold_20260902.yaml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    config = load_oob_config(args.config)
    output_root = resolve_repo_path(args.out or config["outputs"]["root"])
    results: dict[str, dict] = {}

    def protocol_gate():
        validate_oob_protocol(config)
        return {"models": list(PRIMARY8)}

    def fold_gate():
        ids = load_ids(config["data"]["basin_ids"])
        assignment = resolve_repo_path(config["folds"]["assignment_file"])
        contract = validate_fold_contract(
            ids,
            read_fold_assignment(assignment),
            n_folds=N_FOLDS,
            expected_count=int(config["folds"].get("expected_basin_count", EXPECTED_BASINS)),
        )
        return {"basin_count": contract.basin_count, "fold_sizes": list(contract.fold_sizes), "assignment": str(assignment)}

    def source_gate():
        repository = BasinDataRepository(config)
        attributes = repository.load_canonical_attributes(load_ids(config["data"]["basin_ids"])[:1])
        if attributes.shape != (1, 35) or not np.isfinite(attributes).all():
            raise ValueError(f"canonical Caravan attribute row is invalid: {attributes.shape}")
        set_phase(Phase.PREPARE)
        try:
            forcings, target = repository._load_bundle()
            if forcings.shape[0] != 671 or target.shape[:2] != forcings.shape[:2]:
                raise ValueError(f"canonical arrays do not align: forcing={forcings.shape}, target={target.shape}")
            source_shape = {"forcing_shape": list(forcings.shape), "target_shape": list(target.shape)}
        finally:
            set_phase(Phase.TRAIN)
        return {"attributes_shape": list(attributes.shape), "canonical_source": str(repository.attributes_path), **source_shape}

    def calendar_gate():
        dates = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[D]")
        base = torch.zeros((2, 1, 3), dtype=torch.float64)
        mopex = add_calendar_forcing(base, dates, model_name="mopex4")[0]
        ordinary = add_calendar_forcing(base, dates, model_name="alpine2")[0]
        if mopex.shape != (2, 1, 4) or ordinary.shape != (2, 1, 3):
            raise ValueError(f"calendar forcing shapes are mopex4={tuple(mopex.shape)}, alpine2={tuple(ordinary.shape)}")
        for model in PRIMARY8:
            get_spec(model)
        if "mopex4" not in CALENDAR_MODELS:
            raise ValueError("mopex4 is not registered as a calendar model")
        return {"mopex4_channels": int(mopex.shape[-1]), "ordinary_channels": int(ordinary.shape[-1])}

    run_gate("protocol", protocol_gate, results)
    run_gate("fold-contract", fold_gate, results)
    run_gate("canonical-data-and-attributes", source_gate, results)
    run_gate("calendar-forcing-and-model-registry", calendar_gate, results)
    overall = all(row["status"] == "PASS" for row in results.values())
    payload = {"overall": "PASS" if overall else "FAIL", "gates": results, "training_started": False}
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "OOB_LOCAL_PREFLIGHT.json", payload)
    print(json.dumps(payload, sort_keys=True))
    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
