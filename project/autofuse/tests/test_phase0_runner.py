"""Unit and integration tests for Phase 0 Runner & Orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
import yaml

from project.autofuse.phase0_runner import (
    Phase0Orchestrator,
    load_phase0_config,
    prepare_inputs,
)


@pytest.fixture
def smoke_config_dict(tmp_path: Path) -> dict:
    return {
        "schema_version": "autofuse-phase0-config-v1",
        "run_id": "test_smoke_run",
        "output_dir": str(tmp_path / "runs"),
        "cache_dir": str(tmp_path / "cache"),
        "seed": 20260901,
        "device": "cpu",
        "dtype": "float32",
        "data": {
            "manifest_path": "project/autofuse/docs/landscape_12catchment_manifest.json",
            "inputs_dir": "project/autofuse/docs/landscape_inputs",
            "target_basin_count": 2,
            "periods": {
                "forcing": ["1987-01-01", "2009-12-31"],
                "warmup": ["1987-01-01", "1988-12-31"],
                "calibration": ["1989-01-01", "1998-12-31"],
                "evaluation": ["1999-01-01", "2009-12-31"],
            },
        },
        "time_window": {
            "total_days": 8,
            "warmup_days": 4,
            "scored_days": 4,
        },
        "model": {
            "attribute_dim": 35,
            "hidden_dim": 64,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
        },
        "registry": {
            "path": "dfuse/specs/structures_78.json",
            "name": "structures_78",
            "anchors": [2],
        },
        "stages": {
            "p0_a": {"enabled": True},
            "p0_b_throughput": {
                "enabled": True,
                "batch_size": 2,
                "warmup_steps": 0,
                "measured_steps": 1,
                "structures": [2],
            },
            "p0_c_compile": {
                "enabled": True,
                "batch_size": 2,
                "structures": [2],
                "test_eager": True,
                "test_compile": False, # on CPU mock
            },
            "p0_d_validation": {
                "enabled": True,
                "structures": [2],
                "eval_period": "evaluation",
                "eval_days": 10,
                "batch_size": 2,
                "candidate_eval_intervals": [50, 100],
            },
            "p0_e_pilot": {
                "enabled": True,
                "batch_size": 2,
                "cycles": 1,
                "structures": [2],
            },
            "p0_f_synthesis": {"enabled": True},
        },
    }


def test_load_phase0_config(tmp_path: Path, smoke_config_dict: dict):
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(yaml.dump(smoke_config_dict))
    loaded = load_phase0_config(cfg_file)
    assert loaded["schema_version"] == "autofuse-phase0-config-v1"
    assert loaded["run_id"] == "test_smoke_run"


def test_prepare_inputs(smoke_config_dict: dict):
    device = torch.device("cpu")
    inputs, dates = prepare_inputs(smoke_config_dict, device)
    assert len(inputs) == 2
    for b_id, b_data in inputs.items():
        assert "ppt" in b_data
        assert "pet" in b_data
        assert "temp" in b_data
        assert "q_obs" in b_data
        assert "attributes" in b_data
        assert b_data["attributes"].shape == (35,)


def test_phase0_orchestrator_stage_p0_a(tmp_path: Path, smoke_config_dict: dict):
    orchestrator = Phase0Orchestrator(smoke_config_dict)
    res_a = orchestrator.run_stage_p0_a()
    assert res_a["status"] == "PASS"
    assert "git" in res_a
    assert "kernel_hashes" in res_a
    assert "kernel.py" in res_a["kernel_hashes"]

    # Test resume/skip
    res_a2 = orchestrator.run_stage_p0_a()
    assert res_a2["status"] == "PASS"


def test_phase0_orchestrator_stage_p0_b(tmp_path: Path, smoke_config_dict: dict):
    device = torch.device("cpu")
    inputs, dates = prepare_inputs(smoke_config_dict, device)
    orchestrator = Phase0Orchestrator(smoke_config_dict)
    res_b = orchestrator.run_stage_p0_b(inputs, dates)
    assert res_b["status"] == "PASS"
    assert "aggregate_statistics" in res_b
    assert len(res_b["per_structure_records"]) == 1


def test_phase0_orchestrator_stage_p0_d(tmp_path: Path, smoke_config_dict: dict):
    device = torch.device("cpu")
    inputs, dates = prepare_inputs(smoke_config_dict, device)
    orchestrator = Phase0Orchestrator(smoke_config_dict)
    res_d = orchestrator.run_stage_p0_d(inputs, dates)
    assert res_d["status"] == "PASS"
    assert len(res_d["candidate_interval_analysis"]) == 2
    assert "kge_mean" in res_d["records"][0]


def test_phase0_orchestrator_stage_p0_e(tmp_path: Path, smoke_config_dict: dict):
    device = torch.device("cpu")
    inputs, dates = prepare_inputs(smoke_config_dict, device)
    orchestrator = Phase0Orchestrator(smoke_config_dict)
    res_e = orchestrator.run_stage_p0_e(inputs, dates)
    assert res_e["status"] == "PASS"
    assert res_e["total_steps"] == 1
    assert "clipping_overall" in res_e


def test_phase0_orchestrator_stage_p0_f(tmp_path: Path, smoke_config_dict: dict):
    device = torch.device("cpu")
    inputs, dates = prepare_inputs(smoke_config_dict, device)
    orchestrator = Phase0Orchestrator(smoke_config_dict)
    orchestrator.run_stage_p0_a()
    orchestrator.run_stage_p0_b(inputs, dates)
    orchestrator.run_stage_p0_d(inputs, dates)
    orchestrator.run_stage_p0_e(inputs, dates)
    res_f = orchestrator.run_stage_p0_f()
    assert res_f["status"] == "PASS"
    assert (orchestrator.output_dir / "phase0_synthesis.md").is_file()
