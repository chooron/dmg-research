"""Targeted tests for formal CAMELS-531 100-epoch experiment configurations.

Verifies:
1. All five formal configuration files exist and load valid 531-basin data.
2. Base and Full use FixedWeightMopex (Candidate E-S0) with ParamRoutingNet and MyTrainer.
3. Flex variants use LearnedWeightMopexE with LearnedStructureNetPureAttrEncoder and CFTrainer.
4. Lambda is accurately configured (0.005, 0.007, 0.010) and is the single source of truth.
5. All runs have seed=42, epochs=100, test_epoch=100, save_epoch=1, and isolated output paths.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config
from project.flexmopex.local_model_handler import FlexMopexModelHandler
from project.flexmopex.run_model import _build_data_loader, _build_loss


@pytest.mark.parametrize(
    "config_name,expected_variant,expected_lambda,expected_trainer,expected_phy,expected_nn,expected_weights",
    [
        (
            "config_formal_531_base.yaml",
            "base",
            0.0,
            "MyTrainer",
            ["FixedWeightMopex"],
            "ParamRoutingNet",
            {"w_phen": 0.0, "w_int": 0.0, "w_snow": 0.0, "w_sub": 0.0},
        ),
        (
            "config_formal_531_full.yaml",
            "full",
            0.0,
            "MyTrainer",
            ["FixedWeightMopex"],
            "ParamRoutingNet",
            {"w_phen": 1.0, "w_int": 1.0, "w_snow": 1.0, "w_sub": 1.0},
        ),
        (
            "config_formal_531_flex_lambda0005.yaml",
            "flex",
            0.005,
            "CFTrainer",
            ["LearnedWeightMopexE"],
            "LearnedStructureNetPureAttrEncoder",
            None,
        ),
        (
            "config_formal_531_flex_lambda0007.yaml",
            "flex",
            0.007,
            "CFTrainer",
            ["LearnedWeightMopexE"],
            "LearnedStructureNetPureAttrEncoder",
            None,
        ),
        (
            "config_formal_531_flex_lambda0010.yaml",
            "flex",
            0.010,
            "CFTrainer",
            ["LearnedWeightMopexE"],
            "LearnedStructureNetPureAttrEncoder",
            None,
        ),
    ],
)
def test_formal_531_config_structure(
    config_name,
    expected_variant,
    expected_lambda,
    expected_trainer,
    expected_phy,
    expected_nn,
    expected_weights,
):
    config_path = PROJECT_DIR / "conf" / config_name
    assert config_path.exists(), f"Missing config file: {config_path}"

    cfg = load_config(config_path)

    # 1. Dataset verification
    assert cfg["observations"]["name"] == "camels_531"
    assert "531sub_id.txt" in str(cfg["observations"]["subset_path"])

    # 2. General / Training verification
    assert cfg["random_seed"] == 42
    assert cfg["train"]["epochs"] == 100
    assert cfg["test"]["test_epoch"] == 100
    assert cfg["train"]["save_epoch"] == 1
    assert cfg["train"]["optimizer"] == "Adadelta"
    assert cfg["train"]["learning_rate"] == 1.0

    # 3. Model components
    assert cfg["model"]["phy"]["name"] == expected_phy
    assert cfg["model"]["nn"]["name"] == expected_nn
    assert cfg["model"]["phy"]["interception_semantics"] == "S0"
    assert cfg.get("trainer") == expected_trainer

    # 4. Lambda verification
    assert cfg["loss_function"]["aic_alpha"] == expected_lambda

    # 5. Fixed weights if applicable
    if expected_weights is not None:
        assert cfg["model"]["phy"]["fixed_weights"] == expected_weights
        assert cfg["counterfactual_supervision"] is False
    else:
        assert cfg["counterfactual_supervision"] is True
        assert cfg["confidence_weighted_cf_loss"] is True
        assert cfg["cf_loss_weight"] == 1.0


def test_unique_output_directories_across_all_configs():
    """Verify all 5 configs write to distinct, isolated results directories."""
    config_names = [
        "config_formal_531_base.yaml",
        "config_formal_531_full.yaml",
        "config_formal_531_flex_lambda0005.yaml",
        "config_formal_531_flex_lambda0007.yaml",
        "config_formal_531_flex_lambda0010.yaml",
    ]
    save_paths = set()
    model_paths = set()

    for name in config_names:
        cfg = load_config(PROJECT_DIR / "conf" / name)
        save_path = cfg["save_path"]
        model_path = cfg["trained_model"]

        assert "canonical_freeze" not in save_path, f"{name} must not reuse canonical_freeze"
        assert "camels_671" not in save_path, f"{name} must not reuse 671 path"
        assert save_path not in save_paths, f"Duplicate save_path: {save_path}"
        assert model_path not in model_paths, f"Duplicate model_path: {model_path}"

        save_paths.add(save_path)
        model_paths.add(model_path)
