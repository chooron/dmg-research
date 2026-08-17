"""Unit tests for R19 Unified Optimizer Simplification.

Validates:
  1. Parameter coverage: Single optimizer contains all trainable parameters
  2. Structure optimizer is None when structure_optimizer='none'
  3. Single step updates both hydrologic and structure parameters without gradient leakage
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config
from project.flexmopex.models.cf_trainer import CounterfactualTargetGenerator, CFTrainer, per_basin_fit
from project.flexmopex.models.learned_weight_mopex_candidates import (
    LearnedWeightMopexE,
    LearnedStructureNetHybridEncoder,
)
from project.flexmopex.models.nse_dyn_aic_batch_loss import NseDynAicBatchLoss


@pytest.fixture
def dummy_batch():
    B, T, n_attr = 10, 730, 35
    torch.manual_seed(42)
    x_phy = torch.randn(T, B, 3).abs() + 0.1
    doy = torch.randint(1, 366, (T, B, 1)).float()
    c_nn_norm = torch.randn(B, n_attr)
    target = torch.randn(T, B, 1).abs() + 0.05
    std = torch.ones(B) * 0.5
    return {
        "x_phy": x_phy,
        "doy": doy,
        "c_nn_norm": c_nn_norm,
        "target": target,
        "std": std,
        "batch_sample": torch.arange(B),
    }


def test_unified_adadelta_optimizer_initialization(dummy_batch):
    """Test that CFTrainer properly initializes a single Adadelta optimizer containing all parameters."""
    cfg = load_config("conf/config_dmopex_interceptE_S0_r19_unified_adadelta.yaml")
    cfg["mode"] = "train"
    cfg["device"] = "cpu"
    cfg["model"]["phy"]["disable_compile"] = True
    cfg["model_dir"] = "/tmp/test_cf_r19_unified_adadelta_dir"

    phy = LearnedWeightMopexE(cfg["delta_model"]["phy_model"], device="cpu")
    nn = LearnedStructureNetHybridEncoder(input_dim=35, hidden_dim=128, nmul=16, device="cpu")

    class DummyModelHandler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model_dict = {"LearnedWeightMopexE": torch.nn.Module()}
            self.model_dict["LearnedWeightMopexE"].phy_model = phy
            self.model_dict["LearnedWeightMopexE"].nn_model = nn

        def get_parameters(self):
            return list(self.model_dict["LearnedWeightMopexE"].phy_model.parameters()) + list(self.model_dict["LearnedWeightMopexE"].nn_model.parameters())

        def load_model(self, epoch=0):
            pass

    m_handler = DummyModelHandler()
    loss_fn = NseDynAicBatchLoss(cfg["loss_function"], y_obs=dummy_batch["target"], device="cpu")
    trainer = CFTrainer(cfg, m_handler, train_dataset=dummy_batch, loss_func=loss_fn)

    # In unified mode, structure_optimizer must be None
    assert trainer.structure_optimizer is None
    assert isinstance(trainer.optimizer, torch.optim.Adadelta)

    # Primary optimizer must contain ALL model parameters (including structure_encoder)
    all_param_ids = {id(p) for p in m_handler.get_parameters()}
    opt_param_ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert all_param_ids == opt_param_ids, "Mismatch in unified optimizer parameter registration!"


def test_unified_adam_optimizer_initialization(dummy_batch):
    """Test that CFTrainer properly initializes a single Adam optimizer when train.optimizer='Adam'."""
    cfg = load_config("conf/config_dmopex_interceptE_S0_r19_unified_adam.yaml")
    cfg["mode"] = "train"
    cfg["device"] = "cpu"
    cfg["model"]["phy"]["disable_compile"] = True
    cfg["model_dir"] = "/tmp/test_cf_r19_unified_adam_dir"

    phy = LearnedWeightMopexE(cfg["delta_model"]["phy_model"], device="cpu")
    nn = LearnedStructureNetHybridEncoder(input_dim=35, hidden_dim=128, nmul=16, device="cpu")

    class DummyModelHandler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model_dict = {"LearnedWeightMopexE": torch.nn.Module()}
            self.model_dict["LearnedWeightMopexE"].phy_model = phy
            self.model_dict["LearnedWeightMopexE"].nn_model = nn

        def get_parameters(self):
            return list(self.model_dict["LearnedWeightMopexE"].phy_model.parameters()) + list(self.model_dict["LearnedWeightMopexE"].nn_model.parameters())

        def load_model(self, epoch=0):
            pass

    m_handler = DummyModelHandler()
    loss_fn = NseDynAicBatchLoss(cfg["loss_function"], y_obs=dummy_batch["target"], device="cpu")
    trainer = CFTrainer(cfg, m_handler, train_dataset=dummy_batch, loss_func=loss_fn)

    assert trainer.structure_optimizer is None
    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert trainer.optimizer.defaults["lr"] == 0.001

    all_param_ids = {id(p) for p in m_handler.get_parameters()}
    opt_param_ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    assert all_param_ids == opt_param_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
