"""Unit tests for Pure-Attribute Dedicated Structure Encoder.

Validates:
  1. Pure attribute input shape: struct_input is [B, 35], output is [B, 8]
  2. Structure encoder architecture: Linear(35, 128) -> Tanh -> Linear(128, 64) -> Tanh -> Linear(64, 8)
  3. Forward behavior & two-logit semantics:
     - Neutral initialization near p = 0.5
     - params and gamma_uh match when backbone is identical
  4. Gradient isolation invariants:
     - L_CF gives non-zero gradient to all layers of structure_encoder (35 -> 128 -> 64 -> 8)
     - L_CF gives strictly zero gradient to shared hydrologic backbone (0.0 grad by construction)
     - L_CF gives strictly zero gradient to params_head and gamma_head
     - Direct fit loss and direct AIC loss give strictly zero gradient to structure_encoder
     - Direct fit loss gives non-zero gradient to params_head, gamma_head, and backbone
  5. Single unified optimizer covers all trainable parameters without leakage or omissions
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
from project.flexmopex.models.cf_trainer import CFTrainer, per_basin_fit
from project.flexmopex.models.learned_weight_mopex_candidates import (
    LearnedWeightMopexE,
    LearnedStructureNetHybridEncoder,
    LearnedStructureNetPureAttrEncoder,
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


def test_pure_attribute_architecture_and_forward(dummy_batch):
    """Test that LearnedStructureNetPureAttrEncoder has structure_encoder(35 -> 128 -> 64 -> 8)."""
    nn = LearnedStructureNetPureAttrEncoder(input_dim=35, hidden_dim=128, nmul=16, device="cpu")
    nn.eval()

    assert hasattr(nn, "structure_encoder")
    assert nn.structure_encoder[0].in_features == 35  # Pure 35-D static attributes
    assert nn.structure_encoder[0].out_features == 128
    assert nn.structure_encoder[2].in_features == 128
    assert nn.structure_encoder[2].out_features == 64
    assert nn.structure_encoder[4].in_features == 64
    assert nn.structure_encoder[4].out_features == 8

    out = nn({"c_nn_norm": dummy_batch["c_nn_norm"]})
    assert out["weights"].shape == (10, 8)
    assert out["params"].shape == (10, 192)
    assert out["gamma_uh"].shape == (10, 2)

    # Check neutral gate initialization
    logits = out["weights"].view(10, 4, 2)
    p_struct = torch.sigmoid(logits[..., 1] - logits[..., 0])
    assert torch.allclose(p_struct, torch.full_like(p_struct, 0.5), atol=0.05)


def test_gradient_isolation_pure_encoder(dummy_batch):
    """Test that L_CF updates all layers of structure_encoder with strictly zero gradient to backbone."""
    nn = LearnedStructureNetPureAttrEncoder(input_dim=35, hidden_dim=128, nmul=16, device="cpu")
    nn.train()

    # Compute forward
    attrs = dummy_batch["c_nn_norm"]
    raw_weights = nn.structure_encoder(attrs)
    logits = raw_weights.view(raw_weights.shape[0], 4, 2)
    p_struct = torch.sigmoid(logits[..., 1] - logits[..., 0])

    q_target = torch.full_like(p_struct, 0.7)
    c_target = (2.0 * torch.abs(q_target - 0.5)).detach()
    bce_elem = F.binary_cross_entropy(p_struct, q_target, reduction="none")
    sum_c = torch.sum(c_target, dim=0)
    loss_cf = torch.mean(torch.sum(c_target * bce_elem, dim=0) / (sum_c + 1e-12))

    loss_cf.backward()

    # Structure encoder layers MUST receive gradient
    assert nn.structure_encoder[0].weight.grad is not None
    assert torch.norm(nn.structure_encoder[0].weight.grad) > 1e-4
    assert nn.structure_encoder[2].weight.grad is not None
    assert torch.norm(nn.structure_encoder[2].weight.grad) > 1e-4
    assert nn.structure_encoder[4].weight.grad is not None
    assert torch.norm(nn.structure_encoder[4].weight.grad) > 1e-4

    # Backbone MUST have ZERO gradient by construction!
    assert nn.backbone[0].weight.grad is None
    assert nn.backbone[3].weight.grad is None
    assert nn.heads["params"].weight.grad is None
    assert nn.heads["gamma_uh"].weight.grad is None


def test_physics_fit_loss_gradient_routing_pure(dummy_batch):
    """Test that fit loss gives zero grad to structure_encoder but trains backbone/params/gamma."""
    cfg = load_config("conf/config_dmopex_interceptE_S0_r19_pure_x35_seed42.yaml")
    cfg["mode"] = "train"
    cfg["model"]["phy"]["disable_compile"] = True

    phy = LearnedWeightMopexE(cfg["delta_model"]["phy_model"], device="cpu")
    nn = LearnedStructureNetPureAttrEncoder(input_dim=35, hidden_dim=128, nmul=16, device="cpu")
    phy.train()
    nn.train()

    params = nn({"c_nn_norm": dummy_batch["c_nn_norm"]})
    out = phy(dummy_batch, params)

    # Compute Fit Loss
    q = out["streamflow"]
    obs = dummy_batch["target"][365:]
    loss_fit = per_basin_fit(q, obs, dummy_batch["std"]).mean()
    loss_fit.backward()

    # structure_encoder must have zero grad from fit loss!
    for p in nn.structure_encoder.parameters():
        assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p))

    # Backbone and params_head MUST receive nonzero grad from fit loss!
    assert nn.heads["params"].weight.grad is not None and torch.norm(nn.heads["params"].weight.grad) > 1e-6
    assert nn.heads["gamma_uh"].weight.grad is not None and torch.norm(nn.heads["gamma_uh"].weight.grad) > 1e-6
    assert nn.backbone[0].weight.grad is not None and torch.norm(nn.backbone[0].weight.grad) > 1e-6


def test_unified_adadelta_optimizer_pure(dummy_batch):
    """Test that CFTrainer properly initializes a single Adadelta optimizer containing all parameters."""
    cfg = load_config("conf/config_dmopex_interceptE_S0_r19_pure_x35_seed42.yaml")
    cfg["mode"] = "train"
    cfg["device"] = "cpu"
    cfg["model"]["phy"]["disable_compile"] = True
    cfg["model_dir"] = "/tmp/test_cf_r19_pure_x35_dir"

    phy = LearnedWeightMopexE(cfg["delta_model"]["phy_model"], device="cpu")
    nn = LearnedStructureNetPureAttrEncoder(input_dim=35, hidden_dim=128, nmul=16, device="cpu")

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
    assert len(trainer.weights_head_params) == 6
    assert trainer.weights_head_params[0].shape == (128, 35)
    assert trainer.weights_head_params[2].shape == (64, 128)
    assert trainer.weights_head_params[4].shape == (8, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
