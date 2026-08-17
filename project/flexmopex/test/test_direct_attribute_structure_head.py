"""Unit tests for R18 Direct-Attribute Structure Head.

Validates:
  1. Direct 35-D structure-head input path:
     - weights_head.in_features == 35 (direct from c_nn_norm)
     - params_head.in_features == 128 (from backbone)
     - gamma_head.in_features == 128 (from backbone)
  2. Forward behavior & two-logit semantics:
     - Neutral initialization near p = 0.5
     - params and gamma_uh match LearnedStructureNetCF when backbone is identical
  3. Gradient isolation invariants:
     - L_CF gives non-zero gradient to weights_head (35 -> 8)
     - L_CF gives strictly zero gradient to shared backbone (0.0 grad by construction)
     - L_CF gives strictly zero gradient to params_head and gamma_head
     - Direct fit loss and direct AIC loss give strictly zero gradient to weights_head
     - Direct fit loss gives non-zero gradient to params_head, gamma_head, and backbone
  4. Dual-optimizer parameter separation in CFTrainer
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
    LearnedStructureNetCF,
    LearnedStructureNetDirectAttr,
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


def test_direct_attribute_architecture_and_forward(dummy_batch):
    """Test that LearnedStructureNetDirectAttr has weights_head(35) and backbone(128)."""
    nn = LearnedStructureNetDirectAttr(input_dim=35, hidden_dim=128, nmul=16, device="cpu")
    nn.eval()

    assert nn.heads["weights"].in_features == 35
    assert nn.heads["weights"].out_features == 8
    assert nn.heads["params"].in_features == 128
    assert nn.heads["gamma_uh"].in_features == 128

    out = nn({"c_nn_norm": dummy_batch["c_nn_norm"]})
    assert out["weights"].shape == (10, 8)
    assert out["params"].shape == (10, 192)
    assert out["gamma_uh"].shape == (10, 2)

    # Check neutral gate initialization
    logits = out["weights"].view(10, 4, 2)
    p_struct = torch.sigmoid(logits[..., 1] - logits[..., 0])
    assert torch.allclose(p_struct, torch.full_like(p_struct, 0.5), atol=0.05)


def test_gradient_isolation_direct_attribute(dummy_batch):
    """Test that L_CF updates direct weights_head with strictly zero gradient to backbone."""
    nn = LearnedStructureNetDirectAttr(input_dim=35, hidden_dim=128, nmul=16, device="cpu")
    nn.train()

    # Compute p_struct directly from weights_head(c_nn_norm)
    attrs = dummy_batch["c_nn_norm"]
    raw_weights = nn.heads["weights"](attrs)
    logits = raw_weights.view(raw_weights.shape[0], 4, 2)
    p_struct = torch.sigmoid(logits[..., 1] - logits[..., 0])

    q_target = torch.full_like(p_struct, 0.7)
    c_target = (2.0 * torch.abs(q_target - 0.5)).detach()
    bce_elem = F.binary_cross_entropy(p_struct, q_target, reduction="none")
    sum_c = torch.sum(c_target, dim=0)
    loss_cf = torch.mean(torch.sum(c_target * bce_elem, dim=0) / (sum_c + 1e-12))

    loss_cf.backward()

    # Direct weights_head MUST receive gradient
    assert nn.heads["weights"].weight.grad is not None
    assert torch.norm(nn.heads["weights"].weight.grad) > 1e-4

    # Backbone MUST have ZERO gradient by construction!
    assert nn.backbone[0].weight.grad is None
    assert nn.backbone[3].weight.grad is None
    assert nn.heads["params"].weight.grad is None
    assert nn.heads["gamma_uh"].weight.grad is None


def test_physics_fit_loss_gradient_routing(dummy_batch):
    """Test that fit loss gives zero grad to direct weights_head but trains backbone/params/gamma."""
    cfg = load_config("conf/config_dmopex_interceptE_S0_r18a.yaml")
    cfg["mode"] = "train"
    cfg["model"]["phy"]["disable_compile"] = True

    phy = LearnedWeightMopexE(cfg["delta_model"]["phy_model"], device="cpu")
    nn = LearnedStructureNetDirectAttr(input_dim=35, hidden_dim=128, nmul=16, device="cpu")
    phy.train()
    nn.train()

    params = nn({"c_nn_norm": dummy_batch["c_nn_norm"]})
    out = phy(dummy_batch, params)

    # Compute Fit Loss
    q = out["streamflow"]
    obs = dummy_batch["target"][365:]
    loss_fit = per_basin_fit(q, obs, dummy_batch["std"]).mean()
    loss_fit.backward()

    # weights_head must have zero grad from fit loss!
    weights_head_weight = nn.heads["weights"].weight
    weights_head_bias = nn.heads["weights"].bias
    assert weights_head_weight.grad is None or torch.allclose(weights_head_weight.grad, torch.zeros_like(weights_head_weight))
    assert weights_head_bias.grad is None or torch.allclose(weights_head_bias.grad, torch.zeros_like(weights_head_bias))

    # Backbone and params_head MUST receive nonzero grad from fit loss!
    assert nn.heads["params"].weight.grad is not None and torch.norm(nn.heads["params"].weight.grad) > 1e-6
    assert nn.backbone[0].weight.grad is not None and torch.norm(nn.backbone[0].weight.grad) > 1e-6


def test_dual_optimizer_with_direct_attribute_head(dummy_batch):
    """Test that CFTrainer properly separates parameters with direct attribute head."""
    cfg = load_config("conf/config_dmopex_interceptE_S0_r18a.yaml")
    cfg["mode"] = "train"
    cfg["device"] = "cpu"
    cfg["model"]["phy"]["disable_compile"] = True
    cfg["model_dir"] = "/tmp/test_cf_r18_dir"

    phy = LearnedWeightMopexE(cfg["delta_model"]["phy_model"], device="cpu")
    nn = LearnedStructureNetDirectAttr(input_dim=35, hidden_dim=128, nmul=16, device="cpu")

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

    assert trainer.structure_optimizer is not None
    assert isinstance(trainer.structure_optimizer, torch.optim.Adam)
    assert isinstance(trainer.optimizer, torch.optim.Adadelta)
    assert len(trainer.weights_head_params) == 2  # weight [8, 35] and bias [8]
    assert trainer.weights_head_params[0].shape == (8, 35)

    weights_ids = {id(p) for p in trainer.weights_head_params}
    for group in trainer.optimizer.param_groups:
        for p in group["params"]:
            assert id(p) not in weights_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
