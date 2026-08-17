"""Unit tests for R15 Counterfactual Structural Supervision.

Validates:
  1. Forward Invariants (output values, loss, streamflow identical to baseline)
  2. Gradient Invariants:
     - Direct fit loss -> weights_head grad == 0
     - Direct AIC loss -> weights_head grad == 0
     - L_CF -> backbone grad == 0
     - L_CF -> weights_head grad != 0
     - Fit loss -> parameter/routing heads grad != 0
     - Fit loss -> backbone grad != 0
  3. Target Sanity (DeltaJ, T, q in (0, 1), detached)
  4. Default-Off Backward Compatibility
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
)
from project.flexmopex.models.parameter_nets import LearnedStructureNet
from project.flexmopex.model_builder import build_phy_model, build_nn_model
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


def test_forward_invariants(dummy_batch):
    """Test that counterfactual_supervision preserves exact forward values."""
    cfg = load_config("conf/config_dmopex_interceptE_S0.yaml")
    cfg["mode"] = "train"
    cfg["model"]["phy"]["disable_compile"] = True
    cfg["model"]["phy"]["counterfactual_supervision"] = False

    # Baseline model
    phy_base = LearnedWeightMopexE(cfg["model"]["phy"], device="cpu")
    nn_base = LearnedStructureNet(input_dim=35, hidden_dim=128, nmul=16, device="cpu")
    phy_base.train()
    nn_base.train()

    # CF model
    cfg_cf = copy.deepcopy(cfg)
    cfg_cf["model"]["phy"]["counterfactual_supervision"] = True
    phy_cf = LearnedWeightMopexE(cfg_cf["model"]["phy"], device="cpu")
    nn_cf = LearnedStructureNetCF(input_dim=35, hidden_dim=128, nmul=16, device="cpu")
    phy_cf.load_state_dict(phy_base.state_dict())
    nn_cf.load_state_dict(nn_base.state_dict())
    phy_cf.train()
    nn_cf.train()

    # Forward
    torch.manual_seed(123)
    params_base = nn_base({"c_nn_norm": dummy_batch["c_nn_norm"]})
    torch.manual_seed(123)
    params_cf = nn_cf({"c_nn_norm": dummy_batch["c_nn_norm"]})

    # Logits match
    assert torch.allclose(params_base["weights"], params_cf["weights"], atol=1e-6)
    assert torch.allclose(params_base["params"], params_cf["params"], atol=1e-6)
    assert torch.allclose(params_base["gamma_uh"], params_cf["gamma_uh"], atol=1e-6)

    # Outputs match
    torch.manual_seed(456)
    out_base = phy_base(dummy_batch, params_base)
    torch.manual_seed(456)
    out_cf = phy_cf(dummy_batch, params_cf)

    assert torch.allclose(out_base["streamflow"], out_cf["streamflow"], atol=1e-6)
    for p in ["w_phen", "w_int", "w_snow", "w_sub"]:
        assert torch.allclose(out_base[p], out_cf[p], atol=1e-6)


def test_gradient_invariants_fit_and_aic_detached(dummy_batch):
    """Test that under counterfactual_supervision, fit & AIC give strictly zero gradient to weights_head."""
    cfg = load_config("conf/config_dmopex_interceptE_S0.yaml")
    cfg["mode"] = "train"
    cfg["model"]["phy"]["disable_compile"] = True
    cfg["model"]["phy"]["counterfactual_supervision"] = True

    phy = LearnedWeightMopexE(cfg["model"]["phy"], device="cpu")
    nn = LearnedStructureNetCF(input_dim=35, hidden_dim=128, nmul=16, device="cpu")
    phy.train()
    nn.train()

    params = nn({"c_nn_norm": dummy_batch["c_nn_norm"]})
    out = phy(dummy_batch, params)

    # 1. Compute Fit Loss
    q = out["streamflow"]
    obs = dummy_batch["target"][365:]
    loss_fit = per_basin_fit(q, obs, dummy_batch["std"]).mean()
    loss_fit.backward(retain_graph=True)

    # weights_head must have zero grad from fit loss!
    weights_head_weight = nn.heads["weights"].weight
    weights_head_bias = nn.heads["weights"].bias
    assert weights_head_weight.grad is None or torch.allclose(weights_head_weight.grad, torch.zeros_like(weights_head_weight))
    assert weights_head_bias.grad is None or torch.allclose(weights_head_bias.grad, torch.zeros_like(weights_head_bias))

    # Parameter & routing heads and backbone MUST receive nonzero grad from fit loss!
    params_head_weight = nn.heads["params"].weight
    assert params_head_weight.grad is not None and torch.norm(params_head_weight.grad) > 1e-6
    backbone_layer0_weight = nn.backbone[0].weight
    assert backbone_layer0_weight.grad is not None and torch.norm(backbone_layer0_weight.grad) > 1e-6

    # 2. Verify Direct AIC Loss is completely detached (requires_grad == False)
    nn.zero_grad()
    phy.zero_grad()
    loss_aic = 0.01 * (out["w_phen"].mean() * 2.0 + out["w_int"].mean() * 2.0 + out["w_snow"].mean() * 2.0 + out["w_sub"].mean() * 1.0)
    assert not loss_aic.requires_grad, "AIC loss must not require grad (completely detached from weights_head)"
    assert out["w_int"].grad_fn is None, "w_int output must be detached"


def test_gradient_invariants_l_cf_only_weights_head(dummy_batch):
    """Test that L_CF updates weights_head but gives strictly zero gradient to shared backbone."""
    nn = LearnedStructureNetCF(input_dim=35, hidden_dim=128, nmul=16, device="cpu")
    nn.train()

    # Compute p_struct
    params = nn({"c_nn_norm": dummy_batch["c_nn_norm"]})
    raw_weights = params["weights"]
    logits = raw_weights.view(raw_weights.shape[0], 4, 2)
    p_struct = torch.sigmoid(logits[..., 1] - logits[..., 0])

    # Dummy target q
    q_target = torch.full_like(p_struct, 0.7)
    loss_cf = F.binary_cross_entropy(p_struct, q_target)
    loss_cf.backward()

    # weights_head must have NONZERO grad!
    assert nn.heads["weights"].weight.grad is not None
    assert torch.norm(nn.heads["weights"].weight.grad) > 1e-4

    # Backbone MUST have ZERO grad from L_CF!
    assert nn.backbone[0].weight.grad is None or torch.allclose(nn.backbone[0].weight.grad, torch.zeros_like(nn.backbone[0].weight.grad))
    assert nn.backbone[3].weight.grad is None or torch.allclose(nn.backbone[3].weight.grad, torch.zeros_like(nn.backbone[3].weight.grad))

    # Parameter & routing heads MUST have ZERO grad from L_CF!
    assert nn.heads["params"].weight.grad is None
    assert nn.heads["gamma_uh"].weight.grad is None


def test_target_generator_sanity():
    """Test CounterfactualTargetGenerator output shapes and values."""
    cfg = load_config("conf/config_dmopex_interceptE_S0.yaml")
    cfg["mode"] = "train"
    cfg["model"]["phy"]["disable_compile"] = True

    phy = LearnedWeightMopexE(cfg["model"]["phy"], device="cpu")
    nn = LearnedStructureNet(input_dim=35, hidden_dim=128, nmul=16, device="cpu")

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model_dict = {"LearnedWeightMopexE": torch.nn.Module()}
            self.model_dict["LearnedWeightMopexE"].phy_model = phy
            self.model_dict["LearnedWeightMopexE"].nn_model = nn

    m = DummyModel()
    B, T, n_attr = 10, 730, 35
    torch.manual_seed(42)
    td = {
        "x_phy": torch.randn(T, B, 3).abs() + 0.1,
        "doy": torch.randint(1, 366, (T, B, 1)).float(),
        "xc_nn_norm": torch.randn(1, B, n_attr + 3),
        "target": torch.randn(T, B, 1).abs() + 0.05,
    }

    gen = CounterfactualTargetGenerator(cfg, device="cpu")
    q_tensor, diag = gen.generate_targets(m, td)

    assert q_tensor.shape == (B, 4)
    assert (q_tensor >= 0.0).all() and (q_tensor <= 1.0).all()
    assert not q_tensor.requires_grad

    for proc in ["w_phen", "w_int", "w_snow", "w_sub"]:
        assert proc in diag
        assert diag[proc]["T_scale"] > 0
        assert 0.0 <= diag[proc]["q_mean"] <= 1.0
        assert 0.0 <= diag[proc]["frac_q_gt05"] <= 1.0


def test_default_off_backward_compatibility():
    """Test that when counterfactual_supervision is False, normal behavior is retained."""
    cfg = load_config("conf/config_dmopex_interceptE_S0.yaml")
    cfg["model"]["phy"]["counterfactual_supervision"] = False
    phy = LearnedWeightMopexE(cfg["model"]["phy"], device="cpu")
    assert phy.counterfactual_supervision is False


def test_dual_optimizer_initialization_and_stepping(dummy_batch):
    """Test that CFTrainer properly initializes dual optimizers (Adadelta for primary, Adam for weights_head)."""
    cfg = load_config("conf/config_dmopex_interceptE_S0_r17a.yaml")
    cfg["mode"] = "train"
    cfg["device"] = "cpu"
    cfg["model"]["phy"]["disable_compile"] = True
    cfg["model_dir"] = "/tmp/test_cf_model_dir"

    phy = LearnedWeightMopexE(cfg["delta_model"]["phy_model"], device="cpu")
    nn = LearnedStructureNetCF(input_dim=35, hidden_dim=128, nmul=16, device="cpu")

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
    assert len(trainer.weights_head_params) == 2  # weight and bias
    assert trainer.structure_lr == 0.01

    # Verify that primary optimizer does NOT contain weights_head parameters
    weights_ids = {id(p) for p in trainer.weights_head_params}
    for group in trainer.optimizer.param_groups:
        for p in group["params"]:
            assert id(p) not in weights_ids, "weights_head param found in primary optimizer!"
def test_confidence_weighted_cf_loss_properties():
    """Test mathematical and gradient properties of confidence-weighted CF loss."""
    # 1. Exact confidence formula test: c = 2 * |q - 0.5|
    q = torch.tensor([0.5, 0.0, 1.0, 0.25, 0.75, 0.49, 0.51])
    c = (2.0 * torch.abs(q - 0.5)).detach()

    assert torch.isclose(c[0], torch.tensor(0.0)), "c must be 0 at q=0.5"
    assert torch.isclose(c[1], torch.tensor(1.0)), "c must be 1 at q=0.0"
    assert torch.isclose(c[2], torch.tensor(1.0)), "c must be 1 at q=1.0"
    assert torch.isclose(c[3], torch.tensor(0.5)), "c must be 0.5 at q=0.25"
    assert torch.isclose(c[4], torch.tensor(0.5)), "c must be 0.5 at q=0.75"
    assert (c >= 0.0).all() and (c <= 1.0).all(), "c must be bounded in [0, 1]"
    assert not c.requires_grad, "c must be detached"

    # 2. Process-wise weighted BCE reduction
    B, P = 10, 4
    torch.manual_seed(42)
    p_struct = torch.rand(B, P, requires_grad=True)
    q_batch = torch.rand(B, P)
    c_batch = (2.0 * torch.abs(q_batch - 0.5)).detach()

    bce_elem = F.binary_cross_entropy(p_struct, q_batch, reduction="none")
    sum_c = torch.sum(c_batch, dim=0)
    weighted_bce_per_p = torch.sum(c_batch * bce_elem, dim=0) / (sum_c + 1e-12)
    loss_cf = torch.mean(weighted_bce_per_p)

    assert loss_cf.shape == ()
    assert loss_cf.requires_grad
    loss_cf.backward()
    assert p_struct.grad is not None
    assert torch.all(torch.isfinite(p_struct.grad))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
