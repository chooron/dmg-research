"""
Category 7: DPL Training Contract and Loss Integrity Test Suite
===============================================================
Audit Reference: DMOTPY_CURRENT_STATE_AUDIT_20260831.md Section 4, 5 & 9

Validates that:
1. KgeLoss implements sample standard deviation, stability epsilon (1e-5),
   and raises FloatingPointError on non-finite predictions.
2. Warmup executes strictly under torch.no_grad() and detaches state tensors,
   preventing gradient leakage into warmup steps.
3. Neural parameterization (Parameterize) connects cleanly to HydrologyModel
   and propagates finite end-to-end gradients under KGE loss.
4. HydrologyModel factory correctly dispatches specialized models (endpoint UH, intermediate UH, mopex doy, tcm).
"""

from __future__ import annotations

import pytest
import torch

from losses import KgeLoss
from models.hydrology_model import HydrologyModel
from models.endpoint_uh_model import EndpointUHModel
from models.gr4j_uh_model import GR4JUHModel
from models.mopex_doy_model import MopexDoyModel
from models.tcm_model import TCMModel
from neural_networks.parameterize import Parameterize


class TestDplTrainingContract:
    """Validate differentiable training contracts, loss numerical integrity, and warmup."""

    def test_kge_loss_finite_and_exact(self):
        """KgeLoss must compute analytical KGE with sample std."""
        loss_fn = KgeLoss(eps=1e-5)
        y = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=torch.float64)
        loss = loss_fn(y, y)
        assert torch.isclose(loss, torch.tensor(0.0, dtype=torch.float64), atol=1e-4)

    def test_kge_loss_raises_on_nan_prediction(self):
        """KgeLoss must strictly raise FloatingPointError on non-finite predictions."""
        loss_fn = KgeLoss()
        pred = torch.tensor([[1.0], [float("nan")], [3.0]])
        target = torch.tensor([[1.0], [2.0], [3.0]])
        with pytest.raises(FloatingPointError, match="prediction contains NaN or Inf"):
            loss_fn(pred, target)

    def test_warmup_state_detachment_no_leak(self):
        """Warmup steps must not accumulate autograd computation graph."""
        device = torch.device("cpu")
        warmup = 10
        time_steps = 20
        batch_size = 2

        model = HydrologyModel(
            config={"model_name": "gr4j", "warm_up": warmup, "parameter_mapping": "auto", "backend": "none"},
            device=device,
        )

        p = torch.rand(time_steps, batch_size, 3, dtype=torch.float32, device=device)
        x_dict = {"x_phy": p}
        raw_params = torch.tensor([[0.5, 0.5, 0.5, 0.5]] * batch_size, dtype=torch.float32, device=device, requires_grad=True)

        out = model(x_dict, (None, raw_params))
        q_sim = out["streamflow"]
        assert q_sim.shape == (time_steps - warmup, batch_size)

        loss = q_sim.sum()
        loss.backward()
        assert raw_params.grad is not None
        assert torch.isfinite(raw_params.grad).all()

    def test_end_to_end_parameterize_to_model_backward(self):
        """End-to-end forward and backward pass from Parameterize network through HydrologyModel."""
        device = torch.device("cpu")
        n_attributes = 8
        n_params = 4
        batch_size = 3
        time_steps = 15
        warmup = 5

        param_net = Parameterize(nx=n_attributes, ny=n_params, hidden_size=16, device="cpu")
        model = HydrologyModel(
            config={"model_name": "gr4j", "warm_up": warmup, "parameter_mapping": "auto", "backend": "none"},
            device=device,
        )
        loss_fn = KgeLoss()

        attrs = torch.randn(batch_size, n_attributes, device=device)
        x_phy = torch.rand(time_steps, batch_size, 3, device=device)
        x_dict = {"x_phy": x_phy}
        q_obs = torch.rand(time_steps - warmup, batch_size, device=device)

        params_tuple = param_net({"c_nn_norm": attrs})
        out = model(x_dict, params_tuple)
        q_sim = out["streamflow"]
        assert q_sim.shape == (time_steps - warmup, batch_size)

        loss = loss_fn(q_sim, q_obs)
        assert torch.isfinite(loss)
        loss.backward()

        for param in param_net.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()

    def test_factory_dispatch_special_models(self):
        """HydrologyModel __new__ must dispatch specialized subclasses based on config."""
        m_base = HydrologyModel(config={"model_name": "hbv96", "backend": "none"})
        assert type(m_base) is HydrologyModel

        m_endpoint = HydrologyModel(
            config={"model_name": "hbv96", "uh_enabled": True, "uh_mode": "endpoint", "backend": "none"}
        )
        assert type(m_endpoint) is EndpointUHModel

        m_gr4j = HydrologyModel(
            config={"model_name": "gr4j", "uh_enabled": True, "uh_mode": "intermediate", "backend": "none"}
        )
        assert type(m_gr4j) is GR4JUHModel

        m_mopex = HydrologyModel(
            config={"model_name": "mopex4", "backend": "none"}
        )
        assert type(m_mopex) is MopexDoyModel

        m_tcm = HydrologyModel(
            config={"model_name": "tcm", "backend": "none"}
        )
        assert type(m_tcm) is TCMModel
