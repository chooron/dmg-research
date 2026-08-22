"""Tests for the formal deterministic DFlex-CF/BCE path."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import torch.nn.functional as F

from project.flexmopex.models.dflex_weight_mopex import DFlexWeightMopexCF
from project.flexmopex.models.learned_weight_mopex_candidates import LearnedStructureNetPureAttrEncoder


def _batch(batch_size: int = 6, steps: int = 730) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    return {
        "x_phy": torch.rand(steps, batch_size, 3) + 0.1,
        "doy": torch.randint(1, 366, (steps, batch_size, 1)).float(),
        "c_nn_norm": torch.randn(batch_size, 35),
        "target": torch.rand(steps, batch_size, 1) + 0.1,
        "batch_sample": torch.arange(batch_size),
    }


def _models():
    cfg = {
        "variables": ["prcp", "tmean", "pet"],
        "nmul": 16,
        "warm_up": 365,
        "warm_up_states": False,
        "interception_semantics": "S0",
        "counterfactual_supervision": True,
        "disable_compile": True,
        "nearzero": 1e-5,
    }
    phy = DFlexWeightMopexCF(cfg, device="cpu")
    nn = LearnedStructureNetPureAttrEncoder(input_dim=35, hidden_dim=128, nmul=16, device="cpu")
    return phy, nn


def test_dflex_hard_forward_and_gradient_isolation():
    batch = _batch()
    phy, nn = _models()
    phy.train()
    nn.train()
    parameters = nn({"c_nn_norm": batch["c_nn_norm"]})
    output = phy(batch, parameters)

    assert set(torch.unique(output["hard_gates"]).tolist()) <= {0.0, 1.0}
    assert set(torch.unique(output["z_struct"]).tolist()) <= {0.0, 1.0}
    assert torch.allclose(output["hard_gates"], torch.stack([
        output["w_phen"][..., 0],
        output["w_int"][..., 0],
        output["w_snow"][..., 0],
        output["w_sub"][..., 0],
    ], dim=-1))
    assert output["p_struct"].shape[-1] == 4
    assert torch.all((output["p_struct"] >= 0.0) & (output["p_struct"] <= 1.0))

    # Hydrologic fit sees the hard-gate simulator, but cannot update the
    # structure encoder because the CF protocol detaches simulator gates.
    fit_loss = output["streamflow"].square().mean()
    fit_loss.backward()
    structure_grads = [p.grad for p in nn.structure_encoder.parameters()]
    assert all(g is None or torch.allclose(g, torch.zeros_like(g)) for g in structure_grads)
    assert nn.heads["params"].weight.grad is not None
    assert torch.linalg.vector_norm(nn.heads["params"].weight.grad) > 0
    assert nn.heads["gamma_uh"].weight.grad is not None
    assert torch.linalg.vector_norm(nn.heads["gamma_uh"].weight.grad) > 0

    for parameter in list(nn.parameters()) + list(phy.parameters()):
        parameter.grad = None

    # CF/BCE uses the continuous p_struct score and reaches only the pure
    # structure branch; hydrologic encoder/head gradients remain zero.
    q = torch.full_like(output["p_struct"][0], 0.8)
    bce = F.binary_cross_entropy(output["p_struct"][0], q)
    bce.backward()
    assert all(
        p.grad is not None and torch.linalg.vector_norm(p.grad) > 0
        for p in nn.structure_encoder.parameters()
        if p.requires_grad
    )
    assert all(p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad)) for p in nn.backbone.parameters())
    assert nn.heads["params"].weight.grad is None
    assert nn.heads["gamma_uh"].weight.grad is None


def test_dflex_does_not_use_legacy_l0_attributes():
    phy, _ = _models()
    assert not hasattr(phy, "_p_nonzero")
    assert not hasattr(phy, "temperature")
    assert not hasattr(phy, "_last_log_alpha")
    assert phy.is_dflex is True
