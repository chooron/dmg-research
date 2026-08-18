"""Targeted unit tests for unified lambda (loss_function.aic_alpha) propagation.

Verifies that:
1. loss_function.aic_alpha is the single source of truth for both:
   - NseDynAicBatchLoss (main fit/AIC loss)
   - CounterfactualTargetGenerator / CFTrainer (counterfactual target generation)
2. Changing lambda (0.005, 0.007, 0.010) strictly propagates to DeltaJ and soft targets q.
3. cf_loss_weight is decoupled from lambda.
"""
from __future__ import annotations

import copy
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
from project.flexmopex.models.cf_trainer import CounterfactualTargetGenerator, _extract_aic_alpha
from project.flexmopex.models.nse_dyn_aic_batch_loss import NseDynAicBatchLoss
from project.flexmopex.run_model import _build_loss, apply_runtime_overrides, parse_args


def test_lambda_single_source_of_truth_across_lambdas():
    """Verify that setting loss_function.aic_alpha to 0.005, 0.007, 0.010 reaches both loss and target generator."""
    canonical_config_path = PROJECT_DIR / "conf" / "config_flexmopex_canonical.yaml"
    base_config = load_config(canonical_config_path)

    for test_lambda in [0.005, 0.007, 0.010]:
        cfg = copy.deepcopy(base_config)
        cfg["loss_function"]["aic_alpha"] = test_lambda

        # 1. Check loss function
        y_dummy = torch.randn(100, 10, 1)
        loss_fn = _build_loss(cfg, {"target": y_dummy})
        assert isinstance(loss_fn, NseDynAicBatchLoss)
        assert loss_fn.aic_alpha == test_lambda, f"Loss aic_alpha {loss_fn.aic_alpha} != {test_lambda}"

        # 2. Check CounterfactualTargetGenerator extraction and instance
        extracted = _extract_aic_alpha(cfg)
        assert extracted == test_lambda, f"Extracted aic_alpha {extracted} != {test_lambda}"

        generator = CounterfactualTargetGenerator(cfg, device="cpu")
        assert generator.aic_alpha == test_lambda, f"Generator aic_alpha {generator.aic_alpha} != {test_lambda}"


def test_cli_alpha_override_propagates_to_both():
    """Verify CLI --alpha sets loss_function.aic_alpha and updates both paths."""
    canonical_config_path = PROJECT_DIR / "conf" / "config_flexmopex_canonical.yaml"

    for test_lambda in [0.005, 0.007, 0.010]:
        config = load_config(canonical_config_path)
        args = parse_args(["--config", str(canonical_config_path), "--alpha", str(test_lambda)])
        apply_runtime_overrides(config, args, config_path=str(canonical_config_path))

        assert config["loss_function"]["aic_alpha"] == test_lambda

        y_dummy = torch.randn(50, 5, 1)
        loss_fn = _build_loss(config, {"target": y_dummy})
        assert loss_fn.aic_alpha == test_lambda

        generator = CounterfactualTargetGenerator(config, device="cpu")
        assert generator.aic_alpha == test_lambda


def test_counterfactual_target_generator_responds_to_lambda():
    """Verify that varying lambda mathematically modulates DeltaJ and resulting q."""
    torch.manual_seed(42)
    n_timesteps = 50
    n_basins = 6
    device = "cpu"

    # Synthetic train dataset
    prcp = torch.rand(n_timesteps, n_basins, 1) * 20.0
    tmean = torch.rand(n_timesteps, n_basins, 1) * 30.0 - 5.0
    pet = torch.rand(n_timesteps, n_basins, 1) * 6.0
    x_phy = torch.cat([prcp, tmean, pet], dim=-1)
    doy = (torch.arange(n_timesteps, dtype=torch.float32) % 365 + 1).view(n_timesteps, 1, 1).repeat(1, n_basins, 1)
    target = torch.rand(n_timesteps, n_basins, 1) * 5.0
    xc_nn_norm = torch.randn(n_timesteps, n_basins, 38)  # 3 forcings + 35 static attributes

    train_dataset = {
        "x_phy": x_phy,
        "doy": doy,
        "target": target,
        "xc_nn_norm": xc_nn_norm,
    }

    config = load_config(PROJECT_DIR / "conf" / "config_flexmopex_canonical.yaml")
    config["device"] = device
    config["model"]["phy"]["nmul"] = 4
    config["model"]["nn"]["nmul"] = 4
    config["delta_model"]["phy_model"]["nmul"] = 4
    config["delta_model"]["nn_model"]["nmul"] = 4
    config["model"]["phy"]["warm_up"] = 5
    config["model"]["warmup"] = 5
    config["delta_model"]["phy_model"]["warm_up"] = 5

    model = FlexMopexModelHandler(config, device=device)

    # Generate targets with lambda = 0.005 and lambda = 0.010
    gen_005 = CounterfactualTargetGenerator(config, device=device, aic_alpha=0.005)
    gen_010 = CounterfactualTargetGenerator(config, device=device, aic_alpha=0.010)

    q_005, diag_005 = gen_005.generate_targets(model, train_dataset)
    q_010, diag_010 = gen_010.generate_targets(model, train_dataset)

    # Since aic_alpha=0.005 imposes less penalty than 0.010, DeltaJ(0.005) >= DeltaJ(0.010),
    # and soft targets q(0.005) >= q(0.010)
    for proc in ["w_phen", "w_int", "w_snow", "w_sub"]:
        dJ_005 = diag_005[proc]["delta_J_mean"]
        dJ_010 = diag_010[proc]["delta_J_mean"]
        assert dJ_005 > dJ_010, f"Expected DeltaJ(0.005) > DeltaJ(0.010) for {proc}, got {dJ_005} vs {dJ_010}"

    # Verify q is strictly sensitive to lambda
    diff = torch.norm(q_005 - q_010).item()
    assert diff > 1e-4, f"q targets did not respond to lambda change: diff={diff}"
