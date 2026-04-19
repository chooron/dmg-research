from __future__ import annotations

import types
import unittest

import torch

from project.parameterize.implements.distributional_param_trainer import (
    DistributionalParamTrainer,
)
from project.parameterize.implements.param_models import DistributionalParamModel


class _FakeModel:
    def __init__(self, nn_model: DistributionalParamModel) -> None:
        self.nn_model = nn_model

    def __call__(self, dataset_sample):
        target = dataset_sample["target"]
        return {"streamflow": target.clone()}


class TestDistributionalParamTrainer(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5)
        self.param_model = DistributionalParamModel(
            {
                "hidden_size": 16,
                "output_activation": "sigmoid",
                "static_pool": "last",
                "distribution": {"logstd_min": -4.0, "logstd_max": 1.0},
            },
            nx=5,
            ny=14,
        )
        self.trainer = DistributionalParamTrainer.__new__(DistributionalParamTrainer)
        self.trainer.config = {"distribution": {"beta_kl": 1e-3, "kl_warmup_epochs": 10}}
        self.trainer.model = _FakeModel(self.param_model)
        self.trainer.loss_func = lambda y_pred, y_obs: torch.mean((y_pred - y_obs) ** 2)

        self.dataset_sample = {
            "xc_nn_norm": torch.rand(6, 4, 5, dtype=torch.float32),
            "target": torch.rand(6, 4, 1, dtype=torch.float32),
        }

    def test_beta_effective_warmup_scales_linearly(self) -> None:
        self.assertEqual(self.trainer._beta_effective(0), 0.0)
        self.assertAlmostEqual(self.trainer._beta_effective(5), 5e-4)
        self.assertAlmostEqual(self.trainer._beta_effective(10), 1e-3)
        self.assertAlmostEqual(self.trainer._beta_effective(30), 1e-3)

    def test_compute_step_metrics_emits_distributional_logging_fields(self) -> None:
        metrics = self.trainer._compute_step_metrics(self.dataset_sample, epoch=5)
        self.assertIn("loss_total", metrics)
        self.assertIn("loss_hydro", metrics)
        self.assertIn("loss_kl", metrics)
        self.assertIn("beta_effective", metrics)
        self.assertIn("latent_logstd_mean", metrics)
        self.assertIn("latent_logstd_std", metrics)
        self.assertGreaterEqual(float(metrics["loss_kl"]), 0.0)
        self.assertAlmostEqual(float(metrics["beta_effective"]), 5e-4)
        self.assertTrue(torch.isfinite(metrics["latent_logstd_mean"]))
        self.assertTrue(torch.isfinite(metrics["latent_logstd_std"]))


if __name__ == "__main__":
    unittest.main()
