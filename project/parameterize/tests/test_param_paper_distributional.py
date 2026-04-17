from __future__ import annotations

import unittest

import torch

from project.parameterize.paper_variants import DistributionalParamModel


class TestParamPaperDistributional(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.model = DistributionalParamModel(
            {
                "hidden_size": 16,
                "output_activation": "sigmoid",
                "static_pool": "last",
                "distribution": {"logstd_min": -4.0, "logstd_max": 1.0},
            },
            nx=5,
            ny=14,
        )
        self.x = torch.rand(6, 4, 5, dtype=torch.float32)

    def test_distributional_model_exposes_required_heads(self) -> None:
        self.assertTrue(hasattr(self.model, "latent_mu_head"))
        self.assertTrue(hasattr(self.model, "latent_logstd_head"))

    def test_distributional_model_returns_time_major_bounded_tensor(self) -> None:
        parameters, latent_mu, latent_logstd = self.model.sample_parameters_with_stats(
            self.x
        )

        self.assertEqual(parameters.shape, (6, 4, 14))
        self.assertEqual(latent_mu.shape, (6, 4, 14))
        self.assertEqual(latent_logstd.shape, (6, 4, 14))
        self.assertGreaterEqual(float(parameters.detach().min()), 0.0)
        self.assertLessEqual(float(parameters.detach().max()), 1.0)

    def test_distributional_model_supports_backprop_through_sampling(self) -> None:
        parameters = self.model(self.x)
        loss = parameters.mean()
        loss.backward()

        self.assertIsNotNone(self.model.latent_mu_head.weight.grad)
        self.assertIsNotNone(self.model.latent_logstd_head.weight.grad)
        self.assertGreater(
            float(self.model.latent_mu_head.weight.grad.detach().abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(self.model.latent_logstd_head.weight.grad.detach().abs().sum()),
            0.0,
        )

    def test_distributional_model_estimates_parameter_space_moments(self) -> None:
        param_mean, param_std = self.model.estimate_parameter_moments(
            self.x,
            n_samples=8,
        )

        self.assertEqual(param_mean.shape, (6, 4, 14))
        self.assertEqual(param_std.shape, (6, 4, 14))
        self.assertGreaterEqual(float(param_mean.detach().min()), 0.0)
        self.assertLessEqual(float(param_mean.detach().max()), 1.0)
        self.assertGreaterEqual(float(param_std.detach().min()), 0.0)


if __name__ == "__main__":
    unittest.main()
