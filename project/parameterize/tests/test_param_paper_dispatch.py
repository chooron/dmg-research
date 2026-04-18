from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from project.parameterize.paper_variants import build_paper_dpl
from project.parameterize.train_param_paper import _run_model_preflight


def _identity_compile(fn, *args, **kwargs):
    return fn


def _make_config(nn_name: str) -> dict:
    return {
        "device": "cpu",
        "model": {
            "phy": {
                "warm_up": 2,
                "warm_up_states": True,
                "nmul": 1,
                "nearzero": 1e-5,
                "forcings": ["prcp", "tmean", "pet"],
            },
            "nn": {
                "name": nn_name,
                "hidden_size": 32,
                "hidden_layers": 2,
                "dropout": 0.2,
                "output_activation": "sigmoid",
                "static_pool": "last",
                "forcings": [],
                "attributes": ["a", "b", "c", "d"],
                "distribution": {"logstd_min": -5.0, "logstd_max": 2.0},
            },
        },
        "paper": {"variant": "test", "split": "main"},
    }


class TestParamPaperDispatch(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.nt = 7
        self.nb = 3
        self.xc_nn_norm = torch.rand(self.nt, self.nb, 4, dtype=torch.float32)
        self.x_phy = torch.rand(self.nt, self.nb, 3, dtype=torch.float32)

    def _assert_model_runs(self, nn_name: str) -> None:
        with patch("torch.compile", _identity_compile):
            model = build_paper_dpl(_make_config(nn_name))

        parameters = model.nn_model(self.xc_nn_norm)
        self.assertEqual(parameters.shape[0], self.nt)
        self.assertEqual(parameters.shape[1], self.nb)
        self.assertEqual(parameters.shape[2], model.phy_model.learnable_param_count)

        predictions = model(
            {
                "xc_nn_norm": self.xc_nn_norm,
                "x_phy": self.x_phy,
            },
            eval=True,
        )
        self.assertIn("streamflow", predictions)
        self.assertEqual(predictions["streamflow"].shape, (self.nt - 2, self.nb, 1))

    def test_deterministic_variant_dispatch(self) -> None:
        self._assert_model_runs("DeterministicParamModel")

    def test_mc_dropout_variant_dispatch(self) -> None:
        self._assert_model_runs("McMlpModel")

    def test_distributional_variant_dispatch(self) -> None:
        self._assert_model_runs("DistributionalParamModel")

    def test_preflight_runs_on_minimal_training_sample(self) -> None:
        with patch("torch.compile", _identity_compile):
            model = build_paper_dpl(_make_config("DeterministicParamModel"))

        _run_model_preflight(
            model,
            {
                "xc_nn_norm": self.xc_nn_norm,
                "x_phy": self.x_phy,
            },
            {"device": "cpu"},
        )


if __name__ == "__main__":
    unittest.main()
