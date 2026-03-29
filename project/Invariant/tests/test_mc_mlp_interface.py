from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from implements.causal_dpl_model import CausalDplModel
from models.hbv_static import HbvStatic
from models.nns.mc_mlp import McMlpModel


def _identity_compile(fn, *args, **kwargs):
    return fn


class TestMcMlpInterface(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.nt = 8
        self.nb = 3
        self.nx = 30
        self.ny = 14

        self.xc_nn_norm = torch.rand(self.nt, self.nb, self.nx, dtype=torch.float32)
        prcp = torch.rand(self.nt, self.nb, dtype=torch.float32) * 10.0
        tmean = torch.rand(self.nt, self.nb, dtype=torch.float32) * 20.0 - 2.0
        pet = torch.rand(self.nt, self.nb, dtype=torch.float32) * 5.0
        self.x_phy = torch.stack([prcp, tmean, pet], dim=-1)

        self.mc_model = McMlpModel(
            {
                'hidden_size': 32,
                'dropout': 0.2,
                'output_activation': 'sigmoid',
                'static_pool': 'last',
            },
            nx=self.nx,
            ny=self.ny,
        )

    def _build_hbv(self) -> HbvStatic:
        with patch('torch.compile', _identity_compile):
            return HbvStatic(
                config={
                    'warm_up': 2,
                    'warm_up_states': True,
                    'nmul': 1,
                    'nearzero': 1e-5,
                    'forcings': ['prcp', 'tmean', 'pet'],
                },
                device='cpu',
            )

    def test_mc_mlp_returns_time_major_parameter_tensor(self) -> None:
        parameters = self.mc_model(self.xc_nn_norm)

        self.assertEqual(parameters.shape, (self.nt, self.nb, self.ny))
        self.assertTrue(parameters.is_contiguous())
        torch.testing.assert_close(parameters[0], parameters[-1])
        self.assertGreaterEqual(float(parameters.detach().min()), 0.0)
        self.assertLessEqual(float(parameters.detach().max()), 1.0)

    def test_hbv_static_accepts_mc_mlp_output(self) -> None:
        hbv = self._build_hbv()
        parameters = self.mc_model(self.xc_nn_norm)
        predictions = hbv({'x_phy': self.x_phy}, parameters)

        self.assertEqual(parameters.shape[-1], hbv.learnable_param_count)
        self.assertIn('streamflow', predictions)
        self.assertEqual(predictions['streamflow'].shape, (self.nt - 2, self.nb, 1))
        self.assertFalse(bool(torch.isnan(predictions['streamflow']).any()))

    def test_causal_dpl_model_eval_path_matches_interface(self) -> None:
        hbv = self._build_hbv()
        model = CausalDplModel(
            phy_model=hbv,
            nn_model=self.mc_model,
            config={},
            device='cpu',
        )
        model.eval()

        predictions = model(
            {
                'xc_nn_norm': self.xc_nn_norm,
                'x_phy': self.x_phy,
            },
            eval=True,
        )

        self.assertIsInstance(predictions, dict)
        self.assertIn('streamflow', predictions)
        self.assertEqual(predictions['streamflow'].shape, (self.nt - 2, self.nb, 1))


if __name__ == '__main__':
    unittest.main()
