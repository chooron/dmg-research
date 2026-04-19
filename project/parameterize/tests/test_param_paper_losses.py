from __future__ import annotations

import unittest

import torch

from project.parameterize.implements.losses import HybridNseBatchLoss, LogNseBatchLoss


def _base_config() -> dict:
    return {
        "device": "cpu",
        "eps": 1e-6,
        "train": {"loss_function": {"name": "HybridNseBatchLoss"}},
    }


class TestParamPaperLosses(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _base_config()
        self.y_obs = torch.tensor(
            [[0.2, 1.0, 5.0], [0.1, 0.8, 4.0], [0.3, 1.2, 6.0]],
            dtype=torch.float32,
        )
        self.y_pred = torch.tensor(
            [[0.15, 1.1, 4.7], [0.08, 0.7, 4.2], [0.25, 1.0, 5.8]],
            dtype=torch.float32,
        )

    def test_log_nse_batch_loss_is_finite(self) -> None:
        loss_fn = LogNseBatchLoss(self.config, device="cpu")
        loss = loss_fn(y_pred=self.y_pred, y_obs=self.y_obs)
        self.assertTrue(torch.isfinite(loss))

    def test_hybrid_nse_batch_loss_is_finite(self) -> None:
        loss_fn = HybridNseBatchLoss(self.config, device="cpu")
        loss = loss_fn(y_pred=self.y_pred, y_obs=self.y_obs)
        self.assertTrue(torch.isfinite(loss))

    def test_log_nse_emphasizes_low_flow_differences(self) -> None:
        loss_fn = LogNseBatchLoss(self.config, device="cpu")
        better_low_flow = self.y_pred.clone()
        worse_low_flow = self.y_pred.clone()
        better_low_flow[:, 0] = self.y_obs[:, 0]
        worse_low_flow[:, 0] = self.y_obs[:, 0] + 0.5

        loss_better = loss_fn(y_pred=better_low_flow, y_obs=self.y_obs)
        loss_worse = loss_fn(y_pred=worse_low_flow, y_obs=self.y_obs)
        self.assertLess(float(loss_better), float(loss_worse))


if __name__ == "__main__":
    unittest.main()
