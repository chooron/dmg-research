from __future__ import annotations

import unittest

import numpy as np
import torch

from project.parameterize.implements.losses import (
    HybridNseBatchLoss,
    LogNseBatchLoss,
    NseBatchLoss,
)


def _base_config() -> dict:
    return {
        "device": "cpu",
        "eps": 1e-6,
        "log_nse_eps": 1e-6,
        "train": {"loss_function": {"name": "HybridNseBatchLoss"}},
    }


def _manual_batch_scaled_loss(
    y_pred: torch.Tensor,
    y_obs: torch.Tensor,
    reference: np.ndarray,
    sample_ids: np.ndarray,
    eps: float,
) -> float:
    std = np.nanstd(reference, axis=0)
    tiled_std = np.tile(std[sample_ids].T, (y_obs.shape[0], 1))
    pred_np = y_pred.detach().cpu().numpy()
    obs_np = y_obs.detach().cpu().numpy()
    mask = ~np.isnan(obs_np)
    sq_res = (pred_np[mask] - obs_np[mask]) ** 2
    norm_res = sq_res / (tiled_std[mask] + eps) ** 2
    return float(np.mean(norm_res))


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
        self.full_y_obs = self.y_obs.unsqueeze(-1)
        self.sample_ids = np.array([0, 1, 2], dtype=int)

    def test_nse_batch_loss_matches_official_batch_style(self) -> None:
        loss_fn = NseBatchLoss(self.config, device="cpu", y_obs=self.full_y_obs)
        loss = loss_fn(y_pred=self.y_pred, y_obs=self.y_obs, sample_ids=self.sample_ids)
        expected = _manual_batch_scaled_loss(
            self.y_pred,
            self.y_obs,
            self.full_y_obs[:, :, 0].numpy(),
            self.sample_ids,
            eps=self.config["eps"],
        )
        self.assertAlmostEqual(float(loss), expected, places=6)

    def test_log_nse_batch_loss_matches_log_scaled_manual_formula(self) -> None:
        loss_fn = LogNseBatchLoss(self.config, device="cpu", y_obs=self.full_y_obs)
        loss = loss_fn(y_pred=self.y_pred, y_obs=self.y_obs, sample_ids=self.sample_ids)

        log_eps = self.config["log_nse_eps"]
        expected = _manual_batch_scaled_loss(
            torch.log(torch.clamp(self.y_pred, min=0.0) + log_eps),
            torch.log(torch.clamp(self.y_obs, min=0.0) + log_eps),
            np.log(np.clip(self.full_y_obs[:, :, 0].numpy(), a_min=0.0, a_max=None) + log_eps),
            self.sample_ids,
            eps=self.config["eps"],
        )
        self.assertAlmostEqual(float(loss), expected, places=6)

    def test_hybrid_nse_batch_loss_matches_weighted_components(self) -> None:
        loss_fn = HybridNseBatchLoss(self.config, device="cpu", y_obs=self.full_y_obs)
        nse_loss = NseBatchLoss(self.config, device="cpu", y_obs=self.full_y_obs)(
            y_pred=self.y_pred,
            y_obs=self.y_obs,
            sample_ids=self.sample_ids,
        )
        log_loss = LogNseBatchLoss(self.config, device="cpu", y_obs=self.full_y_obs)(
            y_pred=self.y_pred,
            y_obs=self.y_obs,
            sample_ids=self.sample_ids,
        )
        hybrid_loss = loss_fn(
            y_pred=self.y_pred,
            y_obs=self.y_obs,
            sample_ids=self.sample_ids,
        )
        expected = 0.5 * float(nse_loss) + 0.5 * float(log_loss)
        self.assertAlmostEqual(float(hybrid_loss), expected, places=6)

    def test_log_nse_emphasizes_low_flow_differences(self) -> None:
        loss_fn = LogNseBatchLoss(self.config, device="cpu", y_obs=self.full_y_obs)
        better_low_flow = self.y_pred.clone()
        worse_low_flow = self.y_pred.clone()
        better_low_flow[:, 0] = self.y_obs[:, 0]
        worse_low_flow[:, 0] = self.y_obs[:, 0] + 0.5

        loss_better = loss_fn(
            y_pred=better_low_flow,
            y_obs=self.y_obs,
            sample_ids=self.sample_ids,
        )
        loss_worse = loss_fn(
            y_pred=worse_low_flow,
            y_obs=self.y_obs,
            sample_ids=self.sample_ids,
        )
        self.assertLess(float(loss_better), float(loss_worse))


if __name__ == "__main__":
    unittest.main()
