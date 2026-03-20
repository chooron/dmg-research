"""CausalDplModel — DplModel subclass for IRM training."""

from __future__ import annotations

from typing import Optional

import torch

from dmg.models.delta_models.dpl_model import DplModel


class CausalDplModel(DplModel):
    """DplModel extended for IRM training.

    forward accepts an optional ``env_datasets`` list. When provided, runs the
    model on each environment separately and returns per-environment
    ``(y_pred_e, y_obs_e)`` pairs alongside the full prediction.
    """

    def forward(
        self,
        data_dict: dict[str, torch.Tensor],
        env_datasets: Optional[list[dict[str, torch.Tensor]]] = None,
        eval: bool = False,
        return_main_prediction: bool = True,
    ):
        if env_datasets is None or eval:
            parameters = self.nn_model(data_dict['xc_nn_norm'])
            predictions = self.phy_model(data_dict, parameters)
            return predictions

        predictions = None
        if return_main_prediction:
            parameters = self.nn_model(data_dict['xc_nn_norm'])
            predictions = self.phy_model(data_dict, parameters)

        env_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for env_data in env_datasets:
            env_params = self.nn_model(env_data['xc_nn_norm'])
            env_pred = self.phy_model(env_data, env_params)
            y_pred_e = env_pred['streamflow'].squeeze(-1)
            y_obs_e = env_data['target'][-y_pred_e.shape[0]:, :, 0]
            env_pairs.append((y_pred_e, y_obs_e))

        return predictions, env_pairs
