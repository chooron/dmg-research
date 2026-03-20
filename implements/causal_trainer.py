"""CausalTrainer — causal training with effective-cluster holdout evaluation."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import numpy as np
import tqdm
from numpy.typing import NDArray

import torch

from dmg.core.calc.metrics import Metrics
from dmg.core.utils.utils import save_outputs, save_train_state
from dmg.trainers.trainer import Trainer

from implements.causal_dpl_model import CausalDplModel
from implements.gnann_splitter import GnannEnvironmentSplitter
from implements.irm_kge_loss import IRMKgeBatchLoss
from implements.vrex_kge_loss import VRExKgeBatchLoss

_LOSS_REGISTRY = {
    'IRMKgeBatchLoss':  IRMKgeBatchLoss,
    'VRExKgeBatchLoss': VRExKgeBatchLoss,
}

log = logging.getLogger(__name__)


class CausalTrainer(Trainer):
    """Trainer for Causal-dPL with effective-cluster leave-one-cluster CV.

    Parameters
    ----------
    config : dict
        Standard dmg config dict, extended with:
        - ``causal.cluster_csv``    : path to Gnann cluster CSV
        - ``causal.holdout_cluster``: str, effective cluster to hold out (A-G)
        - ``causal.basin_ids_path`` : path to gage_id.npy or gage_id.txt
        - ``lambda_irm``            : IRM penalty coefficient
        - ``irm_warmup_epochs``     : epochs before IRM activates
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: CausalDplModel,
        train_dataset: Optional[dict] = None,
        eval_dataset: Optional[dict] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        **kwargs,
    ) -> None:
        if optimizer is None:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config['train']['lr'],
            )
        if scheduler is None:
            lr_cfg = config['train'].get('lr_scheduler', {})
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=lr_cfg.get('T_max', 100),
                eta_min=lr_cfg.get('eta_min', 1e-5),
            )
        loss_name = config['train']['loss_function']['name']
        loss_cls = _LOSS_REGISTRY.get(loss_name)
        if loss_cls is None:
            raise ValueError(f"Unknown loss '{loss_name}'. Available: {list(_LOSS_REGISTRY)}")
        super().__init__(
            config,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss_func=loss_cls(config, device=config['device']),
            optimizer=optimizer,
            scheduler=scheduler,
            **kwargs,
        )
        causal_cfg     = config.get('causal', {})
        basin_ids      = self._load_basin_ids(causal_cfg['basin_ids_path'])
        self.splitter  = GnannEnvironmentSplitter(
            cluster_csv=causal_cfg['cluster_csv'],
            basin_ids=basin_ids,
            holdout_cluster=causal_cfg.get('holdout_cluster', None),
        )

        if train_dataset is not None:
            self.env_train_datasets = self.splitter.split_dataset(train_dataset)
            # Restrict train_dataset to non-holdout basins only
            train_idx = np.concatenate(list(self.splitter.train_env_indices.values()))
            train_idx = np.sort(train_idx)
            self.train_dataset = GnannEnvironmentSplitter._index_dataset(
                train_dataset, train_idx
            )
            log.info(
                "Environments for training: %s",
                {e: len(idx) for e, idx in self.splitter.train_env_indices.items()},
            )
            if len(self.splitter.unassigned_indices) > 0:
                log.info(
                    "Excluded %d basins without effective-cluster labels.",
                    len(self.splitter.unassigned_indices),
                )
            holdout_cluster = causal_cfg.get('holdout_cluster', None)
            if holdout_cluster is not None:
                log.info(
                    "Held-out effective cluster %s: %d basins",
                    holdout_cluster, len(self.splitter.holdout_indices),
                )

    # ------------------------------------------------------------------
    @staticmethod
    def _load_basin_ids(path: str) -> NDArray:
        if path.endswith('.npy'):
            return np.load(path, allow_pickle=True).astype(int)
        return np.loadtxt(path, dtype=int)

    # ------------------------------------------------------------------
    def train(self) -> None:
        """Train for a fixed number of epochs without validation-based early stopping."""
        from dmg.core.data.data import create_training_grid
        self.is_in_train = True
        n_samples, n_minibatch, n_timesteps = create_training_grid(
            self.train_dataset['xc_nn_norm'], self.config,
        )
        log.info(f"Training model: Beginning {self.start_epoch} of {self.epochs} epochs")

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.train_one_epoch(epoch, n_samples, n_minibatch, n_timesteps)

        self.exp_logger.finalize()

    # ------------------------------------------------------------------
    def train_one_epoch(self, epoch, n_samples, n_minibatch, n_timesteps) -> None:
        start_time = time.perf_counter()
        self.current_epoch = epoch
        self.total_loss = 0.0

        prog_bar = tqdm.tqdm(
            range(1, n_minibatch + 1),
            desc=f"Epoch {epoch}/{self.epochs}",
            leave=False,
            dynamic_ncols=True,
        )

        for mb in prog_bar:
            dataset_sample = self.sampler.get_training_sample(
                self.train_dataset, n_samples, n_timesteps,
            )
            env_samples = self._sample_envs(n_timesteps)

            _, env_pairs = self.model(
                dataset_sample, env_datasets=env_samples,
            )
            loss_environments = self._build_loss_environments(env_pairs, env_samples)

            loss = self.loss_func(environments=loss_environments, current_epoch=epoch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.total_loss += loss.item()

        if self.use_scheduler:
            self.scheduler.step()

        self._log_epoch_stats(
            epoch,
            {self.model.phy_model.name: self.total_loss},
            n_minibatch,
            start_time,
        )

        if (epoch % self.config['train']['save_epoch'] == 0) and self.write_out:
            os.makedirs(self.config['model_dir'], exist_ok=True)
            torch.save(
                self.model.state_dict(),
                os.path.join(self.config['model_dir'], f'model_epoch{epoch}.pt'),
            )
            save_train_state(
                self.config['model_dir'],
                epoch=epoch,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                clear_prior=True,
            )

    def _sample_envs(self, n_timesteps: int) -> list[dict[str, torch.Tensor]]:
        env_samples = []
        for env_dataset in self.env_train_datasets.values():
            n_env = env_dataset['xc_nn_norm'].shape[1]
            if n_env == 0:
                continue
            env_samples.append(
                self.sampler.get_training_sample(env_dataset, n_env, n_timesteps)
            )
        return env_samples

    def _build_loss_environments(
        self,
        env_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        env_samples: list[dict[str, torch.Tensor]],
    ) -> list[tuple]:
        """Attach basin counts for VREx, keep IRM format unchanged."""
        if isinstance(self.loss_func, VRExKgeBatchLoss):
            return [
                (y_pred_e, y_obs_e, int(env_sample['xc_nn_norm'].shape[1]))
                for (y_pred_e, y_obs_e), env_sample in zip(env_pairs, env_samples)
            ]
        return env_pairs

    # ------------------------------------------------------------------
    def evaluate_holdout(self) -> dict[str, float]:
        if self.eval_dataset is None:
            raise ValueError("eval_dataset required for holdout evaluation.")

        holdout_data = self.splitter.holdout_dataset(self.eval_dataset)
        n_samples = holdout_data['xc_nn_norm'].shape[1]
        if n_samples == 0:
            log.warning("Holdout dataset is empty.")
            return {}

        batch_size  = self.config['test']['batch_size']
        batch_start = np.arange(0, n_samples, batch_size)
        batch_end   = np.append(batch_start[1:], n_samples)

        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for s, e in zip(batch_start, batch_end):
                sample = self.sampler.get_validation_sample(holdout_data, int(s), int(e))
                pred = self.model(sample, eval=True)
                all_preds.append({k: v.detach().cpu() for k, v in pred.items()})
        self.model.train()

        observations = holdout_data['target']
        save_outputs(self.config, all_preds, observations)
        metrics = self._build_metrics(all_preds, observations)
        metrics.dump_metrics(self.config['output_dir'])
        self._save_holdout_results(metrics)
        return {}

    def _build_metrics(
        self,
        batch_predictions: list[dict[str, torch.Tensor]],
        observations: torch.Tensor,
    ) -> Metrics:
        target_name = self.config['train']['target'][0]
        warm_up = self.config['model'].get('warm_up', 0)
        predictions = self._batch_data(batch_predictions, target_name)
        target = np.expand_dims(observations[:, :, 0].cpu().numpy(), 2)
        target = target[warm_up:, :]

        return Metrics(
            np.swapaxes(predictions.squeeze(), 1, 0),
            np.swapaxes(target.squeeze(), 1, 0),
        )

    def calc_metrics(
        self,
        batch_predictions: list[dict[str, torch.Tensor]],
        observations: torch.Tensor,
    ) -> None:
        metrics = self._build_metrics(batch_predictions, observations)
        metrics.dump_metrics(self.config['output_dir'])

    def _save_holdout_results(self, metrics: Metrics) -> None:
        import pandas as pd

        holdout_cluster = self.splitter.holdout_cluster
        if holdout_cluster is None:
            return

        basin_meta = self.splitter.basin_metadata(self.splitter.holdout_indices)
        basin_count = len(basin_meta['basin_id'])
        if basin_count != len(metrics.kge):
            raise ValueError(
                "Holdout basin metadata and metric lengths do not match: "
                f"{basin_count} vs {len(metrics.kge)}."
            )

        results_df = pd.DataFrame(
            {
                'basin_id': basin_meta['basin_id'],
                'kge': metrics.kge,
                'effective_cluster': basin_meta['effective_cluster'],
                'gauge_cluster': basin_meta['gauge_cluster'],
                'held_out_cluster': holdout_cluster,
                'seed': int(self.config['seed']),
            }
        )

        output_path = os.path.join(
            self.config['output_dir'],
            f"results_held_out_{holdout_cluster}_seed{self.config['seed']}.csv",
        )
        os.makedirs(self.config['output_dir'], exist_ok=True)
        results_df.to_csv(output_path, index=False)
        log.info("Saved held-out basin KGE results to %s", output_path)
