"""CausalTrainer — IRM-aware trainer with group-holdout cross-validation."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import numpy as np
import tqdm
from numpy.typing import NDArray

import torch

from dmg.core.utils.utils import save_outputs, save_train_state
from dmg.trainers.trainer import Trainer
from dmg.models.criterion.kge_batch_loss import KgeBatchLoss

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
    """Trainer for Causal-dPL with group-holdout cross-validation.

    Parameters
    ----------
    config : dict
        Standard dmg config dict, extended with:
        - ``causal.cluster_csv``    : path to Gnann cluster CSV
        - ``causal.holdout_group``  : int, group to hold out (1-4)
        - ``causal.use_groups``     : bool (default True)
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
        self._val_loss_func = KgeBatchLoss(config, device=config['device'])

        causal_cfg     = config.get('causal', {})
        basin_ids      = self._load_basin_ids(causal_cfg['basin_ids_path'])
        self.splitter  = GnannEnvironmentSplitter(
            cluster_csv   = causal_cfg['cluster_csv'],
            basin_ids     = basin_ids,
            use_groups    = causal_cfg.get('use_groups', True),
            holdout_group = causal_cfg.get('holdout_group', None),
        )

        # Early stopping state
        es_cfg = config.get('early_stopping', {})
        self.es_patience  = es_cfg.get('patience', 0)   # 0 = disabled
        self.es_min_delta = es_cfg.get('min_delta', 0.0)
        self._es_best     = float('inf')
        self._es_counter  = 0
        self._es_stopped  = False

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
            holdout_group = causal_cfg.get('holdout_group', None)
            if holdout_group is not None:
                log.info(
                    "Holdout group %d: %d basins",
                    holdout_group, len(self.splitter.holdout_indices),
                )

    # ------------------------------------------------------------------
    @staticmethod
    def _load_basin_ids(path: str) -> NDArray:
        if path.endswith('.npy'):
            return np.load(path, allow_pickle=True).astype(int)
        return np.loadtxt(path, dtype=int)

    # ------------------------------------------------------------------
    def _val_loss(self) -> float:
        """Compute mean KGE loss on the validation dataset."""
        if self.eval_dataset is None:
            return float('inf')
        self.model.eval()
        n_samples = self.eval_dataset['xc_nn_norm'].shape[1]
        batch_size = self.config['val']['batch_size']
        batch_start = np.arange(0, n_samples, batch_size)
        batch_end = np.append(batch_start[1:], n_samples)
        total_loss = 0.0
        with torch.no_grad():
            for s, e in zip(batch_start, batch_end):
                sample = self.sampler.get_validation_sample(
                    self.eval_dataset, int(s), int(e),
                )
                pred = self.model(sample, eval=True)
                y_pred = pred['streamflow'].squeeze(-1)
                y_obs = sample['target'][-y_pred.shape[0]:, :, 0]
                total_loss += self._val_loss_func(y_pred=y_pred, y_obs=y_obs).item()
        self.model.train()
        return total_loss / len(batch_start)

    def train(self) -> None:
        """Train with optional early stopping based on validation loss."""
        from dmg.core.data.data import create_training_grid
        self.is_in_train = True
        n_samples, n_minibatch, n_timesteps = create_training_grid(
            self.train_dataset['xc_nn_norm'], self.config,
        )
        log.info(f"Training model: Beginning {self.start_epoch} of {self.epochs} epochs")

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.train_one_epoch(epoch, n_samples, n_minibatch, n_timesteps)

            if self.es_patience > 0:
                val_loss = self._val_loss()
                if val_loss < self._es_best - self.es_min_delta:
                    self._es_best = val_loss
                    self._es_counter = 0
                else:
                    self._es_counter += 1
                    log.info(
                        f"Early stopping counter: {self._es_counter}/{self.es_patience} "
                        f"(val_loss={val_loss:.6f}, best={self._es_best:.6f})"
                    )
                    if self._es_counter >= self.es_patience:
                        log.info(f"Early stopping triggered at epoch {epoch}.")
                        self._es_stopped = True
                        break

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

            predictions, env_pairs = self.model(
                dataset_sample, env_datasets=env_samples,
            )

            loss = self.loss_func(environments=env_pairs, current_epoch=epoch)
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
        self.calc_metrics(all_preds, observations)
        return {}
