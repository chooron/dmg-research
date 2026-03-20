"""CausalTrainer — causal training with effective-cluster holdout evaluation."""

from __future__ import annotations

import logging
import os
import time
from contextlib import nullcontext
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
        self.use_amp = self._should_use_amp(config)
        self.amp_dtype = self._resolve_amp_dtype(config)
        self.use_amp, self.amp_dtype = self._normalize_amp_settings(
            self.use_amp,
            self.amp_dtype,
        )
        self.grad_scaler = (
            torch.amp.GradScaler('cuda')
            if self.use_amp and self.amp_dtype == torch.float16
            else None
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

    @staticmethod
    def _should_use_amp(config: dict[str, Any]) -> bool:
        if not str(config['device']).startswith('cuda') or not torch.cuda.is_available():
            return False
        amp_cfg = config['train'].get('amp', None)
        if amp_cfg is None:
            return bool(getattr(torch.cuda, 'is_bf16_supported', lambda: False)())
        return bool(amp_cfg)

    @staticmethod
    def _resolve_amp_dtype(config: dict[str, Any]) -> torch.dtype:
        amp_dtype = str(config['train'].get('amp_dtype', 'bfloat16')).lower()
        if amp_dtype in ('bf16', 'bfloat16'):
            return torch.bfloat16
        if amp_dtype in ('fp16', 'float16', 'half'):
            return torch.float16
        return torch.float32

    @staticmethod
    def _normalize_amp_settings(
        use_amp: bool,
        amp_dtype: torch.dtype,
    ) -> tuple[bool, torch.dtype]:
        if not use_amp:
            return False, torch.float32

        if not torch.cuda.is_available():
            log.warning("AMP requested but CUDA is unavailable; disabling AMP.")
            return False, torch.float32

        if amp_dtype == torch.bfloat16 and not bool(
            getattr(torch.cuda, 'is_bf16_supported', lambda: False)()
        ):
            log.warning(
                "Requested bfloat16 AMP on %s, but native bf16 is unsupported; "
                "falling back to float16.",
                torch.cuda.get_device_name(0),
            )
            return True, torch.float16

        if amp_dtype == torch.float32:
            return False, torch.float32

        return True, amp_dtype

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(
            device_type='cuda',
            dtype=self.amp_dtype,
        )

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
            with self._autocast_context():
                loss_environments = self._sample_loss_environments(n_timesteps)
                loss = self.loss_func(
                    environments=loss_environments,
                    current_epoch=epoch,
                )

            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
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

    def _sample_loss_environments(self, n_timesteps: int) -> list[tuple]:
        """Sample each training environment and keep only tensors needed for loss."""
        environments = []
        for env_dataset in self.env_train_datasets.values():
            n_env = env_dataset['xc_nn_norm'].shape[1]
            if n_env == 0:
                continue

            env_sample = self.sampler.get_training_sample(env_dataset, n_env, n_timesteps)
            env_params = self.model.nn_model(env_sample['xc_nn_norm'])
            env_pred = self.model.phy_model(env_sample, env_params)
            y_pred_e = env_pred['streamflow'].squeeze(-1)
            y_obs_e = env_sample['target'][-y_pred_e.shape[0]:, :, 0]

            if isinstance(self.loss_func, VRExKgeBatchLoss):
                environments.append((y_pred_e, y_obs_e, int(n_env)))
            else:
                environments.append((y_pred_e, y_obs_e))

        return environments

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
                with self._autocast_context():
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
