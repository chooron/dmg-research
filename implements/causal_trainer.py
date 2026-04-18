"""CausalTrainer — causal training with effective-cluster holdout evaluation."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import tqdm
from numpy.typing import NDArray

import torch
import torch.nn as nn

from dmg.core.calc.metrics import Metrics
from dmg.core.utils.utils import save_train_state
from dmg.trainers.trainer import Trainer

from implements.causal_dpl_model import CausalDplModel
from implements.gnann_splitter import GnannEnvironmentSplitter
from implements.hybrid_nse_batch_loss import HybridNseBatchLoss
from implements.log_nse_batch_loss import LogNseBatchLoss
from implements.vrex_kge_loss import VRExKgeBatchLoss
from dmg.models.criterion.kge_batch_loss import KgeBatchLoss

try:
    from implements.irm_kge_loss import IRMKgeBatchLoss
except ModuleNotFoundError:
    IRMKgeBatchLoss = None

_LOSS_REGISTRY = {
    'VRExKgeBatchLoss': VRExKgeBatchLoss,
    'KgeBatchLoss': KgeBatchLoss,
    'LogNseBatchLoss': LogNseBatchLoss,
    'HybridNseBatchLoss': HybridNseBatchLoss,
}

if IRMKgeBatchLoss is not None:
    _LOSS_REGISTRY['IRMKgeBatchLoss'] = IRMKgeBatchLoss

_MODEL_CHECKPOINT_RE = re.compile(r'model_epoch(\d+)\.pt$')
_TRAINER_STATE_RE = re.compile(r'trainer_state_ep(\d+)\.pt$')

log = logging.getLogger(__name__)


def _resolve_loss_class(loss_name: str):
    loss_cls = _LOSS_REGISTRY.get(loss_name)
    if loss_cls is not None:
        return loss_cls

    if loss_name == 'IRMKgeBatchLoss' and IRMKgeBatchLoss is None:
        raise ValueError(
            "Loss 'IRMKgeBatchLoss' is configured, but "
            "'implements/irm_kge_loss.py' has been removed. "
            "Update train.loss_function.name to a supported loss."
        )

    raise ValueError(
        f"Unknown loss '{loss_name}'. Available: {list(_LOSS_REGISTRY)}"
    )


def _parse_env_bool(name: str) -> Optional[bool]:
    value = os.environ.get(name)
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    return None


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
        if 'train' in config['mode']:
            self._configure_training_resume(config)

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
        loss_cls = _resolve_loss_class(loss_name)
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

        test_cfg = config.get('test', {})
        self.mc_samples = max(int(test_cfg.get('mc_samples', 100)), 1)
        self.mc_selection_metric = str(
            test_cfg.get('mc_selection_metric', 'mse')
        ).lower()
        self.enable_mc_dropout = bool(test_cfg.get('mc_dropout', True))
        self.mc_seed_base = int(test_cfg.get('mc_seed_base', config['seed']))
        self.full_train_dataset = train_dataset

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
    def _find_latest_epoch_file(
        directory: str,
        pattern: re.Pattern[str],
    ) -> tuple[int, str | None]:
        latest_epoch = -1
        latest_path = None

        if not os.path.isdir(directory):
            return latest_epoch, latest_path

        for file_name in os.listdir(directory):
            match = pattern.fullmatch(file_name)
            if match is None:
                continue

            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_path = os.path.join(directory, file_name)

        return latest_epoch, latest_path

    def _configure_training_resume(self, config: dict[str, Any]) -> bool:
        target_epochs = int(config['train']['epochs'])
        latest_model_epoch, latest_model_path = self._find_latest_epoch_file(
            config['model_dir'],
            _MODEL_CHECKPOINT_RE,
        )

        if latest_model_path is None:
            return False

        config['train']['start_epoch'] = latest_model_epoch
        completed = latest_model_epoch >= target_epochs
        if completed:
            log.info(
                "Found existing checkpoint at epoch %d (target %d); training will be skipped.",
                latest_model_epoch,
                target_epochs,
            )
        else:
            log.info(
                "Found existing checkpoint at epoch %d/%d; resuming from epoch %d.",
                latest_model_epoch,
                target_epochs,
                latest_model_epoch + 1,
            )
        return completed

    def load_states(self) -> None:
        """Load model and trainer states from the latest discovered checkpoint."""
        resume_epoch = self.start_epoch - 1
        if resume_epoch < 1:
            return

        model_path = os.path.join(self.config['model_dir'], f'model_epoch{resume_epoch}.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model checkpoint for epoch {resume_epoch}.")

        log.info("Loading model checkpoint from epoch %d", resume_epoch)
        model_state = torch.load(model_path, map_location=self.config['device'])
        self.model.load_state_dict(model_state)

        trainer_state_path = os.path.join(
            self.config['model_dir'],
            f'trainer_state_ep{resume_epoch}.pt',
        )
        if not os.path.exists(trainer_state_path):
            if resume_epoch < self.epochs:
                log.warning(
                    "Found model checkpoint at epoch %d but no matching trainer state; "
                    "resuming from weights only.",
                    resume_epoch,
                )
            return

        checkpoint = torch.load(trainer_state_path, map_location='cpu')
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        scheduler_state = checkpoint.get('scheduler_state_dict')
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        random_state = checkpoint.get('random_state')
        if random_state is not None:
            torch.set_rng_state(random_state)

        if str(self.config['device']).startswith('cuda') and torch.cuda.is_available():
            cuda_state = checkpoint.get('cuda_state')
            if cuda_state is None:
                cuda_state = checkpoint.get('cuda_random_state')
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, device=self.config['device'])

    def _save_checkpoint(self, epoch: int, clear_prior: bool) -> None:
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
            clear_prior=clear_prior,
        )

    def _save_final_checkpoint_if_needed(self) -> None:
        final_epoch = int(self.epochs)
        model_path = os.path.join(self.config['model_dir'], f'model_epoch{final_epoch}.pt')
        trainer_state_path = os.path.join(
            self.config['model_dir'],
            f'trainer_state_ep{final_epoch}.pt',
        )
        if os.path.exists(model_path) and os.path.exists(trainer_state_path):
            return

        os.makedirs(self.config['model_dir'], exist_ok=True)
        if not os.path.exists(model_path):
            torch.save(self.model.state_dict(), model_path)
        if not os.path.exists(trainer_state_path):
            save_train_state(
                self.config['model_dir'],
                epoch=final_epoch,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                clear_prior=False,
            )
        log.info("Saved final checkpoint for epoch %d to %s", final_epoch, self.config['model_dir'])

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

    def _should_disable_progress(self) -> bool:
        env_override = _parse_env_bool('TQDM_DISABLE')
        if env_override is not None:
            return env_override

        env_override = _parse_env_bool('DMG_DISABLE_TQDM')
        if env_override is not None:
            return env_override

        return not sys.stderr.isatty()

    def _progress(self, iterable, **kwargs):
        disable = kwargs.pop('disable', self._should_disable_progress())
        return tqdm.tqdm(
            iterable,
            disable=disable,
            dynamic_ncols=not disable,
            file=sys.stderr,
            **kwargs,
        )

    # ------------------------------------------------------------------
    def train(self) -> None:
        """Train for a fixed number of epochs without validation-based early stopping."""
        from dmg.core.data.data import create_training_grid
        self.is_in_train = True

        if self.start_epoch > self.epochs:
            log.info(
                "Skipping training because checkpoint already reached epoch %d.",
                self.start_epoch - 1,
            )
            self.exp_logger.finalize()
            return

        n_samples, n_minibatch, n_timesteps = create_training_grid(
            self.train_dataset['xc_nn_norm'], self.config,
        )
        log.info(f"Training model: Beginning {self.start_epoch} of {self.epochs} epochs")

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.train_one_epoch(epoch, n_samples, n_minibatch, n_timesteps)

        if self.write_out:
            self._save_final_checkpoint_if_needed()

        self.exp_logger.finalize()

    # ------------------------------------------------------------------
    def train_one_epoch(self, epoch, n_samples, n_minibatch, n_timesteps) -> None:
        start_time = time.perf_counter()
        self.current_epoch = epoch
        self.total_loss = 0.0

        prog_bar = self._progress(
            range(1, n_minibatch + 1),
            desc=f"Epoch {epoch}/{self.epochs}",
            leave=False,
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
            self._save_checkpoint(epoch, clear_prior=True)

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
    def evaluate(self) -> None:
        if self.eval_dataset is None:
            raise ValueError('eval_dataset required for evaluation.')
        self.is_in_train = False
        self._evaluate_dataset(
            eval_data=self.eval_dataset,
            reference_data=self.full_train_dataset,
            holdout=False,
        )

    def evaluate_holdout(self) -> dict[str, float]:
        if self.eval_dataset is None:
            raise ValueError('eval_dataset required for holdout evaluation.')

        holdout_data = self.splitter.holdout_dataset(self.eval_dataset)
        n_samples = holdout_data['xc_nn_norm'].shape[1]
        if n_samples == 0:
            log.warning('Holdout dataset is empty.')
            return {}

        reference_data = None
        if self.full_train_dataset is not None:
            reference_data = self.splitter.holdout_dataset(self.full_train_dataset)

        self.is_in_train = False
        self._evaluate_dataset(
            eval_data=holdout_data,
            reference_data=reference_data,
            holdout=True,
        )
        return {}

    def _evaluate_dataset(
        self,
        eval_data: dict[str, torch.Tensor],
        reference_data: Optional[dict[str, torch.Tensor]] = None,
        holdout: bool = False,
    ) -> Metrics | None:
        n_samples = eval_data['xc_nn_norm'].shape[1]
        if n_samples == 0:
            log.warning('Evaluation dataset is empty.')
            return None

        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(self.config['sim_dir'], exist_ok=True)

        seeds = self._mc_seeds()
        best_pass_idx = None
        best_scores = None

        if reference_data is not None:
            ref_n_samples = reference_data['xc_nn_norm'].shape[1]
            if ref_n_samples != n_samples:
                log.warning(
                    'Skipping MC pass selection because reference/eval basin counts differ: %d vs %d.',
                    ref_n_samples,
                    n_samples,
                )
            else:
                best_pass_idx, best_scores = self._select_best_passes_from_reference(
                    reference_data,
                    seeds,
                )

        target_name = self.config['train']['target'][0]
        obs_array = eval_data['target'][:, :, 0].cpu().numpy()
        mc_path = os.path.join(self.config['sim_dir'], f'{target_name}_mc.npy')

        mean_accumulator = None
        mc_memmap = None
        selected_prediction = None
        mc_raw_metrics = []
        mc_agg_metrics = []

        eval_iter = self._progress(
            enumerate(self._forward_mc_predictions(eval_data, seeds)),
            total=len(seeds),
            desc='MC eval',
            leave=False,
        )
        for pass_idx, batch_predictions in eval_iter:
            prediction = self._batch_data(batch_predictions, target_name).astype(
                np.float32,
                copy=False,
            )

            if mc_memmap is None:
                mc_shape = (len(seeds),) + prediction.shape
                mc_memmap = np.lib.format.open_memmap(
                    mc_path,
                    mode='w+',
                    dtype=np.float32,
                    shape=mc_shape,
                )
                mean_accumulator = np.zeros_like(prediction, dtype=np.float64)
                if best_pass_idx is not None:
                    selected_prediction = np.empty_like(prediction)

            mc_memmap[pass_idx] = prediction
            mean_accumulator += prediction

            metrics = self._build_metrics_from_prediction_array(
                prediction,
                eval_data['target'],
            )
            mc_raw_metrics.append(
                {
                    'pass_index': pass_idx,
                    'metrics': self._serialize_metrics(metrics),
                }
            )
            mc_agg_metrics.append(
                {
                    'pass_index': pass_idx,
                    'agg_stats': self._jsonify(metrics.calc_stats()),
                }
            )

            if best_pass_idx is not None and selected_prediction is not None:
                basin_mask = best_pass_idx == pass_idx
                if np.any(basin_mask):
                    selected_prediction[:, basin_mask, ...] = prediction[:, basin_mask, ...]

        if mc_memmap is None or mean_accumulator is None:
            log.warning('No MC predictions were produced during evaluation.')
            return None

        mc_memmap.flush()
        mean_prediction = (mean_accumulator / len(seeds)).astype(np.float32)
        np.save(os.path.join(self.config['sim_dir'], f'{target_name}.npy'), mean_prediction)
        np.save(os.path.join(self.config['sim_dir'], f'{target_name}_obs.npy'), obs_array)
        log.info('Saved MC prediction stack to %s', mc_path)

        mean_metrics = self._build_metrics_from_prediction_array(
            mean_prediction,
            eval_data['target'],
        )
        self._write_metrics_files(mean_metrics, prefix='metrics')
        self._write_mc_metrics(mc_raw_metrics, mc_agg_metrics)

        if selected_prediction is not None and best_scores is not None:
            np.save(
                os.path.join(self.config['sim_dir'], f'{target_name}_mc_selected.npy'),
                selected_prediction,
            )
            selected_metrics = self._build_metrics_from_prediction_array(
                selected_prediction,
                eval_data['target'],
            )
            self._write_metrics_files(selected_metrics, prefix='metrics_avg')
            self._write_selection_artifacts(best_pass_idx, best_scores)
            if holdout:
                self._save_holdout_results(selected_metrics, file_suffix='_avg')
        else:
            log.warning(
                'Skipping metrics_avg because no aligned reference dataset was available for MC pass selection.'
            )

        if holdout:
            self._save_holdout_results(mean_metrics)

        return mean_metrics

    def _mc_seeds(self) -> list[int]:
        return [self.mc_seed_base + i for i in range(self.mc_samples)]

    def _seed_mc_pass(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _forward_mc_predictions(
        self,
        data: dict[str, torch.Tensor],
        seeds: list[int],
    ):
        batch_size = self.config['test']['batch_size']
        n_samples = data['xc_nn_norm'].shape[1]
        batch_start = np.arange(0, n_samples, batch_size)
        batch_end = np.append(batch_start[1:], n_samples)

        was_training = self.model.training
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = None
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state_all()

        self.model.eval()
        if self.enable_mc_dropout and self.mc_samples > 1:
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.train(True)

        try:
            with torch.no_grad():
                for seed in seeds:
                    self._seed_mc_pass(seed)
                    yield self._forward_single_pass(data, batch_start, batch_end)
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            self.model.train(was_training)

    def _forward_single_pass(
        self,
        data: dict[str, torch.Tensor],
        batch_start: NDArray,
        batch_end: NDArray,
    ) -> list[dict[str, torch.Tensor]]:
        batch_predictions = []
        for start, end in zip(batch_start, batch_end):
            dataset_sample = self.sampler.get_validation_sample(
                data,
                int(start),
                int(end),
            )
            with self._autocast_context():
                prediction = self.model(dataset_sample, eval=True)
            batch_predictions.append(
                {key: tensor.detach().cpu() for key, tensor in prediction.items()}
            )
        return batch_predictions

    def _select_best_passes_from_reference(
        self,
        reference_data: dict[str, torch.Tensor],
        seeds: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        score_matrix = None
        ref_iter = self._progress(
            enumerate(self._forward_mc_predictions(reference_data, seeds)),
            total=len(seeds),
            desc='MC ref',
            leave=False,
        )
        target_name = self.config['train']['target'][0]
        for pass_idx, batch_predictions in ref_iter:
            prediction = self._batch_data(batch_predictions, target_name).astype(
                np.float32,
                copy=False,
            )
            scores = self._compute_selection_scores(
                prediction,
                reference_data['target'],
            )
            if score_matrix is None:
                score_matrix = np.full(
                    (len(seeds), scores.shape[0]),
                    np.nan,
                    dtype=np.float32,
                )
            score_matrix[pass_idx] = scores.astype(np.float32, copy=False)

        if score_matrix is None:
            raise RuntimeError('Reference MC selection produced no scores.')
        return self._select_best_passes(score_matrix)

    def _compute_selection_scores(
        self,
        prediction: np.ndarray,
        observations: torch.Tensor,
    ) -> np.ndarray:
        metric = self.mc_selection_metric
        if metric == 'kge':
            metrics = self._build_metrics_from_prediction_array(prediction, observations)
            return metrics.kge.astype(np.float32, copy=False)

        pred = self._prepare_prediction_array(prediction)
        target = self._prepare_target_array(observations)
        sq_error = (pred - target) ** 2
        mse = np.nanmean(sq_error, axis=0)

        if metric in {'loss', 'mse'}:
            return mse.astype(np.float32, copy=False)
        if metric == 'rmse':
            return np.sqrt(mse).astype(np.float32, copy=False)
        raise ValueError(
            f"Unsupported mc_selection_metric '{metric}'. Expected mse, rmse, loss, or kge."
        )

    def _select_best_passes(
        self,
        score_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        metric = self.mc_selection_metric
        n_passes, n_basins = score_matrix.shape
        best_idx = np.zeros(n_basins, dtype=np.int64)
        best_scores = np.full(n_basins, np.nan, dtype=np.float32)
        valid_mask = ~np.all(np.isnan(score_matrix), axis=0)

        if np.any(valid_mask):
            basin_idx = np.where(valid_mask)[0]
            if metric == 'kge':
                filled = np.where(np.isnan(score_matrix[:, basin_idx]), -np.inf, score_matrix[:, basin_idx])
                best_idx[basin_idx] = np.argmax(filled, axis=0)
            else:
                filled = np.where(np.isnan(score_matrix[:, basin_idx]), np.inf, score_matrix[:, basin_idx])
                best_idx[basin_idx] = np.argmin(filled, axis=0)
            best_scores[basin_idx] = score_matrix[best_idx[basin_idx], basin_idx]

        if np.any(~valid_mask):
            log.warning(
                'MC selection metric is NaN for %d basins; defaulting those basins to pass 0.',
                int((~valid_mask).sum()),
            )

        if n_passes == 0:
            raise RuntimeError('No Monte Carlo passes were available for selection.')
        return best_idx, best_scores

    def _build_metrics(
        self,
        batch_predictions: list[dict[str, torch.Tensor]],
        observations: torch.Tensor,
    ) -> Metrics:
        target_name = self.config['train']['target'][0]
        predictions = self._batch_data(batch_predictions, target_name)
        return self._build_metrics_from_prediction_array(predictions, observations)

    def _build_metrics_from_prediction_array(
        self,
        predictions: np.ndarray,
        observations: torch.Tensor,
    ) -> Metrics:
        pred = self._prepare_prediction_array(predictions)
        target = self._prepare_target_array(observations)
        return Metrics(
            np.swapaxes(pred, 1, 0),
            np.swapaxes(target, 1, 0),
        )

    def calc_metrics(
        self,
        batch_predictions: list[dict[str, torch.Tensor]],
        observations: torch.Tensor,
    ) -> None:
        metrics = self._build_metrics(batch_predictions, observations)
        self._write_metrics_files(metrics, prefix='metrics')

    def _prepare_target_array(self, observations: torch.Tensor) -> np.ndarray:
        warm_up = self.config['model'].get('warm_up', 0)
        target = observations[:, :, 0].cpu().numpy()
        target = target[warm_up:, :]
        if target.ndim == 1:
            target = target[:, np.newaxis]
        return target

    @staticmethod
    def _prepare_prediction_array(predictions: np.ndarray) -> np.ndarray:
        pred = np.asarray(predictions)
        if pred.ndim == 3 and pred.shape[-1] == 1:
            pred = pred[:, :, 0]
        if pred.ndim == 1:
            pred = pred[:, np.newaxis]
        if pred.ndim != 2:
            raise ValueError(
                f"Expected prediction array with shape [T, B] or [T, B, 1], got {pred.shape}."
            )
        return pred

    def _write_metrics_files(self, metrics: Metrics, prefix: str) -> None:
        raw_path = os.path.join(self.config['output_dir'], f'{prefix}.json')
        agg_path = os.path.join(self.config['output_dir'], f'{prefix}_agg.json')
        with open(raw_path, 'w') as f:
            json.dump(self._serialize_metrics(metrics), f, indent=4)
        with open(agg_path, 'w') as f:
            json.dump(self._jsonify(metrics.calc_stats()), f, indent=4)

    def _write_mc_metrics(
        self,
        raw_metrics: list[dict[str, Any]],
        agg_metrics: list[dict[str, Any]],
    ) -> None:
        raw_path = os.path.join(self.config['output_dir'], 'metrics_mc.json')
        agg_path = os.path.join(self.config['output_dir'], 'metrics_mc_agg.json')
        raw_payload = {
            'mc_samples': self.mc_samples,
            'selection_metric': self.mc_selection_metric,
            'passes': self._jsonify(raw_metrics),
        }
        agg_payload = {
            'mc_samples': self.mc_samples,
            'selection_metric': self.mc_selection_metric,
            'passes': self._jsonify(agg_metrics),
        }
        with open(raw_path, 'w') as f:
            json.dump(raw_payload, f, indent=4)
        with open(agg_path, 'w') as f:
            json.dump(agg_payload, f, indent=4)

    def _write_selection_artifacts(
        self,
        best_pass_idx: np.ndarray,
        best_scores: np.ndarray,
    ) -> None:
        path = os.path.join(self.config['output_dir'], 'mc_selection.json')
        payload = {
            'mc_samples': self.mc_samples,
            'selection_metric': self.mc_selection_metric,
            'best_pass_index': self._jsonify(best_pass_idx),
            'best_reference_score': self._jsonify(best_scores),
        }
        with open(path, 'w') as f:
            json.dump(payload, f, indent=4)

    @staticmethod
    def _serialize_metrics(metrics: Metrics) -> dict[str, Any]:
        if hasattr(metrics, 'model_dump'):
            model_dict = metrics.model_dump()
        else:
            model_dict = metrics.dict()
        model_dict.pop('pred', None)
        model_dict.pop('target', None)
        return CausalTrainer._jsonify(model_dict)

    @staticmethod
    def _jsonify(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: CausalTrainer._jsonify(v) for k, v in value.items()}
        if isinstance(value, list):
            return [CausalTrainer._jsonify(v) for v in value]
        if isinstance(value, tuple):
            return [CausalTrainer._jsonify(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    def _save_holdout_results(self, metrics: Metrics, file_suffix: str = '') -> None:
        import pandas as pd

        holdout_cluster = self.splitter.holdout_cluster
        if holdout_cluster is None:
            return

        basin_meta = self.splitter.basin_metadata(self.splitter.holdout_indices)
        basin_count = len(basin_meta['basin_id'])
        if basin_count != len(metrics.kge):
            raise ValueError(
                'Holdout basin metadata and metric lengths do not match: '
                f'{basin_count} vs {len(metrics.kge)}.'
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
            f'results{file_suffix}_held_out_{holdout_cluster}_seed{self.config["seed"]}.csv',
        )
        os.makedirs(self.config['output_dir'], exist_ok=True)
        results_df.to_csv(output_path, index=False)
        log.info('Saved held-out basin KGE results to %s', output_path)
