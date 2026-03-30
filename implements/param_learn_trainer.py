"""Trainer for static-attribute parameter learning on a fixed basin subset."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import pandas as pd

import torch

from dmg.models.criterion.kge_batch_loss import KgeBatchLoss
from dmg.trainers.trainer import Trainer

from implements.basin_utils import load_basin_ids, subset_dataset_by_basin_ids
from implements.causal_trainer import CausalTrainer

log = logging.getLogger(__name__)


class ParamLearnTrainer(CausalTrainer):
    """Baseline trainer with basin-subset filtering and time-only train/test splits."""

    def __init__(
        self,
        config: dict[str, Any],
        model: torch.nn.Module,
        train_dataset: Optional[dict] = None,
        eval_dataset: Optional[dict] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        **kwargs,
    ) -> None:
        if "train" in config["mode"]:
            self._configure_training_resume(config)

        if optimizer is None:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config["train"]["lr"],
            )
        if scheduler is None:
            lr_cfg = config["train"].get("lr_scheduler", {})
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=lr_cfg.get("T_max", config["train"]["epochs"]),
                eta_min=lr_cfg.get("eta_min", 1e-5),
            )

        data_cfg = config.get("data", {})
        subset_path = data_cfg.get("basin_ids_path")
        reference_path = data_cfg.get("basin_ids_reference_path")
        if not subset_path or not reference_path:
            raise ValueError(
                "ParamLearnTrainer requires data.basin_ids_path and "
                "data.basin_ids_reference_path in the config."
            )

        self.reference_basin_ids = load_basin_ids(reference_path)
        self.subset_basin_ids = load_basin_ids(subset_path)

        filtered_train_dataset = None
        self.train_basin_ids = self.subset_basin_ids.copy()
        if train_dataset is not None:
            filtered_train_dataset, self.train_basin_ids = subset_dataset_by_basin_ids(
                train_dataset,
                self.reference_basin_ids,
                self.subset_basin_ids,
            )

        filtered_eval_dataset = None
        self.eval_basin_ids = self.subset_basin_ids.copy()
        if eval_dataset is not None:
            filtered_eval_dataset, self.eval_basin_ids = subset_dataset_by_basin_ids(
                eval_dataset,
                self.reference_basin_ids,
                self.subset_basin_ids,
            )

        Trainer.__init__(
            self,
            config,
            model=model,
            train_dataset=filtered_train_dataset,
            eval_dataset=filtered_eval_dataset,
            loss_func=KgeBatchLoss(config, device=config["device"]),
            optimizer=optimizer,
            scheduler=scheduler,
            **kwargs,
        )

        self.use_amp = self._should_use_amp(config)
        self.amp_dtype = self._resolve_amp_dtype(config)
        self.use_amp, self.amp_dtype = self._normalize_amp_settings(
            self.use_amp,
            self.amp_dtype,
        )
        self.grad_scaler = (
            torch.amp.GradScaler("cuda")
            if self.use_amp and self.amp_dtype == torch.float16
            else None
        )

        test_cfg = config.get("test", {})
        self.mc_samples = max(int(test_cfg.get("mc_samples", 100)), 1)
        self.mc_selection_metric = str(
            test_cfg.get("mc_selection_metric", "mse")
        ).lower()
        self.enable_mc_dropout = bool(test_cfg.get("mc_dropout", True))
        self.mc_seed_base = int(test_cfg.get("mc_seed_base", config["seed"]))
        self.full_train_dataset = self.train_dataset

        log.info(
            "ParamLearnTrainer using %d subset basins from %s",
            len(self.subset_basin_ids),
            subset_path,
        )

    def train_one_epoch(self, epoch, n_samples, n_minibatch, n_timesteps) -> None:
        start_time = time.perf_counter()
        self.current_epoch = epoch
        self.total_loss = 0.0
        self.model.train()

        prog_bar = self._progress(
            range(1, n_minibatch + 1),
            desc=f"Epoch {epoch}/{self.epochs}",
            leave=False,
        )

        for _ in prog_bar:
            dataset_sample = self.sampler.get_training_sample(
                self.train_dataset,
                n_samples,
                n_timesteps,
            )

            with self._autocast_context():
                predictions = self.model(dataset_sample)
                y_pred = predictions["streamflow"].squeeze(-1)
                y_obs = dataset_sample["target"][-y_pred.shape[0] :, :, 0]
                loss = self.loss_func(y_pred=y_pred, y_obs=y_obs)

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
            {type(self.loss_func).__name__: self.total_loss},
            n_minibatch,
            start_time,
        )

        if (epoch % self.config["train"]["save_epoch"] == 0) and self.write_out:
            self._save_checkpoint(epoch, clear_prior=True)

    def evaluate(self):
        if self.eval_dataset is None:
            raise ValueError("eval_dataset required for evaluation.")

        self.is_in_train = False
        metrics = self._evaluate_dataset(
            eval_data=self.eval_dataset,
            reference_data=self.full_train_dataset,
            holdout=False,
        )
        if metrics is not None:
            self._save_eval_results(metrics)
        return metrics

    def _save_eval_results(self, metrics, file_suffix: str = "") -> None:
        basin_count = len(self.eval_basin_ids)
        if basin_count != len(metrics.kge):
            raise ValueError(
                "Eval basin ids and metric lengths do not match: "
                f"{basin_count} vs {len(metrics.kge)}."
            )

        results_df = pd.DataFrame(
            {
                "basin_id": self.eval_basin_ids,
                "kge": metrics.kge,
                "seed": int(self.config["seed"]),
                "nn_model": str(self.config["model"]["nn"]["name"]),
            }
        )
        output_path = os.path.join(
            self.config["output_dir"],
            f"results{file_suffix}_seed{self.config['seed']}.csv",
        )
        os.makedirs(self.config["output_dir"], exist_ok=True)
        results_df.to_csv(output_path, index=False)
        log.info("Saved basin-level evaluation results to %s", output_path)
