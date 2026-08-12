import logging
import os
import sys
import time
from typing import Any, Optional

import torch

from .common_trainer import CommonTrainer
from .checkpoint import load_training_checkpoint

log = logging.getLogger(__name__)


class CalTrainer(CommonTrainer):
    """Calibration trainer with standard training logic."""

    def init_optimizer(self) -> torch.optim.Optimizer:
        name = self.config["train"]["optimizer"]
        optimizer_dict = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "Adadelta": torch.optim.Adadelta,
            "RMSprop": torch.optim.RMSprop,
        }

        cls = optimizer_dict[name]
        if cls is None:
            raise ValueError(
                f"Optimizer '{name}' not recognized. Available options are: {list(optimizer_dict.keys())}"
            )

        try:
            self.optimizer = cls(
                self.model.get_parameters(),
                lr=self.config["train"]["learning_rate"],
                weight_decay=self.config["train"].get("weight_decay", 0.0),
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Error initializing optimizer: {exc}") from exc
        return self.optimizer

    def init_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        name = self.config["delta_model"]["nn_model"]["lr_scheduler"]
        scheduler_dict = {
            "StepLR": torch.optim.lr_scheduler.StepLR,
            "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
            "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
            "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        }

        cls = scheduler_dict[name]
        if cls is None:
            raise ValueError(
                f"Scheduler '{name}' not recognized. Available options are: {list(scheduler_dict.keys())}"
            )

        try:
            self.scheduler = cls(
                self.optimizer,
                **self.config["delta_model"]["nn_model"]["lr_scheduler_params"],
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Error initializing scheduler: {exc}") from exc
        return self.scheduler

    def load_states(self) -> None:
        path = self.config["model_path"]
        files = sorted(
            (os.path.join(path, file) for file in os.listdir(path)),
            key=lambda item: item,
        ) if os.path.isdir(path) else []
        files = [file for file in files if os.path.basename(file).startswith("trainer_state_ep")]
        if not files:
            self.start_epoch = 1
            self.global_step = 0
            return
        checkpoint = load_training_checkpoint(
            files[-1],
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            sampler=self.sampler,
            map_location=self.config["device"],
        )
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint["global_step"])
        log.debug(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def train(self) -> None:
        self.is_in_train = True
        n_samples, n_minibatch, n_timesteps = self._setup_training_grid()
        max_batches = self.config.get("train", {}).get("max_batches")
        if max_batches is not None:
            n_minibatch = min(n_minibatch, max(int(max_batches), 1))
        n_basins = self.train_dataset["xc_nn_norm"].shape[1]

        self._log_training_start(n_basins)
        self._train_start_time = time.perf_counter()
        self._final_loss = 0.0

        self._emit_progress(
            f"Training model: Beginning {self.start_epoch} of {self.epochs} epochs"
        )
        sys.stdout.flush()
        sys.stderr.flush()

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.train_one_epoch(epoch, n_samples, n_minibatch, n_timesteps)

        total_time = time.perf_counter() - self._train_start_time
        self._log_training_end(total_time, self._final_loss)

    def train_one_epoch(
        self,
        epoch: int,
        n_samples: int,
        n_minibatch: int,
        n_timesteps: int,
    ) -> None:
        start_time = time.perf_counter()
        self._train_one_epoch_core(epoch, n_samples, n_minibatch, n_timesteps)
        self._log_epoch_stats(epoch, self.model.loss_dict, n_minibatch, start_time)
        self._save_checkpoint(epoch)

    def _log_epoch_stats(
        self,
        epoch: int,
        loss_dict: dict[str, float],
        n_minibatch: int,
        start_time: float,
    ) -> None:
        """Log epoch statistics including loss and time."""
        log_interval = self.config["train"].get("log_interval", 1)
        if epoch % log_interval != 0:
            return

        lr = self.optimizer.param_groups[0]["lr"]
        avg_loss = self._final_loss
        elapsed = time.perf_counter() - start_time

        if torch.cuda.is_available() and str(self.config["device"]).startswith("cuda"):
            mem_mb = int(
                torch.cuda.memory_reserved(device=self.config["device"]) * 0.000001
            )
        else:
            mem_mb = 0

        warm_restart_tag = ""
        if self.use_scheduler and isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        ):
            t_cur = self.scheduler.T_0
            while t_cur <= epoch:
                if epoch == t_cur:
                    warm_restart_tag = "  <- Warm Restart"
                    break
                t_cur += self.scheduler.T_0 * (
                    self.scheduler.T_mult ** (t_cur // self.scheduler.T_0)
                )

        self._emit_progress(
            f"[Epoch {epoch:>4}/{self.epochs}] loss={avg_loss:.4f} | "
            f"lr={lr:.2e} | time={elapsed:.1f}s | mem={mem_mb}MB{warm_restart_tag}"
        )
        sys.stdout.flush()
        sys.stderr.flush()
