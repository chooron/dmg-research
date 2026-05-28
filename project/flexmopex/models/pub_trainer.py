"""Self-contained PubTrainer for LORO experiments.

This version avoids importing from dmg (which requires cartopy) by using
dmg utilities only where they are safe (no cartopy dep), and reimplementing
the minimal training loop needed for FlexMOPEX LORO runs.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import numpy as np
import torch
import tqdm
from numpy.typing import NDArray

log = logging.getLogger(__name__)


class PubTrainer:
    """Trainer adapted for PUB (Predictions in Ungauged Basins) / LORO experiments.

    Designed to work with FlexMopexModelHandler and FlexMopexPubSampler.
    Does NOT inherit from BaseTrainer to avoid dmg cartopy dependency.
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: torch.nn.Module = None,
        train_dataset: Optional[dict] = None,
        eval_dataset: Optional[dict] = None,
        dataset: Optional[dict] = None,
        loss_func: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.nn.Module] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.dataset = dataset
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.verbose = verbose
        self.is_in_train = False
        self.sampler = None  # Set externally after construction
        self.start_epoch = 0
        self.predictions = None

        if "train" in config.get("mode", "train"):
            if not self.train_dataset:
                raise ValueError("'train_dataset' required for training mode.")

            self.epochs = self.config["train"]["epochs"]

            # Loss function (caller should pass it directly)
            if self.loss_func is not None:
                self.model.loss_func = self.loss_func

            # Optimizer
            self.optimizer = optimizer or self._init_optimizer()

            # LR scheduler
            lr_scheduler_name = config.get("delta_model", {}).get("nn_model", {}).get("lr_scheduler")
            if lr_scheduler_name:
                self.use_scheduler = True
                self.scheduler = scheduler or self._init_scheduler()
            else:
                self.use_scheduler = False

            self._load_states()

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------
    def _init_optimizer(self) -> torch.optim.Optimizer:
        name = self.config["train"]["optimizer"]
        lr = self.config["delta_model"]["nn_model"]["learning_rate"]
        optimizer_dict = {
            "Adadelta": torch.optim.Adadelta,
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
        }
        cls = optimizer_dict.get(name)
        if cls is None:
            raise ValueError(f"Optimizer '{name}' not recognized.")
        return cls(self.model.get_parameters(), lr=lr)

    def _init_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        name = self.config["delta_model"]["nn_model"]["lr_scheduler"]
        params = self.config["delta_model"]["nn_model"].get("lr_scheduler_params", {})
        scheduler_dict = {
            "StepLR": torch.optim.lr_scheduler.StepLR,
            "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
            "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
        }
        cls = scheduler_dict.get(name)
        if cls is None:
            raise ValueError(f"Scheduler '{name}' not recognized.")
        return cls(self.optimizer, **params)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------
    def _load_states(self) -> None:
        path = self.config.get("model_path", "")
        if not path or not os.path.isdir(path):
            self.start_epoch = 0
            return
        for file in os.listdir(path):
            if "train_state" in file:
                checkpoint = torch.load(os.path.join(path, file), weights_only=False)
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.model.load_model(epoch=checkpoint["epoch"])
                self.start_epoch = checkpoint["epoch"] + 1
                print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
                if self.scheduler and "scheduler_state_dict" in checkpoint:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                torch.set_rng_state(checkpoint["random_state"])
                if torch.cuda.is_available() and "cuda_random_state" in checkpoint:
                    torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
                return
        self.start_epoch = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Train the model for all epochs."""
        from dmg.core.data import create_training_grid  # safe: no cartopy dep

        self.is_in_train = True
        n_samples, n_minibatch, n_timesteps = create_training_grid(
            self.train_dataset["xc_nn_norm"], self.config
        )
        log.info(f"LORO training: epochs {self.start_epoch}–{self.epochs}")
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_one_epoch(epoch, n_minibatch, n_timesteps)

    def _train_one_epoch(self, epoch: int, n_minibatch: int, n_timesteps: int) -> None:
        start_time = time.perf_counter()
        self.total_loss = 0.0

        for mb in tqdm.tqdm(
            range(1, n_minibatch + 1),
            desc=f"Epoch {epoch}/{self.epochs}",
            leave=False,
            dynamic_ncols=True,
        ):
            dataset_sample = self.sampler.get_training_sample(
                dataset=self.train_dataset, ngrid_train=None, nt=n_timesteps
            )
            _ = self.model(dataset_sample)
            loss = self.model.calc_loss(dataset_sample, loss_func=self.loss_func)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.total_loss += loss.item()

            if self.verbose and mb % 10 == 0:
                tqdm.tqdm.write(f"Epoch {epoch}, batch {mb} | loss: {loss.item():.6f}")

        if self.use_scheduler:
            self.scheduler.step()

        self._log_epoch_stats(epoch, getattr(self.model, "loss_dict", {}), n_minibatch, start_time)

        save_epoch = self.config["train"].get("save_epoch", 10)
        if epoch % save_epoch == 0:
            self.model.save_model(epoch)
            self._save_train_state(epoch)

    def _save_train_state(self, epoch: int) -> None:
        from dmg.core.utils.utils import save_train_state  # safe: no cartopy dep
        save_train_state(
            self.config,
            epoch=epoch,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            clear_prior=False,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self) -> None:
        """Evaluate on all holdout (val) basins basin-by-basin."""
        from dmg.core.calc.metrics import Metrics  # safe: no cartopy dep
        from dmg.core.utils.utils import save_outputs  # safe: no cartopy dep

        self.is_in_train = False
        val_indices = self.sampler.val_indices
        log.info(f"Evaluating on {len(val_indices)} holdout basins.")

        model_name = self.config["delta_model"]["phy_model"]["model"][0]
        all_preds: list[dict] = []

        for basin_idx in tqdm.tqdm(
            val_indices, desc="Evaluating holdout basins", leave=False, dynamic_ncols=True
        ):
            sample = self._get_val_sample(basin_idx)
            pred = self.model(sample, eval=True)
            pred_basin = {
                k: v.cpu().detach() for k, v in pred[model_name].items()
            }
            all_preds.append(pred_basin)

        self.predictions = self._concat_preds(all_preds)

        observations = self.eval_dataset["target"]
        save_outputs(self.config, [self.predictions], observations)
        self._calc_metrics(self.predictions, val_indices)

    def _get_val_sample(self, basin_idx: int) -> dict:
        """Get full time-series sample for one validation basin."""
        dataset = self.eval_dataset
        sample = {}
        for key, value in dataset.items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 3:
                    sample[key] = value[:, [basin_idx], :]
                elif value.ndim == 2:
                    sample[key] = value[[basin_idx], :]
                else:
                    sample[key] = value
            else:
                sample[key] = value
        return sample

    def _concat_preds(self, preds: list[dict]) -> dict:
        """Concatenate per-basin predictions along basin dimension (dim=1)."""
        result = {}
        for key in preds[0]:
            tensors = [p[key] for p in preds]
            dim = 1 if tensors[0].ndim == 3 else 0
            result[key] = torch.cat(tensors, dim=dim).numpy()
        return result

    def _calc_metrics(self, predictions: dict, val_indices) -> None:
        from dmg.core.calc.metrics import Metrics  # safe: no cartopy dep

        target_name = self.config["train"]["target"][0]
        preds = predictions[target_name]  # (T, N, 1) or (T, N)

        warmup = self.config["delta_model"]["phy_model"]["warm_up"]
        obs = self.eval_dataset["target"][:, val_indices, :].cpu().numpy()
        obs = obs[warmup:, :, :]
        if obs.ndim == 3:
            obs_2d = obs[:, :, 0]
        else:
            obs_2d = obs
        if preds.ndim == 3:
            preds_2d = preds[:, :, 0]
        else:
            preds_2d = preds

        # Align lengths
        n = min(obs_2d.shape[0], preds_2d.shape[0])
        metrics = Metrics(
            np.swapaxes(preds_2d[:n], 1, 0),
            np.swapaxes(obs_2d[:n], 1, 0),
        )
        metrics.dump_metrics(self.config["out_path"])

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log_epoch_stats(
        self, epoch: int, loss_dict: dict, n_minibatch: int, start_time: float
    ) -> None:
        avg = {k: v / n_minibatch for k, v in loss_dict.items()} if loss_dict else {}
        loss_str = ", ".join(f"{k}: {v:.6f}" for k, v in avg.items()) or f"total: {self.total_loss / n_minibatch:.6f}"
        elapsed = time.perf_counter() - start_time
        mem_mb = 0
        try:
            device = self.config.get("device", "cpu")
            if str(device).startswith("cuda"):
                mem_mb = int(torch.cuda.memory_reserved(device=device) / 1e6)
        except Exception:
            pass
        log.info(f"Epoch {epoch}: {loss_str} | {elapsed:.1f}s | {mem_mb}MB GPU")
        print(f"[Epoch {epoch}/{self.epochs}] {loss_str} | {elapsed:.1f}s")
