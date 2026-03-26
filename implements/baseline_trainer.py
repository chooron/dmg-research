"""BaselineTrainer — ERM training with MC-dropout evaluation."""

from __future__ import annotations

import time

import tqdm

from implements.causal_trainer import CausalTrainer


class BaselineTrainer(CausalTrainer):
    """Baseline trainer that keeps holdout evaluation but removes env-group loss."""

    def train_one_epoch(self, epoch, n_samples, n_minibatch, n_timesteps) -> None:
        start_time = time.perf_counter()
        self.current_epoch = epoch
        self.total_loss = 0.0
        self.model.train()

        prog_bar = tqdm.tqdm(
            range(1, n_minibatch + 1),
            desc=f"Epoch {epoch}/{self.epochs}",
            leave=False,
            dynamic_ncols=True,
        )

        for _ in prog_bar:
            dataset_sample = self.sampler.get_training_sample(
                self.train_dataset,
                n_samples,
                n_timesteps,
            )

            with self._autocast_context():
                predictions = self.model(dataset_sample)
                y_pred = predictions['streamflow'].squeeze(-1)
                y_obs = dataset_sample['target'][-y_pred.shape[0]:, :, 0]
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

        if (epoch % self.config['train']['save_epoch'] == 0) and self.write_out:
            self._save_checkpoint(epoch, clear_prior=True)
