import logging
import os
import sys
import time
from typing import Any, Optional

import numpy as np
import torch
from numpy.typing import NDArray

from dmg.core.calc.metrics import Metrics
from dmg.core.utils.utils import save_outputs, save_train_state

from .common_trainer import CommonTrainer

log = logging.getLogger(__name__)


class FasterTrainer(CommonTrainer):
    """Optimized trainer built on top of the common trainer base."""

    def _get_group_reduction(self) -> str:
        return self.config.get("train", {}).get("group_reduce", "sum")

    def _prepare_targets_for_loss(
        self,
        target: torch.Tensor,
        n_groups: int,
    ) -> torch.Tensor:
        return target

    def _prepare_observations_for_metrics(
        self,
        observations: torch.Tensor,
        n_groups: int,
    ) -> torch.Tensor:
        return observations

    def _prepare_model_outputs_for_loss(
        self,
        dataset_sample: dict[str, torch.Tensor],
        n_groups: int,
    ) -> None:
        if n_groups <= 1:
            return

        target_name = self._get_target_name()
        n_basins = dataset_sample["target"].shape[1]
        reduction = self._get_group_reduction()

        for output in self.model.output_dict.values():
            if target_name not in output:
                continue
            output[target_name] = self._reduce_grouped_prediction(
                output[target_name],
                n_basins,
                n_groups,
                reduction=reduction,
            )

    def _postprocess_prediction_dict(
        self,
        prediction: dict[str, torch.Tensor],
        n_basins: int,
        n_groups: int,
    ) -> dict[str, torch.Tensor]:
        if n_groups <= 1:
            return prediction

        target_name = self._get_target_name()
        if target_name not in prediction:
            return prediction

        updated = dict(prediction)
        updated[target_name] = self._reduce_grouped_prediction(
            prediction[target_name],
            n_basins,
            n_groups,
            reduction=self._get_group_reduction(),
        )
        return updated

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
        name = self.config["delta_model"]["train"]["lr_scheduler"]
        scheduler_dict = {
            "StepLR": torch.optim.lr_scheduler.StepLR,
            "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
            "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
        }

        cls = scheduler_dict[name]
        if cls is None:
            raise ValueError(
                f"Scheduler '{name}' not recognized. Available options are: {list(scheduler_dict.keys())}"
            )

        try:
            self.scheduler = cls(
                self.optimizer,
                **self.config["delta_model"]["train"]["lr_scheduler_params"],
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Error initializing scheduler: {exc}") from exc
        return self.scheduler

    def load_states(self) -> None:
        path = self.config["model_path"]
        for file in os.listdir(path):
            if "train_state" in file:
                checkpoint = torch.load(
                    os.path.join(path, file),
                    map_location=self.config.get("device", "cpu"),
                )

                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.model.load_model(epoch=checkpoint["epoch"])
                self.start_epoch = checkpoint["epoch"] + 1

                if self.scheduler:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

                torch.set_rng_state(checkpoint["random_state"])
                if torch.cuda.is_available() and "cuda_random_state" in checkpoint:
                    torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
                print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
                return
            self.start_epoch = 1

    def train(self) -> None:
        self.is_in_train = True
        n_samples, n_minibatch, n_timesteps = self._setup_training_grid()
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
        avg_loss_dict = {
            key: value / max(n_minibatch, 1) for key, value in loss_dict.items()
        }
        loss = ", ".join(f"{key}: {value:.6f}" for key, value in avg_loss_dict.items())
        elapsed = time.perf_counter() - start_time

        if torch.cuda.is_available() and str(self.config["device"]).startswith("cuda"):
            mem_alloc = int(
                torch.cuda.memory_reserved(device=self.config["device"]) * 0.000001
            )
        else:
            mem_alloc = 0

        self._emit_progress(
            f"[Epoch {epoch:>4}/{self.epochs}] loss={self._final_loss:.4f} | "
            f"time={elapsed:.1f}s | mem={mem_alloc}MB"
        )
        sys.stdout.flush()
        sys.stderr.flush()

    def evaluate(self) -> None:
        self.is_in_train = False

        observations = self._prepare_observations_for_metrics(
            self.eval_dataset["target"],
            self._get_num_parameter_groups(),
        )
        n_samples = self.eval_dataset["xc_nn_norm"].shape[1]
        batch_start = np.arange(0, n_samples, self.config["test"]["batch_size"])
        batch_end = np.append(batch_start[1:], n_samples)

        log.info(f"Validating Model: Forwarding {len(batch_start)} batches")
        batch_predictions = self._forward_loop(
            self.eval_dataset,
            batch_start,
            batch_end,
        )

        log.info("Saving model outputs + Calculating metrics")
        save_outputs(self.config, batch_predictions, observations, create_dirs=True)
        self.predictions = self._batch_data(batch_predictions)
        self.calc_metrics(batch_predictions, observations)

    def evaluate_mc_dropout(self, n_samples: int = 100) -> None:
        self.is_in_train = False
        log.info(f"Starting MC Dropout evaluation with {n_samples} samples")

        model_name = list(self.model.model_dict.keys())[0]
        dpl_model = self.model.model_dict[model_name]
        nn_model = dpl_model.nn_model
        phy_model = dpl_model.phy_model

        nn_model.train()
        phy_model.eval()

        out_path = self.config["out_path"]
        mc_dropout_dir = os.path.join(out_path, "mc_dropout")
        os.makedirs(mc_dropout_dir, exist_ok=True)

        for dataset_name, dataset in [
            ("train", self.train_dataset),
            ("eval", self.eval_dataset),
        ]:
            if dataset is None:
                continue

            log.info(f"Processing {dataset_name} dataset with MC Dropout")
            all_params_samples, all_preds_samples, all_metrics = (
                self._mc_dropout_forward(
                    dataset,
                    n_samples,
                )
            )
            self._save_mc_dropout_results(
                mc_dropout_dir,
                dataset_name,
                all_params_samples,
                all_preds_samples,
                all_metrics,
                dataset["target"],
            )

        log.info(f"MC Dropout evaluation complete. Results saved to {mc_dropout_dir}")

    def _mc_dropout_forward(
        self,
        dataset: dict,
        n_samples: int,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        model_name = list(self.model.model_dict.keys())[0]
        dpl_model = self.model.model_dict[model_name]
        nn_model = dpl_model.nn_model
        phy_model = dpl_model.phy_model

        n_basins = dataset["xc_nn_norm"].shape[1]
        n_timesteps_full = dataset["x_phy"].shape[0]
        warm_up = self.config["delta_model"]["phy_model"]["warm_up"]
        _ = n_timesteps_full - warm_up

        all_params_samples = []
        all_preds_samples = []
        all_metrics = []

        batch_size = n_basins
        batch_start = np.arange(0, n_basins, batch_size)
        batch_end = np.append(batch_start[1:], n_basins)

        for sample_idx in range(n_samples):
            batch_predictions = []
            batch_parameters = []

            for index in range(len(batch_start)):
                dataset_sample = self.sampler.get_validation_sample(
                    dataset,
                    batch_start[index],
                    batch_end[index],
                )

                with torch.no_grad():
                    _, params = nn_model(dataset_sample)
                    reduced_params = (
                        phy_model.reduce_parameters_for_analysis(params)
                        if hasattr(phy_model, "reduce_parameters_for_analysis")
                        else params.squeeze(-1)
                    )
                    batch_parameters.append(reduced_params.cpu())

                    output = phy_model(dataset_sample, (None, params))
                    batch_predictions.append(
                        self._postprocess_prediction_dict(
                            output,
                            n_basins,
                            params.shape[-1] if params.dim() == 3 else 1,
                        )
                    )

            params_full = torch.cat(batch_parameters, dim=0).numpy()
            all_params_samples.append(params_full)

            preds_full = self._batch_data(batch_predictions, target_key="streamflow")
            all_preds_samples.append(preds_full)

            n_groups = params.shape[-1] if params.dim() == 3 else 1
            metrics = self._calc_sample_metrics(preds_full, dataset["target"], n_groups)
            all_metrics.append(metrics)

        return (
            np.stack(all_params_samples, axis=0),
            np.stack(all_preds_samples, axis=0),
            all_metrics,
        )

    def _calc_sample_metrics(
        self,
        predictions: np.ndarray,
        observations: torch.Tensor,
        n_groups: int,
    ) -> dict:
        prepared_obs = self._prepare_observations_for_metrics(observations, n_groups)
        obs_np = (
            prepared_obs.cpu().numpy()
            if isinstance(prepared_obs, torch.Tensor)
            else prepared_obs
        )
        metrics_to_compute = self.config["test"].get("metrics", None)

        if predictions.ndim == 2 and obs_np.ndim == 3:
            obs_np = obs_np.squeeze(-1)

        if obs_np.shape[0] > predictions.shape[0]:
            obs_np = obs_np[: predictions.shape[0], :]

        metrics_calc = Metrics(
            np.swapaxes(predictions, 1, 0),
            np.swapaxes(obs_np, 1, 0),
            metrics_to_compute,
        )

        result = {}
        if metrics_to_compute is None:
            result["nse_mean"] = float(np.nanmean(metrics_calc.nse))
            result["nse_basin"] = metrics_calc.nse
            result["kge_mean"] = float(np.nanmean(metrics_calc.kge))
            result["kge_basin"] = metrics_calc.kge
            return result

        for metric_name in metrics_to_compute:
            if hasattr(metrics_calc, metric_name):
                values = getattr(metrics_calc, metric_name)
                result[f"{metric_name}_mean"] = float(np.nanmean(values))
                result[f"{metric_name}_basin"] = values
        return result

    def _save_mc_dropout_results(
        self,
        save_dir: str,
        dataset_name: str,
        params_samples: np.ndarray,
        preds_samples: np.ndarray,
        metrics_list: list,
        observations: torch.Tensor,
    ) -> None:
        params_file = os.path.join(save_dir, f"{dataset_name}_parameters_samples.npz")
        np.savez_compressed(params_file, samples=params_samples)
        log.info(f"Saved parameters samples to {params_file}")

        preds_file = os.path.join(save_dir, f"{dataset_name}_predictions_samples.npz")
        np.savez_compressed(preds_file, samples=preds_samples)
        log.info(f"Saved predictions samples to {preds_file}")

        metrics_file = os.path.join(save_dir, f"{dataset_name}_metrics_samples.npz")
        metrics_mean = {}
        metrics_basin = {}

        for key in metrics_list[0].keys():
            if key.endswith("_mean"):
                metric_name = key[:-5]
                metrics_mean[metric_name] = np.array(
                    [metric[key] for metric in metrics_list]
                )
            elif key.endswith("_basin"):
                metric_name = key[:-6]
                metrics_basin[metric_name] = np.array(
                    [metric[key] for metric in metrics_list]
                )

        save_dict = {}
        for metric_name, values in metrics_mean.items():
            save_dict[f"{metric_name}_mean"] = values
        for metric_name, values in metrics_basin.items():
            save_dict[f"{metric_name}_basin"] = values

        np.savez_compressed(metrics_file, **save_dict)
        log.info(f"Saved metrics samples to {metrics_file}")
        log.info(f"  - Global mean metrics: {list(metrics_mean.keys())}")
        log.info(f"  - Basin-level metrics: {list(metrics_basin.keys())}")

        params_stats_file = os.path.join(
            save_dir, f"{dataset_name}_parameters_stats.npz"
        )
        np.savez_compressed(
            params_stats_file,
            mean=params_samples.mean(axis=0),
            std=params_samples.std(axis=0),
            p10=np.percentile(params_samples, 10, axis=0),
            p90=np.percentile(params_samples, 90, axis=0),
        )
        log.info(f"Saved parameters statistics to {params_stats_file}")

        preds_stats_file = os.path.join(
            save_dir, f"{dataset_name}_predictions_stats.npz"
        )
        np.savez_compressed(
            preds_stats_file,
            mean=preds_samples.mean(axis=0),
            std=preds_samples.std(axis=0),
            p10=np.percentile(preds_samples, 10, axis=0),
            p90=np.percentile(preds_samples, 90, axis=0),
        )
        log.info(f"Saved predictions statistics to {preds_stats_file}")

        summary_file = os.path.join(save_dir, f"{dataset_name}_metrics_summary.txt")
        with open(summary_file, "w", encoding="utf-8") as handle:
            handle.write(
                f"MC Dropout Evaluation Summary - {dataset_name.upper()} Dataset\n"
            )
            handle.write("=" * 60 + "\n\n")
            handle.write("Global Mean Metrics (averaged across all basins):\n")
            handle.write("-" * 60 + "\n")
            for metric_name, values in metrics_mean.items():
                handle.write(f"{metric_name.upper()}:\n")
                handle.write(f"  Mean: {values.mean():.4f}\n")
                handle.write(f"  Std:  {values.std():.4f}\n")
                handle.write(f"  Min:  {values.min():.4f}\n")
                handle.write(f"  Max:  {values.max():.4f}\n")
                handle.write(f"  P10:  {np.percentile(values, 10):.4f}\n")
                handle.write(f"  P90:  {np.percentile(values, 90):.4f}\n\n")

            handle.write("\nBasin-Level Metrics Statistics:\n")
            handle.write("-" * 60 + "\n")
            for metric_name, values in metrics_basin.items():
                handle.write(f"{metric_name.upper()}:\n")
                handle.write(f"  Overall Mean: {np.nanmean(values):.4f}\n")
                handle.write(f"  Overall Std:  {np.nanstd(values):.4f}\n")
                handle.write(
                    f"  Best Basin Mean: {np.nanmax(np.nanmean(values, axis=0)):.4f}\n"
                )
                handle.write(
                    f"  Worst Basin Mean: {np.nanmin(np.nanmean(values, axis=0)):.4f}\n\n"
                )

        log.info(f"Saved metrics summary to {summary_file}")

    def _forward_loop(
        self,
        data: dict[str, torch.Tensor],
        batch_start: NDArray,
        batch_end: NDArray,
    ):
        batch_predictions = []
        for index in range(len(batch_start)):
            self.current_batch = index
            dataset_sample = self.sampler.get_validation_sample(
                data,
                batch_start[index],
                batch_end[index],
            )

            if self.config["test"]["split_dataset"]:
                total_time_steps = dataset_sample["x_phy"].shape[0]
                prediction_time_chunks = []
                prediction_length = self.config["delta_model"]["rho"]
                warmup_length = self.config["delta_model"]["phy_model"]["warm_up"]
                time_starts = range(
                    0,
                    total_time_steps - prediction_length - warmup_length + 1,
                    prediction_length,
                )

                for t_start in time_starts:
                    t_end = t_start + prediction_length + warmup_length
                    time_window_input = {
                        key: tensor[t_start:t_end, ...]
                        if len(tensor.shape) > 2
                        else tensor
                        for key, tensor in dataset_sample.items()
                    }
                    prediction_window = self.model(time_window_input, eval=True)
                    prediction_valid_part = {
                        key: tensor.cpu().detach()
                        for key, tensor in prediction_window.items()
                    }
                    prediction_time_chunks.append(prediction_valid_part)

                collated_chunks = {key: [] for key in prediction_time_chunks[0]}
                for chunk in prediction_time_chunks:
                    for key, tensor in chunk.items():
                        collated_chunks[key].append(tensor)

                prediction = {
                    key: torch.cat(tensors, dim=0)
                    for key, tensors in collated_chunks.items()
                }
                batch_predictions.append(
                    self._postprocess_prediction_dict(
                        prediction,
                        dataset_sample["target"].shape[1],
                        self._get_num_parameter_groups(),
                    )
                )
            else:
                prediction = self.model(dataset_sample, eval=True)
                prediction = {
                    key: tensor.cpu().detach() for key, tensor in prediction.items()
                }
                batch_predictions.append(
                    self._postprocess_prediction_dict(
                        prediction,
                        dataset_sample["target"].shape[1],
                        self._get_num_parameter_groups(),
                    )
                )
        return batch_predictions

    def calc_metrics(
        self,
        batch_predictions: list[dict[str, torch.Tensor]],
        observations: torch.Tensor,
    ) -> None:
        target_name = self._get_target_name()
        predictions = self._batch_data(batch_predictions, target_name)
        target = np.expand_dims(observations[:, :, 0].cpu().numpy(), 2)
        target = target[: len(predictions), :]

        metrics_to_compute = self.config["test"].get("metrics", None)
        metrics = Metrics(
            np.swapaxes(predictions.squeeze(), 1, 0),
            np.swapaxes(target.squeeze(), 1, 0),
            metrics_to_compute,
        )
        metrics.dump_metrics(self.config["out_path"])
