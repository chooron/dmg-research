import logging
import os
import sys
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from numpy.typing import NDArray

from dmg.core.calc.metrics import Metrics
from dmg.core.data import create_training_grid

create_dl_training_grid = create_training_grid
from dmg.core.utils.factory import import_data_sampler
from dmg.core.utils.utils import save_outputs

save_outputsv2 = save_outputs
from dmg.models.model_handler import ModelHandler
from dmg.trainers.base import BaseTrainer

from losses import build_loss_from_config

from .checkpoint import save_training_checkpoint

log = logging.getLogger(__name__)


class CommonTrainer(BaseTrainer):
    """Base class for trainers with common training logic."""

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
        self._normalize_dmotpy_config()
        self.model = model or ModelHandler(config)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.verbose = verbose
        self.sampler = import_data_sampler(config["data_sampler"])(config)
        self.is_in_train = False

        if "train" in config["mode"]:
            if not self.train_dataset:
                raise ValueError("'train_dataset' required for training mode.")

            self.epochs = self.config["train"]["epochs"]
            self.loss_func = loss_func or build_loss_from_config(config["loss_function"])
            self.model.loss_func = self.loss_func

            self.optimizer = optimizer or self.init_optimizer()
            if config["delta_model"]["nn_model"]["lr_scheduler"]:
                self.use_scheduler = True
                self.scheduler = scheduler or self.init_scheduler()
            else:
                self.use_scheduler = False

            self.load_states()
        elif "test" in config["mode"]:
            self.load_test_states()

    def _normalize_dmotpy_config(self) -> None:
        """Bridge the dMoT config schema to the legacy Trainer internals.

        The external dMG package uses ``delta_model`` while the dMoT entry
        point uses ``model``.  This is a schema adapter, not a model-specific
        branch, and keeps the Trainer's public lifecycle uniform.
        """
        train = self.config.setdefault("train", {})
        model = self.config.setdefault("model", {})
        phy = model.get("phy", {}) or {}
        nn_model = model.get("nn", {}) or {}
        if model.get("warmup") is None:
            model["warmup"] = int(model.get("warm_up", phy.get("warm_up", 0)) or 0)
        model_name = phy.get("model_name") or phy.get("name") or "unknown"
        if isinstance(model_name, (list, tuple)):
            model_name = model_name[0]
        scheduler = train.get("lr_scheduler")
        if isinstance(scheduler, dict):
            scheduler_name = scheduler.get("name")
            scheduler_params = dict(scheduler)
            scheduler_params.pop("name", None)
        else:
            scheduler_name = scheduler
            scheduler_params = {}
        config_delta = self.config.setdefault("delta_model", {})
        config_delta.setdefault("rho", int(model.get("rho", 365)))
        config_delta.setdefault("phy_model", {"model": model_name, "warm_up": int(model.get("warm_up", phy.get("warm_up", 0)))})
        config_delta.setdefault("nn_model", {"lr_scheduler": scheduler_name})
        config_delta.setdefault("train", {"lr_scheduler": scheduler_name, "lr_scheduler_params": scheduler_params})
        self.config.setdefault("model_path", self.config.get("model_dir", "."))
        self.config.setdefault("out_path", self.config.get("output_dir", "."))
        train.setdefault("learning_rate", train.get("lr", 1e-3))
        train.setdefault("save_epoch", 1)
        self.config.setdefault("loss_function", train.get("loss_function", {"name": "NseBatchLoss"}))

    @abstractmethod
    def init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize the optimizer as named in the config."""
        raise NotImplementedError(
            "Derived classes must implement `init_optimizer` method."
        )

    @abstractmethod
    def init_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Initialize the learning rate scheduler."""
        raise NotImplementedError(
            "Derived classes must implement `init_scheduler` method."
        )

    @abstractmethod
    def load_states(self) -> None:
        """Load training states from checkpoint."""
        raise NotImplementedError(
            "Derived classes must implement `load_states` method."
        )

    def load_test_states(self) -> None:
        path = self.config["model_path"]
        test_epoch = self.config["test"].get("test_epoch", None)
        if test_epoch is None:
            raise ValueError("'test_epoch' must be set in config['test'].")

        model_name = self.config["delta_model"]["phy_model"]["model"]
        if isinstance(model_name, list):
            model_name = model_name[0]

        checkpoint_file = f"d{model_name}_Ep{int(test_epoch)}.pt"
        checkpoint_path = os.path.join(path, checkpoint_file)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"{checkpoint_path} not found.")

        self.model.load_model(epoch=int(test_epoch))
        print(f"Loaded test checkpoint: {checkpoint_path}")

    def _emit_progress(self, message: str) -> None:
        if log.hasHandlers() and log.isEnabledFor(logging.INFO):
            log.info(message)
        else:
            print(message, flush=True)

    def _get_primary_model_name(self) -> str:
        model_name = self.config["delta_model"]["phy_model"]["model"]
        return model_name[0] if isinstance(model_name, list) else model_name

    def _get_primary_dpl_model(self):
        model_name = self._get_primary_model_name()
        return self.model.model_dict.get(model_name, None)

    def _get_primary_nn_model(self):
        dpl_model = self._get_primary_dpl_model()
        return getattr(dpl_model, "nn_model", None)

    def _get_primary_phy_model(self):
        dpl_model = self._get_primary_dpl_model()
        return getattr(dpl_model, "phy_model", None)

    def _get_num_parameter_groups(self) -> int:
        nn_model = self._get_primary_nn_model()
        return getattr(nn_model, "num_start", 1)

    def _should_expand_targets(self, n_groups: int) -> bool:
        phy_model = self._get_primary_phy_model()
        if phy_model is not None and hasattr(phy_model, "should_expand_targets"):
            return phy_model.should_expand_targets(n_groups)
        return n_groups > 1

    def _prepare_targets_for_loss(
        self,
        target: torch.Tensor,
        n_groups: int,
    ) -> torch.Tensor:
        if self._should_expand_targets(n_groups):
            return target.repeat_interleave(n_groups, dim=1)
        return target

    def _prepare_observations_for_metrics(
        self,
        observations: torch.Tensor,
        n_groups: int,
    ) -> torch.Tensor:
        phy_model = self._get_primary_phy_model()
        if phy_model is not None and hasattr(
            phy_model, "prepare_observations_for_metrics"
        ):
            return phy_model.prepare_observations_for_metrics(observations, n_groups)
        return self._prepare_targets_for_loss(observations, n_groups)

    def _get_target_name(self) -> str:
        return self.config["train"]["target"][0]

    def _reshape_grouped_prediction(
        self,
        prediction: torch.Tensor,
        n_basins: int,
        n_groups: int,
    ) -> torch.Tensor:
        if n_groups <= 1:
            return prediction
        if prediction.dim() == 3:
            return prediction
        if prediction.dim() != 2:
            return prediction
        if prediction.shape[1] != n_basins * n_groups:
            return prediction
        return prediction.view(prediction.shape[0], n_basins, n_groups)

    def _reduce_grouped_prediction(
        self,
        prediction: torch.Tensor,
        n_basins: int,
        n_groups: int,
        reduction: str = "mean",
    ) -> torch.Tensor:
        grouped = self._reshape_grouped_prediction(prediction, n_basins, n_groups)
        if grouped.dim() != 3:
            return grouped
        if reduction == "sum":
            return grouped.sum(dim=-1)
        return grouped.mean(dim=-1)

    def _prepare_model_outputs_for_loss(
        self,
        dataset_sample: dict[str, torch.Tensor],
        n_groups: int,
    ) -> None:
        return None

    @staticmethod
    def _assert_finite_output(output: Any, context: str) -> None:
        """Fail at the first non-finite forward value with training context."""
        if isinstance(output, torch.Tensor):
            if not torch.isfinite(output).all():
                raise FloatingPointError(f"{context}: model output contains NaN or Inf")
        elif isinstance(output, dict):
            for key, value in output.items():
                CommonTrainer._assert_finite_output(value, f"{context}.{key}")
        elif isinstance(output, (tuple, list)):
            for index, value in enumerate(output):
                CommonTrainer._assert_finite_output(value, f"{context}[{index}]")

    @staticmethod
    def _align_loss_tensors(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if prediction.ndim > 2 and prediction.shape[-1] == 1:
            prediction = prediction.squeeze(-1)
        if target.ndim > 2 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        if mask is not None and mask.ndim > 2 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        if prediction.ndim != target.ndim:
            raise ValueError(
                f"prediction/target rank mismatch for loss: {prediction.shape} vs {target.shape}"
            )
        if prediction.shape[0] != target.shape[0]:
            n = min(prediction.shape[0], target.shape[0])
            prediction = prediction[-n:]
            target = target[-n:]
            if mask is not None:
                mask = mask[-n:]
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction/target shape mismatch for loss: {prediction.shape} vs {target.shape}"
            )
        if mask is not None and mask.shape != target.shape:
            raise ValueError(f"mask shape {mask.shape} does not match target {target.shape}")
        return prediction, target, mask

    def _calculate_loss(self, dataset_sample: dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply every configured loss through the same explicit contract.

        This bypasses the installed dMG ModelHandler's legacy loss call so the
        dMoT Trainer owns mask, sample and temporal semantics consistently.
        """
        if not hasattr(self.model, "output_dict"):
            raise TypeError("training model must expose output_dict after forward")
        target_name = self._get_target_name()
        target = dataset_sample["target"]
        mask = dataset_sample.get("mask")
        if mask is None:
            mask = torch.isfinite(target)
        sample_ids = dataset_sample.get("batch_sample")
        losses: list[torch.Tensor] = []
        for model_name, output in self.model.output_dict.items():
            if target_name not in output:
                raise KeyError(f"model '{model_name}' has no output '{target_name}'")
            prediction, target_aligned, mask_aligned = self._align_loss_tensors(
                output[target_name], target, mask
            )
            losses.append(
                self.loss_func(
                    prediction,
                    target_aligned,
                    mask=mask_aligned,
                    sample_ids=sample_ids,
                    basin_ids=sample_ids,
                    time_index=dataset_sample.get("time_index"),
                )
            )
        if not losses:
            raise RuntimeError("no model outputs were available for loss computation")
        return torch.stack(losses).sum()

    def _postprocess_prediction_dict(
        self,
        prediction: dict[str, torch.Tensor],
        n_basins: int,
        n_groups: int,
    ) -> dict[str, torch.Tensor]:
        return prediction

    def _setup_training_grid(self) -> tuple[int, int, int]:
        """Setup training grid based on configuration."""
        if self.config.get("data_sampler") == "DlSampler":
            return create_dl_training_grid(
                self.train_dataset["xc_nn_norm"],
                self.config,
            )
        return create_training_grid(
            self.train_dataset["xc_nn_norm"],
            self.config,
        )

    def _log_training_start(self, n_basins: int) -> None:
        """Log training start information."""
        optimizer_name = self.config["train"]["optimizer"]
        lr = self.config["train"]["learning_rate"]
        scheduler_name = self.config["delta_model"]["nn_model"].get(
            "lr_scheduler",
            "None",
        )
        self._emit_progress(
            f"[Train Start] epochs={self.epochs} | optimizer={optimizer_name} | lr={lr} | "
            f"scheduler={scheduler_name} | n_basins={n_basins}"
        )
        sys.stdout.flush()
        sys.stderr.flush()

    def _log_training_end(self, total_time: float, final_loss: float) -> None:
        """Log training end information."""
        self._emit_progress(
            f"[Train End] total_time={total_time:.1f}s | final_loss={final_loss:.4f}"
        )
        sys.stdout.flush()
        sys.stderr.flush()

    def _train_one_epoch_core(
        self,
        epoch: int,
        n_samples: int,
        n_minibatch: int,
        n_timesteps: int,
    ) -> None:
        """Core training logic for one epoch (without progress tracking)."""
        self.current_epoch = epoch
        self.total_loss = 0.0
        self.model.loss_dict = {key: 0.0 for key in self.model.loss_dict}
        num_start = self._get_num_parameter_groups()

        for mb in range(1, n_minibatch + 1):
            self.current_batch = mb
            dataset_sample = self.sampler.get_training_sample(
                self.train_dataset,
                n_samples,
                n_timesteps,
            )

            dataset_sample["target"] = self._prepare_targets_for_loss(
                dataset_sample["target"],
                num_start,
            )

            model_output = self.model(dataset_sample)
            self._assert_finite_output(
                model_output,
                f"model={self._get_primary_model_name()} batch={mb}",
            )
            self._prepare_model_outputs_for_loss(dataset_sample, num_start)
            loss = self._calculate_loss(dataset_sample)

            if not torch.isfinite(loss).all():
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Batch {mb}: loss is non-finite; refusing to skip or sanitize it"
                )

            loss.backward()

            model_parameters = list(self.model.get_parameters())
            bad_gradient_indices = [
                index
                for index, param in enumerate(model_parameters)
                if param.grad is not None and not torch.isfinite(param.grad).all()
            ]
            if bad_gradient_indices:
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Batch {mb}: non-finite gradients in parameter indices "
                    f"{bad_gradient_indices}; gradients were not sanitized"
                )

            torch.nn.utils.clip_grad_norm_(model_parameters, max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.total_loss += loss.item()

            if self.use_scheduler and isinstance(
                self.scheduler,
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            ):
                self.scheduler.step(epoch - 1 + mb / n_minibatch)

        if self.use_scheduler and not isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        ):
            self.scheduler.step()

        self._final_loss = self.total_loss / max(n_minibatch, 1)

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint if needed."""
        if epoch % self.config["train"]["save_epoch"] == 0:
            self.model.save_model(epoch)
            save_training_checkpoint(
                self.config["model_dir"],
                model=self.model,
                epoch=epoch,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                config=self.config,
                sampler=self.sampler,
                clear_prior=True,
            )

    def _evaluate_dataset(
        self,
        dataset: dict,
        out_path: Path,
        start_time: str,
        end_time: str,
    ) -> None:
        num_start = self._get_num_parameter_groups()

        observations = dataset["target"]
        observations = self._prepare_observations_for_metrics(observations, num_start)

        n_samples = dataset["xc_nn_norm"].shape[1]
        batch_start = np.arange(0, n_samples, self.config["test"]["batch_size"])
        batch_end = np.append(batch_start[1:], n_samples)

        log.info(f"Evaluating {start_time} ~ {end_time}: {len(batch_start)} batches")
        batch_predictions = self._forward_loop(dataset, batch_start, batch_end)

        orig_out_path = self.config["out_path"]
        self.config["out_path"] = out_path
        out_path.mkdir(parents=True, exist_ok=True)

        log.info("Saving model outputs + Calculating metrics")
        if self.config.get("save_output", False):
            save_outputsv2(
                self.config,
                batch_predictions,
                observations,
                create_dirs=True,
            )
        self.calc_metrics(batch_predictions, observations)
        self.config["out_path"] = orig_out_path

        del batch_predictions, observations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def evaluate(self) -> None:
        self.is_in_train = False

        base_outpath = Path(self.config["out_path"]).parents[0]
        test_epoch = self.config["test"].get("test_epoch", "")
        datasets_to_eval = []

        if self.train_dataset is not None:
            train_start = self.config["train"].get("start_time", "1989/01/01")
            train_end = self.config["train"].get("end_time", "1998/12/31")
            s_year = train_start.split("/")[0]
            e_year = train_end.split("/")[0]
            folder = base_outpath / f"train{s_year}-{e_year}_Ep{test_epoch}"
            datasets_to_eval.append(
                (self.train_dataset, folder, train_start, train_end)
            )

        if self.eval_dataset is not None:
            eval_start = self.config["test"].get("start_time", "1999/01/01")
            eval_end = self.config["test"].get("end_time", "2009/12/31")
            s_year = eval_start.split("/")[0]
            e_year = eval_end.split("/")[0]
            folder = base_outpath / f"test{s_year}-{e_year}_Ep{test_epoch}"
            datasets_to_eval.append((self.eval_dataset, folder, eval_start, eval_end))

        for dataset, out_path, start_time, end_time in datasets_to_eval:
            self._evaluate_dataset(dataset, out_path, start_time, end_time)
            print(f"Metrics and predictions saved to {out_path}")

    def inference(self):
        self.is_in_train = False

        n_samples = self.dataset["xc_nn_norm"].shape[1]
        batch_start = np.arange(0, n_samples, self.config["simulation"]["batch_size"])
        batch_end = np.append(batch_start[1:], n_samples)

        log.info(f"Inference: Forwarding {len(batch_start)} batches")
        batch_predictions = self._forward_loop(self.dataset, batch_start, batch_end)

        log.info("Saving model outputs")
        save_outputs(self.config, batch_predictions)
        self.predictions = self._batch_data(batch_predictions)
        return self.predictions

    def _batch_data(
        self,
        batch_list: list[dict[str, torch.Tensor]],
        target_key: str = None,
    ):
        data = {}
        try:
            if target_key:
                return (
                    torch.cat([x[target_key] for x in batch_list], dim=1).cpu().numpy()
                )

            for key in batch_list[0].keys():
                data[key] = torch.cat([d[key] for d in batch_list], dim=1).cpu().numpy()
            return data
        except ValueError as exc:
            raise ValueError(f"Error concatenating batch data: {exc}") from exc

    def _forward_loop(
        self,
        data: dict[str, torch.Tensor],
        batch_start: NDArray,
        batch_end: NDArray,
    ):
        batch_predictions = []
        model_name = self._get_primary_model_name()

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
                        key: tensor[warmup_length:, ...].cpu().detach()
                        if tensor.shape[0] > warmup_length
                        else tensor.cpu().detach()
                        for key, tensor in prediction_window[model_name].items()
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

        target = target[self.config["delta_model"]["phy_model"]["warm_up"] :, :]
        target = target[: len(predictions), :]

        metrics_to_compute = self.config["test"].get("metrics", None)
        metrics = Metrics(
            np.swapaxes(predictions.squeeze(), 1, 0),
            np.swapaxes(target.squeeze(), 1, 0),
            metrics_to_compute,
        )
        metrics.dump_metrics(self.config["out_path"])

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

        self._emit_progress(
            f"[Epoch {epoch:>4}/{self.epochs}] loss={avg_loss:.4f} | "
            f"lr={lr:.2e} | time={elapsed:.1f}s"
        )
        sys.stdout.flush()
        sys.stderr.flush()
