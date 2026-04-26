import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import tqdm
from numpy.typing import NDArray

from dmg.core.calc.metrics import Metrics
from dmg.core.data import create_training_grid
# Make sure your factory can import your new PubSampler
from dmg.core.utils.factory import import_data_sampler, load_criterion
from dmg.core.utils.utils import save_outputs, save_train_state
from dmg.models.model_handler import ModelHandler
from dmg.trainers.base import BaseTrainer

log = logging.getLogger(__name__)


class PubTrainer(BaseTrainer):
    """
    Generic trainer adapted for PUB (Predictions in Ungauged Basins) experiments.
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
        super().__init__(config, model)
        self.config = config
        self.model = model or ModelHandler(config)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.verbose = verbose
        self.is_in_train = False
        self.sampler = import_data_sampler(config["data_sampler"])(config)

        if "train" in config["mode"]:
            if not self.train_dataset:
                raise ValueError("'train_dataset' required for training mode.")

            # Now, initialize the PubSampler with the n_basins argument.
            log.info("Initializing experiment")
            self.epochs = self.config["train"]["epochs"]

            # Loss function
            self.loss_func = loss_func or load_criterion(
                self.train_dataset["target"],
                config["loss_function"],
                device=config["device"],
            )
            self.model.loss_func = self.loss_func

            # Optimizer and learning rate scheduler
            self.optimizer = optimizer or self.init_optimizer()
            if config["delta_model"]["nn_model"]["lr_scheduler"]:
                self.use_scheduler = True
                self.scheduler = scheduler or self.init_scheduler()
            else:
                self.use_scheduler = False

            # Resume model training by loading prior states.
            # self.start_epoch = self.config["train"]["start_epoch"] + 1
            # if self.start_epoch > 1:
            self.load_states()
        elif "test" in config["mode"]:
            self.load_test_states()

    def _model_dir(self) -> str:
        return self.config.get("model_dir", self.config["model_path"])

    def _sync_output_dir(self, path: Path | str) -> tuple[str, str]:
        path_str = str(path)
        original = (self.config["out_path"], self.config["sim_dir"])
        self.config["out_path"] = path_str
        self.config["sim_dir"] = path_str
        return original

    # ... (init_optimizer, init_scheduler, load_states methods remain unchanged) ...
    def init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize a state optimizer."""
        name = self.config["train"]["optimizer"]
        learning_rate = self.config["delta_model"]["nn_model"]["learning_rate"]
        optimizer_dict = {
            # 'SGD': torch.optim.SGD,
            # 'Adam': torch.optim.Adam,
            # 'AdamW': torch.optim.AdamW,
            "Adadelta": torch.optim.Adadelta,
            # 'RMSprop': torch.optim.RMSprop,
        }
        cls = optimizer_dict.get(name)
        if cls is None:
            raise ValueError(f"Optimizer '{name}' not recognized.")
        self.optimizer = cls(self.model.get_parameters(), lr=learning_rate)
        return self.optimizer

    def init_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Initialize a learning rate scheduler."""
        name = self.config["delta_model"]["nn_model"]["lr_scheduler"]
        scheduler_dict = {
            "StepLR": torch.optim.lr_scheduler.StepLR,
            "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
            "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
        }
        cls = scheduler_dict.get(name)
        if cls is None:
            raise ValueError(f"Scheduler '{name}' not recognized.")
        self.scheduler = cls(
            self.optimizer, **self.config["delta_model"]["nn_model"]["lr_scheduler_params"]
        )
        return self.scheduler

    def load_states(self) -> None:
        """
        Load model, optimizer, and scheduler states from a checkpoint to resume
        training if a checkpoint file exists.
        """
        path = self._model_dir()
        os.makedirs(path, exist_ok=True)
        for file in os.listdir(path):
            # Check for state checkpoint: looks like `train_state_epoch_XX.pt`.
            if ("train_state" in file): #  and (str(self.start_epoch - 1) in file)
                # log.info("Loading trainer states --> Resuming Training from" /
                #          f" epoch {self.start_epoch}")
                checkpoint = torch.load(os.path.join(path, file))
                # Restore optimizer states
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                # # Restore model
                self.model.load_model(epoch=checkpoint['epoch'])
                self.start_epoch = checkpoint['epoch'] + 1
                print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
                
                if self.scheduler:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

                # Restore random states
                torch.set_rng_state(checkpoint["random_state"])
                if torch.cuda.is_available() and "cuda_random_state" in checkpoint:
                    torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
                return 
            else:
                self.start_epoch = 0

    def load_test_states(self) -> None:
        """Load model states for testing using the configured test epoch."""
        path = self._model_dir()
        test_epoch = self.config["test"].get("test_epoch", None)

        if test_epoch is None:
            raise ValueError("'test_epoch' must be set in config['test'].")

        model_name = self.config["delta_model"]["phy_model"]["model"]
        if isinstance(model_name, list):
            model_name = model_name[0]

        checkpoint_file = f"{str(model_name).lower()}_ep{int(test_epoch)}.pt"
        checkpoint_path = os.path.join(path, checkpoint_file)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"{checkpoint_path} not found.")

        self.model.load_model(epoch=int(test_epoch))
        print(f"Loaded test checkpoint: {checkpoint_path}")

    def train(self) -> None:
        """Train the model."""
        self.is_in_train = True
        n_samples, n_minibatch, n_timesteps = create_training_grid(
            self.train_dataset["xc_nn_norm"], self.config
        )
        log.info(f"Training model: Beginning {self.start_epoch} of {self.epochs} epochs")
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.train_one_epoch(epoch, n_samples, n_minibatch, n_timesteps)

    def train_one_epoch(self, epoch, n_samples, n_minibatch, n_timesteps) -> None:
        """Train model for one epoch."""
        start_time = time.perf_counter()
        prog_str = f"Epoch {epoch}/{self.epochs}"
        self.current_epoch = epoch
        self.total_loss = 0.0

        for mb in tqdm.tqdm(
            range(1, n_minibatch + 1), desc=prog_str, leave=False, dynamic_ncols=True
        ):
            self.current_batch = mb

            dataset_sample = self.sampler.get_training_sample(
                dataset=self.train_dataset, nt=n_timesteps
            )

            _ = self.model(dataset_sample)
            loss = self.model.calc_loss(dataset_sample)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.total_loss += loss.item()

            if self.verbose:
                if mb % 10 == 0:
                    tqdm.tqdm.write(f"Epoch {epoch}, batch {mb} | loss: {loss.item()}")

        if self.use_scheduler:
            self.scheduler.step()

        self._log_epoch_stats(epoch, self.model.loss_dict, n_minibatch, start_time)

        if epoch % self.config["train"]["save_epoch"] == 0:
            self.model.save_model(epoch)
            save_train_state(
                self._model_dir(),
                epoch=epoch,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                clear_prior=False,
            )

    def _get_dataset_indices(self, dataset: dict[str, torch.Tensor], dataset_name: str) -> np.ndarray:
        """Return the basin indices to evaluate for the given PUB dataset."""
        if dataset_name == "eval" and hasattr(self.sampler, "val_indices"):
            return np.asarray(self.sampler.val_indices)
        return np.arange(dataset["target"].shape[1])

    def _evaluate_dataset(
        self,
        dataset: dict[str, torch.Tensor],
        dataset_name: str,
        out_path: Path,
    ) -> None:
        """Evaluate one PUB dataset and save outputs under out_path."""
        basin_indices = self._get_dataset_indices(dataset, dataset_name)
        if basin_indices.size == 0:
            log.warning(f"No basins selected for {dataset_name} dataset evaluation.")
            return

        log.info(f"Validating PUB model on {len(basin_indices)} basins -> {out_path}")
        selected_observations = dataset["target"][:, basin_indices, :]
        model_name = self.config["delta_model"]["phy_model"]["model"][0]

        if self.config["test"]["split_dataset"]:
            dataset_sample = self.sampler.get_validation_sample(dataset, basin_indices)
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
                    key: tensor[t_start:t_end, ...] if len(tensor.shape) > 2 else tensor
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
                key: torch.cat(tensors, dim=0) for key, tensors in collated_chunks.items()
            }
            batch_predictions = [prediction]
            self.predictions = prediction
        else:
            batch_predictions = []
            for basin_idx in tqdm.tqdm(
                basin_indices, desc="Evaluating Basins", leave=False, dynamic_ncols=True
            ):
                dataset_sample = self.sampler.get_validation_sample(dataset, int(basin_idx))
                prediction = self.model(dataset_sample, eval=True)
                prediction = {
                    key: tensor.cpu().detach() for key, tensor in prediction[model_name].items()
                }
                batch_predictions.append(prediction)
            self.predictions = self._batch_data(batch_predictions)

        original_out_path, original_sim_dir = self._sync_output_dir(out_path)
        out_path.mkdir(parents=True, exist_ok=True)

        try:
            log.info("Saving model outputs + Calculating metrics")
            save_outputs(self.config, [self.predictions], selected_observations)
            self.calc_metrics(batch_predictions, selected_observations)
        finally:
            self.config["out_path"] = original_out_path
            self.config["sim_dir"] = original_sim_dir

    def evaluate(self) -> None:
        """Run PUB evaluation on both train and eval datasets when available."""
        self.is_in_train = False

        base_outpath = Path(self.config["out_path"]).parents[0]
        test_epoch = self.config["test"].get("test_epoch", "")
        test_group_id = self.config["test"].get("test_group_id", "")
        datasets_to_eval = []

        if self.train_dataset is not None:
            train_start = self.config["train"].get("start_time", "1989/01/01")
            train_end = self.config["train"].get("end_time", "1999/12/31")
            folder = base_outpath / (
                f"train{train_start.split('/')[0]}-{train_end.split('/')[0]}_Ep{test_epoch}"
            )
            datasets_to_eval.append(("train", self.train_dataset, folder))

        if self.eval_dataset is not None:
            eval_start = self.config["test"].get("start_time", "1989/01/01")
            eval_end = self.config["test"].get("end_time", "1999/12/31")
            folder = base_outpath / (
                f"test{eval_start.split('/')[0]}-{eval_end.split('/')[0]}_Ep{test_epoch}_group{test_group_id}"
            )
            datasets_to_eval.append(("eval", self.eval_dataset, folder))

        for dataset_name, dataset, out_path in datasets_to_eval:
            self._evaluate_dataset(dataset, dataset_name, out_path)
            print(f"{dataset_name} metrics and predictions saved to {out_path}")

    # ... (inference, _batch_data, _forward_loop methods remain) ...
    # NOTE: The _forward_loop is no longer used by evaluate(). If you use inference(),
    # it may need a similar modification to work correctly for PUB.
    def inference(self) -> None:
        """Run batch model inference and save model outputs."""
        self.is_in_train = False

        # Track overall predictions
        batch_predictions = []

        # Get start and end indices for each batch
        n_samples = self.dataset["xc_nn_norm"].shape[1]
        batch_start = np.arange(0, n_samples, self.config["sim"]["batch_size"])
        batch_end = np.append(batch_start[1:], n_samples)

        # Model forward
        log.info(f"Inference: Forwarding {len(batch_start)} batches")
        batch_predictions = self._forward_loop(self.dataset, batch_start, batch_end)

        # Save predictions
        log.info("Saving model outputs")
        save_outputs(self.config, batch_predictions)
        self.predictions = self._batch_data(batch_predictions)

        return self.predictions

    def _batch_data(self, batch_list: list[dict[str, torch.Tensor]], target_key: str = None) -> Any:
        """Merge batch data into a single dictionary."""
        data = {}
        try:
            if target_key:
                # For PUB, each item in batch_list has a 'batch' dim of 1. Cat will merge them.
                return torch.cat([x[target_key] for x in batch_list], dim=1).numpy()
            for key in batch_list[0].keys():
                # Assuming spatial dimension is always 1 for this kind of data
                dim = 1 if len(batch_list[0][key].shape) == 3 else 0
                data[key] = torch.cat([d[key] for d in batch_list], dim=dim).cpu().numpy()
            return data
        except ValueError as e:
            raise ValueError(f"Error concatenating batch data: {e}") from e

    def _forward_loop(self, data: dict, batch_start: NDArray, batch_end: NDArray) -> list:
        """Forward loop for batched evaluation/inference. No longer used by evaluate()."""
        ...

    def calc_metrics(
        self, all_basin_predictions: list[dict[str, torch.Tensor]], observations: torch.Tensor
    ) -> None:
        """Calculate and save model performance metrics for the provided basins."""
        target_name = self.config["train"]["target"][0]
        predictions = self._batch_data(all_basin_predictions, target_name)

        target = np.expand_dims(observations[:, :, 0].cpu().numpy(), 2)
        target = target[self.config["delta_model"]["phy_model"]["warm_up"] :, :]
        target = target[: len(predictions), :, :]

        metrics = Metrics(
            np.swapaxes(predictions.squeeze(), 1, 0),
            np.swapaxes(target.squeeze(), 1, 0),
        )
        metrics.dump_metrics(self.config["out_path"])

    # ... (_log_epoch_stats remains unchanged) ...
    def _log_epoch_stats(
        self, epoch: int, loss_dict: dict, n_minibatch: int, start_time: float
    ) -> None:
        """Log statistics after each epoch."""
        avg_loss_dict = {key: value / n_minibatch for key, value in loss_dict.items()}
        loss_str = ", ".join(f"{key}: {value:.6f}" for key, value in avg_loss_dict.items())
        elapsed = time.perf_counter() - start_time
        mem_alloc = int(torch.cuda.memory_reserved(device=self.config["device"]) / 1e6)
        log.info(
            f"Loss after epoch {epoch}: {loss_str}\n~ Runtime {elapsed:.2f} s, {mem_alloc} Mb reserved GPU memory"
        )
