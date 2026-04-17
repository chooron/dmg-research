from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from implements.causal_dpl_model import CausalDplModel
from models.hbv_static import HbvStatic
from models.nns.mc_mlp import McMlpModel


_PAPER_VARIANT_TO_NN = {
    "deterministic": "DeterministicParamModel",
    "mc_dropout": "McMlpModel",
    "distributional": "DistributionalParamModel",
}


def _normalize_seed_list(raw_value: Any) -> list[int]:
    if raw_value is None:
        return []
    if isinstance(raw_value, int):
        return [int(raw_value)]
    if isinstance(raw_value, str):
        tokens = raw_value.replace(",", " ").split()
        return [int(token) for token in tokens]
    return [int(value) for value in raw_value]


def normalize_paper_config(raw_config: dict[str, Any]) -> None:
    """Normalize paper-facing config into the main-split runtime contract.

    Mutates ``raw_config`` in place. Expects a nested config dict and rewrites
    paper selectors into existing runtime keys such as ``trainer``,
    ``model.nn.name``, and MC-evaluation flags.
    """
    paper_cfg = raw_config.setdefault("paper", {})
    variant = str(paper_cfg.get("variant", "mc_dropout")).lower()
    if variant not in _PAPER_VARIANT_TO_NN:
        raise ValueError(
            f"Unsupported paper.variant '{variant}'. Expected one of: "
            f"{', '.join(sorted(_PAPER_VARIANT_TO_NN))}."
        )

    split = str(paper_cfg.get("split", "main")).lower()
    if split != "main":
        raise ValueError(
            f"Unsupported paper.split '{split}'. Slice 1 only supports 'main'."
        )

    seeds = _normalize_seed_list(paper_cfg.get("seeds", raw_config.get("seeds")))
    if not seeds:
        seeds = [int(raw_config.get("seed", 42))]
    paper_cfg["variant"] = variant
    paper_cfg["split"] = split
    paper_cfg["seeds"] = seeds

    raw_config["seed"] = int(raw_config.get("seed", seeds[0]))
    raw_config["seeds"] = seeds
    raw_config["trainer"] = "ParamLearnTrainer"
    raw_config.setdefault("data_loader", "HydroLoader")
    raw_config.setdefault("data_sampler", "HydroSampler")

    model_cfg = raw_config.setdefault("model", {})
    phy_cfg = model_cfg.setdefault("phy", {})
    nn_cfg = model_cfg.setdefault("nn", {})
    nn_cfg["name"] = _PAPER_VARIANT_TO_NN[variant]
    nn_cfg.setdefault("forcings", [])
    nn_cfg.setdefault("attributes", [])
    nn_cfg["output_activation"] = "sigmoid"
    nn_cfg.setdefault("static_pool", "last")
    phy_cfg["nmul"] = int(phy_cfg.get("nmul", 1) or 1)

    test_cfg = raw_config.setdefault("test", {})
    if variant == "mc_dropout":
        test_cfg["mc_dropout"] = True
        test_cfg["mc_samples"] = max(int(test_cfg.get("mc_samples", 100)), 1)
    else:
        test_cfg["mc_dropout"] = False
        test_cfg["mc_samples"] = 1


def validate_paper_config(config: dict[str, Any]) -> None:
    """Validate the normalized paper config for main-split parameter learning.

    Parameters
    ----------
    config : dict
        Runtime config expected to target ``ParamLearnTrainer`` with static-basin
        inputs and ``HbvStatic``-compatible parameter outputs.
    """
    paper_cfg = config.get("paper", {})
    variant = str(paper_cfg.get("variant", "")).lower()
    if variant not in _PAPER_VARIANT_TO_NN:
        raise ValueError(f"Unknown paper.variant '{variant}'.")

    if str(config.get("trainer", "")) != "ParamLearnTrainer":
        raise ValueError("Paper stack must target ParamLearnTrainer.")

    data_cfg = config.get("data", {})
    for key in ("basin_ids_path", "basin_ids_reference_path"):
        if not data_cfg.get(key):
            raise ValueError(
                f"Paper stack requires data.{key} for ParamLearnTrainer runtime."
            )

    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("Paper stack requires a 'model' mapping in the config.")

    nn_cfg = model_cfg.get("nn")
    if not isinstance(nn_cfg, dict):
        raise ValueError("Paper stack requires a 'model.nn' mapping in the config.")

    phy_cfg = model_cfg.get("phy")
    if not isinstance(phy_cfg, dict):
        raise ValueError("Paper stack requires a 'model.phy' mapping in the config.")

    if nn_cfg.get("forcings"):
        raise ValueError(
            "Paper stack expects static basin attributes only; set model.nn.forcings to []."
        )
    if not nn_cfg.get("attributes"):
        raise ValueError(
            "Paper stack requires at least one static attribute in model.nn.attributes."
        )
    if str(nn_cfg.get("output_activation", "sigmoid")).lower() != "sigmoid":
        raise ValueError(
            "Paper stack expects output_activation='sigmoid' for HbvStatic."
        )
    if int(phy_cfg.get("nmul", 1) or 1) != 1:
        raise ValueError("Paper stack currently requires model.phy.nmul == 1.")


class _StaticParamBase(nn.Module):
    def __init__(self, config: dict[str, Any], nx: int, ny: int) -> None:
        super().__init__()
        self.nx = int(nx)
        self.ny = int(ny)
        self.hidden_size = int(config.get("hidden_size", 128))
        self.output_activation = str(config.get("output_activation", "sigmoid")).lower()
        self.static_pool = str(config.get("static_pool", "last")).lower()

    def _pool_static_input(self, x: torch.Tensor) -> tuple[torch.Tensor, int | None]:
        """Pool static inputs into a basin-level feature tensor.

        Parameters
        ----------
        x : torch.Tensor
            Either ``[B, nx]`` or ``[T, B, nx]`` static-basin features.

        Returns
        -------
        tuple[torch.Tensor, int | None]
            ``(pooled, nt)`` where ``pooled`` has shape ``[B, nx]`` and
            ``nt`` is the original time dimension when ``x`` was 3D.
        """
        if x.ndim == 2:
            if x.shape[-1] != self.nx:
                raise ValueError(
                    f"{type(self).__name__} expected {self.nx} input features, got "
                    f"{x.shape[-1]}."
                )
            return x, None
        if x.ndim != 3:
            raise ValueError(
                f"{type(self).__name__} expects a 2D or 3D tensor, got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.nx:
            raise ValueError(
                f"{type(self).__name__} expected {self.nx} input features, got "
                f"{x.shape[-1]}."
            )
        nt = x.shape[0]
        pooled = x.mean(dim=0) if self.static_pool == "mean" else x[-1]
        return pooled, nt

    def _apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation == "softplus":
            return F.softplus(x)
        if self.output_activation in {"identity", "none"}:
            return x
        raise ValueError(
            f"Unsupported output activation '{self.output_activation}'."
        )

    def _repeat_over_time(self, output: torch.Tensor, nt: int | None) -> torch.Tensor:
        if nt is None:
            return output
        return output.unsqueeze(0).repeat(nt, 1, 1).contiguous()


class DeterministicParamModel(_StaticParamBase):
    def __init__(self, config: dict[str, Any], nx: int, ny: int) -> None:
        super().__init__(config, nx, ny)
        self.name = "DeterministicParamModel"
        self.layers = nn.Sequential(
            nn.Linear(self.nx, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.ny),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, nt = self._pool_static_input(x)
        output = self._apply_output_activation(self.layers(pooled))
        if output.shape[-1] != self.ny:
            raise RuntimeError(
                f"DeterministicParamModel produced {output.shape[-1]} outputs, "
                f"expected {self.ny}."
            )
        return self._repeat_over_time(output, nt)


class DistributionalParamModel(_StaticParamBase):
    def __init__(self, config: dict[str, Any], nx: int, ny: int) -> None:
        super().__init__(config, nx, ny)
        self.name = "DistributionalParamModel"
        self.logstd_min = float(config.get("distribution", {}).get("logstd_min", -5.0))
        self.logstd_max = float(config.get("distribution", {}).get("logstd_max", 2.0))
        self.backbone = nn.Sequential(
            nn.Linear(self.nx, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        self.latent_mu_head = nn.Linear(self.hidden_size, self.ny)
        self.latent_logstd_head = nn.Linear(self.hidden_size, self.ny)

    def _distribution_stats(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int | None]:
        pooled, nt = self._pool_static_input(x)
        hidden = self.backbone(pooled)
        latent_mu = self.latent_mu_head(hidden)
        latent_logstd = torch.clamp(
            self.latent_logstd_head(hidden),
            self.logstd_min,
            self.logstd_max,
        )
        return latent_mu, latent_logstd, nt

    def sample_parameters(self, x: torch.Tensor) -> torch.Tensor:
        """Sample one bounded parameter tensor in parameter space.

        Parameters
        ----------
        x : torch.Tensor
            Static-basin inputs with shape ``[B, nx]`` or ``[T, B, nx]``.

        Returns
        -------
        torch.Tensor
            Sampled bounded parameter tensor with shape ``[B, ny]`` or
            ``[T, B, ny]`` compatible with ``HbvStatic``.
        """
        parameters, _, _ = self.sample_parameters_with_stats(x)
        return parameters

    def sample_parameters_with_stats(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one bounded parameter tensor plus latent-space statistics.

        Parameters
        ----------
        x : torch.Tensor
            Static-basin inputs with shape ``[B, nx]`` or ``[T, B, nx]``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(parameters, latent_mu, latent_logstd)`` where each tensor is
            ``[B, ny]`` or ``[T, B, ny]`` after time expansion.
        """
        latent_mu, latent_logstd, nt = self._distribution_stats(x)
        eps = torch.randn_like(latent_mu)
        latent_sample = latent_mu + torch.exp(latent_logstd) * eps
        parameters = self._apply_output_activation(latent_sample)
        parameters = self._repeat_over_time(parameters, nt)

        latent_mu = self._repeat_over_time(latent_mu, nt)
        latent_logstd = self._repeat_over_time(latent_logstd, nt)
        return parameters, latent_mu, latent_logstd

    def estimate_parameter_moments(
        self,
        x: torch.Tensor,
        n_samples: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate parameter-space moments by repeated bounded sampling.

        Parameters
        ----------
        x : torch.Tensor
            Static-basin inputs with shape ``[B, nx]`` or ``[T, B, nx]``.
        n_samples : int, default=100
            Number of latent-space draws used to estimate parameter-space mean
            and standard deviation.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Empirical ``(param_mean, param_std)`` in parameter space, each with
            shape ``[B, ny]`` or ``[T, B, ny]``.
        """
        if int(n_samples) < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}.")

        samples = [self.sample_parameters(x) for _ in range(int(n_samples))]
        parameter_samples = torch.stack(samples, dim=0)
        param_mean = parameter_samples.mean(dim=0)
        param_std = parameter_samples.std(dim=0, unbiased=False)
        return param_mean, param_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sample_parameters(x)


def build_paper_dpl(config: dict[str, Any]) -> CausalDplModel:
    """Build the paper-facing DPL model on top of the shared HBV boundary.

    Parameters
    ----------
    config : dict
        Runtime config with ``model.phy`` and ``model.nn`` sections.

    Returns
    -------
    CausalDplModel
        Model whose NN emits ``[T, B, ny]`` parameter tensors for ``HbvStatic``.
    """
    phy_cfg = config["model"]["phy"]
    nn_cfg = config["model"]["nn"]
    phy_model = HbvStatic(config=phy_cfg, device=config["device"])
    nx = len(nn_cfg["forcings"]) + len(nn_cfg["attributes"])
    ny = phy_model.learnable_param_count
    nn_name = str(nn_cfg.get("name", ""))

    if nn_name == "DeterministicParamModel":
        nn_model = DeterministicParamModel(nn_cfg, nx=nx, ny=ny)
    elif nn_name in {"McMlpModel", "McMlp", "mc_mlp"}:
        nn_model = McMlpModel(nn_cfg, nx=nx, ny=ny)
    elif nn_name == "DistributionalParamModel":
        nn_model = DistributionalParamModel(nn_cfg, nx=nx, ny=ny)
    else:
        raise ValueError(f"Unsupported paper nn model '{nn_name}'.")

    return CausalDplModel(
        phy_model=phy_model,
        nn_model=nn_model,
        config=config,
        device=config["device"],
    )


def write_run_metadata(config: dict[str, Any]) -> str:
    """Write a flat JSON summary for one paper-stack run.

    Parameters
    ----------
    config : dict
        Runtime config containing paper selector, model, train/test, and data
        sections.

    Returns
    -------
    str
        Absolute path to the written ``run_meta.json`` file.
    """
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_cfg = config.get("paper", {})
    loss_cfg = config.get("train", {}).get("loss_function", {})
    test_cfg = config.get("test", {})
    nn_cfg = config.get("model", {}).get("nn", {})
    data_cfg = config.get("data", {})
    payload = {
        "paper_variant": str(paper_cfg.get("variant", "")),
        "seed": int(config["seed"]),
        "split": str(paper_cfg.get("split", "main")),
        "nn_name": str(nn_cfg.get("name", "")),
        "loss_name": str(loss_cfg.get("name", "unknown")),
        "mc_samples": int(test_cfg.get("mc_samples", 1)),
        "output_activation": str(nn_cfg.get("output_activation", "")),
        "static_pool": str(nn_cfg.get("static_pool", "")),
        "paper_seeds": list(_normalize_seed_list(paper_cfg.get("seeds", []))),
        "data_basin_ids_path": str(data_cfg.get("basin_ids_path", "")),
    }
    path = output_dir / "run_meta.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return os.fspath(path)
