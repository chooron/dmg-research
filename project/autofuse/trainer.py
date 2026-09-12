"""Shared-dPL global-step trainer, diagnostics, and validation module."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from dfuse import PARAMETER_NAMES, StructureSpec, enumerate_structures, get_structure, simulate_coupled_rk2_batched
from dmg.trainers.base import BaseTrainer
from project.autofuse.dpl import DPLConfig, StructureConditionedParameterizer
from project.autofuse.loader import StochasticTimeWindowLoader, TimeWindowConfig
from project.autofuse.metrics import kgecomp_batched
from project.autofuse.parameter_contract import get_parameter_contract
from project.autofuse.samplers import GlobalBasinSampler, ShuffledStructureSampler

log = logging.getLogger(__name__)


@dataclass
class TrainerDiagnostics:
    """Tracks exposure, sparse-head activations, and gradient clipping diagnostics."""
    structure_exposure: dict[int, int] = field(default_factory=dict)
    basin_exposure: dict[str, int] = field(default_factory=dict)
    option_exposure: dict[str, dict[str, int]] = field(default_factory=dict)
    head_activation_count: dict[str, int] = field(default_factory=lambda: {name: 0 for name in PARAMETER_NAMES})
    head_last_activation_step: dict[str, int] = field(default_factory=lambda: {name: -1 for name in PARAMETER_NAMES})
    head_activation_gaps: dict[str, list[int]] = field(default_factory=lambda: {name: [] for name in PARAMETER_NAMES})
    head_update_magnitudes: dict[str, list[float]] = field(default_factory=lambda: {name: [] for name in PARAMETER_NAMES})
    gradient_clipping_history: list[dict[str, Any]] = field(default_factory=list)


class SharedDPLTrainer(BaseTrainer):
    """Global-step shared-dPL trainer for multi-structure FUSE parameter learning.

    Implements the v0 frozen algorithm:
    - K=1: Exactly one legal FUSE structure per global optimization step;
    - Shuffled legal-structure cycle with independent structure RNG;
    - Single global basin cycle with independent basin RNG (no per-structure cursor, no reset on boundary);
    - Stochastic 730-day (365 warmup + 365 scored) time-window sampling;
    - Autograd graph spans full 730 days; loss is evaluated strictly on scored 365 days;
    - Independent parameter coordinate heads ensure inactive parameters have .grad is None;
    - Monotonically increasing global step;
    - Exact checkpointing of model, optimizer, samplers, and RNG states.
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: StructureConditionedParameterizer | None = None,
        *,
        structure_sampler: ShuffledStructureSampler | None = None,
        basin_sampler: GlobalBasinSampler | None = None,
        loader: StochasticTimeWindowLoader | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        compile_step: bool = False,
    ) -> None:
        super().__init__(config, model)
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = getattr(torch, config.get("dtype", "float64")) if isinstance(config.get("dtype", "float64"), str) else config.get("dtype", torch.float64)

        dpl_cfg = DPLConfig(
            attribute_dim=config.get("attribute_dim", 35),
            hidden_dim=config.get("hidden_dim", 64),
            learning_rate=config.get("lr", 1e-3),
        )
        self.model = (model or StructureConditionedParameterizer(dpl_cfg)).to(device=self.device, dtype=self.dtype)

        # Samplers
        self.structure_sampler = structure_sampler or ShuffledStructureSampler(
            config.get("structures", "structures_78"),
            seed=config.get("structure_seed", 20260901),
        )
        self.basin_sampler = basin_sampler or GlobalBasinSampler(
            config.get("basin_ids", ["USA_09447800", "USA_14138900"]),
            batch_size=config.get("batch_size", 100),
            seed=config.get("basin_seed", 20260902),
        )
        self.loader = loader

        self.global_step: int = 0
        self.compile_step = compile_step
        self.max_grad_norm: float | None = config.get("max_grad_norm", None)

        self.optimizer = optimizer or self.init_optimizer()
        self.scheduler = scheduler

        self.diagnostics = TrainerDiagnostics()
        self._init_diagnostics()

    def _init_diagnostics(self) -> None:
        from dfuse.spec import DECISION_ORDER
        for d in DECISION_ORDER:
            self.diagnostics.option_exposure[d] = {}
        for s in self.structure_sampler.structures:
            self.diagnostics.structure_exposure[s] = 0
        for b in self.basin_sampler.basin_ids:
            self.diagnostics.basin_exposure[b] = 0

    def init_optimizer(self) -> torch.optim.Optimizer:
        lr = float(self.config.get("lr", 1e-3))
        opt_name = str(self.config.get("optimizer", "Adam")).lower()
        if opt_name == "adadelta":
            return torch.optim.Adadelta(self.model.parameters(), lr=lr)
        return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=float(self.config.get("weight_decay", 0.0)))

    def init_scheduler(self) -> Any:
        return None

    def train_step(self) -> dict[str, Any]:
        """Execute exactly one global optimization step (K=1)."""
        self.model.train()
        t0_step = time.perf_counter()

        # 1. Sample structure from shuffled cycle (K=1)
        model_id = self.structure_sampler.next_structure()
        spec = get_structure(model_id)
        contract = get_parameter_contract(model_id)

        # 2. Sample basin batch from single global basin cycle
        basin_batch_ids = self.basin_sampler.next_batch()

        # 3. Sample stochastic 730-day time window
        if self.loader is None:
            raise RuntimeError("loader is required for train_step")
        batch = self.loader.sample_batch(basin_batch_ids)

        # 4. Option-wise parameter prediction
        t0_fwd = time.perf_counter()
        forcing = batch.forcing.to(device=self.device, dtype=self.dtype)
        target_scored = batch.target_scored.to(device=self.device, dtype=self.dtype)
        epsilon = batch.epsilon.to(device=self.device, dtype=self.dtype)
        attributes = batch.attributes.to(device=self.device, dtype=self.dtype)

        params = self.model(attributes, model_id)

        # 5. Differentiable simulator forward
        res = simulate_coupled_rk2_batched(
            model_id,
            forcing,
            params,
            basin_ids=batch.basin_ids,
            compile_step=self.compile_step,
            compile_backend=self.config.get("compile_backend", "inductor"),
            compile_fullgraph=bool(self.config.get("compile_fullgraph", True)),
            output_mode="q_only",
        )
        t_fwd = time.perf_counter() - t0_fwd

        # 6. Scored-interval loss computation (last 365 days)
        warmup_len = self.loader.config.warmup_days
        q_scored = res.q[:, warmup_len:]
        kge_scores = kgecomp_batched(q_scored, target_scored, epsilon=epsilon)
        loss = 1.0 - kge_scores.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss ({loss.item()}) encountered at global step {self.global_step} "
                f"for structure {model_id} across basin batch {basin_batch_ids}. Halting training immediately (fail-fast)."
            )

        # 7. Backward pass
        t0_bwd = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        t_bwd = time.perf_counter() - t0_bwd

        # 8. Diagnostics & Gradient Clipping
        pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float("inf")).item())
        if not np.isfinite(pre_clip_norm):
            raise RuntimeError(
                f"Non-finite gradient norm ({pre_clip_norm}) encountered at global step {self.global_step} "
                f"for structure {model_id}. Halting training immediately (fail-fast)."
            )
        clipping_triggered = False
        post_clip_norm = pre_clip_norm

        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            if pre_clip_norm > self.max_grad_norm:
                clipping_triggered = True
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                post_clip_norm = self.max_grad_norm

        # Snapshot weights of active heads before step
        active_heads_before = {
            name: self.model.heads[PARAMETER_NAMES.index(name)].weight.clone()
            for name in contract.active_parameters
        }

        # 9. Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        t_total = time.perf_counter() - t0_step

        # 10. Update exposure and diagnostic accounting
        self.diagnostics.structure_exposure[model_id] = self.diagnostics.structure_exposure.get(model_id, 0) + 1
        for b in basin_batch_ids:
            self.diagnostics.basin_exposure[b] = self.diagnostics.basin_exposure.get(b, 0) + 1
        for d, opt in spec.decisions.items():
            if d in self.diagnostics.option_exposure:
                self.diagnostics.option_exposure[d][opt] = self.diagnostics.option_exposure[d].get(opt, 0) + 1

        for name in PARAMETER_NAMES:
            idx = PARAMETER_NAMES.index(name)
            if name in contract.active_parameters:
                # Active head
                last_step = self.diagnostics.head_last_activation_step[name]
                if last_step >= 0:
                    gap = self.global_step - last_step
                    self.diagnostics.head_activation_gaps[name].append(gap)
                self.diagnostics.head_last_activation_step[name] = self.global_step
                self.diagnostics.head_activation_count[name] += 1

                # Update magnitude
                w_after = self.model.heads[idx].weight
                delta = float((w_after - active_heads_before[name]).norm().item())
                self.diagnostics.head_update_magnitudes[name].append(delta)

        self.diagnostics.gradient_clipping_history.append({
            "global_step": self.global_step,
            "model_id": model_id,
            "pre_clip_norm": pre_clip_norm,
            "clipping_triggered": clipping_triggered,
            "post_clip_norm": post_clip_norm,
        })

        step_output = {
            "global_step": self.global_step,
            "model_id": model_id,
            "batch_size": len(basin_batch_ids),
            "loss": float(loss.item()),
            "mean_kge": float(kge_scores.mean().item()),
            "pre_clip_norm": pre_clip_norm,
            "clipping_triggered": clipping_triggered,
            "timing": {
                "forward_seconds": t_fwd,
                "backward_seconds": t_bwd,
                "total_step_seconds": t_total,
            },
        }

        self.global_step += 1
        return step_output

    def train(self, total_steps: int | None = None) -> list[dict[str, Any]]:
        """Run training for a specified number of global steps."""
        steps = total_steps or int(self.config.get("total_steps", 100))
        history = []
        for _ in range(steps):
            record = self.train_step()
            history.append(record)
        return history

    def evaluate(
        self,
        *,
        structure_subset: Sequence[int | StructureSpec] | None = None,
        basin_subset: Sequence[str] | None = None,
        tail_metric: str = "worst_decile",
    ) -> dict[str, Any]:
        """Run validation evaluation supporting tail-sensitive metrics (worst-decile, mean, etc.)."""
        self.model.eval()
        structs = [int(s.model_id) if isinstance(s, StructureSpec) else int(s) for s in (structure_subset or self.structure_sampler.structures)]
        basins = list(basin_subset or self.basin_sampler.basin_ids)

        kge_by_structure: dict[int, list[float]] = {}
        all_kges: list[float] = []

        if self.loader is None:
            raise RuntimeError("loader is required for evaluate")

        with torch.no_grad():
            for m_id in structs:
                kge_by_structure[m_id] = []
                # In validation, evaluate over the full evaluation horizon or test batch
                batch = self.loader.sample_batch(basins)
                forcing = batch.forcing.to(device=self.device, dtype=self.dtype)
                target_scored = batch.target_scored.to(device=self.device, dtype=self.dtype)
                epsilon = batch.epsilon.to(device=self.device, dtype=self.dtype)
                attributes = batch.attributes.to(device=self.device, dtype=self.dtype)

                params = self.model(attributes, m_id)
                res = simulate_coupled_rk2_batched(
                    m_id,
                    forcing,
                    params,
                    basin_ids=batch.basin_ids,
                    compile_step=self.compile_step,
                    output_mode="q_only",
                )
                warmup_len = self.loader.config.warmup_days
                q_scored = res.q[:, warmup_len:]
                kge_scores = kgecomp_batched(q_scored, target_scored, epsilon=epsilon).detach().cpu().tolist()
                kge_by_structure[m_id].extend(kge_scores)
                all_kges.extend(kge_scores)

        all_kge_arr = np.asarray(all_kges, dtype=np.float64)
        mean_kge = float(np.mean(all_kge_arr))
        median_kge = float(np.median(all_kge_arr))
        p10_kge = float(np.percentile(all_kge_arr, 10))
        p25_kge = float(np.percentile(all_kge_arr, 25))

        return {
            "evaluated_structures": structs,
            "evaluated_basins_count": len(basins),
            "total_evaluations": len(all_kges),
            "mean_kge": mean_kge,
            "median_kge": median_kge,
            "worst_decile_p10_kge": p10_kge,
            "p25_kge": p25_kge,
            "selected_metric_value": p10_kge if tail_metric == "worst_decile" else mean_kge,
            "kge_by_structure_mean": {m_id: float(np.mean(scores)) for m_id, scores in kge_by_structure.items()},
        }

    def inference(self) -> None:
        raise NotImplementedError("inference is implemented via evaluate()")

    def calc_metrics(self, batch_predictions: list[dict[str, Tensor]], observations: Tensor) -> None:
        pass

    def save_checkpoint(self, path: str | Path) -> None:
        """Save full deterministic checkpoint of trainer state."""
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "structure_sampler_state": self.structure_sampler.state_dict(),
            "basin_sampler_state": self.basin_sampler.state_dict(),
            "loader_state": self.loader.state_dict() if self.loader else None,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "diagnostics": {
                "structure_exposure": self.diagnostics.structure_exposure,
                "basin_exposure": self.diagnostics.basin_exposure,
                "option_exposure": self.diagnostics.option_exposure,
                "head_activation_count": self.diagnostics.head_activation_count,
                "head_last_activation_step": self.diagnostics.head_last_activation_step,
                "head_activation_gaps": self.diagnostics.head_activation_gaps,
                "head_update_magnitudes": self.diagnostics.head_update_magnitudes,
            },
        }
        torch.save(payload, ckpt_path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Exact deterministic restoration from checkpoint."""
        ckpt_path = Path(path)
        payload = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.global_step = int(payload["global_step"])
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if self.scheduler and payload.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])

        self.structure_sampler.load_state_dict(payload["structure_sampler_state"])
        self.basin_sampler.load_state_dict(payload["basin_sampler_state"])
        if self.loader and payload.get("loader_state"):
            self.loader.load_state_dict(payload["loader_state"])

        if "torch_rng_state" in payload and payload["torch_rng_state"] is not None:
            rng = payload["torch_rng_state"]
            if isinstance(rng, torch.Tensor):
                rng = rng.cpu().to(torch.uint8)
            torch.set_rng_state(rng)
        if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
            cuda_rng = payload["cuda_rng_state"]
            if isinstance(cuda_rng, torch.Tensor):
                cuda_rng = cuda_rng.cpu().to(torch.uint8)
            torch.cuda.set_rng_state(cuda_rng)

        if "diagnostics" in payload:
            d = payload["diagnostics"]
            self.diagnostics.structure_exposure = {int(k): int(v) for k, v in d.get("structure_exposure", {}).items()}
            self.diagnostics.basin_exposure = {str(k): int(v) for k, v in d.get("basin_exposure", {}).items()}
            self.diagnostics.option_exposure = d.get("option_exposure", {})
            self.diagnostics.head_activation_count = d.get("head_activation_count", {})
            self.diagnostics.head_last_activation_step = d.get("head_last_activation_step", {})
            self.diagnostics.head_activation_gaps = d.get("head_activation_gaps", {})
            self.diagnostics.head_update_magnitudes = d.get("head_update_magnitudes", {})
