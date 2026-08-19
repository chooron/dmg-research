"""Finite-value optimizer transactions for dPL training.

The transaction deliberately never sanitizes a non-finite value.  A failed
pre-step gate clears gradients and either returns a rejected result or raises;
a post-step failure raises and leaves the last successfully saved checkpoint
as the recovery point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import math
from typing import Any, Iterable, Sequence

import torch

_LOG = logging.getLogger(__name__)


@dataclass
class TransactionResult:
    """Structured result for one attempted optimizer update."""

    success: bool
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


class FiniteOptimizerTransaction:
    """Guard one backward/clip/step transaction.

    ``step`` owns ``loss.backward()`` so the finite checks are ordered tightly
    around the optimizer update.  ``failure_policy='raise'`` is the production
    default; ``'skip'`` is useful for callers that already implement a
    skip-bad-batch policy.  Neither policy calls ``optimizer.step`` after a
    failed gate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: Iterable[torch.nn.Parameter],
        *,
        clip_norm: float | None = None,
        scaler: Any | None = None,
        failure_policy: str = "raise",
        logger: logging.Logger | None = None,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]] | None = None,
    ) -> None:
        if failure_policy not in {"raise", "skip"}:
            raise ValueError("failure_policy must be 'raise' or 'skip'")
        if clip_norm is not None and (not torch.isfinite(torch.tensor(clip_norm)) or clip_norm <= 0):
            raise ValueError("clip_norm must be positive and finite when configured")
        self.optimizer = optimizer
        self.parameters = [p for p in parameters if p.requires_grad]
        self._named = list(named_parameters) if named_parameters is not None else [
            (f"parameter[{i}]", p) for i, p in enumerate(self.parameters)
        ]
        self.clip_norm = clip_norm
        self.scaler = scaler
        self.failure_policy = failure_policy
        self.logger = logger or _LOG
        self.successful_steps = 0
        self.rejected_steps = 0
        self.aborted_steps = 0
        self.last_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _finite_tensor(value: Any) -> bool:
        return not torch.is_tensor(value) or bool(torch.isfinite(value).all())

    def parameter_issues(self) -> list[str]:
        return [name for name, p in self._named_parameters() if not self._finite_tensor(p)]

    def gradient_issues(self) -> list[str]:
        return [name for name, p in self._named_parameters() if p.grad is not None and not self._finite_tensor(p.grad)]

    def optimizer_state_issues(self) -> list[str]:
        issues: list[str] = []
        for group_index, group in enumerate(self.optimizer.param_groups):
            for param_index, parameter in enumerate(group["params"]):
                for key, value in self.optimizer.state.get(parameter, {}).items():
                    if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                        issues.append(f"group[{group_index}].param[{param_index}].{key}")
        return issues

    def _named_parameters(self) -> Sequence[tuple[str, torch.nn.Parameter]]:
        return self._named

    def _grad_norm(self) -> float:
        total = torch.zeros((), dtype=torch.float64, device=self.parameters[0].device if self.parameters else "cpu")
        for p in self.parameters:
            if p.grad is not None:
                total = total + p.grad.detach().to(torch.float64).square().sum()
        value = torch.sqrt(total)
        return float(value.detach().cpu())

    def _diagnostics(
        self,
        *,
        reason: str = "",
        epoch: int | None = None,
        batch_index: int | None = None,
        basin_ids: Iterable[Any] | None = None,
        loss: Any | None = None,
        pre_clip_norm: float | None = None,
        post_clip_norm: float | None = None,
        parameter_issues: list[str] | None = None,
        gradient_issues: list[str] | None = None,
        state_issues: list[str] | None = None,
        clipped: bool = False,
    ) -> dict[str, Any]:
        loss_finite = self._finite_tensor(loss) if loss is not None else None
        return {
            "reason": reason,
            "epoch": epoch,
            "batch_index": batch_index,
            "basin_ids": [str(x) for x in basin_ids] if basin_ids is not None else [],
            "loss_finite": loss_finite,
            "pre_clip_grad_norm": pre_clip_norm,
            "post_clip_grad_norm": post_clip_norm,
            "clipped": bool(clipped),
            "nonfinite_parameter_count": len(parameter_issues or []),
            "nonfinite_gradient_count": len(gradient_issues or []),
            "nonfinite_optimizer_state_count": len(state_issues or []),
            "offending_parameters": parameter_issues or [],
            "offending_gradients": gradient_issues or [],
            "offending_optimizer_state": state_issues or [],
            "successful_steps": self.successful_steps,
            "rejected_steps": self.rejected_steps,
            "aborted_steps": self.aborted_steps,
        }

    def _reject(self, diagnostics: dict[str, Any], *, abort: bool = False) -> TransactionResult:
        self.last_diagnostics = diagnostics
        self.rejected_steps += 1
        if abort:
            self.aborted_steps += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.logger.warning("dpl_optimizer_transaction %s", json.dumps(diagnostics, sort_keys=True, default=str))
        if self.failure_policy == "raise" or abort:
            raise FloatingPointError(json.dumps(diagnostics, sort_keys=True, default=str))
        return TransactionResult(False, diagnostics.get("reason", "rejected"), diagnostics)

    def step(
        self,
        loss: torch.Tensor,
        *,
        epoch: int | None = None,
        batch_index: int | None = None,
        basin_ids: Iterable[Any] | None = None,
    ) -> TransactionResult:
        """Run a finite-gated backward, clip, optimizer step, and post-check."""
        if not torch.is_tensor(loss) or loss.numel() != 1 or not bool(torch.isfinite(loss).all()):
            return self._reject(self._diagnostics(reason="nonfinite_loss", epoch=epoch, batch_index=batch_index, basin_ids=basin_ids, loss=loss))

        parameter_issues = self.parameter_issues()
        state_issues = self.optimizer_state_issues()
        if parameter_issues or state_issues:
            return self._reject(self._diagnostics(
                reason="nonfinite_pre_step_state", epoch=epoch, batch_index=batch_index,
                basin_ids=basin_ids, loss=loss, parameter_issues=parameter_issues,
                state_issues=state_issues,
            ))
        if self.scaler is not None:
            # Scaled backward must precede unscale_; clipping and finite checks
            # then operate on true gradients.
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        gradient_issues = self.gradient_issues()
        if gradient_issues:
            return self._reject(self._diagnostics(
                reason="nonfinite_gradient", epoch=epoch, batch_index=batch_index,
                basin_ids=basin_ids, loss=loss, gradient_issues=gradient_issues,
            ))

        pre_clip_norm = self._grad_norm()
        if not torch.isfinite(torch.tensor(pre_clip_norm)):
            return self._reject(self._diagnostics(
                reason="nonfinite_gradient_norm", epoch=epoch, batch_index=batch_index,
                basin_ids=basin_ids, loss=loss, pre_clip_norm=pre_clip_norm,
            ))

        clipped = False
        if self.clip_norm is not None and pre_clip_norm > self.clip_norm:
            # Scale gradients from the float64 norm explicitly.  The generic
            # clip_grad_norm_ implementation can overflow while accumulating
            # very large finite float32 gradients before it applies the scale.
            scale = self.clip_norm / pre_clip_norm
            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            clipped = True
        post_clip_norm = self._grad_norm()
        gradient_issues = self.gradient_issues()
        state_issues = self.optimizer_state_issues()
        if gradient_issues or state_issues or not torch.isfinite(torch.tensor(post_clip_norm)):
            return self._reject(self._diagnostics(
                reason="nonfinite_post_clip_state", epoch=epoch, batch_index=batch_index,
                basin_ids=basin_ids, loss=loss, pre_clip_norm=pre_clip_norm,
                post_clip_norm=post_clip_norm, gradient_issues=gradient_issues,
                state_issues=state_issues, clipped=clipped,
            ))

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        parameter_issues = self.parameter_issues()
        state_issues = self.optimizer_state_issues()
        if parameter_issues or state_issues:
            # This is intentionally fail-fast.  A corrupted Adam state must
            # never be allowed to influence another batch or checkpoint.
            return self._reject(self._diagnostics(
                reason="post_step_corruption", epoch=epoch, batch_index=batch_index,
                basin_ids=basin_ids, loss=loss, pre_clip_norm=pre_clip_norm,
                post_clip_norm=post_clip_norm, parameter_issues=parameter_issues,
                state_issues=state_issues, clipped=clipped,
            ), abort=True)

        self.optimizer.zero_grad(set_to_none=True)
        self.successful_steps += 1
        self.last_diagnostics = self._diagnostics(
            reason="ok", epoch=epoch, batch_index=batch_index, basin_ids=basin_ids,
            loss=loss, pre_clip_norm=pre_clip_norm, post_clip_norm=post_clip_norm,
            clipped=clipped,
        )
        return TransactionResult(True, "ok", self.last_diagnostics)


def validate_finite_training_state(
    parameters: Iterable[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    *,
    loss: torch.Tensor | float | None = None,
) -> None:
    """Reject state that must not be serialized as a normal checkpoint."""
    bad_parameters = [str(i) for i, p in enumerate(parameters) if not bool(torch.isfinite(p).all())]
    bad_state: list[str] = []
    for i, state in enumerate(optimizer.state.values()):
        for key, value in state.items():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                bad_state.append(f"state[{i}].{key}")
    if loss is None:
        loss_bad = False
    elif torch.is_tensor(loss):
        loss_bad = not bool(torch.isfinite(loss).all())
    else:
        loss_bad = not math.isfinite(float(loss))
    if bad_parameters or bad_state or loss_bad:
        raise FloatingPointError(json.dumps({
            "reason": "checkpoint_nonfinite_state",
            "nonfinite_parameters": bad_parameters,
            "nonfinite_optimizer_state": bad_state,
            "loss_finite": not loss_bad,
        }, sort_keys=True))
