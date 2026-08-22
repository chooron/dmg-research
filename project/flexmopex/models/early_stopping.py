"""Shared training-side early-stopping controller for formal Flex-MOPEX runs.

The controller deliberately consumes only the scalar loss used for the current
optimization step.  It has no access to validation/test data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EarlyStoppingController:
    """Track a training objective after a mandatory minimum epoch."""

    enabled: bool
    min_epochs: int = 50
    patience: int = 20
    min_delta: float = 1.0e-4
    monitor: str = "train_loss"
    mode: str = "min"
    best_value: float | None = None
    best_epoch: int | None = None
    wait_count: int = 0
    stop_epoch: int | None = None
    reason: str = "max_epochs_reached"
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EarlyStoppingController":
        settings = config.get("train", {}).get("early_stopping", {}) or {}
        return cls(
            enabled=bool(settings.get("enabled", False)),
            min_epochs=max(1, int(settings.get("min_epochs", 50))),
            patience=max(1, int(settings.get("patience", 20))),
            min_delta=max(0.0, float(settings.get("min_delta", 1.0e-4))),
            monitor=str(settings.get("monitor", "train_loss")),
            mode=str(settings.get("mode", "min")).lower(),
        )

    def update(self, epoch: int, value: float) -> bool:
        """Record one epoch and return whether training should stop now."""
        value = float(value)
        if self.mode not in {"min", "max"}:
            raise ValueError(f"Unsupported early-stopping mode: {self.mode!r}")

        eligible = self.enabled and epoch >= self.min_epochs
        is_finite = value == value and abs(value) != float("inf")
        improved = False
        if eligible and is_finite:
            if self.best_value is None:
                improved = True
            elif self.mode == "min":
                improved = value < self.best_value - self.min_delta
            else:
                improved = value > self.best_value + self.min_delta

            if improved:
                self.best_value = value
                self.best_epoch = epoch
                self.wait_count = 0
            else:
                self.wait_count += 1
        elif eligible:
            self.wait_count += 1

        row = {
            "epoch": int(epoch),
            "monitor": self.monitor,
            "value": value,
            "eligible": eligible,
            "finite": is_finite,
            "improved": improved,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "wait_count": self.wait_count,
        }
        self.history.append(row)

        should_stop = bool(
            self.enabled
            and epoch >= self.min_epochs
            and self.best_epoch is not None
            and self.wait_count >= self.patience
        )
        if should_stop:
            self.stop_epoch = epoch
            self.reason = "patience_exhausted"
        return should_stop

    def finalize(self, stop_epoch: int, max_epochs: int) -> None:
        if self.stop_epoch is None:
            self.stop_epoch = int(stop_epoch)
        if self.enabled and self.stop_epoch < max_epochs and self.reason == "max_epochs_reached":
            self.reason = "patience_exhausted"
        elif not self.enabled:
            self.reason = "disabled"
        else:
            self.reason = "max_epochs_reached"

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "monitor": self.monitor,
            "mode": self.mode,
            "min_epochs": self.min_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "stop_epoch": self.stop_epoch,
            "early_stop_reason": self.reason,
            "history": self.history,
        }
