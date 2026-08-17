"""WarmupTrainer: MyTrainer variant that pushes the current 1-based training
epoch into the physics models so that ``structure_warmup_epochs`` (full-process
parameter warm-up) and ``gate_aic_delay_epochs`` (delayed gate-AIC gradient
exposure) can be honored inside the model forward.

Only the standard training path (``run_model.py`` standard mode) uses this
trainer, and only when the config enables a positive ``structure_warmup_epochs``
or ``gate_aic_delay_epochs``.  With both fields absent or 0, ``MyTrainer`` is
used unchanged.
"""
from __future__ import annotations

from project.bettermodel.implements.my_trainer import MyTrainer


class WarmupTrainer(MyTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._warmup_was_active = False
        self._warmup_epochs = 0
        self._aic_delay_was_active = False
        self._aic_delay_epochs = 0
        for model in getattr(self.model, "model_dict", {}).values():
            phy = getattr(model, "phy_model", None)
            self._warmup_epochs = max(
                self._warmup_epochs, int(getattr(phy, "structure_warmup_epochs", 0) or 0)
            )
            self._aic_delay_epochs = max(
                self._aic_delay_epochs, int(getattr(phy, "gate_aic_delay_epochs", 0) or 0)
            )

    def _sync_epoch(self, epoch: int) -> None:
        for model in getattr(self.model, "model_dict", {}).values():
            phy = getattr(model, "phy_model", None)
            if phy is not None and hasattr(phy, "set_current_epoch"):
                phy.set_current_epoch(epoch)
        active = self._warmup_epochs > 0 and epoch <= self._warmup_epochs
        if active:
            print(f"[structure_warmup] active: epoch {epoch}/{self._warmup_epochs} "
                  f"(effective gates = 1, gate logits frozen)")
        elif self._warmup_was_active:
            print(f"[structure_warmup] released after epoch {epoch - 1}; "
                  f"joint parameter + structure training resumes")
        self._warmup_was_active = active
        delay_active = self._aic_delay_epochs > 0 and epoch <= self._aic_delay_epochs
        if delay_active:
            print(f"[gate_aic_delay] masked: epoch {epoch}/{self._aic_delay_epochs} "
                  f"(AIC value kept in loss; AIC gradient to gates blocked)")
        elif self._aic_delay_was_active:
            print(f"[gate_aic_delay] released after epoch {epoch - 1}; "
                  f"full (fit + AIC) gate gradient restored")
        self._aic_delay_was_active = delay_active

    def train_one_epoch(self, epoch, n_samples, n_minibatch, n_timesteps):
        self._sync_epoch(epoch)
        super().train_one_epoch(epoch, n_samples, n_minibatch, n_timesteps)
