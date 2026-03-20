from typing import Any, Optional

import torch

from dmg.models.criterion.base import BaseCriterion


class IRMKgeBatchLoss(BaseCriterion):
    """IRM-augmented KGE loss for Causal-dPL training.

    Extends KgeBatchLoss with Invariant Risk Minimization (IRM) penalty
    (Arjovsky et al. 2019). Enforces that the parameter prediction network
    learns attribute-parameter relationships that are invariant across
    hydrological behavior environments (Gnann et al. cluster groups).

    Total loss
    ----------
    L_total = (1/K) * Σ_e L_KGE^e  +  λ * Σ_e ||∇_w L_KGE^e(w · ŷ_e)||²

    The second term (IRM penalty) uses the "dummy scalar" trick: a scalar
    w=1.0 is introduced to scale predictions; its gradient norm measures how
    much the loss landscape differs across environments. A large gradient means
    the current predictor is not at a simultaneous optimum across environments,
    i.e., some relationships only hold in specific environments (spurious).

    Parameters
    ----------
    config : dict
        Configuration dictionary. Relevant keys:

        - ``lambda_irm`` : IRM penalty coefficient. Default: ``1.0``.
        - ``irm_warmup_epochs`` : Epochs before IRM penalty activates.
          During warmup the model trains with pure KGE (ERM) to reach a
          reasonable initialisation before invariance is enforced.
          Default: ``10``.
        - ``eps`` : Stability term for KGE computation. Default: ``0.1``.

    device : str, optional
        Device to run on. Default: ``'cpu'``.

    Notes
    -----
    Calling convention (training loop)::

        loss_fn = IRMKgeBatchLoss(config, device)

        # env_data: list of (y_pred_e, y_obs_e) per environment
        loss = loss_fn(
            y_pred=None,
            y_obs=None,
            environments=env_data,
            current_epoch=epoch,
        )

    Fallback: if ``environments`` is not provided, behaves as standard
    KgeBatchLoss (pure ERM, IRM penalty = 0). This keeps baseline and
    Causal-dPL runs using the same loss class.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = 'cpu',
        **kwargs: Any,
    ) -> None:
        super().__init__(config, device)
        self.name = 'IRM KGE Batch Loss'
        self.eps = kwargs.get('eps', config.get('eps', 0.1))
        self.lambda_irm = kwargs.get(
            'lambda_irm', config.get('lambda_irm', 1.0)
        )
        self.irm_warmup_epochs = kwargs.get(
            'irm_warmup_epochs', config.get('irm_warmup_epochs', 10)
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        y_pred: Optional[torch.Tensor] = None,
        y_obs: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute IRM-augmented KGE loss.

        Parameters
        ----------
        y_pred : torch.Tensor, optional
            Predicted values ``[T, B]``. Used only in ERM fallback mode.
        y_obs : torch.Tensor, optional
            Observed values ``[T, B]``. Used only in ERM fallback mode.
        **kwargs
            - ``environments`` : list of ``(y_pred_e, y_obs_e)`` tuples,
              one per training environment. If absent, falls back to ERM.
            - ``current_epoch`` : int. Controls warmup and lambda schedule.

        Returns
        -------
        torch.Tensor
            Scalar loss value (differentiable).
        """
        environments: Optional[list] = kwargs.get('environments', None)
        current_epoch: int = kwargs.get('current_epoch', 9999)

        # ---- Fallback: standard KGE (ERM) ----
        if environments is None:
            assert y_pred is not None and y_obs is not None, (
                "Either 'environments' or (y_pred, y_obs) must be provided."
            )
            return self._kge_loss(y_pred, y_obs)

        # ---- IRM mode ----
        assert len(environments) >= 2, (
            f"IRM requires >= 2 environments, got {len(environments)}."
        )

        # IRM penalty — zero during warmup for stable initialisation
        if current_epoch < self.irm_warmup_epochs:
            env_losses = [
                self._kge_loss(y_pred_e, y_obs_e)
                for y_pred_e, y_obs_e in environments
            ]
            erm_loss = torch.stack(env_losses).mean()
            penalty = torch.tensor(0.0, device=self.device)
        else:
            effective_lambda = self.get_lambda(current_epoch)
            erm_loss, irm_penalty = self._erm_and_irm_penalty(environments)
            penalty = effective_lambda * irm_penalty

        return erm_loss + penalty

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _kge_loss(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Single-environment KGE loss (1 - KGE).

        Identical computation to KgeBatchLoss.forward; separated here so
        it can be called both for the ERM term and inside _irm_penalty.
        """
        prediction, target = self._format(y_pred, y_obs)

        mask = ~torch.isnan(target)
        p_sub = prediction[mask]
        t_sub = target[mask]

        mean_p = torch.mean(p_sub)
        mean_t = torch.mean(t_sub)
        std_p  = torch.std(p_sub)
        std_t  = torch.std(t_sub)

        numerator   = torch.sum((p_sub - mean_p) * (t_sub - mean_t))
        denominator = torch.sqrt(
            torch.sum((p_sub - mean_p) ** 2)
            * torch.sum((t_sub - mean_t) ** 2)
        )
        r     = numerator / (denominator + self.eps)
        beta  = mean_p / (mean_t + self.eps)
        gamma = std_p  / (std_t  + self.eps)

        kge = 1 - torch.sqrt(
            (r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2
        )
        return 1 - kge

    def _erm_and_irm_penalty(
        self,
        environments: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ERM and IRM penalty in one pass to reduce graph duplication.

        For each environment e:
            w_e  = scalar 1.0  (requires_grad=True, fresh each call)
            L_e  = KGE_loss(w_e * ŷ_e, y_obs_e)
            p_e  = ||dL_e / dw_e||²

        Intuition: if the predictor is simultaneously optimal across all
        environments, the gradient of L w.r.t. w is zero everywhere.
        A non-zero gradient means the predictor could still improve in
        that environment, i.e., it relies on environment-specific correlations.

        Parameters
        ----------
        environments : list of (y_pred_e, y_obs_e)
            Per-environment prediction/observation pairs. Predictions must
            already have ``requires_grad=True`` (retained from model forward).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Mean ERM loss and summed IRM penalty across environments.
        """
        penalty = torch.tensor(0.0, device=self.device)
        erm_loss = torch.tensor(0.0, device=self.device)

        for y_pred_e, y_obs_e in environments:
            # Fresh scalar per environment — NOT shared
            w_e = torch.ones(1, requires_grad=True, device=self.device)

            # Scale predictions through w_e to build the gradient path
            y_scaled = w_e * y_pred_e

            loss_e = self._kge_loss(y_scaled, y_obs_e)

            # create_graph=True: keep the computation graph for backprop
            # through the penalty into the parameter prediction network
            grad_e = torch.autograd.grad(
                outputs=loss_e,
                inputs=w_e,
                create_graph=True,
            )[0]

            erm_loss = erm_loss + loss_e
            penalty = penalty + grad_e.pow(2).sum()

        erm_loss = erm_loss / len(environments)
        return erm_loss, penalty

    # ------------------------------------------------------------------
    # Lambda schedule
    # ------------------------------------------------------------------

    def get_lambda(self, current_epoch: int) -> float:
        """Linear ramp schedule for the IRM penalty coefficient.

        - Epochs  0 … warmup-1  : λ_eff = 0   (pure ERM, stable init)
        - Epochs  warmup … 2×warmup-1 : λ_eff ramps 0 → λ_irm
        - Epochs  ≥ 2×warmup    : λ_eff = λ_irm  (full penalty)

        Call from the training loop to update ``self.lambda_irm`` dynamically,
        or pass ``current_epoch`` to ``forward`` and let it call this
        internally via ``_irm_penalty``.

        Parameters
        ----------
        current_epoch : int
            Current training epoch (0-indexed).

        Returns
        -------
        float
            Effective lambda for the current epoch.

        Example
        -------
        >>> for epoch in range(n_epochs):
        ...     loss_fn.lambda_irm = loss_fn.get_lambda(epoch)
        """
        if current_epoch < self.irm_warmup_epochs:
            return 0.0
        ramp = (current_epoch - self.irm_warmup_epochs) / max(self.irm_warmup_epochs, 1)
        return float(min(self.lambda_irm, self.lambda_irm * ramp))
