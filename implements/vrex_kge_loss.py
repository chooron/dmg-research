from typing import Any, Optional

import torch

from dmg.models.criterion.base import BaseCriterion


class VRExKgeBatchLoss(BaseCriterion):
    """VREx-augmented KGE loss for Invariant-dPL training.

    Extends KgeBatchLoss with Variance Risk Extrapolation (VREx) penalty
    (Krueger et al. 2021), motivated by Invariant Causal Prediction theory
    (Peters et al. 2016). Penalises the variance of per-environment losses,
    encouraging solutions that perform equally well across all hydrological
    behavior environments (Gnann et al. cluster groups).

    Total loss
    ----------
    L_total = (1/K) * Σ_e L_KGE^e  +  λ * Var_w({L_KGE^e})

    where Var_w is basin-count-weighted variance across environment losses.
    Weighting prevents small clusters from dominating the penalty signal.

    Compared with IRM, VREx does not require second-order gradients, making
    it compatible with torch.compile and stable under small environment counts.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Relevant keys:

        - ``lambda_vrex`` : VREx penalty coefficient. Default: ``1.0``.
        - ``vrex_warmup_epochs`` : Epochs of pure ERM before penalty activates.
          After warmup, lambda ramps linearly from 0 to ``lambda_vrex`` over
          the next ``vrex_warmup_epochs`` epochs. Default: ``10``.
        - ``eps`` : Stability term for KGE computation. Default: ``0.1``.

    device : str, optional
        Device to run on. Default: ``'cpu'``.

    Notes
    -----
    Calling convention (training loop)::

        loss_fn = VRExKgeBatchLoss(config, device)

        environments = [
            (y_pred_e, y_obs_e, n_basins_e),   # one tuple per environment
            ...
        ]
        loss = loss_fn(
            environments=environments,
            current_epoch=epoch,
        )

    Fallback: if ``environments`` is not provided, behaves as standard
    KgeBatchLoss (pure ERM, VREx penalty = 0). Baseline and Invariant-dPL
    runs can therefore share the same loss class.

    Basin count weighting
    ---------------------
    Each environment loss is weighted by its basin count before variance
    computation. This prevents small clusters (≈30 basins) from dominating
    the penalty signal relative to large clusters (≈150 basins).

    Weighted mean  : L̄  = Σ_e n_e * L_e / Σ_e n_e
    Weighted var   : V_w = Σ_e n_e * (L_e - L̄)² / (Σ_e n_e - 1)
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = 'cpu',
        **kwargs: Any,
    ) -> None:
        super().__init__(config, device)
        self.name = 'VREx KGE Batch Loss'
        self.eps = kwargs.get('eps', config.get('eps', 0.1))
        self.lambda_vrex = kwargs.get(
            'lambda_vrex', config.get('lambda_vrex', 1.0)
        )
        self.vrex_warmup_epochs = kwargs.get(
            'vrex_warmup_epochs', config.get('vrex_warmup_epochs', 10)
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
        """Compute VREx-augmented KGE loss.

        Parameters
        ----------
        y_pred : torch.Tensor, optional
            Predicted values ``[T, B]``. Used only in ERM fallback mode.
        y_obs : torch.Tensor, optional
            Observed values ``[T, B]``. Used only in ERM fallback mode.
        **kwargs
            - ``environments`` : list of ``(y_pred_e, y_obs_e, n_basins_e)``
              tuples, one per training environment.
              ``n_basins_e`` (int) is the number of basins in environment e,
              used for weighted variance. If absent, falls back to ERM.
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

        # ---- VREx mode ----
        assert len(environments) >= 2, (
            f"VREx requires >= 2 environments, got {len(environments)}."
        )

        # Unpack environments — support both 2-tuple and 3-tuple
        env_preds, env_obs, env_sizes = self._unpack_environments(environments)

        # Per-environment KGE losses (ERM term)
        env_losses = [
            self._kge_loss(y_pred_e, y_obs_e)
            for y_pred_e, y_obs_e in zip(env_preds, env_obs)
        ]
        erm_loss = torch.stack(env_losses).mean()

        # VREx penalty — zero during warmup for stable initialisation
        effective_lambda = self.get_lambda(current_epoch)  # 0.0 during warmup
        if effective_lambda == 0.0:
            penalty = torch.tensor(0.0, device=self.device)
        else:
            penalty = effective_lambda * self._vrex_penalty(env_losses, env_sizes)

        return erm_loss + penalty

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unpack_environments(
        self,
        environments: list,
    ) -> tuple[list, list, list[int]]:
        """Unpack environments into predictions, observations, and basin counts.

        Supports two calling formats:
            - 2-tuple: (y_pred_e, y_obs_e)         → equal weighting
            - 3-tuple: (y_pred_e, y_obs_e, n_basins_e) → weighted variance

        Parameters
        ----------
        environments : list of 2-tuple or 3-tuple

        Returns
        -------
        env_preds : list of torch.Tensor
        env_obs   : list of torch.Tensor
        env_sizes : list of int
            Basin counts. Inferred from y_pred shape if not provided.
        """
        env_preds, env_obs, env_sizes = [], [], []
        for env in environments:
            if len(env) == 3:
                y_pred_e, y_obs_e, n_e = env
            else:
                y_pred_e, y_obs_e = env
                # Infer from tensor shape [T, B] → B is basin count
                n_e = y_pred_e.shape[-1] if y_pred_e.ndim >= 2 else 1
            env_preds.append(y_pred_e)
            env_obs.append(y_obs_e)
            env_sizes.append(int(n_e))
        return env_preds, env_obs, env_sizes

    def _kge_loss(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Single-environment KGE loss (1 - KGE)."""
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

    def _vrex_penalty(
        self,
        env_losses: list[torch.Tensor],
        env_sizes: list[int],
    ) -> torch.Tensor:
        """Basin-count-weighted variance of per-environment KGE losses.

        Intuition: the penalty is zero only when all environments have
        identical loss — i.e., the model extrapolates uniformly. Large
        variance means the model fits some environments much better than
        others, implying it exploits environment-specific correlations.

        Weighted formulation prevents small clusters from dominating:
            L̄  = Σ_e n_e * L_e / N          (weighted mean)
            V_w = Σ_e n_e * (L_e - L̄)² / (N - 1)  (weighted unbiased var)
        where N = Σ_e n_e (total basin count across training environments).

        Parameters
        ----------
        env_losses : list of torch.Tensor
            Scalar KGE loss for each environment.
        env_sizes : list of int
            Number of basins in each environment.

        Returns
        -------
        torch.Tensor
            Scalar weighted variance (VREx penalty).
        """
        stacked = torch.stack(env_losses)   # [K]

        weights = torch.tensor(
            env_sizes, dtype=torch.float32, device=self.device
        )                                   # [K]
        N = weights.sum()                   # total basins

        # Weighted mean
        weighted_mean = (weights * stacked).sum() / N

        # Weighted unbiased variance: N in denominator uses (N-1) for Bessel
        weighted_var = (weights * (stacked - weighted_mean) ** 2).sum() / (N - 1)

        return weighted_var

    # ------------------------------------------------------------------
    # Lambda schedule
    # ------------------------------------------------------------------

    def get_lambda(self, current_epoch: int) -> float:
        """Linear ramp schedule for the VREx penalty coefficient.

        Epoch range            Effective λ
        ─────────────────────────────────────────
        [0,       warmup)    : 0               (pure ERM)
        [warmup,  2×warmup)  : ramps 0 → λ_vrex
        [2×warmup, ∞)        : λ_vrex          (full penalty)

        Bug fix vs. original: the ramp now starts at 1/warmup on the first
        post-warmup epoch rather than 0, so lambda is non-zero immediately
        after warmup ends.

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
        Typical usage — let ``forward`` handle it via ``current_epoch``::

            loss = loss_fn(environments=envs, current_epoch=epoch)

        Or manually override for custom schedules::

            loss_fn.lambda_vrex = loss_fn.get_lambda(epoch)
        """
        if current_epoch < self.vrex_warmup_epochs:
            return 0.0
        # FIX: +1 ensures ramp starts at 1/warmup, not 0
        ramp = (current_epoch - self.vrex_warmup_epochs + 1) / max(self.vrex_warmup_epochs, 1)
        return float(min(self.lambda_vrex, self.lambda_vrex * ramp))