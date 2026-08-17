"""Experimental model classes for the interception 2x2 study.

- ``LearnedWeightMopexV1``          : V1 PET semantics, original parameterization
- ``LearnedWeightMopexDecoupled``   : V0 PET semantics, amplitude-decoupled interception
- ``LearnedWeightMopexV1Decoupled`` : V1 PET semantics, amplitude-decoupled interception

Production ``LearnedWeightMopex`` (``models/learned_weight_mopex.py``) is not
modified.  These classes keep the identical network interface: same parameter
ordering (12 MOPEX params + 2 routing + 4x2 structural logits), same output
dimension, same learnable-parameter count, same gate initialization
convention, same descale path (sigmoid -> hydrodl2 ``change_param_range``).
"""
from __future__ import annotations

from typing import Any

import torch

from project.flexmopex.models import mopex_core_v1
from project.flexmopex.models.learned_weight_mopex import LearnedWeightMopex


class LearnedWeightMopexV1(LearnedWeightMopex):
    """V1: same inputs/outputs and learnable parameters as V0; interception loss
    is independent of the soil PET budget (production interception computation
    and cap preserved)."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(config, device)
        self.name = "LearnedWeightMopexV1"
        self.step_fn = self._compile_step(mopex_core_v1.mopex_step_v1)


class _DecoupledLoopMixin:
    """Shared decoupled forward loop: precompute the annual normalization once
    per forward, then feed the per-timestep normalized seasonal shape into the
    decoupled step.  Mirrors ``BaseMopex._run_weighted_loop`` exactly except
    for the interception seasonal-shape source."""

    def _run_weighted_loop(
        self,
        P: torch.Tensor,
        T: torch.Tensor,
        PET: torch.Tensor,
        doy: torch.Tensor,
        params: dict[str, torch.Tensor],
        weights_on: torch.Tensor,
        n_steps: int,
        n_grid: int,
    ) -> torch.Tensor:
        Sb1 = params["Sb1"]
        tw = params["tw"]
        tu = params["tu"]
        Se = params["Se"]
        tc = params["tc"]
        ddf = params["ddf"]
        tcrit = params["tcrit"]
        Sb2 = params["Sb2"]
        alpha = params["alpha"]
        is_time = params["is_time"]
        tmin = params["tmin"]
        tmax = params["tmax"]
        w_phen = weights_on[:, 0].unsqueeze(-1).expand(-1, self.nmul)
        w_int = weights_on[:, 1].unsqueeze(-1).expand(-1, self.nmul)
        w_snow = weights_on[:, 2].unsqueeze(-1).expand(-1, self.nmul)
        w_sub = weights_on[:, 3].unsqueeze(-1).expand(-1, self.nmul)
        S1, S2, Sc1, Sc2, Sn = self._initial_states(n_grid)
        effective_warmup = min(self.warm_up, n_steps)

        # Annual normalization of g_raw over the fixed phase grid (once per
        # forward; differentiable w.r.t. alpha; no detach).
        norm_mean = mopex_core_v1.decoupled_norm_mean(alpha, is_time)

        with torch.no_grad():
            for t in range(effective_warmup):
                season_shape = mopex_core_v1.decoupled_shape(
                    doy[t], alpha, is_time, norm_mean
                )
                _, _, S1, S2, Sc1, Sc2, Sn = self.step_fn(
                    P[t], T[t], PET[t], doy[t],
                    w_phen, w_int, w_snow, w_sub,
                    Sb1, tw, tu, Se, tc, ddf, tcrit, Sb2, alpha, is_time, tmin, tmax,
                    S1, S2, Sc1, Sc2, Sn, season_shape, self.nearzero,
                )

        S1 = S1.detach()
        S2 = S2.detach()
        Sc1 = Sc1.detach()
        Sc2 = Sc2.detach()
        Sn = Sn.detach()
        Q_list = []

        for t in range(effective_warmup, n_steps):
            season_shape = mopex_core_v1.decoupled_shape(
                doy[t], alpha, is_time, norm_mean
            )
            Q, _, S1, S2, Sc1, Sc2, Sn = self.step_fn(
                P[t], T[t], PET[t], doy[t],
                w_phen, w_int, w_snow, w_sub,
                Sb1, tw, tu, Se, tc, ddf, tcrit, Sb2, alpha, is_time, tmin, tmax,
                S1, S2, Sc1, Sc2, Sn, season_shape, self.nearzero,
            )
            Q_list.append(Q)

        return torch.stack(Q_list, dim=0)


class LearnedWeightMopexDecoupled(_DecoupledLoopMixin, LearnedWeightMopex):
    """V0 PET semantics + amplitude-decoupled interception (arm B)."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(config, device)
        self.name = "LearnedWeightMopexDecoupled"
        self.step_fn = self._compile_step(mopex_core_v1.mopex_step_decoupled)


class LearnedWeightMopexV1Decoupled(_DecoupledLoopMixin, LearnedWeightMopexV1):
    """V1 PET semantics + amplitude-decoupled interception (arm D)."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(config, device)
        self.name = "LearnedWeightMopexV1Decoupled"
        self.step_fn = self._compile_step(mopex_core_v1.mopex_step_v1_decoupled)
