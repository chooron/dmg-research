from __future__ import annotations

from typing import Any

import torch

from project.flexmopex.models import mopex_core
from project.flexmopex.models.base_mopex import BaseMopex


class StaticMopex(BaseMopex):
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(config, device)
        self.name = "StaticMopex"
        self.step_fn = self._compile_step(mopex_core.mopex_step_static)

    def _run_loop(
        self,
        P: torch.Tensor,
        T: torch.Tensor,
        PET: torch.Tensor,
        doy: torch.Tensor,
        params: dict[str, torch.Tensor],
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
        S1, S2, Sc1, Sc2, Sn = self._initial_states(n_grid)
        effective_warmup = min(self.warm_up, n_steps)

        with torch.no_grad():
            for t in range(effective_warmup):
                _, _, S1, S2, Sc1, Sc2, Sn = self.step_fn(
                    P[t],
                    T[t],
                    PET[t],
                    doy[t],
                    Sb1,
                    tw,
                    tu,
                    Se,
                    tc,
                    ddf,
                    tcrit,
                    Sb2,
                    alpha,
                    is_time,
                    tmin,
                    tmax,
                    S1,
                    S2,
                    Sc1,
                    Sc2,
                    Sn,
                    self.nearzero,
                )

        S1 = S1.detach()
        S2 = S2.detach()
        Sc1 = Sc1.detach()
        Sc2 = Sc2.detach()
        Sn = Sn.detach()
        Q_list = []

        for t in range(effective_warmup, n_steps):
            Q, _, S1, S2, Sc1, Sc2, Sn = self.step_fn(
                P[t],
                T[t],
                PET[t],
                doy[t],
                Sb1,
                tw,
                tu,
                Se,
                tc,
                ddf,
                tcrit,
                Sb2,
                alpha,
                is_time,
                tmin,
                tmax,
                S1,
                S2,
                Sc1,
                Sc2,
                Sn,
                self.nearzero,
            )
            Q_list.append(Q)

        return torch.stack(Q_list, dim=0)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        mopex_params = self._descale_mopex_params(parameters["params"])
        routing_params = self._descale_routing_params(parameters["gamma_uh"])
        P, T, PET, doy, n_steps, n_grid = self._prepare_forcings(x_dict)
        Q_mopex = self._run_loop(P, T, PET, doy, mopex_params, n_steps, n_grid)
        Qrouted = self._apply_routing(Q_mopex.mean(-1), routing_params)
        return {"streamflow": Qrouted}
