from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .hydrology_model import HydrologyModel


ENDPOINT_UH_SCHEMES = {
    "newzealand2": {"kind": "total", "uhs": [("tri4", "d_delay")]},
    "hillslope": {"kind": "surface_baseflow", "uhs": [("tri3", "th")]},
    "plateau": {"kind": "surface_baseflow", "uhs": [("tri3", "tp")]},
    "smar": {"kind": "surface_baseflow", "uhs": [("gamma6", "nk_delay")]},
    "ihacres": {"kind": "exp_delay_chain", "uhs": [("exp5", "tau_q"), ("exp5", "tau_s")]},
    "hbv96": {"kind": "total", "uhs": [("uniform7", "maxbas")]},
}


class EndpointUHModel(HydrologyModel):
    """HydrologyModel with endpoint unit-hydrograph routing."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "compile",
    ) -> None:
        super().__init__(config, device, backend)
        self._setup_uh()

    def _setup_uh(self) -> None:
        from .unithydro import UH_MAP

        if self.model_name not in ENDPOINT_UH_SCHEMES:
            raise NotImplementedError(f"Endpoint UH routing not implemented for {self.model_name}")

        self._endpoint_scheme = ENDPOINT_UH_SCHEMES[self.model_name]
        self.uh_modules = nn.ModuleList()
        for uh_kind, param_name in self._endpoint_scheme["uhs"]:
            uh_cls = UH_MAP[uh_kind]
            max_lag = int(self.parameter_bounds[param_name][1])
            self.uh_modules.append(uh_cls(max_lag=max_lag))

    def _run_model(
        self,
        x_dict: Dict[str, torch.Tensor],
        states: Tuple[torch.Tensor, ...],
        params_dict: Dict[str, torch.Tensor],
        n_groups: int,
    ) -> Dict[str, torch.Tensor]:
        forcing = x_dict["x_phy"]
        n_steps, n_grid = forcing.shape[:2]
        effective_warmup = min(self.warm_up, n_steps)

        p_seq, t_seq, pet_seq = self._make_forcing_sequences(forcing, n_groups)

        param_values = [params_dict[name] for name in self.phy_param_names]
        curr_states = list(states)

        kind = self._endpoint_scheme["kind"]
        need_split = kind in ("surface_baseflow", "exp_delay_chain")

        qsim_list = []
        surface_list = [] if need_split else None
        baseflow_list = [] if need_split else None

        for t in range(n_steps):
            kwargs: dict = {}
            if need_split:
                kwargs["return_routing_fluxes"] = True

            outputs = self.step_fn(
                p_seq[t], t_seq[t], pet_seq[t], *param_values, *curr_states, self.nearzero, **kwargs
            )

            if need_split:
                qsim_list.append(outputs[0])
                fluxes = outputs[-1]
                surface_list.append(fluxes[0])
                baseflow_list.append(fluxes[1])
                curr_states = list(outputs[2:-1])
            else:
                qsim_list.append(outputs[0])
                curr_states = list(outputs[2:])

        if kind == "total":
            stack = torch.stack(qsim_list, dim=0)
            b_total = n_grid * n_groups
            flat = stack.permute(1, 2, 0).reshape(b_total, n_steps)
            uh_param_name = self._endpoint_scheme["uhs"][0][1]
            uh_param_val = params_dict[uh_param_name].expand(n_grid, n_groups).reshape(b_total, 1)
            routed = self.uh_modules[0](flat, uh_param_val)
            streamflow = routed.view(n_grid, n_groups, n_steps).permute(2, 0, 1)[effective_warmup:]
            return self._finalize_output(streamflow)

        if kind == "surface_baseflow":
            surf_stack = torch.stack(surface_list, dim=0)
            base_stack = torch.stack(baseflow_list, dim=0)
            b_total = n_grid * n_groups
            surf_flat = surf_stack.permute(1, 2, 0).reshape(b_total, n_steps)

            if self._endpoint_scheme["uhs"][0][0] == "gamma6":
                n_res = params_dict["n_res"].expand(n_grid, n_groups).reshape(b_total, 1)
                nk_delay = params_dict["nk_delay"].expand(n_grid, n_groups).reshape(b_total, 1)
                k_val = nk_delay / (n_res + self.nearzero)
                uh_params = torch.cat([n_res, k_val], dim=1)
            else:
                uh_param_name = self._endpoint_scheme["uhs"][0][1]
                uh_params = params_dict[uh_param_name].expand(n_grid, n_groups).reshape(b_total, 1)

            routed = self.uh_modules[0](surf_flat, uh_params)
            routed_surf = routed.view(n_grid, n_groups, n_steps).permute(2, 0, 1)
            streamflow = (routed_surf + base_stack)[effective_warmup:]
            return self._finalize_output(streamflow)

        if kind == "exp_delay_chain":
            uq_stack = torch.stack(surface_list, dim=0)
            us_stack = torch.stack(baseflow_list, dim=0)
            b_total = n_grid * n_groups

            uq_flat = uq_stack.permute(1, 2, 0).reshape(b_total, n_steps)
            us_flat = us_stack.permute(1, 2, 0).reshape(b_total, n_steps)

            tau_q_flat = params_dict["tau_q"].expand(n_grid, n_groups).reshape(b_total, 1)
            tau_s_flat = params_dict["tau_s"].expand(n_grid, n_groups).reshape(b_total, 1)

            routed_uq = self.uh_modules[0](uq_flat, tau_q_flat)
            routed_us = self.uh_modules[1](us_flat, tau_s_flat)
            summed = routed_uq + routed_us
            streamflow = summed.view(n_grid, n_groups, n_steps).permute(2, 0, 1)[effective_warmup:]
            return self._finalize_output(streamflow)

        raise ValueError(f"Unknown endpoint routing kind: {kind}")
