from typing import Dict, Optional, Tuple

import torch

from .hydrology_model import HydrologyModel


class TCMModel(HydrologyModel):
    """Specialized model for TCM, which requires climatological mean precipitation.

    ``mean_P`` is the basin-level static precipitation mean (CAMELS/Caravan
    ``p_mean`` attribute, physical mm/d, non-trainable).  The caller supplies
    it through ``x_dict["p_mean"]`` with shape ``(n_grid,)`` (or
    ``(n_grid, n_groups)``).  For backward compatibility with legacy direct
    calls that omit it, a fallback recomputes the window mean from forcing,
    but production paths must always pass the static attribute.
    """

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        parameters: Tuple[Optional[torch.Tensor], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        raw = self.unpack_parameters(parameters)
        n_groups = self.get_parameter_group_count(raw)
        n_grid = x_dict["x_phy"].size(1)
        states = self._init_states(n_grid, n_groups)
        params_dict = self._descale_params(raw)

        forcing = x_dict["x_phy"]
        if "p_mean" in x_dict:
            # Canonical static basin p_mean (physical mm/d).  Constant,
            # requires_grad=False, identical across warmup/train/val/test.
            p_mean = x_dict["p_mean"]
            if p_mean.dim() == 1:
                p_mean = p_mean.unsqueeze(-1)
            mean_p = p_mean.expand(n_grid, n_groups)
        else:
            # Legacy fallback: window precipitation mean.  Kept only so that
            # direct low-level calls keep working; production must pass p_mean.
            mean_p = forcing[..., 0].mean(dim=0, keepdim=False).unsqueeze(-1).expand(n_grid, n_groups)
        return self._run_model_with_mean_p(forcing, states, params_dict, n_groups, mean_p)

    def _run_model_with_mean_p(
        self,
        forcing: torch.Tensor,
        states: Tuple[torch.Tensor, ...],
        params_dict: Dict[str, torch.Tensor],
        n_groups: int,
        mean_p: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        n_steps, n_grid = forcing.shape[:2]
        effective_warmup = min(self.warm_up, n_steps)

        p_seq, t_seq, pet_seq = self._make_forcing_sequences(forcing, n_groups)
        param_values = [params_dict[name] for name in self.phy_param_names]

        curr_states = states
        with torch.no_grad():
            for t in range(effective_warmup):
                outputs = self.step_fn(
                    p_seq[t],
                    t_seq[t],
                    pet_seq[t],
                    *param_values,
                    *curr_states,
                    self.nearzero,
                    mean_P=mean_p,
                )
                curr_states = tuple(outputs[2:])
        curr_states = tuple(state.detach() for state in curr_states)

        n_train = n_steps - effective_warmup
        streamflow = torch.empty((n_train, n_grid, n_groups), device=forcing.device, dtype=forcing.dtype)

        for offset, t in enumerate(range(effective_warmup, n_steps)):
            outputs = self.step_fn(
                p_seq[t],
                t_seq[t],
                pet_seq[t],
                *param_values,
                *curr_states,
                self.nearzero,
                mean_P=mean_p,
            )
            streamflow[offset] = outputs[0]
            curr_states = outputs[2:]

        return self._finalize_output(streamflow)
