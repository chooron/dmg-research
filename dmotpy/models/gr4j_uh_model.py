from typing import Dict, Tuple

import torch

from .hydrology_model import _maybe_compile
from .intermediate_uh_model import INTERMEDIATE_UH_CONFIG, IntermediateUHModel


class GR4JUHModel(IntermediateUHModel):
    """GR4J-specific intermediate UH model with analytical dual-branch routing."""

    def _setup_uh(self) -> None:
        cfg = INTERMEDIATE_UH_CONFIG[self.model_name]

        self.step_pre_fn = _maybe_compile(cfg["step_pre"], self.backend)
        self.step_post_fn = _maybe_compile(cfg["step_post"], self.backend)

        from .unithydro.uh_full_2 import DplFull2
        from .unithydro.uh_half_1 import DplHalf1

        max_lag_val = int(self.parameter_bounds["x4"][1])
        self.uh_half = DplHalf1(max_lag=max_lag_val + 1)
        self.uh_full = DplFull2(max_lag=max_lag_val * 2 + 2)
        self._pre_param_names = cfg["pre_params"]
        self._post_param_names = cfg["post_params"]
        self._n_pre_states = cfg["n_pre_states"]
        self._n_post_states = cfg["n_post_states"]
        self._n_pre_passthru = cfg["n_pre_passthru"]

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

        forcing_seqs = self._make_forcing_sequences(forcing, n_groups)
        p_seq, t_seq, pet_seq = forcing_seqs[:3]

        pre_params = [params_dict[name] for name in self._pre_param_names]
        post_params = [params_dict[name] for name in self._post_param_names]

        pre_states = list(states[: self._n_pre_states])
        post_states = list(states[self._n_pre_states :])

        pr_list = []
        ephys_list = []
        for t in range(n_steps):
            flux_pr, e_physical, *pre_new = self.step_pre_fn(
                p_seq[t], t_seq[t], pet_seq[t], *pre_params, *pre_states, self.nearzero
            )
            pre_states = list(pre_new)
            pr_list.append(flux_pr)
            ephys_list.append(e_physical)

        pr_stack = torch.stack(pr_list, dim=0)
        b_total = n_grid * n_groups
        pr_flat = pr_stack.permute(1, 2, 0).reshape(b_total, n_steps)

        flux_q9 = pr_flat * 0.9
        flux_q1 = pr_flat * 0.1

        x4_flat = params_dict["x4"].expand(n_grid, n_groups).reshape(b_total, 1)
        routed_q9 = self.uh_half(flux_q9, x4_flat)
        routed_q1 = self.uh_full(flux_q1, x4_flat * 2.0)

        q9_seq = routed_q9.view(n_grid, n_groups, n_steps).permute(2, 0, 1).unbind(0)
        q1_seq = routed_q1.view(n_grid, n_groups, n_steps).permute(2, 0, 1).unbind(0)

        qsim_list = []
        for t in range(n_steps):
            qsim, _ea, *post_new = self.step_post_fn(
                q9_seq[t], q1_seq[t], *post_states, *post_params, ephys_list[t], self.nearzero
            )
            post_states = list(post_new)
            qsim_list.append(qsim)

        qsim_out = torch.stack(qsim_list, dim=0)
        streamflow = qsim_out[effective_warmup:]
        return self._finalize_output(streamflow)
