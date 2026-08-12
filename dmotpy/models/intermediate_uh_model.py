from typing import Any, Dict, Optional, Tuple

import torch

from .core.flexb import flexb_step_post, flexb_step_pre
from .core.flexi import flexi_step_post, flexi_step_pre
from .core.flexis import flexis_step_post, flexis_step_pre
from .core.gr4j import gr4j_step_post, gr4j_step_pre
from .hydrology_model import HydrologyModel, _maybe_compile


INTERMEDIATE_UH_CONFIG = {
    "flexi": {
        "step_pre": flexi_step_pre,
        "step_post": flexi_step_post,
        "pre_params": ["smax", "beta", "d_split", "percmax", "lp", "imax"],
        "post_params": ["kf", "ks"],
        "n_pre_states": 2,
        "n_post_states": 2,
        "n_pre_passthru": 2,
    },
    "flexb": {
        "step_pre": flexb_step_pre,
        "step_post": flexb_step_post,
        "pre_params": ["s1max", "beta", "d_split", "percmax", "lp"],
        "post_params": ["kf", "ks"],
        "n_pre_states": 1,
        "n_post_states": 2,
        "n_pre_passthru": 1,
    },
    "flexis": {
        "step_pre": flexis_step_pre,
        "step_post": flexis_step_post,
        "pre_params": ["smax", "beta", "d_split", "percmax", "lp", "imax", "tt", "ddf"],
        "post_params": ["kf", "ks"],
        "n_pre_states": 3,
        "n_post_states": 2,
        "n_pre_passthru": 2,
    },
    "gr4j": {
        "step_pre": gr4j_step_pre,
        "step_post": gr4j_step_post,
        "pre_params": ["x1"],
        "post_params": ["x2", "x3"],
        "n_pre_states": 1,
        "n_post_states": 1,
        "n_pre_passthru": 1,
    },
}


class IntermediateUHModel(HydrologyModel):
    """HydrologyModel with intermediate unit-hydrograph routing."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "compile",
    ) -> None:
        super().__init__(config, device, backend)
        self._setup_uh()

    def _setup_uh(self) -> None:
        cfg = INTERMEDIATE_UH_CONFIG[self.model_name]

        self.step_pre_fn = _maybe_compile(cfg["step_pre"], self.backend)
        self.step_post_fn = _maybe_compile(cfg["step_post"], self.backend)

        from .unithydro.uh_tri_3 import DplTri3

        self.uh_fast = DplTri3(max_lag=int(self.parameter_bounds["nlagf"][1]))
        self.uh_slow = DplTri3(max_lag=int(self.parameter_bounds["nlags"][1]))
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

        uh_inputs = [[], []]
        passthru_lists = [[] for _ in range(self._n_pre_passthru)]
        for t in range(n_steps):
            result = self.step_pre_fn(
                p_seq[t], t_seq[t], pet_seq[t], *pre_params, *pre_states, self.nearzero
            )
            uh_inputs[0].append(result[0])
            uh_inputs[1].append(result[1])
            for i in range(self._n_pre_passthru):
                passthru_lists[i].append(result[2 + i])
            pre_states = list(result[2 + self._n_pre_passthru :])

        rf_stack = torch.stack(uh_inputs[0], dim=0)
        rsl_stack = torch.stack(uh_inputs[1], dim=0)

        b_total = n_grid * n_groups
        rf_flat = rf_stack.permute(1, 2, 0).reshape(b_total, n_steps)
        rsl_flat = rsl_stack.permute(1, 2, 0).reshape(b_total, n_steps)

        nlagf_flat = params_dict["nlagf"].expand(n_grid, n_groups).reshape(b_total, 1)
        nlags_flat = params_dict["nlags"].expand(n_grid, n_groups).reshape(b_total, 1)

        routed_rf = self.uh_fast(rf_flat, nlagf_flat)
        routed_rsl = self.uh_slow(rsl_flat, nlags_flat)

        rf_seq = routed_rf.view(n_grid, n_groups, n_steps).permute(2, 0, 1).unbind(0)
        rsl_seq = routed_rsl.view(n_grid, n_groups, n_steps).permute(2, 0, 1).unbind(0)

        qsim_list = []
        for t in range(n_steps):
            passthru_vals = [passthru_lists[i][t] for i in range(self._n_pre_passthru)]
            qsim, _ea, *post_new = self.step_post_fn(
                rf_seq[t], rsl_seq[t], *passthru_vals, *post_states, *post_params, self.nearzero
            )
            post_states = list(post_new)
            qsim_list.append(qsim)

        qsim_out = torch.stack(qsim_list, dim=0)
        streamflow = qsim_out[effective_warmup:]
        return self._finalize_output(streamflow)
