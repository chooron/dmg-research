from typing import Dict, Optional, Tuple

import math

import torch

from .hydrology_model import HydrologyModel
from .flux.mopex import mopex_training_context


class MopexDoyModel(HydrologyModel):
    """Specialized model for MOPEX variants requiring day-of-year forcing."""

    _SCALAR_PHASE = "scalar"
    _ATAN2_PHASE = "atan2"
    _CIRCULAR_PHASE = "circular"

    def __init__(self, config=None, device=None, backend: str = "compile"):
        super().__init__(config, device=device, backend=backend)
        self.phase_parameterization = (
            config.get("phase_parameterization", self._SCALAR_PHASE)
            if config is not None
            else self._SCALAR_PHASE
        )
        if self.model_name in {"mopex4"} and self._phase_parameterization() != self._SCALAR_PHASE:
            raise ValueError(
                "MOPEX4 / MOPEX4.1 interception has no diagnostic seasonal phase; "
                "call the legacy helpers explicitly for F0 reproduction."
            )
        self.continuation_lambda_i = float(config.get("continuation_lambda_i", 1.0)) if config else 1.0
        self.continuation_lambda_p = float(config.get("continuation_lambda_p", 1.0)) if config else 1.0
        self.continuation_beta = float(config.get("continuation_beta", 50.0)) if config else 50.0

    def unpack_parameters(
        self,
        parameters: Tuple[Optional[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Keep the extra diagnostic phase coordinate when it is requested."""
        _, raw = parameters
        static_count = len(self.phy_param_names)
        expected = static_count + (self._phase_parameterization() != self._SCALAR_PHASE)
        if raw.dim() == 3:
            return raw[:, :expected, :]
        if raw.dim() != 2:
            raise ValueError(f"Unexpected parameter shape: {tuple(raw.shape)}")
        if raw.shape[1] == expected:
            return raw
        actual_nmul = max(raw.shape[1] // expected, 1)
        if actual_nmul == 1:
            return raw[:, :expected]
        return raw[:, : expected * actual_nmul].view(raw.shape[0], expected, actual_nmul)

    def _phase_parameterization(self) -> str:
        value = str(getattr(self, "phase_parameterization", self._SCALAR_PHASE)).lower()
        if value not in {self._SCALAR_PHASE, self._ATAN2_PHASE, self._CIRCULAR_PHASE}:
            raise ValueError(f"unsupported MOPEX phase_parameterization: {value}")
        return value

    def _split_phase_parameters(
        self, raw: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Convert diagnostic-only two-component phase inputs to scalar slots.

        The normal model has one ``is_time`` slot.  The diagnostic circular
        variants replace that slot with two normalized network outputs.  The
        rest of the physical parameter ordering is unchanged.
        """
        representation = self._phase_parameterization()
        static_count = len(self.phy_param_names)
        if representation == self._SCALAR_PHASE:
            return raw[..., :static_count, :] if raw.dim() == 3 else raw[..., :static_count], None

        expected = static_count + 1
        actual = raw.shape[1]
        if actual != expected:
            raise ValueError(
                f"{representation} phase diagnostics require {expected} raw outputs, got {actual}"
            )
        phase_raw = raw[:, 5:7, :] if raw.dim() == 3 else raw[:, 5:7]
        # Network outputs are normalized coordinates.  Map those two entries
        # to signed components before normalising to the unit circle.
        phase_components = 2.0 * phase_raw - 1.0
        phase_cos, phase_sin = phase_components.unbind(dim=1)
        scalar_placeholder = raw[:, 5:6, :] if raw.dim() == 3 else raw[:, 5:6]
        static_raw = torch.cat((raw[:, :5, :] if raw.dim() == 3 else raw[:, :5], scalar_placeholder,
                                raw[:, 7:, :] if raw.dim() == 3 else raw[:, 7:]), dim=1)

        if representation == self._ATAN2_PHASE:
            radius = torch.sqrt(phase_cos.square() + phase_sin.square() + self.nearzero)
            phi = torch.atan2(phase_sin / radius, phase_cos / radius)
            day = torch.remainder(phi, 2.0 * math.pi) * (365.25 / (2.0 * math.pi))
            # atan2 is used only for candidate B.  Convert to the established
            # scalar physical slot before using the standard hydrologic path.
            day = day.unsqueeze(1) if raw.dim() == 3 else day.unsqueeze(1)
            static_raw = static_raw.clone()
            bounds = self.parameter_bounds["is_time"]
            static_raw[:, 5:6, :] = (day - bounds[0]) / (bounds[1] - bounds[0]) if raw.dim() == 3 else (day - bounds[0]) / (bounds[1] - bounds[0])
            return static_raw, None
        return static_raw, (phase_cos, phase_sin)

    def _make_doy_sequences(self, forcing: torch.Tensor, n_groups: int) -> tuple:
        if n_groups > 1:
            return forcing[..., 3:4].expand(-1, -1, n_groups).unbind(0)
        return forcing[..., 3:4].unbind(0)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        parameters: Tuple[Optional[torch.Tensor], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        raw = self.unpack_parameters(parameters)
        raw, circular_phase = self._split_phase_parameters(raw)
        n_groups = self.get_parameter_group_count(raw)
        n_grid = x_dict["x_phy"].size(1)
        states = self._init_states(n_grid, n_groups)
        params_dict = self._descale_params(raw)

        forcing = x_dict["x_phy"]
        if forcing.shape[-1] < 4:
            doy = x_dict["doy"]
            if doy.dim() == 2:
                doy = doy.unsqueeze(-1)
            elif doy.dim() != 3:
                doy = doy.view(forcing.shape[0], n_grid, 1)
            forcing = torch.cat([forcing, doy.to(forcing.device)], dim=-1)

        return self._run_model({"x_phy": forcing}, states, params_dict, n_groups, circular_phase)

    def _run_model(
        self,
        x_dict: Dict[str, torch.Tensor],
        states: Tuple[torch.Tensor, ...],
        params_dict: Dict[str, torch.Tensor],
        n_groups: int,
        circular_phase: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        forcing = x_dict["x_phy"]
        n_steps, n_grid = forcing.shape[:2]
        effective_warmup = min(self.warm_up, n_steps)

        p_seq, t_seq, pet_seq = self._make_forcing_sequences(forcing, n_groups)
        doy_seq = self._make_doy_sequences(forcing, n_groups)
        param_values = [params_dict[name] for name in self.phy_param_names]

        # MOPEX4/5 initializers return (Sn, S1, S2, Sc1, Sc2) while their step
        # functions take (S1, S2, Sc1, Sc2, Sn) positionally.  Reorder once at
        # initialization so each store reaches the correct step slot; the
        # per-step outputs are already in step order.
        if self.model_name in {"mopex4", "mopex5"} and len(states) == 5:
            states = (states[1], states[2], states[3], states[4], states[0])
        lambda_i = float(getattr(self, "continuation_lambda_i", 1.0))
        lambda_p = float(getattr(self, "continuation_lambda_p", 1.0))
        beta = float(getattr(self, "continuation_beta", 50.0))
        with mopex_training_context(lambda_i=lambda_i, lambda_p=lambda_p, beta=beta):
            curr_states = states
            with torch.no_grad():
                for t in range(effective_warmup):
                    phase_kwargs = {} if circular_phase is None else {
                        "phase_cos": circular_phase[0], "phase_sin": circular_phase[1],
                    }
                    outputs = self.step_fn(
                        p_seq[t], t_seq[t], pet_seq[t], *param_values, *curr_states,
                        delta_t=1.0, nearzero=self.nearzero, doy=doy_seq[t], **phase_kwargs,
                    )
                    curr_states = tuple(outputs[2:])
            curr_states = tuple(state.detach() for state in curr_states)

            n_train = n_steps - effective_warmup
            streamflow = torch.empty((n_train, n_grid, n_groups), device=forcing.device, dtype=forcing.dtype)
            for offset, t in enumerate(range(effective_warmup, n_steps)):
                phase_kwargs = {} if circular_phase is None else {
                    "phase_cos": circular_phase[0], "phase_sin": circular_phase[1],
                }
                outputs = self.step_fn(
                    p_seq[t], t_seq[t], pet_seq[t], *param_values, *curr_states,
                    delta_t=1.0, nearzero=self.nearzero, doy=doy_seq[t], **phase_kwargs,
                )
                streamflow[offset] = outputs[0]
                curr_states = outputs[2:]

        return self._finalize_output(streamflow)
