import importlib.util
from functools import wraps
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .registry import INIT_INFO, PARAM_INFO, STATE_INFO, STFN_INFO


def _maybe_compile(fn, backend: str):
    if backend == "compile" and hasattr(torch, "compile"):
        if importlib.util.find_spec("setuptools") is None:
            raise RuntimeError("torch.compile requested but setuptools is unavailable")

        compiled_fn = torch.compile(fn)

        @wraps(fn)
        def compiled_only(*args, **kwargs):
            return compiled_fn(*args, **kwargs)

        setattr(compiled_only, "_compile_enabled", True)
        setattr(compiled_only, "_compile_backend", "torch_default")
        return compiled_only
    if backend == "jit":
        return torch.jit.script(fn)
    return fn


class HydrologyModel(nn.Module):
    """Unified hydrology model base for calibration and prediction modes.

    Main differences between the legacy v1/v2 paths are now config-driven:
    - parameter inputs can be ``(batch, n_params)`` or ``(batch, n_params, n_groups)``
    - routing can treat groups independently or average them before routing
    - trainers can inspect the model to decide whether targets should be
      expanded to every parameter group or kept at basin scale
    """

    GROUP_INDIVIDUAL = "individual"
    GROUP_MEAN_BEFORE_ROUTING = "mean_before_routing"

    LOSS_PER_GROUP = "per_group"
    LOSS_MEAN_OVER_GROUPS = "mean_over_groups"

    ANALYSIS_MEAN = "mean"
    ANALYSIS_KEEP = "keep"

    def __new__(
        cls,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "compile",
    ) -> "HydrologyModel":
        if cls is not HydrologyModel:
            return super().__new__(cls)
        if config and config.get("model_name", "").lower() in {"mopex4", "mopex5", "vic"}:
            from .mopex_doy_model import MopexDoyModel

            return super().__new__(MopexDoyModel)
        if config and config.get("model_name", "").lower() == "tcm":
            from .tcm_model import TCMModel

            return super().__new__(TCMModel)
        if config and config.get("uh_enabled"):
            from .endpoint_uh_model import EndpointUHModel
            from .gr4j_uh_model import GR4JUHModel
            from .intermediate_uh_model import IntermediateUHModel

            uh_mode = config.get("uh_mode")
            model_name = config.get("model_name", "")
            if uh_mode == "endpoint":
                return super().__new__(EndpointUHModel)
            if uh_mode == "intermediate":
                if model_name == "gr4j":
                    return super().__new__(GR4JUHModel)
                return super().__new__(IntermediateUHModel)
        return super().__new__(cls)

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        backend: str = "compile",
    ) -> None:
        super().__init__()

        self.config = config or {}
        self.model_name = self.config.get("model_name", "hbv96").lower()
        self.name = f"HydrologyModel_{self.model_name}"

        if self.model_name not in PARAM_INFO:
            raise ValueError(f"Unknown model_name: {self.model_name}. Available: {list(PARAM_INFO.keys())}")

        self.parameter_bounds = PARAM_INFO[self.model_name]
        self.raw_step_fn = STFN_INFO[self.model_name]
        self.init_fn = INIT_INFO[self.model_name]
        self.n_states = STATE_INFO[self.model_name]

        self.nmul = 1
        self.warm_up = 0
        self.warm_up_states = True
        self.variables = ["prcp", "tmean", "pet"]
        self.nearzero = 1e-5
        self.check_water_balance = False
        self.parameter_mapping = "linear"
        self.log_mapping_span_threshold = 100.0

        self.group_routing_strategy = self.GROUP_INDIVIDUAL
        self.group_loss_strategy = self.LOSS_PER_GROUP
        self.group_analysis_strategy = self.ANALYSIS_MEAN

        self.backend = self.config.get("backend", backend)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step_fn = _maybe_compile(self.raw_step_fn, self.backend)

        self._load_config(self.config)
        self.phy_param_names = list(self.parameter_bounds.keys())
        self.learnable_param_count = len(self.phy_param_names)

    def _load_config(self, config: Dict[str, Any]) -> None:
        attrs = [
            "warm_up",
            "warm_up_states",
            "variables",
            "nearzero",
            "check_water_balance",
            "parameter_mapping",
            "log_mapping_span_threshold",
            "nmul",
            "group_routing_strategy",
            "group_loss_strategy",
            "group_analysis_strategy",
        ]
        for attr in attrs:
            if attr in config:
                setattr(self, attr, config[attr])

        if "routing_strategy" in config:
            self.group_routing_strategy = config["routing_strategy"]
        if "loss_strategy" in config:
            self.group_loss_strategy = config["loss_strategy"]
        if "mc_dropout_group_strategy" in config:
            self.group_analysis_strategy = config["mc_dropout_group_strategy"]

        self.nearzero = float(self.nearzero)
        self.parameter_mapping = str(self.parameter_mapping).lower()
        self.log_mapping_span_threshold = float(self.log_mapping_span_threshold)

    # ------------------------------------------------------------------
    # Parameter / state helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _should_use_log_mapping(
        bounds: list[float],
        mapping: str,
        span_threshold: float,
    ) -> bool:
        lower = float(bounds[0])
        upper = float(bounds[1])
        if mapping in {"linear", "none"}:
            return False
        if mapping not in {"auto", "auto_log", "log_auto"}:
            raise ValueError(
                f"Unsupported parameter_mapping '{mapping}'. "
                "Use 'linear' or 'auto_log'."
            )
        return lower > 0.0 and upper > lower and (upper / lower) >= span_threshold

    def _init_states(
        self,
        n_grid: int,
        n_groups: Optional[int] = None,
    ) -> Tuple[torch.Tensor, ...]:
        return self.init_fn(
            n_grid,
            n_groups if n_groups is not None else self.nmul,
            self.device,
            self.nearzero,
        )

    @staticmethod
    def _change_param_range(
        param: torch.Tensor,
        bounds: list[float],
        mapping: str = "linear",
        span_threshold: float = 100.0,
    ) -> torch.Tensor:
        if HydrologyModel._should_use_log_mapping(bounds, mapping, span_threshold):
            lower = torch.as_tensor(bounds[0], dtype=param.dtype, device=param.device)
            upper = torch.as_tensor(bounds[1], dtype=param.dtype, device=param.device)
            log_lower = torch.log(lower)
            log_upper = torch.log(upper)
            return torch.exp(log_lower + param * (log_upper - log_lower))
        return param * (bounds[1] - bounds[0]) + bounds[0]

    def _descale_params(self, raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        if raw.dim() == 2:
            return {
                name: self._change_param_range(
                    raw[:, index : index + 1],
                    self.parameter_bounds[name],
                    self.parameter_mapping,
                    self.log_mapping_span_threshold,
                )
                for index, name in enumerate(self.phy_param_names)
            }

        bounds = self.parameter_bounds
        return {
            name: self._change_param_range(
                raw[:, index, :],
                bounds[name],
                self.parameter_mapping,
                self.log_mapping_span_threshold,
            )
            for index, name in enumerate(self.phy_param_names)
        }

    def get_parameter_group_count(self, raw: torch.Tensor) -> int:
        return raw.shape[-1] if raw.dim() == 3 else 1

    def uses_independent_parameter_groups(self) -> bool:
        return self.group_routing_strategy == self.GROUP_INDIVIDUAL

    def should_expand_targets(self, n_groups: int) -> bool:
        return n_groups > 1 and self.group_loss_strategy == self.LOSS_PER_GROUP

    def reduce_parameter_groups(self, values: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
        if values.dim() != 3 or values.shape[-1] == 1:
            return values
        return values.mean(dim=-1, keepdim=keepdim)

    def reduce_parameters_for_analysis(self, params: torch.Tensor) -> torch.Tensor:
        if params.dim() != 3 or params.shape[-1] == 1:
            return params.squeeze(-1) if params.dim() == 3 else params
        if self.group_analysis_strategy == self.ANALYSIS_KEEP:
            return params
        return params.mean(dim=-1)

    def maybe_reduce_before_routing(self, values: torch.Tensor) -> torch.Tensor:
        if self.group_routing_strategy != self.GROUP_MEAN_BEFORE_ROUTING:
            return values
        return self.reduce_parameter_groups(values, keepdim=True)

    def prepare_observations_for_metrics(self, observations: torch.Tensor, n_groups: int) -> torch.Tensor:
        if self.should_expand_targets(n_groups):
            return observations.repeat_interleave(n_groups, dim=1)
        return observations

    def unpack_parameters(
        self,
        parameters: Tuple[Optional[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        _, raw = parameters
        static_count = len(self.phy_param_names)

        if raw.dim() == 3:
            return raw[:, :static_count, :]

        if raw.dim() != 2:
            raise ValueError(f"Unexpected parameter shape: {tuple(raw.shape)}")

        if raw.shape[1] == static_count:
            return raw[:, :static_count]

        actual_nmul = max(raw.shape[1] // static_count, 1)
        if actual_nmul == 1:
            return raw[:, :static_count]
        return raw[:, : static_count * actual_nmul].view(raw.shape[0], static_count, actual_nmul)

    # ------------------------------------------------------------------
    # Forward / simulation
    # ------------------------------------------------------------------

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
        return self._run_model(x_dict, states, params_dict, n_groups)

    def _run_warmup(
        self,
        step_fn,
        n_steps: int,
        states: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        effective_warmup = min(self.warm_up, n_steps)
        curr_states = states
        with torch.no_grad():
            for t in range(effective_warmup):
                curr_states = step_fn(t, curr_states)
        return tuple(state.detach() for state in curr_states)

    def _make_forcing_sequences(
        self,
        forcing: torch.Tensor,
        n_groups: int,
    ) -> tuple:
        if n_groups > 1:
            p_seq = forcing[..., 0:1].expand(-1, -1, n_groups).unbind(0)
            t_seq = forcing[..., 1:2].expand(-1, -1, n_groups).unbind(0)
            pet_seq = forcing[..., 2:3].expand(-1, -1, n_groups).unbind(0)
            return p_seq, t_seq, pet_seq

        p_seq = forcing[..., 0:1].unbind(0)
        t_seq = forcing[..., 1:2].unbind(0)
        pet_seq = forcing[..., 2:3].unbind(0)
        return p_seq, t_seq, pet_seq

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
        curr_states = states
        with torch.no_grad():
            for t in range(effective_warmup):
                outputs = self.step_fn(
                    p_seq[t],
                    t_seq[t],
                    pet_seq[t],
                    *param_values,
                    *curr_states,
                    nearzero=self.nearzero,
                )
                curr_states = tuple(outputs[2:])
        curr_states = tuple(state.detach() for state in curr_states)

        n_train = n_steps - effective_warmup
        streamflow = torch.empty(
            (n_train, n_grid, n_groups),
            device=forcing.device,
            dtype=forcing.dtype,
        )

        for offset, t in enumerate(range(effective_warmup, n_steps)):
            outputs = self.step_fn(
                p_seq[t],
                t_seq[t],
                pet_seq[t],
                *param_values,
                *curr_states,
                nearzero=self.nearzero,
            )
            streamflow[offset] = outputs[0]
            curr_states = outputs[2:]

        return self._finalize_output(streamflow)

    def _finalize_output(self, streamflow: torch.Tensor) -> Dict[str, torch.Tensor]:
        if streamflow.dim() == 3 and streamflow.shape[-1] > 1:
            if self.uses_independent_parameter_groups():
                return {"streamflow": streamflow.flatten(start_dim=1)}
            return {"streamflow": streamflow.mean(dim=-1)}

        if streamflow.dim() == 3:
            return {"streamflow": streamflow.squeeze(-1)}
        return {"streamflow": streamflow}
