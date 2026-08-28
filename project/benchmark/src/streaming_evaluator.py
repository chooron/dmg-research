from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, ContextManager

import torch

from dmotpy.models.endpoint_uh_model import EndpointUHModel
from dmotpy.models.gr4j_uh_model import GR4JUHModel
from dmotpy.models.hydrology_model import HydrologyModel
from dmotpy.models.intermediate_uh_model import IntermediateUHModel
from dmotpy.models.mopex_doy_model import MopexDoyModel
from dmotpy.models.tcm_model import TCMModel

from .objective import (
    StreamingKGEState,
    finalize_streaming_kge_tensors,
    initialize_streaming_kge,
    update_streaming_kge_tensors,
)


_COMPILED_UH_STEP: Callable | None = None
_COMPILED_KGE_UPDATE: Callable | None = None


def _stream_uh_step(
    flux: torch.Tensor,
    history: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one causal UH step to a flattened candidate batch."""
    window = torch.cat((flux.unsqueeze(1), history), dim=1)
    routed = (window * weights).sum(dim=1)
    new_history = torch.cat((flux.unsqueeze(1), history[:, :-1]), dim=1)
    return routed, new_history


def _make_uh_step() -> Callable:
    global _COMPILED_UH_STEP
    if _COMPILED_UH_STEP is None:
        if not hasattr(torch, "compile"):
            raise RuntimeError("streaming evaluation requires torch.compile")
        _COMPILED_UH_STEP = torch.compile(
            _stream_uh_step, backend="inductor", mode="default", fullgraph=True
        )
    return _COMPILED_UH_STEP


def _make_kge_update() -> Callable:
    global _COMPILED_KGE_UPDATE
    if _COMPILED_KGE_UPDATE is None:
        if not hasattr(torch, "compile"):
            raise RuntimeError("streaming evaluation requires torch.compile")
        _COMPILED_KGE_UPDATE = torch.compile(
            update_streaming_kge_tensors,
            backend="inductor",
            mode="default",
            fullgraph=True,
        )
    return _COMPILED_KGE_UPDATE


def _expand_forcing(forcing: torch.Tensor, timestep: int, groups: int) -> tuple[torch.Tensor, ...]:
    row = forcing[timestep]
    return tuple(row[:, index : index + 1].expand(-1, groups) for index in range(3))


class _StreamingExecution:
    """Family hook contract: advance all model state and return one prediction."""

    def __init__(
        self,
        model: HydrologyModel,
        forcing: torch.Tensor,
        states: tuple[torch.Tensor, ...],
        params: dict[str, torch.Tensor],
        groups: int,
    ) -> None:
        self.model = model
        self.forcing = forcing
        self.states = tuple(states)
        self.params = params
        self.groups = groups
        self.param_values = [params[name] for name in model.phy_param_names]

    def __enter__(self) -> "_StreamingExecution":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> bool:
        return False

    def warmup_step(self, timestep: int) -> None:
        self.train_step(timestep)

    def train_step(self, timestep: int) -> torch.Tensor:
        raise NotImplementedError


class _BaseExecution(_StreamingExecution):
    """HydrologyModel family: one compiled hydrology step owns all routing state."""

    def _step_kwargs(self) -> dict[str, torch.Tensor | float]:
        return {"nearzero": self.model.nearzero}

    def train_step(self, timestep: int) -> torch.Tensor:
        p, tmean, pet = _expand_forcing(self.forcing, timestep, self.groups)
        outputs = self.model.step_fn(
            p,
            tmean,
            pet,
            *self.param_values,
            *self.states,
            **self._step_kwargs(),
        )
        self.states = tuple(outputs[2:])
        return outputs[0]


class _TCMExecution(_BaseExecution):
    """TCM family: the only extra input is mean precipitation over the full window."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mean_p = self.forcing[..., 0].mean(dim=0).unsqueeze(-1).expand(
            self.forcing.shape[1], self.groups
        )

    def _step_kwargs(self) -> dict[str, torch.Tensor | float]:
        return {"nearzero": self.model.nearzero, "mean_P": self.mean_p}


class _IntermediateUHExecution(_StreamingExecution):
    """Intermediate UH family with bounded causal history for each branch."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        model = self.model
        self.pre_states = list(self.states[: model._n_pre_states])
        self.post_states = list(self.states[model._n_pre_states :])
        self.pre_params = [self.params[name] for name in model._pre_param_names]
        self.post_params = [self.params[name] for name in model._post_param_names]
        self.uh_step = _make_uh_step()
        flat_count = self.forcing.shape[1] * self.groups
        self.routes: list[dict[str, object]] = []

        if hasattr(model, "uh_fast"):
            specs = (
                (model.uh_fast, self.params["nlagf"], 0, 1.0),
                (model.uh_slow, self.params["nlags"], 1, 1.0),
            )
        elif isinstance(model, GR4JUHModel):
            specs = (
                (model.uh_half, self.params["x4"], 0, 0.9),
                (model.uh_full, self.params["x4"] * 2.0, 0, 0.1),
            )
        else:
            raise TypeError(f"unsupported intermediate UH model: {type(model).__name__}")

        for module, parameter, flux_index, scale in specs:
            uh_parameter = parameter.expand(self.forcing.shape[1], self.groups).reshape(flat_count, 1)
            weights = module.get_weights(uh_parameter).squeeze(1)
            self.routes.append({
                "weights": weights,
                "history": torch.zeros(
                    (flat_count, weights.shape[-1] - 1),
                    device=self.forcing.device,
                    dtype=self.forcing.dtype,
                ),
                "flux_index": flux_index,
                "scale": scale,
            })

    def _route(self, fluxes: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        routed: list[torch.Tensor] = []
        for route in self.routes:
            flux = fluxes[int(route["flux_index"])] * float(route["scale"])
            flat, history = self.uh_step(
                flux.reshape(-1), route["history"], route["weights"]
            )
            route["history"] = history
            routed.append(flat.view_as(flux))
        return routed

    def train_step(self, timestep: int) -> torch.Tensor:
        p, tmean, pet = _expand_forcing(self.forcing, timestep, self.groups)
        result = self.model.step_pre_fn(
            p,
            tmean,
            pet,
            *self.pre_params,
            *self.pre_states,
            self.model.nearzero,
        )
        n_fluxes = len(result) - self.model._n_pre_passthru - self.model._n_pre_states
        fluxes = tuple(result[:n_fluxes])
        passthrough = tuple(result[n_fluxes : n_fluxes + self.model._n_pre_passthru])
        self.pre_states = list(result[n_fluxes + self.model._n_pre_passthru :])
        routed = self._route(fluxes)
        if isinstance(self.model, GR4JUHModel):
            post_result = self.model.step_post_fn(
                *routed,
                *self.post_states,
                *self.post_params,
                *passthrough,
                self.model.nearzero,
            )
        else:
            post_result = self.model.step_post_fn(
                *routed,
                *passthrough,
                *self.post_states,
                *self.post_params,
                self.model.nearzero,
            )
        self.post_states = list(post_result[2:])
        self.states = tuple(self.pre_states + self.post_states)
        return post_result[0]


class _EndpointUHExecution(_StreamingExecution):
    """Endpoint UH family with one or two bounded causal routing histories."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.endpoint_kind = self.model._endpoint_scheme["kind"]
        self.uh_step = _make_uh_step()
        flat_count = self.forcing.shape[1] * self.groups
        self.routes: list[dict[str, object]] = []
        for index, (uh_kind, parameter_name) in enumerate(self.model._endpoint_scheme["uhs"]):
            if uh_kind == "gamma6":
                n_res = self.params["n_res"].expand(self.forcing.shape[1], self.groups).reshape(flat_count, 1)
                nk_delay = self.params["nk_delay"].expand(self.forcing.shape[1], self.groups).reshape(flat_count, 1)
                parameter = torch.cat((n_res, nk_delay / (n_res + self.model.nearzero)), dim=1)
            else:
                parameter = self.params[parameter_name].expand(
                    self.forcing.shape[1], self.groups
                ).reshape(flat_count, 1)
            weights = self.model.uh_modules[index].get_weights(parameter).squeeze(1)
            self.routes.append({
                "weights": weights,
                "history": torch.zeros(
                    (flat_count, weights.shape[-1] - 1),
                    device=self.forcing.device,
                    dtype=self.forcing.dtype,
                ),
            })

    def _route(self, flux: torch.Tensor, route: dict[str, object]) -> torch.Tensor:
        flat, history = self.uh_step(
            flux.reshape(-1), route["history"], route["weights"]
        )
        route["history"] = history
        return flat.view_as(flux)

    def train_step(self, timestep: int) -> torch.Tensor:
        p, tmean, pet = _expand_forcing(self.forcing, timestep, self.groups)
        if self.endpoint_kind in ("surface_baseflow", "exp_delay_chain"):
            outputs = self.model.step_fn(
                p,
                tmean,
                pet,
                *self.param_values,
                *self.states,
                self.model.nearzero,
                return_routing_fluxes=True,
            )
            fluxes = outputs[-1]
            self.states = tuple(outputs[2:-1])
        else:
            outputs = self.model.step_fn(
                p,
                tmean,
                pet,
                *self.param_values,
                *self.states,
                self.model.nearzero,
            )
            fluxes = None
            self.states = tuple(outputs[2:])

        if self.endpoint_kind == "total":
            return self._route(outputs[0], self.routes[0])
        if self.endpoint_kind == "surface_baseflow":
            routed_surface = self._route(fluxes[0], self.routes[0])
            return routed_surface + fluxes[1]
        if self.endpoint_kind == "exp_delay_chain":
            return self._route(fluxes[0], self.routes[0]) + self._route(fluxes[1], self.routes[1])
        raise ValueError(f"unknown endpoint routing kind: {self.endpoint_kind}")


class _MopexDoyExecution(_StreamingExecution):
    """MOPEX4/5 family: day-of-year forcing, context globals, and state reordering."""

    def __init__(self, *args, circular_phase=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.doy = self.forcing[..., 3]
        self.circular_phase = circular_phase
        if self.model.model_name in {"mopex4", "mopex5"} and len(self.states) == 5:
            self.states = (self.states[1], self.states[2], self.states[3], self.states[4], self.states[0])
        self._context: ContextManager | None = None

    def __enter__(self) -> "_MopexDoyExecution":
        try:
            from dmotpy.models.flux.mopex import mopex_training_context
        except ImportError:
            self._context = nullcontext()
        else:
            self._context = mopex_training_context(
                lambda_i=float(getattr(self.model, "continuation_lambda_i", 1.0)),
                lambda_p=float(getattr(self.model, "continuation_lambda_p", 1.0)),
                beta=float(getattr(self.model, "continuation_beta", 50.0)),
            )
        self._context.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self._context is not None:
            self._context.__exit__(exc_type, exc_value, traceback)
            self._context = None
        return False

    def train_step(self, timestep: int) -> torch.Tensor:
        p, tmean, pet = _expand_forcing(self.forcing, timestep, self.groups)
        doy = self.doy[timestep].unsqueeze(-1).expand(-1, self.groups)
        phase_kwargs = {} if self.circular_phase is None else {
            "phase_cos": self.circular_phase[0],
            "phase_sin": self.circular_phase[1],
        }
        outputs = self.model.step_fn(
            p,
            tmean,
            pet,
            *self.param_values,
            *self.states,
            delta_t=1.0,
            nearzero=self.model.nearzero,
            doy=doy,
            **phase_kwargs,
        )
        self.states = tuple(outputs[2:])
        return outputs[0]


def _prepare_parameters(
    model: HydrologyModel,
    latent: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], int, tuple[torch.Tensor, ...] | None]:
    raw = torch.sigmoid(latent).permute(0, 3, 1, 2).reshape(
        latent.shape[0], latent.shape[-1], latent.shape[1] * latent.shape[2]
    ).to(torch.float64)
    raw = model.unpack_parameters((None, raw))
    circular_phase = None
    if isinstance(model, MopexDoyModel):
        raw, circular_phase = model._split_phase_parameters(raw)
    groups = model.get_parameter_group_count(raw)
    return model._descale_params(raw), groups, circular_phase


def _make_execution(
    model: HydrologyModel,
    forcing: torch.Tensor,
    params: dict[str, torch.Tensor],
    groups: int,
    circular_phase: tuple[torch.Tensor, ...] | None,
) -> _StreamingExecution:
    states = model._init_states(forcing.shape[1], groups)
    if isinstance(model, MopexDoyModel):
        return _MopexDoyExecution(model, forcing, states, params, groups, circular_phase=circular_phase)
    if isinstance(model, TCMModel):
        return _TCMExecution(model, forcing, states, params, groups)
    if isinstance(model, EndpointUHModel):
        return _EndpointUHExecution(model, forcing, states, params, groups)
    if isinstance(model, IntermediateUHModel):
        return _IntermediateUHExecution(model, forcing, states, params, groups)
    return _BaseExecution(model, forcing, states, params, groups)


@torch.inference_mode()
def compute_streaming_fitness(
    model: HydrologyModel,
    forcing: torch.Tensor,
    observation: torch.Tensor,
    latent: torch.Tensor,
    *,
    warmup_days: int,
    invalid_penalty: float = -1_000_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run any registered benchmark model with bounded streaming state."""
    if getattr(model, "backend", "compile") != "compile":
        raise RuntimeError("streaming production evaluation requires backend=compile")
    forcing = forcing if forcing.dtype == torch.float64 else forcing.to(torch.float64)
    observation = observation if observation.dtype == torch.float64 else observation.to(torch.float64)
    latent = latent if latent.dtype == torch.float64 else latent.to(torch.float64)
    model.to(dtype=torch.float64)
    model.compute_dtype = torch.float64
    if latent.ndim != 4:
        raise ValueError(f"expected latent [B,S,P,D], got {tuple(latent.shape)}")
    if forcing.ndim != 3 or forcing.shape[-1] < 3:
        raise ValueError(f"expected forcing [T,B,>=3], got {tuple(forcing.shape)}")
    if forcing.shape[1] != latent.shape[0]:
        raise ValueError("forcing and latent basin dimensions do not match")
    if latent.shape[-1] != len(model.phy_param_names):
        raise ValueError(
            f"latent dimension {latent.shape[-1]} does not match model dimension {len(model.phy_param_names)}"
        )

    n_steps = int(forcing.shape[0])
    warmup = min(int(warmup_days), n_steps)
    if observation.shape[0] != n_steps - warmup:
        raise ValueError(
            f"observation length {observation.shape[0]} does not match training length {n_steps - warmup}"
        )

    params, groups, circular_phase = _prepare_parameters(model, latent)
    execution = _make_execution(model, forcing, params, groups, circular_phase)
    state = initialize_streaming_kge((latent.shape[0], groups), forcing.device)
    metric_step = _make_kge_update()

    with execution:
        for timestep in range(warmup):
            execution.warmup_step(timestep)
        for timestep in range(warmup, n_steps):
            prediction = execution.train_step(timestep)
            values = metric_step(
                state.count,
                state.sum_pred,
                state.sum_obs,
                state.sum_pred2,
                state.sum_obs2,
                state.sum_cross,
                state.invalid_prediction,
                prediction,
                observation[timestep - warmup],
            )
            state = StreamingKGEState(*values)

    score, invalid = finalize_streaming_kge_tensors(
        state.count,
        state.sum_pred,
        state.sum_obs,
        state.sum_pred2,
        state.sum_obs2,
        state.sum_cross,
        state.invalid_prediction,
        invalid_penalty=invalid_penalty,
    )
    return score.view(latent.shape[0], latent.shape[1], latent.shape[2]), invalid.view(
        latent.shape[0], latent.shape[1], latent.shape[2]
    )


def compute_flexi_streaming_fitness(
    model: HydrologyModel,
    forcing: torch.Tensor,
    observation: torch.Tensor,
    latent: torch.Tensor,
    *,
    warmup_days: int,
    invalid_penalty: float = -1_000_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible FlexI entry point backed by the family evaluator."""
    if model.model_name != "flexi":
        raise ValueError(f"streaming FlexI evaluator received {model.model_name}")
    return compute_streaming_fitness(
        model,
        forcing,
        observation,
        latent,
        warmup_days=warmup_days,
        invalid_penalty=invalid_penalty,
    )
