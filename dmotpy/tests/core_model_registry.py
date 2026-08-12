from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from models import core


STATE_SIGN_OVERRIDES = {
    "ihacres": (-1.0,),
    "penman": (1.0, -1.0, 1.0),
    "tcm": (1.0, -1.0, 1.0, 1.0),
    "topmodel": (1.0, -1.0),
}


DISABLED_MODELS = {
    "shm": "File is empty and does not define a runnable core model implementation.",
}


@dataclass(frozen=True)
class CoreModelEntry:
    model_name: str
    model_file: str
    step_fn: Callable
    init_fn: Callable
    param_bounds: dict[str, list[float]]
    state_names: tuple[str, ...]
    state_signs: tuple[float, ...]
    uses_snow: bool
    supports_diagnostics: bool
    enabled: bool
    skip_reason: str


def _actual_state_count(init_fn: Callable) -> int:
    return len(init_fn(1, 1, torch.device("cpu")))


def _extract_state_names(init_fn: Callable) -> tuple[str, ...]:
    source = inspect.getsource(init_fn)
    matches = re.findall(r"S(\d+)\s*:\s*([^\n]+)", source)
    try:
        actual_count = _actual_state_count(init_fn)
    except Exception:
        actual_count = None
    if matches and (actual_count is None or len(matches) == actual_count):
        names = []
        for index_text, description in matches:
            names.append(f"S{index_text}: {description.strip()}")
        return tuple(names)

    signature = inspect.signature(init_fn)
    annotation = signature.return_annotation
    count = len(getattr(annotation, "__args__", ()))
    if count:
        return tuple(f"S{i + 1}" for i in range(count))
    return ("S1",)


def _uses_snow(step_fn: Callable, param_bounds: dict[str, list[float]]) -> bool:
    source = inspect.getsource(step_fn).lower()
    param_names = {name.lower() for name in param_bounds}
    snow_markers = {"tt", "tti", "ttm", "ddf", "tcrit", "cfmax", "whc", "cfr"}
    return (
        "snow" in source
        or "snowfall" in source
        or "melt" in source
        or "refreeze" in source
        or bool(param_names & snow_markers)
    )


def _entry_from_runtime(
    model_name: str,
    step_fn: Callable,
    init_fn: Callable,
    param_bounds: dict[str, list[float]],
    enabled: bool = True,
    skip_reason: str = "",
) -> CoreModelEntry:
    model_file = Path(inspect.getsourcefile(step_fn)).name
    state_names = _extract_state_names(init_fn)
    actual_state_count = _actual_state_count(init_fn)
    if len(state_names) != actual_state_count:
        state_names = tuple(f"S{i + 1}" for i in range(actual_state_count))
    state_signs = STATE_SIGN_OVERRIDES.get(model_name, tuple(1.0 for _ in range(actual_state_count)))
    return CoreModelEntry(
        model_name=model_name,
        model_file=model_file,
        step_fn=step_fn,
        init_fn=init_fn,
        param_bounds=param_bounds,
        state_names=state_names,
        state_signs=state_signs,
        uses_snow=_uses_snow(step_fn, param_bounds),
        supports_diagnostics=(
            "return_diagnostics" in inspect.signature(step_fn).parameters
            and inspect.signature(step_fn).parameters["return_diagnostics"].kind
            is inspect.Parameter.KEYWORD_ONLY
        ),
        enabled=enabled,
        skip_reason=skip_reason,
    )


def build_core_model_registry() -> dict[str, CoreModelEntry]:
    registry: dict[str, CoreModelEntry] = {}

    for model_name, step_fn in core.STFN_INFO.items():
        init_fn = getattr(core, f"{model_name}_init")
        enabled = model_name not in DISABLED_MODELS
        registry[model_name] = _entry_from_runtime(
            model_name=model_name,
            step_fn=step_fn,
            init_fn=init_fn,
            param_bounds=core.PARAM_INFO[model_name],
            enabled=enabled,
            skip_reason=DISABLED_MODELS.get(model_name, ""),
        )

    registry["shm"] = CoreModelEntry(
        model_name="shm",
        model_file="shm.py",
        step_fn=lambda *args, **kwargs: None,
        init_fn=lambda *args, **kwargs: (),
        param_bounds={},
        state_names=(),
        state_signs=(),
        uses_snow=False,
        supports_diagnostics=False,
        enabled=False,
        skip_reason=DISABLED_MODELS["shm"],
    )
    return registry


CORE_MODEL_REGISTRY = build_core_model_registry()
