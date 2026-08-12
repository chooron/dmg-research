from __future__ import annotations

import ast
import importlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from tests.core_model_registry import CORE_MODEL_REGISTRY


REPO_ROOT = Path(__file__).resolve().parents[1]
FLUX_ROOT = REPO_ROOT / "models" / "flux"
CORE_ROOT = REPO_ROOT / "models" / "core"
SPECIAL_ROOT = REPO_ROOT / "models" / "special"

DEFAULT_NEARZERO = 1.0e-6
DEFAULT_DEVICE = "cpu"
DEFAULT_DTYPE = torch.float64
FIXED_SEED = 20260624


GENERIC_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "p1": (0.05, 5.0),
    "p2": (0.05, 5.0),
    "p3": (0.05, 5.0),
    "p4": (0.0, 5.0),
    "p5": (0.0, 5.0),
    "k": (0.0, 1.0),
    "ddf": (0.0, 20.0),
    "tcrit": (-3.0, 3.0),
    "tt": (-3.0, 3.0),
    "tti": (0.0, 4.0),
    "ttm": (-3.0, 3.0),
    "tmin": (-10.0, 0.0),
    "trange": (1.0, 20.0),
    "lp": (0.05, 2000.0),
    "Smax": (1.0, 2000.0),
    "S1max": (1.0, 2000.0),
    "S2max": (1.0, 2000.0),
    "Smin": (0.0, 50.0),
    "St": (0.0, 50.0),
    "fc": (0.05, 0.95),
    "alpha": (0.0, 1.0),
    "phi": (0.0, 1.0),
    "beta": (0.0, 5.0),
    "gamma": (0.0, 5.0),
    "gam": (0.0, 1.0),
    "rc": (1.0, 2000.0),
    "x1": (1.0, 2000.0),
    "x2": (-20.0, 20.0),
    "x3": (1.0, 300.0),
    "x4": (0.5, 15.0),
    "d": (1.0, 5.0),
    "c": (0.0, 1.0),
    "r": (0.0, 1.0),
    "is_time": (1.0, 365.0),
    "doy": (1.0, 365.0),
    "n": (1.0, 5.0),
}


NONNEGATIVE_TOKEN_HINTS = (
    "baseflow",
    "capillary",
    "depression",
    "effective",
    "evap",
    "excess",
    "exchange",
    "infiltration",
    "interception",
    "interflow",
    "melt",
    "percolation",
    "phenology",
    "rainfall",
    "recharge",
    "refreeze",
    "saturation",
    "snowfall",
    "soilmoisture",
    "split",
)


@dataclass(frozen=True)
class FluxFunctionInfo:
    flux_file: str
    function_name: str
    module_name: str
    line_start: int
    line_end: int
    function_signature: str
    arg_names: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class FluxUsageContext:
    flux_function: str
    flux_file: str
    model_name: str
    module_type: str
    call_site: str
    parameter_mapping: dict[str, str]
    state_variable_mapping: dict[str, str]
    forcing_variable_mapping: dict[str, str]
    parameter_bounds: dict[str, tuple[float, float]]
    inferred_or_exact: str
    active_usage_status: str


@dataclass(frozen=True)
class FluxWrapperContext:
    flux_function: str
    flux_file: str
    model_context: str
    active_usage_status: str
    parameter_mapping: dict[str, str]
    state_variable_mapping: dict[str, str]
    forcing_variable_mapping: dict[str, str]
    parameter_bounds: dict[str, tuple[float, float]]
    inferred_or_exact: str
    call_site: str


@dataclass(frozen=True)
class FluxWrapper:
    flux_info: FluxFunctionInfo
    context: FluxWrapperContext
    callable_fn: Callable[..., torch.Tensor]
    differentiable_inputs: tuple[str, ...]
    manual_review_required: bool
    manual_review_reason: str
    expected_nonnegative: bool
    available_storage_inputs: tuple[str, ...]
    incoming_flux_inputs: tuple[str, ...]
    threshold_inputs: tuple[str, ...]


def _module_path_from_file(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT)
    return ".".join(rel.with_suffix("").parts)


def _source_segment(src: str, node: ast.FunctionDef) -> str:
    return ast.get_source_segment(src, node) or ""


def _infer_formula_type(flux_file: str, function_name: str) -> str:
    stem = Path(flux_file).stem
    return f"{stem}:{function_name}"


def _load_flux_inventory() -> dict[str, FluxFunctionInfo]:
    inventory: dict[str, FluxFunctionInfo] = {}
    for path in sorted(FLUX_ROOT.glob("*.py")):
        if path.name == "__init__.py":
            continue
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
        module_name = _module_path_from_file(path)
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            sig = None
            mod = importlib.import_module(module_name)
            fn = getattr(mod, node.name)
            sig = str(inspect.signature(fn))
            source = _source_segment(src, node)
            inventory[node.name] = FluxFunctionInfo(
                flux_file=str(path.relative_to(REPO_ROOT)),
                function_name=node.name,
                module_name=module_name,
                line_start=node.lineno,
                line_end=getattr(node, "end_lineno", node.lineno),
                function_signature=sig,
                arg_names=tuple(inspect.signature(fn).parameters.keys()),
                source=source,
            )
    return inventory


def _safe_literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _extract_bounds_from_module(module_name: str) -> dict[str, tuple[float, float]]:
    module = importlib.import_module(module_name)
    bounds: dict[str, tuple[float, float]] = {}
    for name, value in vars(module).items():
        if name.endswith("_PARAMS_BOUNDS") and isinstance(value, dict):
            for key, pair in value.items():
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    bounds[key] = (float(pair[0]), float(pair[1]))
    return bounds


def _collect_import_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if "flux" not in module:
                continue
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name
    return aliases


def _expr_name(expr: ast.AST) -> str:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Constant):
        return repr(expr.value)
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return ast.unparse(expr)


def _classify_arg(arg_name: str) -> str:
    lower = arg_name.lower()
    if lower in {"p", "prcp", "precip", "incoming_flux", "flux", "in", "ep", "pet", "t", "doy"}:
        return "forcing"
    if lower.startswith("s") or lower in {"storage", "sn"}:
        return "state"
    if lower in {"nearzero", "dt"}:
        return "meta"
    return "parameter"


def _expand_contexts_from_module(path: Path, module_type: str, inventory: dict[str, FluxFunctionInfo]) -> list[FluxUsageContext]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    aliases = _collect_import_aliases(tree)
    module_name = _module_path_from_file(path)
    model_name = path.stem
    param_bounds = _extract_bounds_from_module(module_name)

    contexts: list[FluxUsageContext] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        alias_name = node.func.id
        flux_name = aliases.get(alias_name)
        if flux_name is None or flux_name not in inventory:
            continue

        flux_info = inventory[flux_name]
        parameter_mapping: dict[str, str] = {}
        state_mapping: dict[str, str] = {}
        forcing_mapping: dict[str, str] = {}
        call_arg_names = [arg.arg for arg in node.keywords if arg.arg is not None]
        positional_exprs = list(node.args)
        for arg_index, flux_arg_name in enumerate(flux_info.arg_names):
            if flux_arg_name == "nearzero":
                continue
            expr = None
            if arg_index < len(positional_exprs):
                expr = positional_exprs[arg_index]
            else:
                for kw in node.keywords:
                    if kw.arg == flux_arg_name:
                        expr = kw.value
                        break
            if expr is None:
                continue
            expr_name = _expr_name(expr)
            category = _classify_arg(expr_name)
            if category == "state":
                state_mapping[flux_arg_name] = expr_name
            elif category == "forcing":
                forcing_mapping[flux_arg_name] = expr_name
            elif category != "meta":
                parameter_mapping[flux_arg_name] = expr_name

        resolved_bounds: dict[str, tuple[float, float]] = {}
        exact = True
        for flux_arg, source_name in parameter_mapping.items():
            if source_name in param_bounds:
                resolved_bounds[flux_arg] = param_bounds[source_name]
                continue
            generic = GENERIC_PARAM_RANGES.get(source_name) or GENERIC_PARAM_RANGES.get(flux_arg)
            if generic is not None:
                resolved_bounds[flux_arg] = generic
                exact = False
            else:
                resolved_bounds[flux_arg] = (0.05, 5.0)
                exact = False

        active = "active_registered_model" if module_type == "core" and model_name in CORE_MODEL_REGISTRY and CORE_MODEL_REGISTRY[model_name].enabled else (
            "special_wrapper" if module_type == "special" else "inactive_or_unregistered"
        )
        contexts.append(
            FluxUsageContext(
                flux_function=flux_name,
                flux_file=flux_info.flux_file,
                model_name=model_name,
                module_type=module_type,
                call_site=f"{path.relative_to(REPO_ROOT)}:{node.lineno}",
                parameter_mapping=parameter_mapping,
                state_variable_mapping=state_mapping,
                forcing_variable_mapping=forcing_mapping,
                parameter_bounds=resolved_bounds,
                inferred_or_exact="exact" if exact else "inferred",
                active_usage_status=active,
            )
        )
    return contexts


def load_flux_usage_contexts() -> list[FluxUsageContext]:
    inventory = load_flux_inventory()
    contexts: list[FluxUsageContext] = []
    for path in sorted(CORE_ROOT.glob("*.py")):
        if path.name == "__init__.py":
            continue
        contexts.extend(_expand_contexts_from_module(path, "core", inventory))
    for path in sorted(SPECIAL_ROOT.glob("*.py")):
        if path.name == "__init__.py":
            continue
        contexts.extend(_expand_contexts_from_module(path, "special", inventory))
    return contexts


_FLUX_INVENTORY_CACHE: dict[str, FluxFunctionInfo] | None = None
_FLUX_CONTEXT_CACHE: list[FluxUsageContext] | None = None


def load_flux_inventory() -> dict[str, FluxFunctionInfo]:
    global _FLUX_INVENTORY_CACHE
    if _FLUX_INVENTORY_CACHE is None:
        _FLUX_INVENTORY_CACHE = _load_flux_inventory()
    return _FLUX_INVENTORY_CACHE


def _load_context_cache() -> list[FluxUsageContext]:
    global _FLUX_CONTEXT_CACHE
    if _FLUX_CONTEXT_CACHE is None:
        _FLUX_CONTEXT_CACHE = load_flux_usage_contexts()
    return _FLUX_CONTEXT_CACHE


def _tensor_from_case(
    name: str,
    kind: str,
    low: float,
    high: float,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: str,
    generator: torch.Generator,
) -> torch.Tensor:
    low = float(low)
    high = float(high)
    span = max(high - low, 1.0e-9)
    if kind == "lower":
        value = low
    elif kind == "upper":
        value = high
    elif kind == "mid":
        value = 0.5 * (low + high)
    elif kind == "near_lower":
        value = low + 0.01 * span
    elif kind == "near_upper":
        value = high - 0.01 * span
    elif kind == "zero":
        value = 0.0
    elif kind == "nearzero":
        value = DEFAULT_NEARZERO
    elif kind == "random":
        rand = torch.rand(shape, dtype=dtype, device=device, generator=generator)
        return low + rand * span
    elif kind == "threshold_minus":
        value = low - 0.01 * max(abs(low), 1.0)
    elif kind == "threshold_plus":
        value = low + 0.01 * max(abs(low), 1.0)
    else:
        value = 0.5 * (low + high)
    return torch.full(shape, float(value), dtype=dtype, device=device)


def _state_range_for_name(name: str) -> tuple[float, float]:
    lower_name = name.lower()
    if lower_name in {"t", "temp"}:
        return (-10.0, 20.0)
    if lower_name == "doy":
        return (1.0, 365.0)
    if lower_name in {"ep", "pet"}:
        return (0.0, 15.0)
    if lower_name in {"p", "incoming_flux", "flux", "in"}:
        return (0.0, 200.0)
    if "max" in lower_name:
        return (1.0, 2000.0)
    if lower_name.startswith("s"):
        return (0.0, 2000.0)
    return (0.0, 100.0)


def _make_callable(module_name: str, function_name: str) -> Callable[..., torch.Tensor]:
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def _wrapper_contexts_for_function(function_name: str) -> list[FluxWrapperContext]:
    contexts = [ctx for ctx in _load_context_cache() if ctx.flux_function == function_name]
    if contexts:
        wrapped = []
        for ctx in contexts:
            wrapped.append(
                FluxWrapperContext(
                    flux_function=ctx.flux_function,
                    flux_file=ctx.flux_file,
                    model_context=ctx.model_name,
                    active_usage_status=ctx.active_usage_status,
                    parameter_mapping=ctx.parameter_mapping,
                    state_variable_mapping=ctx.state_variable_mapping,
                    forcing_variable_mapping=ctx.forcing_variable_mapping,
                    parameter_bounds=ctx.parameter_bounds,
                    inferred_or_exact=ctx.inferred_or_exact,
                    call_site=ctx.call_site,
                )
            )
        return wrapped

    inventory = load_flux_inventory()
    info = inventory[function_name]
    generic_param_bounds = {}
    for arg in info.arg_names:
        if arg == "nearzero":
            continue
        if arg in {"P", "T", "PET", "incoming_flux", "flux", "Ep", "doy"}:
            continue
        if arg.startswith("S") or arg.startswith("s") and arg not in GENERIC_PARAM_RANGES:
            continue
        generic_param_bounds[arg] = GENERIC_PARAM_RANGES.get(arg, (0.05, 5.0))
    return [
        FluxWrapperContext(
            flux_function=function_name,
            flux_file=info.flux_file,
            model_context="generic_unused",
            active_usage_status="unused",
            parameter_mapping={k: k for k in generic_param_bounds},
            state_variable_mapping={},
            forcing_variable_mapping={},
            parameter_bounds=generic_param_bounds,
            inferred_or_exact="inferred",
            call_site="",
        )
    ]


def _expected_nonnegative(flux_info: FluxFunctionInfo) -> bool:
    stem = Path(flux_info.flux_file).stem
    return stem in NONNEGATIVE_TOKEN_HINTS


def _storage_inputs(arg_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in arg_names if name.startswith("S"))


def _incoming_flux_inputs(arg_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in arg_names if name in {"incoming_flux", "flux", "In", "fin", "P"})


def _threshold_inputs(arg_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        name
        for name in arg_names
        if name.lower() in {"x1", "x3"}
        or any(token in name.lower() for token in ("threshold", "tcrit", "tt", "fc", "st", "min", "max", "lp", "p2"))
    )


def build_flux_wrapper(function_name: str, model_context: str) -> FluxWrapper:
    inventory = load_flux_inventory()
    flux_info = inventory[function_name]
    contexts = _wrapper_contexts_for_function(function_name)
    context = next((ctx for ctx in contexts if ctx.model_context == model_context), None)
    if context is None:
        raise KeyError(f"No wrapper context for {function_name} / {model_context}")

    callable_fn = _make_callable(flux_info.module_name, function_name)
    differentiable_inputs = tuple(name for name in flux_info.arg_names if name != "nearzero")

    manual_review_required = any(
        token in flux_info.source for token in ("integral(", "route(", "uh_", "TODO")
    )
    manual_reason = "signature contains unresolved runtime behavior" if manual_review_required else ""

    return FluxWrapper(
        flux_info=flux_info,
        context=context,
        callable_fn=callable_fn,
        differentiable_inputs=differentiable_inputs,
        manual_review_required=manual_review_required,
        manual_review_reason=manual_reason,
        expected_nonnegative=_expected_nonnegative(flux_info),
        available_storage_inputs=_storage_inputs(flux_info.arg_names),
        incoming_flux_inputs=_incoming_flux_inputs(flux_info.arg_names),
        threshold_inputs=_threshold_inputs(flux_info.arg_names),
    )


def iter_all_flux_wrappers() -> list[FluxWrapper]:
    wrappers: list[FluxWrapper] = []
    for function_name in sorted(load_flux_inventory()):
        for ctx in _wrapper_contexts_for_function(function_name):
            wrappers.append(build_flux_wrapper(function_name, ctx.model_context))
    return wrappers


def build_wrapper_inputs(
    wrapper: FluxWrapper,
    parameter_case: str = "mid",
    state_case: str = "mid",
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    shape: tuple[int, ...] = (9,),
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device if device != "cpu" else "cpu")
    generator.manual_seed(FIXED_SEED)
    values: dict[str, torch.Tensor] = {}

    for name in wrapper.flux_info.arg_names:
        if name == "nearzero":
            continue

        if name in wrapper.context.parameter_bounds:
            low, high = wrapper.context.parameter_bounds[name]
            tensor = _tensor_from_case(name, parameter_case, low, high, shape, dtype, device, generator)
        elif name in wrapper.context.state_variable_mapping or name in wrapper.context.forcing_variable_mapping:
            low, high = _state_range_for_name(name)
            tensor = _tensor_from_case(name, state_case, low, high, shape, dtype, device, generator)
        else:
            low, high = _state_range_for_name(name)
            case = parameter_case if name.startswith("p") or name in GENERIC_PARAM_RANGES else state_case
            tensor = _tensor_from_case(name, case, low, high, shape, dtype, device, generator)

        values[name] = tensor.requires_grad_(name != "nearzero")

    return values


def evaluate_wrapper(wrapper: FluxWrapper, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    ordered_args = []
    kwargs = {}
    for name in wrapper.flux_info.arg_names:
        if name == "nearzero":
            kwargs["nearzero"] = DEFAULT_NEARZERO
        else:
            ordered_args.append(inputs[name])
    output = wrapper.callable_fn(*ordered_args, **kwargs)
    # Shared flux helpers may return several coupled flux fractions.  Stack
    # them so the generic finite/gradient audit can inspect every component.
    if isinstance(output, tuple):
        return torch.stack(output, dim=0)
    return output


def wrapper_registry_summary() -> list[dict[str, Any]]:
    rows = []
    for wrapper in iter_all_flux_wrappers():
        rows.append(
            {
                "flux_function": wrapper.flux_info.function_name,
                "flux_file": wrapper.flux_info.flux_file,
                "model_context": wrapper.context.model_context,
                "active_usage_status": wrapper.context.active_usage_status,
                "manual_review_required": wrapper.manual_review_required,
                "manual_review_reason": wrapper.manual_review_reason,
                "parameter_bounds": json.dumps(wrapper.context.parameter_bounds, ensure_ascii=False),
            }
        )
    return rows
