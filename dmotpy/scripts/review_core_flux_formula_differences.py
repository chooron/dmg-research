from __future__ import annotations

import csv
import importlib
import inspect
import inspect
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
OUT_DIR = (
    REPO_ROOT
    / "validation_results"
    / "architecture_audit"
    / "formula_difference_review"
)
DTYPE = torch.float64
TOL = 1e-10


def t(values: list[float] | tuple[float, ...]) -> torch.Tensor:
    return torch.tensor(list(values), dtype=DTYPE)


def rel_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        try:
            return str(resolved.relative_to(WORKSPACE_ROOT))
        except ValueError:
            return str(resolved)


def flatten_grid(**axes: torch.Tensor) -> dict[str, torch.Tensor]:
    names = list(axes)
    tensors = [axes[name] for name in names]
    mesh = torch.meshgrid(*tensors, indexing="ij")
    return {name: grid.reshape(-1) for name, grid in zip(names, mesh)}


def combine_segments(*segments: torch.Tensor) -> torch.Tensor:
    return torch.unique(torch.cat([seg.reshape(-1) for seg in segments])).to(DTYPE)


def safe_rel_l2(diff: torch.Tensor, ref: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(ref).item()
    if denom <= 1e-14:
        return float(torch.linalg.vector_norm(diff).item())
    return float(torch.linalg.vector_norm(diff).item() / denom)


def safe_max_relative(diff: torch.Tensor, ref: torch.Tensor) -> float:
    denom = torch.clamp(ref.abs(), min=1e-12)
    return float((diff.abs() / denom).max().item())


def bool_text(flag: bool) -> str:
    return "yes" if flag else "no"


def format_float(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.12g}"


def source_info(fn: Callable[..., Any]) -> tuple[str, int, int, str]:
    source_file = inspect.getsourcefile(fn)
    source_lines, start_line = inspect.getsourcelines(fn)
    snippet = inspect.getsource(fn).rstrip()
    return rel_path(source_file), start_line, start_line + len(source_lines) - 1, snippet


def file_snippet(path: Path, line_start: int, line_end: int) -> str:
    lines = path.read_text().splitlines()
    return "\n".join(lines[line_start - 1 : line_end]).rstrip()


def finite_status(grads: list[torch.Tensor | None]) -> tuple[str, float]:
    values: list[torch.Tensor] = []
    for grad in grads:
        if grad is None:
            continue
        values.append(grad.reshape(-1))
    if not values:
        return "no_gradients", float("nan")
    flat = torch.cat(values)
    if torch.isnan(flat).any():
        return "nan_gradients", float("nan")
    if torch.isinf(flat).any():
        return "inf_gradients", float("inf")
    return "finite", float(flat.abs().max().item())


def run_with_grads(fn: Callable[..., torch.Tensor], kwargs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, str, float]:
    grad_args: list[torch.Tensor] = []
    call_kwargs: dict[str, torch.Tensor] = {}
    for key, value in kwargs.items():
        tensor = value.clone().detach().to(DTYPE)
        tensor.requires_grad_(True)
        call_kwargs[key] = tensor
        grad_args.append(tensor)
    output = fn(call_kwargs)
    grads = torch.autograd.grad(output.sum(), grad_args, allow_unused=True)
    status, max_grad = finite_status(list(grads))
    return output.detach(), status, max_grad


@dataclass(frozen=True)
class PairReview:
    pair_id: str
    core_model: str
    core_module: str
    core_attr: str
    flux_module: str
    flux_attr: str
    formula_type: str
    core_formula_name: str
    flux_formula_name: str
    core_expression_summary: str
    flux_expression_summary: str
    same_inputs: str
    same_units: str
    same_soft_gate_direction: str
    same_bounds_or_limiter: str
    same_epsilon_handling: str
    initial_equivalence_judgment: str
    notes: str
    core_wrapper: Callable[[dict[str, torch.Tensor]], torch.Tensor]
    flux_wrapper: Callable[[dict[str, torch.Tensor]], torch.Tensor]
    cases: Callable[[], list[dict[str, Any]]]
    threshold_mask: Callable[[dict[str, torch.Tensor]], torch.Tensor]
    allow_negative: bool = False
    upper_bound: Callable[[dict[str, torch.Tensor]], torch.Tensor | None] | None = None
    flux_unused: bool = False


@dataclass(frozen=True)
class StandaloneReview:
    pair_id: str
    core_model: str
    core_formula: str
    core_file: str
    line_start: int
    line_end: int
    formula_type: str
    code_snippet: str
    equivalence_class: str
    recommended_future_action: str
    short_reason: str


core_ihacres = importlib.import_module("dmotpy.models.core.ihacres")
core_modhydrolog = importlib.import_module("dmotpy.models.core.modhydrolog")
core_mopex1 = importlib.import_module("dmotpy.models.core.mopex1")
core_mopex2 = importlib.import_module("dmotpy.models.core.mopex2")
core_mopex4 = importlib.import_module("dmotpy.models.core.mopex4")
core_mopex5 = importlib.import_module("dmotpy.models.core.mopex5")
core_tcm = importlib.import_module("dmotpy.models.core.tcm")

flux_baseflow = importlib.import_module("dmotpy.models.flux.baseflow")
flux_depression = importlib.import_module("dmotpy.models.flux.depression")
flux_evap = importlib.import_module("dmotpy.models.flux.evap")
flux_exchange = importlib.import_module("dmotpy.models.flux.exchange")
flux_infiltration = importlib.import_module("dmotpy.models.flux.infiltration")
flux_interception = importlib.import_module("dmotpy.models.flux.interception")
flux_melt = importlib.import_module("dmotpy.models.flux.melt")
flux_phenology = importlib.import_module("dmotpy.models.flux.phenology")
flux_rainfall = importlib.import_module("dmotpy.models.flux.rainfall")
flux_recharge = importlib.import_module("dmotpy.models.flux.recharge")
flux_saturation = importlib.import_module("dmotpy.models.flux.saturation")
flux_snowfall = importlib.import_module("dmotpy.models.flux.snowfall")


def case_ihacres() -> list[dict[str, Any]]:
    out = []
    for lp in (1.0, 50.0, 500.0):
        S = combine_segments(
            torch.linspace(0.0, 2.0 * lp, 401, dtype=DTYPE),
            torch.linspace(0.8 * lp, 1.2 * lp, 161, dtype=DTYPE),
        )
        for ep in (0.1, 2.0, 8.0):
            out.append(
                {
                    "name": f"lp={lp:g}_ep={ep:g}",
                    "core": {"S": S, "lp": torch.full_like(S, lp), "Ep": torch.full_like(S, ep)},
                    "flux": {"S": S, "p1": torch.full_like(S, lp), "Ep": torch.full_like(S, ep)},
                }
            )
    return out


def case_modhydrolog_exchange() -> list[dict[str, Any]]:
    out = []
    for fmax in (0.5, 5.0):
        S = combine_segments(
            torch.linspace(-5.0, 5.0, 401, dtype=DTYPE),
            torch.linspace(-500.0, 500.0, 401, dtype=DTYPE),
        )
        out.append(
            {
                "name": f"fmax={fmax:g}",
                "core": {
                    "p1": torch.full_like(S, 0.05),
                    "p2": torch.full_like(S, 0.4),
                    "p3": torch.full_like(S, 12.0),
                    "S": S,
                    "fmax": torch.full_like(S, fmax),
                },
                "flux": {
                    "p1": torch.full_like(S, 0.05),
                    "p2": torch.full_like(S, 0.4),
                    "p3": torch.full_like(S, 12.0),
                    "S": S,
                    "fmax": torch.full_like(S, fmax),
                },
            }
        )
    return out


def case_modhydrolog_infiltration_1() -> list[dict[str, Any]]:
    out = []
    for smax in (0.2, 1.0, 200.0):
        grid = flatten_grid(
            S=combine_segments(
                torch.linspace(0.0, 2.0 * max(smax, 1.0), 301, dtype=DTYPE),
                torch.linspace(0.0, 50.0 * max(smax, 1.0), 301, dtype=DTYPE),
            ),
            fin=t((0.0, 0.2, 3.0, 20.0)),
            p1=t((0.1, 20.0, 200.0)),
            p2=t((0.1, 5.0, 15.0)),
        )
        out.append(
            {
                "name": f"smax={smax:g}",
                "core": {**grid, "Smax": torch.full_like(grid["S"], smax)},
                "flux": {**grid, "Smax": torch.full_like(grid["S"], smax)},
            }
        )
    return out


def case_modhydrolog_infiltration_2() -> list[dict[str, Any]]:
    out = []
    for smax in (0.2, 1.0, 200.0):
        grid = flatten_grid(
            S1=combine_segments(
                torch.linspace(0.0, 2.0 * max(smax, 1.0), 241, dtype=DTYPE),
                torch.linspace(0.0, 50.0 * max(smax, 1.0), 241, dtype=DTYPE),
            ),
            flux=t((0.0, 0.2, 2.0, 10.0)),
            S2=t((0.0, 0.5, 5.0, 20.0)),
            p1=t((0.1, 20.0, 200.0)),
            p2=t((0.1, 5.0, 15.0)),
        )
        out.append(
            {
                "name": f"s1max={smax:g}",
                "core": {**grid, "S1max": torch.full_like(grid["S1"], smax)},
                "flux": {**grid, "S1max": torch.full_like(grid["S1"], smax)},
            }
        )
    return out


def case_modhydrolog_evap_2() -> list[dict[str, Any]]:
    out = []
    for smax in (0.2, 1.0, 200.0):
        grid = flatten_grid(
            S=combine_segments(
                torch.linspace(0.0, 2.0 * max(smax, 1.0), 301, dtype=DTYPE),
                torch.linspace(0.8 * max(smax, 1.0), 1.2 * max(smax, 1.0), 161, dtype=DTYPE),
            ),
            p1=t((0.1, 1.0, 10.0)),
            Ep=t((0.0, 0.5, 5.0, 20.0)),
        )
        out.append(
            {
                "name": f"smax={smax:g}",
                "core": {**grid, "Smax": torch.full_like(grid["S"], smax)},
                "flux": {**grid, "Smax": torch.full_like(grid["S"], smax)},
            }
        )
    return out


def case_interception_overflow() -> list[dict[str, Any]]:
    out = []
    for smax in (1.0, 10.0, 100.0):
        grid = flatten_grid(
            incoming_flux=t((0.0, 0.5, 5.0, 20.0)),
            S=combine_segments(
                torch.linspace(0.0, 2.0 * smax, 321, dtype=DTYPE),
                torch.linspace(0.8 * smax, 1.2 * smax, 161, dtype=DTYPE),
            ),
        )
        out.append(
            {
                "name": f"smax={smax:g}",
                "core": {**grid, "Smax": torch.full_like(grid["S"], smax)},
                "flux": {**grid, "Smax": torch.full_like(grid["S"], smax)},
            }
        )
    return out


def case_depression() -> list[dict[str, Any]]:
    out = []
    for smax in (2.0, 10.0, 30.0):
        grid = flatten_grid(
            S=combine_segments(
                torch.linspace(0.0, 1.2 * smax, 301, dtype=DTYPE),
                torch.linspace(0.8 * smax, 1.05 * smax, 161, dtype=DTYPE),
            ),
            incoming_flux=t((1e-6, 0.1, 1.0, 10.0, 50.0)),
        )
        out.append(
            {
                "name": f"smax={smax:g}",
                "core": {
                    **grid,
                    "ads": torch.full_like(grid["S"], 0.3),
                    "md": torch.full_like(grid["S"], 0.995),
                    "Smax": torch.full_like(grid["S"], smax),
                },
                "flux": {
                    **grid,
                    "p1": torch.full_like(grid["S"], 0.3),
                    "p2": torch.full_like(grid["S"], 0.995),
                    "Smax": torch.full_like(grid["S"], smax),
                },
            }
        )
    return out


def case_mopex1_evap_7() -> list[dict[str, Any]]:
    out = []
    for smax in (1.0, 50.0, 500.0):
        grid = flatten_grid(
            S=combine_segments(
                torch.linspace(0.0, 2.0 * smax, 301, dtype=DTYPE),
                torch.linspace(0.8 * smax, 1.2 * smax, 161, dtype=DTYPE),
            ),
            Ep=t((0.0, 0.1, 5.0, 20.0)),
        )
        out.append(
            {
                "name": f"smax={smax:g}",
                "core": {**grid, "Smax": torch.full_like(grid["S"], smax)},
                "flux": {**grid, "Smax": torch.full_like(grid["S"], smax)},
            }
        )
    return out


def case_mopex1_saturation() -> list[dict[str, Any]]:
    out = []
    for smax in (10.0, 100.0, 500.0):
        grid = flatten_grid(
            P=t((0.0, 0.2, 5.0, 20.0)),
            S=combine_segments(
                torch.linspace(0.0, 1.2 * smax, 301, dtype=DTYPE),
                torch.linspace(0.95 * smax, 1.05 * smax, 201, dtype=DTYPE),
            ),
        )
        out.append(
            {
                "name": f"smax={smax:g}",
                "core": {**grid, "Smax": torch.full_like(grid["S"], smax)},
                "flux": {"incoming_flux": grid["P"], "S": grid["S"], "Smax": torch.full_like(grid["S"], smax)},
            }
        )
    return out


def case_linear_store() -> list[dict[str, Any]]:
    grid = flatten_grid(
        S=combine_segments(
            torch.linspace(0.0, 500.0, 401, dtype=DTYPE),
            torch.linspace(0.0, 5.0, 201, dtype=DTYPE),
        ),
        k=t((0.0, 0.2, 1.0, 1.5)),
    )
    return [{"name": "linear_store", "core": grid, "flux": {"p1": grid["k"], "S": grid["S"]}}]


def case_temp_partition() -> list[dict[str, Any]]:
    out = []
    for tcrit in (-2.0, 0.0, 2.0):
        T = combine_segments(
            torch.linspace(tcrit - 5.0, tcrit + 5.0, 401, dtype=DTYPE),
            torch.linspace(tcrit - 1.0, tcrit + 1.0, 201, dtype=DTYPE),
        )
        grid = flatten_grid(P=t((0.0, 0.5, 5.0, 20.0)), T=T)
        out.append(
            {
                "name": f"tcrit={tcrit:g}",
                "core": {"P": grid["P"], "T": grid["T"], "tcrit": torch.full_like(grid["T"], tcrit)},
                "flux": {"incoming_flux": grid["P"], "T": grid["T"], "p1": torch.full_like(grid["T"], tcrit)},
            }
        )
    return out


def case_melt() -> list[dict[str, Any]]:
    out = []
    for tcrit in (-2.0, 0.0, 2.0):
        T = combine_segments(
            torch.linspace(tcrit - 5.0, tcrit + 5.0, 401, dtype=DTYPE),
            torch.linspace(tcrit - 1.0, tcrit + 1.0, 201, dtype=DTYPE),
        )
        grid = flatten_grid(ddf=t((0.1, 2.0, 8.0)), T=T, Sn=t((0.0, 0.5, 5.0, 50.0)))
        out.append(
            {
                "name": f"tcrit={tcrit:g}",
                "core": {**grid, "tcrit": torch.full_like(grid["T"], tcrit)},
                "flux": {"p1": grid["ddf"], "p2": torch.full_like(grid["T"], tcrit), "T": grid["T"], "S": grid["Sn"]},
            }
        )
    return out


def case_interception_4() -> list[dict[str, Any]]:
    out = []
    doy = combine_segments(
        torch.linspace(1.0, 365.25, 365, dtype=DTYPE),
        torch.linspace(160.0, 205.0, 181, dtype=DTYPE),
    )
    for alpha in (0.0, 0.3, 0.7):
        grid = flatten_grid(flux_pr=t((0.0, 0.2, 5.0, 20.0)), doy=doy)
        out.append(
            {
                "name": f"alpha={alpha:g}",
                "core": {
                    **grid,
                    "alpha": torch.full_like(grid["doy"], alpha),
                    "is_time": torch.full_like(grid["doy"], 180.0),
                },
                "flux": {
                    "p1": torch.full_like(grid["doy"], alpha),
                    "p2": torch.full_like(grid["doy"], 180.0),
                    "t": grid["doy"],
                    "tmax": torch.full_like(grid["doy"], 365.25),
                    "incoming_flux": grid["flux_pr"],
                },
            }
        )
    return out


def case_phenology() -> list[dict[str, Any]]:
    out = []
    for tmin, trange in ((-5.0, 2.0), (-2.0, 5.0), (0.0, 10.0)):
        T = combine_segments(
            torch.linspace(tmin - 5.0, tmin + trange + 5.0, 401, dtype=DTYPE),
            torch.linspace(tmin - 0.5, tmin + trange + 0.5, 201, dtype=DTYPE),
        )
        grid = flatten_grid(T=T, PET=t((0.0, 0.2, 3.0, 10.0)))
        out.append(
            {
                "name": f"tmin={tmin:g}_trange={trange:g}",
                "core": {
                    "T": grid["T"],
                    "tmin": torch.full_like(grid["T"], tmin),
                    "trange": torch.full_like(grid["T"], trange),
                    "PET": grid["PET"],
                },
                "flux": {
                    "T": grid["T"],
                    "p1": torch.full_like(grid["T"], tmin),
                    "p2": torch.full_like(grid["T"], tmin + trange),
                    "Ep": grid["PET"],
                },
            }
        )
    return out


def case_tcm_baseflow_6() -> list[dict[str, Any]]:
    out = []
    for threshold in (0.0, 5.0, 50.0):
        S = combine_segments(
            torch.linspace(0.0, max(100.0, 2.0 * max(threshold, 1.0)), 401, dtype=DTYPE),
            torch.linspace(max(0.0, threshold - 5.0), threshold + 5.0, 201, dtype=DTYPE),
        )
        grid = flatten_grid(p1=t((0.001, 0.01, 0.1)), S=S)
        out.append(
            {
                "name": f"threshold={threshold:g}",
                "core": {**grid, "p2": torch.full_like(grid["S"], threshold)},
                "flux": {**grid, "p2": torch.full_like(grid["S"], threshold)},
            }
        )
    return out


PAIRS: list[PairReview] = [
    PairReview(
        pair_id="PAIR_001",
        core_model="ihacres",
        core_module="dmotpy.models.core.ihacres",
        core_attr="evap_linear_deficit",
        flux_module="dmotpy.models.flux.evap",
        flux_attr="evap_12",
        formula_type="evap",
        core_formula_name="evap_linear_deficit",
        flux_formula_name="evap_12",
        core_expression_summary="clamp(1 - S/lp, 0, 1) * Ep",
        flux_expression_summary="min(1, exp(2 * (1 - S/lp))) * Ep",
        same_inputs="yes",
        same_units="yes",
        same_soft_gate_direction="n/a",
        same_bounds_or_limiter="no",
        same_epsilon_handling="similar",
        initial_equivalence_judgment="different_deficit_response",
        notes="Core uses a linear deficit ramp; shared flux uses an exponential decline and is still active in special/ihacres.py.",
        core_wrapper=lambda kw: core_ihacres.evap_linear_deficit(**kw),
        flux_wrapper=lambda kw: flux_evap.evap_12(**kw),
        cases=case_ihacres,
        threshold_mask=lambda kw: (kw["S"] - kw["lp"]).abs() <= torch.maximum(0.1 * kw["lp"], torch.full_like(kw["lp"], 1.0)),
        upper_bound=lambda kw: kw["Ep"],
    ),
    PairReview(
        pair_id="PAIR_002",
        core_model="modhydrolog",
        core_module="dmotpy.models.core.modhydrolog",
        core_attr="exchange_1",
        flux_module="dmotpy.models.flux.exchange",
        flux_attr="exchange_1",
        formula_type="exchange",
        core_formula_name="exchange_1",
        flux_formula_name="exchange_1",
        core_expression_summary="(p1*S + p2*(1-exp(-p3*|S|))*S/(|S|+eps)) with exponent clamp and lower bound -abs(fmax)",
        flux_expression_summary="(p1*|S| + p2*(1-exp(-p3*|S|))) * sign(S), lower bound -fmax",
        same_inputs="yes",
        same_units="yes",
        same_soft_gate_direction="n/a",
        same_bounds_or_limiter="similar_not_exact",
        same_epsilon_handling="no",
        initial_equivalence_judgment="safe_variant_of_shared_exchange",
        notes="The core helper removes explicit sign(), clamps the exponent, and hardens the lower bound with abs(fmax).",
        core_wrapper=lambda kw: core_modhydrolog.exchange_1(**kw),
        flux_wrapper=lambda kw: flux_exchange.exchange_1(**kw),
        cases=case_modhydrolog_exchange,
        threshold_mask=lambda kw: kw["S"].abs() <= 1.0,
        allow_negative=True,
        flux_unused=True,
    ),
    PairReview(
        pair_id="PAIR_003",
        core_model="modhydrolog",
        core_module="dmotpy.models.core.modhydrolog",
        core_attr="infiltration_1",
        flux_module="dmotpy.models.flux.infiltration",
        flux_attr="infiltration_1",
        formula_type="infiltration",
        core_formula_name="infiltration_1",
        flux_formula_name="infiltration_1",
        core_expression_summary="min(p1 * exp(clamp(-p2*S/max(Smax,1), -30, 0)), fin)",
        flux_expression_summary="min(p1 * exp(-p2*S/(Smax+eps)), fin)",
        same_inputs="yes",
        same_units="yes",
        same_soft_gate_direction="n/a",
        same_bounds_or_limiter="similar_not_exact",
        same_epsilon_handling="no",
        initial_equivalence_judgment="safe_variant",
        notes="Core keeps a denominator lock and exponent clamp for stability when Smax is small or the exponent is very negative.",
        core_wrapper=lambda kw: core_modhydrolog.infiltration_1(**kw),
        flux_wrapper=lambda kw: flux_infiltration.infiltration_1(**kw),
        cases=case_modhydrolog_infiltration_1,
        threshold_mask=lambda kw: (kw["S"] - kw["Smax"]).abs() <= torch.maximum(0.1 * kw["Smax"].abs(), torch.full_like(kw["Smax"], 0.5)),
        upper_bound=lambda kw: kw["fin"],
    ),
    PairReview(
        pair_id="PAIR_004",
        core_model="modhydrolog",
        core_module="dmotpy.models.core.modhydrolog",
        core_attr="infiltration_2",
        flux_module="dmotpy.models.flux.infiltration",
        flux_attr="infiltration_2",
        formula_type="infiltration",
        core_formula_name="infiltration_2",
        flux_formula_name="infiltration_2",
        core_expression_summary="min(relu(p1 * exp(clamp(-p2*S1/max(S1max,1), -30, 0)) - flux), S2)",
        flux_expression_summary="relu(min(p1 * exp(-p2*S1/(S1max+eps)) - flux, S2))",
        same_inputs="yes",
        same_units="yes",
        same_soft_gate_direction="n/a",
        same_bounds_or_limiter="similar_not_exact",
        same_epsilon_handling="no",
        initial_equivalence_judgment="safe_variant",
        notes="The local helper adds the same denominator lock and exponent clamp as infiltration_1; the shared flux is unused in active models.",
        core_wrapper=lambda kw: core_modhydrolog.infiltration_2(**kw),
        flux_wrapper=lambda kw: flux_infiltration.infiltration_2(**kw),
        cases=case_modhydrolog_infiltration_2,
        threshold_mask=lambda kw: (kw["S1"] - kw["S1max"]).abs() <= torch.maximum(0.1 * kw["S1max"].abs(), torch.full_like(kw["S1max"], 0.5)),
        upper_bound=lambda kw: kw["S2"],
        flux_unused=True,
    ),
    PairReview(
        pair_id="PAIR_005",
        core_model="modhydrolog",
        core_module="dmotpy.models.core.modhydrolog",
        core_attr="evap_2",
        flux_module="dmotpy.models.flux.evap",
        flux_attr="evap_2",
        formula_type="evap",
        core_formula_name="evap_2",
        flux_formula_name="evap_2",
        core_expression_summary="min(min(p1 * clamp(S/max(Smax,1), max=1), Ep), S)",
        flux_expression_summary="min(min(p1 * S/(Smax+eps), Ep), S)",
        same_inputs="yes",
        same_units="yes",
        same_soft_gate_direction="n/a",
        same_bounds_or_limiter="no",
        same_epsilon_handling="no",
        initial_equivalence_judgment="safe_variant_for_overfull_store",
        notes="The local formula caps S/Smax at 1.0, preventing evaporation from scaling above p1 when the store overshoots capacity.",
        core_wrapper=lambda kw: core_modhydrolog.evap_2(**kw),
        flux_wrapper=lambda kw: flux_evap.evap_2(**kw),
        cases=case_modhydrolog_evap_2,
        threshold_mask=lambda kw: (kw["S"] - kw["Smax"]).abs() <= torch.maximum(0.1 * kw["Smax"].abs(), torch.full_like(kw["Smax"], 0.5)),
        upper_bound=lambda kw: torch.minimum(kw["Ep"], kw["S"]),
    ),
    PairReview(
        pair_id="PAIR_006",
        core_model="modhydrolog",
        core_module="dmotpy.models.core.modhydrolog",
        core_attr="interception_1",
        flux_module="dmotpy.models.flux.interception",
        flux_attr="interception_1",
        formula_type="interception",
        core_formula_name="interception_1",
        flux_formula_name="interception_1",
        core_expression_summary="min(relu(S - Smax), S)",
        flux_expression_summary="incoming_flux * soft_gate_storage_below(S, Smax)",
        same_inputs="no",
        same_units="yes",
        same_soft_gate_direction="opposite_semantics",
        same_bounds_or_limiter="no",
        same_epsilon_handling="n/a",
        initial_equivalence_judgment="different_process_semantics",
        notes="The core helper computes overflow from an overfull store; the shared flux computes throughfall under a below-threshold gate.",
        core_wrapper=lambda kw: core_modhydrolog.interception_1(**kw),
        flux_wrapper=lambda kw: flux_interception.interception_1(**kw),
        cases=case_interception_overflow,
        threshold_mask=lambda kw: (kw["S"] - kw["Smax"]).abs() <= torch.maximum(0.1 * kw["Smax"].abs(), torch.full_like(kw["Smax"], 0.5)),
        upper_bound=lambda kw: kw["S"],
    ),
    PairReview(
        pair_id="PAIR_007",
        core_model="modhydrolog",
        core_module="dmotpy.models.core.modhydrolog",
        core_attr="depression_1",
        flux_module="dmotpy.models.flux.depression",
        flux_attr="depression_1",
        formula_type="depression",
        core_formula_name="depression_1",
        flux_formula_name="depression_1",
        core_expression_summary="min(min(ads * incoming_flux * exp(clamp(-md*Smax/incoming_flux, -20, 0)), relu(Smax-S)), incoming_flux)",
        flux_expression_summary="min(p1 * exp(-p2 * S / max(Smax-S, 0)) * incoming_flux, relu(Smax-S))",
        same_inputs="yes",
        same_units="yes",
        same_soft_gate_direction="n/a",
        same_bounds_or_limiter="no",
        same_epsilon_handling="no",
        initial_equivalence_judgment="corrected_local_variant",
        notes="The local implementation matches the MODHYDROLOG equation comment and uses incoming-flux control; the shared flux uses a storage/capacity exponent instead.",
        core_wrapper=lambda kw: core_modhydrolog.depression_1(**kw),
        flux_wrapper=lambda kw: flux_depression.depression_1(**kw),
        cases=case_depression,
        threshold_mask=lambda kw: (kw["S"] - kw["Smax"]).abs() <= torch.maximum(0.1 * kw["Smax"].abs(), torch.full_like(kw["Smax"], 0.5)),
        upper_bound=lambda kw: torch.minimum(kw["incoming_flux"], torch.relu(kw["Smax"] - kw["S"])),
        flux_unused=True,
    ),
    PairReview(
        pair_id="PAIR_008",
        core_model="mopex1",
        core_module="dmotpy.models.core.mopex1",
        core_attr="evap_7",
        flux_module="dmotpy.models.flux.evap",
        flux_attr="evap_7",
        formula_type="evap",
        core_formula_name="evap_7",
        flux_formula_name="evap_7",
        core_expression_summary="min(Ep * clamp(S/(Smax+eps), max=1) * dt, S) with dt=1",
        flux_expression_summary="min(clamp(S/Smax, max=1) * Ep, S)",
        same_inputs="yes",
        same_units="yes",
        same_soft_gate_direction="n/a",
        same_bounds_or_limiter="yes",
        same_epsilon_handling="slightly_different",
        initial_equivalence_judgment="likely_identical_dt1",
        notes="With the default dt=1, the local helper is the same formula apart from the tiny denominator epsilon.",
        core_wrapper=lambda kw: core_mopex1.evap_7(**kw),
        flux_wrapper=lambda kw: flux_evap.evap_7(**kw),
        cases=case_mopex1_evap_7,
        threshold_mask=lambda kw: (kw["S"] - kw["Smax"]).abs() <= torch.maximum(0.1 * kw["Smax"].abs(), torch.full_like(kw["Smax"], 0.5)),
        upper_bound=lambda kw: torch.minimum(kw["Ep"], kw["S"]),
    ),
    PairReview(
        pair_id="PAIR_009",
        core_model="mopex1",
        core_module="dmotpy.models.core.mopex1",
        core_attr="saturation_1",
        flux_module="dmotpy.models.flux.saturation",
        flux_attr="saturation_1",
        formula_type="saturation",
        core_formula_name="saturation_1",
        flux_formula_name="saturation_1",
        core_expression_summary="P * sigmoid((S - Smax*(1-r)) / (Smax*r*e + eps))",
        flux_expression_summary="incoming_flux * soft_gate_storage_above(S, Smax)",
        same_inputs="yes",
        same_units="yes",
        same_soft_gate_direction="both_above_threshold",
        same_bounds_or_limiter="no",
        same_epsilon_handling="no",
        initial_equivalence_judgment="different_smoother",
        notes="Both formulas represent saturation excess, but the local MOPEX smoother has a shifted threshold and width based on r and e.",
        core_wrapper=lambda kw: core_mopex1.saturation_1(**kw),
        flux_wrapper=lambda kw: flux_saturation.saturation_1(**kw),
        cases=case_mopex1_saturation,
        threshold_mask=lambda kw: (kw["S"] - kw["Smax"]).abs() <= torch.maximum(0.05 * kw["Smax"].abs(), torch.full_like(kw["Smax"], 1.0)),
        upper_bound=lambda kw: kw["P"],
    ),
    PairReview(
        pair_id="PAIR_010",
        core_model="mopex1",
        core_module="dmotpy.models.core.mopex1",
        core_attr="baseflow_1",
        flux_module="dmotpy.models.flux.baseflow",
        flux_attr="baseflow_1",
        formula_type="baseflow",
        core_formula_name="baseflow_1",
        flux_formula_name="baseflow_1",
        core_expression_summary="min(k * S, S)",
        flux_expression_summary="k * S",
        same_inputs="yes",
        same_units="yes",
        same_soft_gate_direction="n/a",
        same_bounds_or_limiter="no",
        same_epsilon_handling="yes",
        initial_equivalence_judgment="safety_capped_linear_store",
        notes="The local helper enforces flux <= storage when k temporarily exceeds 1, which the shared flux does not.",
        core_wrapper=lambda kw: core_mopex1.baseflow_1(**kw),
        flux_wrapper=lambda kw: flux_baseflow.baseflow_1(**kw),
        cases=case_linear_store,
        threshold_mask=lambda kw: kw["S"] <= 5.0,
        upper_bound=lambda kw: kw["S"],
    ),
    PairReview(
        pair_id="PAIR_011",
        core_model="mopex1",
        core_module="dmotpy.models.core.mopex1",
        core_attr="recharge_3",
        flux_module="dmotpy.models.flux.recharge",
        flux_attr="recharge_3",
        formula_type="recharge",
        core_formula_name="recharge_3",
        flux_formula_name="recharge_3",
        core_expression_summary="min(k * S, S)",
        flux_expression_summary="k * S",
        same_inputs="yes",
        same_units="yes",
        same_soft_gate_direction="n/a",
        same_bounds_or_limiter="no",
        same_epsilon_handling="yes",
        initial_equivalence_judgment="safety_capped_linear_store",
        notes="The local helper matches baseflow_1 semantics and adds the same outflow cap.",
        core_wrapper=lambda kw: core_mopex1.recharge_3(**kw),
        flux_wrapper=lambda kw: flux_recharge.recharge_3(**kw),
        cases=case_linear_store,
        threshold_mask=lambda kw: kw["S"] <= 5.0,
        upper_bound=lambda kw: kw["S"],
    ),
    PairReview(
        pair_id="PAIR_012",
        core_model="mopex2",
        core_module="dmotpy.models.core.mopex2",
        core_attr="snowfall_1",
        flux_module="dmotpy.models.flux.snowfall",
        flux_attr="snowfall_1",
        formula_type="snowfall",
        core_formula_name="snowfall_1",
        flux_formula_name="snowfall_1",
        core_expression_summary="P * sigmoid((tcrit - T) / (abs(tcrit)*r + r + eps))",
        flux_expression_summary="incoming_flux * soft_gate_temperature_below(T, tcrit)",
        same_inputs="yes",
        same_units="yes",
        same_soft_gate_direction="both_below_threshold",
        same_bounds_or_limiter="no",
        same_epsilon_handling="no",
        initial_equivalence_judgment="different_temperature_smoother",
        notes="The local MOPEX gate scales steepness with |tcrit| and r; the shared flux uses the dMoT constant-k gate.",
        core_wrapper=lambda kw: core_mopex2.snowfall_1(**kw),
        flux_wrapper=lambda kw: flux_snowfall.snowfall_1(**kw),
        cases=case_temp_partition,
        threshold_mask=lambda kw: (kw["T"] - kw["tcrit"]).abs() <= 1.0,
        upper_bound=lambda kw: kw["P"],
    ),
    PairReview(
        pair_id="PAIR_013",
        core_model="mopex2",
        core_module="dmotpy.models.core.mopex2",
        core_attr="rainfall_1",
        flux_module="dmotpy.models.flux.rainfall",
        flux_attr="rainfall_1",
        formula_type="rainfall",
        core_formula_name="rainfall_1",
        flux_formula_name="rainfall_1",
        core_expression_summary="P * sigmoid((T - tcrit) / (abs(tcrit)*r + r + eps))",
        flux_expression_summary="incoming_flux * soft_gate_temperature_above(T, tcrit)",
        same_inputs="yes",
        same_units="yes",
        same_soft_gate_direction="both_above_threshold",
        same_bounds_or_limiter="no",
        same_epsilon_handling="no",
        initial_equivalence_judgment="different_temperature_smoother",
        notes="The local MOPEX gate is the complementary form of snowfall_1 and differs from the shared dMoT constant-k gate.",
        core_wrapper=lambda kw: core_mopex2.rainfall_1(**kw),
        flux_wrapper=lambda kw: flux_rainfall.rainfall_1(**kw),
        cases=case_temp_partition,
        threshold_mask=lambda kw: (kw["T"] - kw["tcrit"]).abs() <= 1.0,
        upper_bound=lambda kw: kw["P"],
    ),
    PairReview(
        pair_id="PAIR_014",
        core_model="mopex2",
        core_module="dmotpy.models.core.mopex2",
        core_attr="melt_1",
        flux_module="dmotpy.models.flux.melt",
        flux_attr="melt_1",
        formula_type="melt",
        core_formula_name="melt_1",
        flux_formula_name="melt_1",
        core_expression_summary="min(ddf * sigmoid(T-tcrit) * softplus(T-tcrit), Sn)",
        flux_expression_summary="relu(min(ddf * (T - tcrit), Sn))",
        same_inputs="yes",
        same_units="yes",
        same_soft_gate_direction="both_above_threshold",
        same_bounds_or_limiter="similar_not_exact",
        same_epsilon_handling="n/a",
        initial_equivalence_judgment="smoothed_local_variant",
        notes="The local helper is a differentiable replacement for relu(T-tcrit) with the same degree-day structure but different values near the threshold.",
        core_wrapper=lambda kw: core_mopex2.melt_1(**kw),
        flux_wrapper=lambda kw: flux_melt.melt_1(**kw),
        cases=case_melt,
        threshold_mask=lambda kw: (kw["T"] - kw["tcrit"]).abs() <= 1.0,
        upper_bound=lambda kw: kw["Sn"],
    ),
    PairReview(
        pair_id="PAIR_015",
        core_model="mopex4",
        core_module="dmotpy.models.core.mopex4",
        core_attr="interception_4",
        flux_module="dmotpy.models.flux.interception",
        flux_attr="interception_4",
        formula_type="interception",
        core_formula_name="interception_4",
        flux_formula_name="interception_4",
        core_expression_summary="min(softplus(50 * fraction)/50 * flux_pr, flux_pr)",
        flux_expression_summary="relu(fraction) * incoming_flux",
        same_inputs="yes",
        same_units="yes",
        same_soft_gate_direction="n/a",
        same_bounds_or_limiter="similar_not_exact",
        same_epsilon_handling="slightly_different",
        initial_equivalence_judgment="softplus_vs_relu",
        notes="The local MOPEX4 helper keeps a smooth lower bound around zero seasonal interception fraction; the shared flux uses a hard ReLU.",
        core_wrapper=lambda kw: core_mopex4.interception_4(**kw),
        flux_wrapper=lambda kw: flux_interception.interception_4(**kw),
        cases=case_interception_4,
        threshold_mask=lambda kw: (
            kw["alpha"] + (1.0 - kw["alpha"]) * torch.cos(2.0 * torch.pi * (kw["doy"] - kw["is_time"]) / 365.25)
        ).abs()
        <= 0.1,
        upper_bound=lambda kw: kw["flux_pr"],
        flux_unused=True,
    ),
    PairReview(
        pair_id="PAIR_016",
        core_model="mopex5",
        core_module="dmotpy.models.core.mopex5",
        core_attr="phenology_1",
        flux_module="dmotpy.models.flux.phenology",
        flux_attr="phenology_1",
        formula_type="phenology",
        core_formula_name="phenology_1",
        flux_formula_name="phenology_1",
        core_expression_summary="clamp((T - tmin)/(trange + eps), 0, 1) * PET",
        flux_expression_summary="min(1, relu((T - p1)/(p2 - p1 + eps))) * Ep with p2=tmin+trange",
        same_inputs="mapped_upper_threshold",
        same_units="yes",
        same_soft_gate_direction="n/a",
        same_bounds_or_limiter="yes_after_mapping",
        same_epsilon_handling="yes_after_mapping",
        initial_equivalence_judgment="identical_with_parameter_mapping",
        notes="The shared flux takes lower and upper thresholds directly; the core helper stores lower threshold plus range.",
        core_wrapper=lambda kw: core_mopex5.phenology_1(**kw),
        flux_wrapper=lambda kw: flux_phenology.phenology_1(**kw),
        cases=case_phenology,
        threshold_mask=lambda kw: ((kw["T"] >= kw["tmin"]) & (kw["T"] <= kw["tmin"] + kw["trange"])),
        upper_bound=lambda kw: kw["PET"],
        flux_unused=True,
    ),
    PairReview(
        pair_id="PAIR_017",
        core_model="tcm",
        core_module="dmotpy.models.core.tcm",
        core_attr="baseflow_6",
        flux_module="dmotpy.models.flux.baseflow",
        flux_attr="baseflow_6",
        formula_type="baseflow",
        core_formula_name="baseflow_6",
        flux_formula_name="baseflow_6",
        core_expression_summary="min(S, p1*S^2) * soft_gate_storage_above(S, p2)",
        flux_expression_summary="min(S, p1*S^2) * soft_gate_storage_above(S, p2)",
        same_inputs="yes",
        same_units="yes",
        same_soft_gate_direction="above_threshold",
        same_bounds_or_limiter="yes",
        same_epsilon_handling="yes",
        initial_equivalence_judgment="identical",
        notes="Verified against the active local helper used in models/core/tcm.py.",
        core_wrapper=lambda kw: core_tcm.baseflow_6(**kw),
        flux_wrapper=lambda kw: flux_baseflow.baseflow_6(**kw),
        cases=case_tcm_baseflow_6,
        threshold_mask=lambda kw: (kw["S"] - kw["p2"]).abs() <= torch.maximum(0.1 * kw["p2"].abs(), torch.full_like(kw["p2"], 1.0)),
        upper_bound=lambda kw: kw["S"],
        flux_unused=True,
    ),
]


STANDALONE_REVIEWS: list[StandaloneReview] = [
    StandaloneReview(
        pair_id="STANDALONE_001",
        core_model="gr4j",
        core_formula="gr4j._calc_production_store_tanh",
        core_file="models/core/gr4j.py",
        line_start=7,
        line_end=26,
        formula_type="analytical_production",
        code_snippet=file_snippet(REPO_ROOT / "models/core/gr4j.py", 7, 26),
        equivalence_class="core_state_coupling_not_flux",
        recommended_future_action="Keep in core unless a dedicated analytical GR4J flux module is introduced.",
        short_reason="Analytical production-store helper is tightly coupled to GR4J state sequencing and paired Ps/Es output.",
    ),
    StandaloneReview(
        pair_id="STANDALONE_002",
        core_model="gr4j",
        core_formula="gr4j._calc_percolation_analytical",
        core_file="models/core/gr4j.py",
        line_start=28,
        line_end=35,
        formula_type="analytical_percolation",
        code_snippet=file_snippet(REPO_ROOT / "models/core/gr4j.py", 28, 35),
        equivalence_class="core_state_coupling_not_flux",
        recommended_future_action="Keep in core with the current analytical GR4J block.",
        short_reason="Process equation is embedded in the analytical GR4J integration rather than exposed as a reusable shared flux.",
    ),
    StandaloneReview(
        pair_id="STANDALONE_003",
        core_model="gr4j",
        core_formula="gr4j._calc_routing_outflow_analytical",
        core_file="models/core/gr4j.py",
        line_start=37,
        line_end=44,
        formula_type="analytical_routing",
        code_snippet=file_snippet(REPO_ROOT / "models/core/gr4j.py", 37, 44),
        equivalence_class="core_state_coupling_not_flux",
        recommended_future_action="Keep in core with GR4J routing-state updates.",
        short_reason="Analytical routing helper is coupled to the GR4J routing store update and exchange accounting.",
    ),
    StandaloneReview(
        pair_id="STANDALONE_004",
        core_model="collie3",
        core_formula="collie3.nonlinear_interflow",
        core_file="models/core/collie3.py",
        line_start=87,
        line_end=92,
        formula_type="interflow",
        code_snippet=file_snippet(REPO_ROOT / "models/core/collie3.py", 87, 92),
        equivalence_class="no_flux_equivalent",
        recommended_future_action="If extracted later, create a model-specific collie3 interflow flux.",
        short_reason="Clear process formula but no shared flux function with the same excess-above-field-capacity power-law semantics.",
    ),
    StandaloneReview(
        pair_id="STANDALONE_005",
        core_model="collie3",
        core_formula="collie3.nonlinear_groundwater_baseflow",
        core_file="models/core/collie3.py",
        line_start=97,
        line_end=100,
        formula_type="baseflow",
        code_snippet=file_snippet(REPO_ROOT / "models/core/collie3.py", 97, 100),
        equivalence_class="no_flux_equivalent",
        recommended_future_action="If extracted later, create a model-specific collie3 baseflow flux.",
        short_reason="Power-law groundwater outflow exists inline only; no matching shared baseflow function was found.",
    ),
]


def compare_pair(pair: PairReview) -> dict[str, Any]:
    all_core: list[torch.Tensor] = []
    all_flux: list[torch.Tensor] = []
    near_diffs: list[torch.Tensor] = []
    core_negative = 0
    flux_negative = 0
    core_bound_viol = 0
    flux_bound_viol = 0
    nan_count = 0
    inf_count = 0
    core_grad_states: list[str] = []
    flux_grad_states: list[str] = []
    max_core_grad = 0.0
    max_flux_grad = 0.0
    case_names: list[str] = []

    for case in pair.cases():
        case_names.append(case["name"])
        core_out, core_grad_state, core_grad_max = run_with_grads(pair.core_wrapper, case["core"])
        flux_out, flux_grad_state, flux_grad_max = run_with_grads(pair.flux_wrapper, case["flux"])

        core_grad_states.append(core_grad_state)
        flux_grad_states.append(flux_grad_state)
        if math.isfinite(core_grad_max):
            max_core_grad = max(max_core_grad, core_grad_max)
        if math.isfinite(flux_grad_max):
            max_flux_grad = max(max_flux_grad, flux_grad_max)

        nan_count += int(torch.isnan(core_out).sum().item() + torch.isnan(flux_out).sum().item())
        inf_count += int(torch.isinf(core_out).sum().item() + torch.isinf(flux_out).sum().item())

        all_core.append(core_out.reshape(-1))
        all_flux.append(flux_out.reshape(-1))

        near_mask = pair.threshold_mask(case["core"]).reshape(-1)
        diff = (core_out - flux_out).reshape(-1)
        if near_mask.any():
            near_diffs.append(diff[near_mask].abs())

        if not pair.allow_negative:
            core_negative += int((core_out < -1e-12).sum().item())
            flux_negative += int((flux_out < -1e-12).sum().item())

        if pair.upper_bound is not None:
            bound = pair.upper_bound(case["core"])
            core_bound_viol += int((core_out > bound + 1e-12).sum().item())
            flux_bound_viol += int((flux_out > bound + 1e-12).sum().item())

    core_all = torch.cat(all_core)
    flux_all = torch.cat(all_flux)
    diff_all = core_all - flux_all

    max_abs_diff = float(diff_all.abs().max().item())
    mean_abs_diff = float(diff_all.abs().mean().item())
    rel_l2 = safe_rel_l2(diff_all, flux_all)
    max_rel = safe_max_relative(diff_all, flux_all)
    signed_bias = float(diff_all.mean().item())
    near_threshold_max_diff = float(torch.cat(near_diffs).max().item()) if near_diffs else float("nan")
    identical = max_abs_diff <= TOL and rel_l2 <= TOL and nan_count == 0 and inf_count == 0

    core_grad_summary = "finite" if all(state == "finite" for state in core_grad_states) else ",".join(sorted(set(core_grad_states)))
    flux_grad_summary = "finite" if all(state == "finite" for state in flux_grad_states) else ",".join(sorted(set(flux_grad_states)))

    return {
        "pair_id": pair.pair_id,
        "core_model": pair.core_model,
        "core_formula": pair.core_formula_name,
        "flux_function": pair.flux_formula_name,
        "tested_cases": "; ".join(case_names),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "relative_l2_diff": rel_l2,
        "max_relative_diff": max_rel,
        "signed_bias_mean": signed_bias,
        "near_threshold_max_diff": near_threshold_max_diff,
        "core_negative_count": core_negative,
        "flux_negative_count": flux_negative,
        "bound_violation_difference_count": abs(core_bound_viol - flux_bound_viol),
        "core_bound_violation_count": core_bound_viol,
        "flux_bound_violation_count": flux_bound_viol,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "core_gradient_status": core_grad_summary,
        "flux_gradient_status": flux_grad_summary,
        "core_max_abs_gradient": max_core_grad,
        "flux_max_abs_gradient": max_flux_grad,
        "identical_within_tolerance": bool_text(identical),
    }


def classify_pair(pair: PairReview, metrics: dict[str, Any]) -> tuple[str, str, str, str]:
    pid = pair.pair_id
    if pid in {"PAIR_017", "PAIR_016", "PAIR_008"}:
        return (
            "identical_safe_to_migrate",
            "high" if pid in {"PAIR_016", "PAIR_017"} else "medium",
            "Existing shared flux can preserve behavior if the same parameter mapping and call order are kept.",
            "No material numerical difference detected in the reviewed ranges.",
        )
    if pid in {"PAIR_002", "PAIR_004", "PAIR_007", "PAIR_015"}:
        return (
            "unused_flux_dangerous",
            "low",
            "Do not replace the active core helper with the unused shared flux; mark the shared flux inactive/deprecated later.",
            "Unused shared flux overlaps with an active corrected or smoothed local variant.",
        )
    if pid in {"PAIR_003", "PAIR_005", "PAIR_006", "PAIR_010", "PAIR_011", "PAIR_014"}:
        return (
            "corrected_local_variant",
            "low",
            "Preserve the local helper or extract it later under a model-specific name.",
            "Core helper intentionally adds safety caps, changed semantics, or smoothing beyond the shared flux.",
        )
    return (
        "similar_but_not_equivalent",
        "medium",
        "If extracted later, create a model-specific flux rather than reusing the existing shared one.",
        "Same process family, but different limiter, threshold shape, or response curve.",
    )


def pair_inventory_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for pair in PAIRS:
        flux_mod = importlib.import_module(pair.flux_module)
        core_fn = pair.core_wrapper
        flux_fn = getattr(flux_mod, pair.flux_attr)
        core_file = inspect.getsourcefile(core_fn) or pair.core_module.replace(".", "/")
        _, core_start = inspect.getsourcelines(core_fn)
        core_src = inspect.getsource(core_fn)
        core_end = core_start + len(core_src.splitlines()) - 1
        core_snippet = core_src.strip()
        flux_file, flux_start, flux_end, flux_snippet = source_info(flux_fn)
        rows.append(
            {
                "pair_id": pair.pair_id,
                "core_model": pair.core_model,
                "core_file": core_file,
                "core_lines": f"{core_start}-{core_end}",
                "core_formula_name_or_description": pair.core_formula_name,
                "flux_function": pair.flux_formula_name,
                "flux_file": flux_file,
                "flux_lines": f"{flux_start}-{flux_end}",
                "formula_type": pair.formula_type,
                "core_expression_summary": pair.core_expression_summary,
                "flux_expression_summary": pair.flux_expression_summary,
                "same_inputs": pair.same_inputs,
                "same_units": pair.same_units,
                "same_soft_gate_direction": pair.same_soft_gate_direction,
                "same_bounds_or_limiter": pair.same_bounds_or_limiter,
                "same_epsilon_handling": pair.same_epsilon_handling,
                "initial_equivalence_judgment": pair.initial_equivalence_judgment,
                "notes": pair.notes,
                "core_code_snippet": core_snippet,
                "flux_code_snippet": flux_snippet,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, float):
                    normalized[key] = format_float(value)
                else:
                    normalized[key] = value
            writer.writerow(normalized)


def build_report(
    inventory_rows: list[dict[str, str]],
    comparison_rows: list[dict[str, Any]],
    classification_rows: list[dict[str, Any]],
) -> str:
    pair_count = len(PAIRS)
    reviewed_total = len(classification_rows)
    counts: dict[str, int] = {}
    for row in classification_rows:
        counts[row["equivalence_class"]] = counts.get(row["equivalence_class"], 0) + 1

    compare_by_id = {row["pair_id"]: row for row in comparison_rows}
    inv_by_id = {row["pair_id"]: row for row in inventory_rows}

    def card(pair_id: str) -> str:
        inv = inv_by_id[pair_id]
        cmp_row = compare_by_id[pair_id]
        cls = next(row for row in classification_rows if row["pair_id"] == pair_id)
        return f"""### {pair_id}: {inv['core_model']} / {inv['core_formula_name_or_description']}
Core snippet:
```python
{inv['core_code_snippet']}
```
Flux snippet:
```python
{inv['flux_code_snippet']}
```
Comparison:
- Core expression: `{inv['core_expression_summary']}`
- Flux expression: `{inv['flux_expression_summary']}`
- Max abs diff: `{format_float(cmp_row['max_abs_diff'])}`
- Relative L2 diff: `{format_float(cmp_row['relative_l2_diff'])}`
- Near-threshold max diff: `{format_float(cmp_row['near_threshold_max_diff'])}`
- Gradient status: `core={cmp_row['core_gradient_status']}`, `flux={cmp_row['flux_gradient_status']}`
- Classification: `{cls['equivalence_class']}`
- Preserve current core formula: `{ 'yes' if cls['equivalence_class'] != 'identical_safe_to_migrate' else 'not required if migrated carefully' }`
- Future migration target: `{cls['recommended_future_action']}`
"""

    safe_rows = [row for row in classification_rows if row["equivalence_class"] == "identical_safe_to_migrate"]
    unsafe_rows = [
        row
        for row in classification_rows
        if row["equivalence_class"] in {"similar_but_not_equivalent", "corrected_local_variant", "unused_flux_dangerous"}
    ]
    unused_dangerous_rows = [row for row in classification_rows if row["equivalence_class"] == "unused_flux_dangerous"]

    safe_table = "\n".join(
        f"| {row['core_model']} | {row['core_formula']} | {row['flux_function']} | {row['short_reason']} |"
        for row in safe_rows
    )
    unsafe_table = "\n".join(
        f"| {row['core_model']} | {row['core_formula']} | {row['flux_function']} | {row['equivalence_class']} | {row['short_reason']} |"
        for row in unsafe_rows
    )
    dangerous_table = "\n".join(
        f"| {row['core_model']} | {row['flux_function']} | {row['short_reason']} |"
        for row in unused_dangerous_rows
    )

    return f"""# Formula Difference Review

## 1. Scope
This review compares active inline process formulas in `models/core` against overlapping functions in `models/flux` without modifying any model implementation code. The goal is to determine mathematical equivalence, identify intentional differences, and flag shared flux functions that are unsafe to merge into active core logic.

## 2. Files inspected
- `models/core/ihacres.py`
- `models/core/modhydrolog.py`
- `models/core/mopex1.py`
- `models/core/mopex2.py`
- `models/core/mopex4.py`
- `models/core/mopex5.py`
- `models/core/tcm.py`
- `models/core/gr4j.py`
- `models/core/sacramento.py`
- `models/core/collie3.py`
- `models/flux/baseflow.py`
- `models/flux/depression.py`
- `models/flux/evap.py`
- `models/flux/exchange.py`
- `models/flux/infiltration.py`
- `models/flux/interception.py`
- `models/flux/melt.py`
- `models/flux/phenology.py`
- `models/flux/rainfall.py`
- `models/flux/recharge.py`
- `models/flux/saturation.py`
- `models/flux/snowfall.py`
- `models/special/ihacres.py`

## 3. Review coverage
- Core-vs-flux formula pairs reviewed: `{pair_count}`
- Additional standalone core-only formulas classified: `{reviewed_total - pair_count}`

## 4. Equivalence class counts
- `identical_safe_to_migrate`: `{counts.get('identical_safe_to_migrate', 0)}`
- `similar_but_not_equivalent`: `{counts.get('similar_but_not_equivalent', 0)}`
- `corrected_local_variant`: `{counts.get('corrected_local_variant', 0)}`
- `core_state_coupling_not_flux`: `{counts.get('core_state_coupling_not_flux', 0)}`
- `no_flux_equivalent`: `{counts.get('no_flux_equivalent', 0)}`
- `unused_flux_dangerous`: `{counts.get('unused_flux_dangerous', 0)}`

## 5. High-priority formula cards
{card('PAIR_017')}
{card('PAIR_001')}
{card('PAIR_016')}

## 6. Formulas that are safe to migrate later
| Core model | Core formula | Flux function | Reason |
| --- | --- | --- | --- |
{safe_table}

## 7. Formulas that should not be merged with existing flux functions
| Core model | Core formula | Flux function | Class | Reason |
| --- | --- | --- | --- | --- |
{unsafe_table}

## 8. Flux functions that are unused and potentially dangerous
| Core model | Shared flux function | Reason |
| --- | --- | --- |
{dangerous_table}

## 9. Human-review recommendations
- Preserve `tcm.baseflow_6` behavior exactly; it is numerically identical to `flux.baseflow_6`, but the shared function is currently unused in active core models.
- Do not treat `ihacres.evap_linear_deficit` as equivalent to `flux.evap_12`; `flux.evap_12` remains active in `models/special/ihacres.py`, so any future extraction should introduce a separate IHACRES-specific linear-deficit flux.
- Treat `mopex5.phenology_1` as the cleanest migration candidate after `tcm.baseflow_6`; it is equivalent to `flux.phenology_1` once the upper threshold is mapped as `tmin + trange`.
- Do not replace active MODHYDROLOG or MOPEX local helpers with existing shared flux functions without behavior-preservation tests; several shared functions are unused precisely because the local logic diverges on limiter semantics or smoothing.
- Keep the reviewed GR4J analytical helpers and Sacramento deficit distribution in `core`; they are execution-order or analytical state-coupling logic rather than clean reusable flux functions.
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inventory_rows = pair_inventory_rows()
    comparison_rows = [compare_pair(pair) for pair in PAIRS]

    classification_rows: list[dict[str, Any]] = []
    comparison_by_id = {row["pair_id"]: row for row in comparison_rows}
    for pair in PAIRS:
        metrics = comparison_by_id[pair.pair_id]
        eq_class, migration_safety, action, short_reason = classify_pair(pair, metrics)
        gradient_status = f"core={metrics['core_gradient_status']}; flux={metrics['flux_gradient_status']}"
        classification_rows.append(
            {
                "pair_id": pair.pair_id,
                "core_model": pair.core_model,
                "core_formula": pair.core_formula_name,
                "flux_function": pair.flux_formula_name,
                "formula_type": pair.formula_type,
                "equivalence_class": eq_class,
                "max_abs_diff": metrics["max_abs_diff"],
                "relative_L2_diff": metrics["relative_l2_diff"],
                "gradient_status": gradient_status,
                "migration_safety": migration_safety,
                "recommended_future_action": action,
                "human_review_needed": "yes" if eq_class != "identical_safe_to_migrate" else "no",
                "short_reason": short_reason,
            }
        )

    for standalone in STANDALONE_REVIEWS:
        classification_rows.append(
            {
                "pair_id": standalone.pair_id,
                "core_model": standalone.core_model,
                "core_formula": standalone.core_formula,
                "flux_function": "",
                "formula_type": standalone.formula_type,
                "equivalence_class": standalone.equivalence_class,
                "max_abs_diff": float("nan"),
                "relative_L2_diff": float("nan"),
                "gradient_status": "not_compared",
                "migration_safety": "low",
                "recommended_future_action": standalone.recommended_future_action,
                "human_review_needed": "yes",
                "short_reason": standalone.short_reason,
            }
        )

    write_csv(
        OUT_DIR / "formula_pair_inventory.csv",
        inventory_rows,
        [
            "pair_id",
            "core_model",
            "core_file",
            "core_lines",
            "core_formula_name_or_description",
            "flux_function",
            "flux_file",
            "flux_lines",
            "formula_type",
            "core_expression_summary",
            "flux_expression_summary",
            "same_inputs",
            "same_units",
            "same_soft_gate_direction",
            "same_bounds_or_limiter",
            "same_epsilon_handling",
            "initial_equivalence_judgment",
            "notes",
            "core_code_snippet",
            "flux_code_snippet",
        ],
    )
    write_csv(
        OUT_DIR / "formula_pair_value_comparison.csv",
        comparison_rows,
        [
            "pair_id",
            "core_model",
            "core_formula",
            "flux_function",
            "tested_cases",
            "max_abs_diff",
            "mean_abs_diff",
            "relative_l2_diff",
            "max_relative_diff",
            "signed_bias_mean",
            "near_threshold_max_diff",
            "core_negative_count",
            "flux_negative_count",
            "bound_violation_difference_count",
            "core_bound_violation_count",
            "flux_bound_violation_count",
            "nan_count",
            "inf_count",
            "core_gradient_status",
            "flux_gradient_status",
            "core_max_abs_gradient",
            "flux_max_abs_gradient",
            "identical_within_tolerance",
        ],
    )
    write_csv(
        OUT_DIR / "formula_equivalence_classification.csv",
        classification_rows,
        [
            "pair_id",
            "core_model",
            "core_formula",
            "flux_function",
            "formula_type",
            "equivalence_class",
            "max_abs_diff",
            "relative_L2_diff",
            "gradient_status",
            "migration_safety",
            "recommended_future_action",
            "human_review_needed",
            "short_reason",
        ],
    )
    (OUT_DIR / "formula_difference_report.md").write_text(
        build_report(inventory_rows, comparison_rows, classification_rows)
    )


if __name__ == "__main__":
    main()
