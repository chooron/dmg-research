#!/usr/bin/env python3
"""Read-only evidence collection for dPL gradient trainability.

This script deliberately does not import the dPL trainer, change model source,
or write checkpoints.  It runs the active ``src.model_registry.build_model``
path with real CAMELS forcing and records hard-operator branch occupancy through
``TorchDispatchMode``.  Run from the repository root, for example::

    python project/benchmark/scripts/diagnostics/dpl_gradient_evidence.py

The default output directory is date-stamped and contains only generated CSV,
JSON, and Markdown evidence.  ``--mask-epoch`` is intentionally opt-in because
it evaluates 169 full 730-day batches per model without taking optimizer steps.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import importlib
import inspect
import json
import pickle
import re
import sys
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch


SCRIPT = Path(__file__).resolve()
BENCHMARK_ROOT = SCRIPT.parents[2]
REPO_ROOT = BENCHMARK_ROOT.parents[1]
DATA_DIR = REPO_ROOT / "data"
CMA_ROOT = (
    BENCHMARK_ROOT
    / "dmotpy/experiments/cmaes_36models/downloads"
    / "full300_20260729_160112_partial_20260730/checkpoints_latest"
)
TARGET_MODELS = ("collie3", "newzealand1", "penman", "flexi", "flexis", "hbv96")
WARMUP_DAYS = 365
SCORED_DAYS = 365

sys.path[:0] = [str(BENCHMARK_ROOT), str(BENCHMARK_ROOT / "src")]

from dmotpy.models.endpoint_uh_model import ENDPOINT_UH_SCHEMES  # noqa: E402
from dmotpy.models.intermediate_uh_model import INTERMEDIATE_UH_CONFIG  # noqa: E402
from dmotpy.models.registry import PARAM_INFO, STATE_INFO  # noqa: E402
from src.data_selection import load_ids, select_pilot_basins  # noqa: E402
from src.model_registry import NPARAM_INFO_36, build_model, get_spec  # noqa: E402


def _csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _source_line(path: Path, line: int) -> str:
    lines = path.read_text().splitlines()
    return lines[line - 1].strip() if 0 < line <= len(lines) else "<line unavailable>"


def _relative(path: Path | str) -> str:
    return str(Path(path).resolve().relative_to(BENCHMARK_ROOT))


def _module_for_model(model: str):
    return importlib.import_module(f"dmotpy.models.core.{model}")


def _description_map(model: str) -> dict[str, str]:
    module = _module_for_model(model)
    matches = [value for name, value in vars(module).items() if name.endswith("_PARAMS_DESC")]
    return dict(matches[0]) if len(matches) == 1 else {}


def _statement_nodes(tree: ast.AST) -> list[ast.stmt]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.stmt)]


def _contains_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))


def _operator_tags(text: str) -> str:
    tags: list[str] = []
    patterns = {
        "torch.where": "torch.where",
        "torch.clamp": "torch.clamp",
        "clamp_min": "clamp_min",
        "clamp_max": "clamp_max",
        "torch.minimum": "torch.minimum",
        "torch.maximum": "torch.maximum",
        "torch.relu": "torch.relu",
        "F.relu": "F.relu",
        "F.softplus": "F.softplus",
        "torch.nan_to_num": "torch.nan_to_num",
        "nan_to_num": "nan_to_num",
        ".pow(": "torch.pow",
        " ** ": "fractional-power candidate",
    }
    for needle, tag in patterns.items():
        if needle in text:
            tags.append(tag)
    return "; ".join(dict.fromkeys(tags)) or "none"


def _symbolic_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for model in TARGET_MODELS:
        module = _module_for_model(model)
        path = Path(inspect.getsourcefile(module) or "")
        tree = ast.parse(path.read_text())
        statements = _statement_nodes(tree)
        for parameter in PARAM_INFO[model]:
            for node in statements:
                # An expression statement inside comments/docstrings cannot carry Name nodes.
                if not _contains_name(node, parameter):
                    continue
                # Keep executable statements only and avoid duplicating nested statement text.
                if not hasattr(node, "lineno") or isinstance(node, (ast.FunctionDef, ast.If, ast.For, ast.While)):
                    continue
                text = ast.get_source_segment(path.read_text(), node) or _source_line(path, node.lineno)
                text = " ".join(text.strip().split())
                if len(text) > 420:
                    text = text[:417] + "..."
                rows.append(
                    {
                        "model": model,
                        "parameter": parameter,
                        "file_line": f"{_relative(path)}:{node.lineno}",
                        "expression": text,
                        "blocking_operator_type": _operator_tags(text),
                        "can_yield_zero_gradient": "yes" if _operator_tags(text) != "none" else "not directly",
                        "note": "direct parameter occurrence in active core step/pre/post function",
                    }
                )
    return rows


# Flux bodies that receive generic p1/p2 names cannot be recovered from a core
# AST name scan.  These rows make the parameter-to-flux link explicit.
FLUX_DEPENDENCIES = {
    "collie3": {
        "a": [("dmotpy/models/flux/interflow.py", 30, "out = p1 * (S + nearzero).pow(p2)")],
        "b": [("dmotpy/models/flux/interflow.py", 30, "out = p1 * (S + nearzero).pow(p2)")],
        "lambda_par": [("dmotpy/models/flux/split.py", 10, "return p1 * incoming_flux")],
    },
    "newzealand1": {
        "a": [("dmotpy/models/flux/interflow.py", 107, "excess = F.relu(S - p2)"), ("dmotpy/models/flux/interflow.py", 108, "out = (p1 * excess + nearzero).pow(p3)"), ("dmotpy/models/flux/interflow.py", 109, "return torch.minimum(excess, out)")],
        "b": [("dmotpy/models/flux/interflow.py", 107, "excess = F.relu(S - p2)"), ("dmotpy/models/flux/interflow.py", 108, "out = (p1 * excess + nearzero).pow(p3)"), ("dmotpy/models/flux/interflow.py", 109, "return torch.minimum(excess, out)")],
    },
    "penman": {
        "gam": [("dmotpy/models/flux/evap.py", 258, "evap = p1 * Ep"), ("dmotpy/models/flux/evap.py", 259, "return torch.minimum(evap * smooth_threshold_storage_logistic(S2, S2min, nearzero=nearzero), S1)")],
    },
    "flexi": {
        "imax": [("dmotpy/models/flux/interception.py", 15, "sf = smooth_threshold_storage_logistic(S, Smax, nearzero=nearzero)"), ("dmotpy/models/flux/interception.py", 16, "return incoming_flux * sf")],
        "beta": [("dmotpy/models/flux/saturation.py", 36, "out_frac = torch.sigmoid((ratio + 0.5) / (p1 + nearzero))")],
        "percmax": [("dmotpy/models/flux/percolation.py", 20, "return torch.minimum(S, p1 * S / (Smax + nearzero))")],
    },
    "flexis": {
        "imax": [("dmotpy/models/flux/interception.py", 15, "sf = smooth_threshold_storage_logistic(S, Smax, nearzero=nearzero)"), ("dmotpy/models/flux/interception.py", 16, "return incoming_flux * sf")],
        "beta": [("dmotpy/models/flux/saturation.py", 36, "out_frac = torch.sigmoid((ratio + 0.5) / (p1 + nearzero))")],
        "percmax": [("dmotpy/models/flux/percolation.py", 20, "return torch.minimum(S, p1 * S / (Smax + nearzero))")],
    },
}


def _augment_flux_rows(rows: list[dict[str, str]]) -> None:
    for model, parameters in FLUX_DEPENDENCIES.items():
        for parameter, expressions in parameters.items():
            for rel_path, line, expression in expressions:
                tags = _operator_tags(expression)
                rows.append(
                    {
                        "model": model,
                        "parameter": parameter,
                        "file_line": f"{rel_path}:{line}",
                        "expression": expression,
                        "blocking_operator_type": tags,
                        "can_yield_zero_gradient": "yes" if tags != "none" else "depends on upstream flux",
                        "note": "linked generic flux parameter; source line verified from active flux module",
                    }
                )


RISK_TOKENS = ("nan_to_num", "detach()", ".data", "with torch.no_grad()", "requires_grad_(False)")


def _enclosing_functions(path: Path) -> list[tuple[int, int, str]]:
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    spans: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            spans.append((node.lineno, getattr(node, "end_lineno", node.lineno), node.name))
    return spans


def _static_risk_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in BENCHMARK_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            lines = path.read_text().splitlines()
        except UnicodeDecodeError:
            continue
        spans = _enclosing_functions(path)
        for line_no, line in enumerate(lines, start=1):
            for token in RISK_TOKENS:
                if token == ".data":
                    found = bool(re.search(r"(?<![A-Za-z0-9_])\.data(?![A-Za-z0-9_])", line))
                else:
                    found = token in line
                if not found:
                    continue
                owner = next((name for start, end, name in spans if start <= line_no <= end), "<module>")
                rel = _relative(path)
                active = (
                    (rel.startswith("dmotpy/models/core/")
                     or rel.startswith("dmotpy/models/flux/")
                     or rel in {
                         "dmotpy/models/hydrology_model.py",
                         "dmotpy/models/mopex_doy_model.py",
                         "dmotpy/models/tcm_model.py",
                         "dmotpy/models/endpoint_uh_model.py",
                         "dmotpy/models/intermediate_uh_model.py",
                         "dmotpy/models/gr4j_uh_model.py",
                     })
                    or rel == "scripts/run_dpl_benchmark_dmg_native.py"
                    or rel == "dpl/attributes.py"
                ) and "/test/" not in rel and not rel.startswith("validation_results/")
                if rel == "dpl/build_caravan_671_matrix.py":
                    active = False
                rows.append(
                    {
                        "token": token,
                        "file_line": f"{rel}:{line_no}",
                        "function_or_class": owner,
                        "source": line.strip(),
                        "on_active_forward_or_training_path": str(active).lower(),
                    }
                )
    return rows


def _parameter_bound_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in NPARAM_INFO_36:
        description = _description_map(model)
        for name, (lo, hi) in PARAM_INFO[model].items():
            ratio: str | float = "undefined (lo=0)" if float(lo) == 0 else float(hi) / float(lo)
            rows.append(
                {
                    "model": model,
                    "parameter": name,
                    "lo": lo,
                    "hi": hi,
                    "hi_over_lo": ratio,
                    "unit_or_physical_meaning": description.get(name, "not declared in model module"),
                }
            )
    return rows


def _warmup_rows() -> list[dict[str, str]]:
    rows = [
        {"class": "HydrologyModel", "file_line": "dmotpy/models/hydrology_model.py:297-300", "uses_no_grad": "yes", "detaches_states": "yes", "models": "all base models"},
        {"class": "MopexDoyModel", "file_line": "dmotpy/models/mopex_doy_model.py:54-67", "uses_no_grad": "yes", "detaches_states": "yes", "models": "mopex4, mopex5"},
        {"class": "TCMModel", "file_line": "dmotpy/models/tcm_model.py:41-53", "uses_no_grad": "yes", "detaches_states": "yes", "models": "tcm"},
        {"class": "EndpointUHModel", "file_line": "dmotpy/models/endpoint_uh_model.py:53-130", "uses_no_grad": "no", "detaches_states": "no", "models": ", ".join(ENDPOINT_UH_SCHEMES)},
        {"class": "IntermediateUHModel", "file_line": "dmotpy/models/intermediate_uh_model.py:89-138", "uses_no_grad": "no", "detaches_states": "no", "models": ", ".join(INTERMEDIATE_UH_CONFIG)},
    ]
    routes: list[dict[str, str]] = []
    for model in NPARAM_INFO_36:
        if model in ENDPOINT_UH_SCHEMES:
            wrapper = "EndpointUHModel"
        elif model in INTERMEDIATE_UH_CONFIG:
            wrapper = "IntermediateUHModel"
        elif model in {"mopex4", "mopex5"}:
            wrapper = "MopexDoyModel"
        elif model == "tcm":
            wrapper = "TCMModel"
        else:
            wrapper = "HydrologyModel"
        routes.append({"model": model, "warmup_path": wrapper})
    return rows, routes


def _checkpoint_path(model: str) -> Path:
    return CMA_ROOT / model / "chunk_0_gen_300.pt"


def _checkpoint_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in NPARAM_INFO_36:
        path = _checkpoint_path(model)
        row: dict[str, Any] = {"model": model, "path": str(path), "exists": path.exists()}
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = payload.get("solver", {}).get("state", {})
            best = state.get("best_latent")
            ids = payload.get("basin_ids")
            dimension = int(payload.get("solver", {}).get("dimension", -1))
            row.update(
                generation=payload.get("generation"),
                basin_ids=len(ids) if ids is not None else 0,
                dimension=dimension,
                best_latent_shape=list(best.shape) if isinstance(best, torch.Tensor) else None,
                normalized_space="no: best_latent is unbounded; normalized theta = sigmoid(best_latent)",
                coverage_note=("complete 531-basin archive" if ids is not None and len(ids) == 531 else "partial or non-531 archive"),
            )
        else:
            row.update(generation=None, basin_ids=0, dimension=None, best_latent_shape=None, normalized_space="unavailable", coverage_note="no latest CMA checkpoint in local archive")
        rows.append(row)
    return rows


def _best_cma_theta(model: str, basin_ids: np.ndarray) -> tuple[torch.Tensor | None, str]:
    path = _checkpoint_path(model)
    if not path.exists():
        return None, f"CMA checkpoint unavailable: {path}"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved_ids = np.asarray(payload["basin_ids"], dtype=np.int64)
    state = payload["solver"]["state"]
    best_latent = state["best_latent"].to(torch.float64)
    fitness = state["best_fitness"].to(torch.float64)
    dim = best_latent.shape[1]
    if best_latent.shape[0] % len(saved_ids) != 0:
        return None, "CMA archive has incompatible units/basin_ids shape"
    starts = best_latent.shape[0] // len(saved_ids)
    latent = best_latent.reshape(len(saved_ids), starts, dim)
    scores = fitness.reshape(len(saved_ids), starts)
    index = scores.argmax(dim=1)
    theta = torch.sigmoid(latent[torch.arange(len(saved_ids)), index]).to(torch.float32)
    lookup = {int(basin): position for position, basin in enumerate(saved_ids)}
    missing = [int(basin) for basin in basin_ids if int(basin) not in lookup]
    if missing:
        return None, f"CMA archive covers {len(saved_ids)} basins but misses requested IDs, e.g. {missing[:5]}"
    return theta[[lookup[int(basin)] for basin in basin_ids]], f"{path} (best of {starts} starts by best_fitness)"


def _load_forcing(basin_ids: np.ndarray, start: str) -> tuple[torch.Tensor, torch.Tensor]:
    bundle = DATA_DIR / "camels_dataset"
    if not bundle.exists():
        bundle = DATA_DIR / "camels_dataset.pkl"
    with bundle.open("rb") as handle:
        data = pickle.load(handle)
    if isinstance(data, dict):
        forcings, streamflow = data["forcings"], data["streamflow"]
    else:
        forcings, streamflow = data[:2]
    all_ids = np.asarray(np.load(DATA_DIR / "gage_id.npy"), dtype=np.int64)
    lookup = {int(value): index for index, value in enumerate(all_ids)}
    positions = np.asarray([lookup[int(basin)] for basin in basin_ids], dtype=np.int64)
    dates = pd.date_range("1980-10-01", "2014-09-30", freq="D")
    left = int(dates.get_loc(pd.Timestamp(start)))
    right = left + WARMUP_DAYS + SCORED_DAYS
    if right > len(dates):
        raise ValueError(f"{start} + 730 days exceeds CAMELS extent")
    x = np.asarray(forcings)[positions, left:right, :3]
    q = np.asarray(streamflow)[positions, left:right, 0]
    # The loader used by the active dPL runner takes area from Caravan attributes.
    from dpl.attributes import CatchmentAttributeBuilder

    area = CatchmentAttributeBuilder().load_raw_attributes([int(x) for x in basin_ids])[:, 11]
    q = q * (0.0283168 * 86400.0 * 1000.0 / (area[:, None] * 1e6))
    return torch.as_tensor(np.transpose(x, (1, 0, 2)), dtype=torch.float32), torch.as_tensor(q.T, dtype=torch.float32)


def _load_full_training() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the same 1980-10-01..1995-09-30 training span as the runner."""
    bundle = DATA_DIR / "camels_dataset"
    if not bundle.exists():
        bundle = DATA_DIR / "camels_dataset.pkl"
    with bundle.open("rb") as handle:
        data = pickle.load(handle)
    if isinstance(data, dict):
        forcings, streamflow = data["forcings"], data["streamflow"]
    else:
        forcings, streamflow = data[:2]
    ids = load_ids(DATA_DIR / "531sub_id.txt")
    all_ids = np.asarray(np.load(DATA_DIR / "gage_id.npy"), dtype=np.int64)
    lookup = {int(value): index for index, value in enumerate(all_ids)}
    positions = np.asarray([lookup[int(basin)] for basin in ids], dtype=np.int64)
    dates = pd.date_range("1980-10-01", "2014-09-30", freq="D")
    left = int(dates.get_loc(pd.Timestamp("1980-10-01")))
    right = int(dates.get_loc(pd.Timestamp("1995-09-30"))) + 1
    x = np.asarray(forcings)[positions, left:right, :3]
    q = np.asarray(streamflow)[positions, left:right, 0]
    from dpl.attributes import CatchmentAttributeBuilder

    area = CatchmentAttributeBuilder().load_raw_attributes([int(value) for value in ids])[:, 11]
    q = q * (0.0283168 * 86400.0 * 1000.0 / (area[:, None] * 1e6))
    return x, q, ids


def _window_catalog(observations: np.ndarray, length: int = SCORED_DAYS) -> list[np.ndarray]:
    values = np.asarray(observations, dtype=np.float64)[:, WARMUP_DAYS:]
    valid = np.isfinite(values) & (values >= 0.0)
    clean = np.where(valid, values, 0.0)
    count_cs = np.concatenate((np.zeros((values.shape[0], 1)), np.cumsum(valid, axis=1)), axis=1)
    sum_cs = np.concatenate((np.zeros((values.shape[0], 1)), np.cumsum(clean, axis=1)), axis=1)
    square_cs = np.concatenate((np.zeros((values.shape[0], 1)), np.cumsum(clean * clean, axis=1)), axis=1)
    count = count_cs[:, length:] - count_cs[:, :-length]
    total = sum_cs[:, length:] - sum_cs[:, :-length]
    square_total = square_cs[:, length:] - square_cs[:, :-length]
    safe = np.maximum(count, 1.0)
    variance = np.maximum(square_total / safe - (total / safe) ** 2, 0.0)
    catalog: list[np.ndarray] = []
    for basin in range(values.shape[0]):
        starts = np.flatnonzero((count[basin] >= 30) & (variance[basin] >= 0.01**2))
        if starts.size == 0:
            score = np.where(count[basin] >= 30, variance[basin], -1.0)
            starts = np.asarray([int(np.argmax(score))], dtype=np.int64)
        catalog.append(starts.astype(np.int64))
    return catalog


def _mask_epoch(model_name: str, seed: int = 42, n_batches: int = 169) -> dict[str, Any]:
    """Run one 169-minibatch inference epoch; never constructs a backward graph."""
    memory_reason = _cgroup_memory_reason()
    if memory_reason is not None:
        return {
            "model": model_name,
            "regime": "theta_0.5",
            "scope": f"NOT RUN: requested {n_batches} full forward-only minibatches",
            "observation_missing_fraction": "N/A",
            "prediction_nonfinite_fraction": "N/A",
            "prediction_negative_fraction": "N/A",
            "any_masked_fraction": "N/A",
            "invalid_kge_basin_fraction": "N/A",
            "epoch_batches": n_batches,
            "epoch_points": 0,
            "status": "NOT_RUN",
            "reason": memory_reason,
        }
    forcing_np, target_np, basin_ids = _load_full_training()
    catalog = _window_catalog(target_np)
    rng = np.random.default_rng(seed)
    model = build_model(model_name, "cpu", warm_up=WARMUP_DAYS, backend="eager")
    dim = NPARAM_INFO_36[model_name]
    totals = defaultdict(float)
    total_points = 0
    invalid_basins = 0
    batch_size = 100
    with torch.no_grad():
        for _ in range(n_batches):
            selected = rng.choice(len(basin_ids), size=batch_size, replace=False)
            starts = [int(rng.choice(catalog[int(basin)])) for basin in selected]
            x_batch = torch.stack(
                [torch.as_tensor(forcing_np[basin, start : start + WARMUP_DAYS + SCORED_DAYS, :3], dtype=torch.float32) for basin, start in zip(selected, starts)], dim=1
            )
            y_batch = torch.stack(
                [torch.as_tensor(target_np[basin, start + WARMUP_DAYS : start + WARMUP_DAYS + SCORED_DAYS], dtype=torch.float32) for basin, start in zip(selected, starts)], dim=1
            )
            raw = torch.full((batch_size, dim, 1), 0.5, dtype=torch.float32)
            q = model({"x_phy": x_batch}, (None, raw))["streamflow"].squeeze(-1).squeeze(-1)
            obs_missing = ~torch.isfinite(y_batch)
            pred_nonfinite = ~torch.isfinite(q)
            pred_negative = torch.isfinite(q) & (q < 0.0)
            valid = ~(obs_missing | pred_nonfinite | pred_negative)
            n_valid = valid.sum(dim=0)
            invalid_basins += int((n_valid <= 30).sum())
            totals["observation_missing_fraction"] += float(obs_missing.sum())
            totals["prediction_nonfinite_fraction"] += float(pred_nonfinite.sum())
            totals["prediction_negative_fraction"] += float(pred_negative.sum())
            totals["any_masked_fraction"] += float((~valid).sum())
            total_points += int(valid.numel())
    return {
        "model": model_name,
        "regime": "theta_0.5",
        "scope": f"forward-only: {n_batches} random 100-basin minibatches x 730 days",
        "observation_missing_fraction": totals["observation_missing_fraction"] / total_points,
        "prediction_nonfinite_fraction": totals["prediction_nonfinite_fraction"] / total_points,
        "prediction_negative_fraction": totals["prediction_negative_fraction"] / total_points,
        "any_masked_fraction": totals["any_masked_fraction"] / total_points,
        "invalid_kge_basin_fraction": invalid_basins / (n_batches * batch_size),
        "epoch_batches": n_batches,
        "epoch_points": total_points,
        "status": "COMPLETE",
        "reason": "",
    }


def _cgroup_memory_reason() -> str | None:
    """Avoid an OOM kill when unrelated processes consume the shared cgroup."""
    current_path = Path("/sys/fs/cgroup/memory.current")
    limit_path = Path("/sys/fs/cgroup/memory.max")
    if not current_path.exists():
        current_path = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        limit_path = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if not current_path.exists() or not limit_path.exists():
        return None
    try:
        current = int(current_path.read_text().strip())
        raw_limit = limit_path.read_text().strip()
        if raw_limit == "max":
            return None
        limit = int(raw_limit)
    except (OSError, ValueError):
        return None
    remaining = limit - current
    minimum = 800 * 1024 * 1024
    if remaining < minimum:
        return f"cgroup memory headroom {remaining / 1024**2:.1f} MiB < 800 MiB; unrelated processes prevent full CAMELS epoch load"
    return None


@dataclass
class _BranchTotal:
    taken: float = 0.0
    count: int = 0
    detail: str = ""


class BranchTraceMode(torch.utils._python_dispatch.TorchDispatchMode):
    """Collect actual branch choices without editing any model source file."""

    def __init__(self) -> None:
        super().__init__()
        self.totals: dict[tuple[str, str], _BranchTotal] = {}

    @staticmethod
    def _location() -> str:
        frame = inspect.currentframe()
        while frame is not None:
            filename = Path(frame.f_code.co_filename)
            try:
                rel = filename.resolve().relative_to(BENCHMARK_ROOT)
            except ValueError:
                frame = frame.f_back
                continue
            if str(rel).startswith("dmotpy/models/"):
                return f"{rel}:{frame.f_lineno}"
            frame = frame.f_back
        return "<unknown>"

    @staticmethod
    def _scalar(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return float(value.detach().reshape(-1)[0].item()) if value.numel() else None
        return float(value)

    def _record(self, operator: str, active: torch.Tensor, detail: str) -> None:
        key = (self._location(), operator)
        total = self.totals.setdefault(key, _BranchTotal(detail=detail))
        total.taken += float(active.detach().to(torch.float64).sum().item())
        total.count += int(active.numel())

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        kwargs = kwargs or {}
        name = str(func)
        if "aten.relu" in name and args and isinstance(args[0], torch.Tensor):
            self._record("relu", args[0] > 0, "active means input > 0")
        elif "aten.minimum" in name and len(args) >= 2 and isinstance(args[0], torch.Tensor):
            self._record("minimum:left", args[0] <= args[1], "active means left argument selected")
            self._record("minimum:right", args[1] < args[0], "active means right argument selected")
        elif "aten.maximum" in name and len(args) >= 2 and isinstance(args[0], torch.Tensor):
            self._record("maximum:left", args[0] >= args[1], "active means left argument selected")
            self._record("maximum:right", args[1] > args[0], "active means right argument selected")
        elif "aten.clamp" in name and args and isinstance(args[0], torch.Tensor):
            lower = self._scalar(args[1] if len(args) > 1 else kwargs.get("min"))
            upper = self._scalar(args[2] if len(args) > 2 else kwargs.get("max"))
            if lower is not None:
                self._record("clamp:above_min", args[0] > lower, f"active means input > min={lower:g}")
            if upper is not None:
                self._record("clamp:below_max", args[0] < upper, f"active means input < max={upper:g}")
        return func(*args, **kwargs)


STATE_CAPACITY = {
    "collie3": {"S1": "smax"},
    "newzealand1": {"S1": "s1max"},
    "penman": {"S1": "smax"},
    "flexi": {"S1": "imax", "S2": "smax"},
    "flexis": {"S2": "imax", "S3": "smax"},
    "hbv96": {"S3": "fc"},
}


STATE_NAMES = {
    "collie3": ("S1", "S2"),
    "newzealand1": ("S1",),
    "penman": ("S1", "S2_deficit", "S3"),
    "flexi": ("S1", "S2", "S3", "S4"),
    "flexis": ("S1_snow", "S2", "S3", "S4", "S5"),
    "hbv96": ("S1_snow", "S2_liquid", "S3", "S4", "S5"),
}


class StateCapture:
    def __init__(self, model: str, raw: torch.Tensor) -> None:
        self.model = model
        self.raw = raw
        self.records: dict[str, list[torch.Tensor]] = defaultdict(list)

    def record(self, outputs: tuple[torch.Tensor, ...], state_offset: int, names: tuple[str, ...]) -> None:
        for index, state in enumerate(outputs[state_offset : state_offset + len(names)]):
            name = names[index]
            self.records[name].append(state.detach().reshape(-1).cpu())

    def rows(self, regime: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for state, values in self.records.items():
            x = torch.cat(values).numpy()
            capacity_parameter = STATE_CAPACITY.get(self.model, {}).get(state)
            ratio = None
            if capacity_parameter is not None:
                parameter_index = list(PARAM_INFO[self.model]).index(capacity_parameter)
                lo, hi = list(PARAM_INFO[self.model].values())[parameter_index]
                capacity = float(lo) + self.raw[:, parameter_index, :].detach().reshape(-1).numpy() * (float(hi) - float(lo))
                # The capture sequence is time-major; repeat each basin capacity
                # for every recorded timestep before calculating ratios.
                ratio = x / np.tile(capacity, x.size // len(capacity))
            rows.append(
                {
                    "model": self.model,
                    "regime": regime,
                    "state": state,
                    "capacity_parameter": capacity_parameter or "not_declared",
                    "p05_state": float(np.quantile(x, 0.05)),
                    "p50_state": float(np.quantile(x, 0.50)),
                    "p95_state": float(np.quantile(x, 0.95)),
                    "p05_state_over_capacity": float(np.quantile(ratio, 0.05)) if ratio is not None else "N/A",
                    "p50_state_over_capacity": float(np.quantile(ratio, 0.50)) if ratio is not None else "N/A",
                    "p95_state_over_capacity": float(np.quantile(ratio, 0.95)) if ratio is not None else "N/A",
                    "n": int(x.size),
                }
            )
        return rows


def _install_state_capture(model: torch.nn.Module, name: str, capture: StateCapture) -> None:
    if name in INTERMEDIATE_UH_CONFIG:
        original_pre = model.step_pre_fn
        original_post = model.step_post_fn
        pre_states = STATE_NAMES[name][: INTERMEDIATE_UH_CONFIG[name]["n_pre_states"]]
        post_states = STATE_NAMES[name][INTERMEDIATE_UH_CONFIG[name]["n_pre_states"] :]

        def pre(*args, **kwargs):
            result = original_pre(*args, **kwargs)
            capture.record(result, 2 + INTERMEDIATE_UH_CONFIG[name]["n_pre_passthru"], pre_states)
            return result

        def post(*args, **kwargs):
            result = original_post(*args, **kwargs)
            capture.record(result, 2, post_states)
            return result

        model.step_pre_fn = pre
        model.step_post_fn = post
        return
    original = model.step_fn

    def step(*args, **kwargs):
        result = original(*args, **kwargs)
        capture.record(result, 2, STATE_NAMES[name])
        return result

    model.step_fn = step


def _runner_kge(q_sim: torch.Tensor, q_obs: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Exact arithmetic/mask contract from run_dpl_benchmark_dmg_native.py:178-219."""
    eps_sq = eps * eps
    mask_obs_missing = ~torch.isfinite(q_obs)
    mask_pred_nonfinite = ~torch.isfinite(q_sim)
    mask_pred_negative = torch.isfinite(q_sim) & (q_sim < 0.0)
    mask = torch.isfinite(q_obs) & torch.isfinite(q_sim) & (q_obs >= 0.0) & (q_sim >= 0.0)
    mask_f = mask.to(dtype=q_sim.dtype)
    n_valid = mask_f.sum(dim=0).clamp_min(1.0)
    obs_safe = torch.where(mask, q_obs, torch.zeros_like(q_obs))
    sim_safe = torch.where(mask, q_sim, torch.zeros_like(q_sim))
    mean_obs, mean_sim = obs_safe.sum(0) / n_valid, sim_safe.sum(0) / n_valid
    obs_diff, sim_diff = (obs_safe - mean_obs) * mask_f, (sim_safe - mean_sim) * mask_f
    std_obs = torch.sqrt((obs_diff.square().sum(0) / n_valid) + eps_sq)
    std_sim = torch.sqrt((sim_diff.square().sum(0) / n_valid) + eps_sq)
    r = (obs_diff * sim_diff).sum(0) / n_valid / (std_obs * std_sim)
    alpha, beta = std_sim / std_obs, mean_sim / (mean_obs + eps)
    kge = 1.0 - torch.sqrt((r - 1.0).square() + (alpha - 1.0).square() + (beta - 1.0).square() + eps_sq)
    valid_basins = (n_valid > 30) & torch.isfinite(kge)
    loss = 1.0 - torch.where(valid_basins, kge, torch.zeros_like(kge)).sum() / valid_basins.sum().clamp_min(1)
    stats = {
        "observation_missing_fraction": float(mask_obs_missing.float().mean()),
        "prediction_nonfinite_fraction": float(mask_pred_nonfinite.float().mean()),
        "prediction_negative_fraction": float(mask_pred_negative.float().mean()),
        "any_masked_fraction": float((~mask).float().mean()),
        "invalid_kge_basin_fraction": float((~valid_basins).float().mean()),
    }
    return loss, kge, stats


def _forward_branch_audit(model_name: str, x: torch.Tensor, raw: torch.Tensor, regime: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    model = build_model(model_name, "cpu", warm_up=WARMUP_DAYS, backend="eager")
    capture = StateCapture(model_name, raw)
    _install_state_capture(model, model_name, capture)
    trace = BranchTraceMode()
    with torch.no_grad(), trace:
        q = model({"x_phy": x}, (None, raw))["streamflow"].squeeze(-1).squeeze(-1)
    branches = []
    for (location, operator), total in sorted(trace.totals.items()):
        fraction = total.taken / total.count if total.count else float("nan")
        branches.append(
            {
                "model": model_name,
                "regime": regime,
                "file_line": location,
                "operator": operator,
                "active_frac": fraction,
                "n": total.count,
                "flag_lt_1pct": fraction < 0.01,
                "definition": total.detail,
            }
        )
    return branches, capture.rows(regime), {"prediction_nonfinite_fraction": float((~torch.isfinite(q)).float().mean())}


def _gradient_rows(model_name: str, x: torch.Tensor, y: torch.Tensor, raw_values: torch.Tensor, regime: str) -> tuple[list[dict[str, Any]], dict[str, float]]:
    raw = raw_values.detach().clone().requires_grad_(True)
    model = build_model(model_name, "cpu", warm_up=WARMUP_DAYS, backend="eager")
    q = model({"x_phy": x}, (None, raw))["streamflow"].squeeze(-1).squeeze(-1)
    loss, _kge, mask_stats = _runner_kge(q, y[WARMUP_DAYS:])
    loss.backward()
    rows: list[dict[str, Any]] = []
    for index, (parameter, (lo, hi)) in enumerate(PARAM_INFO[model_name].items()):
        grad = raw.grad[:, index, 0]
        theta = raw[:, index, 0]
        physical = float(lo) + theta * (float(hi) - float(lo))
        width = float(hi) - float(lo)
        abs_grad = grad.abs()
        # Chain rule: dL/d ln(p) = dL/dtheta * p / (hi-lo).
        log_grad = abs_grad * physical / width
        rows.append(
            {
                "model": model_name,
                "regime": regime,
                "parameter": parameter,
                "lo": lo,
                "hi": hi,
                "dphysical_dtheta": width,
                "mean_abs_dloss_dtheta": float(abs_grad.mean()),
                "median_abs_dloss_dtheta": float(abs_grad.median()),
                "zero_gradient_basin_fraction": float((grad == 0).float().mean()),
                "mean_abs_dloss_dlnp": float(log_grad.mean().detach()),
                "median_abs_dloss_dlnp": float(log_grad.median().detach()),
                "loss": float(loss.detach()),
                "formula": "abs(dL/dtheta) * p / (hi-lo)",
            }
        )
    return rows, mask_stats


def _loss_diff_markdown() -> str:
    runner = BENCHMARK_ROOT / "scripts/run_dpl_benchmark_dmg_native.py"
    losses = BENCHMARK_ROOT / "dmotpy/losses.py"
    runner_lines = runner.read_text().splitlines()
    loss_lines = losses.read_text().splitlines()
    runner_code = "\n".join(runner_lines[177:219])
    loss_code = "\n".join(loss_lines[15:148])
    return f"""## E. Loss path consistency

Active runner implementation: `scripts/run_dpl_benchmark_dmg_native.py:178-219`.

```python
{runner_code}
```

Shared loss contract: `dmotpy/losses.py:16-148`.

```python
{loss_code}
```

Observed source-level differences:

| Aspect | Runner | Shared KgeLoss contract |
|---|---|---|
| Prediction non-finite | Included in `mask`, then silently removed | Raises `FloatingPointError` before masking |
| Negative prediction | Included in `mask`, then silently removed | Not rejected by `_prepare`; retained if finite |
| Observation non-finite | Masked | Masked |
| Minimum observations | `n_valid > 30` basin eligibility | skips a column with fewer than 2 valid points |
| Variance stabilization | `sqrt(var + eps^2)` | sample std and denominator `+ eps` |
| KGE aggregation | mean over valid basin KGEs | columnwise KGE with configurable reduction |
| Weighting | none | none |
"""


def _descale_source_markdown() -> str:
    path = BENCHMARK_ROOT / "dmotpy/models/hydrology_model.py"
    lines = path.read_text().splitlines()
    return "\n".join(lines[197:219])


def _write_report(output: Path, metadata: dict[str, Any], mask_rows: list[dict[str, Any]]) -> None:
    symbolic = list(csv.DictReader((output / "symbolic_parameter_expressions.csv").open()))
    branches = list(csv.DictReader((output / "branch_activation.csv").open())) if (output / "branch_activation.csv").exists() else []
    gradients = list(csv.DictReader((output / "gradient_conditioning.csv").open())) if (output / "gradient_conditioning.csv").exists() else []
    red = [row for row in branches if row["flag_lt_1pct"] == "True"]
    lines = [
        "# dPL gradient evidence (read-only)",
        "",
        f"- Created: {date.today().isoformat()}",
        f"- Real forcing: {metadata['start']} to 730 days later; {metadata['basin_count']} basins; 365 warm-up + 365 scored days.",
        "- Active model path: `src.model_registry.build_model(..., backend='eager')`, the same model factory used by the dPL runner.",
        "- No model source, loss source, training configuration, checkpoint, or log was changed.",
        "",
        "## A. Symbol-level trace",
        "",
        f"`symbolic_parameter_expressions.csv` contains {len(symbolic)} parameter-expression rows. It includes direct core-step occurrences and explicit links to generic flux arguments for Collie3 a/b/lambda, NewZealand1 a/b, Penman gam, and Flexi/Flexis imax/beta/percmax.",
        "`static_gradient_risks.csv` is the requested repository-wide Python-code grep for `nan_to_num`, `detach()`, `.data`, `with torch.no_grad()`, and `requires_grad_(False)`. The active-path column distinguishes source from historical diagnostics/tests.",
        "",
        "## B. Real-forcing branch activation",
        "",
        f"`branch_activation.csv` has {len(branches)} actual ATen branch records. `flag_lt_1pct=True` is the requested red condition; {len(red)} records satisfy it. Each record is keyed by the actual source call site captured while the active forward ran, not by a reconstructed formula.",
        "",
        "## C. Parameter mapping",
        "",
        "`parameter_bounds_36models.csv` lists all active registry bounds and descriptions. `HydrologyModel._descale_params` is implemented in `dmotpy/models/hydrology_model.py:198-216`; with the active `parameter_mapping='linear'`, every parameter uses `p = lo + theta * (hi-lo)` through `_change_param_range` at `:177-196`.",
        "\n```python\n" + _descale_source_markdown() + "\n```\n",
        f"`gradient_conditioning.csv` has {len(gradients)} per-parameter real-KGE gradients. Its log-space column uses the chain rule `abs(dL/dtheta) * p/(hi-lo)`, not an invalid dimensional division by `hi-lo` alone.",
        "",
        "## D. Warm-up contract",
        "",
        "`warmup_contract.csv` and `warmup_model_routes.csv` list each wrapper and all 36 model routes. Base/MOPEX/TCM warm-up uses no-grad plus state detach; endpoint/intermediate UH wrappers retain the whole sequence graph.",
        "",
        _loss_diff_markdown(),
        "## F. CMA-ES archive inventory",
        "",
        "`cmaes_archive_inventory.csv` reports local checkpoint coverage. A checkpoint contains unbounded `best_latent`; normalized theta is `sigmoid(best_latent)`. Missing or partial archives remain explicitly unavailable in runtime rows.",
        "",
        "## Mask evidence",
        "",
        "`loss_mask_statistics.csv` contains the exact runner mask categories for each executed diagnostic segment. An epoch-wide 169-minibatch pass is only produced when `--mask-epoch` is supplied; it is not synthesized from the 32-basin audit.",
    ]
    (output / "dpl_gradient_evidence_report.md").write_text("\n".join(lines) + "\n")
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _candidate_basins() -> np.ndarray:
    # The Flexi/Flexis CMA snapshot is partial.  The common archive intersection
    # gives both midpoint and CMA regimes a real 32-basin comparison set.
    present: list[set[int]] = []
    for model in ("collie3", "newzealand1", "penman", "flexi", "flexis"):
        path = _checkpoint_path(model)
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            present.append(set(map(int, payload["basin_ids"])))
    if not present:
        raise RuntimeError("No CMA checkpoint is available to construct the common diagnostic basin set")
    common = np.asarray(sorted(set.intersection(*present)), dtype=np.int64)
    if len(common) < 32:
        raise RuntimeError(f"Only {len(common)} basins are common to the available CMA archives")
    # Deterministic coverage across the common ID ordering; reported verbatim.
    return common[np.linspace(0, len(common) - 1, 32, dtype=np.int64)]


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output_dir) if args.output_dir else BENCHMARK_ROOT / "results" / f"dpl_gradient_evidence_{date.today():%Y%m%d}"
    output.mkdir(parents=True, exist_ok=True)
    symbolic = _symbolic_rows()
    _augment_flux_rows(symbolic)
    _csv(output / "symbolic_parameter_expressions.csv", symbolic, ["model", "parameter", "file_line", "expression", "blocking_operator_type", "can_yield_zero_gradient", "note"])
    _csv(output / "static_gradient_risks.csv", _static_risk_rows(), ["token", "file_line", "function_or_class", "source", "on_active_forward_or_training_path"])
    _csv(output / "parameter_bounds_36models.csv", _parameter_bound_rows(), ["model", "parameter", "lo", "hi", "hi_over_lo", "unit_or_physical_meaning"])
    wrappers, routes = _warmup_rows()
    _csv(output / "warmup_contract.csv", wrappers, ["class", "file_line", "uses_no_grad", "detaches_states", "models"])
    _csv(output / "warmup_model_routes.csv", routes, ["model", "warmup_path"])
    _csv(output / "cmaes_archive_inventory.csv", _checkpoint_inventory(), ["model", "path", "exists", "generation", "basin_ids", "dimension", "best_latent_shape", "normalized_space", "coverage_note"])

    basin_ids = _candidate_basins() if args.basin_ids is None else np.asarray([int(x) for x in args.basin_ids.split(",")], dtype=np.int64)
    if len(basin_ids) != 32:
        raise ValueError("This audit requires exactly 32 comma-separated basin IDs, or omit --basin-ids")
    x, y = _load_forcing(basin_ids, args.start)
    branch_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    cma_runtime: list[dict[str, Any]] = []

    selected_models = tuple(args.models.split(",")) if args.models else TARGET_MODELS
    unknown = sorted(set(selected_models) - set(TARGET_MODELS))
    if unknown:
        raise ValueError(f"--models only accepts {TARGET_MODELS}; got {unknown}")
    for model in selected_models:
        dim = NPARAM_INFO_36[model]
        midpoint = torch.full((len(basin_ids), dim, 1), 0.5, dtype=torch.float32)
        for regime, theta in (("theta_0.5", midpoint),):
            branches, states, forward_stats = _forward_branch_audit(model, x, theta, regime)
            branch_rows.extend(branches)
            state_rows.extend(states)
            gradients, mask = _gradient_rows(model, x, y, theta, regime)
            gradient_rows.extend(gradients)
            mask_rows.append({"model": model, "regime": regime, "scope": "32 basins x one 365+365 real window", **forward_stats, **mask})

        cma_theta, source = _best_cma_theta(model, basin_ids)
        cma_runtime.append({"model": model, "regime": "cmaes_best", "availability": "available" if cma_theta is not None else "unavailable", "source_or_reason": source})
        if cma_theta is not None:
            cma_raw = cma_theta.unsqueeze(-1)
            branches, states, forward_stats = _forward_branch_audit(model, x, cma_raw, "cmaes_best")
            branch_rows.extend(branches)
            state_rows.extend(states)
            gradients, mask = _gradient_rows(model, x, y, cma_raw, "cmaes_best")
            gradient_rows.extend(gradients)
            mask_rows.append({"model": model, "regime": "cmaes_best", "scope": "32 basins x one 365+365 real window", **forward_stats, **mask})
        # Each daily recurrent graph can be large.  Results above are Python
        # scalars/CPU arrays only; collecting now prevents model-to-model RSS
        # accumulation without changing any forward or backward calculation.
        del midpoint
        if cma_theta is not None:
            del cma_theta, cma_raw
        gc.collect()

    _csv(output / "branch_activation.csv", branch_rows, ["model", "regime", "file_line", "operator", "active_frac", "n", "flag_lt_1pct", "definition"])
    _csv(output / "state_capacity_quantiles.csv", state_rows, ["model", "regime", "state", "capacity_parameter", "p05_state", "p50_state", "p95_state", "p05_state_over_capacity", "p50_state_over_capacity", "p95_state_over_capacity", "n"])
    _csv(output / "gradient_conditioning.csv", gradient_rows, ["model", "regime", "parameter", "lo", "hi", "dphysical_dtheta", "mean_abs_dloss_dtheta", "median_abs_dloss_dtheta", "zero_gradient_basin_fraction", "mean_abs_dloss_dlnp", "median_abs_dloss_dlnp", "loss", "formula"])
    if args.mask_epoch:
        for model in selected_models:
            mask_rows.append(_mask_epoch(model, n_batches=args.epoch_batches))
    _csv(output / "loss_mask_statistics.csv", mask_rows, ["model", "regime", "scope", "observation_missing_fraction", "prediction_nonfinite_fraction", "prediction_negative_fraction", "any_masked_fraction", "invalid_kge_basin_fraction", "epoch_batches", "epoch_points", "status", "reason"])
    _csv(output / "cmaes_runtime_availability.csv", cma_runtime, ["model", "regime", "availability", "source_or_reason"])

    metadata = {"start": args.start, "basin_count": int(len(basin_ids)), "basin_ids": basin_ids.tolist(), "warmup_days": WARMUP_DAYS, "scored_days": SCORED_DAYS, "models": list(selected_models), "mask_epoch_ran": bool(args.mask_epoch)}
    _write_report(output, metadata, mask_rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1989-01-01", help="CAMELS start date for the 730-day real forcing slice")
    parser.add_argument("--basin-ids", help="exactly 32 comma-separated IDs; default is the common CMA archive subset")
    parser.add_argument("--models", help="comma-separated subset of collie3,newzealand1,penman,flexi,flexis,hbv96")
    parser.add_argument("--output-dir", help="directory for read-only generated evidence")
    parser.add_argument("--mask-epoch", action="store_true", help="run full forward-only minibatches to quantify mask causes")
    parser.add_argument("--epoch-batches", type=int, default=169, help="number of inference minibatches for --mask-epoch (default: 169)")
    args = parser.parse_args()
    if not args.models:
        # A separate interpreter per model prevents PyTorch's dispatch and
        # operator caches from accumulating across six long recurrent graphs.
        # The child runs use exactly the same read-only code path below.
        output = Path(args.output_dir) if args.output_dir else BENCHMARK_ROOT / "results" / f"dpl_gradient_evidence_{date.today():%Y%m%d}"
        staging = output.with_name(output.name + "_per_model")
        staging.mkdir(parents=True, exist_ok=True)
        for model in TARGET_MODELS:
            child = staging / model
            command = [sys.executable, str(SCRIPT), "--models", model, "--output-dir", str(child), "--start", args.start]
            if args.basin_ids:
                command.extend(["--basin-ids", args.basin_ids])
            if args.mask_epoch:
                command.append("--mask-epoch")
                command.extend(["--epoch-batches", str(args.epoch_batches)])
            subprocess.run(command, check=True)
        output.mkdir(parents=True, exist_ok=True)
        static_files = [
            "symbolic_parameter_expressions.csv", "static_gradient_risks.csv",
            "parameter_bounds_36models.csv", "warmup_contract.csv",
            "warmup_model_routes.csv", "cmaes_archive_inventory.csv",
        ]
        for filename in static_files:
            shutil.copy2(staging / TARGET_MODELS[0] / filename, output / filename)
        dynamic_files = [
            "branch_activation.csv", "state_capacity_quantiles.csv",
            "gradient_conditioning.csv", "loss_mask_statistics.csv",
            "cmaes_runtime_availability.csv",
        ]
        for filename in dynamic_files:
            source = staging / TARGET_MODELS[0] / filename
            with source.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames or []
                rows = list(reader)
            for model in TARGET_MODELS[1:]:
                with (staging / model / filename).open(newline="") as handle:
                    rows.extend(csv.DictReader(handle))
            _csv(output / filename, rows, fields)
        basin_ids = [int(x) for x in args.basin_ids.split(",")] if args.basin_ids else _candidate_basins().tolist()
        metadata = {"start": args.start, "basin_count": len(basin_ids), "basin_ids": basin_ids, "warmup_days": WARMUP_DAYS, "scored_days": SCORED_DAYS, "models": list(TARGET_MODELS), "mask_epoch_ran": bool(args.mask_epoch), "execution": "isolated subprocess per model"}
        _write_report(output, metadata, [])
        print(output)
        return
    output = run(args)
    print(output)


if __name__ == "__main__":
    main()
