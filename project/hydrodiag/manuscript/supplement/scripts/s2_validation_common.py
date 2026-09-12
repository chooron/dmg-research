#!/usr/bin/env python3
"""Shared read-only helpers for the S2 validation closure."""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPLEMENT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = SUPPLEMENT_ROOT / "results"
LOG_ROOT = SUPPLEMENT_ROOT / "logs"

ACTIVE_CASES = (
    ("XAJ", "Base"), ("XAJ", "TGD"), ("XAJ", "CN"),
    ("GR4J", "Base"), ("GR4J", "TGD"), ("GR4J", "CN"),
    ("SIMHYD", "Base"), ("SIMHYD", "TGD"), ("SIMHYD", "CN"),
    ("HBV", "reference"),
)


def add_project_path(root: Path = PROJECT_ROOT) -> None:
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def ensure_dirs() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def git_commit(root: Path = PROJECT_ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return f"unavailable:{type(exc).__name__}"


def environment(root: Path = PROJECT_ROOT) -> dict[str, str]:
    try:
        import torch
        torch_version = torch.__version__
        cuda = str(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        torch_version = f"unavailable:{type(exc).__name__}"
        cuda = "unknown"
    return {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch_version,
        "cuda_available": cuda,
        "git_commit": git_commit(root),
    }


def imports() -> dict[str, Any]:
    add_project_path()
    from models import (
        GR4J, GR4JWithCemaNeige, GR4JWithTemperatureConditionedDelay,
        HBV, SIMHYD, SIMHYDWithCemaNeige,
        SIMHYDWithTemperatureConditionedDelay, XAJ,
        XAJWithCemaNeige, XAJWithTemperatureConditionedDelay,
    )
    from models.parameter_specs import (
        GR4J_CN_PARAM_SPECS, GR4J_PARAM_SPECS, GR4J_TGD_PARAM_SPECS,
        HBV_PARAM_SPECS, SIMHYD_CN_PARAM_SPECS, SIMHYD_PARAM_SPECS,
        SIMHYD_TGD_PARAM_SPECS, XAJ_CN_PARAM_SPECS, XAJ_PARAM_SPECS,
        XAJ_TGD_PARAM_SPECS,
    )
    return {
        "classes": {
            "XAJ_Base": XAJ, "XAJ_TGD": XAJWithTemperatureConditionedDelay,
            "XAJ_CN": XAJWithCemaNeige, "GR4J_Base": GR4J,
            "GR4J_TGD": GR4JWithTemperatureConditionedDelay,
            "GR4J_CN": GR4JWithCemaNeige, "SIMHYD_Base": SIMHYD,
            "SIMHYD_TGD": SIMHYDWithTemperatureConditionedDelay,
            "SIMHYD_CN": SIMHYDWithCemaNeige, "HBV_reference": HBV,
        },
        "specs": {
            "XAJ_Base": XAJ_PARAM_SPECS, "XAJ_TGD": XAJ_TGD_PARAM_SPECS,
            "XAJ_CN": XAJ_CN_PARAM_SPECS, "GR4J_Base": GR4J_PARAM_SPECS,
            "GR4J_TGD": GR4J_TGD_PARAM_SPECS, "GR4J_CN": GR4J_CN_PARAM_SPECS,
            "SIMHYD_Base": SIMHYD_PARAM_SPECS, "SIMHYD_TGD": SIMHYD_TGD_PARAM_SPECS,
            "SIMHYD_CN": SIMHYD_CN_PARAM_SPECS, "HBV_reference": HBV_PARAM_SPECS,
        },
    }


def case_key(model: str, structure: str) -> str:
    return f"{model}_{structure}"


def make_params(specs: dict[str, dict[str, Any]], dtype, batch: int = 1, requires_grad: bool = False):
    import torch
    values = {}
    for name, spec in specs.items():
        if name.endswith("tau"):
            value = (float(spec["lower"]) * float(spec["upper"])) ** 0.5
        else:
            value = 0.5 * (float(spec["lower"]) + float(spec["upper"]))
        values[name] = torch.full((batch,), value, dtype=dtype, requires_grad=requires_grad)
    return values


def make_forcing(name: str, dtype, batch: int = 1, steps: int = 12):
    import torch
    t = torch.arange(steps, dtype=dtype)
    if name == "zero_precip":
        p = torch.zeros(steps, dtype=dtype); pet = torch.full_like(p, 2.0); temp = torch.full_like(p, 5.0)
    elif name == "constant_precip":
        p = torch.full((steps,), 5.0, dtype=dtype); pet = torch.full_like(p, 2.0); temp = torch.full_like(p, 8.0)
    elif name == "mixed":
        p = torch.tensor([0., 4., 12., 1., 0., 8., 2., 15., 0., 5., 3., 9.], dtype=dtype)[:steps]
        pet = torch.tensor([2., 2., 3., 5., 6., 1., 4., 2., 7., 3., 4., 2.], dtype=dtype)[:steps]
        temp = torch.tensor([8., 5., 12., 3., -2., 1., 10., 0., -5., 6., 2., 9.], dtype=dtype)[:steps]
    elif name == "rain_snow_transition":
        p = torch.full((steps,), 6.0, dtype=dtype); pet = torch.full_like(p, 1.5)
        temp = torch.cat((torch.full((steps // 2,), -5.0, dtype=dtype), torch.full((steps - steps // 2,), 8.0, dtype=dtype)))
    elif name == "persistent_cold":
        p = torch.full((steps,), 4.0, dtype=dtype); pet = torch.full_like(p, 1.0); temp = torch.full_like(p, -8.0)
    elif name == "sudden_warm":
        p = torch.cat((torch.full((steps // 2,), 4.0, dtype=dtype), torch.zeros(steps - steps // 2, dtype=dtype)))
        pet = torch.full((steps,), 1.0, dtype=dtype); temp = torch.cat((torch.full((steps // 2,), -8.0, dtype=dtype), torch.full((steps - steps // 2,), 12.0, dtype=dtype)))
    elif name == "high_pet":
        p = torch.full((steps,), 3.0, dtype=dtype); pet = torch.full((steps,), 20.0, dtype=dtype); temp = torch.full_like(p, 10.0)
    elif name == "empty_state":
        p = torch.tensor([0., 0., 1., 0., 2., 0., 0., 1., 0., 0., 0., 0.], dtype=dtype)[:steps]
        pet = torch.full((steps,), 1.0, dtype=dtype); temp = torch.full_like(p, 5.0)
    elif name == "moderate_state":
        p = 3.0 + 2.0 * torch.sin(t * 0.7); p = torch.clamp(p, min=0.0)
        pet = 1.5 + 0.5 * torch.cos(t * 0.3); temp = 2.0 + 7.0 * torch.sin(t * 0.4)
    elif name == "uh_tail_impulse":
        p = torch.zeros(steps, dtype=dtype); p[0] = 40.0; pet = torch.zeros_like(p); temp = torch.full_like(p, 8.0)
    else:
        raise KeyError(name)
    forcing = {"precip": p.expand(batch, -1).clone(), "pet": pet.expand(batch, -1).clone(), "temp": temp.expand(batch, -1).clone()}
    forcing["temp_mean_train"] = torch.zeros(batch, dtype=dtype)
    forcing["temp_std_train"] = torch.full((batch,), 4.0, dtype=dtype)
    return forcing


def run_case(model: str, structure: str, forcing, dtype, return_states: bool = True):
    bundle = imports()
    key = case_key(model, structure)
    instance = bundle["classes"][key]().to(dtype=dtype)
    params = make_params(bundle["specs"][key], dtype, forcing["precip"].shape[0])
    return instance, params, instance(forcing, params, return_states=return_states)


def structure_target(model: str, structure: str) -> str:
    if model == "HBV":
        return "HBV"
    return f"{model}_{structure}"

