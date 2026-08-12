from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch

from dmotpy.models.endpoint_uh_model import ENDPOINT_UH_SCHEMES
from dmotpy.models.hydrology_model import HydrologyModel
from dmotpy.models.intermediate_uh_model import INTERMEDIATE_UH_CONFIG
from dmotpy.models.registry import NPARAM_INFO, PARAM_INFO

NPARAM_INFO_36 = {
    "alpine1": 4, "alpine2": 6, "australia": 8, "collie1": 1, "collie2": 4,
    "collie3": 6, "flexb": 9, "flexi": 10, "flexis": 12, "gr4j": 4, "gsfb": 8,
    "hbv96": 15, "hillslope": 7, "hymod": 5, "ihacres": 6, "modhydrolog": 15,
    "mopex1": 5, "mopex2": 7, "mopex3": 8, "mopex4": 10, "mopex5": 12,
    "newzealand1": 6, "newzealand2": 8, "penman": 4, "plateau": 8, "simhyd": 7,
    "smar": 8, "susannah1": 6, "susannah2": 6, "tank": 12, "tcm": 6,
    "topmodel": 7, "us1": 5, "vic": 10, "wetland": 4, "xinanjiang": 12,
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    dimension: int
    bounds: torch.Tensor
    routed_kind: str


def model_config(name: str, *, warm_up: int = 365, backend: str = "eager") -> dict:
    key = name.lower()
    if key not in NPARAM_INFO_36:
        raise KeyError(f"Not a fixed 36-model calibration target: {name}")
    cfg = {"model_name": key, "warm_up": int(warm_up), "warm_up_states": True,
           "variables": ["prcp", "tmean", "pet"], "nearzero": 1e-6,
           "nmul": 1, "parameter_mapping": "linear", "backend": backend}
    if key in ENDPOINT_UH_SCHEMES:
        cfg.update(uh_enabled=True, uh_mode="endpoint")
    elif key in INTERMEDIATE_UH_CONFIG:
        cfg.update(uh_enabled=True, uh_mode="intermediate")
    return cfg


def get_spec(name: str, device: torch.device | str = "cpu") -> ModelSpec:
    key = name.lower()
    expected = NPARAM_INFO_36[key]
    if NPARAM_INFO.get(key) != expected:
        raise RuntimeError(f"registry dimension mismatch for {key}: expected {expected}, got {NPARAM_INFO.get(key)}")
    entries = PARAM_INFO.get(key)
    if entries is None or len(entries) != expected:
        raise RuntimeError(f"bound count mismatch for {key}: {0 if entries is None else len(entries)} != {expected}")
    flat = [[float(lo), float(hi)] for lo, hi in entries.values()]
    if not all(math.isfinite(x) and lo < hi for (lo, hi) in flat for x in (lo, hi)):
        raise RuntimeError(f"non-finite or unordered bounds for {key}")
    kind = "endpoint" if key in ENDPOINT_UH_SCHEMES else "intermediate" if key in INTERMEDIATE_UH_CONFIG else "base"
    return ModelSpec(key, expected, torch.tensor(flat, dtype=torch.float64, device=device), kind)


def build_model(name: str, device: torch.device | str, *, warm_up: int, backend: str = "eager") -> HydrologyModel:
    return HydrologyModel(model_config(name, warm_up=warm_up, backend=backend), device=torch.device(device), backend=backend).to(device)


def audit_registry(names: Iterable[str] = NPARAM_INFO_36) -> list[dict]:
    rows: list[dict] = []
    for name in names:
        spec = get_spec(name)
        rows.append({"model": name, "dimension": spec.dimension, "bound_count": int(spec.bounds.shape[0]),
                     "bounds_finite": True, "routing": spec.routed_kind})
    if len(rows) != 36:
        raise RuntimeError(f"fixed registry must contain exactly 36 models, got {len(rows)}")
    return rows
