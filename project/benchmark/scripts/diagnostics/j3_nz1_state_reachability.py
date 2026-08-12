#!/usr/bin/env python3
"""J3: compare NZ1 B0/B1 checkpoint state reachability on the original 32 basins."""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from dmotpy.models.core.newzealand1 import create_initial_state
from dmotpy.models.flux.baseflow import baseflow_1
from dmotpy.models.flux.evap import evap_5, evap_6
from dmotpy.models.flux.interflow import interflow_9
from dmotpy.models.flux.saturation import saturation_1
from src.data_selection import load_ids
from src.model_registry import build_model

DEVICE = torch.device("cuda")
OUT = ROOT / "results/dpl_training_pilot_20260801/j3_nz1"
CKPT = ROOT / "results/dpl_training_pilot_20260801/h1_v2/checkpoints/newzealand1"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module; spec.loader.exec_module(module)
    return module


ROUND3 = load_module(ROOT / "scripts/diagnostics/dpl_third_round_diagnostics.py", "j3_round3")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def step(P, T, PET, s1max, sfc_frac, m, a, b, tcbf, S1):
    qse = torch.minimum(torch.clamp(saturation_1(P, S1, s1max), min=0.0), P)
    ea = torch.clamp(evap_6(m, sfc_frac, S1, s1max, PET) + evap_5(m, S1, s1max, PET), min=0.0)
    qss = torch.clamp(interflow_9(S1, a, sfc_frac * s1max, b), min=0.0)
    qbf = torch.clamp(baseflow_1(tcbf, S1), min=0.0)
    available = torch.clamp(S1 + P - 1e-6, min=0.0)
    scale = torch.minimum(torch.ones_like(qse), available / (ea + qse + qss + qbf + 1e-6))
    ea, qse, qss, qbf = ea * scale, qse * scale, qss * scale, qbf * scale
    return torch.clamp(S1 + P - ea - qse - qss - qbf, min=1e-6), qse, ea, qss, qbf


STEP = torch.compile(step)


def parameters(branch: str, ids32: list[int]) -> tuple[torch.Tensor, dict[str, torch.Tensor], str]:
    all_ids = [int(value) for value in load_ids("data/531sub_id.txt")]
    attrs = CatchmentAttributeBuilder().build_normalized_attributes(all_ids, device="cuda", method="zscore")
    network = CatchmentParameterizer(attrs.shape[1], 6, hidden_dims=[256, 256], dropout=.05).to(DEVICE)
    checkpoint = CKPT / f"{branch}_epoch_020.pt"
    payload = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    network.load_state_dict(payload["parameterizer"]); network.eval()
    with torch.no_grad(): theta_all = network(attrs)
    lookup = {basin: index for index, basin in enumerate(all_ids)}
    theta = theta_all[torch.as_tensor([lookup[basin] for basin in ids32], device=DEVICE)]
    mapping = "linear" if branch == "B0" else "auto"
    hydro = build_model("newzealand1", DEVICE, warm_up=365, backend="compile", parameter_mapping=mapping)
    with torch.no_grad(): physical = hydro._descale_params(theta)
    return theta, physical, str(checkpoint)


def diagnose(x: torch.Tensor, physical: dict[str, torch.Tensor]) -> dict[str, float]:
    p = [physical[name] for name in ("s1max", "sfc_frac", "m", "a", "b", "tcbf")]
    state = create_initial_state(x.shape[1], 1, DEVICE, 1e-6)[0].to(dtype=x.dtype)
    states, active = [], []
    for t in range(730):
        before = state
        state, *_ = STEP(x[t, :, 0:1], x[t, :, 1:2], x[t, :, 2:3], *p, state)
        if t >= 365:
            states.append(before); active.append(before > p[1] * p[0])
    ratio = torch.stack(states) / p[0].view(1, -1, 1)
    return {"interflow_active_fraction": float(torch.stack(active).double().mean()),
            "s1_over_s1max_p05": float(ratio.quantile(.05)), "s1_over_s1max_p50": float(ratio.quantile(.5)),
            "s1_over_s1max_p95": float(ratio.quantile(.95)),
            "s1max_physical_p05": float(p[0].quantile(.05)), "s1max_physical_p50": float(p[0].quantile(.5)),
            "s1max_physical_p95": float(p[0].quantile(.95))}


def main() -> None:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    OUT.mkdir(parents=True, exist_ok=True)
    ids32 = [int(value) for value in ROUND3.ids32().cpu().tolist()]
    x, _ = ROUND3.dpl_data(); x = x.to(dtype=torch.float32)
    rows = []
    for branch in ("B0", "B1"):
        _theta, physical, checkpoint = parameters(branch, ids32)
        rows.append({"branch": branch, "checkpoint": checkpoint, "checkpoint_epoch": 20, **diagnose(x, physical)})
    write_csv(OUT / "j3_nz1_b0_b1_state_reachability.csv", rows)


if __name__ == "__main__":
    main()
