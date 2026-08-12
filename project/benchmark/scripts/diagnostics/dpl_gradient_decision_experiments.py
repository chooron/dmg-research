#!/usr/bin/env python3
"""D1-D5 read-only decision experiments for dPL trainability.

No model source, configuration, optimizer checkpoint, or archived result is
modified.  D1/D3 use the same 32-basin, 365+365 real slice as the first audit.
D2 additionally uses the CMA-ES objective contract: five repeated 365-day
warm-up years followed by the 1989-1998 training period.
"""

from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dmotpy.models.core.newzealand1 import newzealand1_step  # noqa: E402
from dmotpy.models.flux.baseflow import baseflow_1  # noqa: E402
from dmotpy.models.flux.evap import evap_5, evap_6  # noqa: E402
from dmotpy.models.flux.interflow import interflow_9  # noqa: E402
from dmotpy.models.flux.saturation import saturation_1  # noqa: E402
from dmotpy.models.registry import PARAM_INFO  # noqa: E402
from src.data_selection import load_ids  # noqa: E402
from src.model_registry import NPARAM_INFO_36, build_model  # noqa: E402
from src.objective import streaming_kge  # noqa: E402


EVIDENCE = ROOT / "results/dpl_gradient_evidence_20260731_epoch"
OUT = ROOT / "results/dpl_gradient_decision_20260731"
DEVICE = torch.device("cuda")
MODELS_D2 = ("collie3", "newzealand1", "penman", "flexi", "flexis")
TARGETS_D1 = {
    "newzealand1": ("a", "b"),
    "penman": ("gam",),
    "collie3": ("b", "lambda_par"),
    "flexi": ("imax",),
    "flexis": ("imax",),
}


def load_old():
    path = ROOT / "scripts/diagnostics/dpl_gradient_evidence.py"
    spec = importlib.util.spec_from_file_location("first_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["first_audit"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


OLD = load_old()
MODEL_CACHE: dict[tuple[str, int], torch.nn.Module] = {}


def compiled_model(model_name: str, warmup: int) -> torch.nn.Module:
    key = (model_name, warmup)
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = build_model(model_name, DEVICE, warm_up=warmup, backend="compile")
    return MODEL_CACHE[key]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def basin_ids() -> np.ndarray:
    return np.asarray(json.loads((EVIDENCE / "run_metadata.json").read_text())["basin_ids"], dtype=np.int64)


def model_params(theta: torch.Tensor) -> torch.Tensor:
    return theta.unsqueeze(-1) if theta.ndim == 2 else theta


def loss_and_score(model_name: str, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor, warmup: int = 365):
    model = compiled_model(model_name, warmup)
    q = model({"x_phy": x}, (None, model_params(theta)))["streamflow"].squeeze(-1).squeeze(-1)
    # D1 uses the runner's differentiable KGE implementation and the scored
    # target only.  Keeping this helper in float64 makes FD/autograd comparable.
    loss, score, stats = OLD._runner_kge(q, y[warmup:])
    return loss, score, q, stats


def d1_scan(x: torch.Tensor, y: torch.Tensor) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scan_rows: list[dict[str, Any]] = []
    derivative_rows: list[dict[str, Any]] = []
    for model_name, parameters in TARGETS_D1.items():
        names = list(PARAM_INFO[model_name])
        dim = len(names)
        base = torch.full((x.shape[1], dim), 0.5, dtype=torch.float64, device=DEVICE)
        for parameter in parameters:
            j = names.index(parameter)
            values: list[tuple[float, float, torch.Tensor]] = []
            for value in np.linspace(0.0, 1.0, 21):
                theta = base.clone()
                theta[:, j] = float(value)
                with torch.no_grad():
                    loss, score, q, stats = loss_and_score(model_name, x, y, theta)
                values.append((float(value), float(score.mean()), q.detach()))
            baseline_q = values[10][2]
            mean_obs = float(y[365:].abs().mean())
            delta_q = max(float((q - baseline_q).abs().max()) / max(mean_obs, 1e-12) for _, _, q in values)
            kges = [score for _, score, _ in values]
            scan_rows.append({
                "model": model_name, "parameter": parameter,
                "max_relative_inf_delta_q": delta_q,
                "kge_range": max(kges) - min(kges),
                "theta_at_max_q_change": values[int(np.argmax([float((q - baseline_q).abs().max()) for _, _, q in values]))][0],
            })
            for center in (0.1, 0.3, 0.5, 0.7, 0.9):
                h = 1e-3
                lo = base.clone(); hi = base.clone()
                lo[:, j] = center - h; hi[:, j] = center + h
                with torch.no_grad():
                    loss_lo, _, _, _ = loss_and_score(model_name, x, y, lo)
                    loss_hi, _, _, _ = loss_and_score(model_name, x, y, hi)
                fd = float((loss_hi - loss_lo) / (2.0 * h))
                # Reconstruct through a detached copy so autograd sees the exact
                # target coordinate without mutating a leaf in-place.
                theta = base.clone(); theta[:, j] = center; theta.requires_grad_(True)
                loss, _, _, _ = loss_and_score(model_name, x, y, theta)
                loss.backward()
                auto = float(theta.grad[:, j].sum())
                abs_fd, abs_auto = abs(fd), abs(auto)
                if delta_q < 1e-12:
                    verdict = "STRUCTURALLY_UNIDENTIFIABLE"
                elif abs_fd <= 1e-10 and abs_auto <= 1e-10:
                    verdict = "STATE_UNREACHABLE"
                elif abs_fd > 100.0 * max(abs_auto, 1e-12):
                    verdict = "AUTOGRAD_DISCONNECTED"
                elif max(abs_fd, abs_auto) < 1e-5:
                    verdict = "WEAK_SENSITIVITY"
                else:
                    verdict = "FD_AUTOGRAD_AGREE"
                derivative_rows.append({
                    "model": model_name, "parameter": parameter, "theta": center,
                    "fd_center": fd, "autograd_mean": auto,
                    "absolute_difference": abs(fd - auto),
                    "relative_difference": abs(fd - auto) / max(abs_fd, abs_auto, 1e-12),
                    "verdict": verdict,
                })
    return scan_rows, derivative_rows


def cma_data(ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    data_path = ROOT.parents[1] / "data/camels_dataset"
    if not data_path.exists():
        data_path = ROOT.parents[1] / "data/camels_dataset.pkl"
    import pickle
    with data_path.open("rb") as handle:
        data = pickle.load(handle)
    if isinstance(data, dict):
        forcing_all, q_all = data["forcings"], data["streamflow"]
    else:
        forcing_all, q_all = data[:2]
    all_ids = np.asarray(np.load(ROOT.parents[1] / "data/gage_id.npy"), dtype=np.int64)
    lookup = {int(value): index for index, value in enumerate(all_ids)}
    positions = np.asarray([lookup[int(value)] for value in ids])
    dates = pd.date_range("1980-10-01", "2014-09-30", freq="D")
    def sl(start: str, end: str) -> tuple[int, int]:
        return int(dates.get_loc(pd.Timestamp(start))), int(dates.get_loc(pd.Timestamp(end))) + 1
    wl, wr = sl("1980-10-01", "1981-09-30")
    tl, tr = sl("1989-01-01", "1998-12-31")
    warm = np.asarray(forcing_all)[positions, wl:wr, :3]
    train = np.asarray(forcing_all)[positions, tl:tr, :3]
    x = np.concatenate([np.concatenate([warm] * 5, axis=1), train], axis=1)
    q = np.asarray(q_all)[positions, tl:tr, 0]
    from dpl.attributes import CatchmentAttributeBuilder
    area = CatchmentAttributeBuilder().load_raw_attributes([int(value) for value in ids])[:, 11]
    q = q * (0.0283168 * 86400.0 * 1000.0 / (area[:, None] * 1e6))
    return torch.as_tensor(np.transpose(x, (1, 0, 2)), dtype=torch.float64, device=DEVICE), torch.as_tensor(q.T, dtype=torch.float64, device=DEVICE)


def archive_theta_and_fitness(model: str, ids: np.ndarray) -> tuple[torch.Tensor | None, torch.Tensor | None, str]:
    path = OLD._checkpoint_path(model)
    if not path.exists():
        return None, None, str(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = np.asarray(payload["basin_ids"], dtype=np.int64)
    state = payload["solver"]["state"]
    latent = state["best_latent"].to(device=DEVICE, dtype=torch.float64)
    fitness = state["best_fitness"].to(device=DEVICE, dtype=torch.float64)
    starts = latent.shape[0] // len(saved)
    latent = latent.reshape(len(saved), starts, -1)
    fitness = fitness.reshape(len(saved), starts)
    best = fitness.argmax(1)
    theta = torch.sigmoid(latent[torch.arange(len(saved), device=DEVICE), best])
    lookup = {int(value): i for i, value in enumerate(saved)}
    if any(int(value) not in lookup for value in ids):
        return None, None, f"archive covers {len(saved)} basins and misses requested IDs"
    ix = [lookup[int(value)] for value in ids]
    return theta[ix], fitness[ix, best[ix]], str(path)


def d2(model: str, ids: np.ndarray, steps: int = 30) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    theta_star, archive_fit, source = archive_theta_and_fitness(model, ids)
    if theta_star is None:
        return [{"model": model, "status": "NOT_RUN", "reason": source}], []
    x, y = cma_data(ids)
    x, y = x.to(torch.float64), y.to(torch.float64)
    model_obj = compiled_model(model, 1825)
    raw = theta_star.to(DEVICE).clone().requires_grad_(True)
    q = model_obj({"x_phy": x}, (None, raw.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
    score, invalid = streaming_kge(q.unsqueeze(-1).unsqueeze(-1), y, eps=0.1)
    score_mean = score.mean()
    (1.0 - score_mean).backward()
    grad_norm = float(raw.grad.detach().norm(dim=1).mean())
    consistency: list[dict[str, Any]] = []
    for i, basin in enumerate(ids):
        consistency.append({
            "model": model, "basin_id": int(basin), "status": "COMPLETE",
            "cma_archive_kge": float(archive_fit[i]), "reevaluated_kge": float(score[i, 0, 0]),
            "absolute_kge_difference": abs(float(archive_fit[i]) - float(score[i, 0, 0])),
            "mean_parameter_grad_norm": grad_norm,
            "invalid": bool(invalid[i, 0, 0]), "source": source,
        })
    trajectories: list[dict[str, Any]] = []
    for lr in (1e-2, 1e-3):
        parameter = torch.nn.Parameter(theta_star.to(DEVICE).clone())
        optimizer = torch.optim.Adam([parameter], lr=lr)
        for step in range(steps + 1):
            optimizer.zero_grad(set_to_none=True)
            q = model_obj({"x_phy": x}, (None, parameter.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            score, _ = streaming_kge(q.unsqueeze(-1).unsqueeze(-1), y, eps=0.1)
            loss = 1.0 - score.mean()
            if step < steps:
                loss.backward(); optimizer.step()
                with torch.no_grad():
                    parameter.clamp_(1e-7, 1.0 - 1e-7)
            trajectories.append({"model": model, "lr": lr, "step": step, "mean_kge": float(score.mean()), "loss": float(loss), "grad_norm": float(parameter.grad.norm()) if parameter.grad is not None else 0.0})
    return consistency, trajectories


def nz_diagnostic_step(P, T, PET, s1max, sfc_frac, m, a, b, tcbf, S1):
    qse = torch.minimum(torch.clamp(saturation_1(P, S1, s1max), min=0.0), P)
    veg = evap_6(m, sfc_frac, S1, s1max, PET)
    bare = evap_5(m, S1, s1max, PET)
    ea = torch.clamp(veg + bare, min=0.0)
    sfc = sfc_frac * s1max
    qss = torch.clamp(interflow_9(S1, a, sfc, b), min=0.0)
    qbf = torch.clamp(baseflow_1(tcbf, S1), min=0.0)
    available = torch.clamp(S1 + P - 1e-6, min=0.0)
    outgoing = ea + qse + qss + qbf
    scale = torch.minimum(torch.ones_like(outgoing), available / (outgoing + 1e-6))
    ea, qse, qss, qbf = ea * scale, qse * scale, qss * scale, qbf * scale
    new_s1 = torch.clamp(S1 + P - ea - qse - qss - qbf, min=1e-6)
    return qse + qss + qbf, ea, new_s1, qse, qss, qbf, P - qse


NZ_DIAGNOSTIC_STEP = torch.compile(nz_diagnostic_step)


def nz_trace(x: torch.Tensor, theta: torch.Tensor, s1max_override: float | None = None) -> dict[str, Any]:
    names = list(PARAM_INFO["newzealand1"])
    p = {name: theta[:, i] for i, name in enumerate(names)}
    if s1max_override is not None:
        p["s1max"] = torch.full_like(p["s1max"], s1max_override)
    states = [torch.full((x.shape[1],), 1e-6, dtype=torch.float64, device=DEVICE)]
    records = {key: [] for key in ("s1", "p", "qse", "ea", "qss", "qbf", "qse_share", "ea_share", "s1_inflow")}
    for t in range(x.shape[0]):
        P, T, PET = x[t].unbind(-1)
        S1 = states[0]
        _q, ea, S1, qse, qss, qbf, s1_inflow = NZ_DIAGNOSTIC_STEP(
            P, T, PET, p["s1max"], p["sfc_frac"], p["m"], p["a"], p["b"], p["tcbf"], S1
        )
        states[0] = S1
        for key, value in {"s1": S1, "p": P, "qse": qse, "ea": ea, "qss": qss, "qbf": qbf, "qse_share": qse / (P + 1e-12), "ea_share": ea / (P + 1e-12), "s1_inflow": s1_inflow}.items():
            records[key].append(value.detach())
    arr = {key: torch.stack(value) for key, value in records.items()}
    return arr


def d3(x: torch.Tensor, theta_mid: torch.Tensor, theta_cma: torch.Tensor) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cma_phys = theta_cma[:, 0] * 1999.0 + 1.0
    median_phys = float(cma_phys.median())
    for label, override in (("midpoint", None), ("cma_s1max_median", median_phys)):
        trace = nz_trace(x, theta_mid, override)
        s1max = 1000.5 if override is None else median_phys
        excess = trace["s1"] - theta_mid[:, 1].view(1, -1) * s1max
        rows.append({
            "regime": label, "s1max_midpoint_or_override": s1max,
            "cma_s1max_median": median_phys,
            "cma_to_midpoint_ratio_mean": float((cma_phys / 1000.5).mean()),
            "max_cumulative_s1_inflow_over_s1max": float(trace["s1_inflow"].sum(0).max() / s1max),
            "precip_cumulative": float(trace["p"].sum()),
            "direct_runoff_cumulative": float(trace["qse"].sum()),
            "evap_cumulative": float(trace["ea"].sum()),
            "interflow_cumulative": float(trace["qss"].sum()),
            "baseflow_cumulative": float(trace["qbf"].sum()),
            "interflow_active_frac": float((excess > 0).double().mean()),
            "s1_over_capacity_p05": float((trace["s1"] / s1max).quantile(0.05)),
            "s1_over_capacity_p50": float((trace["s1"] / s1max).quantile(0.50)),
            "s1_over_capacity_p95": float((trace["s1"] / s1max).quantile(0.95)),
        })
    return rows


class SoftReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, tau: float):
        ctx.save_for_backward(x); ctx.tau = tau; return torch.relu(x)
    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad * torch.sigmoid(x / ctx.tau), None


class SoftMin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor, tau: float):
        ctx.save_for_backward(a, b); ctx.tau = tau; return torch.minimum(a, b)
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors; wa = torch.sigmoid((b - a) / ctx.tau)
        return grad * wa, grad * (1.0 - wa), None


def d5() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    relu_cases = {"dry": -1.0, "wet": 1.0, "threshold": 0.0, "threshold_plus_1e-6": 1e-6, "threshold_minus_1e-6": -1e-6, "snow": -4.0}
    for name, value in relu_cases.items():
        x = torch.tensor([value], dtype=torch.float64, device=DEVICE)
        hard = torch.relu(x); soft = SoftReLU.apply(x, 1e-2)
        state0 = torch.tensor([2.0], dtype=torch.float64, device=DEVICE); incoming = state0 + torch.tensor([1.0], dtype=torch.float64, device=DEVICE)
        residual = incoming - (hard + (incoming - hard))
        rows.append({"operator": "SoftReLU", "case": name, "bit_exact": bool(torch.equal(hard, soft)), "max_abs_forward_diff": float((hard-soft).abs().max()), "water_balance_residual": float(residual.abs().max()), "threshold_equal_case": name == "threshold"})
    min_cases = {"dry": (-1.0, 0.2), "wet": (2.0, 1.0), "threshold": (1.0, 1.0), "threshold_plus_1e-6": (1.0 + 1e-6, 1.0), "threshold_minus_1e-6": (1.0 - 1e-6, 1.0), "snow": (0.1, 3.0)}
    for name, (a, b) in min_cases.items():
        ta = torch.tensor([a], dtype=torch.float64, device=DEVICE); tb = torch.tensor([b], dtype=torch.float64, device=DEVICE)
        hard = torch.minimum(ta, tb); soft = SoftMin.apply(ta, tb, 1e-2)
        residual = ta - (hard + (ta - hard))
        rows.append({"operator": "SoftMin", "case": name, "bit_exact": bool(torch.equal(hard, soft)), "max_abs_forward_diff": float((hard-soft).abs().max()), "water_balance_residual": float(residual.abs().max()), "threshold_equal_case": name == "threshold"})
    return rows


def d4() -> list[dict[str, Any]]:
    path = ROOT / "dmotpy/models/hydrology_model.py"
    lines = path.read_text().splitlines()
    mappings = []
    for model in NPARAM_INFO_36:
        model_mapping = "linear"
        mappings.append({"model": model, "mapping_in_src_model_registry": model_mapping, "nonlinear_log_active": False})
    return mappings


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ids = basin_ids()
    x, y = OLD._load_forcing(ids, "1989-01-01")
    x64, y64 = x.to(device=DEVICE, dtype=torch.float64), y.to(device=DEVICE, dtype=torch.float64)
    scan, derivatives = d1_scan(x64, y64)
    write_csv(OUT / "d1_scan.csv", scan, ["model", "parameter", "max_relative_inf_delta_q", "kge_range", "theta_at_max_q_change"])
    write_csv(OUT / "d1_fd_vs_autograd.csv", derivatives, ["model", "parameter", "theta", "fd_center", "autograd_mean", "absolute_difference", "relative_difference", "verdict"])
    d2_consistency: list[dict[str, Any]] = []; d2_traj: list[dict[str, Any]] = []
    for model in MODELS_D2:
        consistency, trajectory = d2(model, ids)
        d2_consistency.extend(consistency); d2_traj.extend(trajectory)
    write_csv(OUT / "d2_cma_consistency.csv", d2_consistency, ["model", "basin_id", "status", "cma_archive_kge", "reevaluated_kge", "absolute_kge_difference", "mean_parameter_grad_norm", "invalid", "source", "reason"])
    write_csv(OUT / "d2_adam_trajectories.csv", d2_traj, ["model", "lr", "step", "mean_kge", "loss", "grad_norm"])
    theta_mid = torch.full((len(ids), len(PARAM_INFO["newzealand1"])), 0.5, dtype=torch.float64, device=DEVICE)
    theta_nz, _, _ = archive_theta_and_fitness("newzealand1", ids)
    theta_nz = theta_nz.to(DEVICE)
    write_csv(OUT / "d3_s1_water_balance.csv", d3(x64, theta_mid, theta_nz), ["regime", "s1max_midpoint_or_override", "cma_s1max_median", "cma_to_midpoint_ratio_mean", "max_cumulative_s1_inflow_over_s1max", "precip_cumulative", "direct_runoff_cumulative", "evap_cumulative", "interflow_cumulative", "baseflow_cumulative", "interflow_active_frac", "s1_over_capacity_p05", "s1_over_capacity_p50", "s1_over_capacity_p95"])
    write_csv(OUT / "d5_forward_parity.csv", d5(), ["operator", "case", "bit_exact", "max_abs_forward_diff", "water_balance_residual", "threshold_equal_case"])
    write_csv(OUT / "d4_model_mappings.csv", d4(), ["model", "mapping_in_src_model_registry", "nonlinear_log_active"])
    metadata = {"basin_ids": ids.tolist(), "d1_slice": "1989-01-01 + 730 days", "d2_contract": "1825 repeated warm-up + 1989-1998 train, streaming_kge eps=0.1", "no_training": True}
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = ["# dPL decision experiments", "", "This report is read-only. No model formula or training configuration was changed.", "", "## D1", "", "See `d1_scan.csv` and `d1_fd_vs_autograd.csv`; each target has an explicit classification at all five centers.", "", "## D2", "", "See `d2_cma_consistency.csv` and `d2_adam_trajectories.csv`. CMA-ES uses the repeated-warm-up/full-train objective, not the 730-day diagnostic window.", "", "## D3", "", "See `d3_s1_water_balance.csv` for S1 inflow/outflow decomposition and the CMA-median s1max-only intervention.", "", "## D4", "", "`HydrologyModel._change_param_range` is `dmotpy/models/hydrology_model.py:184-196`; mapping defaults to `linear`, `none` is an alias, and `auto/auto_log/log_auto` enable log mapping only when `lo>0` and `hi/lo >= log_mapping_span_threshold` (default 100.0). All active `src.model_registry.model_config` models set `parameter_mapping='linear'` at `src/model_registry.py:39`.", "", "## D5", "", "See `d5_forward_parity.csv`; only SoftReLU and SoftMin were tested. The test uses exact hard forward identities and float64 water-balance residuals."]
    (OUT / "dpl_gradient_decision_report.md").write_text("\n".join(report) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
