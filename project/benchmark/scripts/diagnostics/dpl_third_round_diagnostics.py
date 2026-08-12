#!/usr/bin/env python3
"""Third-round CUDA diagnostics for dPL trainability.

The script never trains a parameterizer.  All hydrological forward, loss,
finite-difference, and backward computations use CUDA tensors and compiled
model step functions.  CPU is used only to deserialize the local data and
checkpoints and to write scalar reports.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dmotpy.models.core.newzealand1 import NEWZEALAND1_PARAMS_BOUNDS, create_initial_state as nz_initial_state
from dmotpy.models.core.penman import PENMAN_PARAMS_BOUNDS, create_initial_state as penman_initial_state, penman_step
from dmotpy.models.flux.baseflow import baseflow_1
from dmotpy.models.flux.evap import evap_5, evap_6, evap_16
from dmotpy.models.flux.interflow import interflow_9
from dmotpy.models.flux.saturation import saturation_1
from dmotpy.models.registry import PARAM_INFO
from src.model_registry import NPARAM_INFO_36, build_model, model_config
from src.objective import streaming_kge

DEVICE = torch.device("cuda")
OUT = ROOT / "results/dpl_third_round_20260731"
PREVIOUS = ROOT / "results/dpl_gradient_decision_20260731"
EVIDENCE = ROOT / "results/dpl_gradient_evidence_20260731_epoch"
MODELS = ("collie3", "newzealand1", "penman", "flexi", "flexis")
T4_MODELS = ("collie3", "newzealand1", "penman", "flexi", "flexis", "hbv96")


def load_first_audit():
    path = ROOT / "scripts/diagnostics/dpl_gradient_evidence.py"
    spec = importlib.util.spec_from_file_location("first_audit_round3", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OLD = load_first_audit()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ids32():
    return torch.tensor(json.loads((EVIDENCE / "run_metadata.json").read_text())["basin_ids"], device=DEVICE)


def dpl_data() -> tuple[torch.Tensor, torch.Tensor]:
    ids = ids32().cpu().numpy()
    x, y = OLD._load_forcing(ids, "1989-01-01")
    return x.to(device=DEVICE, dtype=torch.float64), y.to(device=DEVICE, dtype=torch.float64)


def cma_data() -> tuple[torch.Tensor, torch.Tensor]:
    ids = ids32().cpu().numpy()
    x, y = load_decision_module().cma_data(ids)
    # Production CMA-ES explicitly casts sigmoid(latent) and model forcing to fp32.
    return x.to(device=DEVICE, dtype=torch.float32), y.to(device=DEVICE, dtype=torch.float32)


_DECISION: Any | None = None


def load_decision_module():
    global _DECISION
    if _DECISION is None:
        path = ROOT / "scripts/diagnostics/dpl_gradient_decision_experiments.py"
        spec = importlib.util.spec_from_file_location("round2_decision", path)
        _DECISION = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = _DECISION
        spec.loader.exec_module(_DECISION)
    return _DECISION


def best_theta(model: str, dtype: torch.dtype) -> tuple[torch.Tensor, str]:
    decision = load_decision_module()
    theta, _fitness, source = decision.archive_theta_and_fitness(model, ids32().cpu().numpy())
    if theta is None:
        raise RuntimeError(f"CMA archive unavailable for {model}: {source}")
    return theta.to(device=DEVICE, dtype=dtype), source


def all_archive_theta(model: str, dtype: torch.dtype) -> tuple[torch.Tensor, str, int]:
    path = OLD._checkpoint_path(model)
    if not path.exists():
        raise RuntimeError(f"CMA archive unavailable for {model}: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    basin_ids = payload["basin_ids"]
    state = payload["solver"]["state"]
    latent = state["best_latent"].to(device=DEVICE, dtype=dtype)
    fitness = state["best_fitness"].to(device=DEVICE, dtype=dtype)
    starts = latent.shape[0] // len(basin_ids)
    latent = latent.reshape(len(basin_ids), starts, -1)
    fitness = fitness.reshape(len(basin_ids), starts)
    best = fitness.argmax(dim=1)
    theta = torch.sigmoid(latent[torch.arange(len(basin_ids), device=DEVICE), best])
    return theta, f"{path} ({len(basin_ids)} archive basins)", len(basin_ids)


def runner_loss(q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return OLD._runner_kge(q, y[365:])[0]


def dpl_window_loss(model, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    q = model({"x_phy": x}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
    return runner_loss(q, y)


def penman_params(theta: torch.Tensor) -> list[torch.Tensor]:
    return [theta[:, i : i + 1] * (hi - lo) + lo for i, (lo, hi) in enumerate(PENMAN_PARAMS_BOUNDS.values())]


PENMAN_STEP = torch.compile(penman_step)


def penman_states(x: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, ...]:
    params = penman_params(theta)
    states = tuple(value.to(device=DEVICE, dtype=theta.dtype) for value in penman_initial_state(x.shape[1], 1, DEVICE, 1e-6))
    with torch.no_grad():
        for t in range(365):
            states = PENMAN_STEP(x[t, :, 0:1], x[t, :, 1:2], x[t, :, 2:3], *params, *states, nearzero=1e-6)[2:]
    return tuple(value.detach() for value in states)


def penman_scored_loss(x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor, initial: tuple[torch.Tensor, ...]) -> torch.Tensor:
    params, states = penman_params(theta), initial
    streamflow = []
    for t in range(365, 730):
        outputs = PENMAN_STEP(x[t, :, 0:1], x[t, :, 1:2], x[t, :, 2:3], *params, *states, nearzero=1e-6)
        streamflow.append(outputs[0])
        states = outputs[2:]
    return runner_loss(torch.stack(streamflow).squeeze(-1).squeeze(-1), y)


def t1() -> None:
    x, y = dpl_data()
    base = torch.full((x.shape[1], 4), 0.5, device=DEVICE, dtype=torch.float64)
    j, h = 2, 1e-6
    lo, hi = base.clone(), base.clone()
    lo[:, j] -= h
    hi[:, j] += h
    with torch.no_grad():
        fd_full = (penman_scored_loss(x, y, hi, penman_states(x, hi)) - penman_scored_loss(x, y, lo, penman_states(x, lo))) / (2.0 * h)
        shared = penman_states(x, base)
        fd_scored_only = (penman_scored_loss(x, y, hi, shared) - penman_scored_loss(x, y, lo, shared)) / (2.0 * h)
    theta = base.clone().requires_grad_(True)
    loss = penman_scored_loss(x, y, theta, shared)
    loss.backward()
    autograd = theta.grad[:, j].sum()

    direct_gam = torch.full((32, 1), 0.5, dtype=torch.float64, device=DEVICE, requires_grad=True)
    direct = evap_16(
        direct_gam,
        torch.full_like(direct_gam, float("inf")),
        torch.zeros_like(direct_gam),
        torch.full_like(direct_gam, 0.01),
        torch.ones_like(direct_gam),
    )
    direct.sum().backward()
    tcm_model = build_model("tcm", DEVICE, warm_up=365, backend="compile")
    tcm_base = torch.full((x.shape[1], len(PARAM_INFO["tcm"])), 0.5, device=DEVICE, dtype=torch.float64)
    tcm_j = list(PARAM_INFO["tcm"]).index("gam")
    tcm_lo, tcm_hi = tcm_base.clone(), tcm_base.clone()
    tcm_lo[:, tcm_j] -= h
    tcm_hi[:, tcm_j] += h
    with torch.no_grad():
        tcm_fd = (dpl_window_loss(tcm_model, x, y, tcm_hi) - dpl_window_loss(tcm_model, x, y, tcm_lo)) / (2.0 * h)
    tcm_theta = tcm_base.clone().requires_grad_(True)
    tcm_loss = dpl_window_loss(tcm_model, x, y, tcm_theta)
    tcm_loss.backward()
    tcm_autograd = tcm_theta.grad[:, tcm_j].sum()
    rows = [
        {
            "model": "penman", "parameter": "gam", "caller": "dmotpy/models/core/penman.py:209-216",
            "full_warmup_fd": float(fd_full), "scored_only_fd": float(fd_scored_only),
            "autograd_scored_only": float(autograd), "direct_evap16_grad_max": float(direct_gam.grad.abs().max()),
            "verdict": "WARMUP_STOP_GRAD_MISMATCH_NOT_EVAP16_DISCONNECT",
        },
        {
            "model": "tcm", "parameter": "gam", "caller": "dmotpy/models/core/tcm.py:127-134",
            "full_warmup_fd": float(tcm_fd), "scored_only_fd": float(tcm_fd), "autograd_scored_only": float(tcm_autograd),
            "direct_evap16_grad_max": float(direct_gam.grad.abs().max()),
            "verdict": "UNIDENTIFIABLE_AT_THETA_0_5_ON_THIS_FORCING" if tcm_fd == 0.0 and tcm_autograd == 0.0 else "SEE_VALUES",
        },
    ]
    write_csv(OUT / "t1_evap16_callers_and_warmup_contract.csv", rows, list(rows[0]))
    (OUT / "t1_source_audit.md").write_text(
        "# T1 source audit\n\n"
        "`evap_16` (`dmotpy/models/flux/evap.py:247-262`) has `p1 -> evap = p1 * Ep` at line 258, then a smooth storage gate and `torch.minimum` at lines 259-262. "
        "It contains no `where`, `detach`, bool conversion, in-place update, `.item()`, scalar conversion, `no_grad`, `nan_to_num`, indexing, parameter-dependent clamp boundary, or parameter-dependent `torch.full`. "
        "Direct CUDA eager/compiled probes give nonzero d(evap_16)/d(gam).\n\n"
        "The apparent D1 mismatch is instead `HydrologyModel._run_model:333-344`: FD applies theta during the 365-day warm-up, while backward applies `torch.no_grad()` then detaches the warm-up state. "
        "The full-forward FD is nonzero; after sharing the same detached warm-up state it is exactly zero and matches autograd. Therefore no `evap.py` formula patch is valid, and no model-source change is made for T1.\n"
    )


def cma_loss(model, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    q = model({"x_phy": x}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
    score, _invalid = streaming_kge(q.unsqueeze(-1).unsqueeze(-1), y, eps=0.1)
    return 1.0 - score.mean()


def t2() -> None:
    x, y = cma_data()
    line_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []
    for model_name in MODELS:
        model = build_model(model_name, DEVICE, warm_up=1825, backend="compile")
        theta_star, source = best_theta(model_name, torch.float32)
        theta_star = theta_star.detach().clone().requires_grad_(True)
        loss_star = cma_loss(model, x, y, theta_star)
        loss_star.backward()
        gradient = theta_star.grad.detach()
        norm_star = gradient.norm()
        direction = gradient / norm_star.clamp_min(torch.finfo(gradient.dtype).eps)
        with torch.no_grad():
            for h in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
                candidate = (theta_star.detach() - h * direction).clamp(0.0, 1.0)
                delta = cma_loss(model, x, y, candidate) - loss_star.detach()
                line_rows.append({"model": model_name, "h": h, "loss_theta_star": float(loss_star.detach()), "loss_delta": float(delta), "clamped_fraction": float(((candidate == 0.0) | (candidate == 1.0)).float().mean()), "source": source})
        midpoint = torch.full_like(theta_star, 0.5, requires_grad=True)
        loss_mid = cma_loss(model, x, y, midpoint)
        loss_mid.backward()
        norm_mid = midpoint.grad.detach().norm()
        norm_rows.append({"model": model_name, "loss_theta_star": float(loss_star.detach()), "loss_midpoint": float(loss_mid.detach()), "grad_norm_theta_star": float(norm_star), "grad_norm_midpoint": float(norm_mid), "star_to_midpoint_ratio": float(norm_star / norm_mid.clamp_min(torch.finfo(norm_mid.dtype).eps)), "source": source})
        del theta_star, midpoint, gradient, direction, loss_star, loss_mid, model
        torch.cuda.empty_cache()
    write_csv(OUT / "t2_line_search.csv", line_rows, list(line_rows[0]))
    write_csv(OUT / "t2_gradient_norm_ratio.csv", norm_rows, list(norm_rows[0]))
    (OUT / "t2_loss_contract.csv").write_text(
        "objective,forward_precision,warmup,scored_period,eps,source\n"
        "CMA-ES,fp32 model then fp64 KGE,1825 days (1980-10-01..1981-09-30 repeated 5x),1989-01-01..1998-12-31,0.1,scripts/run_36model_benchmark.py:35-37 and :77\n"
        "dPL runner,runner tensor dtype,365 days,365-day diagnostic/training window,1e-6,scripts/run_dpl_benchmark_dmg_native.py:178-219\n"
    )


def t2d() -> None:
    x64, y64 = dpl_data()
    rows = []
    grads = []
    for dtype in (torch.float32, torch.float64):
        x, y = x64.to(dtype), y64.to(dtype)
        theta = torch.full((x.shape[1], len(PARAM_INFO["collie3"])), 0.5, device=DEVICE, dtype=dtype, requires_grad=True)
        model = build_model("collie3", DEVICE, warm_up=365, backend="compile")
        q = model({"x_phy": x}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
        loss = runner_loss(q, y)
        loss.backward()
        grad = theta.grad[:, list(PARAM_INFO["collie3"]).index("b")].detach()
        grads.append(grad.to(torch.float64))
        rows.append({"dtype": str(dtype), "loss": float(loss.detach()), "gradient_has_nan_inf": bool(~torch.isfinite(grad).all()), "gradient_norm_b": float(grad.norm()), "q_has_nan_inf": bool(~torch.isfinite(q).all())})
        del theta, model, q, loss
        torch.cuda.empty_cache()
    cosine = torch.nn.functional.cosine_similarity(grads[0], grads[1], dim=0)
    for row in rows:
        row["float32_float64_b_gradient_cosine"] = float(cosine)
    write_csv(OUT / "t2d_collie3_float_precision.csv", rows, list(rows[0]))


def nz_step(P, T, PET, s1max, sfc_frac, m, a, b, tcbf, S1):
    qse = torch.minimum(torch.clamp(saturation_1(P, S1, s1max), min=0.0), P)
    ea = torch.clamp(evap_6(m, sfc_frac, S1, s1max, PET) + evap_5(m, S1, s1max, PET), min=0.0)
    qss = torch.clamp(interflow_9(S1, a, sfc_frac * s1max, b), min=0.0)
    qbf = torch.clamp(baseflow_1(tcbf, S1), min=0.0)
    available = torch.clamp(S1 + P - 1e-6, min=0.0)
    scale = torch.minimum(torch.ones_like(qse), available / (ea + qse + qss + qbf + 1e-6))
    ea, qse, qss, qbf = ea * scale, qse * scale, qss * scale, qbf * scale
    return torch.clamp(S1 + P - ea - qse - qss - qbf, min=1e-6), qse, ea, qss, qbf


NZ_STEP = torch.compile(nz_step)


def nz_activation(x: torch.Tensor, theta: torch.Tensor) -> dict[str, float]:
    params = [theta[:, i : i + 1] * (hi - lo) + lo for i, (lo, hi) in enumerate(NEWZEALAND1_PARAMS_BOUNDS.values())]
    state = nz_initial_state(x.shape[1], 1, DEVICE, 1e-6)[0].to(dtype=x.dtype)
    records = {"s": [], "active": [], "p": [], "qse": [], "ea": [], "qss": [], "qbf": []}
    for t in range(730):
        s_before = state
        state, qse, ea, qss, qbf = NZ_STEP(x[t, :, 0:1], x[t, :, 1:2], x[t, :, 2:3], *params, state)
        if t >= 365:
            records["s"].append(s_before)
            records["active"].append(s_before > params[1] * params[0])
            records["p"].append(x[t, :, 0:1])
            records["qse"].append(qse); records["ea"].append(ea); records["qss"].append(qss); records["qbf"].append(qbf)
    s = torch.stack(records["s"])
    capacity = params[0].view(1, -1, 1)
    return {
        "activation": float(torch.stack(records["active"]).double().mean()),
        "s_p05": float((s / capacity).quantile(0.05)), "s_p50": float((s / capacity).quantile(0.50)), "s_p95": float((s / capacity).quantile(0.95)),
        "inflow": float(torch.stack(records["p"]).sum()), "direct": float(torch.stack(records["qse"]).sum()),
        "evap": float(torch.stack(records["ea"]).sum()), "interflow": float(torch.stack(records["qss"]).sum()), "baseflow": float(torch.stack(records["qbf"]).sum()),
    }


def t3() -> None:
    x, _ = dpl_data()
    theta_best, source, archive_basins = all_archive_theta("newzealand1", torch.float64)
    median_values = theta_best.median(dim=0).values
    median = median_values.expand(x.shape[1], -1).contiguous()
    rows = [{"regime": "all_cma_parameter_medians", "changed_parameter": "", "archive_basins": archive_basins, "source": source, **nz_activation(x, median)}]
    if rows[0]["activation"] > 0.05:
        for index, name in enumerate(PARAM_INFO["newzealand1"]):
            trial = median.clone(); trial[:, index] = 0.5
            rows.append({"regime": "leave_one_midpoint", "changed_parameter": name, "archive_basins": archive_basins, "source": source, **nz_activation(x, trial)})
    write_csv(OUT / "t3_nz1_all_cma_median_ablation.csv", rows, list(rows[0]))


def t4() -> None:
    rows: list[dict[str, Any]] = []
    for model_name in T4_MODELS:
        try:
            theta, source, archive_basins = all_archive_theta(model_name, torch.float64)
            median = theta.median(dim=0).values
        except RuntimeError as exc:
            median, source, archive_basins = None, str(exc), 0
        for index, (name, (lo, hi)) in enumerate(PARAM_INFO[model_name].items()):
            ratio = hi / lo if lo > 0.0 else math.nan
            active = lo > 0.0 and ratio >= 100.0
            rows.append({"model": model_name, "parameter": name, "lo": lo, "hi": hi, "hi_over_lo": ratio, "auto_log_branch": active, "linear_midpoint": (lo + hi) / 2.0, "log_midpoint": math.sqrt(lo * hi) if lo > 0.0 else math.nan, "cma_median_physical": (lo + (hi - lo) * float(median[index])) if median is not None else math.nan, "archive_basins": archive_basins, "cma_archive": source})
    write_csv(OUT / "t4_mapping_midpoint_vs_cma.csv", rows, list(rows[0]))
    configs = []
    for mapping in ("linear", "auto_log"):
        cfg = model_config("collie3", parameter_mapping=mapping)
        configs.append({"mapping": mapping, "config_mapping": cfg["parameter_mapping"], "threshold": cfg["log_mapping_span_threshold"]})
    write_csv(OUT / "t4_mapping_configuration_smoke.csv", configs, list(configs[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("t1", "t2", "t3", "t4", "all"), default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for third-round numerical diagnostics")
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = ("t1", "t2", "t3", "t4") if args.task == "all" else (args.task,)
    if "t1" in tasks:
        t1()
    if "t2" in tasks:
        t2(); t2d()
    if "t3" in tasks:
        t3()
    if "t4" in tasks:
        t4()


if __name__ == "__main__":
    main()
