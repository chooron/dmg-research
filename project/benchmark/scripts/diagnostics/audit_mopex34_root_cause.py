#!/usr/bin/env python3
"""Lightweight MOPEX3 -> MOPEX4 root-cause audit.

This audit intentionally uses CPU and four representative basins only. It
checks the call chain, lambda_i=0 structural equivalence, parameter-head
mapping, shared-trunk gradient interference, a fixed-common lambda sweep,
and a freeze-common/direct interception-only probe. It never modifies
production code or launches full-basin training.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

BENCHMARK = Path(__file__).resolve().parents[2]
REPO = BENCHMARK.parents[1]
sys.path[:0] = [str(REPO), str(BENCHMARK), str(BENCHMARK / "src"),
                str(BENCHMARK / "scripts" / "diagnostics")]

from dmotpy.models.hydrology_model import HydrologyModel
from dmotpy.models.registry import PARAM_INFO
from dmotpy.models.core.mopex3 import mopex3_step
from dmotpy.models.core.mopex4 import mopex4_step
from dmotpy.data_contract import add_calendar_forcing
from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from project.benchmark.scripts.run_dpl_benchmark_dmg_native import (
    load_camels_time_series, compute_differentiable_kge,
)
from project.benchmark.src.data_selection import load_ids
from project.benchmark.src.model_registry import model_config
from dmotpy.models.flux.mopex import mopex_training_context

OUT = BENCHMARK / "results" / "mopex45_phase_fix" / "root_cause_audit"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cpu")
WARMUP, SCORED, START = 365, 365, 1825
BASIN_IDX = [391, 373, 269, 530]
M3_TO_M4 = [0, 1, 2, 3, 6, 7, 8, 9]  # target indices for m3 [tcrit,ddf,s2max,tw,tu,se,s3max,tc]
M4_COMMON = [0, 1, 2, 3, 6, 7, 8, 9]
M4_INTERCEPTION = [4, 5]


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    with (OUT / name).open("w", newline="") as f:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def load_context():
    ids = [int(v) for v in load_ids("data/531sub_id.txt")]
    tx, ty, _, _ = load_camels_time_series(ids)
    x = torch.as_tensor(tx, dtype=torch.float32, device=DEVICE)
    y = torch.as_tensor(ty, dtype=torch.float32, device=DEVICE)
    dates = __import__("pandas").date_range("1980-10-01", "1995-09-30", freq="D")
    x, _ = add_calendar_forcing(x, dates, model_name="mopex4")
    b = BASIN_IDX
    return ids, x[:, b], y[:, b], b


def bounds(name):
    return list(PARAM_INFO[name].values())


def norm_to_phys(raw, name):
    bs = bounds(name)
    lo = torch.tensor([v[0] for v in bs], dtype=raw.dtype)
    hi = torch.tensor([v[1] for v in bs], dtype=raw.dtype)
    return lo + raw * (hi - lo)


def make_model(name, warmup=WARMUP, lambda_i=1.0):
    cfg = model_config(name, warm_up=warmup, backend="python", parameter_mapping="auto",
                       warmup_grad_mode="detach")
    if name == "mopex4":
        cfg.update(continuation_lambda_i=lambda_i, continuation_lambda_p=1.0,
                   continuation_beta=50.0)
    return HydrologyModel(cfg, device=DEVICE, backend="python").to(DEVICE)


def mapped_m3_network():
    path = BENCHMARK / "results/dpl_round13_20260805/auto100/checkpoints/mopex3/epoch_100.pt"
    net3 = CatchmentParameterizer(35, 8, hidden_dims=[256, 256], dropout=.05)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    net3.load_state_dict(payload["network"])
    net4 = CatchmentParameterizer(35, 10, hidden_dims=[256, 256], dropout=.05)
    with torch.no_grad():
        s3, s4 = net3.state_dict(), net4.state_dict()
        for key in ("net.0.weight", "net.0.bias", "net.1.weight", "net.1.bias",
                    "net.4.weight", "net.4.bias", "net.5.weight", "net.5.bias"):
            s4[key].copy_(s3[key])
        s4["net.8.weight"].zero_(); s4["net.8.bias"].zero_()
        for src, dst in enumerate(M3_TO_M4):
            s4["net.8.weight"][dst].copy_(s3["net.8.weight"][src])
            s4["net.8.bias"][dst].copy_(s3["net.8.bias"][src])
        # New interception outputs are a neutral midpoint for lambda_i=0.
        s4["net.8.bias"][4:6].zero_()
        net4.load_state_dict(s4)
    return net3.eval(), net4.eval(), str(path)


def fixed_raw_from_network(net, attrs):
    with torch.no_grad():
        return net(attrs)


def wrapper_q(model, x, raw, model_name):
    model.eval()
    if model_name == "mopex4":
        xx = x
    else:
        xx = x[..., :3]
    with torch.no_grad():
        q = model({"x_phy": xx}, (None, raw.unsqueeze(-1))) ["streamflow"]
    return q.squeeze(-1) if q.dim() == 3 else q


def wrapper_q_grad(model, x, raw, model_name):
    model.eval()
    xx = x if model_name == "mopex4" else x[..., :3]
    q = model({"x_phy": xx}, (None, raw.unsqueeze(-1)))["streamflow"]
    return q.squeeze(-1) if q.dim() == 3 else q


def core_equivalence():
    torch.manual_seed(7)
    n = 4
    P = torch.rand(n); T = torch.randn(n) * 4; PET = torch.rand(n) * 4
    doy = torch.randint(1, 365, (n,), dtype=torch.float32)
    common_raw = torch.full((n, 8), .5)
    common = norm_to_phys(common_raw, "mopex3")
    m4 = torch.full((n, 10), .5)
    m4[:, M4_COMMON] = common_raw
    p4 = norm_to_phys(m4, "mopex4")
    p3 = common
    # Random nonzero states expose state-order and update-order errors.
    sn, soil, sub, fast, slow = [torch.rand(n) for _ in range(5)]
    with mopex_training_context(lambda_i=0.0, lambda_p=1.0, beta=50.0):
        o3 = mopex3_step(P, T, PET, *p3.t(), sn, soil, sub, fast, slow, nearzero=1e-6)
        # mopex4 state signature is soil, sub, fast, slow, snow.
        o4 = mopex4_step(P, T, PET, *p4.t(), soil, sub, fast, slow, sn,
                         doy=doy, nearzero=1e-6)
    qdiff = float((o3[0] - o4[0]).abs().max())
    etdiff = float((o3[1] - o4[1]).abs().max())
    # M3 state order: snow, soil, sub, fast, slow; M4 output: soil, sub, fast, slow, snow.
    sdiff = max(float((o3[2] - o4[6]).abs().max()), float((o3[3] - o4[2]).abs().max()),
                float((o3[4] - o4[3]).abs().max()), float((o3[5] - o4[4]).abs().max()),
                float((o3[6] - o4[5]).abs().max()))
    return {"level": "core", "q_max_abs_diff": qdiff, "et_max_abs_diff": etdiff,
            "state_max_abs_diff": sdiff, "status": "PASS" if max(qdiff, etdiff, sdiff) < 1e-6 else "FAIL"}


def wrapper_equivalence(x, y, attrs):
    net3, net4, source = mapped_m3_network()
    raw3 = fixed_raw_from_network(net3, attrs)
    raw4 = fixed_raw_from_network(net4, attrs)
    m3 = make_model("mopex3")
    m4 = make_model("mopex4", lambda_i=0.0)
    q3 = wrapper_q(m3, x, raw3, "mopex3")
    q4 = wrapper_q(m4, x, raw4, "mopex4")
    d = (q3 - q4).abs()
    # q shapes are (T-WARMUP,B)
    return {"level": "wrapper", "q_shape_m3": str(tuple(q3.shape)), "q_shape_m4": str(tuple(q4.shape)),
            "q_max_abs_diff": float(d.max()), "q_rmse": float(d.square().mean().sqrt()),
            "status": "PASS" if float(d.max()) < 1e-4 else "FAIL", "source_checkpoint": source}


def dpl_path_equivalence(x, attrs):
    net3, net4, source = mapped_m3_network()
    raw3 = fixed_raw_from_network(net3, attrs)
    raw4 = fixed_raw_from_network(net4, attrs)
    m3 = make_model("mopex3")
    m4 = make_model("mopex4", lambda_i=0.0)
    # This is the normal neural path with a semantically safe M3->M4 map.
    q3 = wrapper_q(m3, x, raw3, "mopex3")
    q4 = wrapper_q(m4, x, raw4, "mopex4")
    d = (q3 - q4).abs()
    return {"level": "dpl_parameter_path", "q_max_abs_diff": float(d.max()),
            "q_rmse": float(d.square().mean().sqrt()),
            "status": "PASS" if float(d.max()) < 1e-4 else "FAIL", "source_checkpoint": source}


def mapping_audit(attrs):
    rows = []
    net3, net4mapped, source = mapped_m3_network()
    # random/init, trained M3-derived, M4 baseline, M4 continuation
    ckpts = {
        "M3_trained_mapped": (net4mapped, source),
        "M4_baseline": (None, str(BENCHMARK / "results/dpl_round13_20260805/auto100/checkpoints/mopex4/epoch_100.pt")),
        "M4_continuation41": (None, str(BENCHMARK / "results/mopex45_phase_fix/full_continuation/runs/seed_41/checkpoints/J2/seed_41/epoch_100.pt")),
    }
    for label, (net, path) in ckpts.items():
        if net is None:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            net = CatchmentParameterizer(35, 10, hidden_dims=[256, 256], dropout=.05)
            net.load_state_dict(payload["network"])
        net.eval()
        with torch.no_grad(): out = net(attrs)
        for idx, name in enumerate(PARAM_INFO["mopex4"]):
            v = out[:, idx]
            rows.append({"source": label, "parameter_index": idx, "parameter": name,
                         "mean_normalized": float(v.mean()), "std_normalized": float(v.std()),
                         "p10": float(v.quantile(.1)), "p50": float(v.quantile(.5)),
                         "p90": float(v.quantile(.9)),
                         "boundary_fraction": float(((v < .02) | (v > .98)).float().mean()),
                         "checkpoint": path})
    return rows


def gradient_interference(x, y, attrs):
    _, net4, _ = mapped_m3_network()
    model = make_model("mopex4", lambda_i=0.0)
    model.train(); net4.train(False)
    # Fixed representative batch/window, no stochastic sampling.
    xb = x[:WARMUP + SCORED]
    yb = y[WARMUP:WARMUP + SCORED]
    raw0 = net4(attrs)
    rows = []
    g0 = None
    for lam in (0.0, .25, .5, .75, 1.0):
        net4.zero_grad(set_to_none=True)
        model.continuation_lambda_i = lam
        raw = net4(attrs)
        raw.retain_grad()
        q = model({"x_phy": xb}, (None, raw.unsqueeze(-1)))["streamflow"]
        q = q.squeeze(-1) if q.dim() == 3 else q
        loss, _ = compute_differentiable_kge(q, yb, warmup_days=0)
        loss.backward()
        def flat_named(predicate):
            vals = []
            for name, p in net4.named_parameters():
                if p.grad is not None and predicate(name): vals.append(p.grad.detach().reshape(-1))
            return torch.cat(vals) if vals else torch.zeros(1)
        trunk = flat_named(lambda n: not n.startswith("net.8."))
        common_head = torch.cat([net4.net[-1].weight.grad[M4_COMMON].reshape(-1),
                                 net4.net[-1].bias.grad[M4_COMMON].reshape(-1)])
        int_head = torch.cat([net4.net[-1].weight.grad[M4_INTERCEPTION].reshape(-1),
                              net4.net[-1].bias.grad[M4_INTERCEPTION].reshape(-1)])
        if g0 is None: g0 = trunk.clone()
        cosine = float(torch.dot(trunk, g0) / (trunk.norm() * g0.norm() + 1e-12))
        rows.append({"lambda_i": lam, "loss": float(loss),
                     "trunk_grad_norm": float(trunk.norm()), "trunk_cosine_vs_lambda0": cosine,
                     "trunk_relative_delta": float((trunk - g0).norm() / (g0.norm() + 1e-12)),
                     "common_head_grad_norm": float(common_head.norm()),
                     "interception_head_grad_norm": float(int_head.norm()),
                     "raw_alpha_grad_norm": float(raw.grad[:, 4].norm()),
                     "raw_is_time_grad_norm": float(raw.grad[:, 5].norm()),
                     "raw_common_grad_norm": float(raw.grad[:, M4_COMMON].norm()),
                     "q_finite": bool(torch.isfinite(q).all())})
    return rows


def lambda_sweep_and_interception_only(x, y, attrs):
    _, net4, _ = mapped_m3_network()
    raw = net4(attrs).detach()
    model = make_model("mopex4", lambda_i=0.0)
    xb = x[:WARMUP + SCORED]; yb = y[WARMUP:WARMUP + SCORED]
    rows = []
    for lam in torch.linspace(0, 1, 11):
        model.continuation_lambda_i = float(lam)
        q = wrapper_q(model, xb, raw, "mopex4")
        _, kge = compute_differentiable_kge(q, yb, warmup_days=0)
        rows.append({"lambda_i": float(lam), "median_kge": float(kge.median()), "mean_kge": float(kge.mean()),
                     "loss": float(1 - kge.mean())})
    # Directly train only alpha/is_time normalized outputs with all common outputs fixed.
    theta = raw[:, M4_INTERCEPTION].clone().requires_grad_(True)
    opt = torch.optim.AdamW([theta], lr=1e-2)
    for step in range(31):
        opt.zero_grad(set_to_none=True)
        current = raw.detach().clone(); current[:, M4_INTERCEPTION] = theta
        model.continuation_lambda_i = 1.0
        q = wrapper_q_grad(model, xb, current, "mopex4")
        loss, kge = compute_differentiable_kge(q, yb, warmup_days=0)
        loss.backward(); opt.step()
        with torch.no_grad(): theta.clamp_(0, 1)
        if step in (0, 10, 20, 30):
            rows.append({"lambda_i": "interception_only", "step": step,
                         "median_kge": float(kge.median()), "mean_kge": float(kge.mean()),
                         "loss": float(loss)})
    return rows


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(2); torch.manual_seed(123)
    ids, x, y, b = load_context()
    attrs_all = CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cpu", method="zscore")
    attrs = attrs_all[b]
    x = x[START:START + WARMUP + SCORED]; y = y[START:START + WARMUP + SCORED]
    core = core_equivalence()
    wrap = wrapper_equivalence(x, y, attrs)
    dpl = dpl_path_equivalence(x, attrs)
    write_csv("mopex3_mopex4_lambda0_equivalence.csv", [core, wrap, dpl])
    map_rows = mapping_audit(attrs)
    write_csv("mopex34_parameter_mapping_audit.csv", map_rows)
    grad_rows = gradient_interference(x, y, attrs)
    write_csv("mopex4_shared_gradient_interference.csv", grad_rows)
    sweep_rows = lambda_sweep_and_interception_only(x, y, attrs)
    write_csv("mopex3_to_mopex4_lambda_sweep.csv", sweep_rows)
    summary = {"basin_indices": b, "basin_ids": [ids[i] for i in b],
               "core": core, "wrapper": wrap, "dpl_path": dpl,
               "parameter_order_m3": list(PARAM_INFO["mopex3"]),
               "parameter_order_m4": list(PARAM_INFO["mopex4"]),
               "m3_to_m4_common_map": dict(zip(list(PARAM_INFO["mopex3"]), M3_TO_M4)),
               "gpu_used": False, "note": "representative CPU audit; no full-basin training"}
    (OUT / "root_cause_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
