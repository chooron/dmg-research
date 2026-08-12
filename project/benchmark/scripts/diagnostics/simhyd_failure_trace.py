"""Replay SIMHYD from epoch 60 and locate the first non-finite value."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
spec = importlib.util.spec_from_file_location("k_full_retrain", ROOT / "scripts/diagnostics/k_full_retrain.py")
k = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(k)
from dmotpy.models.core.simhyd import simhyd_step

OUT = ROOT / "results/dpl_round13_20260805/final"
DEVICE = torch.device("cuda")


def scalar(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def finite_stats(x: torch.Tensor) -> dict[str, float | int | bool]:
    y = x.detach()
    finite = torch.isfinite(y)
    z = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return {"finite": bool(finite.all()), "finite_count": int(finite.sum()),
            "total_count": int(y.numel()), "min": float(z.min()), "max": float(z.max())}


def network_stats(net) -> dict[str, float | int | bool]:
    values = [p.detach() for p in net.parameters()]
    flat = torch.cat([x.reshape(-1) for x in values])
    return finite_stats(flat)


def named_gradient_stats(net) -> dict[str, dict[str, float | int | bool]]:
    return {name: finite_stats(param.grad) for name, param in net.named_parameters() if param.grad is not None}


def trace_batch(hydro, theta, x, basins, starts, ids, epoch: int, step: int, source: str) -> dict:
    params = {name: value.squeeze(-1) for name, value in hydro._descale_params(theta).items()}
    states = hydro._init_states(len(basins), 1)
    soil, groundwater = states
    first = None
    for t in range(x.shape[0]):
        p = x[t, :, 0:1]
        temp = x[t, :, 1:2]
        pet = x[t, :, 2:3]
        before_soil = soil.clone()
        before_groundwater = groundwater.clone()
        q, ea, soil, groundwater = simhyd_step(
            p, temp, pet, params["insc"].unsqueeze(-1), params["coeff"].unsqueeze(-1),
            params["sq"].unsqueeze(-1), params["smsc"].unsqueeze(-1),
            params["sub"].unsqueeze(-1), params["crak"].unsqueeze(-1),
            params["k"].unsqueeze(-1), soil, groundwater, nearzero=hydro.nearzero,
        )
        values = {"streamflow": q, "evaporation": ea, "soil_new": soil, "groundwater_new": groundwater}
        bad = {name: ~torch.isfinite(value) for name, value in values.items()}
        any_bad = torch.zeros_like(q, dtype=torch.bool)
        for value in bad.values():
            any_bad |= value
        if bool(any_bad.any()):
            bi, gi = torch.where(any_bad)
            b = int(bi[0])
            first = {
                "first_bad_time_in_window": t,
                "first_bad_basin_batch_index": b,
                "first_bad_basin_id": int(ids[int(basins[b])]),
                "epoch": epoch, "batch_step": step, "source": source,
                "forcing_P": scalar(p[b, 0]), "forcing_T": scalar(temp[b, 0]), "forcing_PET": scalar(pet[b, 0]),
                "before_soil": scalar(before_soil[b, 0]),
                "before_groundwater": scalar(before_groundwater[b, 0]),
                "params": {name: scalar(value[b]) for name, value in params.items()},
                "bad_variables": [name for name, mask in bad.items() if bool(mask.any())],
                "values": {name: finite_stats(value[b]) for name, value in values.items()},
            }
            break
    return first or {"epoch": epoch, "batch_step": step, "source": source, "no_bad_value": True}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ids = [int(x) for x in k.load_ids("data/531sub_id.txt")]
    attrs = k.CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")
    tx, ty, _, _ = k.NATIVE.load_camels_time_series(ids)
    train_x = torch.as_tensor(tx, dtype=torch.float32, device=DEVICE)
    train_y = torch.as_tensor(ty, dtype=torch.float32, device=DEVICE)
    catalog, lengths = k.H1.make_catalog(train_y[k.WARMUP:])
    hydro = k.build_model("simhyd", DEVICE, warm_up=k.WARMUP, backend="compile", parameter_mapping="auto", warmup_grad_mode="detach")
    # Use eager only for the target batch so anomaly detection can expose the
    # underlying backward operator instead of a single compiled-function node.
    hydro_trace = k.build_model("simhyd", DEVICE, warm_up=k.WARMUP, backend="eager", parameter_mapping="auto", warmup_grad_mode="detach")
    net = k.CatchmentParameterizer(attrs.shape[1], 7, hidden_dims=[256, 256], dropout=.05).to(DEVICE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ckpt = ROOT / "results/dpl_round13_20260805/auto100/checkpoints/simhyd/epoch_060.pt"
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(payload["network"]); optimizer.load_state_dict(payload["optimizer"])
    torch.random.set_rng_state(payload["cpu_rng"]); torch.cuda.set_rng_state(payload["cuda_rng"], device=DEVICE)

    trace = None
    net.train()
    for epoch in range(61, 65):
        for step in range(k.STEPS):
            basins = torch.randperm(len(ids), device=DEVICE)[:k.BATCH]
            choices = (torch.rand(k.BATCH, device=DEVICE) * lengths[basins]).long()
            starts = catalog[basins, choices]
            x = k.H1.gather_window(train_x, starts, basins)
            y = k.H1.gather_window(train_y, starts, basins)
            optimizer.zero_grad(set_to_none=True)
            theta = net(attrs[basins]); theta.retain_grad()
            anomaly = epoch == 64 and step == 102
            if anomaly:
                torch.autograd.set_detect_anomaly(True)
            active_hydro = hydro_trace if anomaly else hydro
            q = active_hydro({"x_phy": x}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            if not bool(torch.isfinite(q).all()):
                trace = trace_batch(hydro, theta.detach(), x.detach(), basins, starts, ids, epoch, step, "train")
                trace["q_forward_stats"] = finite_stats(q)
                break
            loss, _ = k.NATIVE.compute_differentiable_kge(q, y[k.WARMUP:], warmup_days=0)
            loss.backward()
            if anomaly:
                torch.autograd.set_detect_anomaly(False)
            grad_values = [p.grad.detach() for p in net.parameters() if p.grad is not None]
            grad_stats = finite_stats(torch.cat([x.reshape(-1) for x in grad_values]))
            grad_norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0))
            optimizer.step()
            if not network_stats(net)["finite"]:
                theta_bad = ~torch.isfinite(theta.grad)
                bad_basin, bad_param = torch.where(theta_bad)
                trace = {
                    "epoch": epoch, "batch_step": step, "source": "optimizer_step",
                    "loss": scalar(loss), "q_stats": finite_stats(q),
                    "gradient_stats_before_clip": grad_stats, "gradient_norm_before_clip": grad_norm,
                    "named_gradient_stats": named_gradient_stats(net),
                    "theta_gradient_stats": finite_stats(theta.grad),
                    "theta_bad_entries": [
                        {"batch_index": int(b), "basin_id": int(ids[int(basins[b])]), "parameter_index": int(j),
                         "theta": scalar(theta[b, j])}
                        for b, j in zip(bad_basin[:32], bad_param[:32])
                    ],
                    "network_stats_after_step": network_stats(net),
                }
                break
        if trace is not None:
            break
    if trace is None:
        trace = {"result": "no_nonfinite_reproduced_through_epoch_64"}
    (OUT / "simhyd_failure_trace.json").write_text(json.dumps(trace, indent=2, sort_keys=True))
    print(json.dumps(trace, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
