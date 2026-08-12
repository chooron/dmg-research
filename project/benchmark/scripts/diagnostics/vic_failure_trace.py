"""Replay VIC until its first round-13 non-finite update."""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
spec = importlib.util.spec_from_file_location("k_full_retrain", ROOT / "scripts/diagnostics/k_full_retrain.py")
k = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(k)
OUT = ROOT / "results/dpl_round13_20260805/final"; DEVICE = torch.device("cuda")

def stats(x):
    f = torch.isfinite(x.detach()); z = torch.nan_to_num(x.detach())
    return {"finite": bool(f.all()), "finite_count": int(f.sum()), "total_count": int(x.numel()), "min": float(z.min()), "max": float(z.max())}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = [int(x) for x in k.load_ids("data/531sub_id.txt")]
    attrs = k.CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")
    tx, ty, _, _ = k.NATIVE.load_camels_time_series(ids)
    train_x = torch.as_tensor(tx, dtype=torch.float32, device=DEVICE); train_y = torch.as_tensor(ty, dtype=torch.float32, device=DEVICE)
    catalog, lengths = k.H1.make_catalog(train_y[k.WARMUP:])
    hydro = k.build_model("vic", DEVICE, warm_up=k.WARMUP, backend="compile", parameter_mapping="auto", warmup_grad_mode="detach")
    hydro_trace = k.build_model("vic", DEVICE, warm_up=k.WARMUP, backend="eager", parameter_mapping="auto", warmup_grad_mode="detach")
    net = k.CatchmentParameterizer(attrs.shape[1], 10, hidden_dims=[256, 256], dropout=.05).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    payload = torch.load(ROOT / "results/dpl_round13_20260805/auto100/checkpoints/vic/epoch_050.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(payload["network"]); opt.load_state_dict(payload["optimizer"])
    torch.random.set_rng_state(payload["cpu_rng"]); torch.cuda.set_rng_state(payload["cuda_rng"], device=DEVICE)
    result = None
    for epoch in range(51, 58):
        for step in range(k.STEPS):
            basins = torch.randperm(len(ids), device=DEVICE)[:k.BATCH]
            choices = (torch.rand(k.BATCH, device=DEVICE) * lengths[basins]).long(); starts = catalog[basins, choices]
            x = k.H1.gather_window(train_x, starts, basins); y = k.H1.gather_window(train_y, starts, basins)
            opt.zero_grad(set_to_none=True); theta = net(attrs[basins]); theta.retain_grad()
            anomaly = epoch == 57 and step == 145
            if anomaly:
                torch.autograd.set_detect_anomaly(True)
            q = (hydro_trace if anomaly else hydro)({"x_phy": x}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            if not bool(torch.isfinite(q).all()):
                result = {"epoch": epoch, "batch_step": step, "source": "forward", "q_stats": stats(q), "theta_stats": stats(theta)}; break
            loss, _ = k.NATIVE.compute_differentiable_kge(q, y[k.WARMUP:], warmup_days=0)
            try:
                loss.backward()
            except RuntimeError as error:
                result = {"epoch": epoch, "batch_step": step, "source": "autograd_anomaly", "loss": float(loss), "error": str(error), "q_stats": stats(q), "theta_stats": stats(theta)}
                break
            finally:
                if anomaly:
                    torch.autograd.set_detect_anomaly(False)
            g = torch.cat([p.grad.detach().reshape(-1) for p in net.parameters() if p.grad is not None])
            if not bool(torch.isfinite(g).all()):
                result = {"epoch": epoch, "batch_step": step, "source": "backward", "loss": float(loss), "q_stats": stats(q), "gradient_stats": stats(g), "theta_gradient_stats": stats(theta.grad)}; break
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0); opt.step()
            p = torch.cat([v.detach().reshape(-1) for v in net.parameters()])
            if not bool(torch.isfinite(p).all()):
                result = {"epoch": epoch, "batch_step": step, "source": "optimizer_step", "loss": float(loss), "gradient_stats": stats(g), "network_stats_after_step": stats(p)}; break
        if result is not None: break
    (OUT / "vic_failure_trace.json").write_text(json.dumps(result or {"result":"not_reproduced"}, indent=2))
    print(json.dumps(result or {"result":"not_reproduced"}, indent=2))

if __name__ == "__main__": main()
