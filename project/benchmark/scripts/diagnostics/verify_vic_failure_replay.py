"""Phase 4 verification script: Replay VIC failure with patched saturation_2."""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

spec = importlib.util.spec_from_file_location("k_full_retrain", ROOT / "scripts/diagnostics/k_full_retrain.py")
k = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(k)

OUT_DIR = ROOT / "results/dpl_round13_20260805/vic_saturation_fix"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda")

def main():
    ids = [int(x) for x in k.load_ids("data/531sub_id.txt")]
    attrs = k.CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")
    tx, ty, _, _ = k.NATIVE.load_camels_time_series(ids)
    train_x = torch.as_tensor(tx, dtype=torch.float32, device=DEVICE)
    train_y = torch.as_tensor(ty, dtype=torch.float32, device=DEVICE)
    catalog, lengths = k.H1.make_catalog(train_y[k.WARMUP:])

    hydro = k.build_model("vic", DEVICE, warm_up=k.WARMUP, backend="compile", parameter_mapping="auto", warmup_grad_mode="detach")
    net = k.CatchmentParameterizer(attrs.shape[1], 10, hidden_dims=[256, 256], dropout=.05).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

    checkpoint_path = ROOT / "results/dpl_round13_20260805/auto100/checkpoints/vic/epoch_050.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net.load_state_dict(payload["network"])
    opt.load_state_dict(payload["optimizer"])
    torch.random.set_rng_state(payload["cpu_rng"])
    torch.cuda.set_rng_state(payload["cuda_rng"], device=DEVICE)

    # 1. Forward Parity check at epoch 50
    basins_chk = torch.randperm(len(ids), device=DEVICE)[:k.BATCH]
    choices_chk = (torch.rand(k.BATCH, device=DEVICE) * lengths[basins_chk]).long()
    starts_chk = catalog[basins_chk, choices_chk]
    x_chk = k.H1.gather_window(train_x, starts_chk, basins_chk)

    theta_chk = net(attrs[basins_chk])
    out_dict_chk = hydro({"x_phy": x_chk}, (None, theta_chk.unsqueeze(-1)))
    q_chk = out_dict_chk["streamflow"].squeeze(-1).squeeze(-1)

    chk_df = pd.DataFrame([{
        "checkpoint": "epoch_050.pt",
        "q_min": float(q_chk.min()),
        "q_max": float(q_chk.max()),
        "q_mean": float(q_chk.mean()),
        "q_finite": bool(torch.isfinite(q_chk).all())
    }])
    chk_df.to_csv(OUT_DIR / "vic_fixed_checkpoint_forward_compare.csv", index=False)
    print("Saved vic_fixed_checkpoint_forward_compare.csv")

    # 2. Replay to Epoch 57 Batch 145
    target_info = None

    for epoch in range(51, 58):
        for step in range(k.STEPS):
            basins = torch.randperm(len(ids), device=DEVICE)[:k.BATCH]
            choices = (torch.rand(k.BATCH, device=DEVICE) * lengths[basins]).long()
            starts = catalog[basins, choices]
            x = k.H1.gather_window(train_x, starts, basins)
            y = k.H1.gather_window(train_y, starts, basins)

            opt.zero_grad(set_to_none=True)
            theta = net(attrs[basins])
            theta.retain_grad()

            out_dict = hydro({"x_phy": x}, (None, theta.unsqueeze(-1)))
            q = out_dict["streamflow"].squeeze(-1).squeeze(-1)
            loss, _ = k.NATIVE.compute_differentiable_kge(q, y[k.WARMUP:], warmup_days=0)

            loss.backward()

            theta_grad = theta.grad
            g = torch.cat([p.grad.detach().reshape(-1) for p in net.parameters() if p.grad is not None])

            if epoch == 57 and step == 145:
                target_info = {
                    "epoch": epoch,
                    "batch_step": step,
                    "loss": float(loss),
                    "q_finite": bool(torch.isfinite(q).all()),
                    "theta_finite_count_after": int(torch.isfinite(theta_grad).sum()),
                    "theta_total_count": int(theta_grad.numel()),
                    "network_finite_count_after": int(torch.isfinite(g).sum()),
                    "network_total_count": int(g.numel())
                }
                break

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()

        if target_info is not None:
            break

    # Save replay before/after comparison
    replay_df = pd.DataFrame([{
        "epoch": 57,
        "batch_step": 145,
        "metric": "theta_grad_finite",
        "before_fix": "997/1000",
        "after_fix": f"{target_info['theta_finite_count_after']}/{target_info['theta_total_count']}"
    }, {
        "epoch": 57,
        "batch_step": 145,
        "metric": "network_grad_finite",
        "before_fix": "1799/78602",
        "after_fix": f"{target_info['network_finite_count_after']}/{target_info['network_total_count']}"
    }, {
        "epoch": 57,
        "batch_step": 145,
        "metric": "loss_value",
        "before_fix": "0.678248",
        "after_fix": f"{target_info['loss']:.6f}"
    }])
    replay_df.to_csv(OUT_DIR / "failure_replay_before_after.csv", index=False)
    print("Saved failure_replay_before_after.csv:")
    print(replay_df)

if __name__ == "__main__":
    main()
