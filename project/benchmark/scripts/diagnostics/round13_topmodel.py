"""Round-13 topmodel boundary and numerical-shape audit."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
spec = importlib.util.spec_from_file_location("k_full_retrain", ROOT / "scripts/diagnostics/k_full_retrain.py")
k = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(k)

DEVICE = torch.device("cuda")
OUT = ROOT / "results/dpl_round13_20260805"


def main() -> None:
    model = "topmodel"
    OUT.mkdir(parents=True, exist_ok=True)
    ids = [int(x) for x in k.load_ids("data/531sub_id.txt")]
    attrs = k.CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")
    tx, ty, vx, vy = k.NATIVE.load_camels_time_series(ids)
    train_x = torch.as_tensor(tx, dtype=torch.float32, device=DEVICE)
    train_y = torch.as_tensor(ty, dtype=torch.float32, device=DEVICE)
    val_x = torch.as_tensor(vx, dtype=torch.float32, device=DEVICE)
    val_y = torch.as_tensor(vy, dtype=torch.float32, device=DEVICE)
    hydro = k.build_model(model, DEVICE, warm_up=365, backend="compile", parameter_mapping="auto", warmup_grad_mode="detach")
    net = k.CatchmentParameterizer(attrs.shape[1], 7, hidden_dims=[256, 256], dropout=.05).to(DEVICE)
    ckpt = sorted((ROOT / "results/dpl_full_retrain_20260804/auto100/checkpoints/topmodel").glob("epoch_*.pt"))[-1]
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["network"])
    net.eval()
    with torch.no_grad():
        theta = net(attrs)
        q = hydro({"x_phy": val_x}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
        _, kge = k.NATIVE.compute_differentiable_kge(q, val_y, warmup_days=365)
    names = list(k.PARAM_INFO[model])
    rows = []
    for j, name in enumerate(names):
        mask = (theta[:, j] < .02) | (theta[:, j] > .98)
        if not bool(mask.any()):
            continue
        forced = theta.detach().clone()
        upper = theta[:, j] > .5
        forced[mask, j] = torch.where(upper[mask], torch.ones_like(theta[mask, j]), torch.zeros_like(theta[mask, j]))
        with torch.no_grad():
            q_forced = hydro({"x_phy": val_x}, (None, forced.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            _, kge_forced = k.NATIVE.compute_differentiable_kge(q_forced, val_y, warmup_days=365)
        changed = (q_forced - q).abs().max(dim=0).values > 0
        rows.append({"parameter": name, "boundary_basin_count": int(mask.sum()),
                     "theta_min_boundary": float(theta[mask, j].min()), "theta_max_boundary": float(theta[mask, j].max()),
                     "forced_lower_count": int((mask & ~upper).sum()), "forced_upper_count": int((mask & upper).sum()),
                     "physical_lower": float(k.PARAM_INFO[model][name][0]), "physical_upper": float(k.PARAM_INFO[model][name][1]),
                     "pin_kge_delta": float(kge_forced.median() - kge.median()),
                     "q_finite_count": int(torch.isfinite(q_forced).sum()), "q_nan_count": int(torch.isnan(q_forced).sum()),
                     "q_inf_count": int(torch.isinf(q_forced).sum()),
                     "q_abs_max": float(torch.nan_to_num(q_forced, nan=0.0, posinf=0.0, neginf=0.0).abs().max()),
                     "changed_basin_count": int(changed.sum())})
        if name == "f":
            torch.save({"theta": theta.detach().cpu(), "q": q.detach().cpu(), "q_forced": q_forced.detach().cpu(),
                        "kge": kge.detach().cpu(), "kge_forced": kge_forced.detach().cpu(), "boundary": mask.detach().cpu()},
                       OUT / "topmodel_f_boundary_tensors.pt")
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "topmodel_boundary_diagnostic.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
