"""Continuous local perturbation audit for TOPMODEL f at its best checkpoint."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
spec = importlib.util.spec_from_file_location("k_full_retrain", ROOT / "scripts/diagnostics/k_full_retrain.py")
k = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(k)

OUT = ROOT / "results/dpl_round13_20260805/final"
DEVICE = torch.device("cuda")


def kge_parts(q: torch.Tensor, y: torch.Tensor, warmup: int = 365) -> tuple[torch.Tensor, dict[str, float]]:
    if y.shape[0] == q.shape[0] + warmup:
        y = y[warmup:]
    elif q.shape[0] == y.shape[0] + warmup:
        q = q[warmup:]
    mask = torch.isfinite(q) & torch.isfinite(y) & (q >= 0) & (y >= 0)
    count = mask.sum(0).clamp_min(1).to(q.dtype)
    qs = torch.where(mask, q, torch.zeros_like(q)); ys = torch.where(mask, y, torch.zeros_like(y))
    mq, my = qs.sum(0) / count, ys.sum(0) / count
    dq, dy = (qs - mq) * mask, (ys - my) * mask
    sq = torch.sqrt((dq.square().mean(0)) + 1e-12)
    sy = torch.sqrt((dy.square().mean(0)) + 1e-12)
    r = (dq * dy).mean(0) / (sq * sy)
    alpha = sq / sy
    beta = mq / (my + 1e-6)
    kg = 1.0 - torch.sqrt((r-1).square() + (alpha-1).square() + (beta-1).square() + 1e-12)
    valid = (count > 30) & torch.isfinite(kg)
    return kg, {"correlation_median": float(r[valid].median()), "variability_ratio_median": float(alpha[valid].median()), "bias_ratio_median": float(beta[valid].median())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ids = [int(x) for x in k.load_ids("data/531sub_id.txt")]
    attrs = k.CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")
    _, _, vx, vy = k.NATIVE.load_camels_time_series(ids)
    val_x = torch.as_tensor(vx, dtype=torch.float32, device=DEVICE)
    val_y = torch.as_tensor(vy, dtype=torch.float32, device=DEVICE)
    hydro = k.build_model("topmodel", DEVICE, warm_up=365, backend="compile", parameter_mapping="auto", warmup_grad_mode="detach")
    net = k.CatchmentParameterizer(attrs.shape[1], 7, hidden_dims=[256, 256], dropout=.05).to(DEVICE)
    # Epoch 88 is the scalar best point, but the runner checkpoints every 10
    # epochs. Use the reproducible epoch-100 checkpoint and record that limit.
    ckpt = ROOT / "results/dpl_round13_20260805/auto100/checkpoints/topmodel/epoch_100.pt"
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["network"]); net.eval()
    with torch.no_grad():
        theta = net(attrs)
        q0 = hydro({"x_phy": val_x}, (None, theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
        k0 = kge_parts(q0, val_y)[0]

    def state_stats(raw: torch.Tensor) -> dict[str, float]:
        """Replay the same forward step to retain the two internal states."""
        params = hydro._descale_params(raw)
        states = hydro._init_states(val_x.size(1), 1)
        p_seq, t_seq, pet_seq = hydro._make_forcing_sequences(val_x, 1)
        param_values = [params[name] for name in hydro.phy_param_names]
        minima = [float("inf"), float("inf")]
        maxima = [float("-inf"), float("-inf")]
        medians = [[], []]
        for t in range(val_x.size(0)):
            out = hydro.step_fn(p_seq[t], t_seq[t], pet_seq[t], *param_values, *states, nearzero=hydro.nearzero)
            states = tuple(out[2:])
            if t >= min(hydro.warm_up, val_x.size(0)):
                for i, state in enumerate(states[:2]):
                    minima[i] = min(minima[i], float(state.min()))
                    maxima[i] = max(maxima[i], float(state.max()))
                    medians[i].append(float(state.median()))
        return {
            "s1_min": minima[0], "s1_max": maxima[0], "s1_median": float(torch.tensor(medians[0]).median()),
            "s2_min": minima[1], "s2_max": maxima[1], "s2_median": float(torch.tensor(medians[1]).median()),
        }

    factors = [("current", 1.0), ("0.9x", .9), ("0.5x", .5), ("0.1x", .1), ("positive_floor_1pct_current", None), ("exact_zero", 0.0)]
    rows = []
    for label, factor in factors:
        forced = theta.detach().clone()
        if factor is None:
            forced[:, 4] = torch.clamp(theta[:, 4] * .01, min=1e-6)
        else:
            forced[:, 4] = theta[:, 4] * factor
        with torch.no_grad():
            q = hydro({"x_phy": val_x}, (None, forced.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            kg, parts = kge_parts(q, val_y)
            states = state_stats(forced)
        valid_kg = kg[torch.isfinite(kg)]
        delta_kg = valid_kg - k0[torch.isfinite(k0)]
        finite = torch.isfinite(q)
        rows.append({"perturbation": label, "factor": factor, "f_theta_median": float(forced[:,4].median()),
                     "f_theta_min": float(forced[:,4].min()), "f_theta_max": float(forced[:,4].max()),
                     "validation_kge_median": float(kg.median()), "delta_vs_current": float(kg.median()-k0.median()),
                     "kge_q05": float(torch.quantile(valid_kg, .05)), "kge_q50": float(torch.quantile(valid_kg, .50)),
                     "kge_q95": float(torch.quantile(valid_kg, .95)), "delta_kge_q05": float(torch.quantile(delta_kg, .05)),
                     "delta_kge_q50": float(torch.quantile(delta_kg, .50)), "delta_kge_q95": float(torch.quantile(delta_kg, .95)),
                     "q_finite": bool(finite.all()), "q_min": float(torch.nan_to_num(q).min()),
                     "q_max": float(torch.nan_to_num(q).max()), "q_median": float(torch.nan_to_num(q).median()),
                     **parts, **states})
    result = pd.DataFrame(rows)
    result.insert(0, "checkpoint_epoch", 100)
    result.to_csv(OUT / "topmodel_local_boundary_sensitivity.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
