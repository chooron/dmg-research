#!/usr/bin/env python3
"""Phase 6: Compute and Memory Cost Benchmarking for R14 Counterfactual Structural Target."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config  # noqa: E402
from project.flexmopex.run_model import (  # noqa: E402
    apply_runtime_overrides, parse_args, _build_data_loader,
)
from scripts.diagnose_wint_collapse import build_handler, build_forward, run_loop  # noqa: E402

OUT_DIR = Path("results/feasibility_r14")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def per_basin_fit(q: torch.Tensor, obs: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    o = torch.nan_to_num(obs, nan=0.0)
    sq = (q - o) ** 2 / (std.view(1, -1, 1) ** 2)
    mask = ~torch.isnan(obs)
    n_valid = mask.sum(dim=0).clamp(min=1)
    sq = torch.where(mask, sq, torch.zeros_like(sq))
    return sq.sum(dim=0) / n_valid


def main() -> None:
    dev = "cuda:0"
    cfg_path = "conf/config_dmopex_interceptE_S0_aicdelay2.yaml"
    c = load_config(cfg_path)
    c_cli = parse_args(["--config", cfg_path, "--gpu-id", "0",
                        "--output-root", "results/intercept_aicdelay",
                        "--run-name", "E_S0_aicdelay2"])
    apply_runtime_overrides(c, c_cli, config_path=cfg_path)
    c["mode"] = "train"
    c["model"]["phy"]["disable_compile"] = True
    dl = _build_data_loader(c)

    td, ed = dl.train_dataset, dl.eval_dataset
    B = td["x_phy"].shape[1]
    n_attr = td["xc_nn_norm"].shape[-1] - 3
    n_out = int(ed["x_phy"].shape[0]) - 365
    std_train = (np.nanstd(td["target"][:, :, 0].cpu().numpy(), axis=0) + 0.1).astype(np.float32)
    std_t = torch.from_numpy(std_train).to(dev)

    handler = build_handler(c)
    handler.load_model(2)
    m = next(iter(handler.model_dict.values()))
    phy = m.phy_model

    sample_batch_size = 100
    n_trials = 20

    sample_b100 = {
        "x_phy": ed["x_phy"][:, :sample_batch_size].to(dev),
        "doy": ed["doy"][:, :sample_batch_size].to(dev),
        "c_nn_norm": td["xc_nn_norm"][0, :sample_batch_size, -n_attr:].to(dev),
        "target": ed["target"][:, :sample_batch_size].to(dev),
    }

    # 1. Standard training step (Forward + Backward on 100 basins)
    m.train()
    optimizer = torch.optim.Adadelta(m.parameters(), lr=1.0)

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        p, logits, w_on, m_p, r_p = build_forward(phy, m.nn_model, sample_b100)
        out = run_loop(phy, sample_b100, w_on, m_p, r_p)
        q = out["streamflow"]
        obs = sample_b100["target"][365:365 + n_out]
        loss = per_basin_fit(q, obs, std_t[:sample_batch_size]).mean()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_trials):
        optimizer.zero_grad()
        p, logits, w_on, m_p, r_p = build_forward(phy, m.nn_model, sample_b100)
        out = run_loop(phy, sample_b100, w_on, m_p, r_p)
        q = out["streamflow"]
        obs = sample_b100["target"][365:365 + n_out]
        loss = per_basin_fit(q, obs, std_t[:sample_batch_size]).mean()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    t_std_step = (time.time() - t0) / n_trials

    # 2. Counterfactual evaluation: 1 process ON/OFF (S=2) on 100 basins
    m.eval()
    for _ in range(3):
        with torch.no_grad():
            p_raw = m.nn_model({"c_nn_norm": sample_b100["c_nn_norm"]})
            w_learn = F.softmax(p_raw["weights"].view(sample_batch_size, 4, 2).clamp(-10, 10), dim=-1)[..., 1]
            m_p = phy._descale_mopex_params(p_raw["params"])
            r_p = phy._descale_routing_params(p_raw["gamma_uh"])

            S = 2
            w_cf = w_learn.repeat(S, 1)
            w_cf[:sample_batch_size, 1] = 0.0
            w_cf[sample_batch_size:, 1] = 1.0

            p_rep = {k: v.repeat(S, 1) for k, v in m_p.items()}
            r_rep = {k: v.repeat(S) for k, v in r_p.items()}
            s_rep = {"x_phy": sample_b100["x_phy"].repeat(1, S, 1), "doy": sample_b100["doy"].repeat(1, S, 1)}

            P, T, PET, doy, n_steps, _ = phy._prepare_forcings(s_rep)
            Q = phy._run_weighted_loop(P, T, PET, doy, p_rep, w_cf, n_steps, sample_batch_size * S)
            Qr = phy._apply_routing(Q.mean(-1), r_rep)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_trials):
        with torch.no_grad():
            p_raw = m.nn_model({"c_nn_norm": sample_b100["c_nn_norm"]})
            w_learn = F.softmax(p_raw["weights"].view(sample_batch_size, 4, 2).clamp(-10, 10), dim=-1)[..., 1]
            m_p = phy._descale_mopex_params(p_raw["params"])
            r_p = phy._descale_routing_params(p_raw["gamma_uh"])

            S = 2
            w_cf = w_learn.repeat(S, 1)
            w_cf[:sample_batch_size, 1] = 0.0
            w_cf[sample_batch_size:, 1] = 1.0

            p_rep = {k: v.repeat(S, 1) for k, v in m_p.items()}
            r_rep = {k: v.repeat(S) for k, v in r_p.items()}
            s_rep = {"x_phy": sample_b100["x_phy"].repeat(1, S, 1), "doy": sample_b100["doy"].repeat(1, S, 1)}

            P, T, PET, doy, n_steps, _ = phy._prepare_forcings(s_rep)
            Q = phy._run_weighted_loop(P, T, PET, doy, p_rep, w_cf, n_steps, sample_batch_size * S)
            Qr = phy._apply_routing(Q.mean(-1), r_rep)
    torch.cuda.synchronize()
    t_1proc_cf = (time.time() - t0) / n_trials

    # 3. Counterfactual evaluation: all 4 processes (S=8: ON/OFF per process) vectorized on 100 basins
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_trials):
        with torch.no_grad():
            p_raw = m.nn_model({"c_nn_norm": sample_b100["c_nn_norm"]})
            w_learn = F.softmax(p_raw["weights"].view(sample_batch_size, 4, 2).clamp(-10, 10), dim=-1)[..., 1]
            m_p = phy._descale_mopex_params(p_raw["params"])
            r_p = phy._descale_routing_params(p_raw["gamma_uh"])

            S = 8  # 4 processes * 2 endpoints
            w_cf = w_learn.repeat(S, 1)
            for p_idx in range(4):
                w_cf[(2 * p_idx) * sample_batch_size:(2 * p_idx + 1) * sample_batch_size, p_idx] = 0.0
                w_cf[(2 * p_idx + 1) * sample_batch_size:(2 * p_idx + 2) * sample_batch_size, p_idx] = 1.0

            p_rep = {k: v.repeat(S, 1) for k, v in m_p.items()}
            r_rep = {k: v.repeat(S) for k, v in r_p.items()}
            s_rep = {"x_phy": sample_b100["x_phy"].repeat(1, S, 1), "doy": sample_b100["doy"].repeat(1, S, 1)}

            P, T, PET, doy, n_steps, _ = phy._prepare_forcings(s_rep)
            Q = phy._run_weighted_loop(P, T, PET, doy, p_rep, w_cf, n_steps, sample_batch_size * S)
            Qr = phy._apply_routing(Q.mean(-1), r_rep)
    torch.cuda.synchronize()
    t_4proc_cf = (time.time() - t0) / n_trials

    # Memory usage
    mem_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)

    cost_benchmark = {
        "batch_size": sample_batch_size,
        "n_valid_days": n_out,
        "time_std_training_step_sec": float(t_std_step),
        "time_1proc_cf_sec": float(t_1proc_cf),
        "time_4proc_cf_vectorized_sec": float(t_4proc_cf),
        "ratio_1proc_cf_over_std_step": float(t_1proc_cf / t_std_step),
        "ratio_4proc_cf_over_std_step": float(t_4proc_cf / t_std_step),
        "gpu_max_mem_mb": float(mem_allocated),
        "cost_strategies": {
            "every_batch": {
                "additional_cost_per_epoch_sec": float(t_4proc_cf * 7),
                "epoch_overhead_pct": float(t_4proc_cf / t_std_step * 100),
            },
            "every_epoch_epoch_start": {
                "cost_per_epoch_sec": float(t_4proc_cf * (671 / 100)),
                "epoch_overhead_pct": float((t_4proc_cf * 6.71) / (t_std_step * 7) * 100),
            },
            "every_2_epochs": {
                "cost_per_epoch_sec": float(t_4proc_cf * 6.71 / 2),
                "epoch_overhead_pct": float((t_4proc_cf * 6.71 / 2) / (t_std_step * 7) * 100),
            }
        }
    }

    p6_json = OUT_DIR / "phase6_compute_memory_cost.json"
    p6_json.write_text(json.dumps(cost_benchmark, indent=2))
    print(f"[Phase 6 Complete] Saved compute benchmark to {p6_json}")
    print(json.dumps(cost_benchmark, indent=2))


if __name__ == "__main__":
    main()
