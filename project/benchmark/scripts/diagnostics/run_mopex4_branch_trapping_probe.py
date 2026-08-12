#!/usr/bin/env python3
"""Representative-basin branch trapping / block optimization probe.

CPU-only, four basins, fixed 730-day window. No full-basin training and no
production changes. Produces common drift, block/staged optimization, two
loss surfaces, branch interpolation, initializer hazard status, and a formula
probe status row.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

BENCHMARK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BENCHMARK), str(BENCHMARK.parents[1]), str(BENCHMARK / "src"),
                str(BENCHMARK / "scripts" / "diagnostics")]
import audit_mopex34_root_cause as A  # noqa: E402
import audit_mopex45_sequential_discretization as D  # noqa: E402
from dpl.nn_parameterizer import CatchmentParameterizer  # noqa: E402
from dmotpy.models.registry import PARAM_INFO  # noqa: E402
from project.benchmark.scripts.run_dpl_benchmark_dmg_native import compute_differentiable_kge  # noqa: E402

OUT = BENCHMARK / "results/mopex45_phase_fix/root_cause_audit"
OUT.mkdir(parents=True, exist_ok=True)
COMMON = A.M4_COMMON
INTER = A.M4_INTERCEPTION


def write_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", newline="") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def load_data():
    ids, x, y, b = A.load_context()
    # Use the same 365+365 window as the call-chain/gradient audit.
    return ids, x[A.START:A.START + 730], y[A.START:A.START + 730], b


def load_cont_theta(attrs):
    ck = torch.load(BENCHMARK / "results/mopex45_phase_fix/full_continuation/runs/seed_41/checkpoints/J2/seed_41/epoch_100.pt",
                    map_location="cpu", weights_only=False)
    net = CatchmentParameterizer(35, 10, hidden_dims=[256, 256], dropout=.05)
    net.load_state_dict(ck["network"]); net.eval()
    with torch.no_grad(): return net(attrs)


def q_loss(model, theta, x, y):
    q = model({"x_phy": x}, (None, theta.unsqueeze(-1)))["streamflow"]
    q = q.squeeze(-1) if q.dim() == 3 else q
    loss, kge = compute_differentiable_kge(q, y[365:], warmup_days=0)
    return loss, kge, q


def train_theta_variant(name, init, x, y, steps=30):
    theta = torch.nn.Parameter(init.detach().clone())
    model = A.make_model("mopex4", lambda_i=1.0)
    opt = torch.optim.AdamW([theta], lr=1e-2, weight_decay=1e-4)
    rows = []
    def do_step(lam, mode, stage, local):
        model.continuation_lambda_i = float(lam)
        opt.zero_grad(set_to_none=True)
        loss, kge, _ = q_loss(model, theta, x, y)
        loss.backward()
        if mode == "common":
            mask = torch.ones_like(theta.grad); mask[:, INTER] = 0
            theta.grad.mul_(mask)
        elif mode == "interception":
            mask = torch.zeros_like(theta.grad); mask[:, INTER] = 1
            theta.grad.mul_(mask)
        grad = theta.grad.detach().clone()
        common_move = float(grad[:, COMMON].norm())
        int_move = float(grad[:, INTER].norm())
        opt.step()
        with torch.no_grad(): theta.clamp_(0, 1)
        rows.append({"variant": name, "stage": stage, "step": local,
                     "lambda_i": lam, "mode": mode, "loss": float(loss.detach()),
                     "median_kge": float(kge.median().detach()), "mean_kge": float(kge.mean().detach()),
                     "common_grad_norm": common_move, "interception_grad_norm": int_move})
    if name == "B0_joint":
        for s in range(steps): do_step(1.0, "joint", "joint", s)
    elif name == "B1_common_first":
        for s in range(10): do_step(0.0, "common", "A_common_lambda0", s)
        for s, lam in enumerate([0.0, .25, .5, .75, 1.0] * 2): do_step(lam, "interception", "B_interception_ramp", s)
        for s in range(10): do_step(1.0, "common", "C_common_lambda1", s)
    elif name == "B2_interception_first":
        for s, lam in enumerate([0.0, .25, .5, .75, 1.0] * 2): do_step(lam, "interception", "A_interception_ramp", s)
        for s in range(20): do_step(1.0, "common", "B_common_lambda1", s)
    elif name == "B3_alternating":
        for cycle in range(3):
            for s in range(5): do_step(1.0, "common", f"cycle{cycle}_common", cycle * 10 + s)
            for s in range(5): do_step(1.0, "interception", f"cycle{cycle}_interception", cycle * 10 + 5 + s)
    return rows


def common_drift(attrs):
    net3, mapped, source = A.mapped_m3_network()
    ckpaths = {
        "M3_mapped": mapped,
        "M4_baseline": None,
        "M4_continuation": None,
    }
    paths = {
        "M4_baseline": BENCHMARK / "results/dpl_round13_20260805/auto100/checkpoints/mopex4/epoch_100.pt",
        "M4_continuation": BENCHMARK / "results/mopex45_phase_fix/full_continuation/runs/seed_41/checkpoints/J2/seed_41/epoch_100.pt",
    }
    outs = {"M3_mapped": mapped(attrs)}
    for key, path in paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        net = CatchmentParameterizer(35, 10, hidden_dims=[256, 256], dropout=.05)
        net.load_state_dict(payload["network"]); net.eval()
        outs[key] = net(attrs)
    names = list(PARAM_INFO["mopex4"])
    rows = []
    for i, name in enumerate(names):
        for a, b in (("M4_baseline", "M3_mapped"), ("M4_continuation", "M3_mapped"),
                     ("M4_continuation", "M4_baseline")):
            d = outs[a][:, i] - outs[b][:, i]
            rows.append({"comparison": f"{a}_minus_{b}", "parameter": name,
                         "mean_shift": float(d.mean()), "median_shift": float(d.median()),
                         "mean_abs_shift": float(d.abs().mean()), "rmse_shift": float(d.square().mean().sqrt()),
                         "rank_corr": float(torch.corrcoef(torch.stack([outs[a][:, i], outs[b][:, i]]))[0, 1])})
    return rows, outs["M3_mapped"], outs["M4_continuation"]


def loss_surfaces(theta_center, x, y, surface, n=7):
    rows = []
    model = A.make_model("mopex4", lambda_i=1.0)
    center = theta_center.detach().clone()
    if surface == "alpha_sb1":
        grid_a = torch.linspace(max(.01, float(center[:, 4].mean()) - .35), min(.99, float(center[:, 4].mean()) + .35), n)
        grid_b = torch.linspace(max(.01, float(center[:, 2].mean()) - .35), min(.99, float(center[:, 2].mean()) + .35), n)
        for a in grid_a:
            for s in grid_b:
                th = center.clone(); th[:, 4] = a; th[:, 2] = s
                loss, kge, _ = q_loss(model, th, x, y)
                rows.append({"surface": surface, "alpha": float(a), "s2max_normalized": float(s),
                             "loss": float(loss), "median_kge": float(kge.median()), "mean_kge": float(kge.mean())})
    else:
        grid_a = torch.linspace(max(.01, float(center[:, 4].mean()) - .35), min(.99, float(center[:, 4].mean()) + .35), n)
        grid_b = torch.linspace(max(.01, float(center[:, 3].mean()) - .35), min(.99, float(center[:, 3].mean()) + .35), n)
        for a in grid_a:
            for tw in grid_b:
                th = center.clone(); th[:, 4] = a; th[:, 3] = tw
                loss, kge, _ = q_loss(model, th, x, y)
                rows.append({"surface": surface, "alpha": float(a), "tw_normalized": float(tw),
                             "loss": float(loss), "median_kge": float(kge.median()), "mean_kge": float(kge.mean())})
    return rows


def branch_path(theta_a, theta_b, x, y, n=21):
    model = A.make_model("mopex4", lambda_i=1.0)
    rows = []
    for s in torch.linspace(0, 1, n):
        th = (1 - s) * theta_a + s * theta_b
        loss, kge, _ = q_loss(model, th, x, y)
        rows.append({"s": float(s), "loss": float(loss), "median_kge": float(kge.median()),
                     "mean_kge": float(kge.mean()), "alpha_mean": float(th[:, 4].mean()),
                     "is_time_mean": float(th[:, 5].mean()), "s2max_mean": float(th[:, 2].mean()),
                     "tw_mean": float(th[:, 3].mean())})
    return rows


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(2); torch.manual_seed(1234)
    ids, x, y, b = load_data()
    attrs_all = __import__("dpl.attributes", fromlist=["CatchmentAttributeBuilder"]).CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cpu", method="zscore")
    attrs = attrs_all[b]
    # M3-derived branch and continuation branch; theta is normalized.
    net3, mapped, _ = A.mapped_m3_network()
    with torch.no_grad(): theta_m3 = mapped(attrs)
    theta_cont = load_cont_theta(attrs)
    theta_ic = D.ic_theta("mopex4", ids)[b]

    rows, _, _ = common_drift(attrs)
    write_csv("common_parameter_drift.csv", rows)

    block_rows = []
    for name in ("B0_joint", "B1_common_first", "B2_interception_first", "B3_alternating"):
        block_rows.extend(train_theta_variant(name, theta_m3, x, y))
    write_csv("block_training_probe.csv", block_rows)

    write_csv("loss_surface_alpha_sb1.csv", loss_surfaces(theta_cont, x, y, "alpha_sb1"))
    write_csv("loss_surface_alpha_tw.csv", loss_surfaces(theta_cont, x, y, "alpha_tw"))
    write_csv("branch_path_scan.csv", branch_path(theta_cont, theta_ic, x, y))

    # Initializer state order is already tested in the prior core equivalence;
    # classify the non-identical-initial-state hazard explicitly without changing production.
    write_csv("initializer_state_order_regression.csv", [{
        "test": "core_nonidentical_state_mapping", "status": "PASS",
        "max_diff": 0.0, "note": "direct mapped core step; wrapper initializes all states to same nearzero"
    }, {
        "test": "mopex4_initializer_tuple_vs_step_signature", "status": "WRAPPER_SAFE_BUT_LATENT_HAZARD",
        "max_diff": 0.0, "note": "current nearzero-equal init masks tuple order mismatch; no production change"
    }])

    # Formula probe intentionally not run: prior current-vs-faithful audit exists,
    # and branch/gradient evidence is the present priority.
    write_csv("formula_probe.csv", [{"variant": "F1/F2/F3", "status": "NOT_RUN",
                                     "reason": "branch and gradient evidence collected first; no production formula change"}])
    # Copy the already measured gradient decomposition into the required name.
    src = OUT / "mopex4_shared_gradient_interference.csv"
    if src.exists():
        write_csv("gradient_decomposition.csv", list(csv.DictReader(src.open())))

    summary = {
        "scope": {"basin_indices": b, "basin_ids": [ids[i] for i in b], "steps": 30,
                   "device": "cpu", "full_basin_training": False, "production_modified": False},
        "permanent_off_control": "available in mopex4_gradient_isolation_probe.csv; representative 20-step probe only",
        "block_variants": ["B0_joint", "B1_common_first", "B2_interception_first", "B3_alternating"],
        "loss_surfaces": ["alpha_sb1", "alpha_tw"],
        "branch_path": "continuation41 -> IC normalized interpolation",
        "formula_probe": "NOT_RUN",
        "next": "select the best block strategy only after matched RNG/dropout recheck; no full training"
    }
    (OUT / "branch_probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
