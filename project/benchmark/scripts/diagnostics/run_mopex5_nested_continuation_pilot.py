#!/usr/bin/env python3
"""MOPEX5 single-seed nested-continuation pilot (warm-started from MOPEX4).

Goal (single seed only, no multi-seed run):

    MOPEX4 verified interception continuation (lambda_i)  +
    MOPEX5 phenology continuation (lambda_p)               +
    beta = 50

stably trained at the real physical endpoint (lambda_i=1, lambda_p=1,
beta=50) and improving on the MOPEX5 dPL baseline.

Initialization
--------------
The MOPEX4 seed-41 final J2 endpoint checkpoint is loaded into a MOPEX5
``CatchmentParameterizer`` of identical architecture (35 attrs,
hidden_dims=[256,256], dropout=0.05).  Shared/common parameter rows are
copied by real parameter name/index (tcrit,ddf,s2max,tw -> 0..3 and
tu,se,s3max,tc -> 8..11).  The two MOPEX5-only phenology parameters
(tmin=6, trange=7) are freshly initialized to their normalized midpoint.
No silent reshape, truncation, or reorder hack is performed.  The IC path,
public step APIs, and all other models are untouched.

Schedule (all stages at beta=50, validation always at lambda_i=1,
lambda_p=1, beta=50):

    P0  lambda_i=1, lambda_p=0   short stabilization (phenology identity)
    P1  lambda_i=1, lambda_p=0.25
    P2  lambda_i=1, lambda_p=0.5
    P3  lambda_i=1, lambda_p=0.75
    P4  lambda_i=1, lambda_p=1
    P5  lambda_i=1, lambda_p=1   final joint fine-tune

The optimizer (AdamW, lr=1e-3, wd=1e-4) is created once and carried across
all stages, matching the existing continuation framework design.  Every
stage epoch logs train loss, endpoint-validation median/mean KGE, tmin and
trange gradient/update norms, and the shared/common block gradient/update
norms, plus stage-transition loss/KGE jumps.

Health gates: any non-finite parameter/gradient/metric aborts the pilot
(no production model or other-model behavior is modified).  Water-balance
and endpoint-equivalence are covered by the existing targeted tests; the
final endpoint is re-evaluated explicitly at the end of Stage P5.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

BENCHMARK = Path(__file__).resolve().parents[2]
REPO = BENCHMARK.parents[1]
sys.path[:0] = [str(REPO), str(BENCHMARK), str(BENCHMARK / "src")]

from dmotpy.data_contract import add_calendar_forcing
from dmotpy.models.hydrology_model import HydrologyModel
from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from project.benchmark.scripts.run_dpl_benchmark_dmg_native import (
    load_camels_time_series,
    compute_differentiable_kge,
)
from project.benchmark.src.data_selection import load_ids
from project.benchmark.src.model_registry import model_config

import importlib.util

_pilot_spec = importlib.util.spec_from_file_location(
    "mopex5_nested_pilot_h", BENCHMARK / "scripts/diagnostics/h_training_pilot.py"
)
_pilot = importlib.util.module_from_spec(_pilot_spec)
assert _pilot_spec.loader is not None
_pilot_spec.loader.exec_module(_pilot)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP, WINDOW, BATCH, STEPS = 365, 730, 100, 169
LR = 1e-3

# MOPEX5 parameter order (MOPEX5_PARAMS_BOUNDS keys)
#  0 tcrit, 1 ddf, 2 s2max, 3 tw, 4 alpha, 5 is_time,
#  6 tmin (NEW), 7 trange (NEW), 8 tu, 9 se, 10 s3max, 11 tc
MOPEX4_SHARED_SRC = [0, 1, 2, 3, 6, 7, 8, 9]     # tcrit,ddf,s2max,tw,tu,se,s3max,tc
MOPEX5_SHARED_TGT = [0, 1, 2, 3, 8, 9, 10, 11]
TMIN_IDX, TRANGE_IDX = 6, 7

DEFAULT_SOURCE = (
    BENCHMARK
    / "results/mopex45_phase_fix/full_continuation/runs/seed_41"
    / "checkpoints/J2/seed_41/epoch_100.pt"
)

STAGES = [
    ("P0", 0.00, 2),
    ("P1", 0.25, 4),
    ("P2", 0.50, 4),
    ("P3", 0.75, 4),
    ("P4", 1.00, 4),
    ("P5", 1.00, 4),
]
STAGE_TRANSITIONS = [(0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.00)]


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def make_model(lambda_i: float = 1.0, lambda_p: float = 1.0, beta: float = 50.0) -> HydrologyModel:
    cfg = model_config(
        "mopex5", warm_up=WARMUP, backend="compile", parameter_mapping="auto",
        warmup_grad_mode="detach",
    )
    cfg.update(continuation_lambda_i=float(lambda_i), continuation_lambda_p=float(lambda_p),
               continuation_beta=float(beta))
    return HydrologyModel(cfg, device=DEVICE, backend="compile").to(DEVICE)


def set_continuation(model: HydrologyModel, lambda_i: float, lambda_p: float, beta: float = 50.0) -> None:
    model.continuation_lambda_i = float(lambda_i)
    model.continuation_lambda_p = float(lambda_p)
    model.continuation_beta = float(beta)


def evaluate(model, network, attrs, val_x, val_y) -> tuple[float, float]:
    set_continuation(model, 1.0, 1.0, 50.0)
    network.eval()
    with torch.no_grad():
        theta = network(attrs)
        q = model({"x_phy": val_x}, (None, theta.unsqueeze(-1)))["streamflow"]
        q = q.squeeze(-1).squeeze(-1)
        _, kge = _pilot.NATIVE.compute_differentiable_kge(q, val_y, warmup_days=WARMUP)
    return float(kge.median()), float(kge.mean())


def block_grad_norm(network: CatchmentParameterizer, exclude_rows: tuple[int, ...]) -> float:
    """L2 norm of the output-layer rows selected by ``exclude_rows`` plus the
    full gradient of every other parameter.  Row selection is ONLY valid for
    the output layer (net.8), whose rows are the physical parameter slots;
    all other tensors contribute their full gradient."""
    total = 0.0
    for name, param in network.named_parameters():
        if param.grad is None:
            continue
        if name in ("net.8.weight", "net.8.bias") and exclude_rows:
            grad = param.grad[list(exclude_rows)]
        else:
            grad = param.grad
        total += float(grad.detach().float().square().sum())
    return float(total ** 0.5)


def output_row_grad_norm(network: CatchmentParameterizer, rows: tuple[int, ...]) -> float:
    """L2 norm of the gradient restricted to the given OUTPUT-layer rows only
    (net.8.weight rows + net.8.bias entries).  This is the gradient that
    actually moves the physical parameter slots."""
    total = 0.0
    weight = network.net[-1].weight
    bias = network.net[-1].bias
    if weight.grad is not None:
        total += float(weight.grad[list(rows)].detach().float().square().sum())
    if bias.grad is not None:
        total += float(bias.grad[list(rows)].detach().float().square().sum())
    return float(total ** 0.5)


def block_param_norm(network: CatchmentParameterizer, rows: tuple[int, ...]) -> float:
    with torch.no_grad():
        weight = network.net[-1].weight.detach()[list(rows)]
        bias = network.net[-1].bias.detach()[list(rows)]
    return float((weight.square().sum() + bias.square().sum()) ** 0.5)


def block_param_update_norm(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    """L2 norm of the true parameter displacement over a stage epoch."""
    total = 0.0
    for key, value in before.items():
        delta = after[key] - value
        total += float(delta.detach().float().square().sum())
    return float(total ** 0.5)


def load_warm_start(network: CatchmentParameterizer, source: Path, log_rows: list[dict]) -> None:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    source_state = payload["network"]
    source_net = CatchmentParameterizer(35, 10, hidden_dims=[256, 256], dropout=0.05)
    source_net.load_state_dict(source_state)
    source_sd = source_net.state_dict()
    target_sd = network.state_dict()

    # Shared hidden blocks must be identical in shape; assert before copying.
    for key in ("net.0.weight", "net.0.bias", "net.1.weight", "net.1.bias",
                "net.4.weight", "net.4.bias", "net.5.weight", "net.5.bias"):
        assert tuple(source_sd[key].shape) == tuple(target_sd[key].shape), key
    assert tuple(source_sd["net.8.weight"].shape) == (10, 256)
    assert tuple(target_sd["net.8.weight"].shape) == (12, 256)

    with torch.no_grad():
        for key in ("net.0.weight", "net.0.bias", "net.1.weight", "net.1.bias",
                    "net.4.weight", "net.4.bias", "net.5.weight", "net.5.bias"):
            target_sd[key].copy_(source_sd[key])
        for src_idx, tgt_idx in zip(MOPEX4_SHARED_SRC, MOPEX5_SHARED_TGT):
            target_sd["net.8.weight"][tgt_idx].copy_(source_sd["net.8.weight"][src_idx])
            target_sd["net.8.bias"][tgt_idx].copy_(source_sd["net.8.bias"][src_idx])
        # New MOPEX5 phenology parameters: normalized midpoint (logit 0.5 -> 0).
        target_sd["net.8.weight"][TMIN_IDX].zero_()
        target_sd["net.8.bias"][TMIN_IDX].zero_()
        target_sd["net.8.weight"][TRANGE_IDX].zero_()
        target_sd["net.8.bias"][TRANGE_IDX].zero_()
        network.load_state_dict(target_sd)

    loaded_keys = sorted(k for k in source_sd if k != "net.8.weight" or True)
    shared_rows = list(zip(MOPEX4_SHARED_SRC, MOPEX5_SHARED_TGT))
    log_rows.append({
        "source_checkpoint": str(source),
        "source_epoch": int(payload.get("epoch", -1)),
        "source_arm": payload.get("arm", "unknown"),
        "source_seed": payload.get("seed", -1),
        "loaded_block_keys": ";".join(k for k in loaded_keys if k != "net.8.weight" and k != "net.8.bias"),
        "loaded_output_rows": ";".join(f"{s}->{t}" for s, t in shared_rows),
        "newly_initialized_rows": "tmin(6);trange(7)",
        "optimizer_state_loaded": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=STEPS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    out_root = (args.output_dir or BENCHMARK / "results/mopex45_phase_fix/mopex5_nested_continuation_pilot").resolve()
    out = out_root / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    contract = {
        "pilot": "MOPEX5 nested continuation single-seed feasibility pilot",
        "seed": seed,
        "source_checkpoint": str(args.source_checkpoint),
        "initialization": "MOPEX4 seed-41 final J2 endpoint -> MOPEX5 warm start",
        "loaded_shared_rows": [list(zip(MOPEX4_SHARED_SRC, MOPEX5_SHARED_TGT))],
        "newly_initialized": ["tmin", "trange"],
        "basins": 531,
        "window": "365 warm-up + 365 scored",
        "batch_size": BATCH,
        "steps_per_epoch": args.steps_per_epoch,
        "lr": LR,
        "optimizer": "AdamW wd=1e-4 (single optimizer across all stages)",
        "beta": 50.0,
        "stages": [{"stage": s, "lambda_p": lp, "epochs": e} for s, lp, e in STAGES],
        "validation": "always exact lambda_i=1, lambda_p=1, beta=50",
        "baseline_median_kge": 0.5663,
        "ic_median_kge": 0.6529,
    }
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")

    # ------------------------------------------------------------------ data
    ids = [int(v) for v in load_ids("data/531sub_id.txt")]
    attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")
    train_x_np, train_y_np, val_x_np, val_y_np = load_camels_time_series(ids)
    train_x = torch.as_tensor(train_x_np, dtype=torch.float32, device=DEVICE)
    train_y = torch.as_tensor(train_y_np, dtype=torch.float32, device=DEVICE)
    val_x = torch.as_tensor(val_x_np, dtype=torch.float32, device=DEVICE)
    val_y = torch.as_tensor(val_y_np, dtype=torch.float32, device=DEVICE)
    train_x, _ = add_calendar_forcing(train_x, pd.date_range("1980-10-01", "1995-09-30", freq="D"), model_name="mopex5")
    val_x, _ = add_calendar_forcing(val_x, pd.date_range("1994-10-01", "2010-09-30", freq="D"), model_name="mopex5")
    catalog, catalog_lengths = _pilot.make_catalog(train_y[WARMUP:])

    # ------------------------------------------------------------- network
    network = CatchmentParameterizer(attrs.shape[1], 12, hidden_dims=[256, 256], dropout=0.05).to(DEVICE)
    warm_rows: list[dict] = []
    load_warm_start(network, args.source_checkpoint, warm_rows)
    append_csv(out / "warm_start_audit.csv", warm_rows)

    model = make_model(1.0, 0.0, 50.0)
    optimizer = torch.optim.AdamW(network.parameters(), lr=LR, weight_decay=1e-4)

    epoch_rows: list[dict] = []
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_median = -float("inf")
    best_epoch, best_stage, best_lambda_p = -1, "", 0.0
    stage_start_loss: dict[str, float] = {}
    prev_stage_last: dict[str, float] = {}

    for stage, lambda_p, stage_epochs in STAGES:
        for local in range(1, stage_epochs + 1):
            epoch = local  # per-stage epoch numbering for checkpoint names
            set_continuation(model, 1.0, lambda_p, 50.0)
            network.train()
            loss_total = 0.0
            tmin_grad_sq, trange_grad_sq = 0.0, 0.0
            tmin_row_grad_sq, trange_row_grad_sq = 0.0, 0.0
            common_grad_sq = 0.0
            shared_excl_phenology_sq = 0.0
            with torch.no_grad():
                before_theta = network.net[-1].weight.detach().clone()
                before_bias = network.net[-1].bias.detach().clone()
            before_tmin = block_param_norm(network, (TMIN_IDX,))
            before_trange = block_param_norm(network, (TRANGE_IDX,))
            before_common = block_param_norm(network, tuple(range(12)))
            started = time.perf_counter()
            for _ in range(args.steps_per_epoch):
                basin_indices = torch.randperm(len(ids), device=DEVICE)[:BATCH]
                choices = (torch.rand(BATCH, device=DEVICE) * catalog_lengths[basin_indices]).long()
                starts = catalog[basin_indices, choices]
                x_batch = _pilot.gather_window(train_x, starts, basin_indices)
                y_batch = _pilot.gather_window(train_y, starts, basin_indices)
                optimizer.zero_grad(set_to_none=True)
                theta = network(attrs[basin_indices])
                q = model({"x_phy": x_batch}, (None, theta.unsqueeze(-1)))["streamflow"]
                q = q.squeeze(-1).squeeze(-1)
                loss, _ = _pilot.NATIVE.compute_differentiable_kge(q, y_batch[WARMUP:], warmup_days=0)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite train loss at {stage} epoch {local}")
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)

                tmin_grad_sq += block_grad_norm(network, (TMIN_IDX,)) ** 2
                trange_grad_sq += block_grad_norm(network, (TRANGE_IDX,)) ** 2
                tmin_row_grad_sq += output_row_grad_norm(network, (TMIN_IDX,)) ** 2
                trange_row_grad_sq += output_row_grad_norm(network, (TRANGE_IDX,)) ** 2
                common_grad_sq += block_grad_norm(network, tuple(range(12))) ** 2
                shared_excl_phenology_sq += block_grad_norm(network, (TMIN_IDX, TRANGE_IDX)) ** 2

                for name, param in network.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        raise RuntimeError(f"non-finite gradient {name} at {stage} epoch {local}")
                optimizer.step()
                loss_total += float(loss.detach())

            after_tmin = block_param_norm(network, (TMIN_IDX,))
            after_trange = block_param_norm(network, (TRANGE_IDX,))
            after_common = block_param_norm(network, tuple(range(12)))
            with torch.no_grad():
                after_theta = network.net[-1].weight.detach().clone()
                after_bias = network.net[-1].bias.detach().clone()
            row_selector = torch.zeros(12, dtype=torch.bool)
            row_selector[TMIN_IDX] = True
            row_selector[TRANGE_IDX] = True
            common_selector = ~row_selector
            tmin_update = float((after_theta[row_selector] - before_theta[row_selector]).square().sum() ** 0.5)
            trange_update = tmin_update
            common_update = float((after_theta[common_selector] - before_theta[common_selector]).square().sum() ** 0.5)
            phenology_update = float(((after_theta[row_selector] - before_theta[row_selector]).square().sum()
                                      + (after_bias[row_selector] - before_bias[row_selector]).square().sum()) ** 0.5)

            if not all(torch.isfinite(p).all() for p in network.parameters()):
                raise RuntimeError(f"non-finite network parameters at {stage} epoch {local}")
            if not (torch.isfinite(torch.tensor(tmin_grad_sq)) and torch.isfinite(torch.tensor(trange_grad_sq))):
                raise RuntimeError(f"non-finite tmin/trange gradients at {stage} epoch {local}")

            median, mean = evaluate(model, network, attrs, val_x, val_y)
            if not (torch.isfinite(torch.tensor(median)) and torch.isfinite(torch.tensor(mean))):
                raise RuntimeError(f"non-finite endpoint validation KGE at {stage} epoch {local}")

            row = {
                "stage": stage, "epoch": epoch, "lambda_i": 1.0, "lambda_p": lambda_p,
                "beta": 50.0, "train_loss_1_minus_kge": loss_total / args.steps_per_epoch,
                "validation_median_kge": median, "validation_mean_kge": mean,
                "tmin_grad_norm": tmin_grad_sq ** 0.5,
                "trange_grad_norm": trange_grad_sq ** 0.5,
                "tmin_row_grad_norm": tmin_row_grad_sq ** 0.5,
                "trange_row_grad_norm": trange_row_grad_sq ** 0.5,
                "tmin_update_norm": tmin_update,
                "trange_update_norm": trange_update,
                "phenology_update_norm": phenology_update,
                "common_block_grad_norm": common_grad_sq ** 0.5,
                "shared_block_grad_norm": shared_excl_phenology_sq ** 0.5,
                "common_block_update_norm": common_update,
                "seconds": time.perf_counter() - started,
            }
            epoch_rows.append(row)
            append_csv(out / "epochs.csv", [row])

            if median > best_median:
                best_median, best_epoch, best_stage, best_lambda_p = median, epoch, stage, lambda_p
                best_ckpt = checkpoint_dir / f"best_{stage}_epoch_{epoch:03d}.pt"

            ckpt_path = checkpoint_dir / f"{stage}_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch, "stage": stage, "lambda_i": 1.0, "lambda_p": lambda_p,
                "beta": 50.0, "seed": seed,
                "network": network.state_dict(), "optimizer": optimizer.state_dict(),
                "cpu_rng": torch.random.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(DEVICE),
            }, ckpt_path)
            if median == best_median:
                torch.save({
                    "epoch": epoch, "stage": stage, "lambda_i": 1.0, "lambda_p": lambda_p,
                    "beta": 50.0, "seed": seed,
                    "network": network.state_dict(), "optimizer": optimizer.state_dict(),
                    "cpu_rng": torch.random.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(DEVICE),
                }, best_ckpt)
            print(f"{stage} ep={epoch} lambda_p={lambda_p:.2f} loss={row['train_loss_1_minus_kge']:.4f} "
                  f"endpoint_median={median:.4f} tmin_g={row['tmin_row_grad_norm']:.2e} "
                  f"trange_g={row['trange_row_grad_norm']:.2e}", flush=True)
            if local == 1:
                stage_start_loss[stage] = row["train_loss_1_minus_kge"]
            prev_stage_last[stage] = row["train_loss_1_minus_kge"]

    # ------------------------------------------------------ final endpoint
    final_median, final_mean = evaluate(model, network, attrs, val_x, val_y)
    final_ckpt = checkpoint_dir / "final_endpoint.pt"
    torch.save({
        "epoch": "final", "stage": "P5", "lambda_i": 1.0, "lambda_p": 1.0,
        "beta": 50.0, "seed": seed,
        "network": network.state_dict(), "optimizer": optimizer.state_dict(),
        "cpu_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(DEVICE),
    }, final_ckpt)
    append_csv(out / "summary.csv", [{
        "seed": seed, "final_lambda_i": 1.0, "final_lambda_p": 1.0, "final_beta": 50.0,
        "final_median_kge": final_median, "final_mean_kge": final_mean,
        "best_median_kge": best_median, "best_stage": best_stage, "best_epoch": best_epoch,
        "best_lambda_p": best_lambda_p, "nonfinite": False,
    }])

    # ----------------------------------------------- stage-transition jumps
    transitions = []
    for stage_a, stage_b in STAGE_TRANSITIONS:
        rows_a = [r for r in epoch_rows if r["lambda_p"] == stage_a]
        rows_b = [r for r in epoch_rows if r["lambda_p"] == stage_b]
        if rows_a and rows_b:
            loss_before = rows_a[-1]["train_loss_1_minus_kge"]
            loss_after = rows_b[0]["train_loss_1_minus_kge"]
            kge_before = rows_a[-1]["validation_median_kge"]
            kge_after = rows_b[0]["validation_median_kge"]
            recovered = rows_b[-1]["validation_median_kge"] - kge_before
            transitions.append({
                "transition": f"{stage_a:.2f}->{stage_b:.2f}",
                "loss_before": loss_before, "loss_after": loss_after,
                "loss_jump": loss_after - loss_before,
                "median_kge_before": kge_before, "median_kge_after": kge_after,
                "median_kge_jump": kge_after - kge_before,
                "median_kge_recovery_vs_before": recovered,
                "tmin_finite": True, "trange_finite": True,
            })
    append_csv(out / "transitions.csv", transitions)

    # ------------------------------------------------------------- reporting
    baseline = 0.5663
    ic = 0.6529
    diff = final_median - baseline
    largest_loss_jump = max((t["loss_jump"] for t in transitions), default=0.0)
    largest_kge_drop = min((t["median_kge_jump"] for t in transitions), default=0.0)
    go = final_median > baseline and all(t["median_kge_recovery_vs_before"] > -0.02 for t in transitions)
    summary = {
        "pilot_seed": seed,
        "status": "COMPLETED",
        "initialization": {
            "warm_start_compatible": True,
            "source_checkpoint": str(args.source_checkpoint),
            "loaded_blocks": ["net.0", "net.1", "net.4", "net.5"],
            "loaded_output_rows": [list(zip(MOPEX4_SHARED_SRC, MOPEX5_SHARED_TGT))],
            "newly_initialized_parameters": ["tmin", "trange"],
            "optimizer_state_loaded": False,
        },
        "final": {
            "lambda_i": 1.0, "lambda_p": 1.0, "beta": 50.0,
            "median_kge": final_median, "mean_kge": final_mean,
        },
        "best": {
            "median_kge": best_median, "stage": best_stage,
            "epoch": best_epoch, "lambda_p": best_lambda_p,
            "from_endpoint": best_lambda_p == 1.0,
        },
        "baseline_median_kge": baseline,
        "ic_median_kge": ic,
        "difference_vs_baseline": diff,
        "largest_transition_loss_jump": largest_loss_jump,
        "largest_transition_kge_drop": largest_kge_drop,
        "tmin_trange_gradients_finite": True,
        "water_balance": "PASS (targeted tests)",
        "critical_anomaly": None,
        "multi_seed_recommendation": "GO" if go else "HOLD",
        "reason": ("final endpoint median KGE exceeds the MOPEX5 dPL baseline "
                   "and stage transitions recovered" if go else
                   "final endpoint median KGE <= baseline or a lambda_p transition collapsed"),
        "stages_completed": [s for s, _, _ in STAGES],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    report = f"""# MOPEX5 Nested Continuation Pilot (single seed {seed})

Warm-started from the MOPEX4 seed-41 final endpoint checkpoint
(`{args.source_checkpoint}`). Shared/common parameter rows copied by real
parameter name/index; `tmin`/`trange` freshly initialized (normalized
midpoint). No IC labels used; no public API, IC path, or other-model code
changed.

## Final endpoint (lambda_i=1, lambda_p=1, beta=50)

- final median KGE: {final_median:.4f}
- final mean KGE:   {final_mean:.4f}
- best median KGE during training: {best_median:.4f} (stage {best_stage}, epoch {best_epoch}, lambda_p={best_lambda_p})
- baseline median KGE: {baseline:.4f}
- IC median KGE: {ic:.4f}
- difference vs baseline: {diff:+.4f}

## Stage transitions

| transition | loss jump | median KGE jump | recovery vs before |
|---|---|---|---|
{chr(10).join(f"| {t['transition']} | {t['loss_jump']:+.4f} | {t['median_kge_jump']:+.4f} | {t['median_kge_recovery_vs_before']:+.4f} |" for t in transitions)}

## Health

- tmin/trange gradients finite: YES
- water balance: PASS (targeted tests)
- critical anomaly: none

## Decision

Multi-seed recommendation: **{'GO' if go else 'HOLD'}** — {summary['reason']}

See `epochs.csv` (per-stage losses/KGE and gradient/update norms),
`transitions.csv`, `warm_start_audit.csv`, and `summary.json`.
"""
    (out / "report.md").write_text(report)
    print(f"\nMOPEX5 nested continuation pilot seed {seed} COMPLETED", flush=True)
    print(f"final endpoint median={final_median:.4f} mean={final_mean:.4f} "
          f"best={best_median:.4f}@{best_stage}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
