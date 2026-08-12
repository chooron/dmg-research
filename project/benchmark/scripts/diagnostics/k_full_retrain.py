#!/usr/bin/env python3
"""K1/K2 full 36-model CUDA dPL health retraining runner.

Primary arm: auto mapping + 100 epochs. Control arm: linear + 100 epochs for
the selected representative models. Numerical model/loss/gradient work stays
on CUDA; host work only loads immutable arrays and writes scalar records.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from dpl.attributes import CatchmentAttributeBuilder
from dpl.nn_parameterizer import CatchmentParameterizer
from dmotpy.models.registry import PARAM_INFO
from dmotpy.data_contract import CALENDAR_MODELS, add_calendar_forcing
from src.data_selection import load_ids
from src.model_registry import NPARAM_INFO_36, build_model

DEVICE = torch.device("cuda")
OUT = ROOT / "results/dpl_full_retrain_20260804"
MODELS = tuple(NPARAM_INFO_36)
CONTROL_MODELS = ("collie1", "gr4j", "mopex1", "ihacres", "mopex4", "hillslope")
BATCH, STEPS, WINDOW, WARMUP, SEED = 100, 169, 730, 365, 42
STOP_ON_PLATEAU = True


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


H1 = load_module(ROOT / "scripts/diagnostics/h_training_pilot.py", "k_h1_helpers")
NATIVE = H1.NATIVE


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        if not exists: writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def paths(arm: str) -> tuple[Path, Path, Path, Path]:
    base = OUT / arm
    return base / "epochs.csv", base / "parameter_gradients.csv", base / "status.csv", base / "health.csv"


def guard_path(arm: str) -> Path:
    return OUT / arm / "gradient_guard.csv"


def checkpoint(arm: str, model: str, epoch: int) -> Path:
    return OUT / arm / "checkpoints" / model / f"epoch_{epoch:03d}.pt"


def latest_checkpoint(arm: str, model: str) -> Path | None:
    files = sorted((OUT / arm / "checkpoints" / model).glob("epoch_*.pt"))
    return files[-1] if files else None


def health_exists(arm: str, model: str) -> bool:
    _epochs, _gradients, _status, health = paths(arm)
    if not health.exists(): return False
    with health.open() as handle:
        # A short smoke run records health evidence but is deliberately not a
        # terminal training result. Only a plateau stop or epoch-100 run can
        # suppress a resumable invocation.
        return any(
            row["model"] == model and
            (row["status"] == "PLATEAU_STOP" or int(row["stop_epoch"]) >= 100)
            for row in csv.DictReader(handle)
        )


def initialize_midpoint(network: CatchmentParameterizer) -> None:
    layer = network.net[-1]
    if not isinstance(layer, nn.Linear): raise TypeError("output layer must be Linear")
    with torch.no_grad(): layer.weight.zero_(); layer.bias.zero_()


def summarize_health(arm: str, model: str, status: str, stop_epoch: int, invalid_train: int, invalid_val: int) -> dict[str, Any]:
    epoch_path, gradient_path, _status_path, _health = paths(arm)
    epochs = [row for row in csv.DictReader(epoch_path.open()) if row["model"] == model]
    gradients = [row for row in csv.DictReader(gradient_path.open()) if row["model"] == model]
    best = max(epochs, key=lambda row: float(row["validation_median_kge"]))
    first, last = min(epochs, key=lambda r: int(r["epoch"])), max(epochs, key=lambda r: int(r["epoch"]))
    params = list(PARAM_INFO[model])
    permanently_zero = [name for name in params if all(float(row["zero_gradient_basin_fraction"]) == 1.0 for row in gradients if row["parameter"] == name)]
    conditional = [name for name in params if any(0.0 < float(row["zero_gradient_basin_fraction"]) < 1.0 for row in gradients if row["parameter"] == name)]
    best_epoch, final_epoch = int(best["epoch"]), int(last["epoch"])
    final_boundary = float(last["theta_boundary_fraction"])
    return {"model": model, "arm": arm, "status": status, "stop_epoch": stop_epoch,
            "best_epoch": best_epoch, "best_validation_median_kge": float(best["validation_median_kge"]),
            "epoch1_validation_median_kge": float(first["validation_median_kge"]),
            "final_validation_median_kge": float(last["validation_median_kge"]),
            "best_minus_epoch1": float(best["validation_median_kge"]) - float(first["validation_median_kge"]),
            "best_minus_final": float(best["validation_median_kge"]) - float(last["validation_median_kge"]),
            "final_boundary_fraction": final_boundary, "train_nonfinite_prediction_count": invalid_train,
            "validation_nonfinite_prediction_count": invalid_val,
            "permanently_zero_parameters": ";".join(permanently_zero),
            "conditional_zero_parameters": ";".join(conditional),
            "pass_integrity": status in {"COMPLETED", "PLATEAU_STOP"} and invalid_train == 0 and invalid_val == 0,
            "pass_learning": float(best["validation_median_kge"]) - float(first["validation_median_kge"]) > .05,
            "pass_no_dead_parameters": not permanently_zero,
            "pass_no_saturation": final_boundary < .20,
            "pass_convergence_budget": not (best_epoch >= final_epoch - 4),
            "pass_no_degradation": float(best["validation_median_kge"]) - float(last["validation_median_kge"]) <= .05}


def run_model(arm: str, model: str, epochs: int, lr: float) -> dict[str, Any]:
    if health_exists(arm, model): return {"model": model, "arm": arm, "status": "ALREADY_COMPLETE"}
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    ids = [int(x) for x in load_ids("data/531sub_id.txt")]
    attrs = CatchmentAttributeBuilder().build_normalized_attributes(ids, device="cuda", method="zscore")
    train_x_np, train_y_np, val_x_np, val_y_np = NATIVE.load_camels_time_series(ids)
    train_x, train_y = torch.as_tensor(train_x_np, dtype=torch.float32, device=DEVICE), torch.as_tensor(train_y_np, dtype=torch.float32, device=DEVICE)
    val_x, val_y = torch.as_tensor(val_x_np, dtype=torch.float32, device=DEVICE), torch.as_tensor(val_y_np, dtype=torch.float32, device=DEVICE)
    if model in CALENDAR_MODELS:
        # MOPEX4/5 consume a date-aligned fourth forcing channel on the GPU.
        train_x, _ = add_calendar_forcing(
            train_x, pd.date_range("1980-10-01", "1995-09-30", freq="D"), model_name=model,
        )
        val_x, _ = add_calendar_forcing(
            val_x, pd.date_range("1994-10-01", "2010-09-30", freq="D"), model_name=model,
        )
    catalog, lengths = H1.make_catalog(train_y[WARMUP:])
    mapping = "auto" if arm == "auto100" else "linear"
    warm_mode = "truncate:90" if model == "penman" and arm == "auto100" else "detach"
    hydro = build_model(model, DEVICE, warm_up=WARMUP, backend="compile", parameter_mapping=mapping, warmup_grad_mode=warm_mode)
    network = CatchmentParameterizer(attrs.shape[1], NPARAM_INFO_36[model], hidden_dims=[256,256], dropout=.05).to(DEVICE)
    initialize_midpoint(network)
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-4)
    start, invalid_train, invalid_val = 1, 0, 0
    old = latest_checkpoint(arm, model)
    if old is not None:
        payload=torch.load(old,map_location="cpu",weights_only=False)
        network.load_state_dict(payload["network"]); optimizer.load_state_dict(payload["optimizer"])
        torch.random.set_rng_state(payload["cpu_rng"]); torch.cuda.set_rng_state(payload["cuda_rng"],device=DEVICE)
        start=int(payload["epoch"])+1; invalid_train=int(payload["invalid_train"]); invalid_val=int(payload["invalid_val"])
    epoch_path, gradient_path, status_path, health_path = paths(arm)
    plateau_values=[]; status="COMPLETED"; final_epoch=start-1
    for epoch in range(start, epochs+1):
        network.train(); loss_total=0.; elapsed=0.
        observed=torch.zeros((len(ids),NPARAM_INFO_36[model]),dtype=torch.bool,device=DEVICE)
        for _ in range(STEPS):
            basins=torch.randperm(len(ids),device=DEVICE)[:BATCH]
            choices=(torch.rand(BATCH,device=DEVICE)*lengths[basins]).long(); starts=catalog[basins,choices]
            x=H1.gather_window(train_x,starts,basins); y=H1.gather_window(train_y,starts,basins)
            optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize(); now=time.perf_counter()
            theta=network(attrs[basins]); theta.retain_grad()
            q=hydro({"x_phy":x},(None,theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            invalid_train += int((~torch.isfinite(q)).sum().detach())
            loss,_=NATIVE.compute_differentiable_kge(q,y[WARMUP:],warmup_days=0); loss.backward()
            gradients = [p.grad for p in network.parameters() if p.grad is not None]
            finite_grad = all(bool(torch.isfinite(g).all()) for g in gradients)
            if not finite_grad:
                bad_theta = theta.grad.detach() if theta.grad is not None else torch.empty(0, device=DEVICE)
                bad_rows = (torch.nonzero(~torch.isfinite(bad_theta).all(dim=1), as_tuple=False).flatten()
                            if bad_theta.numel() else torch.empty(0, dtype=torch.long, device=DEVICE))
                append_csv(guard_path(arm), [{"model": model, "epoch": epoch, "batch": int(_),
                    "basin_ids": ";".join(str(int(ids[int(basins[i])])) for i in bad_rows[:64].tolist()),
                    "finite_gradient": False, "action": "SKIP_BATCH"}])
                optimizer.zero_grad(set_to_none=True)
                continue
            observed[basins] |= theta.grad.detach() != 0
            nn.utils.clip_grad_norm_(network.parameters(),max_norm=1.0); optimizer.step(); torch.cuda.synchronize()
            elapsed += time.perf_counter()-now; loss_total += float(loss.detach())
        network.eval()
        with torch.no_grad():
            val_theta=network(attrs); val_q=hydro({"x_phy":val_x},(None,val_theta.unsqueeze(-1)))["streamflow"].squeeze(-1).squeeze(-1)
            invalid_val += int((~torch.isfinite(val_q)).sum().detach())
            _loss,kge=NATIVE.compute_differentiable_kge(val_q,val_y,warmup_days=WARMUP)
        median=float(kge.median()); plateau_values.append(median); final_epoch=epoch
        append_csv(epoch_path,[{"model":model,"arm":arm,"epoch":epoch,"status":"COMPLETED_EPOCH","validation_median_kge":median,
            "validation_mean_kge":float(kge.mean()),"train_loss_1_minus_kge":loss_total/STEPS,
            "theta_boundary_fraction":float(((val_theta<.02)|(val_theta>.98)).float().mean()),"seconds_per_train_step":elapsed/STEPS,
            "parameter_mapping":mapping,"warmup_grad_mode":warm_mode,"train_nonfinite_cumulative":invalid_train,"validation_nonfinite_cumulative":invalid_val}])
        append_csv(gradient_path,[{"model":model,"arm":arm,"epoch":epoch,"parameter":p,
            "zero_gradient_basin_fraction":float((~observed[:,j]).float().mean()),
            "theta_boundary_basin_fraction":float(((val_theta[:,j]<.02)|(val_theta[:,j]>.98)).float().mean())} for j,p in enumerate(PARAM_INFO[model])])
        if epoch%10==0 or epoch==epochs:
            dst=checkpoint(arm,model,epoch); dst.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"epoch":epoch,"network":network.state_dict(),"optimizer":optimizer.state_dict(),"cpu_rng":torch.random.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(DEVICE),"invalid_train":invalid_train,"invalid_val":invalid_val},dst)
        if STOP_ON_PLATEAU and len(plateau_values)>=21 and plateau_values[-1]-plateau_values[-21] < .002:
            status="PLATEAU_STOP"; break
    summary=summarize_health(arm,model,status,final_epoch,invalid_train,invalid_val)
    append_csv(health_path,[summary]); append_csv(status_path,[{"model":model,"arm":arm,"status":status,"last_epoch":final_epoch,"warmup_grad_mode":warm_mode}])
    del hydro,network,optimizer,train_x,train_y,val_x,val_y; torch.cuda.empty_cache(); return summary


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--arm",choices=("auto100","linear100"),required=True); parser.add_argument("--model",choices=MODELS,default=None); parser.add_argument("--epochs",type=int,default=100); parser.add_argument("--lr",type=float,default=1e-3)
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    base=OUT/args.arm; base.mkdir(parents=True,exist_ok=True)
    selected=(args.model,) if args.model else (MODELS if args.arm=="auto100" else CONTROL_MODELS)
    (base/"contract.json").write_text(json.dumps({"arm":args.arm,"models":list(selected),"epochs":args.epochs,"batch_size":BATCH,"steps_per_epoch":STEPS,"parameter_mapping":"auto" if args.arm=="auto100" else "linear","penman_warmup":"truncate:90 only in auto100","other_warmup":"detach","attributes":"Caravan, zscore all531","hidden_dims":[256,256]},indent=2)+"\n")
    for model in selected: run_model(args.arm,model,args.epochs,args.lr)


if __name__=="__main__": main()
