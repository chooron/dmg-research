#!/usr/bin/env python
"""Compute train/validation IC-vs-dPL deltas by snow/non-snow basin group.

All reported metrics use complete continuous sequences. Window slicing is not
used for the final metrics; the 366-day warmup is shared by IC and dPL.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from models import GR4J, HBV, XAJ
from models.composed import GR4JWithCemaNeige, XAJWithCemaNeige
from models.parameter_specs import (
    GR4J_PARAM_SPECS, HBV_PARAM_SPECS, XAJ_PARAM_SPECS,
    GR4J_CN_PARAM_SPECS, XAJ_CN_PARAM_SPECS,
)
from optimization.pycma_calibrator_v3 import compute_kge_fp64
from training.data_contract import load_dates, load_gage_ids


BASE_CONFIG = PROJECT_DIR / "configs/ic_xnes_production_v1.json"
CONFIG = json.loads(BASE_CONFIG.read_text())
OUT = PROJECT_DIR / "outputs/delta_train_val_by_snow_uh90"

CASES = {
    "GR4J": {
        "model_cls": GR4J,
        "specs": GR4J_PARAM_SPECS,
        "ic": PROJECT_DIR / "results/archive/outputs_ic/ic_pilot_5models_v4_remote/GR4J/KGE_Q/best_parameters_physical.npz",
        "dpl": PROJECT_DIR / "outputs/dpl_unified_365d_v1/GR4J/best_parameters_physical.npz",
    },
    "GR4J_CN": {
        "model_cls": GR4JWithCemaNeige,
        "specs": GR4J_CN_PARAM_SPECS,
        "ic": PROJECT_DIR / "results/archive/outputs_ic/ic_pilot_5models_v4_remote/GR4J_CN/KGE_Q/best_parameters_physical.npz",
        "dpl": PROJECT_DIR / "outputs/dpl_unified_365d_v1/GR4J_CN/best_parameters_physical.npz",
    },
    "HBV": {
        "model_cls": HBV,
        "specs": HBV_PARAM_SPECS,
        "ic": PROJECT_DIR / "results/archive/outputs_ic/ic_pilot_5models_v4_remote/HBV/KGE_Q/best_parameters_physical.npz",
        "dpl": PROJECT_DIR / "outputs/dpl_hbv_kgeq_365d_v1/best_parameters_physical.npz",
    },
    "XAJ": {
        "model_cls": XAJ,
        "specs": XAJ_PARAM_SPECS,
        "ic": PROJECT_DIR / "results/archive/outputs_ic/ic_xaj_kgeq_3restart_pop60_uh90_v1/XAJ/KGE_Q/best_parameters_physical.npz",
        "dpl": PROJECT_DIR / "outputs/dpl_xaj_float32_fix_uh90_full/best_parameters_physical.npz",
    },
    "XAJ_CN": {
        "model_cls": XAJWithCemaNeige,
        "specs": XAJ_CN_PARAM_SPECS,
        "ic": PROJECT_DIR / "results/archive/outputs_ic/ic_xaj_cn_kgeq_3restart_uh90_v1/XAJ_CN/KGE_Q/best_parameters_physical.npz",
        "dpl": PROJECT_DIR / "outputs/dpl_xaj_cn_float32_fix_uh90_full/best_parameters_physical.npz",
    },
}


def load_data():
    raw = np.load(CONFIG["data_npz"], allow_pickle=True)
    forcing = np.asarray(raw["forcing"], dtype=np.float32)
    target = np.asarray(raw["target"], dtype=np.float32)
    dates = pd.to_datetime(load_dates(CONFIG["dates_path"]))
    with open(CONFIG["data_basin_ids"]) as f:
        basin_ids = [str(v).zfill(8) for v in json.load(f)]
    n = len(basin_ids)
    date_indices = {}
    for label, start, end in (
        ("train", "1989-01-01", "1998-12-31"),
        ("val", "1999-01-01", "2009-12-31"),
    ):
        si = int(np.where(dates == pd.Timestamp(start))[0][0])
        ei = int(np.where(dates == pd.Timestamp(end))[0][0])
        date_indices[label] = (si, ei)

    # Existing IC protocol: snow if basin snow fraction >= 0.1.
    with open(CONFIG["data_pkl_dataset"], "rb") as f:
        _, _, attributes = pickle.load(f)
    full_ids = load_gage_ids(CONFIG["gage_ids_path"])
    id_to_idx = {bid: i for i, bid in enumerate(full_ids)}
    selected = np.array([id_to_idx[bid] for bid in basin_ids])
    snow_frac = np.asarray(attributes, dtype=np.float32)[selected, 3]
    snow = snow_frac >= 0.1
    return forcing[:, :n], target[:, :n, 0], basin_ids, date_indices, snow


def make_period_arrays(forcing, target, si, ei, warmup=366):
    start = si - warmup
    fc = {
        "precip": forcing[start:ei + 1, :, 0].T.copy(),
        "pet": forcing[start:ei + 1, :, 2].T.copy(),
        "temp": forcing[start:ei + 1, :, 1].T.copy(),
    }
    obs = target[si:ei + 1].T.copy()
    return fc, obs


def evaluate(model_cls, specs, params_np, period_arrays, device, batch_size=64):
    names = list(specs)
    model = model_cls().to(device=device, dtype=torch.float64)
    model.eval()
    kges = np.full(params_np.shape[0], np.nan, dtype=np.float64)
    fc_np, obs_np = period_arrays
    warmup = 366
    with torch.no_grad():
        for start in range(0, params_np.shape[0], batch_size):
            stop = min(start + batch_size, params_np.shape[0])
            params = {
                name: torch.from_numpy(params_np[start:stop, j]).to(device=device, dtype=torch.float64)
                for j, name in enumerate(names)
            }
            fc = {
                key: torch.from_numpy(value[start:stop]).to(device=device, dtype=torch.float64)
                for key, value in fc_np.items()
            }
            qsim, _ = model(forcings=fc, params=params)
            qsim = qsim[:, warmup:].cpu().numpy()
            for j in range(stop - start):
                kges[start + j] = compute_kge_fp64(qsim[j], obs_np[start + j])[0]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return kges


def aggregate(values, mask):
    x = np.asarray(values)[mask]
    return {"mean": float(np.nanmean(x)), "median": float(np.nanmedian(x)), "n": int(x.size)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forcing, target, basin_ids, periods, snow = load_data()
    period_arrays = {
        label: make_period_arrays(forcing, target, *bounds, warmup=366)
        for label, bounds in periods.items()
    }

    all_rows = []
    for model_name, case in CASES.items():
        print(f"Evaluating {model_name}...", flush=True)
        ic_params = np.asarray(np.load(case["ic"])["params"], dtype=np.float64)
        dpl_params = np.asarray(np.load(case["dpl"])["params"], dtype=np.float64)
        expected_dim = len(case["specs"])
        assert ic_params.shape == (len(basin_ids), expected_dim), (model_name, "IC", ic_params.shape)
        assert dpl_params.shape == (len(basin_ids), expected_dim), (model_name, "dPL", dpl_params.shape)
        scores = {}
        for method, params in (("IC", ic_params), ("dPL", dpl_params)):
            scores[method] = {}
            for period in ("train", "val"):
                scores[method][period] = evaluate(
                    case["model_cls"], case["specs"], params,
                    period_arrays[period], device)
                np.savez_compressed(
                    OUT / f"{model_name}_{method}_{period}.npz",
                    kge=scores[method][period], basin_ids=np.asarray(basin_ids))

        for group, mask in (("snow", snow), ("non_snow", ~snow)):
            row = {"model": model_name, "group": group, "n": int(mask.sum())}
            for period in ("train", "val"):
                ic = aggregate(scores["IC"][period], mask)
                dpl = aggregate(scores["dPL"][period], mask)
                row[f"ic_{period}_mean"] = ic["mean"]
                row[f"ic_{period}_median"] = ic["median"]
                row[f"dpl_{period}_mean"] = dpl["mean"]
                row[f"dpl_{period}_median"] = dpl["median"]
                row[f"delta_{period}_mean"] = ic["mean"] - dpl["mean"]
                row[f"delta_{period}_median"] = ic["median"] - dpl["median"]
            row["drop_mean"] = row["delta_train_mean"] - row["delta_val_mean"]
            row["drop_median"] = row["delta_train_median"] - row["delta_val_median"]
            all_rows.append(row)

    fields = list(all_rows[0])
    with (OUT / "group_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    with (OUT / "basin_scores.csv").open("w", newline="") as f:
        fields_b = ["model", "basin_id", "group", "ic_train", "dpl_train",
                    "ic_val", "dpl_val", "delta_train", "delta_val", "drop"]
        writer = csv.DictWriter(f, fieldnames=fields_b)
        writer.writeheader()
        for model_name in CASES:
            arrays = {}
            for method in ("IC", "dPL"):
                for period in ("train", "val"):
                    arrays[f"{method}_{period}"] = np.load(
                        OUT / f"{model_name}_{method}_{period}.npz")["kge"]
            for i, bid in enumerate(basin_ids):
                dt = arrays["IC_train"][i] - arrays["dPL_train"][i]
                dv = arrays["IC_val"][i] - arrays["dPL_val"][i]
                writer.writerow({
                    "model": model_name, "basin_id": bid,
                    "group": "snow" if snow[i] else "non_snow",
                    "ic_train": arrays["IC_train"][i], "dpl_train": arrays["dPL_train"][i],
                    "ic_val": arrays["IC_val"][i], "dpl_val": arrays["dPL_val"][i],
                    "delta_train": dt, "delta_val": dv, "drop": dt - dv,
                })

    lines = [
        "# Train/validation Delta by snow group",
        "",
        "Final metrics use complete continuous sequences, 366-day shared warmup, and FP64 KGE(Q).",
        "",
        "See `group_summary.csv` for the complete mean/median table and `basin_scores.csv` for basin-level values.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    (OUT / "COMPLETE").touch()
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
