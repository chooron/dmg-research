#!/usr/bin/env python
"""Evaluate dPL physical parameters through the IC model forward path.

The dPL checkpoints contain one physical parameter vector per basin.  This
script bypasses the dPL network and feeds those vectors directly to the
current IC model implementation, using the same 1999--2009 evaluation data.
Both the exact dPL 365-day evaluation warmup and the IC protocol's 366-day
warmup are reported so that an apparent mismatch cannot be caused by the
one-day warmup convention.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PROJECT_DIR))

from models import XAJ
from models.composed import XAJWithCemaNeige
from models.parameter_specs import XAJ_CN_PARAM_SPECS, XAJ_PARAM_SPECS
from optimization.pycma_calibrator_v3 import compute_kge_fp64
from training.data_contract import FORCING_NAMES, load_dates

CASES = {
    "XAJ": {
        "model_cls": XAJ,
        "specs": XAJ_PARAM_SPECS,
        "dpl_output": PROJECT_DIR / "outputs/dpl_xaj_float32_fix_uh90_full",
    },
    "XAJ_CN": {
        "model_cls": XAJWithCemaNeige,
        "specs": XAJ_CN_PARAM_SPECS,
        "dpl_output": PROJECT_DIR / "outputs/dpl_xaj_cn_float32_fix_uh90_full",
    },
}


def load_evaluation(config: dict, warmup_days: int):
    raw = np.load(config["data_npz"], allow_pickle=True)
    forcing = np.asarray(raw["forcing"], dtype=np.float32)
    target = np.asarray(raw["target"], dtype=np.float32)
    dates = pd.to_datetime(load_dates(config["dates_path"]))
    periods = config["time_periods"]
    cal_end = pd.Timestamp(periods["calibration"]["end"])
    eval_start = pd.Timestamp(periods["evaluation"]["start"])
    eval_end = pd.Timestamp(periods["evaluation"]["end"])
    cal_end_i = int(np.where(dates == cal_end)[0][0])
    eval_start_i = int(np.where(dates == eval_start)[0][0])
    eval_end_i = int(np.where(dates == eval_end)[0][0])
    assert eval_start_i == cal_end_i + 1
    assert eval_end_i - eval_start_i + 1 == 4018
    forcing_start = eval_start_i - warmup_days
    assert forcing_start >= 0
    axis = {
        "precip": FORCING_NAMES.index("P"),
        "temp": FORCING_NAMES.index("T"),
        "pet": FORCING_NAMES.index("PET"),
    }
    fc = {
        key: forcing[forcing_start : eval_end_i + 1, :, axis[key]].transpose().copy()
        for key in ("precip", "pet", "temp")
    }
    obs = target[eval_start_i : eval_end_i + 1, :, 0].transpose().copy()
    with open(config["data_basin_ids"]) as f:
        basin_ids = [str(v).zfill(8) for v in json.load(f)]
    n = len(basin_ids)
    # The transpose above already puts basins on axis 0: [N, T].
    return fc, obs, basin_ids


def evaluate_case(
    case_name: str,
    config: dict,
    warmup_days: int,
    device: torch.device,
    batch_size: int,
):
    case = CASES[case_name]
    params_path = case["dpl_output"] / "best_parameters_physical.npz"
    dpl_params = np.asarray(np.load(params_path)["params"], dtype=np.float64)
    fc, obs, basin_ids = load_evaluation(config, warmup_days)
    n = len(basin_ids)
    assert dpl_params.shape[0] == n
    names = list(case["specs"])
    assert dpl_params.shape[1] == len(names)

    model = case["model_cls"]().to(device=device, dtype=torch.float64)
    model.eval()
    kges = np.full(n, np.nan, dtype=np.float64)
    components = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            params_np = dpl_params[start:stop]
            params = {
                name: torch.from_numpy(params_np[:, j]).to(
                    device=device, dtype=torch.float64
                )
                for j, name in enumerate(names)
            }
            forcings = {
                key: torch.from_numpy(value[start:stop]).to(
                    device=device, dtype=torch.float64
                )
                for key, value in fc.items()
            }
            qsim, _ = model(forcings=forcings, params=params)
            q_np = qsim[:, warmup_days:].cpu().numpy()
            for j, basin_index in enumerate(range(start, stop)):
                kge, comp = compute_kge_fp64(q_np[j], obs[basin_index])
                kges[basin_index] = kge
                components.append((basin_index, comp))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return basin_ids, dpl_params, kges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/ic_xnes_production_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs/dpl_parameters_ic_reproduction_uh90",
    )
    # 64 keeps the compiled XAJ+CemaNeige FP64 path below the available GPU
    # memory while preserving the same per-basin computation.
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    all_rows = []
    report_lines = [
        "# dPL physical parameters evaluated through IC forward path\n",
        "Test period: 1999-01-01 to 2009-12-31 (4018 days).\n",
        "Metric: FP64 KGE(Q). Parameters: dPL best_parameters_physical.npz.\n",
    ]
    for case_name in ("XAJ", "XAJ_CN"):
        for label, warmup in (("dpl_exact_365d", 365), ("ic_protocol_366d", 366)):
            print(f"Evaluating {case_name}, {label}, device={device}...", flush=True)
            basin_ids, params, kges = evaluate_case(
                case_name, config, warmup, device, args.batch_size
            )
            report_lines.append(
                f"## {case_name} / {label}\n\n"
                f"Warmup: {warmup} days\n\n"
                f"KGE mean={np.nanmean(kges):.6f}, median={np.nanmedian(kges):.6f}, "
                f"min={np.nanmin(kges):.6f}, max={np.nanmax(kges):.6f}\n"
            )
            np.savez_compressed(
                args.output_dir / f"{case_name}_{label}_kge.npz",
                kge=kges,
                params=params,
                basin_ids=np.asarray(basin_ids),
            )
            for i, bid in enumerate(basin_ids):
                all_rows.append(
                    {
                        "model": case_name,
                        "evaluation_path": label,
                        "warmup_days": warmup,
                        "basin_id": bid,
                        "basin_index": i,
                        "kge": float(kges[i]),
                    }
                )

    with (args.output_dir / "basin_kge.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (args.output_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    (args.output_dir / "COMPLETE").touch()
    print("\n".join(report_lines), flush=True)


if __name__ == "__main__":
    main()
