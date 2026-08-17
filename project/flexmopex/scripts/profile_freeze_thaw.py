#!/usr/bin/env python3
"""Agent B — freeze-thaw mechanism test: objective profiles + trajectories.

Runs the frozen w_int sweeps (32-basin window at epochs 0/1/2/3/5/10, and the
128-basin/3-yr window at epochs 0/2/3/10) on the freeze-thaw E-S0 run, plus the
eval gradient decomposition at the decision epochs, and compares gate/param
trajectories with the standard E-S0 control (collapse_diagnosis.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from project.flexmopex import load_config  # noqa: E402
from project.flexmopex.run_model import (  # noqa: E402
    apply_runtime_overrides, parse_args, _build_data_loader, _build_loss,
)
from scripts.diagnose_wint_collapse import (  # noqa: E402
    build_handler, diagnostic_sample, frozen_wint_profile,
    eval_gate_decomposition, epoch_param_drift, W_GRID, GATE_NAMES,
)

EPOCHS_ALL = [0, 1, 2, 3, 5, 10]
EPOCHS_128 = [0, 2, 3, 10]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/config_dmopex_interceptE_S0_freeze.yaml")
    ap.add_argument("--output-root", default="results/intercept_freeze")
    ap.add_argument("--run-name", default="E_S0_freeze")
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.output_root)
    arm_dir = root / args.run_name

    cli = parse_args(["--config", args.config, "--gpu-id", str(args.gpu_id),
                      "--output-root", str(args.output_root), "--run-name", args.run_name])
    config = load_config(args.config)
    apply_runtime_overrides(config, cli, config_path=args.config)
    config["mode"] = "train"
    if str(config["device"]).startswith("cuda"):
        torch.cuda.set_device(config["device"])

    dl = _build_data_loader(config)
    td = dl.train_dataset
    loss_tot = _build_loss(config, td)
    loss_fit = _build_loss(config, td)
    loss_fit.aic_alpha = 0.0

    sample32 = diagnostic_sample(td, config["device"])
    n_attr = td["xc_nn_norm"].shape[-1] - 3
    B128 = list(range(128))
    sample128 = {
        "x_phy": td["x_phy"][730:1825, B128, :].to(config["device"]),
        "doy": td["doy"][730:1825, B128, :].to(config["device"]),
        "c_nn_norm": td["xc_nn_norm"][0, B128, -n_attr:].to(config["device"]),
        "target": td["target"][730 + 365:1825, B128, :].to(config["device"]),
        "batch_sample": np.asarray(B128, dtype=np.int64),
    }
    handler = build_handler(config)

    summary = {}
    for epoch in EPOCHS_ALL:
        try:
            handler.load_model(epoch)
        except FileNotFoundError:
            print(f"[warn] epoch {epoch} checkpoint missing; skipping")
            continue
        for m in handler.model_dict.values():
            m.eval()
        prof32 = frozen_wint_profile(handler, loss_fit, loss_tot, sample32)
        drift = epoch_param_drift(handler, config, td)
        entry = {"profile_32b": prof32, "drift": drift}
        if epoch in (2, 3):
            entry["grad_32b"] = eval_gate_decomposition(handler, loss_fit, loss_tot, sample32)
        if epoch in EPOCHS_128:
            entry["profile_128b"] = frozen_wint_profile(handler, loss_fit, loss_tot, sample128)
        summary[epoch] = entry
        p = entry["profile_32b"]
        print(f"ep{epoch}: fit_best={p['fit_best_w']} total_best={p['total_best_w']} "
              f"delta_Lfit={p['delta_Lfit_best_pos_minus_w0']:+.2e} "
              f"slope_tot={p['slope_Ltotal_near_learned']:+.2e} "
              f"learned_w={p['learned_w_int_median']:.3f}", flush=True)
        if "profile_128b" in entry:
            p128 = entry["profile_128b"]
            print(f"  [128b] fit_best={p128['fit_best_w']} total_best={p128['total_best_w']} "
                  f"Lfit0={p128['L_fit_w0']:.5f} Lfit1={p128['L_fit_best_positive']:.5f}", flush=True)

    (arm_dir / "freeze_profiles.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[done] -> {arm_dir / 'freeze_profiles.json'}")


if __name__ == "__main__":
    main()
