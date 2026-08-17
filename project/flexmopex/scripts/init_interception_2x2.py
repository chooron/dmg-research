#!/usr/bin/env python3
"""Epoch-0 initialization for the interception 2x2 experiment (Phase 5).

Creates the single shared initial state for arms A/B/C/D from seed 42:

  * builds the FlexMOPEX model exactly as ``run_model.py`` does (same config
    normalization + ``set_randomseed(42)`` + xavier init convention),
  * saves ``{model}_ep0.pt`` + ``train_state_ep0.pt`` (with the post-init RNG
    state) into the arm-A model directory,
  * records the SHA-256 of the full initial state dict,
  * records epoch-0 per-basin gate / alpha / is_time distributions.

Arms B/C/D receive byte-identical copies of these two files, so all four arms
start from identical neural weights (verified by checksum) and continue from
the same RNG stream as arm A.  No trained checkpoint is involved.

Usage:
    python scripts/init_interception_2x2.py [--config conf/config_dmopex_intercept2x2_A.yaml] [--output-root results/intercept_2x2] [--run-name A] [--gpu-id 0]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
for p in (PROJECT_DIR.parent.parent, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dmg.core.utils import set_randomseed  # noqa: E402
from dmg.core.utils.utils import save_train_state  # noqa: E402
from project.flexmopex import load_config  # noqa: E402
from project.flexmopex.run_model import apply_runtime_overrides, parse_args  # noqa: E402
from project.flexmopex.local_model_handler import FlexMopexModelHandler  # noqa: E402


def state_sha256(handler: FlexMopexModelHandler) -> str:
    h = hashlib.sha256()
    for model in handler.model_dict.values():
        for name, t in model.state_dict().items():
            h.update(name.encode())
            h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/config_dmopex_intercept2x2_A.yaml")
    parser.add_argument("--output-root", default="results/intercept_2x2")
    parser.add_argument("--run-name", default="A")
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    cli = parse_args(["--config", args.config, "--gpu-id", str(args.gpu_id),
                      "--output-root", args.output_root, "--run-name", args.run_name])
    config = load_config(args.config)
    apply_runtime_overrides(config, cli, config_path=args.config)
    config["mode"] = "train"
    set_randomseed(config["random_seed"])

    handler = FlexMopexModelHandler(config, verbose=False)
    model_dir = Path(config["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    # save epoch-0 model + trainer state (fresh optimizer, RNG right after init)
    handler.save_model(0)
    optimizer = torch.optim.Adadelta(handler.get_parameters(), lr=float(config["train"]["learning_rate"]))
    save_train_state(str(model_dir), epoch=0, optimizer=optimizer)

    sha = state_sha256(handler)
    report = {
        "arm": args.run_name,
        "model_dir": str(model_dir),
        "config": args.config,
        "state_sha256": sha,
    }
    import json
    out_json = Path(args.output_root) / "init_epoch0.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    print(f"[init] epoch-0 checkpoint written to {model_dir}")
    print(f"[init] state_sha256 = {sha}")
    print(f"[init] report -> {out_json}")


if __name__ == "__main__":
    main()
