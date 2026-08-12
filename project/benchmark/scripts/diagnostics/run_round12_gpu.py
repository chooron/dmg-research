"""Round-12 GPU jobs without changing the production model or loss code.

The existing K1 runner is reused for checkpoint continuation and the linear
control arm.  Output roots are isolated so the archived K1 run remains intact.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

spec = importlib.util.spec_from_file_location("k_full_retrain", ROOT / "scripts/diagnostics/k_full_retrain.py")
k = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(k)


def latest_source(_arm: str, model: str) -> Path | None:
    source = ROOT / "results/dpl_full_retrain_20260804/auto100/checkpoints" / model
    files = sorted(source.glob("epoch_*.pt"))
    return files[-1] if files else None


def run_l3(models: list[str]) -> None:
    k.OUT = ROOT / "results/dpl_round12_20260805/l3_auto200"
    k.latest_checkpoint = latest_source
    (k.OUT / "auto100").mkdir(parents=True, exist_ok=True)
    for model in models:
        print(f"[L3] {model}: resume auto100 checkpoint -> epoch 200", flush=True)
        print(k.run_model("auto100", model, 200, 1e-3), flush=True)


def run_l4(models: list[str]) -> None:
    k.OUT = ROOT / "results/dpl_round12_20260805/l4_linear100"
    k.latest_checkpoint = lambda _arm, _model: None
    k.STOP_ON_PLATEAU = False
    (k.OUT / "linear100").mkdir(parents=True, exist_ok=True)
    for model in models:
        print(f"[L4] {model}: linear control -> epoch 100", flush=True)
        print(k.run_model("linear100", model, 100, 1e-3), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", choices=("l3", "l4"), required=True)
    parser.add_argument("--models", required=True)
    args = parser.parse_args()
    models = [item.strip().lower() for item in args.models.split(",") if item.strip()]
    if args.job == "l3":
        run_l3(models)
    else:
        run_l4(models)


if __name__ == "__main__":
    main()
