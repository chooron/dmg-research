"""Rerun a failed round-13 model with the finite-gradient batch guard."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
spec = importlib.util.spec_from_file_location("round15_runner", ROOT / "scripts/diagnostics/k_full_retrain.py")
k = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(k)

OUT = ROOT / "results/dpl_round15_20260808/guarded_retrain"
SOURCE = ROOT / "results/dpl_round13_20260805/auto100/checkpoints"
RESUME_EPOCH = {"simhyd": 60, "vic": 50}


def latest_checkpoint(arm: str, model: str):
    own = sorted((OUT / arm / "checkpoints" / model).glob("epoch_*.pt"))
    if own:
        return own[-1]
    return SOURCE / model / f"epoch_{RESUME_EPOCH[model]:03d}.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(RESUME_EPOCH), required=True)
    args = parser.parse_args()
    if not k.torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    k.OUT = OUT
    (OUT / "auto100").mkdir(parents=True, exist_ok=True)
    k.STOP_ON_PLATEAU = False
    k.latest_checkpoint = latest_checkpoint
    print(k.run_model("auto100", args.model, 100, 1e-3), flush=True)


if __name__ == "__main__":
    main()
