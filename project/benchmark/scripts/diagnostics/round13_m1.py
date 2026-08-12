"""Round-13 unified 100-epoch arms."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
spec = importlib.util.spec_from_file_location("k_full_retrain", ROOT / "scripts/diagnostics/k_full_retrain.py")
k = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(k)
k.STOP_ON_PLATEAU = False

OUT = ROOT / "results/dpl_round13_20260805"
AUTO_SOURCE = ROOT / "results/dpl_full_retrain_20260804/auto100"
LINEAR_SOURCE = ROOT / "results/dpl_round12_20260805/l4_linear100/linear100"


def source_root(arm: str) -> Path:
    return AUTO_SOURCE if arm == "auto100" else LINEAR_SOURCE


def latest_source(arm: str, model: str) -> Path | None:
    files = sorted((source_root(arm) / "checkpoints" / model).glob("epoch_*.pt"))
    return files[-1] if files else None


def seed_history(arm: str, models: list[str]) -> None:
    target = OUT / arm
    target.mkdir(parents=True, exist_ok=True)
    source = source_root(arm)
    for name in ("epochs.csv", "parameter_gradients.csv"):
        src = source / name
        dst = target / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    contract_path = target / "contract.json"
    try:
        previous = json.loads(contract_path.read_text()) if contract_path.exists() else {}
    except json.JSONDecodeError:
        previous = {}
    model_list = sorted(set(previous.get("models", [])) | set(models))
    contract_path.write_text(json.dumps({
        "epochs": 100,
        "stop_rule": "force exactly 100; early stopping disabled",
        "arm": arm,
        "models": model_list,
        "source": str(source),
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("auto100", "linear100"), required=True)
    parser.add_argument("--models", required=True)
    args = parser.parse_args()
    models = [x.strip().lower() for x in args.models.split(",") if x.strip()]
    seed_history(args.arm, models)
    k.OUT = OUT
    k.latest_checkpoint = latest_source
    for model in models:
        print(f"[M1] {args.arm} {model} -> epoch 100", flush=True)
        print(k.run_model(args.arm, model, 100, 1e-3), flush=True)


if __name__ == "__main__":
    main()
