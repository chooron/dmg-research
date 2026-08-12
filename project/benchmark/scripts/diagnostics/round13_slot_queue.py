"""Keep three round-13 auto training slots filled until the nine models finish."""
from __future__ import annotations

import csv
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/dpl_round13_20260805/auto100"
LOG = Path("/tmp/round13_auto_queue")
MODELS = ["tcm", "topmodel", "us1", "vic", "wetland", "xinanjiang"]


def completed(model: str) -> bool:
    path = OUT / "status.csv"
    if not path.exists():
        return False
    with path.open() as handle:
        return any(row["model"] == model and row["status"] == "COMPLETED" for row in csv.DictReader(handle))


def launch(model: str) -> subprocess.Popen:
    LOG.mkdir(parents=True, exist_ok=True)
    handle = (LOG / f"{model}.log").open("a")
    return subprocess.Popen(
        ["python", "scripts/diagnostics/round13_m1.py", "--arm", "auto100", "--models", model],
        cwd=ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def main() -> None:
    pending = [model for model in MODELS if not completed(model)]
    active: dict[str, subprocess.Popen] = {}
    while pending or active:
        while pending and len(active) < 3:
            model = pending.pop(0)
            if not completed(model):
                active[model] = launch(model)
        if not active:
            time.sleep(5)
            continue
        time.sleep(20)
        for model, process in list(active.items()):
            if process.poll() is None:
                continue
            del active[model]
            if not completed(model):
                pending.append(model)
    (LOG / "COMPLETE").touch()


if __name__ == "__main__":
    main()
