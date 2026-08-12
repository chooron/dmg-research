from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/cmaes_36models"
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(ROOT))


def load_yaml(path: Path) -> dict:
    import yaml
    with path.open() as handle: return yaml.safe_load(handle)


def settings() -> dict:
    return load_yaml(EXPERIMENT / "configs/default.yaml")
