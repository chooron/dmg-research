from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
import torch


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temp = Path(handle.name)
    try:
        torch.save(payload, temp)
        os.replace(temp, path)
    finally:
        if temp.exists(): temp.unlink(missing_ok=True)
    return path


def load_checkpoint(path: str | Path, device: str | torch.device) -> dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)
