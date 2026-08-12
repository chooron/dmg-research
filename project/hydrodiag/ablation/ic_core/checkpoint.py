from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .result_io import atomic_write_json


class CheckpointStore:
    """Minimal atomic completion protocol; optimizer state is intentionally optional."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def state_path(self) -> Path:
        return self.root / "checkpoint_state.json"

    @property
    def complete_path(self) -> Path:
        return self.root / "COMPLETE"

    @property
    def failed_path(self) -> Path:
        return self.root / "FAILED"

    def is_complete(self) -> bool:
        return self.complete_path.exists()

    def save_state(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.state_path, state)

    def mark_complete(self, state: dict[str, Any]) -> None:
        self.save_state({**state, "status": "complete"})
        atomic_write_json(self.complete_path, {"status": "complete", **state})
        if self.failed_path.exists():
            self.failed_path.unlink()

    def mark_failed(self, state: dict[str, Any], reason: str) -> None:
        payload = {**state, "status": "failed", "failure_reason": reason}
        self.save_state(payload)
        atomic_write_json(self.failed_path, payload)

    def load_state(self) -> dict[str, Any] | None:
        if not self.state_path.exists():
            return None
        return json.loads(self.state_path.read_text())
