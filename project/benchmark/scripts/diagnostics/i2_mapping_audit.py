#!/usr/bin/env python3
"""I2: enumerate all 36-model auto-log mapping branches."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from dmotpy.models.registry import PARAM_INFO

OUT = ROOT / "results/dpl_training_pilot_20260801/i2_mapping"
THRESHOLDS = (10.0, 30.0, 100.0)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []
    for model, entries in PARAM_INFO.items():
        triggered = 0
        for parameter, (lo, hi) in entries.items():
            ratio = hi / lo if lo > 0 else float("inf")
            auto = lo > 0 and ratio >= 100
            triggered += int(auto)
            rows.append({"model": model, "parameter": parameter, "lo": lo, "hi": hi,
                         "hi_over_lo": ratio, "triggers_auto_log_default_100": auto,
                         "linear_midpoint_physical": (lo + hi) / 2,
                         "log_midpoint_physical": (lo * hi) ** .5 if lo > 0 else None})
        summary.append({"model": model, "parameter_count": len(entries), "auto_log_count_default_100": triggered,
                        "auto_log_fraction_default_100": triggered / len(entries)})
    threshold_rows = []
    for threshold in THRESHOLDS:
        for model, entries in PARAM_INFO.items():
            count = sum(float(lo) > 0 and float(hi) / float(lo) >= threshold for lo, hi in entries.values())
            threshold_rows.append({"threshold": threshold, "model": model, "parameter_count": len(entries),
                                   "auto_log_count": count, "auto_log_fraction": count / len(entries)})
    write_csv(OUT / "i2_parameter_mapping_271.csv", rows)
    write_csv(OUT / "i2_model_summary.csv", summary)
    write_csv(OUT / "i2_threshold_sensitivity.csv", threshold_rows)
    (OUT / "i2_contract.json").write_text(json.dumps({
        "source": "dmotpy/models/hydrology_model.py:164-178",
        "default_condition": "lo > 0 and hi/lo >= 100",
        "thresholds": list(THRESHOLDS), "total_parameters": len(rows),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
