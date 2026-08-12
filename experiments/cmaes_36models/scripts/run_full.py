from __future__ import annotations
from _common import EXPERIMENT
import argparse, json


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--model"); parser.add_argument("--basin-id", type=int); parser.add_argument("--starts", type=int); parser.add_argument("--retry-failed", action="store_true"); args=parser.parse_args()
    pilot=EXPERIMENT/"results/pilot_gate.json"
    if not pilot.exists() or not json.loads(pilot.read_text()).get("passed"):
        raise SystemExit("Refusing formal run: Stage B pilot/compiled-kernel gate is not passed. See reports/pilot_summary.md.")
    raise SystemExit("Formal runner is intentionally unavailable until a successful Stage B manifest is present.")

if __name__ == "__main__": main()
