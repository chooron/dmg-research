from __future__ import annotations
from _common import EXPERIMENT
import argparse, json
from src.checkpointing import load_checkpoint


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); args=parser.parse_args()
    payload=load_checkpoint(args.checkpoint,"cpu")
    print(json.dumps({"model":payload["model"],"generation":payload["solver"]["state"]["generation"],"basins":len(payload["basin_ids"]),"starts":payload["starts"]},indent=2))

if __name__ == "__main__": main()
