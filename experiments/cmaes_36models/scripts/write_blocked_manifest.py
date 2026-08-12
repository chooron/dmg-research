from __future__ import annotations
from _common import EXPERIMENT
from src.reporting import write_blocked_manifest

if __name__ == "__main__": write_blocked_manifest(EXPERIMENT, "torch.compile validation failed on installed Torch 2.9.1+cu128")
