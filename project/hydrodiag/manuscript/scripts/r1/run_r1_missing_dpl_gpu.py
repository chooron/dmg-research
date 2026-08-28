"""Run missing R1 dPL XAJ daily forward passes on CUDA with torch.compile.

This is inference from existing local trained checkpoints only; it does not train
or calibrate.  The data root is explicitly authorized for this run and is read
from the sibling repository data directory.  Outputs remain in hydrodiag's
manuscript cache and are never promoted automatically.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT.parents[1] / "data"
RESULTS_ROOT = PROJECT / "results"
OUTPUT_ROOT = PROJECT / "manuscript" / "cache" / "r1_rebuild_audit_staged" / "daily_dpl_gpu_compile"
BATCH_SIZE = int(os.environ.get("R1_GPU_BATCH_SIZE", "128"))

if str(PROJECT / "manuscript" / "scripts" / "r1") not in sys.path:
    sys.path.insert(0, str(PROJECT / "manuscript" / "scripts" / "r1"))


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable")
    required = [
        DATA_ROOT / "camels_dataset",
        DATA_ROOT / "camels_dates.npy",
        DATA_ROOT / "gage_id.npy",
        DATA_ROOT / "531sub_id.txt",
        RESULTS_ROOT / "dpl_camels_531_lite_v2" / "XAJ",
        RESULTS_ROOT / "dpl_camels_531_lite_v2" / "XAJ_CN",
        RESULTS_ROOT / "dpl_camels_531_lite_v3_tgd2_dpl_audited" / "XAJ_TGD2",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing authorized R1 GPU inputs: " + "; ".join(missing))

    from r1_daily_inference import run_daily_export

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print(f"torch={torch.__version__} hydrological_step_compile=True batch_size={BATCH_SIZE}")
    print(f"data_root={DATA_ROOT}")
    print(f"output_root={OUTPUT_ROOT}")
    result = run_daily_export(
        project_root=PROJECT,
        results_root=RESULTS_ROOT,
        data_root=DATA_ROOT,
        output_root=OUTPUT_ROOT,
        device="cuda",
        batch_size=BATCH_SIZE,
        model_keys=("XAJ", "XAJ_TGD2", "XAJ_CN"),
        paradigm="dpl",
    )
    print(f"status={result.get('status', 'unknown')}")
    print("training_launched=False")
    print("calibration_launched=False")
    print("inference_mode=existing_checkpoints_only")


if __name__ == "__main__":
    main()
