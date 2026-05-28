#!/usr/bin/env bash
# ============================================================
# run_distributional_param.sh
#
# Train DistributionalParamModel (from param_models.py) on all
# 531 CAMELS basins simultaneously, using CUDA.
#
# Usage:
#   bash scripts/run_distributional_param.sh [MODEL_ID] [EPOCHS]
#
# Example:
#   bash scripts/run_distributional_param.sh hbv96 100
#   bash scripts/run_distributional_param.sh all 100      # loop over all 36 models
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$BENCHMARK_DIR")")"

MODEL_ID="${1:-hbv96}"
EPOCHS="${2:-100}"
DEVICE="cuda"
HIDDEN_SIZE=128
BATCH_SIZE=531          # all basins at once
NUM_WORKERS=4
SEED=42

TRAIN_START="1989-01-01"
TRAIN_END="1998-12-31"
VAL_START="1999-01-01"
VAL_END="2003-12-31"
TEST_START="2004-01-01"
TEST_END="2009-12-31"

OUTPUT_ROOT="${BENCHMARK_DIR}/outputs/distributional_param"
DATA_PATH="${REPO_ROOT}/data/camels_dataset"
BASIN_IDS_PATH="${REPO_ROOT}/data/531sub_id.txt"
BASIN_IDS_REF_PATH="${REPO_ROOT}/data/gage_id.npy"

echo "============================================================"
echo "  DistributionalParamModel + dmotpy HydrologyModel"
echo "  Device  : ${DEVICE}"
echo "  Basins  : 531 (all, full-batch)"
echo "  Epochs  : ${EPOCHS}"
echo "  Hidden  : ${HIDDEN_SIZE}"
echo "============================================================"

python3 - <<PYEOF
import sys, os
sys.path.insert(0, "${REPO_ROOT}")
sys.path.insert(0, "${BENCHMARK_DIR}")

import torch
import torch.nn.functional as F
from pathlib import Path
import json, csv, time

# ---- imports -------------------------------------------------------
from benchmark.param_models import DistributionalParamModel
from benchmark.models import available_model_ids, build_hydrology_model
from dmotpy.models.registry import PARAM_INFO

DEVICE      = "${DEVICE}"
HIDDEN_SIZE = ${HIDDEN_SIZE}
EPOCHS      = ${EPOCHS}
SEED        = ${SEED}
MODEL_ID    = "${MODEL_ID}"
OUTPUT_ROOT = Path("${OUTPUT_ROOT}")

torch.manual_seed(SEED)

# ---- helper: NSE loss ----------------------------------------------
def nse_loss(sim: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Batch NSE loss (minimise 1 - NSE)."""
    mask  = ~torch.isnan(obs)
    sim_  = sim[mask]
    obs_  = obs[mask]
    mean_obs = obs_.mean()
    nse   = 1.0 - ((obs_ - sim_).pow(2).sum() /
                   (obs_ - mean_obs).pow(2).sum().clamp(min=1e-8))
    return 1.0 - nse

# ---- helper: KL-regularised loss -----------------------------------
def kl_nse_loss(model, sim, obs, beta=1e-3):
    return nse_loss(sim, obs) + beta * model.kl_div_loss()

# ---- model list ----------------------------------------------------
all_models  = available_model_ids()
model_ids   = all_models if MODEL_ID == "all" else [MODEL_ID]

# ---- nx: number of static attributes ------------------------------
# Uses the 35 CAMELS attributes defined in benchmark.yaml
NX_ATTRS = 35
NX_FORC  = 3   # prcp, tmean, pet
NX       = NX_ATTRS   # static only (DistributionalParamModel pools in time)

print(f"Model list: {model_ids}")
print(f"nx (static attributes) = {NX}")
print()

for mid in model_ids:
    ny = len(PARAM_INFO[mid]["bounds"])
    print(f"=== {mid}  (ny={ny}) ===")

    # -- build hydrology model (CPU for data prep, then GPU) --------
    hydro_cfg = {
        "model": {
            "warm_up": 365,
            "warm_up_states": True,
            "forcings": ["prcp", "tmean", "pet"],
            "nearzero": 1e-5,
            "backend": "eager",
        }
    }
    phy_model = build_hydrology_model(hydro_cfg, mid, DEVICE)

    # -- build DistributionalParamModel -----------------------------
    param_model = DistributionalParamModel(
        config={"hidden_size": HIDDEN_SIZE,
                "output_activation": "sigmoid",
                "distribution": {"logstd_min": -5.0, "logstd_max": 2.0}},
        nx=NX,
        ny=ny,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in param_model.parameters() if p.requires_grad)
    print(f"  DistributionalParamModel parameters: {n_params:,}")

    optimizer = torch.optim.Adam(param_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5)

    # -- NOTE: actual data loading requires dmg/HydroLoader ---------
    # The loop below shows the FULL training structure; replace the
    # synthetic tensors with real HydroLoader batches.
    # batch shape: attr  [B, NX]  (B=531 basins)
    #              forc  [T, B, NX_FORC]
    #              obs   [T, B]

    B, T = 531, 3652   # basins, ~10 yr daily
    attr_dummy  = torch.randn(B, NX,       device=DEVICE)
    obs_dummy   = torch.rand(T, B,         device=DEVICE)  # replace with real obs

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        param_model.train()
        optimizer.zero_grad()

        # sample parameters for all basins
        params = param_model(attr_dummy)          # [B, ny]  (or [T, B, ny])
        if params.ndim == 3:
            params = params[0]                    # take single time-step slice

        # run physical model  (interface: phy_model needs x_phy dict)
        # Here we pass params as a placeholder; wire to phy_model.forward() as needed
        # sim = phy_model(x_phy_dict, params)
        # loss = kl_nse_loss(param_model, sim, obs_dummy)

        # --- placeholder loss (replace with real simulation) ---
        loss = params.mean() * 0 + param_model.kl_divergence(attr_dummy)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(param_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:4d}/{EPOCHS}  loss={loss.item():.4f}  "
                  f"elapsed={elapsed:.1f}s")

    # -- save checkpoint --------------------------------------------
    out_dir = OUTPUT_ROOT / mid
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "distributional_param_model.pt"
    torch.save(param_model.state_dict(), ckpt_path)
    print(f"  Saved checkpoint: {ckpt_path}")
    print()

print("All done.")
PYEOF
