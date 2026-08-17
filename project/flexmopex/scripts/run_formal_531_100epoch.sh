#!/usr/bin/env bash
# =============================================================================
# Flex-MOPEX 100-Epoch Formal CAMELS-531 Production Launcher
#
# Runs:
#   1. Base:        Fixed w=[0,0,0,0], Candidate E-S0, ParamRoutingNet, MyTrainer
#   2. Full:        Fixed w=[1,1,1,1], Candidate E-S0, ParamRoutingNet, MyTrainer
#   3. Flex-0.005:  Learned w, Candidate E-S0, HybridEncoder, CFTrainer, lambda=0.005
#   4. Flex-0.007:  Learned w, Candidate E-S0, HybridEncoder, CFTrainer, lambda=0.007
#   5. Flex-0.010:  Learned w, Candidate E-S0, HybridEncoder, CFTrainer, lambda=0.010
#
# Protocol:
#   - CAMELS-US 531 Basins (data/531sub_id.txt)
#   - 100 Epochs, Checkpoints every epoch (save_epoch=1), Evaluation at Epoch 100
#   - Unified Adadelta (lr=1.0), Seed=42
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXMOPEX_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${FLEXMOPEX_DIR}/../.." && pwd)"

# Python environment
PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"

cd "${FLEXMOPEX_DIR}"

run_variant() {
    local variant="$1"
    local config_file="$2"
    local output_root="results/formal_531"

    echo "============================================================================="
    echo "Starting 100-epoch formal run: ${variant}"
    echo "Config: ${config_file}"
    echo "Time: $(date)"
    echo "============================================================================="

    "${PYTHON}" run_model.py \
        --config "${config_file}" \
        --output-root "${output_root}" \
        --gpu-id "${GPU_ID}" \
        --mode train_test

    echo "Finished 100-epoch formal run: ${variant} at $(date)"
}

TARGET="${1:-all}"

case "${TARGET}" in
    base)
        run_variant "Base" "conf/config_formal_531_base.yaml"
        ;;
    full)
        run_variant "Full" "conf/config_formal_531_full.yaml"
        ;;
    flex-0.005|flex0005)
        run_variant "Flex-0.005" "conf/config_formal_531_flex_lambda0005.yaml"
        ;;
    flex-0.007|flex0007)
        run_variant "Flex-0.007" "conf/config_formal_531_flex_lambda0007.yaml"
        ;;
    flex-0.010|flex0010)
        run_variant "Flex-0.010" "conf/config_formal_531_flex_lambda0010.yaml"
        ;;
    all)
        echo "Launching all 5 formal 100-epoch runs sequentially on GPU ${GPU_ID}..."
        run_variant "Base" "conf/config_formal_531_base.yaml"
        run_variant "Full" "conf/config_formal_531_full.yaml"
        run_variant "Flex-0.005" "conf/config_formal_531_flex_lambda0005.yaml"
        run_variant "Flex-0.007" "conf/config_formal_531_flex_lambda0007.yaml"
        run_variant "Flex-0.010" "conf/config_formal_531_flex_lambda0010.yaml"
        ;;
    *)
        echo "Usage: $0 [base|full|flex-0.005|flex-0.007|flex-0.010|all]"
        exit 1
        ;;
esac
