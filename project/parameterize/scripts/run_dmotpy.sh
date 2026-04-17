#!/usr/bin/env bash
# Run dmotpy hydrology models with Calibrate/Parameterize neural networks.
# Usage:
#   bash scripts/run_dmotpy_test.sh
#   bash scripts/run_dmotpy_test.sh --max-jobs 4
#   bash scripts/run_dmotpy_test.sh --models hbv96 gr4j --nn Calibrate
#
# Optional environment overrides:
#   CONFIG=./conf/config_dmotpy_test.yaml
#   MODE=train_test
#   SEEDS="42 123 456"
#   EPOCHS=100
#   MAX_JOBS=4  # maximum parallel jobs (default: number of CPUs)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

CONFIG=${CONFIG:-${PROJECT_DIR}/conf/config_dmotpy_test.yaml}
SCRIPT=${SCRIPT:-train_dmotpy.py}
MODE=${MODE:-train_test}
EPOCHS=${EPOCHS:-}
MAX_JOBS=${MAX_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}

# Parse command line arguments
MODELS=()
NN_MODELS=()
SEEDS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --models)
            shift
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --nn)
            shift
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                NN_MODELS+=("$1")
                shift
            done
            ;;
        --seeds)
            shift
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                SEEDS+=("$1")
                shift
            done
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default values if not provided
if [[ ${#MODELS[@]} -eq 0 ]]; then
    # All dmotpy models
    MODELS=(
        alpine1 alpine2 australia collie1 collie2 collie3
        flexb flexi flexis gr4j gsfb hbv96 hillslope hymod
        ihacres modhydrolog mopex1 mopex2 mopex3 mopex4 mopex5
        newzealand1 newzealand2 penman plateau simhyd smar
        susannah1 susannah2 tank tcm topmodel us1 vic wetland xinanjiang
    )
fi

if [[ ${#NN_MODELS[@]} -eq 0 ]]; then
    NN_MODELS=(Calibrate Parameterize)
fi

if [[ ${#SEEDS[@]} -eq 0 ]]; then
    SEEDS=(42)
fi

cd "${PROJECT_DIR}"

# Build extra arguments
build_extra_args() {
    local extra_args=()
    extra_args+=(--mode "${MODE}")
    if [[ -n "${EPOCHS:-}" ]]; then
        extra_args+=(--epochs "${EPOCHS}")
    fi
    echo "${extra_args[@]}"
}

# Run a single job
run_job() {
    local model_name=$1
    local nn_model=$2
    local seed=$3
    local extra_args
    extra_args=$(build_extra_args)
    
    echo "========================================"
    echo " dmotpy | model=${model_name} | nn=${nn_model} | seed=${seed}"
    echo " Config: ${CONFIG} | mode: ${MODE}"
    echo "========================================"
    
    uv run python "${SCRIPT}" \
        --config "${CONFIG}" \
        --model-name "${model_name}" \
        --nn-model "${nn_model}" \
        --seed "${seed}" \
        ${extra_args}
}

# Generate all job combinations
generate_jobs() {
    for model in "${MODELS[@]}"; do
        for nn in "${NN_MODELS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                echo "${model} ${nn} ${seed}"
            done
        done
    done
}

# Run jobs in parallel with limited concurrency
run_parallel() {
    local pids=()
    local active_jobs=0
    local total_jobs=0
    local completed_jobs=0
    
    # Count total jobs
    total_jobs=$(generate_jobs | wc -l)
    echo "Total jobs: ${total_jobs}"
    echo "Max parallel jobs: ${MAX_JOBS}"
    echo "Models: ${MODELS[*]}"
    echo "NN models: ${NN_MODELS[*]}"
    echo "Seeds: ${SEEDS[*]}"
    echo ""
    
    while IFS=' ' read -r model nn seed; do
        # Wait if we've reached max jobs
        while [[ ${active_jobs} -ge ${MAX_JOBS} ]]; do
            wait -n || true
            ((active_jobs--))
            ((completed_jobs++))
        done
        
        # Start job in background
        run_job "${model}" "${nn}" "${seed}" &
        ((active_jobs++))
        ((total_jobs++))
    done < <(generate_jobs)
    
    # Wait for all remaining jobs
    wait
    echo "All jobs completed."
}

# Main execution
if [[ $# -ge 3 ]]; then
    # Run specific job
    run_job "$1" "$2" "$3"
else
    # Run all combinations in parallel
    run_parallel
fi