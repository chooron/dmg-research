#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

CONFIG=${CONFIG:-${PROJECT_DIR}/conf/config_param_paper.yaml}
SCRIPT=${SCRIPT:-train_param_paper.py}
MODE=${MODE:-train_test}
EPOCHS=${EPOCHS:-}
MC_SAMPLES=${MC_SAMPLES:-}

VARIANTS=(deterministic mc_dropout distributional)
SEEDS=(42 123)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variants)
            VARIANTS=()
            shift
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                VARIANTS+=("$1")
                shift
            done
            ;;
        --seeds)
            SEEDS=()
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
        --mc-samples)
            MC_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "${PROJECT_DIR}"

for variant in "${VARIANTS[@]}"; do
    pids=()
    for seed in "${SEEDS[@]}"; do
        echo "========================================"
        echo " paper-stack | variant=${variant} | seed=${seed}"
        echo " config=${CONFIG} | mode=${MODE}"
        echo "========================================"

        extra_args=(--config "${CONFIG}" --variant "${variant}" --seed "${seed}" --mode "${MODE}")
        if [[ -n "${EPOCHS:-}" ]]; then
            extra_args+=(--epochs "${EPOCHS}")
        fi
        if [[ -n "${MC_SAMPLES:-}" ]]; then
            extra_args+=(--mc-samples "${MC_SAMPLES}")
        fi

        uv run python "${SCRIPT}" "${extra_args[@]}" &
        pids+=("$!")
    done

    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
done
