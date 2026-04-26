#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

ACTION=${ACTION:-train}
CONFIG=${CONFIG:-${PROJECT_DIR}/conf/neuralhydrology_lstm_config.yml}
RUN_DIR=${RUN_DIR:-}

cd "${PROJECT_DIR}"

case "${ACTION}" in
    train)
        echo "bettermodel | pure_lstm | action=train | config=${CONFIG}"
        nh-run train --config-file "${CONFIG}"
        ;;
    evaluate)
        if [[ -z "${RUN_DIR}" ]]; then
            echo "RUN_DIR is required when ACTION=evaluate" >&2
            exit 1
        fi
        echo "bettermodel | pure_lstm | action=evaluate | run_dir=${RUN_DIR}"
        nh-run evaluate --run-dir "${RUN_DIR}"
        ;;
    both)
        if [[ -z "${RUN_DIR}" ]]; then
            echo "RUN_DIR is required when ACTION=both" >&2
            exit 1
        fi
        echo "bettermodel | pure_lstm | action=train | config=${CONFIG}"
        nh-run train --config-file "${CONFIG}"
        echo "bettermodel | pure_lstm | action=evaluate | run_dir=${RUN_DIR}"
        nh-run evaluate --run-dir "${RUN_DIR}"
        ;;
    *)
        echo "Unsupported ACTION=${ACTION}. Use train, evaluate, or both." >&2
        exit 1
        ;;
esac
