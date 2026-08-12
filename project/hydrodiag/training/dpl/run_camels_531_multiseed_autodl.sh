#!/usr/bin/env bash
# Run the full dPL model registry on CAMELS-531 with a bounded worker pool.
# Intended for /root/dmg-research/project/hydro_structure_diagnosis on AutoDL.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_DIR/training/dpl/base_config_camels_531_autodl.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/autodl-fs/data/dmg_hydro_structure_diagnosis/dpl_camels_531_multiseed_v1}"
MAX_JOBS="${MAX_JOBS:-6}"
LITE_MODELS="${LITE_MODELS:-0}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"
INDUCTOR_CACHE="${INDUCTOR_CACHE:-/autodl-fs/data/dmg_hydro_structure_diagnosis/torch_inductor_cache_retry}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%dT%H%M%S)}"

MODELS=(XAJ XAJ_CN XAJ_TGD2 GR4J GR4J_CN HBV SIMHYD SIMHYD_CN)
SEEDS=(42 123 2026)

mkdir -p "$OUTPUT_ROOT" "$INDUCTOR_CACHE"
cd "$PROJECT_DIR"

manifest="$OUTPUT_ROOT/manifest.tsv"
if [[ ! -f "$manifest" ]]; then
    printf 'model\tseed\toutput_dir\tstatus\n' > "$manifest"
fi

run_job() {
    local model="$1"
    local seed="$2"
    local output_dir="$OUTPUT_ROOT/${model}/seed_${seed}"

    # Keep successful runs untouched.  Failed/incomplete runs are resumed by
    # run_dpl_model.py from their newest periodic checkpoint.
    if [[ -f "$output_dir/COMPLETE" ]]; then
        printf '%s\t%s\t%s\tskipped_complete\n' "$model" "$seed" "$output_dir" >> "$manifest"
        return 0
    fi

    mkdir -p "$output_dir"
    if [[ -f "$output_dir/train.log" ]]; then
        mv "$output_dir/train.log" "$output_dir/train.log.pre_resume_${RUN_TAG}"
    fi
    if [[ -f "$output_dir/LAUNCHER_FAILED" ]]; then
        mv "$output_dir/LAUNCHER_FAILED" "$output_dir/LAUNCHER_FAILED.pre_resume_${RUN_TAG}"
    fi
    if [[ -f "$output_dir/best_checkpoint.pt" && ! -f "$output_dir/best_checkpoint.pt.pre_resume_${RUN_TAG}" ]]; then
        cp "$output_dir/best_checkpoint.pt" "$output_dir/best_checkpoint.pt.pre_resume_${RUN_TAG}"
    fi
    local task_cache="$INDUCTOR_CACHE/${model}/seed_${seed}"
    mkdir -p "$task_cache"
    printf '%s\t%s\t%s\tstarted\n' "$model" "$seed" "$output_dir" >> "$manifest"

    # A checkpoint written by the pre-fix loss can contain very large but
    # finite weights.  If resuming it still produces a non-finite gradient,
    # retain it with a disabled suffix and retry from the next newest result.
    local attempt=0
    local succeeded=0
    local lite_args=()
    if [[ "$LITE_MODELS" == "1" ]]; then
        lite_args=(--lite)
    fi
    while (( attempt < 4 )); do
        attempt=$((attempt + 1))
        local resume_args=()
        if compgen -G "$output_dir/checkpoint_epoch_*.pt" > /dev/null || [[ -f "$output_dir/best_checkpoint.pt" ]]; then
            resume_args=(--resume)
        fi
        if TORCHINDUCTOR_CACHE_DIR="$task_cache" "$PYTHON_BIN" training/dpl/run_dpl_model.py \
            --config "$CONFIG_PATH" --model "$model" --seed "$seed" --output-dir "$output_dir" \
            "${resume_args[@]}" \
            "${lite_args[@]}" \
            > "$output_dir/train.log" 2>&1; then
            succeeded=1
            break
        fi
        if (( attempt >= 4 )); then
            break
        fi
        mv "$output_dir/train.log" "$output_dir/train.log.resume_failed_${RUN_TAG}_attempt${attempt}"
        local newest
        newest="$(find "$output_dir" -maxdepth 1 -type f -name 'checkpoint_epoch_*.pt' -printf '%f\n' | sort -V | tail -n 1)"
        if [[ -z "$newest" && -f "$output_dir/best_checkpoint.pt" ]]; then
            newest="best_checkpoint.pt"
        fi
        if [[ -z "$newest" ]]; then
            break
        fi
        mv "$output_dir/$newest" "$output_dir/${newest%.pt}.resume_failed_${RUN_TAG}_attempt${attempt}.pt.disabled"
        printf '%s\t%s\t%s\tfallback_checkpoint=%s\n' \
            "$model" "$seed" "$output_dir" "$newest" >> "$manifest"
    done
    if (( succeeded )); then
        touch "$output_dir/LAUNCHER_SUCCESS"
        printf '%s\t%s\t%s\tsuccess\n' "$model" "$seed" "$output_dir" >> "$manifest"
    else
        touch "$output_dir/LAUNCHER_FAILED"
        printf '%s\t%s\t%s\tfailed\n' "$model" "$seed" "$output_dir" >> "$manifest"
        return 1
    fi
}

overall_status=0
for seed in "${SEEDS[@]}"; do
    pending_tasks=()
    for model in "${MODELS[@]}"; do
        output_dir="$OUTPUT_ROOT/${model}/seed_${seed}"
        if [[ ! -f "$output_dir/COMPLETE" ]]; then
            pending_tasks+=("$model $seed")
        fi
    done

    printf 'seed=%s queued_pending_tasks=%s max_parallel=%s\n' \
        "$seed" "${#pending_tasks[@]}" "$MAX_JOBS" >> "$manifest"

    # Seed barrier: do not launch the next seed until this seed's full model
    # set has finished.  Within one seed, keep a dynamic worker pool.
    active=0
    for task in "${pending_tasks[@]}"; do
        read -r model task_seed <<< "$task"
        run_job "$model" "$task_seed" &
        active=$((active + 1))
        if (( active >= MAX_JOBS )); then
            wait -n || overall_status=1
            active=$((active - 1))
        fi
    done
    while (( active > 0 )); do
        wait -n || overall_status=1
        active=$((active - 1))
    done
done

printf 'finished_at=%s\nstatus=%s\n' "$(date -Is)" "$overall_status" > "$OUTPUT_ROOT/launcher_status.txt"
sync
if [[ "$AUTO_SHUTDOWN" == "1" ]]; then
    shutdown -h now
fi
exit "$overall_status"
