#!/usr/bin/env bash
# Launch the 4 controlled CN+XAJ dPL variants (D_E/G_E/D_R/G_R) on CAMELS-531,
# one seed (42), concurrently on the single AutoDL GPU.
#
# Protocol: exact copy of the canonical CN+XAJ dPL run
#   /autodl-fs/data/dmg_hydro_structure_diagnosis/dpl_camels_531_lite_v2/XAJ_CN/seed_42
# (100 epochs, batch 128, AdamW lr 1e-3 wd 1e-4, cosine to 1e-4, grad clip 1.0,
#  balanced_valid_kge_windows sampling, 365+365 windows, val/ckpt every 10 epochs,
#  float32 forward / float64 metric, lite mode, seed 42).
#
# Launch detached from an interactive shell:
#   setsid nohup bash launch_dpl_controlled_531_v1.sh > launcher.out 2>&1 < /dev/null &
#
# Each model: independent output dir, train.log, pid file, inductor cache,
# LAUNCHER_SUCCESS / LAUNCHER_FAILED markers, resume from newest checkpoint.

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/hydro_structure_diagnosis}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/autodl-fs/data/dmg_hydro_structure_diagnosis/dpl_controlled_531_v1}"
SEED="${SEED:-42}"
INDUCTOR_CACHE="${INDUCTOR_CACHE:-/autodl-fs/data/dmg_hydro_structure_diagnosis/torch_inductor_cache_controlled_531}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%dT%H%M%S)}"
CONFIG_DIR="$PROJECT_DIR/training/dpl/generated_configs"

MODELS=(XAJ_D_E_CN XAJ_G_E_CN XAJ_D_R_CN XAJ_G_R_CN)

mkdir -p "$OUTPUT_ROOT" "$INDUCTOR_CACHE"
cd "$PROJECT_DIR"

manifest="$OUTPUT_ROOT/manifest.tsv"
[[ -f "$manifest" ]] || printf 'model\tseed\toutput_dir\tstatus\n' > "$manifest"

run_job() {
    local model="$1"
    local output_dir="$OUTPUT_ROOT/${model}/seed_${SEED}"
    local config="$CONFIG_DIR/dpl_controlled_531_v1_${model}_seed_${SEED}.json"
    local task_cache="$INDUCTOR_CACHE/${model}/seed_${SEED}"
    mkdir -p "$output_dir" "$task_cache"

    if [[ -f "$output_dir/COMPLETE" ]]; then
        printf '%s\t%s\t%s\tskipped_complete\n' "$model" "$SEED" "$output_dir" >> "$manifest"
        return 0
    fi
    if [[ -f "$output_dir/train.log" ]]; then
        mv "$output_dir/train.log" "$output_dir/train.log.pre_launch_${RUN_TAG}"
    fi
    if [[ -f "$output_dir/LAUNCHER_FAILED" ]]; then
        mv "$output_dir/LAUNCHER_FAILED" "$output_dir/LAUNCHER_FAILED.pre_launch_${RUN_TAG}"
    fi
    if [[ -f "$output_dir/best_checkpoint.pt" && ! -f "$output_dir/best_checkpoint.pt.pre_launch_${RUN_TAG}" ]]; then
        cp "$output_dir/best_checkpoint.pt" "$output_dir/best_checkpoint.pt.pre_launch_${RUN_TAG}"
    fi

    local resume_args=()
    if compgen -G "$output_dir/checkpoint_epoch_*.pt" > /dev/null || [[ -f "$output_dir/best_checkpoint.pt" ]]; then
        resume_args=(--resume)
    fi

    printf '%s\t%s\t%s\tstarted\n' "$model" "$SEED" "$output_dir" >> "$manifest"

    # Each model is one independent background process on the shared GPU.
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments keeps 4 processes from
    # fragmenting the 12 GiB device (project-probed: 4-way dPL peak ~1.6 GiB).
    local attempt=0
    local succeeded=0
    while (( attempt < 4 )); do
        attempt=$((attempt + 1))
        local resume_args2=()
        if compgen -G "$output_dir/checkpoint_epoch_*.pt" > /dev/null || [[ -f "$output_dir/best_checkpoint.pt" ]]; then
            resume_args2=(--resume)
        fi
        if env TORCHINDUCTOR_CACHE_DIR="$task_cache" \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
            "$PYTHON_BIN" training/dpl/run_dpl_model.py \
            --config "$config" --model "$model" --seed "$SEED" \
            --output-dir "$output_dir" --lite \
            ${resume_args2[@]+"${resume_args2[@]}"} \
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
        printf '%s\t%s\t%s\tfallback_checkpoint=%s\n' "$model" "$SEED" "$output_dir" "$newest" >> "$manifest"
    done
    if (( succeeded )); then
        touch "$output_dir/LAUNCHER_SUCCESS"
        printf '%s\t%s\t%s\tsuccess\n' "$model" "$SEED" "$output_dir" >> "$manifest"
    else
        touch "$output_dir/LAUNCHER_FAILED"
        printf '%s\t%s\t%s\tfailed\n' "$model" "$SEED" "$output_dir" >> "$manifest"
        return 1
    fi
}

# Save resolved protocol / version stamp once at launch time.
{
    echo "{"
    echo "  \"protocol\": \"dpl_camels_531_flexmopex_period_v1\","
    echo "  \"reference_baseline\": \"dpl_camels_531_lite_v2/XAJ_CN/seed_42\","
    echo "  \"seed\": $SEED,"
    echo "  \"models\": [\"XAJ_D_E_CN\", \"XAJ_G_E_CN\", \"XAJ_D_R_CN\", \"XAJ_G_R_CN\"],"
    echo "  \"git_commit_local\": \"${GIT_COMMIT:-no-git}\","
    echo "  \"launched_at\": \"$(date -Is)\""
    echo "}"
} > "$OUTPUT_ROOT/run_meta.json"

overall_status=0
pids=()
for model in "${MODELS[@]}"; do
    output_dir="$OUTPUT_ROOT/${model}/seed_${SEED}"
    mkdir -p "$output_dir"
    run_job "$model" &
    pids+=("$!")
    echo "$!" > "$output_dir/launcher_pid.txt"
done

for pid in "${pids[@]}"; do
    wait "$pid" || overall_status=1
done

printf 'finished_at=%s\nstatus=%s\n' "$(date -Is)" "$overall_status" > "$OUTPUT_ROOT/launcher_status.txt"
sync
exit "$overall_status"
