#!/usr/bin/env bash
# Restart-safe watchdog for the 4 controlled CN+XAJ dPL jobs.
#
# The container can be restarted by the host (AutoDL/Seetacloud) at any time,
# which kills the launcher and all four training processes.  Checkpoints are
# written every 10 epochs, so a re-launch resumes from the newest checkpoint.
#
# Behaviour:
#   - single instance enforced with flock;
#   - every 5 minutes: if any model lacks COMPLETE and neither the launcher nor
#     any run_dpl_model.py process is alive -> relaunch the launcher (which
#     skips COMPLETE models and resumes the rest from newest checkpoints);
#   - a 10-minute cooldown after each relaunch prevents double-launch races;
#   - exits once all four models have COMPLETE (writes WATCHDOG_DONE).
#
# Launch detached:
#   setsid nohup bash resume_dpl_controlled_531_v1.sh > /dev/null 2>&1 < /dev/null &

set -uo pipefail

OUTPUT_ROOT="/autodl-fs/data/dmg_hydro_structure_diagnosis/dpl_controlled_531_v1"
PROJECT_DIR="/root/hydro_structure_diagnosis"
SEED=42
MODELS=(XAJ_D_E_CN XAJ_G_E_CN XAJ_D_R_CN XAJ_G_R_CN)
LOG="$OUTPUT_ROOT/watchdog.log"
GIT_COMMIT="22d8ebb4911e197e4567c5653a7563ffd4a8dbbf"
COOLDOWN_S=600
CHECK_S=300

exec 9>"$OUTPUT_ROOT/watchdog.lock"
flock -n 9 || { echo "$(date -Is) another watchdog holds the lock; exiting" >> "$LOG"; exit 0; }

echo "$(date -Is) watchdog started (pid $$)" >> "$LOG"

last_relaunch=0
while true; do
    all_done=1
    for M in "${MODELS[@]}"; do
        [[ -f "$OUTPUT_ROOT/$M/seed_$SEED/COMPLETE" ]] || all_done=0
    done
    if (( all_done )); then
        echo "$(date -Is) all models COMPLETE; watchdog exiting" >> "$LOG"
        touch "$OUTPUT_ROOT/WATCHDOG_DONE"
        exit 0
    fi

    launcher_alive=0
    pgrep -f 'training/dpl/launch_dpl_controlled_531_v1.sh' > /dev/null && launcher_alive=1
    jobs_alive=0
    pgrep -f 'python training/dpl/run_dpl_model.py' > /dev/null && jobs_alive=1

    now=$(date +%s)
    if (( launcher_alive || jobs_alive )); then
        : # batch still running; nothing to do
    elif (( now - last_relaunch < COOLDOWN_S )); then
        : # just relaunched; avoid double-launch
    else
        echo "$(date -Is) launcher/jobs down, incomplete models present -> relaunching (resume from newest checkpoints)" >> "$LOG"
        ( exec 9>&-; cd "$PROJECT_DIR" && GIT_COMMIT="$GIT_COMMIT" \
            setsid nohup bash training/dpl/launch_dpl_controlled_531_v1.sh \
            >> "$OUTPUT_ROOT/launcher.out" 2>&1 < /dev/null & )
        last_relaunch=$now
    fi
    sleep "$CHECK_S" 9>&-
done
