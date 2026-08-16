#!/bin/bash
# dPL 36-model scheduler: 6 workers, dynamic fill, resume-aware, retry-on-fail
set -u
PY=/root/miniconda3/bin/python
CODE=/autodl-fs/data/dpl_run_20260814
RES=/autodl-fs/data/dmg-research-results/dpl_full_retrain_20260813
RUN=$RES/run_20260815
mkdir -p $RUN/logs
QUEUE=$RUN/queue.txt
LOCK=$RUN/pop.lock
touch $QUEUE

log() { echo "[$(date +%F" "%T)] $*" >> $RUN/master.log; }

# ---- build queue: all 36 minus models with terminal health rows ----
cd $CODE/project/benchmark
PYTHONPATH=$CODE $PY - "$RES/auto100/health.csv" "$QUEUE" <<'PYEOF'
import csv, sys
from src.model_registry import NPARAM_INFO_36
health_path, queue_path = sys.argv[1], sys.argv[2]
done = set()
try:
    with open(health_path) as f:
        for row in csv.DictReader(f):
            if row["status"] in ("COMPLETED", "PLATEAU_STOP") or int(row.get("stop_epoch", 0)) >= 50:
                done.add(row["model"])
except FileNotFoundError:
    pass
models = [m for m in NPARAM_INFO_36 if m not in done]
with open(queue_path, "w") as f:
    f.write("\n".join(models) + "\n")
print("QUEUE_BUILT models=%d skipped=%s" % (len(models), sorted(done)))
PYEOF

pop_next() {
  while ! mkdir $LOCK 2>/dev/null; do sleep 1; done
  local m=""
  if [ -s $QUEUE ]; then
    m=$(head -1 $QUEUE)
    sed -i "1d" $QUEUE
  fi
  rmdir $LOCK
  echo "$m"
}

worker() {
  local id=$1
  while true; do
    local m=$(pop_next)
    [ -z "$m" ] && break
    local tries=0
    while [ $tries -lt 3 ]; do
      log "worker$id start $m try=$((tries+1))"
      ( cd $CODE/project/benchmark && PYTHONPATH=$CODE $PY scripts/diagnostics/k_full_retrain.py --arm auto100 --model $m ) >> $RUN/logs/$m.log 2>&1
      local rc=$?
      log "worker$id end $m rc=$rc"
      if grep -q "^$m," $RES/auto100/health.csv 2>/dev/null; then break; fi
      tries=$((tries+1))
      [ $tries -lt 3 ] && sleep 5
    done
    if [ $tries -ge 3 ]; then echo "$m" >> $RUN/failed.txt; log "worker$id GAVE_UP $m"; fi
  done
  log "worker$id exit"
}

log "SCHEDULER_START workers=6"
for i in $(seq 1 6); do worker $i & done
wait
log "SCHEDULER_ALL_DONE"
