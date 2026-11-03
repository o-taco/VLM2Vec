#!/bin/bash
# After the main 2B reproduction (pid $1) exits: 2 extra init seeds and the 5 component
# ablations, each on the identical ordered pool, then dump + routing baselines (no latency).
# One log per job; a failed job is recorded and the queue continues.
cd /workspace/VLM2Vec
while kill -0 "$1" 2>/dev/null; do sleep 60; done
JOBS=("SEED=26091616" "ABLATION=selected_depth" "ABLATION=no_film" "ABLATION=independent"
      "ABLATION=no_last" "ABLATION=no_lora" "SEED=26091617")
for job in "${JOBS[@]}"; do
  log=early_exit_probe/logs/queue_${job#*=}.log
  echo "[queue] $(date -u +%FT%TZ) start $job"
  if env $job SKIP_LATENCY=1 ./early_exit_probe/run_train_eval.sh > "$log" 2>&1; then
    echo "[queue] $(date -u +%FT%TZ) done $job"
  else
    echo "[queue] $(date -u +%FT%TZ) FAILED $job (see $log)"
  fi
done
echo "[queue] ALL DONE"
