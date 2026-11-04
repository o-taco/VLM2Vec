#!/bin/bash
# The 5 component ablations on the 5-domain pool (DOMAINS=all), seed 26091615, same ordered
# pool as the 5-domain main run; dump + routing baselines, no latency.
# One log per job; a failed job is recorded and the queue continues.
cd /workspace/VLM2Vec
for abl in selected_depth no_film independent no_last no_lora; do
  log=early_exit_probe/logs/repro2b_5d_$abl.log
  echo "[queue5d-abl] $(date -u +%FT%TZ) start $abl"
  if DOMAINS=all ABLATION=$abl SKIP_LATENCY=1 ./early_exit_probe/run_train_eval.sh > "$log" 2>&1; then
    echo "[queue5d-abl] $(date -u +%FT%TZ) done $abl"
  else
    echo "[queue5d-abl] $(date -u +%FT%TZ) FAILED $abl (see $log)"
  fi
done
echo "[queue5d-abl] ALL DONE"
