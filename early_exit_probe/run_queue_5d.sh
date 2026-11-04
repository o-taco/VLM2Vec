#!/bin/bash
# 5-domain (full Table 3 pool) reproduction: 3 init seeds, latency skipped (it depends on the
# exit depths, not the training data, and was measured on the 3-domain run).
cd /workspace/VLM2Vec
for seed in 26091615 26091616 26091617; do
  log=early_exit_probe/logs/repro2b_5d_s$seed.log
  echo "[queue5d] $(date -u +%FT%TZ) start seed $seed"
  if DOMAINS=all SEED=$seed SKIP_LATENCY=1 ./early_exit_probe/run_train_eval.sh > "$log" 2>&1; then
    echo "[queue5d] $(date -u +%FT%TZ) done seed $seed"
  else
    echo "[queue5d] $(date -u +%FT%TZ) FAILED seed $seed (see $log)"
  fi
done
echo "[queue5d] ALL DONE"
