#!/bin/bash
# Runs after the confidence re-dump: confidence-routing baselines, then the every-layer
# latency profile on a clean GPU (paper 336x336 images), then the paper-depth latency estimate.
set -e
cd /workspace/VLM2Vec
C=early_exit_probe/outputs/phase3_pool_run1/checkpoint-1000
until grep -qE "wrote .*routing_diagnosis_conf|Traceback" early_exit_probe/logs/diagnose_conf.log; do sleep 30; done
grep -q Traceback early_exit_probe/logs/diagnose_conf.log && { echo "re-dump failed"; exit 1; }
/venv/main/bin/python -m early_exit_probe.routing_baselines --rows $C/routing_diagnosis_conf_rows.pt --sha_split 200 --out $C/baselines/sha200_conf
EARLY_EXIT_IMAGE_SIZE=336 /venv/main/bin/python -m early_exit_probe.latency --checkpoint $C --profile_every_layer --generation --limit 30 --warmup 3 --out $C/latency/profile_every_layer_bs1_336.json
/venv/main/bin/python -m early_exit_probe.paper_latency --profile $C/latency/profile_every_layer_bs1_336.json --out $C/latency/paper_2b_estimate_336.json
echo "[queue] ALL DONE"
