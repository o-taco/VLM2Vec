#!/bin/bash
# Reproduce the paper's Qwen3-VL-2B every-layer row on this V100, then run the analyses
# that need its per-row outputs. ImageNet/iNaturalist rows are not on this box, so the pool
# is Table 3's A-OKVQA/CUB-200/ScienceQA rows only (7,511 rows, 939 updates).
set -e
cd /workspace/VLM2Vec
export EARLY_EXIT_IMAGE_SIZE=336
SEED=${SEED:-26091615}
ABLATION=${ABLATION:-none}
TAG=$([ "$ABLATION" = none ] && echo "" || echo "_$ABLATION")
OUT=early_exit_probe/outputs/repro2b_el${TAG}_s$SEED
PY=/venv/main/bin/python
$PY -m early_exit_probe.train_every_layer --domains A-OKVQA CUB-200 ScienceQA --seed $SEED --ablation $ABLATION --output_dir $OUT
CK=$(ls -d $OUT/checkpoint-* | sort -t- -k2 -n | tail -1)
echo "[repro] final checkpoint $CK"
$PY -m early_exit_probe.diagnose_routing --checkpoint $CK --limit 1000
$PY -m early_exit_probe.routing_baselines --rows $CK/routing_diagnosis_rows.pt --sha_split 200 \
    --write_thresholds $CK/thresholds.json --out $CK/baselines/sha200
[ "${SKIP_LATENCY:-0}" = 1 ] && { echo "[repro] ALL DONE (latency skipped)"; exit 0; }
$PY -m early_exit_probe.latency --checkpoint $CK --profile_every_layer --generation --limit 30 --warmup 3 \
    --out $CK/latency/profile_every_layer_bs1.json
$PY -m early_exit_probe.paper_latency --profile $CK/latency/profile_every_layer_bs1.json --out $CK/latency/paper_2b_estimate.json
$PY -m early_exit_probe.latency --checkpoint $CK --limit 50 --warmup 3 --out $CK/latency/halting_bs1.json
echo "[repro] ALL DONE"
