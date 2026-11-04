#!/bin/bash
# Reproduce the paper's Qwen3-VL-2B every-layer row on this V100, then run the analyses
# that need its per-row outputs. Default pool is Table 3's A-OKVQA/CUB-200/ScienceQA rows only
# (7,511 rows, 939 updates); DOMAINS=all adds ImageNet + iNaturalist (8,192 rows, 1,024 updates;
# needs HF_TOKEN in /workspace/.env for the gated ImageNet repo).
set -e
cd /workspace/VLM2Vec
export EARLY_EXIT_IMAGE_SIZE=336
set -a; [ -f /workspace/.env ] && . /workspace/.env; set +a
SEED=${SEED:-26091615}
ABLATION=${ABLATION:-none}
TAG=$([ "$ABLATION" = none ] && echo "" || echo "_$ABLATION")
if [ "${DOMAINS:-}" = all ]; then
  DOM="A-OKVQA CUB-200 ScienceQA ImageNet iNaturalist"; TAG="5d$TAG"
else
  DOM="A-OKVQA CUB-200 ScienceQA"
fi
OUT=early_exit_probe/outputs/repro2b_el${TAG}_s$SEED
PY=/venv/main/bin/python
$PY -m early_exit_probe.train_every_layer --domains $DOM --seed $SEED --ablation $ABLATION --output_dir $OUT
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
