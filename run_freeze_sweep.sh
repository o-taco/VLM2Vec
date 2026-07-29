#!/bin/bash
# Adapter-coverage sweep on Qwen3-VL-2B: full 28-layer model kept intact, LoRA
# restricted to the LAST k% of decoder layers (rest stay frozen but still run
# forward -- unlike run_keepfirst_sweep.sh, which physically removes layers).
#
# Point of comparison: at 0% frozen this should reproduce the existing 0%-pruned
# LoRA point (~85% in-batch acc, see figures/depth_pruning/pruning_accuracy_keepfirst_only.png).
# Whether the intermediate points fall smoothly toward the teammate's linear-probe
# floor is the open question this sweep is meant to answer -- it can't reach that
# floor itself (100% frozen = zero trainable params in this codebase, a no-op run),
# so only 0/25/50/75% frozen are covered here.
set -uo pipefail
cd /workspace/VLM2Vec
source /venv/main/bin/activate

TOTAL_LAYERS=28
RESULTS=/tmp/freeze_sweep_results.txt
touch "$RESULTS"

run_config() {
  local PCT_FROZEN=$1 TAG=$2
  local NUM_ADAPTED=$(( TOTAL_LAYERS * (100 - PCT_FROZEN) / 100 ))
  local FIRST_ADAPTED=$(( TOTAL_LAYERS - NUM_ADAPTED ))
  local LAYERS=""
  if [ "$NUM_ADAPTED" -gt 0 ]; then
    LAYERS=$(python -c "print(','.join(str(i) for i in range($FIRST_ADAPTED, $TOTAL_LAYERS)))")
  fi
  local OUT_DIR="output/aokvqa_qwen3vl_freeze_${TAG}"

  echo "=================================================="
  echo "=== Starting ${PCT_FROZEN}% frozen ($TAG): adapting layers [$LAYERS] $(date) ==="
  echo "=================================================="

  echo "--- training ---"
  TAG=$TAG LAYERS=$LAYERS bash examples/qwen3_vl/run_train_aokvqa_freeze.sh
  if [ $? -ne 0 ]; then echo "TRAIN FAILED for $TAG"; return 1; fi

  echo "--- scanning checkpoints ---"
  local BEST_ACC=-1 BEST_STEP="" LATEST_ACC="" step CKPT OUT ACC IS_BETTER
  for step in 320 340 360 380 400 420 440 460 480 500; do
    CKPT="$OUT_DIR/checkpoint-$step"
    if [ ! -d "$CKPT" ]; then continue; fi
    OUT=$(python compute_train_accuracy.py \
      --model_name "Qwen/Qwen3-VL-2B-Instruct" \
      --model_backbone qwen3_vl \
      --checkpoint_path "$CKPT" \
      --batch_size 4 --num_batches 50 --seed 42 2>&1)
    ACC=$(echo "$OUT" | grep -oP 'top-1 accuracy.*?: \K[0-9.]+')
    echo "checkpoint-$step acc=$ACC"
    if [ -z "$ACC" ]; then continue; fi
    if [ "$step" == "500" ]; then LATEST_ACC=$ACC; fi
    IS_BETTER=$(python -c "print(1 if $ACC > $BEST_ACC else 0)")
    if [ "$IS_BETTER" == "1" ]; then BEST_ACC=$ACC; BEST_STEP=$step; fi
  done
  echo "best: checkpoint-$BEST_STEP acc=$BEST_ACC ; latest: checkpoint-500 acc=$LATEST_ACC"
  if [ -z "$BEST_STEP" ]; then echo "NO VALID CHECKPOINTS for $TAG"; return 1; fi

  echo "$PCT_FROZEN $TAG num_adapted=$NUM_ADAPTED best_step=$BEST_STEP best_acc=$BEST_ACC latest_acc=$LATEST_ACC" >> "$RESULTS"
  echo "=== Finished $TAG $(date) ==="
  df -h /workspace | tail -1
}

run_config 0  "0pct_frozen_all_adapted"
run_config 25 "25pct_frozen_last75pct"
run_config 50 "50pct_frozen_last50pct"
run_config 75 "75pct_frozen_last25pct"

echo "ALL DONE $(date)"
