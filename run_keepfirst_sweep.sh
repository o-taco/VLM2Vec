#!/bin/bash
# Orchestrates the full keep-first-N contiguous-drop sweep across the remaining
# pruning ratios (25%, 28.6%, 35.7%, 50%, 75% -- 32.1% already done separately).
# For each ratio: prune -> train 500 steps -> scan checkpoints for best in-batch
# accuracy -> rename best/latest -> run similarity diagnostics -> dump raw
# matrix -> free the pruned base model dir before moving to the next ratio.
set -uo pipefail
cd /workspace/VLM2Vec
source /venv/main/bin/activate

RESULTS=/tmp/keepfirst_sweep_results.txt
touch "$RESULTS"

run_ratio() {
  local N=$1 LABEL=$2
  echo "=================================================="
  echo "=== Starting N=$N layers ($LABEL) $(date) ==="
  echo "=================================================="

  local KEEP_IDX MODEL_DIR OUT_DIR
  KEEP_IDX=$(python -c "print(','.join(str(i) for i in range($N)))")
  MODEL_DIR="pruned_models/qwen3vl-2b-${N}layer-keepfirst"
  OUT_DIR="output/aokvqa_qwen3vl_${N}layer_keepfirst"

  echo "--- pruning ---"
  python prune_model.py --keep_idx "$KEEP_IDX" --out_dir "$MODEL_DIR"
  if [ $? -ne 0 ]; then echo "PRUNE FAILED for $N"; return 1; fi

  echo "--- training ---"
  bash examples/qwen3_vl/run_train_aokvqa_${N}layer_keepfirst.sh
  if [ $? -ne 0 ]; then echo "TRAIN FAILED for $N"; return 1; fi

  echo "--- scanning checkpoints ---"
  local BEST_ACC=-1 BEST_STEP="" LATEST_ACC="" step CKPT OUT ACC IS_BETTER
  for step in 320 340 360 380 400 420 440 460 480 500; do
    CKPT="$OUT_DIR/checkpoint-$step"
    if [ ! -d "$CKPT" ]; then continue; fi
    OUT=$(python compute_train_accuracy.py \
      --model_name "$MODEL_DIR" \
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
  if [ -z "$BEST_STEP" ]; then echo "NO VALID CHECKPOINTS for $N"; return 1; fi

  echo "--- renaming checkpoints ---"
  local BEST_CKPT_DIR
  mv "$OUT_DIR/checkpoint-500" "$OUT_DIR/checkpoint-latest-500"
  if [ "$BEST_STEP" == "500" ]; then
    BEST_CKPT_DIR="$OUT_DIR/checkpoint-latest-500"
  else
    mv "$OUT_DIR/checkpoint-$BEST_STEP" "$OUT_DIR/checkpoint-best-$BEST_STEP"
    BEST_CKPT_DIR="$OUT_DIR/checkpoint-best-$BEST_STEP"
  fi
  find "$OUT_DIR" -maxdepth 1 -name 'checkpoint-[0-9]*' -exec rm -rf {} +

  echo "--- diagnostics ---"
  python inspect_similarity.py \
    --model_name "$MODEL_DIR" \
    --model_backbone qwen3_vl \
    --checkpoint_path "$BEST_CKPT_DIR" \
    --batch_size 32 --seed 42 --tag "${LABEL}_keepfirst" 2>&1 | tee /tmp/inspect_${LABEL}_keepfirst.log

  python dump_similarity_matrix.py \
    --model_name "$MODEL_DIR" \
    --model_backbone qwen3_vl \
    --checkpoint_path "$BEST_CKPT_DIR" \
    --batch_size 32 --seed 42 \
    --out /tmp/sim_matrix_${LABEL}_keepfirst.npy

  echo "--- cleanup pruned base ---"
  rm -rf "$MODEL_DIR"

  echo "$N $LABEL best_step=$BEST_STEP best_acc=$BEST_ACC latest_acc=$LATEST_ACC" >> "$RESULTS"
  echo "=== Finished N=$N ($LABEL) $(date) ==="
  df -h /workspace | tail -1
}

run_ratio 21 "25pct"
run_ratio 20 "28.6pct"
run_ratio 18 "35.7pct"
run_ratio 14 "50pct"
run_ratio 7 "75pct"

echo "ALL DONE $(date)"
