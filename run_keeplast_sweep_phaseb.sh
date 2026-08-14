#!/bin/bash
# Phase B of the front-drop ("keeplast") sweep: the ratio-matched points at
# the exact same nominal prune % as the existing keepfirst sweep (25 / 28.6 /
# 35.7 / 50 / 75pct, i.e. N=21/20/18/14/7), so the two curves can be compared
# point-by-point at matching x, not just by overall shape. Phase A (N=27/26/
# 25/23, 3.6-17.9pct) already showed a smooth, roughly linear decline toward
# the 25% chance floor by ~32.1% (keepmid); this extends that curve out to
# keepfirst's full range to see whether keeplast bottoms out at chance and
# stays there, or does something else at higher prune ratios.
set -uo pipefail
cd /workspace/VLM2Vec
source /venv/main/bin/activate

RESULTS=logs/depth_pruning_keeplast/keeplast_sweep_results.txt
mkdir -p logs/depth_pruning_keeplast
touch "$RESULTS"

run_ratio() {
  local N=$1 LABEL=$2
  echo "=================================================="
  echo "=== Starting N=$N layers ($LABEL) $(date) ==="
  echo "=================================================="

  local START KEEP_IDX MODEL_DIR OUT_DIR
  START=$((31 - N))
  KEEP_IDX=$(python -c "print(','.join(str(i) for i in [0,1,2]+list(range($START,28))))")
  MODEL_DIR="pruned_models/qwen3vl-2b-${N}layer-keeplast"
  OUT_DIR="output/aokvqa_qwen3vl_${N}layer_keeplast"

  echo "--- pruning (keep_idx=$KEEP_IDX) ---"
  python prune_model.py --keep_idx "$KEEP_IDX" --out_dir "$MODEL_DIR"
  if [ $? -ne 0 ]; then echo "PRUNE FAILED for $N"; return 1; fi

  echo "--- training ---"
  bash examples/qwen3_vl/run_train_aokvqa_${N}layer_keeplast.sh
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
    --batch_size 32 --seed 42 --tag "${LABEL}_keeplast" 2>&1 | tee logs/depth_pruning_keeplast/inspect_${LABEL}_keeplast.log

  python dump_similarity_matrix.py \
    --model_name "$MODEL_DIR" \
    --model_backbone qwen3_vl \
    --checkpoint_path "$BEST_CKPT_DIR" \
    --batch_size 32 --seed 42 \
    --out logs/depth_pruning_keeplast/sim_matrix_${LABEL}_keeplast.npy

  echo "--- cleanup pruned base ---"
  rm -rf "$MODEL_DIR"

  echo "$N $LABEL keeplast best_step=$BEST_STEP best_acc=$BEST_ACC latest_acc=$LATEST_ACC" >> "$RESULTS"
  echo "=== Finished N=$N ($LABEL) $(date) ==="
  df -h /workspace | tail -1
}

run_ratio 21 "25pct"
run_ratio 20 "28.6pct"
run_ratio 18 "35.7pct"
run_ratio 14 "50pct"
run_ratio 7 "75pct"

echo "ALL DONE $(date)"
