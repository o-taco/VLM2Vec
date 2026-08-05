#!/bin/bash
# Recovers eval results for capacity-control checkpoints that already trained
# successfully but whose scan/rename/diagnostics step failed because the
# orchestrator's OUT_DIR variable didn't match the training script's actual
# (hardcoded) output path. Checkpoints in output/aokvqa_qwen3vl8b_${N}layer_
# ffnhead_only survived; only the pruned base model was deleted (needed to
# reconstruct the architecture for eval). Re-prunes just that base, then runs
# the normal scan -> rename -> diagnostics -> results -> cleanup tail.
set -uo pipefail
cd /workspace/VLM2Vec
source /venv/main/bin/activate

RESULTS=/tmp/sweep_ffnhead_capacity_control_results.txt
touch "$RESULTS"

HF_HUB_DIR="${HF_HOME:-/workspace/.hf_home}/hub"
free_cache() { rm -rf "${HF_HUB_DIR}/models--${1//\//--}"; }
check_disk() { df -h /workspace | tail -1; }

recover_config() {
  local N=$1
  local MODEL_DIR="pruned_models/qwen3vl-8b-${N}layer-keepfirst"
  local OUT_DIR="output/aokvqa_qwen3vl8b_${N}layer_ffnhead_only"

  echo "=================================================="
  echo "=== Recovering N=${N} $(date) ==="
  echo "=================================================="

  if [ ! -d "$OUT_DIR" ] || [ -z "$(find "$OUT_DIR" -maxdepth 1 -name 'checkpoint-[0-9]*' 2>/dev/null)" ]; then
    echo "no surviving numbered checkpoints for N=${N} in $OUT_DIR, skipping"
    return 1
  fi

  echo "--- re-pruning base model (needed to evaluate the surviving checkpoints) ---"
  local KEEP_IDX
  KEEP_IDX=$(python -c "print(','.join(str(i) for i in range($N)))")
  free_cache "Qwen/Qwen3-VL-2B-Instruct"
  python prune_model_generic.py --backbone Qwen/Qwen3-VL-8B-Instruct --keep_idx "$KEEP_IDX" --out_dir "$MODEL_DIR" --free_cache
  if [ $? -ne 0 ]; then echo "RE-PRUNE FAILED for N=${N}"; return 1; fi
  check_disk

  echo "--- scanning checkpoints ---"
  local BEST_ACC=-1 BEST_STEP="" LATEST_ACC="" step CKPT OUT ACC IS_BETTER
  for step in 320 340 360 380 400 420 440 460 480 500; do
    CKPT="$OUT_DIR/checkpoint-$step"
    if [ ! -d "$CKPT" ]; then continue; fi
    OUT=$(python compute_train_accuracy.py \
      --model_name "$MODEL_DIR" \
      --model_backbone qwen3_vl \
      --checkpoint_path "$CKPT" \
      --no_lora \
      --batch_size 4 --num_batches 50 --seed 42 2>&1)
    ACC=$(echo "$OUT" | grep -oP 'top-1 accuracy.*?: \K[0-9.]+')
    echo "checkpoint-$step acc=$ACC"
    if [ -z "$ACC" ]; then continue; fi
    if [ "$step" == "500" ]; then LATEST_ACC=$ACC; fi
    IS_BETTER=$(python -c "print(1 if $ACC > $BEST_ACC else 0)")
    if [ "$IS_BETTER" == "1" ]; then BEST_ACC=$ACC; BEST_STEP=$step; fi
  done
  echo "best: checkpoint-$BEST_STEP acc=$BEST_ACC ; latest: checkpoint-500 acc=$LATEST_ACC"
  if [ -z "$BEST_STEP" ]; then echo "NO VALID CHECKPOINTS for N=${N} (real failure this time)"; rm -rf "$MODEL_DIR" "$OUT_DIR"; return 1; fi

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
    --no_lora \
    --batch_size 32 --seed 42 --tag "8b_${N}layer_ffnhead_capctrl" 2>&1 | tee /tmp/inspect_8b_${N}layer_ffnhead_capctrl.log

  python dump_similarity_matrix.py \
    --model_name "$MODEL_DIR" \
    --model_backbone qwen3_vl \
    --checkpoint_path "$BEST_CKPT_DIR" \
    --no_lora \
    --batch_size 32 --seed 42 \
    --out /tmp/sim_matrix_8b_${N}layer_ffnhead_capctrl.npy

  echo "--- cleanup ---"
  rm -rf "$MODEL_DIR" "$OUT_DIR"

  echo "8b ${N} best_step=${BEST_STEP} best_acc=${BEST_ACC} latest_acc=${LATEST_ACC}" >> "$RESULTS"
  echo "=== Recovered N=${N} $(date) ==="
  check_disk
}

recover_config "$1"
