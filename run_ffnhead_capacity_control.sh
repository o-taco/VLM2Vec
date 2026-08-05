#!/bin/bash
# Controls for the FFN head's parameter count when comparing head-only vs LoRA
# across model sizes. The head's params scale with hidden_size^2, so 8B's
# default head (hidden_dim=4096, 33.5M params) has ~4x the params of 2B's
# (hidden_dim=2048, 8.39M params) -- confounding "8B needs less internal
# adaptation" with "8B's head just had more capacity to work with".
#
# The head's outer dim is pinned to the backbone's hidden_size (4096 for 8B,
# can't change), so --ffn_hidden_dim only controls the bottleneck width, not
# the total param count directly. To match 2B's exact head param count
# (2*dim*h + h + dim = 8,392,704) on 8B (dim=4096), solve for h: 8193h =
# 8,388,608 -> h=1024 gives 8,393,728 params, within 0.01% of 2B's head.
# (hidden_dim=2048 -- the naive "same width as 2B's own hidden_size" choice --
# actually gives 16,783,360 params, 2x too many; caught and fixed before
# burning the ~4h run on the wrong control.)
#
# Reruns 8B N=3 and N=9 (the ends of the tested range) with
# --ffn_hidden_dim 1024 to see whether head-only still matches/beats LoRA once
# head capacity is actually controlled for.
set -uo pipefail
cd /workspace/VLM2Vec
source /venv/main/bin/activate

RESULTS=/tmp/sweep_ffnhead_capacity_control_results.txt
touch "$RESULTS"

HF_HUB_DIR="${HF_HOME:-/workspace/.hf_home}/hub"
free_cache() { rm -rf "${HF_HUB_DIR}/models--${1//\//--}"; }
check_disk() { df -h /workspace | tail -1; }

run_config() {
  local N=$1
  echo "=================================================="
  echo "=== FFN-head (capacity-matched, hidden_dim=2048): 8b N=${N} $(date) ==="
  echo "=================================================="

  local KEEP_IDX MODEL_DIR OUT_DIR
  KEEP_IDX=$(python -c "print(','.join(str(i) for i in range($N)))")
  MODEL_DIR="pruned_models/qwen3vl-8b-${N}layer-keepfirst"
  OUT_DIR="output/aokvqa_qwen3vl8b_${N}layer_ffnhead_capctrl"

  echo "--- pruning ---"
  free_cache "Qwen/Qwen3-VL-2B-Instruct"
  python prune_model_generic.py --backbone Qwen/Qwen3-VL-8B-Instruct --keep_idx "$KEEP_IDX" --out_dir "$MODEL_DIR" --free_cache
  if [ $? -ne 0 ]; then echo "PRUNE FAILED for N=${N}"; return 1; fi
  check_disk

  echo "--- training (frozen backbone + FFN head, hidden_dim=1024, matches 2B's ~8.39M head params) ---"
  N=$N BACKBONE_SIZE=8b bash examples/qwen3_vl/run_train_aokvqa_ffnhead_only.sh --ffn_hidden_dim 1024
  if [ $? -ne 0 ]; then echo "TRAIN FAILED for N=${N}"; rm -rf "$MODEL_DIR" "$OUT_DIR"; return 1; fi

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
  if [ -z "$BEST_STEP" ]; then echo "NO VALID CHECKPOINTS for N=${N}"; rm -rf "$MODEL_DIR" "$OUT_DIR"; return 1; fi

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
  echo "=== Finished capacity-control N=${N} $(date) ==="
  check_disk
}

run_config 3
run_config 9

echo "ALL PHASES DONE $(date)"
