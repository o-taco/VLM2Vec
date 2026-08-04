#!/bin/bash
# Overnight unattended sweep, run while the user sleeps (~18h budget). Three phases:
#
#   Phase 1: FFN-head-only ablation on Qwen3-VL-2B, N=4,5,6,7 (frozen backbone,
#            train only a small residual FFN readout head -- isolates how much
#            of the LoRA training gain is "reshaping the embedding space" vs.
#            "adapting internal representations").
#   Phase 2: same ablation on Qwen3-VL-8B, N=3,4,6,7,9 (skips 5,8 to fit the
#            time budget; still spans below-floor/at-floor/above-floor points
#            that already have LoRA numbers to compare against).
#   Phase 3: mechanistic test -- at 8B N=6 (known collapsed under contiguous
#            keep-first-6, 51.5% acc / 0.766 collapse), does a "spread" keep
#            pattern (same total layer count, spread across depth instead of
#            packed at the front) also collapse? Tests whether the floor is a
#            pure depth-budget effect or specific to which layers are kept.
#            Uses the normal LoRA training path (not the FFN head) so it's a
#            clean apples-to-apples comparison against the existing keepfirst
#            LoRA N=6 result.
#
# CRITICAL DISK-SAFETY FIX: earlier sweep scripts (run_8b_sweep.sh etc.) only
# deleted the pruned_models/ dir after each config, never the output/
# checkpoint dir -- those silently accumulated across many sessions and filled
# the 32GB disk to 100% earlier tonight, killing an in-progress 8B download.
# This script deletes BOTH after extracting diagnostics for every config.
set -uo pipefail
cd /workspace/VLM2Vec
source /venv/main/bin/activate

RESULTS_FFNHEAD=/tmp/sweep_ffnhead_results.txt
RESULTS_SPREAD=/tmp/sweep_spread_results.txt
touch "$RESULTS_FFNHEAD" "$RESULTS_SPREAD"

HF_HUB_DIR="${HF_HOME:-/workspace/.hf_home}/hub"

free_cache() {
  # $1 = HF repo id, e.g. Qwen/Qwen3-VL-2B-Instruct
  rm -rf "${HF_HUB_DIR}/models--${1//\//--}"
}

check_disk() {
  df -h /workspace | tail -1
}

# ---------------------------------------------------------------------------
# Phase 1 + 2: FFN-head-only ablation
# ---------------------------------------------------------------------------
run_ffnhead_config() {
  local SIZE=$1 N=$2 BACKBONE=$3
  echo "=================================================="
  echo "=== FFN-head-only: ${SIZE} N=${N} $(date) ==="
  echo "=================================================="

  local KEEP_IDX MODEL_DIR OUT_DIR
  KEEP_IDX=$(python -c "print(','.join(str(i) for i in range($N)))")
  MODEL_DIR="pruned_models/qwen3vl-${SIZE}-${N}layer-keepfirst"
  OUT_DIR="output/aokvqa_qwen3vl${SIZE}_${N}layer_ffnhead_only"

  echo "--- pruning ---"
  if [ "$SIZE" == "8b" ]; then
    free_cache "Qwen/Qwen3-VL-2B-Instruct"
    python prune_model_generic.py --backbone "$BACKBONE" --keep_idx "$KEEP_IDX" --out_dir "$MODEL_DIR" --free_cache
  else
    free_cache "Qwen/Qwen3-VL-8B-Instruct"
    python prune_model.py --keep_idx "$KEEP_IDX" --out_dir "$MODEL_DIR"
  fi
  if [ $? -ne 0 ]; then echo "PRUNE FAILED for ${SIZE} N=${N}"; return 1; fi
  check_disk

  echo "--- training (frozen backbone + FFN head only) ---"
  N=$N BACKBONE_SIZE=$SIZE bash examples/qwen3_vl/run_train_aokvqa_ffnhead_only.sh
  if [ $? -ne 0 ]; then echo "TRAIN FAILED for ${SIZE} N=${N}"; rm -rf "$MODEL_DIR" "$OUT_DIR"; return 1; fi

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
  if [ -z "$BEST_STEP" ]; then echo "NO VALID CHECKPOINTS for ${SIZE} N=${N}"; rm -rf "$MODEL_DIR" "$OUT_DIR"; return 1; fi

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
    --batch_size 32 --seed 42 --tag "${SIZE}_${N}layer_ffnhead_only" 2>&1 | tee /tmp/inspect_${SIZE}_${N}layer_ffnhead_only.log

  python dump_similarity_matrix.py \
    --model_name "$MODEL_DIR" \
    --model_backbone qwen3_vl \
    --checkpoint_path "$BEST_CKPT_DIR" \
    --no_lora \
    --batch_size 32 --seed 42 \
    --out /tmp/sim_matrix_${SIZE}_${N}layer_ffnhead_only.npy

  echo "--- cleanup (pruned base AND checkpoint dir) ---"
  rm -rf "$MODEL_DIR" "$OUT_DIR"

  echo "${SIZE} ${N} best_step=${BEST_STEP} best_acc=${BEST_ACC} latest_acc=${LATEST_ACC}" >> "$RESULTS_FFNHEAD"
  echo "=== Finished ffnhead ${SIZE} N=${N} $(date) ==="
  check_disk
}

# ---------------------------------------------------------------------------
# Phase 3: contiguous vs. spread pruning at matched layer count (LoRA)
# ---------------------------------------------------------------------------
run_spread_config() {
  local SIZE=$1 KEEP_IDX=$2 TAG=$3 BACKBONE=$4
  echo "=================================================="
  echo "=== Spread pruning: ${SIZE} ${TAG} (keep=${KEEP_IDX}) $(date) ==="
  echo "=================================================="

  local N MODEL_DIR OUT_DIR
  N=$(python -c "print(len('$KEEP_IDX'.split(',')))")
  MODEL_DIR="pruned_models/qwen3vl-${SIZE}-${TAG}"
  OUT_DIR="output/aokvqa_qwen3vl${SIZE}_${TAG}"

  echo "--- pruning ---"
  if [ "$SIZE" == "8b" ]; then
    free_cache "Qwen/Qwen3-VL-2B-Instruct"
    python prune_model_generic.py --backbone "$BACKBONE" --keep_idx "$KEEP_IDX" --out_dir "$MODEL_DIR" --free_cache
  else
    free_cache "Qwen/Qwen3-VL-8B-Instruct"
    python prune_model.py --keep_idx "$KEEP_IDX" --out_dir "$MODEL_DIR"
  fi
  if [ $? -ne 0 ]; then echo "PRUNE FAILED for ${SIZE} ${TAG}"; return 1; fi
  check_disk

  echo "--- training (standard LoRA, matching the existing keepfirst sweeps) ---"
  WANDB_DISABLED=true \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train.py \
    --model_name "$MODEL_DIR" \
    --model_backbone qwen3_vl \
    --pooling last \
    --normalize True \
    --temperature 0.02 \
    --lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --bf16 \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --dataset_name TIGER-Lab/MMEB-train \
    --subset_name A-OKVQA \
    --split_name original \
    --image_dir data \
    --image_resolution 336 \
    --output_dir "$OUT_DIR" \
    --max_steps 500 \
    --per_device_train_batch_size 64 \
    --grad_cache True \
    --gc_q_chunk_size 8 \
    --gc_p_chunk_size 8 \
    --max_len 512 \
    --learning_rate 1e-5 \
    --logging_steps 10 \
    --save_steps 20 \
    --save_total_limit 10 \
    --remove_unused_columns False \
    --report_to none
  if [ $? -ne 0 ]; then echo "TRAIN FAILED for ${SIZE} ${TAG}"; rm -rf "$MODEL_DIR" "$OUT_DIR"; return 1; fi

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
  if [ -z "$BEST_STEP" ]; then echo "NO VALID CHECKPOINTS for ${SIZE} ${TAG}"; rm -rf "$MODEL_DIR" "$OUT_DIR"; return 1; fi

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
    --batch_size 32 --seed 42 --tag "${SIZE}_${TAG}" 2>&1 | tee /tmp/inspect_${SIZE}_${TAG}.log

  python dump_similarity_matrix.py \
    --model_name "$MODEL_DIR" \
    --model_backbone qwen3_vl \
    --checkpoint_path "$BEST_CKPT_DIR" \
    --batch_size 32 --seed 42 \
    --out /tmp/sim_matrix_${SIZE}_${TAG}.npy

  echo "--- cleanup ---"
  rm -rf "$MODEL_DIR" "$OUT_DIR"

  echo "${SIZE} ${TAG} keep=${KEEP_IDX} N=${N} best_step=${BEST_STEP} best_acc=${BEST_ACC} latest_acc=${LATEST_ACC}" >> "$RESULTS_SPREAD"
  echo "=== Finished spread ${SIZE} ${TAG} $(date) ==="
  check_disk
}

echo "##################################################"
echo "# PHASE 1: FFN-head-only ablation, Qwen3-VL-2B"
echo "##################################################"
run_ffnhead_config 2b 4 Qwen/Qwen3-VL-2B-Instruct
run_ffnhead_config 2b 5 Qwen/Qwen3-VL-2B-Instruct
run_ffnhead_config 2b 6 Qwen/Qwen3-VL-2B-Instruct
run_ffnhead_config 2b 7 Qwen/Qwen3-VL-2B-Instruct

echo "##################################################"
echo "# PHASE 2: FFN-head-only ablation, Qwen3-VL-8B"
echo "##################################################"
run_ffnhead_config 8b 3 Qwen/Qwen3-VL-8B-Instruct
run_ffnhead_config 8b 4 Qwen/Qwen3-VL-8B-Instruct
run_ffnhead_config 8b 6 Qwen/Qwen3-VL-8B-Instruct
run_ffnhead_config 8b 7 Qwen/Qwen3-VL-8B-Instruct
run_ffnhead_config 8b 9 Qwen/Qwen3-VL-8B-Instruct

echo "##################################################"
echo "# PHASE 3: contiguous vs. spread pruning (LoRA), N=6 total layers"
echo "##################################################"
# 8B: 36 total layers. Deepstack requires 0,1,2 kept (by list position).
# spread6 keeps 0,1,2 (mandatory) + 3 more spread through the rest of the depth.
run_spread_config 8b "0,1,2,12,23,35" spread6_keepfirst "Qwen/Qwen3-VL-8B-Instruct"
# 2B: 28 total layers, same idea.
run_spread_config 2b "0,1,2,9,18,27" spread6_keepfirst "Qwen/Qwen3-VL-2B-Instruct"

echo "ALL PHASES DONE $(date)"
