#!/bin/bash
# Extends the contiguous-vs-spread ("no free lunch") comparison beyond the
# single existing N=6 pair per model size. Contiguous (keep-first-N) results
# already exist at these N's (2B: logs/depth_pruning_keepfirst/sweep_2b_floor_results.txt
# N=4/34.0%, N=7 via original keepfirst sweep/69.0%; 8B:
# sweep_8b_boundary_results.txt N=4/33.0%, sweep_8b_results.txt N=9/70.5%) --
# this only runs the missing spread half, at the same N's, bracketing the
# existing N=6 spread pair from both sides.
#
# Spread keep_idx uses the same rule that reproduces the original hand-picked
# N=6 spread configs exactly (0,1,2,9,18,27 for 2B; 0,1,2,12,23,35 for 8B):
# keep 0,1,2 (deepstack prefix) + round(i * (n_layers-1) / tail_keep_count)
# for i=1..tail_keep_count -- i.e. evenly divide the *full* depth range
# 0..n_layers-1 into tail_keep_count segments, unlike prune_model.py's own
# default heuristic which spreads only the tail (3..n_layers-1) and always
# includes layer 3, making it overlap with contiguous at low N. Verified
# against the known N=6 values before use.
set -uo pipefail
cd /workspace/VLM2Vec
source /venv/main/bin/activate

RESULTS=logs/spread_pruning/sweep_spread_results.txt
mkdir -p logs/spread_pruning
touch "$RESULTS"
HF_HUB_DIR="${HF_HOME:-/workspace/.hf_home}/hub"

free_cache() {
  rm -rf "${HF_HUB_DIR}/models--${1//\//--}"
}

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
    python prune_model_generic.py --backbone "$BACKBONE" --keep_idx "$KEEP_IDX" --out_dir "$MODEL_DIR" --free_cache
  else
    python prune_model.py --keep_idx "$KEEP_IDX" --out_dir "$MODEL_DIR"
  fi
  if [ $? -ne 0 ]; then echo "PRUNE FAILED for ${SIZE} ${TAG}"; return 1; fi
  df -h /workspace | tail -1

  echo "--- training (standard LoRA, matching the existing spread6/keepfirst sweeps) ---"
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
    --batch_size 32 --seed 42 --tag "${SIZE}_${TAG}" 2>&1 | tee logs/spread_pruning/inspect_${SIZE}_${TAG}.log

  python dump_similarity_matrix.py \
    --model_name "$MODEL_DIR" \
    --model_backbone qwen3_vl \
    --checkpoint_path "$BEST_CKPT_DIR" \
    --batch_size 32 --seed 42 \
    --out logs/spread_pruning/sim_matrix_${SIZE}_${TAG}.npy

  echo "--- cleanup ---"
  rm -rf "$MODEL_DIR" "$OUT_DIR"

  echo "${SIZE} ${TAG} keep=${KEEP_IDX} N=${N} best_step=${BEST_STEP} best_acc=${BEST_ACC} latest_acc=${LATEST_ACC}" >> "$RESULTS"
  echo "=== Finished spread ${SIZE} ${TAG} $(date) ==="
  df -h /workspace | tail -1
}

# 2B configs first (model already cached, no download / cache-free needed)
run_spread_config 2b "0,1,2,27" spread4 "Qwen/Qwen3-VL-2B-Instruct"
run_spread_config 2b "0,1,2,7,14,20,27" spread7 "Qwen/Qwen3-VL-2B-Instruct"

# Free the 2B cache before downloading 8B (disk headroom)
free_cache "Qwen/Qwen3-VL-2B-Instruct"

# 8B configs
run_spread_config 8b "0,1,2,35" spread4 "Qwen/Qwen3-VL-8B-Instruct"
run_spread_config 8b "0,1,2,6,12,18,23,29,35" spread9 "Qwen/Qwen3-VL-8B-Instruct"

echo "ALL DONE $(date)"
