#!/bin/bash
# Cross-size pruning comparison: Qwen3-VL-8B depth-pruned two ways, compared
# against the already-completed Qwen3-VL-2B (full and keep-first-N) results.
#   - 3layer:  keep only the deepstack-required prefix (0,1,2) -- 91.7% of
#              decoder layers dropped, ~2.26B total params, roughly matched to
#              full 2B (~2.07B). Tests "prune a bigger model down to ~2B size
#              vs just use the full 2B".
#   - 9layer:  keep-first-9 (75% dropped, same ratio as the 2B sweep's most
#              extreme point). ~3.4B total params (8B's untied embed+lm_head
#              is a ~1.24B fixed cost that pruning depth can't remove).
#
# Disk-safety: this box has a 32G non-persistent disk and Qwen3-VL-8B's raw
# weights are ~17.5G. prune_model_generic.py --free_cache deletes the base-8B
# HF cache (after cloning every tensor into independent RAM, decoupled from
# any mmap) BEFORE writing the pruned save, so the full download and the
# pruned copy never coexist on disk. We also free the (already fully-mined)
# 2B HF cache up front for extra margin.
set -uo pipefail
cd /workspace/VLM2Vec
source /venv/main/bin/activate

RESULTS=/tmp/sweep_8b_results.txt
touch "$RESULTS"

echo "--- freeing 2B HF cache for disk headroom (already fully mined, redownloadable) ---"
rm -rf "${HF_HOME:-/workspace/.hf_home}/hub/models--Qwen--Qwen3-VL-2B-Instruct"
df -h /workspace | tail -1

run_config() {
  local N=$1 LABEL=$2
  echo "=================================================="
  echo "=== Starting Qwen3-VL-8B, keep-first-$N ($LABEL) $(date) ==="
  echo "=================================================="

  local KEEP_IDX MODEL_DIR OUT_DIR
  KEEP_IDX=$(python -c "print(','.join(str(i) for i in range($N)))")
  MODEL_DIR="pruned_models/qwen3vl-8b-${N}layer-keepfirst"
  OUT_DIR="output/aokvqa_qwen3vl8b_${N}layer_keepfirst"

  echo "--- pruning (downloading Qwen3-VL-8B-Instruct fresh, ~17.5G, then freeing its cache before save) ---"
  python prune_model_generic.py \
    --backbone Qwen/Qwen3-VL-8B-Instruct \
    --keep_idx "$KEEP_IDX" \
    --out_dir "$MODEL_DIR" \
    --free_cache
  if [ $? -ne 0 ]; then echo "PRUNE FAILED for $N"; return 1; fi
  df -h /workspace | tail -1

  echo "--- training ---"
  bash examples/qwen3_vl/run_train_aokvqa_8b_${N}layer_keepfirst.sh
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
    --batch_size 32 --seed 42 --tag "8b_${LABEL}_keepfirst" 2>&1 | tee /tmp/inspect_8b_${LABEL}_keepfirst.log

  python dump_similarity_matrix.py \
    --model_name "$MODEL_DIR" \
    --model_backbone qwen3_vl \
    --checkpoint_path "$BEST_CKPT_DIR" \
    --batch_size 32 --seed 42 \
    --out /tmp/sim_matrix_8b_${LABEL}_keepfirst.npy

  echo "--- cleanup pruned base (frees disk before next config's 17.5G download) ---"
  rm -rf "$MODEL_DIR"

  echo "$N $LABEL best_step=$BEST_STEP best_acc=$BEST_ACC latest_acc=$LATEST_ACC" >> "$RESULTS"
  echo "=== Finished N=$N ($LABEL) $(date) ==="
  df -h /workspace | tail -1
}

run_config 3 "sizematch2B"
run_config 9 "75pctratio"

echo "ALL DONE $(date)"
