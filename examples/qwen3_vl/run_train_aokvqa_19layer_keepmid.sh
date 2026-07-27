#!/bin/bash
# Same capacity (19 layers, 32.1% pruned) as run_train_aokvqa_19layer.sh and
# run_train_aokvqa_19layer_keepfirst.sh, testing the "seam count" hypothesis:
# keep-first-19 (drop 19..27) has ZERO seams -- every kept layer's input still
# comes from its true original predecessor, it's just a shallower model. This
# config instead drops a single contiguous block (3..11) in the middle, so
# there is exactly ONE seam (layer 2's output now feeds layer 12's weights).
# Deepstack recipients 0,1,2 stay in place. If accuracy here lands between the
# 0-seam (77%) and ~9-seam alternating (47-50%) results, that's a dose-response
# curve for seam count as the causal variable, not raw pruning ratio.
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

WANDB_DISABLED=true \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train.py \
  --model_name pruned_models/qwen3vl-2b-19layer-keepmid \
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
  --output_dir output/aokvqa_qwen3vl_19layer_keepmid \
  --max_steps "${MAX_STEPS:-500}" \
  --per_device_train_batch_size "${BATCH_SIZE:-64}" \
  --grad_cache True \
  --gc_q_chunk_size "${GC_CHUNK_SIZE:-8}" \
  --gc_p_chunk_size "${GC_CHUNK_SIZE:-8}" \
  --max_len 512 \
  --learning_rate 1e-5 \
  --logging_steps 10 \
  --save_steps "${SAVE_STEPS:-20}" \
  --save_total_limit 10 \
  --remove_unused_columns False \
  --report_to none \
  "$@"
