#!/bin/bash
# Same recipe as run_train_aokvqa.sh, but on the depth-pruned 19-layer backbone
# (see prune_model.py --n_keep 19) instead of the full 28-layer Qwen3-VL-2B-Instruct.
# Capacity-ablation point: ~32.1% of decoder layers removed (~params see prune log;
# deepstack recipient layers 0,1,2 kept in place). This is the last integer-layer
# point between the 20-layer (28.6% pruned, P@1~7-7.5%) and 18-layer (35.7% pruned,
# P@1~0.3-0.6%) points -- pins down the P@1 cliff to single-layer resolution.
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

WANDB_DISABLED=true \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train.py \
  --model_name pruned_models/qwen3vl-2b-19layer \
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
  --output_dir output/aokvqa_qwen3vl_19layer \
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
