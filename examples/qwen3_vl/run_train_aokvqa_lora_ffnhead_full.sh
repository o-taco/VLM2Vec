#!/bin/bash
# LoRA + FFN-head combined, FULL (unpruned, 28-layer) Qwen3-VL-2B backbone --
# tests whether stacking a trainable residual FFN head on top of an
# already-LoRA-adapted backbone adds anything beyond LoRA alone. Same
# lora_r/alpha/target_modules and learning_rate as run_train_aokvqa.sh (the
# plain-LoRA full-model run behind the purple curve in
# ffnhead_vs_lora_full_model.png) so only the +add_ffn_head variable changes;
# same max_steps/save_steps as run_train_aokvqa_ffnhead_only_full.sh so all
# three full-depth runs are directly comparable at step 500.
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

WANDB_DISABLED=true \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train.py \
  --model_name Qwen/Qwen3-VL-2B-Instruct \
  --model_backbone qwen3_vl \
  --pooling last \
  --normalize True \
  --temperature 0.02 \
  --lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.0 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --add_ffn_head \
  --bf16 \
  --gradient_checkpointing \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --dataset_name TIGER-Lab/MMEB-train \
  --subset_name A-OKVQA \
  --split_name original \
  --image_dir data \
  --image_resolution 336 \
  --output_dir output/aokvqa_qwen3vl2b_full_lora_ffnhead \
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
