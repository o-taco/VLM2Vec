#!/bin/bash
# Single-GPU (V100 32GB, Volta) A-OKVQA training run, Qwen3-VL-2B-Instruct backbone.
# Mirrors examples/llava_next/run_train_aokvqa.sh's batch=64/grad_cache fix
# (see that script's header for why per_device_train_batch_size must be >>
# gc_q/p_chunk_size). Qwen3-VL is ~4x smaller than the 7B LLaVA-Next backbone,
# so there's more memory headroom -- BATCH_SIZE is env-overridable if you want
# to push it higher once this baseline is confirmed working.
#
# lora_target_modules is overridden from the repo default: Qwen3-VL's decoder
# uses separate q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj (no
# fused qkv_proj/gate_up_proj like Phi3V), and the shared default in
# arguments.py doesn't include gate_proj/up_proj at all.
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
  --bf16 \
  --gradient_checkpointing \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --dataset_name TIGER-Lab/MMEB-train \
  --subset_name A-OKVQA \
  --split_name original \
  --image_dir data \
  --image_resolution 336 \
  --output_dir output/aokvqa_qwen3vl \
  --max_steps "${MAX_STEPS:-100}" \
  --per_device_train_batch_size "${BATCH_SIZE:-64}" \
  --grad_cache True \
  --gc_q_chunk_size "${GC_CHUNK_SIZE:-8}" \
  --gc_p_chunk_size "${GC_CHUNK_SIZE:-8}" \
  --max_len 512 \
  --learning_rate 1e-5 \
  --logging_steps 10 \
  --save_steps "${SAVE_STEPS:-20}" \
  --remove_unused_columns False \
  --report_to none \
  "$@"
