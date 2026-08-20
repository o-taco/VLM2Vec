#!/bin/bash
# Genuine linear-probe run on the FULL (unpruned, 36-layer) Qwen3-VL-8B
# backbone -- --add_linear_head attaches a single residual nn.Linear (no
# hidden layer, no GELU) after pooling, unlike --add_ffn_head's 2-layer
# nonlinear MLP (see run_train_aokvqa_8b_ffnhead_only_full.sh). This is the
# one that actually matches the "A-OKVQA-trained frozen linear probe" table
# row's name.
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

WANDB_DISABLED=true \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train.py \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --model_backbone qwen3_vl \
  --pooling last \
  --normalize True \
  --temperature 0.02 \
  --freeze_backbone \
  --add_linear_head \
  --bf16 \
  --gradient_checkpointing \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --dataset_name TIGER-Lab/MMEB-train \
  --subset_name A-OKVQA \
  --split_name original \
  --image_dir data \
  --image_resolution 336 \
  --output_dir output/aokvqa_qwen3vl8b_full_linear_only \
  --max_steps "${MAX_STEPS:-500}" \
  --per_device_train_batch_size "${BATCH_SIZE:-64}" \
  --grad_cache True \
  --gc_q_chunk_size "${GC_CHUNK_SIZE:-8}" \
  --gc_p_chunk_size "${GC_CHUNK_SIZE:-8}" \
  --max_len 512 \
  --learning_rate "${LR:-1e-4}" \
  --logging_steps 10 \
  --save_steps "${SAVE_STEPS:-20}" \
  --save_total_limit 10 \
  --remove_unused_columns False \
  --report_to none \
  "$@"
