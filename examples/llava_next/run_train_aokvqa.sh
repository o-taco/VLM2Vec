#!/bin/bash
# Single-GPU (V100 32GB, Volta) A-OKVQA training run.
# Hyperparameters recovered from the prior working Colab run that produced
# o-taco/vlm2vec-llava-next-aokvqa-v2 (checkpoint-100..500 => max_steps=500,
# save_steps=100). See aokvqa-single-gpu branch commit history for the
# Volta/sdpa, use_dora, and truncation patches this run depends on.
#
# BATCH_SIZE/GC_CHUNK_SIZE default to 64/2: that prior run used
# per_device_train_batch_size=4 with gc_q/p_chunk_size=4, so grad_cache never
# actually chunked anything (batch == chunk size) -- only 3 in-batch
# negatives per step, too weak a contrastive signal to learn anything that
# generalized (confirmed empirically: accuracy stayed at ~25% chance for a
# 4-way task from step 0 through step 300, despite training loss dropping to
# ~0.06). Raising the batch size while keeping the chunk size small lets
# grad_cache do its actual job: many more in-batch negatives per step at
# roughly the same peak memory, since the frozen 7B backbone dominates
# memory either way, not the batch size.
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

WANDB_DISABLED=true \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train.py \
  --model_name llava-hf/llava-v1.6-vicuna-7b-hf \
  --model_backbone llava_next \
  --pooling last \
  --normalize True \
  --temperature 0.02 \
  --lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.0 \
  --bf16 \
  --gradient_checkpointing \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --dataset_name TIGER-Lab/MMEB-train \
  --subset_name A-OKVQA \
  --split_name original \
  --image_dir data \
  --image_resolution 336 \
  --output_dir output/aokvqa \
  --max_steps "${MAX_STEPS:-500}" \
  --per_device_train_batch_size "${BATCH_SIZE:-64}" \
  --grad_cache True \
  --gc_q_chunk_size "${GC_CHUNK_SIZE:-2}" \
  --gc_p_chunk_size "${GC_CHUNK_SIZE:-2}" \
  --max_len 512 \
  --learning_rate 1e-5 \
  --logging_steps 10 \
  --save_steps "${SAVE_STEPS:-100}" \
  --remove_unused_columns False \
  --report_to none \
  "$@"
