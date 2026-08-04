#!/bin/bash
# Embedding-space-only ablation: freeze the entire (pruned) backbone and train
# only a small residual FFN head on the pooled last-token embedding. Compares
# against the LoRA runs (which adapt internal representations) to isolate how
# much of VLM2Vec's training gain comes from just reshaping the readout
# embedding space. Parameterized by $N (kept layers, matching the keepfirst
# pruned model naming) and $BACKBONE_SIZE (2b or 8b).
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root
: "${N:?set N to the number of kept layers}"
: "${BACKBONE_SIZE:?set BACKBONE_SIZE to 2b or 8b}"

WANDB_DISABLED=true \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train.py \
  --model_name "pruned_models/qwen3vl-${BACKBONE_SIZE}-${N}layer-keepfirst" \
  --model_backbone qwen3_vl \
  --pooling last \
  --normalize True \
  --temperature 0.02 \
  --freeze_backbone \
  --add_ffn_head \
  --bf16 \
  --gradient_checkpointing \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --dataset_name TIGER-Lab/MMEB-train \
  --subset_name A-OKVQA \
  --split_name original \
  --image_dir data \
  --image_resolution 336 \
  --output_dir "output/aokvqa_qwen3vl${BACKBONE_SIZE}_${N}layer_ffnhead_only" \
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
