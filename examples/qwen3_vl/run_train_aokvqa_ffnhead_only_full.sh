#!/bin/bash
# FFN-head-only on the FULL (unpruned, 28-layer) Qwen3-VL-2B backbone --
# counterpart to run_train_aokvqa.sh (the LoRA full-model run behind
# o-taco/qwen3vl-aokvqa / qwen3vl_train_loss.png) for a head-vs-LoRA
# comparison at full depth, extending the crossover sweep past N=7/9.
# Same freeze_backbone+add_ffn_head recipe as run_train_aokvqa_ffnhead_only.sh
# (LR=1e-4, matching the rest of that ablation), just pointed at the
# un-pruned base model instead of pruned_models/qwen3vl-*-Nlayer-keepfirst.
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
  --output_dir output/aokvqa_qwen3vl2b_full_ffnhead_only \
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
