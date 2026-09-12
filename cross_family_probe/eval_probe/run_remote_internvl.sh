#!/bin/bash
set -euo pipefail
cd /workspace/cross_family_probe
export HF_HOME=/workspace/.hf_home
export PYTHONUNBUFFERED=1
mkdir -p outputs/internvl

run_extract() {
  python eval_probe/internvl3_candidate_probe.py \
    --model-id "$1" --depth "$2" --split "$3" --batch-size 4 --max-patches 1 \
    ${5:+--limit "$5"} --output "$4"
}

run_extract OpenGVLab/InternVL3-2B 28 train outputs/internvl/internvl3_2b_d28_train.npz 8000
run_extract OpenGVLab/InternVL3-2B 28 validation outputs/internvl/internvl3_2b_d28_validation.npz
python eval_probe/cross_family_candidate_probe.py fit \
  --train-cache outputs/internvl/internvl3_2b_d28_train.npz \
  --val-cache outputs/internvl/internvl3_2b_d28_validation.npz \
  --result outputs/internvl/internvl3_2b_d28_result.json

run_extract OpenGVLab/InternVL3-8B 7 train outputs/internvl/internvl3_8b_d7_train.npz 8000
run_extract OpenGVLab/InternVL3-8B 7 validation outputs/internvl/internvl3_8b_d7_validation.npz
python eval_probe/cross_family_candidate_probe.py fit \
  --train-cache outputs/internvl/internvl3_8b_d7_train.npz \
  --val-cache outputs/internvl/internvl3_8b_d7_validation.npz \
  --result outputs/internvl/internvl3_8b_d7_result.json

date -u +%FT%TZ > outputs/internvl/COMPLETE
