#!/bin/bash
set -uo pipefail
cd /workspace/cross_family_probe
mkdir -p logs outputs/internvl
source /venv/main/bin/activate
uv pip install --python /venv/main/bin/python \
  torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /venv/main/bin/python \
  'transformers==4.57.6' datasets scikit-learn pillow sentencepiece einops timm packaging tqdm
chmod +x eval_probe/run_remote_internvl.sh
bash eval_probe/run_remote_internvl.sh
status=$?
printf '%s\n' "$status" > outputs/internvl/EXIT_STATUS
exit "$status"
