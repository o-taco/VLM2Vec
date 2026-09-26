#!/bin/bash
# Phase 3: A-OKVQA + ScienceQA pool -> train -> calibrate (pooled held-out) -> evals.
# Zero-shot is skipped in the evals: it doesn't depend on the checkpoint (see phase1_run1 results).
set -e
source /venv/main/bin/activate
cd /workspace/VLM2Vec
OUT=early_exit_probe/outputs/phase3_pool_run1
CK=$OUT/checkpoint-1000
python -m early_exit_probe.train --pool aokvqa+scienceqa --max_steps 1000 --output_dir $OUT
python -m early_exit_probe.calibrate --checkpoint $CK --pool aokvqa+scienceqa
python -m early_exit_probe.eval --checkpoint $CK --skip_zero_shot
python -m early_exit_probe.eval_mmeb --checkpoint $CK --skip_zero_shot --limit 1000
echo "[phase3] ALL DONE"
