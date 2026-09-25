"""Fits a per-depth agreement-probability threshold on the held-out calibration split:
the lowest threshold that keeps empirical precision (fraction of routed examples whose
prediction truly agrees with depth 28) >= --target (default 0.90), maximizing coverage
subject to that precision floor. Matches the deck's "Held-out calibration fits per-depth
agreement thresholds; exit at earliest crossing."
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from early_exit_probe.data_aokvqa import AOKVQADataset, load_or_build_split_index
from early_exit_probe.model import DEPTH_TAPS, DepthProbe, load_backbone, load_processor
from early_exit_probe.inference import run_batch_all_depths, load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_INDEX_PATH = REPO_ROOT / "early_exit_probe" / "outputs" / "aokvqa_split_index.json"


def fit_threshold(probs: torch.Tensor, agrees: torch.Tensor, target: float):
    """Lowest threshold s.t. precision among prob>=threshold is >= target. probs/agrees
    are 1-D tensors over the calibration set for one depth."""
    order = torch.argsort(probs, descending=True)
    probs_sorted = probs[order]
    agrees_sorted = agrees[order].float()
    cum_agree = torch.cumsum(agrees_sorted, dim=0)
    counts = torch.arange(1, len(probs_sorted) + 1, dtype=torch.float32)
    precision = cum_agree / counts
    ok = precision >= target
    if not ok.any():
        return {"threshold": 1.01, "coverage": 0.0, "precision": None}
    # last index (largest prefix / lowest threshold) satisfying precision >= target
    last_ok = ok.nonzero().max().item()
    threshold = probs_sorted[last_ok].item()
    coverage = (last_ok + 1) / len(probs_sorted)
    return {"threshold": threshold, "coverage": coverage, "precision": precision[last_ok].item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--target", type=float, default=0.90)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = load_processor()
    backbone = load_backbone(lora=True).to(device)
    backbone.eval()
    probe = DepthProbe().to(device)
    load_checkpoint(backbone, probe, args.checkpoint)
    probe.eval()

    split_index = load_or_build_split_index(SPLIT_INDEX_PATH, cache_dir=args.cache_dir)
    calib_ds = AOKVQADataset("train", "calibration", split_index, cache_dir=args.cache_dir, limit=args.limit)
    print(f"[calibrate] calibration rows={len(calib_ds)}", flush=True)

    probs_by_depth = {d: [] for d in DEPTH_TAPS}
    agrees_by_depth = {d: [] for d in DEPTH_TAPS}
    for start in range(0, len(calib_ds), args.batch_size):
        examples = [calib_ds[i] for i in range(start, min(start + args.batch_size, len(calib_ds)))]
        per_depth, _ = run_batch_all_depths(backbone, probe, processor, examples, device)
        for d in DEPTH_TAPS:
            probs_by_depth[d].append(per_depth[d]["agreement_prob"].cpu())
            agrees_by_depth[d].append(per_depth[d]["agrees_with_28"].cpu())
        if (start // args.batch_size) % 10 == 0:
            print(f"[calibrate] {start + len(examples)}/{len(calib_ds)}", flush=True)

    thresholds = {}
    for d in DEPTH_TAPS:
        probs = torch.cat(probs_by_depth[d])
        agrees = torch.cat(agrees_by_depth[d])
        thresholds[d] = fit_threshold(probs, agrees, args.target)
        print(f"[calibrate] depth={d} -> {thresholds[d]}", flush=True)

    out_path = Path(args.output) if args.output else Path(args.checkpoint) / "thresholds.json"
    with open(out_path, "w") as f:
        json.dump({"target": args.target, "thresholds": thresholds}, f, indent=2)
    print(f"[calibrate] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
