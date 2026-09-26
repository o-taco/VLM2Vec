"""Evaluates a trained early-exit probe on the MMEB-eval classification sets with native
(large) candidate lists -- the top half of the deck's page-7 results table.

Per dataset it reports: chance, frozen Qwen3-VL-2B zero-shot (adapters disabled; generates
the option number/letter), the probe's accuracy at every tapped depth, full-depth-28
accuracy ("Joint 2-head"), and calibrated early-exit routed accuracy / mean depth
("90% agreement"). Thresholds default to the checkpoint's own thresholds.json, so a
checkpoint calibrated on a different distribution (e.g. A-OKVQA only) will route
miscalibrated -- that is reported as-is, not corrected.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from early_exit_probe.data_mmeb_eval import ALL_DATASETS, MMEBClassificationDataset
from early_exit_probe.model import DEPTH_TAPS, DepthProbe, load_backbone, load_processor
from early_exit_probe.inference import load_checkpoint, run_batch_all_depths
from early_exit_probe.batching import collate_for_generation
from early_exit_probe.data_aokvqa import CHOICE_LETTERS
from early_exit_probe.eval import route


def default_batch_size(n_choices: int) -> int:
    return 4 if n_choices >= 100 else 8


@torch.no_grad()
def zero_shot_batch(backbone, processor, examples, device):
    n = len(examples[0].choices)
    if n <= len(CHOICE_LETTERS):
        suffix, pattern = " Respond with only the letter.", re.compile(r"[A-H]")
    else:
        suffix, pattern = " Respond with only the number of the correct option.", re.compile(r"\d+")
    batch, correct_idx = collate_for_generation(examples, processor, suffix=suffix)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with backbone.disable_adapter():
        out = backbone.generate(**batch, max_new_tokens=5, do_sample=False)
    in_len = batch["input_ids"].shape[1]
    correct = []
    for row, gold in zip(out, correct_idx.tolist()):
        text = processor.tokenizer.decode(row[in_len:], skip_special_tokens=True)
        m = pattern.search(text.strip())
        pred = None
        if m:
            pred = CHOICE_LETTERS.index(m.group()) if n <= len(CHOICE_LETTERS) else int(m.group()) - 1
        correct.append(pred == gold)
    return correct


def eval_dataset(name, backbone, probe, processor, thresholds, device, limit, batch_size, skip_zero_shot):
    ds = MMEBClassificationDataset(name, limit=limit)
    n, n_choices = len(ds), ds.n_choices
    bs = batch_size or default_batch_size(n_choices)
    # a batch must share one option count (ScienceQA is 2-5-way), so batch within count groups
    groups = {}
    for i, c in enumerate(ds.choice_counts):
        groups.setdefault(c, []).append(i)
    batches = [idxs[s:s + bs] for _, idxs in sorted(groups.items()) for s in range(0, len(idxs), bs)]
    per_depth_correct = {d: 0 for d in DEPTH_TAPS}
    routed_correct, routed_depths, fallbacks, zs_correct = 0, [], 0, 0
    for b, batch_idx in enumerate(batches):
        examples = [ds[i] for i in batch_idx]
        per_depth, correct_idx = run_batch_all_depths(backbone, probe, processor, examples, device)
        for d in DEPTH_TAPS:
            per_depth_correct[d] += per_depth[d]["correct"].sum().item()
        exited, final_pred = route(per_depth, thresholds, DEPTH_TAPS)
        routed_correct += (torch.tensor(final_pred, device=device) == correct_idx).sum().item()
        routed_depths.extend(exited)
        fallbacks += sum(1 for d in exited if d == max(DEPTH_TAPS))
        if not skip_zero_shot:
            zs_correct += sum(zero_shot_batch(backbone, processor, examples, device))
        if b % 25 == 0:
            print(f"[{name}] batch {b + 1}/{len(batches)}", flush=True)
    return {
        "n": n,
        "n_choices": n_choices,
        "min_choices": min(ds.choice_counts),
        "chance": sum(1.0 / c for c in ds.choice_counts) / n,
        "zero_shot_accuracy": None if skip_zero_shot else zs_correct / n,
        "per_depth_accuracy": {str(d): per_depth_correct[d] / n for d in DEPTH_TAPS},
        "full_depth28_accuracy": per_depth_correct[max(DEPTH_TAPS)] / n,
        "routed_accuracy": routed_correct / n,
        "routed_mean_depth": sum(routed_depths) / n,
        "routed_coverage_before_fallback": 1 - fallbacks / n,
        "approximate_candidate_set": name == "ImageNet-A",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--datasets", nargs="+", default=list(ALL_DATASETS))
    ap.add_argument("--limit", type=int, default=None, help="first N rows per dataset (deck uses 1,000)")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--skip_zero_shot", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = load_processor()
    processor.tokenizer.padding_side = "left"
    backbone = load_backbone(lora=True).to(device).eval()
    probe = DepthProbe().to(device)
    load_checkpoint(backbone, probe, args.checkpoint)
    probe.eval()

    thr_path = Path(args.thresholds) if args.thresholds else Path(args.checkpoint) / "thresholds.json"
    thresholds = json.load(open(thr_path))["thresholds"]

    out_path = Path(args.output) if args.output else Path(args.checkpoint) / "eval_mmeb_results.json"
    results = json.load(open(out_path)) if out_path.exists() else {}
    for name in args.datasets:
        results[name] = eval_dataset(
            name, backbone, probe, processor, thresholds, device, args.limit, args.batch_size, args.skip_zero_shot
        )
        print(json.dumps({name: results[name]}, indent=2), flush=True)
        json.dump(results, open(out_path, "w"), indent=2)  # incremental: a crash keeps finished sets
    print(f"[eval_mmeb] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
