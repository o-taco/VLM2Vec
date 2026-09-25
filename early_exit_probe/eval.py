"""Evaluates the trained probe on the held-out A-OKVQA validation split (never used in
training or calibration): full-depth-28 accuracy ("Joint 2-head" row), calibrated
early-exit routed accuracy/coverage/mean-depth ("N% agreement" row, with full-depth
fallback when no threshold crossing fires), and a frozen Qwen3-VL-2B zero-shot baseline
(LoRA adapters disabled, letter-logit argmax) as a sanity cross-check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from early_exit_probe.data_aokvqa import AOKVQADataset, CHOICE_LETTERS
from early_exit_probe.model import DEPTH_TAPS, DepthProbe, load_backbone, load_processor
from early_exit_probe.inference import run_batch_all_depths, load_checkpoint
from early_exit_probe.batching import collate_for_generation

REPO_ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def zero_shot_batch(backbone, processor, examples, device):
    # collate()'s prompt is built for the *probe* (ends mid-template, no assistant turn
    # opened -- it never generates, just reads hidden states at fixed positions). Using it
    # for a real next-token logit readout scored at chance (24.5%) because the raw
    # next-token distribution there is off-distribution, dominated by chat-template
    # continuation tokens rather than an actual answer. collate_for_generation() opens a
    # real assistant turn and adds an explicit instruction, which recovers a proper
    # zero-shot signal (verified empirically: ~85% on a 20-example spot check, matching
    # the deck's reported ballpark of 78.40 for this baseline).
    batch, correct_idx = collate_for_generation(examples, processor, suffix=" Respond with only the letter.")
    correct_idx = correct_idx.to(device)
    batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with backbone.disable_adapter():
        out = backbone(**batch_dev, return_dict=True, use_cache=False)
    last_logits = out.logits[:, -1, :]  # left-padded -> last column is the real last position

    tokenizer = processor.tokenizer
    letter_ids = []
    for letter in CHOICE_LETTERS[:4]:
        ids = tokenizer(f" {letter}", add_special_tokens=False)["input_ids"]
        letter_ids.append(ids[0])
    letter_ids = torch.tensor(letter_ids, device=device)
    choice_logits = last_logits[:, letter_ids]  # (batch, 4)
    pred = choice_logits.argmax(dim=-1)
    return (pred == correct_idx)


def route(per_depth, thresholds, depths):
    """per_depth[d] tensors are (batch,). Returns exited_depth (list[int]), pred (Tensor),
    used_fallback (list[bool])."""
    batch_size = per_depth[depths[0]]["pred"].shape[0]
    exited_depth = [None] * batch_size
    final_pred = [None] * batch_size
    for d in depths:
        thr = thresholds.get(str(d), thresholds.get(d))["threshold"]
        prob = per_depth[d]["agreement_prob"]
        pred = per_depth[d]["pred"]
        for i in range(batch_size):
            if exited_depth[i] is None and prob[i].item() >= thr:
                exited_depth[i] = d
                final_pred[i] = pred[i].item()
    max_depth = max(depths)
    for i in range(batch_size):
        if exited_depth[i] is None:
            exited_depth[i] = max_depth
            final_pred[i] = per_depth[max_depth]["pred"][i].item()
    return exited_depth, final_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--thresholds", type=str, default=None, help="defaults to <checkpoint>/thresholds.json")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--skip_zero_shot", action="store_true")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = load_processor()
    backbone = load_backbone(lora=True).to(device)
    backbone.eval()
    probe = DepthProbe().to(device)
    load_checkpoint(backbone, probe, args.checkpoint)
    probe.eval()

    thresholds_path = Path(args.thresholds) if args.thresholds else Path(args.checkpoint) / "thresholds.json"
    with open(thresholds_path) as f:
        thresholds = json.load(f)["thresholds"]

    val_ds = AOKVQADataset("validation", split_filter=None, cache_dir=args.cache_dir, limit=args.limit)
    print(f"[eval] validation rows={len(val_ds)}", flush=True)

    n = len(val_ds)
    full28_correct = 0
    routed_correct = 0
    routed_depths = []
    fallback_count = 0
    zero_shot_correct = 0

    for start in range(0, n, args.batch_size):
        examples = [val_ds[i] for i in range(start, min(start + args.batch_size, n))]
        per_depth, correct_idx = run_batch_all_depths(backbone, probe, processor, examples, device)

        full28_correct += per_depth[max(DEPTH_TAPS)]["correct"].sum().item()

        exited_depth, final_pred = route(per_depth, thresholds, DEPTH_TAPS)
        final_pred_t = torch.tensor(final_pred, device=device)
        routed_correct += (final_pred_t == correct_idx).sum().item()
        routed_depths.extend(exited_depth)
        fallback_count += sum(1 for d in exited_depth if d == max(DEPTH_TAPS))

        if not args.skip_zero_shot:
            zs_correct = zero_shot_batch(backbone, processor, examples, device)
            zero_shot_correct += zs_correct.sum().item()

        if (start // args.batch_size) % 10 == 0:
            print(f"[eval] {start + len(examples)}/{n}", flush=True)

    results = {
        "n": n,
        "full_depth28_accuracy": full28_correct / n,
        "routed_accuracy": routed_correct / n,
        "routed_mean_depth": sum(routed_depths) / n,
        "routed_coverage_before_fallback": 1 - fallback_count / n,
        "zero_shot_accuracy": (zero_shot_correct / n) if not args.skip_zero_shot else None,
        "thresholds_used": thresholds,
    }
    print(json.dumps(results, indent=2), flush=True)

    out_path = Path(args.output) if args.output else Path(args.checkpoint) / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
