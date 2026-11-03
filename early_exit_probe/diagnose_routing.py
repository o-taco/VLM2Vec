"""Diagnoses why calibrated early exit loses accuracy off-distribution: for every evaluation
set, checks whether the agreement head's promise actually holds there. The thresholds are
fitted so that >= 90% of rows clearing a depth's threshold agree with depth 28 on the
calibration split; this measures that same precision on each evaluation set, plus what the
rows that really exit at each depth cost in accuracy.

Per-row outputs are saved (rows.pt) so thresholds can be re-fitted or swept later without
another GPU pass. Nothing here writes to eval_results.json / eval_mmeb_results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from early_exit_probe.data_aokvqa import AOKVQADataset
from early_exit_probe.data_mmeb_eval import ALL_DATASETS, MMEBClassificationDataset
from early_exit_probe.eval_mmeb import default_batch_size
from early_exit_probe.inference import load_checkpoint, probe_for_checkpoint, run_batch_all_depths
from early_exit_probe.model import DEPTH_TAPS, load_backbone, load_processor

AOKVQA = "A-OKVQA"
FIELDS = ("agreement_prob", "answer_maxprob", "answer_margin", "pred", "agrees_with_28", "correct")


def collect(name, backbone, probe, processor, device, limit, batch_size, depths=DEPTH_TAPS):
    """Returns {depth: {field: 1-D cpu tensor over rows}} for one evaluation set."""
    if name == AOKVQA:
        ds = AOKVQADataset("validation", split_filter=None, limit=limit)  # first `limit` val rows
        counts = [4] * len(ds)
    else:
        ds = MMEBClassificationDataset(name, limit=limit)
        counts = ds.choice_counts
    return collect_dataset(name, ds, counts, backbone, probe, processor, device, batch_size, depths)


def collect_dataset(name, ds, counts, backbone, probe, processor, device, batch_size=None, depths=DEPTH_TAPS):
    bs = batch_size or default_batch_size(max(counts))
    groups = {}
    for i, c in enumerate(counts):
        groups.setdefault(c, []).append(i)
    batches = [idxs[s:s + bs] for _, idxs in sorted(groups.items()) for s in range(0, len(idxs), bs)]
    out = {d: {f: [] for f in FIELDS} for d in depths}
    for b, batch_idx in enumerate(batches):
        per_depth, _ = run_batch_all_depths(backbone, probe, processor, [ds[i] for i in batch_idx], device, depths)
        for d in depths:
            for f in FIELDS:
                out[d][f].append(per_depth[d][f].cpu())
        if b % 25 == 0:
            print(f"[{name}] batch {b + 1}/{len(batches)}", flush=True)
    # Batches are grouped by choice count; scatter rows back to dataset order so a row's
    # position is its dataset index (the SHA-256 calibration split keys on it).
    order = torch.tensor([i for batch_idx in batches for i in batch_idx])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(len(order))
    return {d: {f: torch.cat(v)[inverse] for f, v in fs.items()} for d, fs in out.items()}


def auroc(scores: torch.Tensor, labels: torch.Tensor):
    """P(score of an agreeing row > score of a disagreeing row); None if one class is empty."""
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = torch.argsort(torch.argsort(scores)).float() + 1
    return ((ranks[labels].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))).item()


def summarize(rows, thresholds):
    depths = sorted(rows)
    top = max(depths)
    n = len(rows[top]["pred"])
    correct28 = rows[top]["correct"]
    remaining = torch.ones(n, dtype=torch.bool)
    summary = {"n": n, "full_depth28_accuracy": correct28.float().mean().item(), "per_depth": {}}
    routed_correct, routed_depth = torch.zeros(n, dtype=torch.bool), torch.full((n,), top)
    for d in depths:
        r = rows[d]
        if str(d) not in thresholds:
            continue
        thr = thresholds[str(d)]["threshold"]
        clears = r["agreement_prob"] >= thr
        exits = clears & remaining
        remaining &= ~clears
        routed_correct[exits], routed_depth[exits] = r["correct"][exits], d
        k = int(exits.sum())
        summary["per_depth"][str(d)] = {
            "accuracy": r["correct"].float().mean().item(),
            "agreement_rate": r["agrees_with_28"].float().mean().item(),  # with no gating at all
            "agreement_auroc": auroc(r["agreement_prob"], r["agrees_with_28"]),
            "threshold": thr,
            "calibration_coverage": thresholds[str(d)]["coverage"],
            "coverage": clears.float().mean().item(),
            "precision": r["agrees_with_28"][clears].float().mean().item() if clears.any() else None,
            "exit_share": k / n,
            "exit_precision": r["agrees_with_28"][exits].float().mean().item() if k else None,
            "exit_accuracy": r["correct"][exits].float().mean().item() if k else None,
            "exit_accuracy_at_28": correct28[exits].float().mean().item() if k else None,
            # points of whole-set accuracy lost by exiting these rows here instead of at 28
            "accuracy_cost_points": 100 * (correct28[exits].sum() - r["correct"][exits].sum()).item() / n,
        }
    routed_correct[remaining] = correct28[remaining]
    summary["routed_accuracy"] = routed_correct.float().mean().item()
    summary["routed_mean_depth"] = routed_depth.float().mean().item()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--datasets", nargs="+", default=list(ALL_DATASETS) + [AOKVQA])
    ap.add_argument("--limit", type=int, default=1000, help="first N rows per MMEB set (matches eval_mmeb)")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--rows_file", default="routing_diagnosis_rows.pt",
                    help="per-row cache inside the checkpoint dir; a new name forces a fresh GPU pass")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    thr_path = Path(args.thresholds) if args.thresholds else ckpt / "thresholds.json"
    # Every-layer checkpoints are calibrated offline (routing_baselines.py) from these rows.
    thresholds = json.load(open(thr_path))["thresholds"] if thr_path.exists() else None
    rows_path, out_path = ckpt / args.rows_file, ckpt / Path(args.rows_file).with_suffix(".json").name.replace("_rows", "")
    all_rows = torch.load(rows_path, weights_only=True) if rows_path.exists() else {}

    todo = [n for n in args.datasets if n not in all_rows]
    if todo:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = load_processor()
        processor.tokenizer.padding_side = "left"
        backbone = load_backbone(lora=True).to(device).eval()
        probe, depths = probe_for_checkpoint(args.checkpoint)
        probe = probe.to(device)
        load_checkpoint(backbone, probe, args.checkpoint)
        probe.eval()
        for name in todo:
            all_rows[name] = collect(name, backbone, probe, processor, device, args.limit, args.batch_size, depths)
            torch.save(all_rows, rows_path)  # incremental: a crash keeps finished sets

    if thresholds is None:
        print(f"[diagnose] rows saved to {rows_path}; no thresholds.json, skipping summary", flush=True)
        return
    results = {name: summarize(all_rows[name], thresholds) for name in args.datasets}
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"[diagnose] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
