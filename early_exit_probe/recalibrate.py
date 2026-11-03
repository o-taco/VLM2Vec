"""Refits the per-depth exit thresholds on a wider held-out calibration pool and re-scores
routing from the per-row outputs saved by diagnose_routing.py (no evaluation-set GPU pass).

Per-row calibration outputs are cached per source in <checkpoint>/calibration_rows.pt, so
adding a source or trying another pool only costs that source's forward passes. Writes
<checkpoint>/thresholds_<pool>.json and <checkpoint>/recalibration.json; the original
thresholds.json and evaluation result files are left untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from early_exit_probe.calibrate import SPLIT_INDEX_PATH, fit_threshold
from early_exit_probe.data_aokvqa import AOKVQADataset, load_or_build_split_index
from early_exit_probe.data_classification import SOURCES, ClassificationTrainDataset
from early_exit_probe.data_scienceqa import ScienceQATrainDataset
from early_exit_probe.diagnose_routing import collect_dataset, summarize
from early_exit_probe.inference import load_checkpoint
from early_exit_probe.model import DEPTH_TAPS, DepthProbe, load_backbone, load_processor

CLS = ("N24News", "HatefulMemes", "VOC2007", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211")
VQA = ("A-OKVQA", "Visual7W", "ScienceQA")


def calibration_dataset(source):
    if source == "A-OKVQA":
        ds = AOKVQADataset("train", "calibration", load_or_build_split_index(SPLIT_INDEX_PATH))
        return ds, [len(c) for c in ds.ds["choices"]]
    ds = ScienceQATrainDataset("calibration") if source == "ScienceQA" else ClassificationTrainDataset(source, "calibration")
    return ds, ds.choice_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pools", nargs="+", default=["A-OKVQA+ScienceQA", "A-OKVQA+ScienceQA+CUB-200"],
                    help=f"'+'-joined calibration sources from: A-OKVQA, ScienceQA, {', '.join(SOURCES)}")
    ap.add_argument("--target", type=float, default=0.90)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    cal_path = ckpt / "calibration_rows.pt"
    cal = torch.load(cal_path, weights_only=True) if cal_path.exists() else {}
    todo = [s for s in dict.fromkeys(s for p in args.pools for s in p.split("+")) if s not in cal]
    if todo:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = load_processor()
        processor.tokenizer.padding_side = "left"
        backbone = load_backbone(lora=True).to(device).eval()
        probe = DepthProbe().to(device)
        load_checkpoint(backbone, probe, args.checkpoint)
        probe.eval()
        for source in todo:
            ds, counts = calibration_dataset(source)
            cal[source] = collect_dataset(f"cal:{source}", ds, counts, backbone, probe, processor, device)
            torch.save(cal, cal_path)

    eval_rows = torch.load(ckpt / "routing_diagnosis_rows.pt", weights_only=True)
    report = {}
    for pool in args.pools:
        sources = pool.split("+")
        thresholds = {}
        for d in DEPTH_TAPS:
            probs = torch.cat([cal[s][d]["agreement_prob"] for s in sources]).float()
            agrees = torch.cat([cal[s][d]["agrees_with_28"] for s in sources])
            thresholds[str(d)] = fit_threshold(probs, agrees, args.target)
        n_rows = {s: len(cal[s][max(DEPTH_TAPS)]["pred"]) for s in sources}
        with open(ckpt / f"thresholds_{pool}.json", "w") as f:
            json.dump({"target": args.target, "pool": pool, "n_calibration_rows": n_rows,
                       "thresholds": thresholds}, f, indent=2)
        per_set = {name: summarize(rows, thresholds) for name, rows in eval_rows.items()}
        report[pool] = {"n_calibration_rows": n_rows, "thresholds": thresholds, "per_set": per_set}
        print(f"\n[recalibrate] pool={pool} rows={n_rows}")
        for d in DEPTH_TAPS[:-1]:
            t = thresholds[str(d)]
            print(f"  depth {d}: threshold {t['threshold']:.4f} coverage {t['coverage']:.3f}")
        for label, group in (("classification", CLS), ("vqa", VQA)):
            g = [per_set[n] for n in group if n in per_set]
            mean = lambda k: sum(s[k] for s in g) / len(g)
            print(f"  {label}: full {100 * mean('full_depth28_accuracy'):.2f}  routed "
                  f"{100 * mean('routed_accuracy'):.2f}  mean depth {mean('routed_mean_depth'):.2f}")
    json.dump(report, open(ckpt / "recalibration.json", "w"), indent=2)
    print(f"[recalibrate] wrote {ckpt / 'recalibration.json'}", flush=True)


if __name__ == "__main__":
    main()
