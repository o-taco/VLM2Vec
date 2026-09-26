"""Joint training of the LoRA-adapted backbone + depth-conditioned FiLM probe
(answer head + agreement head) on the answer-fit / confidence-fit splits of the chosen pool (A-OKVQA, optionally + ScienceQA).

Usage:
  python -m early_exit_probe.train --smoke_test
  python -m early_exit_probe.train --max_steps 1000 --output_dir early_exit_probe/outputs/run1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from early_exit_probe.data_pool import TrainingPool
from early_exit_probe.model import DEPTH_TAPS, DepthProbe, load_backbone, load_processor
from early_exit_probe.batching import collate, extract_depth_concats

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_INDEX_PATH = REPO_ROOT / "early_exit_probe" / "outputs" / "aokvqa_split_index.json"


def run_step(backbone, probe, processor, device, answer_examples, confidence_examples, agreement_weight):
    examples = answer_examples + confidence_examples
    n_answer = len(answer_examples)
    batch, choice_pos, correct_idx = collate(examples, processor)
    correct_idx = correct_idx.to(device)

    depth_concats = extract_depth_concats(backbone, batch, choice_pos, device, depths=DEPTH_TAPS)

    answer_logits_by_depth = {}
    z_by_depth = {}
    for d in DEPTH_TAPS:
        logits, z = probe(depth_concats[d], d)
        answer_logits_by_depth[d] = logits
        z_by_depth[d] = z

    pred_28 = answer_logits_by_depth[max(DEPTH_TAPS)].argmax(dim=-1).detach()

    total_answer_loss = 0.0
    total_agreement_loss = 0.0
    for d in DEPTH_TAPS:
        logits = answer_logits_by_depth[d]
        z = z_by_depth[d]

        answer_logits = logits[:n_answer]
        answer_targets = correct_idx[:n_answer]
        total_answer_loss = total_answer_loss + F.cross_entropy(answer_logits, answer_targets)

        conf_logits = logits[n_answer:]
        conf_z = z[n_answer:]
        pred_d = conf_logits.argmax(dim=-1)
        target = (pred_d == pred_28[n_answer:]).float().detach()
        agreement_logit = probe.agreement_logits_for_predicted(conf_z, pred_d)
        total_agreement_loss = total_agreement_loss + F.binary_cross_entropy_with_logits(agreement_logit, target)

    total_answer_loss = total_answer_loss / len(DEPTH_TAPS)
    total_agreement_loss = total_agreement_loss / len(DEPTH_TAPS)
    loss = total_answer_loss + agreement_weight * total_agreement_loss
    return loss, total_answer_loss.item(), total_agreement_loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=8, help="split evenly between answer-fit and confidence-fit")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--agreement_weight", type=float, default=1.0)
    ap.add_argument("--pool", default="aokvqa", choices=["aokvqa", "aokvqa+scienceqa"],
                    help="training sources; each step draws from one (source, n_choices) group")
    ap.add_argument("--seed", type=int, default=26091622)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "early_exit_probe" / "outputs" / "run1"))
    ap.add_argument("--cache_dir", type=str, default=None)
    args = ap.parse_args()

    if args.smoke_test:
        args.max_steps = 10
        args.batch_size = 2
        args.log_every = 1
        args.output_dir = str(REPO_ROOT / "early_exit_probe" / "outputs" / "smoke_test")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] loading processor + backbone (device={device}, lora_r={args.lora_r})", flush=True)
    processor = load_processor()
    backbone = load_backbone(lora=True, lora_r=args.lora_r, lora_alpha=args.lora_alpha).to(device)
    backbone.train()
    probe = DepthProbe().to(device)
    probe.train()

    limit = 40 if args.smoke_test else None
    pool = TrainingPool(args.pool, SPLIT_INDEX_PATH, cache_dir=args.cache_dir, limit=limit)
    print(f"[train] pool={args.pool} rows per (source, split, n_choices):\n{pool.describe()}", flush=True)

    trainable = [p for p in backbone.parameters() if p.requires_grad] + list(probe.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[train] trainable params: {n_trainable:,}", flush=True)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    half = max(1, args.batch_size // 2)
    log_path = out_dir / "train_log.jsonl"
    started = time.perf_counter()
    with open(log_path, "w") as logf:
        for step in range(1, args.max_steps + 1):
            answer_examples, confidence_examples = pool.sample_step(rng, half)

            loss, answer_loss, agreement_loss = run_step(
                backbone, probe, processor, device, answer_examples, confidence_examples, args.agreement_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % args.log_every == 0 or step == 1:
                elapsed = time.perf_counter() - started
                rec = {
                    "step": step, "loss": loss.item(), "answer_loss": answer_loss,
                    "agreement_loss": agreement_loss, "elapsed_s": elapsed,
                }
                print(f"[train] step {step}/{args.max_steps} loss={loss.item():.4f} "
                      f"answer={answer_loss:.4f} agreement={agreement_loss:.4f} ({elapsed:.1f}s)", flush=True)
                logf.write(json.dumps(rec) + "\n")
                logf.flush()

            if step % args.save_every == 0 or step == args.max_steps:
                save_checkpoint(backbone, probe, out_dir, step)

    print(f"[train] done. checkpoints in {out_dir}", flush=True)


def save_checkpoint(backbone, probe, out_dir: Path, step: int):
    ckpt_dir = out_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    backbone.save_pretrained(str(ckpt_dir / "lora_adapter"))
    torch.save(probe.state_dict(), ckpt_dir / "probe.pt")
    print(f"[train] saved checkpoint at step {step} -> {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
