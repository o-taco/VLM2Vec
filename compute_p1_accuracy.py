"""
Compute real Precision@1 on MMEB-eval's A-OKVQA test set — the paper's actual
metric (ground truth vs. a large per-query candidate pool, top-1 hit rate),
as opposed to the in-batch 4-way accuracy proxy used during training/smoke-testing.

Mirrors compute_val_loss.py's structure, swapping cross-entropy loss for argmax
top-1 accuracy. --limit and --num_candidates let this scale from a cheap pilot
(few queries, capped candidate pool) up to the full ~1000-query / ~900-candidate
paper-faithful run.
"""
import argparse
import os
import random

import torch
from PIL import Image
from datasets import load_dataset

from src.arguments import ModelArguments
from src.collator import process_vlm_inputs
from src.dataset import process_image
from src.model import MMEBModel
from src.model_utils import PHI3V, load_processor, vlm_image_tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--model_backbone", required=True)
    p.add_argument("--checkpoint_path", default=None)
    p.add_argument("--no_lora", action="store_true", help="evaluate the raw pretrained backbone (step-0 baseline) instead of a LoRA checkpoint")
    p.add_argument("--dataset_name", default="TIGER-Lab/MMEB-eval")
    p.add_argument("--subset_name", default="A-OKVQA")
    p.add_argument("--dataset_split", default="test")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--image_resolution", default="336")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--num_candidates", type=int, default=500, help="candidate pool size per query (paper uses ~900-1000); cap this down for a cheap pilot")
    p.add_argument("--limit", type=int, default=30, help="only evaluate the first N queries (for a cheap pilot; full set is 1000)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    model_args = ModelArguments(
        model_name=args.model_name,
        model_backbone=args.model_backbone,
        checkpoint_path=args.checkpoint_path,
        pooling="last",
        normalize=True,
        lora=not args.no_lora,
    )
    model = MMEBModel.load(model_args, is_trainable=False)
    model = model.to("cuda", dtype=torch.bfloat16)
    model.eval()

    processor = load_processor(model_args)

    ds = load_dataset(args.dataset_name, args.subset_name, split=args.dataset_split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"Loaded {len(ds)} eval rows")

    n_correct = 0
    pool_sizes = []
    with torch.no_grad():
        for i, row in enumerate(ds):
            qry_text = row["qry_text"].replace(vlm_image_tokens[PHI3V], vlm_image_tokens[args.model_backbone])
            qry_image = Image.open(os.path.join(args.image_dir, row["qry_img_path"])).convert("RGB")
            qry_image = process_image(qry_image, args.image_resolution)

            tgt_texts = row["tgt_text"]
            n = min(args.num_candidates, len(tgt_texts))
            other_idx = random.sample(range(1, len(tgt_texts)), n - 1)
            cand_idx = [0] + other_idx  # keep ground truth at index 0
            cand_texts = [tgt_texts[j] for j in cand_idx]
            pool_sizes.append(n)

            qry_inputs = process_vlm_inputs(
                {"text": [qry_text], "image": [qry_image]},
                processor=processor, backbone_name=args.model_backbone, max_length=args.max_len,
            )
            tgt_inputs = process_vlm_inputs(
                {"text": cand_texts, "image": [None] * len(cand_texts)},
                processor=processor, backbone_name=args.model_backbone, max_length=args.max_len,
            )

            qry_rep = model.encode_input(qry_inputs)  # (1, dim)
            tgt_rep = model.encode_input(tgt_inputs)  # (n, dim)

            scores = model.compute_similarity(qry_rep, tgt_rep)  # (1, n)
            pred = scores.argmax(dim=-1).item()
            if pred == 0:
                n_correct += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(ds):
                running = n_correct / (i + 1)
                print(f"[{i + 1}/{len(ds)}] running P@1={running:.4f} ({n_correct}/{i + 1})")

    p1 = n_correct / len(ds)
    avg_pool = sum(pool_sizes) / len(pool_sizes)
    chance = 1.0 / avg_pool
    print(
        f"\nFinal P@1 over {len(ds)} queries, avg candidate pool size={avg_pool:.1f} "
        f"(chance={chance:.4f}): {p1:.4f} ({n_correct}/{len(ds)})"
    )


if __name__ == "__main__":
    main()
