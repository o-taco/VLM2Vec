"""
Compute a contrastive validation loss on MMEB-eval's A-OKVQA test set.

MMEB-eval is shaped for retrieval-style P@1 evaluation (each query paired with
a long candidate-answer list, ground truth always at index 0) rather than the
single-positive-pair shape MMEB-train uses. This reuses the same in-batch-style
contrastive cross-entropy loss MMEBModel.forward() computes during training,
but applied per-query against a subsampled candidate pool instead of in-batch
negatives, since there's no natural "batch of paired examples" here.
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
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--dataset_name", default="TIGER-Lab/MMEB-eval")
    p.add_argument("--subset_name", default="A-OKVQA")
    p.add_argument("--dataset_split", default="test")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--image_resolution", default="336")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--num_candidates", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--limit", type=int, default=None, help="only evaluate the first N queries (for smoke testing)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    model_args = ModelArguments(
        model_name=args.model_name,
        model_backbone=args.model_backbone,
        checkpoint_path=args.checkpoint_path,
        pooling="last",
        normalize=True,
        lora=True,
    )
    model = MMEBModel.load(model_args, is_trainable=False)
    model = model.to("cuda", dtype=torch.bfloat16)
    model.eval()

    processor = load_processor(model_args)

    ds = load_dataset(args.dataset_name, args.subset_name, split=args.dataset_split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"Loaded {len(ds)} eval rows")

    losses = []
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

            scores = model.compute_similarity(qry_rep, tgt_rep) / args.temperature  # (1, n)
            target = torch.zeros(1, dtype=torch.long, device=scores.device)  # ground truth at index 0
            loss = torch.nn.functional.cross_entropy(scores.float(), target)
            losses.append(loss.item())

            if (i + 1) % 100 == 0 or (i + 1) == len(ds):
                running = sum(losses) / len(losses)
                print(f"[{i + 1}/{len(ds)}] running val_loss={running:.4f}")

    val_loss = sum(losses) / len(losses)
    print(
        f"\nFinal val_loss over {len(losses)} queries "
        f"(temperature={args.temperature}, num_candidates={args.num_candidates}): {val_loss:.4f}"
    )


if __name__ == "__main__":
    main()
