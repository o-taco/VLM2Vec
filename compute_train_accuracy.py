"""
Compute training-loss and top-1 accuracy under the EXACT setup used during
training: batches of `batch_size` (qry, pos) pairs with in-batch negatives.

The collator (src/collator.py TrainTextImageDataCollator) loads neg_text but
never returns it — only (qry_inputs, pos_inputs) reach the model, so the
"negatives" during training are simply the other examples' pos_text within
the same batch. This script reproduces that: for each batch of size N, build
an NxN similarity matrix (qry_i vs pos_j) and use the diagonal as target,
exactly like MMEBModel.forward() does.
"""
import argparse
import random

import torch
import pandas as pd
from PIL import Image
import os

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
    p.add_argument("--parquet_path", default="data/A-OKVQA/original-00000-of-00001.parquet")
    p.add_argument("--image_dir", default="data")
    p.add_argument("--image_resolution", default="336")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4, help="must match training's per_device_train_batch_size")
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--num_batches", type=int, default=50, help="50 batches x batch_size 4 = 200 examples, matching the val-loss sample size")
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
        temperature=args.temperature,
    )
    model = MMEBModel.load(model_args, is_trainable=False)
    model = model.to("cuda", dtype=torch.bfloat16)
    model.eval()

    processor = load_processor(model_args)

    df = pd.read_parquet(args.parquet_path)
    idx = list(range(len(df)))
    random.shuffle(idx)
    n_needed = args.batch_size * args.num_batches
    idx = idx[:n_needed]
    rows = df.iloc[idx].reset_index(drop=True)
    print(f"Loaded {len(rows)} train rows -> {args.num_batches} batches of {args.batch_size} (matches training's per_device_train_batch_size)")

    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for b in range(args.num_batches):
            batch = rows.iloc[b * args.batch_size:(b + 1) * args.batch_size]

            qry_texts, qry_images, pos_texts = [], [], []
            for row in batch.itertuples(index=False):
                qry_text = row.qry.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[args.model_backbone])
                qry_image = Image.open(os.path.join(args.image_dir, row.qry_image_path)).convert("RGB")
                qry_image = process_image(qry_image, args.image_resolution)
                qry_texts.append(qry_text)
                qry_images.append(qry_image)
                pos_texts.append(row.pos_text)

            qry_inputs = process_vlm_inputs(
                {"text": qry_texts, "image": qry_images},
                processor=processor, backbone_name=args.model_backbone, max_length=args.max_len,
            )
            pos_inputs = process_vlm_inputs(
                {"text": pos_texts, "image": [None] * len(pos_texts)},
                processor=processor, backbone_name=args.model_backbone, max_length=args.max_len,
            )

            qry_rep = model.encode_input(qry_inputs)  # (batch_size, dim)
            pos_rep = model.encode_input(pos_inputs)  # (batch_size, dim)

            scores = model.compute_similarity(qry_rep, pos_rep) / args.temperature  # (batch_size, batch_size)
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)  # diagonal = correct match
            loss = torch.nn.functional.cross_entropy(scores.float(), target)
            losses.append(loss.item())
            correct += (scores.argmax(dim=-1) == target).sum().item()
            total += scores.size(0)

            if (b + 1) % 10 == 0 or (b + 1) == args.num_batches:
                running_loss = sum(losses) / len(losses)
                running_acc = correct / total
                print(f"[batch {b + 1}/{args.num_batches}] running loss={running_loss:.4f} acc={running_acc:.4f}")

    final_loss = sum(losses) / len(losses)
    final_acc = correct / total
    print(
        f"\nFinal train loss over {args.num_batches} batches of {args.batch_size} "
        f"(in-batch negatives, temperature={args.temperature}): {final_loss:.4f}"
    )
    print(f"Final train top-1 accuracy (in-batch, {total} examples): {final_acc:.4f}")


if __name__ == "__main__":
    main()
