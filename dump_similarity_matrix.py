"""
Same batch/encode setup as inspect_similarity.py, but saves the raw NxN
cosine-similarity matrix (qry vs pos) to disk as .npy for heatmap plotting,
instead of just printing summary stats.
"""
import argparse
import random

import numpy as np
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
    p.add_argument("--no_lora", action="store_true")
    p.add_argument("--parquet_path", default="data/A-OKVQA/original-00000-of-00001.parquet")
    p.add_argument("--image_dir", default="data")
    p.add_argument("--image_resolution", default="336")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    random.seed(args.seed)

    model_args = ModelArguments(
        model_name=args.model_name,
        model_backbone=args.model_backbone,
        checkpoint_path=args.checkpoint_path,
        pooling="last",
        normalize=True,
        lora=not args.no_lora,
        temperature=0.02,
    )
    model = MMEBModel.load(model_args, is_trainable=False)
    model = model.to("cuda", dtype=torch.bfloat16)
    model.eval()

    processor = load_processor(model_args)

    df = pd.read_parquet(args.parquet_path)
    idx = list(range(len(df)))
    random.shuffle(idx)
    rows = df.iloc[idx[:args.batch_size]].reset_index(drop=True)

    qry_texts, qry_images, pos_texts = [], [], []
    for row in rows.itertuples(index=False):
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

    with torch.no_grad():
        qry_rep = model.encode_input(qry_inputs)
        pos_rep = model.encode_input(pos_inputs)
        scores = model.compute_similarity(qry_rep, pos_rep).float()

    np.save(args.out, scores.cpu().numpy())
    print(f"saved {scores.shape} matrix to {args.out}")


if __name__ == "__main__":
    main()
