"""Extract InternVL3 A-OKVQA candidate features for the frozen probe.

The resulting NPZ has the same feature/label/question/choice schema consumed by
``cross_family_candidate_probe.py fit``.  InternVL3 uses repository-provided
model and conversation code, so loading it intentionally enables
``trust_remote_code``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def candidate_text(question: str, candidate: str) -> str:
    return f"Question: {question}\nCandidate answer: {candidate}"


def build_transform(input_size: int) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda image: image.convert("RGB")),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def dynamic_preprocess(
    image: Image.Image,
    min_num: int,
    max_num: int,
    image_size: int,
    use_thumbnail: bool,
) -> list[Image.Image]:
    """Official InternVL aspect-ratio tiling, adapted to accept a PIL image."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda ratio: ratio[0] * ratio[1])

    best_ratio = (1, 1)
    best_ratio_diff = float("inf")
    area = orig_width * orig_height
    for ratio in target_ratios:
        ratio_diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]:
            best_ratio = ratio

    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]
    resized = image.resize((target_width, target_height))
    tiles = []
    for index in range(blocks):
        left = (index % (target_width // image_size)) * image_size
        upper = (index // (target_width // image_size)) * image_size
        tiles.append(resized.crop((left, upper, left + image_size, upper + image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def preprocess_image(image: Image.Image, config, max_patches: int | None = None) -> torch.Tensor:
    image_size = config.force_image_size or config.vision_config.image_size
    tiles = dynamic_preprocess(
        image.convert("RGB"),
        min_num=config.min_dynamic_patch if config.dynamic_image_size else 1,
        max_num=(max_patches or config.max_dynamic_patch) if config.dynamic_image_size else 1,
        image_size=image_size,
        use_thumbnail=config.use_thumbnail,
    )
    transform = build_transform(image_size)
    return torch.stack([transform(tile) for tile in tiles])


def internvl_prompt(model, question: str, candidate: str, num_patches: int) -> str:
    """Reproduce InternVLChatModel.chat without invoking generation."""
    template = model.conv_template.copy()
    template.system_message = model.system_message
    user_text = "<image>\n" + candidate_text(question, candidate)
    template.append_message(template.roles[0], user_text)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    image_tokens = "<img>" + IMG_CONTEXT_TOKEN * (model.num_image_token * num_patches) + "</img>"
    return query.replace("<image>", image_tokens, 1)


def load_internvl(model_id: str, depth: int, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
    )
    layers = model.language_model.model.layers
    total_depth = len(layers)
    if not 1 <= depth <= total_depth:
        raise ValueError(f"depth must be in [1, {total_depth}], got {depth}")
    if depth < total_depth:
        model.language_model.model.layers = nn.ModuleList(list(layers[:depth]))
        model.language_model.config.num_hidden_layers = depth
        model.config.llm_config.num_hidden_layers = depth
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.eval().to(device)
    return model, tokenizer, total_depth


@torch.inference_mode()
def encode_batch(model, tokenizer, questions, candidates, images, max_patches=None, device: str = "cuda"):
    tile_batches = [preprocess_image(image, model.config, max_patches) for image in images]
    num_patches = [tiles.shape[0] for tiles in tile_batches]
    prompts = [
        internvl_prompt(model, question, candidate, patches)
        for question, candidate, patches in zip(questions, candidates, num_patches)
    ]
    tokenizer.padding_side = "left"
    tokens = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    pixel_values = torch.cat(tile_batches).to(device=device, dtype=torch.bfloat16)
    image_flags = torch.ones((pixel_values.shape[0], 1), dtype=torch.long, device=device)
    # This is InternVLChatModel.forward's image-token injection, followed by the
    # Qwen2 base model rather than Qwen2ForCausalLM.  It returns the normalized
    # state after the selected final layer without allocating vocabulary logits
    # or retaining all intermediate hidden states.
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
        vision_embeds = model.extract_feature(pixel_values)
        vision_embeds = vision_embeds[image_flags.squeeze(-1) == 1]
        batch, sequence, width = input_embeds.shape
        flat_embeds = input_embeds.reshape(batch * sequence, width)
        image_positions = input_ids.reshape(-1) == model.img_context_token_id
        expected = int(image_positions.sum().item())
        available = vision_embeds.numel() // width
        if expected != available:
            raise RuntimeError(
                f"image-token mismatch: prompt has {expected} IMG_CONTEXT tokens "
                f"but vision encoder produced {available} embeddings"
            )
        flat_embeds[image_positions] = vision_embeds.reshape(-1, width)
        output = model.language_model.model(
            inputs_embeds=flat_embeds.reshape(batch, sequence, width),
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    hidden = output.last_hidden_state
    positions = torch.arange(hidden.shape[1], device=device).expand_as(attention_mask)
    last_attended = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
    reps = hidden[torch.arange(hidden.shape[0], device=device), last_attended]
    return reps.float().cpu().numpy()


def extract(args) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        print(f"using existing cache: {output_path}")
        return

    model, tokenizer, total_depth = load_internvl(args.model_id, args.depth)
    dataset = load_dataset("HuggingFaceM4/A-OKVQA", split=args.split)
    n_questions = min(len(dataset), args.limit or len(dataset))
    features, labels, question_ids, choice_ids = [], [], [], []
    pending_q, pending_c, pending_img = [], [], []
    pending_y, pending_qid, pending_cid = [], [], []
    started = time.perf_counter()

    def flush() -> None:
        if not pending_q:
            return
        features.append(encode_batch(
            model, tokenizer, pending_q, pending_c, pending_img, args.max_patches
        ))
        labels.extend(pending_y)
        question_ids.extend(pending_qid)
        choice_ids.extend(pending_cid)
        pending_q.clear(); pending_c.clear(); pending_img.clear()
        pending_y.clear(); pending_qid.clear(); pending_cid.clear()

    for qid in tqdm(range(n_questions), desc=f"{args.split} questions"):
        row = dataset[qid]
        image = row["image"].convert("RGB")
        for cid, choice in enumerate(row["choices"]):
            pending_q.append(row["question"]); pending_c.append(choice); pending_img.append(image)
            pending_y.append(int(cid == row["correct_choice_idx"]))
            pending_qid.append(qid); pending_cid.append(cid)
            if len(pending_q) >= args.batch_size:
                flush()
    flush()
    metadata = {
        "family": "internvl3", "model_id": args.model_id, "split": args.split,
        "depth": args.depth, "total_depth": total_depth, "n_questions": n_questions,
        "max_patches": args.max_patches,
        "elapsed_seconds": time.perf_counter() - started,
    }
    np.savez_compressed(
        output_path,
        features=np.concatenate(features).astype(np.float16),
        labels=np.asarray(labels, dtype=np.int8),
        question_ids=np.asarray(question_ids, dtype=np.int32),
        choice_ids=np.asarray(choice_ids, dtype=np.int8),
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--depth", required=True, type=int)
    parser.add_argument("--split", required=True, choices=["train", "validation"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-patches", type=int)
    parser.add_argument("--overwrite", action="store_true")
    extract(parser.parse_args())


if __name__ == "__main__":
    main()
