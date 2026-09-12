"""Extract per-candidate, per-layer hidden states for the layer-agreement analysis.

Unlike ``cross_family_candidate_probe.py`` (which truncates the decoder to a
fixed depth and re-runs a forward pass per depth), this script runs ONE
full-depth forward pass per batch with ``output_hidden_states=True`` and
records the last-token representation at every depth (the embedding output
plus all 28 decoder layers, 29 vectors total). This is the cheap way to get
"a probe at every layer": a single frozen backbone pass supplies features for
all 29 per-layer probes at once.

Output NPZ schema (consumed by ``fit_layer_probes.py``):
  features:     (N, n_layers+1, hidden_dim) float16
  labels:       (N,) int8            -- 1 if this candidate is correct
  question_ids: (N,) int32
  choice_ids:   (N,) int8
  metadata:     json string
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def candidate_text(question: str, candidate: str) -> str:
    return f"Question: {question}\nCandidate answer: {candidate}"


def qwen_prompt(processor, question: str, candidate: str) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": candidate_text(question, candidate)},
    ]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def load_qwen(model_id: str):
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model.eval().to("cuda")
    n_layers = model.config.text_config.num_hidden_layers
    return model, processor, n_layers


@torch.inference_mode()
def encode_batch_all_layers(model, processor, questions, candidates, images):
    prompts = [qwen_prompt(processor, q, c) for q, c in zip(questions, candidates)]
    inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
    inputs = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    last = inputs["attention_mask"].sum(dim=1) - 1
    idx = torch.arange(last.shape[0], device=last.device)
    # hidden_states: tuple of (n_layers + 1) tensors, each (batch, seq, hidden)
    per_layer = [hs[idx, last].float().cpu().numpy() for hs in out.hidden_states]
    # -> (batch, n_layers + 1, hidden)
    return np.stack(per_layer, axis=1)


def extract(args):
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        print(f"using existing cache: {out_path}")
        return

    model, processor, n_layers = load_qwen(args.model_id)
    ds = load_dataset("HuggingFaceM4/A-OKVQA", split=args.split)
    n_total = len(ds)
    if args.limit and args.limit < n_total:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(n_total, size=args.limit, replace=False))
    else:
        indices = np.arange(n_total)

    features, labels, question_ids, choice_ids = [], [], [], []
    pending_q, pending_c, pending_img = [], [], []
    pending_y, pending_qid, pending_cid = [], [], []
    started = time.perf_counter()

    def flush():
        if not pending_q:
            return
        features.append(encode_batch_all_layers(model, processor, pending_q, pending_c, pending_img))
        labels.extend(pending_y)
        question_ids.extend(pending_qid)
        choice_ids.extend(pending_cid)
        pending_q.clear(); pending_c.clear(); pending_img.clear()
        pending_y.clear(); pending_qid.clear(); pending_cid.clear()

    for qid in tqdm(indices.tolist(), desc=f"{args.split} questions"):
        row = ds[int(qid)]
        image = row["image"].convert("RGB")
        for cid, choice in enumerate(row["choices"]):
            pending_q.append(row["question"])
            pending_c.append(choice)
            pending_img.append(image)
            pending_y.append(int(cid == row["correct_choice_idx"]))
            pending_qid.append(int(qid))
            pending_cid.append(cid)
            if len(pending_q) >= args.batch_size:
                flush()
    flush()
    elapsed = time.perf_counter() - started

    metadata = {
        "model_id": args.model_id, "split": args.split, "n_layers": n_layers,
        "n_questions": len(indices), "seed": args.seed, "elapsed_seconds": elapsed,
    }
    np.savez_compressed(
        out_path,
        features=np.concatenate(features).astype(np.float16),
        labels=np.asarray(labels, dtype=np.int8),
        question_ids=np.asarray(question_ids, dtype=np.int32),
        choice_ids=np.asarray(choice_ids, dtype=np.int8),
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--split", required=True, choices=["train", "validation"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    extract(parser.parse_args())


if __name__ == "__main__":
    main()
