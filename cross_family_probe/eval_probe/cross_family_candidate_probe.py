"""Frozen candidate-scoring probes for cross-family A-OKVQA comparisons.

Each question becomes four independent (image, question, candidate) examples.
A binary logistic-regression scorer is trained on cached hidden states and the
highest-scoring candidate is selected per validation question.

Currently implements Qwen3-VL. InternVL3 and Gemma 4 adapters are added after
their checkpoints are available for local structure validation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


SPLITS = {"train": 17056, "validation": 1145}


def candidate_text(question: str, candidate: str) -> str:
    return f"Question: {question}\nCandidate answer: {candidate}"


def qwen_prompt(processor, question: str, candidate: str) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": candidate_text(question, candidate)},
    ]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def load_qwen(model_id: str, depth: int):
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    layers = model.model.language_model.layers
    total_depth = len(layers)
    if not 1 <= depth <= total_depth:
        raise ValueError(f"depth must be in [1, {total_depth}], got {depth}")
    if depth < total_depth:
        model.model.language_model.layers = nn.ModuleList(list(layers[:depth]))
        model.model.language_model.config.num_hidden_layers = depth
        model.config.text_config.num_hidden_layers = depth
    model.eval().to("cuda")
    return model, processor, total_depth


@torch.inference_mode()
def encode_qwen_batch(model, processor, questions, candidates, images):
    prompts = [qwen_prompt(processor, q, c) for q, c in zip(questions, candidates)]
    inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
    inputs = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    hidden = out.hidden_states[-1]
    last = inputs["attention_mask"].sum(dim=1) - 1
    reps = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    return reps.float().cpu().numpy()


def extract(args):
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        print(f"using existing cache: {out_path}")
        return

    if args.family != "qwen3vl":
        raise NotImplementedError(f"adapter not yet implemented: {args.family}")
    model, processor, total_depth = load_qwen(args.model_id, args.depth)
    ds = load_dataset("HuggingFaceM4/A-OKVQA", split=args.split)
    n_questions = min(len(ds), args.limit or len(ds))

    features, labels, question_ids, choice_ids = [], [], [], []
    pending_q, pending_c, pending_img = [], [], []
    pending_y, pending_qid, pending_cid = [], [], []
    started = time.perf_counter()

    def flush():
        if not pending_q:
            return
        features.append(encode_qwen_batch(model, processor, pending_q, pending_c, pending_img))
        labels.extend(pending_y)
        question_ids.extend(pending_qid)
        choice_ids.extend(pending_cid)
        pending_q.clear(); pending_c.clear(); pending_img.clear()
        pending_y.clear(); pending_qid.clear(); pending_cid.clear()

    for qid in tqdm(range(n_questions), desc=f"{args.split} questions"):
        row = ds[qid]
        image = row["image"].convert("RGB")
        for cid, choice in enumerate(row["choices"]):
            pending_q.append(row["question"])
            pending_c.append(choice)
            pending_img.append(image)
            pending_y.append(int(cid == row["correct_choice_idx"]))
            pending_qid.append(qid)
            pending_cid.append(cid)
            if len(pending_q) >= args.batch_size:
                flush()
    flush()
    elapsed = time.perf_counter() - started

    metadata = {
        "family": args.family, "model_id": args.model_id, "split": args.split,
        "depth": args.depth, "total_depth": total_depth,
        "n_questions": n_questions, "elapsed_seconds": elapsed,
    }
    np.savez_compressed(
        out_path, features=np.concatenate(features).astype(np.float16),
        labels=np.asarray(labels, dtype=np.int8),
        question_ids=np.asarray(question_ids, dtype=np.int32),
        choice_ids=np.asarray(choice_ids, dtype=np.int8),
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(json.dumps(metadata, indent=2))


def fit_and_score(args):
    train = np.load(args.train_cache)
    val = np.load(args.val_cache)
    x_train, y_train = train["features"].astype(np.float32), train["labels"]
    x_val = val["features"].astype(np.float32)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=args.c, solver="lbfgs", max_iter=500, class_weight="balanced"),
    )
    started = time.perf_counter()
    clf.fit(x_train, y_train)
    scores = clf.predict_proba(x_val)[:, 1]
    qids, cids, labels = val["question_ids"], val["choice_ids"], val["labels"]
    correct = total = 0
    for qid in np.unique(qids):
        mask = qids == qid
        pred_cid = cids[mask][np.argmax(scores[mask])]
        gold_cid = cids[mask][np.argmax(labels[mask])]
        correct += int(pred_cid == gold_cid)
        total += 1
    result = {"accuracy": correct / total, "correct": correct, "total": total,
              "fit_score_seconds": time.perf_counter() - started, "C": args.c}
    print(json.dumps(result, indent=2))
    if args.result:
        Path(args.result).write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    ext = sub.add_parser("extract")
    ext.add_argument("--family", required=True, choices=["qwen3vl", "internvl3", "gemma4"])
    ext.add_argument("--model-id", required=True)
    ext.add_argument("--depth", required=True, type=int)
    ext.add_argument("--split", required=True, choices=list(SPLITS))
    ext.add_argument("--output", required=True)
    ext.add_argument("--limit", type=int)
    ext.add_argument("--batch-size", type=int, default=16)
    ext.add_argument("--overwrite", action="store_true")
    fit = sub.add_parser("fit")
    fit.add_argument("--train-cache", required=True)
    fit.add_argument("--val-cache", required=True)
    fit.add_argument("--result")
    fit.add_argument("--c", type=float, default=1e-3)
    args = parser.parse_args()
    extract(args) if args.command == "extract" else fit_and_score(args)


if __name__ == "__main__":
    main()
