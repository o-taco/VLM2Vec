"""A-OKVQA data loading and image-disjoint split construction for the early-exit probe.

HuggingFaceM4/A-OKVQA's `train` split has no explicit image-id column, and A-OKVQA
predominantly has one question per image (a small minority of images have two), so we
split by row (question) rather than by a deduplicated image key. This is a documented
approximation of the deck's stricter "image-disjoint" rule, not an exact match.

`train` -> partitioned into answer-fit / confidence-fit / calibration (70/15/15).
`validation` -> held out entirely as the eval split (never touched during training or
calibration), matching the deck's "every MMEB evaluation set is evaluation-only".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

SplitName = Literal["answer_fit", "confidence_fit", "calibration"]

SPLIT_FRACTIONS = {"answer_fit": 0.70, "confidence_fit": 0.15, "calibration": 0.15}


def build_split_index(seed: int = 26091622, cache_dir: str | None = None) -> dict:
    """Returns {"train_question_ids": [...], "split_of": {question_id: split_name}}."""
    ds = load_dataset("HuggingFaceM4/A-OKVQA", split="train", cache_dir=cache_dir)
    qids = ds["question_id"]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(qids))
    n = len(qids)
    n_answer = int(round(n * SPLIT_FRACTIONS["answer_fit"]))
    n_conf = int(round(n * SPLIT_FRACTIONS["confidence_fit"]))
    split_of = {}
    for rank, idx in enumerate(order):
        if rank < n_answer:
            split_of[qids[idx]] = "answer_fit"
        elif rank < n_answer + n_conf:
            split_of[qids[idx]] = "confidence_fit"
        else:
            split_of[qids[idx]] = "calibration"
    return {"seed": seed, "n_total": n, "split_of": split_of}


def load_or_build_split_index(path: Path, cache_dir: str | None = None) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    index = build_split_index(cache_dir=cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(index, f)
    return index


@dataclass
class Example:
    question_id: str
    question: str
    choices: list[str]
    correct_choice_idx: int
    image: "object"  # PIL.Image


class AOKVQADataset(Dataset):
    """Wraps a HF A-OKVQA split, optionally filtered to one probe split ('answer_fit',
    'confidence_fit', 'calibration'). Pass split_filter=None for the raw HF split as-is
    (used for the held-out 'validation' eval split, which has no sub-split)."""

    def __init__(
        self,
        hf_split: str,
        split_filter: SplitName | None = None,
        split_index: dict | None = None,
        cache_dir: str | None = None,
        limit: int | None = None,
    ):
        self.ds = load_dataset("HuggingFaceM4/A-OKVQA", split=hf_split, cache_dir=cache_dir)
        if split_filter is not None:
            assert split_index is not None, "split_index required when split_filter is set"
            split_of = split_index["split_of"]
            keep = [i for i, qid in enumerate(self.ds["question_id"]) if split_of.get(qid) == split_filter]
            self.ds = self.ds.select(keep)
        if limit is not None and limit < len(self.ds):
            self.ds = self.ds.select(range(limit))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx) -> Example:
        row = self.ds[idx]
        return Example(
            question_id=row["question_id"],
            question=row["question"],
            choices=row["choices"],
            correct_choice_idx=row["correct_choice_idx"],
            image=row["image"].convert("RGB"),
        )


CHOICE_LETTERS = "ABCDEFGH"


def build_prompt(question: str, choices: list[str]) -> str:
    lines = [question]
    for i, choice in enumerate(choices):
        lines.append(f"{CHOICE_LETTERS[i]}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)
